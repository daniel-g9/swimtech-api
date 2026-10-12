"""Stage 1.4: lane tracker and crop. Follow one chosen swimmer and produce a steady crop.

ByteTrack-style two-pass association over the detections; keep only the track inside
the chosen lane; fill gaps up to 1 s by linear interpolation; smooth the box center with a
15-frame moving average; fix the crop size to the 95th-percentile box height x 1.6.

Output: <race_id>.crop.mp4 at the normalized fps and <race_id>.crop.parquet with
columns frame, x0, y0, w, h (plus `tracked`) so crop coordinates map back to the
normalized frame. The crop size is constant for the whole race.

Why the tracker is in-house: Roboflow's `trackers` package (Apache 2.0) depends on the
non-headless OpenCV build, which clashes with the headless one this repo ships, and
supervision's own ByteTrack is deprecated for removal. The association logic is ~100
lines, has no learned parts, and is tested here on synthetic data; swapping in a library
tracker later only changes `ByteTracker`.

Teaching notes
--------------
ByteTrack's idea: match confident boxes first, then let the *low-confidence* boxes that
every other tracker throws away rescue the tracks that didn't find a confident match.
Splash makes a swimmer a low-confidence box for a few frames; this is what survives it.
Each track predicts where it will be next (constant velocity) so a box that moved is
still matched by overlap. The lane constraint turns a hard multi-object problem into
"one swimmer in a corridor", so ID swaps become nearly impossible. A fixed crop size
matters: pose models are sensitive to scale jitter, and a crop that breathes with the
box makes the skeleton jitter too.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from swimtech.detect import iou_matrix

log = logging.getLogger("swimtech.track")

TRACK_COLUMNS = ["frame", "cx", "cy", "w", "h", "score", "tracked"]
CROP_COLUMNS = ["frame", "x0", "y0", "w", "h", "tracked"]

# `tracked` values in the track and crop tables.
TRACKED_HELD = 0  # no box within max_gap: position held from the nearest known frame
TRACKED_INTERP = 1  # inside a short gap, linearly interpolated
TRACKED_DETECTED = 2  # a detection was associated on this frame


@dataclass(frozen=True)
class LaneSelection:
    """Where the swimmer of interest is. One of:
    - `box`: x0, y0, x1, y1 drawn on the first frame (normalized-video pixels), or
    - `polygon`: the lane quadrilateral, normally the four calibration corners (1.5), or
    - nothing: pick the longest-lived track and warn (fine for a single-lane clip).
    `lane_number` is a label carried into the results; it does not locate anything by
    itself because the camera can be anywhere in the stands."""

    box: tuple[float, float, float, float] | None = None
    polygon: list[list[float]] | None = None
    lane_number: int | None = None

    def corridor(self) -> np.ndarray | None:
        if self.polygon is not None:
            return np.asarray(self.polygon, dtype=float).reshape(-1, 2)
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        return None


@dataclass(frozen=True)
class TrackConfig:
    max_gap_s: float = 1.0  # gaps up to this are interpolated; longer ones are held
    center_smooth_frames: int = 15
    crop_scale: float = 1.6
    crop_height_percentile: float = 95.0
    # ByteTrack association
    high_score: float = 0.6  # first pass uses boxes at or above this
    low_score: float = 0.1  # second pass uses boxes between low and high
    match_iou_high: float = 0.3
    match_iou_low: float = 0.2
    lost_buffer_s: float = 1.0  # a track survives this long without a match
    min_hits: int = 2  # a track needs this many detections to count
    min_in_lane: float = 0.6  # fraction of a track's centers inside the corridor


# ----------------------------------------------------------------------------- tracker


@dataclass
class _Track:
    id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    lost: int = 0

    @property
    def last_frame(self) -> int:
        return self.frames[-1]

    @property
    def last_box(self) -> np.ndarray:
        return self.boxes[-1]

    def predict(self, frame: int) -> np.ndarray:
        return self.last_box + self.velocity * (frame - self.last_frame)

    def add(self, frame: int, box: np.ndarray, score: float) -> None:
        if self.frames:
            dt = max(frame - self.last_frame, 1)
            step = (box - self.last_box) / dt
            # Exponential average: smooth enough to ignore box jitter, fast enough for a turn.
            self.velocity = 0.7 * self.velocity + 0.3 * step
        self.frames.append(frame)
        self.boxes.append(np.asarray(box, dtype=float))
        self.scores.append(float(score))
        self.lost = 0


def _assign(tracks: list[_Track], boxes: np.ndarray, frame: int, iou_min: float) -> list[tuple]:
    """Hungarian matching on 1 - IoU between predicted track boxes and detections."""
    if not tracks or len(boxes) == 0:
        return []
    pred = np.stack([t.predict(frame) for t in tracks])
    iou = iou_matrix(pred, boxes[:, :4])
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= iou_min]


class ByteTracker:
    """Online multi-object tracker. Call `update` once per frame in order; read
    `finished` + `active` afterwards."""

    def __init__(self, fps: float, cfg: TrackConfig = TrackConfig()) -> None:
        self.cfg = cfg
        self.lost_buffer = max(1, int(round(cfg.lost_buffer_s * fps)))
        self.active: list[_Track] = []
        self.finished: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, frame: int) -> None:
        """`boxes`: (k, 5) x0, y0, x1, y1, score on this frame (may be empty)."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
        high = boxes[boxes[:, 4] >= self.cfg.high_score]
        low = boxes[(boxes[:, 4] >= self.cfg.low_score) & (boxes[:, 4] < self.cfg.high_score)]

        # Pass 1: confident boxes against every active track.
        matched_t: set[int] = set()
        matched_high: set[int] = set()
        for r, c in _assign(self.active, high, frame, self.cfg.match_iou_high):
            self.active[r].add(frame, high[c, :4], high[c, 4])
            matched_t.add(r)
            matched_high.add(c)

        # Pass 2: low-confidence boxes against the tracks still unmatched.
        rest_idx = [i for i in range(len(self.active)) if i not in matched_t]
        rest = [self.active[i] for i in rest_idx]
        for r, c in _assign(rest, low, frame, self.cfg.match_iou_low):
            rest[r].add(frame, low[c, :4], low[c, 4])
            matched_t.add(rest_idx[r])

        # Unmatched confident boxes start new tracks.
        for c in range(len(high)):
            if c not in matched_high:
                t = _Track(self._next_id)
                self._next_id += 1
                t.add(frame, high[c, :4], high[c, 4])
                self.active.append(t)

        # Age the rest; retire the ones lost for too long.
        still: list[_Track] = []
        for i, t in enumerate(self.active):
            if i in matched_t or t.frames[-1] == frame:
                still.append(t)
                continue
            t.lost += 1
            (still if t.lost <= self.lost_buffer else self.finished).append(t)
        self.active = still

    @property
    def tracks(self) -> list[_Track]:
        return [t for t in self.finished + self.active if len(t.frames) >= self.cfg.min_hits]


# ---------------------------------------------------------------------- lane selection


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray casting, vectorized. pts (n, 2), poly (k, 2) -> (n,) bool."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def _centers(t: _Track) -> np.ndarray:
    b = np.stack(t.boxes)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])


def select_lane_tracks(
    tracks: list[_Track], corridor: np.ndarray | None, cfg: TrackConfig = TrackConfig()
) -> list[_Track]:
    """Pick the track(s) that are the swimmer of interest.

    With a corridor: every track whose centers lie mostly inside it, merged in time
    order if they don't overlap (the same swimmer re-acquired after a long gap).
    Without one: the longest-lived track, with a warning."""
    if not tracks:
        return []
    if corridor is None:
        best = max(tracks, key=lambda t: len(t.frames))
        log.warning(
            "no lane given: following the longest-lived track (id %d, %d frames). "
            "Pass --lane-box or calibration corners to choose a lane.",
            best.id,
            len(best.frames),
        )
        return [best]
    scored = []
    for t in tracks:
        frac = float(points_in_polygon(_centers(t), corridor).mean())
        if frac >= cfg.min_in_lane:
            scored.append((len(t.frames) * frac, t))
    if not scored:
        log.warning("no track stays inside the lane corridor; nothing to crop")
        return []
    scored.sort(key=lambda s: -s[0])
    chosen: list[_Track] = []
    for _, t in scored:
        if all(t.last_frame < c.frames[0] or t.frames[0] > c.last_frame for c in chosen):
            chosen.append(t)
    chosen.sort(key=lambda t: t.frames[0])
    log.info("lane track(s): %s", [(t.id, t.frames[0], t.last_frame) for t in chosen])
    return chosen


# ------------------------------------------------------------------------ the stage


def track(
    det: pd.DataFrame,
    lane: LaneSelection,
    fps: float,
    cfg: TrackConfig = TrackConfig(),
    n_frames: int | None = None,
) -> pd.DataFrame:
    """Detections -> one dense per-frame track (frame, cx, cy, w, h, score, tracked) for the
    chosen swimmer. Frames without a box within `max_gap_s` hold the nearest known box
    with tracked = 0 so the crop never jumps to nothing."""
    if n_frames is None:
        n_frames = int(det["frame"].max()) + 1 if len(det) else 0
    tracker = ByteTracker(fps, cfg)
    by_frame = dict(tuple(det.groupby("frame"))) if len(det) else {}
    for f in range(n_frames):
        g = by_frame.get(f)
        boxes = g[["x0", "y0", "x1", "y1", "score"]].to_numpy(dtype=float) if g is not None else []
        tracker.update(np.asarray(boxes, dtype=float).reshape(-1, 5), f)

    out = pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype=int),
            "cx": np.nan,
            "cy": np.nan,
            "w": np.nan,
            "h": np.nan,
            "score": 0.0,
            "tracked": TRACKED_HELD,
        }
    )
    for t in select_lane_tracks(tracker.tracks, lane.corridor(), cfg):
        b = np.stack(t.boxes)
        idx = np.asarray(t.frames)
        out.loc[idx, "cx"] = (b[:, 0] + b[:, 2]) / 2
        out.loc[idx, "cy"] = (b[:, 1] + b[:, 3]) / 2
        out.loc[idx, "w"] = b[:, 2] - b[:, 0]
        out.loc[idx, "h"] = b[:, 3] - b[:, 1]
        out.loc[idx, "score"] = t.scores
        out.loc[idx, "tracked"] = TRACKED_DETECTED
    return fill_gaps(out, max_gap=int(round(cfg.max_gap_s * fps)))


def fill_gaps(track_df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Linearly interpolate runs of missing frames no longer than `max_gap` (tracked = 1);
    hold the nearest known box across longer runs and at the ends (tracked = 0)."""
    df = track_df.copy()
    known = df["tracked"].to_numpy() == TRACKED_DETECTED
    if not known.any():
        return df
    tracked = df["tracked"].to_numpy().copy()
    cols = ["cx", "cy", "w", "h", "score"]
    vals = np.array(df[cols].to_numpy(dtype=float))  # a writable copy (pandas may give a view)
    n = len(df)
    i = 0
    while i < n:
        if known[i]:
            i += 1
            continue
        j = i
        while j < n and not known[j]:
            j += 1
        if 0 < i and j < n and (j - i) <= max_gap:
            for k in range(i, j):
                a = (k - (i - 1)) / (j - (i - 1))
                vals[k] = vals[i - 1] * (1 - a) + vals[j] * a
            tracked[i:j] = TRACKED_INTERP
        i = j
    df[cols] = vals
    df["tracked"] = tracked
    # Hold across whatever is still missing (long gaps, leading/trailing frames).
    df[cols] = df[cols].ffill().bfill()
    df.loc[df["tracked"] == TRACKED_HELD, "score"] = 0.0
    return df


def crop_boxes(
    track_df: pd.DataFrame, frame_w: int, frame_h: int, cfg: TrackConfig = TrackConfig()
) -> pd.DataFrame:
    """Track -> fixed-size square crop boxes clamped to the frame. Pure; testable without video.

    Side = 95th-percentile tracked box height x crop_scale, even-sized for the encoder,
    never larger than the frame. The center is a centered moving average of the track."""
    df = track_df.sort_values("frame").reset_index(drop=True)
    det = df[df["tracked"] == TRACKED_DETECTED]
    if det.empty:
        raise ValueError("no tracked frames; cannot size the crop")
    side = float(np.percentile(det["h"], cfg.crop_height_percentile)) * cfg.crop_scale
    side = min(side, float(frame_w), float(frame_h))
    side = int(np.ceil(side / 2.0) * 2)
    side = max(side, 2)

    win = max(1, int(cfg.center_smooth_frames))
    cx = df["cx"].rolling(win, center=True, min_periods=1).mean()
    cy = df["cy"].rolling(win, center=True, min_periods=1).mean()
    x0 = np.clip(np.round(cx - side / 2), 0, max(frame_w - side, 0)).astype(int)
    y0 = np.clip(np.round(cy - side / 2), 0, max(frame_h - side, 0)).astype(int)
    return pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "x0": x0,
            "y0": y0,
            "w": side,
            "h": side,
            "tracked": df["tracked"].astype(int),
        }
    )[CROP_COLUMNS]


def write_crop_video(
    norm_video: Path, crops: pd.DataFrame, out: Path, fps: float | None = None
) -> int:
    """Cut the crop out of every frame and encode it. Uses ffmpeg (H.264) when it is on
    PATH, otherwise OpenCV's built-in mp4v encoder; both are readable by cv2 in 2.1.
    Returns the number of frames written."""
    cap = cv2.VideoCapture(str(norm_video))
    if not cap.isOpened():
        raise FileNotFoundError(norm_video)
    fps = fps or float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    by_frame = crops.set_index("frame")
    side_w, side_h = int(by_frame["w"].iloc[0]), int(by_frame["h"].iloc[0])
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    proc: subprocess.Popen | None = None
    writer: cv2.VideoWriter | None = None
    if ffmpeg:
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{side_w}x{side_h}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
    else:
        log.warning("ffmpeg not on PATH; writing the crop video with OpenCV mp4v instead")
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (side_w, side_h))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open a video writer; install ffmpeg")

    n = 0
    f = 0
    last = int(by_frame.index.max())
    try:
        while f <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if f in by_frame.index:
                x0, y0 = int(by_frame.at[f, "x0"]), int(by_frame.at[f, "y0"])
                patch = frame[y0 : y0 + side_h, x0 : x0 + side_w]
                if patch.shape[0] != side_h or patch.shape[1] != side_w:  # edge safety
                    patch = cv2.copyMakeBorder(
                        patch,
                        0,
                        side_h - patch.shape[0],
                        0,
                        side_w - patch.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(np.ascontiguousarray(patch).tobytes())
                elif writer is not None:
                    writer.write(patch)
                n += 1
            f += 1
    finally:
        cap.release()
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
        if writer is not None:
            writer.release()
    log.info("crop video: %d frames of %dx%d -> %s", n, side_w, side_h, out)
    return n


def crop_to_frame(xy_crop: np.ndarray, crops: pd.DataFrame) -> np.ndarray:
    """Map (n, k, 2) crop-space keypoints back to normalized-frame pixels."""
    off = crops[["x0", "y0"]].to_numpy(dtype=float)[:, None, :]
    return xy_crop + off
