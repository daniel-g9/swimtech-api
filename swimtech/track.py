"""Stage 1.4: swimmer tracker and crop. Follow one chosen swimmer with a moving camera
and produce a steady close-up.

ByteTrack-style two-pass association over the detections, run in a stabilized "world"
frame (the pixels of frame 0, using the camera motion from `motion.py`) so that a pan
or zoom does not look like the swimmer jumping. The swimmer is chosen by one point
tapped on them in the first frame (or, for a fixed camera, a lane box or polygon).
When the track is lost (turn, underwater, splash, someone walking in front) the next
track that starts near where the swimmer was heading is chained on. Gaps up to 1 s are
interpolated; longer gaps hold the last position. The crop follows the swimmer's size
(so zooming is fine) and is written as a fixed-size video.

Output: <race_id>.crop.mp4 and <race_id>.crop.parquet with columns frame, x0, y0, w, h,
scale, tracked. Crop pixel (u, v) maps back to normalized-frame pixel
(x0 + u * scale, y0 + v * scale); `tracked` is 2 detected, 1 interpolated, 0 held.

Why the tracker is in-house: Roboflow's `trackers` package (Apache 2.0) depends on the
non-headless OpenCV build, which clashes with the headless one this repo ships, and
supervision's own ByteTrack is deprecated for removal. The association logic is ~100
lines, has no learned parts, and is tested here on synthetic data.

Teaching notes
--------------
ByteTrack's idea: match confident boxes first, then let the *low-confidence* boxes that
every other tracker throws away rescue the tracks that didn't find a confident match.
Splash makes a swimmer a low-confidence box for a few frames; this is what survives it.
On real footage the detector is unsure about swimmers (median score ~0.25) and sure
about spectators, so new tracks are allowed to start from moderately confident boxes.
Each track predicts where it will be next (constant velocity) so a box that moved is
still matched by overlap. Doing all of this in world coordinates is what makes a
handheld, panning camera work: the background holds still, only the swimmers move.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from swimtech.detect import dedupe_boxes, iou_matrix
from swimtech.motion import (
    apply,
    boxes_to_frame,
    boxes_to_world,
    cumulative,
    water_fraction,
)

log = logging.getLogger("swimtech.track")

TRACK_COLUMNS = ["frame", "cx", "cy", "w", "h", "score", "tracked"]
CROP_COLUMNS = ["frame", "x0", "y0", "w", "h", "scale", "tracked"]

# `tracked` values in the track and crop tables.
TRACKED_HELD = 0  # no box within max_gap: position held from the nearest known frame
TRACKED_INTERP = 1  # inside a short gap, linearly interpolated
TRACKED_DETECTED = 2  # a detection was associated on this frame


@dataclass(frozen=True)
class SwimmerSelection:
    """Which swimmer to follow. One of:
    - `point`: a tap on the swimmer, in pixels of frame `point_frame` (the normal way), or
    - `box` / `polygon`: a lane region in frame-0 pixels (fixed camera only), or
    - nothing: the longest-lived track, with a warning.
    `lane_number` is a label carried into the results; it locates nothing by itself."""

    point: tuple[float, float] | None = None
    point_frame: int = 0
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


LaneSelection = SwimmerSelection  # older name


@dataclass(frozen=True)
class TrackConfig:
    max_gap_s: float = 1.0  # gaps up to this are interpolated; longer ones are held
    center_smooth_frames: int = 15
    crop_scale: float = 1.3  # crop side = this x the box's longer side
    crop_size_smooth_s: float = 4.0  # rolling median window for the crop size
    crop_min_frac: float = 0.12  # never smaller than this share of the frame's short side
    # ByteTrack association
    high_score: float = 0.4  # first pass and new tracks use boxes at or above this
    low_score: float = 0.1  # second pass uses boxes between low and high
    match_iou_high: float = 0.3
    match_iou_low: float = 0.2
    lost_buffer_s: float = 1.0  # a track survives this long without a match
    min_hits: int = 2  # a track needs this many detections to count
    # Selection
    min_in_lane: float = 0.6  # corridor mode: fraction of centers inside the region
    reacquire_s: float = 15.0  # point mode: how long after a loss we still look
    reacquire_radius: float = 4.0  # ... within this many box long-sides of the prediction
    reacquire_max_radius: float = 10.0  # ... never more than this many, however long the gap
    lane_tolerance: float = (
        0.75  # ... and within this many box heights of the swimmer's own lane line
    )
    dedupe_iou: float = 0.7  # overlapping detector boxes above this IoU are merged
    water_min: float = 0.3  # a box must be at least this much on water to be a swimmer
    # Corridor mode: there is only one swimmer in the lane, so a track may start from a
    # weak box and survive a longer gap without risk of grabbing a lane-mate.
    corridor_high_score: float = 0.15
    corridor_lost_buffer_s: float = 2.0
    corridor_max_height: float = 1.5  # box height over lane width; bigger is not a swimmer
    corridor_max_width: float = 4.0  # box width over lane width
    corridor_gate: float = 1.0  # search radius around the prediction, in lane widths...
    corridor_gate_growth: float = 4.0  # ... growing by this many lane widths per second missing (scored on the truth clicks: 1.5 left clip 2 at 38%, 4.0 gives 77%)
    corridor_gate_max: float = 6.0  # ... up to this
    corridor_coast_s: float = 1.0  # velocity decays to zero over this long when missing
    lane_margin: float = (
        0.15  # corridor mode: boxes may sit this far outside the ropes (lane widths)
    )


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


def _center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) / 2


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
    `tracks` afterwards. Coordinates are whatever you feed it (world, normally)."""

    def __init__(self, fps: float, cfg: TrackConfig = TrackConfig()) -> None:
        self.cfg = cfg
        self.lost_buffer = max(1, int(round(cfg.lost_buffer_s * fps)))
        self.active: list[_Track] = []
        self.finished: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, frame: int, seed: np.ndarray | None = None) -> None:
        """`boxes`: (k, 5) x0, y0, x1, y1, score on this frame (may be empty).
        `seed`: a point the user tapped on this frame; any box containing it may start
        a track regardless of score (the detector is unsure about swimmers)."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 5)
        is_high = boxes[:, 4] >= self.cfg.high_score
        if seed is not None and len(boxes):
            under = (
                (boxes[:, 0] <= seed[0])
                & (seed[0] <= boxes[:, 2])
                & (boxes[:, 1] <= seed[1])
                & (seed[1] <= boxes[:, 3])
                & (boxes[:, 4] >= self.cfg.low_score)
            )
            if under.any():
                # Only the smallest box under the point: the others are duplicates or
                # a lane-mate's box that happens to overlap.
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                pick = int(np.flatnonzero(under)[np.argmin(areas[under])])
                is_high[pick] = True
        high = boxes[is_high]
        low = boxes[~is_high & (boxes[:, 4] >= self.cfg.low_score)]

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


# ---------------------------------------------------------------------- selection


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
    """Fixed-camera mode. With a corridor: every track whose centers lie mostly inside
    it, merged in time order if they don't overlap. Without one: the longest-lived
    track, with a warning."""
    if not tracks:
        return []
    if corridor is None:
        best = max(tracks, key=lambda t: len(t.frames))
        log.warning(
            "no swimmer or lane given: following the longest-lived track (id %d, %d frames). "
            "Pass --swimmer x,y to choose.",
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
        log.warning("no track stays inside the lane region; nothing to crop")
        return []
    scored.sort(key=lambda s: -s[0])
    chosen: list[_Track] = []
    for _, t in scored:
        if all(t.last_frame < c.frames[0] or t.frames[0] > c.last_frame for c in chosen):
            chosen.append(t)
    chosen.sort(key=lambda t: t.frames[0])
    log.info("lane track(s): %s", [(t.id, t.frames[0], t.last_frame) for t in chosen])
    return chosen


def select_by_point(
    tracks: list[_Track],
    point: np.ndarray,
    frame: int,
    fps: float,
    cfg: TrackConfig = TrackConfig(),
    lane_dir: np.ndarray | None = None,
    use_lane_axis: bool = True,
) -> list[_Track]:
    """Point mode. `lane_dir` (unit vector along the lane ropes, world coordinates)
    is the lane prior until the swimmer has moved far enough to define their own line.
    Start from the track whose box contains the tapped point at (or
    nearest to) `frame`; whenever that track ends, chain on the track that starts
    within `reacquire_s` closest to where the swimmer was heading (or to where they
    vanished, which is what a turn at the wall looks like)."""
    if not tracks:
        return []
    point = np.asarray(point, dtype=float)
    best: _Track | None = None
    best_d = np.inf
    for t in tracks:
        fr = np.asarray(t.frames)
        i = int(np.argmin(np.abs(fr - frame)))
        if abs(fr[i] - frame) > fps / 2:
            continue
        b = t.boxes[i]
        diag = max(float(np.hypot(b[2] - b[0], b[3] - b[1])), 1.0)
        d = float(np.linalg.norm(_center(b) - point)) / diag
        inside = b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
        if not inside:
            d += 1.0
        if d < best_d:
            best, best_d = t, d
    if best is None or best_d > 3.0:
        # Nothing on that frame. Treat the tap as "last seen here": take the first
        # track that starts within the re-acquisition window close to the point.
        best, best_d = None, np.inf
        for t in tracks:
            if t.frames[0] < frame or t.frames[0] - frame > cfg.reacquire_s * fps:
                continue
            b = t.boxes[0]
            diag = max(float(np.hypot(b[2] - b[0], b[3] - b[1])), 1.0)
            rel = _center(b) - point
            if lane_dir is not None:
                off_lane = abs(float(lane_dir[0] * rel[1] - lane_dir[1] * rel[0]))
                if off_lane > cfg.lane_tolerance * max(float(b[3] - b[1]), 1.0):
                    continue
                d = abs(float(rel @ lane_dir)) / diag  # distance along the lane only
            else:
                d = float(np.linalg.norm(rel)) / diag
            d += 0.5 * (t.frames[0] - frame) / fps  # sooner is better
            if d < best_d:
                best, best_d = t, d
        if best is None or best_d > 3.0:
            log.warning("no track near the tapped point at frame %d; nothing to crop", frame)
            return []
        log.info("no track at the tapped frame; starting from frame %d instead", best.frames[0])

    chain = [best]
    while True:
        cur = chain[-1]
        end = cur.last_frame
        c_end = _center(cur.last_box)
        w_end = max(float(cur.last_box[2] - cur.last_box[0]), 1.0)
        h_end = max(float(cur.last_box[3] - cur.last_box[1]), 1.0)
        size = max(w_end, h_end)
        v = (cur.velocity[:2] + cur.velocity[2:]) / 2
        # The lane. In world coordinates the pool holds still, so the path the swimmer
        # has traced so far is a line along their lane. A re-acquired track must start
        # within half a lane of that line: lane-mates swim parallel, at the same speed,
        # and are otherwise indistinguishable from "the nearest swimmer heading our way".
        if use_lane_axis:
            lane_u, lane_pt, lane_h = _lane_axis(chain, lane_dir, point)
        else:
            lane_u, lane_pt, lane_h = None, point, 1.0
        nxt: _Track | None = None
        nxt_cost = np.inf
        for t in tracks:
            if t in chain or t.frames[0] <= end:
                continue
            gap = t.frames[0] - end
            if gap > cfg.reacquire_s * fps:
                continue
            b0 = t.boxes[0]
            # Same kind of object: a swimmer seen from the stands is a wide, flat box;
            # a spectator standing on deck is tall. Reject a shape change over 2.5x.
            w0 = max(float(b0[2] - b0[0]), 1.0)
            h0 = max(float(b0[3] - b0[1]), 1.0)
            aspect_ratio = (w0 / h0) / (w_end / h_end)
            size_ratio = max(w0, h0) / size
            if not (1 / 2.5 <= aspect_ratio <= 2.5) or not (1 / 2.5 <= size_ratio <= 2.5):
                continue
            c0 = _center(b0)
            if lane_u is not None:
                rel = c0 - lane_pt
                off_lane = abs(float(lane_u[0] * rel[1] - lane_u[1] * rel[0]))
                if off_lane > cfg.lane_tolerance * lane_h:
                    continue
            d = min(
                float(np.linalg.norm(c0 - (c_end + v * gap))),
                float(np.linalg.norm(c0 - c_end)),
            )
            radius = min(
                cfg.reacquire_radius * size + 0.5 * float(np.linalg.norm(v)) * gap,
                cfg.reacquire_max_radius * size,
            )
            if d > radius:
                continue
            cost = d / radius + 0.3 * gap / (cfg.reacquire_s * fps)
            cost -= 0.2 * min(len(t.frames), fps) / fps  # prefer tracks that last
            if cost < nxt_cost:
                nxt, nxt_cost = t, cost
        if nxt is None:
            break
        chain.append(nxt)
    log.info("swimmer track chain: %s", [(t.id, t.frames[0], t.last_frame) for t in chain])
    return chain


def _lane_axis(
    chain: list[_Track], default_dir: np.ndarray | None, default_pt: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """The swimmer's lane line: direction (unit vector), a point on it, and the median
    box height. From the swimmer's own path once they have moved a few box-lengths
    (robust to perspective); until then from the lane-rope direction through the
    tapped point; with neither, no constraint."""
    centers = np.vstack([_centers(t) for t in chain])
    heights = np.concatenate([np.stack(t.boxes)[:, 3] - np.stack(t.boxes)[:, 1] for t in chain])
    h = float(np.median(heights))
    mean = centers.mean(axis=0)
    if len(centers) >= 5:
        _, s, vt = np.linalg.svd(centers - mean, full_matrices=False)
        spread = s[0] / np.sqrt(len(centers))  # std along the main axis
        if spread >= 3.0 * max(h, 1.0):
            return vt[0] / np.linalg.norm(vt[0]), mean, h
    if default_dir is not None:
        return np.asarray(default_dir, dtype=float), np.asarray(default_pt, dtype=float), h
    return None, mean, h


# ------------------------------------------------------------------------ the stage


def track(
    det: pd.DataFrame,
    sel: SwimmerSelection,
    fps: float,
    cfg: TrackConfig = TrackConfig(),
    n_frames: int | None = None,
    motion: pd.DataFrame | None = None,
    water: np.ndarray | None = None,
    frame_size: tuple[int, int] | None = None,
    lane_dir: np.ndarray | None = None,
    lane: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detections (+ camera motion, + water mask) -> one dense per-frame track of the
    chosen swimmer in normalized-frame pixels: frame, cx, cy, w, h, score, tracked.

    `water`: (n_frames, gh, gw) uint8 grid from motion.estimate_water, with `frame_size`
    (w, h). Boxes that are mostly not on water (spectators, the deck, someone in front
    of the lens) are dropped before tracking: swimmers are always in the water.

    `lane`: the per-frame corridor from lanes.track_corridor. Boxes whose center is
    outside the swimmer's lane are dropped too, so lane-mates never enter the tracker
    and re-acquisition after a loss cannot pick one."""
    if n_frames is None:
        n_frames = int(det["frame"].max()) + 1 if len(det) else 0
    cum = (
        cumulative(motion, n_frames)
        if motion is not None
        else np.repeat(np.eye(3)[None], max(n_frames, 1), 0)
    )

    det = det[["frame", "x0", "y0", "x1", "y1", "score"]]
    if water is not None and len(water) and float(water.mean()) < 0.1:
        # An unusual pool color, or the camera never pointed at the water. Filtering
        # would throw everything away, so don't.
        log.warning("water filter: could not find water in the clip; not filtering boxes")
        water = None
    if water is not None and frame_size is not None and len(det):
        keep = np.ones(len(det), dtype=bool)
        for f, idx in det.groupby("frame").indices.items():
            if 0 <= int(f) < len(water):
                wf = water_fraction(
                    det.iloc[idx][["x0", "y0", "x1", "y1"]].to_numpy(), water[int(f)], *frame_size
                )
                keep[idx] = wf >= cfg.water_min
        log.info("water filter: kept %d of %d boxes", int(keep.sum()), len(det))
        det = det[keep]

    use_lane_axis = True
    if lane is not None and len(lane) and len(det):
        from swimtech.lanes import corridor_for_frame

        keep = np.ones(len(det), dtype=bool)
        for f, idx in det.groupby("frame").indices.items():
            c = corridor_for_frame(lane, int(f))
            if c is None:
                continue
            b = det.iloc[idx]
            centers = np.column_stack([(b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2])
            # Box extent across and along the lane, in lane widths: a swimmer is never
            # taller than the lane, so a head in front of the lens or a lifeguard chair
            # whose center happens to fall in the corridor is rejected here.
            nrm = c.normal()
            tan = np.array([-nrm[1], nrm[0]])
            corners = np.stack(
                [
                    np.column_stack([b["x0"], b["y0"]]),
                    np.column_stack([b["x1"], b["y0"]]),
                    np.column_stack([b["x0"], b["y1"]]),
                    np.column_stack([b["x1"], b["y1"]]),
                ]
            )  # (4, k, 2)
            across = (corners @ nrm).max(axis=0) - (corners @ nrm).min(axis=0)
            along = (corners @ tan).max(axis=0) - (corners @ tan).min(axis=0)
            w = max(c.width, 1.0)
            keep[idx] = (
                c.contains(centers, margin=cfg.lane_margin)
                & (across <= cfg.corridor_max_height * w)
                & (along <= cfg.corridor_max_width * w)
            )
        log.info("lane filter: kept %d of %d boxes", int(keep.sum()), len(det))
        det = det[keep]
        use_lane_axis = False  # the corridor already enforces the lane, drift-free
        cfg = replace(
            cfg, high_score=cfg.corridor_high_score, lost_buffer_s=cfg.corridor_lost_buffer_s
        )
        if sel.point is not None:
            out = _track_in_corridor(det, lane, motion, sel, fps, n_frames, cfg)
            out[["cx", "cy", "w", "h"]] = out[["cx", "cy", "w", "h"]].ffill().bfill()
            return out

    # Everything below happens in world (frame-0) coordinates.
    detw = det.copy()
    if len(detw):
        frames = detw["frame"].to_numpy(dtype=int)
        detw[["x0", "y0", "x1", "y1"]] = boxes_to_world(
            detw[["x0", "y0", "x1", "y1"]].to_numpy(), cum, frames
        )
    point_w = None
    pf = 0
    if sel.point is not None:
        pf = min(max(int(sel.point_frame), 0), max(n_frames - 1, 0))
        point_w = apply(np.linalg.inv(cum[pf]), np.asarray(sel.point, dtype=float))

    tracker = ByteTracker(fps, cfg)
    by_frame = dict(tuple(detw.groupby("frame"))) if len(detw) else {}
    for f in range(n_frames):
        g = by_frame.get(f)
        boxes = g[["x0", "y0", "x1", "y1", "score"]].to_numpy(dtype=float) if g is not None else []
        boxes = dedupe_boxes(np.asarray(boxes, dtype=float).reshape(-1, 5), cfg.dedupe_iou)
        tracker.update(boxes, f, seed=point_w if (point_w is not None and f == pf) else None)

    if point_w is not None:
        chosen = select_by_point(
            tracker.tracks, point_w, pf, fps, cfg, lane_dir=lane_dir, use_lane_axis=use_lane_axis
        )
    else:
        chosen = select_lane_tracks(tracker.tracks, sel.corridor(), cfg)

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
    for t in chosen:
        b = np.stack(t.boxes)
        idx = np.asarray(t.frames)
        out.loc[idx, "cx"] = (b[:, 0] + b[:, 2]) / 2
        out.loc[idx, "cy"] = (b[:, 1] + b[:, 3]) / 2
        out.loc[idx, "w"] = b[:, 2] - b[:, 0]
        out.loc[idx, "h"] = b[:, 3] - b[:, 1]
        out.loc[idx, "score"] = t.scores
        out.loc[idx, "tracked"] = TRACKED_DETECTED
    out = fill_gaps(out, max_gap=int(round(cfg.max_gap_s * fps)))

    # Back to the pixels of each frame.
    if out["cx"].notna().any():
        frames = out["frame"].to_numpy(dtype=int)
        half = out[["w", "h"]].to_numpy() / 2
        c = out[["cx", "cy"]].to_numpy()
        bf = boxes_to_frame(np.hstack([c - half, c + half]), cum, frames)
        out["cx"] = (bf[:, 0] + bf[:, 2]) / 2
        out["cy"] = (bf[:, 1] + bf[:, 3]) / 2
        out["w"] = bf[:, 2] - bf[:, 0]
        out["h"] = bf[:, 3] - bf[:, 1]
    return out


def _track_in_corridor(
    det: pd.DataFrame,
    lane: pd.DataFrame,
    motion: pd.DataFrame | None,
    sel: SwimmerSelection,
    fps: float,
    n_frames: int,
    cfg: TrackConfig,
) -> pd.DataFrame:
    """One swimmer in a corridor: follow them frame by frame.

    Predict where the swimmer is from where they were, their own velocity, and the
    camera motion since the last frame; take the best box near the prediction; when
    none is near, coast (velocity decaying to zero, as at a wall) and widen the search
    the longer they have been missing. Runs forward from the tap frame, then backward.
    Lane-mates cannot be picked because they are not in `det` any more; the rest is a
    tie between the swimmer and their own splash, which the crop absorbs.

    Works in each frame's own pixels: only the frame-to-frame camera motion is used,
    so cumulative drift never enters."""
    from swimtech.lanes import corridor_for_frame
    from swimtech.motion import _as_matrix

    steps: dict[int, np.ndarray] = {}
    if motion is not None and len(motion):
        for _, r in motion.iterrows():
            steps[int(r["frame"])] = _as_matrix(r["a"], r["b"], r["tx"], r["ty"])
    by_frame = (
        {
            int(f): g[["x0", "y0", "x1", "y1", "score"]].to_numpy(dtype=float)
            for f, g in det.groupby("frame")
        }
        if len(det)
        else {}
    )
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
    pf = min(max(int(sel.point_frame), 0), max(n_frames - 1, 0))
    point = np.asarray(sel.point, dtype=float)
    frame_wh = (
        np.array([det["x1"].max(), det["y1"].max()], dtype=float)
        if len(det)
        else np.array([1e9, 1e9])
    )

    def lane_width(f: int) -> float:
        c = corridor_for_frame(lane, f)
        return max(c.width, 1.0) if c is not None else 100.0

    def pick2(f: int, pred_w: np.ndarray, pred_s: np.ndarray, gate: float, size: float | None):
        boxes = dedupe_boxes(by_frame.get(f, np.zeros((0, 5))), cfg.dedupe_iou)
        if len(boxes) == 0:
            return None
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
        d = (
            np.minimum(
                np.linalg.norm(centers - pred_w, axis=1), np.linalg.norm(centers - pred_s, axis=1)
            )
            / gate
        )
        cost = d - 0.5 * boxes[:, 4]
        if size is not None:
            sz = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            cost += 0.3 * np.abs(np.log(np.maximum(sz, 1.0) / size))
        ok = d <= 1.0
        if not ok.any():
            return None
        return boxes[int(np.argmin(np.where(ok, cost, np.inf)))]

    def pick(f: int, pred: np.ndarray, gate: float, size: float | None):
        boxes = dedupe_boxes(by_frame.get(f, np.zeros((0, 5))), cfg.dedupe_iou)
        if len(boxes) == 0:
            return None
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
        d = np.linalg.norm(centers - pred, axis=1) / gate
        cost = d - 0.5 * boxes[:, 4]
        if size is not None:
            sz = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
            cost += 0.3 * np.abs(np.log(np.maximum(sz, 1.0) / size))
        ok = d <= 1.0
        if not ok.any():
            return None
        i = int(np.argmin(np.where(ok, cost, np.inf)))
        return boxes[i]

    def start_box():
        """The box under the tap, or the nearest box within a lane width on frames
        around the tap frame (the swimmer may not be detected on that exact frame)."""
        for radius in range(0, int(fps * 3)):
            for f in {pf + radius, pf - radius}:
                if not (0 <= f < n_frames):
                    continue
                boxes = dedupe_boxes(by_frame.get(f, np.zeros((0, 5))), cfg.dedupe_iou)
                if len(boxes) == 0:
                    continue
                under = (
                    (boxes[:, 0] <= point[0])
                    & (point[0] <= boxes[:, 2])
                    & (boxes[:, 1] <= point[1])
                    & (point[1] <= boxes[:, 3])
                )
                if under.any():
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    return f, boxes[int(np.flatnonzero(under)[np.argmin(areas[under])])]
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
                d = np.linalg.norm(centers - point, axis=1)
                if d.min() <= lane_width(f):
                    return f, boxes[int(np.argmin(d))]
        return None, None

    f0, b0 = start_box()
    if b0 is None:
        log.warning("corridor tracker: no box near the tapped point; nothing to crop")
        return out

    def write(f: int, b: np.ndarray) -> None:
        out.loc[f, ["cx", "cy", "w", "h", "score", "tracked"]] = [
            (b[0] + b[2]) / 2,
            (b[1] + b[3]) / 2,
            b[2] - b[0],
            b[3] - b[1],
            b[4],
            TRACKED_DETECTED,
        ]

    def sweep(frames: range, forward: bool) -> None:
        # Two hypotheses about where a missing swimmer is: fixed in the world (the
        # camera panned, they stayed put: pos follows the camera motion) and fixed on
        # screen (the camera is following them: pos stays). A handheld camera at a meet
        # does both, so a box near either prediction counts, and while they are missing
        # neither prediction is allowed to run off the frame.
        pos_w = (b0[:2] + b0[2:4]) / 2  # world-static hypothesis, in current pixels
        pos_s = pos_w.copy()  # screen-static hypothesis
        vel = np.zeros(2)  # on-screen velocity while tracked
        size = float(max(b0[2] - b0[0], b0[3] - b0[1]))
        missing = 0
        prev = f0
        for f in frames:
            m = steps.get(f if forward else prev, np.eye(3))
            if not forward:
                m = np.linalg.inv(m)
            lin = m[:2, :2]
            pos_w = lin @ pos_w + m[:2, 2]
            w = lane_width(f)
            vcap = 0.25 * w
            if np.linalg.norm(vel) > vcap:
                vel = vel / np.linalg.norm(vel) * vcap
            pred_w = pos_w + vel
            pred_s = pos_s + vel
            gate = (
                min(
                    cfg.corridor_gate + cfg.corridor_gate_growth * missing / fps,
                    cfg.corridor_gate_max,
                )
                * w
            )
            b = pick2(f, pred_w, pred_s, gate, size)
            if b is not None:
                c = (b[:2] + b[2:4]) / 2
                step_v = c - pos_s
                vel = 0.6 * vel + 0.4 * step_v if missing == 0 else step_v / (missing + 1)
                pos_w = c
                pos_s = c
                size = 0.9 * size + 0.1 * float(max(b[2] - b[0], b[3] - b[1]))
                missing = 0
                write(f, b)
            else:
                missing += 1
                vel *= max(0.0, 1.0 - 1.0 / max(cfg.corridor_coast_s * fps, 1.0))
                pos_w = np.clip(pos_w + vel, 0, frame_wh)
                pos_s = np.clip(pos_s + vel, 0, frame_wh)
                # While missing, the box follows the screen-static prediction: a camera
                # at a meet is following the swimmer, so where they were on screen is
                # the best guess (linear interpolation between two detections a second
                # apart was measured to land 2-5 lane widths off during a pan).
                out.loc[f, ["cx", "cy", "w", "h", "score", "tracked"]] = [
                    pos_s[0],
                    pos_s[1],
                    size,
                    size,
                    0.0,
                    TRACKED_INTERP if missing <= cfg.max_gap_s * fps else TRACKED_HELD,
                ]
            prev = f

    write(f0, b0)
    sweep(range(f0 + 1, n_frames), forward=True)
    sweep(range(f0 - 1, -1, -1), forward=False)
    n_det = int((out["tracked"] == TRACKED_DETECTED).sum())
    log.info("corridor tracker: swimmer detected on %d of %d frames", n_det, n_frames)
    return out


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
    track_df: pd.DataFrame,
    frame_w: int,
    frame_h: int,
    cfg: TrackConfig = TrackConfig(),
    fps: float = 30.0,
) -> pd.DataFrame:
    """Track -> square crop boxes clamped to the frame. Pure; testable without video.

    The side follows the swimmer's size: a rolling median (over `crop_size_smooth_s`)
    of the box's LONGER side, times `crop_scale`. Seen from the side a swimmer is a
    long flat box, 2-3x wider than tall, and the recovery arm reaches beyond it; sizing
    from the height alone gave head-and-shoulders crops with no limbs in them. The
    median keeps frame-to-frame jitter out; the center is a centered moving average.
    `scale` is side / output video side; the output side is the 95th percentile."""
    df = track_df.sort_values("frame").reset_index(drop=True)
    det = df[df["tracked"] == TRACKED_DETECTED]
    if det.empty:
        raise ValueError("no tracked frames; cannot size the crop")
    win_size = max(1, int(round(cfg.crop_size_smooth_s * fps)))
    long_side = np.maximum(df["w"].to_numpy(dtype=float), df["h"].to_numpy(dtype=float))
    side = pd.Series(long_side).rolling(win_size, center=True, min_periods=1).median()
    side = side.to_numpy() * cfg.crop_scale
    frame_min = min(frame_w, frame_h)
    side = np.minimum(np.maximum(side, cfg.crop_min_frac * frame_min), frame_min)
    side = (np.ceil(side / 2.0) * 2).astype(int)
    out_side = int(np.ceil(np.percentile(side, 95) / 2.0) * 2)

    win = max(1, int(cfg.center_smooth_frames))
    cx = df["cx"].rolling(win, center=True, min_periods=1).mean().to_numpy()
    cy = df["cy"].rolling(win, center=True, min_periods=1).mean().to_numpy()
    x0 = np.clip(np.round(cx - side / 2), 0, np.maximum(frame_w - side, 0)).astype(int)
    y0 = np.clip(np.round(cy - side / 2), 0, np.maximum(frame_h - side, 0)).astype(int)
    return pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "x0": x0,
            "y0": y0,
            "w": side,
            "h": side,
            "scale": side / out_side,
            "tracked": df["tracked"].astype(int),
        }
    )[CROP_COLUMNS]


def tracking_report(track_df: pd.DataFrame, fps: float) -> dict:
    """What the user needs to know about whether we could see the swimmer.
    Goes into the probe JSON after the track stage."""
    t = track_df["tracked"].to_numpy()
    held = t == TRACKED_HELD
    longest = 0
    run = 0
    for h in held:
        run = run + 1 if h else 0
        longest = max(longest, run)
    det = track_df[track_df["tracked"] == TRACKED_DETECTED]
    height = float(det["h"].median()) if len(det) else 0.0
    rep = {
        "detected_frac": round(float((t == TRACKED_DETECTED).mean()), 3),
        "interpolated_frac": round(float((t == TRACKED_INTERP).mean()), 3),
        "held_frac": round(float(held.mean()), 3),
        "longest_gap_s": round(longest / fps, 1),
        "swimmer_height_px": round(height, 0),
        "warnings": [],
    }
    if rep["detected_frac"] + rep["interpolated_frac"] < 0.6:
        rep["warnings"].append(
            f"Swimmer found in only {100 * (1 - rep['held_frac']):.0f}% of frames. "
            "Keep them in view and closer to the camera."
        )
    if rep["longest_gap_s"] >= 5:
        rep["warnings"].append(
            f"Swimmer lost for {rep['longest_gap_s']:.0f} s at a stretch (turn, obstruction or "
            "underwater phase)."
        )
    if 0 < height < 60:
        rep["warnings"].append(
            f"Swimmer is only ~{height:.0f} px tall; limbs will be hard to see. Move closer or "
            "use the telephoto lens."
        )
    return rep


def write_crop_video(
    norm_video: Path, crops: pd.DataFrame, out: Path, fps: float | None = None
) -> int:
    """Cut the crop out of every frame, resize to the fixed output side, encode. Uses
    ffmpeg (H.264) when it is on PATH, otherwise OpenCV's built-in mp4v encoder; both
    are readable by cv2 in 2.1. Returns the number of frames written."""
    cap = cv2.VideoCapture(str(norm_video))
    if not cap.isOpened():
        raise FileNotFoundError(norm_video)
    fps = fps or float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    by_frame = crops.set_index("frame")
    scale_col = by_frame["scale"] if "scale" in by_frame else pd.Series(1.0, index=by_frame.index)
    out_side = int(round(float(by_frame["w"].iloc[0]) / float(scale_col.iloc[0])))
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
                f"{out_side}x{out_side}",
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
        writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_side, out_side)
        )
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
                w, h = int(by_frame.at[f, "w"]), int(by_frame.at[f, "h"])
                patch = frame[y0 : y0 + h, x0 : x0 + w]
                if patch.shape[0] != h or patch.shape[1] != w:  # edge safety
                    patch = cv2.copyMakeBorder(
                        patch,
                        0,
                        h - patch.shape[0],
                        0,
                        w - patch.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                if (h, w) != (out_side, out_side):
                    patch = cv2.resize(patch, (out_side, out_side), interpolation=cv2.INTER_AREA)
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
    log.info("crop video: %d frames of %dx%d -> %s", n, out_side, out_side, out)
    return n


def crop_to_frame(xy_crop: np.ndarray, crops: pd.DataFrame) -> np.ndarray:
    """Map (n, k, 2) crop-video keypoints back to normalized-frame pixels."""
    off = crops[["x0", "y0"]].to_numpy(dtype=float)[:, None, :]
    scale = crops["scale"].to_numpy(dtype=float)[:, None, None] if "scale" in crops else 1.0
    return xy_crop * scale + off
