"""Camera motion estimation: how the whole picture moved between consecutive frames.

The camera is handheld and follows the swimmer, so a box that stays in the same place
on screen belongs to a swimmer who moved, and a box that jumped may belong to a
swimmer who stayed still. The tracker (1.4) therefore works in a stabilized "world"
frame: the pixel coordinates of frame 0, with every later frame mapped back into it.

Output: <race_id>.motion.parquet with one row per frame: the similarity transform
(uniform scale + rotation + translation) that maps pixels of frame f-1 to pixels of
frame f, as a, b, tx, ty with  x' = a*x - b*y + tx,  y' = b*x + a*y + ty.  Frame 0 is
the identity. `inliers` is how many background features agreed; 0 means we fell back
to a plain translation or gave up (identity).

Teaching notes
--------------
We find a few hundred corner-like points on a small grayscale copy of the previous
frame (lane ropes, deck edges, flags), follow them into the current frame with
pyramidal Lucas-Kanade optical flow, and fit one similarity transform to all of them
with RANSAC. RANSAC keeps the majority motion, which is the background; the swimmers
are a minority and get discarded as outliers. Scale captures zooming, which a plain
phase correlation cannot see. Working at 480 px wide makes this a few milliseconds a
frame; the transform is then rescaled to full resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("swimtech.motion")

MOTION_COLUMNS = ["frame", "a", "b", "tx", "ty", "inliers"]


@dataclass(frozen=True)
class MotionConfig:
    work_width: int = 480  # estimate on a frame downscaled to this width
    max_features: int = 500
    min_inliers: int = 15  # below this, fall back to phase correlation
    min_inlier_frac: float = 0.5  # ... and at least this share of tracked points must agree
    ransac_px: float = 3.0  # inlier threshold at work scale


def identity_rows(n_frames: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype=int),
            "a": 1.0,
            "b": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "inliers": 0,
        }
    )


def _estimate_pair(
    prev: np.ndarray, cur: np.ndarray, cfg: MotionConfig, mask: np.ndarray | None = None
) -> tuple[float, float, float, float, int]:
    """Similarity from `prev` to `cur` (both grayscale, work scale). `mask` (same size,
    uint8) marks where features may be taken from: 0 inside detected people."""
    pts = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=cfg.max_features,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=mask,
    )
    if pts is not None and len(pts) >= cfg.min_inliers:
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, cur, pts, None, winSize=(21, 21), maxLevel=3
        )
        ok = status.reshape(-1) == 1
        if ok.sum() >= cfg.min_inliers:
            src = pts.reshape(-1, 2)[ok]
            dst = nxt.reshape(-1, 2)[ok]
            m, inl = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_px
            )
            n_inl = int(inl.sum()) if inl is not None else 0
            # A real camera motion is agreed on by most background points. If only a
            # minority agrees, we are probably looking at a moving object, not the camera.
            if (
                m is not None
                and n_inl >= cfg.min_inliers
                and n_inl >= cfg.min_inlier_frac * ok.sum()
            ):
                return float(m[0, 0]), float(m[1, 0]), float(m[0, 2]), float(m[1, 2]), n_inl
    # Fallback: translation only (works on texture-poor frames, blind to zoom). The
    # detected people are blanked out here too, so they cannot drive the estimate.
    p = prev.astype(np.float32)
    c = cur.astype(np.float32)
    if mask is not None:
        keep = (mask > 0).astype(np.float32)
        p = p * keep
        c = c * keep
    if p.std() >= 4.0:
        (dx, dy), resp = cv2.phaseCorrelate(p, c)
        if resp > 0.1:
            return 1.0, 0.0, float(dx), float(dy), 0
    return 1.0, 0.0, 0.0, 0.0, 0


def _mask_for(
    boxes: np.ndarray | None, size: tuple[int, int], scale: float, pad: int = 6
) -> np.ndarray | None:
    """uint8 mask at work scale with detected boxes (padded) zeroed out."""
    if boxes is None or len(boxes) == 0:
        return None
    w, h = size
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in np.asarray(boxes, dtype=float).reshape(-1, 4) * scale:
        cv2.rectangle(
            mask,
            (int(x0) - pad, int(y0) - pad),
            (int(np.ceil(x1)) + pad, int(np.ceil(y1)) + pad),
            0,
            -1,
        )
    return mask


def estimate_motion(
    video: Path,
    out: Path | None = None,
    cfg: MotionConfig = MotionConfig(),
    exclude: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The stage function. Reads the normalized video once; writes the Parquet table.

    `exclude`: the detections table (frame, x0, y0, x1, y1, ...). People move on their
    own, so features inside their boxes are not allowed to vote on camera motion."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return identity_rows(0)
    h, w = frame.shape[:2]
    scale = min(1.0, cfg.work_width / w)
    size = (int(round(w * scale)), int(round(h * scale)))
    by_frame: dict[int, np.ndarray] = {}
    if exclude is not None and len(exclude):
        by_frame = {
            int(f): g[["x0", "y0", "x1", "y1"]].to_numpy() for f, g in exclude.groupby("frame")
        }

    def small(bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else g

    rows = [(0, 1.0, 0.0, 0.0, 0.0, 0)]
    prev = small(frame)
    f = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = small(frame)
        mask = _mask_for(by_frame.get(f - 1), size, scale) if by_frame else None
        a, b, tx, ty, inl = _estimate_pair(prev, cur, cfg, mask)
        rows.append((f, a, b, tx / scale, ty / scale, inl))
        prev = cur
        f += 1
    cap.release()
    df = pd.DataFrame(rows, columns=MOTION_COLUMNS)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    moved = np.hypot(df["tx"], df["ty"]).mean()
    log.info("motion: %d frames, mean %.1f px/frame camera movement -> %s", len(df), moved, out)
    return df


# ------------------------------------------------------------------ world transforms


def _as_matrix(a: float, b: float, tx: float, ty: float) -> np.ndarray:
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=float)


def cumulative(motion: pd.DataFrame, n_frames: int | None = None) -> np.ndarray:
    """(n, 3, 3) matrices C_f with  pixel_in_frame_f = C_f @ pixel_in_frame_0.

    Frames beyond the table (or an empty table) get the last known transform."""
    n = n_frames if n_frames is not None else len(motion)
    cum = np.repeat(np.eye(3)[None], max(n, 1), axis=0)
    if len(motion) == 0:
        return cum[:n]
    m = motion.sort_values("frame")
    cur = np.eye(3)
    rows = dict(zip(m["frame"].astype(int), m[["a", "b", "tx", "ty"]].to_numpy(), strict=True))
    for f in range(n):
        if f in rows and f > 0:
            cur = _as_matrix(*rows[f]) @ cur
        cum[f] = cur
    return cum[:n]


def apply(mat: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 (or (n,3,3) per-row) transform to (..., 2) points."""
    xy = np.asarray(xy, dtype=float)
    if mat.ndim == 2:
        out = xy @ mat[:2, :2].T + mat[:2, 2]
        return out
    # per-row: mat (n,3,3), xy (n, ..., 2)
    out = np.einsum("nij,n...j->n...i", mat[:, :2, :2], xy) + mat[:, None, :2, 2].reshape(
        (len(mat),) + (1,) * (xy.ndim - 2) + (2,)
    )
    return out


def scale_of(mat: np.ndarray) -> np.ndarray:
    """Uniform scale factor(s) of 3x3 similarity transform(s)."""
    if mat.ndim == 2:
        return float(np.hypot(mat[0, 0], mat[1, 0]))
    return np.hypot(mat[:, 0, 0], mat[:, 1, 0])


def boxes_to_world(boxes: np.ndarray, cum: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """(k, 4) xyxy boxes seen on `frames` -> frame-0 ("world") coordinates.
    Transforms the center and rescales the size; rotation between frames is tiny."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes
    inv = np.linalg.inv(cum[frames])
    c = apply(inv, ((boxes[:, :2] + boxes[:, 2:]) / 2)[:, None, :])[:, 0, :]
    s = scale_of(inv)[:, None]
    half = (boxes[:, 2:] - boxes[:, :2]) / 2 * s
    return np.hstack([c - half, c + half])


def boxes_to_frame(boxes_w: np.ndarray, cum: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Inverse of boxes_to_world."""
    boxes_w = np.asarray(boxes_w, dtype=float).reshape(-1, 4)
    if len(boxes_w) == 0:
        return boxes_w
    mats = cum[frames]
    c = apply(mats, ((boxes_w[:, :2] + boxes_w[:, 2:]) / 2)[:, None, :])[:, 0, :]
    s = scale_of(mats)[:, None]
    half = (boxes_w[:, 2:] - boxes_w[:, :2]) / 2 * s
    return np.hstack([c - half, c + half])


# ------------------------------------------------------------------------ water mask

WATER_GRID_W = 80  # the mask is stored at this width (height follows the aspect ratio)


def water_mask(bgr: np.ndarray, grid_w: int = WATER_GRID_W) -> np.ndarray:
    """Where the water is, as a small uint8 grid (1 = water). Pool water is the one
    region that is reliably cyan-to-blue and saturated; spectators, the deck and a
    person in front of the lens are not. Holes from swimmers, splash and lane marks
    are closed, so a box on a swimmer still counts as mostly water.

    Teaching notes: HSV hue in OpenCV runs 0-180; cyan is ~90, blue ~120. Saturation
    separates water from white foam and grey tiles; value drops out shadows and dark
    clothing. A morphological close with a swimmer-sized kernel fills the holes."""
    small = cv2.resize(
        bgr,
        (320, max(int(round(320 * bgr.shape[0] / bgr.shape[1])), 8)),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 70) & (h <= 130) & (s >= 40) & (v >= 60)).astype(np.uint8)
    k_close = max(3, small.shape[1] // 12)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    k_dil = max(3, small.shape[1] // 40)
    m = cv2.dilate(m, np.ones((k_dil, k_dil), np.uint8))
    grid_h = max(int(round(grid_w * bgr.shape[0] / bgr.shape[1])), 2)
    return (
        cv2.resize(m.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA) >= 0.5
    ).astype(np.uint8)


def water_fraction(boxes: np.ndarray, mask: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """Fraction of each (k, 4) xyxy box (frame pixels) that lies on water, per the grid."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros(0)
    gh, gw = mask.shape
    sx, sy = gw / frame_w, gh / frame_h
    out = np.zeros(len(boxes))
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        gx0, gx1 = int(np.floor(x0 * sx)), int(np.ceil(x1 * sx))
        gy0, gy1 = int(np.floor(y0 * sy)), int(np.ceil(y1 * sy))
        gx0, gy0 = max(gx0, 0), max(gy0, 0)
        gx1, gy1 = min(max(gx1, gx0 + 1), gw), min(max(gy1, gy0 + 1), gh)
        cell = mask[gy0:gy1, gx0:gx1]
        out[i] = float(cell.mean()) if cell.size else 0.0
    return out


def estimate_water(video: Path, out: Path | None = None) -> np.ndarray:
    """Water mask for every frame: (n_frames, grid_h, grid_w) uint8. Saved as .npz."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    masks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        masks.append(water_mask(frame))
    cap.release()
    arr = np.stack(masks) if masks else np.zeros((0, 2, WATER_GRID_W), np.uint8)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, water=arr)
    log.info(
        "water: %d frames, %.0f%% of the picture is water on average",
        len(arr),
        100 * arr.mean() if len(arr) else 0,
    )
    return arr


def load_water(path: Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    return np.load(path)["water"]
