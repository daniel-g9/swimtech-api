"""Stage 1.5: lane-plane calibration for `_rect` metrics.

The user clicks the four lane corners (two rope ends at each wall) on the empty-lane
clip, or as a fallback on the race's first frame. We compute a homography from image
pixels to a top-down lane frame in meters: x along the lane (0 at the near wall),
y across the lane (0 at the left rope as seen from the camera).

Teaching notes
--------------
A homography maps points on one plane to another plane; four non-collinear
correspondences fix it. Only points on the water surface (shoulders, hips, the wrist at
entry) lie on the lane plane, so only those are rectified. A recovering arm is above the
water and its rectified position would be wrong, which is why metric names say which
space they use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LANE_WIDTH_M = 2.5
LANE_WIDTH_7FT_M = 2.13
POOL_LENGTH_25YD_M = 22.86
POOL_LENGTH_25M_M = 25.0
POOL_LENGTH_50M_M = 50.0


@dataclass
class Calibration:
    """Image -> lane-plane homography. `corners_px` are in the order:
    near-left, near-right, far-right, far-left (as seen from the camera)."""

    corners_px: list[list[float]]
    lane_width_m: float = LANE_WIDTH_M
    pool_length_m: float = POOL_LENGTH_25YD_M
    source: str = "empty_lane_clip"  # or "race_first_frame"
    H: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.corners_px) != 4:
            raise ValueError("need exactly four lane corners")
        if not self.H:
            self.H = compute_homography(
                np.asarray(self.corners_px, dtype=np.float32),
                self.lane_width_m,
                self.pool_length_m,
            ).tolist()

    def to_rect(self, x: float, y: float) -> tuple[float, float]:
        """Map one image point to lane meters. NaN in, NaN out."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return float("nan"), float("nan")
        pt = np.array([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, np.asarray(self.H, dtype=np.float64))
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def to_rect_array(self, xy: np.ndarray) -> np.ndarray:
        """Vectorized version for an (..., 2) array. NaNs propagate."""
        shape = xy.shape
        flat = xy.reshape(-1, 2).astype(np.float64)
        ok = np.isfinite(flat).all(axis=1)
        out = np.full_like(flat, np.nan)
        if ok.any():
            pts = flat[ok].reshape(1, -1, 2)
            out[ok] = cv2.perspectiveTransform(pts, np.asarray(self.H, dtype=np.float64))[0]
        return out.reshape(shape)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "corners_px": self.corners_px,
                    "lane_width_m": self.lane_width_m,
                    "pool_length_m": self.pool_length_m,
                    "source": self.source,
                    "H": self.H,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> Calibration:
        return cls(**json.loads(Path(path).read_text()))


def compute_homography(
    corners_px: np.ndarray, lane_width_m: float, pool_length_m: float
) -> np.ndarray:
    """Four image corners (near-left, near-right, far-right, far-left) -> 3x3 H."""
    dst = np.array(
        [
            [0.0, 0.0],
            [0.0, lane_width_m],
            [pool_length_m, lane_width_m],
            [pool_length_m, 0.0],
        ],
        dtype=np.float32,
    )
    # OpenCV happily returns a matrix for collinear points, so check the quadrilateral
    # has real area first (shoelace formula), relative to its extent.
    pts = corners_px.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    extent = max(np.ptp(x), np.ptp(y), 1.0)
    if area < 0.01 * extent * extent:
        raise ValueError("degenerate lane corners: the four points do not form a quadrilateral")
    H, _ = cv2.findHomography(pts.astype(np.float32), dst)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("degenerate lane corners; homography failed")
    return H


def calibrate(corners_px: list[list[float]], out_path: Path, **kw) -> Calibration:
    """The stage function: build the calibration and write <race_id>.calib.json."""
    cal = Calibration(corners_px=corners_px, **kw)
    cal.save(out_path)
    return cal
