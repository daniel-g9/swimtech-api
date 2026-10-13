"""Lane geometry from the picture itself: which way the lane ropes run.

The tracker (1.4) needs to tell "same lane" from "next lane" when it re-acquires a
swimmer after a loss, because lane-mates swim parallel at the same speed and are
otherwise indistinguishable. A line through the tapped swimmer along the ropes is that
lane, and the user never has to draw it.

Teaching notes
--------------
Lane ropes are the longest straight edges on the water, so the length-weighted median
orientation of Hough line segments inside the water mask is the lane direction.
Perspective makes far lanes converge, so one direction is approximate; the tracker only
needs it near the swimmer, and switches to the swimmer's own path once they have moved.
"""

from __future__ import annotations

import cv2
import numpy as np


def lane_direction(bgr: np.ndarray, water: np.ndarray | None = None) -> np.ndarray | None:
    """Unit vector along the lane ropes in this frame, or None if no ropes are found."""
    h = max(int(round(640 * bgr.shape[0] / bgr.shape[1])), 8)
    small = cv2.resize(bgr, (640, h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    if water is not None:
        m = cv2.resize(
            water.astype(np.uint8) * 255,
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        edges[m == 0] = 0
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=small.shape[1] // 5, maxLineGap=12
    )
    if lines is None or len(lines) < 3:
        return None
    seg = lines.reshape(-1, 4).astype(float)
    d = seg[:, 2:] - seg[:, :2]
    length = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ang = np.where(ang > 90, ang - 180, np.where(ang < -90, ang + 180, ang))
    order = np.argsort(ang)
    cum = np.cumsum(length[order])
    med = float(ang[order][np.searchsorted(cum, cum[-1] / 2)])
    return np.array([np.cos(np.radians(med)), np.sin(np.radians(med))])


def angle_deg(direction: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(direction[1], direction[0])))
