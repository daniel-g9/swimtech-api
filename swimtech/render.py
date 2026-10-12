"""Stage 2.6: annotated video renderer. The overlay video the app plays back and shares.

Draws the skeleton (grey when vis < 2), the three angle readouts at each sampled
instant, cycle counters and finding markers; encodes at the source resolution by piping
frames to ffmpeg; also writes a thumbnail.

Status: STUB. Implemented in plan task 2.6. `draw_frame` is pure (frame in, frame out)
so it can be unit-tested on a blank image, and so the Vue overlay (4.3) can mirror it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from swimtech.types import SKELETON, VIS_VISIBLE, Cycle, Finding

COLOR_VISIBLE = (40, 220, 90)  # BGR
COLOR_OCCLUDED = (160, 160, 160)
COLOR_FINDING = (30, 60, 240)


def draw_frame(
    frame: np.ndarray,
    xy: np.ndarray,
    vis: np.ndarray,
    labels: list[tuple[str, tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Draw one frame's skeleton on a BGR image. Pure: returns a new array."""
    import cv2

    out = frame.copy()
    for a, b in SKELETON:
        pa, pb = xy[a], xy[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        color = (
            COLOR_VISIBLE if (vis[a] == VIS_VISIBLE and vis[b] == VIS_VISIBLE) else COLOR_OCCLUDED
        )
        cv2.line(out, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 3, cv2.LINE_AA)
    for j in range(xy.shape[0]):
        p = xy[j]
        if np.isfinite(p).all():
            color = COLOR_VISIBLE if vis[j] == VIS_VISIBLE else COLOR_OCCLUDED
            cv2.circle(out, tuple(p.astype(int)), 5, color, -1, cv2.LINE_AA)
    for text, (x, y) in labels or []:
        cv2.putText(
            out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
    return out


def render(
    norm_video: Path,
    xy: np.ndarray,
    vis: np.ndarray,
    cycles: list[Cycle],
    findings: list[Finding],
    out_video: Path,
    out_thumb: Path,
) -> None:
    """The stage function."""
    raise NotImplementedError("plan task 2.6: renderer")
