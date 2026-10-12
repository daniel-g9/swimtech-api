"""Shared helpers: synthetic videos and a fake detector, so Stage 1 is tested without
real footage, a GPU, or torch.

The "swimmer" is a bright rectangle on a dark pool; the fake detector thresholds it.
That exercises every line of detect/track except the model call itself.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> Path:
    """Write BGR frames with OpenCV's built-in mp4v encoder (no ffmpeg needed)."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter"
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def blob_frames(
    n: int, w: int, h: int, boxes_at: Callable[[int], list[Box]], bg: int = 30
) -> list[np.ndarray]:
    """Dark frames with one bright rectangle per box returned by boxes_at(frame)."""
    frames = []
    for f in range(n):
        img = np.full((h, w, 3), bg, dtype=np.uint8)
        for x0, y0, x1, y1 in boxes_at(f):
            img[y0:y1, x0:x1] = 255
        frames.append(img)
    return frames


def blob_detector(frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
    """A `Detector`: every bright connected blob is a 'person' with score 0.9."""
    out = []
    for rgb in frames_rgb:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh < 16:
                continue
            rows.append([x, y, x + bw, y + bh, 0.9])
        out.append(np.asarray(rows, dtype=float).reshape(-1, 5))
    return out


@pytest.fixture
def moving_blob_video(tmp_path):
    """60 frames, 320x240 @ 30 fps; a 40x20 'swimmer' moving right along y=120,
    and a second one in another lane (y=40) moving left."""

    def boxes(f: int) -> list[Box]:
        x = 20 + 4 * f
        return [(x, 110, x + 40, 130), (280 - 4 * f, 30, 320 - 4 * f, 50)]

    path = write_video(tmp_path / "blob.mp4", blob_frames(60, 320, 240, boxes))
    return path, boxes
