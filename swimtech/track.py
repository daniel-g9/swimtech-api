"""Stage 1.4: lane tracker and crop. Follow one chosen swimmer and produce a steady crop.

ByteTrack over the detections; keep only the track inside the chosen lane; fill gaps up
to 1 s by linear interpolation; smooth the box center with a 15-frame moving average;
fix the crop size to the 95th-percentile box height x 1.6.

Output: <race_id>.crop.mp4 at the normalized fps and <race_id>.crop.parquet with
columns frame, x0, y0, w, h so crop coordinates map back to the normalized frame.

Status: STUB. Implemented in plan task 1.4.

Teaching notes
--------------
ByteTrack keeps low-confidence boxes as a second association pass, which is exactly what
survives splash. The lane constraint turns a hard multi-object problem into "one swimmer
in a corridor", so ID swaps become nearly impossible. A fixed crop size matters: pose
models are sensitive to scale jitter, and a crop that breathes with the box makes the
skeleton jitter too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CROP_COLUMNS = ["frame", "x0", "y0", "w", "h"]


@dataclass(frozen=True)
class LaneSelection:
    """Either a box drawn on the first frame (x0, y0, x1, y1 in normalized pixels) or a
    lane number resolved through the calibration."""

    box: tuple[float, float, float, float] | None = None
    lane_number: int | None = None


@dataclass(frozen=True)
class TrackConfig:
    max_gap_s: float = 1.0
    center_smooth_frames: int = 15
    crop_scale: float = 1.6
    crop_height_percentile: float = 95.0


def track(
    det: pd.DataFrame, lane: LaneSelection, fps: float, cfg: TrackConfig = TrackConfig()
) -> pd.DataFrame:
    """Detections -> one track (frame, cx, cy, h) for the chosen swimmer."""
    raise NotImplementedError("plan task 1.4")


def crop_boxes(
    track_df: pd.DataFrame, frame_w: int, frame_h: int, cfg: TrackConfig = TrackConfig()
) -> pd.DataFrame:
    """Track -> fixed-size crop boxes clamped to the frame. Pure; testable without video."""
    raise NotImplementedError("plan task 1.4")


def write_crop_video(norm_video: Path, crops: pd.DataFrame, out: Path) -> None:
    raise NotImplementedError("plan task 1.4")


def crop_to_frame(xy_crop: np.ndarray, crops: pd.DataFrame) -> np.ndarray:
    """Map (n, k, 2) crop-space keypoints back to normalized-frame pixels."""
    off = crops[["x0", "y0"]].to_numpy(dtype=float)[:, None, :]
    return xy_crop + off
