"""Stage 2.1: pose inference. 13 keypoints per frame with model confidence.

Runs RTMPose-l (MMPose, Apache 2.0) exported to ONNX on each crop frame; maps COCO-17
to our 13-point order (types.COCO17_TO_SWIM13); converts crop -> normalized-frame
coordinates with the crop offsets. Fallback behind a flag: ViTPose-B.

Output: <race_id>.kp.raw.parquet with columns frame, then for each keypoint name
<name>_x, <name>_y, <name>_conf.

Status: STUB. Implemented in plan task 2.1 (needs `--extra models`).

Open issue 12: a 1.6x crop often includes a lane-mate's arm. When the model returns
more than one instance, pick the one whose hip center is closest to the crop center,
not the highest-scoring one.

Teaching notes
--------------
Top-down pose = detect the person, crop, estimate keypoints in the crop. RTMPose is a
SimCC-style model: it predicts x and y as separate 1-D classifications, which is fast and
robust to blur. Model confidence is NOT visibility; vis is derived in 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swimtech.types import KEYPOINT_NAMES


@dataclass(frozen=True)
class PoseConfig:
    model: str = "rtmpose-l"  # or "vitpose-b"
    onnx_path: Path | None = None  # resolved from models/ when None
    input_size: tuple[int, int] = (288, 384)  # RTMPose-l default (w, h)


def pose(
    crop_video: Path, crops: pd.DataFrame, out: Path, cfg: PoseConfig = PoseConfig()
) -> pd.DataFrame:
    """The stage function. Returns the raw keypoints table and writes it as Parquet."""
    raise NotImplementedError("plan task 2.1: pose inference")


def raw_table_to_array(df: pd.DataFrame) -> np.ndarray:
    """Parquet table -> (n_frames, 13, 3) array of x, y, conf, ordered by frame."""
    df = df.sort_values("frame")
    cols = []
    for name in KEYPOINT_NAMES:
        cols.append(df[[f"{name}_x", f"{name}_y", f"{name}_conf"]].to_numpy(dtype=float))
    return np.stack(cols, axis=1)


def array_to_raw_table(arr: np.ndarray, frames: np.ndarray | None = None) -> pd.DataFrame:
    """(n, 13, 3) -> the Parquet table layout. Inverse of raw_table_to_array."""
    n = arr.shape[0]
    data: dict[str, np.ndarray] = {"frame": np.arange(n) if frames is None else frames}
    for j, name in enumerate(KEYPOINT_NAMES):
        data[f"{name}_x"] = arr[:, j, 0]
        data[f"{name}_y"] = arr[:, j, 1]
        data[f"{name}_conf"] = arr[:, j, 2]
    return pd.DataFrame(data)
