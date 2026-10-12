"""Stage 1.3: swimmer detector. Finds every person box in every frame.

Wraps RF-DETR (Apache 2.0) pretrained on COCO `person`. Runs at `detect_fps` on a
downscaled frame and linearly interpolates boxes between runs.

Output: <race_id>.det.parquet with columns frame, x0, y0, x1, y1, score, interpolated.

Status: STUB. Implemented in plan task 1.3 once real clips exist (needs `--extra models`).

Teaching notes
--------------
Detectors trained on COCO see a lot of people but few half-submerged ones; recall drops
mostly on the underwater phases, which we don't need for surface technique. Running at
15 fps and interpolating is 4x cheaper than 60 fps and loses nothing for localization
because the crop has margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DET_COLUMNS = ["frame", "x0", "y0", "x1", "y1", "score", "interpolated"]


@dataclass(frozen=True)
class DetectConfig:
    detect_fps: float = 15.0
    max_side_px: int = 1280  # downscale the long side to this before inference
    score_min: float = 0.3
    model: str = "rfdetr-base"  # resolved by swimtech/models.py in 1.3


def detect(video: Path, out: Path, cfg: DetectConfig = DetectConfig()) -> pd.DataFrame:
    """The stage function. Returns the detections table and writes it as Parquet."""
    raise NotImplementedError("plan task 1.3: swimmer detector (needs real clips)")


def interpolate_boxes(det: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Fill frames between detection runs by linear interpolation per track.

    Pure helper so it can be tested without a model. Implemented alongside 1.3.
    """
    raise NotImplementedError("plan task 1.3")
