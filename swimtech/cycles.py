"""Stage 2.3: cycle and phase segmentation for freestyle.

For each wrist, build the "height above the shoulder line" signal (NOT raw image y; see
docs/metrics.md and open issue 1), band-pass 0.5-2.5 Hz, find peaks with a minimum
distance of 0.5 s. mid-recovery = the peak; entry = the local minimum after a peak that
coincides with the wrist crossing the shoulder line forward; exit = the preceding
minimum near the hip. Breath = head-yaw proxy (ear-to-ear distance ratio) exceeding a
threshold for >= 4 frames. Length boundaries come from the swimmer's direction reversal
in track x.

Output: the `cycles` list for the results JSON.

Status: STUB. Implemented in plan task 2.3. The signal builders below are pure and can
be written and tested before real data arrives.

Teaching notes
--------------
Freestyle wrists trace a near-sinusoid relative to the body; the band-pass keeps stroke
frequencies and drops drift and jitter. Everything here is signal processing, no
learning, which means it is inspectable: when a cycle is missed you can plot the signal
and see why. Other strokes (7.6) reuse this module with different rules: butterfly has
both arms in phase, breaststroke has no recovery peak above the shoulder line, backstroke
is freestyle with the sign flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from swimtech.metrics import geometry as g
from swimtech.types import KP, Cycle


@dataclass(frozen=True)
class CycleConfig:
    band_hz: tuple[float, float] = (0.5, 2.5)
    min_cycle_s: float = 0.5
    breath_yaw_threshold: float = 0.35  # |log(d_L / d_R)| above this = head turned
    breath_min_frames: int = 4


def wrist_height_signal(xy: np.ndarray, arm: str) -> np.ndarray:
    """Per-frame wrist height above the shoulder line, positive away from the hips.
    xy: (n, 13, 2). Returns (n,) with NaN where any input is missing."""
    w = KP.L_WRIST if arm == "L" else KP.R_WRIST
    out = np.full(xy.shape[0], np.nan)
    for i in range(xy.shape[0]):
        hip_mid = g.midpoint(xy[i, KP.L_HIP], xy[i, KP.R_HIP])
        out[i] = g.height_above_line(xy[i, w], xy[i, KP.L_SHOULDER], xy[i, KP.R_SHOULDER], hip_mid)
    return out


def head_yaw_signal(xy: np.ndarray) -> np.ndarray:
    """log(|nose - L ear| / |nose - R ear|) per frame; 0 = facing down, +/- = turned."""
    d_l = np.linalg.norm(xy[:, KP.NOSE] - xy[:, KP.L_EAR], axis=1)
    d_r = np.linalg.norm(xy[:, KP.NOSE] - xy[:, KP.R_EAR], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(d_l / d_r)


def bandpass(x: np.ndarray, fps: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase Butterworth band-pass. NaNs are filled by linear interpolation first
    (the caller should treat those regions as low-visibility)."""
    x = x.astype(float).copy()
    nan = ~np.isfinite(x)
    if nan.all():
        return x
    if nan.any():
        idx = np.arange(len(x))
        x[nan] = np.interp(idx[nan], idx[~nan], x[~nan])
    sos = butter(2, band, btype="band", fs=fps, output="sos")
    return sosfiltfilt(sos, x)


def segment_cycles(
    xy: np.ndarray,
    vis: np.ndarray,
    track_x: np.ndarray,
    fps: float,
    cfg: CycleConfig = CycleConfig(),
) -> list[Cycle]:
    """The stage function: smoothed keypoints -> cycles with entry/midrec/exit/breath frames."""
    raise NotImplementedError("plan task 2.3: cycle segmentation")
