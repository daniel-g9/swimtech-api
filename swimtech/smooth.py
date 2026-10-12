"""Stage 2.2: temporal smoothing, gap bridging, and visibility.

Input : (n_frames, 13, 3) array of x, y, conf from pose (2.1)
Output: (n_frames, 13, 2) smoothed x, y   and   (n_frames, 13) int8 vis

Teaching notes
--------------
Savitzky-Golay fits a local polynomial (here order 3 over 9 frames = 0.15 s at 60 fps)
and takes its value at the center. Unlike a moving average it keeps real peaks, so the
wrist's turnaround at mid-recovery and exit stays where it was; a moving average would
flatten it and shift every angle sample a few frames late.

Gaps: frames with conf < 0.3 are treated as missing. Gaps of up to `max_gap` frames
(12 = 0.2 s) are bridged by linear interpolation and marked vis = 1 (occluded). Longer
gaps stay NaN with vis = 0: a wrist hidden for longer than that is genuinely unknown and
the app should say so rather than draw it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from swimtech.types import VIS_OCCLUDED, VIS_OUT_OF_FRAME, VIS_VISIBLE


@dataclass(frozen=True)
class SmoothConfig:
    window: int = 9  # frames, must be odd
    order: int = 3
    conf_min: float = 0.3  # below this the detection is treated as missing
    conf_visible: float = 0.5  # at or above this (and not interpolated) -> vis = 2
    max_gap: int = 12  # frames; longer gaps stay NaN


def _fill_short_gaps(
    x: np.ndarray, missing: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate runs of `missing` no longer than max_gap.

    Returns the filled signal and a boolean mask of which samples were interpolated.
    Leading and trailing gaps are never filled (there is nothing to interpolate from).
    """
    x = x.astype(float).copy()
    filled = np.zeros_like(missing, dtype=bool)
    n = len(x)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        gap = j - i
        if 0 < i and j < n and gap <= max_gap:
            x[i:j] = np.linspace(x[i - 1], x[j], gap + 2)[1:-1]
            filled[i:j] = True
        else:
            x[i:j] = np.nan
        i = j
    return x, filled


def _savgol_nan_aware(x: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay over each contiguous non-NaN run; runs shorter than the window
    are left as they are."""
    out = x.copy()
    n = len(x)
    valid = np.isfinite(x)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        if j - i >= window:
            out[i:j] = savgol_filter(x[i:j], window, order, mode="interp")
        i = j
    return out


def smooth_keypoints(
    raw: np.ndarray, cfg: SmoothConfig = SmoothConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """The stage function.

    raw: (n, 13, 3) float array of x, y, conf. NaN x/y or conf < conf_min means missing.
    Returns (xy, vis): xy (n, 13, 2) float with NaN where unknown; vis (n, 13) int8.
    """
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"expected (n, k, 3), got {raw.shape}")
    if cfg.window % 2 == 0 or cfg.order >= cfg.window:
        raise ValueError("window must be odd and larger than order")

    n, k, _ = raw.shape
    xy = np.full((n, k, 2), np.nan)
    vis = np.full((n, k), VIS_OUT_OF_FRAME, dtype=np.int8)

    conf = raw[:, :, 2]
    missing = (~np.isfinite(raw[:, :, 0])) | (~np.isfinite(raw[:, :, 1])) | (conf < cfg.conf_min)

    for j in range(k):
        xs, fx = _fill_short_gaps(raw[:, j, 0], missing[:, j], cfg.max_gap)
        ys, fy = _fill_short_gaps(raw[:, j, 1], missing[:, j], cfg.max_gap)
        interpolated = fx | fy
        xs = _savgol_nan_aware(xs, cfg.window, cfg.order)
        ys = _savgol_nan_aware(ys, cfg.window, cfg.order)
        xy[:, j, 0] = xs
        xy[:, j, 1] = ys

        known = np.isfinite(xs) & np.isfinite(ys)
        v = np.full(n, VIS_OUT_OF_FRAME, dtype=np.int8)
        v[known] = VIS_OCCLUDED
        v[known & ~interpolated & (conf[:, j] >= cfg.conf_visible)] = VIS_VISIBLE
        vis[:, j] = v

    return xy, vis
