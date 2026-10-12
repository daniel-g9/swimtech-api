import numpy as np

from swimtech.smooth import SmoothConfig, smooth_keypoints
from swimtech.types import N_KEYPOINTS, VIS_OCCLUDED, VIS_OUT_OF_FRAME, VIS_VISIBLE


def _sine_track(n=240, fps=60.0, hz=1.2, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    raw = np.zeros((n, N_KEYPOINTS, 3))
    for j in range(N_KEYPOINTS):
        raw[:, j, 0] = 500 + 10 * j + rng.normal(0, noise, n)
        raw[:, j, 1] = 300 + 80 * np.sin(2 * np.pi * hz * t + j) + rng.normal(0, noise, n)
        raw[:, j, 2] = 0.9
    return raw, t


def test_clean_signal_passes_through_and_is_visible():
    raw, _ = _sine_track()
    xy, vis = smooth_keypoints(raw)
    assert xy.shape == (240, N_KEYPOINTS, 2)
    assert (vis == VIS_VISIBLE).all()
    np.testing.assert_allclose(xy[:, :, 1], raw[:, :, 1], atol=1.0)


def test_smoothing_reduces_noise_but_keeps_peaks():
    raw, t = _sine_track(noise=4.0)
    clean = 300 + 80 * np.sin(2 * np.pi * 1.2 * t)
    xy, _ = smooth_keypoints(raw)
    err_raw = np.abs(raw[:, 0, 1] - clean).mean()
    err_smooth = np.abs(xy[:, 0, 1] - clean).mean()
    assert err_smooth < err_raw * 0.7
    # Peak amplitude survives (a moving average would shrink it)
    assert xy[:, 0, 1].max() > 300 + 80 * 0.95


def test_short_gap_is_interpolated_and_marked_occluded():
    raw, _ = _sine_track()
    raw[100:106, 7, 2] = 0.1  # 6 frames of low confidence on the L wrist
    xy, vis = smooth_keypoints(raw)
    assert np.isfinite(xy[100:106, 7]).all()
    assert (vis[100:106, 7] == VIS_OCCLUDED).all()
    assert vis[99, 7] == VIS_VISIBLE and vis[106, 7] == VIS_VISIBLE


def test_long_gap_stays_nan():
    raw, _ = _sine_track()
    raw[100:130, 8, 0] = np.nan  # 30 frames missing, above max_gap = 12
    xy, vis = smooth_keypoints(raw)
    assert np.isnan(xy[100:130, 8]).all()
    assert (vis[100:130, 8] == VIS_OUT_OF_FRAME).all()
    assert np.isfinite(xy[99, 8]).all() and np.isfinite(xy[130, 8]).all()


def test_mid_confidence_is_kept_but_not_visible():
    raw, _ = _sine_track()
    raw[50:60, 3, 2] = 0.4  # between conf_min (0.3) and conf_visible (0.5)
    xy, vis = smooth_keypoints(raw)
    assert np.isfinite(xy[50:60, 3]).all()
    assert (vis[50:60, 3] == VIS_OCCLUDED).all()


def test_leading_gap_is_not_filled():
    raw, _ = _sine_track()
    raw[:5, 0, 2] = 0.0
    xy, vis = smooth_keypoints(raw)
    assert np.isnan(xy[:5, 0]).all()
    assert (vis[:5, 0] == VIS_OUT_OF_FRAME).all()


def test_bad_config_rejected():
    raw, _ = _sine_track()
    try:
        smooth_keypoints(raw, SmoothConfig(window=8))
    except ValueError:
        return
    raise AssertionError("even window should be rejected")
