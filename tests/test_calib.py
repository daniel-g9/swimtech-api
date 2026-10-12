import math

import numpy as np
import pytest

from swimtech.calib import Calibration, calibrate

# A lane seen obliquely: a trapezoid in the image. Order: near-left, near-right,
# far-right, far-left.
CORNERS = [[100.0, 900.0], [700.0, 900.0], [600.0, 200.0], [200.0, 200.0]]


def test_corners_map_to_lane_rectangle():
    cal = Calibration(corners_px=CORNERS, lane_width_m=2.5, pool_length_m=25.0)
    assert cal.to_rect(100, 900) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert cal.to_rect(700, 900) == pytest.approx((0.0, 2.5), abs=1e-6)
    assert cal.to_rect(600, 200) == pytest.approx((25.0, 2.5), abs=1e-6)
    assert cal.to_rect(200, 200) == pytest.approx((25.0, 0.0), abs=1e-6)


def test_lane_center_at_near_wall_is_mid_width():
    cal = Calibration(corners_px=CORNERS)
    x, y = cal.to_rect(400, 900)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(1.25, abs=1e-6)


def test_vectorized_matches_scalar_and_propagates_nan():
    cal = Calibration(corners_px=CORNERS)
    pts = np.array([[400.0, 900.0], [np.nan, 500.0], [400.0, 200.0]])
    out = cal.to_rect_array(pts)
    assert out[0] == pytest.approx(cal.to_rect(400, 900))
    assert math.isnan(out[1, 0]) and math.isnan(out[1, 1])
    assert out[2] == pytest.approx(cal.to_rect(400, 200))


def test_round_trip_through_json(tmp_path):
    path = tmp_path / "x.calib.json"
    cal = calibrate(CORNERS, path, lane_width_m=2.13, source="race_first_frame")
    loaded = Calibration.load(path)
    assert loaded.lane_width_m == 2.13
    assert loaded.source == "race_first_frame"
    assert loaded.to_rect(400, 500) == pytest.approx(cal.to_rect(400, 500))


def test_degenerate_corners_rejected():
    with pytest.raises(ValueError):
        Calibration(corners_px=[[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        Calibration(corners_px=CORNERS[:3])
