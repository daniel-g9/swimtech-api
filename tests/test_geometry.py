import math

import numpy as np
import pytest

from swimtech.metrics import geometry as g


def test_right_angle():
    assert g.angle_deg([0, 1], [0, 0], [1, 0]) == pytest.approx(90.0)


def test_straight_arm_is_180():
    assert g.angle_deg([0, 0], [1, 0], [2, 0]) == pytest.approx(180.0)


def test_folded_arm_is_0():
    assert g.angle_deg([2, 0], [0, 0], [2, 0]) == pytest.approx(0.0)


def test_angle_is_rotation_invariant():
    a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    th = 0.7
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    assert g.angle_deg(rot @ a, rot @ b, rot @ c) == pytest.approx(90.0)


def test_nan_propagates_not_raises():
    assert math.isnan(g.angle_deg([np.nan, 0], [0, 0], [1, 0]))
    assert math.isnan(g.angle_deg([0, 0], [0, 0], [1, 0]))  # zero-length vector


def test_line_angle_ignores_direction():
    assert g.line_angle_deg([1, 0], [-1, 0]) == pytest.approx(0.0)
    assert g.line_angle_deg([1, 0], [0, 1]) == pytest.approx(90.0)
    assert g.line_angle_deg([1, 0], [1, 1]) == pytest.approx(45.0)


def test_height_above_line_positive_away_from_hips():
    # Image coords: y grows downward. Shoulders at y=100, hips below at y=200.
    ls, rs = np.array([0.0, 100.0]), np.array([50.0, 100.0])
    hip_mid = np.array([25.0, 200.0])
    wrist_up = np.array([25.0, 60.0])  # above shoulders on screen = away from hips
    wrist_down = np.array([25.0, 150.0])
    assert g.height_above_line(wrist_up, ls, rs, hip_mid) == pytest.approx(40.0)
    assert g.height_above_line(wrist_down, ls, rs, hip_mid) == pytest.approx(-50.0)


def test_height_above_line_works_when_swimmer_is_upside_down_on_screen():
    # Same body, camera flipped: hips now above the shoulders on screen.
    ls, rs = np.array([0.0, 200.0]), np.array([50.0, 200.0])
    hip_mid = np.array([25.0, 100.0])
    wrist_away_from_hips = np.array([25.0, 240.0])
    assert g.height_above_line(wrist_away_from_hips, ls, rs, hip_mid) == pytest.approx(40.0)


def test_lateral_offset_sign_and_magnitude():
    origin, direction = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert g.lateral_offset([5.0, 2.0], origin, direction) == pytest.approx(2.0)
    assert g.lateral_offset([5.0, -2.0], origin, direction) == pytest.approx(-2.0)
    assert g.lateral_offset([5.0, 0.0], origin, direction) == pytest.approx(0.0)
