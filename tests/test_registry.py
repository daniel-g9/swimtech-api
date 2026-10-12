import math

import numpy as np
import pytest

from swimtech.metrics.registry import REGISTRY, MetricContext
from swimtech.types import KP, N_KEYPOINTS

EXPECTED = {
    "elbow_angle_exit",
    "elbow_angle_midrec",
    "elbow_angle_entry",
    "wrist_height_midrec",
    "hand_entry_offset_rect",
    "forearm_entry_angle",
    "head_angle",
    "head_lift_breath",
    "shoulder_line_angle",
    "shoulder_foreshortening",
    "hip_shoulder_offset_rect",
    "breath_side",
    "breath_frequency",
    "breath_cycle_penalty",
    "stroke_rate",
    "cycle_cv",
}


def test_registry_matches_spec():
    assert set(REGISTRY) == EXPECTED


def test_rect_metrics_are_named_rect():
    for spec in REGISTRY.values():
        assert (spec.space == "rect") == spec.name.endswith("_rect"), spec.name


def test_per_arm_specs_have_right_side_keypoints():
    for spec in REGISTRY.values():
        if spec.per_arm:
            assert spec.keypoints_r, spec.name
            assert spec.keypoints("L") != spec.keypoints("R"), spec.name
        else:
            assert spec.keypoints("L") == spec.keypoints("R") == spec.keypoints(None)


def _upright_body():
    """A swimmer seen from above, shoulders horizontal, hips below, L arm bent 90 deg."""
    kp = np.full((N_KEYPOINTS, 2), np.nan)
    kp[KP.NOSE] = [50, 60]
    kp[KP.L_EAR] = [40, 70]
    kp[KP.R_EAR] = [60, 70]
    kp[KP.L_SHOULDER] = [20, 100]
    kp[KP.R_SHOULDER] = [80, 100]
    kp[KP.L_ELBOW] = [0, 100]
    kp[KP.L_WRIST] = [0, 60]
    kp[KP.R_ELBOW] = [100, 100]
    kp[KP.R_WRIST] = [140, 100]
    kp[KP.L_HIP] = [30, 200]
    kp[KP.R_HIP] = [70, 200]
    kp[KP.L_KNEE] = [30, 260]
    kp[KP.R_KNEE] = [70, 260]
    return kp


def test_elbow_angle_left_and_right():
    kp = _upright_body()
    spec = REGISTRY["elbow_angle_exit"]
    assert spec.fn(kp, MetricContext(arm="L")) == pytest.approx(90.0)
    assert spec.fn(kp, MetricContext(arm="R")) == pytest.approx(180.0)


def test_wrist_height_in_shoulder_widths():
    kp = _upright_body()
    spec = REGISTRY["wrist_height_midrec"]
    # L wrist is 40 px above the shoulder line; shoulder width is 60 px.
    assert spec.fn(kp, MetricContext(arm="L")) == pytest.approx(40 / 60)
    # With an explicit reference width the normalization uses it instead.
    assert spec.fn(kp, MetricContext(arm="L", shoulder_width_ref=80)) == pytest.approx(0.5)
    # R wrist is on the shoulder line.
    assert spec.fn(kp, MetricContext(arm="R")) == pytest.approx(0.0)


def test_head_angle_facing_forward_is_perpendicular_to_shoulders():
    kp = _upright_body()
    assert REGISTRY["head_angle"].fn(kp, MetricContext()) == pytest.approx(90.0)


def test_shoulder_line_angle_needs_lane_dir():
    kp = _upright_body()
    spec = REGISTRY["shoulder_line_angle"]
    assert math.isnan(spec.fn(kp, MetricContext()))
    assert spec.fn(kp, MetricContext(lane_dir=np.array([0.0, 1.0]))) == pytest.approx(90.0)


def test_missing_keypoint_gives_nan_not_error():
    kp = _upright_body()
    kp[KP.L_WRIST] = [np.nan, np.nan]
    assert math.isnan(REGISTRY["elbow_angle_exit"].fn(kp, MetricContext(arm="L")))


def test_unimplemented_metrics_say_which_task():
    kp = _upright_body()
    with pytest.raises(NotImplementedError, match="2.4"):
        REGISTRY["hand_entry_offset_rect"].fn(kp, MetricContext(arm="L"))
