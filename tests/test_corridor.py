"""Per-frame lane corridor: finding the ropes, following them, and filtering boxes."""

import cv2
import numpy as np
import pandas as pd

from swimtech.detect import DET_COLUMNS
from swimtech.lanes import (
    LANE_COLUMNS,
    Corridor,
    corridor_at_point,
    corridor_for_frame,
    ropes_in_frame,
    track_corridor,
)
from swimtech.motion import identity_rows
from swimtech.track import TRACKED_DETECTED, SwimmerSelection, track
from tests.conftest import write_video

FPS = 30.0


def rope_frame(w=640, h=480, slope=0.1, offsets=(120, 220, 320, 420), shift=0.0):
    img = np.full((h, w, 3), (200, 160, 40), dtype=np.uint8)  # water
    for y in offsets:
        y0 = int(y + shift)
        cv2.line(img, (0, y0), (w, int(y0 - slope * w)), (30, 30, 220), 4)
    return img


def test_ropes_in_frame_finds_each_rope():
    det = ropes_in_frame(rope_frame(), None)
    assert det is not None
    phi, rhos, support = det
    assert len(rhos) == 4
    # Offsets are perpendicular distances, roughly the y-intercepts times cos(angle).
    assert np.all(np.diff(rhos) > 80)
    c = corridor_at_point(phi, rhos, support, np.array([320.0, 240.0]))
    assert c is not None
    assert 80 < c.width < 120
    assert c.contains(np.array([[320.0, 240.0]]))[0]
    assert not c.contains(np.array([[320.0, 100.0]]))[0]


def test_corridor_follows_a_panning_camera(tmp_path):
    # Ropes drift 2 px down per frame (camera tilting up); the corridor must follow.
    frames = [rope_frame(shift=2 * f) for f in range(30)]
    video = write_video(tmp_path / "ropes.mp4", frames, FPS)
    motion = identity_rows(30)
    motion.loc[1:, "ty"] = 2.0
    lane = track_corridor(video, None, motion, np.array([320.0, 240.0]), 0)
    assert list(lane.columns) == LANE_COLUMNS and len(lane) == 30
    assert (lane["snapped"] > 0).mean() > 0.8
    c0, c29 = corridor_for_frame(lane, 0), corridor_for_frame(lane, 29)
    assert c29.rho_above - c0.rho_above > 40  # moved with the ropes (58 px expected)
    assert c29.contains(np.array([[320.0, 240.0 + 58]]))[0]
    assert not c29.contains(np.array([[320.0, 240.0 - 60]]))[0]


def test_lane_filter_keeps_only_the_corridor():
    """Two swimmers; the lane-mate is closer after a loss but outside the corridor."""
    rows = [(f, 20 + 4 * f, 110, 60 + 4 * f, 130, 0.9, False) for f in range(30)]
    rows += [(f, 150, 60, 190, 80, 0.9, False) for f in range(40, 90)]  # lane-mate (y 70)
    rows += [(f, 250, 110, 290, 130, 0.9, False) for f in range(60, 90)]  # ours (y 120)
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    # Horizontal ropes at y = 100 and y = 140 in every frame: phi = 90 deg, rho = y.
    lane = pd.DataFrame(
        {"frame": range(90), "phi": np.pi / 2, "rho_above": 100.0, "rho_below": 140.0, "snapped": 2}
    )
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90, lane=lane)
    # The lane-mate (y 70) never enters the tracker: the gap is bridged along our lane.
    assert (t.loc[40:59, "tracked"] != TRACKED_DETECTED).all()
    assert (t.loc[60:89, "tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t["cy"], 120)


def test_corridor_helpers():
    c = Corridor(np.pi / 2, 100.0, 140.0)
    assert c.width == 40
    assert np.allclose(c.offset(np.array([[0.0, 120.0]])), 0.5)
    assert corridor_for_frame(pd.DataFrame(columns=LANE_COLUMNS), 3) is None


def test_giant_box_in_corridor_is_not_a_swimmer():
    """A lifeguard chair or a head in front of the lens makes a box far taller than the
    lane; it must not become the swimmer even when its center lies in the corridor."""
    rows = [(f, 20 + 4 * f, 110, 60 + 4 * f, 130, 0.9, False) for f in range(30)]
    rows += [(f, 100, 20, 300, 220, 0.9, False) for f in range(31, 90)]  # 200x200 monster
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    lane = pd.DataFrame(
        {"frame": range(90), "phi": np.pi / 2, "rho_above": 100.0, "rho_below": 140.0, "snapped": 2}
    )
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90, lane=lane)
    assert (t.loc[31:89, "tracked"] != TRACKED_DETECTED).all()
