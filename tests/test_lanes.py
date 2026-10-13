"""Lane-rope direction and the lane prior in point-mode selection."""

import cv2
import numpy as np
import pandas as pd
import pytest

from swimtech.detect import DET_COLUMNS
from swimtech.lanes import angle_deg, lane_direction
from swimtech.track import TRACKED_DETECTED, SwimmerSelection, track

FPS = 30.0


def test_lane_direction_from_ropes():
    img = np.full((480, 640, 3), (200, 160, 40), dtype=np.uint8)  # water
    drop = int(640 * np.tan(np.radians(10)))
    for y in range(120, 480, 90):  # ropes rising 10 degrees to the right
        cv2.line(img, (0, y), (640, y - drop), (30, 30, 220), 4)
    d = lane_direction(img)
    assert d is not None
    assert angle_deg(d) == pytest.approx(-10, abs=1.5)


def test_no_ropes_gives_none():
    assert lane_direction(np.full((240, 320, 3), 120, dtype=np.uint8)) is None


def test_lane_prior_constrains_the_first_hop():
    """Lost before moving far: without a prior the nearest lane-mate wins; with the
    rope direction the same-lane continuation wins even though it appears later."""
    rows = [(f, 20 + 4 * f, 110, 60 + 4 * f, 130, 0.9, False) for f in range(6)]
    rows += [(f, 60, 75, 100, 95, 0.9, False) for f in range(20, 60)]  # lane-mate, lane up
    rows += [(f, 140, 110, 180, 130, 0.9, False) for f in range(40, 90)]  # our lane
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    t = track(
        det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90, lane_dir=np.array([1.0, 0.0])
    )
    assert (t.loc[40:89, "tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t.loc[40:89, "cy"], 120)
