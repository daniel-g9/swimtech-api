"""Point-mode selection, camera-motion compensation, zoom-following crop and the
tracking report (the moving-camera rework of 1.4)."""

import numpy as np
import pandas as pd
import pytest

from swimtech.detect import DET_COLUMNS
from swimtech.motion import identity_rows
from swimtech.track import (
    TRACKED_DETECTED,
    TRACKED_HELD,
    SwimmerSelection,
    TrackConfig,
    crop_boxes,
    crop_to_frame,
    track,
    tracking_report,
)
from tests.test_track import two_lane_detections

FPS = 30.0


def test_point_picks_the_tapped_swimmer():
    det = two_lane_detections(n=60)
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=60)
    assert (t["tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t["cy"], 120)
    t2 = track(det, SwimmerSelection(point=(300, 40)), FPS, n_frames=60)
    assert np.allclose(t2["cy"], 40)


def test_point_reacquires_after_long_gap_and_ignores_other_lane():
    # Lost for 45 frames (> 1 s buffer): the tracker starts a new id; point mode must
    # chain it on and must not jump to the swimmer in the other lane meanwhile.
    det = two_lane_detections(n=120, a_missing=range(30, 75))
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=120)
    assert (t.loc[0:29, "tracked"] == TRACKED_DETECTED).all()
    assert (t.loc[30:74, "tracked"] == TRACKED_HELD).all()
    assert (t.loc[75:119, "tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t["cy"], 120)  # never the y=40 lane


def test_point_far_from_any_swimmer_gives_nothing(caplog):
    det = two_lane_detections(n=30)
    t = track(det, SwimmerSelection(point=(160, 230)), FPS, n_frames=30)
    assert (t["tracked"] == TRACKED_HELD).all()
    assert t["cx"].isna().all()


def _panning_detections(n=40, jump_at=20, jump=60):
    """One swimmer, world x = 20 + 4 f, y = 120. The camera jumps `jump` px right at
    `jump_at`, so on screen the box shifts left by `jump` in one frame."""
    rows = []
    motion = identity_rows(n)
    for f in range(n):
        x = 20 + 4 * f - (jump if f >= jump_at else 0)
        rows.append((f, x, 110, x + 40, 130, 0.9, False))
    motion.loc[jump_at, "tx"] = -jump
    return pd.DataFrame(rows, columns=DET_COLUMNS), motion


def test_camera_jump_is_compensated():
    det, motion = _panning_detections()
    # Point mode would re-chain even if the id broke, so check the tracker itself: in
    # world coordinates the box never moves more than 4 px, so one id must cover all.
    from swimtech.motion import boxes_to_world, cumulative
    from swimtech.track import ByteTracker

    cum = cumulative(motion)
    tr = ByteTracker(FPS, TrackConfig())
    for f in range(40):
        b = det[det.frame == f][["x0", "y0", "x1", "y1"]].to_numpy()
        bw = boxes_to_world(b, cum, np.array([f]))
        tr.update(np.hstack([bw, [[0.9]]]), f)
    assert len(tr.tracks) == 1 and len(tr.tracks[0].frames) == 40

    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=40, motion=motion)
    assert (t["tracked"] == TRACKED_DETECTED).all()
    # Output is back in screen pixels: the jump is visible there.
    assert t.loc[19, "cx"] == pytest.approx(20 + 4 * 19 + 20)
    assert t.loc[20, "cx"] == pytest.approx(20 + 4 * 20 + 20 - 60)


def test_point_given_on_a_later_frame_with_motion():
    det, motion = _panning_detections()
    # Tap at frame 25 in screen pixels (after the jump).
    x = 20 + 4 * 25 - 60 + 20
    t = track(
        det, SwimmerSelection(point=(x, 120), point_frame=25), FPS, n_frames=40, motion=motion
    )
    assert (t["tracked"] == TRACKED_DETECTED).all()


def test_crop_follows_zoom():
    n = 300
    h = np.linspace(20, 60, n)
    df = pd.DataFrame(
        {
            "frame": range(n),
            "cx": 300.0,
            "cy": 200.0,
            "w": h * 2,
            "h": h,
            "score": 0.9,
            "tracked": TRACKED_DETECTED,
        }
    )
    crops = crop_boxes(df, 640, 480, TrackConfig(), fps=30)
    assert crops["w"].iloc[0] < crops["w"].iloc[-1]
    assert crops["w"].iloc[-1] == pytest.approx(156, abs=14)  # longer side 120 * 1.3
    assert (crops["w"] % 2 == 0).all()
    out_side = crops["w"].iloc[-1] / crops["scale"].iloc[-1]
    assert out_side == pytest.approx(np.percentile(crops["w"], 95), abs=2)
    assert np.allclose(crops["w"] / crops["scale"], out_side)
    # A keypoint at the crop-video center maps to the swimmer center on every frame.
    kp = np.full((n, 1, 2), out_side / 2)
    back = crop_to_frame(kp, crops)
    assert np.allclose(back[:, 0, 0], 300, atol=2) and np.allclose(back[:, 0, 1], 200, atol=2)


def test_tracking_report_warnings():
    good = pd.DataFrame(
        {
            "frame": range(300),
            "cx": 1.0,
            "cy": 1.0,
            "w": 100.0,
            "h": 100.0,
            "score": 0.9,
            "tracked": TRACKED_DETECTED,
        }
    )
    r = tracking_report(good, 30)
    assert r["detected_frac"] == 1.0 and r["warnings"] == []
    bad = good.copy()
    bad.loc[30:240, "tracked"] = TRACKED_HELD
    bad["h"] = 40.0
    r = tracking_report(bad, 30)
    assert r["held_frac"] == pytest.approx(211 / 300, abs=0.01)
    assert r["longest_gap_s"] == pytest.approx(7.0, abs=0.1)
    texts = " ".join(r["warnings"])
    assert "only" in texts and "lost for 7 s" in texts and "px tall" in texts


def test_tap_promotes_a_low_confidence_box():
    """The detector scores swimmers low. A box under the tapped point may start a track
    anyway, and the second association pass keeps it alive on low scores."""
    det = two_lane_detections(n=60, a_lowconf=range(0, 60))  # swimmer A never above 0.3
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=60)
    assert (t["tracked"] == TRACKED_DETECTED).mean() > 0.95
    assert np.allclose(t["cy"], 120)


def test_tap_on_a_frame_without_the_swimmer_starts_later():
    det = two_lane_detections(n=90, a_missing=range(0, 15))
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90)
    assert (t.loc[15:89, "tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t["cy"], 120)


def test_chain_rejects_a_different_shape():
    """After losing a flat swimmer box, a tall spectator box nearby must not be chained."""
    rows = []
    for f in range(30):
        x = 20 + 4 * f
        rows.append((f, x, 110, x + 40, 130, 0.9, False))  # 40x20 swimmer
    for f in range(45, 90):
        rows.append((f, 150, 60, 170, 140, 0.9, False))  # 20x80 standing person
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90)
    assert (t.loc[45:89, "tracked"] == TRACKED_HELD).all()


def test_dedupe_boxes():
    from swimtech.detect import dedupe_boxes

    boxes = np.array(
        [[0, 0, 100, 50, 0.3], [2, 1, 101, 52, 0.6], [300, 300, 340, 320, 0.2]], dtype=float
    )
    out = dedupe_boxes(boxes, 0.7)
    assert len(out) == 2 and out[0, 4] == 0.6 and out[1, 4] == 0.2


def test_water_mask_and_fraction():
    import cv2

    from swimtech.motion import water_fraction, water_mask

    img = np.full((240, 320, 3), (90, 90, 90), dtype=np.uint8)  # grey deck
    img[120:, :] = (200, 160, 40)  # cyan-blue water (BGR) in the bottom half
    cv2.rectangle(img, (150, 170), (190, 190), (255, 255, 255), -1)  # white splash hole
    m = water_mask(img)
    assert m.shape[1] == 80
    assert m[: m.shape[0] // 2 - 4].mean() < 0.1 and m[m.shape[0] // 2 + 4 :].mean() > 0.9
    wf = water_fraction(
        np.array([[10, 10, 60, 40], [100, 150, 200, 220], [150, 170, 190, 190]]), m, 320, 240
    )
    assert wf[0] < 0.2 and wf[1] > 0.8 and wf[2] > 0.8  # the splash hole is closed


def test_boxes_off_water_are_ignored():
    """A confident box on the deck must not be tracked or chained, even when it is the
    nearest thing after the swimmer is lost."""
    rows = []
    for f in range(30):
        x = 20 + 4 * f
        rows.append((f, x, 110, x + 40, 130, 0.9, False))  # swimmer, in water
    for f in range(40, 90):
        rows.append((f, 150, 20, 190, 40, 0.9, False))  # same shape, on the deck
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    water = np.zeros((90, 24, 32), dtype=np.uint8)
    water[:, 8:, :] = 1  # water is the bottom two thirds of a 320x240 frame
    t = track(
        det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90, water=water, frame_size=(320, 240)
    )
    assert (t.loc[0:29, "tracked"] == TRACKED_DETECTED).all()
    assert (t.loc[40:89, "tracked"] == TRACKED_HELD).all()


def test_chain_prefers_own_lane_over_nearer_lane_mate():
    """After a loss, a lane-mate that reappears sooner and closer must lose to the
    continuation in the swimmer's own lane."""
    rows = []
    for f in range(40):  # our swimmer, lane y=120, moving right
        x = 20 + 4 * f
        rows.append((f, x, 110, x + 40, 130, 0.9, False))
    for f in range(50, 90):  # lane-mate, one lane up (y=85), near where we vanished
        rows.append((f, 190, 75, 230, 95, 0.9, False))
    for f in range(70, 90):  # our swimmer again, same lane, further along
        x = 20 + 4 * f
        rows.append((f, x, 110, x + 40, 130, 0.9, False))
    det = pd.DataFrame(rows, columns=DET_COLUMNS)
    t = track(det, SwimmerSelection(point=(40, 120)), FPS, n_frames=90)
    assert (t.loc[70:89, "tracked"] == TRACKED_DETECTED).all()
    assert np.allclose(t.loc[70:89, "cy"], 120)
    # The 30-frame gap is bridged, but never through the lane-mate's position.
    assert (t.loc[40:69, "tracked"] != TRACKED_DETECTED).all()
    assert np.allclose(t["cy"], 120)
