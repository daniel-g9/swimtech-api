import cv2
import numpy as np
import pandas as pd
import pytest

from swimtech.detect import DET_COLUMNS
from swimtech.track import (
    CROP_COLUMNS,
    TRACKED_DETECTED,
    TRACKED_HELD,
    TRACKED_INTERP,
    ByteTracker,
    LaneSelection,
    TrackConfig,
    crop_boxes,
    crop_to_frame,
    fill_gaps,
    points_in_polygon,
    select_lane_tracks,
    track,
    write_crop_video,
)
from tests.conftest import blob_frames, write_video

FPS = 30.0


def two_lane_detections(n=60, a_missing=(), a_lowconf=(), b_missing=()) -> pd.DataFrame:
    """Swimmer A in lane y~120 moving right, swimmer B in lane y~40 moving left."""
    rows = []
    for f in range(n):
        if f not in a_missing:
            x = 20 + 4 * f
            rows.append((f, x, 110, x + 40, 130, 0.3 if f in a_lowconf else 0.9, False))
        if f not in b_missing:
            x = 280 - 4 * f
            rows.append((f, x, 30, x + 40, 50, 0.9, False))
    return pd.DataFrame(rows, columns=DET_COLUMNS)


def _run_tracker(det: pd.DataFrame, cfg=TrackConfig()) -> ByteTracker:
    tr = ByteTracker(FPS, cfg)
    for f, g in det.groupby("frame"):
        tr.update(g[["x0", "y0", "x1", "y1", "score"]].to_numpy(), int(f))
    return tr


def test_two_swimmers_get_two_stable_ids():
    tr = _run_tracker(two_lane_detections())
    assert len(tr.tracks) == 2
    assert sorted(len(t.frames) for t in tr.tracks) == [60, 60]


def test_low_confidence_boxes_keep_the_track_alive():
    """ByteTrack's second pass: the splashy frames (score 0.3) still attach to the track
    instead of being dropped, so there is no gap and no new id."""
    tr = _run_tracker(two_lane_detections(a_lowconf=range(20, 30)))
    assert len(tr.tracks) == 2
    a = max(tr.tracks, key=lambda t: t.boxes[0][1])  # larger y = lane A
    assert len(a.frames) == 60
    assert set(range(20, 30)) <= set(a.frames)


def test_short_occlusion_bridges_without_new_id():
    tr = _run_tracker(two_lane_detections(a_missing=range(20, 26)))
    assert len(tr.tracks) == 2
    a = max(tr.tracks, key=lambda t: t.boxes[0][1])
    assert len(a.frames) == 54 and 19 in a.frames and 26 in a.frames


def test_points_in_polygon():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    pts = np.array([[5, 5], [15, 5], [0.1, 9.9], [-1, -1]])
    assert points_in_polygon(pts, poly).tolist() == [True, False, True, False]


def test_select_lane_picks_the_track_in_the_corridor():
    tr = _run_tracker(two_lane_detections())
    lane_a = np.array([[0, 100], [320, 100], [320, 140], [0, 140]])
    chosen = select_lane_tracks(tr.tracks, lane_a)
    assert len(chosen) == 1
    assert chosen[0].boxes[0][1] == pytest.approx(110)
    # No corridor: longest-lived wins (both equal here, so just one is returned).
    assert len(select_lane_tracks(tr.tracks, None)) == 1
    empty_lane = np.array([[0, 200], [320, 200], [320, 240], [0, 240]])
    assert select_lane_tracks(tr.tracks, empty_lane) == []


def test_select_lane_merges_reacquired_track_after_long_gap():
    # Lost for 45 frames (> 1 s buffer at 30 fps): the tracker starts a new id; the lane
    # selection should glue the two pieces together because they don't overlap in time.
    det = two_lane_detections(n=120, a_missing=range(30, 75))
    tr = _run_tracker(det)
    lane_a = np.array([[0, 100], [600, 100], [600, 140], [0, 140]])
    chosen = select_lane_tracks(tr.tracks, lane_a)
    assert len(chosen) == 2
    assert chosen[0].frames[0] == 0 and chosen[1].frames[0] == 75


def test_track_fills_short_gaps_and_holds_long_ones():
    det = two_lane_detections(n=120, a_missing=list(range(20, 26)) + list(range(60, 100)))
    t = track(det, LaneSelection(box=(0, 100, 600, 140)), FPS, n_frames=120)
    assert len(t) == 120 and t["frame"].tolist() == list(range(120))
    assert (t.loc[20:25, "tracked"] == TRACKED_INTERP).all()
    assert t.loc[22, "cx"] == pytest.approx((20 + 4 * 22) + 20, abs=0.5)  # linear
    assert (t.loc[60:99, "tracked"] == TRACKED_HELD).all()
    assert (t.loc[60:99, "cx"] == t.loc[59, "cx"]).all()  # held, not extrapolated
    assert (t.loc[60:99, "score"] == 0).all()
    assert (t.loc[0:19, "tracked"] == TRACKED_DETECTED).all()
    assert t["cx"].notna().all()


def test_fill_gaps_holds_leading_and_trailing_frames():
    df = pd.DataFrame(
        {
            "frame": range(6),
            "cx": [np.nan, np.nan, 10, 12, np.nan, np.nan],
            "cy": [np.nan, np.nan, 5, 5, np.nan, np.nan],
            "w": 4.0,
            "h": 4.0,
            "score": [0, 0, 0.9, 0.9, 0, 0],
            "tracked": [0, 0, 2, 2, 0, 0],
        }
    )
    out = fill_gaps(df, max_gap=5)
    assert out["cx"].tolist() == [10, 10, 10, 12, 12, 12]
    assert out["tracked"].tolist() == [0, 0, 2, 2, 0, 0]


def test_crop_boxes_fixed_size_even_and_clamped():
    n = 100
    cx = np.linspace(-20, 340, n)  # runs off both edges of a 320-wide frame
    df = pd.DataFrame(
        {
            "frame": range(n),
            "cx": cx,
            "cy": 120.0,
            "w": 40.0,
            "h": 20.0,
            "score": 0.9,
            "tracked": TRACKED_DETECTED,
        }
    )
    crops = crop_boxes(df, 320, 240, TrackConfig(crop_scale=1.6))
    assert list(crops.columns) == CROP_COLUMNS
    assert crops["w"].nunique() == 1 and crops["h"].nunique() == 1
    side = int(crops["w"].iloc[0])
    assert side == 32  # ceil(20 * 1.6) = 32, already even
    assert (crops["x0"] >= 0).all() and (crops["x0"] + side <= 320).all()
    assert (crops["y0"] >= 0).all() and (crops["y0"] + side <= 240).all()
    assert crops["x0"].iloc[0] == 0 and crops["x0"].iloc[-1] == 320 - side


def test_crop_boxes_smooths_center_jitter():
    rng = np.random.default_rng(0)
    n = 300
    cx = 160 + rng.normal(0, 6, n)
    df = pd.DataFrame(
        {
            "frame": range(n),
            "cx": cx,
            "cy": 120.0,
            "w": 40.0,
            "h": 50.0,
            "score": 0.9,
            "tracked": TRACKED_DETECTED,
        }
    )
    crops = crop_boxes(df, 640, 480, TrackConfig(center_smooth_frames=15))
    assert crops["x0"].std() < 0.5 * cx.std()


def test_crop_boxes_never_exceed_frame():
    df = pd.DataFrame(
        {
            "frame": range(10),
            "cx": 50.0,
            "cy": 50.0,
            "w": 90.0,
            "h": 90.0,
            "score": 0.9,
            "tracked": TRACKED_DETECTED,
        }
    )
    crops = crop_boxes(df, 100, 100)
    assert int(crops["w"].iloc[0]) == 100 and (crops["x0"] == 0).all()


def test_crop_boxes_needs_tracked_frames():
    df = pd.DataFrame(
        {
            "frame": range(3),
            "cx": 1.0,
            "cy": 1.0,
            "w": 1.0,
            "h": 1.0,
            "score": 0.0,
            "tracked": TRACKED_HELD,
        }
    )
    with pytest.raises(ValueError):
        crop_boxes(df, 100, 100)


def test_write_crop_video_and_map_back(tmp_path):
    def boxes(f):
        x = 20 + 4 * f
        return [(x, 110, x + 40, 130)]

    video = write_video(tmp_path / "v.mp4", blob_frames(20, 320, 240, boxes), FPS)
    crops = pd.DataFrame(
        {
            "frame": range(20),
            "x0": [10 + 4 * f for f in range(20)],
            "y0": 80,
            "w": 100,
            "h": 100,
            "tracked": TRACKED_DETECTED,
        }
    )
    out = tmp_path / "v.crop.mp4"
    n = write_crop_video(video, crops, out, fps=FPS)
    assert n == 20 and out.exists()
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (100, 100)
    # The blob (x 20..60, y 110..130) lands at crop x 10..50, y 30..50 on frame 0.
    assert frame[40, 30].mean() > 150 and frame[90, 90].mean() < 100
    # And crop coordinates map back to the normalized frame.
    kp = np.full((20, 13, 2), 25.0)
    back = crop_to_frame(kp, crops)
    assert back[0, 0].tolist() == [35.0, 105.0]
    assert back[19, 0].tolist() == [10 + 4 * 19 + 25.0, 105.0]
