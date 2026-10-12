import numpy as np
import pandas as pd
import pytest

from swimtech.detect import (
    DET_COLUMNS,
    DetectConfig,
    detect,
    greedy_match,
    interpolate_boxes,
    iou_matrix,
    sample_frames,
)
from tests.conftest import blob_detector


def test_sample_frames_strides_and_keeps_last():
    idx = sample_frames(40, fps=30, detect_fps=15)
    assert idx[0] == 0 and idx[1] == 2
    assert idx[-1] == 39  # the last frame is always sampled
    assert len(sample_frames(0, 30, 15)) == 0
    assert list(sample_frames(5, 60, 60)) == [0, 1, 2, 3, 4]


def test_iou_matrix_and_greedy_match():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
    b = np.array([[0, 0, 10, 10], [25, 25, 35, 35], [100, 100, 110, 110]])
    iou = iou_matrix(a, b)
    assert iou.shape == (2, 3)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 1] == pytest.approx(25 / 175)
    assert iou[0, 2] == 0.0
    assert greedy_match(iou, 0.1) == [(0, 0), (1, 1)]
    assert greedy_match(iou, 0.5) == [(0, 0)]
    assert greedy_match(np.zeros((0, 3)), 0.1) == []


def _row(frame, box, score=0.9):
    return (frame, *box, score, False)


def test_interpolate_boxes_fills_between_sampled_frames():
    sparse = pd.DataFrame(
        [_row(0, (0, 0, 10, 10)), _row(4, (4, 0, 14, 10)), _row(4, (200, 200, 210, 210))],
        columns=DET_COLUMNS,
    )
    dense = interpolate_boxes(sparse, n_frames=5)
    assert list(dense.columns) == DET_COLUMNS
    mid = dense[(dense["frame"] == 2)]
    assert len(mid) == 1  # the unmatched far-away box is not extended backwards
    assert mid["x0"].iloc[0] == pytest.approx(2.0)
    assert bool(mid["interpolated"].iloc[0]) is True
    assert dense["frame"].tolist() == [0, 1, 2, 3, 4, 4]
    assert not dense[dense["frame"].isin([0, 4])]["interpolated"].any()


def test_interpolate_boxes_empty_and_out_of_range():
    empty = interpolate_boxes(pd.DataFrame(columns=DET_COLUMNS), n_frames=10)
    assert len(empty) == 0 and list(empty.columns) == DET_COLUMNS
    sparse = pd.DataFrame([_row(0, (0, 0, 10, 10)), _row(50, (0, 0, 10, 10))], columns=DET_COLUMNS)
    dense = interpolate_boxes(sparse, n_frames=20)  # frame 50 is beyond the clip
    assert dense["frame"].max() < 20


def test_detect_runs_at_reduced_rate_and_scales_boxes_back(moving_blob_video, tmp_path):
    video, boxes = moving_blob_video
    out = tmp_path / "x.det.parquet"
    # max_side_px=160 forces a 0.5x downscale of the 320-wide frame, so this also checks
    # that boxes come back in normalized-frame pixels.
    cfg = DetectConfig(detect_fps=15, max_side_px=160, batch_size=4)
    df = detect(video, out, cfg, detector=blob_detector)
    assert out.exists()
    assert list(df.columns) == DET_COLUMNS
    assert df["frame"].nunique() == 60  # interpolation made it dense
    sampled = df[~df["interpolated"]]["frame"].unique()
    assert set(sampled) <= set(range(0, 60, 2)) | {59}
    # Frame 10: swimmer at x = 20 + 40 = 60..100, y 110..130 (lane 2 blob is elsewhere).
    f10 = df[(df["frame"] == 10) & (df["y0"] > 80)]
    assert len(f10) == 1
    assert f10["x0"].iloc[0] == pytest.approx(60, abs=4)
    assert f10["x1"].iloc[0] == pytest.approx(100, abs=4)
    assert f10["y0"].iloc[0] == pytest.approx(110, abs=4)
    # Interpolated frames exist and sit between their neighbours.
    f11 = df[(df["frame"] == 11) & (df["y0"] > 80)]
    assert bool(f11["interpolated"].iloc[0]) is True
    assert 60 <= f11["x0"].iloc[0] <= 70
