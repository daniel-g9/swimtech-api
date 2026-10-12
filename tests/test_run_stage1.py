"""End to end through Stage 1: a synthetic 'phone clip' -> ingest -> quality -> detect
-> track -> crop video, with the fake detector injected. Needs ffmpeg for ingest."""

import json
import shutil

import cv2
import pandas as pd
import pytest

from swimtech.detect import DetectConfig, detect
from swimtech.run import RunOptions, default_stages, resolve_lane, run
from swimtech.track import TRACKED_DETECTED
from swimtech.types import RacePaths
from tests.conftest import blob_detector

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed (see docs/setup.md)",
)


def _stages_with_fake_detector():
    stages = default_stages()

    def _detect(o, p):
        detect(p.norm_video, p.detections, DetectConfig(detect_fps=15), detector=blob_detector)

    stages["detect"] = _detect
    return stages


def test_stage1_end_to_end(moving_blob_video, tmp_path):
    clip, boxes = moving_blob_video
    work = tmp_path / "data"
    opts = RunOptions(
        src=clip,
        race_id="t1",
        work_dir=work,
        lane_box=(0, 100, 320, 140),
        lane_number=4,
        to="track",
    )
    paths = run(opts, stages=_stages_with_fake_detector())
    assert paths.crop_video.exists() and paths.crop_offsets.exists()

    probe = json.loads(paths.probe_json.read_text())
    assert "quality" in probe and probe["target_fps"] == 30

    crops = pd.read_parquet(paths.crop_offsets)
    assert len(crops) == 60
    assert (crops["tracked"] == TRACKED_DETECTED).mean() > 0.9
    side = int(crops["w"].iloc[0])
    # The crop follows the lane-4 swimmer (y 110..130), not the other lane (y 30..50).
    for f in (5, 30, 55):
        x0, y0, x1, y1 = boxes(f)[0]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        row = crops[crops["frame"] == f].iloc[0]
        assert abs(row["x0"] + side / 2 - cx) < 12
        assert abs(row["y0"] + side / 2 - cy) < 8

    cap = cv2.VideoCapture(str(paths.crop_video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (side, side)


def test_second_run_is_cached(moving_blob_video, tmp_path, caplog):
    clip, _ = moving_blob_video
    opts = RunOptions(
        src=clip, race_id="t2", work_dir=tmp_path, lane_box=(0, 100, 320, 140), to="track"
    )
    run(opts, stages=_stages_with_fake_detector())
    with caplog.at_level("INFO", logger="swimtech"):
        run(opts, stages=_stages_with_fake_detector())
    cached = [r.message for r in caplog.records if "cached" in r.message]
    assert any("[detect]" in m for m in cached) and any("[track]" in m for m in cached)


def test_resolve_lane_priority(tmp_path):
    paths = RacePaths(work_dir=tmp_path, race_id="r")
    corners = [[0, 100], [320, 100], [320, 140], [0, 140]]
    sel = resolve_lane(RunOptions(src=tmp_path, race_id="r", calib_corners=corners), paths)
    assert sel.polygon == corners and paths.calib_json.exists()
    # Saved calibration is reused on the next run without corners.
    sel2 = resolve_lane(RunOptions(src=tmp_path, race_id="r"), paths)
    assert sel2.polygon == corners
    # An explicit box wins over the calibration.
    sel3 = resolve_lane(RunOptions(src=tmp_path, race_id="r", lane_box=(1, 2, 3, 4)), paths)
    assert sel3.box == (1, 2, 3, 4) and sel3.polygon is None
    # A race with no calibration and no box has no corridor: the tracker picks the
    # longest-lived swimmer and warns.
    other = RacePaths(work_dir=tmp_path, race_id="other")
    assert resolve_lane(RunOptions(src=tmp_path, race_id="other"), other).corridor() is None
