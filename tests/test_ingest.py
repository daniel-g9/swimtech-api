"""Ingest tests need ffmpeg. They generate a tiny synthetic clip so no real footage is
required, and skip cleanly when ffmpeg is not on PATH."""

import json
import shutil
import subprocess

import pytest

from swimtech.ingest import ingest, probe
from swimtech.types import RacePaths

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed (see docs/setup.md)",
)


@pytest.fixture
def synthetic_clip(tmp_path):
    """2 seconds of a test pattern at 30 fps, 320x240, H.264."""
    src = tmp_path / "IMG_0001.MOV"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    return src


def test_probe_reads_basic_fields(synthetic_clip):
    p = probe(synthetic_clip)
    assert p.codec == "h264"
    assert (p.width, p.height) == (320, 240)
    assert p.avg_fps == pytest.approx(30.0)
    assert p.duration_s == pytest.approx(2.0, abs=0.1)
    assert p.target_fps == 30  # never upsample a slower source
    assert not p.is_vfr


def test_ingest_writes_norm_video_and_probe_json(synthetic_clip, tmp_path):
    paths = RacePaths(work_dir=tmp_path / "out", race_id="t1")
    p = ingest(synthetic_clip, paths)
    assert paths.norm_video.exists()
    assert paths.probe_json.exists()
    saved = json.loads(paths.probe_json.read_text())
    assert saved["target_fps"] == p.target_fps
    out = probe(paths.norm_video)
    assert out.codec == "h264"
    assert out.pix_fmt == "yuv420p"
    assert out.avg_fps == pytest.approx(30.0)
    assert out.rotation == 0


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nope.MOV", RacePaths(work_dir=tmp_path, race_id="x"))
