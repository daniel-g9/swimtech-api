"""Stage 1.1: turn a raw phone clip into the normalized video every later stage reads.

Output: <race_id>.norm.mp4 (H.264, 8-bit yuv420p, constant frame rate, rotation baked in)
        <race_id>.probe.json (fps, duration, resolution, codec, whether the source was VFR)

Teaching notes
--------------
Phones record variable frame rate (VFR): frame timestamps drift, so "frame 600" is not
reliably 10 s. `-fps_mode cfr -r 60` resamples to a constant rate so frame index maps to
time. iPhones write HEVC 10-bit with a rotation tag; ffmpeg's default autorotate bakes
it in so OpenCV sees upright frames. `-pix_fmt yuv420p` keeps every decoder downstream
happy. We cap at 60 fps and never upsample a slower source: a 30 fps clip stays 30 fps
(and the probe JSON records that, so `frame / fps` is still correct).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from swimtech.types import TARGET_FPS, RacePaths


class FfmpegMissingError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FfmpegMissingError(
            f"{tool} not found on PATH. See docs/setup.md for how to install ffmpeg."
        )
    return path


@dataclass
class Probe:
    """What ffprobe tells us about a clip, plus what we decided to do with it."""

    source: str
    codec: str
    pix_fmt: str
    width: int
    height: int
    rotation: int
    duration_s: float
    avg_fps: float
    r_fps: float
    is_vfr: bool
    target_fps: float
    n_frames_estimate: int

    def to_json(self) -> dict:
        return asdict(self)


def _parse_fps(text: str | None) -> float:
    if not text or text in ("0/0", "0"):
        return 0.0
    return float(Fraction(text))


def probe(src: Path) -> Probe:
    """Run ffprobe on a clip and summarize the video stream."""
    ffprobe = _require("ffprobe")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate,duration,"
        "nb_frames:stream_side_data=rotation:stream_tags=rotate:format=duration",
        "-of",
        "json",
        str(src),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    stream = info["streams"][0]

    rotation = 0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and "rotate" in (stream.get("tags") or {}):
        rotation = int(stream["tags"]["rotate"])

    avg_fps = _parse_fps(stream.get("avg_frame_rate"))
    r_fps = _parse_fps(stream.get("r_frame_rate"))
    # avg (frames/duration) and r (container tick rate) differ on VFR clips.
    is_vfr = avg_fps > 0 and r_fps > 0 and abs(avg_fps - r_fps) / r_fps > 0.02

    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    target = min(float(TARGET_FPS), round(avg_fps) if avg_fps else TARGET_FPS)
    width, height = int(stream["width"]), int(stream["height"])
    if rotation % 180 != 0:
        width, height = height, width

    return Probe(
        source=str(src),
        codec=stream.get("codec_name", "?"),
        pix_fmt=stream.get("pix_fmt", "?"),
        width=width,
        height=height,
        rotation=rotation,
        duration_s=duration,
        avg_fps=avg_fps,
        r_fps=r_fps,
        is_vfr=is_vfr,
        target_fps=target,
        n_frames_estimate=int(round(duration * target)),
    )


def normalize(src: Path, dst: Path, target_fps: float, *, crf: int = 18) -> None:
    """Re-encode to H.264 8-bit constant-frame-rate with rotation baked in."""
    ffmpeg = _require("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-fps_mode",
        "cfr",
        "-r",
        f"{target_fps:g}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def ingest(src: Path, paths: RacePaths) -> Probe:
    """The stage function: probe, normalize, write both outputs. Returns the probe."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    p = probe(src)
    normalize(src, paths.norm_video, p.target_fps)
    paths.probe_json.write_text(json.dumps(p.to_json(), indent=2))
    return p
