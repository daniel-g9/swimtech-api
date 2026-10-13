"""Small drawing helpers that are not part of the pipeline: a gridded frame for picking
the swimmer by eye (task 1.6, until the Vue picker in 4.2), and a preview video that
draws the tracked box on the full frame so a human can judge whether it followed the
right swimmer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def save_frame_with_grid(
    video: Path, out: Path, index: int = 0, grid: int = 200
) -> tuple[int, int]:
    """Write frame `index` of `video` as a JPEG with a labelled pixel grid.
    Returns (width, height)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"could not read frame {index} from {video}")
    h, w = frame.shape[:2]
    if grid > 0:
        scale = max(w, h) / 1920.0
        thick = max(1, int(round(scale)))
        font = 0.6 * max(scale, 0.75)
        for x in range(0, w, grid):
            cv2.line(frame, (x, 0), (x, h), (0, 255, 255), thick)
            cv2.putText(
                frame,
                str(x),
                (x + 4, int(24 * scale) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font,
                (0, 255, 255),
                thick,
            )
        for y in range(0, h, grid):
            cv2.line(frame, (0, y), (w, y), (0, 255, 255), thick)
            cv2.putText(
                frame,
                str(y),
                (4, y - 4 if y else int(24 * scale) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font,
                (0, 255, 255),
                thick,
            )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"could not write {out}")
    return w, h


_COLORS = {2: (0, 255, 0), 1: (0, 200, 255), 0: (0, 0, 255)}  # detected, interpolated, held
_LABELS = {2: "tracked", 1: "gap filled", 0: "LOST (holding last position)"}


def save_track_preview(
    norm_video: Path, crops: pd.DataFrame, out: Path, max_width: int = 720
) -> int:
    """Full-frame video with the crop box drawn: green while tracked, orange across a
    short gap, red when the swimmer is lost. Returns frames written."""
    cap = cv2.VideoCapture(str(norm_video))
    if not cap.isOpened():
        raise FileNotFoundError(norm_video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    ow, oh = int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2
    by_frame = crops.set_index("frame")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    proc = writer = None
    if ffmpeg:
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{ow}x{oh}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            stdin=subprocess.PIPE,
        )
    else:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    n = 0
    f = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f in by_frame.index:
                c = by_frame.loc[f]
                tr = int(c["tracked"])
                x0, y0, s = int(c["x0"]), int(c["y0"]), int(c["w"])
                cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), _COLORS[tr], max(2, w // 300))
                cv2.putText(
                    frame,
                    f"{f / fps:5.1f}s  {_LABELS[tr]}",
                    (20, int(0.05 * h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, h / 900),
                    _COLORS[tr],
                    max(2, w // 400),
                )
            small = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            if proc is not None and proc.stdin is not None:
                proc.stdin.write(np.ascontiguousarray(small).tobytes())
            elif writer is not None:
                writer.write(small)
            n += 1
            f += 1
    finally:
        cap.release()
        if proc is not None and proc.stdin is not None:
            proc.stdin.close()
            proc.wait()
        if writer is not None:
            writer.release()
    return n
