"""Small drawing helpers that are not part of the pipeline: a gridded frame for picking a
lane box by eye (task 1.6, until the Vue lane picker in 4.2), and a crop preview."""

from __future__ import annotations

from pathlib import Path

import cv2


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
