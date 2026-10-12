"""Stage 1.2: clip quality check. Warn before processing when a clip is unusable.

Samples 20 frames and computes sharpness (Laplacian variance), mean brightness,
clipped-highlight fraction (glare), and inter-frame border motion (camera shake).

Teaching notes
--------------
The Laplacian variance of a grayscale frame is a cheap focus measure; under ~100 on 4K
footage usually means motion blur or missed focus. Border motion is estimated with
cv2.phaseCorrelate on the top strip of consecutive frames; anything over ~2 px/frame
means the tripod moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    sharpness_min: float = 100.0
    brightness_min: float = 60.0
    brightness_max: float = 200.0
    glare_max: float = 0.05  # fraction of pixels at >= 250
    shake_max_px: float = 2.0
    shake_texture_min: float = 4.0  # grey-level std of the top strip needed to measure shake
    n_samples: int = 20


@dataclass
class Quality:
    score: float  # 0-1
    sharpness: float
    brightness: float
    glare: float
    shake_px: float
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "score": round(self.score, 3),
            "sharpness": round(self.sharpness, 1),
            "brightness": round(self.brightness, 1),
            "glare": round(self.glare, 4),
            "shake_px": round(self.shake_px, 2),
            "warnings": self.warnings,
        }


def _sample_indices(n_frames: int, n: int) -> list[int]:
    if n_frames <= 0:
        return []
    return sorted(set(np.linspace(0, max(n_frames - 2, 0), n).astype(int).tolist()))


def check_quality(video: Path, th: QualityThresholds = QualityThresholds()) -> Quality:
    """The stage function. Reads the normalized video; returns a score and warnings."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sharp, bright, glare, shake = [], [], [], []

    for idx in _sample_indices(n_frames, th.n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, a = cap.read()
        ok2, b = cap.read()
        if not ok:
            continue
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(ga, cv2.CV_64F).var()))
        bright.append(float(ga.mean()))
        glare.append(float((ga >= 250).mean()))
        if ok2:
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            strip = slice(0, max(ga.shape[0] // 10, 8))
            # Phase correlation on a featureless strip (plain wall, ceiling, sky) returns
            # noise, not motion. Only measure where there is texture to align.
            if ga[strip].std() >= th.shake_texture_min:
                (dx, dy), _ = cv2.phaseCorrelate(
                    ga[strip].astype(np.float32), gb[strip].astype(np.float32)
                )
                shake.append(float(np.hypot(dx, dy)))
    cap.release()

    if not sharp:
        return Quality(0.0, 0.0, 0.0, 0.0, 0.0, ["Could not read any frames from the clip."])

    q = Quality(
        score=1.0,
        sharpness=float(np.median(sharp)),
        brightness=float(np.median(bright)),
        glare=float(np.median(glare)),
        shake_px=float(np.median(shake)) if shake else 0.0,
    )
    penalties = 0.0
    if q.sharpness < th.sharpness_min:
        q.warnings.append(
            "Footage looks soft or motion-blurred. Lock focus and use a faster shutter."
        )
        penalties += 0.4
    if q.brightness < th.brightness_min:
        q.warnings.append("Footage is dark. Raise exposure or film with more light.")
        penalties += 0.2
    elif q.brightness > th.brightness_max:
        q.warnings.append("Footage is very bright; details may be washed out.")
        penalties += 0.2
    if q.glare > th.glare_max:
        q.warnings.append("Glare on the water. Sit with windows behind you or use a polarizer.")
        penalties += 0.2
    if q.shake_px > th.shake_max_px:
        q.warnings.append("The camera moved during the race. Use a tripod and don't touch it.")
        penalties += 0.3
    q.warnings = q.warnings[:3]
    q.score = max(0.0, 1.0 - penalties)
    return q
