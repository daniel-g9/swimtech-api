"""Quality check (1.2) on synthetic clips: a sharp textured clip is clean; blurred, dark
and shaky clips each trip exactly the right warning."""

import cv2
import numpy as np

from swimtech.quality import QualityThresholds, check_quality
from tests.conftest import write_video

TH = QualityThresholds(n_samples=5)


def _texture(w=320, h=240, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 0/200 rather than 0/255: pure white would (correctly) trip the glare check.
    noise = rng.integers(0, 2, (h // 4, w // 4), dtype=np.uint8) * 200
    img = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_sharp_static_clip_is_clean(tmp_path):
    base = _texture()
    q = check_quality(write_video(tmp_path / "ok.mp4", [base] * 12), TH)
    assert q.warnings == []
    assert q.score == 1.0
    assert q.shake_px < 1.0


def test_blurred_clip_warns_about_focus(tmp_path):
    blurred = cv2.GaussianBlur(_texture(), (0, 0), 12)
    q = check_quality(write_video(tmp_path / "blur.mp4", [blurred] * 12), TH)
    assert any("soft" in w for w in q.warnings)
    assert q.score < 1.0


def test_dark_clip_warns(tmp_path):
    dark = (_texture() * 0.1).astype(np.uint8)
    q = check_quality(write_video(tmp_path / "dark.mp4", [dark] * 12), TH)
    assert any("dark" in w for w in q.warnings)


def test_featureless_top_strip_is_not_mistaken_for_shake(tmp_path):
    """A plain wall above the pool gives phase correlation nothing to align; that must
    not read as camera movement. Only the swimmer moves here."""
    frames = []
    for f in range(12):
        img = np.full((240, 320, 3), 120, dtype=np.uint8)
        img[150:200, 20 + 8 * f : 60 + 8 * f] = 220
        frames.append(img)
    q = check_quality(write_video(tmp_path / "wall.mp4", frames), TH)
    assert q.shake_px == 0.0
    assert not any("moved" in w for w in q.warnings)


def test_shaky_clip_warns(tmp_path):
    base = _texture(w=400)
    frames = [np.ascontiguousarray(np.roll(base, 6 * f, axis=1)) for f in range(12)]
    q = check_quality(write_video(tmp_path / "shake.mp4", frames), TH)
    assert q.shake_px > 2.0
    assert any("moved" in w for w in q.warnings)
