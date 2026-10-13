"""Camera motion estimation on a synthetic panning / zooming clip."""

import cv2
import numpy as np
import pytest

from swimtech.motion import (
    MotionConfig,
    apply,
    boxes_to_frame,
    boxes_to_world,
    cumulative,
    estimate_motion,
    identity_rows,
    scale_of,
)
from tests.conftest import write_video


def _scene(w=1600, h=1200, seed=1) -> np.ndarray:
    """A big textured background (lane ropes + random blobs) to pan a window over."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 140, dtype=np.uint8)
    for y in range(100, h, 120):
        cv2.line(img, (0, y), (w, y - 60), (40, 40, 200), 6)
    for _ in range(300):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(
            img,
            (int(x), int(y)),
            int(rng.integers(3, 10)),
            tuple(int(c) for c in rng.integers(0, 255, 3)),
            -1,
        )
    return img


def pan_frames(n: int, cam, size=(320, 240), scene=None) -> list[np.ndarray]:
    """cam(f) -> (x, y, zoom): the top-left of the window in the scene and the zoom factor
    (window is size/zoom in scene pixels, resized up to `size`)."""
    scene = _scene() if scene is None else scene
    w, h = size
    out = []
    for f in range(n):
        x, y, z = cam(f)
        ww, hh = int(round(w / z)), int(round(h / z))
        win = scene[int(y) : int(y) + hh, int(x) : int(x) + ww]
        out.append(cv2.resize(win, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def test_identity_rows_and_cumulative_shapes():
    m = identity_rows(5)
    cum = cumulative(m)
    assert cum.shape == (5, 3, 3)
    assert np.allclose(cum, np.eye(3))
    assert cumulative(identity_rows(0), 3).shape == (3, 3, 3)


def test_apply_and_scale_of():
    m = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0], [0, 0, 1]])
    assert apply(m, np.array([1.0, 1.0])).tolist() == [12.0, 7.0]
    assert scale_of(m) == pytest.approx(2.0)
    stack = np.stack([np.eye(3), m])
    out = apply(stack, np.array([[[1.0, 1.0]], [[1.0, 1.0]]]))
    assert out[1, 0].tolist() == [12.0, 7.0] and out[0, 0].tolist() == [1.0, 1.0]


def test_estimate_pan(tmp_path):
    # Pan 3 px right and 1 px down per frame.
    frames = pan_frames(20, lambda f: (100 + 3 * f, 100 + 1 * f, 1.0))
    m = estimate_motion(write_video(tmp_path / "pan.mp4", frames), tmp_path / "m.parquet")
    assert len(m) == 20 and (tmp_path / "m.parquet").exists()
    body = m.iloc[1:]
    # The scene moves left/up on screen when the camera pans right/down.
    assert body["tx"].mean() == pytest.approx(-3.0, abs=0.4)
    assert body["ty"].mean() == pytest.approx(-1.0, abs=0.4)
    assert body["a"].mean() == pytest.approx(1.0, abs=0.01)
    assert (body["inliers"] > 15).all()
    # Cumulative: a scene point fixed in the world drifts by the full pan on screen.
    cum = cumulative(m)
    p = apply(cum[19], np.array([160.0, 120.0]))
    assert p[0] == pytest.approx(160 - 3 * 19, abs=3)
    assert p[1] == pytest.approx(120 - 19, abs=3)


def test_estimate_zoom(tmp_path):
    frames = pan_frames(15, lambda f: (200, 200, 1.0 + 0.02 * f))
    m = estimate_motion(write_video(tmp_path / "zoom.mp4", frames), None)
    s = scale_of(cumulative(m)[14])
    assert s == pytest.approx(1.28, abs=0.06)


def test_boxes_world_roundtrip():
    m = identity_rows(3)
    m.loc[1, ["a", "tx", "ty"]] = [1.0, -10.0, 0.0]
    m.loc[2, ["a", "tx", "ty"]] = [1.0, -10.0, 0.0]
    cum = cumulative(m)
    boxes = np.array([[50, 50, 70, 90], [50, 50, 70, 90]], dtype=float)
    frames = np.array([0, 2])
    w = boxes_to_world(boxes, cum, frames)
    assert w[0].tolist() == [50, 50, 70, 90]
    assert w[1].tolist() == [70, 50, 90, 90]  # screen moved 20 left => world is 20 right
    back = boxes_to_frame(w, cum, frames)
    assert np.allclose(back, boxes)


def test_featureless_clip_gives_identity(tmp_path):
    flat = [np.full((240, 320, 3), 90, dtype=np.uint8)] * 6
    m = estimate_motion(write_video(tmp_path / "flat.mp4", flat), None, MotionConfig())
    assert np.allclose(m[["tx", "ty"]].to_numpy(), 0.0)
    assert np.allclose(m["a"].to_numpy(), 1.0)
