"""Stage 1.3: swimmer detector. Finds every person box in every frame.

Wraps RF-DETR (Apache 2.0) pretrained on COCO `person`. Runs at `detect_fps` on a
downscaled frame and linearly interpolates boxes between runs.

Output: <race_id>.det.parquet with columns frame, x0, y0, x1, y1, score, interpolated.
Boxes are pixels in the normalized video's frame (never the downscaled one).

The model is injected through the `Detector` protocol so the stage can be tested with a
fake detector and no torch install; `RFDetrDetector` is the real one.

Teaching notes
--------------
Detectors trained on COCO see a lot of people but few half-submerged ones; recall drops
mostly on the underwater phases, which we don't need for surface technique. Running at
15 fps and interpolating is 4x cheaper than 60 fps and loses nothing for localization
because the crop has margin. Interpolation needs to know which box at frame t is "the
same" as which at frame t+4; we match by overlap (IoU), greedily, which is enough when
boxes move a few pixels between runs. Identity over the whole race is the tracker's job
(1.4), not ours.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("swimtech.detect")

DET_COLUMNS = ["frame", "x0", "y0", "x1", "y1", "score", "interpolated"]
_DET_DTYPES = {
    "frame": int,
    "x0": float,
    "y0": float,
    "x1": float,
    "y1": float,
    "score": float,
    "interpolated": bool,
}


@dataclass(frozen=True)
class DetectConfig:
    detect_fps: float = 15.0
    max_side_px: int = 1280  # downscale the long side to this before inference
    score_min: float = 0.1  # boxes below this are dropped; low ones feed ByteTrack's 2nd pass
    model: str = "rfdetr-medium"  # resolved by swimtech/models.py
    batch_size: int = 8
    match_iou_min: float = 0.3  # for interpolation between detection runs


class Detector(Protocol):
    """Takes a list of RGB uint8 frames, returns one (k, 5) array per frame:
    x0, y0, x1, y1, score, in the pixel coordinates of the frame it was given."""

    def __call__(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]: ...


class RFDetrDetector:
    """The real detector. Lazy-loads rfdetr so this module imports without torch."""

    def __init__(self, model: str = "rfdetr-medium", device: str | None = None) -> None:
        from swimtech.models import COCO_PERSON_CLASS_ID, load_rfdetr

        self._model: Any = load_rfdetr(model, device)
        self._person = COCO_PERSON_CLASS_ID

    def __call__(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        if not frames_rgb:
            return []
        # A low threshold on purpose: ByteTrack's second association pass (1.4) uses the
        # low-confidence boxes to ride through splash. DetectConfig.score_min trims later.
        dets = self._model.predict(frames_rgb, threshold=0.05, include_source_image=False)
        if not isinstance(dets, list):
            dets = [dets]
        out = []
        for d in dets:
            if len(d) == 0:
                out.append(np.zeros((0, 5), dtype=float))
                continue
            if d.class_id is not None:
                keep = np.asarray(d.class_id) == self._person
            else:
                keep = np.ones(len(d), dtype=bool)
            xyxy = np.asarray(d.xyxy, dtype=float)[keep]
            conf = np.asarray(d.confidence, dtype=float)[keep]
            out.append(np.column_stack([xyxy, conf]))
        return out


def sample_frames(n_frames: int, fps: float, detect_fps: float) -> np.ndarray:
    """Which frame indices to run the model on: every `stride`-th frame, plus the last."""
    if n_frames <= 0:
        return np.zeros(0, dtype=int)
    stride = max(1, int(round(fps / detect_fps))) if detect_fps > 0 else 1
    idx = np.arange(0, n_frames, stride)
    if idx[-1] != n_frames - 1:
        idx = np.append(idx, n_frames - 1)
    return idx


def _read_frames(cap: cv2.VideoCapture, wanted: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Sequential read (no seeking: seeking H.264 is slow and inexact). Yields BGR."""
    wanted_set = {int(i) for i in wanted}
    last = max(wanted_set) if wanted_set else -1
    f = 0
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if f in wanted_set:
            yield f, frame
        f += 1


def _downscale(frame_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))))
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), scale


def detect(
    video: Path,
    out: Path,
    cfg: DetectConfig = DetectConfig(),
    detector: Detector | None = None,
) -> pd.DataFrame:
    """The stage function. Returns the detections table and writes it as Parquet."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
    if detector is None:
        detector = RFDetrDetector(cfg.model)

    wanted = sample_frames(n_frames, fps, cfg.detect_fps)
    log.info(
        "detect: %d of %d frames at %.3g fps (model %s)", len(wanted), n_frames, fps, cfg.model
    )
    rows: list[tuple] = []
    batch_idx: list[int] = []
    batch_img: list[np.ndarray] = []
    batch_scale: list[float] = []

    def flush() -> None:
        if not batch_img:
            return
        for f, s, boxes in zip(batch_idx, batch_scale, detector(batch_img), strict=True):
            for x0, y0, x1, y1, score in np.asarray(boxes, dtype=float).reshape(-1, 5):
                if score < cfg.score_min:
                    continue
                rows.append((f, x0 / s, y0 / s, x1 / s, y1 / s, score, False))
        batch_idx.clear()
        batch_img.clear()
        batch_scale.clear()

    for f, frame in _read_frames(cap, wanted):
        rgb, scale = _downscale(frame, cfg.max_side_px)
        batch_idx.append(f)
        batch_img.append(rgb)
        batch_scale.append(scale)
        if len(batch_img) >= cfg.batch_size:
            flush()
    flush()
    cap.release()

    sparse = pd.DataFrame(rows, columns=DET_COLUMNS).astype(_DET_DTYPES)
    dense = interpolate_boxes(sparse, n_frames, iou_min=cfg.match_iou_min)
    out.parent.mkdir(parents=True, exist_ok=True)
    dense.to_parquet(out, index=False)
    log.info("detect: %d boxes on %d frames -> %s", len(dense), dense["frame"].nunique(), out)
    return dense


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) xyxy boxes -> (n, m) intersection over union."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def greedy_match(iou: np.ndarray, iou_min: float) -> list[tuple[int, int]]:
    """Pair rows to columns by descending IoU, each used at most once."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    used_r: set[int] = set()
    used_c: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        r, c = divmod(int(flat), iou.shape[1])
        if iou[r, c] < iou_min:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return pairs


def interpolate_boxes(det: pd.DataFrame, n_frames: int, iou_min: float = 0.3) -> pd.DataFrame:
    """Fill the frames between detection runs by linear interpolation.

    Boxes on consecutive sampled frames are paired by greedy IoU; each pair is
    interpolated across the frames in between and marked `interpolated=True`. Unpaired
    boxes (someone appeared or vanished) are not extended. Pure; tested without a model.
    """
    if len(det) == 0:
        return pd.DataFrame(columns=DET_COLUMNS).astype(_DET_DTYPES)
    det = det[DET_COLUMNS].copy()
    det["interpolated"] = det["interpolated"].astype(bool)
    sampled = sorted(int(f) for f in det["frame"].unique())
    by_frame = dict(tuple(det.groupby("frame")))
    new_rows: list[tuple] = []
    for fa, fb in zip(sampled[:-1], sampled[1:], strict=True):
        gap = fb - fa
        if gap <= 1:
            continue
        a, b = by_frame[fa], by_frame[fb]
        ba = a[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
        bb = b[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float)
        sa = a["score"].to_numpy(dtype=float)
        sb = b["score"].to_numpy(dtype=float)
        for r, c in greedy_match(iou_matrix(ba, bb), iou_min):
            for k in range(1, gap):
                t = k / gap
                box = ba[r] * (1 - t) + bb[c] * t
                new_rows.append((fa + k, *box, sa[r] * (1 - t) + sb[c] * t, True))
    if new_rows:
        extra = pd.DataFrame(new_rows, columns=DET_COLUMNS).astype(_DET_DTYPES)
        det = pd.concat([det, extra], ignore_index=True)
    det = det[(det["frame"] >= 0) & (det["frame"] < max(n_frames, 1))]
    det = det.sort_values(["frame", "score"], ascending=[True, False]).reset_index(drop=True)
    return det[DET_COLUMNS].astype(_DET_DTYPES)
