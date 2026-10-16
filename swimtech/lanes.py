"""Lane geometry from the picture itself: the lane ropes, found in every frame.

Lane-mates swim parallel at the same speed and are otherwise indistinguishable, so the
tracker (1.4) must know which lane the swimmer is in. A line estimated once from the
first frame drifts as a handheld camera pans and tilts (measured: 150-330 px off by the
second length). So the two ropes bounding the swimmer's lane are found in every frame
and followed: the corridor between them is the lane, in that frame's own pixels, and
only boxes inside it are allowed to be the swimmer.

Representation: a rope is a straight line in normal form  x*cos(phi) + y*sin(phi) = rho
where phi is the normal angle. All ropes in one frame share (approximately) one
direction; each has its own rho. The corridor is (rho_above, rho_below) around the
swimmer, plus the direction.

Teaching notes
--------------
Lane ropes are the longest straight edges on the water, so Hough line segments inside
the water mask, within a few degrees of the dominant direction, are rope pieces. Their
perpendicular offsets cluster into one value per rope. Frame to frame, the ropes move
with the camera, so the previous corridor is pushed through the camera-motion transform
and snapped to the nearest detected ropes; if splash hides a rope, the prediction stands.
Because the ropes converge with perspective, the corridor is re-measured rather than
assumed parallel, which is also why this is done per frame and not once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("swimtech.lanes")

LANE_COLUMNS = ["frame", "phi", "rho_above", "rho_below", "snapped"]


def lane_direction(bgr: np.ndarray, water: np.ndarray | None = None) -> np.ndarray | None:
    """Unit vector along the lane ropes in this frame, or None if no ropes are found."""
    segs = rope_segments(bgr, water)
    if segs is None:
        return None
    ang = _dominant_angle(segs)
    return np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])


def angle_deg(direction: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


# ------------------------------------------------------------------ rope detection


def rope_segments(bgr: np.ndarray, water: np.ndarray | None = None) -> np.ndarray | None:
    """(k, 4) line segments x0, y0, x1, y1 in full-frame pixels, or None."""
    full_w, full_h = bgr.shape[1], bgr.shape[0]
    h = max(int(round(640 * full_h / full_w)), 8)
    small = cv2.resize(bgr, (640, h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    if water is not None:
        m = cv2.resize(water.astype(np.uint8) * 255, (640, h), interpolation=cv2.INTER_NEAREST)
        edges[m == 0] = 0
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=640 // 6, maxLineGap=12
    )
    if lines is None or len(lines) < 3:
        return None
    seg = lines.reshape(-1, 4).astype(float)
    seg *= full_w / 640.0
    return seg


def _dominant_angle(seg: np.ndarray) -> float:
    d = seg[:, 2:] - seg[:, :2]
    length = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ang = np.where(ang > 90, ang - 180, np.where(ang < -90, ang + 180, ang))
    order = np.argsort(ang)
    cum = np.cumsum(length[order])
    return float(ang[order][np.searchsorted(cum, cum[-1] / 2)])


def ropes_in_frame(
    bgr: np.ndarray,
    water: np.ndarray | None,
    phi_hint: float | None = None,
    angle_tol_deg: float = 8.0,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Find the ropes: returns (phi, rhos, support) where rhos are the perpendicular
    offsets of each rope (sorted) and support the total segment length behind each."""
    seg = rope_segments(bgr, water)
    if seg is None:
        return None
    d = seg[:, 2:] - seg[:, :2]
    length = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ang = np.where(ang > 90, ang - 180, np.where(ang < -90, ang + 180, ang))
    dom = _dominant_angle(seg) if phi_hint is None else np.degrees(phi_hint) - 90.0
    keep = np.abs(((ang - dom + 90) % 180) - 90) <= angle_tol_deg
    if keep.sum() < 2:
        return None
    seg, length = seg[keep], length[keep]
    phi = np.radians(dom + 90.0)
    nrm = np.array([np.cos(phi), np.sin(phi)])
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    rho = mid @ nrm
    # Cluster offsets: sort, split where the gap exceeds `gap`.
    order = np.argsort(rho)
    rho, length = rho[order], length[order]
    gap = 0.012 * max(bgr.shape[:2])
    groups = np.split(np.arange(len(rho)), np.flatnonzero(np.diff(rho) > gap) + 1)
    rhos, support = [], []
    for g in groups:
        w = length[g]
        rhos.append(float(np.average(rho[g], weights=w)))
        support.append(float(w.sum()))
    return float(phi), np.asarray(rhos), np.asarray(support)


# ------------------------------------------------------------------ corridor tracking


@dataclass
class Corridor:
    phi: float  # normal angle of the ropes (radians)
    rho_above: float  # the rope on the smaller-rho side of the swimmer
    rho_below: float  # the rope on the larger-rho side

    @property
    def width(self) -> float:
        return self.rho_below - self.rho_above

    def normal(self) -> np.ndarray:
        return np.array([np.cos(self.phi), np.sin(self.phi)])

    def offset(self, xy: np.ndarray) -> np.ndarray:
        """Signed position across the lane: 0 at the upper rope, 1 at the lower."""
        rho = np.asarray(xy, dtype=float).reshape(-1, 2) @ self.normal()
        return (rho - self.rho_above) / max(self.width, 1e-6)

    def contains(self, xy: np.ndarray, margin: float = 0.15) -> np.ndarray:
        off = self.offset(xy)
        return (off >= -margin) & (off <= 1 + margin)


def corridor_at_point(
    phi: float, rhos: np.ndarray, support: np.ndarray, point: np.ndarray
) -> Corridor | None:
    """The two ropes around `point`. If only one side was detected, the other is placed
    one median rope spacing away."""
    if len(rhos) == 0:
        return None
    nrm = np.array([np.cos(phi), np.sin(phi)])
    r = float(np.asarray(point, dtype=float) @ nrm)
    above = rhos[rhos < r]
    below = rhos[rhos > r]
    spacing = float(np.median(np.diff(rhos))) if len(rhos) >= 2 else None
    if len(above) and len(below):
        return Corridor(phi, float(above.max()), float(below.min()))
    if spacing is None:
        return None
    if len(above):
        return Corridor(phi, float(above.max()), float(above.max()) + spacing)
    return Corridor(phi, float(below.min()) - spacing, float(below.min()))


def _push_line(phi: float, rho: float, m: np.ndarray) -> tuple[float, float]:
    """Move a line x cos(phi) + y sin(phi) = rho through a 2D similarity m (3x3)."""
    nrm = np.array([np.cos(phi), np.sin(phi)])
    p0 = rho * nrm  # a point on the line
    tangent = np.array([-nrm[1], nrm[0]])
    p1 = p0 + 100.0 * tangent
    q0 = (m @ np.array([*p0, 1.0]))[:2]
    q1 = (m @ np.array([*p1, 1.0]))[:2]
    t = q1 - q0
    new_phi = float(np.arctan2(t[0], -t[1]))  # normal of the moved tangent
    new_nrm = np.array([np.cos(new_phi), np.sin(new_phi)])
    new_rho = float(q0 @ new_nrm)
    # Keep the normal pointing the same way as before (rho sign convention).
    if new_nrm @ nrm < 0:
        new_phi += np.pi
        new_rho = -new_rho
    return new_phi, new_rho


def track_corridor(
    video_path,
    water: np.ndarray | None,
    motion: pd.DataFrame | None,
    point: np.ndarray,
    point_frame: int = 0,
    snap_frac: float = 0.3,
) -> pd.DataFrame:
    """Follow the swimmer's lane corridor through the whole clip.

    Starts at `point_frame` from the ropes around `point`, then for every other frame
    pushes the previous corridor through the camera motion and snaps each rope to the
    nearest detected rope within `snap_frac` of the lane width. Returns one row per
    frame: phi, rho_above, rho_below, snapped (how many of the two ropes were seen)."""
    from swimtech.motion import _as_matrix  # local import: motion imports nothing from here

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    # Frames are read once, sequentially; only the frames before the tap frame are
    # buffered (for the backward pass), which keeps memory small when the tap is early.
    pf = max(point_frame, 0)
    before: list[np.ndarray] = []
    seq: list[np.ndarray] = []  # frames from pf on, consumed as we go
    f_read = 0
    while f_read < pf:
        ok, img = cap.read()
        if not ok:
            break
        before.append(img)
        f_read += 1
    frames = _FrameSource(cap, before)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release()
        return pd.DataFrame(columns=LANE_COLUMNS)

    steps: dict[int, np.ndarray] = {}
    if motion is not None and len(motion):
        for _, r in motion.iterrows():
            steps[int(r["frame"])] = _as_matrix(r["a"], r["b"], r["tx"], r["ty"])

    def detect(f: int, phi_hint: float | None):
        w = water[f] if water is not None and f < len(water) else None
        img = frames.get(f)
        if img is None:
            return None
        return ropes_in_frame(img, w, phi_hint)

    pf = min(pf, max(n - 1, 0))
    first = frames.get(pf)
    if first is None:
        cap.release()
        return pd.DataFrame(columns=LANE_COLUMNS)
    center = np.array([first.shape[1] / 2.0, first.shape[0] / 2.0])
    det0 = detect(pf, None)
    if det0 is None:
        log.warning("lane corridor: no ropes found on frame %d", pf)
        return pd.DataFrame(columns=LANE_COLUMNS)
    start = corridor_at_point(*det0, np.asarray(point, dtype=float))
    if start is None:
        log.warning("lane corridor: could not place the lane around the tapped point")
        return pd.DataFrame(columns=LANE_COLUMNS)
    rows: dict[int, tuple] = {pf: (start.phi, start.rho_above, start.rho_below, 2)}

    zoom = {pf: 1.0}  # cumulative scale from the tap frame, forward and backward

    def step(cur: Corridor, f_from: int, f_to: int) -> tuple[Corridor, int]:
        # Predict: push both ropes through the camera motion (forward uses M_{f_to},
        # backward its inverse). Only the offsets are trusted from this; the angle is
        # re-measured from the frame, because small rotation errors accumulate.
        if f_to > f_from:
            m = steps.get(f_to, np.eye(3))
        else:
            m = np.linalg.inv(steps.get(f_from, np.eye(3)))
        zoom[f_to] = zoom.get(f_from, 1.0) * float(np.hypot(m[0, 0], m[1, 0]))
        expected = start.width * zoom[f_to]  # what the lane width should be now
        phi_p, rho_a = _push_line(cur.phi, cur.rho_above, m)
        _, rho_b = _push_line(cur.phi, cur.rho_below, m)
        pred = Corridor(phi_p, rho_a, rho_b)
        det = detect(f_to, None)
        if det is None:
            return pred, 0
        phi, rhos, _support = det
        d_ang = abs(((np.degrees(phi - cur.phi) + 90) % 180) - 90)
        if d_ang > 15.0:
            return pred, 0  # not the ropes (a bad frame); keep the prediction
        nrm_new = np.array([np.cos(phi), np.sin(phi)])

        # Re-express the predicted ropes with the freshly measured normal, anchored at
        # the point of each line nearest the frame center (anchoring at the foot of the
        # perpendicular from the origin would amplify any angle difference by ~rho).
        def _rho_with(rho_old: float, n_old: np.ndarray) -> float:
            foot = center + (rho_old - center @ n_old) * n_old
            return float(foot @ nrm_new)

        ra = _rho_with(pred.rho_above, pred.normal())
        rb = _rho_with(pred.rho_below, pred.normal())
        width = max(rb - ra, 1.0)
        tol = snap_frac * width
        ia = int(np.argmin(np.abs(rhos - ra)))
        ib = int(np.argmin(np.abs(rhos - rb)))
        ok_a = abs(rhos[ia] - ra) <= tol
        ok_b = abs(rhos[ib] - rb) <= tol and ib != ia
        if ok_a and ok_b and 0.7 <= (rhos[ib] - rhos[ia]) / width <= 1.4:
            lo, hi = float(rhos[ia]), float(rhos[ib])
            # A rope well inside means the corridor has swallowed two lanes: keep the
            # half the previous corridor's center was in.
            inner = rhos[(rhos > lo + 0.3 * (hi - lo)) & (rhos < hi - 0.3 * (hi - lo))]
            if len(inner) and (hi - lo) > 1.4 * expected:
                prev_mid = (ra + rb) / 2
                mid = float(inner[np.argmin(np.abs(inner - prev_mid))])
                lo, hi = (lo, mid) if prev_mid < mid else (mid, hi)
            return Corridor(phi, lo, hi), 2
        # One rope seen: place the other at the expected width.
        w_new = float(np.clip(width, 0.6 * expected, 1.6 * expected))
        if ok_a:
            return Corridor(phi, float(rhos[ia]), float(rhos[ia]) + w_new), 1
        if ok_b:
            return Corridor(phi, float(rhos[ib]) - w_new, float(rhos[ib])), 1
        return Corridor(phi, ra, ra + w_new), 0

    cur = start
    f = pf + 1
    while frames.get(f) is not None:
        cur, snapped = step(cur, f - 1, f)
        rows[f] = (cur.phi, cur.rho_above, cur.rho_below, snapped)
        f += 1
    cur = start
    for f in range(pf - 1, -1, -1):
        cur, snapped = step(cur, f + 1, f)
        rows[f] = (cur.phi, cur.rho_above, cur.rho_below, snapped)
    cap.release()
    n = max(rows) + 1
    df = pd.DataFrame([(f, *rows[f]) for f in range(n)], columns=LANE_COLUMNS)
    seen = (df["snapped"] > 0).mean()
    log.info("lane corridor: ropes re-found on %.0f%% of frames", 100 * seen)
    return df


class _FrameSource:
    """Sequential access to video frames: buffered frames before the tap frame, and
    frames from the tap frame on read from the capture on demand (each exactly once,
    in order). `get(f)` returns None past the end."""

    def __init__(self, cap: cv2.VideoCapture, before: list[np.ndarray]) -> None:
        self._cap = cap
        self._before = before
        self._next = len(before)
        self._cur: np.ndarray | None = None
        self._cur_idx = -1

    def get(self, f: int) -> np.ndarray | None:
        if f < len(self._before):
            return self._before[f]
        while self._cur_idx < f:
            ok, img = self._cap.read()
            if not ok:
                return None
            self._cur_idx += 1
            self._cur = img
        return self._cur if self._cur_idx == f else None


def corridor_for_frame(lane_df: pd.DataFrame, frame: int) -> Corridor | None:
    if lane_df is None or len(lane_df) == 0:
        return None
    r = lane_df[lane_df["frame"] == frame]
    if r.empty:
        return None
    r = r.iloc[0]
    return Corridor(float(r["phi"]), float(r["rho_above"]), float(r["rho_below"]))
