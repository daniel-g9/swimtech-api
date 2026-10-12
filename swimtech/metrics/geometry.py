"""Pure 2D geometry used by every metric. No I/O, no state, fully unit-tested.

All functions take points as arrays of shape (2,) = (x, y) in pixels (or meters after
rectification) and return floats. NaN in, NaN out: a missing keypoint must never raise,
it must propagate so the metric's `vis` explains the gap.

Teaching notes
--------------
The interior angle at joint B from points A, B, C is
    degrees(acos(dot(A-B, C-B) / (|A-B| |C-B|)))
We clip the cosine to [-1, 1] because floating-point error can push it a hair outside
and acos would return NaN for a perfectly valid straight arm.

Image y grows downward. "Height above the shoulder line" is therefore a signed
perpendicular distance from the line through the two shoulders, with the sign chosen so
that positive means toward the head side... except that from the stands we cannot know
which side is "up" from two points alone. We use the hip midpoint as the reference: the
side of the shoulder line away from the hips is positive. That works in every camera
orientation and both swim directions.
"""

from __future__ import annotations

import numpy as np

Point = np.ndarray  # shape (2,)


def angle_deg(a: Point, b: Point, c: Point) -> float:
    """Interior angle at B, in degrees, from points A, B, C."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if not np.isfinite(na) or not np.isfinite(nc) or na == 0 or nc == 0:
        return float("nan")
    cos = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def vector_angle_deg(u: Point, v: Point) -> float:
    """Unsigned angle between two direction vectors, 0-180 degrees."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not np.isfinite(nu) or not np.isfinite(nv) or nu == 0 or nv == 0:
        return float("nan")
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def line_angle_deg(u: Point, v: Point) -> float:
    """Angle between two *lines* (direction ignored), 0-90 degrees."""
    a = vector_angle_deg(u, v)
    if np.isnan(a):
        return a
    return float(min(a, 180.0 - a))


def midpoint(a: Point, b: Point) -> Point:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def distance(a: Point, b: Point) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def signed_perp_distance(p: Point, line_a: Point, line_b: Point, positive_side: Point) -> float:
    """Signed perpendicular distance from P to the line through A and B.

    Positive when P is on the same side of the line as `positive_side`. Callers that
    want "away from the hips" use `height_above_line` below, which flips the sign.
    """
    a = np.asarray(line_a, dtype=float)
    b = np.asarray(line_b, dtype=float)
    p = np.asarray(p, dtype=float)
    d = b - a
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n == 0:
        return float("nan")
    normal = np.array([-d[1], d[0]]) / n
    dist = float(np.dot(p - a, normal))
    ref = float(np.dot(np.asarray(positive_side, dtype=float) - a, normal))
    if not np.isfinite(ref) or ref == 0:
        return float("nan")
    return dist if ref > 0 else -dist


def height_above_line(p: Point, line_a: Point, line_b: Point, hip_mid: Point) -> float:
    """Perpendicular distance of P from the shoulder line, positive on the side away
    from the hips (i.e. "above" the shoulders in body terms)."""
    v = signed_perp_distance(p, line_a, line_b, hip_mid)
    return float("nan") if np.isnan(v) else -v


def lateral_offset(p: Point, axis_origin: Point, axis_dir: Point) -> float:
    """Signed distance of P from the line through `axis_origin` along `axis_dir`.

    Used for hand entry offset and hip-shoulder offset after rectification, where the
    axis is the swimmer's body line (hip mid -> shoulder mid). Positive = to the left of
    the direction of travel in a right-handed (x right, y up) frame.
    """
    o = np.asarray(axis_origin, dtype=float)
    d = np.asarray(axis_dir, dtype=float)
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n == 0:
        return float("nan")
    normal = np.array([-d[1], d[0]]) / n
    return float(np.dot(np.asarray(p, dtype=float) - o, normal))
