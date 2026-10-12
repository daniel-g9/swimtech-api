"""The metric registry: one entry per metric in docs/metrics.md (plan task 0.1).

Each metric is a pure function
    fn(kp: np.ndarray, ctx: MetricContext) -> float
where `kp` is the (13, 2) array of keypoint positions at the sampled instant (already
rectified if the metric's space is "rect") and `ctx` carries anything else the metric
needs (which arm, the median shoulder width for this length, cycle timing...).

Adding a metric is one function and one `register(...)` call. Task 2.4 fills in the
bodies that still raise NotImplementedError; the angle family is implemented here
because it is the spec's own worked example.

`vis` is NOT computed by the metric. The runner computes it as the mean visibility of
`spec.keypoints(arm)` at the sampled frame, so a metric can never forget to report it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from swimtech.metrics import geometry as g
from swimtech.types import KP, Arm, Instant, MetricSpace


@dataclass
class MetricContext:
    arm: Arm | None = None
    # Median shoulder width (px or m) over the current length; normalizes body-relative
    # distances so near and far lengths are comparable.
    shoulder_width_ref: float = float("nan")
    # Unit direction of travel along the lane in the metric's space, if known.
    lane_dir: np.ndarray | None = None
    # Cycle timing, for rhythm metrics (seconds).
    cycle_duration_s: float = float("nan")
    extra: dict = field(default_factory=dict)


MetricFn = Callable[[np.ndarray, MetricContext], float]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    instant: Instant | None  # None = per-cycle/per-length aggregate, not sampled at a frame
    unit: str
    space: MetricSpace
    per_arm: bool
    keypoints_l: tuple[int, ...]  # keypoints used when arm == "L" (or when not per_arm)
    keypoints_r: tuple[int, ...] = ()
    description: str = ""

    def keypoints(self, arm: Arm | None) -> tuple[int, ...]:
        if self.per_arm and arm == "R":
            return self.keypoints_r
        return self.keypoints_l


REGISTRY: dict[str, MetricSpec] = {}


def register(spec: MetricSpec) -> MetricSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"duplicate metric name {spec.name!r}")
    REGISTRY[spec.name] = spec
    return spec


def metric_names() -> list[str]:
    return list(REGISTRY)


def _not_implemented(name: str) -> MetricFn:
    def fn(kp: np.ndarray, ctx: MetricContext) -> float:
        raise NotImplementedError(f"metric {name!r} is implemented in plan task 2.4")

    return fn


# ----------------------------------------------------------------------------------
# Arm selection helpers
# ----------------------------------------------------------------------------------

_ARM = {
    "L": (KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST),
    "R": (KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST),
}


def _arm_pts(kp: np.ndarray, arm: Arm | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm not in _ARM:
        raise ValueError("per-arm metric called without ctx.arm")
    s, e, w = _ARM[arm]
    return kp[s], kp[e], kp[w]


def _shoulders(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return kp[KP.L_SHOULDER], kp[KP.R_SHOULDER]


def _hip_mid(kp: np.ndarray) -> np.ndarray:
    return g.midpoint(kp[KP.L_HIP], kp[KP.R_HIP])


# ----------------------------------------------------------------------------------
# Implemented: the elbow angle family (the spec's worked example)
# ----------------------------------------------------------------------------------


def elbow_angle(kp: np.ndarray, ctx: MetricContext) -> float:
    s, e, w = _arm_pts(kp, ctx.arm)
    return g.angle_deg(s, e, w)


def wrist_height_midrec(kp: np.ndarray, ctx: MetricContext) -> float:
    """Wrist height above the shoulder line, in shoulder widths."""
    _, _, w = _arm_pts(kp, ctx.arm)
    ls, rs = _shoulders(kp)
    h = g.height_above_line(w, ls, rs, _hip_mid(kp))
    ref = ctx.shoulder_width_ref
    if not np.isfinite(ref) or ref == 0:
        ref = g.distance(ls, rs)
    return h / ref if ref else float("nan")


def forearm_entry_angle(kp: np.ndarray, ctx: MetricContext) -> float:
    """Angle between the forearm (elbow -> wrist) and the body axis (hip mid -> shoulder mid)."""
    _, e, w = _arm_pts(kp, ctx.arm)
    ls, rs = _shoulders(kp)
    body_axis = g.midpoint(ls, rs) - _hip_mid(kp)
    return g.vector_angle_deg(w - e, body_axis)


def head_angle(kp: np.ndarray, ctx: MetricContext) -> float:
    """Angle between the head direction (ear mid -> nose) and the shoulder line."""
    ear_mid = g.midpoint(kp[KP.L_EAR], kp[KP.R_EAR])
    ls, rs = _shoulders(kp)
    return g.line_angle_deg(kp[KP.NOSE] - ear_mid, rs - ls)


def shoulder_line_angle(kp: np.ndarray, ctx: MetricContext) -> float:
    """Angle of the shoulder line vs the lane direction; roll proxy."""
    if ctx.lane_dir is None:
        return float("nan")
    ls, rs = _shoulders(kp)
    return g.line_angle_deg(rs - ls, ctx.lane_dir)


def shoulder_foreshortening(kp: np.ndarray, ctx: MetricContext) -> float:
    """Apparent shoulder width over its reference; < 1 means the body is rolled."""
    ls, rs = _shoulders(kp)
    ref = ctx.shoulder_width_ref
    if not np.isfinite(ref) or ref == 0:
        return float("nan")
    return g.distance(ls, rs) / ref


# ----------------------------------------------------------------------------------
# Registration. Order here is the order the UI's metric picker shows them.
# ----------------------------------------------------------------------------------

_L_ARM = (KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST)
_R_ARM = (KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST)
_SHOULDERS = (KP.L_SHOULDER, KP.R_SHOULDER)
_HIPS = (KP.L_HIP, KP.R_HIP)
_HEAD = (KP.NOSE, KP.L_EAR, KP.R_EAR)

for _instant in ("exit", "midrec", "entry"):
    register(
        MetricSpec(
            name=f"elbow_angle_{_instant}",
            fn=elbow_angle,
            instant=_instant,  # type: ignore[arg-type]
            unit="deg",
            space="image",
            per_arm=True,
            keypoints_l=_L_ARM,
            keypoints_r=_R_ARM,
            description=f"Interior elbow angle at {_instant}.",
        )
    )

register(
    MetricSpec(
        name="wrist_height_midrec",
        fn=wrist_height_midrec,
        instant="midrec",
        unit="sw",
        space="image",
        per_arm=True,
        keypoints_l=(KP.L_WRIST, *_SHOULDERS, *_HIPS),
        keypoints_r=(KP.R_WRIST, *_SHOULDERS, *_HIPS),
        description="Wrist height above the shoulder line at mid-recovery, in shoulder widths.",
    )
)

register(
    MetricSpec(
        name="hand_entry_offset_rect",
        fn=_not_implemented("hand_entry_offset_rect"),
        instant="entry",
        unit="m",
        space="rect",
        per_arm=True,
        keypoints_l=(KP.L_WRIST, *_SHOULDERS, *_HIPS),
        keypoints_r=(KP.R_WRIST, *_SHOULDERS, *_HIPS),
        description="Lateral offset of the wrist from the body line at entry, rectified.",
    )
)

register(
    MetricSpec(
        name="forearm_entry_angle",
        fn=forearm_entry_angle,
        instant="entry",
        unit="deg",
        space="image",
        per_arm=True,
        keypoints_l=(KP.L_ELBOW, KP.L_WRIST, *_SHOULDERS, *_HIPS),
        keypoints_r=(KP.R_ELBOW, KP.R_WRIST, *_SHOULDERS, *_HIPS),
        description="Angle between the forearm and the body axis at entry.",
    )
)

register(
    MetricSpec(
        name="head_angle",
        fn=head_angle,
        instant="midrec",
        unit="deg",
        space="image",
        per_arm=False,
        keypoints_l=(*_HEAD, *_SHOULDERS),
        description="Head direction vs the shoulder line, sampled on non-breath cycles.",
    )
)

register(
    MetricSpec(
        name="head_lift_breath",
        fn=_not_implemented("head_lift_breath"),
        instant="breath",
        unit="sw",
        space="image",
        per_arm=False,
        keypoints_l=(KP.NOSE, *_SHOULDERS, *_HIPS),
        description="Nose height above the shoulder line on a breath minus non-breath baseline.",
    )
)

register(
    MetricSpec(
        name="shoulder_line_angle",
        fn=shoulder_line_angle,
        instant="entry",
        unit="deg",
        space="image",
        per_arm=False,
        keypoints_l=_SHOULDERS,
        description="Shoulder line vs lane direction; roll proxy.",
    )
)

register(
    MetricSpec(
        name="shoulder_foreshortening",
        fn=shoulder_foreshortening,
        instant="entry",
        unit="ratio",
        space="image",
        per_arm=False,
        keypoints_l=_SHOULDERS,
        description="Apparent shoulder width over the length's median; roll proxy.",
    )
)

register(
    MetricSpec(
        name="hip_shoulder_offset_rect",
        fn=_not_implemented("hip_shoulder_offset_rect"),
        instant="entry",
        unit="m",
        space="rect",
        per_arm=False,
        keypoints_l=(*_SHOULDERS, *_HIPS),
        description="Lateral offset of hip midpoint from shoulder midpoint across the lane.",
    )
)

# Rhythm and breathing metrics are per-cycle or per-length aggregates computed from the
# cycles array, not sampled at a frame. Their `fn` receives an empty kp array and reads
# ctx; see docs/metrics.md.
for _name, _unit, _desc in (
    ("breath_side", "side", "Which side the swimmer breathed on this cycle: -1 L, 0 none, +1 R."),
    ("breath_frequency", "per_cycle", "Breaths per cycle over a length."),
    ("breath_cycle_penalty", "s", "Breath-cycle duration minus mean non-breath cycle duration."),
    ("stroke_rate", "spm", "Strokes per minute over a length (both arms)."),
    ("cycle_cv", "ratio", "Coefficient of variation of cycle duration over a length."),
):
    register(
        MetricSpec(
            name=_name,
            fn=_not_implemented(_name),
            instant=None,
            unit=_unit,
            space="image",
            per_arm=False,
            keypoints_l=(KP.L_WRIST, KP.R_WRIST, *_HEAD),
            description=_desc,
        )
    )
