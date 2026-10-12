"""Shared constants and data types. Every stage imports from here; nothing here imports a stage.

Conventions (see README.md and the plan's "Shared conventions"):
- 13 keypoints in a fixed order. Never reorder: Parquet files, the Vue overlay and the
  training config all rely on these indices.
- vis: 0 = not in frame, 1 = occluded/interpolated, 2 = visible.
- Coordinates are pixels in the normalized video's frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
)
N_KEYPOINTS = len(KEYPOINT_NAMES)


class KP(IntEnum):
    """Keypoint index. Use these instead of bare integers in every stage."""

    NOSE = 0
    L_EAR = 1
    R_EAR = 2
    L_SHOULDER = 3
    R_SHOULDER = 4
    L_ELBOW = 5
    R_ELBOW = 6
    L_WRIST = 7
    R_WRIST = 8
    L_HIP = 9
    R_HIP = 10
    L_KNEE = 11
    R_KNEE = 12


# Map from the COCO-17 order that RTMPose / RF-DETR emit to our 13-point order.
# COCO: 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
#       7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
#       13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
COCO17_TO_SWIM13: tuple[int, ...] = (0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

# Limb segments for drawing and for sanity checks (pairs of KP indices).
SKELETON: tuple[tuple[int, int], ...] = (
    (KP.L_EAR, KP.NOSE),
    (KP.R_EAR, KP.NOSE),
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_SHOULDER, KP.L_ELBOW),
    (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW),
    (KP.R_ELBOW, KP.R_WRIST),
    (KP.L_SHOULDER, KP.L_HIP),
    (KP.R_SHOULDER, KP.R_HIP),
    (KP.L_HIP, KP.R_HIP),
    (KP.L_HIP, KP.L_KNEE),
    (KP.R_HIP, KP.R_KNEE),
)

# Left/right pairs, used by the flip augmentation (3.4) and the L/R toggle.
LR_PAIRS: tuple[tuple[int, int], ...] = (
    (KP.L_EAR, KP.R_EAR),
    (KP.L_SHOULDER, KP.R_SHOULDER),
    (KP.L_ELBOW, KP.R_ELBOW),
    (KP.L_WRIST, KP.R_WRIST),
    (KP.L_HIP, KP.R_HIP),
    (KP.L_KNEE, KP.R_KNEE),
)

Arm = Literal["L", "R"]
Stroke = Literal["free", "back", "breast", "fly"]
Instant = Literal["entry", "midrec", "exit", "breath"]
MetricSpace = Literal["image", "rect"]

# Visibility values stored per keypoint.
VIS_OUT_OF_FRAME = 0
VIS_OCCLUDED = 1
VIS_VISIBLE = 2

# The UI greys out any metric whose mean visibility is below this.
VIS_GREY_THRESHOLD = 0.6

TARGET_FPS = 60


@dataclass(frozen=True)
class RacePaths:
    """Every file one race produces, derived from a work directory and a race id.

    Stages read and write only through these paths, so caching (2.7) can check
    whether an output exists before re-running a stage.
    """

    work_dir: Path
    race_id: str

    def _p(self, suffix: str) -> Path:
        return self.work_dir / f"{self.race_id}.{suffix}"

    @property
    def norm_video(self) -> Path:
        return self._p("norm.mp4")

    @property
    def probe_json(self) -> Path:
        return self._p("probe.json")

    @property
    def detections(self) -> Path:
        return self._p("det.parquet")

    @property
    def crop_video(self) -> Path:
        return self._p("crop.mp4")

    @property
    def crop_offsets(self) -> Path:
        return self._p("crop.parquet")

    @property
    def calib_json(self) -> Path:
        return self._p("calib.json")

    @property
    def keypoints_raw(self) -> Path:
        return self._p("kp.raw.parquet")

    @property
    def keypoints(self) -> Path:
        return self._p("kp.parquet")

    @property
    def results_json(self) -> Path:
        return self._p("results.json")

    @property
    def annotated_video(self) -> Path:
        return self._p("annot.mp4")

    @property
    def thumbnail(self) -> Path:
        return self._p("jpg")


@dataclass
class Cycle:
    """One stroke cycle of one arm. Frame indices, not seconds."""

    arm: Arm
    n: int
    entry_f: int
    exit_f: int
    length: int
    midrec_f: int | None = None
    breath_f: int | None = None

    def instant_frame(self, instant: Instant) -> int | None:
        return {
            "entry": self.entry_f,
            "exit": self.exit_f,
            "midrec": self.midrec_f,
            "breath": self.breath_f,
        }[instant]


@dataclass
class MetricValue:
    """One metric sampled on one cycle. Matches the `metrics` entries in results JSON."""

    cycle: str  # e.g. "L3"; "race" for whole-race aggregates, "len2" for per-length
    name: str
    value: float
    unit: str
    vis: float  # mean visibility of the keypoints used, 0-1

    def to_json(self) -> dict:
        return {
            "cycle": self.cycle,
            "name": self.name,
            "value": round(float(self.value), 3),
            "unit": self.unit,
            "vis": round(float(self.vis), 3),
        }


@dataclass
class Finding:
    frame: int
    severity: Literal["low", "med", "high"]
    text: str
    metric: str | None = None

    def to_json(self) -> dict:
        out = {"frame": self.frame, "severity": self.severity, "text": self.text}
        if self.metric:
            out["metric"] = self.metric
        return out


@dataclass
class Results:
    """The results JSON contract between the API and the Vue app."""

    race_id: str
    fps: float
    n_frames: int
    stroke: Stroke
    keypoints: str
    cycles: list[Cycle] = field(default_factory=list)
    metrics: list[MetricValue] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    media: dict[str, str] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "race_id": self.race_id,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "stroke": self.stroke,
            "keypoints": self.keypoints,
            "cycles": [
                {
                    "arm": c.arm,
                    "n": c.n,
                    "entry_f": c.entry_f,
                    "exit_f": c.exit_f,
                    "length": c.length,
                    **({"midrec_f": c.midrec_f} if c.midrec_f is not None else {}),
                    **({"breath_f": c.breath_f} if c.breath_f is not None else {}),
                }
                for c in self.cycles
            ],
            "metrics": [m.to_json() for m in self.metrics],
            "findings": [f.to_json() for f in self.findings],
            "media": self.media,
            "summary": self.summary,
        }
