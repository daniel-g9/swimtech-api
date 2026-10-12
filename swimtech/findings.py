"""Stage 2.5: rule-based findings. Turn the metrics table into a short list a coach
should look at, each tied to a frame.

Rules v1: flag any metric whose race slope exceeds a threshold, any L/R asymmetry over
a threshold, breath-side bias, cycle-variability spikes. Rank by magnitude x visibility,
cap at 8, name the frame where the finding is most visible.

Status: STUB. Implemented in plan task 2.5. Thresholds live in `FindingThresholds` so a
coach can tune them without touching the rules.

Teaching notes
--------------
Rules first, LLM later (7.4). Rules are predictable and tunable with a coach in the
room; an LLM is for wording, not detection.
"""

from __future__ import annotations

from dataclasses import dataclass

from swimtech.types import Cycle, Finding, MetricValue

MAX_FINDINGS = 8


@dataclass(frozen=True)
class FindingThresholds:
    slope_deg_per_cycle: float = 0.5  # elbow angle drift across the race
    asym_deg: float = 10.0
    asym_sw: float = 0.15
    breath_bias: float = 0.8  # fraction of breaths on one side
    cycle_cv: float = 0.12


def generate_findings(
    metrics: list[MetricValue],
    cycles: list[Cycle],
    summary: dict,
    th: FindingThresholds = FindingThresholds(),
) -> list[Finding]:
    """The stage function."""
    raise NotImplementedError("plan task 2.5: findings")
