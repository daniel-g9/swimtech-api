"""Stage 2.7: pipeline runner. Chains the stages with caching: a stage is skipped when
its output exists and is newer than its inputs.

Stage order and what each one reads/writes is the only knowledge this module has; it
never computes anything itself. Use `--to <stage>` to stop early (task 1.6 runs
`--to track` on a laptop CPU).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from swimtech.types import RacePaths, Stroke

log = logging.getLogger("swimtech")

STAGES: tuple[str, ...] = (
    "ingest",
    "quality",
    "detect",
    "track",
    "pose",
    "smooth",
    "cycles",
    "metrics",
    "findings",
    "render",
    "results",
)


@dataclass
class RunOptions:
    src: Path
    race_id: str
    work_dir: Path = Path("data")
    stroke: Stroke = "free"
    lane_box: tuple[float, float, float, float] | None = None
    lane_number: int | None = None
    calib_corners: list[list[float]] | None = None
    to: str = "results"
    force: bool = False


def _is_fresh(outputs: list[Path], inputs: list[Path]) -> bool:
    if not outputs or not all(o.exists() for o in outputs):
        return False
    newest_in = max((i.stat().st_mtime for i in inputs if i.exists()), default=0.0)
    return all(o.stat().st_mtime >= newest_in for o in outputs)


def run(
    opts: RunOptions, stages: dict[str, Callable[[RunOptions, RacePaths], None]] | None = None
) -> RacePaths:
    """Run stages in order up to `opts.to`. `stages` can be injected for tests."""
    if opts.to not in STAGES:
        raise ValueError(f"unknown stage {opts.to!r}; choose from {STAGES}")
    paths = RacePaths(work_dir=Path(opts.work_dir), race_id=opts.race_id)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    impl = stages if stages is not None else default_stages()

    for name in STAGES:
        io = STAGE_IO[name](paths, opts)
        if not opts.force and _is_fresh(io["out"], io["in"]):
            log.info("[%s] cached", name)
        else:
            log.info("[%s] running", name)
            impl[name](opts, paths)
        if name == opts.to:
            break
    return paths


# Which files each stage reads and writes. Used only for cache freshness.
STAGE_IO: dict[str, Callable[[RacePaths, RunOptions], dict[str, list[Path]]]] = {
    "ingest": lambda p, o: {"in": [Path(o.src)], "out": [p.norm_video, p.probe_json]},
    "quality": lambda p, o: {
        "in": [p.norm_video],
        "out": [],
    },  # writes into probe_json; always re-run
    "detect": lambda p, o: {"in": [p.norm_video], "out": [p.detections]},
    "track": lambda p, o: {"in": [p.detections], "out": [p.crop_video, p.crop_offsets]},
    "pose": lambda p, o: {"in": [p.crop_video], "out": [p.keypoints_raw]},
    "smooth": lambda p, o: {"in": [p.keypoints_raw], "out": [p.keypoints]},
    "cycles": lambda p, o: {"in": [p.keypoints], "out": []},
    "metrics": lambda p, o: {"in": [p.keypoints], "out": []},
    "findings": lambda p, o: {"in": [p.keypoints], "out": []},
    "render": lambda p, o: {"in": [p.keypoints], "out": [p.annotated_video, p.thumbnail]},
    "results": lambda p, o: {"in": [p.keypoints], "out": [p.results_json]},
}


def default_stages() -> dict[str, Callable[[RunOptions, RacePaths], None]]:
    """Bind each stage name to its module. Imports are inside so that a stub stage that
    needs heavy deps does not break `swimtech ingest` on a laptop."""

    def _ingest(o: RunOptions, p: RacePaths) -> None:
        from swimtech.ingest import ingest

        ingest(Path(o.src), p)

    def _quality(o: RunOptions, p: RacePaths) -> None:
        import json

        from swimtech.quality import check_quality

        q = check_quality(p.norm_video)
        probe = json.loads(p.probe_json.read_text())
        probe["quality"] = q.to_json()
        p.probe_json.write_text(json.dumps(probe, indent=2))
        for w in q.warnings:
            log.warning("[quality] %s", w)

    def _todo(task: str) -> Callable[[RunOptions, RacePaths], None]:
        def fn(o: RunOptions, p: RacePaths) -> None:
            raise NotImplementedError(f"plan task {task} is not implemented yet")

        return fn

    return {
        "ingest": _ingest,
        "quality": _quality,
        "detect": _todo("1.3"),
        "track": _todo("1.4"),
        "pose": _todo("2.1"),
        "smooth": _todo("2.2 (stage function exists; wiring lands with 2.1)"),
        "cycles": _todo("2.3"),
        "metrics": _todo("2.4"),
        "findings": _todo("2.5"),
        "render": _todo("2.6"),
        "results": _todo("2.7"),
    }
