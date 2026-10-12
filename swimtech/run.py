"""Stage 2.7: pipeline runner. Chains the stages with caching: a stage is skipped when
its output exists and is newer than its inputs.

Stage order and what each one reads/writes is the only knowledge this module has; it
never computes anything itself. Use `--to <stage>` to stop early (task 1.6 runs
`--to track` on a laptop CPU).
"""

from __future__ import annotations

import json
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
    detect_model: str | None = None  # None -> DetectConfig default
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
    "track": lambda p, o: {
        "in": [p.detections, p.norm_video, p.calib_json],
        "out": [p.crop_video, p.crop_offsets],
    },
    "pose": lambda p, o: {"in": [p.crop_video], "out": [p.keypoints_raw]},
    "smooth": lambda p, o: {"in": [p.keypoints_raw], "out": [p.keypoints]},
    "cycles": lambda p, o: {"in": [p.keypoints], "out": []},
    "metrics": lambda p, o: {"in": [p.keypoints], "out": []},
    "findings": lambda p, o: {"in": [p.keypoints], "out": []},
    "render": lambda p, o: {"in": [p.keypoints], "out": [p.annotated_video, p.thumbnail]},
    "results": lambda p, o: {"in": [p.keypoints], "out": [p.results_json]},
}


def resolve_lane(opts: RunOptions, paths: RacePaths):
    """Turn the run options into a LaneSelection (1.4). Priority: an explicit box on the
    first frame, then calibration corners (given now, or saved earlier as calib.json),
    then nothing (the tracker follows the longest-lived swimmer and warns)."""
    from swimtech.calib import Calibration, calibrate
    from swimtech.track import LaneSelection

    polygon = None
    if opts.calib_corners is not None:
        cal = calibrate(opts.calib_corners, paths.calib_json)
        polygon = cal.corners_px
    elif paths.calib_json.exists():
        polygon = Calibration.load(paths.calib_json).corners_px
    if opts.lane_box is not None:
        return LaneSelection(box=opts.lane_box, lane_number=opts.lane_number)
    return LaneSelection(polygon=polygon, lane_number=opts.lane_number)


def default_stages() -> dict[str, Callable[[RunOptions, RacePaths], None]]:
    """Bind each stage name to its module. Imports are inside so that a stub stage that
    needs heavy deps does not break `swimtech ingest` on a laptop."""

    def _ingest(o: RunOptions, p: RacePaths) -> None:
        from swimtech.ingest import ingest

        ingest(Path(o.src), p)

    def _quality(o: RunOptions, p: RacePaths) -> None:
        from swimtech.quality import check_quality

        q = check_quality(p.norm_video)
        probe = json.loads(p.probe_json.read_text())
        probe["quality"] = q.to_json()
        p.probe_json.write_text(json.dumps(probe, indent=2))
        for w in q.warnings:
            log.warning("[quality] %s", w)

    def _detect(o: RunOptions, p: RacePaths) -> None:
        from swimtech.detect import DetectConfig, detect

        cfg = DetectConfig(model=o.detect_model) if o.detect_model else DetectConfig()
        detect(p.norm_video, p.detections, cfg)

    def _track(o: RunOptions, p: RacePaths) -> None:
        import pandas as pd

        from swimtech.track import crop_boxes, track, write_crop_video

        probe = json.loads(p.probe_json.read_text())
        fps = float(probe["target_fps"])
        det = pd.read_parquet(p.detections)
        n_frames = _count_frames(p.norm_video)
        t = track(det, resolve_lane(o, p), fps, n_frames=n_frames)
        crops = crop_boxes(t, int(probe["width"]), int(probe["height"]))
        crops.to_parquet(p.crop_offsets, index=False)
        write_crop_video(p.norm_video, crops, p.crop_video, fps=fps)

    def _todo(task: str) -> Callable[[RunOptions, RacePaths], None]:
        def fn(o: RunOptions, p: RacePaths) -> None:
            raise NotImplementedError(f"plan task {task} is not implemented yet")

        return fn

    return {
        "ingest": _ingest,
        "quality": _quality,
        "detect": _detect,
        "track": _track,
        "pose": _todo("2.1"),
        "smooth": _todo("2.2 (stage function exists; wiring lands with 2.1)"),
        "cycles": _todo("2.3"),
        "metrics": _todo("2.4"),
        "findings": _todo("2.5"),
        "render": _todo("2.6"),
        "results": _todo("2.7"),
    }


def _count_frames(video: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
