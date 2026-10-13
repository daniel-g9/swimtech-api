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
    "motion",
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
    swimmer_point: tuple[float, float] | None = None  # tap on the swimmer (normal way)
    swimmer_frame: int = 0
    lane_box: tuple[float, float, float, float] | None = None  # fixed camera only
    lane_number: int | None = None
    calib_corners: list[list[float]] | None = None  # fixed camera only
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


# Which files each stage reads and writes. Used only for cache freshness. Stages with
# no "out" always run (they write into probe_json or are not implemented yet).
STAGE_IO: dict[str, Callable[[RacePaths, RunOptions], dict[str, list[Path]]]] = {
    "ingest": lambda p, o: {"in": [Path(o.src)], "out": [p.norm_video, p.probe_json]},
    "quality": lambda p, o: {"in": [p.norm_video], "out": []},
    "detect": lambda p, o: {"in": [p.norm_video], "out": [p.detections]},
    # Motion masks out detected people, so it runs after detect.
    "motion": lambda p, o: {"in": [p.norm_video, p.detections], "out": [p.motion, p.water]},
    # Track is cheap and depends on the selection options, so it always re-runs.
    "track": lambda p, o: {"in": [p.detections, p.motion, p.norm_video], "out": []},
    "pose": lambda p, o: {"in": [p.crop_video], "out": [p.keypoints_raw]},
    "smooth": lambda p, o: {"in": [p.keypoints_raw], "out": [p.keypoints]},
    "cycles": lambda p, o: {"in": [p.keypoints], "out": []},
    "metrics": lambda p, o: {"in": [p.keypoints], "out": []},
    "findings": lambda p, o: {"in": [p.keypoints], "out": []},
    "render": lambda p, o: {"in": [p.keypoints], "out": [p.annotated_video, p.thumbnail]},
    "results": lambda p, o: {"in": [p.keypoints], "out": [p.results_json]},
}


def resolve_selection(opts: RunOptions, paths: RacePaths):
    """Turn the run options into a SwimmerSelection (1.4). Priority: a tap on the
    swimmer, then a box on the first frame, then calibration corners (given now, or
    saved earlier as calib.json), then nothing (longest-lived track, with a warning).
    Box and corners only make sense for a fixed camera."""
    from swimtech.calib import Calibration, calibrate
    from swimtech.track import SwimmerSelection

    if opts.swimmer_point is not None:
        return SwimmerSelection(
            point=opts.swimmer_point, point_frame=opts.swimmer_frame, lane_number=opts.lane_number
        )
    if opts.lane_box is not None:
        return SwimmerSelection(box=opts.lane_box, lane_number=opts.lane_number)
    polygon = None
    if opts.calib_corners is not None:
        polygon = calibrate(opts.calib_corners, paths.calib_json).corners_px
    elif paths.calib_json.exists():
        polygon = Calibration.load(paths.calib_json).corners_px
    return SwimmerSelection(polygon=polygon, lane_number=opts.lane_number)


resolve_lane = resolve_selection  # older name


def _update_probe(paths: RacePaths, key: str, value: dict) -> None:
    probe = json.loads(paths.probe_json.read_text()) if paths.probe_json.exists() else {}
    probe[key] = value
    paths.probe_json.write_text(json.dumps(probe, indent=2))


def default_stages() -> dict[str, Callable[[RunOptions, RacePaths], None]]:
    """Bind each stage name to its module. Imports are inside so that a stub stage that
    needs heavy deps does not break `swimtech ingest` on a laptop."""

    def _ingest(o: RunOptions, p: RacePaths) -> None:
        from swimtech.ingest import ingest

        ingest(Path(o.src), p)

    def _quality(o: RunOptions, p: RacePaths) -> None:
        from swimtech.quality import check_quality

        q = check_quality(p.norm_video)
        _update_probe(p, "quality", q.to_json())
        for w in q.warnings:
            log.warning("[quality] %s", w)

    def _motion(o: RunOptions, p: RacePaths) -> None:
        import pandas as pd

        from swimtech.motion import estimate_motion, estimate_water

        det = pd.read_parquet(p.detections) if p.detections.exists() else None
        estimate_motion(p.norm_video, p.motion, exclude=det)
        estimate_water(p.norm_video, p.water)

    def _detect(o: RunOptions, p: RacePaths) -> None:
        from swimtech.detect import DetectConfig, detect

        cfg = DetectConfig(model=o.detect_model) if o.detect_model else DetectConfig()
        detect(p.norm_video, p.detections, cfg)

    def _track(o: RunOptions, p: RacePaths) -> None:
        import pandas as pd

        from swimtech.track import crop_boxes, track, tracking_report, write_crop_video

        probe = json.loads(p.probe_json.read_text())
        fps = float(probe["target_fps"])
        det = pd.read_parquet(p.detections)
        from swimtech.motion import load_water

        motion = pd.read_parquet(p.motion) if p.motion.exists() else None
        n_frames = _count_frames(p.norm_video)
        t = track(
            det,
            resolve_selection(o, p),
            fps,
            n_frames=n_frames,
            motion=motion,
            water=load_water(p.water),
            frame_size=(int(probe["width"]), int(probe["height"])),
            lane_dir=_lane_dir_at(p.norm_video, o.swimmer_frame if o.swimmer_point else 0),
        )
        report = tracking_report(t, fps)
        _update_probe(p, "tracking", report)
        for w in report["warnings"]:
            log.warning("[track] %s", w)
        if report["detected_frac"] == 0:
            raise RuntimeError(
                "the swimmer was not found anywhere in the clip. Check the --swimmer point "
                "against `swimtech frame`, or pick a frame where they are clearly visible "
                "with --swimmer-frame."
            )
        crops = crop_boxes(t, int(probe["width"]), int(probe["height"]), fps=fps)
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
        "motion": _motion,
        "track": _track,
        "pose": _todo("2.1"),
        "smooth": _todo("2.2 (stage function exists; wiring lands with 2.1)"),
        "cycles": _todo("2.3"),
        "metrics": _todo("2.4"),
        "findings": _todo("2.5"),
        "render": _todo("2.6"),
        "results": _todo("2.7"),
    }


def _lane_dir_at(video: Path, frame: int):
    """Lane-rope direction on one frame (see lanes.lane_direction), or None."""
    import cv2

    from swimtech.lanes import angle_deg, lane_direction
    from swimtech.motion import water_mask

    cap = cv2.VideoCapture(str(video))
    try:
        if frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    d = lane_direction(img, water_mask(img))
    if d is not None:
        log.info("[track] lane direction from the ropes: %.1f deg", angle_deg(d))
    return d


def _count_frames(video: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
