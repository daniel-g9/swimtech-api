"""Command line entry point: `uv run swimtech <command>`.

swimtech ingest IMG_1234.MOV --race-id meet1_r3
swimtech frame data/test1.norm.mp4 --out first.jpg      # pick a lane box by eye
swimtech run clip.MOV --race-id test1 --swimmer 560,890 --to track
swimtech metrics          # list the registry
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_box(text: str | None) -> tuple[float, float, float, float] | None:
    if not text:
        return None
    try:
        vals = tuple(float(v) for v in text.split(","))
    except ValueError as e:
        raise typer.BadParameter("--lane-box needs four numbers: x0,y0,x1,y1") from e
    if len(vals) != 4:
        raise typer.BadParameter("--lane-box needs four numbers: x0,y0,x1,y1")
    return vals  # type: ignore[return-value]


def _parse_point(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError as e:
        raise typer.BadParameter("--swimmer needs two numbers: x,y") from e
    return (x, y)


def _parse_corners(text: str | None) -> list[list[float]] | None:
    """'x,y;x,y;x,y;x,y' in the order near-left, near-right, far-right, far-left."""
    if not text:
        return None
    pts = []
    for chunk in text.split(";"):
        try:
            x, y = (float(v) for v in chunk.split(","))
        except ValueError as e:
            raise typer.BadParameter("--calib-corners needs 'x,y;x,y;x,y;x,y'") from e
        pts.append([x, y])
    if len(pts) != 4:
        raise typer.BadParameter("--calib-corners needs exactly four points")
    return pts


@app.command()
def ingest(
    src: Path = typer.Argument(..., exists=True, readable=True, help="Phone clip (.MOV/.MP4)"),
    race_id: str = typer.Option(..., "--race-id", help="Id used to name every output file"),
    work_dir: Path = typer.Option(Path("data"), help="Where outputs go"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Normalize a clip (plan task 1.1)."""
    from swimtech.ingest import ingest as _ingest
    from swimtech.types import RacePaths

    _setup_logging(verbose)
    paths = RacePaths(work_dir=work_dir, race_id=race_id)
    p = _ingest(src, paths)
    typer.echo(
        f"{p.codec} {p.width}x{p.height} rot={p.rotation} {p.avg_fps:.2f} fps "
        f"({'VFR' if p.is_vfr else 'CFR'}) {p.duration_s:.1f}s "
        f"-> {paths.norm_video} @ {p.target_fps:g} fps"
    )


@app.command()
def run(
    src: Path = typer.Argument(..., exists=True, readable=True),
    race_id: str = typer.Option(..., "--race-id"),
    work_dir: Path = typer.Option(Path("data")),
    stroke: str = typer.Option("free", help="free | back | breast | fly (v1: free)"),
    swimmer: str | None = typer.Option(
        None, help="x,y of the swimmer on the first frame (see `swimtech frame`)"
    ),
    swimmer_frame: int = typer.Option(0, help="Frame index the --swimmer point was read on"),
    lane: int | None = typer.Option(None, help="Lane number (a label only)"),
    lane_box: str | None = typer.Option(
        None, help="Fixed camera only: x0,y0,x1,y1 enclosing the swimmer's lane"
    ),
    calib_corners: str | None = typer.Option(
        None,
        help="Fixed camera only: lane corners 'x,y;x,y;x,y;x,y' "
        "(near-left, near-right, far-right, far-left)",
    ),
    detect_model: str | None = typer.Option(
        None, help="rfdetr-nano | rfdetr-small | rfdetr-medium (default) | rfdetr-large"
    ),
    to: str = typer.Option("results", help="Stop after this stage"),
    force: bool = typer.Option(False, help="Ignore cached outputs"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run the pipeline end to end, or up to --to (plan task 2.7)."""
    from swimtech.run import RunOptions
    from swimtech.run import run as _run

    _setup_logging(verbose)
    opts = RunOptions(
        src=src,
        race_id=race_id,
        work_dir=work_dir,
        stroke=stroke,  # type: ignore[arg-type]
        swimmer_point=_parse_point(swimmer),
        swimmer_frame=swimmer_frame,
        lane_number=lane,
        lane_box=_parse_box(lane_box),
        calib_corners=_parse_corners(calib_corners),
        detect_model=detect_model,
        to=to,
        force=force,
    )
    from swimtech.models import ModelsNotInstalledError

    try:
        paths = _run(opts)
    except NotImplementedError as e:
        typer.secho(f"stopped: {e}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=2) from e
    except ModelsNotInstalledError as e:
        typer.secho(f"stopped: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=3) from e
    except RuntimeError as e:
        typer.secho(f"stopped: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=4) from e
    typer.echo(f"done up to '{to}'; outputs in {paths.work_dir}")


@app.command()
def frame(
    video: Path = typer.Argument(..., exists=True, readable=True, help="A .norm.mp4"),
    out: Path = typer.Option(Path("frame.jpg"), help="Where to write the JPEG"),
    index: int = typer.Option(0, help="Frame index to extract"),
    grid: int = typer.Option(200, help="Grid spacing in pixels (0 = no grid)"),
) -> None:
    """Save one frame with a labelled pixel grid, to read off --lane-box or lane corners
    by eye until the Vue lane picker (4.2) exists."""
    from swimtech.viz import save_frame_with_grid

    w, h = save_frame_with_grid(video, out, index=index, grid=grid)
    typer.echo(f"wrote {out} ({w}x{h}); coordinates are normalized-video pixels")


@app.command()
def preview(
    race_id: str = typer.Argument(..., help="A race that has been run to the track stage"),
    work_dir: Path = typer.Option(Path("data")),
    out: Path | None = typer.Option(None, help="Default: data/<race_id>.track.mp4"),
) -> None:
    """Write a full-frame video with the tracked box drawn (green = tracked, orange = gap
    filled, red = lost), to check by eye that it followed the right swimmer."""
    import pandas as pd

    from swimtech.types import RacePaths
    from swimtech.viz import save_track_preview

    paths = RacePaths(work_dir=work_dir, race_id=race_id)
    if not paths.crop_offsets.exists():
        raise typer.BadParameter(
            f"{paths.crop_offsets} not found; run `swimtech run ... --to track`"
        )
    target = out or (paths.work_dir / f"{race_id}.track.mp4")
    n = save_track_preview(paths.norm_video, pd.read_parquet(paths.crop_offsets), target)
    typer.echo(f"wrote {target} ({n} frames)")


@app.command()
def mark(
    race_id: str = typer.Argument(..., help="A race that has been ingested"),
    work_dir: Path = typer.Option(Path("data")),
    every: float = typer.Option(2.0, help="Seconds between frames to mark"),
) -> None:
    """Write data/<race_id>.mark.html: open it in a browser, click your swimmer on each
    frame, download the CSV, save it as data/<race_id>.truth.csv, then `swimtech score`."""
    from swimtech.truth import write_mark_page
    from swimtech.types import RacePaths

    paths = RacePaths(work_dir=work_dir, race_id=race_id)
    if not paths.norm_video.exists():
        raise typer.BadParameter(f"{paths.norm_video} not found; run `swimtech ingest` first")
    out = paths.work_dir / f"{race_id}.mark.html"
    n = write_mark_page(paths.norm_video, out, race_id, every_s=every)
    typer.echo(f"wrote {out} with {n} frames; open it in a browser")


@app.command()
def score(
    race_id: str = typer.Argument(..., help="A race run to the track stage, with a truth CSV"),
    work_dir: Path = typer.Option(Path("data")),
    truth: Path | None = typer.Option(None, help="Default: data/<race_id>.truth.csv"),
) -> None:
    """Score the tracked crop against the clicks from `swimtech mark`."""
    import json

    import pandas as pd

    from swimtech.truth import load_truth
    from swimtech.truth import score as _score
    from swimtech.types import RacePaths

    paths = RacePaths(work_dir=work_dir, race_id=race_id)
    truth_path = truth or (paths.work_dir / f"{race_id}.truth.csv")
    if not truth_path.exists():
        raise typer.BadParameter(f"{truth_path} not found; download it from the mark page")
    if not paths.crop_offsets.exists():
        raise typer.BadParameter(
            f"{paths.crop_offsets} not found; run `swimtech run ... --to track`"
        )
    fps = float(json.loads(paths.probe_json.read_text())["target_fps"])
    summary, table = _score(pd.read_parquet(paths.crop_offsets), load_truth(truth_path), fps)
    for _, r in table.iterrows():
        typer.echo(f"{r['sec']:6.1f}s  {r['verdict']}")
    typer.echo("")
    typer.echo(
        f"on target {summary['on_target']:.0%}   wrong target {summary['wrong_target']:.0%}   "
        f"lost while visible {summary['lost_while_visible']:.0%}   "
        f"({summary['visible_marks']} visible marks of {summary['marks']}; "
        f"{summary['false_box']} boxes shown while not visible)"
    )


@app.command()
def metrics() -> None:
    """List the metric registry (docs/metrics.md)."""
    from swimtech.metrics.registry import REGISTRY

    for spec in REGISTRY.values():
        arm = "L/R" if spec.per_arm else "-"
        typer.echo(
            f"{spec.name:28s} {arm:4s} {str(spec.instant):7s} {spec.unit:9s} "
            f"{spec.space:6s} {spec.description}"
        )


if __name__ == "__main__":
    app()
