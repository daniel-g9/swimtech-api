"""Command line entry point: `uv run swimtech <command>`.

swimtech ingest IMG_1234.MOV --race-id meet1_r3
swimtech frame data/test1.norm.mp4 --out first.jpg      # pick a lane box by eye
swimtech run clip.MOV --race-id test1 --lane-box 120,640,3800,980 --to track
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
    lane: int | None = typer.Option(None, help="Lane number (a label; see --lane-box)"),
    lane_box: str | None = typer.Option(
        None, help="x0,y0,x1,y1 on the first frame enclosing the swimmer's lane"
    ),
    calib_corners: str | None = typer.Option(
        None,
        help="Lane corners 'x,y;x,y;x,y;x,y' (near-left, near-right, far-right, far-left)",
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
