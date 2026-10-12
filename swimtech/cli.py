"""Command line entry point: `uv run swimtech <command>`.

swimtech ingest IMG_1234.MOV --race-id meet1_r3
swimtech run clip.MOV --race-id test1 --lane 4 --to track
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
    lane: int | None = typer.Option(None, help="Lane number (needs calibration)"),
    lane_box: str | None = typer.Option(None, help="x0,y0,x1,y1 on the first frame"),
    to: str = typer.Option("results", help="Stop after this stage"),
    force: bool = typer.Option(False, help="Ignore cached outputs"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run the pipeline end to end, or up to --to (plan task 2.7)."""
    from swimtech.run import RunOptions
    from swimtech.run import run as _run

    _setup_logging(verbose)
    box = tuple(float(v) for v in lane_box.split(",")) if lane_box else None
    if box is not None and len(box) != 4:
        raise typer.BadParameter("--lane-box needs four numbers")
    opts = RunOptions(
        src=src,
        race_id=race_id,
        work_dir=work_dir,
        stroke=stroke,  # type: ignore[arg-type]
        lane_number=lane,
        lane_box=box,  # type: ignore[arg-type]
        to=to,
        force=force,
    )
    try:
        paths = _run(opts)
    except NotImplementedError as e:
        typer.secho(f"stopped: {e}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=2) from e
    typer.echo(f"done up to '{to}'; outputs in {paths.work_dir}")


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
