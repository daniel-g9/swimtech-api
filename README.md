# swimtech-api

The AI pipeline and HTTP service behind the swim technique app. Freestyle first; other strokes reuse the same pipeline with different cycle rules (plan task 7.6).

The plan of record lives in the Vue repo at `swim_app_draft/docs/BUILD_PLAN.md`. Task ids in this repo's comments (1.1, 2.3, ...) refer to it.

## Layout

```
swimtech/          the pipeline: one module per stage, pure functions, no global state
  types.py         keypoint order, constants, dataclasses shared by every stage
  ingest.py        1.1  phone clip -> <race_id>.norm.mp4 + probe JSON
  quality.py       1.2  blur / glare / shake warnings
  detect.py        1.3  RF-DETR person boxes at 15 fps, interpolated to every frame
  track.py         1.4  ByteTrack-style association + lane constraint -> steady crop
  calib.py         1.5  lane-plane homography for _rect metrics
  models.py        loads / exports the detector (and later the pose model); lazy torch
  viz.py           gridded frame export for picking a lane box by eye (task 1.6)
  pose.py          2.1  RTMPose 13 keypoints per frame
  smooth.py        2.2  Savitzky-Golay + gap fill + visibility
  cycles.py        2.3  stroke cycle and phase segmentation
  metrics/         0.1 + 2.4  metric registry and implementations
  findings.py      2.5  rule-based findings
  render.py        2.6  annotated video
  run.py           2.7  stage runner with caching
  cli.py           `swimtech ingest`, `swimtech frame`, `swimtech run`, `swimtech metrics`
api/main.py        2.7  FastAPI service
scripts/           eval, prelabel, train (Stage 3)
tests/             pytest; stages are tested alone on synthetic inputs (no footage needed)
docs/              metrics.md (the spec), setup.md (including the task 1.6 walkthrough)
samples/           sample.results.json for the Vue app to build against
```

## Status

| Stage | Task | State |
|---|---|---|
| 0 | 0.1 metric spec | done (`docs/metrics.md`, registry) |
| 1 | 1.1 ingest | done, tested with a synthetic clip |
| 1 | 1.2 quality | done, tested |
| 1 | 1.3 detector | done, tested with a fake detector; real RF-DETR call unverified until footage |
| 1 | 1.4 tracker + crop | done, tested on synthetic two-lane sequences |
| 1 | 1.5 calibration | done, tested |
| 1 | 1.6 run on one clip | yours; see `docs/setup.md` section 4 |
| 2 | 2.2 smoothing | stage function done; wiring lands with 2.1 |
| 2 | everything else | stubs |

## Quick start

See `docs/setup.md`. In short:

```
uv sync
uv run pytest
uv run swimtech ingest path/to/clip.MOV --race-id test1
```

## Conventions

- Every stage is `stage(input) -> output` with no global state.
- Frame index is the primary key everywhere; seconds are `frame / fps`.
- 13 keypoints in the order defined in `swimtech/types.py`. Never reorder.
- Coordinates are always in the normalized video's pixel frame, never the crop.
- Every metric carries `vis` (0-1). No metric is reported without it.
- Permissive licenses only. No Ultralytics (AGPL), no VideoPose3D (non-commercial).
- Heavy model imports (torch, rfdetr) stay inside functions so the CLI, API and tests
  work on a laptop without `--extra models`.
