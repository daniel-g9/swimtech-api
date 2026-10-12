# swimtech-api

The AI pipeline and HTTP service behind the swim technique app. Freestyle first; other strokes reuse the same pipeline with different cycle rules (plan task 7.6).

The plan of record lives in the Vue repo at `swim_app_draft/docs/BUILD_PLAN.md`. Task ids in this repo's comments (1.1, 2.3, ...) refer to it.

## Layout

```
swimtech/          the pipeline: one module per stage, pure functions, no global state
  types.py         keypoint order, constants, dataclasses shared by every stage
  ingest.py        1.1  phone clip -> <race_id>.norm.mp4 + probe JSON
  quality.py       1.2  blur / glare / shake warnings
  detect.py        1.3  RF-DETR person boxes
  track.py         1.4  ByteTrack + lane constraint -> steady crop
  calib.py         1.5  lane-plane homography for _rect metrics
  pose.py          2.1  RTMPose 13 keypoints per frame
  smooth.py        2.2  Savitzky-Golay + gap fill + visibility
  cycles.py        2.3  stroke cycle and phase segmentation
  metrics/         0.1 + 2.4  metric registry and implementations
  findings.py      2.5  rule-based findings
  render.py        2.6  annotated video
  run.py           2.7  stage runner with caching
  cli.py           `swimtech ingest`, `swimtech run`
api/main.py        2.7  FastAPI service
scripts/           eval, prelabel, train (Stage 3)
tests/             pytest; stages are tested alone on synthetic inputs
docs/              metrics.md (the spec), setup.md
samples/           sample.results.json for the Vue app to build against
```

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
