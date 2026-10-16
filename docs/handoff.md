# Handoff: where Stage 1 stands (2026-10-15)

## What works now

- Moving camera: per-frame camera motion (`motion.py`), water mask, and a per-frame
  lane corridor found from the ropes (`lanes.py`), so a handheld, panning, zooming
  camera is fine. The swimmer is picked by one tap (`--swimmer x,y`); inside the
  corridor a single-target tracker follows them (`track.py`, `_track_in_corridor`).
- Scoring against human clicks: `swimtech mark <race>` makes a click page, the CSV
  goes to `data/<race>.truth.csv`, `swimtech score <race>` prints on-target %.
  Every tracker change is scored on both clips before it is reported.

## Scores (crop box contains the clicked swimmer)

| Clip | On target | Wrong target | Lost while visible |
|---|---|---|---|
| `test_1` (IMG_2295, full race, 8 swimmers) | 96% | 4% | 0% |
| `race2` (IMG_2583, heavy panning) | 77% | 23% | 0% |
| `solo` (one swimmer) | tracked on 97% of frames, no gaps | | |

Clip 2's remaining misses are frames where the detector has no box on the swimmer
(18-20 s) and the coasting guess is off, plus one false box after the camera leaves
the pool. Next lever there is detector recall (Stage 3 fine-tuning), not tracking.

## Commands

```
uv run swimtech ingest clip.MOV --race-id race1
uv run swimtech frame data\race1.norm.mp4 --out race1_first.jpg
uv run swimtech run clip.MOV --race-id race1 --swimmer 560,890 --to track
uv run swimtech preview race1          # full-frame video with the box drawn
uv run swimtech mark race1             # click page -> data\race1.truth.csv
uv run swimtech score race1
```

## Next

1. Stage 2 (pose) on the `solo` clip: one clean track, full body in the crop.
2. More truth clips as they are filmed; keep both scores from dropping.
3. Stage 3 detector fine-tuning if recall on small swimmers stays the limit.
