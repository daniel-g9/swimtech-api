# Setup

Exact commands for a laptop. CPU is enough for Stage 1 (task 1.6). A GPU is only
needed for pose at scale, which runs on Modal (task 5.2).

## 1. Tools

**uv** (Python environment manager; it also downloads the right Python for you):

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**ffmpeg** (video normalization, task 1.1, and the crop video, 1.4):

```powershell
# Windows
winget install Gyan.FFmpeg
```
```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
```

Close every terminal (and VS Code) and reopen afterwards so PATH picks them up. Check:

```
uv --version
ffmpeg -version
```

## 2. Environment

```
cd swimtech-api
uv sync            # creates .venv with Python 3.12 and the base dependencies
uv run pytest      # should pass; ffmpeg-dependent tests skip if ffmpeg is missing
```

Model dependencies are optional and heavy (torch is several GB). The detector (1.3)
needs them; ingest, quality and the tests do not:

```
uv sync --extra models
```

The first `swimtech run` downloads the RF-DETR weights (a few hundred MB) into the
Hugging Face cache; later runs are offline.

## 3. Running

```
uv run swimtech ingest path\to\clip.MOV --race-id test1
uv run swimtech run path\to\clip.MOV --race-id test1 --to quality
uv run swimtech metrics
uv run uvicorn api.main:app --reload --port 8000     # then open http://localhost:8000/docs
```

Outputs land in `data/<race_id>.*`. `data/` is git-ignored; never commit video.

The API serves a fake race for front-end work before the pipeline exists:
`GET http://localhost:8000/races/sample` returns `samples/sample.results.json`.

## 4. Task 1.6: run Stage 1 on one clip

Goal: a crop video that stays on your swimmer. About 30 minutes. CPU is fine. The
camera may be handheld, panning and zooming; the pipeline compensates for that.

1. Install the model dependencies once: `uv sync --extra models`.
2. Normalize the clip and save its first frame with a pixel grid:

   ```
   uv run swimtech ingest C:\Users\you\Downloads\IMG_1234.MOV --race-id race1
   uv run swimtech frame data\race1.norm.mp4 --out race1_first.jpg
   ```

3. Open `race1_first.jpg`. The yellow lines are a pixel ruler, labelled every 200 px.
   Find your swimmer and read off a point on their body, for example the cap or the
   middle of the back, as `x,y`. If they are not visible on the first frame (still
   under water after the start), add `--index 150` to the frame command to look at a
   later frame, and pass that frame number as `--swimmer-frame` below.

4. Run detection, camera motion and tracking (the slow part: roughly 10 to 20 minutes
   on a laptop CPU per minute of video with the default `rfdetr-medium`; add
   `--detect-model rfdetr-nano` for a 5x faster, less accurate first look):

   ```
   uv run swimtech run C:\Users\you\Downloads\IMG_1234.MOV --race-id race1 --lane 4 --swimmer 560,890 --to track
   ```

   `--lane 4` is only a label for the results. `--swimmer x,y` is what picks the
   swimmer. If you leave it out, the tracker follows the longest-visible swimmer and
   warns. (`--lane-box` and `--calib-corners` still exist for a camera on a tripod.)

5. Make the review video and watch it:

   ```
   uv run swimtech preview race1
   ```

   Open `data\race1.track.mp4`: the full frame with a box drawn on the swimmer we
   followed, green while tracked, orange across a short gap, red when lost (the box
   then just holds the last position). Tell me where the green box is on the wrong
   swimmer, and where it goes red while your swimmer is clearly visible. Paste the
   console log. `data\race1.crop.mp4` is the close-up the pose model will see, and
   `data\race1.probe.json` has the quality and tracking reports: how much of the race
   the swimmer was found in, the longest gap, and how big they were in pixels.

Re-running the same command reuses the slow detection step; only tracking repeats, in
seconds. So trying a different `--swimmer` point is cheap. Add `--force` to redo
everything.

## 5. Secrets

Copy `.env.example` to `.env` and fill it in. Nothing in this repo reads a secret from
code; `python-dotenv` loads `.env` in local dev and Modal/Supabase secret stores do it
in production (Stage 5).

## If anything fails

Paste the full error and stop. Don't debug the install yourself unless you want to.
Errors you may see and what they mean:

- `ffmpeg not found on PATH`: step 1 above; close and reopen VS Code after installing.
- `rfdetr is not installed. Run uv sync --extra models`: step 4.1 above.
- `Invalid value for 'src': Path ... does not exist`: the clip path is wrong. Drag the
  file into the terminal to paste its exact path.
- `no track near the tapped point`: the swimmer was not detected on that frame. Try a
  point on a frame where they are clearly visible (`--swimmer-frame`).
