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

Open a new terminal afterwards so PATH picks them up. Check:

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

Goal: a crop video that stays on your swimmer. About 30 minutes. CPU is fine.

1. Install the model dependencies once: `uv sync --extra models`.
2. Normalize the clip and save its first frame with a pixel grid:

   ```
   uv run swimtech ingest clip.MOV --race-id test1
   uv run swimtech frame data\test1.norm.mp4 --out first.jpg
   ```

3. Open `first.jpg`. Read off a box that encloses your swimmer's whole lane for the
   full race (both walls, a little margin above and below the lane ropes). The grid
   labels are pixel coordinates. Write it down as `x0,y0,x1,y1` (left, top, right,
   bottom), for example `120,640,3800,980`.

4. Run detection and tracking (this is the slow part: a 2-minute 4K clip takes about
   10 to 30 minutes on a laptop CPU with the default `rfdetr-medium`; add
   `--detect-model rfdetr-nano` for a 5x faster first look):

   ```
   uv run swimtech run clip.MOV --race-id test1 --lane 4 --lane-box 120,640,3800,980 --to track
   ```

   `--lane 4` is only a label for the results; `--lane-box` is what picks the swimmer.
   If you leave both out, the tracker follows the longest-visible swimmer and warns.

5. Open `data\test1.crop.mp4`. Tell me: does it stay on the swimmer? Where does it
   jump? Paste the console log. `data\test1.crop.parquet` has the crop position per
   frame and a `tracked` column (2 = detected, 1 = short gap filled, 0 = held).

Re-running the same command skips stages whose outputs are already there; add
`--force` to redo everything.

## 5. Secrets

Copy `.env.example` to `.env` and fill it in. Nothing in this repo reads a secret from
code; `python-dotenv` loads `.env` in local dev and Modal/Supabase secret stores do it
in production (Stage 5).

## If anything fails

Paste the full error and stop. Don't debug the install yourself unless you want to.
Two errors you may see and what they mean:

- `ffmpeg not found on PATH`: step 1 above; open a new terminal after installing.
- `rfdetr is not installed. Run uv sync --extra models`: step 4.1 above.
