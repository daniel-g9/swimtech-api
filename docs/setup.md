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

**ffmpeg** (video normalization, task 1.1):

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

Model dependencies are optional and heavy (torch is several GB). Only add them when a
stage needs them:

```
uv sync --extra models
```

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

## 4. Secrets

Copy `.env.example` to `.env` and fill it in. Nothing in this repo reads a secret from
code; `python-dotenv` loads `.env` in local dev and Modal/Supabase secret stores do it
in production (Stage 5).

## If anything fails

Paste the full error and stop. Don't debug the install yourself unless you want to.
