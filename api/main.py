"""FastAPI service (plan task 2.7).

Endpoints:
    POST  /races                      register a race, get an upload target
    POST  /races/{id}/process         start processing (lane, calibration)
    GET   /races/{id}                 status + results JSON when done
    GET   /races/{id}/media/{kind}    norm | annot | thumb
    PATCH /races/{id}/keypoints       corrections from the Vue correction tool (4.6)
    GET   /health

Local dev runs processing in-process in a background task; task 5.2 moves it to a GPU
worker. State is an in-memory dict here and moves to Postgres in 5.4.

Run:  uv run uvicorn api.main:app --reload --port 8000   (docs at /docs)
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from swimtech import __version__
from swimtech.types import RacePaths, Stroke

DATA_DIR = Path(os.environ.get("SWIMTECH_DATA_DIR", "data"))
SAMPLE_RESULTS = Path(__file__).resolve().parent.parent / "samples" / "sample.results.json"

app = FastAPI(title="swimtech-api", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SWIMTECH_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

Status = Literal["uploaded", "queued", "processing", "done", "failed"]


class RaceCreate(BaseModel):
    stroke: Stroke = "free"
    swimmer: str | None = None
    filename: str


class RaceCreated(BaseModel):
    race_id: str
    upload_path: str  # local dev: a path on disk. 5.3 replaces this with a presigned URL.


class ProcessRequest(BaseModel):
    lane_number: int | None = None
    swimmer_point: tuple[float, float] | None = None  # tap on the swimmer (normal way)
    swimmer_frame: int = 0  # which frame the tap was made on
    lane_box: tuple[float, float, float, float] | None = None  # fixed camera only
    calib_corners: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)


class KeypointPatch(BaseModel):
    frame: int
    keypoint: int = Field(ge=0, le=12)
    x: float | None = None
    y: float | None = None
    vis: int | None = Field(default=None, ge=0, le=2)


class RaceState(BaseModel):
    race_id: str
    status: Status
    stage: str | None = None
    error: str | None = None
    results: dict | None = None


_RACES: dict[str, RaceState] = {}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.post("/races", response_model=RaceCreated)
def create_race(body: RaceCreate) -> RaceCreated:
    race_id = uuid.uuid4().hex[:12]
    paths = RacePaths(work_dir=DATA_DIR, race_id=race_id)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    upload = paths.work_dir / f"{race_id}.src{Path(body.filename).suffix.lower() or '.mp4'}"
    _RACES[race_id] = RaceState(race_id=race_id, status="uploaded")
    return RaceCreated(race_id=race_id, upload_path=str(upload))


def _process(race_id: str, req: ProcessRequest) -> None:
    from swimtech.run import RunOptions, run

    state = _RACES[race_id]
    paths = RacePaths(work_dir=DATA_DIR, race_id=race_id)
    src = next(paths.work_dir.glob(f"{race_id}.src.*"), None)
    if src is None:
        state.status, state.error = "failed", "no uploaded file"
        return
    state.status = "processing"
    try:
        run(
            RunOptions(
                src=src,
                race_id=race_id,
                work_dir=DATA_DIR,
                lane_number=req.lane_number,
                swimmer_point=req.swimmer_point,
                swimmer_frame=req.swimmer_frame,
                lane_box=req.lane_box,
                calib_corners=req.calib_corners,
            )
        )
        state.results = json.loads(paths.results_json.read_text())
        state.status = "done"
    except NotImplementedError as e:
        state.status, state.error = "failed", str(e)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI; Sentry in 5.5
        state.status, state.error = "failed", f"{type(e).__name__}: {e}"


@app.post("/races/{race_id}/process", response_model=RaceState)
def process_race(race_id: str, body: ProcessRequest, bg: BackgroundTasks) -> RaceState:
    state = _RACES.get(race_id)
    if state is None:
        raise HTTPException(404, "unknown race")
    if state.status in ("queued", "processing"):
        raise HTTPException(409, "already processing")
    state.status = "queued"
    bg.add_task(_process, race_id, body)
    return state


@app.get("/races/{race_id}", response_model=RaceState)
def get_race(race_id: str) -> RaceState:
    if race_id == "sample":
        return RaceState(
            race_id="sample", status="done", results=json.loads(SAMPLE_RESULTS.read_text())
        )
    state = _RACES.get(race_id)
    if state is None:
        raise HTTPException(404, "unknown race")
    return state


_MEDIA = {"norm": "norm_video", "annot": "annotated_video", "thumb": "thumbnail"}


@app.get("/races/{race_id}/media/{kind}")
def get_media(race_id: str, kind: str) -> FileResponse:
    if kind not in _MEDIA:
        raise HTTPException(404, "kind must be norm | annot | thumb")
    path: Path = getattr(RacePaths(work_dir=DATA_DIR, race_id=race_id), _MEDIA[kind])
    if not path.exists():
        raise HTTPException(404, "not rendered yet")
    return FileResponse(path)


@app.patch("/races/{race_id}/keypoints", response_model=RaceState)
def patch_keypoints(race_id: str, patches: list[KeypointPatch]) -> RaceState:
    state = _RACES.get(race_id)
    if state is None:
        raise HTTPException(404, "unknown race")
    raise HTTPException(
        501, "plan task 4.6: keypoint corrections re-run stages 2.2-2.5 for the window"
    )
