"""Ground truth for the tracker, marked by a human, and a score against it.

`swimtech mark <race>` writes data/<race>.mark.html: one frame every `every_s`
seconds with a click target. The user clicks their swimmer on each frame (or marks them
not visible) and downloads data/<race>.truth.csv with columns frame, x, y, visible in
normalized-video pixels. `swimtech score <race>` then reports how often the tracked crop
box contained the clicked point, how often the swimmer was lost while visible, and how
often a box was shown while the swimmer was not visible.

Teaching notes
--------------
Eyeballing a crop video does not tell you whether a change helped. Thirty clicks give a
number that does, and the same clicks score every future version of the tracker.
Scoring on the crop box (not the detector box) is deliberate: the crop is what the pose
model sees, so "the click is inside the crop" is exactly the question that matters.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TRUTH_COLUMNS = ["frame", "x", "y", "visible"]


def sample_seconds(duration_s: float, every_s: float) -> list[float]:
    n = int(np.floor(duration_s / every_s)) + 1
    return [round(i * every_s, 3) for i in range(n) if i * every_s < duration_s]


def _frames_as_jpeg(video: Path, every_s: float, max_width: int) -> tuple[list[dict], int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_width / w)
    out = []
    for sec in sample_seconds(n / fps, every_s):
        f = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        if scale < 1.0:
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(
            {
                "frame": f,
                "sec": sec,
                "src": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
            }
        )
    cap.release()
    return out, w, h


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Mark the swimmer: __RACE__</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:10px 16px;background:#222;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 button{font-size:15px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer}
 #wrap{position:relative;display:inline-block;margin:12px 16px;cursor:crosshair}
 #img{display:block;max-width:calc(100vw - 32px);max-height:calc(100vh - 120px)}
 .dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border:3px solid #0f0;border-radius:50%;pointer-events:none}
 #help{padding:0 16px 12px;color:#bbb;max-width:900px}
 .done{color:#8f8}.todo{color:#fc6}
</style></head><body>
<header>
 <b>__RACE__</b>
 <span id="pos"></span>
 <button id="prev">&larr; Prev</button>
 <button id="next">Next &rarr;</button>
 <button id="nv" style="background:#a33;color:#fff">Not visible</button>
 <button id="dl" style="background:#2a7;color:#fff">Download CSV</button>
 <span id="status"></span>
</header>
<div id="help">Click on your swimmer (anywhere on their body) in each frame, then Next.
If they are not visible or you cannot tell which one is them, press <b>Not visible</b>.
Clicking again moves the mark. Your clicks are saved in this browser as you go; when every
frame is done, press <b>Download CSV</b> and move the file to <code>data/__RACE__.truth.csv</code>.</div>
<div id="wrap"><img id="img"><div id="dot" class="dot" hidden></div></div>
<script>
const FRAMES = __FRAMES__;
const W = __W__, H = __H__, KEY = "swimtech-truth-__RACE__";
let i = 0, marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const img = document.getElementById("img"), dot = document.getElementById("dot");
function save(){ try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} }
function show(){
  const fr = FRAMES[i]; img.src = fr.src;
  const m = marks[fr.frame];
  const done = Object.keys(marks).length;
  document.getElementById("pos").textContent = `frame ${i+1} / ${FRAMES.length}  (${fr.sec.toFixed(1)} s)`;
  document.getElementById("status").innerHTML = `<span class="${done===FRAMES.length?'done':'todo'}">${done} of ${FRAMES.length} marked</span>` + (m ? (m.visible ? " &middot; marked" : " &middot; not visible") : "");
  if (m && m.visible) { dot.hidden = false; placeDot(m.x, m.y); } else { dot.hidden = true; }
}
function placeDot(x, y){ const r = img.getBoundingClientRect(); dot.style.left = (x / W * r.width) + "px"; dot.style.top = (y / H * r.height) + "px"; }
img.addEventListener("click", e => {
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H;
  marks[FRAMES[i].frame] = {x: Math.round(x), y: Math.round(y), visible: 1}; save(); show();
});
document.getElementById("nv").onclick = () => { marks[FRAMES[i].frame] = {x: "", y: "", visible: 0}; save(); if (i < FRAMES.length-1) i++; show(); };
document.getElementById("next").onclick = () => { if (i < FRAMES.length-1) i++; show(); };
document.getElementById("prev").onclick = () => { if (i > 0) i--; show(); };
document.addEventListener("keydown", e => { if (e.key === "ArrowRight") document.getElementById("next").click(); if (e.key === "ArrowLeft") document.getElementById("prev").click(); if (e.key === "n") document.getElementById("nv").click(); });
document.getElementById("dl").onclick = () => {
  let csv = "frame,x,y,visible\\n";
  for (const fr of FRAMES) { const m = marks[fr.frame]; if (m) csv += `${fr.frame},${m.x},${m.y},${m.visible}\\n`; }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([csv], {type: "text/csv"})); a.download = "__RACE__.truth.csv"; a.click();
};
window.addEventListener("resize", show);
show();
</script></body></html>
"""


def write_mark_page(
    norm_video: Path, out_html: Path, race_id: str, every_s: float = 2.0, max_width: int = 720
) -> int:
    """Write the clicking page. Returns the number of frames in it."""
    frames, w, h = _frames_as_jpeg(norm_video, every_s, max_width)
    page = (
        _PAGE.replace("__RACE__", race_id)
        .replace("__FRAMES__", json.dumps(frames))
        .replace("__W__", str(w))
        .replace("__H__", str(h))
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return len(frames)


def load_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["frame"] = df["frame"].astype(int)
    df["visible"] = df["visible"].astype(int)
    return df[TRUTH_COLUMNS]


def score(crops: pd.DataFrame, truth: pd.DataFrame, fps: float) -> tuple[dict, pd.DataFrame]:
    """Compare the crop boxes to the clicks. Returns (summary, per-mark table)."""
    by_frame = crops.set_index("frame")
    rows = []
    for _, t in truth.iterrows():
        f = int(t["frame"])
        if f not in by_frame.index:
            continue
        c = by_frame.loc[f]
        shown = int(c["tracked"]) > 0
        if t["visible"] == 1:
            x, y = float(t["x"]), float(t["y"])
            inside = c["x0"] <= x <= c["x0"] + c["w"] and c["y0"] <= y <= c["y0"] + c["h"]
            if not shown:
                verdict = "lost"
            elif inside:
                verdict = "on target"
            else:
                verdict = "wrong target"
        else:
            verdict = "false box" if shown else "correctly lost"
        rows.append({"sec": round(f / fps, 1), "frame": f, "verdict": verdict})
    table = pd.DataFrame(rows, columns=["sec", "frame", "verdict"])
    vis = table[table["verdict"].isin(["on target", "wrong target", "lost"])]
    n_vis = max(len(vis), 1)
    summary = {
        "marks": len(table),
        "visible_marks": len(vis),
        "on_target": round(float((vis["verdict"] == "on target").mean()), 3) if len(vis) else 0.0,
        "wrong_target": round(float((vis["verdict"] == "wrong target").sum() / n_vis), 3),
        "lost_while_visible": round(float((vis["verdict"] == "lost").sum() / n_vis), 3),
        "false_box": int((table["verdict"] == "false box").sum()),
    }
    return summary, table
