"""The results JSON is the contract with the Vue app. These tests pin its shape so a
change on either side breaks the build, not a screen."""

import json
from pathlib import Path

from swimtech.metrics.registry import REGISTRY
from swimtech.types import VIS_GREY_THRESHOLD, Cycle, Finding, MetricValue, Results

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample.results.json"
TOP_LEVEL = {
    "race_id",
    "fps",
    "n_frames",
    "stroke",
    "keypoints",
    "cycles",
    "metrics",
    "findings",
    "media",
}


def test_sample_has_the_contract_fields():
    data = json.loads(SAMPLE.read_text())
    assert TOP_LEVEL <= set(data)
    for c in data["cycles"]:
        assert {"arm", "n", "entry_f", "exit_f", "length"} <= set(c)
        assert c["arm"] in ("L", "R")
        assert c["entry_f"] < c["exit_f"]
    for m in data["metrics"]:
        assert {"cycle", "name", "value", "unit", "vis"} <= set(m)
        assert 0.0 <= m["vis"] <= 1.0
    for f in data["findings"]:
        assert {"frame", "severity", "text"} <= set(f)
        assert f["severity"] in ("low", "med", "high")


def test_sample_metric_names_exist_in_registry():
    data = json.loads(SAMPLE.read_text())
    for m in data["metrics"]:
        base = m["name"].removesuffix("_asym").removesuffix("_slope")
        assert base in REGISTRY, m["name"]


def test_sample_has_a_greyed_out_metric_for_the_ui_to_handle():
    data = json.loads(SAMPLE.read_text())
    assert any(m["vis"] < VIS_GREY_THRESHOLD for m in data["metrics"])


def test_results_dataclass_serializes_to_contract():
    r = Results(
        race_id="x",
        fps=60,
        n_frames=100,
        stroke="free",
        keypoints="x.kp.parquet",
        cycles=[Cycle(arm="L", n=1, entry_f=10, exit_f=50, length=1, midrec_f=30)],
        metrics=[
            MetricValue(cycle="L1", name="elbow_angle_exit", value=118.4444, unit="deg", vis=0.9)
        ],
        findings=[Finding(frame=30, severity="low", text="t", metric="elbow_angle_exit")],
        media={"annotated": "x.annot.mp4", "thumb": "x.jpg"},
    )
    out = r.to_json()
    assert TOP_LEVEL <= set(out)
    assert out["cycles"][0] == {
        "arm": "L",
        "n": 1,
        "entry_f": 10,
        "exit_f": 50,
        "length": 1,
        "midrec_f": 30,
    }
    assert out["metrics"][0]["value"] == 118.444
    assert out["findings"][0]["metric"] == "elbow_angle_exit"
    json.dumps(out)  # must be plain JSON
