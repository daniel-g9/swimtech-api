import pandas as pd

from swimtech.track import TRACKED_DETECTED, TRACKED_HELD
from swimtech.truth import load_truth, sample_seconds, score, write_mark_page
from tests.conftest import blob_frames, write_video


def test_sample_seconds():
    assert sample_seconds(9.8, 2.0) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert sample_seconds(4.0, 2.0) == [0.0, 2.0]


def test_mark_page_has_every_frame(tmp_path):
    video = write_video(
        tmp_path / "v.mp4", blob_frames(90, 320, 240, lambda f: [(20 + f, 110, 60 + f, 130)])
    )
    n = write_mark_page(video, tmp_path / "v.mark.html", "v", every_s=1.0, max_width=160)
    assert n == 3
    html = (tmp_path / "v.mark.html").read_text(encoding="utf-8")
    assert html.count("data:image/jpeg;base64,") == 3
    assert '"frame": 60' in html and "v.truth.csv" in html
    assert "const W = 320, H = 240" in html  # clicks map back to full-size pixels


def test_score_verdicts(tmp_path):
    crops = pd.DataFrame(
        {
            "frame": [0, 30, 60, 90],
            "x0": [10, 10, 10, 10],
            "y0": [10, 10, 10, 10],
            "w": 50,
            "h": 50,
            "scale": 1.0,
            "tracked": [TRACKED_DETECTED, TRACKED_DETECTED, TRACKED_HELD, TRACKED_DETECTED],
        }
    )
    csv = tmp_path / "t.csv"
    csv.write_text("frame,x,y,visible\n0,30,30,1\n30,200,200,1\n60,30,30,1\n90,,,0\n")
    summary, table = score(crops, load_truth(csv), fps=30)
    assert table["verdict"].tolist() == ["on target", "wrong target", "lost", "false box"]
    assert summary["visible_marks"] == 3
    assert summary["on_target"] == round(1 / 3, 3)
    assert summary["wrong_target"] == round(1 / 3, 3)
    assert summary["lost_while_visible"] == round(1 / 3, 3)
    assert summary["false_box"] == 1
