from pathlib import Path

from pipeline_viz.dataframe_compare import run_datacompy_compare


def test_row_order_compare(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("a,b\n1,x\n", encoding="utf-8")
    b.write_text("a,b\n1,y\n", encoding="utf-8")
    r = run_datacompy_compare(a, b, None)
    assert r.get("matches") is False
    assert "report" in r and len(r["report"]) > 10


def test_merge_key_compare(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("id,v\n1,a\n", encoding="utf-8")
    b.write_text("id,v\n1,b\n", encoding="utf-8")
    r = run_datacompy_compare(a, b, ["id"])
    assert r.get("matches") is False
