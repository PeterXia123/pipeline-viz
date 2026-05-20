"""example_good：多段 py、扇入/扇出、链式 data、无 I/O 的 helper。"""
from __future__ import annotations

from pathlib import Path

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.layout import layered_positions
from pipeline_viz.paths import code_id, data_id

ROOT = Path(__file__).resolve().parent.parent / "example_good"
MANIFEST = ROOT / "pipeline-manifest.yaml"


def test_complex_example_graph_smoke():
    g = build_graph(ROOT, MANIFEST)
    assert len(g.nodes) == 11
    assert len(g.edges) == 12

    nb = "notebooks/pipeline.ipynb"
    calls = [e.target for e in g.edges if e.kind == "notebook_call" and e.source == code_id(nb)]
    assert calls == [
        code_id("pipeline/ingest.py"),
        code_id("pipeline/transform.py"),
        code_id("pipeline/report.py"),
        code_id("lib/helper.py"),
    ]

    es = {(x.source, x.target, x.kind) for x in g.edges}
    assert (data_id("data/raw/orders.csv"), code_id("pipeline/ingest.py"), "input") in es
    assert (data_id("data/staging/merged_orders.csv"), code_id("pipeline/transform.py"), "input") in es
    assert (data_id("data/out/summary.csv"), code_id("pipeline/report.py"), "input") in es

    pos = layered_positions(g.nodes, g.edges)
    assert len(pos) == 11
