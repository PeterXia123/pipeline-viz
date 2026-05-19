"""单 notebook 串联本地 .py：无数据传递的调用边 + 顺序。"""
from __future__ import annotations

from pathlib import Path

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id

ROOT = Path(__file__).resolve().parent.parent / "example_project_nb_py"
MANIFEST = ROOT / "pipeline-manifest.yaml"


def test_notebook_python_call_edges_order():
    g = build_graph(ROOT, MANIFEST)
    nb = "notebooks/run.ipynb"
    nb_id = code_id(nb)

    calls = [e for e in g.edges if e.kind == "notebook_call" and e.source == nb_id]
    targets = [e.target for e in calls]
    assert targets == [code_id("src/step_a.py"), code_id("src/step_b.py")]

    kinds = {n.id: n.kind for n in g.nodes}
    assert kinds[nb_id] == "notebook"
    assert kinds[code_id("src/step_a.py")] == "python"
    assert kinds[code_id("src/step_b.py")] == "python"

    edges = {(e.source, e.target) for e in g.edges}
    assert (data_id("data/in.csv"), nb_id) in edges
    assert (nb_id, data_id("data/out/result.csv")) in edges


def test_layout_covers_all_kinds():
    from pipeline_viz.layout import layered_positions

    g = build_graph(ROOT, MANIFEST)
    pos = layered_positions(g.nodes, g.edges)
    for n in g.nodes:
        assert n.id in pos
