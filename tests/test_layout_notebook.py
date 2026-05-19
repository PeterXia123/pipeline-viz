"""布局：notebook 独占列避免与 source data 重叠；notebook 纵向跨度与所调用 py 的 y 范围一致；无 notebook→中间 data 的伪连接边。"""
from __future__ import annotations

from pathlib import Path

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.layout import compute_layout
from pipeline_viz.paths import code_id, data_id

ROOT_COMPLEX = Path(__file__).resolve().parent.parent / "example_complex_viz"
MANIFEST_COMPLEX = ROOT_COMPLEX / "pipeline-manifest.yaml"

ROOT_NB_PY = Path(__file__).resolve().parent.parent / "example_project_nb_py"
MANIFEST_NB_PY = ROOT_NB_PY / "pipeline-manifest.yaml"


def test_notebook_column_separate_from_source_data():
    g = build_graph(ROOT_COMPLEX, MANIFEST_COMPLEX)
    L = compute_layout(g.nodes, g.edges)
    pos = L["positions"]
    nb = code_id("notebooks/pipeline.ipynb")
    cust = data_id("data/raw/customers.csv")
    assert pos[nb][0] != pos[cust][0]
    assert pos[cust][0] < pos[nb][0]


def test_notebook_span_matches_py_y_extent():
    g = build_graph(ROOT_COMPLEX, MANIFEST_COMPLEX)
    L = compute_layout(g.nodes, g.edges)
    pos = L["positions"]
    nb = code_id("notebooks/pipeline.ipynb")
    py_ids = [e.target for e in g.edges if e.kind == "notebook_call" and e.source == nb]
    ys = [pos[pid][1] for pid in py_ids]
    span = max(ys) - min(ys)
    sp = L["notebook_spans"][nb]
    assert abs(sp["span_y"] - span) < 1e-6
    assert sp["box_height"] >= span


def test_no_direct_edge_notebook_to_pipeline_data():
    """图中不应存在 notebook→staging/merged 等：这些边只应来自 .py I/O。"""
    g = build_graph(ROOT_COMPLEX, MANIFEST_COMPLEX)
    merged = data_id("data/staging/merged_orders.csv")
    nb = code_id("notebooks/pipeline.ipynb")
    for e in g.edges:
        if e.source == nb and e.target == merged:
            raise AssertionError("unexpected notebook -> merged edge")
        if e.source == merged and e.target == nb:
            raise AssertionError("unexpected merged -> notebook edge")


def test_vertical_multi_py_order_preserves_span():
    """example_project_nb_py：两个 py 纵向错开时 notebook 盒子高度覆盖二者。"""
    g = build_graph(ROOT_NB_PY, MANIFEST_NB_PY)
    L = compute_layout(g.nodes, g.edges)
    nb = code_id("notebooks/run.ipynb")
    assert nb in L["notebook_spans"]
    py_ids = [e.target for e in g.edges if e.kind == "notebook_call" and e.source == nb]
    assert len(py_ids) >= 2
    ys = [L["positions"][p][1] for p in py_ids]
    assert L["notebook_spans"][nb]["span_y"] == max(ys) - min(ys)
