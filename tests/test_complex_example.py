"""复杂目录结构：多层级 notebooks、扇入/扇出、manifest 独占边。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id

ROOT = Path(__file__).resolve().parent.parent / "example_project_complex"
MANIFEST = ROOT / "pipeline-manifest.yaml"


@pytest.fixture
def graph():
    return build_graph(ROOT, MANIFEST)


def test_notebook_count(graph):
    codes = [n for n in graph.nodes if n.kind == "notebook"]
    paths = sorted(n.path for n in codes)
    assert paths == [
        "notebooks/analytics/aggregate.ipynb",
        "notebooks/extra/manifest_only.ipynb",
        "notebooks/ingest/merge_sources.ipynb",
        "notebooks/transform/split_wide.ipynb",
    ]


def test_data_nodes_include_chain_and_manifest(graph):
    data_paths = sorted(n.path for n in graph.nodes if n.kind == "data")
    assert "data/raw/orders.csv" in data_paths
    assert "data/raw/customers.csv" in data_paths
    assert "data/raw/product.csv" in data_paths
    assert "data/staging/merged_orders.csv" in data_paths
    assert "data/out/summary.csv" in data_paths
    assert "data/out/detail_wide.csv" in data_paths
    assert "data/out/metrics.json" in data_paths
    assert "data/aux/readme_stub.txt" in data_paths
    assert "data/aux/derived_note.txt" in data_paths


def test_edges_merge_fan_in(graph):
    """merge_sources: 多路 raw（含 product）-> 一路 staging"""
    edges = {(e.source, e.target) for e in graph.edges}
    nb = "notebooks/ingest/merge_sources.ipynb"
    assert (data_id("data/raw/orders.csv"), code_id(nb)) in edges
    assert (data_id("data/raw/customers.csv"), code_id(nb)) in edges
    assert (data_id("data/raw/product.csv"), code_id(nb)) in edges
    assert (code_id(nb), data_id("data/staging/merged_orders.csv")) in edges


def test_edges_split_fan_out(graph):
    """split_wide: 一路 staging -> 两路 out（上下并行在布局里体现）"""
    edges = {(e.source, e.target) for e in graph.edges}
    nb = "notebooks/transform/split_wide.ipynb"
    assert (data_id("data/staging/merged_orders.csv"), code_id(nb)) in edges
    assert (code_id(nb), data_id("data/out/summary.csv")) in edges
    assert (code_id(nb), data_id("data/out/detail_wide.csv")) in edges


def test_edges_aggregate_and_manifest_only(graph):
    edges = {(e.source, e.target) for e in graph.edges}
    agg = "notebooks/analytics/aggregate.ipynb"
    assert (data_id("data/out/summary.csv"), code_id(agg)) in edges
    assert (code_id(agg), data_id("data/out/metrics.json")) in edges

    man = "notebooks/extra/manifest_only.ipynb"
    assert (data_id("data/aux/readme_stub.txt"), code_id(man)) in edges
    assert (code_id(man), data_id("data/aux/derived_note.txt")) in edges


def test_edge_count(graph):
    # merge 4（orders/customers/product -> staging）+ split 3 + aggregate 2 + manifest_only 2
    assert len(graph.edges) == 11


def test_layout_reaches_all_nodes(graph):
    from pipeline_viz.layout import layered_positions

    pos = layered_positions(graph.nodes, graph.edges)
    for n in graph.nodes:
        assert n.id in pos, f"missing position for {n.id}"
