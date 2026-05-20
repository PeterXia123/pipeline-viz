"""Test that runtime_io merges into the pipeline graph."""
from __future__ import annotations

from pathlib import Path

import nbformat

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id


def _make_minimal_notebook(path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def test_runtime_io_adds_data_edges(tmp_path):
    nb_rel = "notebooks/analysis.ipynb"
    _make_minimal_notebook(tmp_path / nb_rel)

    runtime_io = {
        nb_rel: {
            "inputs": ["data/raw/orders.csv", "data/raw/users.csv"],
            "outputs": ["data/out/report.csv"],
        }
    }
    g = build_graph(tmp_path, None, runtime_io=runtime_io)

    node_ids = {n.id for n in g.nodes}
    assert data_id("data/raw/orders.csv") in node_ids
    assert data_id("data/raw/users.csv") in node_ids
    assert data_id("data/out/report.csv") in node_ids
    assert code_id(nb_rel) in node_ids

    edge_tuples = {(e.source, e.target, e.kind) for e in g.edges}
    assert (data_id("data/raw/orders.csv"), code_id(nb_rel), "input") in edge_tuples
    assert (data_id("data/raw/users.csv"), code_id(nb_rel), "input") in edge_tuples
    assert (code_id(nb_rel), data_id("data/out/report.csv"), "output") in edge_tuples


def test_runtime_io_none_is_noop(tmp_path):
    nb_rel = "notebooks/simple.ipynb"
    _make_minimal_notebook(tmp_path / nb_rel)

    g1 = build_graph(tmp_path, None, runtime_io=None)
    g2 = build_graph(tmp_path, None)
    assert len(g1.nodes) == len(g2.nodes)
    assert len(g1.edges) == len(g2.edges)
