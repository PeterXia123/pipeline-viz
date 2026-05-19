"""Cookiecutter-style example: layered data + src package + numbered notebooks."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id

ROOT = Path(__file__).resolve().parent.parent / "example_cookiecutter_ds"
MANIFEST = ROOT / "pipeline-manifest.yaml"


@pytest.fixture
def graph():
    return build_graph(ROOT, MANIFEST)


def test_pipeline_chain_edges(graph):
    edges = {(e.source, e.target) for e in graph.edges}
    assert (data_id("data/raw/customers.csv"), code_id("notebooks/01_ingest_merge.ipynb")) in edges
    assert (data_id("data/interim/merged_orders.csv"), code_id("notebooks/02_features.ipynb")) in edges
    assert (data_id("data/processed/features.csv"), code_id("notebooks/03_train_model.ipynb")) in edges
    assert (data_id("models/metrics.csv"), code_id("notebooks/04_report.ipynb")) in edges
    assert (code_id("notebooks/04_report.ipynb"), data_id("reports/metrics.json")) in edges


def test_notebook_calls_src_package(graph):
    edges = {(e.source, e.target) for e in graph.edges}
    assert (code_id("notebooks/01_ingest_merge.ipynb"), code_id("src/ccds_ds/ingest.py")) in edges
    assert (code_id("notebooks/02_features.ipynb"), code_id("src/ccds_ds/features.py")) in edges
    assert (code_id("notebooks/03_train_model.ipynb"), code_id("src/ccds_ds/training.py")) in edges


def test_geo_lookup_is_input_to_ingest_py_not_notebook(graph):
    """geo_lookup 在 ingest.py 内读取，不应因 manifest 误接到 notebook。"""
    edges = {(e.source, e.target) for e in graph.edges}
    gid = data_id("data/external/geo_lookup.csv")
    assert (gid, code_id("src/ccds_ds/ingest.py")) in edges
    assert (gid, code_id("notebooks/01_ingest_merge.ipynb")) not in edges


def test_edge_count(graph):
    assert len(graph.edges) == 13
