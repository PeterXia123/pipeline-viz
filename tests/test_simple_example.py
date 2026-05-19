from __future__ import annotations

from pathlib import Path

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id

ROOT = Path(__file__).resolve().parent.parent / "example_project"


def test_simple_linear_pipeline():
    g = build_graph(ROOT, ROOT / "pipeline-manifest.yaml")
    edges = {(e.source, e.target) for e in g.edges}
    nb = "notebooks/etl.ipynb"
    assert (data_id("data/raw/in.csv"), code_id(nb)) in edges
    assert (code_id(nb), data_id("data/out/out.csv")) in edges
    assert len(g.nodes) == 3
    assert len(g.edges) == 2
