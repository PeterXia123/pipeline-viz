"""集成：选中 view_snapshot_id 时 /api/graph 应对上一快照算出非空高亮。"""

from __future__ import annotations

from pathlib import Path

from pipeline_viz.main import api_graph
from pipeline_viz.models import GraphEdge, GraphNode, GraphPayload
from pipeline_viz.paths import data_id
from pipeline_viz.snapshot_store import save_snapshot


def test_api_graph_view_snapshot_highlights_edge_only_for_structure_change(tmp_path: Path):
    """图结构变化（新增边）只高亮边，不高亮节点。"""
    nb = "notebooks/a.ipynb"
    (tmp_path / "notebooks").mkdir(parents=True)
    (tmp_path / nb).write_text("{}", encoding="utf-8")

    ta = f"code:{nb}"
    p1 = GraphPayload(
        nodes=[
            GraphNode(id=ta, kind="notebook", label="a.ipynb", path=nb),
        ],
        edges=[],
    )
    td = data_id("data/product.csv")
    p2 = GraphPayload(
        nodes=[
            GraphNode(id=ta, kind="notebook", label="a.ipynb", path=nb),
            GraphNode(id=td, kind="data", label="product.csv", path="data/product.csv"),
        ],
        edges=[GraphEdge(id="e0", source=td, target=ta, kind="input")],
    )
    sid1, _, m1, _ = save_snapshot(tmp_path, p1)
    sid2, _, m2, _ = save_snapshot(tmp_path, p2)
    assert sid1 and m1 == "saved" and sid2 and m2 == "saved"

    data = api_graph(
        project_root=str(tmp_path.resolve()),
        view_snapshot_id=sid2,
    )
    assert data["highlight_debug"]["using_saved_payload"] is True
    assert data["highlight_debug"]["previous_snapshot_id_for_view"] == sid1
    assert data["highlight_counts"]["node"] == 0
    assert data["highlight_counts"]["edge"] > 0
