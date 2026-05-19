"""仅边 id / 顺序抖动不应被视为语义变更（不应产生新快照）。"""

from __future__ import annotations

from pathlib import Path

from pipeline_viz.models import GraphEdge, GraphNode, GraphPayload
from pipeline_viz.snapshot_store import save_snapshot


def test_snapshot_skip_when_only_edge_ids_differ(tmp_path: Path):
    (tmp_path / "n.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "d.csv").write_text("x\n", encoding="utf-8")
    na = "n.ipynb"
    da = "d.csv"
    a = GraphNode(id=f"code:{na}", kind="notebook", label="n", path=na)
    b = GraphNode(id=f"data:{da}", kind="data", label="d", path=da)
    e1 = GraphEdge(id="e0", source=b.id, target=a.id, kind="input")
    e2 = GraphEdge(id="eDifferentId", source=b.id, target=a.id, kind="input")
    p1 = GraphPayload(nodes=[a, b], edges=[e1])
    p2 = GraphPayload(nodes=[a, b], edges=[e2])

    sid1, _, m1, _ = save_snapshot(tmp_path, p1)
    sid2, _, m2, _ = save_snapshot(tmp_path, p2)
    assert sid1 and m1 == "saved"
    assert sid2 is None and m2 == "no_changes"
