from __future__ import annotations

import json
from pathlib import Path

from pipeline_viz.models import GraphEdge, GraphNode, GraphPayload
from pipeline_viz.snapshot_store import SnapshotRecord, commit_edge_highlights, commit_highlights


def _write_snapshot(project_root: Path, rec: SnapshotRecord) -> None:
    hr = project_root / ".pipeline-viz" / "history"
    hr.mkdir(parents=True, exist_ok=True)
    (hr / f"{rec.snapshot_id}.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
    # index.json 只需要能让 previous_snapshot_id 找到顺序
    idx_p = hr / "index.json"
    idx = []
    if idx_p.is_file():
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
    idx.append(
        {
            "snapshot_id": rec.snapshot_id,
            "saved_at": rec.saved_at,
            "label": rec.label,
            "description": rec.description,
        }
    )
    idx_p.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")


def test_commit_highlights_first_snapshot_no_red(tmp_path: Path):
    """第一条全局快照：无上一条可比，不应对节点/边做「变更」标红。"""
    p = GraphPayload(
        nodes=[GraphNode(id="code:notebooks/a.ipynb", kind="notebook", label="a", path="notebooks/a.ipynb")],
        edges=[],
    )
    s1 = SnapshotRecord(
        snapshot_id="S1",
        saved_at="2026-01-01T00:00:00+00:00",
        label="",
        description="",
        payload=p,
        file_hashes={"notebooks/a.ipynb": "h1"},
    )
    _write_snapshot(tmp_path, s1)
    assert commit_highlights(tmp_path, "S1", p) == {}
    assert commit_edge_highlights(tmp_path, "S1", p) == {}


def test_commit_highlights_marks_graph_only_changes(tmp_path: Path):
    """
    只发生“图结构变化”（例如 manifest 改了但 manifest 不跟踪）时，也应有红点/红线提示。
    """
    p1 = GraphPayload(
        nodes=[GraphNode(id="code:notebooks/a.ipynb", kind="notebook", label="a", path="notebooks/a.ipynb")],
        edges=[],
    )
    p2 = GraphPayload(
        nodes=[
            GraphNode(id="code:notebooks/a.ipynb", kind="notebook", label="a", path="notebooks/a.ipynb"),
            GraphNode(id="data:data/product.csv", kind="data", label="product.csv", path="data/product.csv"),
        ],
        edges=[
            GraphEdge(
                id="e1",
                source="data:data/product.csv",
                target="code:notebooks/a.ipynb",
                kind="input",
            )
        ],
    )
    s1 = SnapshotRecord(
        snapshot_id="S1",
        saved_at="2026-01-01T00:00:00+00:00",
        label="",
        description="",
        payload=p1,
        file_hashes={},  # 模拟：没有任何文件 hash 变化
    )
    s2 = SnapshotRecord(
        snapshot_id="S2",
        saved_at="2026-01-02T00:00:00+00:00",
        label="",
        description="",
        payload=p2,
        file_hashes={},  # 模拟：没有任何文件 hash 变化
    )
    _write_snapshot(tmp_path, s1)
    _write_snapshot(tmp_path, s2)

    hi = commit_highlights(tmp_path, "S2", p2)
    assert "data:data/product.csv" in hi  # 新节点应被标红

    ehi = commit_edge_highlights(tmp_path, "S2", p2)
    assert ehi.get("data:data/product.csv|code:notebooks/a.ipynb|input") == "#b71c1c"

