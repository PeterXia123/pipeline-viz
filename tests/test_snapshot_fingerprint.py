import json
from pathlib import Path

from pipeline_viz.models import GraphEdge, GraphNode, GraphPayload
from pipeline_viz.paths import code_id, data_id
from pipeline_viz.snapshot_store import (
    SnapshotRecord,
    history_for_node,
    history_root,
    index_path,
    path_changed_in_commit,
    save_snapshot,
    state_fingerprint,
)


def test_save_snapshot_when_only_edges_change(tmp_path: Path):
    """文件内容未变，仅节点间连线（方向/关系）变化时也应产生新快照。"""
    (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y=2\n", encoding="utf-8")
    ta, tb = "code:a.py", "code:b.py"
    a = GraphNode(id=ta, label="a.py", kind="python", path="a.py")
    b = GraphNode(id=tb, label="b.py", kind="python", path="b.py")
    g1 = GraphPayload(
        nodes=[a, b],
        edges=[GraphEdge(id="e0", source=ta, target=tb, kind="notebook_call")],
    )
    sid1, _, m1, _ = save_snapshot(tmp_path, g1)
    assert sid1 and m1 == "saved"
    g2 = GraphPayload(
        nodes=[a, b],
        edges=[GraphEdge(id="e0", source=tb, target=ta, kind="notebook_call")],
    )
    sid2, _, m2, _ = save_snapshot(tmp_path, g2)
    assert sid2 and m2 == "saved"


def test_fingerprint_stable_for_empty_graph():
    empty = GraphPayload(nodes=[], edges=[])
    assert state_fingerprint(empty, {}) == state_fingerprint(empty, {})
    assert state_fingerprint(empty, {"a.csv": "dead"}) != state_fingerprint(empty, {})


def test_node_history_only_keeps_actual_file_versions(tmp_path: Path):
    code_rel = "pipeline/ingest.py"
    data_rel = "data/raw.csv"

    (tmp_path / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / code_rel).write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / data_rel).write_text("a\n1\n", encoding="utf-8")

    payload = GraphPayload(
        nodes=[
            GraphNode(id=code_id(code_rel), label="ingest.py", kind="python", path=code_rel),
            GraphNode(id=data_id(data_rel), label="raw.csv", kind="data", path=data_rel),
        ],
        edges=[],
    )

    sid1, _, msg1, _ = save_snapshot(tmp_path, payload)
    assert sid1 and msg1 == "saved"

    # 仅 data 变化：全局快照会保存，但 ingest.py 不应新增“该文件版本”
    (tmp_path / data_rel).write_text("a\n2\n", encoding="utf-8")
    sid2, _, msg2, _ = save_snapshot(tmp_path, payload)
    assert sid2 and msg2 == "saved"

    # code 变化：ingest.py 应新增版本点
    (tmp_path / code_rel).write_text("def run():\n    return 3\n", encoding="utf-8")
    sid3, _, msg3, _ = save_snapshot(tmp_path, payload)
    assert sid3 and msg3 == "saved"

    hist = history_for_node(tmp_path, code_rel, is_data=False)
    got = [e["snapshot_id"] for e in hist]
    assert got == [sid3, sid1]


def test_code_history_keeps_single_legacy_fallback_without_hash(tmp_path: Path):
    code_rel = "pipeline/injest.py"
    node = GraphNode(id=code_id(code_rel), label="injest.py", kind="python", path=code_rel)

    hr = history_root(tmp_path)
    hr.mkdir(parents=True, exist_ok=True)
    idx = []
    for sid, ts in (("s1", "2026-01-01T00:00:00+00:00"), ("s2", "2026-01-02T00:00:00+00:00")):
        rec = SnapshotRecord(
            snapshot_id=sid,
            saved_at=ts,
            label="",
            payload=GraphPayload(nodes=[node], edges=[]),
            file_hashes={},
        )
        (hr / f"{sid}.json").write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        idx.append({"snapshot_id": sid, "saved_at": ts, "label": ""})
    index_path(tmp_path).write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")

    hist = history_for_node(tmp_path, code_rel, is_data=False)
    got = [e["snapshot_id"] for e in hist]
    assert got == ["s1"]


def test_path_changed_in_commit_matches_commit_view(tmp_path: Path):
    code_rel = "pipeline/x.py"
    data_rel = "data/x.csv"
    (tmp_path / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / code_rel).write_text("a = 1\n", encoding="utf-8")
    (tmp_path / data_rel).write_text("1\n", encoding="utf-8")

    payload = GraphPayload(
        nodes=[
            GraphNode(id=code_id(code_rel), label="x.py", kind="python", path=code_rel),
            GraphNode(id=data_id(data_rel), label="x.csv", kind="data", path=data_rel),
        ],
        edges=[],
    )
    sid1, _, _, _ = save_snapshot(tmp_path, payload)
    assert sid1

    (tmp_path / data_rel).write_text("2\n", encoding="utf-8")
    sid2, _, _, _ = save_snapshot(tmp_path, payload)
    assert sid2

    assert path_changed_in_commit(tmp_path, sid1, code_rel) is True
    assert path_changed_in_commit(tmp_path, sid1, data_rel) is True
    assert path_changed_in_commit(tmp_path, sid2, code_rel) is False
    assert path_changed_in_commit(tmp_path, sid2, data_rel) is True
