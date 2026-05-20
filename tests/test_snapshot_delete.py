"""删除全局快照：索引、blob、latest 指向。"""

from pathlib import Path

from pipeline_viz.models import GraphNode, GraphPayload
from pipeline_viz.paths import code_id
from pipeline_viz.snapshot_store import (
    delete_snapshot,
    history_root,
    latest_path,
    load_latest,
    save_snapshot,
)


def test_delete_snapshot_removes_files_and_updates_latest(tmp_path: Path):
    code_rel = "pipeline/a.py"
    (tmp_path / "pipeline").mkdir(parents=True, exist_ok=True)
    (tmp_path / code_rel).write_text("x = 1\n", encoding="utf-8")

    node = GraphNode(id=code_id(code_rel), label="a.py", kind="python", path=code_rel)
    payload = GraphPayload(nodes=[node], edges=[])

    sid1, _, _, _ = save_snapshot(tmp_path, payload)
    (tmp_path / code_rel).write_text("x = 2\n", encoding="utf-8")
    sid2, _, _, _ = save_snapshot(tmp_path, payload)

    assert load_latest(tmp_path).snapshot_id == sid2
    assert (history_root(tmp_path) / f"{sid1}.json").is_file()
    # 旧快照的 blobs 在新快照保存时已被清理，只保留最新快照的文件副本
    assert not (history_root(tmp_path) / sid1 / "files").is_dir()
    assert (history_root(tmp_path) / sid2 / "files").is_dir()

    assert delete_snapshot(tmp_path, sid1) is True
    assert not (history_root(tmp_path) / f"{sid1}.json").exists()
    assert not (history_root(tmp_path) / sid1).exists()
    assert load_latest(tmp_path).snapshot_id == sid2

    assert delete_snapshot(tmp_path, sid2) is True
    assert latest_path(tmp_path).is_file() is False
    assert load_latest(tmp_path) is None


def test_delete_unknown_returns_false(tmp_path: Path):
    assert delete_snapshot(tmp_path, "nope") is False
