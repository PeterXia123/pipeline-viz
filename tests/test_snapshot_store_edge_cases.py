"""Edge-case unit tests for pipeline_viz.snapshot_store (Layer 2)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from pipeline_viz.models import GraphEdge, GraphNode, GraphPayload
from pipeline_viz.paths import code_id, data_id
from pipeline_viz.snapshot_store import (
    SnapshotRecord,
    _cleanup_old_blobs,
    _graph_node_id_for_tracked_rel,
    _hashes_fingerprint_str,
    _payload_fingerprint_str,
    _safe_rel,
    blob_path,
    compute_file_hashes,
    delete_snapshot,
    diff_edge_highlights,
    diff_highlights,
    history_root,
    latest_path,
    list_history,
    list_history_desc,
    load_latest,
    load_snapshot,
    node_path_from_graph_id,
    previous_snapshot_id,
    read_blob,
    save_snapshot,
    set_node_meta,
    set_snapshot_label,
    set_snapshot_meta,
    snapshot_json_path,
    state_fingerprint,
    tracked_paths,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_payload(
    nodes_data: list[tuple[str, str, str, str]],
    edges_data: list[tuple[str, str, str, str]] | None = None,
) -> GraphPayload:
    nodes = [
        GraphNode(id=n[0], label=n[1], kind=n[2], path=n[3]) for n in nodes_data
    ]
    edges = [
        GraphEdge(id=e[0], source=e[1], target=e[2], kind=e[3])
        for e in (edges_data or [])
    ]
    return GraphPayload(nodes=nodes, edges=edges)


def _simple_payload() -> GraphPayload:
    return make_payload(
        [
            (data_id("data/raw.csv"), "raw.csv", "data", "data/raw.csv"),
            (code_id("nb.ipynb"), "nb.ipynb", "notebook", "nb.ipynb"),
        ],
        [("e0", data_id("data/raw.csv"), code_id("nb.ipynb"), "input")],
    )


def _setup_project(tmp_path: Path, payload: GraphPayload | None = None) -> GraphPayload:
    """Create minimal files on disk that match payload node paths."""
    p = payload or _simple_payload()
    for n in p.nodes:
        fp = tmp_path / n.path
        fp.parent.mkdir(parents=True, exist_ok=True)
        if not fp.exists():
            fp.write_text(f"content of {n.path}\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SnapshotRecord legacy validator
# ---------------------------------------------------------------------------

class TestSnapshotRecordLegacy:
    def test_missing_snapshot_id_auto_generates(self):
        rec = SnapshotRecord.model_validate(
            {"saved_at": "2025-01-01T00:00:00", "payload": {"nodes": [], "edges": []}}
        )
        assert rec.snapshot_id != ""
        assert "2025-01-01T00-00-00" in rec.snapshot_id

    def test_missing_file_hashes_defaults_to_empty(self):
        rec = SnapshotRecord.model_validate(
            {
                "snapshot_id": "s1",
                "saved_at": "2025-01-01T00:00:00",
                "payload": {"nodes": [], "edges": []},
            }
        )
        assert rec.file_hashes == {}

    def test_missing_label_description_defaults(self):
        rec = SnapshotRecord.model_validate(
            {
                "snapshot_id": "s1",
                "saved_at": "2025-01-01T00:00:00",
                "payload": {"nodes": [], "edges": []},
            }
        )
        assert rec.label == ""
        assert rec.description == ""

    def test_missing_node_labels_defaults(self):
        rec = SnapshotRecord.model_validate(
            {
                "snapshot_id": "s1",
                "saved_at": "2025-01-01T00:00:00",
                "payload": {"nodes": [], "edges": []},
            }
        )
        assert rec.node_labels == {}

    def test_missing_data_diffs_defaults(self):
        rec = SnapshotRecord.model_validate(
            {
                "snapshot_id": "s1",
                "saved_at": "2025-01-01T00:00:00",
                "payload": {"nodes": [], "edges": []},
            }
        )
        assert rec.data_diffs == {}

    def test_provided_fields_preserved(self):
        rec = SnapshotRecord.model_validate(
            {
                "snapshot_id": "myid",
                "saved_at": "2025-06-01T12:00:00",
                "label": "v1",
                "description": "first",
                "payload": {"nodes": [], "edges": []},
                "file_hashes": {"a.csv": "abc123"},
                "node_labels": {"a.csv": {"label": "A"}},
                "data_diffs": {"a.csv": {"diff": True}},
            }
        )
        assert rec.snapshot_id == "myid"
        assert rec.label == "v1"
        assert rec.description == "first"
        assert rec.file_hashes == {"a.csv": "abc123"}


# ---------------------------------------------------------------------------
# _safe_rel
# ---------------------------------------------------------------------------

class TestSafeRel:
    def test_normal_path(self):
        assert _safe_rel("data/raw.csv") == "data/raw.csv"

    def test_path_with_dotdot_raises(self):
        with pytest.raises(ValueError, match="invalid path"):
            _safe_rel("data/../secret.csv")

    def test_path_with_leading_slash_stripped(self):
        assert _safe_rel("/data/raw.csv") == "data/raw.csv"

    def test_backslash_normalized(self):
        result = _safe_rel("data\\raw.csv")
        assert "\\" not in result
        assert "data" in result and "raw.csv" in result

    def test_multiple_slashes_stripped(self):
        result = _safe_rel("///data/raw.csv")
        assert result == "data/raw.csv"


# ---------------------------------------------------------------------------
# node_path_from_graph_id
# ---------------------------------------------------------------------------

class TestNodePathFromGraphId:
    def test_data_prefix(self):
        assert node_path_from_graph_id("data:path/to/file.csv") == "path/to/file.csv"

    def test_code_prefix(self):
        assert node_path_from_graph_id("code:nb.ipynb") == "nb.ipynb"

    def test_invalid_prefix(self):
        assert node_path_from_graph_id("invalid") is None

    def test_data_empty_path(self):
        assert node_path_from_graph_id("data:") == ""

    def test_code_empty_path(self):
        assert node_path_from_graph_id("code:") == ""


# ---------------------------------------------------------------------------
# tracked_paths
# ---------------------------------------------------------------------------

class TestTrackedPaths:
    def test_returns_sorted_unique(self, tmp_path: Path):
        payload = make_payload([
            (data_id("b.csv"), "b.csv", "data", "b.csv"),
            (data_id("a.csv"), "a.csv", "data", "a.csv"),
            (code_id("nb.ipynb"), "nb.ipynb", "notebook", "nb.ipynb"),
        ])
        result = tracked_paths(tmp_path, payload)
        assert result == sorted(result)
        assert len(result) == len(set(result))

    def test_dedup_same_path(self, tmp_path: Path):
        payload = make_payload([
            (data_id("a.csv"), "a.csv", "data", "a.csv"),
            ("data:a.csv_dup", "a.csv", "data", "a.csv"),
        ])
        result = tracked_paths(tmp_path, payload)
        assert result.count("a.csv") == 1


# ---------------------------------------------------------------------------
# compute_file_hashes
# ---------------------------------------------------------------------------

class TestComputeFileHashes:
    def test_existing_file_hashed(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        h = compute_file_hashes(tmp_path, ["data.csv"])
        assert "data.csv" in h
        assert len(h["data.csv"]) == 64  # sha256 hex

    def test_missing_file_empty_hash(self, tmp_path: Path):
        h = compute_file_hashes(tmp_path, ["no_such_file.csv"])
        assert h["no_such_file.csv"] == ""

    def test_ipynb_uses_stable_hash(self, tmp_path: Path):
        nb = {
            "cells": [
                {"cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": [], "execution_count": 1}
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (tmp_path / "nb.ipynb").write_text(json.dumps(nb), encoding="utf-8")
        h = compute_file_hashes(tmp_path, ["nb.ipynb"])
        assert h["nb.ipynb"] != ""

        # change execution_count only -> hash should NOT change
        nb["cells"][0]["execution_count"] = 99
        (tmp_path / "nb.ipynb").write_text(json.dumps(nb), encoding="utf-8")
        h2 = compute_file_hashes(tmp_path, ["nb.ipynb"])
        assert h2["nb.ipynb"] == h["nb.ipynb"]

    def test_path_outside_root_skipped(self, tmp_path: Path):
        h = compute_file_hashes(tmp_path, ["../../../etc/passwd"])
        assert "../../../etc/passwd" not in h


# ---------------------------------------------------------------------------
# _payload_fingerprint_str
# ---------------------------------------------------------------------------

class TestPayloadFingerprint:
    def test_same_nodes_different_order(self):
        p1 = make_payload([
            (data_id("a.csv"), "a", "data", "a.csv"),
            (data_id("b.csv"), "b", "data", "b.csv"),
        ])
        p2 = make_payload([
            (data_id("b.csv"), "b", "data", "b.csv"),
            (data_id("a.csv"), "a", "data", "a.csv"),
        ])
        assert _payload_fingerprint_str(p1) == _payload_fingerprint_str(p2)

    def test_different_nodes_different_fingerprint(self):
        p1 = make_payload([(data_id("a.csv"), "a", "data", "a.csv")])
        p2 = make_payload([(data_id("b.csv"), "b", "data", "b.csv")])
        assert _payload_fingerprint_str(p1) != _payload_fingerprint_str(p2)

    def test_edge_id_ignored(self):
        p1 = make_payload(
            [(data_id("a.csv"), "a", "data", "a.csv"), (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb")],
            [("e0", data_id("a.csv"), code_id("nb.ipynb"), "input")],
        )
        p2 = make_payload(
            [(data_id("a.csv"), "a", "data", "a.csv"), (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb")],
            [("e999", data_id("a.csv"), code_id("nb.ipynb"), "input")],
        )
        assert _payload_fingerprint_str(p1) == _payload_fingerprint_str(p2)


# ---------------------------------------------------------------------------
# _hashes_fingerprint_str
# ---------------------------------------------------------------------------

class TestHashesFingerprint:
    def test_empty_values_filtered(self):
        result = _hashes_fingerprint_str({"a.csv": "abc", "b.csv": ""})
        parsed = json.loads(result)
        keys_in_result = [k for k, v in parsed]
        assert "b.csv" not in keys_in_result

    def test_sorted_output(self):
        r1 = _hashes_fingerprint_str({"z.csv": "z1", "a.csv": "a1"})
        r2 = _hashes_fingerprint_str({"a.csv": "a1", "z.csv": "z1"})
        assert r1 == r2

    def test_all_empty_values(self):
        result = _hashes_fingerprint_str({"a.csv": "", "b.csv": ""})
        assert result == "[]"


# ---------------------------------------------------------------------------
# state_fingerprint
# ---------------------------------------------------------------------------

class TestStateFingerprint:
    def test_consistent_sha256(self):
        payload = make_payload([(data_id("a.csv"), "a", "data", "a.csv")])
        hashes = {"a.csv": "hash1"}
        f1 = state_fingerprint(payload, hashes)
        f2 = state_fingerprint(payload, hashes)
        assert f1 == f2
        assert len(f1) == 64

    def test_different_payload_different_fingerprint(self):
        p1 = make_payload([(data_id("a.csv"), "a", "data", "a.csv")])
        p2 = make_payload([(data_id("b.csv"), "b", "data", "b.csv")])
        hashes = {"a.csv": "hash1"}
        assert state_fingerprint(p1, hashes) != state_fingerprint(p2, hashes)


# ---------------------------------------------------------------------------
# save_snapshot
# ---------------------------------------------------------------------------

class TestSaveSnapshot:
    def test_first_save_succeeds(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, jp, msg, detail = save_snapshot(tmp_path, payload)
        assert sid is not None
        assert jp is not None
        assert msg == "saved"
        assert detail is None
        assert jp.is_file()

    def test_duplicate_save_returns_none(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        assert sid1 is not None

        sid2, jp2, msg2, detail2 = save_snapshot(tmp_path, payload)
        assert sid2 is None
        assert jp2 is None
        assert msg2 == "no_changes"
        assert detail2 is not None
        assert detail2["graph_changed"] is False
        assert detail2["files_changed"] is False

    def test_saves_with_data_diffs(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        diffs = {"data/raw.csv": {"columns_added": ["new_col"]}}
        sid, jp, msg, _ = save_snapshot(tmp_path, payload, data_diffs=diffs)
        assert sid is not None
        rec = load_snapshot(tmp_path, sid)
        assert rec is not None
        assert rec.data_diffs == diffs

    def test_save_after_file_change(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        assert sid1 is not None

        (tmp_path / "data" / "raw.csv").write_text("changed content\n", encoding="utf-8")
        sid2, _, msg2, _ = save_snapshot(tmp_path, payload)
        assert sid2 is not None
        assert msg2 == "saved"


# ---------------------------------------------------------------------------
# load_snapshot
# ---------------------------------------------------------------------------

class TestLoadSnapshot:
    def test_valid_snapshot(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        rec = load_snapshot(tmp_path, sid)
        assert rec is not None
        assert rec.snapshot_id == sid

    def test_nonexistent_returns_none(self, tmp_path: Path):
        assert load_snapshot(tmp_path, "nonexistent_id") is None

    def test_corrupted_json_returns_none(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, jp, _, _ = save_snapshot(tmp_path, payload)
        jp.write_text("{invalid json!!!", encoding="utf-8")
        # Clear cache so it re-reads from disk
        from pipeline_viz.snapshot_store import _SNAPSHOT_CACHE, _cache_key
        _SNAPSHOT_CACHE.pop(_cache_key(tmp_path, sid), None)
        assert load_snapshot(tmp_path, sid) is None


# ---------------------------------------------------------------------------
# load_latest
# ---------------------------------------------------------------------------

class TestLoadLatest:
    def test_no_latest_returns_none(self, tmp_path: Path):
        assert load_latest(tmp_path) is None

    def test_valid_latest(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        rec = load_latest(tmp_path)
        assert rec is not None
        assert rec.snapshot_id == sid


# ---------------------------------------------------------------------------
# delete_snapshot
# ---------------------------------------------------------------------------

class TestDeleteSnapshot:
    def test_delete_existing(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        assert delete_snapshot(tmp_path, sid) is True
        assert load_snapshot(tmp_path, sid) is None

    def test_delete_nonexistent_returns_false(self, tmp_path: Path):
        assert delete_snapshot(tmp_path, "nope") is False

    def test_invalid_id_special_chars_returns_false(self, tmp_path: Path):
        assert delete_snapshot(tmp_path, "../../etc/passwd") is False
        assert delete_snapshot(tmp_path, "") is False
        assert delete_snapshot(tmp_path, "a" * 300) is False

    def test_delete_latest_updates_pointer(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        # Modify file to force a second save
        for n in payload.nodes:
            fp = tmp_path / n.path
            fp.write_text(f"v2 of {n.path}\n", encoding="utf-8")
        sid2, _, _, _ = save_snapshot(tmp_path, payload)
        assert sid2 is not None

        # Delete latest (sid2) -> should point to sid1
        assert delete_snapshot(tmp_path, sid2) is True
        rec = load_latest(tmp_path)
        assert rec is not None
        assert rec.snapshot_id == sid1

    def test_delete_last_snapshot_removes_latest_json(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        assert delete_snapshot(tmp_path, sid) is True
        assert not latest_path(tmp_path).is_file()


# ---------------------------------------------------------------------------
# set_snapshot_meta
# ---------------------------------------------------------------------------

class TestSetSnapshotMeta:
    def test_updates_json_and_index(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        result = set_snapshot_meta(tmp_path, sid, label="v1", description="first version")
        assert result is True

        # Clear cache to re-read
        from pipeline_viz.snapshot_store import _SNAPSHOT_CACHE, _cache_key
        _SNAPSHOT_CACHE.pop(_cache_key(tmp_path, sid), None)

        rec = load_snapshot(tmp_path, sid)
        assert rec.label == "v1"
        assert rec.description == "first version"

        idx = list_history(tmp_path)
        entry = next(e for e in idx if e["snapshot_id"] == sid)
        assert entry["label"] == "v1"
        assert entry["description"] == "first version"

    def test_nonexistent_returns_false(self, tmp_path: Path):
        assert set_snapshot_meta(tmp_path, "nope", label="x", description="y") is False


# ---------------------------------------------------------------------------
# set_snapshot_label
# ---------------------------------------------------------------------------

class TestSetSnapshotLabel:
    def test_delegates_correctly(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        result = set_snapshot_label(tmp_path, sid, "my label")
        assert result is True

        from pipeline_viz.snapshot_store import _SNAPSHOT_CACHE, _cache_key
        _SNAPSHOT_CACHE.pop(_cache_key(tmp_path, sid), None)

        rec = load_snapshot(tmp_path, sid)
        assert rec.label == "my label"

    def test_nonexistent_returns_false(self, tmp_path: Path):
        assert set_snapshot_label(tmp_path, "nope", "label") is False


# ---------------------------------------------------------------------------
# set_node_meta
# ---------------------------------------------------------------------------

class TestSetNodeMeta:
    def test_sets_metadata(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        result = set_node_meta(
            tmp_path, sid, "data/raw.csv", label="Raw Data", description="Input file"
        )
        assert result is True

        from pipeline_viz.snapshot_store import _SNAPSHOT_CACHE, _cache_key
        _SNAPSHOT_CACHE.pop(_cache_key(tmp_path, sid), None)

        rec = load_snapshot(tmp_path, sid)
        assert "data/raw.csv" in rec.node_labels
        assert rec.node_labels["data/raw.csv"]["label"] == "Raw Data"

    def test_updates_latest_if_is_latest(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        set_node_meta(tmp_path, sid, "data/raw.csv", label="LL", description="DD")
        latest = load_latest(tmp_path)
        assert latest.node_labels.get("data/raw.csv", {}).get("label") == "LL"

    def test_nonexistent_returns_false(self, tmp_path: Path):
        assert set_node_meta(tmp_path, "nope", "a.csv", label="x", description="y") is False


# ---------------------------------------------------------------------------
# previous_snapshot_id
# ---------------------------------------------------------------------------

class TestPreviousSnapshotId:
    def test_first_snapshot_returns_none(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        assert previous_snapshot_id(tmp_path, sid1) is None

    def test_second_returns_first(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        # Modify to force second save
        for n in payload.nodes:
            (tmp_path / n.path).write_text("v2\n", encoding="utf-8")
        sid2, _, _, _ = save_snapshot(tmp_path, payload)
        assert previous_snapshot_id(tmp_path, sid2) == sid1

    def test_nonexistent_sid_returns_none(self, tmp_path: Path):
        assert previous_snapshot_id(tmp_path, "nope") is None


# ---------------------------------------------------------------------------
# list_history / list_history_desc
# ---------------------------------------------------------------------------

class TestListHistory:
    def test_empty(self, tmp_path: Path):
        assert list_history(tmp_path) == []

    def test_multiple_entries(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        for n in payload.nodes:
            (tmp_path / n.path).write_text("v2\n", encoding="utf-8")
        sid2, _, _, _ = save_snapshot(tmp_path, payload)
        history = list_history(tmp_path)
        assert len(history) == 2
        assert history[0]["snapshot_id"] == sid1
        assert history[1]["snapshot_id"] == sid2

    def test_desc_is_reversed(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid1, _, _, _ = save_snapshot(tmp_path, payload)
        for n in payload.nodes:
            (tmp_path / n.path).write_text("v2\n", encoding="utf-8")
        sid2, _, _, _ = save_snapshot(tmp_path, payload)
        asc = list_history(tmp_path)
        desc = list_history_desc(tmp_path)
        assert desc == list(reversed(asc))


# ---------------------------------------------------------------------------
# diff_highlights
# ---------------------------------------------------------------------------

class TestDiffHighlights:
    def test_no_previous_empty(self):
        current = _simple_payload()
        assert diff_highlights(current, None) == {}

    def test_new_node_highlighted(self):
        prev = make_payload([
            (data_id("a.csv"), "a", "data", "a.csv"),
        ])
        current = make_payload([
            (data_id("a.csv"), "a", "data", "a.csv"),
            (data_id("b.csv"), "b", "data", "b.csv"),
        ])
        hi = diff_highlights(current, prev)
        assert data_id("b.csv") in hi
        assert hi[data_id("b.csv")] == "#b7f5c8"

    def test_new_edge_endpoints_highlighted(self):
        prev = make_payload([
            (data_id("a.csv"), "a", "data", "a.csv"),
            (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb"),
        ])
        current = make_payload(
            [
                (data_id("a.csv"), "a", "data", "a.csv"),
                (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb"),
            ],
            [("e0", data_id("a.csv"), code_id("nb.ipynb"), "input")],
        )
        hi = diff_highlights(current, prev)
        # Both endpoints should be highlighted
        assert data_id("a.csv") in hi or code_id("nb.ipynb") in hi

    def test_no_changes_empty(self):
        p = _simple_payload()
        assert diff_highlights(p, p) == {}


# ---------------------------------------------------------------------------
# diff_edge_highlights
# ---------------------------------------------------------------------------

class TestDiffEdgeHighlights:
    def test_no_previous_empty(self):
        current = _simple_payload()
        assert diff_edge_highlights(current, None) == {}

    def test_new_edges_highlighted(self):
        prev = make_payload([
            (data_id("a.csv"), "a", "data", "a.csv"),
            (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb"),
        ])
        current = make_payload(
            [
                (data_id("a.csv"), "a", "data", "a.csv"),
                (code_id("nb.ipynb"), "nb", "notebook", "nb.ipynb"),
            ],
            [("e0", data_id("a.csv"), code_id("nb.ipynb"), "input")],
        )
        hi = diff_edge_highlights(current, prev)
        key = f"{data_id('a.csv')}|{code_id('nb.ipynb')}|input"
        assert key in hi
        assert hi[key] == "#2e7d32"

    def test_no_new_edges_empty(self):
        p = _simple_payload()
        assert diff_edge_highlights(p, p) == {}


# ---------------------------------------------------------------------------
# read_blob
# ---------------------------------------------------------------------------

class TestReadBlob:
    def test_existing_blob(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        content = read_blob(tmp_path, sid, "data/raw.csv")
        assert content is not None
        assert isinstance(content, bytes)

    def test_missing_blob(self, tmp_path: Path):
        payload = _setup_project(tmp_path)
        sid, _, _, _ = save_snapshot(tmp_path, payload)
        assert read_blob(tmp_path, sid, "nonexistent.csv") is None

    def test_invalid_snapshot_id(self, tmp_path: Path):
        assert read_blob(tmp_path, "nosuch", "a.csv") is None


# ---------------------------------------------------------------------------
# _graph_node_id_for_tracked_rel
# ---------------------------------------------------------------------------

class TestGraphNodeIdForTrackedRel:
    def test_exact_match(self):
        payload = make_payload([
            (data_id("data/raw.csv"), "raw.csv", "data", "data/raw.csv"),
        ])
        result = _graph_node_id_for_tracked_rel("data/raw.csv", payload)
        assert result == data_id("data/raw.csv")

    def test_dot_slash_prefix_resolved(self):
        payload = make_payload([
            (data_id("data/raw.csv"), "raw.csv", "data", "./data/raw.csv"),
        ])
        result = _graph_node_id_for_tracked_rel("data/raw.csv", payload)
        assert result is not None

    def test_basename_fallback(self):
        payload = make_payload([
            (data_id("some/deep/path/raw.csv"), "raw.csv", "data", "some/deep/path/raw.csv"),
        ])
        result = _graph_node_id_for_tracked_rel("raw.csv", payload)
        assert result == data_id("some/deep/path/raw.csv")

    def test_no_match_returns_none(self):
        payload = make_payload([
            (data_id("data/raw.csv"), "raw.csv", "data", "data/raw.csv"),
        ])
        result = _graph_node_id_for_tracked_rel("totally_different.csv", payload)
        assert result is None


# ---------------------------------------------------------------------------
# _cleanup_old_blobs
# ---------------------------------------------------------------------------

class TestCleanupOldBlobs:
    def test_keeps_specified_removes_others(self, tmp_path: Path):
        hr = history_root(tmp_path)
        hr.mkdir(parents=True, exist_ok=True)

        # Create two snapshot dirs with files
        for sid in ("snap_keep", "snap_remove"):
            files_dir = hr / sid / "files"
            files_dir.mkdir(parents=True)
            (files_dir / "data.csv").write_text("data", encoding="utf-8")

        _cleanup_old_blobs(tmp_path, "snap_keep")

        assert (hr / "snap_keep" / "files").is_dir()
        assert not (hr / "snap_remove" / "files").is_dir()

    def test_no_history_dir_no_error(self, tmp_path: Path):
        _cleanup_old_blobs(tmp_path, "any_sid")  # should not raise
