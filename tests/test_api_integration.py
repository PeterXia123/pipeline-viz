"""Comprehensive API integration tests for pipeline-viz using FastAPI TestClient."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from pipeline_viz.main import app, _graph_cache
from pipeline_viz.snapshot_store import _SNAPSHOT_CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_notebook(cells=None):
    if cells is None:
        cells = [
            {
                "cell_type": "code",
                "source": "print('hello')",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ]
    return json.dumps(
        {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}
    )


def make_manifest(codes: list[dict[str, Any]]) -> str:
    return yaml.dump({"codes": codes})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear snapshot and graph caches before every test."""
    _SNAPSHOT_CACHE.clear()
    _graph_cache._entries.clear()
    yield
    _SNAPSHOT_CACHE.clear()
    _graph_cache._entries.clear()


@pytest.fixture()
def project_env(tmp_path, monkeypatch):
    """Set PIPELINE_VIZ_PROJECT_ROOT and return the tmp dir."""
    monkeypatch.setenv("PIPELINE_VIZ_PROJECT_ROOT", str(tmp_path))
    # Ensure notebook execution is disabled by default
    monkeypatch.delenv("PIPELINE_VIZ_ALLOW_RUN", raising=False)
    # Ensure no API token by default
    monkeypatch.delenv("PIPELINE_VIZ_API_TOKEN", raising=False)
    yield tmp_path


@pytest.fixture()
def simple_project(project_env):
    """Create a minimal project with a notebook, manifest, and CSV data file."""
    root = project_env

    # Create a notebook that reads input.csv and writes output.csv
    nb_content = make_notebook([
        {
            "cell_type": "code",
            "source": "import pandas as pd\ndf = pd.read_csv('data/input.csv')\ndf.to_csv('data/output.csv', index=False)",
            "metadata": {},
            "outputs": [],
            "execution_count": 1,
        }
    ])
    (root / "step1.ipynb").write_text(nb_content, encoding="utf-8")

    # Create manifest
    manifest = make_manifest([
        {
            "path": "step1.ipynb",
            "inputs": ["data/input.csv"],
            "outputs": ["data/output.csv"],
        }
    ])
    (root / "pipeline-manifest.yaml").write_text(manifest, encoding="utf-8")

    # Create data files
    (root / "data").mkdir()
    (root / "data" / "input.csv").write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    (root / "data" / "output.csv").write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    return root


@pytest.fixture()
def client():
    """Return a TestClient for the app."""
    return TestClient(app)


def _save_snapshot(client: TestClient, body: dict | None = None) -> dict:
    """Helper: POST /api/snapshot/save and return JSON."""
    resp = client.post("/api/snapshot/save", json=body or {})
    assert resp.status_code == 200
    return resp.json()


# ===========================================================================
# GET / (index)
# ===========================================================================

class TestIndex:
    def test_index_returns_html_when_static_exists(self, client):
        """If the static dir has index.html, GET / returns 200."""
        from pipeline_viz.main import STATIC
        index_html = STATIC / "index.html"
        if index_html.is_file():
            resp = client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")
        else:
            # Static dir missing -- expect 500
            resp = client.get("/")
            assert resp.status_code == 500
            assert "static UI missing" in resp.json()["detail"]

    def test_index_returns_500_when_static_missing(self, client):
        """Mock STATIC so index.html does not exist."""
        fake = Path("/tmp/_nonexistent_static_dir_for_test")
        with patch("pipeline_viz.main.STATIC", fake):
            resp = client.get("/")
            assert resp.status_code == 500
            assert "static UI missing" in resp.json()["detail"]


# ===========================================================================
# GET /api/graph
# ===========================================================================

class TestApiGraph:
    def test_valid_project_returns_graph(self, client, simple_project):
        resp = client.get("/api/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "graph" in data
        assert "positions" in data
        assert "highlights" in data
        assert "edge_highlights" in data
        assert "highlight_counts" in data
        assert "server_features" in data
        assert "missing_node_ids" in data
        assert "new_node_ids" in data
        graph = data["graph"]
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0

    def test_invalid_project_root_returns_400(self, client, project_env, monkeypatch):
        monkeypatch.delenv("PIPELINE_VIZ_PROJECT_ROOT", raising=False)
        resp = client.get("/api/graph", params={"project_root": "/nonexistent/xyz"})
        assert resp.status_code == 400

    def test_view_snapshot_id_not_found_returns_404(self, client, simple_project):
        resp = client.get(
            "/api/graph", params={"view_snapshot_id": "nonexistent_snap_id"}
        )
        assert resp.status_code == 404

    def test_view_snapshot_id_uses_saved_payload(self, client, simple_project):
        # First save a snapshot
        save_resp = _save_snapshot(client)
        assert save_resp["skipped"] is False
        sid = save_resp["snapshot_id"]

        # Now request graph with that snapshot id
        resp = client.get("/api/graph", params={"view_snapshot_id": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["view_snapshot_id"] == sid
        assert data["highlight_debug"]["using_saved_payload"] is True

    def test_missing_node_detection(self, client, simple_project):
        """When a node's file is deleted from disk, it should appear in missing_node_ids."""
        resp = client.get("/api/graph")
        assert resp.status_code == 200

        # Delete a data file
        (simple_project / "data" / "input.csv").unlink()
        _graph_cache._entries.clear()

        resp2 = client.get("/api/graph")
        assert resp2.status_code == 200
        data = resp2.json()
        assert "data:data/input.csv" in data["missing_node_ids"]

    def test_response_keys(self, client, simple_project):
        resp = client.get("/api/graph")
        data = resp.json()
        expected_keys = {
            "project_root", "manifest", "graph", "positions",
            "notebook_spans", "highlights", "edge_highlights",
            "highlight_counts", "server_features", "parse_note",
            "highlight_debug", "snapshot_time", "snapshot_id",
            "history_count", "view_snapshot_id", "missing_node_ids",
            "new_node_ids",
        }
        assert expected_keys.issubset(set(data.keys()))


# ===========================================================================
# POST /api/snapshot/save
# ===========================================================================

class TestSnapshotSave:
    def test_first_save_returns_snapshot_id(self, client, simple_project):
        data = _save_snapshot(client)
        assert data["skipped"] is False
        assert "snapshot_id" in data
        assert data["nodes"] > 0
        assert data["edges"] > 0

    def test_duplicate_save_is_skipped(self, client, simple_project):
        first = _save_snapshot(client)
        assert first["skipped"] is False

        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()

        second = _save_snapshot(client)
        assert second["skipped"] is True
        assert "reason" in second

    def test_save_with_data_diffs(self, client, simple_project):
        data_diffs = {"data/output.csv": {"rows_added": 2, "rows_removed": 0}}
        data = _save_snapshot(client, body={"data_diffs": data_diffs})
        assert data["skipped"] is False
        sid = data["snapshot_id"]

        # Verify data_diffs are stored
        resp = client.get("/api/snapshot/data-diffs", params={"snapshot_id": sid})
        assert resp.status_code == 200
        assert resp.json()["data_diffs"] == data_diffs


# ===========================================================================
# GET /api/history
# ===========================================================================

class TestHistory:
    def test_empty_history(self, client, project_env):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert resp.json()["items"] == []

    def test_history_after_saves(self, client, simple_project):
        _save_snapshot(client)

        # Modify a file to allow a second save
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n10,20,30\n", encoding="utf-8"
        )
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()
        _save_snapshot(client)

        resp = client.get("/api/history")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2


# ===========================================================================
# POST /api/history/delete
# ===========================================================================

class TestHistoryDelete:
    def test_delete_existing(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.post("/api/history/delete", params={"snapshot_id": sid})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert resp.json()["deleted"] == sid

    def test_delete_nonexistent_returns_404(self, client, project_env):
        resp = client.post(
            "/api/history/delete", params={"snapshot_id": "does_not_exist"}
        )
        assert resp.status_code == 404


# ===========================================================================
# POST /api/history/label
# ===========================================================================

class TestHistoryLabel:
    def test_set_label(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.post(
            "/api/history/label",
            params={"snapshot_id": sid},
            json={"label": "my-label"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["label"] == "my-label"

    def test_set_label_nonexistent_returns_404(self, client, project_env):
        resp = client.post(
            "/api/history/label",
            params={"snapshot_id": "nope"},
            json={"label": "x"},
        )
        assert resp.status_code == 404


# ===========================================================================
# POST /api/history/meta
# ===========================================================================

class TestHistoryMeta:
    def test_update_label_and_description(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.post(
            "/api/history/meta",
            params={"snapshot_id": sid},
            json={"label": "v1.0", "description": "first version"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "v1.0"
        assert body["description"] == "first version"

    def test_partial_update_label_only(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        # Set both first
        client.post(
            "/api/history/meta",
            params={"snapshot_id": sid},
            json={"label": "v1.0", "description": "desc"},
        )
        _SNAPSHOT_CACHE.clear()

        # Update only label
        resp = client.post(
            "/api/history/meta",
            params={"snapshot_id": sid},
            json={"label": "v2.0"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "v2.0"
        # Description should be preserved
        assert body["description"] == "desc"

    def test_partial_update_description_only(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        # Set both first
        client.post(
            "/api/history/meta",
            params={"snapshot_id": sid},
            json={"label": "v1.0", "description": "desc"},
        )
        _SNAPSHOT_CACHE.clear()

        # Update only description
        resp = client.post(
            "/api/history/meta",
            params={"snapshot_id": sid},
            json={"description": "new-desc"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "v1.0"
        assert body["description"] == "new-desc"

    def test_nonexistent_snapshot_returns_404(self, client, project_env):
        resp = client.post(
            "/api/history/meta",
            params={"snapshot_id": "nope"},
            json={"label": "x"},
        )
        assert resp.status_code == 404


# ===========================================================================
# GET /api/node/history
# ===========================================================================

class TestNodeHistory:
    def test_valid_node_id_with_entries(self, client, simple_project):
        _save_snapshot(client)

        resp = client.get(
            "/api/node/history", params={"node_id": "data:data/input.csv"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "data/input.csv"
        assert isinstance(data["entries"], list)

    def test_invalid_node_id_no_prefix_returns_400(self, client, simple_project):
        resp = client.get(
            "/api/node/history", params={"node_id": "no_prefix_here"}
        )
        assert resp.status_code == 400

    def test_view_snapshot_id_filtering(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.get(
            "/api/node/history",
            params={
                "node_id": "data:data/input.csv",
                "view_snapshot_id": sid,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["view_snapshot_id"] == sid
        assert data["commit_filtered"] is True

    def test_view_snapshot_id_not_found_returns_404(self, client, simple_project):
        resp = client.get(
            "/api/node/history",
            params={
                "node_id": "data:data/input.csv",
                "view_snapshot_id": "nonexistent",
            },
        )
        assert resp.status_code == 404


# ===========================================================================
# POST /api/node/meta
# ===========================================================================

class TestNodeMeta:
    def test_set_node_metadata(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.post(
            "/api/node/meta",
            params={"snapshot_id": sid, "node_path": "data/input.csv"},
            json={"label": "Input Data", "description": "Raw input"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["label"] == "Input Data"
        assert body["description"] == "Raw input"

    def test_nonexistent_snapshot_returns_404(self, client, project_env):
        resp = client.post(
            "/api/node/meta",
            params={"snapshot_id": "nope", "node_path": "data/x.csv"},
            json={"label": "x"},
        )
        assert resp.status_code == 404


# ===========================================================================
# GET /api/data/columns
# ===========================================================================

class TestDataColumns:
    def test_csv_file_returns_columns(self, client, simple_project):
        resp = client.get(
            "/api/data/columns", params={"path": "data/input.csv"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == ["a", "b", "c"]
        assert data["total"] == 3

    def test_tsv_file_returns_columns(self, client, simple_project):
        (simple_project / "data" / "test.tsv").write_text(
            "x\ty\tz\n1\t2\t3\n", encoding="utf-8"
        )
        resp = client.get(
            "/api/data/columns", params={"path": "data/test.tsv"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == ["x", "y", "z"]
        assert data["total"] == 3

    def test_nonexistent_file_returns_error(self, client, simple_project):
        resp = client.get(
            "/api/data/columns", params={"path": "data/nope.csv"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is not None
        assert data["columns"] == []

    def test_unsupported_format_returns_error(self, client, simple_project):
        (simple_project / "data" / "test.xyz").write_text("hello")
        resp = client.get(
            "/api/data/columns", params={"path": "data/test.xyz"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["columns"] == []

    def test_path_escaping_root_returns_400(self, client, simple_project):
        resp = client.get(
            "/api/data/columns", params={"path": "../../etc/passwd"}
        )
        assert resp.status_code == 400


# ===========================================================================
# GET /api/file/diff
# ===========================================================================

class TestFileDiff:
    def test_compare_previous_with_two_snapshots(self, client, simple_project):
        _save_snapshot(client)
        # Modify file and save again
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n10,20,30\n", encoding="utf-8"
        )
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()
        save2 = _save_snapshot(client)
        sid2 = save2["snapshot_id"]

        resp = client.get(
            "/api/file/diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid2,
                "compare": "previous",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "unified" in data
        assert "path" in data
        assert data["path"] == "data/output.csv"

    def test_compare_current(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        # Modify file on disk after saving
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n99,99,99\n", encoding="utf-8"
        )

        resp = client.get(
            "/api/file/diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid,
                "compare": "current",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "unified" in data
        assert data["unchanged"] is False

    def test_invalid_compare_returns_400(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        resp = client.get(
            "/api/file/diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid,
                "compare": "invalid",
            },
        )
        assert resp.status_code == 400

    def test_nonexistent_snapshot_returns_404(self, client, simple_project):
        resp = client.get(
            "/api/file/diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": "nope",
                "compare": "previous",
            },
        )
        assert resp.status_code == 404


# ===========================================================================
# GET /api/file/schema-diff
# ===========================================================================

class TestFileSchemaDiff:
    def test_compare_previous_on_csv(self, client, simple_project):
        _save_snapshot(client)

        (simple_project / "data" / "output.csv").write_text(
            "a,b,c,d\n1,2,3,4\n", encoding="utf-8"
        )
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()
        save2 = _save_snapshot(client)
        sid2 = save2["snapshot_id"]

        resp = client.get(
            "/api/file/schema-diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid2,
                "compare": "previous",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "old_profile" in data
        assert "new_profile" in data
        assert "schema_diff" in data

    def test_nonexistent_snapshot_returns_404(self, client, simple_project):
        resp = client.get(
            "/api/file/schema-diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": "nope",
                "compare": "previous",
            },
        )
        assert resp.status_code == 404

    def test_invalid_compare_returns_400(self, client, simple_project):
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]
        resp = client.get(
            "/api/file/schema-diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid,
                "compare": "bogus",
            },
        )
        assert resp.status_code == 400


# ===========================================================================
# GET /api/snapshot/latest
# ===========================================================================

class TestSnapshotLatest:
    def test_no_snapshots_returns_none(self, client, project_env):
        resp = client.get("/api/snapshot/latest")
        assert resp.status_code == 200
        assert resp.json()["snapshot"] is None

    def test_after_save_returns_snapshot(self, client, simple_project):
        _save_snapshot(client)
        resp = client.get("/api/snapshot/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot"] is not None
        assert "snapshot_id" in data["snapshot"]
        assert "saved_at" in data["snapshot"]


# ===========================================================================
# GET /api/snapshot/data-diffs
# ===========================================================================

class TestSnapshotDataDiffs:
    def test_existing_snapshot_with_data_diffs(self, client, simple_project):
        diffs = {"data/output.csv": {"rows_added": 5}}
        save_data = _save_snapshot(client, body={"data_diffs": diffs})
        sid = save_data["snapshot_id"]

        resp = client.get(
            "/api/snapshot/data-diffs", params={"snapshot_id": sid}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot_id"] == sid
        assert data["data_diffs"] == diffs

    def test_nonexistent_snapshot_returns_404(self, client, project_env):
        resp = client.get(
            "/api/snapshot/data-diffs", params={"snapshot_id": "nope"}
        )
        assert resp.status_code == 404


# ===========================================================================
# GET /api/data/datacompy
# ===========================================================================

class TestDatacompy:
    def test_compare_csv_against_snapshot(self, client, simple_project):
        _save_snapshot(client)

        # Modify the CSV on disk
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8"
        )

        resp = client.get(
            "/api/data/datacompy", params={"path": "data/output.csv"}
        )
        assert resp.status_code == 200

    def test_non_dataframe_file_returns_400(self, client, simple_project):
        resp = client.get(
            "/api/data/datacompy", params={"path": "step1.ipynb"}
        )
        assert resp.status_code == 400

    def test_no_snapshots_returns_400(self, client, project_env):
        # Create a csv in the project but no snapshots
        (project_env / "data").mkdir()
        (project_env / "data" / "test.csv").write_text("a\n1\n", encoding="utf-8")
        resp = client.get(
            "/api/data/datacompy", params={"path": "data/test.csv"}
        )
        assert resp.status_code == 400

    def test_file_not_on_disk_returns_400(self, client, simple_project):
        _save_snapshot(client)
        resp = client.get(
            "/api/data/datacompy", params={"path": "data/nonexistent.csv"}
        )
        assert resp.status_code == 400


# ===========================================================================
# GET /api/snapshot/review-compare
# ===========================================================================

class TestSnapshotReviewCompare:
    def test_basic_comparison(self, client, simple_project):
        _save_snapshot(client)
        # Modify file on disk
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8"
        )

        resp = client.get(
            "/api/snapshot/review-compare", params={"path": "data/output.csv"}
        )
        assert resp.status_code == 200

    def test_non_dataframe_returns_400(self, client, simple_project):
        resp = client.get(
            "/api/snapshot/review-compare", params={"path": "step1.ipynb"}
        )
        assert resp.status_code == 400

    def test_no_snapshots_returns_400(self, client, project_env):
        (project_env / "data").mkdir()
        (project_env / "data" / "test.csv").write_text("a\n1\n", encoding="utf-8")
        resp = client.get(
            "/api/snapshot/review-compare", params={"path": "data/test.csv"}
        )
        assert resp.status_code == 400


# ===========================================================================
# GET /api/snapshot/pre-check
# ===========================================================================

class TestSnapshotPreCheck:
    def test_no_previous_snapshot(self, client, simple_project):
        resp = client.get("/api/snapshot/pre-check")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_changes"] is False
        assert data["reason"] == "no_previous_snapshot"

    def test_after_file_changes(self, client, simple_project):
        _save_snapshot(client)
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()

        # Modify a file
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n10,20,30\n40,50,60\n", encoding="utf-8"
        )

        resp = client.get("/api/snapshot/pre-check")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_changes"] is True

    def test_no_changes_after_save(self, client, simple_project):
        _save_snapshot(client)
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()

        resp = client.get("/api/snapshot/pre-check")
        assert resp.status_code == 200
        data = resp.json()
        # Files haven't changed since last save
        assert data["has_changes"] is False


# ===========================================================================
# POST /api/run-notebook
# ===========================================================================

class TestRunNotebook:
    def test_disabled_by_default_returns_403(self, client, simple_project):
        resp = client.post(
            "/api/run-notebook", params={"notebook_rel": "step1.ipynb"}
        )
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower() or "Disabled" in resp.json()["detail"]

    def test_enabled_non_ipynb_returns_400(self, client, simple_project, monkeypatch):
        monkeypatch.setenv("PIPELINE_VIZ_ALLOW_RUN", "1")
        (simple_project / "script.py").write_text("print(1)")
        resp = client.post(
            "/api/run-notebook", params={"notebook_rel": "script.py"}
        )
        assert resp.status_code == 400

    def test_path_escaping_root_returns_400(self, client, simple_project, monkeypatch):
        monkeypatch.setenv("PIPELINE_VIZ_ALLOW_RUN", "1")
        resp = client.post(
            "/api/run-notebook",
            params={"notebook_rel": "../../etc/passwd"},
        )
        assert resp.status_code == 400


# ===========================================================================
# Authentication middleware
# ===========================================================================

class TestAuthMiddleware:
    def test_api_without_token_returns_401(self, client, simple_project, monkeypatch):
        """When PIPELINE_VIZ_API_TOKEN is set, requests without it get 401."""
        # We need to patch the module-level variable since it's read at import
        with patch("pipeline_viz.main._API_TOKEN", "secret123"):
            resp = client.get("/api/graph")
            assert resp.status_code == 401

    def test_api_with_correct_token_succeeds(self, client, simple_project, monkeypatch):
        """Request with correct token succeeds."""
        with patch("pipeline_viz.main._API_TOKEN", "secret123"):
            resp = client.get(
                "/api/graph",
                headers={"Authorization": "Bearer secret123"},
            )
            assert resp.status_code == 200

    def test_non_api_paths_dont_require_token(self, client):
        """Non-API paths (/) should not require a token."""
        with patch("pipeline_viz.main._API_TOKEN", "secret123"):
            resp = client.get("/")
            # Should NOT be 401 (it's either 200 if static exists, or 500)
            assert resp.status_code != 401


# ===========================================================================
# Edge cases and integration scenarios
# ===========================================================================

class TestIntegrationScenarios:
    def test_full_workflow(self, client, simple_project):
        """End-to-end: graph -> save -> history -> label -> node history -> diff."""
        # 1. Get initial graph
        graph_resp = client.get("/api/graph")
        assert graph_resp.status_code == 200

        # 2. Save snapshot
        save1 = _save_snapshot(client)
        assert save1["skipped"] is False
        sid1 = save1["snapshot_id"]

        # 3. Check history
        hist_resp = client.get("/api/history")
        assert len(hist_resp.json()["items"]) == 1

        # 4. Set label
        label_resp = client.post(
            "/api/history/label",
            params={"snapshot_id": sid1},
            json={"label": "baseline"},
        )
        assert label_resp.json()["ok"] is True

        # 5. Set node meta
        node_meta_resp = client.post(
            "/api/node/meta",
            params={"snapshot_id": sid1, "node_path": "data/input.csv"},
            json={"label": "Raw Data", "description": "Input CSV"},
        )
        assert node_meta_resp.json()["ok"] is True

        # 6. Modify data and save again
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n100,200,300\n", encoding="utf-8"
        )
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()
        save2 = _save_snapshot(client)
        assert save2["skipped"] is False
        sid2 = save2["snapshot_id"]

        # 7. Check diff between snapshots
        diff_resp = client.get(
            "/api/file/diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid2,
                "compare": "previous",
            },
        )
        assert diff_resp.status_code == 200
        assert diff_resp.json()["unchanged"] is False

        # 8. Verify history has 2 entries
        hist2 = client.get("/api/history")
        assert len(hist2.json()["items"]) == 2

        # 9. View graph at snapshot 1
        graph_at_s1 = client.get(
            "/api/graph", params={"view_snapshot_id": sid1}
        )
        assert graph_at_s1.status_code == 200

        # 10. Delete first snapshot
        del_resp = client.post(
            "/api/history/delete", params={"snapshot_id": sid1}
        )
        assert del_resp.json()["ok"] is True

        # 11. History should have 1 entry
        hist3 = client.get("/api/history")
        assert len(hist3.json()["items"]) == 1

    def test_snapshot_latest_reflects_most_recent(self, client, simple_project):
        """latest endpoint should always reflect the most recently saved snapshot."""
        save1 = _save_snapshot(client)
        sid1 = save1["snapshot_id"]

        latest = client.get("/api/snapshot/latest").json()
        assert latest["snapshot"]["snapshot_id"] == sid1

        # Modify and save again
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c\n7,8,9\n", encoding="utf-8"
        )
        _SNAPSHOT_CACHE.clear()
        _graph_cache._entries.clear()

        save2 = _save_snapshot(client)
        sid2 = save2["snapshot_id"]

        latest2 = client.get("/api/snapshot/latest").json()
        assert latest2["snapshot"]["snapshot_id"] == sid2

    def test_columns_with_many_columns(self, client, simple_project):
        """Test that column truncation works for wide CSVs."""
        cols = [f"col{i}" for i in range(50)]
        header = ",".join(cols)
        row = ",".join(["0"] * 50)
        (simple_project / "data" / "wide.csv").write_text(
            f"{header}\n{row}\n", encoding="utf-8"
        )
        resp = client.get(
            "/api/data/columns", params={"path": "data/wide.csv"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 50
        assert len(data["columns"]) == 30  # MAX_COLUMNS_PREVIEW
        assert data["truncated"] is True

    def test_schema_diff_compare_current(self, client, simple_project):
        """Schema diff with compare=current compares snapshot blob to disk."""
        save_data = _save_snapshot(client)
        sid = save_data["snapshot_id"]

        # Add a column to the CSV on disk
        (simple_project / "data" / "output.csv").write_text(
            "a,b,c,new_col\n1,2,3,4\n", encoding="utf-8"
        )

        resp = client.get(
            "/api/file/schema-diff",
            params={
                "path": "data/output.csv",
                "snapshot_id": sid,
                "compare": "current",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "schema_diff" in data
        # The new column should be detected
        new_prof = data["new_profile"]
        assert "new_col" in new_prof.get("columns", [])
