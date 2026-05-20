from pathlib import Path

from pipeline_viz.snapshot_store import SnapshotRecord
from pipeline_viz.models import GraphPayload, GraphNode, GraphEdge
from pipeline_viz.paths import code_id, data_id
from pipeline_viz.snapshot_store import save_snapshot
from pipeline_viz.main import api_snapshot_precheck


def test_snapshot_record_has_data_diffs():
    rec = SnapshotRecord(
        snapshot_id="test1",
        saved_at="2026-01-01T00:00:00+00:00",
        payload=GraphPayload(nodes=[], edges=[]),
        data_diffs={
            "data/orders.csv": {
                "schema_diff": {
                    "row_count_old": 100,
                    "row_count_new": 107,
                    "columns_added": ["discount_rate"],
                    "columns_removed": [],
                },
                "datacompy_report": "DataComPy report text here...",
                "datacompy_matches": False,
            }
        },
    )
    assert rec.data_diffs["data/orders.csv"]["schema_diff"]["row_count_new"] == 107
    dumped = rec.model_dump()
    assert "data_diffs" in dumped


def test_snapshot_record_data_diffs_default_empty():
    rec = SnapshotRecord(
        snapshot_id="test2",
        saved_at="2026-01-01T00:00:00+00:00",
        payload=GraphPayload(nodes=[], edges=[]),
    )
    assert rec.data_diffs == {}


def test_legacy_snapshot_without_data_diffs():
    data = {
        "snapshot_id": "old1",
        "saved_at": "2025-01-01T00:00:00+00:00",
        "payload": {"nodes": [], "edges": []},
        "file_hashes": {},
    }
    rec = SnapshotRecord.model_validate(data)
    assert rec.data_diffs == {}


_MINIMAL_NB = '{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[]}'


def test_precheck_detects_changed_data_files(tmp_path: Path):
    nb = "notebooks/a.ipynb"
    csv = "data/orders.csv"
    (tmp_path / "notebooks").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / nb).write_text(_MINIMAL_NB, encoding="utf-8")
    (tmp_path / csv).write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")

    p = GraphPayload(
        nodes=[
            GraphNode(id=code_id(nb), kind="notebook", label="a.ipynb", path=nb),
            GraphNode(id=data_id(csv), kind="data", label="orders.csv", path=csv),
        ],
        edges=[GraphEdge(id="e0", source=code_id(nb), target=data_id(csv), kind="output")],
    )
    save_snapshot(tmp_path, p)

    (tmp_path / csv).write_text("id,amount,discount\n1,10,0.1\n2,20,0.2\n3,30,0.3\n", encoding="utf-8")

    result = api_snapshot_precheck(project_root=str(tmp_path))
    assert result["has_changes"] is True
    changed = result["changed_files"]
    assert len(changed) == 1
    cf = changed[0]
    assert cf["path"] == csv
    assert cf["is_dataframe"] is True
    assert cf["schema_diff"]["row_count_old"] == 2
    assert cf["schema_diff"]["row_count_new"] == 3
    assert "discount" in cf["schema_diff"]["columns_added"]


def test_precheck_no_changes(tmp_path: Path):
    nb = "notebooks/a.ipynb"
    (tmp_path / "notebooks").mkdir(parents=True)
    (tmp_path / nb).write_text(_MINIMAL_NB, encoding="utf-8")
    p = GraphPayload(
        nodes=[GraphNode(id=code_id(nb), kind="notebook", label="a.ipynb", path=nb)],
        edges=[],
    )
    save_snapshot(tmp_path, p)
    result = api_snapshot_precheck(project_root=str(tmp_path))
    assert result["has_changes"] is False


def test_precheck_non_dataframe_file(tmp_path: Path):
    pkl = "models/model.pkl"
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / pkl).write_bytes(b"\x80\x05\x95abc")
    p = GraphPayload(
        nodes=[GraphNode(id=data_id(pkl), kind="data", label="model.pkl", path=pkl)],
        edges=[],
    )
    save_snapshot(tmp_path, p)

    (tmp_path / pkl).write_bytes(b"\x80\x05\x95abcdef")

    result = api_snapshot_precheck(project_root=str(tmp_path))
    assert result["has_changes"] is True
    cf = result["changed_files"][0]
    assert cf["is_dataframe"] is False
    assert cf["file_size_old"] > 0
    assert cf["file_size_new"] > 0


from pipeline_viz.main import api_snapshot_review_compare
from pipeline_viz.snapshot_store import load_latest


def test_save_with_data_diffs(tmp_path: Path):
    csv = "data/orders.csv"
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / csv).write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")

    p = GraphPayload(
        nodes=[GraphNode(id=data_id(csv), kind="data", label="orders.csv", path=csv)],
        edges=[],
    )
    save_snapshot(tmp_path, p)
    (tmp_path / csv).write_text("id,amount\n1,10\n2,25\n3,30\n", encoding="utf-8")

    diffs = {
        csv: {
            "schema_diff": {
                "row_count_old": 2,
                "row_count_new": 3,
                "columns_added": [],
                "columns_removed": [],
            },
            "datacompy_report": "Some report text",
            "datacompy_matches": False,
        }
    }
    sid, _, msg, _ = save_snapshot(tmp_path, p, data_diffs=diffs)
    assert sid is not None
    assert msg == "saved"

    rec = load_latest(tmp_path)
    assert rec.data_diffs == diffs


def test_review_compare_returns_datacompy_result(tmp_path: Path):
    csv = "data/orders.csv"
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / csv).write_text("id,amount\n1,10\n2,20\n", encoding="utf-8")

    p = GraphPayload(
        nodes=[GraphNode(id=data_id(csv), kind="data", label="orders.csv", path=csv)],
        edges=[],
    )
    save_snapshot(tmp_path, p)
    (tmp_path / csv).write_text("id,amount\n1,10\n2,25\n3,30\n", encoding="utf-8")

    result = api_snapshot_review_compare(
        path=csv,
        project_root=str(tmp_path),
        merge_keys="id",
    )
    assert "report" in result
    assert result["matches"] is False
