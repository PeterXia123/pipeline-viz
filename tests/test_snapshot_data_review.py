from pipeline_viz.snapshot_store import SnapshotRecord
from pipeline_viz.models import GraphPayload


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
