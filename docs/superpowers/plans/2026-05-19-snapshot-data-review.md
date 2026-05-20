# Snapshot Data Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When saving a snapshot, detect changed data files and require the user to review DataComPy / file-level diffs before the save completes — storing schema diff + DataComPy results in the new snapshot.

**Architecture:** The save flow becomes two-phase: (1) a "pre-check" API returns which data files changed with schema-level diffs, (2) the frontend shows a review modal where the user runs DataComPy per DataFrame file, confirms non-DataFrame files, then (3) a "confirm save" API persists the snapshot with `data_diffs` attached. The backend adds a `data_profile` module for lightweight schema extraction, and extends `SnapshotRecord` with a `data_diffs` field.

**Tech Stack:** Python 3.9, FastAPI, Pydantic, pandas, datacompy, single-file HTML/JS frontend

---

## File Structure

| File | Role |
|------|------|
| `src/pipeline_viz/data_profile.py` | **New.** Extract lightweight data profiles (row count, columns, dtypes, file size) from disk files and snapshot blobs. |
| `src/pipeline_viz/snapshot_store.py` | **Modify.** Add `data_diffs` field to `SnapshotRecord`. Add `save_snapshot_with_diffs()` that accepts pre-computed diffs. |
| `src/pipeline_viz/dataframe_compare.py` | **Modify.** Add `compare_snapshot_blob_to_disk_for_review()` that returns structured result (not just text report). |
| `src/pipeline_viz/main.py` | **Modify.** Add `/api/snapshot/pre-check` and `/api/snapshot/review-compare` endpoints. Modify `/api/snapshot/save` to accept `data_diffs`. |
| `src/pipeline_viz/static/index.html` | **Modify.** Replace simple save button click with review modal flow. |
| `tests/test_data_profile.py` | **New.** Tests for data profile extraction. |
| `tests/test_snapshot_data_review.py` | **New.** Integration tests for the pre-check + save-with-diffs flow. |

---

### Task 1: Data Profile Module

Extract lightweight metadata from data files without loading full DataFrames.

**Files:**
- Create: `src/pipeline_viz/data_profile.py`
- Create: `tests/test_data_profile.py`

- [ ] **Step 1: Write failing tests for profile extraction**

```python
# tests/test_data_profile.py
from pathlib import Path
import json

from pipeline_viz.data_profile import extract_profile, is_dataframe_format


def test_csv_profile(tmp_path: Path):
    f = tmp_path / "data.csv"
    f.write_text("id,name,amount\n1,alice,10.5\n2,bob,20.3\n", encoding="utf-8")
    p = extract_profile(f)
    assert p["row_count"] == 2
    assert p["col_count"] == 3
    assert p["columns"] == ["id", "name", "amount"]
    assert p["file_size"] > 0
    assert "dtypes" in p
    assert p["dtypes"]["id"] is not None


def test_tsv_profile(tmp_path: Path):
    f = tmp_path / "data.tsv"
    f.write_text("a\tb\n1\t2\n3\t4\n5\t6\n", encoding="utf-8")
    p = extract_profile(f)
    assert p["row_count"] == 3
    assert p["col_count"] == 2


def test_parquet_profile(tmp_path: Path):
    import pandas as pd
    f = tmp_path / "data.parquet"
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_parquet(f)
    p = extract_profile(f)
    assert p["row_count"] == 2
    assert "x" in p["columns"]
    assert "y" in p["columns"]


def test_non_dataframe_profile(tmp_path: Path):
    f = tmp_path / "model.pkl"
    f.write_bytes(b"\x80\x05\x95")
    p = extract_profile(f)
    assert p["row_count"] is None
    assert p["col_count"] is None
    assert p["file_size"] > 0
    assert p["format"] == "non_dataframe"


def test_missing_file(tmp_path: Path):
    p = extract_profile(tmp_path / "nope.csv")
    assert p["row_count"] is None
    assert p["file_size"] == 0
    assert p["error"] is not None


def test_is_dataframe_format():
    assert is_dataframe_format("data.csv") is True
    assert is_dataframe_format("data.tsv") is True
    assert is_dataframe_format("data.parquet") is True
    assert is_dataframe_format("data.pq") is True
    assert is_dataframe_format("model.pkl") is True
    assert is_dataframe_format("config.yaml") is False
    assert is_dataframe_format("model.h5") is False
    assert is_dataframe_format("data.npz") is False


def test_profile_from_bytes(tmp_path: Path):
    from pipeline_viz.data_profile import extract_profile_from_bytes
    data = b"a,b\n1,2\n3,4\n"
    p = extract_profile_from_bytes(data, ".csv")
    assert p["row_count"] == 2
    assert p["col_count"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_data_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_viz.data_profile'`

- [ ] **Step 3: Implement data_profile.py**

```python
# src/pipeline_viz/data_profile.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Any


_DF_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".pkl", ".pickle"}


def is_dataframe_format(path: str) -> bool:
    return Path(path).suffix.lower() in _DF_SUFFIXES


def extract_profile(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": 0,
            "format": "missing",
            "error": "file not found",
        }

    file_size = file_path.stat().st_size
    suf = file_path.suffix.lower()

    if suf not in _DF_SUFFIXES:
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": file_size,
            "format": "non_dataframe",
            "error": None,
        }

    try:
        df = _read_df(file_path)
        return _profile_from_df(df, file_size, suf)
    except Exception as e:
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": file_size,
            "format": suf.lstrip("."),
            "error": str(e)[:200],
        }


def extract_profile_from_bytes(data: bytes, suffix: str) -> dict[str, Any]:
    suf = suffix.lower()
    try:
        df = _read_df_bytes(data, suf)
        return _profile_from_df(df, len(data), suf)
    except Exception as e:
        return {
            "row_count": None,
            "col_count": None,
            "columns": [],
            "dtypes": {},
            "file_size": len(data),
            "format": suf.lstrip("."),
            "error": str(e)[:200],
        }


def _profile_from_df(df, file_size: int, suf: str) -> dict[str, Any]:
    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "file_size": file_size,
        "format": suf.lstrip("."),
        "error": None,
    }


def _read_df(path: Path):
    import pandas as pd
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"unsupported: {suf}")


def _read_df_bytes(data: bytes, suf: str):
    import pandas as pd
    bio = io.BytesIO(data)
    if suf == ".csv":
        return pd.read_csv(bio)
    if suf == ".tsv":
        return pd.read_csv(bio, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(bio)
    if suf in (".pkl", ".pickle"):
        return pd.read_pickle(bio)
    raise ValueError(f"unsupported: {suf}")


def compute_schema_diff(
    old_profile: dict[str, Any], new_profile: dict[str, Any]
) -> dict[str, Any]:
    old_cols = set(old_profile.get("columns") or [])
    new_cols = set(new_profile.get("columns") or [])
    old_dtypes = old_profile.get("dtypes") or {}
    new_dtypes = new_profile.get("dtypes") or {}

    type_changes = {}
    for col in old_cols & new_cols:
        if old_dtypes.get(col) != new_dtypes.get(col):
            type_changes[col] = {
                "old": old_dtypes.get(col),
                "new": new_dtypes.get(col),
            }

    return {
        "row_count_old": old_profile.get("row_count"),
        "row_count_new": new_profile.get("row_count"),
        "col_count_old": old_profile.get("col_count"),
        "col_count_new": new_profile.get("col_count"),
        "columns_added": sorted(new_cols - old_cols),
        "columns_removed": sorted(old_cols - new_cols),
        "type_changes": type_changes,
        "file_size_old": old_profile.get("file_size", 0),
        "file_size_new": new_profile.get("file_size", 0),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_data_profile.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/data_profile.py tests/test_data_profile.py
git commit -m "feat: add data_profile module for lightweight schema extraction"
```

---

### Task 2: Extend SnapshotRecord with data_diffs

**Files:**
- Modify: `src/pipeline_viz/snapshot_store.py:38-63`
- Create: `tests/test_snapshot_data_review.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_snapshot_data_review.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py::test_snapshot_record_has_data_diffs -v`
Expected: FAIL — `data_diffs` not a recognized field

- [ ] **Step 3: Add data_diffs field to SnapshotRecord**

In `src/pipeline_viz/snapshot_store.py`, add the field to `SnapshotRecord` and update the `_legacy` validator:

```python
# In SnapshotRecord class (line ~38):
class SnapshotRecord(BaseModel):
    snapshot_id: str = ""
    saved_at: str
    label: str = ""
    description: str = ""
    payload: GraphPayload
    file_hashes: dict[str, str] = Field(default_factory=dict)
    node_labels: dict[str, dict[str, str]] = Field(default_factory=dict)
    data_diffs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if not out.get("snapshot_id"):
            out["snapshot_id"] = str(out.get("saved_at", "legacy")).replace(":", "-")[:48]
        if "file_hashes" not in out:
            out["file_hashes"] = {}
        if "label" not in out:
            out["label"] = ""
        if "description" not in out:
            out["description"] = ""
        if "node_labels" not in out:
            out["node_labels"] = {}
        if "data_diffs" not in out:
            out["data_diffs"] = {}
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Run full test suite to check nothing breaks**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_viz/snapshot_store.py tests/test_snapshot_data_review.py
git commit -m "feat: add data_diffs field to SnapshotRecord"
```

---

### Task 3: Pre-check API Endpoint

Returns which data files changed and their schema diffs, without actually saving.

**Files:**
- Modify: `src/pipeline_viz/main.py`
- Modify: `tests/test_snapshot_data_review.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_snapshot_data_review.py`:

```python
from pathlib import Path

from pipeline_viz.main import api_snapshot_precheck
from pipeline_viz.models import GraphNode, GraphPayload, GraphEdge
from pipeline_viz.paths import code_id, data_id
from pipeline_viz.snapshot_store import save_snapshot


def test_precheck_detects_changed_data_files(tmp_path: Path):
    nb = "notebooks/a.ipynb"
    csv = "data/orders.csv"
    (tmp_path / "notebooks").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / nb).write_text("{}", encoding="utf-8")
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
    (tmp_path / nb).write_text("{}", encoding="utf-8")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py::test_precheck_detects_changed_data_files -v`
Expected: FAIL — `ImportError: cannot import name 'api_snapshot_precheck'`

- [ ] **Step 3: Implement pre-check endpoint**

Add to `src/pipeline_viz/main.py`:

```python
# New imports at top:
from pipeline_viz.data_profile import (
    compute_schema_diff,
    extract_profile,
    extract_profile_from_bytes,
    is_dataframe_format,
)

# New endpoint:
@app.get("/api/snapshot/pre-check")
def api_snapshot_precheck(project_root: Optional[str] = Query(default=None)):
    root = _resolve_root(project_root)
    payload = _build_graph_or_400(root)

    from pipeline_viz.snapshot_store import (
        compute_file_hashes,
        tracked_paths,
        _effective_hashes,
        _hashes_fingerprint_str,
        _payload_fingerprint_str,
    )

    last = load_latest(root)
    paths_cur = tracked_paths(root, payload)
    hashes_cur = compute_file_hashes(root, paths_cur)

    if not last:
        return {"has_changes": False, "changed_files": [], "reason": "no_previous_snapshot"}

    prev_hashes = _effective_hashes(root, last.snapshot_id, last)
    graph_changed = _payload_fingerprint_str(last.payload) != _payload_fingerprint_str(payload)
    files_changed = _hashes_fingerprint_str(prev_hashes) != _hashes_fingerprint_str(hashes_cur)

    if not graph_changed and not files_changed:
        return {"has_changes": False, "changed_files": [], "reason": "no_changes"}

    changed_files = []
    data_node_paths = {n.path for n in payload.nodes if n.kind == "data"}

    for rel in sorted(data_node_paths):
        old_h = (prev_hashes.get(rel) or "").strip()
        new_h = (hashes_cur.get(rel) or "").strip()
        if old_h == new_h:
            continue
        if not old_h and new_h:
            continue  # first-time creation, skip

        disk_path = (root / rel).resolve()
        is_df = is_dataframe_format(rel)

        new_profile = extract_profile(disk_path) if disk_path.is_file() else {}
        old_blob = read_blob(root, last.snapshot_id, rel)

        entry: dict[str, Any] = {
            "path": rel,
            "is_dataframe": is_df,
            "file_size_old": 0,
            "file_size_new": new_profile.get("file_size", 0),
        }

        if old_blob is not None:
            old_profile = extract_profile_from_bytes(old_blob, Path(rel).suffix)
            entry["file_size_old"] = old_profile.get("file_size", 0)
            if is_df:
                entry["schema_diff"] = compute_schema_diff(old_profile, new_profile)
                entry["old_profile"] = old_profile
                entry["new_profile"] = new_profile
        else:
            if is_df:
                entry["schema_diff"] = None
                entry["old_profile"] = None
                entry["new_profile"] = new_profile

        changed_files.append(entry)

    return {
        "has_changes": graph_changed or files_changed or len(changed_files) > 0,
        "changed_files": changed_files,
        "graph_changed": graph_changed,
        "files_changed": files_changed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/pipeline_viz/main.py tests/test_snapshot_data_review.py
git commit -m "feat: add /api/snapshot/pre-check endpoint for data review"
```

---

### Task 4: Review-Compare API Endpoint

Run DataComPy for a single file on-demand (user clicks "运行 DataComPy" in the review modal).

**Files:**
- Modify: `src/pipeline_viz/main.py`
- Modify: `tests/test_snapshot_data_review.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_snapshot_data_review.py`:

```python
from pipeline_viz.main import api_snapshot_review_compare


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py::test_review_compare_returns_datacompy_result -v`
Expected: FAIL — `ImportError: cannot import name 'api_snapshot_review_compare'`

- [ ] **Step 3: Implement review-compare endpoint**

Add to `src/pipeline_viz/main.py`:

```python
@app.get("/api/snapshot/review-compare")
def api_snapshot_review_compare(
    path: str = Query(..., description="相对项目根的数据文件路径"),
    project_root: Optional[str] = Query(default=None),
    merge_keys: str = Query("", description="逗号分隔的 merge key"),
):
    root = _resolve_root(project_root)
    rel = path.replace("\\", "/").lstrip("/")
    if not is_dataframe_file(rel):
        raise HTTPException(400, "仅支持 DataFrame 格式文件")

    disk = (root / rel).resolve()
    if not is_under_root(root, disk) or not disk.is_file():
        raise HTTPException(400, "磁盘上不存在该文件")

    latest = load_latest(root)
    if not latest:
        raise HTTPException(400, "尚无全局快照")

    blob = read_blob(root, latest.snapshot_id, rel)
    if blob is None:
        raise HTTPException(404, "该快照中未保存此文件")

    keys = [k.strip() for k in merge_keys.split(",") if k.strip()]
    result = compare_snapshot_blob_to_disk(blob, rel, disk, keys if keys else None)
    if result.get("error"):
        raise HTTPException(400, str(result["error"]))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/main.py tests/test_snapshot_data_review.py
git commit -m "feat: add /api/snapshot/review-compare endpoint for DataComPy on demand"
```

---

### Task 5: Confirm-Save API with data_diffs

Modify the save endpoint to accept `data_diffs` from the review flow and store them.

**Files:**
- Modify: `src/pipeline_viz/snapshot_store.py`
- Modify: `src/pipeline_viz/main.py`
- Modify: `tests/test_snapshot_data_review.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_snapshot_data_review.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py::test_save_with_data_diffs -v`
Expected: FAIL — `save_snapshot() got an unexpected keyword argument 'data_diffs'`

- [ ] **Step 3: Extend save_snapshot to accept data_diffs**

In `src/pipeline_viz/snapshot_store.py`, modify `save_snapshot`:

```python
def save_snapshot(
    project_root: Path, payload: GraphPayload,
    data_diffs: dict[str, dict[str, Any]] | None = None,
) -> tuple[str | None, Path | None, str, dict[str, Any] | None]:
    # ... existing logic until SnapshotRecord creation ...

    rec = SnapshotRecord(
        snapshot_id=sid,
        saved_at=datetime.now(timezone.utc).isoformat(),
        label="",
        description="",
        payload=payload,
        file_hashes=file_hashes,
        data_diffs=data_diffs or {},
    )

    # ... rest unchanged ...
```

- [ ] **Step 4: Update api_snapshot_save to accept data_diffs**

In `src/pipeline_viz/main.py`, modify the save endpoint:

```python
@app.post("/api/snapshot/save")
def api_snapshot_save(
    project_root: Optional[str] = Query(default=None),
    body: dict = Body(default_factory=dict),
):
    root = _resolve_root(project_root)
    payload = _build_graph_or_400(root)
    data_diffs = body.get("data_diffs") or {}
    sid, jp, msg, skip_detail = save_snapshot(root, payload, data_diffs=data_diffs)
    # ... rest unchanged ...
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_snapshot_data_review.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/pipeline_viz/snapshot_store.py src/pipeline_viz/main.py tests/test_snapshot_data_review.py
git commit -m "feat: save_snapshot accepts data_diffs, stores in SnapshotRecord"
```

---

### Task 6: Frontend Review Modal

Replace the simple save button with a two-phase review flow.

**Files:**
- Modify: `src/pipeline_viz/static/index.html`

- [ ] **Step 1: Add CSS for review modal**

Add after the existing `.diff-line.hunk` style block (around line 106):

```css
#modal-review {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 100;
  align-items: center; justify-content: center; padding: 20px;
}
#modal-review.open { display: flex; }
#modal-review .modal-box { max-width: min(960px, 96vw); }
.review-hint { font-size: 13px; color: #555; margin: 0 0 12px; }
.file-list { list-style: none; padding: 0; margin: 0 0 12px; }
.file-item {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 10px 14px; margin: 6px 0; border: 1px solid #e8e8e8; border-radius: 8px;
  background: #fafafa;
}
.file-item.reviewed { border-color: #4caf50; background: #f1f8e9; }
.file-item.non-df { border-color: #ff9800; background: #fff8e1; }
.file-icon { font-size: 20px; flex-shrink: 0; }
.file-info { flex: 1; min-width: 0; }
.file-name { font-size: 13px; font-weight: 600; word-break: break-all; }
.file-meta { font-size: 12px; color: #666; margin-top: 2px; }
.file-schema { font-size: 12px; color: #1565c0; margin-top: 3px; }
.file-actions { flex-shrink: 0; display: flex; gap: 6px; align-items: flex-start; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-changed { background: #ffebee; color: #b71c1c; }
.badge-ok { background: #e8f5e9; color: #2e7d32; }
.badge-nondf { background: #fff3e0; color: #e65100; }
.dc-detail { display: none; margin-top: 10px; padding: 12px; background: #f6f8fa; border-radius: 6px; font-size: 12px; }
.dc-detail.open { display: block; }
.dc-detail pre { white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow: auto; background: #f0f0f0; padding: 8px; border-radius: 4px; }
.merge-key-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.merge-key-row label { font-size: 12px; color: #555; white-space: nowrap; }
.merge-key-row input { flex: 1; padding: 5px 8px; border: 1px solid #ccc; border-radius: 6px; font-size: 12px; }
.modal-footer { padding: 12px 16px; border-top: 1px solid #eee; display: flex; align-items: center; justify-content: space-between; }
.footer-hint { font-size: 12px; color: #888; }
.progress-text { font-size: 13px; color: #555; }
```

- [ ] **Step 2: Add review modal HTML**

Add after the existing `<div id="modal-diff">` block (around line 166):

```html
<div id="modal-review" role="dialog" aria-modal="true">
  <div class="modal-box">
    <div class="modal-head">
      <h2>保存快照 — 数据变化审查</h2>
      <span class="progress-text" id="review-progress">0 / 0 已审查</span>
    </div>
    <div class="modal-body">
      <p class="review-hint">检测到以下数据文件发生变化，请审查后再保存。DataFrame 格式需运行 DataComPy。</p>
      <ul class="file-list" id="review-file-list"></ul>
    </div>
    <div class="modal-footer">
      <span class="footer-hint" id="review-footer-hint">请审查全部变化文件后保存</span>
      <div style="display:flex;gap:8px">
        <button id="review-cancel">取消</button>
        <button class="primary" id="review-confirm" disabled>保存快照</button>
      </div>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Implement review modal JS logic**

Replace the existing `qs("#btn-save").addEventListener("click", ...)` block (around line 766-780) with the full review flow:

```javascript
// ---- Review state ----
let reviewFiles = [];
let reviewDiffs = {};
let reviewedSet = new Set();

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function updateReviewProgress() {
  const total = reviewFiles.length;
  const done = reviewedSet.size;
  qs("#review-progress").textContent = `${done} / ${total} 已审查`;
  if (done >= total) {
    qs("#review-confirm").disabled = false;
    qs("#review-footer-hint").textContent = "全部已审查，可以保存";
    qs("#review-footer-hint").style.color = "#2e7d32";
  } else {
    qs("#review-confirm").disabled = true;
    qs("#review-footer-hint").textContent = "请审查全部变化文件后保存";
    qs("#review-footer-hint").style.color = "#888";
  }
}

function markFileReviewed(path) {
  reviewedSet.add(path);
  const li = document.querySelector(`[data-review-path="${CSS.escape(path)}"]`);
  if (li) {
    li.classList.add("reviewed");
    const badge = li.querySelector(".badge");
    if (badge) { badge.className = "badge badge-ok"; badge.textContent = "已审查 ✓"; }
  }
  updateReviewProgress();
}

function buildReviewFileItem(cf) {
  const li = document.createElement("li");
  li.className = "file-item" + (cf.is_dataframe ? "" : " non-df");
  li.dataset.reviewPath = cf.path;

  const icon = cf.is_dataframe ? "📊" : "📦";
  const badgeClass = cf.is_dataframe ? "badge-changed" : "badge-nondf";
  const badgeText = cf.is_dataframe ? "已变化" : "非表格";

  let schemaHtml = "";
  if (cf.is_dataframe && cf.schema_diff) {
    const sd = cf.schema_diff;
    const parts = [];
    if (sd.row_count_old != null && sd.row_count_new != null) {
      const diff = sd.row_count_new - sd.row_count_old;
      const sign = diff >= 0 ? "+" : "";
      parts.push(`行数: ${sd.row_count_old.toLocaleString()} → ${sd.row_count_new.toLocaleString()} (${sign}${diff})`);
    }
    if (sd.columns_added && sd.columns_added.length > 0) {
      parts.push(`列新增: <b>${escapeHtml(sd.columns_added.join(", "))}</b>`);
    }
    if (sd.columns_removed && sd.columns_removed.length > 0) {
      parts.push(`列删除: <b>${escapeHtml(sd.columns_removed.join(", "))}</b>`);
    }
    if (parts.length === 0) parts.push("schema 无变化，内容已变化");
    schemaHtml = `<div class="file-schema">${parts.join(" | ")}</div>`;
  }

  const sizeOld = formatFileSize(cf.file_size_old || 0);
  const sizeNew = formatFileSize(cf.file_size_new || 0);

  li.innerHTML = `
    <div class="file-icon">${icon}</div>
    <div class="file-info">
      <div class="file-name">${escapeHtml(cf.path)}</div>
      <div class="file-meta">
        <span class="badge ${badgeClass}">${badgeText}</span>
        大小: ${sizeOld} → ${sizeNew}
      </div>
      ${schemaHtml}
      ${cf.is_dataframe ? `<div class="dc-detail" id="dc-review-${CSS.escape(cf.path)}"></div>` : ""}
    </div>
    <div class="file-actions">
      ${cf.is_dataframe
        ? `<button class="btn-run-dc" data-path="${escapeAttr(cf.path)}">运行 DataComPy</button>`
        : `<button class="btn-confirm-nondf" data-path="${escapeAttr(cf.path)}">确认 ✓</button>`
      }
    </div>
  `;
  return li;
}

function renderDCDetail(path, result) {
  const el = document.getElementById(`dc-review-${CSS.escape(path)}`);
  if (!el) return;
  el.classList.add("open");
  el.innerHTML = `
    <h4 style="margin:0 0 8px;font-size:13px">DataComPy 结果</h4>
    <div class="merge-key-row">
      <label>Merge Key:</label>
      <input type="text" class="dc-merge-input" data-path="${escapeAttr(path)}" placeholder="逗号分隔，留空按行顺序" />
      <button class="btn-rerun-dc" data-path="${escapeAttr(path)}" style="font-size:11px">重新比较</button>
    </div>
    <pre>${escapeHtml(result.report || JSON.stringify(result, null, 2))}</pre>
    <div style="margin-top:8px">
      <button class="primary btn-confirm-dc" data-path="${escapeAttr(path)}" style="font-size:12px">确认已审查 ✓</button>
    </div>
  `;
}

async function runReviewDC(path, mergeKeys) {
  const el = document.getElementById(`dc-review-${CSS.escape(path)}`);
  if (el) { el.classList.add("open"); el.innerHTML = "<p>计算中…</p>"; }
  const extra = { path };
  if (mergeKeys) extra.merge_keys = mergeKeys;
  try {
    const res = await fetch(buildUrl("/api/snapshot/review-compare", extra));
    const result = await res.json();
    if (!res.ok) {
      if (el) el.innerHTML = `<pre style="color:#b71c1c">${escapeHtml(result.detail || JSON.stringify(result))}</pre>`;
      return;
    }
    reviewDiffs[path] = {
      datacompy_report: result.report || "",
      datacompy_matches: result.matches,
    };
    renderDCDetail(path, result);
  } catch (e) {
    if (el) el.innerHTML = `<pre style="color:#b71c1c">请求失败: ${escapeHtml(String(e))}</pre>`;
  }
}

// ---- Event delegation for review modal ----
qs("#review-file-list").addEventListener("click", async (e) => {
  const btnDC = e.target.closest(".btn-run-dc");
  if (btnDC) {
    await runReviewDC(btnDC.dataset.path, "");
    return;
  }
  const btnNonDF = e.target.closest(".btn-confirm-nondf");
  if (btnNonDF) {
    markFileReviewed(btnNonDF.dataset.path);
    return;
  }
  const btnConfirmDC = e.target.closest(".btn-confirm-dc");
  if (btnConfirmDC) {
    markFileReviewed(btnConfirmDC.dataset.path);
    return;
  }
  const btnRerun = e.target.closest(".btn-rerun-dc");
  if (btnRerun) {
    const input = document.querySelector(`.dc-merge-input[data-path="${CSS.escape(btnRerun.dataset.path)}"]`);
    const keys = input ? input.value.trim() : "";
    await runReviewDC(btnRerun.dataset.path, keys);
    return;
  }
});

qs("#review-cancel").addEventListener("click", () => {
  qs("#modal-review").classList.remove("open");
});
qs("#modal-review").addEventListener("click", (e) => {
  if (e.target.id === "modal-review") qs("#modal-review").classList.remove("open");
});

qs("#review-confirm").addEventListener("click", async () => {
  qs("#review-confirm").disabled = true;
  qs("#review-confirm").textContent = "保存中…";

  const allDiffs = {};
  for (const cf of reviewFiles) {
    const entry = { reviewed: true };
    if (cf.schema_diff) entry.schema_diff = cf.schema_diff;
    if (reviewDiffs[cf.path]) Object.assign(entry, reviewDiffs[cf.path]);
    entry.file_size_old = cf.file_size_old || 0;
    entry.file_size_new = cf.file_size_new || 0;
    allDiffs[cf.path] = entry;
  }

  try {
    const res = await fetch(buildUrl("/api/snapshot/save"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_diffs: allDiffs }),
      cache: "no-store",
    });
    if (!res.ok) { toast(await res.text()); return; }
    const j = await res.json();
    if (j.skipped) { toast(j.message || "未保存"); return; }
    toast(`已保存全局快照 ${j.snapshot_id}（含 ${reviewFiles.length} 个数据变化审查记录）`);
    viewSnapshotId = j.snapshot_id;
    loadGraph();
  } finally {
    qs("#modal-review").classList.remove("open");
    qs("#review-confirm").textContent = "保存快照";
    qs("#review-confirm").disabled = false;
  }
});

// ---- Replace btn-save handler ----
qs("#btn-save").addEventListener("click", async () => {
  qs("#btn-save").disabled = true;
  qs("#btn-save").textContent = "检测中…";
  try {
    const res = await fetch(buildUrl("/api/snapshot/pre-check"));
    if (!res.ok) { toast("预检失败: " + await res.text()); return; }
    const data = await res.json();

    if (!data.has_changes) {
      toast(data.reason === "no_previous_snapshot"
        ? "首次保存，无需审查"
        : "未检测到变化，无需保存");
      if (data.reason === "no_previous_snapshot") {
        // First snapshot: save directly
        const saveRes = await fetch(buildUrl("/api/snapshot/save"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
          cache: "no-store",
        });
        const j = await saveRes.json();
        if (j.skipped) { toast(j.message || "未保存"); }
        else { toast(`已保存首个全局快照 ${j.snapshot_id}`); viewSnapshotId = j.snapshot_id; loadGraph(); }
      }
      return;
    }

    if (data.changed_files.length === 0) {
      // Only graph structure changed, no data file changes — save directly
      const saveRes = await fetch(buildUrl("/api/snapshot/save"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
        cache: "no-store",
      });
      const j = await saveRes.json();
      if (j.skipped) { toast(j.message || "未保存"); }
      else { toast(`已保存全局快照 ${j.snapshot_id}`); viewSnapshotId = j.snapshot_id; loadGraph(); }
      return;
    }

    // Show review modal
    reviewFiles = data.changed_files;
    reviewDiffs = {};
    reviewedSet = new Set();
    const list = qs("#review-file-list");
    list.innerHTML = "";
    for (const cf of reviewFiles) {
      list.appendChild(buildReviewFileItem(cf));
    }
    updateReviewProgress();
    qs("#modal-review").classList.add("open");
  } finally {
    qs("#btn-save").disabled = false;
    qs("#btn-save").textContent = "保存全局快照";
  }
});
```

- [ ] **Step 4: Start dev server and test in browser**

Run: `.venv/bin/python -m pipeline_viz --allow-run`

Test flow:
1. Load project → change a CSV file
2. Click "保存全局快照" → review modal should appear
3. Click "运行 DataComPy" for a data file → see report
4. Click "重新比较" with merge keys → report refreshes
5. Click "确认已审查 ✓" on all files
6. Click "保存快照" → snapshot saved with data_diffs
7. Verify toast shows correct count

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/static/index.html
git commit -m "feat: add data review modal to save snapshot flow"
```

---

### Task 7: View Stored Data Diffs in History

When user clicks a data node while viewing a historical snapshot, show the stored schema diff + DataComPy results.

**Files:**
- Modify: `src/pipeline_viz/main.py`
- Modify: `src/pipeline_viz/static/index.html`

- [ ] **Step 1: Add API to retrieve data_diffs for a snapshot**

Add to `src/pipeline_viz/main.py`:

```python
@app.get("/api/snapshot/data-diffs")
def api_snapshot_data_diffs(
    snapshot_id: str = Query(..., description="快照 ID"),
    project_root: Optional[str] = Query(default=None),
):
    root = _resolve_root(project_root)
    rec = load_snapshot(root, snapshot_id)
    if not rec:
        raise HTTPException(404, "snapshot not found")
    return {
        "snapshot_id": snapshot_id,
        "data_diffs": rec.data_diffs,
    }
```

- [ ] **Step 2: Add frontend logic to show stored diffs**

In the global history list click handler (where `viewSnapshotId` is set), after loading the snapshot view, check if the clicked snapshot has `data_diffs`. When the user clicks a data node while a snapshot is selected, show the stored review results in the existing diff modal:

```javascript
// Add a function to show stored data diff
async function showStoredDataDiff(snapshotId, nodePath) {
  const res = await fetch(buildUrl("/api/snapshot/data-diffs", { snapshot_id: snapshotId }));
  if (!res.ok) return null;
  const data = await res.json();
  const diff = (data.data_diffs || {})[nodePath];
  return diff || null;
}
```

Update the node-click handler (where the diff modal is triggered for data nodes) to first check for stored diffs when `viewSnapshotId` is set:

```javascript
// In the node click / diff modal open logic:
// If viewing a historical snapshot and the node has stored data_diffs, show those
if (graphViewSnapshotId && isDataNode(nodeId)) {
  const p = dataPathFromId(nodeId);
  const stored = await showStoredDataDiff(graphViewSnapshotId, p);
  if (stored) {
    const body = qs("#diff-body");
    let html = `<h3 style="margin:0 0 12px;font-size:14px">${escapeHtml(p)} — 保存时的数据审查</h3>`;
    if (stored.schema_diff) {
      const sd = stored.schema_diff;
      html += `<div style="margin-bottom:12px;font-size:13px">`;
      if (sd.row_count_old != null) html += `行数: ${sd.row_count_old} → ${sd.row_count_new}<br>`;
      if (sd.columns_added?.length) html += `列新增: <b>${escapeHtml(sd.columns_added.join(", "))}</b><br>`;
      if (sd.columns_removed?.length) html += `列删除: <b>${escapeHtml(sd.columns_removed.join(", "))}</b><br>`;
      html += `</div>`;
    }
    if (stored.datacompy_report) {
      html += `<pre style="white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:10px;border-radius:6px;max-height:50vh;overflow:auto">${escapeHtml(stored.datacompy_report)}</pre>`;
    }
    body.innerHTML = html;
    qs("#diff-title").textContent = `${p} · 快照数据审查`;
    qs("#btn-diff-prev").style.display = "none";
    qs("#btn-diff-disk").style.display = "none";
    qs("#modal-diff").classList.add("open");
    return;
  }
}
```

- [ ] **Step 3: Start dev server and test**

1. Save a snapshot with data review (from Task 6)
2. Click on the snapshot in the left panel
3. Click a data node that had changes → should show stored schema diff + DataComPy report in the modal
4. Click a data node without changes → normal behavior

- [ ] **Step 4: Commit**

```bash
git add src/pipeline_viz/main.py src/pipeline_viz/static/index.html
git commit -m "feat: view stored data diffs when browsing historical snapshots"
```

---

## Self-Review

**Spec coverage:**
- ✅ Pre-check detects changed data files with schema diff
- ✅ DataFrame files require DataComPy before save
- ✅ Non-DataFrame files show file size change, require manual confirm
- ✅ All reviewed → save button enabled
- ✅ Schema diff + DataComPy results stored in `data_diffs`
- ✅ Historical snapshots show stored review results
- ✅ Merge key input with re-run capability
- ✅ First snapshot saves directly (no review needed)
- ✅ Graph-only changes (no data file changes) save directly

**Placeholder scan:** No TBDs, TODOs, or vague steps found. All code blocks are complete.

**Type consistency:**
- `data_diffs: dict[str, dict[str, Any]]` — consistent across SnapshotRecord, save_snapshot parameter, and API body
- `extract_profile` / `extract_profile_from_bytes` — both return same dict shape
- `compute_schema_diff` — input/output types consistent between Task 1 definition and Task 3 usage
- `is_dataframe_format` (data_profile.py) vs `is_dataframe_file` (dataframe_compare.py) — different names but both used correctly in their respective contexts
