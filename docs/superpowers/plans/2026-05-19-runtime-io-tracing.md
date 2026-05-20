# Runtime I/O Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically discover file I/O during notebook execution by injecting monkey-patches into the Jupyter kernel, then merge traced paths into the pipeline graph.

**Architecture:** A new `io_tracer.py` module generates Python startup code that patches `open`, pandas, json, numpy, and pickle I/O functions to record paths. After notebook execution, a hidden cell collects the trace. The server persists results to `.pipeline-viz/runtime_io.json` and `build_graph` merges them with static analysis and manifest data.

**Tech Stack:** Python 3.9+, nbclient/nbformat (existing), Pydantic (existing), pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/pipeline_viz/paths.py` | Add shared `DATA_EXT` regex (moved from `notebook_parser.py`) |
| `src/pipeline_viz/notebook_parser.py` | Import `DATA_EXT` from `paths.py` instead of defining `_DATA_EXT` locally |
| `src/pipeline_viz/io_tracer.py` | **New.** Startup code generation, trace output parsing, runtime_io cache |
| `src/pipeline_viz/graph_builder.py` | Accept optional `runtime_io` param, merge into graph |
| `src/pipeline_viz/main.py` | Wire tracing into notebook execution and graph cache |
| `tests/test_io_tracer.py` | **New.** Unit tests for io_tracer |
| `tests/test_runtime_graph_merge.py` | **New.** Test that runtime_io merges into graph correctly |

---

### Task 1: Move `_DATA_EXT` to `paths.py` as shared constant

**Files:**
- Modify: `src/pipeline_viz/paths.py`
- Modify: `src/pipeline_viz/notebook_parser.py`

- [ ] **Step 1: Add `DATA_EXT` to `paths.py`**

Add at the end of `src/pipeline_viz/paths.py`, before the `data_id`/`code_id` functions:

```python
DATA_EXT = re.compile(
    r"\.(csv|tsv|parquet|pq|json|jsonl|ndjson|feather|ftr|pkl|pickle|"
    r"joblib|h5|hdf5|txt|xml|yaml|yml|npz|npy|xlsx|xls|orc|ipc|arrow)$",
    re.IGNORECASE,
)
```

Add `import re` to the imports of `paths.py`.

- [ ] **Step 2: Update `notebook_parser.py` to import from `paths.py`**

In `src/pipeline_viz/notebook_parser.py`, replace the local `_DATA_EXT` definition (lines 67-71) with:

```python
from pipeline_viz.paths import DATA_EXT as _DATA_EXT
```

Remove the old `_DATA_EXT = re.compile(...)` block. All existing references to `_DATA_EXT` in the file remain unchanged.

- [ ] **Step 3: Run existing tests**

Run: `cd /Users/admin/Documents/pipeline-viz && .venv/bin/python -m pytest tests/ -x -q`
Expected: All 61 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/pipeline_viz/paths.py src/pipeline_viz/notebook_parser.py
git commit -m "refactor: move DATA_EXT regex to paths.py as shared constant"
```

---

### Task 2: Create `io_tracer.py` — startup code generation

**Files:**
- Create: `src/pipeline_viz/io_tracer.py`
- Create: `tests/test_io_tracer.py`

- [ ] **Step 1: Write the failing test for `generate_trace_startup_code`**

Create `tests/test_io_tracer.py`:

```python
"""Tests for runtime I/O tracing."""
from __future__ import annotations


def test_generate_trace_startup_code_is_valid_python():
    from pipeline_viz.io_tracer import generate_trace_startup_code

    code = generate_trace_startup_code("/tmp/project")
    compile(code, "<startup>", "exec")


def test_generate_trace_startup_code_contains_trace_dict():
    from pipeline_viz.io_tracer import generate_trace_startup_code

    code = generate_trace_startup_code("/tmp/project")
    assert "__pipeline_viz_io_trace__" in code
    assert '"reads"' in code
    assert '"writes"' in code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py -x -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline_viz.io_tracer'`

- [ ] **Step 3: Write `generate_trace_startup_code`**

Create `src/pipeline_viz/io_tracer.py`:

```python
"""Runtime I/O tracing: inject monkey-patches into Jupyter kernels to capture file paths."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from pipeline_viz.paths import DATA_EXT


def generate_trace_startup_code(project_root: str) -> str:
    return textwrap.dedent(f'''\
        import os as __pviz_os__
        import functools as __pviz_ft__

        __pipeline_viz_io_trace__ = {{"reads": [], "writes": []}}
        __pviz_project_root__ = __pviz_os__.path.abspath({project_root!r})

        def __pviz_record__(path, kind):
            try:
                if isinstance(path, __pviz_os__.PathLike):
                    path = str(path)
                if not isinstance(path, str):
                    return
                path = __pviz_os__.path.abspath(path)
                bucket = __pipeline_viz_io_trace__["reads"] if kind == "r" else __pipeline_viz_io_trace__["writes"]
                if path not in bucket:
                    bucket.append(path)
            except Exception:
                pass

        # --- patch builtins.open ---
        __pviz_orig_open__ = open
        @__pviz_ft__.wraps(__pviz_orig_open__)
        def __pviz_open__(*args, **kwargs):
            try:
                p = args[0] if args else kwargs.get("file")
                mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
                if isinstance(mode, str) and any(c in mode for c in "wax"):
                    __pviz_record__(p, "w")
                else:
                    __pviz_record__(p, "r")
            except Exception:
                pass
            return __pviz_orig_open__(*args, **kwargs)
        import builtins
        builtins.open = __pviz_open__

        # --- patch pandas read ---
        try:
            import pandas as __pviz_pd__
            for __pviz_rn__ in ("read_csv", "read_json", "read_parquet", "read_excel",
                                "read_pickle", "read_feather", "read_hdf"):
                __pviz_orig__ = getattr(__pviz_pd__, __pviz_rn__, None)
                if __pviz_orig__ is None:
                    continue
                def __pviz_make_r__(__orig=__pviz_orig__):
                    @__pviz_ft__.wraps(__orig)
                    def wrapper(*a, **kw):
                        try:
                            __pviz_record__(a[0] if a else kw.get("filepath_or_buffer"), "r")
                        except Exception:
                            pass
                        return __orig(*a, **kw)
                    return wrapper
                setattr(__pviz_pd__, __pviz_rn__, __pviz_make_r__())
        except ImportError:
            pass

        # --- patch pandas write ---
        try:
            import pandas as __pviz_pd2__
            for __pviz_wn__ in ("to_csv", "to_json", "to_parquet", "to_excel", "to_pickle"):
                __pviz_orig_w__ = getattr(__pviz_pd2__.DataFrame, __pviz_wn__, None)
                if __pviz_orig_w__ is None:
                    continue
                def __pviz_make_w__(__orig=__pviz_orig_w__):
                    @__pviz_ft__.wraps(__orig)
                    def wrapper(self, *a, **kw):
                        try:
                            p = a[0] if a else kw.get("path_or_buf") or kw.get("path")
                            __pviz_record__(p, "w")
                        except Exception:
                            pass
                        return __orig(self, *a, **kw)
                    return wrapper
                setattr(__pviz_pd2__.DataFrame, __pviz_wn__, __pviz_make_w__())
        except ImportError:
            pass

        # --- patch json load/dump ---
        try:
            import json as __pviz_json_mod__
            __pviz_orig_jload__ = __pviz_json_mod__.load
            @__pviz_ft__.wraps(__pviz_orig_jload__)
            def __pviz_jload__(fp, *a, **kw):
                try:
                    name = getattr(fp, "name", None)
                    if name:
                        __pviz_record__(name, "r")
                except Exception:
                    pass
                return __pviz_orig_jload__(fp, *a, **kw)
            __pviz_json_mod__.load = __pviz_jload__

            __pviz_orig_jdump__ = __pviz_json_mod__.dump
            @__pviz_ft__.wraps(__pviz_orig_jdump__)
            def __pviz_jdump__(obj, fp, *a, **kw):
                try:
                    name = getattr(fp, "name", None)
                    if name:
                        __pviz_record__(name, "w")
                except Exception:
                    pass
                return __pviz_orig_jdump__(obj, fp, *a, **kw)
            __pviz_json_mod__.dump = __pviz_jdump__
        except Exception:
            pass

        # --- patch numpy ---
        try:
            import numpy as __pviz_np__
            for __pviz_nn__, __pviz_nk__ in [("load", "r"), ("save", "w"), ("savez", "w")]:
                __pviz_norig__ = getattr(__pviz_np__, __pviz_nn__, None)
                if __pviz_norig__ is None:
                    continue
                def __pviz_make_np__(__orig=__pviz_norig__, __k=__pviz_nk__):
                    @__pviz_ft__.wraps(__orig)
                    def wrapper(*a, **kw):
                        try:
                            __pviz_record__(a[0] if a else kw.get("file"), __k)
                        except Exception:
                            pass
                        return __orig(*a, **kw)
                    return wrapper
                setattr(__pviz_np__, __pviz_nn__, __pviz_make_np__())
        except ImportError:
            pass

        # --- patch pickle ---
        try:
            import pickle as __pviz_pickle__
            __pviz_orig_pload__ = __pviz_pickle__.load
            @__pviz_ft__.wraps(__pviz_orig_pload__)
            def __pviz_pload__(fp, *a, **kw):
                try:
                    __pviz_record__(getattr(fp, "name", None), "r")
                except Exception:
                    pass
                return __pviz_orig_pload__(fp, *a, **kw)
            __pviz_pickle__.load = __pviz_pload__

            __pviz_orig_pdump__ = __pviz_pickle__.dump
            @__pviz_ft__.wraps(__pviz_orig_pdump__)
            def __pviz_pdump__(obj, fp, *a, **kw):
                try:
                    __pviz_record__(getattr(fp, "name", None), "w")
                except Exception:
                    pass
                return __pviz_orig_pdump__(obj, fp, *a, **kw)
            __pviz_pickle__.dump = __pviz_pdump__
        except Exception:
            pass
    ''')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py -x -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/io_tracer.py tests/test_io_tracer.py
git commit -m "feat: add io_tracer with startup code generation"
```

---

### Task 3: Add trace collection and parsing to `io_tracer.py`

**Files:**
- Modify: `src/pipeline_viz/io_tracer.py`
- Modify: `tests/test_io_tracer.py`

- [ ] **Step 1: Write failing tests for `generate_trace_collect_code` and `parse_trace_output`**

Append to `tests/test_io_tracer.py`:

```python
def test_generate_trace_collect_code_is_valid_python():
    from pipeline_viz.io_tracer import generate_trace_collect_code

    code = generate_trace_collect_code()
    compile(code, "<collect>", "exec")
    assert "__PIPELINE_VIZ_TRACE__:" in code


def test_parse_trace_output_extracts_paths():
    from pipeline_viz.io_tracer import parse_trace_output

    outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": [
                '__PIPELINE_VIZ_TRACE__:{"reads": ["/project/data/in.csv"], "writes": ["/project/out.csv"]}\n'
            ],
        }
    ]
    reads, writes = parse_trace_output(outputs)
    assert reads == ["/project/data/in.csv"]
    assert writes == ["/project/out.csv"]


def test_parse_trace_output_empty_on_no_trace():
    from pipeline_viz.io_tracer import parse_trace_output

    outputs = [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}]
    reads, writes = parse_trace_output(outputs)
    assert reads == []
    assert writes == []


def test_parse_trace_output_handles_text_as_string():
    from pipeline_viz.io_tracer import parse_trace_output

    outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": '__PIPELINE_VIZ_TRACE__:{"reads": ["/a.csv"], "writes": []}\n',
        }
    ]
    reads, writes = parse_trace_output(outputs)
    assert reads == ["/a.csv"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py -x -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `generate_trace_collect_code` and `parse_trace_output`**

Add to `src/pipeline_viz/io_tracer.py`:

```python
_TRACE_PREFIX = "__PIPELINE_VIZ_TRACE__:"


def generate_trace_collect_code() -> str:
    return (
        "import json as __pviz_json__\n"
        f'print("{_TRACE_PREFIX}" + __pviz_json__.dumps(__pipeline_viz_io_trace__))\n'
    )


def parse_trace_output(cell_outputs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    for out in cell_outputs:
        if out.get("output_type") != "stream" or out.get("name") != "stdout":
            continue
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        for line in text.splitlines():
            if line.startswith(_TRACE_PREFIX):
                try:
                    data = json.loads(line[len(_TRACE_PREFIX):])
                    return data.get("reads", []), data.get("writes", [])
                except (json.JSONDecodeError, TypeError):
                    pass
    return [], []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py -x -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/io_tracer.py tests/test_io_tracer.py
git commit -m "feat: add trace collection code generation and output parsing"
```

---

### Task 4: Add path filtering and runtime_io cache to `io_tracer.py`

**Files:**
- Modify: `src/pipeline_viz/io_tracer.py`
- Modify: `tests/test_io_tracer.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_io_tracer.py`:

```python
from pathlib import Path


def test_filter_project_paths_keeps_data_files_under_root(tmp_path):
    from pipeline_viz.io_tracer import filter_project_paths

    paths = [
        str(tmp_path / "data" / "in.csv"),
        str(tmp_path / "out.json"),
        "/outside/project/secret.csv",
        str(tmp_path / "src" / "main.py"),  # .py not a data ext
    ]
    result = filter_project_paths(tmp_path, paths)
    assert result == {"data/in.csv", "out.json"}


def test_load_save_runtime_io(tmp_path):
    from pipeline_viz.io_tracer import load_runtime_io, update_runtime_io

    assert load_runtime_io(tmp_path) == {}

    update_runtime_io(tmp_path, "nb.ipynb", {"a.csv"}, {"b.csv"})
    data = load_runtime_io(tmp_path)
    assert data == {"nb.ipynb": {"inputs": ["a.csv"], "outputs": ["b.csv"]}}

    update_runtime_io(tmp_path, "nb2.ipynb", {"c.csv"}, set())
    data = load_runtime_io(tmp_path)
    assert "nb.ipynb" in data
    assert data["nb2.ipynb"] == {"inputs": ["c.csv"], "outputs": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py::test_filter_project_paths_keeps_data_files_under_root tests/test_io_tracer.py::test_load_save_runtime_io -x -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `filter_project_paths`, `load_runtime_io`, `update_runtime_io`**

Add to `src/pipeline_viz/io_tracer.py`:

```python
def filter_project_paths(project_root: Path, paths: list[str]) -> set[str]:
    root = project_root.resolve()
    result: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        try:
            p = Path(raw).resolve()
            rel = p.relative_to(root)
        except (ValueError, OSError):
            continue
        s = rel.as_posix()
        if s.startswith(".."):
            continue
        if not DATA_EXT.search(s):
            continue
        result.add(s)
    return result


def _runtime_io_path(project_root: Path) -> Path:
    return project_root / ".pipeline-viz" / "runtime_io.json"


def load_runtime_io(project_root: Path) -> dict[str, dict[str, list[str]]]:
    p = _runtime_io_path(project_root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def update_runtime_io(
    project_root: Path, notebook_rel: str, inputs: set[str], outputs: set[str]
) -> None:
    data = load_runtime_io(project_root)
    data[notebook_rel] = {
        "inputs": sorted(inputs),
        "outputs": sorted(outputs),
    }
    p = _runtime_io_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_io_tracer.py -x -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/io_tracer.py tests/test_io_tracer.py
git commit -m "feat: add path filtering and runtime_io cache"
```

---

### Task 5: Merge runtime_io into `build_graph`

**Files:**
- Modify: `src/pipeline_viz/graph_builder.py`
- Create: `tests/test_runtime_graph_merge.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_runtime_graph_merge.py`:

```python
"""Test that runtime_io merges into the pipeline graph."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

from pipeline_viz.graph_builder import build_graph
from pipeline_viz.paths import code_id, data_id


def _make_minimal_notebook(path: Path) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("x = 1"))
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def test_runtime_io_adds_data_edges(tmp_path):
    nb_rel = "notebooks/analysis.ipynb"
    _make_minimal_notebook(tmp_path / nb_rel)

    runtime_io = {
        nb_rel: {
            "inputs": ["data/raw/orders.csv", "data/raw/users.csv"],
            "outputs": ["data/out/report.csv"],
        }
    }
    g = build_graph(tmp_path, None, runtime_io=runtime_io)

    node_ids = {n.id for n in g.nodes}
    assert data_id("data/raw/orders.csv") in node_ids
    assert data_id("data/raw/users.csv") in node_ids
    assert data_id("data/out/report.csv") in node_ids
    assert code_id(nb_rel) in node_ids

    edge_tuples = {(e.source, e.target, e.kind) for e in g.edges}
    assert (data_id("data/raw/orders.csv"), code_id(nb_rel), "input") in edge_tuples
    assert (data_id("data/raw/users.csv"), code_id(nb_rel), "input") in edge_tuples
    assert (code_id(nb_rel), data_id("data/out/report.csv"), "output") in edge_tuples


def test_runtime_io_none_is_noop(tmp_path):
    nb_rel = "notebooks/simple.ipynb"
    _make_minimal_notebook(tmp_path / nb_rel)

    g1 = build_graph(tmp_path, None, runtime_io=None)
    g2 = build_graph(tmp_path, None)
    assert len(g1.nodes) == len(g2.nodes)
    assert len(g1.edges) == len(g2.edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runtime_graph_merge.py -x -v`
Expected: FAIL — `TypeError: build_graph() got an unexpected keyword argument 'runtime_io'`

- [ ] **Step 3: Modify `build_graph` to accept and merge `runtime_io`**

In `src/pipeline_viz/graph_builder.py`, change the function signature:

```python
def build_graph(
    project_root: Path,
    manifest_path: Path | None,
    runtime_io: dict[str, dict[str, list[str]]] | None = None,
) -> GraphPayload:
```

Add the import at the top of the file:

```python
from pipeline_viz.paths import (
    SCAN_SKIP_DIR_NAMES,
    code_id,
    data_id,
    normalize_rel,
    rel_has_scan_skip_dir,
)
```

(This adds `normalize_rel` to the existing import — it was already imported, verify and adjust if needed.)

Inside `build_graph`, in the notebook loop (after `ins, outs = _merge_io(...)` on line 99), add runtime merge:

```python
        rt = (runtime_io or {}).get(nb_rel, {})
        rt_ins = {normalize_rel(root, x) for x in rt.get("inputs", [])} if rt else set()
        rt_outs = {normalize_rel(root, x) for x in rt.get("outputs", [])} if rt else set()
        ins = ins | {x for x in rt_ins if x}
        outs = outs | {x for x in rt_outs if x}
```

Similarly, in the `.py` loop (after `ins, outs = _merge_io(...)` on line 125), add the same pattern:

```python
        rt = (runtime_io or {}).get(py_rel, {})
        rt_ins = {normalize_rel(root, x) for x in rt.get("inputs", [])} if rt else set()
        rt_outs = {normalize_rel(root, x) for x in rt.get("outputs", [])} if rt else set()
        ins = ins | {x for x in rt_ins if x}
        outs = outs | {x for x in rt_outs if x}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_runtime_graph_merge.py tests/ -x -q`
Expected: All tests pass (existing 61 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline_viz/graph_builder.py tests/test_runtime_graph_merge.py
git commit -m "feat: merge runtime_io into build_graph"
```

---

### Task 6: Wire tracing into notebook execution and graph cache

**Files:**
- Modify: `src/pipeline_viz/main.py`

- [ ] **Step 1: Add imports**

At the top of `src/pipeline_viz/main.py`, add to the imports:

```python
from pipeline_viz.io_tracer import (
    filter_project_paths,
    generate_trace_collect_code,
    generate_trace_startup_code,
    load_runtime_io,
    parse_trace_output,
    update_runtime_io,
)
```

- [ ] **Step 2: Modify `_GraphCache.get_or_build` to load runtime_io**

Change `get_or_build` to load and pass runtime_io:

```python
def get_or_build(self, root: Path, manifest_path: Optional[Path]) -> GraphPayload:
    key = str(root.resolve())
    fps = self._scan_mtimes(root, manifest_path)
    cached = self._entries.get(key)
    if cached is not None and cached[0] == fps:
        return cached[1]
    runtime_io = load_runtime_io(root)
    payload = build_graph(root, manifest_path, runtime_io=runtime_io)
    self._entries[key] = (fps, payload)
    return payload
```

- [ ] **Step 3: Add `runtime_io.json` to cache fingerprint**

In `_scan_mtimes`, add after the manifest check:

```python
rio = root / ".pipeline-viz" / "runtime_io.json"
if rio.is_file():
    try:
        fps[str(rio)] = rio.stat().st_mtime
    except OSError:
        pass
```

- [ ] **Step 4: Modify `_run_notebook_execute_inplace` to inject tracing and collect results**

Change `_run_notebook_execute_inplace` signature to return traced I/O:

```python
def _run_notebook_execute_inplace(project_root: Path, nb_path: Path) -> tuple[set[str], set[str]]:
```

After creating the `ExecutePreprocessor`, before `ep.preprocess(notebook, resources)`, prepend a startup cell:

```python
    startup_src = generate_trace_startup_code(str(project_root.resolve()))
    startup_cell = nbformat.v4.new_code_cell(startup_src)
    startup_cell.metadata["pipeline_viz_trace"] = True
    notebook.cells.insert(0, startup_cell)
```

After `ep.preprocess(notebook, resources)` (inside the try block, before the CellExecutionError except), append a collect cell and execute it:

```python
    collect_cell = nbformat.v4.new_code_cell(generate_trace_collect_code())
    collect_cell.metadata["pipeline_viz_trace"] = True
    notebook.cells.append(collect_cell)
    try:
        ep.preprocess(notebook, resources)
    except CellExecutionError:
        pass
```

Wait — the second `ep.preprocess` would re-run ALL cells. Instead, only the collect cell needs to run. Use a simpler approach: run the collect code in a fresh single-cell notebook on the same kernel. But `nbclient` doesn't expose per-cell execution easily when using `ExecutePreprocessor`.

Better approach: append the collect cell BEFORE `ep.preprocess`, so it runs as part of the notebook. Then parse its output afterward.

Revised implementation — replace the entire function body:

```python
def _run_notebook_execute_inplace(project_root: Path, nb_path: Path) -> tuple[set[str], set[str]]:
    try:
        import nbformat
        from nbclient.exceptions import CellExecutionError
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail="执行 notebook 需要 nbformat、nbconvert、ipykernel。",
        ) from e

    timeout = int(os.environ.get("PIPELINE_VIZ_RUN_TIMEOUT", "600"))
    startup_timeout = int(os.environ.get("PIPELINE_VIZ_KERNEL_STARTUP_TIMEOUT", "120"))
    env = _env_for_notebook_run(project_root)

    with nb_path.open(encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    startup_cell = nbformat.v4.new_code_cell(
        generate_trace_startup_code(str(project_root.resolve()))
    )
    startup_cell.metadata["pipeline_viz_trace"] = True
    collect_cell = nbformat.v4.new_code_cell(generate_trace_collect_code())
    collect_cell.metadata["pipeline_viz_trace"] = True

    notebook.cells.insert(0, startup_cell)
    notebook.cells.append(collect_cell)

    ep = ExecutePreprocessor(
        timeout=timeout,
        startup_timeout=startup_timeout,
        kernel_manager_class="jupyter_client.manager.KernelManager",
    )
    resources = {"metadata": {"path": str(project_root.resolve())}}
    ep.km_kwargs = {"env": env}
    try:
        ep.preprocess(notebook, resources)
    except CellExecutionError as e:
        detail = str(e).strip()
        if len(detail) > 4000:
            detail = detail[:4000] + "\n…(truncated)"
        raise HTTPException(status_code=500, detail=detail) from e

    traced_reads, traced_writes = parse_trace_output(
        notebook.cells[-1].get("outputs", [])
    )
    traced_inputs = filter_project_paths(project_root, traced_reads)
    traced_outputs = filter_project_paths(project_root, traced_writes)

    notebook.cells = [
        c for c in notebook.cells if not c.metadata.get("pipeline_viz_trace")
    ]

    data = json.loads(nbformat.writes(notebook))
    sanitize_notebook_dict(data)
    notebook = nbformat.from_dict(data)
    with nb_path.open("w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    return traced_inputs, traced_outputs
```

- [ ] **Step 5: Update `api_run_notebook` to use traced results**

Replace the `api_run_notebook` function body:

```python
@app.post("/api/run-notebook")
def api_run_notebook(
    notebook_rel: str = Query(..., description="Notebook path relative to project root"),
    project_root: Optional[str] = Query(default=None),
):
    if os.environ.get("PIPELINE_VIZ_ALLOW_RUN", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "Notebook execution disabled. Set PIPELINE_VIZ_ALLOW_RUN=1")
    root = _resolve_root(project_root)
    nb = (root / notebook_rel).resolve()
    if not is_under_root(root, nb):
        raise HTTPException(400, "path escapes project root")
    if not nb.is_file() or nb.suffix.lower() != ".ipynb":
        raise HTTPException(400, "not a .ipynb file under project root")
    traced_inputs, traced_outputs = _run_notebook_execute_inplace(root, nb)
    if traced_inputs or traced_outputs:
        update_runtime_io(root, notebook_rel, traced_inputs, traced_outputs)
    _graph_cache.invalidate(root)
    return {
        "ok": True,
        "notebook": notebook_rel,
        "traced_io": {
            "inputs": sorted(traced_inputs),
            "outputs": sorted(traced_outputs),
        },
    }
```

- [ ] **Step 6: Add `nbformat` import at the top of `main.py`**

Add near the existing imports (this is needed for `nbformat.v4.new_code_cell` inside the function — the existing code imported it locally; now we need it at module level or keep the local import). The existing code already does `import nbformat` inside the function. Keep it local — no change needed.

- [ ] **Step 7: Run all existing tests**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/pipeline_viz/main.py
git commit -m "feat: wire runtime I/O tracing into notebook execution and graph cache"
```

---

### Task 7: End-to-end verification

**Files:** None — manual verification.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -x -v`
Expected: All tests pass (61 existing + 10 new).

- [ ] **Step 2: Manual smoke test with `example_ecommerce_analytics`**

Start the server with execution enabled:

```bash
PIPELINE_VIZ_ALLOW_RUN=1 .venv/bin/python -m pipeline_viz.cli --port 8766
```

Open browser to: `http://127.0.0.1:8766/?project_root=/Users/admin/Documents/pipeline-viz/example_ecommerce_analytics`

1. Click "加载图" — verify graph loads with static analysis edges.
2. Click a notebook node → "运行所选 Notebook" → verify it completes.
3. Click "加载图" again — verify new edges appear (from runtime trace).
4. Check the API response from run-notebook — verify `traced_io` field is present.

- [ ] **Step 3: Commit any fixes from smoke testing**
