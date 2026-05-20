# Runtime I/O Tracing Design

## Problem

The static parser (`notebook_parser.py`) uses regex + AST folding to discover file I/O in notebooks and `.py` modules. This misses:
- Dynamic paths (glob, loops, conditionals)
- Paths from function arguments, config files, environment variables
- Complex path construction that exceeds the AST folder's capabilities

Users must manually declare these edges in `pipeline-manifest.yaml`.

## Solution

Inject lightweight monkey-patches into the Jupyter kernel at execution time. When a notebook runs via `/api/run-notebook`, the patches record every file read/write path. After execution, the server collects the trace, filters it to project-scoped data files, and merges the results into the graph alongside static analysis.

## Architecture

```
  notebook execution (nbclient kernel process)
      |
      |  kernel startup code injects monkey-patches
      |  patches record paths to __pipeline_viz_io_trace__ dict
      |  notebook cells execute normally
      |
      v
  after execution:
      |  server appends a hidden cell to read __pipeline_viz_io_trace__
      |  executes it, parses output, removes the cell
      |
      v
  server-side:
      |  filter: keep only paths under project_root with data extensions
      |  convert to relative POSIX paths
      |  persist to .pipeline-viz/runtime_io.json
      |  invalidate graph cache
      |  next /api/graph call merges: static + manifest + runtime
```

## Components

### 1. `src/pipeline_viz/io_tracer.py` (new file)

Four public functions:

#### `generate_trace_startup_code(project_root: str) -> str`

Returns Python source code to be executed when the kernel starts. This code:

- Creates a global `__pipeline_viz_io_trace__ = {"reads": [], "writes": []}`.
- Monkey-patches the following targets:

| Target | Classification | How |
|--------|---------------|-----|
| `builtins.open` | read if mode in `(r, rb, "")`, write if mode has `w/a/x` | Wrap to record first arg, delegate to original |
| `pandas.read_csv/read_json/read_parquet/read_excel/read_pickle/read_feather/read_hdf` | read | Wrap each; record first positional arg if `str` or `Path` |
| `DataFrame.to_csv/to_json/to_parquet/to_excel/to_pickle` | write | Wrap each; record first positional arg if `str` or `Path` |
| `json.load` | read | Track via the file object's `.name` attribute (set by patched `open`) |
| `json.dump` | write | Track via the file object's `.name` attribute |
| `numpy.load` | read | Record first arg if `str` or `Path` |
| `numpy.save/numpy.savez` | write | Record first arg if `str` or `Path` |
| `pickle.load` | read | Track via file object `.name` |
| `pickle.dump` | write | Track via file object `.name` |

Patches must:
- Be safe: if anything fails, silently fall back to the original function (no breaking user code).
- Use `functools.wraps` to preserve signatures.
- Resolve paths to absolute via `os.path.abspath` before recording.
- Handle both `str` and `pathlib.Path` arguments.

#### `generate_trace_collect_code() -> str`

Returns Python source for a temporary cell appended after execution:

```python
import json as __pviz_json__
print("__PIPELINE_VIZ_TRACE__:" + __pviz_json__.dumps(__pipeline_viz_io_trace__))
```

Uses a unique prefix so the output can be reliably extracted.

#### `parse_trace_output(cell_outputs: list[dict]) -> tuple[list[str], list[str]]`

Scans cell output dicts (nbformat format) for the `__PIPELINE_VIZ_TRACE__:` prefix line. Parses the JSON and returns `(read_paths, write_paths)` as absolute path strings.

Returns `([], [])` if no trace output is found (graceful degradation).

#### `filter_project_paths(project_root: Path, paths: list[str]) -> set[str]`

Filters a list of absolute paths:
1. Must be under `project_root` (using `Path.relative_to`)
2. Must have a data file extension (use `DATA_EXT` from `paths.py` — move the existing `_DATA_EXT` regex from `notebook_parser.py` to `paths.py` as a shared constant to avoid circular imports)
3. Returns relative POSIX paths

### 2. Runtime I/O Cache: `.pipeline-viz/runtime_io.json`

Schema:
```json
{
  "notebooks/01_ingest.ipynb": {
    "inputs": ["data/raw/orders.csv", "data/raw/products.csv"],
    "outputs": ["data/clean/orders_validated.csv"]
  }
}
```

Functions in `io_tracer.py`:

#### `load_runtime_io(project_root: Path) -> dict[str, dict[str, list[str]]]`

Reads and returns the cache. Returns `{}` if file missing or corrupt.

#### `save_runtime_io(project_root: Path, data: dict) -> None`

Writes the cache atomically.

#### `update_runtime_io(project_root: Path, notebook_rel: str, inputs: set[str], outputs: set[str]) -> None`

Updates a single notebook's entry and persists.

### 3. Changes to `main.py` — `_run_notebook_execute_inplace`

Before execution:
- Generate startup code via `generate_trace_startup_code(project_root)`.
- Pass it to `ExecutePreprocessor` via `ep.km_kwargs["startup_code"]` or by prepending a startup cell.

After execution:
- Append a temporary cell with `generate_trace_collect_code()`.
- Execute that single cell via `ep.preprocess` (or `run_cell`).
- Call `parse_trace_output` on its outputs.
- Call `filter_project_paths` to get relative paths.
- Call `update_runtime_io` to persist.
- Remove the temporary cell from the notebook before writing back to disk.

Return value change:
- `_run_notebook_execute_inplace` now returns `tuple[set[str], set[str]]` (traced inputs, outputs).

API change:
- `/api/run-notebook` response adds `traced_io: {inputs: [...], outputs: [...]}`.
- After execution, call `_graph_cache.invalidate(root)` so the next `/api/graph` rebuilds.

### 4. Changes to `graph_builder.py` — `build_graph`

New optional parameter:
```python
def build_graph(
    project_root: Path,
    manifest_path: Path | None,
    runtime_io: dict[str, dict[str, list[str]]] | None = None,
) -> GraphPayload:
```

For each notebook, I/O is now:
```python
static_in, static_out = parse_notebook_io(root, nb_rel)      # static analysis
manifest_in, manifest_out = from manifest_codes                # manifest
runtime_entry = (runtime_io or {}).get(nb_rel, {})             # runtime trace
runtime_in = {normalize_rel(root, x) for x in runtime_entry.get("inputs", [])}
runtime_out = {normalize_rel(root, x) for x in runtime_entry.get("outputs", [])}

ins = static_in | manifest_in | runtime_in
outs = static_out | manifest_out | runtime_out
```

The same merge happens for `.py` files if runtime_io has entries for them (unlikely but supported).

### 5. Changes to `_GraphCache.get_or_build`

Load runtime_io and pass it to `build_graph`:
```python
def get_or_build(self, root, manifest_path):
    ...
    runtime_io = load_runtime_io(root)
    payload = build_graph(root, manifest_path, runtime_io=runtime_io)
    ...
```

Include `runtime_io.json` mtime in the cache fingerprint so the cache invalidates when runtime data changes.

## Data Flow Example

Before tracing (static analysis only):
```
01_ingest.ipynb --> orders_validated.csv
                    (misses: glob("data/raw/*.csv") not detected)
```

After running notebook with tracing:
```
01_ingest.ipynb <-- data/raw/orders.csv      (traced read)
                <-- data/raw/products.csv     (traced read)
                <-- data/raw/users.csv        (traced read)
                --> data/clean/orders_validated.csv  (traced write)
```

## Error Handling

- If startup code injection fails: log warning, execute notebook normally, return empty trace.
- If trace collection cell fails: log warning, return empty trace, still write notebook outputs.
- If monkey-patch causes an exception in user code: the patch catches it silently and delegates to the original function.
- If `runtime_io.json` is corrupt: treat as empty, rebuild on next run.

## Testing Strategy

- Unit tests for `io_tracer.py`: generate code, parse output, filter paths.
- Integration test: create a minimal notebook with `glob.glob` + `pd.read_csv`, run it, verify traced I/O appears in graph.
- Regression: existing tests must pass unchanged (runtime_io defaults to None/empty).

## Files Changed

| File | Change |
|------|--------|
| `src/pipeline_viz/io_tracer.py` | **New** — trace code generation, parsing, cache |
| `src/pipeline_viz/main.py` | Modify `_run_notebook_execute_inplace`, `api_run_notebook`, cache fingerprint |
| `src/pipeline_viz/graph_builder.py` | Add `runtime_io` parameter to `build_graph` |
| `tests/test_io_tracer.py` | **New** — unit tests |
| `tests/test_runtime_trace_integration.py` | **New** — end-to-end test |
