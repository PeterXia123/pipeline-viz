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
