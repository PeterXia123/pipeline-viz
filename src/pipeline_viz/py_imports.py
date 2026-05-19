from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from pipeline_viz.notebook_parser import split_notebook_code_cells
from pipeline_viz.paths import SCAN_SKIP_DIR_NAMES


def build_module_to_py_map(project_root: Path) -> dict[str, str]:
    """
    dotted.module.name -> posix path relative to project (e.g. src/foo.py).
    Prefer shorter path if duplicate keys (unlikely).
    """
    root = project_root.resolve()
    out: dict[str, str] = {}
    for p in root.rglob("*.py"):
        if any(x in SCAN_SKIP_DIR_NAMES for x in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        if rel.endswith(".py"):
            stem = rel[:-3]
        else:
            stem = rel
        name = stem.replace("/", ".")
        if name not in out or len(rel) < len(out[name]):
            out[name] = rel
    return out


def _resolve_module(module_name: str, module_map: dict[str, str]) -> Optional[str]:
    if module_name in module_map:
        return module_map[module_name]
    parts = module_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in module_map:
            return module_map[prefix]
    return None


_split_notebook_cells = split_notebook_code_cells


_MAGIC_RUN = re.compile(r"^\s*%run\s+['\"]?([^\s'\"]+\.py)['\"]?", re.MULTILINE)


def _rels_from_import_node(
    node: ast.Import | ast.ImportFrom,
    module_map: dict[str, str],
) -> list[str]:
    out: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            hit = _resolve_module(alias.name, module_map)
            if hit:
                out.append(hit)
    elif isinstance(node, ast.ImportFrom):
        if node.level and node.level > 0:
            return []
        if node.module is None:
            return []
        base = node.module
        hit = _resolve_module(base, module_map)
        if hit:
            out.append(hit)
            return out
        for alias in node.names:
            if alias.name == "*":
                continue
            hit2 = _resolve_module(f"{base}.{alias.name}", module_map)
            if hit2:
                out.append(hit2)
    return out


def parse_notebook_py_imports_ordered(
    project_root: Path,
    notebook_rel: str,
    module_map: dict[str, str],
) -> list[str]:
    """
    按 notebook 中代码单元顺序，以及单元内「行号 + 列号」顺序，
    解析对项目内 .py 的首次引用（import / from、%run）。
    """
    nb_path = (project_root / notebook_rel).resolve()
    if not nb_path.is_file():
        return []

    ordered: list[str] = []
    seen: set[str] = set()

    def add(rel: Optional[str]) -> None:
        if not rel or rel in seen:
            return
        seen.add(rel)
        ordered.append(rel)

    for src in _split_notebook_cells(nb_path):
        events: list[tuple[int, int, str]] = []

        for m in _MAGIC_RUN.finditer(src):
            rel = _to_rel_py(project_root, m.group(1).strip())
            if rel:
                line = src[: m.start()].count("\n") + 1
                events.append((line, m.start(), rel))

        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None

        if tree:
            imports: list[ast.Import | ast.ImportFrom] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
            imports.sort(key=lambda n: (n.lineno, n.col_offset or 0))
            for node in imports:
                rels = _rels_from_import_node(node, module_map)
                for i, rel in enumerate(rels):
                    events.append((node.lineno, (node.col_offset or 0) + i, rel))

        events.sort(key=lambda t: (t[0], t[1]))
        for _, _, rel in events:
            add(rel)

    return ordered


def _to_rel_py(project_root: Path, raw: str) -> Optional[str]:
    raw = raw.strip().strip('"').strip("'")
    if not raw.endswith(".py"):
        return None
    root = project_root.resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        return None
    s = rel.as_posix()
    if s.startswith(".."):
        return None
    return s
