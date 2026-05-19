from __future__ import annotations

import ast
import posixpath
import re
from pathlib import Path
from typing import Optional

import nbformat

_READ_API_ATTRS = frozenset(
    {
        "read_csv",
        "read_json",
        "read_parquet",
        "read_excel",
        "read_feather",
        "read_hdf",
        "read_pickle",
    }
)
_WRITE_API_ATTRS = frozenset(
    {
        "to_csv",
        "to_json",
        "to_parquet",
        "to_excel",
        "to_hdf",
        "to_pickle",
    }
)

# Heuristic: classify path strings as read vs write from surrounding context.
# 不含 open：open() 的读写由 _open_line_kind 单独判断，避免 open(..., "w") 被误判为读
_READ_CTX = re.compile(
    r"(read_csv|read_json|read_parquet|read_excel|read_feather|read_hdf|read_pickle|"
    r"load|loads|np\.load|json\.load|pickle\.load)\s*\(",
    re.IGNORECASE,
)
_WRITE_CTX = re.compile(
    r"(to_csv|to_json|to_parquet|to_excel|to_hdf|to_pickle|savefig|save|dump|dumps|"
    r"np\.save|json\.dump|pickle\.dump)\s*\(",
    re.IGNORECASE,
)
_OPEN_MODE = re.compile(r"open\s*\([^,]+,\s*['\"]([rwa+x]+)['\"]", re.IGNORECASE)

# First string literal in parentheses (best-effort for common calls).
_STRING_ARG = re.compile(r"['\"]([^'\"]+)['\"]")

# os.chdir("...") 字符串字面量
_CHDIR_OS = re.compile(r"os\.chdir\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE)
# os.chdir(foo) 变量
_CHDIR_VAR = re.compile(r"os\.chdir\s*\(\s*(\w+)\s*\)", re.IGNORECASE)

# read_*(ident,  — 首参为变量名（排除首参已是字符串的情况）
_READ_IDENT_FIRST = re.compile(
    r"(?:read_csv|read_json|read_parquet|read_excel|read_feather|read_hdf|read_pickle)\s*\(\s*(\w+)\s*(?:,|\))",
    re.IGNORECASE,
)
_WRITE_IDENT_FIRST = re.compile(
    r"(?:to_csv|to_json|to_parquet|to_excel|to_hdf|to_pickle)\s*\(\s*(\w+)\s*(?:,|\))",
    re.IGNORECASE,
)
_OPEN_IDENT_FIRST = re.compile(r"\bopen\s*\(\s*(\w+)\s*(?:,|\))", re.IGNORECASE)

# Paths that look like data files (reduce noise from module paths).
from pipeline_viz.paths import DATA_EXT as _DATA_EXT


def split_notebook_code_cells(nb_path: Path) -> list[str]:
    nb = nbformat.read(nb_path, as_version=4)
    parts: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.get("source", "") or ""
        if isinstance(src, list):
            src = "".join(src)
        parts.append(src)
    return parts


def _open_line_kind(line: str) -> Optional[str]:
    """open(...)：有模式则按模式；否则视为读（含 open('f') 单参数）。"""
    if not re.search(r"\bopen\s*\(", line):
        return None
    om = _OPEN_MODE.search(line)
    if om:
        mode = om.group(1).lower()
        if "w" in mode or "a" in mode or "x" in mode:
            return "out"
        return "in"
    return "in"


def _line_kind(line: str) -> Optional[str]:
    if _WRITE_CTX.search(line):
        return "out"
    ok = _open_line_kind(line)
    if ok is not None:
        return ok
    if _READ_CTX.search(line):
        return "in"
    return None


def _strings_in_line(line: str) -> list[str]:
    return [m.group(1) for m in _STRING_ARG.finditer(line)]


def _line_semantic_units(line: str) -> list[str]:
    """按分号拆语句，便于处理 `os.chdir('a'); pd.read_csv('b')` 等同物理行多语句。"""
    line = line.strip()
    if not line:
        return []
    if "#" in line:
        code = line.split("#", 1)[0].strip()
        if not code:
            return []
        line = code
    parts = [p.strip() for p in line.split(";") if p.strip()]
    return parts if parts else [line]


def _lines_to_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    for raw_line in lines:
        for unit in _line_semantic_units(raw_line):
            u = unit.strip()
            if u:
                units.append(u)
    return units


def _notebook_code_units(nb_path: Path) -> list[str]:
    parts: list[str] = []
    for src in split_notebook_code_cells(nb_path):
        parts.extend(_lines_to_units(src.splitlines()))
    return parts


def _strip_inline_comment(unit: str) -> str:
    if "#" in unit:
        return unit.split("#", 1)[0].strip()
    return unit.strip()


def _assignment_string_rhs(unit: str) -> Optional[tuple[str, str]]:
    u = _strip_inline_comment(unit)
    m = re.match(r"^(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s*$", u)
    if m:
        return m.group(1), m.group(2)
    return None


def _assignment_var_rhs(unit: str) -> Optional[tuple[str, str]]:
    u = _strip_inline_comment(unit)
    m = re.match(r"^(\w+)\s*=\s*(\w+)\s*$", u)
    if m:
        return m.group(1), m.group(2)
    return None


def _resolve_name_from_prior_units(
    units: list[str],
    until_idx: int,
    name: str,
    visiting: set[str],
) -> Optional[str]:
    """
    从 until_idx 之前**往上**找最近一次对 name 的赋值：name = "..." 或 name = rhs（再递归 rhs）。
    """
    if name in visiting:
        return None
    visiting.add(name)
    try:
        for j in range(until_idx - 1, -1, -1):
            u = _strip_inline_comment(units[j])
            m = re.match(rf"^{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]\s*$", u)
            if m:
                return m.group(1)
            m2 = re.match(rf"^{re.escape(name)}\s*=\s*(\w+)\s*$", u)
            if m2:
                rhs = m2.group(1)
                v = _resolve_name_from_prior_units(units, j, rhs, visiting)
                if v is not None:
                    return v
        return None
    finally:
        visiting.discard(name)


def _resolve_to_literal_str(
    units: list[str],
    unit_idx: int,
    scope: dict[str, str],
    name: str,
) -> Optional[str]:
    if name in scope:
        return scope[name]
    return _resolve_name_from_prior_units(units, unit_idx, name, set())


def _is_os_path_join_call(call: ast.Call) -> bool:
    return bool(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "join"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "path"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "os"
    )


def _is_os_chdir_call(call: ast.Call) -> bool:
    return bool(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "chdir"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
    )


def _fold_str_expr(
    node: ast.expr | None,
    scope: dict[str, str],
    units: list[str],
    unit_idx: int,
) -> Optional[str]:
    """仅折叠 compile-time 可确定字符串：常量、名、`+`、`/`、`f""`（纯格式化）、`os.path.join`、`Path(...)`。"""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Str):  # Python < 3.8 风格（兼容）
        return node.s
    if isinstance(node, ast.Name):
        return _resolve_to_literal_str(units, unit_idx, scope, node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = _fold_str_expr(node.left, scope, units, unit_idx)
        b = _fold_str_expr(node.right, scope, units, unit_idx)
        if a is not None and b is not None:
            return a + b
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        a = _fold_str_expr(node.left, scope, units, unit_idx)
        b = _fold_str_expr(node.right, scope, units, unit_idx)
        if a is not None and b is not None:
            la = a.rstrip("/").replace("\\", "/")
            rb = b.lstrip("/").replace("\\", "/")
            if la == "":
                return rb
            return f"{la}/{rb}"
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for p in node.values:
            if isinstance(p, ast.Constant) and isinstance(p.value, str):
                parts.append(p.value)
            elif isinstance(p, ast.Str):
                parts.append(p.s)
            elif isinstance(p, ast.FormattedValue):
                if p.conversion != -1 or p.format_spec is not None:
                    return None
                fe = _fold_str_expr(p.value, scope, units, unit_idx)
                if fe is None:
                    return None
                parts.append(fe)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Call):
        if _is_os_path_join_call(node):
            acc: list[str] = []
            for arg in node.args:
                x = _fold_str_expr(arg, scope, units, unit_idx)
                if x is None:
                    return None
                acc.append(x.replace("\\", "/"))
            return posixpath.join(*acc) if acc else None
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
            return _fold_str_expr(node.args[0], scope, units, unit_idx)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "Path"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pathlib"
            and node.args
        ):
            return _fold_str_expr(node.args[0], scope, units, unit_idx)
    return None


def _open_mode_for_ast_call(call: ast.Call, scope: dict[str, str], units: list[str], idx: int) -> str:
    mode = "r"
    if len(call.args) >= 2:
        m = _fold_str_expr(call.args[1], scope, units, idx)
        if isinstance(m, str):
            mode = m
    for kw in call.keywords or []:
        if kw.arg == "mode":
            m = _fold_str_expr(kw.value, scope, units, idx)
            if isinstance(m, str):
                mode = m
    return mode


def _ast_stmt_linear_only(body: list[ast.stmt]) -> bool:
    """仅含可按顺序静态处理的语句；遇控制流或非 Call 的 Expr 则退回正则。"""
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            continue
        return False
    return True


def _ast_io_kind_for_call(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Name) and call.func.id == "open":
        return "open"
    if isinstance(call.func, ast.Name):
        if call.func.id in _READ_API_ATTRS:
            return "in"
        if call.func.id in _WRITE_API_ATTRS:
            return "out"
    if isinstance(call.func, ast.Attribute):
        if call.func.attr in _READ_API_ATTRS:
            return "in"
        if call.func.attr in _WRITE_API_ATTRS:
            return "out"
    return None


def _ast_apply_call_side_effects(
    call: ast.Call,
    *,
    project_root: Path,
    units: list[str],
    unit_idx: int,
    scope: dict[str, str],
    cwd: Path,
    inputs: set[str],
    outputs: set[str],
) -> Path:
    """处理 os.chdir 与 I/O 类 Call；返回更新后的 cwd。"""
    root = project_root.resolve()
    if _is_os_chdir_call(call) and call.args:
        lit = _fold_str_expr(call.args[0], scope, units, unit_idx)
        if lit is not None:
            new_cwd = _resolve_chdir_cwd(root, cwd, lit)
            if new_cwd is not None:
                cwd = new_cwd
        return cwd

    iok = _ast_io_kind_for_call(call)
    if iok == "open" and call.args:
        mode = _open_mode_for_ast_call(call, scope, units, unit_idx)
        kind = "out" if any(x in mode.lower() for x in ("w", "a", "x")) else "in"
        p = _fold_str_expr(call.args[0], scope, units, unit_idx)
        if p is not None:
            _maybe_add_path_literal(project_root, cwd, p, kind, inputs=inputs, outputs=outputs)
        return cwd

    if iok in ("in", "out") and call.args:
        p = _fold_str_expr(call.args[0], scope, units, unit_idx)
        if p is not None:
            _maybe_add_path_literal(project_root, cwd, p, iok, inputs=inputs, outputs=outputs)
    return cwd


def _try_ast_process_unit(
    unit: str,
    unit_idx: int,
    project_root: Path,
    units: list[str],
    scope: dict[str, str],
    cwd: Path,
    inputs: set[str],
    outputs: set[str],
) -> tuple[bool, Path]:
    """
    尝试用 AST 折叠处理本单元全部语句；成功则返回 (True, new_cwd)，调用方应跳过正则。
    """
    try:
        tree = ast.parse(unit)
    except SyntaxError:
        return False, cwd

    if not _ast_stmt_linear_only(tree.body):
        return False, cwd

    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
            continue

        if isinstance(stmt, ast.Assign):
            val = _fold_str_expr(stmt.value, scope, units, unit_idx)
            if val is not None:
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        scope[t.id] = val
            if isinstance(stmt.value, ast.Call):
                cwd = _ast_apply_call_side_effects(
                    stmt.value,
                    project_root=project_root,
                    units=units,
                    unit_idx=unit_idx,
                    scope=scope,
                    cwd=cwd,
                    inputs=inputs,
                    outputs=outputs,
                )
            continue

        if isinstance(stmt, ast.AnnAssign):
            if stmt.target and stmt.value and isinstance(stmt.target, ast.Name):
                val = _fold_str_expr(stmt.value, scope, units, unit_idx)
                if val is not None:
                    scope[stmt.target.id] = val
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            cwd = _ast_apply_call_side_effects(
                stmt.value,
                project_root=project_root,
                units=units,
                unit_idx=unit_idx,
                scope=scope,
                cwd=cwd,
                inputs=inputs,
                outputs=outputs,
            )
            continue

    return True, cwd


def _is_under_project(project_root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(project_root.resolve())
        return True
    except ValueError:
        return False


def _resolve_chdir_cwd(project_root: Path, cwd: Path, literal: str) -> Optional[Path]:
    """
    将 os.chdir 的字符串参数解析为项目内的绝对路径（模拟运行时 cwd）。
    无法解析或跃出项目根时返回 None（保持原 cwd）。
    """
    root = project_root.resolve()
    s = literal.strip().strip('"').strip("'")
    if not s:
        return None
    p = Path(s)
    try:
        if not p.is_absolute():
            np = (cwd / p).resolve()
        else:
            np = p.resolve()
    except OSError:
        return None
    if not _is_under_project(root, np):
        return None
    return np


def _maybe_add_path_literal(
    project_root: Path,
    cwd: Path,
    s: str,
    kind: str,
    *,
    inputs: set[str],
    outputs: set[str],
) -> None:
    if not _DATA_EXT.search(s) and "/" not in s and "\\" not in s:
        return
    if not _DATA_EXT.search(s) and "." not in Path(s).name:
        return
    rel = _to_rel_from_cwd(project_root, cwd, s)
    if not rel:
        return
    if kind == "in":
        inputs.add(rel)
    else:
        outputs.add(rel)


def _maybe_add_path_from_name(
    project_root: Path,
    cwd: Path,
    units: list[str],
    unit_idx: int,
    scope: dict[str, str],
    name: str,
    kind: str,
    *,
    inputs: set[str],
    outputs: set[str],
) -> None:
    lit = _resolve_to_literal_str(units, unit_idx, scope, name)
    if lit is None:
        return
    _maybe_add_path_literal(project_root, cwd, lit, kind, inputs=inputs, outputs=outputs)


def _scan_units_for_io(
    project_root: Path,
    units: list[str],
    *,
    initial_cwd: Path | None = None,
) -> tuple[set[str], set[str], Path]:
    """
    顺序扫描「语义单元」：简单赋值进入 scope；os.chdir 支持字面量与变量（变量从 scope + 向上回溯）；
    读写 API 支持字符串字面量与首参变量。
    """
    root = project_root.resolve()
    cwd = (initial_cwd if initial_cwd is not None else root).resolve()
    inputs: set[str] = set()
    outputs: set[str] = set()
    scope: dict[str, str] = {}

    for i, unit_raw in enumerate(units):
        unit = unit_raw.strip()

        ast_ok, cwd = _try_ast_process_unit(
            unit, i, project_root, units, scope, cwd, inputs, outputs
        )
        if ast_ok:
            continue

        s_as = _assignment_string_rhs(unit)
        if s_as:
            scope[s_as[0]] = s_as[1]
            continue

        av = _assignment_var_rhs(unit)
        if av:
            lhs, rhs = av
            val = _resolve_to_literal_str(units, i, scope, rhs)
            if val is not None:
                scope[lhs] = val
            continue

        ch_lit_m = _CHDIR_OS.search(unit)
        if ch_lit_m:
            new_cwd = _resolve_chdir_cwd(root, cwd, ch_lit_m.group(1).strip())
            if new_cwd is not None:
                cwd = new_cwd
            continue

        ch_var_m = _CHDIR_VAR.search(unit)
        if ch_var_m:
            vlit = _resolve_to_literal_str(units, i, scope, ch_var_m.group(1))
            if vlit is not None:
                new_cwd = _resolve_chdir_cwd(root, cwd, vlit)
                if new_cwd is not None:
                    cwd = new_cwd
            continue

        kind = _line_kind(unit)
        if not kind:
            continue

        for s in _strings_in_line(unit):
            _maybe_add_path_literal(project_root, cwd, s.strip(), kind, inputs=inputs, outputs=outputs)

        if kind == "in":
            ri = _READ_IDENT_FIRST.search(unit)
            if ri:
                _maybe_add_path_from_name(
                    project_root,
                    cwd,
                    units,
                    i,
                    scope,
                    ri.group(1),
                    "in",
                    inputs=inputs,
                    outputs=outputs,
                )
            oi = _OPEN_IDENT_FIRST.search(unit)
            if oi and _open_line_kind(unit) == "in":
                _maybe_add_path_from_name(
                    project_root,
                    cwd,
                    units,
                    i,
                    scope,
                    oi.group(1),
                    "in",
                    inputs=inputs,
                    outputs=outputs,
                )
        else:
            wi = _WRITE_IDENT_FIRST.search(unit)
            if wi:
                _maybe_add_path_from_name(
                    project_root,
                    cwd,
                    units,
                    i,
                    scope,
                    wi.group(1),
                    "out",
                    inputs=inputs,
                    outputs=outputs,
                )
            oi = _OPEN_IDENT_FIRST.search(unit)
            if oi and _open_line_kind(unit) == "out":
                _maybe_add_path_from_name(
                    project_root,
                    cwd,
                    units,
                    i,
                    scope,
                    oi.group(1),
                    "out",
                    inputs=inputs,
                    outputs=outputs,
                )

    return inputs, outputs, cwd


def parse_notebook_io(project_root: Path, notebook_rel: str) -> tuple[set[str], set[str]]:
    """
    Scan .ipynb code cells for path-like string literals near read/write APIs.
    支持：顺序模拟 os.chdir；简单赋值 + 向上回溯；**AST 常量折叠**（`+`、`/`、`f"{x}"`、
    `os.path.join`、`Path(...)`、以及 `x = pd.read_csv(...)` 赋值右侧的 I/O）；
    无法用线性 AST 处理的语句（如 `if`/`def`）仍回退正则。
    Returns sets of paths relative to project_root (POSIX).
    """
    nb_path = (project_root / notebook_rel).resolve()
    if not nb_path.is_file():
        return set(), set()

    units = _notebook_code_units(nb_path)
    di, do, _ = _scan_units_for_io(project_root, units)
    return di, do


def parse_python_file_io(project_root: Path, py_rel: str) -> tuple[set[str], set[str]]:
    """与 notebook 相同的启发式，扫描单个 .py 源文件。"""
    path = (project_root / py_rel).resolve()
    if not path.is_file():
        return set(), set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set(), set()
    units = _lines_to_units(text.splitlines())
    di, do, _ = _scan_units_for_io(project_root, units)
    return di, do


def _to_rel_from_cwd(project_root: Path, cwd: Path, raw: str) -> str:
    """
    将字符串路径规范为相对 project_root 的 POSIX 路径。
    相对路径按当前模拟 cwd 拼接（已执行过的 os.chdir 字面量）；绝对路径按本机根解析后须落在项目根下。
    """
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return ""
    root = project_root.resolve()
    cwd = cwd.resolve()
    p = Path(raw)
    try:
        if not p.is_absolute():
            p = (cwd / p).resolve()
        else:
            p = p.resolve()
    except OSError:
        return ""
    try:
        rel = p.relative_to(root)
    except ValueError:
        return ""
    s = rel.as_posix()
    if s.startswith(".."):
        return ""
    return s
