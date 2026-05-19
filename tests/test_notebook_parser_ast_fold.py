"""AST 常量折叠：字符串拼接、os.path.join、Path /、f-string。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_viz.notebook_parser import parse_notebook_io


def _write_nb(path: Path, cells: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
        },
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": c, "outputs": []}
            for c in cells
        ],
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_ast_string_concat_assign_and_read(tmp_path: Path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "orders.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import pandas as pd\n'
            'p = "data/" + "raw/" + "orders.csv"\n'
            "pd.read_csv(p)\n",
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/raw/orders.csv" in ins


def test_ast_os_path_join(tmp_path: Path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            "import os, pandas as pd\n"
            'p = os.path.join("a", "b", "c.csv")\n'
            "pd.read_csv(p)\n",
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "a/b/c.csv" in ins


def test_ast_pathlib_div(tmp_path: Path):
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "x" / "y" / "z.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            "from pathlib import Path\nimport pandas as pd\n"
            'p = Path("x") / "y" / "z.csv"\n'
            "pd.read_csv(p)\n",
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "x/y/z.csv" in ins


def test_ast_fstring_fold(tmp_path: Path):
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "orders.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'sub = "orders"\n'
            'import pandas as pd\n'
            'p = f"data/{sub}.csv"\n'
            "pd.read_csv(p)\n",
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/orders.csv" in ins


def test_if_stmt_falls_back_to_regex(tmp_path: Path):
    """含 if 的语义行无法用线性 AST / AST 整块解析失败时，退回正则仍能识别字面量 read。"""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "a.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            "import pandas as pd\n"
            'if True: pd.read_csv("data/raw/a.csv")\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/raw/a.csv" in ins
