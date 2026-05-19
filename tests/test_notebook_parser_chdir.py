"""notebook I/O 解析：按顺序模拟 os.chdir，相对路径相对当前模拟目录。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_viz.notebook_parser import parse_notebook_io
from pipeline_viz.paths import normalize_rel


def _write_nb(path: Path, source_cells: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for src in source_cells:
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": src,
                "outputs": [],
            }
        )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3"}},
        "cells": cells,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def test_chdir_then_basename_read_csv(tmp_path: Path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "orders.csv").write_text("a\n", encoding="utf-8")
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import os\nos.chdir("data/raw")\n',
            'import pandas as pd\npd.read_csv("orders.csv")\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/raw/orders.csv" in ins


def test_same_line_chdir_then_read(tmp_path: Path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "x.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [r'import os, pandas as pd; os.chdir("data/raw"); pd.read_csv("x.csv")'],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/raw/x.csv" in ins


def test_chdir_persists_across_cells(tmp_path: Path):
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "sub" / "f.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import os\nos.chdir("d/sub")\n',
            'import pandas as pd\npd.read_csv("f.csv")\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "d/sub/f.csv" in ins


def test_absolute_path_still_project_relative(tmp_path: Path):
    """绝对路径若落在项目根下，仍解析为相对项目根的 POSIX 路径。"""
    root = tmp_path.resolve()
    raw = root / "data" / "a.csv"
    raw.parent.mkdir(parents=True)
    raw.touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(nb, [f'import pandas as pd\npd.read_csv("{root}/data/a.csv")\n'])
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/a.csv" in ins


def test_normalize_rel_manifest_compatible(tmp_path: Path):
    """确保 compose 侧 normalize_rel 仍可与解析结果并集（无回归）。"""
    assert normalize_rel(tmp_path, "data/raw/a.csv") == "data/raw/a.csv"


def test_variable_chdir_resolved_from_prior_assignment(tmp_path: Path):
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "sub" / "f.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import os\nbase = "d"\n',
            'rel = "sub"\n',
            'os.chdir(base)\nos.chdir(rel)\n',
            'import pandas as pd\npd.read_csv("f.csv")\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "d/sub/f.csv" in ins


def test_read_csv_variable_backward_chain(tmp_path: Path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "orders.csv").write_text("a\n", encoding="utf-8")
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import pandas as pd\n',
            'fn = name\n',
            'name = "orders.csv"\n',
            'pd.read_csv(fn)\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "data/raw/orders.csv" not in ins  # fn 先于 name，静态解析也得不到完整路径
    nb2 = tmp_path / "n2.ipynb"
    _write_nb(
        nb2,
        [
            'import pandas as pd\n',
            'name = "data/raw/orders.csv"\n',
            'fn = name\n',
            'pd.read_csv(fn)\n',
        ],
    )
    ins2, _ = parse_notebook_io(tmp_path, "n2.ipynb")
    assert "data/raw/orders.csv" in ins2

    nb3 = tmp_path / "n3.ipynb"
    _write_nb(
        nb3,
        [
            'import os, pandas as pd\n',
            'os.chdir("data/raw")\n',
            'name = "orders.csv"\n',
            'fn = name\n',
            'pd.read_csv(fn)\n',
        ],
    )
    ins3, _ = parse_notebook_io(tmp_path, "n3.ipynb")
    assert "data/raw/orders.csv" in ins3


def test_chdir_variable_os_chdir(tmp_path: Path):
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "x" / "y" / "z.csv").touch()
    nb = tmp_path / "n.ipynb"
    _write_nb(
        nb,
        [
            'import os\nd = "x/y"\n',
            "os.chdir(d)\n",
            'import pandas as pd\npd.read_csv("z.csv")\n',
        ],
    )
    ins, _ = parse_notebook_io(tmp_path, "n.ipynb")
    assert "x/y/z.csv" in ins
