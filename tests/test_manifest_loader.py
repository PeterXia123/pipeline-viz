"""Edge-case tests for pipeline_viz.manifest_loader module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline_viz.manifest_loader import load_manifest, manifest_by_notebook
from pipeline_viz.models import CodeEntry, Manifest


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_valid_yaml_with_codes(self, tmp_path):
        f = tmp_path / "manifest.yaml"
        f.write_text(yaml.dump({
            "project_root": ".",
            "codes": [
                {"path": "notebooks/etl.ipynb", "inputs": ["raw.csv"], "outputs": ["clean.csv"]},
                {"path": "src/transform.py", "inputs": ["clean.csv"], "outputs": ["final.csv"]},
            ],
        }), encoding="utf-8")
        m = load_manifest(f)
        assert isinstance(m, Manifest)
        assert len(m.codes) == 2
        assert m.codes[0].path == "notebooks/etl.ipynb"
        assert m.codes[1].outputs == ["final.csv"]

    def test_empty_yaml_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("", encoding="utf-8")
        m = load_manifest(f)
        assert isinstance(m, Manifest)
        assert m.codes == []
        assert m.project_root == "."

    def test_yaml_with_no_codes_key(self, tmp_path):
        f = tmp_path / "no_codes.yaml"
        f.write_text("project_root: /tmp/myproj\n", encoding="utf-8")
        m = load_manifest(f)
        assert m.project_root == "/tmp/myproj"
        assert m.codes == []

    def test_yaml_with_empty_codes_list(self, tmp_path):
        f = tmp_path / "empty_codes.yaml"
        f.write_text(yaml.dump({"codes": []}), encoding="utf-8")
        m = load_manifest(f)
        assert m.codes == []

    def test_invalid_yaml_syntax_raises(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("codes:\n  - path: 'a.ipynb\n    missing_quote", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            load_manifest(f)

    def test_yaml_only_null(self, tmp_path):
        """A file containing only 'null' should produce a default Manifest."""
        f = tmp_path / "null.yaml"
        f.write_text("null\n", encoding="utf-8")
        m = load_manifest(f)
        assert isinstance(m, Manifest)
        assert m.codes == []


# ---------------------------------------------------------------------------
# manifest_by_notebook
# ---------------------------------------------------------------------------

class TestManifestByNotebook:
    def test_basic_lookup(self):
        m = Manifest(codes=[
            CodeEntry(path="notebooks/etl.ipynb", inputs=["a.csv"]),
            CodeEntry(path="src/transform.py", inputs=["b.csv"]),
        ])
        result = manifest_by_notebook(m)
        assert "notebooks/etl.ipynb" in result
        assert "src/transform.py" in result

    def test_backslash_normalization(self):
        m = Manifest(codes=[
            CodeEntry(path="notebooks\\etl.ipynb"),
        ])
        result = manifest_by_notebook(m)
        assert "notebooks/etl.ipynb" in result

    def test_leading_dot_slash_not_stripped_by_manifest_by_notebook(self):
        """manifest_by_notebook does replace+strip but not remove './'."""
        m = Manifest(codes=[
            CodeEntry(path="./notebooks/etl.ipynb"),
        ])
        result = manifest_by_notebook(m)
        # The function does .replace("\\","/").strip(), so "./" stays
        assert "./notebooks/etl.ipynb" in result

    def test_trailing_whitespace_stripped(self):
        m = Manifest(codes=[
            CodeEntry(path="  notebooks/etl.ipynb  "),
        ])
        result = manifest_by_notebook(m)
        assert "notebooks/etl.ipynb" in result

    def test_duplicate_paths_last_wins(self):
        c1 = CodeEntry(path="nb.ipynb", inputs=["a.csv"])
        c2 = CodeEntry(path="nb.ipynb", inputs=["b.csv"])
        m = Manifest(codes=[c1, c2])
        result = manifest_by_notebook(m)
        assert result["nb.ipynb"].inputs == ["b.csv"]

    def test_empty_manifest(self):
        m = Manifest(codes=[])
        result = manifest_by_notebook(m)
        assert result == {}
