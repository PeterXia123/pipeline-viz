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
