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
