"""Explicit corpus exclusions also apply to the fallback text lane."""
import subprocess

from repowise.core.source_search.indexer import _build_window_chunks


def test_explicit_root_ignore_excludes_labels_but_keeps_operational_windows(tmp_path):
    for path in ["eval/labels.json", "eval/keep.json", "repowise-mcp.service.in"]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"owner": "service"}\n')
    (tmp_path / ".repowiseIgnore").write_text("eval/**\n!eval/keep.json\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    files = {chunk.file_path for chunk in _build_window_chunks(tmp_path, [])}
    assert files == {"eval/keep.json", "repowise-mcp.service.in"}
