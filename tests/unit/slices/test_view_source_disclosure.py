"""The ``full`` view never hands back an empty body without saying why.

Three ways a source read can come up short — the file is gone, the file is
shorter than the index thinks, the member is longer than the per-member cap —
and all three have to be distinguishable from "this function is empty".
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from repowise.core.ingestion.parser import ASTParser
from repowise.core.slices.models import SliceMember
from repowise.core.slices.views import ViewContext, render_member


def _symbol(path: str, start: int, end: int) -> SliceMember:
    return SliceMember(
        node_id=f"{path}::thing",
        node_type="symbol",
        layer="symbol",
        file_path=path,
        distance=1,
        name="thing",
        kind="function",
        start_line=start,
        end_line=end,
        language="python",
    )


def _file_member(path: str) -> SliceMember:
    return SliceMember(
        node_id=path,
        node_type="file",
        layer="file",
        file_path=path,
        distance=0,
        language="python",
    )


def _ctx(tmp_path, *, max_source_lines: int = 200) -> ViewContext:
    return ViewContext(repo_root=tmp_path, max_source_lines=max_source_lines)


def test_a_symbol_body_is_sliced_to_its_own_line_range(tmp_path) -> None:
    (tmp_path / "mod.py").write_text("# heading\n\ndef thing():\n    return 1\n")
    payload = render_member(_symbol("mod.py", 3, 5), "full", _ctx(tmp_path))

    assert payload["source"] == "def thing():\n    return 1"
    assert payload["source_first_line"] == 3
    assert payload["source_lines"] == 2
    assert "source_unavailable" not in payload


def test_a_missing_file_is_named_rather_than_returned_empty(tmp_path) -> None:
    payload = render_member(_symbol("gone.py", 1, 5), "full", _ctx(tmp_path))

    assert payload["source"] is None
    assert "gone.py" in payload["source_unavailable"]
    assert "gone or unreadable" in payload["source_unavailable"]


def test_a_missing_symbol_cannot_use_its_former_line_range(tmp_path) -> None:
    """The index and the working tree disagreeing is not an empty function.

    This is the shape a stale index produces after someone deletes half a file:
    the read succeeds, the slice is fine, and the window is empty. Returning
    ``source: ""`` there would read as 'this function has no body'.
    """
    (tmp_path / "short.py").write_text("only\ntwo\n")
    payload = render_member(_symbol("short.py", 40, 60), "full", _ctx(tmp_path))

    assert payload["source"] is None
    assert "successful live parse" in payload["source_unavailable"]
    assert "short.py::thing" in payload["source_unavailable"]


def test_a_file_member_over_the_cap_says_how_much_was_cut(tmp_path) -> None:
    (tmp_path / "big.py").write_text("\n".join(f"line{i}" for i in range(1, 101)))
    payload = render_member(_file_member("big.py"), "full", _ctx(tmp_path, max_source_lines=10))

    assert payload["source_lines"] == 10
    assert payload["source_truncated"] is True
    assert payload["source_lines_omitted"] == 90
    assert "max_source_lines=10" in payload["source_truncation_note"]


def test_card_and_skeleton_never_read_source_at_all(tmp_path) -> None:
    (tmp_path / "mod.py").write_text("line1\nline2\n")
    member = _symbol("mod.py", 1, 2)
    for view in ("card", "skeleton"):
        payload = render_member(member, view, _ctx(tmp_path))
        assert "source" not in payload
        assert "source_unavailable" not in payload


def test_a_member_path_cannot_read_outside_the_repo(tmp_path) -> None:
    """An index row is not a trust boundary.

    A checked-in symlink to a file outside the repo is listed by the walker
    and indexed under its repo-relative name; ``..`` segments and absolute
    paths can only come from a hostile index. All three must read as gone,
    never as the target's bytes.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "leak.py").symlink_to(outside / "secret.txt")

    from repowise.core.slices.views import ViewContext

    for rel in ("leak.py", "../outside/secret.txt", str(outside / "secret.txt")):
        payload = render_member(_symbol(rel, 1, 1), "full", ViewContext(repo_root=repo))
        assert payload["source"] is None, rel
        assert "source_unavailable" in payload, rel


def test_a_symlinked_repo_root_still_serves_its_own_files(tmp_path) -> None:
    """The containment guard must not refuse the repo's own files.

    The classic regression — comparing the resolved target against the
    UNRESOLVED root — refuses every legitimate read under a symlinked
    checkout, and it fails closed and silent (everything becomes
    ``source_unavailable``). Resolving the root first is what this pins.
    """
    real = tmp_path / "real"
    real.mkdir()
    (real / "mod.py").write_text("def thing():\n    return 1\n")
    link = tmp_path / "rootlink"
    link.symlink_to(real, target_is_directory=True)

    from repowise.core.slices.views import ViewContext

    payload = render_member(_symbol("mod.py", 1, 2), "full", ViewContext(repo_root=link))
    assert payload["source"] == "def thing():\n    return 1"


def test_live_parse_is_once_per_file_and_only_for_full_symbols(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text("def first():\n    return 1\ndef second():\n    return 2\n")
    parse = ASTParser.parse_file
    calls = []

    def counted(self, info, source):
        calls.append(info.path)
        return parse(self, info, source)

    monkeypatch.setattr(ASTParser, "parse_file", counted)
    ctx = _ctx(tmp_path)
    first = replace(_symbol("mod.py", 1, 2), node_id="mod.py::first", name="first")
    second = replace(_symbol("mod.py", 3, 4), node_id="mod.py::second", name="second")
    for view in ("card", "skeleton"):
        render_member(first, view, ctx)
    assert calls == []
    assert ctx._source_cache == {}
    render_member(_file_member("mod.py"), "full", ctx)
    assert calls == []
    assert render_member(first, "full", ctx)["source"] == "def first():\n    return 1"
    assert render_member(second, "full", ctx)["source"] == "def second():\n    return 2"
    assert calls == ["mod.py"]


@pytest.mark.parametrize("failure", ["unsupported", "syntax", "exception"])
def test_unsuccessful_parse_never_falls_back_to_indexed_bounds(tmp_path, monkeypatch, failure):
    source = "def thing():\n    return 1\n"
    member = _symbol("mod.py", 1, 2)
    if failure == "syntax":
        source = "def thing():\n    return ???\n"
    elif failure == "unsupported":
        member = replace(member, language="unknown")
    else:

        def broken(*args):
            raise RuntimeError("parser failed")

        monkeypatch.setattr(ASTParser, "parse_file", broken)
    (tmp_path / "mod.py").write_text(source)
    payload = render_member(member, "full", _ctx(tmp_path))
    assert payload["source"] is None
    assert "successful live parse" in payload["source_unavailable"]
