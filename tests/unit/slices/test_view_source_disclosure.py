"""The ``full`` view never hands back an empty body without saying why.

Three ways a source read can come up short — the file is gone, the file is
shorter than the index thinks, the member is longer than the per-member cap —
and all three have to be distinguishable from "this function is empty".
"""

from __future__ import annotations

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
    (tmp_path / "mod.py").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    payload = render_member(_symbol("mod.py", 3, 5), "full", _ctx(tmp_path))

    assert payload["source"] == "line3\nline4\nline5"
    assert payload["source_first_line"] == 3
    assert payload["source_lines"] == 3
    assert "source_unavailable" not in payload


def test_a_missing_file_is_named_rather_than_returned_empty(tmp_path) -> None:
    payload = render_member(_symbol("gone.py", 1, 5), "full", _ctx(tmp_path))

    assert payload["source"] is None
    assert "gone.py" in payload["source_unavailable"]
    assert "gone or unreadable" in payload["source_unavailable"]


def test_a_line_range_past_the_end_of_the_file_is_named_too(tmp_path) -> None:
    """The index and the working tree disagreeing is not an empty function.

    This is the shape a stale index produces after someone deletes half a file:
    the read succeeds, the slice is fine, and the window is empty. Returning
    ``source: ""`` there would read as 'this function has no body'.
    """
    (tmp_path / "short.py").write_text("only\ntwo\n")
    payload = render_member(_symbol("short.py", 40, 60), "full", _ctx(tmp_path))

    assert payload["source"] is None
    assert "the file changed since it was indexed" in payload["source_unavailable"]
    assert "40-60" in payload["source_unavailable"]


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
