"""Observer-facing query-time exclusion decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from repowise.core import exclusion as exclusion_module
from repowise.core.exclusion import build_exclude_spec, exclusion_decision, is_excluded


@pytest.fixture(autouse=True)
def _clear_compiled_spec_cache():
    """Compile fresh per test.

    ``_compile_spec`` is a process-wide ``lru_cache``, and the tests below
    monkeypatch the rule-source accessor it reads. Clearing on both sides
    keeps a spec built from a patched accessor from outliving its test.
    """
    exclusion_module._compile_spec.cache_clear()
    yield
    exclusion_module._compile_spec.cache_clear()


def test_exclusion_decision_names_winning_rule_source(tmp_path):
    repowise_dir = tmp_path / ".repowise"
    repowise_dir.mkdir()
    (repowise_dir / "config.yaml").write_text(
        'exclude_patterns:\n  - "scratch/**"\n',
        encoding="utf-8",
    )
    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    info = tmp_path / ".git" / "info"
    info.mkdir(parents=True)
    (info / "exclude").write_text("private/\n", encoding="utf-8")

    assert exclusion_decision(tmp_path, "scratch/probe.py").source == "config"
    assert exclusion_decision(tmp_path, "dist/bundle.js").source == "gitignore"
    assert exclusion_decision(tmp_path, "private/note.py").source == "git_info_exclude"


def test_exclusion_decision_honours_cross_source_negation(tmp_path):
    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    info = tmp_path / ".git" / "info"
    info.mkdir(parents=True)
    (info / "exclude").write_text("!dist/keep.py\n", encoding="utf-8")

    decision = exclusion_decision(tmp_path, "dist/keep.py")

    assert decision.matched is True
    assert decision.excluded is False
    assert decision.source == "git_info_exclude"
    assert decision.pattern == "!dist/keep.py"


def test_exclusion_decision_reports_no_match_without_guessing(tmp_path):
    decision = exclusion_decision(tmp_path, "src/app.py")

    assert decision.matched is False
    assert decision.excluded is False
    assert decision.source is None
    assert decision.pattern is None


def test_compile_spec_reads_every_declared_ignore_source(tmp_path, monkeypatch):
    custom_ignore = tmp_path / ".repowiseignore"
    custom_ignore.write_text("custom/**\n", encoding="utf-8")
    monkeypatch.setattr(
        exclusion_module,
        "_gitignore_sources",
        lambda _root: (("gitignore", custom_ignore),),
    )

    decision = exclusion_decision(tmp_path, "custom/probe.py")

    assert decision.excluded is True
    assert decision.source == "gitignore"


def test_is_excluded_honours_a_newly_declared_ignore_source(tmp_path, monkeypatch):
    """A source added to the accessor must reach the compiled spec.

    ``_compile_spec`` once spelled the gitignore stack out inline while
    ``_rule_files`` read it from ``_gitignore_sources``. That fork is invisible
    until a third source is added: the mtime stamp would watch the new file —
    so an edit to it invalidated the cache — while the spec it recompiled
    never contained a single one of its patterns. This asserts at the
    ``is_excluded`` surface, which is what every read path actually calls.
    """
    extra_ignore = tmp_path / ".repowiseignore"
    extra_ignore.write_text("vendored/**\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    real_sources = exclusion_module._gitignore_sources
    monkeypatch.setattr(
        exclusion_module,
        "_gitignore_sources",
        lambda root: (*real_sources(root), ("gitignore", extra_ignore)),
    )

    spec = build_exclude_spec(tmp_path)

    assert is_excluded("vendored/pkg/mod.py", spec) is True
    # The pre-existing sources keep working alongside the added one.
    assert is_excluded("dist/bundle.js", spec) is True
    assert is_excluded("src/app.py", spec) is False


def test_gitignore_sources_are_exactly_the_two_git_stack_files(tmp_path):
    """Pin the source set the regression below is a regression against."""
    assert exclusion_module._gitignore_sources(tmp_path) == (
        ("gitignore", tmp_path / ".gitignore"),
        ("git_info_exclude", tmp_path / ".git" / "info" / "exclude"),
    )


def _write_three_source_repo(root: Path) -> None:
    repowise_dir = root / ".repowise"
    repowise_dir.mkdir()
    (repowise_dir / "config.yaml").write_text(
        'exclude_patterns:\n  - "scratch/**"\n',
        encoding="utf-8",
    )
    (root / ".gitignore").write_text("dist/\n*.log\n", encoding="utf-8")
    info = root / ".git" / "info"
    info.mkdir(parents=True)
    (info / "exclude").write_text("private/\n!dist/keep.py\n", encoding="utf-8")


def test_is_excluded_is_unchanged_across_the_current_rule_sources(tmp_path):
    """Behavioural baseline for the compiled spec, at the surface readers use.

    The ``_compile_spec`` rewrite that unified the rule sources was asserted
    only through ``exclusion_decision``. This pins ``is_excluded`` itself over
    all three current sources, their precedence, and the cross-source negation
    that a per-source-spec implementation would get wrong.
    """
    _write_three_source_repo(tmp_path)
    spec = build_exclude_spec(tmp_path)

    assert is_excluded("scratch/notes.py", spec) is True  # config
    assert is_excluded("dist/bundle.js", spec) is True  # .gitignore
    assert is_excluded("build/debug.log", spec) is True  # .gitignore glob
    assert is_excluded("private/note.py", spec) is True  # .git/info/exclude
    assert is_excluded("dist/keep.py", spec) is False  # later negation wins
    assert is_excluded("src/app.py", spec) is False  # matched by nothing
    assert is_excluded(None, spec) is False
    assert is_excluded("src/app.py", None) is False


def test_is_excluded_memoises_without_changing_its_answer(tmp_path):
    """The per-spec memo must be a cache, not a second decision path."""
    _write_three_source_repo(tmp_path)
    spec = build_exclude_spec(tmp_path)

    first = [is_excluded(p, spec) for p in ("dist/keep.py", "dist/bundle.js", "src/app.py")]
    second = [is_excluded(p, spec) for p in ("dist/keep.py", "dist/bundle.js", "src/app.py")]

    assert first == [False, True, False]
    assert second == first


def test_repowise_ignore_uses_the_traversal_rule_and_reports_its_source(tmp_path):
    from repowise.core.ingestion import FileTraverser

    source = tmp_path / "captured.js"
    source.write_text("export function captured() {}\n")
    (tmp_path / ".repowiseIgnore").write_text("captured.js\n")
    assert FileTraverser(tmp_path).file_info_for_path("captured.js") is None

    assert is_excluded("captured.js", build_exclude_spec(tmp_path))
    decision = exclusion_decision(tmp_path, "captured.js")
    assert decision.excluded
    assert decision.source == "repowise_ignore"
    assert decision.pattern == "captured.js"


def test_repowise_ignore_negation_does_not_reopen_git_excluded_files(tmp_path):
    (tmp_path / ".gitignore").write_text("secret.py\n")
    (tmp_path / ".repowiseIgnore").write_text("*.py\n!keep.py\n!secret.py\n")
    spec = build_exclude_spec(tmp_path)

    assert is_excluded("drop.py", spec)
    assert not is_excluded("keep.py", spec)
    assert is_excluded("secret.py", spec)
    assert exclusion_decision(tmp_path, "secret.py").source == "gitignore"


def test_repowise_ignore_pruned_parent_cannot_be_reopened_by_file_negation(tmp_path):
    from repowise.core.ingestion import FileTraverser

    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "keep.py").write_text("def keep(): pass\n")
    (tmp_path / ".repowiseIgnore").write_text("vendor/\n!vendor/keep.py\n")
    assert not list(FileTraverser(tmp_path).traverse())

    assert is_excluded("vendor/keep.py", build_exclude_spec(tmp_path))
    assert exclusion_decision(tmp_path, "vendor/keep.py").pattern == "vendor/"


def test_repowise_ignore_ancestors_use_pathspec_platform_separators(tmp_path, monkeypatch):
    import pathspec.util

    monkeypatch.setattr(pathspec.util, "NORMALIZE_PATH_SEPS", ["\\"])
    (tmp_path / ".repowiseIgnore").write_text("vendor/\n!vendor/keep.py\n")

    assert is_excluded(r"vendor\keep.py", build_exclude_spec(tmp_path))
    assert exclusion_decision(tmp_path, r"vendor\keep.py").pattern == "vendor/"


def test_repowise_ignore_edit_invalidates_the_compiled_spec(tmp_path):
    ignore = tmp_path / ".repowiseIgnore"
    ignore.write_text("old.py\n")
    before = build_exclude_spec(tmp_path)
    assert is_excluded("old.py", before)
    ignore.write_text("new_source.py\n")
    after = build_exclude_spec(tmp_path)

    assert not is_excluded("old.py", after)
    assert is_excluded("new_source.py", after)


def test_repowise_ignore_read_failure_cannot_be_mistaken_for_an_empty_rule(tmp_path, monkeypatch):
    ignore = tmp_path / ".repowiseIgnore"
    ignore.write_text("captured.js\n")
    read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == ignore:
            raise PermissionError("ignore rules unavailable")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(PermissionError, match="ignore rules unavailable"):
        build_exclude_spec(tmp_path)
