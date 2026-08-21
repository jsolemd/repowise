"""Observer-facing query-time exclusion decisions."""

from __future__ import annotations

from repowise.core.exclusion import exclusion_decision


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
