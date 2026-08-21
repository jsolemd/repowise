"""Query-time exclusion: compile a repo's exclude rules and filter rows.

Excluded files are skipped at ingest time, but DB rows may predate an
``exclude_patterns`` / gitignore change, so read paths (MCP tools, editor-file
generation) filter again at query time. This module is the single home for
that logic; ``repowise.server.mcp_server._helpers`` delegates here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

# Per-spec memo ceiling. Comfortably above any real repo's distinct path count,
# low enough that 64 cached specs cannot pin unbounded memory.
_MEMO_MAX = 200_000

ExclusionRuleSource = Literal["config", "gitignore", "git_info_exclude"]


@dataclass(frozen=True, slots=True)
class ExclusionDecision:
    """The final rule that decided one query-time exclusion check."""

    excluded: bool
    matched: bool
    source: ExclusionRuleSource | None = None
    pattern: str | None = None


def _gitignore_files(root: Path) -> tuple[Path, ...]:
    """The gitignore-stack files unioned into the spec."""
    return (root / ".gitignore", root / ".git" / "info" / "exclude")


def _rule_files(root: Path) -> tuple[Path, ...]:
    """Every file whose contents define the spec, config file included.

    Read from ``repo_config`` rather than spelled out again: a stamp that
    missed the config file would cache a spec across an ``exclude_patterns``
    edit and keep serving rows the user just excluded.
    """
    from repowise.core.repo_config import CONFIG_FILENAME, get_repowise_dir

    return (get_repowise_dir(root) / CONFIG_FILENAME, *_gitignore_files(root))


def _rules_stamp(root: Path) -> tuple[tuple[int, int], ...]:
    """``(mtime_ns, size)`` per rule source, ``(0, 0)`` when absent.

    Size is in the key because mtime alone is not a reliable invalidator on
    Windows: measured on NTFS, 25 of 199 back-to-back writes shared an identical
    ``st_mtime_ns``. A rules edit that landed inside one timestamp tick would
    otherwise pin the stale spec for the life of the process, with no way to
    force a recompile short of restarting the server. A pattern edit essentially
    always changes the file size, and it rides the same ``stat`` call.
    """
    stamp = []
    for path in _rule_files(root):
        try:
            st = path.stat()
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((0, 0))
    return tuple(stamp)


# Sized for workspace mode, not for one repo. A workspace can register many
# repos, and an agent sweeping them round-robin past the cache size gets a 0%
# hit rate — which is worse than no cache, since a miss now also pays
# ``Path.resolve()`` and three ``stat`` calls. Specs are small; the mtime/size
# stamp in the key means stale entries are unreachable rather than merely old.
@lru_cache(maxsize=64)
def _compile_spec(root_key: str, _stamp: tuple[tuple[int, int], ...]) -> Any:
    """Compile and cache one repo's spec. Keyed by root + rule-file mtimes."""
    import pathspec

    from repowise.core.repo_config import load_repo_config

    root = Path(root_key)
    rule_sets: list[tuple[ExclusionRuleSource, list[str]]] = [
        ("config", list(load_repo_config(root).get("exclude_patterns") or [])),
    ]
    ignore_sources: tuple[tuple[ExclusionRuleSource, Path], ...] = (
        ("gitignore", root / ".gitignore"),
        ("git_info_exclude", root / ".git" / "info" / "exclude"),
    )
    for source, ignore_file in ignore_sources:
        lines: list[str] = []
        try:
            if ignore_file.exists():
                lines = ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            pass
        rule_sets.append((source, lines))

    patterns: list[Any] = []
    sources: list[ExclusionRuleSource] = []
    for source, lines in rule_sets:
        compiled = pathspec.PathSpec.from_lines("gitwildmatch", lines)
        patterns.extend(compiled.patterns)
        sources.extend([source] * len(compiled.patterns))
    if not patterns:
        return None
    # Concatenating the compiled patterns preserves the exact cross-source
    # precedence of the former one-shot ``from_lines`` call, including a later
    # negation re-including a path. The parallel source tuple lets an observer
    # name the rule set containing the final deciding pattern.
    spec = pathspec.PathSpec(patterns)
    spec._repowise_rule_sources = tuple(sources)  # type: ignore[attr-defined]
    # Per-path decision memo, carried on the spec so it lives exactly as long as
    # the compiled spec does. ``match_file`` is a regex sweep over every pattern;
    # a repo-wide read filters the same few thousand paths over and over, both
    # within one call (findings outnumber files ~3:1) and across calls.
    spec._repowise_memo = {}  # type: ignore[attr-defined]
    return spec


def build_exclude_spec(repo_path: Path | str) -> Any:
    """Compile the repo's exclusion rules into a PathSpec, or ``None``.

    Unions ``.repowise/config.yaml`` ``exclude_patterns`` with the repo's
    gitignore stack (``.gitignore`` + ``.git/info/exclude``). Indexes built
    before the traverser honoured ``info/exclude`` still contain rows for
    local-only scratch dirs; filtering them at query time keeps those paths
    out of generated output without forcing a reindex.

    Cached per repo root and invalidated by the rule files' mtimes: every MCP
    tool builds this on each request, and compiling it is pure overhead when
    nothing changed.
    """
    root = Path(repo_path)
    try:
        root_key = str(root.resolve())
    except OSError:
        root_key = str(root)
    return _compile_spec(root_key, _rules_stamp(root))


def is_excluded(path: str | None, spec: Any) -> bool:
    """True if *path* matches *spec* (None spec or path -> not excluded)."""
    if spec is None or not path:
        return False
    memo = getattr(spec, "_repowise_memo", None)
    if memo is None:
        return bool(spec.match_file(path))
    cached = memo.get(path)
    if cached is None:
        if len(memo) >= _MEMO_MAX:
            # Bounded rather than merely finite. The natural ceiling is the
            # repo's distinct paths, but this is also fed symbol ids and
            # decision-anchored paths by other tools, so nothing guarantees
            # that. Dropping the whole memo costs one repopulation pass and
            # keeps a long-lived server from growing without limit.
            memo.clear()
        cached = memo[path] = bool(spec.match_file(path))
    return cached


def exclusion_decision(repo_path: Path | str, path: str) -> ExclusionDecision:
    """Return whether *path* is excluded and which rule source decided it.

    The winning source is read from the same combined, ordered PathSpec used by
    :func:`is_excluded`; it is not reconstructed by matching three independent
    specs, which would get cross-file negations wrong.
    """

    spec = build_exclude_spec(repo_path)
    if spec is None:
        return ExclusionDecision(excluded=False, matched=False)

    checked = spec.check_file(path)
    if checked.index is None or checked.include is None:
        return ExclusionDecision(excluded=False, matched=False)

    sources = getattr(spec, "_repowise_rule_sources", ())
    source = sources[checked.index] if checked.index < len(sources) else None
    raw_pattern = getattr(spec.patterns[checked.index], "pattern", None)
    pattern = str(raw_pattern) if raw_pattern is not None else None
    return ExclusionDecision(
        excluded=bool(checked.include),
        matched=True,
        source=source,
        pattern=pattern,
    )


def decision_is_excluded(decision_row: Any, spec: Any) -> bool:
    """True when a DecisionRecord is anchored entirely in excluded paths.

    Decision mining can predate an exclude_patterns / info-exclude change, so
    records anchored in vendored trees (a checked-in venv's site-packages, a
    local-only scratch dir) survive in the DB and would surface as the repo's
    "top decisions". A record whose affected files are ALL excluded is noise;
    one with no affected files at all is kept (nothing to judge it by).
    Paths are normalized to forward slashes — ``affected_files_json`` stores
    OS-native separators.

    Confirmed records are exempt. The file list is where an extractor *read* a
    decision, not what makes it true, and once a human has marked one ``active``
    that provenance has been superseded by their judgement. Measured on this
    repo: all 18 records this rule dropped were ``active``, a quarter of the
    confirmed corpus, and they included both records answering "why ruff check
    and not ruff format" — a rule so plainly repo-wide that it had been written
    into CLAUDE.md by hand while the store already held it. What the rule is
    aimed at is unreviewed machine output from a tree the user said to ignore,
    and that is exactly what ``proposed`` marks.
    """
    if spec is None:
        return False
    if getattr(decision_row, "status", None) == "active":
        return False
    try:
        affected = json.loads(getattr(decision_row, "affected_files_json", None) or "[]")
    except (ValueError, TypeError):
        return False
    paths = [p.replace("\\", "/") for p in affected if isinstance(p, str) and p]
    if not paths:
        return False
    return all(is_excluded(p, spec) for p in paths)
