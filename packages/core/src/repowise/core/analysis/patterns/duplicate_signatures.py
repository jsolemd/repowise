"""``duplicate_signatures`` — the same callable declared in more than one file.

Two functions named ``parse_config`` taking two arguments, in two files,
are one of two things: a helper that was copied, or a helper that should
have been imported. Either way the reader wants to see them side by side
before writing a third.

The name comparison is case- and separator-insensitive so ``parseConfig``
and ``parse_config`` collide, which is exactly the cross-language and
cross-author case this is for. Arity comes from the recorded signature;
when a language's signature does not expose a parameter list the arity is
recorded as unknown and only groups with other unknowns, so an unparsed
signature never manufactures a match.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import Symbol, SymbolGraph

DEFINITION = PatternDefinition(
    name="duplicate_signatures",
    question="Which callables are declared more than once across the repository?",
    predicate=(
        "Two or more symbols of kind function/method/constructor whose normalized name "
        "(case-folded, separators removed, so parseConfig == parse_config) and parameter "
        "count are both identical, declared in at least min_files distinct files. Symbols "
        "whose signature exposes no parameter list carry arity 'unknown' and group only "
        "with other unknowns."
    ),
    ranking="Group size (symbols), then distinct file count, then normalized name.",
    not_this=(
        "Not a clone finding: this compares declarations, not bodies. Two functions with "
        "the same shape and unrelated implementations match here and would not match "
        "find_clones. Run find_clones to learn whether the bodies agree."
    ),
    defaults={
        "min_files": 2,
        "include_tests": False,
        "kinds": ["function", "method", "constructor"],
    },
)

_SEPARATORS = re.compile(r"[^0-9a-z]+")


def normalize_name(name: str) -> str:
    """Case-fold and drop separators so naming conventions do not split a group."""
    return _SEPARATORS.sub("", name.casefold())


def parameter_count(signature: str) -> int | None:
    """Count top-level parameters in *signature*, or ``None`` if unparseable.

    Nested parentheses, brackets and braces are skipped so a default value
    like ``timeout=(5, 30)`` or an annotation like ``dict[str, int]`` does
    not read as two extra parameters.
    """
    open_at = signature.find("(")
    if open_at < 0:
        return None
    depth = 0
    params = 0
    seen_content = False
    for ch in signature[open_at:]:
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth -= 1
            if depth == 0:
                return params + 1 if seen_content else 0
            continue
        if depth == 1:
            if ch == ",":
                params += 1
            elif not ch.isspace():
                seen_content = True
    return None


def _group_key(sym: Symbol) -> tuple[str, int | None]:
    return normalize_name(sym.name), parameter_count(sym.signature)


def run(
    graph: SymbolGraph,
    *,
    min_files: int = 2,
    include_tests: bool = False,
    kinds: tuple[str, ...] | list[str] | None = None,
) -> PatternResult:
    allowed = {k.lower() for k in (kinds or DEFINITION.defaults["kinds"])}
    candidates = [
        s for s in graph.candidates(include_tests=include_tests) if s.kind.lower() in allowed
    ]

    groups: dict[tuple[str, int | None], list[Symbol]] = defaultdict(list)
    for sym in candidates:
        if not sym.name:
            continue
        groups[_group_key(sym)].append(sym)

    matches: list[PatternMatch] = []
    for (norm_name, arity), members in groups.items():
        files = {m.file for m in members}
        if len(files) < min_files:
            continue
        members_sorted = sorted(members, key=lambda s: (s.file, s.start_line))
        evidence: dict[str, Any] = {
            "normalized_name": norm_name,
            "arity": arity if arity is not None else "unknown",
            "file_count": len(files),
            "declarations": [{**m.location(), "signature": m.signature} for m in members_sorted],
        }
        matches.append(
            PatternMatch(
                key=f"{norm_name}/{arity if arity is not None else 'unknown'}",
                evidence=evidence,
                rank=(len(members), len(files)),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], -m.rank[1], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={"min_files": min_files, "include_tests": include_tests, "kinds": sorted(allowed)},
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
