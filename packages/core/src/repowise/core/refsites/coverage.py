"""Per-language coverage declaration for reference-site extraction.

The rule this module exists to enforce: **an uncovered language says so.**
A query that returns zero sites because nobody ever taught the extractor to
read YAML must not be indistinguishable from a query that returns zero sites
because the symbol genuinely has no references. Every query response therefore
carries a coverage block, and every language present in the repository appears
in it with a status and a reason.

Statuses are deliberately four, not two:

``full``
    Gated and measured against a fixture in this build. Definitions, calls,
    imports, bindings and the textual sweep all run.
``partial``
    Gated and measured, but the parse layer supplies less than the full set.
    ``limitations`` names exactly what is missing.
``ungated``
    The parse layer handles the language and extraction will run over it, but
    this build declared no fixture for it. Sites are produced; their
    completeness is unproven.
``none``
    The parse layer produces nothing for this language. Zero sites is a
    structural fact, not a measurement.

The ``none`` set is derived from :data:`REGISTRY` rather than typed out, so a
language that gains a grammar upstream stops being reported as unsupported
without anyone editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repowise.core.ingestion.languages.registry import REGISTRY
from repowise.core.ingestion.special_handlers import SPECIAL_HANDLER_LANGUAGES


class CoverageStatus(StrEnum):
    """How much of a language's reference surface this build claims."""

    FULL = "full"
    PARTIAL = "partial"
    UNGATED = "ungated"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    """The claim this build makes about one language."""

    language: str
    status: CoverageStatus
    kinds: tuple[str, ...]
    limitations: tuple[str, ...]
    reason: str

    @property
    def is_covered(self) -> bool:
        """True when extraction produces sites for this language at all."""
        return self.status is not CoverageStatus.NONE


#: Languages the textual sweep is allowed to run over. The sweep is bounded by
#: the AST-derived name universe, so it needs a language whose comment and
#: string syntax we model; running it on a language we cannot lex would mint
#: identifier sites inside comments and call them references.
SWEEPABLE_LANGUAGES: frozenset[str] = frozenset({"python", "typescript", "javascript", "shell"})

_GATED: dict[str, LanguageCoverage] = {
    "python": LanguageCoverage(
        language="python",
        status=CoverageStatus.FULL,
        kinds=(
            "definition",
            "call",
            "receiver",
            "import",
            "import_binding",
            "identifier",
            "string_ref",
        ),
        limitations=(),
        reason="tree-sitter grammar with call, import and symbol captures",
    ),
    "typescript": LanguageCoverage(
        language="typescript",
        status=CoverageStatus.FULL,
        kinds=(
            "definition",
            "call",
            "receiver",
            "import",
            "import_binding",
            "reexport",
            "type_ref",
            "heritage",
            "identifier",
            "string_ref",
        ),
        limitations=(
            "`.tsx` is the tsx grammar variant of the `typescript` tag, not a separate language",
            "a symbol declared inside an anonymous callable is deleted by the parse layer, so its "
            "definition site is recovered by the textual sweep at reduced confidence",
        ),
        reason="tree-sitter grammar with call, import, type and heritage captures",
    ),
    "javascript": LanguageCoverage(
        language="javascript",
        status=CoverageStatus.FULL,
        kinds=(
            "definition",
            "call",
            "receiver",
            "import",
            "import_binding",
            "heritage",
            "identifier",
            "string_ref",
        ),
        limitations=(
            "a symbol declared inside an anonymous callable is deleted by the parse layer, so its "
            "definition site is recovered by the textual sweep at reduced confidence",
        ),
        reason="tree-sitter grammar with call and import captures",
    ),
    "shell": LanguageCoverage(
        language="shell",
        status=CoverageStatus.PARTIAL,
        kinds=("definition", "call", "import", "identifier", "string_ref"),
        limitations=(
            "shell.scm captures no `@call.receiver` and no `@call.arguments`, so no receiver sites "
            "and no arity",
            "only `source`/`.` produce import sites",
        ),
        reason="tree-sitter bash grammar with a reduced query set",
    ),
    "sql": LanguageCoverage(
        language="sql",
        status=CoverageStatus.PARTIAL,
        kinds=("definition",),
        limitations=(
            "parsed by a sqlglot special handler, not tree-sitter: there is no call-site "
            "vocabulary, so no call sites exist to record",
            "dbt ref()/source() imports carry no line number and are therefore not sited",
            "the textual sweep does not run, so no identifier or string_ref sites",
        ),
        reason="sqlglot DDL handler yields definitions with line numbers and nothing else",
    ),
}


def _derived(language: str) -> LanguageCoverage:
    """Classify a language this build did not gate."""
    spec = REGISTRY.get(language)
    if spec is None:
        return LanguageCoverage(
            language=language,
            status=CoverageStatus.NONE,
            kinds=(),
            limitations=(),
            reason="not a registered language",
        )
    if language in SPECIAL_HANDLER_LANGUAGES:
        return LanguageCoverage(
            language=language,
            status=CoverageStatus.UNGATED,
            kinds=("definition",),
            limitations=("handled by a non-tree-sitter special handler; not gated by a fixture",),
            reason="special handler produces symbols but no call vocabulary",
        )
    if spec.scm_file is None or spec.is_passthrough:
        return LanguageCoverage(
            language=language,
            status=CoverageStatus.NONE,
            kinds=(),
            limitations=(),
            reason=(
                "no tree-sitter query file: the parse layer returns an empty ParsedFile, so no "
                "reference site of any kind can exist for this language"
            ),
        )
    return LanguageCoverage(
        language=language,
        status=CoverageStatus.UNGATED,
        kinds=("definition", "call", "import", "import_binding"),
        limitations=("extraction runs but this build declares no fixture, so recall is unproven",),
        reason=f"tree-sitter grammar present ({spec.scm_file}) but not gated by a fixture",
    )


def coverage_for(language: str) -> LanguageCoverage:
    """Return the coverage claim for *language*."""
    gated = _GATED.get(language)
    return gated if gated is not None else _derived(language)


def gated_languages() -> tuple[str, ...]:
    """The languages this build explicitly declares and tests."""
    return tuple(sorted(_GATED))


def declared_coverage() -> tuple[LanguageCoverage, ...]:
    """Every explicit claim, for a tool response that lists the whole gate."""
    return tuple(_GATED[name] for name in gated_languages())


def is_sweepable(language: str) -> bool:
    """True when the textual sweep may run over *language*."""
    return language in SWEEPABLE_LANGUAGES


__all__ = [
    "SWEEPABLE_LANGUAGES",
    "CoverageStatus",
    "LanguageCoverage",
    "coverage_for",
    "declared_coverage",
    "gated_languages",
    "is_sweepable",
]
