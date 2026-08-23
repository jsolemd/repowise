"""Closed vocabularies for reference sites: kind, origin, and the confidence rubric.

Two vocabularies, deliberately separate:

``ReferenceKind``
    *What syntactic position* the occurrence sits in. Answers "what is this
    text doing here" without any claim about what it binds to.

``ReferenceOrigin``
    *How we decided which symbol it refers to*, and therefore how much a
    rename should trust it. Answers "why do we believe this site names that
    symbol".

``ORIGIN_CONFIDENCE`` is the single source of truth for the numeric scale, so
a site's confidence can never drift from the reason it was assigned. The scale
means exactly one thing: **P(a rename of the resolved target must edit this
site)**. It is not a search relevance score and must never be compared against
one — see the ``confidence follows the order it accompanies`` rule that the
search layer already holds itself to.
"""

from __future__ import annotations

from enum import StrEnum

#: Bumped whenever extraction changes shape. Persisted on every row so a
#: store written by an older extractor is detectable rather than silently
#: mixed with a newer one.
EXTRACTOR_VERSION = "refsites/1"


class ReferenceKind(StrEnum):
    """Syntactic position of an occurrence."""

    #: The definition itself. A rename always edits this.
    DEFINITION = "definition"
    #: An invocation: ``foo()``, ``obj.foo()``, shell ``foo arg``.
    CALL = "call"
    #: The receiver of a method call — ``user`` in ``user.save()``.
    RECEIVER = "receiver"
    #: A whole import/require/source statement.
    IMPORT = "import"
    #: One named binding inside an import — ``foo`` in ``from m import foo``.
    IMPORT_BINDING = "import_binding"
    #: A re-export that forwards a name onward.
    REEXPORT = "reexport"
    #: A type annotation position.
    TYPE_REF = "type_ref"
    #: An extends/implements/trait-impl position.
    HERITAGE = "heritage"
    #: Named without being called (the parser's ``references`` channel).
    REFERENCE = "reference"
    #: A bare identifier recovered by the textual sweep, not AST-attested.
    IDENTIFIER = "identifier"
    #: The name inside a string literal — dynamic dispatch, DI keys, configs.
    STRING_REF = "string_ref"


class ReferenceOrigin(StrEnum):
    """Why this site is believed to name the target symbol."""

    DEFINITION = "definition"
    SAME_FILE = "same_file"
    IMPORT_BINDING = "import_binding"
    IMPORT_SCOPED = "import_scoped"
    UNIQUE_GLOBAL = "unique_global"
    TEXTUAL_IDENTIFIER_IN_SCOPE = "textual_identifier_in_scope"
    AMBIGUOUS_GLOBAL = "ambiguous_global"
    UNRESOLVED = "unresolved"
    TEXTUAL_IDENTIFIER = "textual_identifier"
    TEXTUAL_STRING = "textual_string"


#: origin -> P(a rename of the resolved target must edit this site).
#:
#: The ordering carries the argument:
#:   - ``definition`` is certain; the site *is* the symbol.
#:   - ``same_file`` beats ``import_binding`` because file-local resolution
#:     cannot be defeated by a shadowing re-export.
#:   - ``unique_global`` is the "only one thing in the repo has this name"
#:     tier: right until the repo grows a second one.
#:   - ``textual_identifier_in_scope`` outranks ``ambiguous_global`` because
#:     the file demonstrably imports the target, so the bare identifier is
#:     far more likely to be that binding than a same-named stranger.
#:   - ``unresolved`` is a real reference we could not bind. It stays in the
#:     store on purpose: vanishing would read as "no such site exists".
#:   - the two ``textual_*`` floors are recall, not precision. They exist so a
#:     rename preview can show a human what it cannot prove.
ORIGIN_CONFIDENCE: dict[ReferenceOrigin, float] = {
    ReferenceOrigin.DEFINITION: 1.00,
    ReferenceOrigin.SAME_FILE: 0.95,
    ReferenceOrigin.IMPORT_BINDING: 0.90,
    ReferenceOrigin.IMPORT_SCOPED: 0.90,
    ReferenceOrigin.UNIQUE_GLOBAL: 0.70,
    ReferenceOrigin.TEXTUAL_IDENTIFIER_IN_SCOPE: 0.40,
    ReferenceOrigin.AMBIGUOUS_GLOBAL: 0.35,
    ReferenceOrigin.UNRESOLVED: 0.20,
    ReferenceOrigin.TEXTUAL_IDENTIFIER: 0.15,
    ReferenceOrigin.TEXTUAL_STRING: 0.10,
}

#: At or above this, a site is safe to rewrite without review.
CONFIDENCE_CERTAIN = 0.90
#: Below this, a site is a lead for a human, not an edit.
CONFIDENCE_ADVISORY = 0.40

#: Origins produced by the textual sweep rather than the AST. Kept as a set
#: so callers filter on provenance instead of re-deriving it from ``tier``.
TEXTUAL_ORIGINS: frozenset[ReferenceOrigin] = frozenset(
    {
        ReferenceOrigin.TEXTUAL_IDENTIFIER_IN_SCOPE,
        ReferenceOrigin.TEXTUAL_IDENTIFIER,
        ReferenceOrigin.TEXTUAL_STRING,
    }
)

#: Provenance tier. ``ast`` sites are attested by the parse layer; ``textual``
#: sites are recovered by the sweep and bounded by the AST-derived name
#: universe — the sweep can never invent a name the parser never saw.
TIER_AST = "ast"
TIER_TEXTUAL = "textual"


def confidence_for(origin: ReferenceOrigin) -> float:
    """Return the rubric confidence for *origin*.

    Raises ``KeyError`` rather than defaulting: an origin with no rubric entry
    is a programming error, and silently scoring it 0.0 would let a real site
    sort below a string match.
    """
    return ORIGIN_CONFIDENCE[origin]


__all__ = [
    "CONFIDENCE_ADVISORY",
    "CONFIDENCE_CERTAIN",
    "EXTRACTOR_VERSION",
    "ORIGIN_CONFIDENCE",
    "TEXTUAL_ORIGINS",
    "TIER_AST",
    "TIER_TEXTUAL",
    "ReferenceKind",
    "ReferenceOrigin",
    "confidence_for",
]
