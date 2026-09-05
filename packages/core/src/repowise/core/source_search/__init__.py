"""Source-code retrieval corpus: symbol chunks and file windows.

The wiki index answers questions about a repository in prose the generator
wrote. This package indexes the repository's *own text* — one chunk per
persisted symbol, plus overlapping line windows over the shell scripts,
compose files and unit definitions no grammar produces symbols for — so a
retriever can return the lines that exist rather than a page about them.

Three stores, one recipe:

* :mod:`.chunks` builds the chunk text and is pure — no I/O, no database.
* :mod:`.vector_store` holds the dense leg in LanceDB (``source_chunks``).
* :mod:`.fts` holds the lexical leg in SQLite FTS5 over a camelCase-split
  token stream, so ``parseConfig`` finds ``parse_config`` and vice versa.

:mod:`.manifest` records what a build was made of, and :mod:`.indexer`
orchestrates the four.

Nothing here runs unless ``REPOWISE_SOURCE_SEARCH`` is set. This module is
deliberately import-cheap — stdlib only — so a caller can ask whether the
feature is on without pulling LanceDB, SQLAlchemy or the language registry
onto its path.
"""

from __future__ import annotations

import os

__all__ = [
    "QUERY_FOCUS_ENV",
    "SOURCE_SEARCH_ENV",
    "TRUTHY",
    "query_focus_enabled",
    "source_search_enabled",
]

#: Environment variable that gates every source-search code path.
SOURCE_SEARCH_ENV = "REPOWISE_SOURCE_SEARCH"

#: Environment variable that gates verbose-query focusing. Off by default.
QUERY_FOCUS_ENV = "REPOWISE_SOURCE_QUERY_FOCUS"

#: Values that count as "on", case-insensitively.
TRUTHY = frozenset({"1", "true", "yes", "on"})


def source_search_enabled() -> bool:
    """Whether the source-search corpus is enabled for this process.

    Defaults to **off**, the opposite of
    :func:`repowise.core.analysis.kg_curation.curation_enabled`, and for the
    same reason read in reverse: that feature has an acceptance gate behind
    it and this one does not yet, so an unset variable must leave the
    product exactly as it was.

    Resolved at the call site rather than at import, so a test can set the
    variable and a already-imported module still sees it.
    """
    return os.environ.get(SOURCE_SEARCH_ENV, "").strip().lower() in TRUTHY


def query_focus_enabled() -> bool:
    """Whether a verbose prose query is focused before retrieval.

    Defaults to **off**, and for the reason
    :func:`source_search_enabled` gives: the feature has no acceptance gate
    behind it yet, so an unset variable must leave retrieval exactly as it was.

    Measured on the landing corpus, the heuristic loses real subjects. It drops
    any concept whose exact token has zero document frequency — which is every
    morphological variant the dense leg would have matched (``committing`` when
    the corpus says ``commit``, ``sentence`` when it says ``sentences``) — while
    keeping corpus-common function words like ``an``, then embeds the mangled
    string. Symbol-surfacing verdicts went 10 exact / 1 enclosing / 1 wrong_file
    to 5 / 3 / 4. The code and its tests stay; the default does not.

    Resolved at the call site, so a test can set the variable and an
    already-imported module still sees it.
    """
    return os.environ.get(QUERY_FOCUS_ENV, "").strip().lower() in TRUTHY
