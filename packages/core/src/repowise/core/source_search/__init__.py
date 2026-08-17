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

__all__ = ["SOURCE_SEARCH_ENV", "TRUTHY", "source_search_enabled"]

#: Environment variable that gates every source-search code path.
SOURCE_SEARCH_ENV = "REPOWISE_SOURCE_SEARCH"

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
