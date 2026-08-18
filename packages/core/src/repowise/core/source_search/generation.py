"""Generation identities shared by the source-search stores.

The source corpus spans three persistence engines that cannot share one
transaction.  A generation is therefore an explicit value, not an incidental
timestamp: SQLite, LanceDB, and the publication manifest all carry the same
identity, while the monotonically increasing sequence provides a cheap
visibility predicate for versioned rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "LEGACY_GENERATION",
    "OPEN_ENDED_GENERATION",
    "GenerationRef",
    "vector_table_for_recipe",
    "version_row_id",
]

# SQLite INTEGER and Arrow int64 share this upper bound.  An open version uses
# the sentinel rather than NULL so both stores can use the same branch-free
# predicate: ``valid_from <= active < valid_to``.
OPEN_ENDED_GENERATION = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class GenerationRef:
    """Stable id plus monotonic visibility sequence for one publication."""

    generation_id: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.generation_id:
            raise ValueError("generation_id must not be empty")
        if self.sequence < 0:
            raise ValueError("generation sequence must be non-negative")


LEGACY_GENERATION = GenerationRef("legacy", 0)


def version_row_id(generation: GenerationRef, chunk_id: str) -> str:
    """Deterministic storage key for *chunk_id* in *generation*.

    A retry writes the same row instead of duplicating it.  The full generation
    id (rather than only its integer sequence) keeps separately restored
    databases from accidentally aliasing rows after a copy.
    """

    digest = hashlib.sha256()
    digest.update(generation.generation_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(chunk_id.encode("utf-8"))
    return digest.hexdigest()


def vector_table_for_recipe(recipe_fingerprint: str) -> str:
    """Lance table namespace for one embedding recipe.

    A model or dimensionality change cannot be staged in the active table:
    Arrow fixes vector width in the schema.  A recipe-namespaced table lets the
    replacement build beside the current one and keeps the manifest flip as the
    sole publication point.
    """

    safe = re.sub(r"[^a-f0-9]", "", recipe_fingerprint.lower())[:16]
    if not safe:
        safe = hashlib.sha256(recipe_fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"source_chunks_{safe}"
