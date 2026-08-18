"""Stable handles for execution flows.

A flow had no identity. ``get_execution_flows`` returned a list whose entries
were distinguished by their position and by ``entry_point`` — a graph node id —
and neither survives contact with a second session. Position moves because the
ordering is by a score recomputed at index time; the node id survives a reindex
but says nothing about *which* index produced the trace beside it, so an agent
holding one cannot tell a re-run from a different answer.

A flow id fixes both ends. It is a pure function of the entry point and the
generation of the index the trace was walked from, so:

* the same request in a later session returns the same id, and
* a reindex that changes the call graph changes the id, visibly, instead of
  silently changing what the old id means.

The shape is ``flow:<generation>:<entry point node id>`` — self-describing on
purpose. An opaque digest would need somewhere to store the reverse mapping,
and the only candidates were a new table (heavy, and the schema belongs to the
lifecycle unit) or a scan of the scored entry points (which cannot resolve a
flow traced from an explicit ``entry_point`` argument that never ranked). The
node id is already in the payload as ``entry_point``, so nothing is disclosed
that the caller did not already have, and a human debugging a trace can read
the handle.

Nothing here touches the database or persists anything: an id is derived on
demand from values the caller already holds, which is what makes it a function
rather than a name.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FLOW_ID_PREFIX",
    "GENERATION_DIGEST_CHARS",
    "ParsedFlowId",
    "index_generation",
    "mint_flow_id",
    "parse_flow_id",
]

FLOW_ID_PREFIX = "flow"

#: Width of the generation digest. Twelve hex characters is 48 bits — far more
#: than enough to tell a repository's successive indexes apart, and short enough
#: that the handle stays readable next to the node id it carries.
GENERATION_DIGEST_CHARS = 12

#: Stand-in for a generation input a store has not recorded. An index written
#: before a field existed reads as the empty string rather than as an error, so
#: old stores mint stable ids too — they simply cannot distinguish the axis they
#: never captured.
_ABSENT = ""


@dataclass(frozen=True, slots=True)
class ParsedFlowId:
    """A flow id taken apart: which index generation, and which entry point."""

    generation: str
    entry_point: str


def index_generation(repository: Any) -> str:
    """The generation digest of the graph *repository*'s flows are walked from.

    Two inputs, because two things independently decide what a trace finds: the
    commit the index was built at, and the parser build that wrote the ``calls``
    edges. Re-indexing the same commit with a changed extractor produces a
    different call graph, and a handle that ignored that would go on claiming a
    trace it no longer describes.

    Deliberately not the wall-clock index time: two indexes of the same commit
    by the same parser find the same flows, and an id that changed anyway would
    make every re-index look like a change to a caller diffing handles.
    """
    head = getattr(repository, "head_commit", None) or _ABSENT
    parser = getattr(repository, "graph_edges_parser_fingerprint", None) or _ABSENT
    digest = hashlib.sha256()
    digest.update(f"head_commit={head}\n".encode())
    digest.update(f"graph_edges_parser_fingerprint={parser}\n".encode())
    return digest.hexdigest()[:GENERATION_DIGEST_CHARS]


def mint_flow_id(generation: str, entry_point: str) -> str:
    """The handle for the flow from *entry_point* under *generation*."""
    return f"{FLOW_ID_PREFIX}:{generation}:{entry_point}"


def parse_flow_id(flow_id: str) -> ParsedFlowId | None:
    """Take *flow_id* apart, or None when it is not one.

    Split on the first colon after the prefix and never on the rest: a node id
    is ``file.py::Symbol`` and contains colons of its own, so anything stricter
    would reject exactly the ids this mints.
    """
    text = (flow_id or "").strip()
    prefix = f"{FLOW_ID_PREFIX}:"
    if not text.startswith(prefix):
        return None
    generation, separator, entry_point = text[len(prefix) :].partition(":")
    if not separator or not generation or not entry_point:
        return None
    return ParsedFlowId(generation=generation, entry_point=entry_point)
