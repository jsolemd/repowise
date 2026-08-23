"""A planted repository for the clone and pattern surfaces.

Every surface under test has exactly one deliberate instance in this
tree, so a test can assert on a named plant instead of on a count:

======================  ====================================================
plant                   where
======================  ====================================================
exact clone pair        ``alpha/report.py::compute_total`` and
                        ``beta/report.py::compute_total`` (byte-identical
                        bodies, long enough to clear the detector's window)
near clone pair         ``core/util.py::shared_helper`` and
                        ``gamma/paraphrase.py::scaled_total`` — same routine,
                        renamed and restructured so the token detector
                        cannot match it
duplicate_signatures    the two ``compute_total`` declarations
orphan_exports          ``core/util.py::orphan_helper``
hub_functions           ``core/util.py::shared_helper``
isolated_siblings       ``core/lonely.py::loner``
reuse_candidates        ``core/util.py::shared_helper``
bridge_functions        ``core/gateway.py::bridge``
======================  ====================================================

``mutate_remove_clone`` rewrites the second half of the exact pair, which
is the mutation leg for "a finding that no longer exists stops being
served".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from repowise.core.analysis.clone_service import NearCloneChunk

# The duplicated body. Long enough to clear DEFAULT_WINDOW_TOKENS (50) and
# DEFAULT_MIN_LINES (6) with headroom, so the plant does not depend on a
# borderline token count.
_CLONE_BODY = """
def compute_total(items, rate):
    running = 0
    skipped = 0
    for entry in items:
        if entry is None:
            skipped = skipped + 1
            continue
        if entry < 0:
            skipped = skipped + 1
            continue
        running = running + entry * rate
    if skipped > 0:
        running = running - skipped
    if running < 0:
        running = 0
    return running
"""

_UTIL = '''
"""Shared helpers."""


def shared_helper(value, factor):
    total = 0
    for item in value:
        if item is None:
            continue
        total = total + item * factor
    if total > 100:
        total = 100
    return total


def orphan_helper(value):
    return value * 2
'''

# Same routine as ``shared_helper``, rewritten the way a second author
# would write it: different names, augmented assignment instead of a
# rebind, ``min`` instead of a comparison. The token-kind stream diverges
# often enough that no 50-token window matches, which is what makes this
# a near clone and not an exact one.
_PARAPHRASE = """
def scaled_total(values, multiplier):
    result = 0
    for entry in values:
        if entry is None:
            continue
        result += entry * multiplier
    result = min(result, 100)
    return result
"""

_GATEWAY = """
from core.sink import sink_write


def bridge(payload):
    return sink_write(payload)
"""

_SINK = """
def sink_write(payload):
    return len(payload)
"""

_LONELY = """
def used_one(x):
    return used_two(x) + 1


def used_two(x):
    return x * 3


# Long enough to clear the near-clone size floor so it can serve as the
# below-threshold pair, and structurally unlike anything else planted here
# so it never becomes an accidental clone.
def loner(label, mapping):
    parts = []
    for key in sorted(mapping):
        parts.append("{}={}".format(key, mapping[key]))
    joined = ";".join(parts)
    if not joined:
        return label
    return label + ":" + joined
"""

_ALPHA_TAIL = """

def alpha_entry(items):
    from core.gateway import bridge
    from core.util import shared_helper

    return bridge(shared_helper(items, 2))
"""

_BETA_TAIL = """

def beta_entry(items):
    from core.util import shared_helper

    return shared_helper(items, 3)
"""

# The replacement body for the mutation leg: same name and arity (so the
# duplicate_signatures plant survives) but no shared token window, so the
# clone finding must disappear.
_CLONE_REPLACEMENT = """
def compute_total(items, rate):
    return sum(x for x in items if x) * rate
"""


def build(root: Path) -> Path:
    """Write the planted tree under *root* and return it."""
    files = {
        "core/util.py": _UTIL,
        "core/gateway.py": _GATEWAY,
        "core/sink.py": _SINK,
        "core/lonely.py": _LONELY,
        "gamma/paraphrase.py": _PARAPHRASE,
        "alpha/report.py": _CLONE_BODY + _ALPHA_TAIL,
        "beta/report.py": _CLONE_BODY + _BETA_TAIL,
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.lstrip("\n"), encoding="utf-8")
    return root


def mutate_remove_clone(root: Path) -> None:
    """Rewrite the second half of the exact pair so the clone is gone."""
    (root / "beta/report.py").write_text(
        (_CLONE_REPLACEMENT + _BETA_TAIL).lstrip("\n"), encoding="utf-8"
    )


def scan_entries(root: Path) -> list[tuple[str, str]]:
    """``(repo-relative path, language)`` for every planted file."""
    return sorted(
        (str(p.relative_to(root)).replace("\\", "/"), "python") for p in root.rglob("*.py")
    )


def symbol_graph(root: Path) -> Any:
    """The real ingestion graph for the planted tree.

    Full traverse → parse → build, not a hand-written node list: the
    pattern queries are only worth testing against edges a real resolver
    produced.
    """
    from repowise.core.analysis.patterns import SymbolGraph
    from repowise.core.ingestion import ASTParser, FileTraverser, GraphBuilder

    traverser, parser, builder = FileTraverser(root), ASTParser(), GraphBuilder()
    for file_info in traverser.traverse():
        builder.add_file(parser.parse_file(file_info, Path(file_info.abs_path).read_bytes()))
    return SymbolGraph.from_graph(builder.build())


def symbol_chunks(root: Path) -> list[NearCloneChunk]:
    """One near-clone chunk per real symbol in the planted tree."""
    graph = symbol_graph(root)
    return [
        NearCloneChunk(
            id=sym.id,
            file=sym.file,
            name=sym.name,
            kind=sym.kind,
            start_line=sym.start_line,
            end_line=sym.end_line,
            is_test=sym.is_test,
        )
        for sym in graph.candidates()
    ]


class OracleNearCloneIndex:
    """A :class:`NearCloneIndex` whose similarities the test states outright.

    The production index answers with cosine over learned embeddings, and
    how good those embeddings are is the embedder's property, not this
    module's. What belongs to the near-clone leg is everything downstream
    of a similarity number: the threshold gate, symmetric de-duplication,
    the minimum-size floor, and suppressing pairs the exact detector
    already reported. Stating the similarities makes those assertions
    exact instead of hostage to a stand-in metric's quirks.
    """

    def __init__(
        self,
        chunks: Sequence[NearCloneChunk],
        similarity: Mapping[tuple[str, str], float],
    ) -> None:
        self._chunks = list(chunks)
        self._similarity: dict[frozenset[str], float] = {
            frozenset(pair): score for pair, score in similarity.items()
        }
        self.neighbour_calls = 0

    async def chunks(self) -> Sequence[NearCloneChunk]:
        return self._chunks

    async def neighbours(self, chunk_id: str, *, limit: int) -> Sequence[tuple[str, float]]:
        self.neighbour_calls += 1
        scored = [
            (other.id, self._similarity.get(frozenset({chunk_id, other.id}), 0.0))
            for other in self._chunks
            if other.id != chunk_id
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]


#: The planted symbol ids the tests assert on, by role.
EXACT_PAIR = ("alpha/report.py::compute_total", "beta/report.py::compute_total")
NEAR_PAIR = ("core/util.py::shared_helper", "gamma/paraphrase.py::scaled_total")
UNRELATED_PAIR = ("core/util.py::shared_helper", "core/lonely.py::loner")


def oracle_index(root: Path) -> OracleNearCloneIndex:
    """The default oracle: one near pair, one below-threshold pair, one exact.

    ``EXACT_PAIR`` is scored high on purpose — the exact detector already
    reports it, so the leg must drop it rather than report it twice.
    """
    return OracleNearCloneIndex(
        symbol_chunks(root),
        {
            NEAR_PAIR: 0.93,
            UNRELATED_PAIR: 0.71,
            EXACT_PAIR: 0.99,
        },
    )
