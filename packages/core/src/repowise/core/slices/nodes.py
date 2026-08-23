"""Node classification shared by entry-point resolution and the walk.

Both surfaces ask the same two questions of a ``graph_nodes`` row — where does
it live, and is it test material — and both got the second one wrong in the
same way before this module existed, because the answer differs by layer.
"""

from __future__ import annotations

from repowise.core.ids import SymbolId, parse
from repowise.core.persistence.models import GraphNode
from repowise.core.test_paths import is_test_path


def node_path(node: GraphNode) -> str:
    """Repo-relative path a graph node lives at.

    A file node *is* its path; a symbol node carries one, and falls back to
    the path half of its ``path::Name`` id when the column is empty (which a
    partially rebuilt index can leave it).
    """
    if node.node_type == "file":
        return node.node_id
    if node.file_path:
        return node.file_path
    parsed = parse(node.node_id)
    return parsed.path if isinstance(parsed, SymbolId) else node.node_id


def is_test_node(node: GraphNode) -> bool:
    """Whether a graph node is test material, at either layer.

    ``GraphNode.is_test`` is the canonical answer ingestion records — but it
    records it on **file** nodes only. Measured on a freshly built index,
    every symbol defined inside a test file carries ``is_test = False``, so a
    symbol-layer query filtering on that column alone pulls test functions
    into a result whose file layer correctly excluded their file. Falling back
    to the shared path classifier (:func:`repowise.core.test_paths.is_test_path`
    — the same rule ingestion used to set the flag in the first place) keeps
    the two layers agreeing.
    """
    if node.is_test:
        return True
    return is_test_path(node_path(node), node.language or None)
