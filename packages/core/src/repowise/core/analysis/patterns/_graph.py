"""Read-only symbol-graph view shared by every pattern query.

The six patterns are graph predicates, not analyses: each one is a filter
over "which symbols are related to which, and where do they live". They
need the same three things — the symbol table, the call/reference
adjacency in both directions, and the file each symbol was defined in —
so those are built once here and handed to every pattern.

The input is the node/edge *row* shape that
:func:`repowise.core.persistence.crud.graph.get_all_graph_nodes` and
:func:`~repowise.core.persistence.crud.graph.get_all_graph_edges` already
produce, which is also what an in-memory ``GraphBuilder`` graph decomposes
into. That is deliberate: the same pattern code answers from a persisted
index (the MCP tool) and from a graph that was just built in memory (a
test, a pre-commit run) without a storage adapter in between.

Two shapes in the raw graph need explaining before the patterns read them:

* **Synthetic module symbols** — ingestion mints one ``<file>::__module__``
  symbol per file to own that file's top-level calls. It is real as an
  *edge endpoint* (a top-level call is a genuine use of the callee) and
  meaningless as a *result* (no reader wants "``__module__`` is an orphan
  export"). It stays in the adjacency and is excluded from every candidate
  set, via :attr:`Symbol.synthetic`.
* **Test membership** — ``is_test`` is stamped on file nodes, not symbol
  nodes, so a symbol's test-ness is derived from its defining file here
  rather than read off the symbol row (where it is always ``False``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from posixpath import dirname
from typing import Any

#: Edge types that mean "this symbol uses that symbol". ``calls`` is a
#: resolved invocation; ``references`` is a resolved non-call mention
#: (a decorator, a type annotation, a value passed by name). Both are
#: uses, and a pattern that counted only calls would call a symbol
#: referenced ten times an orphan.
USE_EDGE_TYPES: frozenset[str] = frozenset({"calls", "references"})

#: The visibility values that mean "declared for use outside this scope".
#: ``local`` is the one value that never qualifies — the parser stamps it
#: on function-local definitions, which have no export surface at all.
EXPORTED_VISIBILITY: frozenset[str] = frozenset({"public", "protected", "internal"})

_MODULE_SUFFIX = "::__module__"


@dataclass(frozen=True, slots=True)
class Symbol:
    """One symbol node, flattened to the fields the patterns predicate on."""

    id: str
    name: str
    kind: str
    file: str
    directory: str
    start_line: int
    end_line: int
    visibility: str
    signature: str
    parent_id: str
    is_test: bool
    synthetic: bool

    @property
    def lines(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    @property
    def exported(self) -> bool:
        return self.visibility in EXPORTED_VISIBILITY

    def location(self) -> dict[str, Any]:
        """The identity block every pattern match carries."""
        return {
            "symbol": self.id,
            "name": self.name,
            "kind": self.kind,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "visibility": self.visibility,
        }


@dataclass(frozen=True, slots=True)
class FileNode:
    """The file-level facts the patterns need (test-ness, entry-point-ness)."""

    path: str
    is_test: bool
    is_entry_point: bool
    language: str


@dataclass
class SymbolGraph:
    """Symbols plus their use-adjacency, indexed for the pattern queries."""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    files: dict[str, FileNode] = field(default_factory=dict)
    #: symbol id -> ids of symbols that use it (inbound)
    callers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: symbol id -> ids of symbols it uses (outbound)
    callees: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: file path -> ids of the non-synthetic symbols defined in it
    defined_in: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: (source file, target file) for every direct symbol-level use edge
    file_use_pairs: frozenset[tuple[str, str]] = frozenset()

    # -- construction ----------------------------------------------------

    @classmethod
    def from_rows(
        cls,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        *,
        use_edge_types: frozenset[str] = USE_EDGE_TYPES,
    ) -> SymbolGraph:
        """Build the view from persisted (or rehydrated) node/edge rows.

        Unknown node types and edge types are ignored rather than rejected:
        the graph schema grows new edge kinds over time, and a pattern that
        crashed on one would be a worse outcome than a pattern that does
        not count it.
        """
        files: dict[str, FileNode] = {}
        for row in nodes:
            if row.get("node_type") != "file":
                continue
            path = str(row.get("node_id") or "")
            if not path:
                continue
            files[path] = FileNode(
                path=path,
                is_test=bool(row.get("is_test")),
                is_entry_point=bool(row.get("is_entry_point")),
                language=str(row.get("language") or ""),
            )

        symbols: dict[str, Symbol] = {}
        defined: dict[str, list[str]] = defaultdict(list)
        for row in nodes:
            if row.get("node_type") != "symbol":
                continue
            sid = str(row.get("node_id") or "")
            if not sid:
                continue
            file_path = str(row.get("file_path") or _file_of(sid))
            synthetic = sid.endswith(_MODULE_SUFFIX)
            file_node = files.get(file_path)
            sym = Symbol(
                id=sid,
                name=str(row.get("name") or ""),
                kind=str(row.get("kind") or ""),
                file=file_path,
                directory=dirname(file_path) or ".",
                start_line=int(row.get("start_line") or 0),
                end_line=int(row.get("end_line") or 0),
                visibility=str(row.get("visibility") or ""),
                signature=str(row.get("signature") or ""),
                # A method's scope is its class; everything else is scoped
                # to the file it was defined in.
                parent_id=str(row.get("parent_symbol_id") or "") or file_path,
                is_test=bool(file_node.is_test) if file_node is not None else False,
                synthetic=synthetic,
            )
            symbols[sid] = sym
            if not synthetic:
                defined[file_path].append(sid)

        callers: dict[str, list[str]] = defaultdict(list)
        callees: dict[str, list[str]] = defaultdict(list)
        file_pairs: set[tuple[str, str]] = set()
        for row in edges:
            if str(row.get("edge_type") or "") not in use_edge_types:
                continue
            src = str(row.get("source_node_id") or "")
            dst = str(row.get("target_node_id") or "")
            if src not in symbols or dst not in symbols or src == dst:
                continue
            callees[src].append(dst)
            callers[dst].append(src)
            file_pairs.add((symbols[src].file, symbols[dst].file))

        return cls(
            symbols=symbols,
            files=files,
            callers={k: tuple(dict.fromkeys(v)) for k, v in callers.items()},
            callees={k: tuple(dict.fromkeys(v)) for k, v in callees.items()},
            defined_in={k: tuple(sorted(v)) for k, v in defined.items()},
            file_use_pairs=frozenset(file_pairs),
        )

    @classmethod
    def from_graph(cls, graph: Any, **kwargs: Any) -> SymbolGraph:
        """Build the view from a live ingestion graph (duck-typed NetworkX).

        The same view, one step earlier in the pipeline: a pre-commit or
        test run that has just built the graph in memory can query the
        patterns without persisting it first. Node ids become ``node_id``
        and edge endpoints become ``source_node_id``/``target_node_id``,
        which is the row shape the persisted path already produces.
        """
        nodes = [{**data, "node_id": node_id} for node_id, data in graph.nodes(data=True)]
        edges = [
            {**data, "source_node_id": src, "target_node_id": dst}
            for src, dst, data in graph.edges(data=True)
        ]
        return cls.from_rows(nodes, edges, **kwargs)

    # -- queries ---------------------------------------------------------

    def candidates(self, *, include_tests: bool = False) -> list[Symbol]:
        """Every symbol a pattern may *report*, in a stable order.

        Synthetic module scopes are always excluded — they are edge
        endpoints, not findings — and so are function-local definitions,
        which no reader can act on from outside their enclosing body.
        """
        out = [
            sym
            for sym in self.symbols.values()
            if not sym.synthetic
            and sym.visibility != "local"
            and (include_tests or not sym.is_test)
        ]
        out.sort(key=lambda s: (s.file, s.start_line, s.name))
        return out

    def callers_of(self, symbol_id: str) -> tuple[str, ...]:
        return self.callers.get(symbol_id, ())

    def callees_of(self, symbol_id: str) -> tuple[str, ...]:
        return self.callees.get(symbol_id, ())

    def in_degree(self, symbol_id: str) -> int:
        return len(self.callers.get(symbol_id, ()))

    def out_degree(self, symbol_id: str) -> int:
        return len(self.callees.get(symbol_id, ()))

    def siblings_of(self, symbol: Symbol) -> tuple[Symbol, ...]:
        """The other reportable symbols sharing *symbol*'s parent scope."""
        scope = symbol.parent_id
        if scope == symbol.file:
            pool = self.defined_in.get(symbol.file, ())
        else:
            pool = tuple(
                sid
                for sid in self.defined_in.get(symbol.file, ())
                if self.symbols[sid].parent_id == scope
            )
        return tuple(self.symbols[sid] for sid in pool if sid != symbol.id)

    def module_scope_of(self, file_path: str) -> str | None:
        """The synthetic top-level scope id for *file_path*, when it exists."""
        sid = f"{file_path}{_MODULE_SUFFIX}"
        return sid if sid in self.symbols else None

    def files_of(self, symbol_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for sid in symbol_ids:
            sym = self.symbols.get(sid)
            if sym is not None:
                seen[sym.file] = None
        return tuple(seen)

    def directories_of(self, symbol_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for sid in symbol_ids:
            sym = self.symbols.get(sid)
            if sym is not None:
                seen[sym.directory] = None
        return tuple(seen)


def _file_of(symbol_id: str) -> str:
    """Recover the defining file from a ``<file>::<name>`` symbol id."""
    return symbol_id.rsplit("::", 1)[0] if "::" in symbol_id else ""
