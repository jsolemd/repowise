"""Three fidelity levels for one slice member.

``card``
    Identity and rank only — what the member is, where it lives, why it is in
    the slice. Cheap enough that a two-hundred-member slice still fits a small
    budget, which is what makes "show me everything, briefly" a real option
    rather than a truncated one.
``skeleton``
    The card plus the shape: signature, the first line of the docstring, and —
    for a file member — the signatures of the symbols it defines. Enough to
    decide whether to open it.
``full``
    The skeleton plus source, with symbol identity and bounds checked by a
    live parse cached once per file for this render. Bounded per member by
    ``max_source_lines``: one four-thousand-line file cannot consume a whole slice's
    budget on its own, and when it is cut the cut is stated on the member
    (``source_truncated``), not inferred from a short string.

Everything a view needs beyond the member row is loaded once, in batch, by
:func:`prepare`. Rendering itself touches no database and no disk except the
source cache, so the budgeter can render a member, price it, and throw it away
without paying for it twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.ingestion.models import FileInfo, Symbol
from repowise.core.ingestion.source_text import decode_source
from repowise.core.persistence.models import WikiSymbol
from repowise.core.slices.models import VIEWS, SliceMember

logger = logging.getLogger(__name__)

#: Symbols listed in a file member's skeleton. A barrel re-exporting two
#: hundred names is not more informative at two hundred than at twenty.
_FILE_SKELETON_SYMBOL_CAP = 20

#: Default per-member source cap for the ``full`` view.
DEFAULT_MAX_SOURCE_LINES = 200

#: SQLite parameter-limit-safe batch for the detail queries.
_BATCH = 400


def normalize_view(view: str | None) -> str:
    """Coerce a caller's view name, defaulting to the middle fidelity."""
    candidate = (view or "skeleton").strip().lower()
    return candidate if candidate in VIEWS else "skeleton"


@dataclass
class ViewContext:
    """Batch-loaded material every view of one slice needs."""

    # None means no symbol lookup was requested (the card view). An empty
    # mapping means the lookup found none, so an old source range is unsafe.
    symbols: dict[str, WikiSymbol] | None = None
    file_symbols: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    file_symbol_totals: dict[str, int] = field(default_factory=dict)
    repo_root: Path | None = None
    max_source_lines: int = DEFAULT_MAX_SOURCE_LINES
    _source_cache: dict[str, list[str] | None] = field(default_factory=dict)
    _live_symbols: dict[str, dict[str, Symbol]] = field(default_factory=dict)

    def source_lines(self, rel_path: str) -> list[str] | None:
        """Lines of one repo file, read once per slice render."""
        if rel_path in self._source_cache:
            return self._source_cache[rel_path]
        lines: list[str] | None = None
        if self.repo_root is not None:
            # An index row is not a trust boundary: a checked-in symlink (the
            # walker lists symlinked files), a ``..`` segment, or an absolute
            # path in a member's rel_path must not read outside the repo.
            try:
                root = self.repo_root.resolve()
                target = (root / rel_path).resolve()
                target.relative_to(root)
                data = target.read_bytes()
            except (OSError, ValueError):
                data = None
            if data is not None:
                lines = decode_source(data).splitlines()
        self._source_cache[rel_path] = lines
        return lines

    def live_symbol(self, member: SliceMember) -> Symbol | None:
        """Resolve an exact symbol from the same lines this render will serve."""
        path = member.file_path
        if path in self._live_symbols:
            return self._live_symbols[path].get(member.node_id)
        self._live_symbols[path] = {}
        lines = self.source_lines(path)
        if lines is None:
            return None
        source = "\n".join(lines).encode("utf-8")
        info = FileInfo(
            path=path,
            abs_path=str(self.repo_root / path) if self.repo_root else path,
            language=member.language,
            size_bytes=len(source),
            git_hash="",
            last_modified=datetime.now(UTC),
            is_test=False,
            is_config=False,
            is_api_contract=False,
            is_entry_point=False,
        )
        try:
            from repowise.core.ingestion.parser import ASTParser

            parsed = ASTParser().parse_file(info, source)
        except Exception:
            logger.warning("Could not parse live slice source %s", path, exc_info=True)
            return None
        if parsed.parse_errors:
            return None
        self._live_symbols[path] = {symbol.id: symbol for symbol in parsed.symbols}
        return self._live_symbols[path].get(member.node_id)


def _chunks(values: list[str]) -> list[list[str]]:
    return [values[i : i + _BATCH] for i in range(0, len(values), _BATCH)]


async def prepare(
    session: AsyncSession,
    repo_id: str,
    members: list[SliceMember],
    *,
    view: str,
    repo_root: Path | str | None = None,
    max_source_lines: int = DEFAULT_MAX_SOURCE_LINES,
) -> ViewContext:
    """Load, in batch, everything the requested view needs beyond the members.

    ``card`` needs nothing, so it issues no query at all — the cheapest view
    is cheap all the way down, not just in its output.
    """
    ctx = ViewContext(
        repo_root=Path(repo_root) if repo_root else None,
        max_source_lines=max(1, max_source_lines),
    )
    if view == "card" or not members:
        return ctx

    ctx.symbols = {}
    symbol_ids = [m.node_id for m in members if m.layer == "symbol"]
    for chunk in _chunks(symbol_ids):
        rows = await session.execute(
            select(WikiSymbol).where(
                WikiSymbol.repository_id == repo_id,
                WikiSymbol.symbol_id.in_(chunk),
            )
        )
        for row in rows.scalars().all():
            ctx.symbols[row.symbol_id] = row

    file_paths = [m.node_id for m in members if m.layer == "file"]
    for chunk in _chunks(file_paths):
        rows = await session.execute(
            select(WikiSymbol)
            .where(
                WikiSymbol.repository_id == repo_id,
                WikiSymbol.file_path.in_(chunk),
            )
            .order_by(WikiSymbol.file_path, WikiSymbol.start_line, WikiSymbol.name)
        )
        for row in rows.scalars().all():
            bucket = ctx.file_symbols.setdefault(row.file_path, [])
            ctx.file_symbol_totals[row.file_path] = ctx.file_symbol_totals.get(row.file_path, 0) + 1
            if len(bucket) < _FILE_SKELETON_SYMBOL_CAP:
                bucket.append(
                    {
                        "name": row.name,
                        "kind": row.kind,
                        "signature": row.signature or "",
                        "lines": [row.start_line, row.end_line],
                    }
                )
    return ctx


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0][:240]


def _source_for(member: SliceMember, ctx: ViewContext) -> dict[str, Any]:
    """Source block for a ``full`` member, with every shortfall stated."""
    lines = ctx.source_lines(member.file_path)
    if lines is None:
        return {
            "source": None,
            "source_unavailable": (
                f"{member.file_path} could not be read from the working tree; "
                f"the index knows the member but the file is gone or unreadable."
            ),
        }

    if member.layer == "symbol" and member.start_line:
        start = max(1, member.start_line)
        end = member.end_line or start
        window = lines[start - 1 : end]
        first_line_no = start
    else:
        window = lines
        first_line_no = 1

    if not window:
        # The file read fine and the slice's line range landed outside it, so
        # the index and the working tree disagree about this symbol. An empty
        # ``source`` string would read as "this function has no body"; say what
        # actually happened instead.
        return {
            "source": None,
            "source_unavailable": (
                f"{member.file_path} has {len(lines)} line(s) but the index places this "
                f"member at lines {first_line_no}-{member.end_line or first_line_no}; "
                f"the file changed since it was indexed. Re-index to recover the body."
            ),
        }

    total = len(window)
    truncated = total > ctx.max_source_lines
    if truncated:
        window = window[: ctx.max_source_lines]

    block: dict[str, Any] = {
        "source": "\n".join(window),
        "source_first_line": first_line_no,
        "source_lines": len(window),
    }
    if truncated:
        block["source_truncated"] = True
        block["source_lines_omitted"] = total - len(window)
        block["source_truncation_note"] = (
            f"source cut at max_source_lines={ctx.max_source_lines}; "
            f"{total - len(window)} line(s) not shown"
        )
    return block


def _resolve_member(
    member: SliceMember, view: str, ctx: ViewContext
) -> tuple[SliceMember, str | None]:
    """Keep slice membership/rank while resolving metadata from its current owner."""
    if view == "card" or member.layer != "symbol":
        return member, None
    row = ctx.symbols.get(member.node_id) if ctx.symbols is not None else None
    if ctx.symbols is not None and row is None:
        return member, (
            f"{member.node_id} is no longer in the symbol index; "
            "its former line range cannot identify the current source."
        )
    if row is not None:
        member = replace(member, file_path=row.file_path, language=row.language)
    resolved: WikiSymbol | Symbol | None = row
    if view == "full" and ctx.source_lines(member.file_path) is not None:
        resolved = ctx.live_symbol(member)
        if resolved is None:
            return member, (
                f"{member.node_id} could not be identified in a successful live parse; "
                "the symbol was removed or renamed, the file has parse errors, "
                "or its parser is unavailable. Indexed bounds cannot identify its source."
            )
    if resolved is not None:
        member = replace(
            member,
            name=resolved.name,
            kind=resolved.kind,
            start_line=resolved.start_line,
            end_line=resolved.end_line,
            signature=resolved.signature,
            docstring=resolved.docstring,
        )
    return member, None


def render_member(member: SliceMember, view: str, ctx: ViewContext) -> dict[str, Any]:
    """Render one member at the requested fidelity."""
    member, unavailable = _resolve_member(member, view, ctx)
    payload = member.identity()
    payload["why"] = member.reasons[:3]
    if member.edge_types:
        payload["edge_types"] = sorted(member.edge_types)
    if view == "card":
        return payload

    signature = member.signature
    if signature:
        payload["signature"] = signature
    doc = _first_line(member.docstring)
    if doc:
        payload["doc"] = doc
    if member.layer == "file":
        symbols = ctx.file_symbols.get(member.node_id, [])
        if symbols:
            payload["defines"] = symbols
            total = ctx.file_symbol_totals.get(member.node_id, len(symbols))
            if total > len(symbols):
                payload["defines_truncated"] = total - len(symbols)

    if view == "full":
        if unavailable:
            payload.update(source=None, source_unavailable=unavailable)
        else:
            payload.update(_source_for(member, ctx))
    return payload
