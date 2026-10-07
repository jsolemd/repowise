"""What an MCP client sees, as distinct from what a tool function returns.

The full envelope rides on the JSON-RPC result's ``_meta``. A compact ``trust``
projection stays in the flat payload: hosts do not reliably pass protocol
metadata to their model, and stale or degraded evidence must remain visible.
The shared trust middleware prepares that projection before the response is
budgeted and measured. Timings and other diagnostics remain protocol-only.

:func:`wire` is the outermost middleware layer (see
``mcp_server.tool_middleware``) and the only place the envelope is lifted off.
That placement is the design:

* Outside the savings and budget layers, so the ledger and the response budget
  keep measuring the dict payload as delivered. Neither can measure a
  ``CallToolResult``.
* Only on the registered wrapper, so nothing changes for the tool functions
  themselves. ``repowise search`` awaits ``search_codebase`` in process
  (``cli/tool_bridge.py``) and reads ``payload["_meta"]`` to say when an answer
  came back lexical-only; that path never reaches this layer, so the internal
  contract holds by construction rather than by care.

FastMCP passes a returned ``CallToolResult`` straight through
(``fastmcp/utilities/func_metadata.py`` ``convert_result``, then
``lowlevel/server.py``), validating only its ``structuredContent`` against the
advertised ``outputSchema`` — which :mod:`._signature` has just made the flat
payload's schema. The two halves of the wire change are one change.

A response whose semantic lane did not run leads with a top-level ``warning``.
The embedder notes its own failures on the request (``embedding.outage``), so
every tool that embeds is covered, including the paths that swallow the error
and return keyword hits; a vector leg that failed for another reason is read
from the envelope's ``retrieval_degraded``. ``trust`` alone was not enough: an
agent reads results before metadata, and a lexical miss reads as an absence.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Sequence
from typing import Annotated, Any

import pydantic_core
from mcp.types import CallToolResult, TextContent

from repowise.core.providers.embedding.outage import keyword_only_notices
from repowise.server.mcp_server._meta import agent_trust
from repowise.server.mcp_server._signature import PAYLOAD_ANNOTATION, evaluated_signature, freeze

PRETTY_JSON_ENV = "REPOWISE_MCP_PRETTY_JSON"


def wire(fn: Any) -> Any:
    """Wrap an async MCP tool so it returns a finished protocol result."""
    if not inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        with keyword_only_notices() as notices:
            payload = await fn(*args, **kwargs)
        return as_call_tool_result(payload, keyword_only=notices)

    return freeze(_wrapped, _wire_signature(fn))


def _wire_signature(fn: Any) -> inspect.Signature | None:
    """*fn*'s signature, re-declared as what :func:`wire`'s wrapper returns.

    ``Annotated[CallToolResult, T]`` is FastMCP's own spelling for "this
    callable builds its own protocol result; advertise and validate it as
    ``T``". A bare ``CallToolResult`` annotation would suppress the output
    schema altogether, and claiming to return the payload would be a lie about
    a wrapper that returns a result object.

    ``T`` is the tool's own evaluated return annotation rather than a constant,
    which is what keeps ``_signature.evaluated_signature`` load-bearing: a
    middleware layer that went back to freezing an unevaluated signature would
    put the *string* there, and FastMCP would rebuild the ``{"result": ...}``
    wrapper for all 25 tools.
    """
    signature = evaluated_signature(fn)
    if signature is None:
        return None
    payload = signature.return_annotation
    if payload is inspect.Signature.empty:  # pragma: no cover - every tool annotates
        payload = PAYLOAD_ANNOTATION
    return signature.replace(return_annotation=Annotated[CallToolResult, payload])


def as_call_tool_result(payload: Any, *, keyword_only: Sequence[str] = ()) -> Any:
    """*payload* as a protocol result, with its ``_meta`` promoted off it.

    *keyword_only* holds the warnings the embedder noted on this request.
    """
    if not isinstance(payload, dict):
        # No tool returns one. For a shape this was not written for, FastMCP's
        # own conversion is a better answer than a hand-built result.
        return payload
    envelope = payload.get("_meta")
    warning = _keyword_only_warning(keyword_only, envelope)
    if warning is not None:
        payload = _lead_with(warning, payload)
    if not isinstance(envelope, dict):
        return CallToolResult(content=text_block(payload), structuredContent=payload)
    # A copy rather than a ``pop``: the layers below have already measured,
    # budgeted and recorded this exact dict, and a wrapper that mutates the
    # value it was handed makes the object delivered differ from the object
    # accounted for. One shallow top-level copy against a full serialisation.
    flat = {key: value for key, value in payload.items() if key != "_meta"}
    # Many MCP hosts deliver only content/structuredContent to the model.
    # Preserve actionable trust facts there; protocol-only diagnostics cannot
    # help an agent decide whether it may rely on an answer.
    trust = agent_trust(envelope)
    if trust:
        existing = flat.get("trust")
        flat["trust"] = {**trust, **existing} if isinstance(existing, dict) else trust
    # ``CallToolResult.meta`` is ``Field(alias="_meta")`` on a model that does
    # not populate by field name, so a ``meta=`` keyword is accepted as an
    # *extra* key called ``meta`` and leaves ``_meta`` null — silently, because
    # the model allows extras. The SDK passes the alias for the same reason
    # (``mcp/server/lowlevel/server.py``, on ``ResourceContents``).
    return CallToolResult(content=text_block(flat), structuredContent=flat, **{"_meta": envelope})


def _keyword_only_warning(notices: Sequence[str], envelope: Any) -> str | None:
    if notices:
        return " ".join(notices)
    if isinstance(envelope, dict) and envelope.get("retrieval_degraded"):
        return str(
            envelope.get("retrieval_degraded_reason")
            or "Keyword-only results: semantic retrieval did not run for this query. "
            "A miss here is not evidence of absence."
        )
    return None


def _lead_with(warning: str, payload: dict[str, Any]) -> dict[str, Any]:
    """A copy of *payload* whose first key is ``warning``, keeping any it had."""
    existing = payload.get("warning")
    if existing:
        warning = f"{warning} {existing}"
    return {
        "warning": warning,
        **{key: value for key, value in payload.items() if key != "warning"},
    }


def text_block(payload: dict[str, Any]) -> list[TextContent]:
    """The unstructured content block, mirroring the flat payload.

    Compact JSON avoids spending the agent's context budget on presentation
    whitespace. Source strings retain their own line breaks and indentation.
    The trust projection is visible; the full diagnostic envelope is not
    duplicated in the text.
    """
    pretty = os.environ.get(PRETTY_JSON_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    text = pydantic_core.to_json(payload, fallback=str, indent=2 if pretty else None).decode()
    return [TextContent(type="text", text=text)]


__all__ = ["as_call_tool_result", "text_block", "wire"]
