"""What an MCP client sees, as distinct from what a tool function returns.

The fork's trust envelope — ``contract_version``, ``timing_ms``,
``embedder_degraded``, ``source_search``, ``repo_freshness``,
``index_age_days``, ``index_behind`` and the rest — is metadata *about* an
answer, not part of it. MCP has a channel for exactly that: ``_meta`` on the
JSON-RPC result. Until this module the envelope rode inside the payload under a
``_meta`` key, where an agent reads it as one more field of the answer and has
to know to skip it, and where it was serialised a second time into the text
content block. :func:`wire` moves it to the protocol and sends the payload
flat.

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
"""

from __future__ import annotations

import functools
import inspect
from typing import Annotated, Any

import pydantic_core
from mcp.types import CallToolResult, TextContent

from repowise.server.mcp_server._signature import PAYLOAD_ANNOTATION, evaluated_signature, freeze


def wire(fn: Any) -> Any:
    """Wrap an async MCP tool so it returns a finished protocol result."""
    if not inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        return as_call_tool_result(await fn(*args, **kwargs))

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


def as_call_tool_result(payload: Any) -> Any:
    """*payload* as a protocol result, with its ``_meta`` promoted off it."""
    if not isinstance(payload, dict):
        # No tool returns one. For a shape this was not written for, FastMCP's
        # own conversion is a better answer than a hand-built result.
        return payload
    envelope = payload.get("_meta")
    if not isinstance(envelope, dict) or not envelope:
        return CallToolResult(content=text_block(payload), structuredContent=payload)
    # A copy rather than a ``pop``: the layers below have already measured,
    # budgeted and recorded this exact dict, and a wrapper that mutates the
    # value it was handed makes the object delivered differ from the object
    # accounted for. One shallow top-level copy against a full serialisation.
    flat = {key: value for key, value in payload.items() if key != "_meta"}
    # ``CallToolResult.meta`` is ``Field(alias="_meta")`` on a model that does
    # not populate by field name, so a ``meta=`` keyword is accepted as an
    # *extra* key called ``meta`` and leaves ``_meta`` null — silently, because
    # the model allows extras. The SDK passes the alias for the same reason
    # (``mcp/server/lowlevel/server.py``, on ``ResourceContents``).
    return CallToolResult(content=text_block(flat), structuredContent=flat, **{"_meta": envelope})


def text_block(payload: dict[str, Any]) -> list[TextContent]:
    """The unstructured content block, mirroring the flat payload.

    Rendered exactly as FastMCP renders a dict result
    (``func_metadata._convert_to_content``), so the only change a client sees
    in the text is the absent ``_meta``. Absent on purpose: leaving the
    envelope in the block every client displays would put a second copy of it
    back on the wire, which is the defect this module exists to remove.
    """
    text = pydantic_core.to_json(payload, fallback=str, indent=2).decode()
    return [TextContent(type="text", text=text)]


__all__ = ["as_call_tool_result", "text_block", "wire"]
