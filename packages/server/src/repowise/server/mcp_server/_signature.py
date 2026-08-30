"""One evaluated signature for the MCP tool wrappers.

FastMCP builds a tool's schemas from ``inspect.signature(fn, eval_str=True)``
on the callable it is handed. But ``inspect.signature`` returns a pre-set
``__signature__`` *verbatim* — it never evaluates the annotations inside one,
and ``eval_str`` does not reach them. So a middleware wrapper that froze
``inspect.signature(fn)``, with no ``eval_str``, under
``from __future__ import annotations`` handed FastMCP the *string*
``"dict[str, Any]"`` where it expected a type. A string falls through every
branch of FastMCP's schema builder to its catch-all, which wraps the return
value in ``{"result": ...}`` and advertises that wrapper as the output schema.
That is where the fork's ``structuredContent: {"result": <payload>}`` came
from, on all 25 served tools: nobody wrote those schemas and no tool intended
them.

Freezing an *evaluated* signature is the whole fix. It is load-bearing rather
than decorative: :mod:`._wire` derives the payload type it advertises from the
return annotation that arrives up this chain, so a wrapper that stopped
evaluating would put the ``{"result": ...}`` wrapper straight back — where
``tests/unit/server/mcp/test_solemd_wire_contract.py`` names it.

This lives in its own module, importing nothing but the standard library,
because the three wrappers that need it (``_failure_shield``, ``_rounding``,
``_savings.wrapper``) are also on the CLI's in-process tool path, where
``import mcp.types`` costs ~240ms against ``_failure_shield``'s current ~20ms.
The protocol half of the wire shape is :mod:`._wire`, which no CLI path
imports.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

#: What every MCP tool function returns: a JSON object with string keys.
PAYLOAD_ANNOTATION = dict[str, Any]


def evaluated_signature(fn: Any) -> inspect.Signature | None:
    """*fn*'s signature with its string annotations resolved to real types.

    ``None`` when they cannot be resolved — a name that exists only under
    ``TYPE_CHECKING``, an exotic callable. The caller then leaves its wrapper's
    signature alone, which is no worse than the state this replaces, and the
    served schema is pinned, so a tool that lands there fails the wire contract
    rather than quietly regrowing the wrapper.
    """
    try:
        signature = inspect.signature(fn, eval_str=True)
    except Exception:  # pragma: no cover - defensive; no tool is in this state
        return None
    return signature.replace(return_annotation=_payload(signature.return_annotation))


def _payload(annotation: Any) -> Any:
    """Widen a bare ``dict`` return annotation to ``dict[str, Any]``.

    14 of the 29 tool functions are spelled ``-> dict`` and 15 ``-> dict[str,
    Any]``; both describe the same value. FastMCP can only build a schema from
    the second. Bare ``dict`` is a class with no type hints, so it yields *no*
    ``outputSchema`` at all — a different regression from the one being fixed,
    and a worse one, since a client then has nothing to validate against.
    Normalising the spelling here is one line where editing 14 modules to say
    the same thing would be 14, in files other work is holding.

    Anything else is left exactly as written, so a future tool returning a
    ``TypedDict`` or a ``BaseModel`` keeps the richer schema it earns.
    """
    return PAYLOAD_ANNOTATION if annotation is dict else annotation


def freeze(wrapper: Any, signature: inspect.Signature | None) -> Any:
    """Pin *signature* onto *wrapper* — a no-op for ``None`` — and return it."""
    if signature is not None:
        with contextlib.suppress(ValueError, TypeError):  # pragma: no cover - exotic callables
            wrapper.__signature__ = signature  # type: ignore[attr-defined]
    return wrapper


def preserve(wrapper: Any, fn: Any) -> Any:
    """Pin *fn*'s evaluated signature onto *wrapper*, and return *wrapper*."""
    return freeze(wrapper, evaluated_signature(fn))


__all__ = ["PAYLOAD_ANNOTATION", "evaluated_signature", "freeze", "preserve"]
