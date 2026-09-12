"""Mutation authorization for the docs host's state-changing HTTP routes.

``repowise-docs`` now owns the docs writer outright, so ``/docs/action/index``
enqueues real indexing work instead of forwarding it to another container. That
makes it a state-changing endpoint on this process, and it needs the same
fail-closed policy the retired code-search transport applied to its own
reindex/webhook routes.

Policy (unchanged from the code-search original, renamed into the docs lane):

- token configured      -> require a matching ``X-Doc-Search-Token`` header
- no token + opt-in     -> allow (operator set ``DOC_SEARCH_ALLOW_UNAUTHENTICATED``)
- no token + no opt-in  -> refuse (fail CLOSED)

The policy is a free function so it is unit-testable without constructing a
server, a Starlette request, or a settings object.
"""

from __future__ import annotations

import hmac
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

#: Canonical header for the docs lane.
MUTATION_TOKEN_HEADER = "X-Doc-Search-Token"

#: Compatibility header retained for ONE release. Callers written against the
#: proxy era (which forwarded ``X-Code-Search-Token`` to the ``:8100`` writer)
#: keep working through the cutover. Delete this fallback -- and the
#: ``_LEGACY_MUTATION_TOKEN_HEADER`` lookup below -- in the release after the
#: docs writer relocation ships.
_LEGACY_MUTATION_TOKEN_HEADER = "X-Code-Search-Token"


def mutation_token_from_request(request: Request | Any) -> str | None:
    """Read the mutation token from the canonical header, then the legacy one."""
    headers = request.headers
    return headers.get(MUTATION_TOKEN_HEADER) or headers.get(_LEGACY_MUTATION_TOKEN_HEADER)


def authorize_mutation_response(
    *,
    webhook_token: str | None,
    allow_unauthenticated: bool,
    provided_token: str | None,
) -> JSONResponse | None:
    """Authorize a state-changing docs request (``/docs/action/index``).

    Returns a ``401`` :class:`~starlette.responses.JSONResponse` when the
    request must be rejected, or ``None`` when the caller may proceed.
    """
    if webhook_token:
        # Constant-time compare so a timing side channel can't recover the
        # token byte-by-byte; both operands must be str for compare_digest.
        if not hmac.compare_digest(provided_token or "", webhook_token):
            return JSONResponse(
                {"error": "Invalid or missing auth token"},
                status_code=401,
            )
        return None

    if allow_unauthenticated:
        return None

    return JSONResponse(
        {
            "error": "unauthenticated_mutation_refused",
            "error_description": (
                "This endpoint changes server state. Set DOC_SEARCH_WEBHOOK_TOKEN "
                "and send it as the X-Doc-Search-Token header, or set "
                "DOC_SEARCH_ALLOW_UNAUTHENTICATED=1 to allow token-less access on "
                "a trusted network."
            ),
        },
        status_code=401,
    )


def authorize_mutation_request(
    request: Request | Any,
    settings: Any,
) -> JSONResponse | None:
    """Apply :func:`authorize_mutation_response` to a live request.

    ``getattr`` defaults keep the policy fail-closed for any settings object
    that predates these fields (test doubles included): no token and no opt-in
    means refuse.
    """
    return authorize_mutation_response(
        webhook_token=getattr(settings, "webhook_token", "") or None,
        allow_unauthenticated=bool(getattr(settings, "allow_unauthenticated", False)),
        provided_token=mutation_token_from_request(request),
    )


__all__ = [
    "MUTATION_TOKEN_HEADER",
    "authorize_mutation_request",
    "authorize_mutation_response",
    "mutation_token_from_request",
]
