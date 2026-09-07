"""``/api/docs-libraries``: proxy the documentation service's library inventory.

The dashboard renders a freshness view over the indexed documentation corpus,
but that corpus lives in a separate, on-demand service (``codeatlas-docs``),
not in this server's store. Proxying server-side keeps the browser talking only
to its own origin — the same reason ``routers.feedback`` relays rather than
letting the page call out — and means the dashboard needs no second base URL,
no second API key, and no CORS grant on the docs service.

The docs lane is opt-in and frequently down, so an unreachable service is a
502 with a sentence a reader can act on, never a stack trace or a hang: the
outbound call carries its own timeout.
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException

from repowise.server.deps import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/docs-libraries",
    tags=["docs-libraries"],
    dependencies=[Depends(verify_api_key)],
)

#: Where the documentation service listens when nothing says otherwise. Read
#: per request rather than at import, matching ``routers.webhooks``: there is no
#: central settings module, and a module-level read freezes whatever the
#: environment happened to be when the process imported this file.
_DEFAULT_DOCS_URL = "http://127.0.0.1:8101"

#: Bounded so a wedged docs service degrades to a 502 instead of holding a
#: dashboard request open. The upstream endpoint is a single database read.
_TIMEOUT_SECONDS = 10.0


def _docs_url() -> str:
    return os.environ.get("REPOWISE_DOCS_URL", _DEFAULT_DOCS_URL)


@router.get("")
async def get_docs_libraries() -> dict:
    """Return the docs service's library inventory verbatim.

    The payload is passed through unchanged: this server has no schema of its
    own for it, and re-modelling it here would mean a second place to update
    every time the docs service adds a field.
    """
    url = f"{_docs_url()}/docs/libraries"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        # Covers connect/read errors and non-2xx alike: to the dashboard both
        # mean "the inventory is not available right now".
        logger.warning("docs_libraries_fetch_failed url=%s error=%s", url, exc)
        raise HTTPException(
            status_code=502,
            detail="Couldn't reach the documentation service.",
        ) from exc
