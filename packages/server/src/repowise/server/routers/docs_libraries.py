"""Documentation inventory and tools under the authenticated RepoWise API."""

import json

from fastapi import APIRouter, Depends, HTTPException, Request

from repowise.docs.client import DocsClient, DocsUnavailable
from repowise.server.deps import verify_api_key

router = APIRouter(
    prefix="/api/docs-libraries",
    tags=["docs-libraries"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("")
async def get_docs_libraries() -> dict:
    """Return the worker's live inventory, including freshness and jobs."""
    try:
        return await DocsClient().inventory()
    except DocsUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/tools/{name}")
async def call_docs_tool(name: str, request: Request) -> dict:
    """Use the same bounded worker connection as RepoWise MCP."""
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise HTTPException(status_code=415, detail="Use application/json.")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site requests are not allowed.")
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > 65536:
            raise HTTPException(status_code=413, detail="Documentation request is too large.")
    try:
        arguments = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from exc
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=422, detail="Arguments must be a JSON object.")
    try:
        result = await DocsClient().call_tool(name, arguments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocsUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if result.get("status") == "error":
        payload = result.get("payload", {})
        raise HTTPException(
            status_code=422, detail=payload.get("error", "Documentation request failed.")
        )
    return result
