"""RepoWise's bounded connection to its optional documentation worker.

This module imports no database, indexer, or scraper. Code navigation can boot
and serve while the docs worker and its dependencies are unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class DocsUnavailable(Exception):  # noqa: N818
    """The worker could not provide a valid response within its request budget."""


class DocsClient:
    """Own worker URL, token, deadlines, bounded responses, and error mapping."""

    def __init__(self, *, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (
            base_url or os.environ.get("REPOWISE_DOCS_URL", "http://127.0.0.1:8101")
        ).rstrip("/")
        self.token = token if token is not None else os.environ.get("REPOWISE_DOCS_TOKEN", "")

    async def request(self, method: str, path: str, *, arguments: dict | None = None) -> dict:
        headers = {"X-Doc-Search-Token": self.token} if self.token else {}
        timeout = httpx.Timeout(35.0, connect=3.0, write=10.0, pool=3.0)
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:  # noqa: SIM117
                async with client.stream(
                    method, self.base_url + path, json=arguments, headers=headers
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > 4 * 1024 * 1024:
                            raise DocsUnavailable("Documentation response exceeded its size limit.")
                    payload = json.loads(body)
                    if not isinstance(payload, dict):
                        raise DocsUnavailable("Documentation worker returned an invalid response.")
                    return payload
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                raise DocsUnavailable(
                    "Documentation worker refused access; check the configured worker token."
                ) from exc
            raise DocsUnavailable("Documentation worker could not complete the request.") from exc
        except httpx.HTTPError as exc:
            raise DocsUnavailable(
                "Documentation worker is unavailable. Start it with: solemd infra up"
            ) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise DocsUnavailable("Documentation worker returned an invalid response.") from exc

    async def inventory(self) -> dict:
        return await self.request("GET", "/docs/libraries")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        from repowise.docs.catalog import get_doc_tool_definitions

        if name not in {tool.name for tool in get_doc_tool_definitions()}:
            raise ValueError(f"Unknown documentation tool: {name}")
        return await self.request("POST", f"/docs/tools/{name}", arguments=arguments)
