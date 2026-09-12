"""GitHub remote metadata helpers for doc-search."""

from __future__ import annotations

import logging

import httpx

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None


def _build_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "doc-search/1.0",
    }
    if settings.github_token:
        headers["Authorization"] = f"token {settings.github_token}"
    return headers


def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=10.0, follow_redirects=True)
    return _HTTP_CLIENT


async def close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None


async def get_remote_head_sha(repo: str, branch: str = "main") -> str | None:
    """Get the latest commit SHA from GitHub API without cloning."""
    url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    headers = _build_headers()

    try:
        response = await get_http_client().get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("sha")

        logger.warning(
            "Failed to get remote SHA for %s/%s: HTTP %s",
            repo,
            branch,
            response.status_code,
        )
        return None
    except Exception as exc:
        logger.warning("Error checking remote SHA for %s: %s", repo, exc)
        return None


async def resolve_default_branch(repo: str) -> str | None:
    """Get the default branch for a GitHub repo via API."""
    url = f"https://api.github.com/repos/{repo}"
    headers = _build_headers()

    try:
        response = await get_http_client().get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("default_branch")

        logger.warning(
            "Failed to resolve default branch for %s: HTTP %s",
            repo,
            response.status_code,
        )
    except Exception as exc:
        logger.warning("Error resolving default branch for %s: %s", repo, exc)

    return None
