"""Unit contracts for the docs-lane mutation authorization policy.

Ported from the retired code-search transport, which is why the policy is a
free function: it is checked here without a server, a request, or a settings
object.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repowise.docs.transport.auth import (
    MUTATION_TOKEN_HEADER,
    authorize_mutation_request,
    authorize_mutation_response,
    mutation_token_from_request,
)


def _request(headers: dict[str, str]) -> SimpleNamespace:
    # Starlette lower-cases header lookups; a plain dict with the exact casing
    # the tests send is enough for the policy's ``.get`` calls.
    return SimpleNamespace(headers=headers)


def _body(response) -> dict:
    return json.loads(bytes(response.body))


def test_a_configured_token_must_match() -> None:
    assert (
        authorize_mutation_response(
            webhook_token="secret",
            allow_unauthenticated=False,
            provided_token="secret",
        )
        is None
    )


@pytest.mark.parametrize("provided", [None, "", "nope", "secre", "secrets"])
def test_a_configured_token_rejects_everything_else(provided: str | None) -> None:
    response = authorize_mutation_response(
        webhook_token="secret",
        allow_unauthenticated=False,
        provided_token=provided,
    )
    assert response is not None
    assert response.status_code == 401
    assert _body(response)["error"] == "Invalid or missing auth token"


def test_a_configured_token_wins_over_the_unauthenticated_opt_in() -> None:
    """Setting both must not silently open the endpoint."""
    response = authorize_mutation_response(
        webhook_token="secret",
        allow_unauthenticated=True,
        provided_token=None,
    )
    assert response is not None
    assert response.status_code == 401


def test_no_token_and_no_opt_in_fails_closed() -> None:
    response = authorize_mutation_response(
        webhook_token=None,
        allow_unauthenticated=False,
        provided_token=None,
    )
    assert response is not None
    assert response.status_code == 401
    payload = _body(response)
    assert payload["error"] == "unauthenticated_mutation_refused"
    # The refusal must name the docs-lane knobs, not the retired code-search ones.
    assert "DOC_SEARCH_WEBHOOK_TOKEN" in payload["error_description"]
    assert "X-Doc-Search-Token" in payload["error_description"]
    assert "DOC_SEARCH_ALLOW_UNAUTHENTICATED" in payload["error_description"]
    assert "CODE_SEARCH" not in payload["error_description"]


def test_no_token_with_the_opt_in_allows() -> None:
    assert (
        authorize_mutation_response(
            webhook_token="",
            allow_unauthenticated=True,
            provided_token=None,
        )
        is None
    )


def test_the_canonical_header_is_preferred_over_the_legacy_one() -> None:
    assert mutation_token_from_request(_request({MUTATION_TOKEN_HEADER: "new"})) == "new"
    assert mutation_token_from_request(_request({"X-Code-Search-Token": "old"})) == "old"
    assert (
        mutation_token_from_request(
            _request({MUTATION_TOKEN_HEADER: "new", "X-Code-Search-Token": "old"})
        )
        == "new"
    )
    assert mutation_token_from_request(_request({})) is None


def test_settings_without_the_auth_fields_fail_closed() -> None:
    """A settings double from before this change must not open the endpoint."""
    response = authorize_mutation_request(_request({}), SimpleNamespace())
    assert response is not None
    assert response.status_code == 401
    assert _body(response)["error"] == "unauthenticated_mutation_refused"


def test_an_empty_configured_token_is_treated_as_unset() -> None:
    """Otherwise an empty DOC_SEARCH_WEBHOOK_TOKEN would match an empty header."""
    settings = SimpleNamespace(webhook_token="", allow_unauthenticated=True)
    assert authorize_mutation_request(_request({MUTATION_TOKEN_HEADER: ""}), settings) is None
