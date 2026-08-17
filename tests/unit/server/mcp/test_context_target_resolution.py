"""get_context accepts the same target forms as get_symbol, and says so on a miss.

Two ergonomics gaps the adoption trial hit: a qualified name resolved in
get_symbol but not here, and a mistyped path returned an error with nothing to
retry — the point where an agent abandons the tool and starts globbing.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from repowise.core.persistence.models import Repository, WikiSymbol


async def _add_second_login(session):
    """A second ``login`` so the bare name is genuinely ambiguous."""
    repo = (await session.execute(select(Repository))).scalars().first()
    session.add(
        WikiSymbol(
            id="sym4",
            repository_id=repo.id,
            file_path="src/db/models.py",
            symbol_id="src/db/models.py::login",
            name="login",
            qualified_name="db.models.User.login",
            kind="method",
            signature="def login(self)",
            start_line=20,
            end_line=25,
            language="python",
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_qualified_name_target_resolves(setup_mcp):
    from repowise.server.mcp_server import get_context

    result = await get_context(["AuthService.login"])
    card = result["targets"]["AuthService.login"]

    assert card.get("error") is None
    assert card["type"] == "symbol"
    assert card["docs"]["name"] == "login"
    assert card["docs"]["file_path"] == "src/auth/service.py"


@pytest.mark.asyncio
async def test_qualified_name_resolves_in_any_separator_style(setup_mcp):
    from repowise.server.mcp_server import get_context

    result = await get_context(["AuthService::login", "auth.service.AuthService.login"])
    for target in ("AuthService::login", "auth.service.AuthService.login"):
        card = result["targets"][target]
        assert card.get("error") is None, target
        assert card["docs"]["name"] == "login", target


@pytest.mark.asyncio
async def test_symbol_id_target_still_resolves(setup_mcp):
    from repowise.server.mcp_server import get_context

    result = await get_context(["src/auth/service.py::AuthService"])
    card = result["targets"]["src/auth/service.py::AuthService"]

    assert card["type"] == "symbol"
    assert card["docs"]["name"] == "AuthService"


@pytest.mark.asyncio
async def test_ambiguous_name_is_labelled_not_silently_picked(setup_mcp, session):
    from repowise.server.mcp_server import get_context

    await _add_second_login(session)
    card = (await get_context(["login"]))["targets"]["login"]

    assert card["status"] == "ambiguous"
    assert card["match_count"] == 2
    ids = {c["symbol_id"] for c in card["candidates"]}
    assert ids == {"src/auth/service.py::login", "src/db/models.py::login"}
    for candidate in card["candidates"]:
        assert set(candidate) >= {
            "symbol_id",
            "file",
            "name",
            "qualified_name",
            "kind",
            "start_line",
        }
    # The card still describes one of them, and says which — the caller keeps
    # a usable answer instead of only being told the question was ambiguous.
    assert card["docs"]["name"] == "login"
    assert "candidates" in card["note"]


@pytest.mark.asyncio
async def test_unambiguous_target_carries_no_ambiguity_keys(setup_mcp):
    from repowise.server.mcp_server import get_context

    card = (await get_context(["src/auth/service.py"]))["targets"]["src/auth/service.py"]

    assert "status" not in card
    assert "match_count" not in card


@pytest.mark.asyncio
async def test_mistyped_path_names_the_near_misses(setup_mcp):
    from repowise.server.mcp_server import get_context

    card = (await get_context(["src/auth/servce.py"]))["targets"]["src/auth/servce.py"]

    assert card["status"] == "not_found"
    assert "src/auth/service.py" in card["suggestions"]
    assert "suggestions" in card["error"]


@pytest.mark.asyncio
async def test_mistyped_path_in_the_wrong_directory_still_resolves_by_name(setup_mcp):
    from repowise.server.mcp_server import get_context

    card = (await get_context(["lib/servce.py"]))["targets"]["lib/servce.py"]

    assert card["status"] == "not_found"
    assert "src/auth/service.py" in card["suggestions"]


@pytest.mark.asyncio
async def test_a_path_resembling_nothing_says_that_rather_than_guessing(setup_mcp):
    from repowise.server.mcp_server import get_context

    target = "zzzqqq/wholly_unrelated.py"
    card = (await get_context([target]))["targets"][target]

    assert card["status"] == "not_found"
    assert "suggestions" not in card
    assert "no indexed path resembles it" in card["error"]


@pytest.mark.asyncio
async def test_module_targets_keep_their_opt_in_blocks(setup_mcp, session):
    """A module target has no file, and must not lose blocks to that.

    Pinned because the test-linkage block was briefly written inside the
    triage section and swallowed the governing-decisions lookup: file targets
    kept working, module targets silently lost ``decision_records``, and every
    existing test still passed.
    """
    from repowise.core.persistence.models import DecisionNodeLink
    from repowise.server.mcp_server import get_context

    session.add(
        DecisionNodeLink(
            id="dnl1",
            repository_id="repo1",
            decision_id="dec1",
            node_id="src/auth",
            link_type="module",
        )
    )
    await session.flush()

    card = (await get_context(["src/auth"], include=["decisions"]))["targets"]["src/auth"]

    assert card["type"] == "module"
    assert "Use JWT for authentication" in card["decision_records"]
    # No file behind a module, so no test-linkage claim is made either way.
    assert "tested" not in card
