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


@pytest.mark.parametrize("count", [10, 11, 20, 21])
@pytest.mark.asyncio
async def test_substring_ambiguity_materializes_the_documented_candidate_cap(
    setup_mcp, session, count
):
    from repowise.server.mcp_server import get_context
    from repowise.server.mcp_server._symbol_lookup import MAX_AMBIGUITY_CANDIDATES

    for index in range(count):
        session.add(
            WikiSymbol(
                id=f"substring-boundary-{count}-{index}",
                repository_id="repo1",
                file_path=f"src/generated/needle_{index}.py",
                symbol_id=f"src/generated/needle_{index}.py::needle_match_{index}",
                name=f"needle_match_{index}",
                qualified_name=f"generated.needle_match_{index}",
                kind="function",
                signature=f"def needle_match_{index}()",
                start_line=1,
                end_line=2,
                language="python",
            )
        )
    await session.flush()

    card = (await get_context(["needle_match"]))["targets"]["needle_match"]

    listed = min(count, MAX_AMBIGUITY_CANDIDATES)
    assert card["status"] == "ambiguous"
    assert card["match_count"] == count
    assert len(card["candidates"]) == listed
    if count > MAX_AMBIGUITY_CANDIDATES:
        assert f"Only the first {listed} are listed" in card["note"]
    else:
        assert "Only the first" not in card["note"]


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
async def test_extensionless_path_never_crosses_into_the_symbol_namespace(
    setup_mcp, session, tmp_path, monkeypatch
):
    """A path miss must not become an unrelated same-tail symbol hit.

    Extensionless config files are ordinary path targets.  The slash is the
    caller's namespace choice; resolving ``infra/repowise/env`` as a symbol
    named ``ENV`` is more dangerous than returning no result because the card
    looks valid while describing unrelated code.
    """
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import get_context

    session.add(
        WikiSymbol(
            id="unrelated-env-symbol",
            repository_id="repo1",
            file_path="src/settings.py",
            symbol_id="src/settings.py::ENV",
            name="ENV",
            qualified_name="settings.ENV",
            kind="constant",
            signature="ENV = {}",
            start_line=1,
            end_line=1,
            language="python",
        )
    )
    await session.flush()

    live = tmp_path / "infra" / "repowise" / "env"
    live.parent.mkdir(parents=True)
    live.write_text("REPOWISE_SOURCE_SEARCH=1\n", encoding="utf-8")
    monkeypatch.setattr(mcp_mod, "_repo_path", str(tmp_path))

    live_card = (await get_context(["infra/repowise/env"]))["targets"]["infra/repowise/env"]
    missing_card = (await get_context(["infra/repowise/missing"]))["targets"][
        "infra/repowise/missing"
    ]

    assert live_card["type"] == "file"
    assert live_card["index_status"] == "live_file_without_index_record"
    assert live_card.get("docs", {}).get("name") != "ENV"
    assert missing_card["status"] == "not_found"
    assert missing_card.get("type") != "symbol"


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


@pytest.mark.asyncio
async def test_symbol_enrichments_use_the_canonical_resolved_target(
    setup_mcp, health_data, session, tmp_path, monkeypatch
):
    import repowise.server.mcp_server as mcp_mod
    from repowise.core.persistence.models import GraphEdge, GraphNode
    from repowise.server.mcp_server import get_context

    source_lines = ["# filler" for _ in range(100)]
    source_lines[9] = "class AuthService:"
    source_lines[19] = "    async def login(self, username, password):"
    source_lines[20] = "        return username"
    source_path = tmp_path / "src" / "auth" / "service.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("\n".join(source_lines) + "\n")
    monkeypatch.setattr(mcp_mod, "_repo_path", str(tmp_path))

    canonical = "src/auth/service.py::login"
    caller = "src/auth/caller.py::invoke"
    callee = "src/auth/token.py::issue"
    for node_id, name, file_path in (
        (canonical, "login", "src/auth/service.py"),
        (caller, "invoke", "src/auth/caller.py"),
        (callee, "issue", "src/auth/token.py"),
    ):
        session.add(
            GraphNode(
                id=f"enrich-{name}",
                repository_id="repo1",
                node_id=node_id,
                node_type="symbol",
                language="python",
                kind="method",
                name=name,
                qualified_name=name,
                file_path=file_path,
                start_line=20,
                end_line=21,
                pagerank=0.7,
                betweenness=0.2,
                community_id=1,
            )
        )
    session.add_all(
        [
            GraphEdge(
                id="enrich-caller-edge",
                repository_id="repo1",
                source_node_id=caller,
                target_node_id=canonical,
                edge_type="calls",
                confidence=1.0,
            ),
            GraphEdge(
                id="enrich-callee-edge",
                repository_id="repo1",
                source_node_id=canonical,
                target_node_id=callee,
                edge_type="calls",
                confidence=1.0,
            ),
        ]
    )
    await session.flush()

    include = ["callers", "callees", "metrics", "community", "health", "skeleton"]
    spellings = [
        canonical,
        "AuthService.login",
        "login",
        "auth.service::AuthService::login",
    ]
    cards = {
        target: (await get_context([target], include=include))["targets"][target]
        for target in spellings
    }

    for target in spellings[1:]:
        for key in include:
            assert cards[target][key] == cards[canonical][key], (target, key)
    assert cards[canonical]["callers"]
    assert cards[canonical]["callees"]
    assert cards[canonical]["metrics"] is not None
    assert cards[canonical]["community"] is not None
    assert cards[canonical]["health"] is not None
    assert cards[canonical]["skeleton"]["tokens"] > 0


@pytest.mark.asyncio
async def test_ambiguous_symbol_omits_unique_target_enrichment(setup_mcp, session):
    from repowise.server.mcp_server import get_context

    await _add_second_login(session)
    requested = ["callers", "callees", "metrics", "community", "health", "skeleton"]
    card = (await get_context(["login"], include=["docs", *requested]))["targets"]["login"]

    assert card["status"] == "ambiguous"
    assert card["docs"]["name"] == "login"
    assert card["enrichment_omitted"] == sorted(requested)
    assert all(key not in card for key in requested)
    assert "cannot be attributed" in card["note"]


@pytest.mark.asyncio
async def test_context_symbol_lookup_filters_exclusions_before_selecting_a_card(
    setup_mcp, session, tmp_path, monkeypatch
):
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import get_context

    config_dir = tmp_path / ".repowise"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("exclude_patterns:\n  - src/db/\n")
    monkeypatch.setattr(mcp_mod, "_repo_path", str(tmp_path))
    await _add_second_login(session)
    session.add(
        WikiSymbol(
            id="sym-db-only",
            repository_id="repo1",
            file_path="src/db/models.py",
            symbol_id="src/db/models.py::db_only",
            name="db_only",
            qualified_name="db.models.User.db_only",
            kind="method",
            signature="def db_only(self)",
            start_line=30,
            end_line=31,
            language="python",
        )
    )
    await session.flush()

    visible = (await get_context(["login"]))["targets"]["login"]
    excluded = (await get_context(["db.models.User.db_only"]))["targets"]["db.models.User.db_only"]

    assert visible.get("status") != "ambiguous"
    assert visible["docs"]["file_path"] == "src/auth/service.py"
    assert "error" in excluded
    assert "src/db/models.py" not in str(excluded)
