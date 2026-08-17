"""get_symbol resolves the target forms an agent actually has.

An agent reading a call site has a bare name; one reading a stack trace has a
qualified one; only a response from repowise itself hands back a full
``path::Symbol`` id. Requiring the id it does not have yet costs a round trip
and, on a miss, its trust in the tool. All three forms resolve through the same
ladder, and a target matching several symbols is reported as ambiguous rather
than answered with one of them.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from repowise.core.persistence.models import Repository, WikiSymbol

PKG_SOURCE = '''"""Package module."""


class Gatekeeper:
    def signin(self, user):
        return user


def helper():
    return 1
'''

OTHER_SOURCE = '''"""Another module with the same leaf name."""


class AdminGate:
    def signin(self, user):
        return None
'''


@pytest.fixture
def repo_on_disk(tmp_path, monkeypatch):
    import repowise.server.mcp_server as mcp_mod

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "auth.py").write_text(PKG_SOURCE)
    (tmp_path / "pkg" / "admin.py").write_text(OTHER_SOURCE)
    monkeypatch.setattr(mcp_mod, "_repo_path", str(tmp_path))
    return tmp_path


async def _seed(session, rows):
    repo = (await session.execute(select(Repository))).scalars().first()
    for row in rows:
        session.add(WikiSymbol(repository_id=repo.id, language="python", **row))
    await session.flush()


GATE_SIGNIN = {
    "id": "aaa1",
    "file_path": "pkg/auth.py",
    "symbol_id": "pkg/auth.py::Gatekeeper::signin",
    "name": "signin",
    "qualified_name": "pkg.auth.Gatekeeper.signin",
    "kind": "method",
    "signature": "def signin(self, user)",
    "start_line": 5,
    "end_line": 6,
}
ADMIN_SIGNIN = {
    "id": "bbb1",
    "file_path": "pkg/admin.py",
    "symbol_id": "pkg/admin.py::AdminGate::signin",
    "name": "signin",
    "qualified_name": "pkg.admin.AdminGate.signin",
    "kind": "method",
    "signature": "def signin(self, user)",
    "start_line": 5,
    "end_line": 6,
}
HELPER = {
    "id": "ccc1",
    "file_path": "pkg/auth.py",
    "symbol_id": "pkg/auth.py::helper",
    "name": "helper",
    "qualified_name": "pkg.auth.helper",
    "kind": "function",
    "signature": "def helper()",
    "start_line": 9,
    "end_line": 10,
}


@pytest.mark.asyncio
async def test_bare_name_resolves_without_a_path(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [HELPER])
    result = await get_symbol("helper")

    assert result.get("error") is None
    assert result["symbol_id"] == "pkg/auth.py::helper"
    assert result["file"] == "pkg/auth.py"
    assert "return 1" in result["source"]


@pytest.mark.asyncio
async def test_qualified_name_resolves_in_every_separator_style(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN])
    for target in (
        "Gatekeeper.signin",
        "Gatekeeper::signin",
        "Gatekeeper/signin",
        "pkg.auth.Gatekeeper.signin",
    ):
        result = await get_symbol(target)
        assert result.get("error") is None, target
        assert result["symbol_id"] == "pkg/auth.py::Gatekeeper::signin", target


@pytest.mark.asyncio
async def test_exact_symbol_id_form_is_unchanged(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN])
    result = await get_symbol("pkg/auth.py::Gatekeeper::signin")

    assert result.get("error") is None
    assert result["symbol_id"] == "pkg/auth.py::Gatekeeper::signin"
    assert result["verified"] is True
    assert result["kind"] == "method"


@pytest.mark.asyncio
async def test_ambiguous_bare_name_returns_candidates_not_an_answer(
    setup_mcp, repo_on_disk, session
):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN, ADMIN_SIGNIN])
    result = await get_symbol("signin")

    assert result["status"] == "ambiguous"
    assert result["match_count"] == 2
    # No single answer was fabricated: the top level carries no body.
    assert "source" not in result
    ids = {c["symbol_id"] for c in result["candidates"]}
    assert ids == {"pkg/auth.py::Gatekeeper::signin", "pkg/admin.py::AdminGate::signin"}
    for candidate in result["candidates"]:
        assert set(candidate) >= {
            "symbol_id",
            "file",
            "name",
            "qualified_name",
            "kind",
            "start_line",
        }


@pytest.mark.asyncio
async def test_qualifying_an_ambiguous_name_disambiguates_it(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN, ADMIN_SIGNIN])
    result = await get_symbol("AdminGate.signin")

    assert result.get("status") != "ambiguous"
    assert result["symbol_id"] == "pkg/admin.py::AdminGate::signin"


@pytest.mark.asyncio
async def test_case_folding_is_a_fallback_never_a_first_choice(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    # Two symbols differing only in case: the exact one must win outright, so
    # the case-insensitive rung cannot turn a resolvable query into an
    # ambiguous one.
    await _seed(
        session,
        [
            HELPER,
            {**HELPER, "id": "ccc2", "symbol_id": "pkg/auth.py::HELPER", "name": "HELPER"},
        ],
    )
    exact = await get_symbol("helper")
    assert exact.get("status") != "ambiguous"
    assert exact["symbol_id"] == "pkg/auth.py::helper"

    shouted = await get_symbol("HELPER")
    assert shouted.get("status") != "ambiguous"
    assert shouted["symbol_id"] == "pkg/auth.py::HELPER"


@pytest.mark.asyncio
async def test_case_folded_match_resolves_when_nothing_matches_exactly(
    setup_mcp, repo_on_disk, session
):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN])
    result = await get_symbol("gatekeeper.SIGNIN")

    assert result.get("error") is None
    assert result["symbol_id"] == "pkg/auth.py::Gatekeeper::signin"


@pytest.mark.asyncio
async def test_unresolvable_name_names_no_symbol(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [HELPER])
    result = await get_symbol("definitely_not_indexed")

    assert "error" in result
    assert "symbol_id" not in result or result["symbol_id"] == "definitely_not_indexed"
    assert "source" not in result


@pytest.mark.asyncio
async def test_a_name_match_says_how_it_was_reached(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN])
    result = await get_symbol("Gatekeeper.signin")

    assert result["resolution"] == "qualified_suffix"
    assert result["resolved_from"] == "Gatekeeper.signin"

    folded = await get_symbol("gatekeeper.SIGNIN")
    assert folded["resolution"] == "qualified_suffix_ci"


@pytest.mark.asyncio
async def test_an_exact_id_says_nothing_extra(setup_mcp, repo_on_disk, session):
    from repowise.server.mcp_server import get_symbol

    await _seed(session, [GATE_SIGNIN])
    result = await get_symbol("pkg/auth.py::Gatekeeper::signin")

    assert "resolution" not in result
    assert "resolved_from" not in result


@pytest.mark.asyncio
async def test_qualified_suffix_is_case_sensitive_before_it_is_forgiving(
    setup_mcp, repo_on_disk, session
):
    """Two qualified names differing only in case must not collide.

    SQL ``LIKE`` is case-insensitive for ASCII on SQLite, so the suffix rung's
    query alone cannot tell ``Gatekeeper.signin`` from ``gatekeeper.SIGNIN``.
    Without the Python re-check the exact query would come back ambiguous.
    """
    await _seed(
        session,
        [
            GATE_SIGNIN,
            {
                **GATE_SIGNIN,
                "id": "aaa2",
                "symbol_id": "pkg/auth.py::gatekeeper::SIGNIN",
                "name": "SIGNIN",
                "qualified_name": "pkg.auth.gatekeeper.SIGNIN",
            },
        ],
    )

    from repowise.server.mcp_server import get_symbol

    exact = await get_symbol("Gatekeeper.signin")
    assert exact.get("status") != "ambiguous"
    assert exact["symbol_id"] == "pkg/auth.py::Gatekeeper::signin"

    other = await get_symbol("gatekeeper.SIGNIN")
    assert other.get("status") != "ambiguous"
    assert other["symbol_id"] == "pkg/auth.py::gatekeeper::SIGNIN"

    # Neither spelling exists exactly, so the case-folded rung runs and reports
    # both — the honest answer, and labelled as case-folded.
    folded = await get_symbol("GATEKEEPER.Signin")
    assert folded["status"] == "ambiguous"
    assert folded["resolution"].endswith("_ci")
