"""Naming a served row's nested definitions from the reference-site index.

The coordinator can only name a nested symbol that retrieval surfaced as a
rival for the same file (``test_coordinator.py``). The residue is a file window,
which names no symbol at all, and an enclosing chunk whose helper never entered
the candidate set. Both are spans the reference-site index can describe, and
this is where that description is attached — at the wiring seam, because the
coordinator holds no database handle and must not acquire one.

What is under test here is the *honesty* of that attachment far more than its
happy path. Reference sites are replaced repo-wide while chunks update
incrementally, so the two indexes can sit at different generations. Every test
below that deletes a row, empties a table, or dirties the working tree is
asserting the same thing from a different side: the enrichment adds a name it
can prove and otherwise says nothing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import delete

from repowise.core.persistence.database import get_session
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.schema import ReferenceSite
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import ReferenceKind
from repowise.core.source_search.worktree import WorkingTreeDivergence
from repowise.server import source_search_wiring as w

# ---------------------------------------------------------------------------
# A repository whose only interesting property is that it nests
# ---------------------------------------------------------------------------

DIVERSIFY_PY = """\
def mmr_diversify(items, k):
    \"\"\"Greedy maximal-marginal-relevance selection.\"\"\"

    def _sim(left, right):
        return float(left == right)

    chosen = []
    for item in items:
        if all(_sim(item, other) < 0.5 for other in chosen):
            chosen.append(item)
    return chosen[:k]
"""

#: The file's own text is the snippet a real chunk would carry, so the
#: corroboration check is exercised against the same bytes the reader sees.
DIVERSIFY_PATH = "diversify.py"


@pytest.fixture
def nested_repo(tmp_path: Path) -> Path:
    (tmp_path / DIVERSIFY_PATH).write_text(DIVERSIFY_PY)
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
async def extracted(async_session, repo_id, nested_repo):
    """Extract the nesting repo into the store."""
    store = SqlReferenceSiteStore(async_session)
    await store.replace_repository(repo_id, extract_repository(nested_repo))
    await async_session.commit()
    return store


# ---------------------------------------------------------------------------
# The store method
# ---------------------------------------------------------------------------


async def test_definitions_in_range_finds_the_nested_definition(extracted, repo_id):
    """The whole file's span holds both the function and the helper inside it."""
    sites = await extracted.definitions_in_range(repo_id, DIVERSIFY_PATH, 1, 160)

    chains = {site.target_symbol_id for site in sites}
    assert f"{DIVERSIFY_PATH}::mmr_diversify" in chains
    assert f"{DIVERSIFY_PATH}::mmr_diversify::_sim" in chains


async def test_definitions_in_range_excludes_what_starts_outside_it(extracted, repo_id):
    """A range below the helper's declaration does not get to claim it."""
    sites = await extracted.definitions_in_range(repo_id, DIVERSIFY_PATH, 6, 12)

    assert all("_sim" not in (site.target_symbol_id or "") for site in sites)


async def test_definitions_in_range_returns_only_definitions(extracted, repo_id):
    """Calls to ``_sim`` sit in the same span and are not declarations of it."""
    sites = await extracted.definitions_in_range(repo_id, DIVERSIFY_PATH, 1, 160)

    assert sites
    assert {site.kind for site in sites} == {ReferenceKind.DEFINITION}


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class _Inner:
    """A coordinator that has already ranked; only its response matters here."""

    def __init__(self, response: dict) -> None:
        self._response = response

    async def search(self, *_args, **_kwargs) -> dict:
        return self._response


def _window_response(**overrides) -> dict:
    """A served file-window row: real lines, real snippet, and no symbol named."""
    row = {
        "file": DIVERSIFY_PATH,
        "target_path": DIVERSIFY_PATH,
        "name": DIVERSIFY_PATH,
        "kind": "file_window",
        "source": "file_window",
        "snippet": DIVERSIFY_PY,
        "relevance_score": 0.5,
        "evidence": {},
        "start_line": 1,
        "end_line": 160,
    }
    row.update(overrides)
    return {"results": [row], "mode": "hybrid", "confidence": "confident", "_meta": {}}


def _patch_status(monkeypatch, divergence: WorkingTreeDivergence) -> None:
    """Stand in for the index status read, which is not what these test."""
    from repowise.core.source_search import status as status_mod

    async def fake_inspect(*_args, **_kwargs):
        return SimpleNamespace(
            state="ready",
            generation_id="g1",
            generation_sequence=1,
            pending_updates=0,
            blocked_updates=0,
            building_updates=0,
            ready_updates=0,
            stale_files=(),
            working_tree=divergence,
            last_error=None,
            degraded=False,
        )

    monkeypatch.setattr(status_mod, "inspect_source_index", fake_inspect)


CLEAN = WorkingTreeDivergence(checked=True)


def _coordinator(response: dict, session_factory, repo: Path):
    return w._StatusCoordinator(_Inner(response), repo, object(), object(), session_factory)


@pytest.fixture
def session_factory(async_engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


async def test_a_window_row_is_named_from_the_reference_sites(
    monkeypatch, extracted, session_factory, nested_repo
):
    """The residue case: a window names no symbol, so the index names them for it."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(_window_response(), session_factory, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    assert response["results"][0]["contains_symbols"] == [
        "mmr_diversify",
        "mmr_diversify::_sim",
    ]


async def test_an_enclosing_row_is_named_when_no_rival_survived(
    monkeypatch, extracted, session_factory, nested_repo
):
    """A symbol chunk whose helper never reached the candidate set at all."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(
        _window_response(
            kind="function",
            name="mmr_diversify",
            source="symbol",
            symbol_path="mmr_diversify",
            end_line=12,
        ),
        session_factory,
        nested_repo,
    )

    response = await coordinator.search("how are near duplicates dropped")

    # Its own name is not something it "contains", and the helper is.
    assert response["results"][0]["contains_symbols"] == ["mmr_diversify::_sim"]


async def test_a_name_the_coordinator_already_found_is_left_alone(
    monkeypatch, extracted, session_factory, nested_repo
):
    """Enrichment adds; it never rewrites what retrieval already proved."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(
        _window_response(contains_symbols=["mmr_diversify::_sim"]),
        session_factory,
        nested_repo,
    )

    response = await coordinator.search("how are near duplicates dropped")

    assert response["results"][0]["contains_symbols"] == ["mmr_diversify::_sim"]


# -- the three withdrawals ---------------------------------------------------


async def test_deleting_the_definition_site_withdraws_the_name(
    monkeypatch, extracted, async_session, repo_id, session_factory, nested_repo
):
    """Leg 1. With the row gone, the enrichment has nothing it can prove.

    The response must lose the *name*, not gain a wrong one and not fail: this
    is the shape a repo-wide refsites replace leaves behind mid-flight, and the
    only honest answer to it is silence.
    """
    await async_session.execute(
        delete(ReferenceSite).where(
            ReferenceSite.repository_id == repo_id,
            ReferenceSite.target_symbol_id == f"{DIVERSIFY_PATH}::mmr_diversify::_sim",
        )
    )
    await async_session.commit()
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(_window_response(), session_factory, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    row = response["results"][0]
    assert "_sim" not in str(row.get("contains_symbols", []))
    # The row still serves, and the definition that IS recorded still names it.
    assert row["contains_symbols"] == ["mmr_diversify"]
    assert row["file"] == DIVERSIFY_PATH


async def test_a_file_with_no_sites_at_all_falls_back_rather_than_failing(
    monkeypatch, async_session, repo_id, session_factory, nested_repo
):
    """Leg 2. Nothing extracted for this repository: absence is not an error.

    Deliberately *without* the ``extracted`` fixture, so the table is empty —
    the state a repository sits in before any extraction pass has run.
    """
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(
        _window_response(contains_symbols=["mmr_diversify::_sim"]),
        session_factory,
        nested_repo,
    )

    response = await coordinator.search("how are near duplicates dropped")

    # Whatever the coordinator established survives; nothing is added.
    assert response["results"][0]["contains_symbols"] == ["mmr_diversify::_sim"]


async def test_no_sites_and_no_coordinator_name_serves_a_plain_row(
    monkeypatch, repo_id, session_factory, nested_repo
):
    """Leg 2, the other half: no name from either source is still a served row.

    Takes ``repo_id`` but not ``extracted``, so the repository exists and its
    site table is empty. Without the repository row this would pass for the
    wrong reason — the lookup failing rather than the query finding nothing.
    """
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(_window_response(), session_factory, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    row = response["results"][0]
    assert "contains_symbols" not in row
    assert row["file"] == DIVERSIFY_PATH


@pytest.mark.parametrize(
    "divergence",
    [
        pytest.param(
            WorkingTreeDivergence(checked=True, modified=(DIVERSIFY_PATH,)),
            id="modified",
        ),
        pytest.param(
            WorkingTreeDivergence(checked=True, deleted=(DIVERSIFY_PATH,)),
            id="deleted",
        ),
        pytest.param(
            WorkingTreeDivergence(checked=False, unavailable_reason="not_a_git_repository"),
            id="unchecked",
        ),
    ],
)
async def test_a_diverged_file_is_never_named_from_the_index(
    monkeypatch, extracted, session_factory, nested_repo, divergence
):
    """Leg 3. A file that moved, or a tree nobody checked, gets no assertion.

    The unchecked case is parametrised alongside the two flagged ones on
    purpose. "The check did not run" is not a finding of cleanliness, and
    treating it as one would read silence as evidence — the exact error the
    deleted-site leg forbids, arriving from the freshness side instead.
    """
    _patch_status(monkeypatch, divergence)
    coordinator = _coordinator(_window_response(), session_factory, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    assert "contains_symbols" not in response["results"][0]


# -- corroboration and fail-soft --------------------------------------------


async def test_a_name_the_served_snippet_does_not_show_is_not_asserted(
    monkeypatch, extracted, session_factory, nested_repo
):
    """A recorded definition the current chunk text cannot show goes unnamed.

    This is what makes the generation skew between the two indexes a coverage
    limit rather than a correctness risk: the snippet comes from the chunk
    generation being served, so a name that survives is one the reader can see.
    """
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(
        _window_response(snippet="def mmr_diversify(items, k):\n    ..."),
        session_factory,
        nested_repo,
    )

    response = await coordinator.search("how are near duplicates dropped")

    assert response["results"][0]["contains_symbols"] == ["mmr_diversify"]


async def test_a_substring_of_a_longer_identifier_does_not_corroborate(
    monkeypatch, extracted, session_factory, nested_repo
):
    """``_sim`` inside ``_similarity`` is a different name, not this one."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(
        _window_response(snippet="def mmr_diversify(items, k):\n    _similarity(a, b)"),
        session_factory,
        nested_repo,
    )

    response = await coordinator.search("how are near duplicates dropped")

    assert response["results"][0]["contains_symbols"] == ["mmr_diversify"]


async def test_a_database_that_cannot_be_read_does_not_fail_the_search(
    monkeypatch, extracted, nested_repo
):
    """Fail-soft, like every other enrichment at this seam."""
    _patch_status(monkeypatch, CLEAN)

    def exploding_factory():
        raise RuntimeError("no database")

    coordinator = _coordinator(_window_response(), exploding_factory, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    assert response["results"][0]["file"] == DIVERSIFY_PATH
    assert "contains_symbols" not in response["results"][0]


async def test_without_a_session_factory_the_seam_is_inert(monkeypatch, extracted, nested_repo):
    """A host that wires no database keeps exactly the coordinator's answer."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(_window_response(), None, nested_repo)

    response = await coordinator.search("how are near duplicates dropped")

    assert "contains_symbols" not in response["results"][0]


async def test_the_repository_id_is_resolved_once_per_coordinator(
    monkeypatch, extracted, session_factory, nested_repo
):
    """One repo per coordinator, so the lookup is a constant, not per query."""
    _patch_status(monkeypatch, CLEAN)
    coordinator = _coordinator(_window_response(), session_factory, nested_repo)
    calls = 0

    import repowise.server.mcp_server as mcp_mod

    real = mcp_mod._get_repo

    async def counting(session, repo=None):
        nonlocal calls
        calls += 1
        return await real(session, repo)

    monkeypatch.setattr(mcp_mod, "_get_repo", counting)

    await coordinator.search("q")
    await coordinator.search("q")

    assert calls == 1


async def test_get_session_is_not_opened_when_no_row_needs_naming(
    monkeypatch, extracted, session_factory, nested_repo
):
    """Every served row already named: the naming pass does not touch the database."""
    _patch_status(monkeypatch, CLEAN)
    opened = 0
    real_get_session = get_session

    def counting_get_session(factory):
        nonlocal opened
        opened += 1
        return real_get_session(factory)

    monkeypatch.setattr("repowise.core.persistence.database.get_session", counting_get_session)
    coordinator = _coordinator(
        _window_response(contains_symbols=["mmr_diversify::_sim"]),
        session_factory,
        nested_repo,
    )

    await coordinator.search("q")

    assert opened == 0
