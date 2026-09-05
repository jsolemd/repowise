"""``manage_decision`` — the agent-callable surface over the decision journal.

Two properties carry the whole design and are asserted here directly rather
than through their effects: recording never confirms, and a journal that cannot
be written never looks like a successful write.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from repowise.core.analysis.decisions.journal import DECISIONS_JOURNAL_ENV
from repowise.core.persistence.database import init_db
from repowise.core.persistence.vector_store import InMemoryVectorStore
from repowise.core.providers.embedding.base import MockEmbedder
from tests.unit.server.test_mcp_workspace import _make_repo_context, _MockRegistry

JOURNAL_RELATIVE = ".repowise/decisions.jsonl"

_MCP_STATE = (
    "_registry",
    "_workspace_root",
    "_session_factory",
    "_fts",
    "_vector_store",
    "_decision_store",
    "_repo_path",
    "_vector_store_ready",
    "_cross_repo_enricher",
)


def _read_journal(root: Path) -> list[dict]:
    path = root / JOURNAL_RELATIVE
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.fixture
async def journal_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real on-disk repo with a file-backed store, wired into the MCP globals.

    File-backed rather than in-memory because one of the legs below writes to
    this same database from two separate processes.
    """
    import repowise.server.mcp_server as mcp_mod

    root = tmp_path / "repository"
    (root / ".repowise").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "replacement.py").write_text("VALUE = 2\n", encoding="utf-8")

    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, JOURNAL_RELATIVE)

    db_path = root / ".repowise" / "wiki.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    from repowise.core.persistence import upsert_repository

    async with factory() as session:
        repo = await upsert_repository(session, name="journal-repo", local_path=str(root))
        repo_id = repo.id
        await session.commit()

    mcp_mod._session_factory = factory
    mcp_mod._repo_path = str(root)
    mcp_mod._decision_store = InMemoryVectorStore(embedder=MockEmbedder())

    yield root, repo_id, db_path

    mcp_mod._session_factory = None
    mcp_mod._repo_path = None
    mcp_mod._decision_store = None
    mcp_mod._registry = None
    mcp_mod._workspace_root = None
    mcp_mod._embedder_status = None
    await engine.dispose()


def _install(contexts: dict, default: str, root: Path) -> _MockRegistry:
    import repowise.server.mcp_server as mcp_mod

    registry = _MockRegistry(contexts=contexts, default_alias=default, workspace_root=root)
    primary = contexts[default]
    mcp_mod._registry = registry
    mcp_mod._workspace_root = str(root)
    mcp_mod._session_factory = primary.session_factory
    mcp_mod._fts = primary.fts
    mcp_mod._vector_store = primary.vector_store
    mcp_mod._decision_store = primary.decision_store
    mcp_mod._repo_path = str(primary.path)
    mcp_mod._vector_store_ready = primary.vector_store_ready
    return registry


async def _uninstall(registry: _MockRegistry) -> None:
    import repowise.server.mcp_server as mcp_mod

    await registry.close()
    for attr in _MCP_STATE:
        setattr(mcp_mod, attr, None)


@pytest.fixture
async def decision_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two repo contexts whose journals live under different roots."""
    root = tmp_path / "workspace"
    repo_roots = {alias: root / alias for alias in ("a", "b")}
    for repo_root in repo_roots.values():
        anchor = repo_root / "src" / "service.py"
        anchor.parent.mkdir(parents=True)
        anchor.write_text("VALUE = 1\n", encoding="utf-8")

    contexts = {
        alias: await _make_repo_context(alias, str(repo_root), pages=[], extra_models=[])
        for alias, repo_root in repo_roots.items()
    }
    registry = _install(contexts, "a", root)
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, JOURNAL_RELATIVE)

    yield root, repo_roots["a"], repo_roots["b"]

    await _uninstall(registry)


async def _call(**kwargs):
    from repowise.server.mcp_server.tool_decisions import manage_decision

    return await manage_decision(**kwargs)


async def _record(title="Route reads through the projection", anchors=("src/service.py",), **kw):
    return await _call(
        action="record",
        title=title,
        decision="Read the canonical view through the journal projection.",
        why="Two readers disagreeing about decision state is worse than one slow read.",
        anchors=list(anchors),
        actor="test-agent",
        **kw,
    )


# ---------------------------------------------------------------------------
# Acceptance: record -> list -> confirm -> supersede -> get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_lands_proposed(journal_repo):
    result = await _record()

    assert "error" not in result, result
    assert result["recorded"] is True
    assert result["journal_created"] is True
    assert result["decision"]["status"] == "proposed"
    assert result["decision"]["confirmed"] is False
    assert result["decision"]["confirmed_at"] is None

    # The claim is about the file, not only the projection: an agent-recorded
    # row is unconfirmed *on disk*, which is what a reviewer reads in the diff.
    root, _, _ = journal_repo
    (written,) = _read_journal(root)
    assert written["confirmed_at"] is None
    assert written["status"] == "active"


@pytest.mark.asyncio
async def test_list_shows_the_proposal(journal_repo):
    recorded = await _record()
    listed = await _call(action="list")

    assert listed["total"] == 1
    assert listed["journal_available"] is True
    assert listed["journal_exists"] is True
    assert listed["repo"] == "default"
    (row,) = listed["decisions"]
    assert row["id"] == recorded["decision"]["id"]
    assert row["status"] == "proposed"

    filtered = await _call(action="list", status="proposed")
    assert filtered["total"] == 1
    assert (await _call(action="list", status="active"))["total"] == 0


@pytest.mark.asyncio
async def test_confirm_promotes_and_reanchors(journal_repo):
    recorded = await _record()
    decision_id = recorded["decision"]["id"]

    confirmed = await _call(action="confirm", decision_id=decision_id, actor="jon")

    assert confirmed["confirmed"] is True
    assert confirmed["decision"]["status"] == "active"
    assert confirmed["decision"]["confirmed_at"] is not None

    root, _, _ = journal_repo
    (written,) = _read_journal(root)
    assert written["confirmed_at"] is not None
    # Confirming re-hashes what the decision governs, so the anchor carries the
    # digest of the file as it stood at confirmation.
    assert written["anchors"][0]["file_sha"] is not None

    assert (await _call(action="list", status="active"))["total"] == 1
    assert (await _call(action="list", status="proposed"))["total"] == 0


@pytest.mark.asyncio
async def test_supersede_keeps_both_records_readable(journal_repo):
    first = (await _record(title="Original rule"))["decision"]["id"]
    await _call(action="confirm", decision_id=first, actor="jon")
    second = (await _record(title="Replacement rule", anchors=("src/replacement.py",)))["decision"][
        "id"
    ]

    superseded = await _call(
        action="supersede",
        decision_id=first,
        superseded_by=second,
        actor="jon",
    )

    assert superseded["superseded"] is True
    assert superseded["decision"]["status"] == "superseded"
    assert superseded["decision"]["superseded_by"] == second
    assert superseded["successor"]["supersedes"] == first

    # Nothing was deleted: both rows are still in the file, linked both ways.
    root, _, _ = journal_repo
    rows = {row["id"]: row for row in _read_journal(root)}
    assert set(rows) == {first, second}
    assert rows[first]["superseded_by"] == second
    assert rows[second]["supersedes"] == first

    got = await _call(action="get", decision_id=first)
    assert got["decision"]["id"] == first
    assert got["journal_exists"] is True
    assert got["repo"] == "default"
    assert [link["id"] for link in got["chain"]] == [first, second]

    # Asking from either end returns the same history.
    from_successor = await _call(action="get", decision_id=second)
    assert [link["id"] for link in from_successor["chain"]] == [first, second]


@pytest.mark.asyncio
async def test_record_can_retire_its_predecessor_in_one_write(journal_repo):
    """``record(supersedes=...)`` retires the old record in the same mutation.

    The two-step path leaves a window where both records are active and the
    store contradicts itself; this one has no such window. The successor is
    still a proposal — replacing a rule is not the same act as confirming the
    replacement.
    """
    first = (await _record(title="Original rule"))["decision"]["id"]
    await _call(action="confirm", decision_id=first, actor="jon")

    second = await _record(
        title="Replacement rule",
        anchors=("src/replacement.py",),
        supersedes=first,
    )

    assert second["decision"]["status"] == "proposed"
    assert second["decision"]["supersedes"] == first

    root, _, _ = journal_repo
    rows = {row["id"]: row for row in _read_journal(root)}
    assert rows[first]["status"] == "superseded"
    assert rows[first]["superseded_by"] == second["decision"]["id"]

    chain = await _call(action="get", decision_id=first)
    assert [link["id"] for link in chain["chain"]] == [first, second["decision"]["id"]]


@pytest.mark.asyncio
async def test_a_decision_cannot_supersede_itself(journal_repo):
    only = (await _record())["decision"]["id"]
    result = await _call(action="supersede", decision_id=only, superseded_by=only, actor="jon")
    assert "error" in result
    root, _, _ = journal_repo
    assert _read_journal(root)[0]["status"] == "active"


@pytest.mark.asyncio
async def test_later_record_does_not_claim_journal_creation(journal_repo):
    await _record()

    result = await _record(title="Second rule")

    assert result["journal_created"] is False


# ---------------------------------------------------------------------------
# Leg (a): a journal that is off or unwritable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["record", "confirm", "supersede"])
async def test_env_unset_disables_every_write_visibly(journal_repo, monkeypatch, verb):
    existing = (await _record())["decision"]["id"]
    monkeypatch.delenv(DECISIONS_JOURNAL_ENV)

    kwargs = {
        "record": {
            "action": "record",
            "title": "t",
            "decision": "d",
            "why": "w",
            "anchors": ["src/service.py"],
            "actor": "a",
        },
        "confirm": {"action": "confirm", "decision_id": existing, "actor": "a"},
        "supersede": {
            "action": "supersede",
            "decision_id": existing,
            "superseded_by": existing,
            "actor": "a",
        },
    }[verb]

    result = await _call(**kwargs)

    assert result["journal_available"] is False
    assert "error" in result and DECISIONS_JOURNAL_ENV in result["error"]
    assert result["guidance"]
    # The refusal is the whole point: nothing may report success, and nothing
    # may quietly land in the derived store instead.
    assert "recorded" not in result and "confirmed" not in result
    root, _, _ = journal_repo
    assert len(_read_journal(root)) == 1


@pytest.mark.asyncio
async def test_reads_still_serve_when_the_journal_is_off(journal_repo, monkeypatch):
    decision_id = (await _record())["decision"]["id"]
    monkeypatch.delenv(DECISIONS_JOURNAL_ENV)

    listed = await _call(action="list")
    assert listed["total"] == 1
    assert listed["journal_available"] is False
    assert "not configured" in listed["note"]

    got = await _call(action="get", decision_id=decision_id)
    assert got["decision"]["id"] == decision_id
    assert got["journal_exists"] is False
    assert got["repo"] == "default"
    assert "not configured" in got["note"]


@pytest.mark.asyncio
async def test_misconfigured_journal_path_disables_writes(journal_repo, monkeypatch):
    """An absolute or escaping path is a broken journal, not a missing one."""
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, "../elsewhere/decisions.jsonl")

    result = await _record()

    assert result["journal_available"] is False
    assert "error" in result


def test_journal_status_shields_a_misconfigured_path(tmp_path, monkeypatch):
    from repowise.server.services import decisions_ops as ops

    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, "../elsewhere/decisions.jsonl")

    assert ops.journal_status(tmp_path) == {
        "relative": None,
        "path": None,
        "exists": False,
    }


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
@pytest.mark.asyncio
async def test_unwritable_journal_disables_writes(journal_repo):
    root, _, _ = journal_repo
    await _record()
    journal_dir = root / ".repowise"
    original = journal_dir.stat().st_mode
    os.chmod(journal_dir, 0o500)
    try:
        result = await _record(title="Second rule")
    finally:
        os.chmod(journal_dir, original)

    assert result["journal_available"] is False
    assert "error" in result
    assert "recorded" not in result
    assert len(_read_journal(root)) == 1


# ---------------------------------------------------------------------------
# Leg (b): two processes recording at once
# ---------------------------------------------------------------------------


_CONCURRENT_WRITER = textwrap.dedent(
    """
    import asyncio, json, pathlib, sys, time
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from repowise.core.persistence.database import init_db
    from repowise.server.services import decisions_ops as ops

    root, db_path, repo_id, title, start_at = sys.argv[1:6]

    async def main():
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        await init_db(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        # Both processes aim at one wall-clock instant so the exclusive flock is
        # genuinely contended rather than serialised by interpreter startup.
        while time.time() < float(start_at):
            time.sleep(0.001)
        outcome = {"failure": None}
        try:
            async with factory() as session:
                await ops.record_decision(
                    session,
                    repo_id,
                    repo_root=pathlib.Path(root),
                    actor="writer",
                    title=title,
                    decision="d",
                    why="w",
                    anchors=["src/service.py"],
                )
                await session.commit()
        except Exception as exc:  # classified by the test, never swallowed
            outcome["failure"] = type(exc).__name__
        await engine.dispose()
        print(json.dumps(outcome))

    asyncio.run(main())
    """
)


@pytest.mark.asyncio
async def test_concurrent_records_from_two_processes_both_land(journal_repo, tmp_path):
    """Two processes recording at once: the journal takes both, intact.

    The claim under test is about the JSONL, which is the authority. Whichever
    process loses the race for the exclusive flock waits and then rewrites the
    file whole, so neither record is lost and no line is interleaved.

    A caveat this leg also pins down, deliberately rather than incidentally:
    the *projection* that follows each write is not serialised the way the
    write is. ``refresh_decision_journal`` reads the projected rows, then
    inserts the missing ones, with no lock spanning the two, so two processes
    projecting the same journal can collide on ``decision_records.id`` and one
    of them raises ``IntegrityError``. Its journal write is already durable
    when that happens, and the next read reprojects and heals — which is what
    the final assertion checks. The fix has a design choice in it (widen the
    flock over the projection, retry on IntegrityError, or make the projection
    an upsert) and belongs to whoever owns the projection, so this test states
    the behaviour rather than asserting the collision is fine.
    """
    import time

    root, repo_id, db_path = journal_repo
    script = tmp_path / "writer.py"
    script.write_text(_CONCURRENT_WRITER, encoding="utf-8")

    env = {
        **os.environ,
        DECISIONS_JOURNAL_ENV: JOURNAL_RELATIVE,
        "DO_NOT_TRACK": "1",
        "REPOWISE_TELEMETRY_DISABLED": "1",
    }
    start_at = str(time.time() + 3.0)
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(root), str(db_path), repo_id, title, start_at],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for title in ("Writer one rule", "Writer two rule")
    ]
    failures = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=180)
        assert proc.returncode == 0, stderr.decode()
        failures.append(json.loads(stdout.decode().strip().splitlines()[-1])["failure"])

    rows = _read_journal(root)
    assert len(rows) == 2, rows
    assert {row["title"] for row in rows} == {"Writer one rule", "Writer two rule"}
    # Distinct ids, both unconfirmed, every line parses — the file was replaced
    # whole under the lock rather than interleaved.
    assert len({row["id"] for row in rows}) == 2
    assert all(row["confirmed_at"] is None for row in rows)

    # No process may fail in a way that touches the journal contract itself.
    assert all(failure in (None, "IntegrityError") for failure in failures), failures

    listed = await _call(action="list")
    assert listed["total"] == 2


# ---------------------------------------------------------------------------
# Leg (c): the journal edited outside RepoWise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_edit_is_reconciled_on_the_next_read(journal_repo):
    root, _, _ = journal_repo
    mine = (await _record())["decision"]["id"]

    # A teammate's commit, a merge, a hand edit — the file changes with no call
    # through this surface at all.
    external = {
        "anchors": [{"file": "src/replacement.py", "symbol": None, "file_sha": None}],
        "confirmed_at": "2026-08-01T00:00:00Z",
        "decision": "Landed by someone else entirely.",
        "id": "dec-abcdef01",
        "recorded_at": "2026-08-01T00:00:00Z",
        "status": "active",
        "superseded_by": None,
        "supersedes": None,
        "title": "Recorded outside RepoWise",
        "why": "Because the file is the authority, not the database.",
    }
    path = root / JOURNAL_RELATIVE
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(external, sort_keys=True, separators=(",", ":")) + "\n")

    listed = await _call(action="list")

    assert listed["total"] == 2
    by_id = {row["id"]: row for row in listed["decisions"]}
    assert set(by_id) == {mine, "dec-abcdef01"}
    assert by_id["dec-abcdef01"]["status"] == "active"
    assert by_id[mine]["status"] == "proposed"

    got = await _call(action="get", decision_id="dec-abcdef01")
    assert got["decision"]["title"] == "Recorded outside RepoWise"

    # A row deleted from the file leaves the canonical view on the next read.
    path.write_text(
        json.dumps(external, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    assert (await _call(action="list"))["total"] == 1


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_action_names_the_vocabulary(journal_repo):
    result = await _call(action="delete", decision_id="dec-00000001")
    assert "Unknown action" in result["error"]
    for verb in ("record", "list", "get", "confirm", "supersede"):
        assert verb in result["guidance"]


@pytest.mark.asyncio
async def test_unknown_status_is_named_not_applied(journal_repo):
    await _record()
    result = await _call(action="list", status="propsed")

    # Dropped, not raised: the caller still gets the rows it could have had,
    # and the typo is spelled out rather than reported as an empty result.
    assert result["total"] == 1
    assert result["ignored_arguments"][0]["argument"] == "status"
    assert result["ignored_arguments"][0]["values"] == ["propsed"]


@pytest.mark.asyncio
async def test_an_oversized_limit_is_clamped_out_loud(journal_repo):
    await _record()
    result = await _call(action="list", limit=500)
    assert result["total"] == 1
    assert "clamped to 200" in result["limit_note"]

    assert "limit_note" not in await _call(action="list", limit=10)


@pytest.mark.asyncio
async def test_unparseable_time_bound_is_an_error(journal_repo):
    result = await _call(action="list", recorded_after="last tuesday")
    assert "recorded_after" in result["error"]


@pytest.mark.asyncio
async def test_time_range_filters(journal_repo):
    await _record()
    assert (await _call(action="list", recorded_after="2020-01-01"))["total"] == 1
    assert (await _call(action="list", recorded_before="2020-01-01"))["total"] == 0
    inverted = await _call(action="list", recorded_after="2026-01-01", recorded_before="2020-01-01")
    assert "no decision can match" in inverted["error"]


@pytest.mark.parametrize("timezone", ["America/Los_Angeles", "UTC"])
@pytest.mark.asyncio
async def test_recorded_at_round_trips_as_utc_and_bounds_match_storage(
    journal_repo, monkeypatch: pytest.MonkeyPatch, timezone: str
):
    """SQLite's naive round-trip must not reinterpret UTC as host-local time."""
    previous_timezone = os.environ.get("TZ")
    monkeypatch.setenv("TZ", timezone)
    time.tzset()
    try:
        recorded = await _record()
        root, _, _ = journal_repo
        (journal_row,) = _read_journal(root)
        stamp = journal_row["recorded_at"]

        got = await _call(action="get", decision_id=recorded["decision"]["id"])
        assert got["decision"]["recorded_at"] == stamp

        after = await _call(action="list", recorded_after=stamp)
        assert [row["id"] for row in after["decisions"]] == [recorded["decision"]["id"]]

        one_second_earlier = (
            (datetime.fromisoformat(stamp.replace("Z", "+00:00")) - timedelta(seconds=1))
            .isoformat()
            .replace("+00:00", "Z")
        )
        assert (await _call(action="list", recorded_before=one_second_earlier))["total"] == 0
    finally:
        if previous_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_timezone
        time.tzset()


@pytest.mark.asyncio
async def test_free_text_matches_title_decision_and_why(journal_repo):
    await _record(title="Adopt the projection")
    assert (await _call(action="list", query="PROJECTION"))["total"] == 1
    assert (await _call(action="list", query="disagreeing"))["total"] == 1
    assert (await _call(action="list", query="nothing here"))["total"] == 0
    # LIKE metacharacters are escaped, not honoured.
    assert (await _call(action="list", query="%"))["total"] == 0


@pytest.mark.asyncio
async def test_record_requires_an_anchor_and_an_actor(journal_repo):
    missing_anchor = await _call(
        action="record", title="t", decision="d", why="w", anchors=[], actor="a"
    )
    assert "anchor" in missing_anchor["error"]

    missing_actor = await _call(
        action="record", title="t", decision="d", why="w", anchors=["src/service.py"], actor=""
    )
    assert "actor" in missing_actor["error"]


@pytest.mark.asyncio
async def test_anchor_symbol_syntax(journal_repo):
    result = await _record(anchors=("src/service.py::VALUE",))
    assert result["decision"]["anchors"][0]["symbol"] == "VALUE"

    missing = await _record(anchors=("src/service.py::",))
    assert "::" in missing["error"]


@pytest.mark.asyncio
async def test_anchor_must_exist_on_disk(journal_repo):
    result = await _record(anchors=("src/imaginary.py",))
    assert "error" in result
    root, _, _ = journal_repo
    assert _read_journal(root) == []


@pytest.mark.asyncio
async def test_get_reports_a_missing_id(journal_repo):
    result = await _call(action="get", decision_id="dec-00000001")
    assert "dec-00000001" in result["error"]


# ---------------------------------------------------------------------------
# Governance readers: a proposal is visible where it is labeled, and absent
# where it would count as a rule (the lead's condition 2 on the proposed state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_proposal_never_reaches_answer_context_until_confirmed(journal_repo):
    """The self-reinforcement loop: an agent must not cite its own proposal.

    ``fetch_relevant_decisions`` feeds the decision block ``get_answer``
    injects. In journal mode a ``proposed`` row is an agent's unratified text,
    so it must be excluded there until a person confirms it — while remaining
    visible, labeled, in the status counts.
    """
    from types import SimpleNamespace

    import repowise.server.mcp_server as mcp_mod
    from repowise.core.persistence.crud.decisions import count_decisions_by_status
    from repowise.server.mcp_server._answer_context import fetch_relevant_decisions

    _root, repo_id, _db = journal_repo
    recorded = await _record()
    decision_id = recorded["decision"]["id"]
    await _call(action="list")  # runs the journal projection

    ctx = SimpleNamespace(session_factory=mcp_mod._session_factory)

    injected = await fetch_relevant_decisions(ctx, repo_id, ["src/service.py"])
    assert injected == [], "unconfirmed proposal leaked into answer context"

    # Labeled surfaces still see it — as proposed, never as active.
    async with mcp_mod._session_factory() as session:
        counts = await count_decisions_by_status(session, repo_id)
    assert counts["proposed"] == 1
    assert counts["active"] == 0

    await _call(action="confirm", decision_id=decision_id, actor="reviewer")
    await _call(action="list")

    injected = await fetch_relevant_decisions(ctx, repo_id, ["src/service.py"])
    assert [d["title"] for d in injected] == ["Route reads through the projection"]
    # v0.48.0 replaced the injected ``status`` with a lane, because the row is
    # rendered verbatim into a model prompt. A confirmed journal decision must
    # land in the accepted lane: the mirror of the trap upstream wrote the lane
    # for — a ratified decision that reads to the model as an open candidate.
    assert injected[0]["lane"] == "accepted"
    assert injected[0]["currency"] is not None


# ---------------------------------------------------------------------------
# Workspace routing: each repo owns its journal, and only list fans out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_list_discloses_a_missing_repo_journal(decision_workspace):
    _root, _a_root, _b_root = decision_workspace

    result = await _call(action="list", repo="b")

    assert result["repo"] == "b"
    assert result["journal_available"] is True
    assert result["journal_exists"] is False
    assert result["total"] == 0
    assert JOURNAL_RELATIVE in result["note"]
    assert "no recorded decisions" in result["note"]
    assert "record" in result["note"] and "create" in result["note"]


@pytest.mark.asyncio
async def test_workspace_record_creates_only_the_target_repo_journal(decision_workspace):
    _root, a_root, b_root = decision_workspace
    await _record(repo="a")
    a_journal = a_root / JOURNAL_RELATIVE
    b_journal = b_root / JOURNAL_RELATIVE
    cwd_journal = Path.cwd() / JOURNAL_RELATIVE
    a_before = a_journal.read_bytes()
    cwd_before = cwd_journal.read_bytes() if cwd_journal.exists() else None

    result = await _record(repo="b")

    assert result["journal_created"] is True
    assert result["decision"]["status"] == "proposed"
    assert result["decision"]["confirmed_at"] is None
    assert b_journal.is_file()
    (written,) = _read_journal(b_root)
    assert written["confirmed_at"] is None
    assert a_journal.read_bytes() == a_before
    assert (cwd_journal.read_bytes() if cwd_journal.exists() else None) == cwd_before

    got = await _call(action="get", repo="b", decision_id=result["decision"]["id"])
    assert got["repo"] == "b"
    assert got["journal_exists"] is True


@pytest.mark.asyncio
async def test_workspace_list_all_pages_each_repo_and_tags_rows(decision_workspace):
    _root, _a_root, _b_root = decision_workspace
    await _record(repo="a")
    a_list = await _call(action="list", repo="a")
    b_list = await _call(action="list", repo="b")

    result = await _call(action="list", repo="all")

    assert result["repo"] == "all"
    assert result["total"] == a_list["total"] + b_list["total"]
    assert result["returned"] == len(result["decisions"])
    assert list(result["per_repo"]) == ["a", "b"]
    assert result["per_repo"]["a"]["journal_exists"] is True
    assert result["per_repo"]["b"]["journal_exists"] is False
    assert all(row["repo"] in {"a", "b"} for row in result["decisions"])
    assert result["paging"] == "limit and offset apply within each repo"


@pytest.mark.asyncio
async def test_workspace_list_all_applies_the_limit_within_each_repo(decision_workspace):
    await _record(repo="a")
    await _record(repo="b")

    result = await _call(action="list", repo="all", limit=1)

    assert result["total"] == 2
    assert result["returned"] == 2
    assert result["per_repo"]["a"]["returned"] == 1
    assert result["per_repo"]["b"]["returned"] == 1


@pytest.mark.asyncio
async def test_workspace_get_and_writes_still_refuse_repo_all(decision_workspace):
    root, _a_root, _b_root = decision_workspace

    recorded = await _record(repo="all")
    got = await _call(action="get", repo="all", decision_id="dec-00000001")

    for result in (recorded, got):
        assert "error" in result
        assert "repo='all'" in result["error"]
        assert "a" in result["error"] and "b" in result["error"]
        assert "recorded" not in result
    assert list(root.rglob("decisions.jsonl")) == []


@pytest.mark.asyncio
async def test_workspace_unknown_alias_is_a_readable_error(decision_workspace):
    result = await _call(action="list", repo="nope")

    assert "Unknown repo 'nope'" in result["error"]
    assert "a" in result["error"] and "b" in result["error"]
