"""End-to-end acceptance for the task-slice service on a real indexed repo.

Not a mocked graph: the fixture writes a small Python project to disk, runs
the real traverser / parser / graph builder over it, persists the result into
a real ``wiki.db``, and then builds, re-fetches, extends and renders slices
against those rows. Everything the service claims — entry-point resolution,
the two-layer walk, ranked membership, the three views, budget disclosure,
frontier-resumed extension — is asserted against that index.

The toy project is shaped to exercise the parts that are easy to get wrong:

* a call chain three hops deep (``main`` → ``AuthService.login`` →
  ``TokenStore.issue`` → ``User``) so depth actually matters;
* an upstream caller (``api.py``) that only a reverse hop can find;
* a test file, which must stay out by default;
* a JSON config and a nested ``index.py`` glue leaf, both of which name the
  task's words and neither of which may be chosen as where the task starts —
  that is the :mod:`repowise.core.entry_candidacy` rule doing its job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from repowise.core.persistence.database import create_engine, get_session, init_db
from repowise.core.persistence.models import Repository
from repowise.core.slices import (
    BudgetTooSmallError,
    SliceNotFoundError,
    SliceStore,
    WalkPolicy,
    build_slice,
    extend_slice,
    load_slice,
    nominate_entry_points,
    render_slice,
)
from repowise.core.slices.models import DOWNSTREAM

TASK = "fix the auth login token refresh"

_FILES: dict[str, str] = {
    "app/__init__.py": '"""Toy application package."""\n',
    "app/main.py": '''"""Application entry point."""

from app.service import AuthService


def main() -> int:
    """Start the application and perform one login."""
    service = AuthService()
    service.login("someone", "secret")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    "app/service.py": '''"""Authentication service — the subject of the toy task."""

from app.tokens import TokenStore


class AuthService:
    """Authenticates users and hands out session tokens."""

    def __init__(self) -> None:
        self.tokens = TokenStore()

    def login(self, username: str, password: str) -> str:
        """Authenticate a user and return a fresh session token."""
        return self.tokens.issue(username)

    def logout(self, token: str) -> None:
        """Drop a session token."""
        self.tokens.revoke(token)
''',
    "app/tokens.py": '''"""Token issue, refresh and revocation."""

from app.models import User


class TokenStore:
    """Holds session tokens in memory."""

    def __init__(self) -> None:
        self.issued: dict[str, str] = {}

    def issue(self, username: str) -> str:
        """Mint a token for a user."""
        user = User(username)
        token = f"tok-{user.display()}"
        self.issued[token] = username
        return token

    def refresh(self, token: str) -> str:
        """Exchange a token for a new one."""
        owner = self.issued.pop(token)
        return f"tok-{owner}-2"

    def revoke(self, token: str) -> None:
        """Forget a token."""
        self.issued.pop(token, None)
''',
    "app/models.py": '''"""Domain models."""

from app.storage import Registry


class User:
    """A named user."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.registry = Registry()

    def display(self) -> str:
        """Human-readable name."""
        return self.registry.canonical(self.name)
''',
    "app/storage.py": '''"""Name registry — three hops out from the service."""


class Registry:
    """Canonicalises user names."""

    def canonical(self, name: str) -> str:
        """Lower-case and strip a name."""
        return name.strip().lower()
''',
    "app/api.py": '''"""HTTP surface — reaches the service from above."""

from app.service import AuthService


def handle_login(username: str, password: str) -> str:
    """Route handler that logs a user in."""
    service = AuthService()
    return service.login(username, password)
''',
    "app/util/__init__.py": "",
    "app/util/index.py": '''"""Re-export shim — a glue leaf, never an execution start."""


def auth_login_helper(value: str) -> str:
    """Normalise an auth login value."""
    return value.strip().lower()
''',
    "app/auth_login.json": json.dumps({"login": {"token": {"refresh": 3600}}}, indent=2),
    "tests/test_service.py": '''"""Tests for the auth service."""

from app.service import AuthService


def test_login_returns_a_token() -> None:
    assert AuthService().login("a", "b").startswith("tok-")
''',
}


def _write_toy_repo(root: Path) -> Path:
    for rel, body in _FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


async def _index_toy_repo(root: Path, session: AsyncSession, repo_id: str) -> None:
    """Run the real ingestion over *root* and persist its graph."""
    from repowise.core.ingestion import ASTParser, FileTraverser, GraphBuilder
    from repowise.core.persistence import batch_upsert_graph_edges, batch_upsert_symbols
    from repowise.core.pipeline.persist import persist_graph_nodes

    traverser = FileTraverser(root)
    parser = ASTParser()
    builder = GraphBuilder()

    parsed_files = []
    source_map: dict[str, bytes] = {}
    for file_info in traverser.traverse():
        data = Path(file_info.abs_path).read_bytes()
        source_map[file_info.path] = data
        parsed = parser.parse_file(file_info, data)
        builder.add_file(parsed)
        parsed_files.append(parsed)
    builder.set_source_map(source_map)
    graph = builder.build()

    await persist_graph_nodes(session, repo_id, builder)

    edges = [
        {
            "source_node_id": u,
            "target_node_id": v,
            "imported_names_json": json.dumps(data.get("imported_names", [])),
            "edge_type": data.get("edge_type", "imports"),
            "confidence": data.get("confidence", 1.0),
            "hint_source": data.get("hint_source"),
            "resolution_origin": data.get("resolution_origin"),
        }
        for u, v, data in graph.edges(data=True)
    ]
    if edges:
        await batch_upsert_graph_edges(session, repo_id, edges)

    symbols = []
    for parsed in parsed_files:
        for sym in parsed.symbols:
            if not getattr(sym, "file_path", None):
                sym.file_path = parsed.file_info.path
            symbols.append(sym)
    if symbols:
        await batch_upsert_symbols(session, repo_id, symbols)
    await session.commit()


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory) -> Path:
    return _write_toy_repo(tmp_path_factory.mktemp("slice_toy_repo"))


@pytest.fixture
async def indexed(toy_root: Path):
    """``(session, repo_id, store, root)`` against a freshly indexed toy repo."""
    db_path = toy_root / ".repowise" / "wiki.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with get_session(factory) as session:
        repo = Repository(
            id="toyrepo",
            name="slice-toy",
            url="",
            local_path=str(toy_root),
            default_branch="main",
            settings_json="{}",
        )
        session.add(repo)
        await session.flush()
        await _index_toy_repo(toy_root, session, repo.id)

    store = SliceStore.open_default(toy_root)
    async with get_session(factory) as session:
        yield session, "toyrepo", store, toy_root
    store.close()
    await engine.dispose()
    for stale in (db_path, toy_root / ".repowise" / "slices" / "slices.db"):
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(stale) + suffix)
            if candidate.exists():
                candidate.unlink()


# ---------------------------------------------------------------------------
# The index itself — if these fail nothing below means anything
# ---------------------------------------------------------------------------


async def test_toy_repo_indexes_both_graph_layers(indexed) -> None:
    from sqlalchemy import func, select

    from repowise.core.persistence.models import GraphEdge, GraphNode

    session, repo_id, _store, _root = indexed
    files = await session.scalar(
        select(func.count())
        .select_from(GraphNode)
        .where(GraphNode.repository_id == repo_id, GraphNode.node_type == "file")
    )
    symbols = await session.scalar(
        select(func.count())
        .select_from(GraphNode)
        .where(GraphNode.repository_id == repo_id, GraphNode.node_type == "symbol")
    )
    edges = await session.scalar(
        select(func.count()).select_from(GraphEdge).where(GraphEdge.repository_id == repo_id)
    )
    assert files >= 8, "toy repo should index its Python files and its JSON config"
    assert symbols >= 10, "symbol layer must exist or the walk has only one layer"
    assert edges >= 10


# ---------------------------------------------------------------------------
# Entry-point resolution
# ---------------------------------------------------------------------------


async def test_entry_candidacy_demotes_config_and_glue_leaves(indexed) -> None:
    """A JSON config and a nested index.py name the task and still lose.

    Both carry every term the task does. ``entry_candidacy`` is what keeps
    them out of the top: a config file describes the system and a nested glue
    module dispatches within it, so neither is where a reader enters.
    """
    session, repo_id, _store, _root = indexed
    candidates = await nominate_entry_points(session, repo_id, TASK, limit=20)
    by_node = {c.node_id: c for c in candidates}

    assert candidates, "the task names real symbols; nomination must find them"
    assert candidates[0].language == "python"
    assert candidates[0].node_id not in {
        "app/auth_login.json",
        "app/util/index.py",
        "app/util/index.py::auth_login_helper",
    }

    config = by_node.get("app/auth_login.json")
    if config is not None:
        assert config.score < candidates[0].score
        assert any("not an execution start" in reason for reason in config.reasons)

    glue = by_node.get("app/util/index.py::auth_login_helper")
    if glue is not None:
        assert glue.score < candidates[0].score
        assert any("not an execution start" in reason for reason in glue.reasons)


async def test_explicit_entry_point_beats_the_query(indexed) -> None:
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py::AuthService::login"],
        policy=WalkPolicy(downstream_depth=1, upstream_depth=0),
    )
    assert record.seeds[0] == "app/service.py::AuthService::login"
    seed = record.member("app/service.py::AuthService::login")
    assert seed is not None and seed.is_seed
    assert any("requested explicitly" in r for r in seed.reasons)


async def test_exact_symbol_seed_does_not_promote_sibling_definitions(indexed) -> None:
    """An exact call-path seed spends the slice on its graph, not its filemates."""
    session, repo_id, store, root = indexed
    exact = "app/service.py::AuthService::login"
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task="change login and verify its callers and tests",
        store=store,
        entry_points=[exact],
        policy=WalkPolicy(
            downstream_depth=1,
            upstream_depth=1,
            include_tests=True,
            seed_symbol_fanout=50,
            max_members=30,
        ),
    )

    seeds = {member.node_id for member in record.members if member.is_seed}
    sibling_ids = {
        member.node_id
        for member in record.members
        if member.file_path == "app/service.py" and member.node_id != exact
    }

    assert seeds == {exact}
    assert "app/service.py" in {member.node_id for member in record.members}
    assert not (sibling_ids & seeds), sibling_ids
    assert any(
        member.file_path in {"app/api.py", "app/main.py", "tests/test_service.py"}
        for member in record.members
    ), "the exact seed should reach a caller/test before unrelated siblings"


async def test_unresolvable_entry_point_is_an_error_not_an_empty_slice(indexed) -> None:
    from repowise.core.slices import EntryPointsUnresolvedError

    session, repo_id, store, root = indexed
    with pytest.raises(EntryPointsUnresolvedError) as excinfo:
        await build_slice(
            session,
            repo_id=repo_id,
            repo_path=str(root),
            task=TASK,
            store=store,
            entry_points=["app/does_not_exist.py"],
        )
    assert excinfo.value.details()["unresolved"] == ["app/does_not_exist.py"]


# ---------------------------------------------------------------------------
# Build → walk → rank
# ---------------------------------------------------------------------------


@pytest.fixture
async def built(indexed):
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py"],
        policy=WalkPolicy(downstream_depth=2, upstream_depth=1),
    )
    return session, repo_id, store, root, record


async def test_build_reaches_downstream_and_upstream(built) -> None:
    _session, _repo_id, _store, _root, record = built
    node_ids = {m.node_id for m in record.members}

    assert re.fullmatch(r"sl_[0-9a-f]{12}", record.slice_id)
    assert "app/tokens.py" in node_ids, "one downstream hop from the seed file"
    assert "app/models.py" in node_ids, "two downstream hops"
    assert "app/storage.py" not in node_ids, "three hops is past this policy's depth"
    assert "app/api.py" in node_ids, "one upstream hop — nothing else finds this"
    assert "tests/test_service.py" not in node_ids, "tests stay out by default"
    assert not any(m.file_path.startswith("tests/") for m in record.members), (
        "no test symbol may ride in on the symbol layer either"
    )


async def test_every_member_says_why_it_is_here(built) -> None:
    _session, _repo_id, _store, _root, record = built
    for member in record.members:
        assert member.reasons, f"{member.node_id} entered the slice with no reason"
    downstream = record.member("app/tokens.py")
    assert downstream is not None
    assert any("reached by the slice" in r for r in downstream.reasons)
    upstream = record.member("app/api.py")
    assert upstream is not None
    assert any("reaches the slice" in r for r in upstream.reasons)


async def test_ranking_is_ordered_seeds_first_then_score(built) -> None:
    """The ordering contract, asserted directly.

    This is the test the ranking mutation leg breaks: shuffle ``rank_members``
    and all three of these fail.
    """
    _session, _repo_id, _store, _root, record = built
    members = record.members

    assert [m.rank for m in members] == list(range(1, len(members) + 1))

    seed_positions = [i for i, m in enumerate(members) if m.is_seed]
    reached_positions = [i for i, m in enumerate(members) if not m.is_seed]
    assert not reached_positions or max(seed_positions) < min(reached_positions)

    scores = [m.score for m in members if not m.is_seed]
    assert scores == sorted(scores, reverse=True)

    near = record.member("app/tokens.py")
    far = record.member("app/models.py")
    assert near is not None and far is not None
    assert near.score > far.score, "a closer member must outrank a further one"


async def test_slice_is_a_subgraph_not_a_bag(built) -> None:
    _session, _repo_id, _store, _root, record = built
    node_ids = {m.node_id for m in record.members}
    assert record.edges
    for edge in record.edges:
        assert edge.source in node_ids or edge.source.startswith(("external:", "framework:"))
        assert edge.target in node_ids or edge.target.startswith(("external:", "framework:"))


# ---------------------------------------------------------------------------
# Persistence: fetch by id, and the typed miss
# ---------------------------------------------------------------------------


async def test_slice_round_trips_through_the_store(built) -> None:
    _session, _repo_id, store, _root, record = built
    reloaded = load_slice(store, record.slice_id)

    assert reloaded.slice_id == record.slice_id
    assert reloaded.task == record.task
    assert [m.node_id for m in reloaded.members] == [m.node_id for m in record.members]
    assert [m.rank for m in reloaded.members] == [m.rank for m in record.members]
    assert reloaded.member("app/tokens.py").reasons == record.member("app/tokens.py").reasons
    assert reloaded.frontier(DOWNSTREAM) == record.frontier(DOWNSTREAM)


async def test_unknown_slice_id_raises_rather_than_returning_empty(built) -> None:
    """The mutation leg for missing resources: a bad id is never an empty success."""
    _session, _repo_id, store, _root, record = built
    with pytest.raises(SliceNotFoundError) as excinfo:
        load_slice(store, "sl_000000000000")
    details = excinfo.value.details()
    assert details["slice_id"] == "sl_000000000000"
    assert record.slice_id in details["recent_slice_ids"]


# ---------------------------------------------------------------------------
# Extension resumes from the frontier
# ---------------------------------------------------------------------------


async def test_extend_resumes_from_the_frontier_without_re_walking(indexed) -> None:
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py"],
        policy=WalkPolicy(downstream_depth=1, upstream_depth=0),
    )
    before = {m.node_id for m in record.members}
    frontier = record.frontier(DOWNSTREAM)
    assert frontier, "a depth-1 walk must leave a frontier to resume from"
    assert "app/tokens.py" in before, "one hop out"
    assert "app/models.py" not in before, "depth 1 cannot reach two hops out"

    extended, extension = await extend_slice(
        session, store=store, slice_id=record.slice_id, extra_downstream=1
    )
    after = {m.node_id for m in extended.members}

    assert extension["resumed_from"][DOWNSTREAM] == len(frontier)
    assert extension["re_walked"] is False
    assert extension["members_added"] == len(after - before)
    assert "app/models.py" in after, "the resumed hop reaches the next ring"
    assert before <= after, "extension adds; it never drops what was there"
    assert extended.revision == 2

    # Distances are still measured from the original seeds, not from the
    # frontier the extension resumed at.
    assert extended.member("app/tokens.py").distance == 1
    assert extended.member("app/models.py").distance == 2
    assert extended.member("app/models.py").added_revision == 2
    assert extended.member("app/tokens.py").added_revision == 1


async def test_extend_persists_and_is_visible_to_a_fresh_load(indexed) -> None:
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py"],
        policy=WalkPolicy(downstream_depth=1, upstream_depth=0),
    )
    extended, _ = await extend_slice(
        session, store=store, slice_id=record.slice_id, extra_downstream=1
    )
    reloaded = load_slice(store, record.slice_id)
    assert len(reloaded.members) == len(extended.members)
    assert reloaded.revision == 2
    kinds = [event["kind"] for event in reloaded.events]
    assert kinds == ["build", "extend"]


async def test_extend_can_fold_in_a_new_entry_point(indexed) -> None:
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py"],
        policy=WalkPolicy(downstream_depth=0, upstream_depth=0),
    )
    extended, extension = await extend_slice(
        session,
        store=store,
        slice_id=record.slice_id,
        extra_downstream=0,
        entry_points=["app/main.py"],
    )
    assert "app/main.py" in extension["seeds_added"]
    assert extended.member("app/main.py") is not None
    assert extended.member("app/main.py").is_seed


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


async def test_three_views_are_three_fidelities(built) -> None:
    session, _repo_id, _store, root, record = built

    card = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)
    skeleton = await render_slice(
        session, record, view="skeleton", budget_tokens=20000, repo_root=root
    )
    full = await render_slice(session, record, view="full", budget_tokens=60000, repo_root=root)

    for payload in (card, skeleton, full):
        assert payload["status"] == "ok"
        assert payload["budget"]["truncated"] is False, "budgets here are deliberately generous"

    assert card["budget"]["members_tokens"] < skeleton["budget"]["members_tokens"]
    assert skeleton["budget"]["members_tokens"] < full["budget"]["members_tokens"]

    def find(payload, node_id):
        return next(m for m in payload["members"] if m["node"] == node_id)

    assert "signature" not in find(card, "app/tokens.py")
    assert "source" not in find(card, "app/tokens.py")

    tokens_skeleton = find(skeleton, "app/tokens.py")
    assert [s["name"] for s in tokens_skeleton["defines"]]
    assert "source" not in tokens_skeleton

    tokens_full = find(full, "app/tokens.py")
    assert "class TokenStore" in tokens_full["source"]

    symbol_full = next(
        (m for m in full["members"] if m["node_type"] == "symbol" and m.get("source")), None
    )
    assert symbol_full is not None
    assert symbol_full["source_lines"] <= 200


async def test_full_view_caps_source_per_member_and_says_so(built) -> None:
    session, _repo_id, _store, root, record = built
    payload = await render_slice(
        session,
        record,
        view="full",
        budget_tokens=60000,
        max_source_lines=3,
        repo_root=root,
    )
    truncated = [m for m in payload["members"] if m.get("source_truncated")]
    assert truncated, "a 3-line cap must bite on a real file"
    for member in truncated:
        assert member["source_lines"] == 3
        assert member["source_lines_omitted"] > 0
        assert "max_source_lines=3" in member["source_truncation_note"]


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_shrinking_the_budget_drops_by_rank_and_discloses_it(built) -> None:
    """The budget mutation leg: silence here is the failure being tested for."""
    session, _repo_id, _store, root, record = built

    generous = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)
    assert generous["budget"]["truncated"] is False
    assert len(generous["members"]) == len(record.members)

    envelope = generous["budget"]["envelope_tokens"]
    tight = await render_slice(
        session, record, view="card", budget_tokens=envelope + 200, repo_root=root
    )

    budget = tight["budget"]
    assert budget["truncated"] is True
    assert budget["total_members"] == len(record.members)
    assert budget["included_members"] == len(tight["members"])
    assert budget["included_members"] < budget["total_members"]
    assert budget["dropped_members"] == len(budget["dropped"])
    assert budget["dropped_members"] > 0
    assert "were dropped" in budget["disclosure"]
    assert "budget.dropped" in budget["disclosure"]

    shown_ranks = [m["rank"] for m in tight["members"]]
    dropped_ranks = [d["rank"] for d in budget["dropped"]]
    assert shown_ranks == sorted(shown_ranks)
    assert max(shown_ranks) < min(dropped_ranks), "drops come from the bottom of the ranking"
    assert sorted(shown_ranks + dropped_ranks) == list(range(1, len(record.members) + 1))
    for entry in budget["dropped"]:
        assert entry["node"] and entry["tokens"] > 0


async def test_a_budget_too_small_for_one_member_is_an_error(built) -> None:
    session, _repo_id, _store, root, record = built
    with pytest.raises(BudgetTooSmallError) as excinfo:
        await render_slice(session, record, view="full", budget_tokens=1, repo_root=root)
    details = excinfo.value.details()
    assert details["minimum_budget_tokens"] > 1
    assert details["total_members"] == len(record.members)


async def test_the_member_cap_is_disclosed_rather_than_silent(indexed) -> None:
    """A walk that stops at its ceiling says so; the reader must not infer it."""
    session, repo_id, store, root = indexed
    record = await build_slice(
        session,
        repo_id=repo_id,
        repo_path=str(root),
        task=TASK,
        store=store,
        entry_points=["app/service.py"],
        policy=WalkPolicy(downstream_depth=3, upstream_depth=2, max_members=8),
    )
    assert record.member_cap_hit is True
    assert "app/storage.py" not in {m.node_id for m in record.members}, (
        "depth 3 would reach it; the cap is what stopped the walk"
    )

    payload = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)
    assert "max_members=8" in payload["walk_truncated"]
    assert payload["summary"]["member_cap_hit"] is True


async def test_a_generous_walk_reports_no_cap(built) -> None:
    session, _repo_id, _store, root, record = built
    assert record.member_cap_hit is False
    payload = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)
    assert "walk_truncated" not in payload


async def test_a_slice_outlives_its_index_and_says_which_members_did_not(built) -> None:
    """A stored slice survives a re-index; its vanished members come back labelled.

    Deleting graph rows under a slice is exactly what ``repowise update`` does
    to a renamed file. The slice id must keep working — losing it would throw
    away an agent's working set over a rename — and the members that no longer
    exist must be marked rather than handed back looking current.
    """
    from sqlalchemy import delete

    from repowise.core.persistence.models import GraphNode

    session, repo_id, _store, root, record = built
    doomed = "app/models.py"
    assert record.member(doomed) is not None

    await session.execute(
        delete(GraphNode).where(GraphNode.repository_id == repo_id, GraphNode.node_id == doomed)
    )
    await session.flush()

    payload = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)

    assert payload["status"] == "ok", "drift is a labelled result, not a failure"
    assert payload["summary"]["stale_members"] == 1
    assert "no longer in the index" in payload["index_drift"]

    stale = [m for m in payload["members"] if m.get("stale")]
    assert [m["node"] for m in stale] == [doomed]
    assert "revision" in stale[0]["stale_reason"]

    live = [m for m in payload["members"] if not m.get("stale")]
    assert live, "the rest of the slice is still usable"


async def test_no_drift_marker_when_the_index_still_matches(built) -> None:
    session, _repo_id, _store, root, record = built
    payload = await render_slice(session, record, view="card", budget_tokens=20000, repo_root=root)
    assert "index_drift" not in payload
    assert "stale_members" not in payload["summary"]
    assert not any(m.get("stale") for m in payload["members"])
