"""MCP surface for task slices.

The response *shape* is what these lock down, and one rule dominates: a
failure must never be mistakable for a result. An unknown slice id, an
unresolvable entry point and an impossible budget each come back with a
``status`` and an ``error`` and **no member list at all** — because a tool
that answers a typo with ``{"members": []}`` has told the agent its task
needs no code.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from repowise.server.mcp_server.tool_slices import (
    build_task_slice,
    extend_task_slice,
    get_task_slice,
)

TASK = "fix the auth login flow"


@pytest.fixture
async def slice_env(setup_mcp, tmp_path):
    """The MCP fixture DB, with the slice sidecar redirected into tmp_path."""
    import repowise.server.mcp_server as mcp_mod

    previous = mcp_mod._repo_path
    mcp_mod._repo_path = str(tmp_path)
    yield tmp_path
    mcp_mod._repo_path = previous


@pytest.fixture
async def built(slice_env):
    return await build_task_slice(task=TASK, entry_points=["src/auth/service.py"], view="card")


@pytest.fixture
async def symbol_slice(slice_env, factory, setup_mcp):
    from repowise.core.persistence.models import GraphNode

    async with factory() as session:
        session.add(
            GraphNode(
                repository_id=setup_mcp,
                node_id="src/auth/service.py::login",
                node_type="symbol",
                file_path="src/auth/service.py",
                name="login",
                kind="method",
                language="python",
                start_line=20,
                end_line=40,
                signature="async def login(self, username: str, password: str) -> Token",
            )
        )
        await session.commit()
    return await build_task_slice(
        task=TASK,
        entry_points=["src/auth/service.py::login"],
        view="card",
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_build_returns_a_ranked_slice_under_a_reusable_id(built) -> None:
    assert built["status"] == "ok"
    assert re.fullmatch(r"sl_[0-9a-f]{12}", built["slice_id"])
    assert built["task"] == TASK
    assert built["seeds"] == ["src/auth/service.py"]
    assert built["members"], "the seed alone is already a member"
    assert built["ranking"]["drop_order"] == "lowest rank first"
    assert built["summary"]["total_members"] == built["budget"]["total_members"]
    assert built["_meta"]["timing_ms"] >= 0


async def test_the_walk_reaches_both_directions(slice_env) -> None:
    payload = await build_task_slice(
        task=TASK,
        entry_points=["src/auth/service.py"],
        view="card",
        downstream_depth=1,
        upstream_depth=1,
    )
    nodes = {m["node"] for m in payload["members"]}
    assert "src/db/models.py" in nodes, "one hop along what the seed imports"
    assert "src/auth/middleware.py" in nodes, "one hop along what imports the seed"
    assert "tests/test_service.py" not in nodes, "tests stay out by default"


async def test_every_member_carries_its_reason_and_rank(built) -> None:
    for member in built["members"]:
        assert member["why"], f"{member['node']} has no stated reason"
        assert member["rank"] >= 1
    ranks = [m["rank"] for m in built["members"]]
    assert ranks == sorted(ranks)


async def test_the_slice_can_be_re_read_at_another_fidelity(built, slice_env) -> None:
    again = await get_task_slice(slice_id=built["slice_id"], view="skeleton")
    assert again["status"] == "ok"
    assert again["slice_id"] == built["slice_id"]
    assert again["summary"]["total_members"] == built["summary"]["total_members"]
    assert again["view"] == "skeleton"


async def test_extend_reports_what_it_resumed_from(built, slice_env) -> None:
    extended = await extend_task_slice(slice_id=built["slice_id"], extra_downstream=1)
    assert extended["status"] == "ok"
    assert extended["extension"]["re_walked"] is False
    assert extended["extension"]["revision"] == 2
    assert extended["summary"]["revision"] == 2
    assert set(extended["extension"]) >= {"resumed_from", "rounds", "members_added"}


# ---------------------------------------------------------------------------
# Failures are never empty successes
# ---------------------------------------------------------------------------


async def test_an_unknown_slice_id_is_a_typed_failure_with_no_member_list(built, slice_env) -> None:
    payload = await get_task_slice(slice_id="sl_deadbeef0000")

    assert payload["status"] == "slice_not_found"
    assert "members" not in payload, "a failure must not look like an empty slice"
    assert "budget" not in payload
    assert payload["slice_id"] == "sl_deadbeef0000"
    assert built["slice_id"] in payload["recent_slice_ids"]
    assert "sl_" in payload["error"]


async def test_an_unresolvable_entry_point_is_a_typed_failure(slice_env) -> None:
    payload = await build_task_slice(task=TASK, entry_points=["src/nope/missing.py"])

    assert payload["status"] == "entry_points_unresolved"
    assert "members" not in payload
    assert payload["unresolved"] == ["src/nope/missing.py"]


async def test_a_task_that_matches_nothing_is_a_typed_failure(slice_env) -> None:
    payload = await build_task_slice(task="zzzqqq wibblefrotz nonexistent")

    assert payload["status"] == "entry_points_unresolved"
    assert "members" not in payload
    assert payload["task"] == "zzzqqq wibblefrotz nonexistent"


async def test_extending_an_unknown_slice_is_a_typed_failure(slice_env) -> None:
    payload = await extend_task_slice(slice_id="sl_000000000000")
    assert payload["status"] == "slice_not_found"
    assert "members" not in payload


async def test_repo_all_is_refused_explicitly(slice_env) -> None:
    for payload in (
        await build_task_slice(task=TASK, repo="all"),
        await get_task_slice(slice_id="sl_000000000000", repo="all"),
        await extend_task_slice(slice_id="sl_000000000000", repo="all"),
    ):
        assert "members" not in payload
        assert "error" in payload


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_a_tight_budget_discloses_its_drops_and_makes_them_recoverable(
    built, slice_env
) -> None:
    tight = await get_task_slice(
        slice_id=built["slice_id"],
        view="card",
        budget_tokens=built["budget"]["envelope_tokens"] + 130,
    )

    budget = tight["budget"]
    assert budget["truncated"] is True
    assert budget["included_members"] < budget["total_members"]
    assert budget["dropped"], "a drop with no list is a silent drop"
    assert "budget.dropped" in budget["disclosure"]

    # The shared omission store makes the disclosure recoverable, not just visible.
    assert "omission_marker" in tight
    assert re.search(r"repowise#[0-9a-f]+", tight["omission_marker"])
    assert tight["_meta"]["omitted"]["refs"]
    assert "repowise expand" in tight["_meta"]["omitted"]["restore"]


async def test_a_budget_too_small_for_one_member_is_an_error_not_an_empty_slice(
    built, slice_env
) -> None:
    payload = await get_task_slice(slice_id=built["slice_id"], view="full", budget_tokens=1)

    # ``budget_tokens=1`` is clamped up to the floor, and the floor is still
    # far too small for a full-view member — so this is the error path.
    assert payload["status"] == "budget_too_small"
    assert "members" not in payload
    assert payload["minimum_budget_tokens"] > payload["budget_tokens"]


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


async def test_full_view_says_so_when_the_source_is_gone(built, slice_env) -> None:
    """The fixture index describes files that were never written to disk.

    That is exactly the drift a stale index produces, and the view has to name
    it rather than return a member with an empty ``source``.
    """
    payload = await get_task_slice(slice_id=built["slice_id"], view="full", budget_tokens=8000)
    assert payload["status"] == "ok"
    missing = [m for m in payload["members"] if "source_unavailable" in m]
    assert missing, "no member silently returned empty source"
    for member in missing:
        assert member["source"] is None
        assert member["file"] in member["source_unavailable"]


async def test_reread_uses_current_symbol_bounds_without_rewalking(
    symbol_slice, slice_env, factory
):
    """A persistent slice may outlive a symbol's location and signature."""
    from sqlalchemy import update

    from repowise.core.persistence.models import WikiSymbol
    from repowise.core.slices.store import SliceStore

    symbol_id = "src/auth/service.py::login"
    with SliceStore.open_default(slice_env) as store:
        original = store.load(symbol_slice["slice_id"])
    stored = next(member for member in original.members if member.node_id == symbol_id)
    assert stored.start_line == 20
    path = slice_env / "src/auth/service.py"
    path.parent.mkdir(parents=True)
    path.write_text("def other():\n    return 'unrelated'\n\ndef login(token):\n    return token\n")
    async with factory() as session:
        await session.execute(
            update(WikiSymbol)
            .where(WikiSymbol.symbol_id == symbol_id)
            .values(
                start_line=4,
                end_line=5,
                signature="def login(token)",
                docstring=None,
            )
        )
        await session.commit()

    payload = await get_task_slice(
        slice_id=symbol_slice["slice_id"], view="full", budget_tokens=8000
    )
    member = next(row for row in payload["members"] if row["node"] == symbol_id)
    assert member["source"] == "def login(token):\n    return token"
    assert member["source_first_line"] == 4
    assert member["lines"] == [4, 5]
    assert member["signature"] == "def login(token)"
    assert "doc" not in member
    assert member["rank"] == stored.rank
    with SliceStore.open_default(slice_env) as store:
        assert store.load(symbol_slice["slice_id"]).revision == original.revision


async def test_removed_symbol_does_not_return_another_symbols_body(
    symbol_slice, slice_env, factory
):
    from sqlalchemy import delete

    from repowise.core.persistence.models import WikiSymbol

    symbol_id = "src/auth/service.py::login"
    path = slice_env / "src/auth/service.py"
    path.parent.mkdir(parents=True)
    path.write_text("# preceding code\n" * 19 + "def unrelated():\n    return 1\n")
    async with factory() as session:
        await session.execute(delete(WikiSymbol).where(WikiSymbol.symbol_id == symbol_id))
        await session.commit()

    payload = await get_task_slice(
        slice_id=symbol_slice["slice_id"], view="full", budget_tokens=8000
    )
    member = next(row for row in payload["members"] if row["node"] == symbol_id)
    assert member["source"] is None
    assert "no longer" in member["source_unavailable"]


@pytest.mark.parametrize("change", ["shift", "remove"])
async def test_uncommitted_source_edits_cannot_serve_old_symbol_bounds(
    symbol_slice, slice_env, factory, setup_mcp, change
):
    from sqlalchemy import update

    from repowise.core.persistence.models import Repository

    path = slice_env / "src/auth/service.py"
    path.parent.mkdir(parents=True)
    path.write_text("# padding\n" * 19 + "def login():\n    return 1\n" + "# after\n" * 20)

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=slice_env, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("add", "src/auth/service.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "source")
    head = git("rev-parse", "HEAD")
    async with factory() as session:
        await session.execute(
            update(Repository)
            .where(Repository.id == setup_mcp)
            .values(local_path=str(slice_env), head_commit=head)
        )
        await session.commit()

    replacement = "def unrelated():\n    return 'wrong body'\n\n"
    live_symbol = "def login(token):\n    return token\n" if change == "shift" else ""
    path.write_text("# padding\n" * 19 + replacement + live_symbol + "# after\n" * 20)
    assert git("rev-parse", "HEAD") == head
    assert git("diff", "--name-only") == "src/auth/service.py"

    payload = await get_task_slice(
        slice_id=symbol_slice["slice_id"], view="full", budget_tokens=8000
    )
    assert payload["_meta"]["index_behind"] is False
    member = next(row for row in payload["members"] if row["node"] == "src/auth/service.py::login")
    if change == "shift":
        assert member["source"] == live_symbol.rstrip()
        assert member["lines"] == [23, 24]
        assert member["source_first_line"] == 23
    else:
        assert member["source"] is None
        assert "successful live parse" in member["source_unavailable"]
