"""The slice sidecar: what a stored slice id promises.

A slice id is a promise that the slice can be fetched back exactly as it was —
same members, same ranks, same reasons, same frontier. These check the promise
holds across a close/reopen, and that a broken promise (an id that names
nothing) is a typed failure rather than a plausible-looking empty slice.
"""

from __future__ import annotations

import re

import pytest

from repowise.core.slices.errors import SliceNotFoundError
from repowise.core.slices.models import SliceEdge, SliceMember, SliceRecord, WalkPolicy
from repowise.core.slices.store import SliceStore, default_store_path, new_slice_id


def _record(slice_id: str, *, task: str = "add a retry to the client") -> SliceRecord:
    seed = SliceMember(
        node_id="src/client.py",
        node_type="file",
        layer="file",
        file_path="src/client.py",
        distance=0,
        is_seed=True,
        seed_score=0.9,
        rank=1,
        score=0.9,
        reasons=["entry point (name contains 'client')"],
        frontier_down=False,
        frontier_up=True,
    )
    reached = SliceMember(
        node_id="src/retry.py::backoff",
        node_type="symbol",
        layer="symbol",
        file_path="src/retry.py",
        distance=1,
        name="backoff",
        kind="function",
        signature="def backoff(attempt: int) -> float",
        start_line=10,
        end_line=20,
        language="python",
        reference_count=3,
        max_confidence=0.95,
        query_hits=1,
        rank=2,
        score=0.61,
        reasons=["calls from src/client.py::send; reached by the slice at distance 1"],
        edge_types={"calls"},
        frontier_down=True,
        added_revision=2,
    )
    return SliceRecord(
        slice_id=slice_id,
        repository_id="repo1",
        repo_path="/tmp/does-not-need-to-exist",
        task=task,
        policy=WalkPolicy(downstream_depth=3, upstream_depth=1),
        seeds=["src/client.py"],
        members=[seed, reached],
        edges=[
            SliceEdge(
                source="src/client.py::send",
                target="src/retry.py::backoff",
                edge_type="calls",
                confidence=0.95,
                direction="downstream",
            )
        ],
        external_dependencies=["external:httpx"],
        revision=2,
    )


def test_slice_ids_have_a_recognisable_shape() -> None:
    assert re.fullmatch(r"sl_[0-9a-f]{12}", new_slice_id())
    assert new_slice_id() != new_slice_id()


def test_store_lives_beside_the_other_sidecars(tmp_path) -> None:
    assert default_store_path(tmp_path) == tmp_path / ".repowise" / "slices" / "slices.db"


def test_a_slice_survives_a_close_and_reopen(tmp_path) -> None:
    slice_id = new_slice_id()
    with SliceStore.open_default(tmp_path) as store:
        store.save(_record(slice_id), event={"kind": "build", "members_added": 2})

    with SliceStore.open_default(tmp_path) as store:
        loaded = store.load(slice_id)

    assert loaded.slice_id == slice_id
    assert loaded.task == "add a retry to the client"
    assert loaded.revision == 2
    assert loaded.seeds == ["src/client.py"]
    assert loaded.external_dependencies == ["external:httpx"]
    assert loaded.policy.downstream_depth == 3

    assert [m.node_id for m in loaded.members] == ["src/client.py", "src/retry.py::backoff"]
    assert [m.rank for m in loaded.members] == [1, 2]
    symbol = loaded.member("src/retry.py::backoff")
    assert symbol.reasons == ["calls from src/client.py::send; reached by the slice at distance 1"]
    assert symbol.edge_types == {"calls"}
    assert symbol.signature == "def backoff(attempt: int) -> float"
    assert symbol.added_revision == 2
    assert symbol.max_confidence == pytest.approx(0.95)

    assert len(loaded.edges) == 1
    assert loaded.edges[0].edge_type == "calls"


def test_the_frontier_survives_the_round_trip(tmp_path) -> None:
    """Without this an extension cannot know where the walk stopped."""
    slice_id = new_slice_id()
    with SliceStore.open_default(tmp_path) as store:
        store.save(_record(slice_id))
        loaded = store.load(slice_id)

    assert loaded.frontier("downstream") == {"src/retry.py::backoff": 1}
    assert loaded.frontier("upstream") == {"src/client.py": 0}


def test_an_unknown_id_raises_with_the_ids_that_do_exist(tmp_path) -> None:
    known = new_slice_id()
    with SliceStore.open_default(tmp_path) as store:
        store.save(_record(known))
        with pytest.raises(SliceNotFoundError) as excinfo:
            store.load("sl_ffffffffffff")

    details = excinfo.value.details()
    assert details["slice_id"] == "sl_ffffffffffff"
    assert known in details["recent_slice_ids"]


def test_saving_twice_replaces_rather_than_duplicates(tmp_path) -> None:
    slice_id = new_slice_id()
    with SliceStore.open_default(tmp_path) as store:
        record = _record(slice_id)
        store.save(record, event={"kind": "build"})
        record.members = record.members[:1]
        record.revision = 3
        store.save(record, event={"kind": "extend"})
        loaded = store.load(slice_id)

    assert len(loaded.members) == 1
    assert loaded.revision == 3
    assert [event["kind"] for event in loaded.events] == ["build", "extend"]


def test_listing_and_deleting(tmp_path) -> None:
    with SliceStore.open_default(tmp_path) as store:
        first, second = new_slice_id(), new_slice_id()
        store.save(_record(first, task="first task"))
        store.save(_record(second, task="second task"))

        listed = store.list_slices("repo1")
        assert {row["slice_id"] for row in listed} == {first, second}

        assert store.delete(first) is True
        assert store.delete(first) is False
        with pytest.raises(SliceNotFoundError):
            store.load(first)
        assert store.load(second).task == "second task"


def test_two_repos_get_two_stores(tmp_path) -> None:
    """A slice belongs to the repo it was cut from, not to the machine."""
    repo_a, repo_b = tmp_path / "a", tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()
    slice_id = new_slice_id()

    with SliceStore.open_default(repo_a) as store:
        store.save(_record(slice_id))
    with SliceStore.open_default(repo_b) as store, pytest.raises(SliceNotFoundError):
        store.load(slice_id)
