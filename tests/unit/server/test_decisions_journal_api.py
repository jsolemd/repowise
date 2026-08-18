"""REST/UI read parity and mutation policy for journal-backed decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournal,
)
from tests.unit.server.conftest import create_test_repo


@pytest.fixture
async def journal_api_repo(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    repo = await create_test_repo(client, tmp_path)
    root = Path(repo["local_path"])
    (root / "src").mkdir()
    (root / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "next.py").write_text("VALUE = 2\n", encoding="utf-8")
    return repo, root


async def _post_decision(
    client: AsyncClient,
    repo_id: str,
    *,
    title: str,
    affected_file: str,
):
    return await client.post(
        f"/api/repos/{repo_id}/decisions",
        json={
            "title": title,
            "decision": f"{title} is the chosen implementation.",
            "rationale": "This keeps one durable source of truth.",
            "affected_files": [affected_file],
        },
    )


@pytest.mark.asyncio
async def test_create_list_get_and_health_match_canonical_jsonl(
    client: AsyncClient,
    journal_api_repo,
) -> None:
    repo, root = journal_api_repo
    created_response = await _post_decision(
        client,
        repo["id"],
        title="Use the tracked decision journal",
        affected_file="src/service.py",
    )
    assert created_response.status_code == 201, created_response.text
    created = created_response.json()

    rows = DecisionJournal(root).list()
    assert len(rows) == 1
    canonical = rows[0]
    assert created["id"] == canonical.id
    assert created["source"] == "journal"
    assert created["decision"] == canonical.decision
    assert created["rationale"] == canonical.why
    assert created["status"] == canonical.status
    assert created["supersedes"] == canonical.supersedes
    assert created["superseded_by"] == canonical.superseded_by
    assert created["anchors"] == [anchor.to_dict() for anchor in canonical.anchors]
    assert created["affected_files"] == [anchor.file for anchor in canonical.anchors]

    listed_response = await client.get(f"/api/repos/{repo['id']}/decisions")
    assert listed_response.status_code == 200
    listed = listed_response.json()
    assert [row["id"] for row in listed] == [canonical.id]
    assert listed[0]["anchors"] == created["anchors"]

    detail_response = await client.get(f"/api/repos/{repo['id']}/decisions/{canonical.id}")
    assert detail_response.status_code == 200
    assert detail_response.json() == created

    health_response = await client.get(f"/api/repos/{repo['id']}/decisions/health")
    assert health_response.status_code == 200
    health = health_response.json()["journal"]
    assert health["path"] == str(root / ".codeatlas" / "decisions.jsonl")
    assert len(health["content_hash"]) == 64
    assert health["projected_count"] == 1
    assert health["last_refresh"]
    assert health["lock_acquirable"] is True


@pytest.mark.asyncio
async def test_external_edit_is_visible_on_next_api_read(
    client: AsyncClient,
    journal_api_repo,
) -> None:
    repo, root = journal_api_repo
    journal = DecisionJournal(root)
    external = journal.record(
        decision_id="dec-eeeeeeee",
        title="Externally edited decision",
        decision="Hand edits are projected on the next read.",
        why="Git pull and editor changes cannot require a server restart.",
        anchors=[{"file": "src/service.py", "symbol": "VALUE"}],
    )
    payload = json.loads(journal.path.read_text(encoding="utf-8"))
    payload["decision"] = "  Hand edits are projected on the next read.\n"
    journal.path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    external = journal.get(external.id)

    listed_response = await client.get(f"/api/repos/{repo['id']}/decisions")
    assert listed_response.status_code == 200
    listed = listed_response.json()
    assert [row["id"] for row in listed] == [external.id]
    assert listed[0]["decision"] == external.decision
    assert listed[0]["anchors"][0]["symbol"] == "VALUE"


@pytest.mark.asyncio
async def test_supported_patch_operations_write_jsonl_first(
    client: AsyncClient,
    journal_api_repo,
) -> None:
    repo, root = journal_api_repo
    old_response = await _post_decision(
        client,
        repo["id"],
        title="Original implementation",
        affected_file="src/service.py",
    )
    successor_response = await _post_decision(
        client,
        repo["id"],
        title="Replacement implementation",
        affected_file="src/next.py",
    )
    assert old_response.status_code == successor_response.status_code == 201
    old_id = old_response.json()["id"]
    successor_id = successor_response.json()["id"]

    link_response = await client.patch(
        f"/api/repos/{repo['id']}/decisions/{old_id}",
        json={"affected_files": ["src/service.py", "src/next.py"]},
    )
    assert link_response.status_code == 200, link_response.text
    assert link_response.json()["affected_files"] == ["src/service.py", "src/next.py"]
    assert len(DecisionJournal(root).get(old_id).anchors) == 2

    supersede_response = await client.patch(
        f"/api/repos/{repo['id']}/decisions/{old_id}",
        json={"status": "superseded", "superseded_by": successor_id},
    )
    assert supersede_response.status_code == 200, supersede_response.text
    assert supersede_response.json()["status"] == "superseded"
    assert supersede_response.json()["superseded_by"] == successor_id

    canonical = {row.id: row for row in DecisionJournal(root).list()}
    assert canonical[old_id].superseded_by == successor_id
    assert canonical[successor_id].supersedes == old_id

    successor_detail = await client.get(f"/api/repos/{repo['id']}/decisions/{successor_id}")
    assert successor_detail.status_code == 200
    assert successor_detail.json()["supersedes"] == old_id


@pytest.mark.asyncio
async def test_noncanonical_mutations_are_visibly_disabled(
    client: AsyncClient,
    journal_api_repo,
) -> None:
    repo, root = journal_api_repo
    created = await _post_decision(
        client,
        repo["id"],
        title="Canonical-only updates",
        affected_file="src/service.py",
    )
    assert created.status_code == 201
    decision_id = created.json()["id"]
    before = (root / ".codeatlas" / "decisions.jsonl").read_bytes()

    deprecated = await client.patch(
        f"/api/repos/{repo['id']}/decisions/{decision_id}",
        json={"status": "deprecated"},
    )
    modules = await client.patch(
        f"/api/repos/{repo['id']}/decisions/{decision_id}",
        json={"affected_modules": ["src"]},
    )
    lossy_create = await client.post(
        f"/api/repos/{repo['id']}/decisions",
        json={
            "title": "Unsupported metadata",
            "decision": "Do not silently drop fields.",
            "rationale": "The caller needs an honest error.",
            "affected_files": ["src/service.py"],
            "tags": ["architecture"],
        },
    )

    assert deprecated.status_code == 409
    assert "canonical journal representation" in deprecated.json()["detail"]
    assert modules.status_code == 409
    assert "affected_modules" in modules.json()["detail"]
    assert lossy_create.status_code == 409
    assert "cannot losslessly store fields" in lossy_create.json()["detail"]
    assert (root / ".codeatlas" / "decisions.jsonl").read_bytes() == before
