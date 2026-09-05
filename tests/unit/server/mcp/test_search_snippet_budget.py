"""F64: preserve ranked evidence before spending the budget on full excerpts."""

from __future__ import annotations

import copy
import inspect
import json

import pytest

from repowise.core.distill.store import OmissionStore, default_store_path
from repowise.server.mcp_server._budget import enforce_response_budget
from repowise.server.mcp_server._budget.budgeter import response_chars


def _payload(text: str) -> dict:
    rows = [
        {
            "repo": "infra",
            "target_path": f"src/{i}.py",
            "symbol_path": f"Worker{i}",
            "snippet": f"def Worker{i}():\n{text}",
            "evidence": {"dense_cosine": 0.7, "lane": "source"},
            "relevance_score": 0.5 - i / 100,
        }
        for i in range(10)
    ]
    return {
        "results": rows,
        "candidates": [{"repo": "infra", "path": r["target_path"]} for r in rows],
        "selected_owner": {
            "repo": "infra",
            "file": rows[0]["target_path"],
            "reason": "exact match",
        },
        "confidence": "confident",
        "trust": {"source_search": {"status": "current"}},
    }


def _enforce(payload, root):
    (root / ".repowise").mkdir(exist_ok=True)
    return enforce_response_budget(
        "search_codebase",
        payload,
        signature=inspect.signature(lambda: None),
        args=(),
        kwargs={},
        repo_root=root,
    )


@pytest.mark.parametrize(
    "snippet", ["code\n" * 900, 'code: 雪\\"\n' * 900], ids=["code", "escaped-unicode"]
)
def test_snippets_shrink_before_rows_and_full_text_is_recoverable(tmp_path, snippet):
    payload = _payload(snippet)
    original = copy.deepcopy(payload)
    result = _enforce(payload, tmp_path)

    assert response_chars(result) <= 24_000
    assert len(result["results"]) == 10
    for key in ("candidates", "selected_owner", "confidence", "trust"):
        assert result[key] == original[key]
    trimmed = []
    for row, before in zip(result["results"], original["results"], strict=True):
        for key in ("repo", "target_path", "symbol_path", "relevance_score", "evidence"):
            assert row[key] == before[key]
        assert before["snippet"].startswith(row["snippet"])
        if row.get("snippet_truncated"):
            assert row["snippet_reduced_reason"] == "response_budget"
            assert len(row["snippet"]) >= 800
            trimmed.append(before)
    assert trimmed
    with OmissionStore(default_store_path(tmp_path)) as store:
        restored = "\n".join(store.get(ref) or "" for ref in result["_meta"]["omitted"]["refs"])
    for before in trimmed:
        assert json.dumps(before["snippet"]) in restored
        assert before["target_path"] in restored
    assert result["omission_marker"]
    assert _enforce(copy.deepcopy(original), tmp_path) == result


def test_under_budget_rows_are_unchanged(tmp_path):
    payload = _payload("return True")
    original = copy.deepcopy(payload)
    result = _enforce(payload, tmp_path)
    assert result["results"] == original["results"]
    assert not result.get("truncated")
    assert "omission_marker" not in result


def test_row_shedding_still_discloses_counts_when_evidence_itself_is_large(tmp_path):
    payload = _payload("x" * 3_000)
    for row in payload["results"]:
        row["evidence"]["explanation"] = "e" * 3_000
    original_candidates = copy.deepcopy(payload["candidates"])
    result = _enforce(payload, tmp_path)
    assert 0 < len(result["results"]) < 10
    assert result["results_total"] == 10
    assert result["results_emitted"] == len(result["results"])
    assert result["results_omitted"] == 10 - len(result["results"])
    assert result["candidates"] == original_candidates
    assert all(r["evidence"]["explanation"] == "e" * 3_000 for r in result["results"])


def test_omission_store_failure_never_advertises_recovery(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("read-only store")

    monkeypatch.setattr(OmissionStore, "put", fail)
    result = _enforce(_payload("x" * 4_000), tmp_path)
    assert len(result["results"]) == 10
    assert result["_meta"]["recovery_unavailable"]
    assert "omission_marker" not in result
    assert not result["_meta"].get("omitted")
