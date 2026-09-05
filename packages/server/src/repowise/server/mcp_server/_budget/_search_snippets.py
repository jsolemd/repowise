"""Search excerpts are cheaper to lose than ranked, evidence-bearing rows.

Reduce tail snippets before the shared contract sheds any result. The claim,
rank, evidence, symbol identity, candidate handles and trust remain untouched.
Full snippets go to the existing omission store, identified by repo and path.
"""

from __future__ import annotations

from typing import Any

from repowise.server.mcp_server._budget.budgeter import response_chars
from repowise.server.mcp_server._budget.collector import OmissionCollector

_MIN_SNIPPET_CHARS = 800


def trim_search_snippets(result: dict[str, Any], collector: OmissionCollector, limit: int) -> None:
    rows = result.get("results")
    if not isinstance(rows, list) or response_chars(result) <= limit:
        return
    for index in range(len(rows) - 1, -1, -1):
        if response_chars(result) <= limit:
            break
        row = rows[index]
        if not isinstance(row, dict):
            continue
        original = row.get("snippet")
        if not isinstance(original, str) or len(original) <= _MIN_SNIPPET_CHARS:
            continue
        before = dict(row)
        row["snippet_truncated"] = True
        row["snippet_reduced_reason"] = "response_budget"
        # Count serialized characters, not raw text: quotes, newlines and
        # non-ASCII code can otherwise leave a supposedly fitted answer over.
        low, high = _MIN_SNIPPET_CHARS, len(original)
        row["snippet"] = original[:low]
        if response_chars(row) + 20 >= response_chars(before):
            row.clear()
            row.update(before)
            continue
        result["truncated"] = True
        if response_chars(result) <= limit:
            while low < high:
                middle = (low + high + 1) // 2
                row["snippet"] = original[:middle]
                if response_chars(result) <= limit:
                    low = middle
                else:
                    high = middle - 1
        row["snippet"] = original[:low]
        collector.add(
            f"results[{index}].snippet before response budgeting",
            {
                key: original if key == "snippet" else row[key]
                for key in ("repo", "target_path", "file", "symbol_path", "symbol_id", "snippet")
                if key in row
            },
        )
