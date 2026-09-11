"""I/O ownership and ordering are separate from local dataflow independence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from repowise.core.analysis.health.biomarkers import FileContext
from repowise.core.analysis.health.biomarkers.serial_await_in_loop import BIOMARKER
from repowise.core.analysis.health.complexity import FileComplexity, PerfHit
from repowise.core.analysis.health.perf.actionability import actionability, assess_fix
from repowise.core.analysis.health.perf.promotion import apply_perf_promotions


@pytest.mark.parametrize("receiver", ["session", "connection", "cursor", "client", "writer"])
def test_shared_io_resource_is_never_promoted(tmp_path, receiver):
    path = tmp_path / "shared.py"
    path.write_text(
        f"async def write({receiver}, items):\n"
        "    for item in items:\n"
        f"        await {receiver}.execute(item)\n"
    )
    parsed = SimpleNamespace(file_info=SimpleNamespace(
        path=path.name, abs_path=str(path), language="python",
    ))
    complexity = FileComplexity(
        functions=[], classes=[], perf_hits=[
            PerfHit(kind="serial_await_in_loop", line=3, function="write", detail="db"),
        ],
    )

    apply_perf_promotions([(parsed, complexity)])

    assert complexity.perf_hits[0].promoted is False


def test_legacy_promoted_hit_does_not_assert_fanout():
    context = FileContext(
        file_path="shared.py", language="python", nloc=3,
        has_test_file=False, module=None,
        perf_hits=[PerfHit(
            kind="serial_await_in_loop", line=3, function="write",
            detail="db", promoted=True,
        )],
    )

    finding, = BIOMARKER.detect(context)

    assert not finding.details.get("dataflow_verified")
    assert "if the iterations are independent" in finding.reason


@pytest.mark.parametrize("details", [[], [{}], [{"dataflow_verified": True}]])
@pytest.mark.parametrize("boundary", ["db", "network", "filesystem", "lock", "subprocess"])
def test_local_dataflow_cannot_prove_concurrent_io_safe(details, boundary):
    assessment = assess_fix(
        "serial_await_in_loop", ("serial_await_in_loop",), boundary,
        details, cross_function=False,
    )
    verdict = actionability(assessment, "high")

    if details and all(detail.get("dataflow_verified") for detail in details):
        assert verdict.state == "advisory"
        assert verdict.fix is not None and verdict.fix.safety == "advisory"
    else:
        assert verdict.state == "investigate"
        assert verdict.fix is None
    assert "resource_concurrency_contract" in verdict.prerequisites
    assert "effect_and_failure_ordering" in verdict.prerequisites
    assert "bounded_concurrency" in verdict.prerequisites
