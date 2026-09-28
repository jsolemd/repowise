"""Changed-line evidence keeps new findings from stealing moved findings."""

import pytest

from repowise.core.analysis.change_health.matcher import FindingMatcher
from repowise.core.analysis.change_health.service import ChangeHealthDeltaService, DeltaRequest
from repowise.core.analysis.change_health.sources import FileChange
from repowise.core.analysis.changed_lines import FileDiff
from repowise.core.analysis.health import HealthFindingData, Severity


def _handler(name):
    return (
        f"def {name}(callback):\n"
        "    try:\n"
        "        return callback()\n"
        "    except Exception:\n"
        "        return None\n\n"
    )


@pytest.mark.parametrize("rename", [False, True])
def test_inserted_handler_does_not_charge_unchanged_handlers(make_repo, rename):
    repo = make_repo()
    original = "".join(_handler(name) for name in ("first", "second", "third"))
    repo.commit("seed", {"app.py": original})
    path = "renamed.py" if rename else "app.py"
    if rename:
        repo.move("app.py", path)
    repo.commit("insert handler", {path: _handler("inserted") + original})

    delta = ChangeHealthDeltaService(repo_path=str(repo.path)).compare(
        DeltaRequest(str(repo.path), "HEAD")
    )
    findings = [f for f in delta.findings if f.biomarker_type == "error_handling"]
    assert len(findings) == 1
    assert findings[0].line_start == 4
    assert findings[0].attribution_basis == "added_lines"
    assert findings[0].path == path


@pytest.mark.parametrize("status", ["truncated", "unavailable", "binary"])
def test_unreliable_patch_cannot_override_proximity(status):
    def finding(line):
        return HealthFindingData(
            biomarker_type="error_handling", severity=Severity.LOW,
            file_path="app.py", function_name=None, line_start=line, line_end=line,
            details={}, health_impact=0.1, reason="broad catch", dimension="maintainability",
        )

    base = finding(10)
    nearby, distant = finding(11), finding(30)
    # This incomplete patch would wrongly give line30 the same surviving-line
    # rank as base line10. Its missing hunks make that evidence unusable.
    change = FileChange(
        "app.py", "app.py", "modified",
        diff=FileDiff("app.py", new_lines=set(range(1, 21))), diff_status=status,
    )
    result = FindingMatcher(changes={"app.py": change}).match([base], [nearby, distant])
    assert result.matched[0].base is base
    assert result.matched[1].kind == "introduced"
