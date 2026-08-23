"""Extraction produces a site per occurrence, and says what it cannot read.

Locks the per-language gate, the two tiers, and the three losses the store
exists to recover: same-line call dedup, symbols deleted inside anonymous
callables, and names reachable only through a string.
"""

from __future__ import annotations

from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.taxonomy import (
    TIER_AST,
    TIER_TEXTUAL,
    ReferenceKind,
    ReferenceOrigin,
)
from tests.unit.refsites.conftest import COMPUTE_TOTAL_GROUND_TRUTH


def _sites(result, **match):
    out = []
    for site in result.sites:
        if all(getattr(site, key) == value for key, value in match.items()):
            out.append(site)
    return sorted(out, key=lambda s: (s.file_path, s.start_line, s.start_col))


def test_every_gated_language_is_measured(toy_repo):
    """Each declared language reports a status and a file count for this repo."""
    result = extract_repository(toy_repo)
    by_language = result.by_language()

    assert by_language["python"].status is CoverageStatus.FULL
    assert by_language["typescript"].status is CoverageStatus.FULL
    assert by_language["javascript"].status is CoverageStatus.FULL
    assert by_language["shell"].status is CoverageStatus.PARTIAL
    assert by_language["sql"].status is CoverageStatus.PARTIAL
    assert by_language["yaml"].status is CoverageStatus.NONE

    # Every language with files produced sites, except the one that cannot.
    for language, record in by_language.items():
        assert record.files_seen > 0
        if language == "yaml":
            assert record.sites == 0
            assert record.files_extracted == 0
        else:
            assert record.sites > 0, language


def test_uncovered_language_is_declared_not_silent(toy_repo):
    """YAML files are read, produce nothing, and say why."""
    result = extract_repository(toy_repo)
    yaml = result.by_language()["yaml"]

    assert yaml.files_seen == 1
    assert yaml.sites == 0
    assert yaml.status is CoverageStatus.NONE
    assert "no tree-sitter query file" in yaml.reason
    # The name is in the YAML text; no site exists for it anywhere in that file.
    assert not _sites(result, file_path="pipeline.yaml")


def test_partial_languages_name_their_limitations(toy_repo):
    """A partial claim carries the specific thing it cannot do."""
    result = extract_repository(toy_repo)
    by_language = result.by_language()

    assert any("call.receiver" in text for text in by_language["shell"].limitations)
    assert any("no call-site vocabulary" in text for text in by_language["sql"].limitations)

    # SQL yields definitions and nothing else, which is the claim.
    sql_kinds = {site.kind for site in _sites(result, language="sql")}
    assert sql_kinds == {ReferenceKind.DEFINITION}


def test_sql_definitions_carry_exact_columns(toy_repo):
    """The sqlglot handler gives lines; placement supplies the columns."""
    result = extract_repository(toy_repo)
    names = {site.name: site for site in _sites(result, language="sql")}

    assert set(names) == {"totals", "total_summary"}
    assert all(site.range_exact for site in names.values())
    assert names["totals"].start_line == 1


def test_shell_calls_resolve_same_file(toy_repo):
    """Shell function calls bind to the definition in the same script."""
    result = extract_repository(toy_repo)
    calls = _sites(result, file_path="deploy.sh", kind=ReferenceKind.CALL)

    assert [site.start_line for site in calls] == [9, 10]
    assert all(site.target_symbol_id == "deploy.sh::compute_total" for site in calls)
    assert all(site.origin is ReferenceOrigin.SAME_FILE for site in calls)


def test_same_line_call_pair_is_recovered(toy_repo):
    """Two calls on one line yield two sites, where the parse layer keeps one.

    ``widget.ts`` line 7 is ``[() => computeTotal(1), () => computeTotal(2)]``.
    The parse layer dedups call sites on ``(line, target, receiver)``, so it
    reports one. The store reports both — the second as a textual site, since
    only the first is AST-attested.
    """
    result = extract_repository(toy_repo)
    line_seven = [
        site
        for site in result.sites
        if site.file_path == "widget.ts" and site.start_line == 7 and site.name == "computeTotal"
    ]

    assert len(line_seven) == 2
    assert {site.occurrence_index for site in line_seven} == {0, 1}
    assert {site.tier for site in line_seven} == {TIER_AST, TIER_TEXTUAL}
    assert all(site.range_exact for site in line_seven)
    assert len({site.start_col for site in line_seven}) == 2
    assert all(site.target_symbol_id == "calc.ts::computeTotal" for site in line_seven)


def test_python_same_line_call_pair_is_recovered(toy_repo):
    """``compute_total(a) + compute_total(b)`` is two sites, not one."""
    result = extract_repository(toy_repo)
    paired = [
        site
        for site in result.sites
        if site.file_path == "app.py" and site.name == "compute_total" and site.start_line == 17
    ]

    assert len(paired) == 2
    assert {site.occurrence_index for site in paired} == {0, 1}
    assert all(site.target_symbol_id == "helpers.py::compute_total" for site in paired)


def test_string_reference_is_recorded_at_the_floor(toy_repo):
    """A name reached only through a string is kept, at the lowest tier."""
    result = extract_repository(toy_repo)
    string_refs = _sites(result, kind=ReferenceKind.STRING_REF, name="compute_total")

    assert [site.file_path for site in string_refs] == ["app.py"]
    assert string_refs[0].origin is ReferenceOrigin.TEXTUAL_STRING
    assert string_refs[0].confidence == 0.10
    assert string_refs[0].target_symbol_id == "helpers.py::compute_total"


def test_unresolvable_call_survives_at_low_confidence(toy_repo):
    """A call to a name nothing defines keeps its row rather than vanishing."""
    result = extract_repository(toy_repo)
    orphans = _sites(result, name="not_a_real_function")

    assert len(orphans) == 1
    assert orphans[0].kind is ReferenceKind.CALL
    assert orphans[0].target_symbol_id is None
    assert orphans[0].origin is ReferenceOrigin.UNRESOLVED
    assert orphans[0].confidence == 0.20
    assert orphans[0].range_exact


def test_import_bindings_resolve_across_files(toy_repo):
    """``from helpers import compute_total`` binds to the defining symbol."""
    result = extract_repository(toy_repo)
    bindings = _sites(result, kind=ReferenceKind.IMPORT_BINDING, name="compute_total")

    assert [site.file_path for site in bindings] == ["app.py"]
    assert bindings[0].target_symbol_id == "helpers.py::compute_total"
    assert bindings[0].origin is ReferenceOrigin.IMPORT_BINDING


def test_destructuring_require_is_placed_in_the_source(toy_repo):
    """A statement whose captured text does not start its line still lands."""
    result = extract_repository(toy_repo)
    imports = _sites(result, file_path="legacy.js", kind=ReferenceKind.IMPORT)

    assert len(imports) == 1
    assert imports[0].start_line == 1
    assert imports[0].range_exact
    assert imports[0].start_col > 0


def test_ground_truth_site_list_for_the_rename_target(toy_repo):
    """Every site of ``calc.ts::computeTotal``, hand-counted from the fixture."""
    result = extract_repository(toy_repo)
    observed = tuple(
        (site.file_path, site.start_line, str(site.kind))
        for site in sorted(
            (site for site in result.sites if site.name == "computeTotal"),
            key=lambda s: (s.file_path, s.start_line, str(s.kind)),
        )
    )
    assert observed == COMPUTE_TOTAL_GROUND_TRUTH


def test_extraction_is_deterministic(toy_repo):
    """Two runs over the same tree produce identical rows."""
    first = extract_repository(toy_repo)
    second = extract_repository(toy_repo)

    def key(result):
        return sorted(
            (s.file_path, s.start_line, s.start_col, s.name, str(s.kind), s.target_symbol_id or "")
            for s in result.sites
        )

    assert key(first) == key(second)


def test_no_site_is_minted_inside_a_comment(toy_repo, tmp_path):
    """The sweep skips comment regions; only the real call is a site."""
    (tmp_path / "commented.py").write_text(
        "from helpers import compute_total\n\n\ndef go():\n    # compute_total is mentioned\n"
        "    return compute_total(1)\n"
    )
    result = extract_repository(tmp_path)
    lines = {
        site.start_line
        for site in result.sites
        if site.file_path == "commented.py" and site.name == "compute_total"
    }
    assert 5 not in lines
    assert lines == {1, 6}


def test_bare_call_does_not_bind_to_an_out_of_scope_method(tmp_path):
    """A module-level ``render()`` is not the ``render`` method of a class.

    One file defining both is the case where taking whichever symbol parsed
    first produces a ``same_file`` binding at 0.95 confidence that is simply
    wrong. Scope decides, and a rename must not be handed that mistake.
    """
    (tmp_path / "scoped.py").write_text(
        "class Report:\n"
        "    def render(self):\n"
        "        return self.helper()\n"
        "\n"
        "    def helper(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "def render():\n"
        "    return 2\n"
        "\n"
        "\n"
        "def go():\n"
        "    return render()\n"
    )
    result = extract_repository(tmp_path)

    module_call = next(
        site for site in result.sites if site.name == "render" and site.kind is ReferenceKind.CALL
    )
    assert module_call.start_line == 14
    assert module_call.target_symbol_id == "scoped.py::render"
    assert module_call.origin is ReferenceOrigin.SAME_FILE

    # And the sibling-method case still resolves to the sibling, not to a
    # module-level function that happens to share the name.
    sibling_call = next(
        site for site in result.sites if site.name == "helper" and site.kind is ReferenceKind.CALL
    )
    assert sibling_call.target_symbol_id == "scoped.py::Report::helper"
    assert sibling_call.origin is ReferenceOrigin.SAME_FILE


def test_textual_tier_can_be_switched_off(toy_repo):
    """Tier A alone is a supported, and visibly less complete, configuration."""
    full = extract_repository(toy_repo)
    ast_only = extract_repository(toy_repo, sweep_textual=False)

    assert all(site.tier == TIER_AST for site in ast_only.sites)
    assert len(ast_only.sites) < len(full.sites)

    # The recoveries are exactly what is given up: the same-line siblings, the
    # lambda-scoped declaration, and the name inside a string.
    given_up = {
        (site.file_path, site.start_line, str(site.kind))
        for site in full.sites
        if site.tier == TIER_TEXTUAL
    }
    assert given_up == {
        ("app.py", 17, "identifier"),
        ("app.py", 33, "string_ref"),
        ("widget.ts", 7, "identifier"),
        ("widget.ts", 10, "identifier"),
    }


def test_sweep_is_bounded_by_names_attested_in_the_same_file(tmp_path):
    """A name that is only a symbol *elsewhere* is not swept into this file.

    Without the per-file bound, common identifiers that happen to be somebody's
    symbol name turn nearly every token in the tree into a site.
    """
    (tmp_path / "defines.py").write_text("def value():\n    return 1\n")
    (tmp_path / "unrelated.py").write_text("def go():\n    value = 3\n    return value\n")

    result = extract_repository(tmp_path)
    swept = [
        site for site in result.sites if site.file_path == "unrelated.py" and site.name == "value"
    ]
    assert swept == []
