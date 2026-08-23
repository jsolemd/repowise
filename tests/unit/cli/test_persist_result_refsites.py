"""The CLI ``persist_result`` path keeps the reference-site store current.

The normal single-repo ``repowise init`` persists the INDEX phase
incrementally during the run, so ``persist_result`` takes the
``index_persisted_incrementally=True`` branch. Before the fix that branch
called ``persist_analysis`` + ``persist_generation`` but never refreshed the
reference-site store, so a fresh index ended with ``reference_sites`` empty --
and a subsequent no-change ``repowise update`` skips the re-parse, so the
sites never appeared at all. The two opt-in tools that read the store were
dark on every freshly indexed repo until something forced an incremental
full-tree rebuild.

Same shape as ``test_persist_result_sweep.py``: a real repo-local SQLite DB
driven through ``persist_result``, asserting against the rows that land.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import func, select

from repowise.cli._repo_session import open_repo_db
from repowise.cli.commands.init_cmd.persistence import persist_result
from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.persistence import get_session
from repowise.core.pipeline.persist import persist_reference_sites
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.schema import ReferenceSite, ReferenceSiteCoverage
from repowise.core.refsites.store import SqlReferenceSiteStore

WIDGET_PY = """\
class Widget:
    def __init__(self, name):
        self.name = name

    def render(self):
        return "<" + self.name + ">"


def make_widget(name):
    return Widget(name)
"""

APP_PY = """\
from widget import Widget, make_widget


def build():
    w = make_widget("box")
    other = Widget("panel")
    return w.render() + other.render()


def main():
    print(build())
"""

CALC_TS = """\
export function computeTotal(n: number): number {
  return n + 1;
}
"""

WIDGET_TS = """\
import { computeTotal } from "./calc";

export function widgetTotal(n: number): number {
  return computeTotal(n);
}
"""

SOURCES: dict[str, str] = {
    "widget.py": WIDGET_PY,
    "app.py": APP_PY,
    "calc.ts": CALC_TS,
    "widget.ts": WIDGET_TS,
}


def _write_repo(repo_path: Path) -> Path:
    """A small multi-language tree with real cross-file references.

    Deliberately not a git checkout: ``persist_result`` never consults git, and
    the reference-site extractor works off the parse, so the sites this asserts
    on are the same either way. Its own ``.git`` marker is what gives the
    traverser a root.
    """
    repo_path.mkdir(parents=True, exist_ok=True)
    for name, body in SOURCES.items():
        (repo_path / name).write_text(body)
    (repo_path / ".git").mkdir(exist_ok=True)
    return repo_path


def _pipeline_parse(root: Path) -> list:
    """Parse the way the pipeline does: traverse + ASTParser over raw bytes."""
    parser = ASTParser()
    parsed = []
    for file_info in FileTraverser(root).traverse():
        raw = (root / file_info.path).read_bytes()
        parsed.append(parser.parse_file(file_info, raw))
    return parsed


def _result(repo_name: str, parsed_files: list) -> SimpleNamespace:
    """A minimal PipelineResult stand-in for the ``index_done`` branch.

    Everything the phase persisters read is falsy so they no-op; only the
    reference-site refresh has anything to do.
    """
    return SimpleNamespace(
        repo_name=repo_name,
        index_persisted_incrementally=True,
        parsed_files=parsed_files,
        generated_pages=[],
        tech_stack=None,
        vector_store=None,
        dead_code_report=None,
        health_report=None,
        decision_report=None,
        git_metadata_list=[],
        knowledge_graph_result=None,
        authoritative_page_types=set(),
        preserved_page_ids=set(),
    )


async def _persisted_sites(repo_path: Path) -> list[tuple]:
    """Every persisted site as a comparable tuple, repo-id free and sorted."""
    engine, sf, repo_id = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            rows = (
                await session.execute(
                    select(
                        ReferenceSite.file_path,
                        ReferenceSite.language,
                        ReferenceSite.name,
                        ReferenceSite.kind,
                        ReferenceSite.start_line,
                        ReferenceSite.end_line,
                        ReferenceSite.start_col,
                        ReferenceSite.end_col,
                        ReferenceSite.target_symbol_id,
                        ReferenceSite.enclosing_symbol_id,
                        ReferenceSite.resolution_origin,
                        ReferenceSite.tier,
                        ReferenceSite.occurrence_index,
                    ).where(ReferenceSite.repository_id == repo_id)
                )
            ).all()
    finally:
        await engine.dispose()
    return sorted(tuple(row) for row in rows)


async def test_a_fresh_index_leaves_reference_sites_populated(tmp_path):
    """The regression: ``reference_sites`` must not be empty after an init.

    Pre-fix this fails at zero -- the ``index_done`` branch persisted analysis
    and generation and nothing else, so the store the init was supposed to fill
    stayed empty on every normal single-repo run.
    """
    repo_path = _write_repo(tmp_path / "repo")
    parsed = _pipeline_parse(repo_path)

    await persist_result(_result("r", parsed), repo_path)

    engine, sf, repo_id = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            count = await SqlReferenceSiteStore(session).count(repo_id)
            coverage = (
                await session.execute(
                    select(func.count())
                    .select_from(ReferenceSiteCoverage)
                    .where(ReferenceSiteCoverage.repository_id == repo_id)
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert count > 0
    assert coverage > 0


async def test_the_init_store_matches_a_direct_extraction(tmp_path):
    """What init persists is the whole tree's extraction, not a subset.

    ``persist_reference_sites`` is a full replace over the tree, so a run that
    reached the seam with a changed-file slice would still leave a non-empty
    store -- and the count above would pass while the store was wrong. This is
    the assertion that a partial wiring cannot satisfy.
    """
    repo_path = _write_repo(tmp_path / "repo")
    parsed = _pipeline_parse(repo_path)

    await persist_result(_result("r", parsed), repo_path)

    direct = extract_repository(repo_path)
    persisted = await _persisted_sites(repo_path)
    assert len(persisted) == len(direct.sites)
    assert {row[0] for row in persisted} == {site.file_path for site in direct.sites}


async def test_init_persists_what_the_update_path_would(tmp_path):
    """Equivalence with the path whose sites the workaround produced.

    The bug was worked around by forcing a full re-parse through
    ``repowise update``, whose incremental persister calls the seam as
    ``persist_reference_sites(session, repo_id, parsed_files, repo_path=root)``.
    That literal call is reproduced here against a second database, and the
    persisted rows must match the init path's exactly -- the fix wires a second
    call site rather than unifying the two, so nothing structural stops them
    from drifting apart.
    """
    init_repo = _write_repo(tmp_path / "init")
    update_repo = _write_repo(tmp_path / "update")

    await persist_result(_result("r", _pipeline_parse(init_repo)), init_repo)

    # The update path's call, verbatim (pipeline/incremental.py).
    engine, sf, repo_id = await open_repo_db(update_repo, repo_name="r")
    try:
        async with get_session(sf) as session:
            await persist_reference_sites(
                session, repo_id, _pipeline_parse(update_repo), repo_path=update_repo
            )
            await session.commit()
    finally:
        await engine.dispose()

    assert await _persisted_sites(init_repo) == await _persisted_sites(update_repo)


async def test_an_empty_parse_leaves_the_existing_store_alone(tmp_path):
    """A run with nothing parsed must not wipe the sites already on disk.

    The seam is a full replace, so handing it an empty list is how a resumed
    or degraded run would delete every site it failed to re-derive. The update
    path guards this with ``if parsed_files:``; the init path has to as well.
    """
    repo_path = _write_repo(tmp_path / "repo")
    parsed = _pipeline_parse(repo_path)

    await persist_result(_result("r", parsed), repo_path)
    before = await _persisted_sites(repo_path)
    assert before

    await persist_result(_result("r", []), repo_path)

    assert await _persisted_sites(repo_path) == before
