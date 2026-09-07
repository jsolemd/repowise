"""The no-spend rules around a template wiki, and its module-page reconcile.

Two promises are under test on the embedder side. A run that says it costs
nothing does not put a hosted embedder on the bill, and it does not silently
change which embedder a repo uses between one run and the next. Both are easy
to break by accident, because the embedder is resolved from whatever API key
happens to be in the environment.

The reconcile side covers the other half of an index-only update on a template
wiki: it re-renders file pages and leaves every repo-wide page frozen, so a
module page whose directory was deleted survives every update. Retiring one is
gated on a derived authoritative set, and the tests below pin what happens when
that set is present, absent, and too small to be believed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from repowise.cli.commands.update_cmd.deterministic import (
    deterministic_embedder_name,
    module_reconcile_mode,
    reconcile_module_page_ids,
)
from repowise.core.persistence import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
)
from repowise.core.persistence.crud import upsert_page, upsert_repository
from repowise.core.persistence.database import resolve_db_url
from repowise.core.persistence.models import Page
from repowise.core.pipeline.incremental import persist_incremental_index
from repowise.core.pipeline.prune_state import apply_prune_outcome


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from an environment that infers nothing."""
    for var in (
        "REPOWISE_EMBEDDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestDeterministicEmbedderName:
    def test_config_choice_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The repo was indexed with this embedder, so its store holds vectors
        # of that width. Re-deciding here would rewrite the store.
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert deterministic_embedder_name({"embedder": "openai"}) == "openai"

    def test_ambient_key_alone_does_not_bill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The key is there to pay for a model somewhere else. Nobody asked for
        # 2000 pages of embeddings on a run advertised as free.
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert deterministic_embedder_name({}) == "mock"

    def test_explicit_env_embedder_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOWISE_EMBEDDER", "openai")
        assert deterministic_embedder_name({}) == "openai"

    def test_ollama_needs_no_permission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The keyless one: running it costs nothing, so inferring it is fine.
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        assert deterministic_embedder_name({}) == "ollama"

    def test_nothing_configured_is_mock(self) -> None:
        assert deterministic_embedder_name({}) == "mock"


# ---------------------------------------------------------------------------
# Module-page reconcile on the index-only path
# ---------------------------------------------------------------------------


async def _seed_module_pages(repo_path: Path, *targets: str) -> None:
    """Write one ``module_page`` per target into the repo's own store."""
    engine = create_engine(resolve_db_url(repo_path))
    try:
        await init_db(engine)
        async with get_session(create_session_factory(engine)) as session:
            repo = await upsert_repository(session, name=repo_path.name, local_path=str(repo_path))
            for target in targets:
                await upsert_page(
                    session,
                    page_id=f"module_page:{target}",
                    repository_id=repo.id,
                    page_type="module_page",
                    title=target,
                    content=f"# {target}",
                    summary=f"What {target} does.",
                    target_path=target,
                    source_hash="h",
                    model_name="template",
                    provider_name="template",
                    structural_key=f"concept-{target}",
                )
    finally:
        await engine.dispose()


async def _module_page_ids(repo_path: Path) -> set[str]:
    engine = create_engine(resolve_db_url(repo_path))
    try:
        async with get_session(create_session_factory(engine)) as session:
            rows = await session.execute(select(Page.id).where(Page.page_type == "module_page"))
            return set(rows.scalars().all())
    finally:
        await engine.dispose()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".repowise").mkdir()
    return tmp_path


async def _persist(repo: Path, module_page_ids: set[str] | None, **kwargs):
    """Drive the index-only persist with no graph worth the name.

    Every repo-wide step that needs a real ``graph_builder`` degrades on its
    own, which is deliberate: it isolates the reconcile from the rest of the
    run, and the reconcile itself needs no graph.
    """
    return await persist_incremental_index(
        repo,
        object(),
        {},
        None,
        None,
        [],
        log=lambda _msg: None,
        degraded=kwargs.pop("degraded", []),
        module_page_ids=module_page_ids,
        **kwargs,
    )


class TestModuleReconcileMode:
    def test_default_is_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REPOWISE_MODULE_RECONCILE", raising=False)
        assert module_reconcile_mode() == "on"

    def test_report_and_off_are_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "REPORT")
        assert module_reconcile_mode() == "report"
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "off")
        assert module_reconcile_mode() == "off"

    def test_nonsense_reads_as_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # This runs inside the post-commit hook. Refusing to update because a
        # variable is misspelled costs more than doing the safe default.
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "yes-please")
        assert module_reconcile_mode() == "on"


async def test_index_only_update_retires_module_pages_for_deleted_directories(repo):
    """The live shape: two of three directories are gone from the parse."""
    await _seed_module_pages(repo, "codeatlas/code_search", "codeatlas/chunking", "mcp")

    outcome = await _persist(repo, {"module_page:mcp"})

    assert await _module_page_ids(repo) == {"module_page:mcp"}
    assert set(outcome.swept_page_ids) == {
        "module_page:codeatlas/code_search",
        "module_page:codeatlas/chunking",
    }
    assert outcome.refusals == ()


async def test_index_only_update_without_module_ids_retires_nothing(repo):
    """``None`` is every other caller, and it must keep today's behaviour."""
    seeded = {"module_page:codeatlas/code_search", "module_page:mcp"}
    await _seed_module_pages(repo, "codeatlas/code_search", "mcp")

    outcome = await _persist(repo, None)

    assert await _module_page_ids(repo) == seeded
    assert outcome.swept_page_ids == ()


async def test_reconcile_refusal_reaches_prune_refusals_state(repo):
    """A refused reconcile has to survive into ``state.json``.

    That is the whole reporting chain: the refusal rides the prune outcome,
    ``apply_prune_outcome`` writes it to state, and the index-status tool reads
    it back as ``prune_refused``. A refusal that stopped at the log would leave
    the store quietly holding pages nobody decided to keep.
    """
    targets = [f"src/m{i}" for i in range(30)]
    await _seed_module_pages(repo, *targets)
    degraded: list[str] = []

    outcome = await _persist(repo, {f"module_page:src/m{i}" for i in range(5)}, degraded=degraded)

    assert len(outcome.refusals) == 1
    assert outcome.refusals[0].table == "wiki_module_pages"
    assert len(await _module_page_ids(repo)) == 30
    assert any("prune refused" in entry.lower() for entry in degraded)

    state: dict = {}
    apply_prune_outcome(state, {}, outcome, from_commit="abc1234")
    findings = state["prune_refusals"]["findings"]
    assert [f["table"] for f in findings] == ["wiki_module_pages"]
    assert findings[0]["candidate_paths"] == 25


class TestReconcileGate:
    """What the CLI helper hands persistence, per mode."""

    def _args(self, repo: Path, monkeypatch: pytest.MonkeyPatch, derived: dict[str, str]):
        from repowise.cli.commands.update_cmd import deterministic as det

        monkeypatch.setattr(det, "authoritative_module_pages", lambda **_kw: derived, raising=False)
        monkeypatch.setattr(
            "repowise.core.generation.selection.module_reconcile.authoritative_module_pages",
            lambda **_kw: derived,
        )
        return dict(
            repo_path=repo,
            parsed_files=[],
            graph_builder=object(),
            git_meta_map={},
            kg_modules=None,
            cfg={},
            concurrency=2,
            degraded=[],
        )

    def test_on_mode_enforces_the_derived_set(self, repo, monkeypatch):
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "on")
        args = self._args(repo, monkeypatch, {"module_page:mcp": "concept-mcp"})
        derived, enforced = reconcile_module_page_ids(**args)
        assert derived == {"module_page:mcp": "concept-mcp"}
        assert enforced == {"module_page:mcp"}

    def test_report_mode_derives_but_enforces_nothing(self, repo, monkeypatch):
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "report")
        args = self._args(repo, monkeypatch, {"module_page:mcp": "concept-mcp"})
        derived, enforced = reconcile_module_page_ids(**args)
        assert derived == {"module_page:mcp": "concept-mcp"}
        assert enforced is None

    def test_off_mode_does_not_even_derive(self, repo, monkeypatch):
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "off")

        def _boom(**_kw):
            raise AssertionError("off mode must not derive")

        monkeypatch.setattr(
            "repowise.core.generation.selection.module_reconcile.authoritative_module_pages",
            _boom,
        )
        derived, enforced = reconcile_module_page_ids(
            repo_path=repo,
            parsed_files=[],
            graph_builder=object(),
            git_meta_map={},
            kg_modules=None,
            cfg={},
            concurrency=2,
            degraded=[],
        )
        assert (derived, enforced) == ({}, None)

    def test_a_failed_derivation_enforces_nothing_and_degrades(self, repo, monkeypatch):
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "on")

        def _boom(**_kw):
            raise RuntimeError("grouping exploded")

        monkeypatch.setattr(
            "repowise.core.generation.selection.module_reconcile.authoritative_module_pages",
            _boom,
        )
        degraded: list[str] = []
        derived, enforced = reconcile_module_page_ids(
            repo_path=repo,
            parsed_files=[],
            graph_builder=object(),
            git_meta_map={},
            kg_modules=None,
            cfg={},
            concurrency=2,
            degraded=degraded,
        )
        assert (derived, enforced) == ({}, None)
        assert degraded == ["Module page reconcile: grouping exploded"]

    def test_an_empty_derivation_enforces_nothing(self, repo, monkeypatch):
        """Empty is a legal answer and an under-derivation. Refuse to act on it."""
        monkeypatch.setenv("REPOWISE_MODULE_RECONCILE", "on")
        args = self._args(repo, monkeypatch, {})
        derived, enforced = reconcile_module_page_ids(**args)
        assert (derived, enforced) == ({}, None)


# ---------------------------------------------------------------------------
# Rendering the pages the reconcile would otherwise only retire
#
# These tests are synchronous: ``render_missing_module_pages`` opens its own
# event loop the way every other CLI helper does, so an async test would be
# calling ``asyncio.run`` from inside a running loop.
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(self, page_id: str) -> None:
        self.page_id = page_id
        self.page_type = "module_page"


class _FakeParsed:
    def __init__(self, path: str) -> None:
        self.file_info = SimpleNamespace(path=path)


def _seed_sync(repo: Path, *targets: str) -> None:
    asyncio.run(_seed_module_pages(repo, *targets))


def _render_args(repo: Path, module_ids: dict[str, str], degraded: list[str]) -> dict:
    return dict(
        repo_path=repo,
        parsed_files=[_FakeParsed("a.py"), _FakeParsed("pkg/b.py")],
        source_map={},
        graph_builder=object(),
        repo_structure=object(),
        git_meta_map={},
        module_ids=module_ids,
        cfg={},
        concurrency=2,
        degraded=degraded,
    )


def test_missing_module_page_is_rendered_and_persisted(repo, monkeypatch):
    from repowise.cli.commands.update_cmd import deterministic as det

    _seed_sync(repo, "mcp")
    calls: list[dict] = []
    persisted: list[list] = []

    def _fake_render(**kwargs):
        calls.append(kwargs)
        return [_FakePage("module_page:codeatlas/doc_search")]

    monkeypatch.setattr(det, "_render_pages", _fake_render)
    monkeypatch.setattr(
        det,
        "persist_deterministic_pages",
        lambda **kw: (persisted.append(kw["generated_pages"]), 7)[1],
    )

    rendered, total = det.render_missing_module_pages(
        **_render_args(
            repo,
            {"module_page:mcp": "concept-mcp", "module_page:codeatlas/doc_search": "k2"},
            [],
        )
    )

    assert (rendered, total) == (1, 7)
    assert len(calls) == 1
    assert calls[0]["only_page_ids"] == {"module_page:codeatlas/doc_search"}
    assert calls[0]["file_pages_only"] is False
    # The whole parse, not the changed slice: a module page written from a
    # corner of its subsystem is worse than no page at all.
    assert calls[0]["regenerate_paths"] == ["a.py", "pkg/b.py"]
    assert [p.page_id for p in persisted[0]] == ["module_page:codeatlas/doc_search"]


def test_no_drift_renders_nothing(repo, monkeypatch):
    """A quiet commit must not pay for a full-repo context pass."""
    from repowise.cli.commands.update_cmd import deterministic as det

    _seed_sync(repo, "mcp")

    def _boom(**_kw):
        raise AssertionError("no drift, so no render")

    monkeypatch.setattr(det, "_render_pages", _boom)

    assert det.render_missing_module_pages(
        **_render_args(repo, {"module_page:mcp": "concept-mcp"}, [])
    ) == (0, None)


def test_a_drifted_structural_key_is_re_rendered(repo, monkeypatch):
    from repowise.cli.commands.update_cmd import deterministic as det

    _seed_sync(repo, "mcp")
    calls: list[dict] = []
    monkeypatch.setattr(
        det,
        "_render_pages",
        lambda **kw: (calls.append(kw), [_FakePage("module_page:mcp")])[1],
    )
    monkeypatch.setattr(det, "persist_deterministic_pages", lambda **_kw: 3)

    rendered, _total = det.render_missing_module_pages(
        **_render_args(repo, {"module_page:mcp": "concept-moved"}, [])
    )

    assert rendered == 1
    assert calls[0]["only_page_ids"] == {"module_page:mcp"}


def test_an_unkeyed_stored_page_is_not_drift(repo, monkeypatch):
    """A page written before the structural stamp would re-render forever."""
    from repowise.cli.commands.update_cmd import deterministic as det

    async def _seed_unkeyed() -> None:
        engine = create_engine(resolve_db_url(repo))
        try:
            await init_db(engine)
            async with get_session(create_session_factory(engine)) as session:
                r = await upsert_repository(session, name=repo.name, local_path=str(repo))
                await upsert_page(
                    session,
                    page_id="module_page:mcp",
                    repository_id=r.id,
                    page_type="module_page",
                    title="mcp",
                    content="# mcp",
                    target_path="mcp",
                    source_hash="h",
                    model_name="template",
                    provider_name="template",
                    structural_key=None,
                )
        finally:
            await engine.dispose()

    asyncio.run(_seed_unkeyed())

    def _boom(**_kw):
        raise AssertionError("an unkeyed page is not drift")

    monkeypatch.setattr(det, "_render_pages", _boom)

    assert det.render_missing_module_pages(
        **_render_args(repo, {"module_page:mcp": "concept-mcp"}, [])
    ) == (0, None)
