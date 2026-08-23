"""Fixtures for the reference-site store.

The toy repository is written out in full here rather than assembled per test,
because every acceptance claim in this directory is stated as a hand count
against *this* text. If the fixture drifts, the counts must drift with it and
the tests must be re-derived — that is deliberate.

Each file is annotated with what it is there to prove.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from repowise.core.persistence.crud import upsert_repository
from repowise.core.persistence.database import create_engine, init_db
from repowise.core.refsites.schema import ensure_schema

# --------------------------------------------------------------------------
# The toy repository
# --------------------------------------------------------------------------

CALC_TS = """\
export function computeTotal(n: number): number {
  return n + 1;
}
"""

# Proves, in order: an import binding; an ordinary call; TWO calls to the same
# name on one line (the parse layer dedups these on ``(line, target, receiver)``
# and keeps one); a symbol declared inside an anonymous callable (the parse
# layer deletes it, so neither its definition nor the call to it reaches the
# graph); and a call whose enclosing scope is an arrow, which the parse layer
# attributes to no symbol at all.
WIDGET_TS = """\
import { computeTotal } from "./calc";

export function widgetTotal(n: number): number {
  return computeTotal(n);
}

const handlers = [() => computeTotal(1), () => computeTotal(2)];

describe("suite", () => {
  const makeUser = () => ({ id: 1 });
  it("works", () => {
    expect(makeUser()).toBeTruthy();
    computeTotal(3);
  });
});
"""

# Proves the `.tsx` grammar variant is the same `typescript` language tag.
PANEL_TSX = """\
import { computeTotal } from "./calc";

export const Panel = () => {
  const v = computeTotal(2);
  return <div>{v}</div>;
};
"""

# Proves javascript, and a destructuring `require` whose captured statement
# text does not start its line.
LEGACY_JS = """\
const { computeTotal } = require("./calc");

function legacyTotal(n) {
  return computeTotal(n);
}

const cb = () => computeTotal(9);
"""

HELPERS_PY = """\
def compute_total(x):
    return x + 1
"""

# Proves, in order: a plain call; a call inside a lambda bound to a constant
# (the graph attributes it to the constant, not the lambda); a call inside a
# lambda passed as an argument; TWO calls on one line; a call to a name nothing
# defines; and the target's name inside a string literal.
APP_PY = """\
import os
from helpers import compute_total


def compute_total_wrapper(x):
    return compute_total(x)


HANDLERS = {"a": lambda v: compute_total(v)}


def run(items):
    return list(map(lambda i: compute_total(i), items))


def paired(a, b):
    return compute_total(a) + compute_total(b)


class Report:
    def render(self):
        return compute_total(1)


def orphan():
    return not_a_real_function(3)


def yaml_only_helper():
    return 7


NAME_IN_STRING = "compute_total"
"""

DEPLOY_SH = """\
#!/usr/bin/env bash
source ./lib.sh

compute_total() {
  echo "$1"
}

run_all() {
  compute_total 3
  compute_total 4
}
"""

SCHEMA_SQL = """\
CREATE TABLE totals (id INTEGER PRIMARY KEY, amount INTEGER);

CREATE VIEW total_summary AS
SELECT id, amount FROM totals;
"""

# The uncovered language. It names `yaml_only_helper` and `build_matrix`;
# neither can ever produce a site, which is exactly what the coverage
# disclosure has to say out loud.
PIPELINE_YAML = """\
jobs:
  build_matrix:
    steps:
      - run: python -c "import app; app.yaml_only_helper()"
      - name: compute_total
"""

TOY_REPO: dict[str, str] = {
    "calc.ts": CALC_TS,
    "widget.ts": WIDGET_TS,
    "panel.tsx": PANEL_TSX,
    "legacy.js": LEGACY_JS,
    "helpers.py": HELPERS_PY,
    "app.py": APP_PY,
    "deploy.sh": DEPLOY_SH,
    "schema.sql": SCHEMA_SQL,
    "pipeline.yaml": PIPELINE_YAML,
}

#: The symbol the rename-preview ground truth is stated against.
TARGET_SYMBOL_ID = "calc.ts::computeTotal"

#: Hand-counted from WIDGET_TS/PANEL_TSX/LEGACY_JS/CALC_TS above:
#: ``(file, line, kind)`` for every site a rename of ``computeTotal`` touches.
#: The two entries on widget.ts line 7 are the pair the parse layer collapses.
COMPUTE_TOTAL_GROUND_TRUTH: tuple[tuple[str, int, str], ...] = (
    ("calc.ts", 1, "definition"),
    ("legacy.js", 1, "import_binding"),
    ("legacy.js", 4, "call"),
    ("legacy.js", 7, "call"),
    ("panel.tsx", 1, "import_binding"),
    ("panel.tsx", 4, "call"),
    ("widget.ts", 1, "import_binding"),
    ("widget.ts", 4, "call"),
    ("widget.ts", 7, "call"),
    ("widget.ts", 7, "identifier"),
    ("widget.ts", 13, "call"),
)


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    """Write the toy repository to disk and return its root.

    The bare ``.git`` directory gives the traverser a repo root WITHOUT being
    a real repository, and that non-git-ness is load-bearing: every tool test
    on this fixture exercises the permanent working-tree-unverifiable path
    (``not_a_git_repository``), which is why
    ``test_rename_preview_declares_that_it_changes_nothing`` asserts full
    demotion. Do not "fix" this into a real checkout — the verified-tree
    semantics have their own owner in ``test_tool_freshness.py``, against a
    genuinely committed repo.
    """
    for name, body in TOY_REPO.items():
        (tmp_path / name).write_text(body)
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
async def async_engine():
    """In-memory SQLite with the shared schema plus this package's tables.

    Built through the project's own ``create_engine`` rather than SQLAlchemy's,
    so the connect-time pragmas apply — ``foreign_keys=ON`` in particular. A
    bare ``create_async_engine`` leaves foreign keys off, which would let a
    cascade test pass or fail for reasons that have nothing to do with the
    schema under test.
    """
    engine = create_engine("sqlite+aiosqlite:///:memory:", use_static_pool=True)
    await init_db(engine)
    await ensure_schema(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session(async_engine):
    factory = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session


@pytest.fixture
async def repo_id(async_session) -> str:
    repo = await upsert_repository(
        async_session,
        name="refsites-toy",
        local_path="/tmp/refsites-toy",
        url="https://example.invalid/refsites-toy",
    )
    await async_session.commit()
    return repo.id


@pytest.fixture
async def mcp_state(async_engine, async_session, repo_id):
    """Point the MCP server's single-repo globals at the test database.

    The package forwards attribute writes to ``_state``, so assigning on the
    package is the supported way to configure single-repo mode in a test.
    """
    import repowise.server.mcp_server as mcp_mod

    factory = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
    mcp_mod._session_factory = factory
    mcp_mod._repo_path = "/tmp/refsites-toy"
    yield repo_id
    mcp_mod._session_factory = None
    mcp_mod._repo_path = None
    mcp_mod._registry = None
