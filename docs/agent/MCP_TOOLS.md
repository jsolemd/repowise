# MCP Tools Reference

repowise exposes a curated set of tools via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). These tools give AI coding assistants (Claude Code, Codex, Cursor, Cline, Windsurf) structured access to your codebase intelligence: dependency graph, git history, documentation, and architectural decisions.

29 tools are registered in total. A single-repo server advertises 12 by default: the eleven flagship tools below plus `list_repos`. Workspace mode adds 2 more automatically (`get_architecture`, `get_blast_radius`), for 14. Fifteen further tools are off by default everywhere and must be opted in. The surface is configurable; see [Configuring the tool surface](#configuring-the-tool-surface).

**Start the MCP server:**

```bash
repowise mcp --transport stdio           # for Claude Code, Codex, Cursor, etc.
repowise mcp --transport streamable-http # for HTTP clients on port 7338
repowise mcp --transport sse --port 7338 # legacy SSE transport
```

**Auto-setup:** `repowise init` automatically registers the MCP server and installs proactive hooks for Claude Code. `repowise init --codex` writes project-local Codex MCP config and hooks.

**Opting out:** each Claude config holds a single `repowise` MCP key, so indexing a second repo repoints it rather than adding a second entry. Pass `repowise init --no-editor-setup` (or set `REPOWISE_SKIP_EDITOR_SETUP=1`) for a repo you do not want registered: a scratch clone, a worktree, a CI or benchmark run. Nothing about the index changes, and re-running `repowise init` without the flag registers it later. `init` also prints a notice when it is about to repoint an existing entry.

---

## Contents

**Default tools (single-repo, 12)**
[get_overview](#get_overview) &middot;
[get_answer](#get_answer) &middot;
[get_context](#get_context) &middot;
[get_symbol](#get_symbol) &middot;
[search_codebase](#search_codebase) &middot;
[get_risk](#get_risk) &middot;
[get_change_risk](#get_change_risk) &middot;
[get_why](#get_why) &middot;
[get_dead_code](#get_dead_code) &middot;
[get_health](#get_health) &middot;
[get_index_status](#get_index_status) &middot;
[list_repos](#list_repos)

**Workspace-only tools (added automatically, 2)**
[get_architecture](#get_architecture) &middot;
[get_blast_radius](#get_blast_radius)

**Opt-in tools (off by default everywhere, 15)**
[get_dependents](#get_dependents) &middot;
[get_dependency_path](#get_dependency_path) &middot;
[get_execution_flows](#get_execution_flows) &middot;
[generate_refactoring_code](#generate_refactoring_code) &middot;
[get_conformance](#get_conformance) &middot;
[reindex_repository](#reindex_repository) &middot;
[build_task_slice](#build_task_slice) &middot;
[get_task_slice](#get_task_slice) &middot;
[extend_task_slice](#extend_task_slice) &middot;
[find_clones](#find_clones) &middot;
[find_patterns](#find_patterns) &middot;
[get_query_quality](#get_query_quality) &middot;
[manage_decision](#manage_decision) &middot;
[get_reference_sites](#get_reference_sites) &middot;
[preview_symbol_rename](#preview_symbol_rename)

Also see [Configuring the tool surface](#configuring-the-tool-surface), [Reversible truncation](#reversible-truncation-_metaomitted) and [Unrecognised arguments](#unrecognised-arguments-ignored_arguments).

---

## The eleven flagship tools

| Tool | Purpose | Typical use |
|------|---------|-------------|
| `get_overview` | Architecture summary | First call on any unfamiliar codebase |
| `get_answer` | One-call RAG Q&A | First call on any code question |
| `get_context` | Rich context for targets | Before reading or modifying code |
| `get_symbol` | Raw source bytes for one symbol | When you need one function/class body |
| `search_codebase` | Hybrid symbol / path / concept search | Finding a symbol or file, or discovering code by topic |
| `get_risk` | Modification risk | Before changing hotspot files |
| `get_change_risk` | Live commit or range risk | Before merging a commit or PR range |
| `get_why` | Architectural decisions | Before structural changes |
| `get_dead_code` | Unreachable code | Cleanup tasks |
| `get_health` | Code-health marker scores | Before refactoring, find the worst files |
| `get_index_status` | Source-search publication trust | Before relying on indexed search results |

Also always on by default: `list_repos` (repo aliases). See [Supplementary tools](#supplementary-tools).

---

## Configuring the tool surface

The default surface is deliberately small: fewer, richer tools mean fewer round-trips and less schema overhead per task. What a server advertises is resolved from three things: each tool's `default`/`requires_workspace` metadata, whether the server is in workspace mode, and an optional override.

- **Default (single-repo):** 12 tools, the eleven flagship tools plus `list_repos`.
- **Default (workspace):** those 12 plus `get_architecture` and `get_blast_radius`, added automatically when the server starts inside a workspace. They are never advertised outside one.
- **Opt-in tools:** `get_dependents`, `get_dependency_path`, `get_execution_flows`, `generate_refactoring_code`, `get_conformance`, `reindex_repository`, `build_task_slice`, `get_task_slice`, `extend_task_slice`, `get_query_quality`, `find_clones`, `find_patterns`, `manage_decision`, `get_reference_sites`, and `preview_symbol_rename` are registered but off by default. Turn them on per repo; `get_conformance` only does useful work in workspace mode (name it there), and `manage_decision` only where a decision journal is configured. `reindex_repository` remains separate from the default read surface because it starts background work.

**Configure it in `.repowise/config.yaml`** under an `mcp.tools` key. Four shapes are supported:

```yaml
# Adjust the default set with + / - deltas (the common case):
mcp:
  tools: ["+get_execution_flows", "-get_dead_code"]

# Or give an explicit allowlist (only these tools):
mcp:
  tools: ["get_answer", "get_context", "get_symbol", "search_codebase"]

# Or enable everything available in the current mode:
mcp:
  tools: all

# Or select the agent-lean profile (see below):
mcp:
  tools: lean
```

**Or per launch on the CLI**, which overrides the config block:

```bash
repowise mcp --tools "+get_execution_flows"          # default set plus one
repowise mcp --tools "get_answer,get_context"         # explicit allowlist
repowise mcp --tools lean                             # agent-lean profile
repowise mcp --all                                    # every available tool
```

Workspace-only tools named explicitly in single-repo mode are ignored (they cannot do useful work there). Unknown tool names are ignored with a warning.

**The `lean` profile** is the agent-lean surface: `get_answer`, `get_context`, `get_symbol`, `search_codebase`, `get_risk`, and `get_why`, plus `list_repos` in workspace mode (where repo aliases must be discoverable). `get_why` is part of the lean set because why/history questions are the category no code-search surface can answer from the tree alone; a lean profile without it measurably underperforms on exactly those questions. The profile advertises ~2.1k tokens of schema versus ~4.1k for the default surface. That is small enough to keep always loaded, so when a repo has `mcp.tools: lean` configured, `repowise init` skips the tool-search recommendation (the `ENABLE_TOOL_SEARCH` setting that defers MCP schemas behind a lookup round trip) for Claude Code; the six schemas the agent actually reaches for stay in context on every turn. init never turns an existing `ENABLE_TOOL_SEARCH` setting off, since it applies to every MCP server, not just repowise.

**Or from the dashboard:** the Settings page lists every tool with its description and a per-repo toggle, and writes the same `mcp.tools` config for you.

---

## Reversible truncation: `_meta.omitted`

Tool responses are token-budgeted. When a response is truncated, the dropped
content is no longer silently lost: it is stored in the repo's
[omission store](DISTILL.md#the-omission-store) and the response's `_meta`
envelope lists how to get it back:

```jsonc
"_meta": {
  "omitted": {
    "refs": ["a1b2c3d4e5f6"],
    "tokens": 5840,
    "restore": "repowise expand <ref> (CLI) or get_symbol(\"repowise#<ref>\", query?) (MCP)"
  }
}
```

Truncated skeleton blocks are replaced in place by a `[repowise#<ref>: ...]`
marker; everything else is captured into one combined document per response.
Resolve refs with `repowise expand <ref>` from a shell, or
`get_symbol("repowise#<ref>")` from any MCP client. See
[DISTILL.md](DISTILL.md) for the full reversibility model.

**The `_meta` envelope** (all fields optional, present only when meaningful):

| Field | When present |
|-------|--------------|
| `timing_ms` | Tool wall-time |
| `hint` | A short, conservative follow-up suggestion |
| `cached` | Only when `true` |
| `index_age_days` | Days since the last `repowise update` |
| `indexed_commit` | Short (12-char) SHA the index was built against |
| `live_head` | Only when it differs from `indexed_commit` |
| `stale_warning` | Only on a real signal: HEAD mismatch **that actually changed files**, or age over ~90 days when git is unreachable. Two commits with identical trees (an empty commit, a no-op merge) report `index_behind` with no warning |
| `index_behind` | Whenever the live-vs-indexed comparison ran: `true` if HEAD has moved (alongside `stale_warning` when served content actually changed), `false` if the commits match. Absent means the comparison could not run (no git, or a repo-level tool that serves no file content) |
| `embedder_degraded` | Whenever an embedder is resolved, `true` or `false`. Absent means none was initialised |
| `embedder`, `embedder_warning` | Only when the embedder fell back to a mock/degraded mode |

Silence on `stale_warning` means the index is current; don't infer staleness from its absence. `list_repos`, `get_architecture`, `get_blast_radius`, and `get_conformance` don't carry a freshness envelope at all.

---

## Unrecognised arguments: `ignored_arguments`

A tool never answers a bad argument with a filter that matches nothing. A value
outside a closed vocabulary is **dropped, not applied** — so the response is the
one you would have got without it — and the tool names what it dropped, at the
top level:

```jsonc
"ignored_arguments": [
  { "argument": "kind",
    "values": ["unused_exports"],
    "valid": ["unreachable_file", "unused_export", "unused_internal", "zombie_package"] }
]
```

The key is absent when every argument was understood, so its presence is the
whole signal. One entry per argument, however many of its values missed.

This exists because the alternative is a lie: `get_dead_code(kind="unused_exports")`
used to filter on the plural, match nothing, and recommend *"No dead code found
matching your filters."* beside a summary counting hundreds of unused exports
([#1496](https://github.com/repowise-dev/repowise/issues/1496)). It covers
`get_dead_code` (`kind`, `tier`, `min_confidence`), `get_context` (`include`),
`get_dependency_path` (`mode`), and `search_codebase` (`kind`).

`get_dead_code`'s `min_confidence` additionally accepts the tier names the
response is organised by — `"high"` (0.8), `"medium"` (0.5), `"low"` (0.0) — as
well as a float. `get_health` reports the same thing under its own older name,
`unknown_only_keys`, for the `only` projection.

---

## `get_overview`

Architecture summary, module map, entry points, git health, and community summary.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` |
| `include` | list[string] | No | `"content"` returns the full overview essay in `content_md` instead of the compact summary section |

**Returns:** Architecture description, key modules with purpose and owner, entry points, tech stack, hotspot files, knowledge silos, community summary (top communities by size with labels and cohesion scores). `content_md` is compact by default (summary + tech stack + layers); pass `include=["content"]` for the full essay.

**When to use:** First call on any unfamiliar codebase. Gives the agent a mental map before diving into specifics. Skip on later calls in the same session; it doesn't change mid-session.

**Example calls:**

```
get_overview()
get_overview(include=["content"])
```

---

## `get_answer`

One-call RAG: retrieves over the wiki, gates synthesis on confidence, and returns a cited 2-5 sentence answer.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | Natural language question about the codebase |
| `repo` | string | No | *(workspace only)* Target repo alias |

**Returns:** A synthesized answer with file/symbol citations and a confidence label (`high`, `medium`, `low`). High-confidence answers can be cited directly. Low-confidence answers return ranked wiki excerpts instead.

Two path-bearing blocks, with different jobs:

| Field | Job | Confidence-gated? |
|-------|-----|-------------------|
| `retrieval` | **Evidence.** Enriched hits (summary, snippet, key symbols) to re-read when the prose needs checking. Shrinks as confidence rises, because a trustworthy answer needs less of it. | Yes |
| `candidates` | **Navigation.** The ranked shortlist of files retrieval resolved, one `{path, lines?}` entry each, up to 20. | No |

`candidates` is present whenever retrieval resolved anything, including on high-confidence answers where `retrieval` is deliberately empty. It is where to look next; it is not evidence that the answer is right.

**Retrieval legs:** three, fused by Reciprocal Rank Fusion: full-text and vector search over wiki pages, plus the structural symbol index. The symbol leg is keyed on the content words of the question rather than on whether it happens to carry an identifier-shaped token, so "how does an incremental update persist symbols" reaches the same rows as `_persist_symbols`. It exists because a generated file page renders only the *public* symbol table: a private helper or a local name is not in the text the other two legs index.

**When to use:** First call on any code question. Collapses search, read, and reason into one round-trip. If confidence is low, follow up with `search_codebase` to discover candidate pages.

**Example call:**

```
get_answer(question="How does the authentication flow work?")
```

---

## `get_context`

The workhorse tool. Returns docs, symbols, ownership, freshness, and community membership for any combination of files, modules, or symbols.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `targets` | list[string] | Yes | File paths, module names, or symbols. Batch multiple targets in one call. Symbol targets take exactly the forms `get_symbol` accepts and resolve through the same ladder, so an id from either tool works in the other: a full `"path/to/file.py::Name"`, a qualified `"Class.method"` / `"Class::method"` / `"pkg.mod.Class.method"`, or a bare name. |
| `include` | list[string] | No | Additional data to include: `"full_doc"` (full wiki markdown), `"callers"` (who calls this, symbol targets), `"callees"` (what this calls, symbol targets), `"ownership"` (primary owner, bus factor, contributor count), `"last_change"` (last commit date + author), `"metrics"` (PageRank, betweenness, percentiles), `"community"` (cluster membership + neighbors), `"decisions"` (full decision records; default returns titles only), `"skeleton"` (file targets only; the file with bodies elided: every signature, imports, and the bodies of the most central symbols, token-budgeted; typically ~15% of the full file's tokens) |
| `compact` | boolean | No | Default `true`. Set `false` for full structure block and importer list. |
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` |

**Returns per target:** Documentation summary, symbols defined, ownership percentages, freshness score, co-change partners, architectural decisions governing the file. With `include` options: source code, call graph, graph metrics, community membership.

A file with no indexed symbols (README, config, plain data) gets a
`docs.file_preview` instead of an empty symbol list: line and character counts,
plus the heading spine for markdown or the first non-blank lines otherwise.
Counts and verbatim excerpts only, nothing inferred.

When the symbol half of a `path::Name` target does not resolve but the file
half does, the reply is that file's card with `resolved_to` naming the file and
a `note` saying which symbol was not found. The file's symbol list is where the
correct id is, so this is a partial answer rather than a dead end.

**Test linkage.** Every file and symbol card carries `tested` (bool) and
`test_linkage_basis`. Authoritative `coverage` or `graph` evidence adds
`guarding_tests` (up to 10 paths) with `guarding_test_count`; naming-only
conventions instead add `possible_tests` / `possible_test_count` and leave
`tested: false`. `get_risk`'s `test_gap` is the negation of the same `tested`,
read from the same resolver, so the two tools cannot disagree about a file.
The basis says how strong the claim is:
`coverage` (a coverage run proved these tests execute the file), `graph`
(these test files import it), `naming` (nothing references it, but a
conventionally-named test file exists and no same-stem source collision makes
attribution ambiguous — possible evidence, never a cleared gap), `self` (the
file is itself test material), or `none`. The `health` block's
`has_test_file` is a *different* question — the index-time paired-filename
heuristic that feeds the health score — and carries a note when it diverges
from the linkage.

**Ambiguity.** A name matching several symbols returns `status: "ambiguous"`,
an exact `match_count`, and up to 20 `candidates`
(`symbol_id`/`file`/`name`/`qualified_name`/`kind`/`start_line`). The card
still describes the first match and the `note` says which, so the reply is
usable, but nothing is presented as *the* answer. Requested symbol-specific
`callers`/`callees`/`metrics`/`community`/`health`/`skeleton` blocks are omitted
until one candidate is selected; `enrichment_omitted` names those blocks.

**A path that does not exist** returns `status: "not_found"` with
`suggestions`: files under the target if it looked like a directory, then a
filename match, then the closest indexed paths by edit distance — so a typo
(`src/auth/servce.py`) comes back with `src/auth/service.py` rather than a bare
error. When nothing resembles it, the error says so explicitly.

**When to use:** Before reading or modifying code. Pass all relevant targets in one call to minimize round-trips. In workspace mode, enriched with cross-repo co-change and contract data.

**Example calls:**

```
get_context(targets=["src/auth/middleware.ts"])
get_context(targets=["middleware", "api/routes", "payments"], include=["callers", "metrics"])
get_context(targets=["src/auth"], compact=false, include=["community"])
get_context(targets=["src/big_module.py"], include=["skeleton"])
```

**Skeletons:** with `include=["skeleton"]`, file targets gain a structure-level
rendering sliced from the index's persisted symbol bounds (no parsing at query
time): every signature, the import preamble, and the bodies of the top symbols
ranked by graph centrality / hotspot / query match. Elision markers carry
1-indexed line ranges so you can range-`Read` anything back. For
structure-level questions ("what's in this file", "which function handles X")
this replaces a full file read at a fraction of the cost.

---

## `get_symbol`

Raw source bytes for one indexed symbol with exact line bounds, cheaper and
safer than `Read` + offset math. The only tool that returns actual source code.
Also resolves **omission refs** (`repowise#<12-hex>`) from truncated responses.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol_id` | string \| list[string] | Yes | A target, or a list of up to 20 (see **Batching** below). A target is `"path/to/file.py::SymbolName"` (canonical, from `get_context`'s symbol list; normalises `::` / `.` / `/` separators across languages), a qualified name with no path (`"AuthService.login"`, `"AuthService::login"`, `"auth.service.AuthService.login"`), a bare name (`"reconcile_symbols_for_files"`), `"path/to/file.py:140-180"` (a live range read, 200 lines max), or an omission ref `"repowise#<12-hex>"` / a pasted whole `[repowise#...]` marker. |
| `query` | string | No | Omission refs only: return just the stored lines matching this regex (or substring). Ignored for symbol ids and range reads. |
| `context_lines` | int | No | Extra source lines before/after the symbol (0-50, default 0) |
| `depth` | int | No | Follow the call graph outward from this symbol and include what it calls, with bodies (1-3, default 1 = this symbol only). Out-of-range values clamp. |
| `repo` | string | No | *(workspace only)* Usually omitted; `"all"` is not supported |

**Returns:** For a symbol id or range: the source (bounded at ~600 lines,
each line prefixed with its file line number in the same format as a `Read`
result), its exact start/end line numbers, kind, and a `truncated` flag; on a
miss, an `error` with the closest matches (`fallback_lines` from a live grep).
When several indexed symbols match the target (overloads, re-exports,
conditional definitions, or a leaf name that is simply common) the response has
`status: "ambiguous"`, `ambiguous: true`, an exact `match_count`, and a
`candidates` list — none is silently chosen. Each candidate carries
`symbol_id`, `file`, `name`, `qualified_name`, `kind`, and `start_line`.
Path-less name queries carry bodies with four or fewer matches and otherwise
carry `fetch_with`; at most 20 name candidates are materialized (a bare
`__init__` matches 123 symbols on a real index, and 123 bodies answer nothing).
Path-qualified overload sets retain the legacy envelope: every candidate is
represented, with bodies until the byte budget and `not_rendered` range reads
beyond it. In both forms `match_count` is the exact eligible total, independent
of the candidate cap. For an omission ref: the stored content plus provenance
(`source`, `created_at`, `original_tokens`).

**Resolution order.** Path-qualified rungs first — exact `symbol_id`, then
`(file, qualified_name)`, then `(file, name)`, then a file-path suffix match so
a remembered filename (`answer.py::get_answer`) resolves. If the target names a
file (it contains a `/` or its first segment ends in an extension or special
filename recognized by the language registry)
resolution stops there: `nope/wrong.py::alpha` returns retryable `suggestions`
rather than an `alpha` from some other file, because the caller asserted a path
and meant it. Otherwise the whole target is retried as a name: exact
`qualified_name`, then a qualified tail on a separator boundary
(`Class.method` under `pkg.mod.Class.method`), then the bare `name`, then the
leaf segment alone. Only if all of that comes back empty is the ladder walked
again case-insensitively, so an exact match always outranks a case-folded one
in languages where `Foo` and `foo` are two symbols.

**Batching.** Pass a list to fetch several targets in one round trip. The reply
becomes `{"results": [...], "count", "resolved_count", "ambiguous_count"}` —
one entry per target, in request order, each the same shape a single-target
call returns, plus the `target` string it came from. `resolved_count` counts
only uniquely resolved successes; ambiguity is counted separately. Per-item
`_meta` is folded into one outer envelope whose freshness is evaluated over the
union of canonical served files. When target-scoped checks identify changed
files, `_meta.stale_targets` lists them compactly beside the any-target
`stale_warning`; `replaced_tokens` is summed. A single target returns the flat
shape, unchanged. Over 20 targets, the first 20 are served and the rest are
named in `not_served` rather than the call failing.

With `depth` above 1 the response also carries `callee_bodies`: the symbols
this one calls, transitively, each with its `depth` (hops from the root), its
source, and a `verified` flag. Every symbol appears once, at the shallowest
depth it was reached from. Callees past the response budget are listed in
`not_rendered` with the `fetch_with` range that retrieves them, so a bounded
walk never looks like a complete one.

**When to use:** When you need the body of one function or class: pipe the
`symbol_id` straight from `get_context`'s symbol list. Use the line-range form
for anything that falls between symbols. Or when a response's `_meta.omitted`
lists refs you want back and you have no shell for `repowise expand` (e.g.
Claude Desktop).

Reach for `depth=2` when you are following a call chain: reading a body,
finding the next name in it, then fetching that one. The graph already holds
those edges before the first call, so one `depth=2` call replaces the whole
sequence of round trips.

**Example calls:**

```
get_symbol(symbol_id="src/auth/service.py::AuthService")
get_symbol(symbol_id="src/auth/service.py::login", context_lines=10)
get_symbol(symbol_id="src/auth/service.py::login", depth=2)
get_symbol(symbol_id="src/auth/service.py:140-180")
get_symbol(symbol_id="AuthService.login")          # qualified, no path
get_symbol(symbol_id="reconcile_symbols_for_files") # bare name
get_symbol(symbol_id=["AuthService.login", "src/db/models.py::User"])
get_symbol(symbol_id="repowise#a1b2c3d4e5f6")
get_symbol(symbol_id="repowise#a1b2c3d4e5f6", query="FAILED")
```

---

## `search_codebase`

Hybrid code search over repowise's indexes. A single tool that, depending on
the shape of the query, searches the indexed **symbols**, **file paths**, or
the **wiki**, instead of forcing a fallback to Grep for identifiers.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Identifier, path, or natural-language query |
| `limit` | int | No | Max results (default 5) |
| `mode` | string | No | `auto` (default) \| `concept` \| `symbol` \| `path` \| `hybrid` |
| `kind` | string | No | `implementation` \| `test` \| `config` \| `doc` |
| `symbol_kind` | string | No | Restrict symbol hits by kind (`function`, `class`, `method`, ...) |
| `page_type` | string | No | Restrict to one page type. The two you will reach for are `file_page` (the always-on per-file docs) and `module_page` (the subsystem/concept pages). Other stored types (`repo_overview`, `layer_page`, `scc_page`, `api_contract`, `infra_page`, `symbol_spotlight`) also filter. |
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` to search across the workspace. Against a single-repo server `"all"` means the same as omitting it — there is one repo, and all of it is what an unqualified search returns. |

**Modes:**

- **`auto`** (default) routes by query shape:
  - an **identifier** (`GitIndexer`, `index_repo`) -> searches indexed symbols;
  - a **path** (`core/ingestion/indexer.py`) -> searches file pages;
  - **prose** ("how do we handle retries?") -> wiki-semantic search;
  - mixed prose + identifier -> **hybrid** (symbol hits first, then concept pages).
- **`concept`** forces the original wiki-semantic behavior.
- **`symbol`** / **`path`** force the structural search.

**Returns:**

- *Symbol hits*: `{type: "symbol", symbol_id, name, kind, file, start_line, end_line, signature, next: "get_symbol"}`. Ranked by exact-name/qualified-name match, query-token coverage, then graph centrality (PageRank / betweenness / entry-point); non-test before test unless `kind="test"`.
- *File hits*: `{type: "file", page_id, file, title, next: "get_context"}`.
- *Concept hits*: ranked wiki pages with `relevance_score`, `snippet`, `target_path`, and a `search_method` (`embedding` vs `bm25` fallback). A `symbol_spotlight` page's `target_path` is a page identifier of the form `file.py::Symbol`; those hits also carry `file` with the openable path. **Read `file` when present.** `target_path` is for piping into `get_symbol`, not for opening.

Alongside `results`, the response carries **`candidates`**: up to `limit`
distinct files worth opening next, best first — one `{path}` entry each, plus a
`repo` on a federated workspace call (see below).

Every entry is a real file path, and that is the difference between the two
blocks. `results` ranks *pages*, and a page is not always a file: a
`module_page` is named by a structural group key that reads exactly like a
directory, an `scc_page` by `scc-<hash>`, an `onboarding` page by a slot name.
Ranking those is correct; opening them is not. `candidates` resolves symbol
pages to their file, collapses several symbols of one file to a single entry,
skips every page that names no file, and backfills from below the result
window so a slot spent on a module page does not also cost you a file.

**If your next move is a Read, read `candidates`.** If you are enumerating
matches or resolving a `symbol_id`, read `results`.

Tombstoned and `exclude_patterns`-excluded results are filtered. In workspace
mode, structural and concept searches both federate across repos and merge
(this is the one tool where `repo="all"` is fully supported). **On that
federated call** — `repo="all"` against a workspace, and only that — every
result row and every `candidates` entry carries its `repo`, and identical
relative paths in two repos stay two distinct candidates, because the path
alone is not openable in a workspace. A call scoped to one repo
(`repo=<alias>`) answers from that repo alone and carries no `repo` key, and
neither does any single-repo response.

The federated merge ranks by relevance, never by the workspace config's repo
order, and it carries each repo's own noise demotion: decision records and test
pages that a repo ranked below its real pages stay below them after the merge.

A federated response also carries its freshness per corpus, because a workspace
answer can mix a repo indexed this morning with one indexed six weeks ago.
`_meta.repo_freshness` maps each alias to that repository's own `indexed_commit`,
`index_age_days`, `index_behind` and any `stale_warning`, scoped to the rows
that repo actually contributed; the roll-up beside it reports the oldest age and
warns only when a repo that contributed content is behind. There is deliberately
no workspace-level `indexed_commit`: each repo has its own, and one of them is
not the workspace's.

When the source-search lane is active, a
federated response is composed from per-repo engines: the strongest repo's
results lead, and `confidence` is the winning repo's own class — never more
than that repo itself asserted — demoted to `caution` when two repos are EACH
confident of an owner (the same relative path in two repos is two distinct
claims), with all claimants listed under `competing_owners` in ranked order
with their evidence. A repo whose search read no corpus (`status: "error"`)
is disclosed as broken in `_meta.source_search.repos`, never ranked as an
answer.
`competing_owners` also fires when the query names a kind of file rather than
a repository — ask for "not found page" in a workspace where several repos
have one and you get the rivals listed and `caution`, not one of them picked
silently. Neither the leading repo nor its block takes the whole window:
every other repo that answered keeps a tail slot for its own top row in both
`results` and `candidates`, so a `limit` at least the number of answering
repos shows you something openable from each of them. `repo` omitted means
the workspace's default repo, not all of them.

**When to use:** Locating a function/class/method by name, resolving a
path-shaped query, or discovering pages by topic: the symbol/file shapes pipe
directly into `get_symbol` / `get_context`.

**Example calls:**

```
search_codebase(query="GitIndexer index_repo")          # -> symbol hits
search_codebase(query="core/ingestion/indexer.py")      # -> file hits
search_codebase(query="rate limit OR throttle OR retry") # -> wiki pages
search_codebase(query="login", mode="symbol", symbol_kind="method")
```

---

## `get_risk`

Modification risk assessment for files or a set of changed files.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `targets` | list[string] | No | File paths to assess |
| `changed_files` | list[string] | No | Files in a PR/changeset for blast radius analysis; passing this switches the response into PR-directive mode |
| `repo` | string | No | *(workspace only)* Target repo alias |

**Returns:** Per-file `hotspot_score` (0-1 churn percentile), `health_score` (0-10), hotspot status, dependent count, co-change partners (each with a recency-decayed `weight`, not an integer count), blast radius, recommended reviewers, test gap analysis, security signals. In workspace mode, enriched with cross-repo co-change partners and contract dependencies.

Test-gap analysis is the same resolver `get_context` reports from, so the two
tools always agree about a file: `test_gap` is the negation of `tested`, and
the payload carries the `test_linkage_basis` plus authoritative
`guarding_tests` or naming-only `possible_tests`. Naming alone leaves
`test_gap: true`. See [Test linkage](#get_context) under `get_context` for what
each basis claims.

> **Scales.** Ratios derived from ownership or percentile columns are 0-1 (`hotspot_score`, `owner_pct`, `recent_owner_pct`); coverage and gap fields are 0-100 (`coverage_pct`, `branch_coverage_pct`, `share_of_repo_gap_pct`, `change_entropy_pct`, `churn_percentile`). The `_pct` suffix alone does not tell you which — check this table. Every emitted float is rounded to 4 significant digits.

When `changed_files` is passed, the response leads with a `directive` block. Its core lists are the local blast radius: `will_break` (production files that depend on the diff and are likely to break), `will_break_tests` (test files impacted the same way, kept separate so a burst of broken tests doesn't crowd production impact out of the capped list), `missing_cochanges` (historical co-changers absent from the diff), `missing_tests` (changed files without test coverage), and `tests_to_run` (the positive complement of `missing_tests`: the tests the per-test coverage map proves execute the changed files, as pytest-runnable ids to validate the change; empty until a coverage map is ingested with `repowise coverage add`). In workspace mode that directive also carries the cross-repo fallout of the changed repo:

- `will_break_consumers`: services in *other* repos that depend on this one (structural impact), each with `repo`, `service`, `distance`, `score`, and the edge kinds carrying the impact.
- `missing_cross_repo_cochanges`: services in other repos that historically co-change with this one but aren't in the diff.
- `breaking_changes`: provider contracts in this repo that changed *incompatibly* since the last index (a removed route or field, a type or field-number change, a newly-required field), each with the changed `contract_id`, the change `kind`/`severity`, and the `impacted_consumers` (repo, service, file) it endangers across repos. Schema-level truth, distinct from the topology-level `will_break_consumers`; non-breaking changes (added optional field, new endpoint) never appear. See [Breaking-Change Guard](../scale/WORKSPACES.md#breaking-change-guard).
- `conformance_violations`: declared dependency-rule breaches the diff's repo participates in, each with the offending `source`/`target` services, the `rule` (e.g. `frontend !-> db`), and `edge_kind`. See [Architecture Conformance](../scale/WORKSPACES.md#architecture-conformance).
- `dependency_cycles`: circular service dependencies involving this repo, each with the participating `nodes` and `length`.

**When to use:** Before modifying files, especially hotspots. Understand what could break, who to involve in review, and whether tests cover the affected area.

**Example calls:**

```
get_risk(targets=["src/auth/middleware.ts"])
get_risk(changed_files=["src/api/routes.ts", "src/middleware/cors.ts"])
```

---

## `get_change_risk`

Live risk scoring for one commit or a `base..head` range. Unlike `get_risk`,
which evaluates indexed files and can report blast radius, this scores the
shape of the live diff and needs no index refresh.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `revspec` | string | No | Commit or `base..head` range to score. Omit it to score uncommitted work, or pass `HEAD` when the tree is clean |
| `repo` | string | No | *(workspace only)* Target repo alias |
| `extensions` | list[string] | No | File suffixes to count, such as `[".py", ".ts"]` |
| `exclude_patterns` | list[string] | No | Gitignore-style paths to omit; combined with root `.riskignore` rules |
| `baseline` | int | No | Recent commits to sample for percentile ranking (default `200`; `0` disables percentile ranking) |

**Returns:** `fix_history` first — the recency-weighted bug-fix record of the
files the change touches, with `files` naming where the pressure sits and
`percentile` ranking it against the repo's own fix-bearing files. Triage on
this: it is the part that separates a small edit to a fragile file from a large
edit to a safe one. `available` is false when the history walk could not run.

`score` measures diff size and spread, not where the change lands — see
`score_measures` — and `score_unit` names the unit it is calibrated on (a single
commit, so a PR-sized range reads high by construction). `risk_percentile`,
`review_priority` and `classification` rank that same diff shape against recent
commits. `fallback_band` carries the absolute band and appears only when no
baseline was available. `working_tree` says whether uncommitted work was the
subject. `baseline_sample_size` reports how many filtered commits informed the
percentile; `features`, `drivers`, and combined `exclude_patterns` make the
result auditable.

It also returns `impacted_tests`: the tests the per-test coverage map proves
execute the change's changed *lines* (line-precise, so a narrower set than
`get_risk`'s file-level `tests_to_run`), capped at ten with `total` and
`truncated` reporting any overflow. Its `missing_tests` buckets flag
`untested_changes` (covered file, uncovered change), `stale_test_candidates`
(covered lines whose guarding test file is absent from the diff), `covered`, and
`no_coverage_data` (files absent from the map). When no map is ingested,
`status` is `no_map` and the change is reported as unknown ("run the full
suite"), never as untested. Build the map with `coverage run --contexts=test`
followed by `repowise coverage add`.

When the changed files carry counted bug fixes, the response also holds
`prior_fixes`: per file, how many past bug-fix commits touched it
(`fix_count`), how many of the change's lines fall inside the ranges one of
those fixes replaced (`overlapping_lines`), and how long ago the most recent
was (`last_fix_days_ago`). `total_fixes` counts distinct commits, not rows,
and `files` is capped at ten with `truncated` reporting overflow.

Each file also carries how much of *this* change sits in it — `changed_lines`
and `share_of_change` — with `changed_lines_in_fixed_files` as the total across
them. That join is what lets the response say where the risk sits rather than
only that some touched file has a past: the score is whole-change, so when one
returned file holds at least half the changed lines, `concentration` names it.

`overlapping_lines` is labelled `approximate` in the payload, and that label is
load-bearing: a past fix's ranges are numbered against its own parent commit,
so anything that moved lines in between shifts them. Read it as "this
neighbourhood has been patched before", not "this exact line". The per-file
`fix_count` beside it carries no such caveat. The whole block is aggregate and
never names the commit that introduced a bug: file-level SZZ measured 74.5%
precision on this repo's frozen judgments, which is enough to count fixes and
not enough to accuse one commit of causing them. The block is absent entirely
on an index with no fix history.

**When to use:** Before merging a commit or PR range, especially when you need
to assess the diff itself rather than the risk of an already-indexed file.

**Example calls:**

```
get_change_risk()
get_change_risk(revspec="main..HEAD", extensions=[".py"], exclude_patterns=["tests/"])
```

---

## `get_why`

Architectural decision intelligence. Falls back to git archaeology when no decision records exist for a path, and further to a rationale comment mined live from the source when neither decisions nor git history explain the "why".

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | No | Natural language question about decisions, OR a file/module path |
| `targets` | list[string] | No | File paths to anchor an NL `query` search to |
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` (only when `query` is given) |

**Modes:**

1. **NL search**: pass a question, optionally anchored to `targets`: `get_why(query="why JWT over sessions?")` -> searches decision records.
2. **Path-based**: pass a file path as `query`: `get_why(query="src/auth/service.ts")` -> returns decisions governing that file plus its origin story.
3. **Health dashboard**: no `query`: `get_why()` -> stale decisions, conflicts, ungoverned hotspots.

**Returns:** Matching decision records with title, rationale, alternatives considered, affected files, staleness score. Health mode returns stale decisions, conflicts, and ungoverned hotspots.

**When to use:** Before architectural changes, understand existing intent and constraints. After changes, record new decisions.

**Example calls:**

```
get_why(query="rate limiting")
get_why(query="src/payments/processor.ts")
get_why(query="why is caching split from the eviction path?", targets=["src/cache"])
get_why()
```

---

## `get_dead_code`

Unreachable code, unused exports, unused internals, and zombie packages, sorted by confidence tier with cleanup impact estimates. Flag-based, not include-list-based.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | *(workspace only)* Target repo alias |
| `kind` | string | No | Restrict to one finding kind: `unreachable_file` \| `unused_export` \| `unused_internal` \| `zombie_package` |
| `min_confidence` | float | No | Minimum confidence floor (default `0.4`; `0.7`+ is cleanup-ready only) |
| `safe_only` | boolean | No | Deletion-ready findings only, excluding anything with runtime-load risk (default `false`) |
| `limit` | int | No | Max findings per tier, clamped to 25 (default 20) |
| `tier` | string | No | Restrict to one tier: `high` (>= 0.8) \| `medium` \| `low` |
| `directory` | string | No | Path-prefix filter |
| `owner` | string | No | Primary-owner filter |
| `group_by` | string | No | Roll findings up by `directory` or `owner` instead of listing them flat |
| `include_internals` | boolean | No | Include private/underscore symbols (default `false`) |
| `include_zombie_packages` | boolean | No | Include zombie-package findings (default `true`) |
| `no_unreachable` | boolean | No | Exclude `unreachable_file` findings (default `false`) |
| `no_unused_exports` | boolean | No | Exclude `unused_export` findings (default `false`) |

**Returns:** Dead code findings grouped by confidence tier (high >= 0.8, medium, low). Each finding includes: file path, kind, confidence score, line count, and cleanup impact estimate. In workspace mode, confidence is lowered on findings other repos still import.

**When to use:** Cleanup tasks, not a targeted fix. Conservative by design: `safe_only` excludes dynamically-loaded patterns and framework-decorated functions.

**Example calls:**

```
get_dead_code()
get_dead_code(min_confidence=0.8, tier="high", safe_only=true)
get_dead_code(kind="unused_export", group_by="owner")
```

---

## `get_health`

Code-health marker scores: the same deterministic markers the
`repowise health` CLI computes, across three signals (defect risk,
maintainability, performance), exposed for agentic workflows. Zero LLM calls.
Use it to **self-check a change before opening a PR**: the same signals a
code-health merge-gate judges it on.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `targets` | list[string] | No | File paths, or `module:foo` to expand a module's file set. Empty means dashboard mode. |
| `include` | list[string] | No | Opt-in blocks (default response stays lean): `"biomarkers"` (findings in dashboard mode), `"refactoring"` (structured, graph-aware refactoring plans; see below), `"trend"` (snapshot diff + declining / predicted-decline alerts), `"coverage"`, `"accuracy"` (the "does the score find the bugs?" stat, dashboard mode), `"signals"` (per-file process / people / topology signals, targeted mode), `"churn_complexity"` (churn x complexity quadrant points, dashboard mode), and a dimension name (`"performance"` / `"defect"` / `"maintainability"`) to filter findings to that pillar. |
| `only` | list[string] | No | Keep just these top-level keys. `include` adds blocks, `only` subtracts them. `mode`, `_meta`, `unresolved`, `known_modules` and each kept list's `*_total` sibling always survive. The three `include` **block** names work as aliases: `biomarkers`→`findings`, `accuracy`→`defect_accuracy`, `refactoring`→`refactoring_plans`. The `include` **dimension** names (`performance`, `defect`, `maintainability`) do not — they filter rows inside several blocks and have no single key to resolve to, so they land in `unknown_only_keys`. Nor does `signals`, which merges into `metrics[].signals` — in targeted mode, where `signals` applies, name `metrics` instead. |
| `repo` | string | No | *(workspace only)* Target repo alias |
| `limit` | int | No | Max rows in **every** ranked list (default 20, capped at 50). `0` means no rows; the `*_total` siblings still report the true counts. |

**Returns:** Dashboard mode (no `targets`) returns a `directive`, repo-level KPIs
(hotspot health, average health, worst performer, maintainability / performance
pillar averages), the lowest-scoring files, and a per-module NLOC-weighted
rollup. Targeted mode returns per-file marker findings with severity,
per-dimension scores, and the score breakdown. Each finding carries a `dimension`
(`defect` / `maintainability` / `performance`).

**Lead with `directive`.** Dashboard mode opens with the single file to fix
first, its dominant finding, `recovers_points` / `share_of_repo_gap_pct` (what
fixing it buys the headline), and `then`, the next two by leverage. Every other
block ranks and describes; this one recommends. Same role as `get_risk`'s
`directive`. Rank by `weighted_deficit`, not `score` — the score floors at 1.0.

**Nothing is dropped silently.** Any `targets` entry that matched nothing is
named in `unresolved` with a reason (`not_indexed` → run `repowise update`,
`no_such_path`, `excluded`, `no_such_module`; a missed module name also returns
`known_modules`), so an empty `findings` list means healthy and nothing else. A
target set that resolves to nothing still answers in targeted mode rather than
falling back to the repo dashboard. Every capped list carries a `*_total`
sibling — including under `only`, which retains it automatically. `unresolved`
and `known_modules` survive any `only` projection too, for the same reason
`mode` does: a caller who has to ask for the error report in order to see it
does not have an error report.

`_meta.health_analyzed_at` dates the health pass, which is separate from
indexing and can lag it, and `_meta.health_analyzed_commit` says which commit
those scores were computed against. The incremental update path rescores only
the files that changed, so the metrics table can hold rows from several passes
at once; when it does, `_meta.health_analyzed_commits_distinct` says how many
and the reported commit is the newest pass's. Both fields are omitted rather
than guessed when no row records a commit.

**The response is bounded.** `include` only *adds* blocks, and the dashboard's
five ranked lists compose: `include=['refactoring']` on a mid-size repo lands
near the host's tool-result cap, past which the host rejects the whole result
and you get nothing. Pair `include` with `only` —
`get_health(include=['refactoring'], only=['refactoring_plans'])` is the call
`directive.plan_via` names. Anything that would still overflow is trimmed
longest-ranked-list-first and reported in `_meta.truncated_to_fit`
(`{block: rows_dropped}`), never silently; the `*_total` siblings still describe
what was there, and re-requesting one block with `only` recovers it.

**Test material is bucketed, not hidden.** Every metric row carries `is_test`
(distinct from `has_test_file`: "is this file a test" vs "is this file tested").
In dashboard mode the ranked finding lists are split — `top_findings` /
`findings` carry production findings, `test_findings` carries the test half, and
`top_findings_total + test_findings_total` is the whole open set. Defect risk in
a test asks a different question from defect risk in the code it covers, and at
the default limit a quarter of the headline list was describing the test suite.
Targeted mode is never split: you named the files, so you get their findings.
KPIs, `worst_files` and `high_leverage_files` deliberately still include test
files — excluding them would move the repo's headline score, which is a scoring
change, not a display one.

**Leverage, not just lowness.** `average_health` is NLOC-weighted (the number the
badge and dashboard surface), so a few large low-scoring files hold it down. To
make that actionable rather than a mystery:

- `kpis.average_health_unweighted` is the plain file mean and
  `kpis.average_health_weighting` is `"nloc"`. When the weighted and unweighted
  numbers diverge, the gap is telling you to chase *big* files, not the long tail.
- `gap_analysis` (dashboard mode) reports the net weighted points the average must
  recover to reach the Healthy floor (8.0), how many files sit below it, and how
  few of them carry the whole gap (`files_to_reach_target`) or half of it
  (`files_for_half_gap`). This reframes a repo-wide number as a short worklist.
- Every metric row carries `weighted_deficit = (8 - score) x nloc`: how much the
  repo headline recovers if that file reaches 8.0. `high_leverage_files`
  (dashboard mode) is the top-N ranked by it, distinct from `worst_files`, which
  sorts by raw score and ranks a 30-line file at 1.0 equal to a 1,200-line file at
  1.0 that moves the average ~40x more.
- `weighted_deficit`, `directive.recovers_points` and
  `gap_analysis.weighted_gap_points` share one unit — *score-points x NLOC* —
  which compares against itself and nothing else. Every `high_leverage_files`
  row and the `directive` also carry `share_of_repo_gap_pct`, the same quantity
  with a denominator; that plus `gap_analysis.files_to_reach_target` is what
  answers "is this worth doing".
- `kpis.non_code_files` and `kpis.average_health_code_only` say how much of the
  headline is markdown/JSON/YAML. No biomarker walks those files, so they score
  a mechanical 10.0 meaning "nothing looked at this" — on this repo, 233 of
  3,314 rows, lifting `average_health` from 7.31 to 7.47. `average_health`
  itself deliberately still counts them, so the tool, the badge, the snapshots
  and the web UI all report the same number.

**One score, not two.** A metric row carries `score` (the defect dimension and
the headline), `maintainability_score` and `performance_score`. There is no
`defect_score` in the response: it was set from the same value as `score` on
every row, and two names for one number cost a reader a source dive to pick
between them. The field to rank on is neither — it is `weighted_deficit`.

**`primary_biomarker` names a discrete cause.** It prefers the strongest
*discrete* finding over a continuous one. `coverage_gradient` fires on every
file that has coverage data at all, so on a well-covered repo it used to win the
max-impact tiebreak nearly everywhere and headline the list with "N% of lines
uncovered" — true, and equally true of every other file. The gradient still
counts in full toward `total_deduction` and the score, and still leads a file
that has no discrete finding.

The opt-in enrichments:

- **`accuracy`** returns a `defect_accuracy` block: of the K least-healthy files, how
  many were recently bug-fixed vs the repo-wide base rate (precision@K + `lift`),
  with a per-K table and the flagged files. Silent (`null`) on repos with too
  little history to be honest (< 25 scored files or < 5 recently-fixed files).
- **`signals`** adds a `signals` object on each targeted metric: prior-defect count,
  change scatter, 90-day churn, primary / recent owner, and graph in / out
  degree. Honest `null` per field when the underlying row is absent (never an
  imputed zero).
- **`churn_complexity`** returns `churn_complexity` points (one per recently-changed
  file: 90-day commit count, max CCN, NLOC, score, churn percentile): the
  refactor zone where volatility and tangle collide.
- **`refactoring`** returns ranked, structured refactoring plans (not template
  strings): `extract_class` (the cohesion `groups` to split into), `extract_helper`
  (clone `occurrences` + `suggested_site`), `move_method` (`{method, from_class,
  to_class}`), and `break_cycle` (the import `cut_edges`). Each plan carries its
  `evidence`, `impact_delta`, `effort_bucket`, `blast_radius`, and an `id` you can
  hand to `generate_refactoring_code`. The list is capped to `limit` and ranked
  file-leverage-first (by the file's `weighted_deficit`, then per-plan impact), so
  plans on the files that move the headline surface first; `refactoring_plans_total`
  reports the full count behind the cap. Each plan echoes its
  `file_weighted_deficit`. Full shapes in [`docs/layers/REFACTORING.md`](../layers/REFACTORING.md).
- **dimension filter** narrows the returned findings to one pillar, e.g.
  `include=["biomarkers", "performance"]`.

**Performance findings rank on `perf_rank`, not on `health_impact`.** Every
performance finding carries `health_impact: 0` — the pillar is deliberately
never blended into the score — so ranking them by impact ordered them by nothing
and the cap kept whichever the tie broke to. Each performance row now carries an
integer `perf_rank` (absent on defect and maintainability rows, which rank on
`weighted_deficit`), and the returned list is ordered by it *within* each impact
tier, so the defect ordering is untouched. It is an ordering key, not a score:
nothing is blended into `score` / `performance_score`. It adds three signals the
row already carries, so you can re-rank on your own weights from the same
payload:

| signal | reads | why |
|---|---|---|
| the marker | `biomarker_type` | superlinear (`nested_loop_quadratic` 5) > N×M or lock-serialized (4) > one crossing per iteration, or one crossing proven on a hot path (3) > in-loop CPU/allocation (2) > cheap in-loop idioms (1) |
| the boundary | `details.boundary_kind` | `subprocess` 4 > `network` 3 > `db`/`lock` 2 > `filesystem` 1 — a process spawn in a loop is not a stat in a loop |
| the call shape | `details.cross_function` | +1. An intra-function loop is usually visibly bounded at the call site; a cross-function N+1 is the one nobody sees by reading the loop |

Request-reachability is read off the marker rather than a column:
`hot_path_sync_io` and `nested_loop_quadratic` are only ever emitted for a
function the perf ranker called hot (top-quintile call-graph in-degree, or a
churny/hotspot file), so their presence is already the proof. Deliberately not
`severity` — that column grades `hot_path_sync_io` below `io_in_loop` and takes
only two values across a whole repo's perf findings.
- **`refactoring`** also emits `suggestion_legend`: `biomarker_type` → the prose
  suggestion for that type, once per response rather than per finding. Join on
  `biomarker_type`. It is keyed off the ranked finding head and does not vary
  with `only`, so it can carry an entry for a block a projection dropped —
  extra rows in a lookup table, never a missing one. Note it explains the
  **findings**, not the
  plans it ships beside — the two sets differ (no plan kind is sourced from
  `coverage_gradient`), and `directive.plan_addresses_reason` is what reports
  that gap.

**When to use:** Before opening a PR, to self-check the files you changed
(`targets=[...], include=["signals"]`) and confirm you are not regressing the
worst files. Before refactoring, find the worst-scoring files and what to fix
first (`include=["accuracy", "churn_complexity"]`). Pair with `get_risk` on
hotspots.

**Example calls:**

```
get_health(only=["directive"])                        # cheapest useful call: what to fix first
get_health()                                          # directive, kpis, gap_analysis, worst + high_leverage files
get_health(include=["accuracy", "churn_complexity"])
get_health(include=["biomarkers", "performance"])     # only performance findings
get_health(targets=["src/api/server.py"], include=["signals"])
get_health(targets=["module:src.api"], include=["trend", "refactoring"])
get_health(include=["accuracy"], only=["accuracy"])   # the block, without the dashboard again
get_health(only=["top_findings"])                     # + top_findings_total, automatically
get_health(only=["kpis"], limit=0)                    # headline numbers, no rows at all
```

---

## `get_index_status`

Checks the source-search publication before an agent relies on indexed results. This
tool is read-only and on by default. It never resolves an LLM provider, embeds text,
or starts a job.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | `"status"` (default) or `"path"` |
| `path` | string | Path mode only | One repository-relative path to diagnose |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |

**Status mode returns:**

- `trust.search_results`: `trustworthy`, `stale`, or `unknown`, plus exact reasons.
- Active generation id/sequence, indexed commit versus live HEAD, build/publication
  times, working-tree paths captured by the last index, and per-file stale reasons.
- Exact pending/building/ready/blocked queue totals. Their stated unit is source-index
  update rows after the active generation; there is no pagination or hidden cap on the
  totals themselves.
- The variable-length arrays (`stale_files`, `uncommitted_indexed_paths`, active job
  rows) are capped with the exact total disclosed beside the listed slice
  (`*_count` / `*_listed`), and the dropped tail is restorable through the standard
  `_meta.omitted` mechanism ("Reversible truncation" above). Path mode's eligibility
  block names its `deciding_surface` — the ingestion traversal for the symbol lane,
  `git ls-files` + window eligibility for the window lane.
- Manifest symbol/file-window totals and file coverage, plus independently read FTS
  and vector counts and parity.
- Indexed/runtime embedder and parser identities. A missing identity produces
  `unknown`; it is never guessed from a plausible default.
- `degraded`, `degraded_reason`, and structured `degradation_findings` when a
  publication component is broken. Retrieval-time `failed_legs` remains a distinct
  exception-oriented contract on search responses.

`verify_stores=true` is unconditional. On the frozen SoleMD.Infra mirror (8,200
active chunks over 940 files), six warm checks measured 10.45–13.31 ms with an
11.23 ms median. The first cold check was 830.11 ms including the deferred LanceDB
import. Counting rows makes no embedding or generative call.

**Path mode returns:** exact active-generation symbol/file-window inventory, tracked
and working-tree state, and parser/window lane eligibility. `path_shape_candidate` is
reported only as a file-watcher/diff hint; it never decides source-index membership.
Closed reasons include `indexed`, `parser_failed_stale`, `eligible_not_indexed`,
`untracked_window_only`, and `not_source_eligible`. When a source-lane policy cannot
identify its deciding rule, the result is `unknown` with the missing fact stated; the
handler does not substitute the separate query-time wiki-exclusion policy.

```
get_index_status()
get_index_status(mode="path", path="src/auth/service.py")
```

Pair a stale/unknown result with the opt-in [`reindex_repository`](#reindex_repository)
action only after reviewing its preview.

---

## Supplementary tools

These are registered and on by default (in the modes noted) but are not
part of the eleven-tool headline set.

### `list_repos`

Lists the repos this server is serving. No parameters.

**Returns:** In workspace mode, `workspace: true`, the workspace root, the default repo alias, and every configured repo alias (`repos`). In single-repo mode, `workspace: false` and a single `"default"` alias.

**When to use:** Discovering the `repo` aliases to pass to other tools, especially in workspace mode.

```
list_repos()
```

### Workspace-only tools

*(Available only when the server is started inside a workspace; see [Workspace Mode](#workspace-mode).)*

#### `get_blast_radius`

Cross-repo downstream impact: if you change this service, what breaks across the other repos?

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `targets` | list[string] | Yes | Node ids (`repo` or `repo::service/path`) or repo aliases |
| `max_depth` | int | No | Reachability depth (1-8, default 3) |
| `include_behavioral` | bool | No | Include co-change (behavioral) edges (default `true`) |

**Returns:** The impacted services ranked by impact `score`, each with `distance` (hops), `structural` (a real dependency vs co-change only), and the edge kinds that carried the impact; plus `impacted_repos`, `structural_count` / `behavioral_count`, `total_impacted`, and any `unresolved_targets`.

**When to use:** Before changing a high-fan-out provider, see who consumes it across repo boundaries. Structural impact ("will break") outweighs behavioral co-change ("may drift"). Reads the same system graph the [Live System Map](../scale/WORKSPACES.md#live-system-map) renders.

```
get_blast_radius(targets=["backend"])
get_blast_radius(targets=["mono::services/auth"], max_depth=2, include_behavioral=false)
```

#### `get_conformance`

Architecture governance: does the live system graph obey the declared dependency rules, and are there circular service dependencies?

**Opt-in.** Off by default even in workspace mode; enable with `mcp.tools: ["+get_conformance"]`. Named in single-repo mode it is ignored, since it needs the workspace graph. The same findings still surface in the `get_risk` PR-mode directive (`conformance_violations` / `dependency_cycles`) without opting the tool in.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | Limit findings to those involving this repo alias |

**Returns:** `violations` (each with the offending `source`/`target` services, the `rule_source`/`rule_target` matchers that fired, and the `edge_kind`), `cycles` (each with the participating `nodes` and `length`), and the `violation_count` / `cycle_count` / `rules_evaluated` rollups.

**When to use:** Before a refactor that changes service boundaries, or to audit whether the live architecture still matches the intended one. Rules are declared under `conformance:` in `.repowise-workspace.yaml`. See [Architecture Conformance](../scale/WORKSPACES.md#architecture-conformance).

```
get_conformance()
get_conformance(repo="frontend")
```

#### `get_architecture`

The one evaluative read of the whole system: how coupled is it, where is the architectural core, and a single 1-10 architecture score. Deterministic, structural edges only (co-change excluded). No parameters.

**Returns:** `score` (1-10), `architecture_type` (`core-periphery` or `hierarchical`), `propagation_cost_pct` (share of other services the average service reaches), `core_size` / `core_ratio` / `core_members` (the largest cyclic group), `cycle_count`, `conformance_violations`, a `role_breakdown` (count of Core / Shared / Control / Peripheral services), and a one-line `summary`.

**When to use:** Before a cross-service refactor, or to gauge and compare overall system structure over time. See [Architecture Metrics](../scale/WORKSPACES.md#architecture-metrics).

```
get_architecture()
```

### Opt-in tools

*(Registered but off by default in every mode; enable with `mcp.tools: ["+name"]` or `repowise mcp --tools "+name"`. See [Configuring the tool surface](#configuring-the-tool-surface).)*

#### `get_dependents`

Complete inbound dependents for a file or symbol, with test filtering and honest pagination.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | string | Yes | File path, exact symbol id, qualified symbol name, or unambiguous bare symbol name |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `depth` | int | No | Inbound traversal depth, clamped to 1-8 (default 1) |
| `include_tests` | boolean | No | Include tests and allow traversal through them (default false) |
| `offset` | int | No | Zero-based offset into the ranked result set (default 0) |
| `limit` | int | No | Page size, clamped to 1-100 (default 25) |

**Returns:** `dependents` is the requested page; `total` and `counts_by_depth` describe the complete filtered traversal before pagination. Results rank by `reference_count`, then PageRank. `reference_count` means distinct persisted graph relations into the preceding breadth-first frontier—not source call-site occurrences, because graph rows are aggregated per source/target/edge type. File targets follow the canonical file-dependency edge set; symbol targets follow the canonical symbol-use edge set. `pagination.has_more` and `next_offset` make every remaining row recoverable.

**When to use:** Before changing a file or symbol, to enumerate direct consumers or a bounded transitive impact frontier without test files crowding out production dependencies.

```
get_dependents(target="src/db/models.py")
get_dependents(target="reconcile_project_files", depth=3, limit=50)
get_dependents(target="src/db/models.py", include_tests=true, offset=25)
```

#### `get_dependency_path`

Shortest dependency paths between files, or pure call/reference chains between symbols.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | Source file path, exact symbol id, qualified symbol name, or unambiguous bare symbol name |
| `target` | string | Yes | Target file path, exact symbol id, qualified symbol name, or unambiguous bare symbol name |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `mode` | string | No | `"files"` preserves the existing dependency traversal (default); `"calls"` admits symbol-level `calls`/`references` edges only |
| `limit_paths` | int | No | Distinct shortest chains to return, clamped to 1-5 (default 1) |

**Returns:** The legacy top-level `path` and `distance` remain the first shortest chain. `paths` carries up to `limit_paths` distinct shortest chains and `paths_truncated` says whether more shortest chains exist. Symbol names resolve when unique; ambiguity returns structured candidates and no fabricated path. In `mode="calls"`, every returned relationship is exactly `calls` or `references`; imports and containment can never enter the graph. When no path exists, visual context instead describes nearest common ancestors, shared neighbors, communities, and bridge suggestions.

**When to use:** Understanding how two parts of the codebase are (or aren't) connected, or why an expected dependency doesn't show up.

```
get_dependency_path(source="src/api/routes.py", target="src/db/models.py")
get_dependency_path(source="handle_search_code", target="build_evidence", mode="calls", limit_paths=3)
```

#### `get_execution_flows`

Top entry points and their call traces: how the codebase actually executes.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `top_n` | int | No | Number of top entry points to trace (default 10) |
| `max_depth` | int | No | Max trace depth per flow (default 8) |
| `entry_point` | string | No | Trace from a specific symbol, overriding `top_n` scoring |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |

**Returns:** Scored entry points with BFS call-path traces showing which functions are called in sequence, and whether the flow crosses community boundaries.

**When to use:** Understanding runtime call flow through an unfamiliar system, or tracing what a specific entry point actually does end to end.

```
get_execution_flows()
get_execution_flows(entry_point="src/cli/main.py::main", max_depth=4)
```

#### `generate_refactoring_code`

Turns one structured refactoring plan from `get_health(include=["refactoring"])` into actual generated code and a unified diff, grounded on the plan plus the real source spans it references. For Extract Class, the result includes an LCOM4 before/after self-check.

**Off by default twice over:** it must be opted into the tool surface (`mcp.tools: ["+generate_refactoring_code"]`), and even then returns `{"error": "disabled", ...}` unless `refactoring.llm.enabled: true` is set in the repo's `.repowise/config.yaml`. When enabled, it uses the repo's configured LLM provider/model (bring your own key) and caches results by a content hash, so an unchanged plan never regenerates.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `suggestion_id` | string | Yes | The `id` of a plan returned by `get_health(include=["refactoring"])` |
| `repo` | string | No | *(workspace only)* Target repo alias |

**When to use:** After `get_health(include=["refactoring"])` surfaces a plan you want turned into an applyable diff, and your repo has opted into both the tool and LLM-backed generation.

```
generate_refactoring_code(suggestion_id="a1b2c3d4")
```

#### `reindex_repository`

Previews or queues a non-generative repository `index_only` job. The tool is
off by default and cannot be reached merely by enabling the ordinary read surface.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `confirm` | boolean | No | Defaults to `false`; must be `true` before work is queued |
| `force` | boolean | No | Queue even when verified status is already trustworthy/current |

Without confirmation, the response is read-only: it reports active jobs and a cost
preview based on the active generation's exact file/chunk counts, with zero
generative calls. An already-current repository is a no-op unless `force=true`.
Concurrent requests reuse the repository's pending/running job instead of launching
overlapping work. Confirmed work uses the established `index_only` executor so parsing
and SQL symbols refresh before the derived source stores publish; it never performs a
direct reconcile against possibly stale symbol bounds. `force` only bypasses the
already-current no-op; it does not change the executor into a full rebuild mode.

```
reindex_repository()                    # preview only
reindex_repository(confirm=true)        # queue when stale/unknown
reindex_repository(confirm=true, force=true)
```

---

#### `build_task_slice`

Cuts a *task slice* — the part of the codebase one task needs — and stores it
under an id that survives the call. Entry points resolve from the task text,
the slice grows along the symbol graph (downstream calls, upstream callers),
members are ranked, and the whole result serializes inside `budget_tokens`.
Members the budget drops are disclosed and recoverable through the shared
omission store, never silently gone.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | string | Yes | The task, phrased as work: "add retry to the sync client" |
| `entry_points` | string[] | No | Explicit starting nodes — file paths, `path::Symbol` ids, or unambiguous symbol names. Naming them suppresses nomination from the task text |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `view` | string | No | `card` (default here: `skeleton`), `skeleton`, or `full` fidelity per member |
| `budget_tokens` | integer | No | Serialization budget; drops are ranked and disclosed |
| `downstream_depth` / `upstream_depth` | integer | No | Walk depth from the entry points (defaults 2 / 1) |
| `include_tests` | boolean | No | Include test files as members |
| `max_members` | integer | No | Member cap before the budget pass (seed members are always kept) |
| `include_edges` | boolean | No | Carry the member-to-member edges in the response |

A build that matches nothing is a shaped failure naming what was tried — never
an empty member list, which would read as "this task needs no code".

```
build_task_slice(task="wire the freshness envelope into the CLI status table")
```

---

#### `get_task_slice`

Re-reads a stored slice by id, possibly at a different fidelity or budget than
it was built with. The three views are per-member fidelity levels: `card`
(name, path, role, one line), `skeleton` (signatures and docstrings), `full`
(bounded source). A wrong id is a shaped error, not an empty slice.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `slice_id` | string | Yes | The id `build_task_slice` returned |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `view` | string | No | `card`, `skeleton` (default), or `full` |
| `budget_tokens` | integer | No | Serialization budget for this read |
| `max_source_lines` | integer | No | Per-member source cap in `full` view |
| `include_edges` | boolean | No | Carry the member-to-member edges |

```
get_task_slice(slice_id="sl_3f9a2b7c1d0e", view="full", budget_tokens=12000)
```

---

#### `extend_task_slice`

Grows an existing slice when the first cut was too narrow — deeper along the
graph, or from new entry points, without re-walking what is already there.
Extension is additive: members never leave a slice by extension, and the
response discloses what the extension added versus what the budget then had to
drop. Extending a fully-expanded slice says so rather than pretending growth.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `slice_id` | string | Yes | The slice to grow |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `extra_downstream` / `extra_upstream` | integer | No | Additional depth from the current frontier (defaults 1 / 0) |
| `entry_points` | string[] | No | New entry symbols or paths to walk from |
| `task_addendum` | string | No | Extra task text to resolve new entry points from |
| `view` | string | No | Fidelity of the returned members (default `card`) |
| `budget_tokens` | integer | No | Serialization budget for this read |
| `include_edges` | boolean | No | Carry the member-to-member edges in the response |

```
extend_task_slice(slice_id="sl_3f9a2b7c1d0e", extra_downstream=1,
                  entry_points=["src/sync/client.py::SyncClient"])
```

---

#### `find_clones`

Duplicated regions — exact by construction, near-duplicates on request. The
tool is a thin adapter over the clone service boundary; the detector's on-disk
caches stay behind it and are never exposed or required reading. Near-clones
(dense-similarity pairs over the source index) are off by default and never
mixed silently into exact results: every finding names which detector produced
it. A degraded leg (cold cache, missing source index) is disclosed in the
response, never folded into "no clones".

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `path` | string | No | Confine findings to one file or directory |
| `min_lines` | integer | No | Minimum region size to report |
| `limit` | integer | No | Findings per response (cap 40) |
| `include_near` | boolean | No | Add semantic near-clones from the source index (off by default) |
| `near_threshold` | number | No | Cosine floor for a near pair |
| `cross_directory_only` | boolean | No | Only pairs that span directories |
| `include_intra_file` | boolean | No | Keep same-file pairs (on by default) |
| `include_tests` | boolean | No | Include test files in scope |

```
find_clones(min_lines=15, cross_directory_only=true)
find_clones(include_near=true, near_threshold=0.86)
```

---

#### `find_patterns`

Six named structural queries over the symbol graph: `duplicate_signatures`,
`orphan_exports`, `hub_functions`, `isolated_siblings`, `reuse_candidates`,
`bridge_functions`. Each response carries the predicate that produced it, so
the list is read against the rule that ran rather than whatever the name
suggests. Called with no pattern — or an unknown one — it returns the
catalogue instead of an empty match list, because an empty list is
indistinguishable from "this repository has none".

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pattern` | string | No | One of the six; omit for the catalogue |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `limit` | integer | No | Matches per response (cap 30) |
| `include_tests` | boolean | No | Include test files in scope |
| `min_files` / `min_lines` / `min_callers` / `min_siblings` / `min_directories` | integer | No | Per-pattern thresholds; only the ones the chosen predicate uses apply (`min_directories` is `reuse_candidates`' one tunable) |
| `percentile` | number | No | Percentile cutoff, `hub_functions` only (default 95); clamped 0–100 |

```
find_patterns()                          # the catalogue, with each predicate
find_patterns(pattern="hub_functions", min_callers=12)
```

---

#### `get_query_quality`

Reports what source-search retrieval got wrong, and turns a bucket of it into a
runnable eval suite. Off by default. `report` and `export` read
`.repowise/source_search/query_log.jsonl` and open no index at all, so they work
on a repository whose index is broken — which is when the error bucket is worth
reading. Only `run` needs a working index.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mode` | string | No | `report` (default), `export`, or `run` |
| `bucket` | string | For export | `error`, `wrong_owner`, `no_match`, or `low_confidence` |
| `window` | string | No | Trend granularity: `hour`, `day` (default), `week` |
| `intent` | string | No | `goal` (default) asserts the fixed behaviour; `guard` asserts today's as a floor |
| `limit` | integer | No | Offenders per bucket in report mode; cases in export mode |
| `write_to` | string | No | Filename for the exported suite, confined to `.repowise/source_search/eval/` |
| `suite` | string | For run | Filename of the suite to execute, read from that same directory |
| `verdicts` | string | No | JSON object of query to correct owner, read from that same directory |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |

Four quality buckets, each row counted under exactly one so the rates partition
the log: **error** (the search reached no corpus), **wrong_owner** (a later signal
contradicts the served owner), **no_match** (the corpus was read and declared to
have no answer), **low_confidence** (served with the caution flag). Rows the pass
could not use — unparseable lines, undated rows, rows with no corpus generation —
are declared under `caveats` rather than dropped.

Export never invents ground truth. Where the log does not establish the right
answer the case is emitted as `expect.kind = "todo"` and reports as `pending`,
neither a pass nor a failure, until a human answers it. Writing the observed owner
into the expectation would freeze the defect into a test that passes forever
because it asserts the bug.

Reads and writes are confined to `.repowise/source_search/eval/`; a path outside it
is refused rather than followed.

```
get_query_quality()                                             # report
get_query_quality(mode="report", window="week", verdicts="v.json")
get_query_quality(mode="export", bucket="no_match", write_to="week31.json")
get_query_quality(mode="run", suite="week31.json")
```

---

#### `manage_decision`

Record, review, confirm, and retire this repository's architectural decisions. Five verbs
over one store: a git-tracked JSONL journal at `$REPOWISE_DECISIONS_JOURNAL`, whose diff a
person reviews and commits.

**Recording is not confirming.** `record` always lands `proposed` — an agent that inferred a
decision from a diff has not reviewed it, and the store has to tell that apart from a rule
the team stands behind. `confirm` is the separate verb that promotes it, re-hashing the
anchors so staleness restarts from that moment. No parameter on `record` can promote a
record on the way in.

**Nothing is ever deleted.** `supersede` flips the old record to `superseded`, links it to
its successor in both directions, and leaves it readable. `get` returns the whole chain from
either end.

**Writes require the journal.** With `REPOWISE_DECISIONS_JOURNAL` unset, pointing outside the
repository, or naming an unwritable path, every mutating verb returns
`{"error": ..., "journal_available": false}` rather than falling back to the derived SQLite
table — a decision written only to a local derived store never reaches git, so no teammate
ever sees it. `list` and `get` keep serving the last projected state and say so.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | Yes | `record`, `list`, `get`, `confirm`, or `supersede` |
| `decision_id` | string | For get/confirm/supersede | The `dec-xxxxxxxx` id |
| `title` | string | For record | Short name |
| `decision` | string | For record | What was chosen |
| `why` | string | For record | What forced the choice — the part not re-derivable from the code |
| `anchors` | string[] | For record | Repo-relative files this governs: `path` or `path::Symbol`. At least one, and each must exist on disk |
| `supersedes` | string | No | Id this new record replaces, retired in the same write (record) |
| `superseded_by` | string | For supersede | Id of the successor |
| `actor` | string | For writes | Who is asking. Reported in the server log; durable attribution is the git commit that lands the diff |
| `status` | string | No | Filter: `proposed`, `active`, `superseded` (list) |
| `query` | string | No | Case-insensitive match over title, decision, and why (list) |
| `recorded_after` / `recorded_before` | string | No | ISO-8601 instant or bare date (list) |
| `limit` / `offset` | int | No | Page size (capped at 200) and offset (list) |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |

Rows come back in a total order: confirmed rules first, then proposals, then history; newest
within each, with the id as a final tiebreak (the journal stamps whole seconds, so
same-second records are routine). An unrecognised `status` is dropped and named in
`ignored_arguments`; an unparseable time bound is an error, because a dropped bound returns
*more* rows than were asked for.

**When to use:** after you had to reason out a non-obvious choice the next reader would
otherwise re-derive. Not for summarising what the code already says. Call `confirm` only when
a person asks you to.

```
manage_decision(action="record", title="Project unconfirmed rows as proposed",
                decision="Map confirmed_at=null to status=proposed in the projection.",
                why="An unconfirmed row projected as active enrols in every reader that counts governance.",
                anchors=["packages/core/src/repowise/core/analysis/decisions/journal_projection.py"],
                actor="claude")
manage_decision(action="list", status="proposed")
manage_decision(action="confirm", decision_id="dec-a1b2c3d4", actor="jon")
manage_decision(action="supersede", decision_id="dec-a1b2c3d4",
                superseded_by="dec-e5f6a7b8", actor="jon")
```

---

#### `get_reference_sites`

Every recorded occurrence of a symbol, with its position, kind and resolution confidence.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | string | Yes | Symbol id (`path::Name`) or an unambiguous bare symbol name |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `kinds` | list[string] | No | Restrict to these reference kinds (`call`, `definition`, `import`, `import_binding`, `receiver`, `reexport`, `type_ref`, `heritage`, `reference`, `identifier`, `string_ref`); omit for all |
| `min_confidence` | float | No | Drop sites below this rubric confidence, 0.0-1.0 (default 0.0) |
| `offset` | int | No | Zero-based result offset (default 0) |
| `limit` | int | No | Page size, clamped to 1-500 (default 50) |

**Returns:** one entry per *occurrence*, not per relation. Four calls in one file are four entries with distinct columns, where `get_dependents` reports a single aggregated edge. Each site carries `start_line`/`start_col`/`end_line`/`end_col`, `range_exact` (false means only the line is trusted), `kind`, `origin`, `confidence`, `tier` (`ast` or `textual`), and `occurrence_index` — non-zero marks a site the parse layer's same-line dedup would have dropped. Occurrences that could not be bound to a definition are returned at low confidence rather than omitted, and counted in `unbound`.

`confidence` is the probability that renaming the resolved target must edit this site. It is not a relevance score and must not be compared against search ranking.

Every response carries a `coverage` block: the languages this build declares, what was actually observed in this repository, and `uncovered_languages_present`. An empty `sites` list always comes with a `status` — `not_indexed`, `not_found`, `ambiguous`, or `coverage_limited` — so it is never ambiguous which of those reasons produced it.

**When to use:** before a rename or a signature change, when you need the call sites themselves rather than the fact that a dependency exists.

```
get_reference_sites(target="src/calc.ts::computeTotal")
get_reference_sites(target="computeTotal", kinds=["call"], min_confidence=0.9)
```

---

#### `preview_symbol_rename`

Every site a rename would touch, with per-site confidence. Reports only — it changes nothing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Symbol id (`path::Name`) or an unambiguous bare symbol name |
| `new_name` | string | No | Proposed name; used only to report collisions, never written |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |
| `min_confidence` | float | No | Drop sites below this rubric confidence, 0.0-1.0 (default 0.0) |
| `limit` | int | No | Maximum sites to return, clamped to 1-500 |

**Returns:** sites bound to the symbol first at their resolution confidence, then sites that merely spell the same name and bind to nothing — an ambiguous global, an unresolved call, a name inside a string literal. `mechanically_safe` marks the difference: true only when confidence clears 0.90 *and* the column range was verified, i.e. a rewriter could patch the site unattended. `summary.needs_review` counts the rest. `applies_changes` is `false` in the payload itself, so a caller cannot infer an apply path from a clean preview.

**`caveats` is the part to read before acting.** It names any language present in the repository that produces no reference sites at all, any partial-coverage language, any site whose column range could not be verified, a missing definition site, and any collision with `new_name`.

**When to use:** to size a rename before doing it, and to see which sites a mechanical rewrite cannot safely own.

```
preview_symbol_rename(symbol="src/calc.ts::computeTotal")
preview_symbol_rename(symbol="computeTotal", new_name="totalOf")
```

---

## Workspace Mode

In workspace mode (initialized with `repowise init .`), all tools accept an optional `repo` parameter:

- **Omit `repo`**: queries the default (primary) repo
- **`repo="backend"`**: targets a specific repo by alias
- **`repo="all"`**: queries across all workspace repos (fully supported by `search_codebase`; `get_context` and `get_overview` also accept it; not supported by `get_symbol`, `get_dependents`, `get_dependency_path`, or `get_execution_flows`)

The MCP server automatically enriches responses with cross-repo intelligence:
- **Co-change partners** from other repos surfaced in `get_context` and `get_risk`
- **API contract links** (HTTP, gRPC, topics) between repos
- **Package dependencies** between repos
- **Cross-repo blast radius** via the workspace-only `get_blast_radius` tool, and a cross-repo `directive` in `get_risk` PR-mode
- **Breaking-change guard**: incompatible provider-contract changes and the consumers they endanger, in the `get_risk` PR-mode `breaking_changes` directive
- **Architecture conformance**: declared dependency-rule violations and dependency cycles via the workspace-only, opt-in `get_conformance` tool, and `conformance_violations` / `dependency_cycles` in the `get_risk` PR-mode directive
- **Architecture metrics**: whole-system coupling (propagation cost), the cyclic core, per-service roles, and a deterministic 1-10 architecture score via the workspace-only `get_architecture` tool

---

## Proactive Hooks (Complementary)

In addition to the MCP tools above, `repowise init` installs AI-agent hooks (Claude Code and Codex) that provide **passive, automatic** context enrichment:

- **Claude Code PostToolUse**: broad or zero-result `Grep`/`Glob` calls can be enriched with graph context, and git operations can trigger stale-wiki notices.
- **Codex SessionStart**: Codex receives concise repowise MCP workflow guidance when a session starts.
- **Codex PostToolUse**: after edits or git operations, Codex receives a freshness reminder when indexed context may be stale.

Hooks are lightweight reminders. MCP tools are for deeper, on-demand investigation. See [Auto-Sync](../scale/AUTO_SYNC.md) and [Codex Integration](CODEX.md) for details.
