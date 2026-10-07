# MCP Tools Reference

repowise exposes a curated set of tools via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). These tools give AI coding assistants (Claude Code, Codex, Cursor, Cline, Windsurf) structured access to your codebase intelligence: dependency graph, git history, documentation, and architectural decisions.

41 tools are registered in total. A single-repo server advertises 11 by default: exactly the canonical tools. Workspace mode adds the `list_repos` discovery utility, for 12. 29 specialist tools are opt-in, subject to mode eligibility. The surface is configurable; see [Configuring the tool surface](#configuring-the-tool-surface).

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

**Canonical tools (default in both modes, 11)**
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
[get_index_status](#get_index_status)

**Workspace discovery utility (default in workspace mode, 1)**
[list_repos](#list_repos)

**Opt-in specialists (off by default everywhere, 29)**
[get_architecture](#get_architecture) &middot;
[get_blast_radius](#get_blast_radius) &middot;
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
[preview_symbol_rename](#preview_symbol_rename) &middot;
[set_finding_status](#set_finding_status) &middot;
[list_doc_files](#list_doc_files) &middot;
[resolve_library_id](#resolve_library_id) &middot;
[search_docs](#search_docs) &middot;
[expand_doc_chunk](#expand_doc_chunk) &middot;
[list_doc_libraries](#list_doc_libraries) &middot;
[read_doc](#read_doc) &middot;
[update_doc_library](#update_doc_library) &middot;
[add_doc_library](#add_doc_library) &middot;
[delete_doc_library](#delete_doc_library) &middot;
[export_doc_bundle](#export_doc_bundle) &middot;
[import_doc_bundle](#import_doc_bundle)

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
| `get_change_risk` | What a commit or range newly made worse | Before merging a commit or PR range |
| `get_why` | Architectural decisions | Before structural changes |
| `get_dead_code` | Unreachable code | Cleanup tasks |
| `get_health` | Code-health marker scores | Before refactoring, find the worst files |
| `get_index_status` | Source-search publication trust | Before relying on indexed search results |

In workspace mode, `list_repos` is also on by default so repository aliases are discoverable. It is unavailable in single-repo mode because the server is already bound to the only repository. See [Supplementary tools](#supplementary-tools).

---

## Configuring the tool surface

The default surface is deliberately small: fewer, richer tools mean fewer round-trips and less schema overhead per task. What a server advertises is resolved from three things: each tool's `default`/`requires_workspace` metadata, whether the server is in workspace mode, and an optional override.

- **Default (single-repo):** 11 tools, the canonical intelligence set plus `get_index_status`.
- **Default (workspace):** those 11 plus `list_repos`, the workspace discovery utility.
- **Opt-in tools:** `get_dependents`, `get_dependency_path`, `get_execution_flows`, `generate_refactoring_code`, `reindex_repository`, `build_task_slice`, `get_task_slice`, `extend_task_slice`, `get_query_quality`, `find_clones`, `find_patterns`, `manage_decision`, `get_reference_sites`, and `preview_symbol_rename` are registered but off by default. `get_architecture`, `get_blast_radius`, and `get_conformance` are workspace-only and, as of v0.46.0, also opt-in rather than workspace defaults. Turn them on per repo; `manage_decision` only does useful work where a decision journal is configured, and `reindex_repository` remains separate from the default read surface because it starts background work.

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
A response that would still oversize sheds whole blocks, in an order each tool
declares cheapest-loss-first, and reports `truncated: true` alongside the refs.

Every tool is budgeted. Most declare their own shed order; the rest meet a
final size guard that trims the largest blocks and records what it took. Either
way no response is returned unbounded and unflagged, and
`_meta.response_budget` reports the ceiling and delivered size when trimming
occurs, enforcement fails or full diagnostics are requested.
Resolve refs with `repowise expand <ref>` from a shell, or
`get_symbol("repowise#<ref>")` from any MCP client. See
[DISTILL.md](DISTILL.md) for the full reversibility model.

**Where the envelope rides.** Over MCP it is protocol metadata, not payload: it
comes back as `_meta` on the JSON-RPC `result`, beside `content` and
`structuredContent`. `structuredContent` is the tool's payload flat — no
`{"result": ...}` wrapper around it, and no `_meta` key inside it — and the text
content block mirrors that same flat payload, so a response carries exactly one
copy of the envelope. A tool function awaited in process, which is what the
`repowise` CLI does, still gets a single dict with `_meta` inside it; the
promotion happens in the served MCP wrapper and nowhere else.

**The `_meta` envelope.** `contract_version` is always present. Routine responses
keep trust, failure and recovery facts; `get_overview` and
`REPOWISE_MCP_DEBUG_META=1` retain full diagnostics. The text block is compact JSON
by default; `REPOWISE_MCP_PRETTY_JSON=1` changes its indentation without changing
structured content or metadata. Other fields are present only when meaningful:

| Field | When present |
|-------|--------------|
| `contract_version` | Always: version 3 of the response contract |
| `timing_ms` | Tool wall-time when full diagnostics are requested |
| `hint` | A short, conservative follow-up suggestion. On a `get_answer` reply that graded `low` while the index is behind live HEAD, it says to run `repowise update` and ask again before trusting the answer |
| `cached` | Only when `true` |
| `index_age_days` | Days since the last `repowise update` |
| `indexed_commit` | Short (12-char) SHA the index was built against |
| `live_head` | Short (12-char) SHA of the current checkout, whenever `.git/HEAD` is readable. Equal to `indexed_commit` when the index is current |
| `stale_warning` | Only on a real signal: HEAD mismatch **that actually changed files**, or age over ~90 days when git is unreachable. Two commits with identical trees (an empty commit, a no-op merge) report `index_behind` with no warning |
| `index_behind` | Whenever the live-vs-indexed comparison ran: `true` if HEAD has moved (alongside `stale_warning` when served content actually changed), `false` if the commits match. Absent means the comparison could not run (no git, or a repo-level tool that serves no file content) |
| `embedder_degraded` | Whenever an embedder is resolved, `true` or `false`. Absent means none was initialised. An **install-level** claim, latched when the embedder was built: it does not move when one query's retrieval fails |
| `embedder`, `embedder_warning` | Only when the embedder fell back to a mock/degraded mode |
| `semantic_search` | Only when `false`: semantic retrieval was unavailable or incomplete for this request. Healthy federated members may still contribute vector hits; `retrieval_degraded_repos` identifies affected native stores |
| `retrieval_degraded` | Only when a retrieval leg fell over on **this** request (`["vector"]`). The results are still served, and a miss in them is not evidence of absence. Embedding requests are retried per call; a failed native store reports its repair or loading state for the selected repository |
| `retrieval_degraded_reason` | Alongside `retrieval_degraded`: what failed, and the cause it reported |
| `retrieval_degraded_repos` | Native-store failures for repositories consulted by this request, with repair or retry hints. An unrelated repository's failure does not degrade a scoped or independent source search |
| `newer_release` | Once per server process, in the first response after the background check sees a newer repowise on PyPI: the available version, the running one, and that the MCP server needs a restart after upgrading. Repeats only for a later, newer version |
| `response_budget` | When trimmed, enforcement failed or full diagnostics are requested: `limit_chars` (the ceiling that applied), `tier` (`default` or `expanded`, chosen by whether the call passed an expansion argument), `serialized_chars` (the size delivered) |
| `scope_hint` | `get_context` and `get_answer`, when knowledge-graph layers exist that contain none of the served paths: one sentence naming up to three of them with file counts, so an agent knows which areas the answer did not touch |
| `complete` | When the response served whole units: how many symbol bodies (bounds verified against the live file) or whole files, and that they need not be re-opened. Sliced bodies and partial ranges are never counted |
| `completeness` | When capped or full diagnostics are requested: `capped`, plus `shown` and `total` summed over every collection a reducing pass counted, and `reason` naming the pass that dropped the most. The sum spans unlike collections and sits inside whatever `limit` the call already applied, so it measures what survived reduction, not what share of the repository you hold. Absent `shown`/`total` means no pass counted rows |
| `floor` | Only when the response carries a count derived by walking the indexed graph: names those fields, whose values are lower bounds because an edge the index failed to resolve is uncounted rather than proven absent |
| `state` | Only when something fired: `degraded` plus `degraded_reasons` mapping each contributing key to its reason (a synthesis reason string, the retrieval legs that broke), `partial`, `truncated`. A coarse roll-up of the response's own flags |

Read the freshness fields when present; absence means the comparison was not evaluated. Workspace search reports freshness per consulted repository in `repo_freshness`, because a federated answer has no single indexed commit.

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
well as a float.

`get_health` is the exception to the shape. It reports a misspelled `only` key
under its own older name, `unknown_only_keys`, and everything else —
`refactoring_*` and `performance_*` filters, `scope`, `counts` — in
`ignored_arguments` as a flat map of argument to the value that was dropped:

```jsonc
"ignored_arguments": { "counts": "code-shape", "refactoring_effort": "tiny" }
```

A detail lookup (`finding_id`, `plan_id`, `opportunity_id`) reports `scope` and
`counts` there too: it answers about one stored row, so a population control has
nothing to act on.

---

## `get_overview`

Architecture summary, module map, and entry points.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` |
| `include` | list[string] | No | Opt-in blocks, any combination of `"content"`, `"outline"`, `"tour"`, `"decisions"`, `"graph"`, `"ownership"` (see below) |

**Returns (default):** `title`, `content_md` (the overview essay's summary section), `key_modules` (name, path, outline section), `entry_points`, `architecture` (layer names, file counts, layer order), `code_health`, `git_health`, `next_actions` (per horizon, week and quarter: the first three stored actions as `id`, `tier`, `title`, `impact`, `done_when`, `target`, with `total` and `by_tier`; `unavailable` names stores the index predates; replaced by `next_actions_reason` when they cannot be read), `_meta`, and in workspace mode a `workspace` footer. The response's `more` field names the opt-in blocks.

**Opt-in blocks** — omitted unless named in `include`, and not computed at all when they are not:

| `include` key | Adds |
|---|---|
| `"content"` | the full overview essay in `content_md` |
| `"outline"` | `outline` — the stored wiki page tree, two rungs deep. `key_modules[].section` indexes into it |
| `"tour"` | `guided_tour` and `reading_order` — onboarding walks |
| `"decisions"` | `key_decisions`. `get_why` is the richer route |
| `"graph"` | `community_summary` — code-community clusters |
| `"ownership"` | `knowledge_map` — top 3 owners by files owned |

**When to use:** First call on any unfamiliar codebase. Gives the agent a mental map before diving into specifics. Skip on later calls in the same session; it doesn't change mid-session.

**Example calls:**

```
get_overview()
get_overview(include=["outline", "content"])
```

> **Output-schema change.** `guided_tour`, `reading_order`, `key_decisions`,
> `community_summary` and `knowledge_map` moved behind `include`; `outline` did
> too (`include=["outline"]` previously only deepened an always-present tree).
> `key_modules` no longer carries `description`, `page_id` or `parent_page_id`,
> and `architecture.layers[].description` is gone.

---

## `get_answer`

One-call RAG: retrieves over the wiki, gates synthesis on confidence, and returns a cited 2-5 sentence answer.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | Yes | Natural language question about the codebase |
| `scope` | string | No | Repository-relative path prefix to restrict retrieval to |
| `include` | list[string] | No | `["evidence"]` returns the expanded projection: no confidence-keyed trimming, and a larger response budget |
| `repo` | string | No | *(workspace only)* Target repo alias |

**Returns:** A synthesized answer with file/symbol citations, a `confidence`
label (`high`, `medium`, `low`) rating the prose, and a `retrieval_quality`
label (`high`, `partial`, `weak`) rating the evidence under it. Every grade
returns an answer; what changes is how much evidence rides along with it.
A `high` answer can be cited directly and sheds `retrieval`, `best_guesses`,
`candidates` and `fallback_targets`, keeping one quote and, when the answer is
not already grounded, one symbol body. A `medium` answer keeps one body, two
quotes and the top two evidence rows. A `low` answer keeps two bodies and the
top three evidence rows, and `fallback_targets` appears only when nothing else
was served. Read the rows the reply names rather than calling `search_codebase`
again. `include=["evidence"]` skips this trimming entirely.

When synthesis cannot run at all, no provider resolvable or the call failed, the
response carries a top-level `degraded` naming the reason, is built from
retrieval and mined rationale with no LLM involved, and keeps the fullest
evidence shape whatever it graded. It also raises `_meta.state.degraded`.
`confidence` there is graded from the retrieval actually served, not from the
missing prose: on `no-llm-provider` it is `medium` unless `retrieval_quality` is
`weak`, in which case `low`. Any other reason, a configured provider whose call
failed, stays `low`, because a retry can still produce a real answer. `high` is
unreachable on this path, since the `answer` string is assembled boilerplate.

`_meta.complete` names the symbol bodies served whole from live source, with
bounds checked against the file; do not re-open those. `_meta.scope_hint` names
up to three knowledge-graph layers holding none of the served paths, so an agent
knows which areas the answer did not touch. When an answer grades `low` and the
index is behind live HEAD, `_meta.hint` says to run `repowise update` and ask
again before trusting it.

Two path-bearing blocks, with different jobs:

| Field | Job | Confidence-gated? |
|-------|-----|-------------------|
| `retrieval` | **Evidence.** Enriched hits (summary, snippet, key symbols) to re-read when the prose needs checking. Shrinks as confidence rises, because a trustworthy answer needs less of it. | Yes |
| `candidates` | **Navigation.** The ranked shortlist of files retrieval resolved, one `{path, lines?}` entry each, up to 20. | Shape-gated: the default projection drops it at every confidence |

`candidates` is built whenever retrieval resolved anything, including on high-confidence answers where `retrieval` is deliberately empty, but the default projection drops it; ask for it with `include=["evidence"]`. It is where to look next; it is not evidence that the answer is right.

**Retrieval legs:** three, fused by Reciprocal Rank Fusion: full-text and vector search over wiki pages, plus the structural symbol index. The symbol leg is keyed on the content words of the question rather than on whether it happens to carry an identifier-shaped token, so "how does an incremental update persist symbols" reaches the same rows as `_persist_symbols`. It exists because a generated file page renders only the *public* symbol table: a private helper or a local name is not in the text the other two legs index.

**When to use:** First call on any code question. Collapses search, read, and reason into one round-trip. On a low grade, start from the evidence rows the reply already carries; reach for `search_codebase` only when `retrieval_quality` is `weak`.

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
| `include` | list[string] | No | Additional data to include: `"full_doc"` (full wiki markdown), `"callers"` (who calls this, symbol targets), `"callees"` (what this calls, symbol targets), `"ownership"` (primary owner, bus factor, contributor count), `"last_change"` (last commit date + author), `"metrics"` (PageRank, betweenness, percentiles), `"community"` (cluster membership + neighbors), `"decisions"` (full decision records; default returns titles only), `"skeleton"` (file targets only; the file with bodies elided: every signature, imports, and the bodies of the most central symbols, token-budgeted; typically ~15% of the full file's tokens), `"health"` (code-health score and biomarkers), `"doc_drift"` (the documents that name this file, and whether those documents carry drift of their own). An empty `callers`, `callees` or `used_by` list sits beside a `*_basis` object: the language, how many call edges the index resolved for it, the share of those that are guesses, and a note that unbound call sites are not counted, so an empty list means no resolved edge, not proof of none |
| `compact` | boolean | No | Default `true`. Set `false` for full structure block and importer list. |
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` is not supported |

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
get_context(targets=["src/auth/service.py"], include=["doc_drift"])
```

**Skeletons:** with `include=["skeleton"]`, file targets gain a structure-level
rendering sliced from the index's persisted symbol bounds (no parsing at query
time): every signature, the import preamble, and the bodies of the top symbols
ranked by graph centrality / hotspot / query match. Elision markers carry
1-indexed line ranges so you can range-`Read` anything back. For
structure-level questions ("what's in this file", "which function handles X")
this replaces a full file read at a fraction of the cost.

**Documentation drift, in reverse:** with `include=["doc_drift"]`, a file target
gains the documents that name it. The drift detector files a finding against the
*document*, so this is the only direction that answers "what documentation would
my change invalidate". Two separate claims ride in the block and must not be
merged: `references` lists documents that name this file and still resolve to
it, while `documents_with_drift` says a listed document carries some assertion
that no longer holds --- anywhere in it, not necessarily about this file.
`references_basis` states both limits, and is emitted on an empty answer too,
since a reference the detector cannot resolve is not listed. Documents dropped
by the repo's exclude rules are counted in `references_excluded` rather than
silently missing.

An answer the store cannot support is a refusal, not an empty list, because "no
document mentions this file" is a far stronger claim than "no drift findings".
`{"unavailable": "not_computed"}` means the table exists but the pass has never
run, which happens between an upgrade and the first update that does any work;
`"index_predates_doc_drift"` means the index is older than the table; and
`"drift_read_failed"` means the read failed for some other reason, where
reindexing is not the fix.

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
| `reference` | object | No | A structured `continuation_reference` or `fetch_reference` emitted by this tool; pass it unchanged to retain both id and repository scope. |

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
| `limit` | int | No | Max results (default 5); native hybrid results group matches from the same file |
| `mode` | string | No | `auto` (default) \| `concept` \| `symbol` \| `path` \| `hybrid` |
| `kind` | string | No | `implementation` \| `test` \| `config` \| `doc` |
| `symbol_kind` | string | No | Restrict symbol hits by kind (`function`, `class`, `method`, ...) |
| `page_type` | string | No | Restrict to one page type. The two you will reach for are `file_page` (the always-on per-file docs) and `module_page` (the subsystem/concept pages). Other stored types (`repo_overview`, `layer_page`, `scc_page`, `api_contract`, `infra_page`, `symbol_spotlight`) also filter. |
| `repo` | string | No | *(workspace only)* Target repo alias, or `"all"` to search across the workspace. Against a single-repo server `"all"` means the same as omitting it — there is one repo, and all of it is what an unqualified search returns. |

**Modes:**

With source search enabled, unfiltered `auto`, `concept` and `hybrid` queries
use source-owner evidence, including bare identifiers. Explicit `symbol` and
`path` modes, filename lookups and filters use native indexed resolvers; an
exact miss never falls through to semantic search.

Without the source lane, `auto` routes identifiers to indexed symbols, paths
to file pages, prose to wiki-semantic search and mixed queries to native hybrid
retrieval. `concept` forces wiki retrieval; `symbol` and `path` force structural
search. Native hybrid rows combine same-file matches, while symbol mode keeps
separate overload rows.

**Returns:**

- Native rows that name an openable file carry `path`. Symbol hits also carry `symbol_id`, line bounds and `signature`; `file` is a compatibility alias. Same-file matches can appear in `symbols`.
- Native concept hits carry `relevance_score`, `snippet` and `sources`. A logical page target remains in `target_path` only where it differs from `path`; it is not necessarily a file.
- Source-lane rows retain `target_path` and `symbol_path`, including nested `Outer::inner` identities, with their source-owner confidence and provenance.

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
(with `get_overview`, `get_dead_code` and query-shaped `get_why`, this accepts `repo="all"`). **On that
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
| `include` | list[string] | No | Opt-in blocks: `graph` (typed `dependents`, `consumers`, `cross_repo_links`, structural `impact_surface`, `direct_risks`), `churn` (`change_magnitude`, `risk_type`, `change_pattern`) |
| `repo` | string | No | *(workspace only)* Target repo alias |

**Returns:** Per-file `hotspot_score` (0-1 churn percentile), `health_score` (0-10), hotspot status, direct directed `dependents_count`, historical `co_change_partners` (each with a recency-decayed `weight`, not an integer count, and a `direction` of `a_to_b`, `b_to_a` or `undirected`, where `a` is the assessed file and `b` the partner; `conf_ab` and `conf_ba` are the share of each file's own commits that touched the other, and both are omitted on an index written before those commit totals were recorded, which also reads as `undirected`), blast radius, recommended reviewers, test gap analysis, and security signals. With `include=["graph"]`, `dependents` preserves direct versus transitive structural reach, `consumers` contains typed contract consumers only, and `cross_repo_links` retains both repository identities, direction, relationship type, evidence kind, and file- or repository-level granularity. Package-manifest links are repository-level and never invent a target file. Every typed relationship collection carries matching total/emitted/truncated fields. `relationship_analysis` distinguishes available-empty analysis from unavailable, degraded, partial, and source-truncated artifacts and retains artifact generation/provenance fields. Structural reach is not proof of runtime breakage.

Test-gap analysis is the same resolver `get_context` reports from, so the two
tools always agree about a file: `test_gap` is the negation of `tested`, and
the payload carries the `test_linkage_basis` plus authoritative
`guarding_tests` or naming-only `possible_tests`. Naming alone leaves
`test_gap: true`. See [Test linkage](#get_context) under `get_context` for what
each basis claims.

> **Scales.** Ratios derived from ownership or percentile columns are 0-1 (`hotspot_score`, `owner_pct`, `recent_owner_pct`); coverage and gap fields are 0-100 (`coverage_pct`, `branch_coverage_pct`, `share_of_repo_gap_pct`, `change_entropy_pct`, `churn_percentile`). The `_pct` suffix alone does not tell you which — check this table. Every emitted float is rounded to 4 significant digits.
> **Unresolved targets.** A target naming no indexed file comes back as
> `{"resolved": false, "unresolved_reason": ...}` with no counts — omitted
> rather than zeroed, since a structural zero reads as a measured one. Reasons:
> `unsupported_target_kind` (a `module:` id; risk is scored per file),
> `directory`, `not_indexed` (run `repowise update`), `no_such_path`.

> **Opt-in blocks.** `impact_surface` and `direct_risks` are pagerank floats an agent cannot rank; `change_magnitude`, `risk_type` and `change_pattern` restate numbers printed beside them. All five are computed regardless and feed `risk_summary`; `include` only decides whether they ship. `global_hotspots` accompanies a multi-target call only, being ambient orientation that a single named file does not need; it ranks by fix history the same way `defect_profile` does.

> **Scales.** Every response carries the facts that stop a misreading: unit,
> range, calibration status, and whether the value is authoritative. The
> per-field dictionary behind them never varies between calls, so it ships only
> with `include=["scales"]` - ask for it once per session, not per call. That
> dictionary describes indexed file values: `hotspot_score`, `owner_pct`, and
> `recent_owner_pct` are 0-1 ratios, while `risk_type` is an
> uncalibrated category. In PR mode, `structural_impact_score` is a deterministic,
> uncalibrated 0-10 structural-exposure heuristic; `localized` is below 4,
> `moderate` is 4 to below 7, and `broad` is 7 or above. It is not a runtime-
> breakage probability and is not authoritative for live change review.
> Deprecated `overall_risk_score` remains an exact alias, with migration metadata.
> Direct rows expose raw, unbounded `structural_score` values in
> pagerank-weighted-hotspot units. These are not comparable to
> `get_change_risk.score`. Coverage and gap fields are percentages from 0-100.
> Every emitted float is rounded to 4 significant digits.

When `changed_files` is passed, the exact serialized response starts with a `directive` block. Its core lists are the local blast radius: `may_break` (production files in structural reverse-import reach of the diff, candidates for review rather than proven breakage), `may_break_tests` (test files reached the same way, kept separate so a burst of tests doesn't crowd production impact out of the capped list), `missing_cochanges` (historical co-changers absent from the diff), compatibility `missing_tests`, and additive typed `test_recommendations`. Every recommendation retains `basis`, all retained `bases`, repository identity, source files, and evidence: `measured` means the per-test coverage map found the test; `inferred` means structural reachability found a candidate and is not coverage proof. `tests_to_run` preserves the older measured-first/fallback id projection and `tests_to_run_basis` remains `measured`, `inferred`, or `none`; `files_without_measured_tests` carries the narrower typed coverage claim. Matching total/emitted/truncated/omitted fields describe each exact pre-cap population. `coverage_analysis`, `test_inference_analysis`, and `test_analysis` distinguish available-empty evidence from unavailable, stale, partial, or degraded analysis; unavailable coverage with `tests_to_run: []` never means that no tests are needed. The full typed population is shared with the REST blast-radius response, while any budget-omitted directive rows are recoverable through the response omission marker. In workspace mode the directive also carries the cross-repo fallout of the changed repo:

- `will_break_consumers`: deprecated compatibility name for services in *other* repos that structurally depend on this one. Rows carry `claim: structural_reach`, `runtime_breakage_claim: false`, both repository roles, direction, distance, and aggregated edge kinds; the sibling `will_break_consumers_semantics` is `structural_reach_only`. Matching total/emitted/truncated fields describe the exact pre-cap structural population, while `cross_repo_relationship_analysis` labels unavailable or partial edge provenance.
- `missing_cross_repo_cochanges`: services in other repos that historically co-change with this one but aren't in the diff.
- `breaking_changes`: compatibility-named list of provider incompatibilities and comparison warnings since the last index. Entries carry `contract_id`, `kind`/`severity`, optional request/response `side`, `comparison_source`/`comparison_key`, and endpoint-exposed `impacted_consumers` (repo, service, file). OpenAPI findings require complete sides with matching fidelity; unsupported or changed extraction evidence becomes uncertainty, not field removals. Consumer links do not prove field use or runtime failure. See [Breaking-Change Guard](../scale/WORKSPACES.md#breaking-change-guard).
- `conformance_violations`: declared dependency-rule breaches the diff's repo participates in, each with the offending `source`/`target` services, the `rule` (e.g. `frontend !-> db`), and `edge_kind`. See [Architecture Conformance](../scale/WORKSPACES.md#architecture-conformance).
- `dependency_cycles`: circular service dependencies involving this repo, each with the participating `nodes` and `length`.

> **Output-schema change.** `directive.will_break` is now `directive.may_break`,
> and `directive.will_break_tests` is now `directive.may_break_tests`. Both are
> a reverse-import reachability walk: `get_risk` is given a file list, never a
> diff, so it cannot know whether the symbol an importer actually uses changed.
> The old name promised a precision the analyzer does not have.
> `will_break_consumers` temporarily keeps its old key for compatibility, but
> it is structural reach only and is explicitly marked deprecated. Only
> `breaking_changes` comes from incompatible contract diffing.

**When to use:** Before modifying files, especially hotspots. Understand what could break, who to involve in review, and whether tests cover the affected area.

**Example calls:**

```
get_risk(targets=["src/auth/middleware.ts"])
get_risk(changed_files=["src/api/routes.ts", "src/middleware/cors.ts"])
get_risk(targets=["src/auth/middleware.ts"], include=["graph", "churn"])
```

---

## `get_change_risk`

Review one commit, a `base..head` range, or uncommitted work. Unlike
`get_risk`, which evaluates indexed files and can report blast radius, this
compares the two revisions directly and needs no index refresh.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `revspec` | string | No | Commit, `base..head` range, or `base...head` (diffed from the merge-base) to score. Omit it to score uncommitted work, or pass `HEAD` when the tree is clean |
| `repo` | string | No | *(workspace only)* Target repo alias |
| `extensions` | list[string] | No | File suffixes to count, such as `[".py", ".ts"]` |
| `exclude_patterns` | list[string] | No | Gitignore-style paths to omit; combined with root `.riskignore` rules |
| `baseline` | int | No | Recent commits to sample for percentile ranking (default `200`; `0` disables every percentile, `risk_percentile` and `fix_history.percentile` alike) |
| `include` | list[string] | No | `"findings"` for every change finding, `"diagnostics"` for the raw score mechanics, `"scales"` for units and calibration |
| `finding_id` | string | No | Expand one `health_delta` finding by its id |

**Returns:** `directive` leads with a `status`
(`review_required`, `review_recommended`, `clear_in_analyzed_scope`, `unknown`),
a headline, bounded reasons, and concrete next actions.

`health_delta` is what the change newly made worse, across defect,
maintainability and performance. Both revisions are analysed from their own
content, so a finding present at head is reported only when the diff explains
it. `scope` counts changed, eligible, analysed, skipped and failed files, and
`status` distinguishes `available` from `partial` and `unavailable` — a
`partial` comparison is never a clean bill, and `skipped` says why each file
was left out. `introduced`, `worsened` and `resolved` are totals;
`top_findings` carries the three most actionable, with `findings_total` and a
recovery call for the rest.

Each finding names its `dimension`, `biomarker`, `severity`, `path`, `symbol`
and head-side `lines`, a `reason`, and an `attribution` — `basis`
(`added_lines`, `changed_symbol`, `changed_call_edge`, `new_file`,
`file_change`, `context_change`, `unknown`) with a `confidence`. Identity
ignores line numbers, so moving code introduces nothing and a rename carries
its findings across. Performance findings carry `opportunity_id` and
`opportunity_rank` and are ordered by opportunity rank and actionability, never
by defect impact, which is zero for them by construction. `inspect` gives the
exact `finding_id` call that expands one; ids are bound to the two revisions
that produced them. A finding that exactly matches a stored one also carries a
`health_reference` for `get_health(finding_id=...)`.

`fix_history` reports the recency-weighted bug-fix record of the files the
change touches, with `files` naming where the pressure sits and `percentile`
ranking it against the same measure over the repo's own recent commits. It is
the part that separates a small edit to a fragile file from a large edit to a
safe one. `available` is false when the history walk could not run.

`diff_shape` is one sentence ranking the diff's size against the repo's recent
commits, alongside the top-level `risk_percentile`, `review_priority`,
`classification` and `is_fix`. It is size and spread, never a danger verdict.
`working_tree` says whether uncommitted work was the subject.

The raw 0-10 `score` is not on the wire by default. It ranks 0.99 against lines
added on every repository measured, so the percentile beside it already carried
the ranking while the number invited being read as a probability. It remains
available, with `fallback_band`, behind `include=["diagnostics"]`.

`include=["diagnostics"]` adds the raw mechanics: `score`, `fallback_band`,
`risk_authority`, `score_measures`, `score_unit`, `baseline_sample_size`,
`features` and `drivers`. `include=["scales"]` adds each field's kind, unit, range,
calibration and thresholds. Both are identical on every call, so ask once.

It also returns `impacted_tests`, whose `tests_to_run` names the tests the
per-test coverage map proves execute the change's changed *lines* (line-precise,
so a narrower set than `get_risk`'s file-level `tests_to_run` — same field name,
because it is the same concern). It is capped at ten, with `total` and
`truncated` reporting the overflow and the tail written to the omission store:
on `truncated: true` the response carries an `omission_marker`, and
`repowise expand <ref>` returns the rest. Its `line_coverage` buckets flag
`untested_changes` (covered file, uncovered change), `stale_test_candidates`
(covered lines whose guarding test file is absent from the diff), `covered`, and
`no_coverage_data` (files absent from the map). When no map is ingested the
change is never reported as untested: `status` becomes `inferred` when the
import graph can name test files reaching the change (candidates, file-level, no
line attribution, and `line_coverage` stays empty because reaching cannot speak
to lines), and `no_map` ("run the full suite") when it cannot. `basis` carries
the same distinction in one word, always present: `measured`, `inferred`, or
`none`. `tests_to_run_kind` says what each entry is: `test_id` (a coverage-map
test id, measured) or `test_file` (inferred), null when `basis` is `none`;
`get_risk`'s directive carries the same field beside `tests_to_run_basis`. Build the
measured map with `coverage run --contexts=test` followed by
`repowise coverage add`.

When the index stores coverage, the response also carries `patch_coverage`:
the share of the change's executable lines the stored coverage ran, the same
computation and JSON shape `repowise coverage check --format json` gates on.
`patch_coverage_pct` is null when no changed line is executable, files the
coverage never names read `not_in_report` rather than 0%, and
`scope.freshness` is `stale` when the coverage was measured at another commit
than the change's head (for uncommitted work: ingested before the newest
edit), so its line numbers may describe other code. `path_gates` lists the
path-scoped gates in
`coverage.gates`, each judged on the changed files its globs match (`gate`
reads `fail` when one that is not informational fails). They are judged only
on coverage measured at the change's head and valid config; otherwise they
read `no_data`, and `scope.config_errors` names each invalid entry. The block
is absent when no coverage is stored.
Each file row carries `risk`: `fix_pressure` (recency-weighted bug-fix weight
from the checkout's git history), `dependents`, `hotspot` and `bug_magnet`
(from the index), `risky`, `reasons` and `basis` (`git_and_index`, `git`,
`index` or `unavailable`). Rows are listed riskiest first, and `risky`
(`file_count`, `covered_line_count`, `coverable_line_count`,
`patch_coverage_pct`, `threshold`, `gate`) summarizes coverage over the risky
files, null when no row's risk was assessed.
Each measured row also carries `hints`, one per uncovered range (the first
eight): `range`, `symbol` (the innermost indexed symbol, null outside any),
`tests` (up to three test files to extend, best first), `basis` (`per_test`,
`call_graph`, `import_graph` or `none`) and `total` (how many qualified before
the cap). `per_test` is measured (per-test coverage ran nearby lines);
`call_graph` and `import_graph` are inferred from the graph. `hints` is null
when the index could not be read. When changed lines are uncovered,
`directive.next_actions` gains one line naming the scope and the tests to
extend, or, when the coverage is stale, saying to re-run the tests first.

Without a `revspec`, `patch_coverage` covers everything a push would bring,
diffed from the merge-base with the CI or default base branch: `scope.label`
`origin/main...working tree` on a dirty tree (untracked files included),
`origin/main...HEAD` on a clean one, plain `working tree` when no base
resolves. For uncommitted work freshness is by time: `current` when the last
coverage ingest came after the newest edit to the changed files, else
`stale`. After a full test run the augment hook re-ingests a fresh report in
the background, and `get_change_risk` reads it once the ingest finishes.

In workspace mode the response also carries `cross_repo`, and every
`cross_repo.consumers[]` row gains a `tests` block: a `state` (`measured`,
`inferred`, `none` or `unresolved`), up to five `tests_to_run` rows carrying
`test_file`, `test_id`, `basis`, `via` and `confidence`, `total` and
`truncated` for the overflow (the tail goes to the omission store), and
`unresolved_reason` / `unresolved_detail` when the join could not be followed.
Only `tests_to_run` is capped at five; `total` is the true number of tests
found for that consumer, so `truncated: true` means `total` minus five went to
the omission store. `unresolved_reason: "lookup_failed"` is the state every row
lands in when the join itself failed, as opposed to one link that could not be
followed, and `unresolved_detail` names what failed.

> **Output-schema change.** `impacted_tests.tests` is now
> `impacted_tests.tests_to_run`, matching `get_risk`'s directive.

`is_fix` is the defect benchmark's keyword rule read over the commit subject,
not the conventional-commit type, so a `feat:` commit whose subject says it
fixes something reads true; the rule is frozen for comparability rather than
tuned. `fix_history.overlap` is the tuned view: it applies a diff-shape filter
on top of that rule, counting only commits that actually edited production
code. `fix_history` itself runs the same unfiltered rule, and `overlap` is the
one part of the three that needs an index.

When the changed files carry counted bug fixes, `fix_history.overlap` reports
per file how many past bug-fix commits touched it
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

In workspace mode the response also holds `cross_repo`, built from the artifacts
of the last `repowise update --workspace` rather than from a graph traversal. It
appears when the commit touches a file that provides a contract some other repo
consumes, or when the compatibility report attributes a finding to one of the
changed files. `consumers[]` names each link with its `provider_file`, the
consumer's `repo`, `file` and `contract_id`, the `contract_type` and the
`match_type` that joined them, plus `provider_symbol_id` and `symbol_id` when the
link is symbol-level; `consumer_repos` lists the other repos in one place.
`breaking_changes[]` carries the `contract_id`, `type`, `kind`, `severity`,
`detail`, `provider_file` and `impacted_repos` of each incompatibility or
comparison warning, together with side/source/key evidence when available.
Impacted repositories are directly linked endpoint exposure, not proof of a
runtime failure. `breaking_changes_available` says whether a detection pass ran at
all, so an empty list reads as silence rather than as an all-clear, and
`breaking_changes_as_of` stamps that half only. `consumers` is capped at ten and
`breaking_changes` at five, with `consumers_truncated` and
`breaking_changes_truncated` counting what the caps left out; the block answers
whether this commit crosses a repo boundary, and `get_blast_radius` is the tool
for the full traversal. It is absent outside workspace mode, without contract
artifacts, and when the commit touches no published file.

`branch_overlap` names the other open branches editing the files this change
edits. It is git-only, so it appears whether or not the repo is indexed; an index
only orders the shared files and adds the history rows. `base` and `current` name
the two ends of the comparison, and each `branches[]` entry carries the branch
name, `ahead` and `behind` commit counts, `last_commit` (the date), and `files[]`.
Every file row states its `basis` in words, either `same file` or
`co-change pair, N of M commits`, the second carrying the `partner` file of this
change it pairs with and appearing only under a branch that already shares a file
directly, at most three per branch. `scanned`, `total` and `truncated` report the
branch scan itself, which is bounded to the newest 50 branches by committer date.
There is no score and no percentage in the block. `branches` is capped at five and
each entry's `files` at ten, both through the shared response budget: a capped
list gains `<key>_total`, `<key>_emitted`, `<key>_truncated`, `<key>_omitted` and
`<key>_reduced_reason` beside it, and the omitted rows go to the omission store,
so on `branches_truncated` or `files_truncated` the response carries an
`omission_marker` and `repowise expand <ref>` returns the rest. `truncated` is a
different fact: it reports the branch scan bound, not a cap. The block is absent when the change has
no counted files, when no other branch edits a shared file, and when the scan
exceeds its 20-second ceiling or git cannot answer.

`independent_changes` says when the diff is several changes rather
than one. It groups the changed files by connectivity, over index edges (imports,
calls, type references, framework and dynamic edges), stored co-change pairs, and,
when `revspec` is a `base..head` range, the files each commit of that range
touched, which links them to each other. A single commit and uncommitted work
carry no commit evidence, so only a range reads it. Only a changed file that is in
the index, is not a test, and is written in a language whose resolver can emit an
import edge is eligible to be grouped: docs, config and data files are never in a
group, not even through a co-change pair, and tests never join or connect one.
`count` is the number of groups; each `groups[]` entry lists its `files` and its
`bridging_files`, the files that alone hold the group together, where moving one
out would split it (named only for groups of three or more files, and most often
the file two commits of the range share). `ungrouped_files` carries every changed
file left out of the grouping, and `summary` names the reasons in those terms:
docs, config, tests, files not in the index, or files it has never linked. `basis` states in words
what was actually checked, and its sentence changes with the subject: it names a
shared commit alongside the import, call, type reference and co-change pair when
the commits of a range were read, and omits it when there were none to read. Under
either wording it is a claim about this index and not about the code. There is no
separate key saying which sentence you got; the sentence itself says it.
`ungrouped_files` is capped at ten through the shared response budget, so a capped
list gains `ungrouped_files_total`, `_emitted`, `_truncated`, `_omitted` and
`_reduced_reason` beside it and the omitted names go to the omission store,
recoverable with `repowise expand <ref>`; nothing else in the block is capped. The
block needs an index and is absent without one, and it is absent whenever the diff
is one change: fewer than two changed files, fewer than two of them eligible to be
grouped, or fewer than two groups surviving. Under a response over budget it is
the first thing shed, ahead of `diff_shape`; `branch_overlap` sheds after
`fix_history` and before `cross_repo`.

The freshness envelope is scoped to the files this change edits, whether or not
the repo is indexed: `branch_overlap` reads files on other branches, and that
never widens what the response is about.

**When to use:** Before merging a commit or PR range, especially when you need
to assess the change itself rather than the risk of an already-indexed file.

**Example calls:**

```
get_change_risk()
get_change_risk(revspec="main..HEAD", extensions=[".py"], exclude_patterns=["tests/"])
get_change_risk(revspec="main..HEAD", include=["findings"])
get_change_risk(revspec="main..HEAD", finding_id="chf_27a13be11e7ee33f")
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
| `id` | string | No | A decision id or `ev_...` evidence id previously emitted by `get_why`; resolves it directly without relevance search |
| `reference` | object | No | An emitted evidence reference object; pass it unchanged to retain both id and repository scope. |

**Modes:**

1. **NL search**: pass a question, optionally anchored to `targets`: `get_why(query="why JWT over sessions?")` -> searches decision records.
2. **Path-based**: pass a file path as `query`: `get_why(query="src/auth/service.ts")` -> returns three lanes, `decisions` (accepted, governing), `candidates` (nobody accepted them) and `history` (accepted and since replaced), plus the file's origin story.
3. **Health dashboard**: no `query`: `get_why()` -> stale decisions, conflicts, ungoverned hotspots, retired records and accepted records that name no file.
4. **Reference lookup**: pass `id`: `get_why(id="ev_...")` -> the exact evidence and supporting decision in one call.

**Returns:** Matching decision records with title, rationale, alternatives considered, affected files, staleness score. Health mode returns stale decisions, conflicts, ungoverned hotspots, `retired_decisions` and `unscoped_decisions`.

Two health-mode lanes `counts` reported as a bare number now name their records: `retired_decisions` (superseded, deprecated, dismissed — each row carries its `lane`) and `unscoped_decisions` (accepted records naming no file). Five rows each, ranked, remainder in `_meta.omitted`; full sizes stay in `counts`, split across the three status keys for the retired lane. `active` stays count-only.

`answer_basis` names the strongest lane the response rests on: `decision`, `episode`, `rationale`, `archaeology`, or `documentation`. Only `decision` is a ruling; the rest are evidence to weigh. Absent when no lane was served, and on the health dashboard.

**The lane a record is in decides whether it binds you, and path mode puts it in one.** `decisions` holds accepted records: somebody accepted each in a recorded event naming the reason, the scope, the evidence and the accepter, so treat them as constraints. `candidates` holds records something inferred and nobody has agreed to; read them as hints and never as rules, and note the `candidates_note` beside them says so too. Nothing produces an acceptance except an explicit `repowise decision confirm` or a committed ADR that says it is accepted, so a candidate that has recurred across fifty sessions is still a candidate.

Do not read the lane off `status`. That column is a projection kept in step for readers that predate the split, and a record can carry `status: "active"` with no acceptance behind it at all. An accepted record instead carries a `currency`: `active` (still describes its code), `needs_review` (its files have moved, and it still binds), `uncheckable` (it names nothing, so nothing can check it), `superseded` or `dismissed`. A candidate carries `review_state: "open"` and no `currency`.

Path mode's `alignment` counts the lanes separately and they sum to `governing_count`, which is every record naming the file: `active_count` is what governs it, `deprecated_count` what was accepted and withdrawn, `uncheckable_count` what was accepted but names nothing, `candidate_count` what is merely awaiting review. `score` is derived from `active_count` alone, so a file with `active_count: 0` is ungoverned however many candidates name it.

The `candidates` and `history` lanes are capped at three rows each and shed first under response-budget pressure, so an absent lane means the budget was tight, not that it was empty. `get_overview`, `get_risk` directives and `get_answer` serve accepted records only; a candidate reaches none of them as an instruction.

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
| `finding_id` | string | No | Resolve an emitted stable finding `id` directly in one call |

**Returns:** Dead code findings grouped by confidence tier (high >= 0.8, medium, low). Each finding includes: file path, kind, confidence score, line count, and cleanup impact estimate. In workspace mode, confidence is lowered on findings other repos still import. `summary.call_resolution_basis` lists, per language, how many call edges the index resolved and what share are guesses, which is the graph the findings rest on.

**When to use:** Cleanup tasks, not a targeted fix. Conservative by design: `safe_only` excludes dynamically-loaded patterns and framework-decorated functions.

**Example calls:**

```
get_dead_code()
get_dead_code(min_confidence=0.8, tier="high", safe_only=true)
get_dead_code(kind="unused_export", group_by="owner")
```

---

## `get_health`

Code-health scores and findings from the stored analysis, across defect risk, maintainability and performance. With no `targets` it returns a dashboard led by `fix_first`, one ranked queue of what to fix first. With `targets` it scores those files. It reads stored analysis. Save edits and run `repowise update --include-working-tree` to refresh uncommitted changes; a commit is not required. Ordinary `repowise update` refreshes committed changes. No LLM calls.

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `targets` | list[string] | none (dashboard) | File paths or `module:<name>`. Misses are named in `unresolved` |
| `include` | list[string] | none | Blocks: `biomarkers`, `refactoring`, `trend`, `coverage`, `accuracy`, `signals`, `churn_complexity`, `doc_drift`, `semantics`, `unverified`. Dimension filters: `performance`, `defect`, `maintainability`, `advisory` |
| `only` | list[string] | none | Keep just these top-level keys; identity, totals and recovery fields always survive |
| `limit` | int | `20` | Max rows in every ranked list, capped at 50; `0` for none |
| `cursor` | int | `0` | Offset into a ranked list; the `recovery` block names the next call |
| `fix_id` | string | none | Open one `fix_first` item in full |
| `finding_id`, `plan_id` | string | none | Open one finding or refactoring plan by id |
| `opportunity_id` | string | none | Open one opportunity: `perf...` for performance, `refop...` for a refactoring |
| `refactoring_view` | string | `"diversified"` | `diversified`, `canonical` or `file_spread` |
| `refactoring_scope` | string | `fix_first` without targets, `all` with | Which open refactoring opportunities to list |
| `refactoring_type`, `refactoring_confidence`, `refactoring_effort` | string | none | Refactoring queue filters |
| `performance_view` | string | `detail` | `detail` or `summary` |
| `performance_context` | string | `production` | `production`, `tooling`, `test`, `unknown` or `all` |
| `performance_boundary`, `performance_confidence`, `performance_actionability`, `performance_sort` | string | none | Performance queue filters |
| `scope` | string | `"all"` | `production` drops test files from every figure |
| `counts` | string | `"everything"` | `code_shape` drops the git-derived half of the score |
| `repo` | string | default repo | Workspace repo alias. `"all"` is not supported |

Only one of `fix_id`, `finding_id`, `plan_id`, `opportunity_id` per call; passing two returns `mode: "conflict"`.

**Key return fields:** `mode`, `fix_first` (`lead`, up to five `items` each with a `next_call`, `totals`), `kpis`, `gap_analysis`, `worst_files`, `high_leverage_files` (ranked by `weighted_deficit`), `top_findings`, `unresolved`, and the opt-in blocks you named. `_meta.health_analysis` says whether stored analysis exists and which commit it describes.

The response is bounded. Pair `include` with `only` to keep one block, e.g. `get_health(include=["refactoring"], only=["refactoring_opportunities"])`.

```
get_health(only=["fix_first"])
get_health(fix_id="fix1_...")
get_health(targets=["src/api/server.py"], include=["signals"])
get_health(include=["performance"], only=["performance_summary"])
get_health(include=["coverage"], only=["coverage"])
```

Performance opportunities remain advisory when static evidence cannot establish a
safe transformation. Database batching or concurrency changes require the
`resource_concurrency_contract` prerequisites; follow the item's `next_call`
and inspect its validation basis before implementation. The ranked `fix_first`
queue replaces the previous independent health, refactoring and performance
directives.

### Coverage: the stored report and how far to trust it

`include=["coverage"]` returns the stored per-file rows and a repo-wide
`summary`. Every row carries `covered_line_count` beside
`total_coverable_lines`; targeted mode adds the `covered_lines` array. The
summary's `freshness.status` is `current` when the report was measured at the
indexed commit, `stale` when at another one (its line numbers may describe
code that has moved), and `unknown` when either commit is missing.
`report_paths` says how the report's own file entries mapped at ingest
(`total`, `matched`, `unmatched`, `ambiguous`, and a short `unmatched_sample`);
it is null for coverage stored before that record existed. `source_formats`
lists every report format merged.

In dashboard mode the block also carries `history`, one point per complete
(not partial) ingest (`ingested_at`, `ingested_commit_sha`, `line_coverage_pct`,
`branch_coverage_pct`), oldest first, newest 10, with `history_total`,
`history_emitted` and `history_reduced_reason: "limit"` when cut.

In targeted mode each row also carries a `decay` block: how many of the
report's covered lines are unchanged since it ran (`confirmed_lines`) and how
many have moved since (`invalidated_lines`, now unknown rather than uncovered).
`decay.drifted` is true once a fifth of a file's measurement has moved. It is a
per-file statement about lines, separate from the summary's commit-level
`freshness`.

### Performance: one lead, then drill down

Use `fix_first` to choose an intervention, then follow its `next_call` or request
`get_health(opportunity_id="perf2_...")` for the cause, evidence, plan steps and
validation. `include=["performance"]` exposes the summary and opportunity queue;
filter it with `only`, `performance_context` and the queue controls above.
A rejected filter value is echoed with accepted values in `ignored_arguments`.
No supported static pattern is not a claim that runtime performance is optimal.

Ids are stable within a performance model version. An older id resolves to
`model_state.state: "stale_model"` with `refresh_required`; it is not silently
translated to a different opportunity. Evidence rows carry public `finding_id`
values that round-trip through the detail selector.

See [Code health](../layers/CODE_HEALTH.md) for scoring, calibration and Fix first,
and [Refactoring](../layers/REFACTORING.md) for plan shapes and limitations.

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

**Returns:** In workspace mode, `workspace: true`, the workspace root, the default repo alias, and every configured repo's `alias`, config-relative `path`, and `absolute_path`. Any of those emitted identities can be passed unchanged as `repo` to workspace-aware tools. In single-repo mode, `workspace: false` and a single `"default"` alias.

**When to use:** Discovering the `repo` aliases to pass to other tools, especially in workspace mode.

```
list_repos()
```

### Workspace-only tools

*(Available only when the server is started inside a workspace; see [Workspace Mode](#workspace-mode).)*

#### `get_blast_radius`

Cross-repo structural and historical reach from a changed service.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `targets` | list[string] | Yes | Node ids (`repo` or `repo::service/path`) or repo aliases |
| `max_depth` | int | No | Reachability depth (1-8, default 3) |
| `include_behavioral` | bool | No | Include co-change (behavioral) edges (default `true`) |

**Returns:** The impacted services ranked by uncalibrated `score` (0-1 relative
path weight, not a breakage probability), each with `distance` (hops),
`structural` (a real dependency vs co-change only), and the edge kinds that
carried the impact; plus `impact_score_semantics`, `impacted_repos`,
`structural_count` / `behavioral_count`, `total_impacted`, and unresolved targets.

Each `symbol_targets[].consumers[]` row also carries a `tests` block of the same
shape as `get_change_risk`'s: which tests in that consumer repo guard the
symbol's contracts, capped at five with the tail in the omission store, and a
named reason when a link could not be followed.

**When to use:** Before changing a high-fan-out provider, see who structurally
consumes it across repo boundaries. Structural reach outweighs historical
co-change in ranking, but neither is a runtime-breakage claim. Reads the same
system graph the [Live System Map](../scale/WORKSPACES.md#live-system-map) renders.

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

**Off by default twice over:** it must be opted into the tool surface (`mcp.tools: ["+generate_refactoring_code"]`), and generation remains unavailable unless `refactoring.llm.enabled: true` is set in the repo's `.repowise/config.yaml`. A valid plan id still resolves while generation is disabled, returning the canonical plan plus `generation.available: false`. When enabled, it uses the repo's configured LLM provider/model (bring your own key) and caches results by a content hash, so an unchanged plan never regenerates.

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

`list` and `get` report `repo` and `journal_exists` — `journal_exists: false` under journal
mode means this repo has no journal file yet, not zero decisions; `record` creates it and
says so with `journal_created: true`.

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
| `repo` | string | No | *(workspace only)* Target repo alias; `"all"` lists across repos (writes and get name one) |

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

Every response carries a `coverage` block: the languages this build declares, what was actually observed in this repository, and `uncovered_languages_present`. An empty `sites` list always comes with a `status` — `not_indexed`, `not_found`, `ambiguous`, or `coverage_limited` — so it is never ambiguous which of those reasons produced it. Served positions are exactly as fresh as the last index; `_meta.working_tree` discloses any drift on the cited files (absent = checked-and-clean; see `preview_symbol_rename` below for both shapes of the block).

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

**Returns:** sites bound to the symbol first at their resolution confidence, then sites that merely spell the same name and bind to nothing — an ambiguous global, an unresolved call, a name inside a string literal. `mechanically_safe` marks the difference: true only when confidence clears 0.90, the column range was verified, *and* the file's working-tree freshness could be verified with the file undiverged since the index was built — i.e. a rewriter could patch the site unattended. A file with uncommitted changes newer than the index demotes every one of its sites, and a working tree that could not be checked at all (no git repository, a diff failure or timeout) demotes all of them. `summary.needs_review` counts the rest. `applies_changes` is `false` in the payload itself, so a caller cannot infer an apply path from a clean preview.

**`caveats` is the part to read before acting.** A freshness problem always leads the list: either the named files changed after the index was built (positions may have shifted; re-index before a mechanical rewrite) or working-tree freshness could not be established at all, with the reason. After that it names any language present in the repository that produces no reference sites, any partial-coverage language, any site whose column range could not be verified, a missing definition site, and any collision with `new_name`.

**`_meta.working_tree`** carries the freshness evidence in both tools. Absent means checked-and-clean — the live comparison ran and found no served file diverged. Present in one of two shapes: `{checked: true, served_modified: [...], served_deleted: [...], note}` when served files have uncommitted changes newer than the index, or `{checked: false, reason, note}` when the working tree could not be read (`not_a_git_repository` is permanent for a corpus indexed from a non-git directory; `git_diff_failed`/`git_diff_timeout` are transient). The commit-scoped fields (`index_age_days`, `indexed_commit`, `stale_warning`) ride `_meta` as on every tool.

**When to use:** to size a rename before doing it, and to see which sites a mechanical rewrite cannot safely own.

```
preview_symbol_rename(symbol="src/calc.ts::computeTotal")
preview_symbol_rename(symbol="computeTotal", new_name="totalOf")
#### `set_finding_status`

Records a durable disposition on one refactoring plan — the write half of the findings triage loop. `get_health(include=["refactoring"])` and the generated task prompts ask an agent to flag false positives, but until this tool there was nowhere to record that verdict from inside the agent loop. The status is stored on the plan row, and the analyzer's finalizer never re-emits a `false_positive` plan, so the triage survives every later run.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `suggestion_id` | string | Yes | The `id` or `public_id` of a plan from `get_health(include=["refactoring"])` |
| `status` | string | Yes | One of `open`, `acknowledged`, `resolved`, `false_positive` |
| `repo` | string | No | *(workspace only)* Target repo alias |
| `reason` | string | No | Free-text audit note stored on the row (defaults to `"agent"`) |

- `false_positive` — the plan is wrong for this repository; it is never re-emitted on future runs.
- `acknowledged` — real, but the team is consciously not acting now; stays visible, stops counting as unheard.
- `resolved` — the change landed (or the code moved on); a person-resolved plan stays resolved even if the detector still fires.
- `open` — reset a prior decision.

**When to use:** After `get_health(include=["refactoring"])` surfaces a plan you have judged, so the verdict becomes durable state instead of a one-off remark.

```
set_finding_status(suggestion_id="a1b2c3d4", status="false_positive", reason="false alarm: the class is a DTO")
```

---

## Workspace Mode

In workspace mode (initialized with `repowise init .`), all tools accept an optional `repo` parameter:

- **Omit `repo`**: queries the default (primary) repo
- **`repo="backend"`**: targets a specific repo by any `alias`, `path`, or `absolute_path` emitted by `list_repos`
- **`repo="all"`**: queries across all workspace repos (fully supported by `search_codebase`; `get_context` and `get_overview` also accept it; not supported by `get_symbol`, `get_dependents`, `get_dependency_path`, or `get_execution_flows`)

The MCP server automatically enriches responses with cross-repo intelligence:
- **Co-change partners** from other repos surfaced in `get_context` and `get_risk`
- **API contract links** (HTTP, gRPC, topics) between repos
- **Package dependencies** between repos
- **Cross-repo blast radius** via the workspace-only `get_blast_radius` tool, and a cross-repo `directive` in `get_risk` PR-mode
- **Breaking-change guard**: provider incompatibilities and explicit comparison uncertainty, plus endpoint-exposed consumers, in the `get_risk` PR-mode `breaking_changes` directive
- **Architecture conformance**: declared dependency-rule violations and dependency cycles via the workspace-only, opt-in `get_conformance` tool, and `conformance_violations` / `dependency_cycles` in the `get_risk` PR-mode directive
- **Architecture metrics**: whole-system coupling (propagation cost), the cyclic core, per-service roles, and a deterministic 1-10 architecture score via the workspace-only `get_architecture` tool

---

## Proactive Hooks (Complementary)

In addition to the MCP tools above, `repowise init` installs AI-agent hooks (Claude Code and Codex) that provide **passive, automatic** context enrichment:

- **Claude Code PostToolUse**: broad or zero-result `Grep`/`Glob` calls can be enriched with graph context, git operations can trigger stale-wiki notices, and with `hooks.coverage_reingest: true` (opt-in) a full test run, passing or failing, re-ingests its fresh coverage report in the background ([details](HOOKS.md#what-gets-written-where)).
- **Codex SessionStart**: Codex receives concise repowise MCP workflow guidance when a session starts.
- **Codex PostToolUse**: after edits or git operations, Codex receives a freshness reminder when indexed context may be stale.

Hooks are lightweight reminders. MCP tools are for deeper, on-demand investigation. See [Auto-Sync](../scale/AUTO_SYNC.md) and [Codex Integration](CODEX.md) for details.

## Library documentation

The optional RepoWise docs worker provides external library references. These tools share the RepoWise MCP endpoint and dashboard at `/docs-libraries`; they are independent of repository code indexing. Configure `REPOWISE_DOCS_URL` (default `http://127.0.0.1:8101`) and optional `REPOWISE_DOCS_TOKEN`. The worker has no public MCP endpoint.

### `list_doc_files`

Browse indexed documentation files within a library, with a path filter and exact pagination totals.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | yes |
| `query` | string | no |
| `offset` | integer | no |
| `limit` | integer | no |
| `output` | string | no |

### `resolve_library_id`

Resolve a library name to its full indexed documentation library ID.

| Argument | Type | Required |
|---|---|---|
| `library_name` | string | yes |
| `query` | string | no |
| `output` | string | no |

### `search_docs`

Search indexed external documentation for one library using hybrid retrieval. Use `exact_match=true` for API/function lookups.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | no |
| `library_ids` | array | no |
| `query` | string | yes |
| `limit` | integer | no |
| `chunk_types` | array | no |
| `include_code_blocks` | boolean | no |
| `exact_match` | boolean | no |
| `output` | string | no |

### `expand_doc_chunk`

Expand one documentation chunk with surrounding file context.

| Argument | Type | Required |
|---|---|---|
| `chunk_id` | string | yes |
| `lines_before` | integer | no |
| `lines_after` | integer | no |
| `output` | string | no |

### `list_doc_libraries`

List indexed documentation libraries and their status. Paginated: pass `offset` + `limit`; `next_offset` in the response points to the next page. Pass `filter` to substring-match against library_id, name, description, or repo (case-insensitive) — useful when the registry has dozens of libraries (e.g. `filter='gsap'`).

| Argument | Type | Required |
|---|---|---|
| `include_stats` | boolean | no |
| `status` | string | no |
| `filter` | string | no |
| `offset` | integer | no |
| `limit` | integer | no |
| `output` | string | no |

### `read_doc`

Read a full indexed documentation file or one anchored section.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | yes |
| `path` | string | yes |
| `file_path` | string | no |
| `section` | string | no |
| `max_tokens` | integer | no |
| `output` | string | no |

### `update_doc_library`

Queue a docs reindex for one library.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | yes |
| `force` | boolean | no |
| `output` | string | no |

### `add_doc_library`

Register a new documentation library and queue its initial index.

| Argument | Type | Required |
|---|---|---|
| `repo` | string | yes |
| `name` | string | yes |
| `description` | string | no |
| `docs_path` | string | no |
| `branch` | string | no |
| `include_patterns` | array | no |
| `exclude_patterns` | array | no |
| `output` | string | no |

### `delete_doc_library`

Delete an indexed documentation library and all of its stored data.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | yes |
| `confirm` | boolean | yes |
| `output` | string | no |

### `export_doc_bundle`

Export one library's indexed docs as a portable bundle.

| Argument | Type | Required |
|---|---|---|
| `library_id` | string | yes |
| `output` | string | no |

### `import_doc_bundle`

Import a previously exported documentation bundle.

| Argument | Type | Required |
|---|---|---|
| `bundle_path` | string | yes |
| `output` | string | no |
