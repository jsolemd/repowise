"""Workspace routing for the source-search lane.

The coordinator is a single-repository engine: every store it opens belongs to
one repo, and its ranking and confidence are calibrated per corpus. A
workspace serves several repos, so this module is where the difference lives —
a scoped call routes to that repo's coordinator, a federated call composes
several coordinators' finished answers. Nothing a coordinator decided is
recomputed or reweighted here; per-repo calibration is a gate-protected
contract, and this layer's whole job is to compose it honestly.

Fusion policy — envelope-ranked repo blocks. Per-repo fused scores are
rank-derived (RRF), so comparing them across corpora manufactures precision
that is not there: every repo's top hit scores alike by construction. Repos
are ranked instead by what IS comparable across corpora — an exact-name hit,
whether the owner's own *name* declares the queried subject, the confidence
class each envelope asserts, and dense cosine under the one shared embedder —
and the merged list presents the winning repo's rows first with their
internal order intact, then the rest in envelope order.

The window is not the winner's block cut at ``limit``. A federated answer
that fills every slot from one repo has silently answered a workspace
question with a repository answer, so each remaining answering repo keeps a
reserved tail slot for its top row — the same bargain
:func:`~repowise.server.mcp_server.tool_search._append_symbol_backed` strikes
for files the page retrievers structurally cannot see.

Confidence composes downward, never upward: the federated class is the
winner's own, except when two repos are EACH confident of an owner — the same
relative path in two repos is still two distinct claims — in which case that
disagreement IS uncertainty, and the response says ``caution`` and lists the
competitors rather than picking one silently. When every repo declines, the
workspace declines; federation must not turn four abstentions into an answer,
and a repo whose search read no corpus at all (the ``status: "error"``
envelope) is disclosed as broken, never ranked as if it had answered.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from repowise.core.source_search.coordinator import NO_MATCH_CONCEPT_COVERAGE

log = logging.getLogger(__name__)

_CONF_ORDER = {"confident": 2, "caution": 1, "no_match": 0}
_CAUTION = "caution"
_NO_MATCH = "no_match"


def _is_error_envelope(env: dict[str, Any]) -> bool:
    """An envelope for a search that reached no corpus at all.

    The coordinator marks it ``status: "error"`` and deliberately says
    ``caution`` rather than ``no_match``, because a search that read nothing
    cannot assert absence. That placeholder caution must never compete with
    envelopes that actually searched: ranked naively it outranks a genuine
    ``no_match``, wins, and the composition would launder a broken repo into
    an ordinary answer with the error block dropped on the floor.
    """
    return env.get("status") == "error"


def _tag_scoped(response: dict[str, Any], alias: str) -> dict[str, Any]:
    """Stamp one repo's identity onto a coordinator response, in place."""
    for row in response.get("results") or []:
        row["repo"] = alias
    for cand in response.get("candidates") or []:
        cand["repo"] = alias
    owner = response.get("selected_owner")
    if owner:
        owner["repo"] = alias
    meta = response.setdefault("_meta", {})
    meta.setdefault("source_search", {})["answered_from"] = alias
    return response


def _owner_evidence(env: dict[str, Any]) -> dict[str, Any]:
    """The evidence dict belonging to the claim this envelope actually makes.

    Read from the selected owner and nowhere else, because that is the
    coordinator's own rule: evidence from another candidate cannot make
    *owner* trustworthy. An envelope with no owner asserts no ownership at all
    — its top row is the only thing it offers, and reading that keeps the key
    total instead of special-cased.
    """
    owner = env.get("selected_owner") or {}
    evidence = owner.get("evidence")
    if isinstance(evidence, dict):
        return evidence
    return ((env.get("results") or [{}])[0].get("evidence")) or {}


def _coverage(evidence: dict[str, Any], key: str) -> float:
    """One coverage fraction, or 0.0 where the envelope carries no such measure.

    Absent is neither a penalty nor a bonus. An evidence dict without the
    coverage keys is the coordinator's real shape for "retrieval succeeded but
    the co-location evidence did not" — a state it already answers with
    ``caution`` — and mapping it to 0.0 leaves the declaration test below
    false, which is what "we could not measure a subject" should mean.
    """
    value = evidence.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _name_tokens(raw_name: str) -> set[str]:
    """Word-level tokens of a file name, separator- and camelCase-split.

    The second camelCase boundary handles acronym prefixes — without it
    "HTTPServer" tokenizes as one word and "server" never matches it. Shared
    by the rival gate and the subject declaration so the two surfaces cannot
    disagree about what a name says.
    """
    return {
        part
        for part in re.split(
            r"[^a-z0-9]+",
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", raw_name).lower(),
        )
        if part
    }


def _declares_subject(env: dict[str, Any]) -> int:
    """Whether this repo's owner is *named* for the subject it was asked about.

    Two conditions, both read off the owner's own evidence:

    * ``concept_coverage >= NO_MATCH_CONCEPT_COVERAGE`` — the engine's own
      floor for a file being a plausible subject at all, reused rather than
      reinvented. Without it a file that merely happens to have one query word
      in its path wins: measured, ``neo4j/writer_retry.py`` takes "dramatiq
      broker wired to Redis with retry middleware and shutdown notifications"
      away from the real broker on the token "retry", and a search component
      takes an infra ranking-helper query on the token "search".
    * The owner's **stem** — the basename minus its extension — carries the
      subject, in one of two ways, both read against the matched tokens of
      the per-concept breakdown with the same tokenizer as the rival gate:
      **two or more matched tokens in the stem** (``mantine-theme.ts`` for a
      Mantine-theme query — a multi-token name match is the subject spelled
      out), or **one matched token together with**
      ``concept_coverage > content_concept_coverage`` (``routes/health.py``
      for a health-check query — the name carries subject the body does not).
      The stem, not the path, and not the extension: a directory segment
      names an area of the tree (measured, ``infra/zotero/service/``'s
      ``api_sync.py`` took make's BetterBibTeX query on the token "zotero"
      alone, away from ``bbt.py`` whose content matched *betterbibtex
      itself*), and an extension names a file format ("config.json" must not
      declare a json query).

    The single-token arm's coverage-difference requirement is what stops one
    incidental name word from outranking a confidence class: a search
    component whose body also carries "search" has coverage equal to content
    and does not declare on that one token. An earlier form used the
    difference test as the ONLY name signal, and that had a measured
    inversion on both sealed and open sets — an owner whose content fully
    carries every matched concept could never declare, so web's
    ``mantine-theme.ts`` (every token content-carried, the stronger dense)
    lost to graph's twin declaring on a name-only "mantine". Being named for
    the subject, in full, cannot be suppressed by also containing it — that
    is the multi-token arm.

    Envelopes without a per-concept breakdown keep the bare difference test
    as the fallback. In production that branch is unreachable — the
    coordinator always emits ``concepts``, and an empty tuple zeroes
    ``concept_coverage`` below the floor first — so its only exercisers are
    the legacy-shape stub tests; if a future wire shape drops the key, the
    weaker rule goes live silently, which this comment exists to flag.

    This does not contradict A3, it answers a different question. A3 governs
    whether a claim inside one corpus may be called *confident*, and there a
    filename may corroborate a subject but never constitute one. Which
    repository *owns* a subject is a separate question, and a file named for
    the subject is the strongest declaration of ownership a workspace has —
    ``docx_template.py`` owns docx templating, and a 493-line file that merely
    mentions docx does not. Nothing here upgrades a confidence class; the
    winner's own class is what the response says, so this only ever moves an
    answer from a confidently wrong repository to a cautiously right one.
    """
    evidence = _owner_evidence(env)
    coverage = _coverage(evidence, "concept_coverage")
    if coverage < NO_MATCH_CONCEPT_COVERAGE:
        return 0
    concepts = evidence.get("concepts")
    content = _coverage(evidence, "content_concept_coverage")
    if isinstance(concepts, list) and concepts:
        matched = {
            str(concept.get("token", "")).lower()
            for concept in concepts
            if isinstance(concept, dict) and concept.get("matched")
        }
        basename = ((env.get("selected_owner") or {}).get("file") or "").rsplit("/", 1)[-1]
        # The stem, not the full basename: "config.json" must not declare a
        # json/config query on its extension — json/sql/yaml/css are all
        # plausible query concepts AND extensions, and a declaration outranks
        # a whole confidence class. The rival gate keeps extensions: a *.json
        # twin collision across repos is real, and there the comparison is
        # symmetric so the extension cancels. The [1:] guard keeps dotfiles
        # (".env") whole.
        stem = basename.rsplit(".", 1)[0] if "." in basename[1:] else basename
        named = matched & _name_tokens(stem)
        if len(named) >= 2:
            return 1
        return 1 if named and coverage > content else 0
    return 1 if coverage > content else 0


def _envelope_strength(env: dict[str, Any]) -> tuple[int, int, int, int, float]:
    """What one repo's answer asserts, in cross-corpus-comparable terms only.

    1. ``exact_name`` — the query named the artifact. The coordinator returns
       ``confident`` on this alone, and nothing here outranks it.
    2. ``declares_subject`` — see :func:`_declares_subject`. Deliberately
       above ``confidence``: measured across the G4 federated set, five of the
       six cross-repo bleeds had the *wrong* repo answering ``confident`` off
       a term-dense file that mentions the whole query vocabulary, while the
       right repo answered ``caution`` from the file named for the subject.
       Ranking confidence first cannot see that, because both classes are
       honestly earned inside their own corpus.
    3. ``confidence`` — the class the repo committed to.
    4. ``has_owner`` — an envelope that selected no owner asserts no
       ownership, and at equal class must not beat one that did on a cosine
       coin flip (see the inline comment at the return).
    5. ``dense_cosine`` — the last resort, and the only magnitude here.

    What is deliberately *absent* is any coverage magnitude. Coverage looks
    cross-corpus comparable and is not: each corpus derives its own concept
    list and its own IDF weights, so the fractions have different denominators
    — for "Next.js metadata route handlers" one repo's concepts are
    ``[js, metadata, handlers]`` and another's are
    ``[next, js, metadata, handlers]``. Comparing those fractions across repos
    is the same error as comparing RRF scores, which is the error this module
    exists to avoid. Only the boolean above, whose inputs are the *query's*
    tokens, and dense cosine under the one shared embedder, survive the trip.
    """
    conf = _CONF_ORDER.get(env.get("confidence"), 0)
    cosines: list[float] = []
    exact = 0
    owner = env.get("selected_owner") or {}
    for evidence in (
        owner.get("evidence") or {},
        ((env.get("results") or [{}])[0].get("evidence") or {}),
    ):
        if evidence.get("exact_name"):
            exact = 1
        cos = evidence.get("dense_cosine")
        if isinstance(cos, (int, float)):
            cosines.append(float(cos))
    # An envelope that selected no owner asserts no ownership. At equal class
    # it must not beat one that did on a cosine coin flip: the composition
    # adopts the winner's owner, and a null one would erase the only real
    # owner in the workspace from the payload entirely.
    has_owner = 1 if owner.get("file") else 0
    return (exact, _declares_subject(env), conf, has_owner, max(cosines, default=-1.0))


def _order_key(item: tuple[str, dict[str, Any]]) -> tuple:
    alias, env = item
    exact, declares, conf, has_owner, cosine = _envelope_strength(env)
    return (-exact, -declares, -conf, -has_owner, -cosine, alias)


def _query_concept_tokens(envelopes: list[tuple[str, dict[str, Any]]]) -> set[str]:
    """The informative tokens the corpora derived from this query.

    Composed-envelope information only — every repo reports the concepts it
    scored against, and their union is as close to "what the query is about"
    as this layer can get without re-parsing the sentence the coordinator
    already parsed.
    """
    tokens: set[str] = set()
    for _, env in envelopes:
        sources = [env.get("selected_owner") or {}]
        sources.extend((env.get("results") or [])[:1])
        for source in sources:
            for concept in (source.get("evidence") or {}).get("concepts") or []:
                token = str(concept.get("token") or "").strip().lower()
                if token:
                    tokens.add(token)
    return tokens


def _same_name_rivals(
    winner_alias: str,
    owner_file: str,
    envelopes: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Other repos that hold a file of the same *name* as the winner's owner.

    A workspace question that names a kind of file rather than a repository —
    "not found page" — is answerable by every repo that has one, and picking
    one silently is the federated version of the disagreement
    ``competing_owners`` already exists to disclose. Measured: three repos
    hold a ``not-found`` page and the federated answer named one of them
    ``confident``, with ``competing_owners`` empty, because that field only
    fired when two repos were each *confident* of the same relative path.

    Matching is on the file name, not the relative path. Repos lay their trees
    out differently — the measured collision is ``apps/web/app/not-found.tsx``
    against ``src/app/not-found.tsx`` — so a relative-path test never fires;
    it was measured at 0 of 17 cases against the federated probe set.

    The name must also carry one of the query's own concepts. Without that
    guard every repo's ``package.json``, ``__init__.py`` and ``index.ts``
    collide by construction and the workspace would hedge on almost every
    query, which trades a rare wrong answer for a constant useless one. With
    it, the collision has to be about what was asked.
    """
    raw_name = (owner_file or "").rsplit("/", 1)[-1]
    name = raw_name.lower()
    if not name:
        return []
    tokens = _query_concept_tokens(envelopes)
    # Whole-token comparison on both sides: concept tokens are word-level, so a
    # substring test lets "db" open the gate against "dashboard.py".
    if not (tokens & _name_tokens(raw_name)):
        return []
    rivals: list[dict[str, Any]] = []
    for alias, env in envelopes:
        if alias == winner_alias:
            continue
        offered = [(env.get("selected_owner") or {}).get("file")]
        offered.extend(cand.get("path") for cand in (env.get("candidates") or []))
        for path in offered:
            if path and path.rsplit("/", 1)[-1].lower() == name:
                rivals.append({"repo": alias, "file": path, "reason": "same filename"})
                break
    return rivals


def _federated_window(
    blocks: list[tuple[str, list[dict[str, Any]]]], limit: int
) -> list[dict[str, Any]]:
    """Merge ranked repo blocks into one window that no repo can starve.

    The winner keeps the head and its internal order; every other answering
    repo is owed one tail slot for its top row, and the rows those slots
    displace are the weakest end of what was already there. Nothing about
    owner selection or within-repo order is touched — this decides only who
    is *visible*.

    When ``limit`` is at least the number of answering repos, every one of
    them is seated. Below that the window cannot pay everyone, and the debt
    is settled in envelope order from the weakest end: the winner never loses
    its top row, and the repos whose envelopes asserted least give way first.
    """
    ordered = [row for _, rows in blocks for row in rows]
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    reserved = [rows[0] for _, rows in blocks[1:] if rows]
    if not reserved:
        return ordered[:limit]

    # Reserving a slot shrinks the head, which can un-seat a row that was
    # paying for its own reservation, so the head length is a fixpoint rather
    # than one subtraction. It only ever shrinks and never past one, so this
    # settles in at most one pass per reserved row. Rows are compared by
    # identity: two repos can return dicts that compare equal and still be two
    # different answers.
    head_len = limit
    while True:
        seated = {id(row) for row in ordered[:head_len]}
        owed = [row for row in reserved if id(row) not in seated]
        shrunk = max(1, limit - len(owed))
        if shrunk >= head_len:
            break
        head_len = shrunk
    return ordered[:head_len] + owed[: limit - head_len]


@dataclass
class _FederatedAnswers:
    """One request's outcomes; failed reads never join the ranked answers."""

    envelopes: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errored: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def record(self, alias: str, result: dict[str, Any] | BaseException) -> None:
        if isinstance(result, BaseException):
            log.warning("source-search: federated leg %s failed: %r", alias, result)
            self.unavailable[alias] = f"search failed: {type(result).__name__}"
        elif _is_error_envelope(result):
            self.errored.append((alias, result))
        else:
            self.envelopes.append((alias, result))

    def source_meta(self, repo_order: list[str]) -> dict[str, Any]:
        """One disclosure policy for partial answers and all-error responses."""
        per_repo = {
            alias: dict((env.get("_meta") or {}).get("source_search") or {})
            for alias, env in [*self.envelopes, *self.errored]
        }
        for alias, env in self.errored:
            per_repo[alias].update(
                {
                    "errored": (env.get("error") or {}).get("code") or "search error",
                    "degraded": True,
                }
            )
        per_repo.update(
            (alias, {"unavailable": reason}) for alias, reason in self.unavailable.items()
        )
        return {"federated": True, "repo_order": repo_order, "repos": per_repo}


async def workspace_source_search(
    query: str,
    *,
    limit: int,
    mode: str,
    repo: str | None,
    registry: Any,
    build_meta: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    """Route, borrow readers, and attach freshness to a composed answer.

    None scopes to the default repository; "all" federates. A missing source
    lane returns None so the caller can use native search. Every borrowed
    reader stays alive through search, composition, and freshness enrichment.
    """
    from repowise.server.source_search_wiring import context_coordinator, coordinator_lease

    resolved = registry.resolve_repo_param(repo)
    async with AsyncExitStack() as readers:
        if isinstance(resolved, str):
            ctx = await registry.get(resolved)
            coordinator = await context_coordinator(ctx)
            if coordinator is None:
                return None
            await readers.enter_async_context(coordinator_lease(coordinator))
            response = await coordinator.search(
                query, limit=limit, mode=mode, base_meta=build_meta()
            )
            _tag_scoped(response, resolved)
            await _attach_workspace_freshness(response, {resolved: ctx}, [resolved])
            return response

        answers = _FederatedAnswers()
        contexts, live = await _collect_readers(resolved, registry, readers, answers.unavailable)
        if not live:
            return None
        searches = await asyncio.gather(
            *(c.search(query, limit=limit, mode=mode, base_meta={}) for _, c in live),
            return_exceptions=True,
        )
        for (alias, _), result in zip(live, searches, strict=True):
            answers.record(alias, result)
        if not answers.envelopes:
            if answers.errored:
                return _compose_all_error(answers, mode=mode, base_meta=build_meta())
            return None

        response = _compose_answer(query, answers, limit=limit, mode=mode, base_meta=build_meta())
        aliases = response["_meta"]["source_search"]["repo_order"]
        await _attach_workspace_freshness(response, contexts, aliases)
        return response


async def _collect_readers(
    aliases: list[str],
    registry: Any,
    readers: AsyncExitStack,
    unavailable: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    """Acquire each coordinator while its context is live, and lease it at once.

    A two-pass load can evict early contexts before their stores are ready on
    workspaces larger than the registry's LRU. Sequential acquisition keeps
    that bound; the caller runs the completed readers' searches in parallel.
    """
    from repowise.server.source_search_wiring import context_coordinator, coordinator_lease

    contexts: dict[str, Any] = {}
    live: list[tuple[str, Any]] = []
    for alias in aliases:
        try:
            ctx = await registry.get(alias)
        except Exception as exc:
            unavailable[alias] = f"context unavailable: {type(exc).__name__}"
            continue
        try:
            coordinator = await context_coordinator(ctx)
        except Exception as exc:
            log.warning("source-search: coordinator construction for %s failed: %r", alias, exc)
            unavailable[alias] = f"coordinator failed: {type(exc).__name__}"
            continue
        if coordinator is None:
            unavailable[alias] = "source index unavailable"
            continue
        await readers.enter_async_context(coordinator_lease(coordinator))
        contexts[alias] = ctx
        live.append((alias, coordinator))
    return contexts, live


def _compose_windows(
    ranked: list[tuple[str, dict[str, Any]]], limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve the same per-repo visibility in result and openable-path windows."""
    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    cand_blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for alias, env in ranked:
        blocks.append((alias, [{**row, "repo": alias} for row in env.get("results") or []]))
        seen: set[str] = set()
        paths: list[dict[str, Any]] = []
        for cand in env.get("candidates") or []:
            path = cand.get("path")
            if path and path not in seen:
                seen.add(path)
                paths.append({"path": path, "repo": alias})
        cand_blocks.append((alias, paths))
    return _federated_window(blocks, limit), _federated_window(cand_blocks, limit)


def _compose_answer(
    query: str,
    answers: _FederatedAnswers,
    *,
    limit: int,
    mode: str,
    base_meta: dict[str, Any],
) -> dict[str, Any]:
    """Compose finished evidence without I/O or modifying a member's answer."""
    ranked = sorted(answers.envelopes, key=_order_key)
    winner_alias, winner = ranked[0]
    results, candidates = _compose_windows(ranked, limit)

    # Copy by override, never rebuild an enumerated schema: fields this layer
    # does not own (including future ones) must survive. Copy the final bounded
    # answer deeply once, below, so callers cannot modify the input envelopes.
    response = dict(winner)
    response["results"] = results
    response["confidence"] = winner.get("confidence") or _CAUTION
    owner = winner.get("selected_owner")
    response["selected_owner"] = {**owner, "repo": winner_alias} if owner else None
    response.setdefault("mode", mode)
    if candidates:
        response["candidates"] = candidates
    else:
        response.pop("candidates", None)

    meta = dict(response.get("_meta") or {})
    meta.update(base_meta)
    meta["source_search"] = answers.source_meta([alias for alias, _ in ranked])
    # Parallel work cannot take less than its slowest leg, including errors.
    leg_timings = [
        value
        for _, env in [*answers.envelopes, *answers.errored]
        for value in [((env.get("_meta") or {}).get("timing_ms"))]
        if isinstance(value, (int, float))
    ]
    if leg_timings:
        meta["timing_ms"] = max(leg_timings)
    response["_meta"] = meta

    _disclose_competition(response, ranked)
    if response["confidence"] == _NO_MATCH:
        scope = (
            "the repositories that completed retrieval"
            if answers.errored or answers.unavailable
            else "any workspace repository"
        )
        response["note"] = (
            f"No indexed match for {query!r} in {scope}. The results (if any) "
            "are nearest neighbours, not evidence."
        )
        if answers.errored or answers.unavailable:
            response["note"] += (
                " Other repositories could not be searched; this is not a workspace-wide "
                "absence claim. trust.source_search.repos names their limitations."
            )
    return copy.deepcopy(response)


def _disclose_competition(
    response: dict[str, Any], ranked: list[tuple[str, dict[str, Any]]]
) -> None:
    """Disagreement can lower confidence, never upgrade a member's claim."""
    confident = {
        alias: env["selected_owner"]
        for alias, env in ranked
        if env.get("confidence") == "confident" and env.get("selected_owner")
    }
    if len(confident) >= 2:
        response["confidence"] = _CAUTION
        response["competing_owners"] = [
            {
                "repo": alias,
                "file": owner.get("file"),
                "reason": owner.get("reason"),
                "evidence": owner.get("evidence"),
            }
            for alias, owner in confident.items()
        ]
        response["note"] = (
            f"{len(confident)} repositories each returned a confident owner for "
            "this query. The federated confidence is caution until the query "
            "names one (repo=<alias>); competing_owners lists them all."
        )
        return

    winner_alias, _ = ranked[0]
    owner_file = (response.get("selected_owner") or {}).get("file") or ""
    rivals = _same_name_rivals(winner_alias, owner_file, ranked)
    if not rivals:
        return
    if _CONF_ORDER.get(response["confidence"], 0) > _CONF_ORDER[_CAUTION]:
        response["confidence"] = _CAUTION
    response["competing_owners"] = rivals
    response["note"] = (
        f"{len(rivals) + 1} repositories hold a file named {owner_file.rsplit('/', 1)[-1]!r}, "
        "and this query names none of them. The owner below is this workspace's best "
        "single answer, not its only plausible one; competing_owners lists the rest. "
        "Name a repository (repo=<alias>) for that repository's own answer."
    )


async def _attach_workspace_freshness(
    response: dict[str, Any],
    contexts_by_alias: dict[str, Any],
    aliases: list[str],
) -> None:
    """Use the native lane's freshness adapter for scoped and federated reads.

    One Repository read per consulted repo, scoped to the contributed paths.
    An enrichment failure never takes down an otherwise useful search.
    """
    from repowise.server.mcp_server.tool_search import _federated_freshness

    contexts = [contexts_by_alias[alias] for alias in aliases if alias in contexts_by_alias]
    if not contexts:
        return
    try:
        freshness = await _federated_freshness(contexts, response.get("results") or [])
    except Exception:
        log.debug("source-search: workspace freshness enrichment failed", exc_info=True)
        return
    if freshness:
        response.setdefault("_meta", {}).update(freshness)


def _compose_all_error(
    answers: _FederatedAnswers, *, mode: str, base_meta: dict[str, Any]
) -> dict[str, Any]:
    """No reachable repo read a corpus: retain the error, not an empty answer."""
    _, first = answers.errored[0]
    response = dict(first)
    response["results"] = []
    response.pop("candidates", None)
    response["selected_owner"] = None
    response.setdefault("mode", mode)
    meta = dict(response.get("_meta") or {})
    meta.update(base_meta)
    meta["source_search"] = answers.source_meta([])
    response["_meta"] = meta
    return copy.deepcopy(response)
