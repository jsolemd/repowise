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
from typing import Any, Callable

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
    * ``concept_coverage > content_concept_coverage`` — some of that subject
      is carried by the path and not by the body. This is the *difference*
      between A3's permissive and strict coverage, so it needs nothing that is
      not already on the wire.

    Together they read: the engine considers this owner a plausible subject,
    and its name declares part of that subject its content does not.

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
    content = _coverage(evidence, "content_concept_coverage")
    declares = coverage >= NO_MATCH_CONCEPT_COVERAGE and coverage > content
    return 1 if declares else 0


def _envelope_strength(env: dict[str, Any]) -> tuple[int, int, int, float]:
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
    4. ``dense_cosine`` — the last resort, and the only magnitude here.

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
            for concept in ((source.get("evidence") or {}).get("concepts") or []):
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
    name_tokens = {
        part
        for part in re.split(
            r"[^a-z0-9]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_name).lower()
        )
        if part
    }
    if not (tokens & name_tokens):
        return []
    rivals: list[dict[str, Any]] = []
    for alias, env in envelopes:
        if alias == winner_alias:
            continue
        offered = [(env.get("selected_owner") or {}).get("file")]
        offered.extend(cand.get("path") for cand in (env.get("candidates") or []))
        for path in offered:
            if path and path.rsplit("/", 1)[-1].lower() == name:
                rivals.append(
                    {"repo": alias, "file": path, "reason": "same filename"}
                )
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


async def workspace_source_search(
    query: str,
    *,
    limit: int,
    mode: str,
    repo: str | None,
    registry: Any,
    build_meta: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    """Serve one source-search call in workspace mode, or ``None`` to fall
    through to the stock path (the same fail-soft contract as single-repo:
    a missing lane must degrade to the search that existed before it, never
    to an error the stock path would not have produced).

    ``repo=None`` scopes to the workspace's default repo and ``repo="all"``
    federates — the same resolution the wiki lane applies, so the two lanes
    cannot disagree about what an argument means.
    """
    from repowise.server.source_search_wiring import context_coordinator

    resolved = registry.resolve_repo_param(repo)

    if isinstance(resolved, str):
        ctx = await registry.get(resolved)
        coordinator = await context_coordinator(ctx)
        if coordinator is None:
            return None
        response = await coordinator.search(
            query, limit=limit, mode=mode, base_meta=build_meta()
        )
        _tag_scoped(response, resolved)
        # One freshness vocabulary for workspace mode, scoped and federated
        # alike: the per-repo map plus the roll-up, never a bare single-repo
        # commit published as though it described the workspace.
        await _attach_workspace_freshness(response, {resolved: ctx}, [resolved])
        return response

    unavailable: dict[str, str] = {}
    contexts_by_alias: dict[str, Any] = {}
    live: list[tuple[str, Any]] = []
    # Context and coordinator are resolved together, per repo, in one pass.
    # Two-pass collection (all contexts, then all coordinators) breaks on a
    # workspace larger than the registry's LRU cap: fetching the sixth context
    # evicts the first while it is still in hand, its background store load is
    # cancelled, and its readiness event never fires. Building the coordinator
    # immediately gives each context its stores while the context is
    # guaranteed live. The searches below still run in parallel.
    for alias in resolved:
        try:
            ctx = await registry.get(alias)
        except Exception as exc:
            unavailable[alias] = f"context unavailable: {type(exc).__name__}"
            continue
        try:
            coordinator = await context_coordinator(ctx)
        except Exception as exc:
            # Guarded for the same reason the search gather below is: one
            # member repo's broken wiring must degrade to that repo being
            # disclosed as unavailable, never to the whole workspace erroring
            # where the stock path would have answered.
            log.warning(
                "source-search: coordinator construction for %s failed: %r",
                alias,
                exc,
            )
            unavailable[alias] = f"coordinator failed: {type(exc).__name__}"
            continue
        if coordinator is None:
            unavailable[alias] = "source index unavailable"
        else:
            contexts_by_alias[alias] = ctx
            live.append((alias, coordinator))
    if not live:
        return None

    searches = await asyncio.gather(
        *(c.search(query, limit=limit, mode=mode, base_meta={}) for _, c in live),
        return_exceptions=True,
    )
    envelopes: list[tuple[str, dict[str, Any]]] = []
    errored: list[tuple[str, dict[str, Any]]] = []
    for (alias, _), result in zip(live, searches):
        if isinstance(result, BaseException):
            log.warning(
                "source-search: federated leg %s failed: %r", alias, result
            )
            unavailable[alias] = f"search failed: {type(result).__name__}"
        elif _is_error_envelope(result):
            # A repo that read no corpus produced no answer. It never ranks,
            # never wins, and never lends its placeholder caution to the
            # composition — it is disclosed beside the other unavailability.
            errored.append((alias, result))
        else:
            envelopes.append((alias, result))

    if not envelopes:
        if errored:
            return _compose_all_error(
                errored, unavailable, mode=mode, build_meta=build_meta
            )
        return None

    ranked = sorted(envelopes, key=_order_key)
    winner_alias, winner = ranked[0]

    confident = [
        (alias, env["selected_owner"])
        for alias, env in envelopes
        if env.get("confidence") == "confident" and env.get("selected_owner")
    ]
    conflict = len({alias for alias, _ in confident}) >= 2
    confidence = _CAUTION if conflict else (winner.get("confidence") or _CAUTION)

    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for alias, env in ranked:
        rows = env.get("results") or []
        for row in rows:
            row["repo"] = alias
        blocks.append((alias, list(rows)))
    results = _federated_window(blocks, limit)

    # Same tail reserve as the results window, for the same reason. This block
    # is documented as the one to Read, so starving the losing repo out of it
    # takes away the only openable path that repo offered — the results row
    # names a file, the candidate is what a caller opens.
    seen: set[tuple[str, str]] = set()
    cand_blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for alias, env in ranked:
        paths: list[dict[str, Any]] = []
        for cand in env.get("candidates") or []:
            path = cand.get("path")
            if not path or (alias, path) in seen:
                continue
            seen.add((alias, path))
            paths.append({"path": path, "repo": alias})
        cand_blocks.append((alias, paths))
    candidates = _federated_window(cand_blocks, limit)

    owner = winner.get("selected_owner")
    if owner:
        owner = dict(owner)
        owner["repo"] = winner_alias

    # A query that names a kind of file rather than a repository is answerable
    # by every repo that has one. Disclose the rest rather than picking one
    # silently, and hedge — downward only, so a caution answer stays caution.
    rivals = _same_name_rivals(winner_alias, (owner or {}).get("file") or "", envelopes)
    if rivals and _CONF_ORDER.get(confidence, 0) > _CONF_ORDER[_CAUTION]:
        confidence = _CAUTION

    per_repo: dict[str, Any] = {}
    for alias, env in envelopes:
        per_repo[alias] = (env.get("_meta") or {}).get("source_search") or {}
    for alias, env in errored:
        per_repo[alias] = {
            "errored": ((env.get("error") or {}).get("code") or "search error"),
            "degraded": True,
        }
    for alias, reason in unavailable.items():
        per_repo[alias] = {"unavailable": reason}

    # Compose by copying the winner's envelope and overriding, never by
    # rebuilding an enumerated key set. Every field the coordinator serves
    # that this layer does not explicitly own — a note, an exact_match flag,
    # anything added later — survives the trip, which is what keeps the
    # federated and scoped lanes speaking the same response contract.
    response: dict[str, Any] = copy.deepcopy(winner)
    response["results"] = results
    response["confidence"] = confidence
    response["selected_owner"] = owner
    response.setdefault("mode", mode)
    if candidates:
        response["candidates"] = candidates
    else:
        response.pop("candidates", None)

    meta = dict(response.get("_meta") or {})
    meta.update(build_meta())
    meta["source_search"] = {
        "federated": True,
        "repo_order": [alias for alias, _ in ranked],
        "repos": per_repo,
    }
    # timing_ms is a field this layer owns and must override: the deep copy
    # carried the WINNER leg's timing, and a 12ms winner beside an 880ms loser
    # published 12ms for a call that took at least 880. The legs ran in
    # parallel, so the call's floor is the slowest leg.
    leg_timings = [
        value
        for _, env in envelopes
        for value in [((env.get("_meta") or {}).get("timing_ms"))]
        if isinstance(value, (int, float))
    ]
    if leg_timings:
        meta["timing_ms"] = max(leg_timings)
    response["_meta"] = meta

    if conflict:
        # Ranked order, with the owners' evidence intact: the reader deciding
        # between claims should not need a second round-trip, and this list
        # must never resurrect the workspace-config order the ranking left.
        confident_by_alias = dict(confident)
        response["competing_owners"] = [
            {
                "repo": alias,
                "file": confident_by_alias[alias].get("file"),
                "reason": confident_by_alias[alias].get("reason"),
                "evidence": confident_by_alias[alias].get("evidence"),
            }
            for alias, _ in ranked
            if alias in confident_by_alias
        ]
        response["note"] = (
            f"{len(confident)} repositories each returned a confident owner for "
            "this query. The federated confidence is caution until the query "
            "names one (repo=<alias>); competing_owners lists them all."
        )
    elif rivals:
        name = ((owner or {}).get("file") or "").rsplit("/", 1)[-1]
        response["competing_owners"] = rivals
        response["note"] = (
            f"{len(rivals) + 1} repositories hold a file named {name!r}, and "
            "this query names none of them. The owner below is this "
            "workspace's best single answer, not its only plausible one; "
            "competing_owners lists the rest. Name a repository "
            "(repo=<alias>) for that repository's own answer."
        )
    elif confidence == _NO_MATCH:
        response["note"] = (
            f"No indexed match for {query!r} in any workspace repository. The "
            "results (if any) are nearest neighbours, not evidence — "
            "_meta.source_search.repos names each corpus and commit consulted."
        )
    await _attach_workspace_freshness(
        response, contexts_by_alias, [alias for alias, _ in ranked]
    )
    return response


async def _attach_workspace_freshness(
    response: dict[str, Any],
    contexts_by_alias: dict[str, Any],
    aliases: list[str],
) -> None:
    """M9: give the workspace answer the freshness envelope it never had.

    Delegates to the wiki lane's adapter so both lanes read one Repository row
    per consulted repo and scope each repo's warning to the paths it actually
    contributed — one implementation, one vocabulary (``repo_freshness`` plus
    the roll-up). Fail-soft as everything here: an enrichment failure leaves
    the answer standing, because freshness is disclosure about the response,
    never a reason not to serve it.
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
    errored: list[tuple[str, dict[str, Any]]],
    unavailable: dict[str, str],
    *,
    mode: str,
    build_meta: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Every reachable repo read no corpus: compose an error, not an answer.

    The per-repo error envelope exists so a caller cannot mistake "nothing
    was searched" for "nothing matched"; a workspace of them must make the
    same statement. The first repo's envelope is copied whole — ``status``,
    ``error`` block, its deliberate ``caution`` — and the federation meta
    discloses every leg. Nothing here is an answer, and no field pretends
    otherwise.
    """
    first_alias, first = errored[0]
    response = copy.deepcopy(first)
    response["results"] = []
    response.pop("candidates", None)
    response["selected_owner"] = None
    response.setdefault("mode", mode)

    per_repo: dict[str, Any] = {}
    for alias, env in errored:
        per_repo[alias] = {
            "errored": ((env.get("error") or {}).get("code") or "search error"),
            "degraded": True,
        }
    for alias, reason in unavailable.items():
        per_repo[alias] = {"unavailable": reason}

    meta = dict(response.get("_meta") or {})
    meta.update(build_meta())
    meta["source_search"] = {
        "federated": True,
        "repo_order": [],
        "repos": per_repo,
    }
    response["_meta"] = meta
    return response
