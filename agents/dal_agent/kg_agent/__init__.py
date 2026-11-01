"""KG Agent — multi-step ReAct orchestrator federating mcp_kg (graph/metadata),
mcp_dal (product/data), and mcp_var (resilience) tool catalogs.

This is the public API the UI's Flask backend consumes. Two entry points:
- `run_agent`: async-native (preferred for any async caller).
- `run_agent_sync`: thin asyncio.run() wrapper for sync Flask handlers.

Both return a `KgOutput` matching the wire shape in
`ui/UI_INTEGRATION.md` §3 — drop-in replacement for the response the UI
gets from `mcp_kg.kg` today.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime
from uuid import uuid4

from .config import (
    KG_AGENT_CACHE_ENABLED,
    KG_AGENT_LOG_LEVEL,
    KG_AGENT_MAX_STEPS,
)
from .llm import resolve_model
from .models import KgOutput, Rendering, Graph, TraceStep
from .react import _strategy as _agent_strategy
from .react import run_react
from .synthesizer import synthesize


logging.basicConfig(
    level=KG_AGENT_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kg_agent")


# ---------------------------------------------------------------------------
# Read-through cache. See CACHE_DESIGN.md.
# Key: (sha256(normalised_question), resolved_model). Substrate: the existing
# `runs` table. No new infra; sub-ms lookup; same store as the audit trail.
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _normalize_question(q: str) -> str:
    """Cache normalisation — collapse whitespace, lowercase, strip. Per the
    design doc this is intentionally weak (no stopword stripping or token
    sorting); paraphrase detection is its own project."""
    return _WS_RE.sub(" ", (q or "").strip().lower())


def _hash_question(q: str) -> str:
    """SHA-256 of the normalised question. Hex string, fits in VARCHAR(64)."""
    return hashlib.sha256(_normalize_question(q).encode("utf-8")).hexdigest()


def _is_cacheable_output(output: KgOutput) -> bool:
    """Per CACHE_DESIGN.md §"What NOT to cache": skip errors, blanks, and
    `**Error:**`-prefixed answers. Filter at write time so the read path
    can be a clean SELECT."""
    if output.status != "completed":
        return False
    body = (output.answer_markdown or "").strip()
    if not body:
        return False
    if body.lstrip().lower().startswith("**error:**"):
        return False
    return True


async def _cache_lookup(question: str, resolved_model: str) -> KgOutput | None:
    """Read-through cache lookup. Returns a KgOutput on hit (with
    `cache_hit=True`) or None on miss / disabled / DB error.

    Best-effort: any exception is logged and treated as a miss so a flaky
    persistence layer never blocks a user query.
    """
    if not KG_AGENT_CACHE_ENABLED:
        return None

    try:
        from sqlalchemy import select
        from .persistence.db import get_session
        from .persistence.models import Run

        question_hash = _hash_question(question)

        async with get_session() as session:
            # No time filter — v2 cache lives forever; only an explicit
            # clear_cache() / POST /api/cache/clear nulls a row's payload.
            # See CACHE_DESIGN_V2.md.
            stmt = (
                select(Run)
                .where(Run.question_hash == question_hash)
                .where(Run.model == resolved_model)
                .where(Run.status == "completed")
                .where(Run.output_json.is_not(None))
                .order_by(Run.finished_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None or row.output_json is None:
                return None

            try:
                output = KgOutput.model_validate_json(row.output_json)
            except Exception as exc:
                # Stored payload doesn't deserialise (KgOutput shape drift,
                # corrupted row, etc.). Treat as miss; the live run will
                # write a fresh, valid envelope.
                logger.warning("cache hit but output_json failed to parse: %s", exc)
                return None

            output.cache_hit = True
            return output
    except Exception as exc:
        logger.warning("cache lookup failed (treating as miss): %s", exc)
        return None


async def clear_cache(
    *,
    question: str | None = None,
    model: str | None = None,
) -> int:
    """Manual cache invalidation. NULLs `output_json` and `question_hash`
    on matching rows so future cache lookups miss; the underlying audit
    history in `runs` is preserved (we don't DELETE).

    - No args: clear ALL cache entries.
    - `question` only: clear all model variants of that question.
    - `model` only: clear all questions for that model.
    - Both: clear that specific (question, model) cell.

    Returns the number of rows touched. See CACHE_DESIGN_V2.md.
    """
    try:
        from sqlalchemy import update
        from .persistence.db import get_session
        from .persistence.models import Run

        async with get_session() as session:
            stmt = update(Run).values(output_json=None, question_hash=None)
            if question:
                stmt = stmt.where(Run.question_hash == _hash_question(question))
            if model:
                stmt = stmt.where(Run.model == resolve_model(model))
            # Always require there to be a cached payload — saves touching
            # the millions of pre-cache rows + the access-log hit rows.
            stmt = stmt.where(Run.output_json.is_not(None))
            result = await session.execute(stmt)
            return result.rowcount or 0
    except Exception as exc:
        logger.warning("clear_cache failed: %s", exc)
        return 0


def clear_cache_sync(
    *,
    question: str | None = None,
    model: str | None = None,
) -> int:
    """Sync wrapper of clear_cache for Flask handlers."""
    return asyncio.run(clear_cache(question=question, model=model))


async def run_agent(
    question: str,
    max_steps: int = KG_AGENT_MAX_STEPS,
    user_id: str | None = None,
    client: str | None = None,
    mcp_servers: list[tuple[str, str]] | None = None,
    model: str | None = None,
) -> KgOutput:
    """Run the ReAct loop, persist the run + steps, return KgOutput.

    `mcp_servers` overrides which MCP servers the agent federates over —
    a list of `(label, url)` pairs. Default (None) uses
    `config.MCP_SERVERS` — mcp_kg :8001, mcp_dal :8002, mcp_var :8003 in
    production. Useful for tests that want to point at a specific subset.

    `model` overrides `PROVIDER_MODEL_NAME` for this single invocation —
    the UI's per-user gating layer routes premium users to a higher-tier
    model without touching the host env. None preserves the env default.

    Persistence is best-effort — a DB outage degrades to "agent works,
    history not recorded" rather than failing the user-visible call.
    """
    resolved_model = resolve_model(model)

    # Read-through cache — short-circuits the ReAct loop for repeats.
    # On a hit, write a `runs` row marked cache_hit=True so the dashboard
    # / analytics see the request even though no LLM/MCP work happened.
    # The original cached row keeps its data; this row is the access log.
    started_at = datetime.utcnow()
    lookup_start = time.monotonic()
    # The cache is keyed by question and model only, so a run under a non-default
    # agent strategy must neither read nor write it.
    default_strategy = _agent_strategy()[0] == "react"
    cached = await _cache_lookup(question, resolved_model) if default_strategy else None
    lookup_elapsed = time.monotonic() - lookup_start
    if cached is not None:
        logger.info(
            "cache hit (model=%s, question_hash=%s…, lookup=%.3fs)",
            resolved_model, _hash_question(question)[:12], lookup_elapsed,
        )
        # Best-effort access-log write. Mirrors the miss-path _persist
        # call but with cache_hit=True, steps_used=0, no trace, and the
        # tool_invoked carried over from the cached envelope.
        await _persist(
            run_id=uuid4().hex,
            question=question,
            status=cached.status,
            started_at=started_at,
            finished_at=datetime.utcnow(),
            elapsed=lookup_elapsed,
            max_steps=max_steps,
            trace=[],
            final_answer=cached.answer_markdown,
            tool_invoked=cached.tool_invoked,
            user_id=user_id,
            client=client,
            model=resolved_model,
            question_hash=_hash_question(question),
            output_json=None,  # the cached row already has it; don't duplicate
            cache_hit=True,
        )
        # Surface the hit elapsed in the response too — UI can show
        # the actual ~3 ms instead of the original miss time.
        cached.elapsed_seconds = round(lookup_elapsed, 3)
        return cached

    run_id = uuid4().hex
    started_at = datetime.utcnow()
    trace, elapsed, tokens = await run_react(question, max_steps, mcp_servers=mcp_servers, model=model)
    output = synthesize(question, trace, elapsed)
    output.model = resolved_model
    finished_at = datetime.utcnow()

    # Stash the cache key + payload only when the result is hittable.
    # Errors, blanks, and `**Error:**`-prefixed answers leave these NULL
    # so they can never be returned by a future cache lookup.
    cacheable = default_strategy and _is_cacheable_output(output)
    await _persist(
        run_id=run_id,
        question=question,
        status=output.status,
        started_at=started_at,
        finished_at=finished_at,
        elapsed=elapsed,
        max_steps=max_steps,
        trace=trace,
        final_answer=output.answer_markdown,
        tool_invoked=output.tool_invoked,
        user_id=user_id,
        client=client,
        model=resolved_model,
        question_hash=_hash_question(question) if cacheable else None,
        output_json=output.model_dump_json() if cacheable else None,
        cache_hit=False,  # this is the original real-work run, not a hit
        tokens=tokens,
    )

    return output


def run_agent_sync(
    question: str,
    max_steps: int = KG_AGENT_MAX_STEPS,
    user_id: str | None = None,
    client: str | None = None,
    mcp_servers: list[tuple[str, str]] | None = None,
    model: str | None = None,
) -> KgOutput:
    """Sync wrapper for Flask. Each call spins up a fresh event loop —
    fine at our request rate (LLM dominates; loop setup is microseconds)."""
    return asyncio.run(run_agent(question, max_steps, user_id, client, mcp_servers, model))


async def _persist(
    *,
    run_id: str,
    question: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    elapsed: float,
    max_steps: int,
    trace: list[TraceStep],
    final_answer: str | None,
    tool_invoked: str | None,
    user_id: str | None,
    client: str | None,
    model: str,
    question_hash: str | None = None,
    output_json: str | None = None,
    cache_hit: bool = False,
    tokens: dict[str, int] | None = None,
) -> None:
    """Best-effort write to runs + run_steps. Logs and swallows errors.

    `question_hash` and `output_json` populate the read-through cache.
    Both are None for uncacheable runs (errors / blanks / `**Error:**`
    answers); the cache lookup filters those out via SQL.

    `cache_hit=True` marks an access-log row written when the cache
    served a request without running the loop. Such rows have empty
    trace and steps_used=0; analytics that want "real work only" filter
    on cache_hit=False.
    """
    try:
        # Imported lazily so import-time of kg_agent doesn't require a DB.
        from .persistence.db import get_session
        from .persistence.models import Run, RunStep

        # Persist token usage only when actually captured; NULL (not 0) means
        # "no LLM work / provider reported nothing" so analytics can tell the
        # difference. A run with an all-zero dict collapses to NULL too.
        total = (tokens or {}).get("total") or None

        async with get_session() as session:
            session.add(Run(
                id=run_id,
                question=question,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                final_answer=final_answer,
                elapsed_seconds=elapsed,
                steps_used=len(trace),
                max_steps=max_steps,
                model=model,
                tool_invoked=tool_invoked,
                user_id=user_id,
                client=client,
                question_hash=question_hash,
                output_json=output_json,
                cache_hit=cache_hit,
                prompt_tokens=(tokens or {}).get("prompt") if total else None,
                completion_tokens=(tokens or {}).get("completion") if total else None,
                total_tokens=total,
            ))
            for step in trace:
                session.add(RunStep(
                    run_id=run_id,
                    step=step.step,
                    thought=step.thought,
                    action=step.action,
                    action_input=step.action_input,
                    observation=step.observation,
                    final_answer=step.final_answer,
                    elapsed_ms=step.elapsed_ms,
                ))
    except Exception as exc:
        logger.warning("persistence failed (run_id=%s): %s", run_id, exc)


__all__ = [
    "run_agent",
    "run_agent_sync",
    "clear_cache",
    "clear_cache_sync",
    "KgOutput",
    "Rendering",
    "Graph",
    "TraceStep",
]
