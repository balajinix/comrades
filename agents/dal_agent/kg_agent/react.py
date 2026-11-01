"""The ReAct loop. Keep this file tight; do not put I/O here that isn't part
of the loop's mechanics. Tool dispatch goes through `mcp_client`; persistence
is handled in `__init__.run_agent`."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import KG_AGENT_MAX_STEPS
from .llm import breaker, build_llm
from .mcp_client import call_tool, list_tools
from .models import TraceStep
from .observation import format_observation
from .prompts import SYSTEM_PROMPT, parse_action_input, parse_step


logger = logging.getLogger("kg_agent.react")


# Agent strategies. ``react`` is the original loop and stays the default, so every
# existing runner is unchanged. ``cot`` adds one chain-of-thought planning turn
# before the loop; ``reflect`` asks the model to critique a proposed Final Answer
# before it is accepted. Both keep the same tools, guards (loop detector, no-tool
# strike, short circuit) and productive-step budget, and their extra LLM turns are
# counted in the same token and message counters. Chosen per process by
# COMRADES_AGENT_STRATEGY so the A2A service and every runner pick it up.
STRATEGIES = ("react", "cot", "reflect")

COT_PLAN = (
    "Before you call any tool, think step by step. Write a short numbered plan: what the "
    "question asks for, which tools from the catalog you will call in which order and "
    "with which arguments, and how you will combine their results into the answer. Do "
    "not emit an Action or a Final Answer yet. Reply with the plan only."
)

COT_GO = (
    "Now carry out your plan. From here on follow the system prompt's format exactly: "
    "Thought, then Action and Action Input, or Thought and Final Answer."
)

REFLECT = (
    "Before this Final Answer is accepted, reflect on it. Check it against the question "
    "and your observations: does it answer what was asked, is every figure taken from an "
    "observation rather than assumed, does it say plainly which data could not be "
    "obtained, and would another tool call fill a gap? Reply with a Thought holding your "
    "critique, then either a Final Answer (the same one if it is right, a corrected one "
    "if not) or an Action and Action Input if you need more data."
)


def _strategy() -> tuple[str, int]:
    """(strategy, max reflection rounds) from the environment."""
    name = os.getenv("COMRADES_AGENT_STRATEGY", "react").strip().lower() or "react"
    if name not in STRATEGIES:
        raise ValueError(f"COMRADES_AGENT_STRATEGY={name!r}; expected one of {STRATEGIES}")
    return name, max(0, int(os.getenv("COMRADES_MAX_REFLECTIONS", "1")))


# Loop-detection normalisation. The LLM often paraphrases when retrying — same
# intent, different wording. Strip punctuation, lowercase, drop common
# stopwords, sort tokens. Two near-paraphrased inputs collapse to the same
# fingerprint; truly different inputs (different time windows, different app
# names) stay distinct because content tokens dominate.
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_WS_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "with", "by",
    "is", "are", "was", "were", "be", "or", "and",
    "me", "my", "our", "your", "us",
    "what", "which", "who", "where", "when", "why", "how",
    "show", "tell", "find", "give", "get", "list", "fetch", "display",
    "please", "kindly",
}


def _fingerprint(action_input: str | None) -> str:
    """Stable fingerprint for an Action Input string — same intent → same
    fingerprint. Used by loop detection so the LLM can't retry the same query
    with a paraphrased input and bypass the dedup check.
    """
    if not action_input:
        return ""
    s = action_input.lower()
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in _WS_RE.sub(" ", s).strip().split() if t and t not in _STOPWORDS]
    return " ".join(sorted(set(tokens)))


def _fingerprint_args(args: dict[str, Any]) -> str:
    """Fingerprint a structured argument dict. Canonical-JSON first (so key
    order doesn't matter), then the same token normalisation as `_fingerprint`
    (so NL-string values are paraphrase-tolerant while structured values —
    app names, dates, resource IDs — survive as distinct content tokens).
    """
    try:
        canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = str(args)
    return _fingerprint(canonical)


def _build_catalog(tools: list[dict[str, Any]]) -> str:
    """Render the federated tool catalog for the system prompt, grouped by
    server so the metadata/data/resilience split is visually reinforced. Each
    tool shows its description and compact arg schema."""
    by_server: dict[str, list[dict[str, Any]]] = {}
    for t in tools:
        by_server.setdefault(t.get("server", "other"), []).append(t)

    _LABELS = {
        "mcp_kg": "mcp_kg — graph & metadata (CEG, Federated Registry, ITOM, temporal-graph)",
        "mcp_dal": "mcp_dal — product & operational data (FinOps, APM, ITSM, Resilience, Infra)",
        "mcp_var": "mcp_var — resilience metrics (calculate_*)",
        "a2a_peers": "a2a_peers — sibling agents you can delegate to "
                     "(schema mapping, GraphQL generation)",
    }

    blocks: list[str] = []
    for server in list(_LABELS) + [s for s in by_server if s not in _LABELS]:
        entries = by_server.get(server)
        if not entries:
            continue
        blocks.append(f"## {_LABELS.get(server, server)}")
        for t in entries:
            blocks.append(f"### {t['name']}")
            if t.get("description"):
                blocks.append(t["description"])
            blocks.append("args:")
            blocks.append(t.get("compact_schema") or "  (no args — pass {})")
        blocks.append("")
    return "\n".join(blocks).strip()


def _usage_from_response(response) -> tuple[int, int, int]:
    """Best-effort (prompt, completion, total) token usage from an LLM response.

    ChatLiteLLM surfaces usage in either LangChain's standard ``usage_metadata``
    (input/output/total) or the raw ``response_metadata['token_usage']``
    (prompt/completion/total); providers that report nothing yield all-zeros.
    """
    um = getattr(response, "usage_metadata", None)
    if isinstance(um, dict) and um:
        p = int(um.get("input_tokens") or 0)
        c = int(um.get("output_tokens") or 0)
        return p, c, int(um.get("total_tokens") or (p + c))
    rm = getattr(response, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    if isinstance(tu, dict) and tu:
        p = int(tu.get("prompt_tokens") or 0)
        c = int(tu.get("completion_tokens") or 0)
        return p, c, int(tu.get("total_tokens") or (p + c))
    return 0, 0, 0


async def run_react(
    question: str,
    max_steps: int = KG_AGENT_MAX_STEPS,
    mcp_servers: list[tuple[str, str]] | None = None,
    model: str | None = None,
    strategy: str | None = None,
) -> tuple[list[TraceStep], float, dict[str, int]]:
    """Run the loop. Returns (trace, elapsed_seconds, tokens).

    `tokens` is a {prompt, completion, total} dict summed over every LLM call in
    the loop (all-zero on the early-exit paths that make no LLM call).

    `mcp_servers` overrides the configured server list (`config.MCP_SERVERS`) —
    a list of `(label, url)` pairs. Useful for tests; production callers leave
    it None.

    `model` overrides `PROVIDER_MODEL_NAME` for this one invocation — the
    per-user gating layer in the UI uses this to route premium users to a
    higher-tier model without touching the host env.

    `strategy` overrides COMRADES_AGENT_STRATEGY (``react``, ``cot`` or ``reflect``).

    Errors at every step are caught and turned into either a terminal trace
    entry or a recoverable observation. The loop never raises — the caller
    can rely on always getting a list back.
    """
    start = time.monotonic()
    env_strategy, max_reflections = _strategy()
    strategy = (strategy or env_strategy).lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    tokens = {"prompt": 0, "completion": 0, "total": 0}
    if not question.strip():
        return [], 0.0, tokens

    def _ms_since(t0: float) -> int:
        return int(round((time.monotonic() - t0) * 1000))

    # Discover tools dynamically across all configured MCP servers. If every
    # server is unreachable, fail fast with a single trace entry so the
    # synthesizer can render an actionable error.
    try:
        tools = await list_tools(servers=mcp_servers)
    except Exception as exc:
        logger.exception("list_tools failed")
        return (
            [TraceStep(
                step=1,
                thought=f"Could not connect to any MCP server: {type(exc).__name__}: {exc}",
                final_answer=f"**Error:** could not reach any MCP server ({type(exc).__name__}: {exc}).",
                elapsed_ms=_ms_since(start),
            )],
            time.monotonic() - start,
            tokens,
        )

    if not tools:
        return (
            [TraceStep(
                step=1,
                thought="MCP servers returned an empty (whitelisted) tool catalog.",
                final_answer="**Error:** no tools available from any MCP server.",
                elapsed_ms=_ms_since(start),
            )],
            time.monotonic() - start,
            tokens,
        )

    # Index for schema/argument-key lookup at call time.
    tool_index: dict[str, dict[str, Any]] = {t["name"]: t for t in tools}

    system = SystemMessage(content=SYSTEM_PROMPT.format(tool_catalog=_build_catalog(tools)))
    messages: list[Any] = [
        system,
        HumanMessage(content=f"Question: {question.strip()}"),
    ]

    trace: list[TraceStep] = []
    llm = build_llm(model=model)

    # Loop-detection state. Two checks fire before a tool is actually called:
    #   - exact-intent repeat: same (action, fingerprint(args)) seen before
    #   - same-action overuse: this action has already run twice in this loop
    # On a hit, we skip the call, inject a coaching observation, and let the
    # LLM re-plan. Coaching turns consume an iteration but NOT the productive
    # step budget (see the while-loop below), so a wasted retry doesn't starve
    # the real data-gathering work.
    seen_signatures: set[str] = set()
    action_call_counts: dict[str, int] = {}

    # Code-level enforcement of the prompt's MUST-emit-an-Action rule.
    # The LLM intermittently emits a Final Answer at step 1 with no tool
    # call (the failure mode is fabricated answers from training data).
    # Reject the first 2 such attempts with a coaching observation; on the
    # 3rd, accept what the LLM gives so the synthesizer can render a
    # meaningful response rather than burning all max_steps.
    no_tool_strikes = 0
    NO_TOOL_STRIKE_LIMIT = 2

    # Code-level enforcement of the JSON Action-Input contract. When the LLM
    # emits malformed JSON for a structured tool (no single-string fallback),
    # coach it with that tool's compact schema and let it retry. After the
    # limit, terminate with a clear error rather than calling with bad args.
    json_strikes = 0
    JSON_STRIKE_LIMIT = 2

    # Step budgeting. `productive` counts only real tool calls and is bounded
    # by max_steps. Coaching/rejection turns (no-tool strike, JSON strike,
    # loop-detection) advance the iteration but DON'T spend the productive
    # budget — so a hallucination-reject or an enum-error retry never starves
    # data gathering. `step` is the monotonic trace index. The strike limits
    # already bound coaching; HARD_ITER_CAP is a final runaway backstop.
    productive = 0
    step = 0
    HARD_ITER_CAP = max_steps + 5

    # Chain of thought: one planning turn before the loop. The plan request rides
    # on the question turn so no provider sees two user turns in a row. The plan is
    # a trace step (counted as a message) and its tokens join the same counters.
    if strategy == "cot":
        step += 1
        HARD_ITER_CAP += 1
        step_t0 = time.monotonic()
        messages[1] = HumanMessage(content=f"Question: {question.strip()}\n\n{COT_PLAN}")
        try:
            response = await llm.ainvoke(messages)
            breaker.record_success()
            p, c, t = _usage_from_response(response)
            tokens["prompt"] += p
            tokens["completion"] += c
            tokens["total"] += t
        except Exception as exc:
            breaker.record_failure(exc)
            logger.exception("LLM planning call failed")
            trace.append(TraceStep(
                step=step,
                thought=f"LLM call failed: {type(exc).__name__}: {exc}",
                final_answer=f"**Error:** LLM call failed while planning: {exc}",
                elapsed_ms=_ms_since(step_t0),
            ))
            return trace, time.monotonic() - start, tokens
        plan = response.content if hasattr(response, "content") else str(response)
        trace.append(TraceStep(step=step, thought=f"Plan: {plan}",
                               elapsed_ms=_ms_since(step_t0)))
        messages.append(AIMessage(content=plan))
        messages.append(HumanMessage(content=COT_GO))

    # Self-reflection: a proposed Final Answer is held back for critique up to
    # `max_reflections` times. A reflection that reopens the search goes through the
    # same loop guard (seen signatures persist), so identical calls stay blocked. If
    # the run then ends in an error, the held answer is kept, so reflecting never
    # leaves the agent worse off than accepting its first answer.
    reflections = 0
    pending_final: str | None = None
    if strategy == "reflect":
        HARD_ITER_CAP += max_reflections

    while productive < max_steps and step < HARD_ITER_CAP:
        step += 1
        step_t0 = time.monotonic()

        # 0. Circuit breaker: if the LLM provider has been failing
        # repeatedly, fail fast with a clear message instead of waiting
        # through another retry-and-backoff cycle.
        if breaker.is_open():
            remaining = breaker.cooldown_remaining()
            trace.append(TraceStep(
                step=step,
                thought=(
                    f"Circuit breaker open after {breaker.consec_failures} "
                    f"consecutive LLM disconnects; skipping the call."
                ),
                final_answer=(
                    f"**Error:** the LLM provider is temporarily unavailable "
                    f"(circuit breaker open, cooldown ~{remaining}s). This is "
                    f"usually a provider-side instability — try again shortly."
                ),
                elapsed_ms=_ms_since(step_t0),
            ))
            break

        # 1. Reason.
        try:
            response = await llm.ainvoke(messages)
            breaker.record_success()
            p, c, t = _usage_from_response(response)
            tokens["prompt"] += p
            tokens["completion"] += c
            tokens["total"] += t
        except Exception as exc:
            breaker.record_failure(exc)
            logger.exception("LLM call failed at step %d", step)
            trace.append(TraceStep(
                step=step,
                thought=f"LLM call failed: {type(exc).__name__}: {exc}",
                final_answer=f"**Error:** LLM call failed at step {step}: {exc}",
                elapsed_ms=_ms_since(step_t0),
            ))
            break

        text = response.content if hasattr(response, "content") else str(response)
        thought, action, action_input, final = parse_step(text)

        # 2. Terminal: final answer.
        if final is not None:
            # Enforce "must call at least one tool first". If we've never
            # successfully run a tool, this Final Answer is fabricated.
            no_tool_call_yet = not any(s.action and s.tool_result for s in trace)

            if no_tool_call_yet and no_tool_strikes < NO_TOOL_STRIKE_LIMIT:
                no_tool_strikes += 1
                coaching = (
                    "REJECTED: You emitted a Final Answer without calling any "
                    "tool. The constraints REQUIRE at least one Action before "
                    "any Final Answer. Restart this turn with a Thought + "
                    "Action + Action Input. Pick the tool whose description "
                    "best matches the question."
                )
                trace.append(TraceStep(
                    step=step,
                    thought=thought,
                    observation=coaching,
                    elapsed_ms=_ms_since(step_t0),
                ))
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=f"Observation: {coaching}"))
                continue

            if no_tool_call_yet:
                # Strikes exhausted; the LLM ignored every coaching message.
                # Don't pass the hallucination through to the user — return
                # an honest error instead.
                error = (
                    f"**Error:** the agent emitted a Final Answer without "
                    f"calling any tool, despite {NO_TOOL_STRIKE_LIMIT} explicit "
                    f"re-prompts. The proposed answer is a hallucination — "
                    f"synthesised from the model's training rather than current "
                    f"data. Try rephrasing the question to make the intended "
                    f"data source clearer (e.g. \"Use APM to show...\" or "
                    f"\"In ITSM, list...\"). Original (rejected) answer: "
                    f"{final[:200]}"
                )
                trace.append(TraceStep(
                    step=step,
                    thought=thought,
                    final_answer=error,
                    elapsed_ms=_ms_since(step_t0),
                ))
                break

            if strategy == "reflect" and reflections < max_reflections:
                reflections += 1
                pending_final = final
                trace.append(TraceStep(
                    step=step,
                    thought=thought,
                    observation=f"REFLECT ({reflections}): proposed answer held for critique.",
                    elapsed_ms=_ms_since(step_t0),
                ))
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=REFLECT))
                continue

            trace.append(TraceStep(
                step=step,
                thought=thought,
                final_answer=final,
                elapsed_ms=_ms_since(step_t0),
            ))
            break

        # 3. Parse failure (no recognisable Action and no Final Answer).
        if not action or action not in tool_index:
            # Salvage: if we ALREADY have a successful tool observation, the
            # LLM has the data — it just rambled instead of emitting a clean
            # `Final Answer:` line (common with query_infra_nl2iql, where
            # the model describes the returned IQL inside a Thought). Treat
            # that Thought as the answer rather than erroring out. Only fall
            # back to a hard error when no tool ever succeeded.
            had_data = any(
                s.tool_result is not None and (s.tool_result or {}).get("status") != "error"
                for s in trace
            )
            if had_data and thought:
                # After a reflection, a bare critique is not an answer; keep the
                # answer that was held for critique instead.
                trace.append(TraceStep(
                    step=step,
                    thought=thought,
                    final_answer=pending_final if pending_final is not None else thought,
                    elapsed_ms=_ms_since(step_t0),
                ))
                break
            trace.append(TraceStep(
                step=step,
                thought=thought,
                final_answer=(
                    f"**Error:** could not parse a valid Action at step {step}. "
                    f"LLM raw output (truncated): `{text[:300]}`"
                ),
                elapsed_ms=_ms_since(step_t0),
            ))
            break

        entry = tool_index[action]

        # 3b. Build structured args from the raw Action Input. Empty input for
        # a single-string tool falls back to the original question (preserves
        # the old single-NL-string behaviour); structured tools require JSON.
        raw_input = action_input
        if not raw_input and entry.get("single_string_key"):
            raw_input = question
        args, arg_err = parse_action_input(raw_input, entry)

        if arg_err is not None:
            # Malformed JSON for a structured tool. Coach with this tool's
            # schema and retry; terminate cleanly after the strike limit.
            if json_strikes < JSON_STRIKE_LIMIT:
                json_strikes += 1
                coaching = f"REJECTED (invalid Action Input). {arg_err}"
                trace.append(TraceStep(
                    step=step,
                    thought=thought,
                    action=action,
                    action_input=raw_input,
                    observation=coaching,
                    elapsed_ms=_ms_since(step_t0),
                ))
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=f"Observation: {coaching}"))
                continue
            trace.append(TraceStep(
                step=step,
                thought=thought,
                action=action,
                action_input=raw_input,
                final_answer=(
                    f"**Error:** the agent could not produce valid JSON arguments "
                    f"for `{action}` after {JSON_STRIKE_LIMIT} attempts. {arg_err}"
                ),
                elapsed_ms=_ms_since(step_t0),
            ))
            break

        args = args or {}
        args_repr = json.dumps(args, default=str)

        # 4. Loop detection — block before spending a tool call on a repeat.
        #   a) Exact intent repeat (same action + arg fingerprint).
        #   b) Same-action overuse: third call to the same action regardless
        #      of args — strong signal the agent is stuck on one tool.
        signature = f"{action}::{_fingerprint_args(args)}"
        prev_count = action_call_counts.get(action, 0)

        if signature in seen_signatures:
            coaching = (
                f"You have already called `{action}` with these same arguments in "
                f"this run. Calling it again returns the same data. Choose a "
                f"DIFFERENT tool or emit a one-sentence Final Answer with what "
                f"you have."
            )
            trace.append(TraceStep(
                step=step,
                thought=thought,
                action=action,
                action_input=args_repr,
                observation=coaching,
                elapsed_ms=_ms_since(step_t0),
            ))
            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(content=f"Observation: {coaching}"))
            continue

        if prev_count >= 2:
            # Short-circuit when every prior call to this action errored. The
            # LLM has nothing useful to synthesise from; if we coach + loop,
            # the LLM burns ~30s producing a long-winded apology. Construct
            # the giveup ourselves and exit.
            prior_real = [
                s for s in trace
                if s.action == action and s.tool_result is not None
            ]
            prior_errors = [
                s for s in prior_real
                if (s.tool_result or {}).get("status") == "error"
            ]
            if prior_real and len(prior_errors) == len(prior_real):
                last_msg = (prior_errors[-1].tool_result or {}).get("message") or \
                           (prior_errors[-1].observation or "(no observation)")
                giveup = (
                    f"**Error:** the agent called `{action}` {prev_count} times; "
                    f"every call returned an error from the tool. Most recent: "
                    f"`{last_msg[:300]}`. This is typically an upstream/backend "
                    f"issue rather than a query issue — check the tool's underlying "
                    f"service or rephrase the question to use a different data source."
                )
                trace.append(TraceStep(
                    step=step,
                    thought=(
                        f"Short-circuit: {prev_count} prior calls to `{action}` "
                        f"all returned errors; skipping further LLM synthesis."
                    ),
                    final_answer=giveup,
                    elapsed_ms=_ms_since(step_t0),
                ))
                break

            coaching = (
                f"You have already called `{action}` twice in this run. Switch "
                f"to a different tool or emit a one-sentence Final Answer — three "
                f"calls to the same action is the soft ceiling per the system prompt."
            )
            trace.append(TraceStep(
                step=step,
                thought=thought,
                action=action,
                action_input=args_repr,
                observation=coaching,
                elapsed_ms=_ms_since(step_t0),
            ))
            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(content=f"Observation: {coaching}"))
            continue

        # 5. Act. Each tool entry carries the URL of the server that
        # advertised it — use that for dispatch so federated calls land on
        # the right server regardless of catalog order. This is the only path
        # that spends the productive-step budget.
        productive += 1
        seen_signatures.add(signature)
        action_call_counts[action] = prev_count + 1

        target_url = entry["url"]
        try:
            tool_result = await call_tool(action, args, target_url)
        except Exception as exc:
            logger.exception("call_tool(%s @ %s) failed at step %d", action, target_url, step)
            tool_result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        observation = format_observation(tool_result)
        trace.append(TraceStep(
            step=step,
            thought=thought,
            action=action,
            action_input=args_repr,
            observation=observation,
            tool_result=tool_result,
            elapsed_ms=_ms_since(step_t0),
        ))

        # 6. Loop: feed observation back as the next user turn.
        messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=f"Observation: {observation}"))

    if pending_final is not None:
        last = trace[-1] if trace else None
        ended_badly = (last is None or not last.final_answer
                       or last.final_answer.startswith("**Error:**"))
        if ended_badly:
            trace.append(TraceStep(
                step=step + 1,
                thought="The run after reflection ended without a usable answer; "
                        "keeping the answer proposed before reflection.",
                final_answer=pending_final,
                elapsed_ms=0,
            ))

    return trace, time.monotonic() - start, tokens
