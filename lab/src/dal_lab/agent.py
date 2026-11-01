"""A neutral, minimal ReAct agent that answers questions through one facade.

The same loop runs against every facade, so differences in cost and accuracy come
from the data-access interface, not from the agent. The protocol is plain JSON
in text (no provider-specific function calling), so any chat model can run it:

- call a tool:  ``{"tool": "<name>", "args": {...}}``
- answer:       ``{"answer": [[v1, v2, ...], ...]}``   (one inner list per row)
- by reference: ``{"answer_handle": "<handle>"}``      (handles facade only)

Every tool result goes back to the model verbatim (after the facade's own
observation transform, such as compaction). Nothing else trims it, so context
growth is real. The run stops on an answer, after ``max_steps`` tool calls, when a
prompt exceeds ``context_limit`` tokens, or when cumulative input passes
``input_budget``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLMGateway
from .meter import Meter, to_json

SYSTEM = """You are a data agent. Answer the user's question using only the tools below.
Reply with exactly ONE JSON object per message and nothing else:
- to call a tool: {{"tool": "<tool name>", "args": {{...}}}}
- to give the final answer: {{"answer": [[...], [...]]}} with one inner list per result row,
  values in the column order the question asks for. Use [] if there are no rows.
{handle_rule}
{extra}
Tools (name, description, JSON schema of args):
{tools}"""

HANDLE_RULE = ('- if the result is large and you hold a result handle that contains exactly '
               'the answer rows, you may answer {"answer_handle": "<handle>"} instead.')

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentRun:
    """Outcome and cost of one agent run."""

    status: str  # answered | max_steps | overflow | budget | parse_error | llm_error
    answer: list | None = None
    answer_handle: str | None = None
    steps: int = 0
    tool_errors: int = 0
    llm: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)


def _parse(text: str) -> dict | None:
    # The first complete JSON object is the action, as in other ReAct loops: models
    # sometimes append a second action or an invented answer in the same reply.
    text = text or ""
    start = text.find("{")
    if start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    m = _JSON.search(text)
    if not m:
        return None
    # Models (notably Llama) sometimes drop trailing braces on nested JSON; repairing
    # that keeps the study about data access rather than about brace counting.
    for pad in range(4):
        try:
            obj = json.loads(m.group(0) + "}" * pad)
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, dict) else None
    return None


def _normalise(action: dict, tools: dict) -> tuple[Any, Any]:
    """Canonical (tool name, args) from the action variants models emit.

    One protocol for every model: Maverick nests ``{"tool": {"name", "args"}}``,
    Granite sends ``args`` as a JSON string, and gpt-oss prefixes tool names with a
    namespace (``repo_browser.run_sql``). Unknown names stay unknown.
    """
    name, args = action.get("tool"), action.get("args", action.get("arguments"))
    if isinstance(name, dict):
        args = name.get("args", name.get("arguments", args))
        name = name.get("name")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    if isinstance(name, str) and name not in tools and "." in name:
        short = name.rsplit(".", 1)[1]
        if short in tools:
            name = short
    return name, args if args is not None else {}


def run_agent(question: str, columns: list[str], descriptors: list[dict],
              tools: dict[str, Callable[..., Any]], meter: Meter, llm: LLMGateway,
              allow_handles: bool = False, max_steps: int = 12,
              context_limit: int = 120_000, input_budget: int = 600_000,
              extra_instructions: str = "",
              on_turn: Callable[[], str | None] | None = None) -> AgentRun:
    """Run the ReAct loop for one question.

    Args:
        question: The information need.
        columns: Expected answer columns, stated to the model so grading is unambiguous.
        descriptors: MCP-style tool descriptors of the facade.
        tools: Metered tool callables.
        meter: The data gateway (already wrapping ``tools``).
        llm: The LLM gateway.
        allow_handles: Whether answering by handle is permitted.
        max_steps: Tool-call cap.
        context_limit: Stop as overflow when one prompt exceeds this many tokens.
        input_budget: Stop when cumulative prompt tokens exceed this.
        extra_instructions: Appended to the system prompt (e.g. a retry playbook).
        on_turn: Called after every model turn; a returned string stops the run
            with that status (e.g. a virtual-time deadline).

    Returns:
        The :class:`AgentRun`.
    """
    llm.reset()
    system = SYSTEM.format(handle_rule=HANDLE_RULE if allow_handles else "",
                           extra=extra_instructions,
                           tools=json.dumps(descriptors, indent=None))
    user = f"{question}\nAnswer columns, in order: {', '.join(columns)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    run = AgentRun(status="max_steps")
    parse_failures = 0
    while True:
        try:
            text = llm.chat(messages)
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            msg = str(exc).lower()
            run.status = "overflow" if ("context" in msg or "token" in msg) else "llm_error"
            run.transcript.append({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            break
        last = llm.turns[-1]
        run.transcript.append({"model": text[:2000]})
        if last.prompt_tokens > context_limit:
            run.status = "overflow"
            break
        if sum(t.prompt_tokens for t in llm.turns) > input_budget:
            run.status = "budget"
            break
        if on_turn is not None:
            stop = on_turn()
            if stop:
                run.status = stop
                break
        action = _parse(text)
        if action is None:
            parse_failures += 1
            if parse_failures > 2:
                run.status = "parse_error"
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Reply with exactly one JSON object."}]
            continue
        if "answer" in action:
            run.status, run.answer = "answered", action["answer"]
            break
        if "answer_handle" in action and allow_handles:
            run.status, run.answer_handle = "answered", str(action["answer_handle"])
            break
        if run.steps >= max_steps:
            run.status = "max_steps"
            break
        name, args = _normalise(action, tools)
        run.steps += 1
        if not isinstance(name, str) or name not in tools or not isinstance(args, dict):
            obs = {"error": f"unknown tool {name!r}"}
            run.tool_errors += 1
        else:
            try:
                obs = tools[name](**args)
            except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
                obs = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                run.tool_errors += 1
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Observation: " + to_json(obs)}]
    run.llm = llm.totals()
    run.data = meter.totals()
    return run
