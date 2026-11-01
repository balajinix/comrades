"""RQ5 driver: AI coding harnesses as data consumers under backend failure.

For each (harness, task, fault rate, seed) this:
  1. sets the fault proxy's per-run control (rate, seed, run_id) so the proxy
     injects faults at the mcp_dal tool boundary and tags every call with run_id;
  2. runs the coding harness headlessly, pointed at the proxy (not mcp_dal), with
     its native tool-use loop unmodified;
  3. records the harness's COMMUNICATION only -- tool calls, tokens, turns, latency,
     and a recover-vs-give-up read -- as a function of the injected fault rate.

Communication only: no safety/enforcement/blast-radius outcome is measured here
(that is the a related project paper's variable). Ground-truth tool-call counts come from the
proxy's JSONL log (mechanism-independent); tokens/turns come from the harness CLI.

Pilot harness: Claude Code (`claude -p ... --output-format stream-json`). Codex /
Gemini / IDE-harness slot into HARNESSES with their own stat parsers.

Run (proxy must already front mcp_dal, e.g. on :8212):
    python harness/run_harness.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
        --tasks harness/tasks.jsonl --out harness/results/rq5.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Import IDE-harness PTY adapter
try:
    from ide_pty_adapter import IdePTYRunner, ideh_cmd, ideh_parse
    IDEH_AVAILABLE = True
except ImportError:
    IDEH_AVAILABLE = False
    print("Warning: IDE-harness PTY adapter not available (pexpect missing)")

PROXY_URL = os.getenv("RQ5_PROXY_URL", "http://localhost:8212/mcp")
PROXY_LOG = Path(os.getenv("RQ5_PROXY_LOG", str(HERE / "proxy_calls_rq5.jsonl")))
CONTROL_TOOL = "__comrades_control__"

# Neutral framing: tell the harness the tools exist and to use them; say NOTHING
# about retrying or giving up, so its native behavior under failure is what we measure.
SYS_PROMPT = (
    "You have MCP tools (server name mcp_dal) that fetch enterprise observability, "
    "cost, ITSM, resilience, and infrastructure data. Use them to complete the task. "
    "Do not ask the user for confirmation; gather the data yourself and then answer."
)


def _mcp_call(url, tool, args):
    """Call one MCP tool over streamable-http (used to set the proxy control)."""
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _call():
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool, args or {})

    return anyio.run(_call)


def set_proxy_control(rate: float, seed: int, run_id: str) -> None:
    _mcp_call(PROXY_URL, CONTROL_TOOL, {"rate": rate, "seed": seed, "run_id": run_id})


def load_tasks(path: str):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# --- harness adapters: (build argv, parse stdout -> stats dict) --------------
def _claude_cmd(prompt: str, mcp_cfg: str, max_turns: int):
    # Scope tool permission to the read-only mcp_dal MCP server only; in headless
    # (-p) mode any tool NOT on this allowlist (Bash, file edits, etc.) is auto-denied.
    # This keeps the harness's data-fetch loop running without the blanket
    # --dangerously-skip-permissions bypass.
    return [
        "claude", "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--mcp-config", mcp_cfg,
        "--append-system-prompt", SYS_PROMPT,
        "--allowedTools", "mcp__mcp_dal",
        "--max-turns", str(max_turns),
    ]


def _claude_parse(stdout: str) -> dict:
    """Pull tokens/turns/latency from Claude's stream-json `result` event, and a
    tool-call cross-check from the assistant `tool_use` blocks."""
    out = {"cli_tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "num_turns": 0,
           "duration_s": 0.0, "cost_usd": 0.0, "is_error": None, "answer": ""}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        t = ev.get("type")
        if t == "assistant":
            for blk in (ev.get("message", {}) or {}).get("content", []) or []:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    out["cli_tool_calls"] += 1
        elif t == "result":
            u = ev.get("usage", {}) or {}
            out["tokens_in"] = (u.get("input_tokens", 0)
                                + u.get("cache_read_input_tokens", 0)
                                + u.get("cache_creation_input_tokens", 0))
            out["tokens_out"] = u.get("output_tokens", 0)
            out["num_turns"] = ev.get("num_turns", 0)
            out["duration_s"] = round((ev.get("duration_ms", 0) or 0) / 1000, 2)
            out["cost_usd"] = round(ev.get("total_cost_usd", 0) or 0, 4)
            out["is_error"] = ev.get("is_error")
            out["answer"] = (ev.get("result") or "")[:300]
    return out


def _codex_cmd(prompt: str, mcp_cfg: str, max_turns: int):
    # Codex reads MCP servers from ~/.codex (cached), so we force the proxy URL with a
    # per-run -c override. exec auto-cancels MCP calls under any sandbox/approval combo
    # except the bypass flag, so we use it here (read-only data tasks, local controlled
    # setup). The prompt is positional; prepend the neutral guidance.
    return [
        "codex", "exec", "--json", "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c", f"mcp_servers.mcp_dal.url={PROXY_URL}",
        f"{SYS_PROMPT}\n\nTask: {prompt}",
    ]


def _codex_parse(stdout: str) -> dict:
    """Pull comms stats from Codex's JSONL events: mcp_tool_call items + the
    turn.completed usage block."""
    out = {"cli_tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "num_turns": 0,
           "duration_s": 0.0, "cost_usd": 0.0, "is_error": None, "answer": ""}
    turn_ok = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        t = ev.get("type")
        item = ev.get("item", {}) or {}
        if t == "item.completed" and item.get("type") == "mcp_tool_call":
            out["cli_tool_calls"] += 1
        elif t == "item.completed" and item.get("type") == "agent_message":
            out["answer"] = (item.get("text") or "")[:300]
        elif t == "turn.completed":
            turn_ok = True
            out["num_turns"] += 1
            u = ev.get("usage") or item.get("usage") or {}
            out["tokens_in"] = (u.get("input_tokens", 0)
                                + u.get("cached_input_tokens", 0))
            out["tokens_out"] = u.get("output_tokens", 0)
        elif t == "turn.failed" or t == "error":
            out["is_error"] = True
    if out["is_error"] is None:
        out["is_error"] = not turn_ok
    return out


def _antigravity_cmd(prompt: str, mcp_cfg: str, max_turns: int):
    # We need to copy/link the config to the workspace-level `.gemini/settings.json`
    # so that gemini CLI picks it up.
    import shutil
    target_dir = HERE.parent / ".gemini"
    target_dir.mkdir(exist_ok=True)
    target_path = target_dir / "settings.json"
    
    # Read mcp_cfg
    with open(mcp_cfg) as f:
        cfg = json.load(f)
    
    # Append trailing slash to URL if it doesn't have one to prevent 307 redirect failure
    url = cfg["mcpServers"]["mcp_dal"]["url"]
    if not url.endswith("/"):
        cfg["mcpServers"]["mcp_dal"]["url"] = url + "/"
        
    # Append other necessary overrides to disable sub-agents
    cfg["experimental"] = {"enableAgents": False}
    cfg["agents"] = {
        "overrides": {
            "generalist": {"enabled": False},
            "codebase_investigator": {"enabled": False},
            "cli_help": {"enabled": False}
        }
    }
    cfg["model"] = {
        "maxSessionTurns": max_turns
    }
    cfg["tools"] = {
        "exclude": [
            "glob",
            "grep_search",
            "list_directory",
            "read_file",
            "run_shell_command",
            "write_file",
            "replace",
            "google_web_search",
            "write_todos",
            "web_fetch",
            "read_many_files",
            "get_internal_docs",
            "activate_skill",
            "ask_user",
            "update_topic",
            "tracker_create_task",
            "tracker_update_task",
            "tracker_get_task",
            "tracker_list_tasks",
            "tracker_add_dependency",
            "tracker_visualize",
            "complete_task",
            "enter_plan_mode",
            "exit_plan_mode",
            "invoke_agent",
            "take_snapshot",
            "read_mcp_resource",
            "list_mcp_resources",
        ]
    }
    
    with open(target_path, "w") as f:
        json.dump(cfg, f, indent=2)
        
    return [
        "gemini", "-p", prompt,
        "-o", "stream-json",
        "--allowed-mcp-server-names", "mcp_dal",
        "--skip-trust",
        "--approval-mode", "yolo"
    ]


def _antigravity_parse(stdout: str) -> dict:
    try:
        settings_path = HERE.parent / ".gemini" / "settings.json"
        if settings_path.exists():
            settings_path.unlink()
    except Exception:
        pass

    out = {"cli_tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "num_turns": 1,
           "duration_s": 0.0, "cost_usd": 0.0, "is_error": None, "answer": ""}
           
    answer_parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
            
        t = ev.get("type")
        if t == "tool_use":
            out["cli_tool_calls"] += 1
        elif t == "tool_result":
            out["num_turns"] += 1
        elif t == "message" and ev.get("role") == "assistant":
            content = ev.get("content") or ""
            if content:
                answer_parts.append(content)
        elif t == "result":
            out["is_error"] = (ev.get("status") != "success")
            stats = ev.get("stats") or {}
            if stats:
                out["tokens_in"] = stats.get("input_tokens", 0)
                out["tokens_out"] = stats.get("output_tokens", 0)
                out["duration_s"] = round(stats.get("duration_ms", 0) / 1000, 2)
                
    out["answer"] = "".join(answer_parts)[:300]
    if out["is_error"] is None:
        out["is_error"] = False
    return out


HARNESSES = {
    "claude": {"cmd": _claude_cmd, "parse": _claude_parse},
    "codex": {"cmd": _codex_cmd, "parse": _codex_parse},
    "antigravity": {"cmd": _antigravity_cmd, "parse": _antigravity_parse},
}

if IDEH_AVAILABLE:
    HARNESSES["ideh"] = {"cmd": ideh_cmd, "parse": ideh_parse, "pty": True}


def proxy_comms(run_id: str, gold_tools=None) -> dict:
    """Ground-truth communication for this run from the proxy log: total tool calls,
    injected faults, recover/give-up, and (if gold_tools given) on-task scoring.

    - calls: tool calls the harness made to the data layer (volume).
    - faulted: injected faults it hit.
    - recovered: it hit >=1 injected fault but still completed >=1 forwarded call
      after a fault (retried and got through) -> over-communication recovery.
    - gave_up: it hit >=1 injected fault and never completed a forwarded call after
      one -> under-communication give-up.
    - on_task: it called at least one of the task's annotated gold tools.
    - gold_obtained: a gold tool returned data (forwarded, not faulted) -> the task's
      data was actually retrieved (a rigorous "data obtained", per the annotation).
    """
    gold = set(gold_tools or [])
    calls = faulted = 0
    forwarded_after_fault = False
    seen_fault = False
    on_task = gold_obtained = False
    if PROXY_LOG.exists():
        with PROXY_LOG.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if r.get("run_id") != run_id:
                    continue
                calls += 1
                tool = r.get("tool")
                if tool in gold:
                    on_task = True
                    if not r.get("faulted"):
                        gold_obtained = True
                if r.get("faulted"):
                    faulted += 1
                    seen_fault = True
                elif seen_fault:
                    forwarded_after_fault = True
    recovered = int(faulted > 0 and forwarded_after_fault)
    gave_up = int(faulted > 0 and not forwarded_after_fault)
    return {"calls": calls, "faulted": faulted, "recovered": recovered,
            "gave_up": gave_up, "on_task": int(on_task),
            "gold_obtained": int(gold_obtained)}


def run_one(harness: str, task: dict, rate: float, seed: int, max_turns: int) -> dict:
    run_id = f"{harness}-{task['id']}-r{rate}-s{seed}"
    set_proxy_control(rate, seed, run_id)
    cfg = {"mcpServers": {"mcp_dal": {"type": "http", "url": PROXY_URL}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(cfg, fh)
        cfg_path = fh.name
    stats = None
    wall = 0.0
    try:
        argv = HARNESSES[harness]["cmd"](task["prompt"], cfg_path, max_turns)
        t0 = time.time()
        
        # Check if this is IDE-harness PTY mode
        if HARNESSES[harness].get("pty") and argv[0] == "__IDE_PTY__":
            # Use PTY runner for IDE-harness
            try:
                runner = IdePTYRunner(cfg_path, max_turns, timeout=600)
                stats = runner.run(task["prompt"])
                wall = stats["duration_s"]
            except Exception as e:
                wall = round(time.time() - t0, 2)
                print(f"    [ideh-pty-error] {run_id}: {e}", flush=True)
                stats = {"cli_tool_calls": 0, "num_turns": 0, "tokens_in": 0,
                        "tokens_out": 0, "cost_usd": 0.0, "duration_s": wall, "is_error": 1}
        else:
            # Standard subprocess mode for other harnesses
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
                wall = round(time.time() - t0, 2)
                stats = HARNESSES[harness]["parse"](proc.stdout)
            except subprocess.TimeoutExpired:
                # A single hung CLI invocation must not abort the whole sweep:
                # record an error row and let the remaining cells run.
                wall = round(time.time() - t0, 2)
                print(f"    [timeout] {run_id} exceeded 600s -> error row", flush=True)
    finally:
        os.unlink(cfg_path)
    comms = proxy_comms(run_id, task.get("gold_tools"))
    if stats is None:
        stats = {"cli_tool_calls": 0, "num_turns": 0, "tokens_in": 0,
                 "tokens_out": 0, "cost_usd": 0.0, "duration_s": wall, "is_error": 1}
    return {
        "harness": harness, "id": task["id"], "family": task.get("family"),
        "fault_rate": rate, "seed": seed,
        "tool_calls": comms["calls"], "faulted": comms["faulted"],
        "recovered": comms["recovered"], "gave_up": comms["gave_up"],
        "on_task": comms["on_task"], "gold_obtained": comms["gold_obtained"],
        "cli_tool_calls": stats["cli_tool_calls"], "num_turns": stats["num_turns"],
        "tokens_in": stats["tokens_in"], "tokens_out": stats["tokens_out"],
        "cost_usd": stats["cost_usd"], "duration_s": stats["duration_s"],
        "wall_s": wall, "is_error": stats["is_error"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    available_harnesses = list(HARNESSES.keys())
    ap.add_argument("--harness", default="claude", choices=available_harnesses)
    ap.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None, help="cap tasks (smoke)")
    ap.add_argument("--out", default=str(HERE / "results" / "rq5.csv"))
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in str(args.seeds).split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["harness", "id", "family", "fault_rate", "seed", "tool_calls", "faulted",
            "recovered", "gave_up", "on_task", "gold_obtained", "cli_tool_calls",
            "num_turns", "tokens_in", "tokens_out", "cost_usd", "duration_s",
            "wall_s", "is_error"]
    n = 0
    print(f"RQ5: harness={args.harness} tasks={len(tasks)} rates={rates} seeds={seeds} "
          f"-> {out}\n", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        f.flush()
        for rate in rates:
            for seed in seeds:
                for task in tasks:
                    row = run_one(args.harness, task, rate, seed, args.max_turns)
                    w.writerow(row)
                    f.flush()
                    n += 1
                    print(f"  [{n}] {task['id']} r={rate} s={seed}: "
                          f"calls={row['tool_calls']} faulted={row['faulted']} "
                          f"rec={row['recovered']} giveup={row['gave_up']} "
                          f"turns={row['num_turns']} tok_out={row['tokens_out']} "
                          f"{row['duration_s']}s", flush=True)
    print(f"\nWrote {n} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
