# RQ5 harness runbook — how we run coding agents as data consumers

This is the recipe for measuring an AI coding harness (Claude Code, Codex, and —
your task — Antigravity) as a **data-consuming agent** under backend failure. It is
**communication-only**: we measure how a harness *talks to the data layer* as faults
rise, not whether it makes a correct code change (that enforcement/blast-radius angle
belongs to a related project, and is deliberately out of scope here).

## 1. The idea in one picture

```
  harness (headless)  ──MCP──>  fault-injecting proxy  ──>  mcp_dal (data layer)
   claude / codex /              harness/fault_proxy.py        ~105 tools over
   antigravity                   port 8212                     cost/APM/ITSM/
                                  - injects faults (seeded)     resilience/infra
                                  - logs EVERY call (truth)
                                  - REPLAY=1: returns canned
                                    success instead of forwarding
```

Every harness points at the **same proxy**, so faults are injected at one boundary and
every tool call is counted identically regardless of which harness made it.

## 2. Components / scripts

| File | Role |
|---|---|
| `harness/fault_proxy.py` | The measurement boundary. A streamable-HTTP MCP server (port 8212) that mirrors `mcp_dal`'s tools, injects faults per call from a seeded RNG, and appends every call to `harness/proxy_calls_rq5.jsonl`. With `FAULT_PROXY_REPLAY=1` it returns a synthetic `ok` on non-faulted calls (clean, VPN-free rate-0 baseline). |
| `harness/tasks.jsonl` | 15 tasks, one JSON per line: `{id, family, prompt, gold_tools}`. `prompt` is a data-fetch ask; `gold_tools` are the `mcp_dal` tools that correctly serve it (used for on-task scoring). |
| `harness/run_harness.py` | The driver. For each (harness × task × rate × seed): sets the proxy's fault control, invokes the harness CLI headlessly, parses its output, joins with the proxy log, writes a CSV row. Harness-specific logic lives in the `HARNESSES` dict. |
| `harness/check_gold_tools.py` | Validates `gold_tools` by ranking every `mcp_dal` tool's description against each task prompt (sentence-transformer similarity). |
| `harness/aggregate_rq5.py` | Reads the result CSV(s), prints per-(harness, fault-rate) LaTeX rows for the paper. |

## 3. Inputs and outputs

**Input to a run:**
- a **task prompt** (from `tasks.jsonl`),
- a neutral **system prompt** (says only: "you have `mcp_dal` MCP tools, use them; don't
  ask for confirmation" — deliberately silent on retry vs. give-up so we measure native
  behavior),
- an **MCP server config** that points the harness at the proxy (`http://localhost:8212/mcp`),
- the proxy's **fault control** (rate, seed, run_id), set right before the run.

**Output per run (one CSV row in `harness/results/rq5_<harness>.csv`):**
`harness, id, family, fault_rate, seed, tool_calls, faulted, recovered, gave_up,
on_task, gold_obtained, cli_tool_calls, num_turns, tokens_in, tokens_out, cost_usd,
duration_s, wall_s, is_error`.

- **`tool_calls`, `faulted`, `recovered`, `gave_up`, `on_task`, `gold_obtained`** come
  from the **proxy log** (ground truth, harness-independent), filtered by `run_id`.
- **`num_turns`, `tokens_*`, `cost_usd`, `cli_tool_calls`** come from parsing the
  **harness's own structured output**.

## 4. The proxy control protocol (how faults are set per run)

The proxy exposes a hidden tool `__comrades_control__`. Before each run, the driver
calls it with `{rate, seed, run_id}`; the proxy reseeds its RNG to `"{seed}:{run_id}"`
and tags every subsequent logged call with that `run_id`. **Runs are sequential** — the
proxy holds one global fault state, so only one harness runs at a time.

```python
# set the proxy's fault control for a run (streamable-http MCP client)
session.call_tool("__comrades_control__", {"rate": 0.4, "seed": 1, "run_id": "codex-itsm-01-r0.4-s1"})
```

## 5. How each harness is invoked

The driver builds an argv and parses stdout per harness (`HARNESSES` dict in
`run_harness.py`). The two working ones:

**Claude Code** — surfaces MCP tools in headless mode; scoped permission via allow-list:
```
claude -p "<prompt>" --output-format stream-json --verbose \
  --mcp-config <tmp.json>            # {"mcpServers":{"mcp_dal":{"type":"http","url":"http://localhost:8212/mcp"}}}
  --append-system-prompt "<SYS_PROMPT>" \
  --allowedTools mcp__mcp_dal        # auto-allows only mcp_dal tools; everything else auto-denied
  --max-turns 14
```
Parse the final `type:"result"` event for `usage` (tokens) and `num_turns`.

**Codex** — `exec` is headless; needs the bypass flag to actually execute MCP calls,
and a `-c` override because it caches MCP servers separately from `config.toml`:
```
codex exec --json --skip-git-repo-check \
  --dangerously-bypass-approvals-and-sandbox \
  -c mcp_servers.mcp_dal.url=http://localhost:8212/mcp \
  "<SYS_PROMPT>\n\nTask: <prompt>"
```
Parse JSONL events: `item.completed` with `item.type=="mcp_tool_call"` (tool calls),
`turn.completed.usage` (tokens).

**Harnesses that did NOT fit (documented limitations):**
- **IDE harness 1.0.4/1.0.5:** non-interactive mode uses a fixed built-in toolset and
  does **not** inject MCP tools at all (by design). Connects but never sees `mcp_dal`.
- **Gemini CLI 0.46:** forces a built-in multi-agent orchestration (`invoke_agent`/
  `update_topic`); the sub-agent doesn't inherit the MCP permission, so the data call is
  blocked, and the meta-tooling pollutes the communication measurement.

## 6. How we evaluate (the DV, communication-only)

Everything is reported **as a function of the injected fault rate** (0.0–0.6):
- **Communication volume:** `tool_calls` to the data layer, `num_turns`, `tokens_out`.
- **Behavior under failure (on runs that hit a fault):**
  `recovered` = retried after a fault and still completed a forwarded call →
  *over-communication recovery*; `gave_up` = hit a fault and never completed after →
  *under-communication give-up*.
- **On-task grounding (from `gold_tools`):** `on_task` = called a gold tool;
  `gold_obtained` = a gold tool returned data un-faulted.

The headline: does the harness **retry (communication amplifies)** or **fail-fast (flat
comms, rising give-up)** as faults rise — i.e., does the RQ1–RQ2 frontier hold for real
coding tools. (Claude pilot: retry/over-communicate — tool calls rose 3.4 → 7.9 across
rates 0 → 0.6.)

## 7. Run it (end to end)

```bash
cd evaluation/comrades

# 1. Start the proxy in replay mode, fronting mcp_dal, on :8212 (one time):
FAULT_PROXY_TARGET=http://localhost:8201/mcp FAULT_PROXY_PORT=8212 \
  FAULT_PROXY_HIDE="" FAULT_PROXY_REPLAY=1 \
  FAULT_PROXY_LOG="$PWD/harness/proxy_calls_rq5.jsonl" \
  ../../agents/dal_agent/.venv/bin/python harness/fault_proxy.py &

# 2. (optional) sanity-check the gold mapping vs tool descriptions:
../../agents/schema_mapping_agent/.venv/bin/python harness/check_gold_tools.py --top 5

# 3. Run a harness over the full sweep (15 tasks x 4 rates x seeds):
../../agents/dal_agent/.venv/bin/python harness/run_harness.py \
  --harness claude --rates 0.0,0.2,0.4,0.6 --seeds 1 \
  --out harness/results/rq5_claude.csv

# 4. Aggregate:
python3 harness/aggregate_rq5.py harness/results/rq5_claude.csv
```

## 8. Adding Antigravity (what to check, in order)

A harness works for us only if it passes these four gates. Test each with ONE task
before wiring the adapter — set the proxy control, run the CLI, then check the proxy log
for a logged call under your `run_id`:

1. **Headless / non-interactive:** can it take a single prompt and run to completion
   without a TTY? (Claude `-p`, Codex `exec`.) ← IDE-harness/Gemini both technically pass this.
2. **Loads MCP tools in headless mode:** does it actually expose `mcp_dal`'s tools to the
   model? (IDE-harness fails here.) Configure its MCP server to point at
   `http://localhost:8212/mcp` (streamable HTTP). Ask it to "list the mcp_dal tools" and
   confirm it can.
3. **Single-agent tool use (no forced orchestration):** it should call `mcp_dal` tools
   directly, not spawn sub-agents that lose the permission. (Gemini fails here.)
4. **Routes through the proxy:** after a run, `grep '"run_id": "<your-id>"'
   harness/proxy_calls_rq5.jsonl` must show the calls. (If the tool ran but nothing is
   logged, the harness is hitting the real `mcp_dal` directly, not the proxy — fix its
   MCP URL / clear any cached config, as we had to for Codex.)

Then add an adapter to `HARNESSES` in `run_harness.py`:
```python
def _antigravity_cmd(prompt, mcp_cfg, max_turns):
    return ["antigravity", "<headless-flag>", ...,
            "<point MCP at http://localhost:8212/mcp>",
            "<auto-approve/scope the mcp_dal tools>",
            f"{SYS_PROMPT}\n\nTask: {prompt}"]

def _antigravity_parse(stdout):
    # return {cli_tool_calls, tokens_in, tokens_out, num_turns, duration_s,
    #         cost_usd, is_error, answer}; rely on the proxy log for the
    #         authoritative tool_calls/faulted/recover/giveup anyway.
    ...

HARNESSES["antigravity"] = {"cmd": _antigravity_cmd, "parse": _antigravity_parse}
```
Even if you can't parse Antigravity's tokens/turns cleanly, the proxy log still gives
the core comms metrics — so a minimal parser is enough to get it into the panel.

## 9. Gotchas we already hit (save yourself the loop)
- **Auth:** harnesses need to be logged in / keyed first (Claude Max login; Gemini reads
  `GEMINI_API_KEY` — we export it from `kg/change-impact-analysis/.env`'s `GOOGLE_API_KEY`;
  Codex uses ChatGPT login). Do the interactive auth once before headless runs.
- **Permission flags:** headless tool use needs the harness's "allow tools" switch
  (Claude `--allowedTools`, Codex `--dangerously-bypass-approvals-and-sandbox`, Gemini
  `--yolo`/`--approval-mode`). Prefer the *scoped* form when one exists.
- **Cached MCP config:** Codex ignored `config.toml` (used a cached server URL); the
  per-run `-c mcp_servers.mcp_dal.url=...` override fixed it. Watch for the same.
- **Dead backend:** the local `mcp_dal` (:8201) has no live backends, so always run the
  proxy with `FAULT_PROXY_REPLAY=1` for a clean rate-0 baseline.
</content>
