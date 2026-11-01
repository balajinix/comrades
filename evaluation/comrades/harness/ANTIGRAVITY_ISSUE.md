# Gemini CLI hangs after MCP tool calls in headless `stream-json` mode

## Summary
When driving the **Gemini CLI** non-interactively (`-p ... -o stream-json`) against a
remote MCP server, the CLI **makes its tool calls successfully but then never
terminates** — it does not emit a final terminal event and the process does not exit.
We wrap each run in a 600 s hard timeout; the majority of runs hit that wall and are
killed, producing an unusable (empty) result. This makes Gemini unreliable as a
headless harness in our benchmark, while Claude Code and Codex run the identical
workload to completion.

## Environment
- **Gemini CLI:** v0.49.0
- **Node:** v23.11.0 ; **OS:** macOS (darwin 25.3)
- **MCP server:** a local streamable-HTTP proxy (`mcp_dal`, ~105 tools) on
  `http://localhost:8212/mcp`, which forwards to a real backend. The proxy logs every
  inbound tool call, so we have ground truth on what the CLI actually sent.

## How we invoke it
Per-run we write a workspace `./.gemini/settings.json` containing:
```jsonc
{
  "mcpServers": { "mcp_dal": { "url": "http://localhost:8212/mcp/" } }, // trailing slash to avoid a 307
  "experimental": { "enableAgents": false },
  "agents": { "overrides": {
    "generalist": { "enabled": false },
    "codebase_investigator": { "enabled": false },
    "cli_help": { "enabled": false }
  }}
}
```
and run:
```bash
gemini -p "<task prompt>" \
       -o stream-json \
       --allowed-mcp-server-names mcp_dal \
       --skip-trust \
       --approval-mode yolo
```
We then parse the emitted JSONL stream, counting `tool_use` / `tool_result` events and
reading the final `{"type":"result", ...}` event for status + token stats.

## Symptom / evidence (66-task sweep, 71 runs observed so far)
- **39 / 71 runs (55%) hit the 600 s timeout** and were killed.
- **36 / 71 runs are empty** — `num_turns = 0`, `tokens_out = 0`: we never received a
  terminal `result` event to parse, even though the process was alive the whole 600 s.
- **61 / 71 runs reached the MCP server and issued ≥1 tool call** (confirmed in the
  proxy log). One run issued **10 tool calls** and then hung for the full 600 s.
- So this is **not** an MCP-connectivity or auth failure, and **not** a "no tools
  loaded" problem — the model connects, lists tools, and calls them. It **hangs after
  tool use**, before/instead of producing the final assistant turn + `result` event.

Representative killed run id: `antigravity-apm-01-r0.0-s1` (single simple task, no
injected faults, still timed out).

## What we've ruled out
- **MCP reachability** — proxy logs show the tool calls landing. ✅
- **Sub-agent fan-out stealing the MCP permission** — we explicitly disable agents
  (`experimental.enableAgents=false` + per-agent overrides). The calls come from the
  top-level model. ✅
- **307 redirect on the MCP URL** — handled by appending the trailing slash. ✅
- **Approval prompts blocking headless** — we pass `--approval-mode yolo --skip-trust`. ✅

## Open questions for the Gemini CLI team
1. In headless `-p ... -o stream-json` mode, after the model issues MCP `tool_use`
   calls and receives `tool_result`s, what conditions cause the CLI to **not emit a
   final `result` event and not exit**? Is there a server-side keep-alive / SSE stream
   that is left open (streamable-HTTP MCP) and blocks termination?
2. Is there a supported **per-run wall-clock or max-turns cap** that forces the CLI to
   emit a terminal `result` (even an error/timeout one) and exit, rather than hanging?
   (Claude Code has `--max-turns`; Codex `exec` exits on completion.)
3. Does `-o stream-json` reliably emit a terminal `{"type":"result"}` event on *every*
   exit path (success, tool-loop exhaustion, error)? In our captures it is frequently
   absent.
4. Are there known issues with **remote streamable-HTTP MCP servers** (as opposed to
   stdio MCP) causing the event loop to stay open after the last tool result?
5. Any recommended headless flags for an **automated benchmark** (deterministic exit,
   bounded tool loop, machine-readable final status)?

## Minimal repro
1. Stand up any streamable-HTTP MCP server that exposes a few tools at
   `http://localhost:PORT/mcp/`.
2. Write the `./.gemini/settings.json` above pointing at it.
3. Run the `gemini -p ... -o stream-json --allowed-mcp-server-names ... --approval-mode yolo`
   command on a prompt that requires one tool call.
4. Observe: the tool call reaches the server, but the CLI process stays alive with no
   further stdout and never exits.

## Impact
Claude Code and Codex complete the same 66-task × 4-fault-rate sweep (264 runs each) in
a couple of hours; Gemini reaches ~70 runs in the same window, ~55% of them empty
timeouts. Until the hang is resolved we cannot include Gemini as a reliable headless
data-consumer harness.
