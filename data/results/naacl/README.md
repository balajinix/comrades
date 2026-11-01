# Raw results behind the NAACL 2027 paper

Each CSV has one row per run. Paths to scripts are relative to `evaluation/comrades/`.

| File | Paper item | Produced by |
|---|---|---|
| `rq1_recorded_seeds.csv` | ReAct data-access agent, fail-fast, rates 0 to 0.6, 3 seeds, replay | `run_faults_parallel.py --retry none` |
| `rq1_live.csv` | live floor of the same agent (0.72 at rate 0) | `run_faults.py`, live backends |
| `rq1_react_retry.csv` | same agent with the shared retry policy in the harness | `run_retry_arm.sh` |
| `rq1_cot.csv`, `rq1_reflect.csv` | chain-of-thought and self-reflection agents, fail-fast | `run_agent_strategies.sh` |
| `rq3_live_direct_mcp.csv` | three models, in-process and MCP, live, pooled over fault rates | `run_rq3_parallel.py` |
| `rq3_a2a.csv` | the same over A2A (success only, replay) | `run_rq3_parallel.py` |
| `control_wide_mcp.csv`, `control_wide_a2a.csv` | gold-scored schema mapping: workflow, skill and free loop over MCP; workflow over A2A | `schema_map/run_control.py` |
| `rq5_codex_default_r06.csv`, `rq5_codex_written_policy_r06.csv` | coding assistant as shipped and with the written retry policy, 66 tasks, rate 0.6, identical seeded faults | `harness/run_harness.py` (policy arm: `RQ5_PROSE_POLICY=shared`) |
| `rq5_claude_default_r06.csv`, `rq5_gemini_default_r06.csv` | the other two coding assistants as shipped, same 66 tasks, rate 0.6 and seeded faults (the `antigravity` harness label is the Gemini CLI) | `harness/run_harness.py` |

Notes:
- `success` / `data_obtained` count a run that obtained non-error data. Schema mapping is scored by
  F1 against gold field pairs, and the coding-assistant runs by `gold_obtained`.
- In `rq3_live_direct_mcp.csv`, a failed in-process (`direct`) run records zero tokens and zero latency.
  Compare cost over successful runs only.
- `evaluation/comrades/harness/results/rq5_*_v2.csv` hold the earlier coding-assistant panel over
  fault rates 0 to 0.6, which the paper's appendix reports. A row with zero `tool_calls` is a run that
  made no data-layer calls. The `rq5_*_r06.csv` files above are a clean rerun at rate 0.6, and a rerun
  at the lower rates is planned.
