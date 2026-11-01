# Firing COMRADES runs against the remote kg box

The runners are now remote-aware. Point them at the live `mcp_dal` (and
`mcp_map_reflect`) on the kg machine and the experiments run against **real
backends** — no canned replay, so success is meaningful (this is the clean fix for
the RQ2 canned-data confound).

## 0. Prereqs
- **VPN / network reachability** from wherever the runners execute to
  `internal-host` on the relevant ports (`:8201` mcp_dal, `:8203`
  mcp_map_reflect). Quick check:
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" http://internal-host:8201/mcp \
    -X POST -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
  ```
- Provider creds in `agents/dal_agent/.env` (the agent LLM).
- Run RQ1/RQ2/RQ3 from the **dal_agent venv**; schema-mapping from a venv with the
  `mcp` client (e.g. `agents/dal_agent/.venv`).

## 1. Set the endpoints (one export each)
```bash
export MCP_DAL_URL=http://internal-host:8201/mcp
# mcp_map_reflect (deployed + verified green 2026-06-26; 8 tools per the contract):
export MCP_MAP_REFLECT_URL=http://internal-host:8203/mcp   # staging
# prod alt: http://internal-host:8203/mcp
# from a laptop off-VPC: ssh -L 8203:localhost:8203 anon@example.com
#   then use http://localhost:8203/mcp
```
Both :8203 endpoints respond (a raw curl returns HTTP 400 = MCP server up, needs the
client handshake). The server uses provider project `e7a5ea6c` from its own `.env`.

> **Coordination:** the schema-mapping team's session runs experiments against the
> *same* `mcp_map_reflect` + provider project. The full schema_map sweep makes many LLM
> calls (find_field_mapping + reflect per row x 521 rows x rates x seeds), so check with
> that session before launching the full sweep to avoid contending on the shared project.
Every runner reads these (falling back to `DAL_MCP_URL`, then localhost). Because the
URL is non-local, **replay defaults OFF** (real backend calls); override with
`COMRADES_REPLAY=1` only if you want canned data.

## 2. RQ1 — failure → communication (single agent)
```bash
cd evaluation/comrades
agents/dal_agent/.venv/bin/python run_faults.py \
  --requests requests.jsonl --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
  --model meta-llama/llama-3-3-70b-instruct --out results/rq1_remote.csv
python3 aggregate.py --rq1 results/rq1_remote.csv
```

## 3. RQ3 — architecture x model
The three architecture runners (`runners/{direct,mcp_only,a2a}.py`) read
`MCP_DAL_URL` and take `--model`. Sweep the three locked models
(`meta-llama/llama-3-3-70b-instruct`, `mistralai/mistral-medium-2505`,
`ibm/granite-4-h-small`) x architecture x fault rate via the RQ3 driver (task #138).

## 4. Schema-mapping family — gold-scored, multi-hop (real mcp_map_reflect)
Drop `--mock`; the runner calls the real `mcp_map_reflect` at `MCP_MAP_REFLECT_URL`:
```bash
cd evaluation/comrades/schema_map
# smoke first (5 rows, 1 fault level):
agents/dal_agent/.venv/bin/python run_schema_map.py --limit 5 --rates 0.0 --retries 0
# full sweep:
agents/dal_agent/.venv/bin/python run_schema_map.py \
  --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 --retries 2 --out results/schema_map_remote.csv
```

## Notes
- Faults are injected at the DAL agent's client-side tool boundary, so they work
  identically whether mcp_dal is local or remote.
- If a run 401s/times out broadly, it's a VPN/cred issue on the kg box's backends,
  not the runner — check `mcp_dal/.env` there (backend credentials are site-specific).
</content>
