# COMRADES

A **data harness** for agentic data access across IT operations platforms, with a
built-in **fault-injection harness** for testing it. COMRADES puts heterogeneous
backends (cost, APM, ITSM, resilience, infrastructure) behind one MCP tool
interface, serves a data-access agent in-process, over MCP or over A2A, and wraps
the agent loop with production controls. Consumption agents (schema mapping,
GraphQL generation) and coding assistants are its clients.

### Coding-assistant results

The coding-assistant table and panel (b) of the cost-and-recovery figure come from the 66-task
runs in `evaluation/comrades/harness/results/rq5_{claude,codex,antigravity}_v2.csv`
(`antigravity` is the Gemini CLI). `data/results/rq4_harness_*.csv` hold an earlier 15-task
pilot and are not used in the paper.

### Harness controls and where they live

| Control | What it does | Code |
|---|---|---|
| Tool-call failure policy | fail-fast or bounded retry of transient faults (never 401) | `evaluation/comrades/policy.py` |
| Loop detection | blocks exact-intent repeats and same-action overuse | `agents/dal_agent/kg_agent/react.py` |
| Step budget and short circuit | counts only productive tool calls; ends a request once a tool keeps failing | `agents/dal_agent/kg_agent/react.py` |
| Answer contract | challenges answers given without a tool call; coaches malformed tool arguments with the tool's schema | `agents/dal_agent/kg_agent/react.py` |
| Model-provider breaker | bounded backoff plus a circuit breaker on the model provider | `agents/dal_agent/kg_agent/llm.py` |
| Observation compaction | caps each tool result fed back to the model; keeps the full result in the trace | `agents/dal_agent/kg_agent/observation.py` |
| Sub-schema selection | surfaces only the relevant part of a large schema | `agents/schema_mapping_agent/app/utils/subschema_selector.py` |
| Answer cache | read-through cache keyed by normalised question and model; skips errors and blanks; manual invalidation | `agents/dal_agent/kg_agent/__init__.py` |
| Traces | every run and step persisted (tool call, raw result, tokens, latency) | `agents/dal_agent/kg_agent/persistence/` |
| Fault injector | seeded faults (timeout, 429, 5xx, 401, partial outage) at the tool boundary | `evaluation/comrades/faults.py`, `evaluation/comrades/harness/fault_proxy.py` |
| Redundancy classifier | splits necessary recovery from redundant re-issues by intent fingerprint | `evaluation/comrades/redundancy.py` |

> **Anonymised for double-blind review.** Vendor product names are replaced by
> generic descriptors throughout code, data and file names (`apm`, `itsm`,
> `finops`, `optimizer`, `k8scost`, `resilience`, `infra`, `graphdb`), and the LLM
> provider's SDK and LangChain integration appear as the placeholders `provider_sdk`
> and `langchain_provider`. Credential templates (`.env.example`), lock files,
> runtime logs, session databases and exploratory scripts that probed live backends
> are not included. The datasets in `data/datasets/` and the result tables in
> `data/results/` are complete; `evaluation/comrades/schema_map/score.py` and
> `aggregate_control.py` run offline on them with the Python standard library.

A live demo (when running) is available at:
[http://internal-host:8080/dev-ui/?app=orchestrator](http://internal-host:8080/dev-ui/?app=orchestrator)

---

## Architecture

```
                 user / dev-ui
                      |  A2A
              UI-Orchestrator (a2a_host_agent)  :8080
          ________________|______________
         |              |                |
   GraphQL Query   Schema Mapping    DAL agent
   :8001           :8002             (federates mcp_dal)  :8005
                                         |  MCP
                                     mcp_dal  :8201
                                     (cost / APM / ITSM /
                                      resilience / infra)
```

| Component | Dir | Port | Role |
|---|---|---|---|
| UI-Orchestrator | `agents/a2a_host_agent/` | 8080 | ADK chat UI; routes requests to agents over A2A |
| GraphQL Query Agent | `agents/graphql_query_agent/` | 8001 | NL → GraphQL query generation |
| Schema Mapping Agent | `agents/schema_mapping_agent/` | 8002 | maps schemas across heterogeneous sources |
| **DAL Agent** | `agents/dal_agent/` | 8005 | federated ReAct agent over the `mcp_dal` data-access layer |
| `mcp_dal` (external) | `../kg/mcp_dal` | 8201 | MCP server: ~100 tools over cost/APM/ITSM/resilience/infra backends |
| COMRADES harness | `evaluation/comrades/` | — | communication-mechanism + failure-injection study |

---

## Prerequisites

- **Python 3.13** and **[uv](https://docs.astral.sh/uv/)**.
- **Provider credentials** for the agents (`PROVIDER_API_KEY`, `PROVIDER_URL`, `PROVIDER_PROJECT_ID`, `PROVIDER_MODEL_NAME`; see `agents/dal_agent/kg_agent/config.py`).
- **`mcp_dal`** — the external Data Access Layer MCP server that the DAL agent
  federates (the ~100 cost/APM/ITSM/resilience/infrastructure tools). It lives in
  its own repository; clone it and point COMRADES at it via `MCP_DAL_DIR`:

  ```bash
  git clone <MCP_DAL_REPO_URL> mcp_dal   # repository withheld for double-blind review
  export MCP_DAL_DIR=/path/to/mcp_dal     # default: ../mcp_dal
  ```

  See `mcp_dal`'s own README for backend credentials and setup.
  `scripts/comrades_up.sh` runs its `setup.sh` and starts it on port 8201 when
  found at `MCP_DAL_DIR`.

---

## Setup

Each agent has its **own** `.venv` (dependency sets differ). Python 3.13.

```bash
# per agent, e.g. schema_mapping_agent:
python -m venv /data/comrades/schema_mapping_agent-venv
ln -s /data/comrades/schema_mapping_agent-venv .venv
source .venv/bin/activate
# (the dal_agent uses uv: see agents/dal_agent/kg_agent/config.py and scripts/comrades_up.sh)
```

**agent-analytics** (instrumentation used by the agents) — install from:
- the agent-analytics SDK repository (withheld for double-blind review) →
  `cp -r agent-analytics/sdk/src/agent_analytics .venv/lib/python3.13/site-packages/`
- the agent-analytics core repository (withheld for double-blind review) →
  `cp -r agent-analytics/core/src/agent_analytics_core .venv/lib/python3.13/site-packages/`

Known install quirk (shared `a2a` package across venvs):
```bash
cp -r schema_mapping_agent-venv/lib/python3.13/site-packages/a2a \
      graphql_query_agent-venv/lib/python3.13/site-packages/
```

---

## Running

### Full multi-agent system

```bash
./start_all_agents.sh      # starts all agents + orchestrator (+ mcp_dal, MCP servers)
./stop_all_agents.sh
```

### Just the DAL agent over A2A (recommended for data access)

```bash
$EDITOR agents/dal_agent/.env     # set the PROVIDER_* variables
scripts/comrades_up.sh                        # starts mcp_dal (:8201) + dal_agent (:8005)
agents/dal_agent/.venv/bin/python scripts/ask.py "list up to 3 ITSM incidents"
scripts/comrades_down.sh
```

### Querying from your own code (A2A)

```python
import asyncio, uuid, httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart

async def ask(text, url="http://localhost:8005"):
    async with httpx.AsyncClient(timeout=180) as hc:
        card = await A2ACardResolver(hc, url).get_agent_card()
        client = A2AClient(hc, card, url=url)
        msg = Message(role=Role.user, messageId=uuid.uuid4().hex,
                      parts=[Part(root=TextPart(text=text))])
        resp = await client.send_message(
            SendMessageRequest(id=uuid.uuid4().hex,
                               params=MessageSendParams(message=msg)))
        return resp.model_dump()["result"]["status"]["message"]["parts"][0]["text"]

print(asyncio.run(ask("list up to 3 ITSM incidents")))
```

The agent card is at `http://localhost:8005/.well-known/agent.json`; point `ask.py`
elsewhere with `DAL_AGENT_URL=http://host:8005`.

### What the DAL agent can reach

`mcp_dal` exposes ~100 tools across five families; the agent picks and chains them:

| Family | Examples |
|---|---|
| **Cloud cost (FinOps)** | cost reports, rightsizing, anomalies, billing accounts (AWS/Azure/GCP/OCI) |
| **Observability / APM** | application & infra metrics, events, SLOs, releases, service topology |
| **IT service mgmt** | incidents, users, knowledge articles |
| **Resilience / security** | resilience posture, compliance risks, CVE lookups, workflows |
| **Infrastructure graph** | natural-language to infrastructure-query-language (IQL) |

Backend credentials (APM/ITSM/FinOps/Resilience/Infra) live in
`mcp_dal/.env`; backend credentials are site-specific and not included. Missing backend
creds just disable those tools; the agent still starts.

---

## Evaluation: the COMRADES experiments

`evaluation/comrades/` studies **agent communication robustness under failure**: by
injecting faults at the agent's tool boundary and sweeping fault rate, failure-handling
design (fail-fast vs. retry), communication architecture, and model, we measure how task
success and communication volume (messages, tokens, latency) move — and how much of the
induced communication is *redundant* (a redundancy classifier separates necessary
recovery from over-communication). This backs the ICSE 2027 paper in [`paper/`](paper).

The experiments are organized as research questions:

| RQ | Question | Driver |
|---|---|---|
| **RQ1** | As the backend fault rate rises, how do a data agent's communication and task success change? | `run_faults.py` |
| **RQ3** | How do communication architecture (in-process / MCP / A2A) and model size shift the under-/over-communication frontier? | `run_rq3.py` |
| **RQ4** | What operating point and guardrails best balance reliability vs. cost for SWE/SRE deploying data agents? | synthesis of RQ1–3, 5 |
| **RQ5** | When AI coding harnesses (Claude Code, Codex, Gemini, IDE-harness) fetch data while completing a task, how does failure reshape their communication? (on hold) | `harness/` |

A gold-scored, multi-hop **schema-mapping task family** (`schema_map/`, 521 annotated
rows) runs the same fault sweep over the `mcp_map_reflect` tools and scores predicted
field mappings against gold (precision / recall / F1) — turning a proxy metric into a
real one.

```bash
cd evaluation/comrades
# RQ1 — failure → communication (single agent):
../../agents/dal_agent/.venv/bin/python run_faults.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3
# RQ3 — architecture × model sweep:
../../agents/dal_agent/.venv/bin/python run_rq3.py --mechanisms direct,mcp_only,a2a --rates 0.0,0.2,0.4,0.6
# schema-mapping task family (gold-scored, multi-hop):
cd schema_map && ../../../agents/dal_agent/.venv/bin/python run_schema_map.py --rates 0.0,0.2,0.4,0.6
```

Plans and runbooks: **[evaluation/COMRADES_PLAN.md](evaluation/COMRADES_PLAN.md)**
(overall), **[evaluation/comrades/EXECUTION_PLAN.md](evaluation/comrades/EXECUTION_PLAN.md)**
(per-RQ method + status), **[evaluation/comrades/REMOTE_RUNS.md](evaluation/comrades/REMOTE_RUNS.md)**
(running against the live `mcp_dal` / `mcp_map_reflect` on the remote box).

---

## Troubleshooting

- **`NameResolutionError` / connection errors in answers** — the agent reasoned and
  called the right tool, but the backend host is unreachable (usually need **corporate VPN**
  or backend creds in `mcp_dal/.env`). Not a code error.
- **403 `user_authorization_failed` from Provider** — your `PROVIDER_PROJECT_ID` doesn't
  list your account as a member; use a project you belong to.
- **Port already in use** — `scripts/comrades_down.sh` frees `:8201`/`:8005`, or set
  `MCP_DAL_PORT` / `DAL_AGENT_PORT`.
- **Instant 0.0s answers** — the read-through cache served a repeated question. Normal
  for interactive use; the benchmark harness disables it.
- **`dal_agent` won't start** — check `logs/dal_agent.log`; usually a missing
  `agents/dal_agent/.env` or Provider creds.
- **`mcp_dal` won't start** — check `logs/mcp_dal.log`; a WIP tool may log a warning and
  be skipped (non-fatal).

---

## Repository layout

- `agents/a2a_host_agent/` — orchestrator (ADK) + A2A routing
- `agents/graphql_query_agent/`, `agents/schema_mapping_agent/` — specialized agents
- `agents/dal_agent/` — federated DAL agent (vendored `kg_agent`) over `mcp_dal`
- `common/` — shared utilities (dataviz, mapping, provider, caching)
- `evaluation/` — datasets, eval scripts, and the COMRADES harness (`comrades/`)
- `scripts/` — `comrades_up.sh`, `comrades_down.sh`, `ask.py`
- `ui/`, `demo/` — UI and demo assets

---

## Appendix G results (`lab/`)

`lab/` holds the code and raw results behind Appendix G (data interfaces and bursty faults).
Install with `cd lab && pip install -e .` (see `lab/pyproject.toml`); the data store for the
interface comparison is built from the public ITBench-Lite SRE snapshots (Apache 2.0).

| Paper | Script | Results |
|---|---|---|
| Table on data interfaces (35 ITBench-Lite questions, three models) | `lab/experiments/analyze_grid.py` | `lab/findings/grid/`, `lab/findings/grid_summary.md` |
| Table on bursty faults (coded workflow, virtual time, no model calls) | `lab/experiments/exp_bursty_faults.py` | `lab/findings/bursty_faults.md`, `bursty_faults_[abc].csv` |
| Retry-placement pilot (six TPC-H questions, seven seeds) | `lab/experiments/exp_llm_faults.py` | `lab/findings/llm_faults_v2.csv` |

The interface table uses the `sf=35` rows of `grid_summary.md` (all 35 scenarios loaded),
seed 0, for generic SQL, fine-grained tools, handles, router, skills, code and GraphQL with
aggregates. The bursty table uses grid A at a marginal fault rate of 0.3.
