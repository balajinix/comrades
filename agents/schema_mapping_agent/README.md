# Schema Mapping Agent

A schema mapping agent that uses natural language to map fields between heterogeneous
GraphQL schemas. It is one of the consumption agents in the [COMRADES](../README.md)
multi-agent system and speaks the A2A protocol so the orchestrator can route to it.

## Setup

1. **Environment Variables**
   Create a `.env` file with:
   ```
   PROVIDER_APIKEY=your_provider_api_key
   PROVIDER_PROJECT_ID=your_project_id
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Schema files**
   The in-process backend reads candidate GraphQL schemas from `./data/graphql/`
   (one `*.graphql` file per source/target). Not needed for the server backend,
   which uses the products pre-loaded on `mcp_map_reflect` (see below).

## Mapping backend: mcp_map_reflect server vs. in-process

`find_mappings` (the agent's core field-mapping step) can run against either of two
backends. Selection is by a single environment variable.

### Default (recommended): delegate to the deployed `mcp_map_reflect` MCP server

This is the default configuration within COMRADES: the agent stays a thin client and
calls the deployed `mcp_map_reflect` server over MCP, mirroring how the DAL agent talks
to `mcp_dal`. The heavy mapping stack (chromadb, sentence-transformers) is **not** needed
on this path.

1. Point the agent at the server by setting one variable in your `.env`
   (`MCP_MAP_REFLECT_URL` is accepted as an alias):
   ```bash
   MAP_REFLECT_MCP_URL=http://internal-host:8203/mcp   # staging
   # prod alt: http://internal-host:8203/mcp
   ```
   See `.env.example` for the full template. Off-VPC, tunnel first:
   ```bash
   ssh -L 8203:localhost:8203 <user>@internal-host
   # then use MAP_REFLECT_MCP_URL=http://localhost:8203/mcp
   ```
2. Install deps (the `mcp` client is declared in `requirements.txt` / `pyproject.toml`):
   ```bash
   pip install -r requirements.txt          # or: uv sync
   ```
3. Run the agent as usual (e.g. `python -m app`). When `MAP_REFLECT_MCP_URL` is set,
   `find_mappings` calls the server's `map_schema` tool once per source type and merges
   the results — no code change required.

**Important — the server maps by *product name*, not by your schema text.** On this path
the agent passes the **product names** (the source/target schema names) and the **source
type names** to the server, which uses its own pre-loaded product lookup. So both
source and target must be products the server knows. Check what's available:
```bash
python - <<'PY'
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
URL = "http://internal-host:8203/mcp"
async def main():
    async with streamablehttp_client(URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.call_tool("list_products", {})
            print(getattr(tools, "structuredContent", None) or tools.content)
anyio.run(main)
PY
```
Known products include `finops`, `optimizer`, `apm`, `resilience`, the cloud
providers (`aws`/`azure`/`gcp`), and others. For an **arbitrary** schema the server has
not pre-loaded, use the in-process backend instead.

### Fallback: in-process map-reflect engine

Leave `MAP_REFLECT_MCP_URL` **unset**. `find_mappings` then runs the vendored map-reflect
engine locally, mapping the actual GraphQL schema text you pass in (so it works for any
schema, not just pre-loaded products). This path needs the full dependency set (chromadb,
sentence-transformers — all in `requirements.txt`) and a Provider key. Tune the mapper LLM
with `SCHEMA_MAPPING_MODEL` (default `meta-llama/llama-3-3-70b-instruct`).

### Quick check (server path)

```bash
MAP_REFLECT_MCP_URL=http://internal-host:8203/mcp python - <<'PY'
from app.agent import find_mappings
src = "type CostReport { totalCost: Float account: String }"
tgt = "type Stats { cost: Float accountId: String }"
res = find_mappings(src, "finops", tgt, "optimizer")
print(res["name"], "->", len(res["properties"]), "source fields mapped")
PY
```

## Layout

```
schema_mapping_agent/
├── app/
│   ├── __main__.py        # A2A (Starlette) server entrypoint, port 8002
│   ├── agent.py           # LangGraph mapping agent + find_mappings (the core step)
│   ├── agent_executor.py  # A2A request handler
│   ├── map_reflect/       # vendored in-process map-reflect engine (fallback backend)
│   └── utils/             # GraphQL parsing, schema indexing, provider helpers
├── requirements.txt
└── pyproject.toml
```

## Running the agent

Starts the A2A server on port 8002:

```bash
python -m app
```

The orchestrator in [`a2a_host_agent/`](../a2a_host_agent) routes mapping requests to
it over A2A. Selection of the mapping backend (server vs. in-process) is by
`MAP_REFLECT_MCP_URL` as described above and applies whichever way the agent is invoked.

## Testing

```bash
python app/test_agent.py     # sends a request to a locally running agent on :8002
```
