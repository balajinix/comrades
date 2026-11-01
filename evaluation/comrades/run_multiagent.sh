#!/usr/bin/env bash
#
# Multi-agent failure sweep: restart the DAL agent at each backend fault rate,
# then have the dashboard (consumption) agent fetch from it over A2A under two
# retry policies, measuring inter-agent message volume + task success.
#
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
cd "$ROOT"

RATES="${RATES:-0.0,0.3,0.6}"
RETRIES="${RETRIES:-0,2}"
REQUESTS="${REQUESTS:-$ROOT/evaluation/comrades/requests.jsonl}"
OUT="${OUT:-$ROOT/evaluation/comrades/results/multiagent.csv}"
DAL_DIR="${DAL_DIR:-$ROOT/agents/dal_agent}"
DASH_DIR="${DASH_DIR:-$ROOT/agents/dashboard_agent}"
[ -f "$OUT" ] && rm -f "$OUT"   # fresh sweep; mafetch appends per rate

# Credentials already in the environment win over the agent's .env file.
[ -n "${PROVIDER_API_KEY:-}" ] || { set -a; source "$DAL_DIR/.env" 2>/dev/null; set +a; }

# Remote mcp_dal: set MCP_DAL_URL=http://<host>:8201/mcp to run against
# the real backends on the kg box. Replay defaults OFF for a remote (real) backend so
# the agent actually queries live data, and ON for a local canned mcp_dal.
MCP_DAL_URL="${MCP_DAL_URL:-${DAL_MCP_URL:-http://localhost:8201/mcp}}"
export DAL_MCP_URL="$MCP_DAL_URL"
case "$MCP_DAL_URL" in
  *localhost*|*127.0.0.1*) REMOTE_DAL=0 ;;
  *) REMOTE_DAL=1 ;;
esac
COMRADES_REPLAY="${COMRADES_REPLAY:-$([ "$REMOTE_DAL" = 1 ] && echo 0 || echo 1)}"
echo "mcp_dal=$MCP_DAL_URL  remote=$REMOTE_DAL  replay=$COMRADES_REPLAY"

ensure_mcp_dal() {
  [ "$REMOTE_DAL" = 1 ] && { echo "using remote mcp_dal; not starting a local server"; return; }
  lsof -nP -iTCP:8201 -sTCP:LISTEN >/dev/null 2>&1 && return
  ( cd $HOME/code/kg/mcp_dal && \
    LITELLM_API_KEY=dummy LITELLM_BASE_URL=http://localhost:9999 \
    nohup venv/bin/dal-mcp-server --transport streamable-http --port 8201 \
      >/tmp/mcp_dal_server.log 2>&1 & ) ; sleep 7
}

restart_dal_with_faults() {  # rate
  kill -9 $(lsof -nP -tiTCP:8005 2>/dev/null) 2>/dev/null; sleep 1
  rm -f /tmp/dal_agent.db
  ( cd "$DAL_DIR" && KG_AGENT_CACHE_ENABLED=false \
    DAL_MCP_URL="$MCP_DAL_URL" COMRADES_REPLAY="$COMRADES_REPLAY" \
    COMRADES_FAULT_RATE="$1" COMRADES_FAULT_SEED=1 \
    nohup .venv/bin/python -m app >/tmp/dal_agent.log 2>&1 & )
  for i in $(seq 1 30); do lsof -nP -iTCP:8005 -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  echo "dal_agent failed to start at rate $1"; return 1
}

ensure_mcp_dal
echo "=== Multi-agent failure sweep: rates=$RATES retries=$RETRIES ==="
IFS=',' read -ra RATE_ARR <<< "$RATES"
for rate in "${RATE_ARR[@]}"; do
  echo "----- DAL backend fault rate = $rate -----"
  restart_dal_with_faults "$rate" || continue
  COMRADES_FAULT_RATE="$rate" "$DASH_DIR/.venv/bin/python" "$DASH_DIR/mafetch.py" \
    --retries "$RETRIES" --requests "$REQUESTS" --out "$OUT" \
    2>&1 | grep -E '=>|fault rate'
done
# leave the DAL agent running fault-free
restart_dal_with_faults 0 >/dev/null 2>&1 || true
echo "=== done ==="
