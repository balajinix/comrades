#!/usr/bin/env bash
#
# comrades_up.sh - single-command bring-up of the COMRADES A2A stack.
#
# Starts:
#   1. mcp_dal    (Data Access Layer MCP server)  -> http://localhost:8201/mcp
#   2. dal_agent  (A2A agent over mcp_dal)         -> http://localhost:8005
#
# Then fire natural-language queries over A2A:
#   agents/dal_agent/.venv/bin/python scripts/ask.py "list up to 3 ITSM incidents"
#
# Requirements: python3.13, uv, and the mcp_dal repo available at MCP_DAL_DIR.
# Config: copy agents/dal_agent/.env.example -> agents/dal_agent/.env and fill Provider creds.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

# --- config (override via env) ---
MCP_DAL_DIR="${MCP_DAL_DIR:-$HOME/src/mcp_dal}"
MCP_DAL_PORT="${MCP_DAL_PORT:-8201}"
DAL_AGENT_PORT="${DAL_AGENT_PORT:-8005}"
# mcp_dal's infra tool wants these present at boot (any value works unless
# you actually query infra); real backends use the creds in mcp_dal/.env.
export LITELLM_API_KEY="${LITELLM_API_KEY:-dummy-key}"
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://localhost:9999}"

say()  { printf '==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
die()  { printf '[error] %s\n' "$*" >&2; exit 1; }

port_up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_port() {  # port timeout
  local port="$1" timeout="${2:-30}" i=0
  while ! port_up "$port"; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && return 1
    sleep 1
  done
  return 0
}

# --- 0. prereqs ---
command -v uv >/dev/null 2>&1 || die "uv not found (https://docs.astral.sh/uv/)"
command -v python3.13 >/dev/null 2>&1 || warn "python3.13 not on PATH; uv will fetch it"
[ -d "$MCP_DAL_DIR" ] || die "mcp_dal not found at MCP_DAL_DIR=${MCP_DAL_DIR} (set MCP_DAL_DIR=/path/to/mcp_dal)"
[ -f agents/dal_agent/.env ] || die "agents/dal_agent/.env missing. Run: cp agents/dal_agent/.env.example agents/dal_agent/.env  then fill Provider creds."

# --- 1. mcp_dal ---
if port_up "$MCP_DAL_PORT"; then
  say "mcp_dal already listening on :${MCP_DAL_PORT} - reusing"
else
  if [ ! -x "$MCP_DAL_DIR/venv/bin/dal-mcp-server" ]; then
    say "Setting up mcp_dal (one-time)..."
    ( cd "$MCP_DAL_DIR" && PYTHON="${PYTHON:-python3.13}" bash setup.sh ) \
      || die "mcp_dal setup failed (see its output)"
  fi
  say "Starting mcp_dal on :${MCP_DAL_PORT} ..."
  ( cd "$MCP_DAL_DIR" && nohup venv/bin/dal-mcp-server \
      --transport streamable-http --port "$MCP_DAL_PORT" \
      > "$ROOT_DIR/logs/mcp_dal.log" 2>&1 & echo $! > "$ROOT_DIR/logs/mcp_dal.pid" )
  wait_for_port "$MCP_DAL_PORT" 40 \
    || die "mcp_dal did not come up; see logs/mcp_dal.log"
  say "mcp_dal up (pid $(cat logs/mcp_dal.pid))"
fi

# --- 2. dal_agent ---
if port_up "$DAL_AGENT_PORT"; then
  say "dal_agent already listening on :${DAL_AGENT_PORT} - reusing"
else
  say "Syncing dal_agent deps..."
  ( cd dal_agent && uv sync --link-mode=copy >/dev/null ) || die "uv sync failed in dal_agent"
  say "Starting dal_agent on :${DAL_AGENT_PORT} ..."
  ( cd dal_agent && nohup uv run --active python -m app \
      > "$ROOT_DIR/logs/dal_agent.log" 2>&1 & echo $! > "$ROOT_DIR/logs/dal_agent.pid" )
  wait_for_port "$DAL_AGENT_PORT" 40 \
    || die "dal_agent did not come up; see logs/dal_agent.log"
  say "dal_agent up (pid $(cat logs/dal_agent.pid))"
fi

cat <<EOF

COMRADES stack is up.
  mcp_dal     http://localhost:${MCP_DAL_PORT}/mcp
  dal_agent   http://localhost:${DAL_AGENT_PORT}   (A2A; card at /.well-known/agent.json)

Fire a query:
  agents/dal_agent/.venv/bin/python scripts/ask.py "list up to 3 ITSM incidents"
  agents/dal_agent/.venv/bin/python scripts/ask.py "what are my top cloud costs this month"

Stop everything:
  scripts/comrades_down.sh
EOF
