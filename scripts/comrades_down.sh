#!/usr/bin/env bash
#
# comrades_down.sh — stop the COMRADES A2A stack started by comrades_up.sh.
#
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

stop() {  # name pidfile port
  local name="$1" pidfile="$2" port="$3"
  if [ -f "$pidfile" ]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "stopped $name (pid $(cat "$pidfile"))"
    rm -f "$pidfile"
  fi
  # belt-and-suspenders: free the port if still held
  local pids; pids="$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null && echo "freed port :$port"
  return 0
}

stop dal_agent logs/dal_agent.pid "${DAL_AGENT_PORT:-8005}"
stop mcp_dal   logs/mcp_dal.pid   "${MCP_DAL_PORT:-8201}"
echo "COMRADES stack stopped."
