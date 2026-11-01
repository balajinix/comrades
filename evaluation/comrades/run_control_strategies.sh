#!/usr/bin/env bash
#
# Rerun the agent rows of the control-discipline table (skill and free, which run
# the data-access agent's loop) under other agent strategies, against the live
# mapping service. Same setup as the paper's table: the 15 rows below x rates
# 0/0.2/0.4/0.6 x seeds 1,2,3, MCP transport, shared retry policy, Llama 70B.
#
# Steps:
#   1. drift check: the coded workflow (no agent model calls) at rate 0, seed 1. It
#      must reproduce the paper's workflow F1 (about 0.68) before anything else runs.
#   2. one stream per (strategy, mechanism), each through its OWN fault proxy on its
#      own port with its own call log, because a proxy serves one run at a time.
# Ports are checked before use, and nothing is sent until the started proxy itself
# is listening. Outputs: results/strategies/control_<mechanism>_<strategy>.csv.
#
# Needs PROVIDER_API_KEY (or PROVIDER_APIKEY), PROVIDER_PROJECT_ID and PROVIDER_URL for
# the agent. Knobs: MAP_REFLECT_URL, STRATEGIES (default "cot reflect"; add react
# for a same-day baseline), BASE_PORT (default 8241), WAIT_FOR (a file to wait for
# before step 2, default the retry arm's output, so the shared key stays near 8).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../../agents/dal_agent/.venv/bin/python"
OUTDIR="$HERE/results/strategies"
LOGDIR="$OUTDIR/control_logs"
mkdir -p "$LOGDIR"
UPSTREAM="${MAP_REFLECT_URL:?set MAP_REFLECT_URL to the mapping service MCP endpoint}"
STRATEGIES="${STRATEGIES:-cot reflect}"
BASE_PORT="${BASE_PORT:-8241}"
WAIT_FOR="${WAIT_FOR:-$OUTDIR/rq1_react_retry.csv}"
ROWS=coverage_00008,coverage_00084,coverage_00119,coverage_00148,coverage_00168,coverage_00181,coverage_00281,coverage_00361,coverage_00362,coverage_00393,coverage_00436,coverage_00438,coverage_00559,coverage_00619,coverage_00642
export PROVIDER_API_KEY="${PROVIDER_API_KEY:-${PROVIDER_APIKEY:-}}"
[ -n "$PROVIDER_API_KEY" ] && [ -n "${PROVIDER_PROJECT_ID:-}" ] && [ -n "${PROVIDER_URL:-}" ] \
  || { echo "provider credentials missing from the environment"; exit 1; }
export PROVIDER_MODEL_NAME=meta-llama/llama-3-3-70b-instruct KG_AGENT_CACHE_ENABLED=false
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

start_proxy() {  # port log -> echoes pid, or fails
  local port="$1" log="$2" pid
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is in use; not starting" >&2; return 1
  fi
  FAULT_PROXY_TARGET="$UPSTREAM" FAULT_PROXY_PORT="$port" FAULT_PROXY_LOG="$log" \
    nohup "$PY" harness/fault_proxy.py >"$LOGDIR/proxy_$port.out" 2>&1 &
  pid=$!
  for _ in $(seq 1 30); do
    lsof -nP -a -p "$pid" -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && { echo "$pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  echo "proxy on $port did not come up" >&2; kill "$pid" 2>/dev/null; return 1
}

# 1. drift check
port=$BASE_PORT
pid=$(start_proxy "$port" "$LOGDIR/drift_wf.jsonl") || exit 1
PIDS+=("$pid")
echo "[$(date +%H:%M:%S)] drift check (workflow, rate 0, seed 1)"
FAULT_PROXY_URL="http://localhost:$port/mcp" FAULT_PROXY_LOG="$LOGDIR/drift_wf.jsonl" \
  "$PY" schema_map/run_control.py --mechanism wf --rates 0.0 --seeds 1 --rows "$ROWS" \
  --out "$OUTDIR/control_drift_wf.csv" 2>&1 | grep -E "^\[wf"
F1=$("$PY" -c "
import csv; r=list(csv.DictReader(open('$OUTDIR/control_drift_wf.csv')))
print(round(sum(float(x['f1']) for x in r)/len(r),3))")
echo "[$(date +%H:%M:%S)] drift check mean row F1 = $F1 (paper seed 1: 0.646, seeds 1-3: 0.680)"
if ! "$PY" -c "import sys; sys.exit(0 if float('$F1') >= 0.55 else 1)"; then
  echo "drift check failed: the service does not reproduce the paper's workflow F1; stopping"
  exit 1
fi

# 2. agent streams
until [ -z "$WAIT_FOR" ] || [ -s "$WAIT_FOR" ]; do sleep 30; done
STREAM_PIDS=()
for s in $STRATEGIES; do
  for mech in skill free; do
    out="$OUTDIR/control_${mech}_${s}.csv"
    [ -s "$out" ] && continue
    port=$((port + 1))
    log="$LOGDIR/${mech}_${s}.jsonl"
    pid=$(start_proxy "$port" "$log") || continue
    PIDS+=("$pid")
    echo "[$(date +%H:%M:%S)] control $mech $s on port $port"
    ( COMRADES_AGENT_STRATEGY="$s" FAULT_PROXY_URL="http://localhost:$port/mcp" FAULT_PROXY_LOG="$log" \
        "$PY" schema_map/run_control.py --mechanism "$mech" --rates 0.0,0.2,0.4,0.6 \
        --seeds 1,2,3 --rows "$ROWS" --fresh-log --out "$out.part" \
        >"$LOGDIR/${mech}_${s}.out" 2>&1 \
      && mv "$out.part" "$out" && echo "[$(date +%H:%M:%S)] control $mech $s done -> $out" ) &
    STREAM_PIDS+=($!)
  done
done
wait "${STREAM_PIDS[@]:-}"
echo "[$(date +%H:%M:%S)] control strategies all done"
