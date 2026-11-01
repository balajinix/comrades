#!/usr/bin/env bash
#
# The ReAct data-access agent under harness-enforced retry (policy.RetryPolicy: retry
# transient faults at the tool boundary, at most 3 attempts, never 401, no backoff),
# on the same replay setup as Table rq1: 50 requests x 4 fault rates x 3 seeds, Llama
# 70B. Writes results/strategies/rq1_react_retry.csv (the rq1 columns plus attempts).
# Waits for results/strategies/rq1_reflect.csv first, so it never runs beside the
# rq1 stage of run_agent_strategies.sh and the shared provider load stays near 8.
# Needs PROVIDER_API_KEY (or PROVIDER_APIKEY), PROVIDER_PROJECT_ID and PROVIDER_URL.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../../agents/dal_agent/.venv/bin/python"
OUTDIR=results/strategies
OUT="$OUTDIR/rq1_react_retry.csv"
[ -s "$OUT" ] && { echo "$OUT exists; nothing to do"; exit 0; }
export PROVIDER_API_KEY="${PROVIDER_API_KEY:-${PROVIDER_APIKEY:-}}"
[ -n "$PROVIDER_API_KEY" ] && [ -n "${PROVIDER_PROJECT_ID:-}" ] && [ -n "${PROVIDER_URL:-}" ] \
  || { echo "provider credentials missing from the environment"; exit 1; }
export COMRADES_REPLAY=1 KG_AGENT_CACHE_ENABLED=false COMRADES_AGENT_STRATEGY=react
export PROVIDER_MODEL_NAME=meta-llama/llama-3-3-70b-instruct
export MCP_DAL_URL=http://localhost:8201/mcp DAL_MCP_URL=http://localhost:8201/mcp
WAIT_FOR="${WAIT_FOR:-$OUTDIR/rq1_reflect.csv}"
until [ -z "$WAIT_FOR" ] || [ -s "$WAIT_FOR" ]; do sleep 30; done
echo "[$(date +%H:%M:%S)] rq1 react retry"
"$PY" run_faults_parallel.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
  --requests requests.jsonl --retry shared --out "$OUT" \
  --shard-dir "$OUTDIR/shards_rq1_react_retry" --concurrency "${CONCURRENCY:-6}"
echo "[$(date +%H:%M:%S)] rq1 react retry done -> $OUT"
