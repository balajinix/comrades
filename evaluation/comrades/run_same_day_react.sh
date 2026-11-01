#!/usr/bin/env bash
#
# Same-day react baselines for the strategy comparison, because react's tokens per
# task drifted since the paper's June runs:
#   1. rq1 react fail-fast (Table rq1 setup) -> results/strategies/rq1_react.csv
#   2. if MAP_REFLECT_URL is set: the control table's agent rows (run_control_strategies.sh)
#   3. rq3 react, direct and mcp_only x 3 models -> results/strategies/rq3_react.csv
# then resumes a paused run_agent_strategies.sh (its pid as $1, optional), which goes
# on to rq3 cot and reflect. One stage at a time, concurrency 6, so the shared
# provider load stays near 8. Finished files are skipped.
# Needs PROVIDER_API_KEY (or PROVIDER_APIKEY), PROVIDER_PROJECT_ID and PROVIDER_URL.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../../agents/dal_agent/.venv/bin/python"
OUTDIR=results/strategies
RESUME_PID="${1:-}"
LLAMA=meta-llama/llama-3-3-70b-instruct
MODELS="$LLAMA,mistralai/mistral-medium-2505,ibm/granite-4-h-small"
export PROVIDER_API_KEY="${PROVIDER_API_KEY:-${PROVIDER_APIKEY:-}}"
[ -n "$PROVIDER_API_KEY" ] && [ -n "${PROVIDER_PROJECT_ID:-}" ] && [ -n "${PROVIDER_URL:-}" ] \
  || { echo "provider credentials missing from the environment"; exit 1; }
export COMRADES_REPLAY=1 KG_AGENT_CACHE_ENABLED=false COMRADES_AGENT_STRATEGY=react
export PROVIDER_MODEL_NAME="$LLAMA"
export MCP_DAL_URL=http://localhost:8201/mcp DAL_MCP_URL=http://localhost:8201/mcp
export COMRADES_RECORD_ATTEMPTS=1  # same "attempts" definition as the retry arm

if [ ! -s "$OUTDIR/rq1_react.csv" ]; then
  echo "[$(date +%H:%M:%S)] rq1 react (fail-fast, same day)"
  "$PY" run_faults_parallel.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
    --requests requests.jsonl --out "$OUTDIR/rq1_react.csv" \
    --shard-dir "$OUTDIR/shards_rq1_react" --concurrency 6
  echo "[$(date +%H:%M:%S)] rq1 react done -> $OUTDIR/rq1_react.csv"
fi
# tab:control agent rows against the live mapping service (drift check first). Runs
# here, between the rq1 and rq3 stages, so the shared key stays near 8 including the
# mapping service's own model calls.
if [ -n "${MAP_REFLECT_URL:-}" ]; then
  echo "[$(date +%H:%M:%S)] control strategies (live mapping service)"
  WAIT_FOR="" ./run_control_strategies.sh
fi
if [ ! -s "$OUTDIR/rq3_react.csv" ]; then
  echo "[$(date +%H:%M:%S)] rq3 react"
  "$PY" run_rq3_parallel.py --mechanisms direct,mcp_only --models "$MODELS" \
    --rates 0.0,0.2,0.4,0.6 --seeds 1 --requests requests.jsonl \
    --out "$OUTDIR/rq3_react.csv" --shard-dir "$OUTDIR/shards_rq3_react" --concurrency 6
  echo "[$(date +%H:%M:%S)] rq3 react done -> $OUTDIR/rq3_react.csv"
fi
if [ -n "$RESUME_PID" ] && kill -0 "$RESUME_PID" 2>/dev/null; then
  kill -CONT "$RESUME_PID" && echo "[$(date +%H:%M:%S)] resumed run_agent_strategies.sh ($RESUME_PID)"
fi
echo "[$(date +%H:%M:%S)] same-day react baselines done"
