#!/usr/bin/env bash
#
# Rerun the paper's ReAct experiments under other agent strategies, on the replay
# substrate, so results do not rest on a single agent design.
#   rq1  fail-fast fault-rate sweep (Table rq1)          run_faults_parallel.py
#   rq2  dashboard agent over A2A, fail-fast vs retry    run_multiagent.sh
#   rq3  direct vs MCP x three models (Table rq3)         run_rq3_parallel.py
# The strategy is set by COMRADES_AGENT_STRATEGY, read by the data-access agent's
# loop (agents/dal_agent/kg_agent/react.py). Every output goes to results/strategies/,
# so the paper's existing CSVs and shards are never touched. Finished files are
# skipped, so the script is safe to rerun after an interruption.
#
# Needs PROVIDER_API_KEY (or PROVIDER_APIKEY), PROVIDER_PROJECT_ID and PROVIDER_URL.
# Knobs: STRATEGIES (default "cot reflect"), RQ3_STRATEGIES (default "react cot reflect",
# since the paper's rq3 ran live and needs a replay react baseline), EXPS (default
# "rq1 rq2 rq3"), CONCURRENCY (default 6; keep the total across sessions near 8).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="$HERE/../../agents/dal_agent/.venv/bin/python"
OUTDIR=results/strategies
mkdir -p "$OUTDIR"
STRATEGIES="${STRATEGIES:-cot reflect}"
RQ3_STRATEGIES="${RQ3_STRATEGIES:-react cot reflect}"
EXPS="${EXPS:-rq1 rq2 rq3}"
CONC="${CONCURRENCY:-6}"
LLAMA=meta-llama/llama-3-3-70b-instruct
MODELS="$LLAMA,mistralai/mistral-medium-2505,ibm/granite-4-h-small"

export PROVIDER_API_KEY="${PROVIDER_API_KEY:-${PROVIDER_APIKEY:-}}"
[ -n "$PROVIDER_API_KEY" ] && [ -n "${PROVIDER_PROJECT_ID:-}" ] && [ -n "${PROVIDER_URL:-}" ] \
  || { echo "provider credentials missing from the environment"; exit 1; }
export COMRADES_REPLAY=1 KG_AGENT_CACHE_ENABLED=false PROVIDER_MODEL_NAME="$LLAMA"
export MCP_DAL_URL=http://localhost:8201/mcp DAL_MCP_URL=http://localhost:8201/mcp

ensure_mcp_dal() {  # the replay substrate still needs the tool catalog
  lsof -nP -iTCP:8201 -sTCP:LISTEN >/dev/null 2>&1 && return
  ( cd "$HOME/code/kg/mcp_dal" && \
    LITELLM_API_KEY=dummy LITELLM_BASE_URL=http://localhost:9999 \
    nohup venv/bin/dal-mcp-server --transport streamable-http --port 8201 \
      >/tmp/mcp_dal_server.log 2>&1 & ); sleep 7
}
ensure_mcp_dal

has() { case " $EXPS " in *" $1 "*) return 0;; esac; return 1; }
stamp() { date "+%H:%M:%S"; }

# Stage by stage (all of rq1, then rq2, then rq3), so the table that matters most
# lands first; each stage writes its CSV as soon as it completes.
for s in $STRATEGIES; do
  export COMRADES_AGENT_STRATEGY="$s"
  if has rq1 && [ ! -s "$OUTDIR/rq1_$s.csv" ]; then
    echo "[$(stamp)] rq1 $s"
    "$PY" run_faults_parallel.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
      --requests requests.jsonl --out "$OUTDIR/rq1_$s.csv" \
      --shard-dir "$OUTDIR/shards_rq1_$s" --concurrency "$CONC"
    echo "[$(stamp)] rq1 $s done -> $OUTDIR/rq1_$s.csv"
  fi
done
for s in $STRATEGIES; do
  export COMRADES_AGENT_STRATEGY="$s"
  if has rq2 && [ ! -s "$OUTDIR/multiagent_$s.csv" ]; then
    echo "[$(stamp)] rq2 $s"
    RATES=0.0,0.3,0.6 RETRIES=0,2 OUT="$HERE/$OUTDIR/multiagent_$s.csv.part" \
      bash run_multiagent.sh && mv "$OUTDIR/multiagent_$s.csv.part" "$OUTDIR/multiagent_$s.csv"
  fi
done
if has rq3; then
  for s in $RQ3_STRATEGIES; do
    export COMRADES_AGENT_STRATEGY="$s"
    [ -s "$OUTDIR/rq3_$s.csv" ] && continue
    echo "[$(stamp)] rq3 $s"
    "$PY" run_rq3_parallel.py --mechanisms direct,mcp_only --models "$MODELS" \
      --rates 0.0,0.2,0.4,0.6 --seeds 1 --requests requests.jsonl \
      --out "$OUTDIR/rq3_$s.csv" --shard-dir "$OUTDIR/shards_rq3_$s" --concurrency "$CONC"
  done
fi
echo "[$(stamp)] all done"
