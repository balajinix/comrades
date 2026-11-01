#!/bin/sh
# Main grid (E2): every question x the core interfaces x scales, one model, one seed.
# Usage: experiments/run_grid.sh <litellm-model> <seed> [temperature]
# Four streams run in parallel; each appends to its own CSV, resumable.
set -eu
MODEL="$1"; SEED="$2"; TEMP="${3:-0}"
TAG=$(basename "$MODEL" | cut -c1-24)
OUT=findings/grid
FAC="generic fine_grained fine_grained+ops handles sql+auto sql+schema router skills code code+prov graphql+agg"
run() {  # schema, scales..., file stem
  schema="$1"; shift; stem="$1"; shift
  .venv/bin/python experiments/exp_llm_pilot.py --model "$MODEL" --schema "$schema" \
    --sf "$@" --all-questions --facades $FAC --seed "$SEED" --temperature "$TEMP" \
    --out "$OUT/${stem}_${TAG}_s${SEED}.csv" >> "$OUT/${stem}_${TAG}_s${SEED}.log" 2>&1
}
run tpch tpch_small 0.01 1 &
run tpch tpch_sf10 10 &
run tpcds tpcds 0.01 1 &
run itbench itbench 1 35 &
wait
