#!/bin/sh
# F1 control: SQL that may answer with its final query, on the 22 large-result
# question-scale pairs. Usage: experiments/run_answer_query.sh <litellm-model> <seed>
set -eu
MODEL="$1"; SEED="$2"
TAG=$(basename "$MODEL" | cut -c1-24)
OUT=findings/answer_query
run() {  # schema, file stem, scale, questions...
  schema="$1"; stem="$2"; sf="$3"; shift 3
  .venv/bin/python experiments/exp_llm_pilot.py --model "$MODEL" --schema "$schema" \
    --sf "$sf" --qids "$@" --facades sql+answer_query --seed "$SEED" --temperature 0 \
    --out "$OUT/${stem}_${TAG}_s${SEED}.csv" >> "$OUT/${stem}_${TAG}_s${SEED}.log" 2>&1
}
run tpch tpch_sf1 1 G1 G2 G3 G4 G5 G6 &
run tpch tpch_sf10 10 G1 G2 G3 G4 G5 G6 &
run tpcds tpcds 1 DS-G1 DS-G2 DS-G3 DS-G4 DS-G5 &
run itbench itbench 35 IT-G1 IT-G2 IT-G3 IT-G4 IT-G5 &
wait
