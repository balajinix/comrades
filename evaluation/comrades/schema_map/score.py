#!/usr/bin/env python3
"""Gold precision/recall/F1 scorer for the COMRADES schema-mapping task family.

Scores a predicted field mapping against the gold `mapping` from the schema-mapping
benchmark (itopsgraph_dataset.jsonl, 521 gold rows). The benchmark is a many-to-many
bipartite field mapping, so we score over the SET of (source_field, target_field)
pairs: precision = correct pairs / predicted pairs, recall = correct / gold, F1 the
harmonic mean. Stdlib-only.

Field ids are fully-qualified `"{product}_{Type}.{field}"` on both sides (the
`mcp_map_reflect` contract guarantees this shape; see MCP_MAP_REFLECT_REQUIREMENTS.md).
"""
from __future__ import annotations

import json
import os


def _pairs(mapping) -> set:
    """Normalize a mapping into a set of (source_field, target_field) pairs.

    Accepts both the gold shape ({src: tgt_str} or {src: [tgt_str, ...]}) and the
    predicted shape ({src: [{"mapProperty": tgt, ...}, ...]}).
    """
    out = set()
    for src, tgts in (mapping or {}).items():
        if isinstance(tgts, str):
            out.add((src, tgts))
        elif isinstance(tgts, dict):
            t = tgts.get("mapProperty")
            if t:
                out.add((src, t))
        elif isinstance(tgts, list):
            for t in tgts:
                if isinstance(t, str):
                    out.add((src, t))
                elif isinstance(t, dict) and t.get("mapProperty"):
                    out.add((src, t["mapProperty"]))
    return {(s, t) for (s, t) in out if s and t}


def prf(pred, gold) -> dict:
    """Precision/recall/F1 of a predicted mapping vs gold, over field-pair sets."""
    P, G = _pairs(pred), _pairs(gold)
    tp = len(P & G)
    precision = tp / len(P) if P else (1.0 if not G else 0.0)
    recall = tp / len(G) if G else (1.0 if not P else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "n_pred": len(P), "n_gold": len(G)}


def aggregate(scores: list[dict]) -> dict:
    """Micro (pooled pairs) and macro (mean per-row) P/R/F1 over many rows."""
    if not scores:
        return {}
    tp = sum(s["tp"] for s in scores)
    npred = sum(s["n_pred"] for s in scores)
    ngold = sum(s["n_gold"] for s in scores)
    micro_p = tp / npred if npred else 0.0
    micro_r = tp / ngold if ngold else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
    macro_f1 = sum(s["f1"] for s in scores) / len(scores)
    return {"n_rows": len(scores), "micro_precision": micro_p, "micro_recall": micro_r,
            "micro_f1": micro_f1, "macro_f1": macro_f1}


def load_dataset(path: str) -> list[dict]:
    """Load the schema-mapping gold dataset (one JSON object per line).

    Returns rows with the fields COMRADES needs: id, utterance, source/target schema,
    expected_target_type, gold mapping, and (from metadata) source/target type names.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            meta = r.get("metadata") or {}
            rows.append({
                "id": r.get("id"),
                "utterance": r.get("utterance"),
                "source_schema": r.get("source_schema"),
                "target_schema": r.get("target_schema"),
                "expected_target_type": r.get("expected_target_type"),
                "source_type": meta.get("source_type"),
                "target_type": meta.get("target_type"),
                "gold": r.get("mapping") or {},
            })
    return rows


if __name__ == "__main__":
    # Self-test: scoring gold against itself must give F1 == 1.0 on every row.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.path.expanduser("~/src/schema-mapping/data/itopsgraph_dataset.jsonl"))
    args = ap.parse_args()
    rows = load_dataset(args.dataset)
    per = [prf(r["gold"], r["gold"]) for r in rows]
    agg = aggregate(per)
    bad = [r["id"] for r, s in zip(rows, per) if s["n_gold"] and s["f1"] != 1.0]
    print(f"loaded {len(rows)} gold rows from {args.dataset}")
    print(f"gold-vs-gold self-score: micro_f1={agg['micro_f1']:.3f} macro_f1={agg['macro_f1']:.3f}")
    print(f"rows with non-trivial gold: {sum(1 for s in per if s['n_gold'])}")
    print(f"self-score failures (should be 0): {len(bad)}")
    # demo: a partial prediction on the first non-empty-gold row
    ex = next((r for r in rows if r["gold"]), None)
    if ex:
        gp = list(_pairs(ex["gold"]))
        partial = {gp[0][0]: [{"mapProperty": gp[0][1]}]} if gp else {}
        print(f"\nexample row {ex['id']}: gold has {len(gp)} pairs; "
              f"scoring a 1-pair partial prediction -> {prf(partial, ex['gold'])}")
