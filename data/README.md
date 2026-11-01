# COMRADES paper data

Copies of the datasets and result files behind the ICSE 2027 COMRADES paper
(`paper/icse_2027.tex`), collected in one place for convenience. These are
duplicates of files under `evaluation/comrades/` (and one external dataset);
the originals remain the source of truth.

## `datasets/` — inputs (paper Table: "The COMRADES dataset")

| File | Rows | Paper element | Source |
|---|---|---|---|
| `requests_curated_50.jsonl` | 50 | Data retrieval, curated (evaluated) | `evaluation/comrades/requests.jsonl` |
| `requests_templated_431.jsonl` | 431 | Data retrieval, templated (released) | `evaluation/comrades/requests_large.jsonl` |
| `harness_tasks_66.jsonl` | 66 | Coding-harness tasks (RQ4) | `evaluation/comrades/harness/tasks_v2.jsonl` |
| `schema_mapping_521.jsonl` | 521 | Schema mapping (multi-hop, gold F1) | external `kg/schema-mapping/data/itopsgraph_dataset.jsonl` |

## `results/` — outputs (behind the paper's tables/figures)

| File | Rows | Paper element | Source |
|---|---|---|---|
| `rq1_success_vs_faultrate.csv` | 600 | RQ1: Table II + Fig. (recorded, 3 seeds × 4 rates × 50) | `results/rq1_recorded_seeds.csv` |
| `rq3_arch_model_inproc.csv` | 1200 | RQ3: Table III, `direct`/`mcp` rows (50-req live, fixed seeding) | `results/rq3_inproc_live_fixed.csv` |
| `rq3_arch_model_a2a.csv` | 600 | RQ3: Table III, `a2a` rows | `results/rq3_a2a.csv` |
| `rq4_harness_claude.csv` | 60 | RQ4: Table (Claude Code) | `harness/results/rq5_claude.csv` |
| `rq4_harness_codex.csv` | 60 | RQ4: Table (Codex) | `harness/results/rq5_codex.csv` |
| `rq4_harness_gemini.csv` | 264 | RQ4: Table (Gemini/antigravity) | `harness/results/rq5_antigravity_v2.csv` |
| `schema_mapping_f1.csv` | 120 | RQ1 schema-map columns (fail-fast/retry F1) | `schema_map/results/schema_map.csv` |
| `control_discipline.csv` | 239 | Control-discipline table (workflow/skill/free) | `schema_map/results/control.csv` |

## Notes

- **RQ4 harness row counts are uneven** (Claude/Codex 60 = 15 tasks × 4 rates;
  Gemini 264 = 66 × 4). The paper reports n=66/rate; if the Claude/Codex panel
  should also be the full 66-task set, those two need a re-run before the numbers
  are apples-to-apples. Copied here as-is.
- RQ1/RQ3 reflect the corrected fault-seeding fix (per-request sub-seed); earlier
  buggy CSVs (`rq1_recorded_seeds.BUGGY.csv`, `rq3.csv`) are NOT copied.
