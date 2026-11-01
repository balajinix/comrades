# Question-level tests

Per question: mean correctness over the scales (grid) or fault seeds (F6) in the cell; exact two-sided sign test; Holm over the 8 headline tests and over all 31.

| finding | comparison | first higher | second higher | p | Holm p (headline) | Holm p (all) |
|---|---|---|---|---|---|---|
| F1 | Llama TPC-H SF 1: handles vs SQL | 6 | 0 | 0.0312 | 0.125 | 0.71 |
| F1 | Llama TPC-H SF 10: handles vs SQL | 7 | 0 | 0.0156 | 0.0781 | 0.391 |
| F1 | Llama large-result questions at scale: handles vs SQL | 12 | 0 | 0.000488 | 0.00342 | 0.0142 |
| F3 | Llama-3.3-70B all 100 questions: handles vs SQL | 18 | 11 | 0.265 | 0.265 | 1 |
| F3 | Llama-3.3-70B ITBench: handles vs SQL | 4 | 5 | 1 |  | 1 |
| F3 | Llama-3.3-70B TPC-DS: handles vs SQL | 7 | 6 | 1 |  | 1 |
| F3 | Llama-3.3-70B TPC-H: handles vs SQL | 7 | 0 | 0.0156 |  | 0.391 |
| F3 | gpt-oss-120b all 100 questions: handles vs SQL | 20 | 5 | 0.00408 | 0.0245 | 0.11 |
| F3 | gpt-oss-120b ITBench: handles vs SQL | 7 | 2 | 0.18 |  | 1 |
| F3 | gpt-oss-120b TPC-DS: handles vs SQL | 5 | 1 | 0.219 |  | 1 |
| F3 | gpt-oss-120b TPC-H: handles vs SQL | 8 | 2 | 0.109 |  | 1 |
| F3 | Granite-4-H-Small all 100 questions: handles vs SQL | 9 | 33 | 0.000272 | 0.00217 | 0.00815 |
| F3 | Granite-4-H-Small ITBench: handles vs SQL | 2 | 10 | 0.0386 |  | 0.771 |
| F3 | Granite-4-H-Small TPC-DS: handles vs SQL | 3 | 9 | 0.146 |  | 1 |
| F3 | Granite-4-H-Small TPC-H: handles vs SQL | 4 | 14 | 0.0309 |  | 0.71 |
| F2 | Llama-3.3-70B ITBench: sql+auto vs best (router) | 2 | 4 | 0.688 |  | 1 |
| F2 | Llama-3.3-70B TPC-DS: router vs best (sql+auto) | 7 | 7 | 1 |  | 1 |
| F2 | Llama-3.3-70B TPC-H: sql+auto vs best (handles) | 0 | 1 | 1 |  | 1 |
| F2 | Llama-3.3-70B TPC-H: router vs best (handles) | 0 | 1 | 1 |  | 1 |
| F2 | gpt-oss-120b ITBench: sql+auto vs best (handles) | 3 | 4 | 1 |  | 1 |
| F2 | gpt-oss-120b ITBench: router vs best (handles) | 2 | 6 | 0.289 |  | 1 |
| F2 | gpt-oss-120b TPC-DS: router vs best (sql+auto) | 1 | 7 | 0.0703 |  | 1 |
| F2 | gpt-oss-120b TPC-H: router vs best (sql+auto) | 2 | 8 | 0.109 |  | 1 |
| F2 | Granite-4-H-Small ITBench: sql+auto vs best (generic) | 4 | 5 | 1 |  | 1 |
| F2 | Granite-4-H-Small ITBench: router vs best (generic) | 5 | 7 | 0.774 |  | 1 |
| F2 | Granite-4-H-Small TPC-DS: sql+auto vs best (generic) | 4 | 7 | 0.549 |  | 1 |
| F2 | Granite-4-H-Small TPC-DS: router vs best (generic) | 3 | 14 | 0.0127 |  | 0.331 |
| F2 | Granite-4-H-Small TPC-H: sql+auto vs best (generic) | 3 | 24 | 4.92e-05 |  | 0.00153 |
| F2 | Granite-4-H-Small TPC-H: router vs best (generic) | 3 | 17 | 0.00258 |  | 0.0722 |
| F6 | bursty faults: coded vs native (per question, 7 seeds) | 6 | 0 | 0.0312 | 0.125 | 0.71 |
| F6 | bursty faults: coded vs prose (per question, 7 seeds) | 4 | 0 | 0.125 | 0.25 | 1 |

## Runs per model (seed 0)

| model | runs | question-scale-interface cells |
|---|---|---|
| Granite-4-H-Small | 2530 | 2530 |
| Llama-3.3-70B | 2530 | 2530 |
| gpt-oss-120b | 2530 | 2530 |

## Questions a router workflow serves

| dataset | questions | with a coded workflow |
|---|---|---|
| TPC-H | 30 | 13 |
| TPC-DS | 35 | 0 |
| ITBench | 35 | 4 |

## Llama, 22 large-result pairs: median seconds per run

| interface | wall_s | llm_latency_s | db_time_s |
|---|---|---|---|
| generic | 233.25 | 233.26 | 0.0 |
| handles | 3.35 | 3.31 | 0.0 |
| sql+auto | 3.25 | 3.2 | 0.0 |
| router | 3.65 | 3.46 | 0.01 |
| code+prov | 9.6 | 4.55 | 1.68 |
| skills | 236.3 | 236.29 | 0.0 |

## Llama, all runs: median wall-clock seconds per run

| interface | ITBench | TPC-DS | TPC-H | all |
|---|---|---|---|---|
| code | 6.3 | 18.1 | 4.0 | 7.2 |
| code+prov | 5.8 | 12.9 | 5.1 | 8.1 |
| fine_grained | 10.4 | 24.9 | 21.6 | 15.9 |
| fine_grained+ops | 18.2 | 45.8 | 10.1 | 20.1 |
| generic | 4.5 | 4.1 | 4.4 | 4.3 |
| graphql+agg | 7.8 | 22.9 | 9.6 | 9.9 |
| handles | 3.3 | 4.2 | 3.2 | 3.5 |
| router | 3.7 | 6.8 | 3.8 | 4.3 |
| skills | 7.0 | 7.8 | 8.1 | 7.5 |
| sql+auto | 3.8 | 3.8 | 3.4 | 3.7 |
| sql+schema | 9.8 | 6.8 | 6.0 | 7.8 |

## Truncation ablation, gpt-oss-120b (second provider), 22 large-result pairs

| arm | correct | confident wrong | no parseable answer |
|---|---|---|---|
| generic | 4 | 6 | 12 |
| sql+labelled | 4 | 4 | 14 |
| sql+silent | 4 | 8 | 10 |

## Truncation ablation, same provider as the grid (flag arm = grid SQL)

| model | arm | correct | confident wrong |
|---|---|---|---|
| Granite-4-H-Small | generic | 4 | 3 |
| Granite-4-H-Small | sql+labelled | 4 | 5 |
| Granite-4-H-Small | sql+silent | 4 | 4 |
| Llama-3.3-70B | generic | 4 | 7 |
| Llama-3.3-70B | sql+labelled | 4 | 7 |
| Llama-3.3-70B | sql+silent | 4 | 6 |
| gpt-oss-120b | generic | 5 | 5 |
| gpt-oss-120b | sql+labelled | 3 | 4 |
| gpt-oss-120b | sql+silent | 3 | 2 |
