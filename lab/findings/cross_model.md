# Cross-model grid (seed 0, temperature 0)

Runs: 7590; per model: gpt-oss-120b 2530, Granite-4-H-Small 2530, Llama-3.3-70B 2530

## 1. Accuracy (scales pooled)

| interface | Llama-3.3-70B ITBench | Llama-3.3-70B TPC-DS | Llama-3.3-70B TPC-H | gpt-oss-120b ITBench | gpt-oss-120b TPC-DS | gpt-oss-120b TPC-H | Granite-4-H-Small ITBench | Granite-4-H-Small TPC-DS | Granite-4-H-Small TPC-H |
|---|---|---|---|---|---|---|---|---|---|
| generic | 0.73 | 0.61 | 0.83 | 0.71 | 0.83 | 0.84 | 0.53 | 0.56 | 0.79 |
| handles | 0.76 | 0.6 | 0.98 | 0.86 | 0.89 | 0.94 | 0.36 | 0.41 | 0.48 |
| sql+auto | 0.73 | 0.63 | 0.97 | 0.83 | 0.9 | 0.96 | 0.47 | 0.47 | 0.29 |
| router | 0.77 | 0.59 | 0.97 | 0.79 | 0.81 | 0.84 | 0.44 | 0.29 | 0.44 |
| sql+schema | 0.39 | 0.54 | 0.82 | 0.54 | 0.74 | 0.74 | 0.26 | 0.33 | 0.69 |
| skills | 0.56 | 0.53 | 0.72 | 0.54 | 0.56 | 0.48 | 0.16 | 0.16 | 0.28 |
| code | 0.51 | 0.36 | 0.9 | 0.67 | 0.76 | 0.91 | 0.19 | 0.29 | 0.41 |
| code+prov | 0.5 | 0.53 | 0.96 | 0.66 | 0.81 | 0.96 | 0.33 | 0.27 | 0.52 |
| fine_grained | 0.27 | 0.24 | 0.32 | 0.34 | 0.21 | 0.22 | 0.14 | 0.17 | 0.28 |
| fine_grained+ops | 0.34 | 0.19 | 0.63 | 0.5 | 0.53 | 0.71 | 0.21 | 0.14 | 0.4 |
| graphql+agg | 0.37 | 0.29 | 0.43 | 0.61 | 0.46 | 0.61 | 0.09 | 0.07 | 0.14 |

## 2. Large-result questions at scale (correct/n, median tokens)

| interface | Llama-3.3-70B | gpt-oss-120b | Granite-4-H-Small |
|---|---|---|---|
| generic | 4/22, 85.3k | 5/22, 76.1k | 4/22, 81.3k |
| handles | 21/22, 2.0k | 22/22, 2.5k | 3/22, 5.3k |
| sql+auto | 21/22, 2.0k | 21/22, 2.8k | 4/22, 2.4k |
| router | 22/22, 3.2k | 19/22, 3.7k | 5/22, 3.1k |
| sql+schema | 3/22, 97.1k | 4/22, 97.8k | 3/22, 75.9k |
| skills | 4/22, 85.9k | 3/22, 55.9k | 1/22, 18.5k |
| code | 15/22, 1.6k | 18/22, 1.9k | 7/22, 2.6k |
| code+prov | 15/22, 2.1k | 18/22, 2.0k | 10/22, 2.7k |
| fine_grained | 1/22, 65.3k | 2/22, 138.9k | 1/22, 106.8k |
| fine_grained+ops | 3/22, 121.8k | 3/22, 100.7k | 0/22, 64.2k |
| graphql+agg | 3/22, 96.5k | 3/22, 18.1k | 0/22, 25.1k |

## 3. Other questions (all scales): correct/n

| model | generic | handles | sql+auto | router |
|---|---|---|---|---|
| gpt-oss-120b | 167/192 | 169/192 | 171/192 | 155/192 |
| Granite-4-H-Small | 131/192 | 86/192 | 83/192 | 82/192 |
| Llama-3.3-70B | 153/192 | 146/192 | 146/192 | 144/192 |

## 4. Router gap to the best interface, and the oracle, per column

| model | column | best | router | router gap | oracle |
|---|---|---|---|---|---|
| gpt-oss-120b | ITBench 1 | handles 0.89 | 0.74 | 0.14 | 0.94 |
| gpt-oss-120b | ITBench 35 | handles 0.83 | 0.83 | 0.0 | 0.94 |
| gpt-oss-120b | TPC-DS 0.01 | handles 0.91 | 0.8 | 0.11 | 0.97 |
| gpt-oss-120b | TPC-DS 1 | sql+auto 0.89 | 0.83 | 0.06 | 0.91 |
| gpt-oss-120b | TPC-H 0.01 | code+prov 1.00 | 0.87 | 0.13 | 1.0 |
| gpt-oss-120b | TPC-H 1 | handles 0.97 | 0.8 | 0.17 | 1.0 |
| gpt-oss-120b | TPC-H 10 | handles 0.97 | 0.87 | 0.1 | 1.0 |
| Granite-4-H-Small | ITBench 1 | generic 0.54 | 0.46 | 0.09 | 0.74 |
| Granite-4-H-Small | ITBench 35 | generic 0.51 | 0.43 | 0.09 | 0.69 |
| Granite-4-H-Small | TPC-DS 0.01 | generic 0.69 | 0.31 | 0.37 | 0.83 |
| Granite-4-H-Small | TPC-DS 1 | generic 0.43 | 0.26 | 0.17 | 0.8 |
| Granite-4-H-Small | TPC-H 0.01 | generic 0.93 | 0.43 | 0.5 | 1.0 |
| Granite-4-H-Small | TPC-H 1 | generic 0.73 | 0.47 | 0.27 | 0.97 |
| Granite-4-H-Small | TPC-H 10 | generic 0.70 | 0.43 | 0.27 | 0.9 |
| Llama-3.3-70B | ITBench 1 | handles 0.77 | 0.77 | 0.0 | 0.91 |
| Llama-3.3-70B | ITBench 35 | router 0.77 | 0.77 | 0.0 | 0.94 |
| Llama-3.3-70B | TPC-DS 0.01 | generic 0.69 | 0.63 | 0.06 | 0.91 |
| Llama-3.3-70B | TPC-DS 1 | sql+auto 0.63 | 0.54 | 0.09 | 0.83 |
| Llama-3.3-70B | TPC-H 0.01 | generic 1.00 | 0.97 | 0.03 | 1.0 |
| Llama-3.3-70B | TPC-H 1 | code+prov 0.97 | 0.97 | 0.0 | 1.0 |
| Llama-3.3-70B | TPC-H 10 | code+prov 0.97 | 0.97 | 0.0 | 0.97 |

## 5. handles vs SQL, paired exact McNemar

| model | dataset | handles only | SQL only | p |
|---|---|---|---|---|
| gpt-oss-120b | ITBench | 12 | 2 | 0.0129 |
| gpt-oss-120b | TPC-DS | 5 | 1 | 0.219 |
| gpt-oss-120b | TPC-H | 13 | 4 | 0.049 |
| Granite-4-H-Small | ITBench | 2 | 14 | 0.00418 |
| Granite-4-H-Small | TPC-DS | 3 | 13 | 0.0213 |
| Granite-4-H-Small | TPC-H | 8 | 36 | 2.54e-05 |
| Llama-3.3-70B | ITBench | 8 | 6 | 0.791 |
| Llama-3.3-70B | TPC-DS | 8 | 9 | 1 |
| Llama-3.3-70B | TPC-H | 13 | 0 | 0.000244 |

## 6. gpt-oss: serving SHOW TABLES / DESCRIBE in on-demand schema SQL

Before excludes runs that ended on a provider rate limit.

| dataset | interface | before | after |
|---|---|---|---|
| itbench | router | 0.77 (n=69) | 0.79 (n=70) |
| itbench | sql+schema | 0.61 (n=69) | 0.54 (n=70) |
| tpcds | router | 0.43 (n=70) | 0.81 (n=70) |
| tpcds | sql+schema | 0.67 (n=69) | 0.74 (n=70) |
| tpch_sf10 | router | 0.83 (n=29) | 0.87 (n=30) |
| tpch_sf10 | sql+schema | 0.54 (n=28) | 0.63 (n=30) |
| tpch_small | router | 0.85 (n=59) | 0.83 (n=60) |
| tpch_small | sql+schema | 0.66 (n=59) | 0.80 (n=60) |
