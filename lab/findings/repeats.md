# Repeat runs: Llama-3.3-70B

Seed 0 at temperature 0; seeds 1-2 at temperature 0.7. Runs per seed: 0: 2530, 1: 2530, 2: 2530

## 1. Spread across runs, per cell

- cells: 77
- median range: 0.07; max: 0.23 (sql+schema itbench 1.0)
- share of cells with range <= 0.10: 0.71

## 2. Accuracy, mean of 3 runs

| interface | itbench 1 | itbench 35 | tpcds 0.01 | tpcds 1 | tpch 0.01 | tpch 1 | tpch 10 |
|---|---|---|---|---|---|---|---|
| generic | 0.69 | 0.69 | 0.68 | 0.47 | 0.98 | 0.79 | 0.76 |
| handles | 0.74 | 0.72 | 0.57 | 0.51 | 1.0 | 0.99 | 0.99 |
| sql+auto | 0.7 | 0.69 | 0.6 | 0.65 | 0.96 | 0.98 | 0.98 |
| router | 0.78 | 0.76 | 0.57 | 0.52 | 0.94 | 0.94 | 0.96 |
| sql+schema | 0.42 | 0.33 | 0.56 | 0.42 | 0.91 | 0.74 | 0.71 |
| skills | 0.51 | 0.49 | 0.57 | 0.5 | 0.87 | 0.67 | 0.62 |
| code | 0.5 | 0.5 | 0.4 | 0.33 | 0.94 | 0.91 | 0.89 |
| code+prov | 0.52 | 0.5 | 0.6 | 0.47 | 0.91 | 0.92 | 0.94 |
| fine_grained | 0.24 | 0.22 | 0.26 | 0.22 | 0.44 | 0.26 | 0.26 |
| fine_grained+ops | 0.39 | 0.34 | 0.22 | 0.14 | 0.72 | 0.54 | 0.52 |
| graphql+agg | 0.35 | 0.39 | 0.3 | 0.25 | 0.59 | 0.34 | 0.37 |

## 3a. Large-result questions at scale: correct of 22

| interface | run 0 | run 1 | run 2 |
|---|---|---|---|
| generic | 4 | 4 | 3 |
| handles | 21 | 21 | 21 |
| sql+auto | 21 | 20 | 21 |
| router | 22 | 22 | 22 |
| sql+schema | 3 | 3 | 3 |
| skills | 4 | 3 | 3 |
| code | 15 | 13 | 15 |
| code+prov | 15 | 13 | 14 |
| fine_grained | 1 | 0 | 0 |
| fine_grained+ops | 3 | 2 | 2 |
| graphql+agg | 3 | 3 | 2 |

## 3b. handles vs SQL on TPC-H, paired exact McNemar

| TPC-H SF | run | handles only | SQL only | p |
|---|---|---|---|---|
| 1 | 0 | 6 | 0 | 0.0312 |
| 1 | 1 | 6 | 0 | 0.0312 |
| 1 | 2 | 6 | 0 | 0.0312 |
| 10 | 0 | 7 | 0 | 0.0156 |
| 10 | 1 | 7 | 0 | 0.0156 |
| 10 | 2 | 7 | 0 | 0.0156 |

## 3c. Router gap to the best interface

| column | gap 0 | best 0 | gap 1 | best 1 | gap 2 | best 2 | gap mean | best mean |
|---|---|---|---|---|---|---|---|---|
| itbench 1 | 0.0 | handles | 0.0 | router | 0.0 | router | 0.0 | router |
| itbench 35 | 0.0 | router | 0.0 | router | 0.0 | router | 0.0 | router |
| tpcds 0.01 | 0.06 | generic | 0.17 | generic | 0.14 | code+prov | 0.1 | generic |
| tpcds 1 | 0.09 | sql+auto | 0.06 | sql+auto | 0.23 | sql+auto | 0.12 | sql+auto |
| tpch 0.01 | 0.03 | generic | 0.07 | handles | 0.07 | handles | 0.06 | handles |
| tpch 1 | 0.0 | code+prov | 0.03 | handles | 0.1 | handles | 0.04 | handles |
| tpch 10 | 0.0 | code+prov | 0.07 | handles | 0.03 | handles | 0.03 | handles |

Max gap per run: 0: 0.09, 1: 0.17, 2: 0.23, mean: 0.12

## 3d. code vs code+prov on TPC-DS

| run | interface | silent failures | accuracy SF 1 |
|---|---|---|---|
| 0 | code | 0.56 | 0.31 |
| 0 | code+prov | 0.36 | 0.46 |
| 1 | code | 0.51 | 0.31 |
| 1 | code+prov | 0.46 | 0.49 |
| 2 | code | 0.57 | 0.37 |
| 2 | code+prov | 0.33 | 0.46 |
