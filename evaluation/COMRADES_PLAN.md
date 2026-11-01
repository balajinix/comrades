# COMRADES — Run Plan to Clear the Paper's Open Items

**Paper:** COMRADES: Benchmarking Communication Robustness among Data Agents in Enterprise Systems
**Goal:** turn the 9 `\todo`s in `icse_2027.tex` into real numbers, figures, and text.
**Principle:** fault runs use the **canned-replay baseline** (`COMRADES_REPLAY=1`), so the
whole study is reproducible **without VPN/live backends** — non-faulted tool calls
return canned success, faulted ones fail. Live data is *not* needed for the robustness measurements.

---

## What already works (pilots verified, n=3)

| Harness | Entry point | Measures | Status |
|---|---|---|---|
| Single-agent fault sweep | `evaluation/comrades/run_faults.py` | task success, msgs, tokens, redundant-frac vs fault rate | OK (RQ1 pilot) |
| Mechanism comparison | `evaluation/comrades/run_comrades.py` | direct / mcp_only / a2a, no faults | OK |
| Fault injector + replay | `agents/dal_agent/app/faults.py` (`COMRADES_FAULT_*`), `evaluation/comrades/faults.py` | — | OK |
| Redundancy classifier | `evaluation/comrades/redundancy.py` | necessary vs redundant calls | OK |

Working model route: `provider/meta-llama/llama-3-3-70b-instruct`.

---

## The 9 todos -> 4 work items

| # | Paper todo | Work item |
|---|---|---|
| 1 | characterize released request set | **W1 Dataset** |
| 2 | models/versions/seeds/repetitions (RQ setup) | W1 + W3 |
| 3 | RQ1 scale beyond n=3 + CIs | **W2 RQ1 run** |
| 4 | RQ2 scale + success/cost frontier + redundant frac | **W2 RQ2 run** |
| 5 | RQ3 amplification+redundancy by architecture x model | **W3 RQ3 run** |
| 6 | RQ4 decision-guidance table | **W4 Analysis** (no new runs) |
| 7 | architecture figure foregrounding the DAL | **W5 Figure** |
| 8 | conclusion/abstract headline numbers | falls out of W2-W4 |
| 9 | AI-disclosure final wording | trivial (camera-ready) |

---

## W1 — Build & characterize the request set  (~1/2 day)

Curate **~50 natural-language data-access requests**, ~10 per backend family
(cost, APM, ITSM, resilience, infrastructure). They need only be plausible
requests the DAL agent will *attempt* — replay supplies the data — so no live
backends required.

- Write `evaluation/comrades/requests.jsonl`: `{"id","prompt","family"}` per line.
- Seed from the marquee examples already in the agent prompt + the old
  `evaluation/itops_dataset.jsonl` utterances (filter to DAL-routable ones).
- **Characterize** (fills todo #1): a 5-row table of count per family; note the
  collection method (curated from production-style utterances) and that difficulty
  is operationalized as backend-hops.

**Small code change:** add a `--requests <jsonl>` loader to `run_faults.py`.

---

## W2 — RQ1 + RQ2 runs (the headline)  (~1-1.5 days compute)

Fixed model `llama-3-3-70b`. Grid kept tractable; widen later if time allows.

### RQ1 — single-agent failure -> communication  (fills `tab:rq1`)
```bash
# add a --seeds loop to run_faults.py, then:
cd evaluation/comrades
../../agents/dal_agent/.venv/bin/python run_faults.py \
    --requests requests.jsonl --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3
```
- Grid: 50 req x 4 rates x 3 seeds = 600 runs (~2-3 h at ~15 s/run).
- Report mean +/- 95% CI (across seeds) for task success, msgs, tokens, redundant-frac.

### RQ2 — inter-agent fail-fast vs retry  (fills RQ2 frontier)
```bash
RATES=0.0,0.2,0.4,0.6 RETRIES=0,1,2,3 \
```
- Grid: 50 req x 4 rates x 4 retry policies x 3 seeds (loop seeds in the script).
- Report task success and **A2A messages/request** per (rate, retry) — the
  robustness<->cost frontier; redundant fraction from the classifier.

> Output both as tidy CSVs in `results/`; a small `aggregate.py` emits the LaTeX rows.

---

## W3 — RQ3: architecture x model  (~2-3 days; biggest lift)  (fills RQ3 table)

Two axes on top of the fault sweep:
- **Architectures:** in-process (`direct`), `mcp_only`, `a2a`.
- **Models:** small / mid / large — e.g. `granite-3-3-8b` (small), `llama-3-3-70b`
  (mid), and one large (e.g. `mistral-large` or a frontier model via the RITS/
  litellm proxy). **Verify access first** — model availability is the main risk.

**Code change:** make `run_faults.py` honor `--mechanism {direct,mcp_only,a2a}`
(in-process mechanisms self-inject; `a2a` uses the env-gated service injector via

- Trim the grid (e.g. 30 req x 3 rates x 2 seeds) to keep 3x3 cells feasible.
- Report how each architecture/model shifts position on the under-/over-comm frontier.

---

## W4 — RQ4 guidance + headline numbers  (~1/2 day, analysis only)  (fills RQ4 + conclusion/abstract)

No new runs. From W2-W3 produce:
- a **decision table**: at which fault rate retry's success gain outweighs its
  message/cost growth; which guardrails (loop cap, circuit-breaking, caching)
  cut redundant thrash without hurting recovery.
- drop the real headline numbers into the abstract + conclusion (replacing the
  pilot figures).

---

## W5 — Architecture figure  (~1/2 day)  (fills `fig:architecture` todo)

Redraw (draw.io or TikZ) to foreground: user -> orchestrator -> {data agents} with
the **consumption->DAL A2A edge** highlighted, DAL -> `mcp_dal` (MCP) -> the five
backend families, and the **fault-injection point** marked at the tool boundary.
Export `figures/comrades_architecture.png`; update the caption (drop the todo).

---

## Ordering & minimal-viable fallback

1. **W1 dataset** (unblocks everything)
2. **W2 RQ1 + RQ2** <- if only this lands, the paper has its core robustness result with CIs
3. **W5 figure** (independent; do in parallel)
4. **W3 RQ3** (model access permitting)
5. **W4 analysis + text**

**MVP if compute/time is tight:** W1 + W2 + W5 + W4-text gives a complete,
defensible paper (RQ1+RQ2 with CIs, real figure, honest guidance); mark RQ3 as
"single-model; multi-model in extended version."

## Risks
- **Model access (W3):** confirm small + large model routes before planning RQ3; fall back to MVP.
- **Compute:** ~15 s/run x grid. Parallelize across requests (the harness is per-request); cap grid sizes as above.
- **Inter-agent harness restarts the DAL service per rate** — slow; batch by rate, not by request.
