"""COMRADES annotation — Streamlit app.

Human-in-the-loop verification of machine-proposed Dataset A candidates, with
explicit feedback on exactly what each Accept/Reject records, and durable
save-to-disk on every action (no lost work, no manual export needed).

Run:
    cd evaluation/comrades/annotation
    streamlit run annotate_app.py
    # or point at a different candidates file:
    CANDIDATES=candidates/dataset_A.jsonl streamlit run annotate_app.py

Writes:
    <candidates>.annotations.json   incremental store (id -> record), resume-safe
    <candidates>.gold.jsonl         accepted rows (verified=true) — the dataset
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
CANDIDATES = Path(os.getenv("CANDIDATES", HERE / "candidates" / "dataset_A.jsonl"))
# v151 enriched catalog (151 tools, reliable backend/family from the [Backend] title
# prefix). Override with CATALOG_ENRICHED. Falls back to the old 105-tool file.
ENRICHED_PATH = Path(os.getenv("CATALOG_ENRICHED", HERE / "catalog_mcp_dal_v151.enriched.json"))
if not ENRICHED_PATH.exists():
    ENRICHED_PATH = HERE / "catalog_mcp_dal.enriched.json"
FAMILIES = ["cost", "apm", "itsm", "resilience", "infra", "other"]
DIFFICULTY = ["", "easy", "medium", "hard"]
REASONING = ["", "lookup", "filter", "aggregate", "compare", "trace", "rank", "other"]
EDIT_FIELDS = ["prompt", "gold_tool", "gold_args", "result_assertion",
               "family", "difficulty", "hops", "reasoning_type", "notes"]

st.set_page_config(page_title="COMRADES annotation", layout="wide")


# ---------- persistence ----------
def store_path() -> Path:
    return CANDIDATES.with_suffix(CANDIDATES.suffix + ".annotations.json")


def gold_path() -> Path:
    return CANDIDATES.with_name(CANDIDATES.stem + ".gold.jsonl")


@st.cache_data
def load_enriched(path: str, mtime: float) -> dict:
    """Tool reference (description, backend, params, API hints) keyed by tool name."""
    p = Path(path)
    if not p.exists():
        return {}
    return {t["name"]: t for t in json.loads(p.read_text())}


@st.cache_data
def load_candidates(path: str, mtime: float) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        o.setdefault("id", f"row-{i}")
        rows.append(o)
    return rows


def load_store() -> dict:
    p = store_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_store(store: dict) -> None:
    store_path().write_text(json.dumps(store, indent=1))


def write_gold(store: dict) -> int:
    accepted = [r for r in store.values() if r.get("__status") == "accept"]
    with gold_path().open("w") as fh:
        for r in accepted:
            o = {k: v for k, v in r.items() if k != "__status"}
            o["verified"] = True
            fh.write(json.dumps(o) + "\n")
    return len(accepted)


# ---------- state ----------
if not CANDIDATES.exists():
    st.error(f"Candidates file not found: {CANDIDATES}\n\nRun gen_candidates.py first.")
    st.stop()

rows = load_candidates(str(CANDIDATES), CANDIDATES.stat().st_mtime)
by_id = {r["id"]: r for r in rows}
if "store" not in st.session_state:
    st.session_state.store = load_store()
if "idx" not in st.session_state:
    # start at first un-annotated row
    st.session_state.idx = next((i for i, r in enumerate(rows)
                                 if r["id"] not in st.session_state.store), 0)
if "annotator" not in st.session_state:
    st.session_state.annotator = ""

store = st.session_state.store
n = len(rows)


def record(status: str, vals: dict) -> dict:
    cand = rows[st.session_state.idx]
    # Normalize the assertion: drop the placeholder, tag the gold level so we can
    # accept tool-level rows now and fill assertions later.
    a = (vals.get("result_assertion") or "").strip()
    if "<EDIT" in a:
        a = ""
    vals = {**vals, "result_assertion": a,
            "gold_level": "tool+assertion" if a else "tool"}
    rec = {**cand, **vals, "__status": status,
           "annotator": st.session_state.annotator or "anon",
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    store[rec["id"]] = rec
    save_store(store)
    write_gold(store)
    st.session_state.last = rec
    return rec


def advance():
    nxt = next((i for i, r in enumerate(rows)
                if i > st.session_state.idx and r["id"] not in store), None)
    st.session_state.idx = nxt if nxt is not None else min(st.session_state.idx + 1, n - 1)


# ---------- sidebar: progress + balance + export ----------
with st.sidebar:
    st.header("COMRADES annotation")
    st.session_state.annotator = st.text_input("Annotator (initials)",
                                                st.session_state.annotator or "anon")
    acc = [r for r in store.values() if r.get("__status") == "accept"]
    rej = sum(1 for r in store.values() if r.get("__status") == "reject")
    skp = sum(1 for r in store.values() if r.get("__status") == "skip")
    done = len(store)
    st.progress(done / n if n else 0, text=f"{done}/{n} reviewed")
    c1, c2, c3 = st.columns(3)
    c1.metric("✓ accepted", len(acc))
    c2.metric("✗ rejected", rej)
    c3.metric("skipped", skp)

    st.caption("Accepted by family (target balance). Read-tool pool is now broad — "
               "itsm 30, apm 27, resilience 20, infra 13, cost 7 — so aim for a balanced "
               "per-family target rather than skipping ITSM surplus.")
    target = st.slider("per-family target", 5, 100, 25, 5)
    fam_counts = {f: 0 for f in FAMILIES}
    for r in acc:
        fam_counts[r.get("family", "other")] = fam_counts.get(r.get("family", "other"), 0) + 1
    for f in FAMILIES:
        st.progress(min(fam_counts[f] / target, 1.0), text=f"{f}: {fam_counts[f]}/{target}")

    with_assert = sum(1 for r in acc if r.get("gold_level") == "tool+assertion")
    st.caption(f"Of {len(acc)} accepted, **{with_assert}** have a result_assertion "
               f"(correctness-gradable); {len(acc) - with_assert} are tool-level only.")

    st.divider()
    st.caption(f"Gold → `{gold_path().name}` (rewritten on every save)")
    st.caption(f"Store → `{store_path().name}`")
    if st.button("Jump to first un-annotated"):
        st.session_state.idx = next((i for i, r in enumerate(rows)
                                     if r["id"] not in store), st.session_state.idx)
        st.rerun()


# ---------- feedback banner: what was just recorded ----------
if "last" in st.session_state:
    last = st.session_state.last
    icon = {"accept": "✅", "reject": "🗑️", "skip": "⏭️"}.get(last["__status"], "•")
    st.success(f"{icon} Recorded **{last['id']}** as **{last['__status'].upper()}** "
               f"by *{last['annotator']}* at {last['ts']} → saved to `{gold_path().name}`")
    with st.expander("Exactly what was written", expanded=False):
        st.json({k: v for k, v in last.items() if k != "__status"})


# ---------- main: current candidate ----------
idx = st.session_state.idx
cand = rows[idx]
prior = store.get(cand["id"])
cur = {**cand, **(prior or {})}

top = st.columns([3, 1, 1, 1])
top[0].subheader(f"Row {idx+1} / {n} — `{cand['id']}`")
top[1].caption(f"source: {cand.get('source','?')}")
status_now = (prior or {}).get("__status")
top[2].caption(f"status: **{status_now or 'pending'}**")
top[3].caption(f"orig tool: `{cand.get('gold_tool','')}`")

# Read-only tool reference for the current gold tool(s) — backend, description,
# params (+ API hints if present) so the annotator can write a natural prompt and a
# correct result_assertion. Supports single-hop (gold_tool) and multi-hop (gold_tools).
_ENRICHED = load_enriched(str(ENRICHED_PATH),
                          ENRICHED_PATH.stat().st_mtime if ENRICHED_PATH.exists() else 0.0)
_tool_names = cur.get("gold_tools") or ([cur["gold_tool"]] if cur.get("gold_tool") else [])
_refs = [_ENRICHED[t] for t in _tool_names if t in _ENRICHED]
if _refs:
    if len(_refs) > 1:
        st.markdown(f"**Multi-hop task — {len(_refs)} tools:** "
                    + " → ".join(f"`{r['name']}`" for r in _refs))
    for ref in _refs:
        with st.expander(f"📖 Tool reference — `{ref['name']}`  ·  {ref.get('backend','?')}",
                         expanded=(len(_refs) == 1)):
            if ref.get("description"):
                st.markdown(f"**What it does:** {ref['description']}")
            if ref.get("api_hints"):
                st.markdown("**API endpoints:** " + "  ".join(f"`{h}`" for h in ref["api_hints"]))
            if ref.get("params"):
                st.markdown("**Parameters:**")
                st.dataframe(
                    [{"param": p["name"], "type": p["type"], "req": "✓" if p["required"] else "",
                      "description": p.get("description", "")} for p in ref["params"]],
                    hide_index=True, use_container_width=True)
            st.caption(f"suggested family: **{ref.get('family_guess','?')}**"
                       + (f" · source: `{ref['source_file']}`" if ref.get("source_file") else ""))
    _missing = [t for t in _tool_names if t not in _ENRICHED]
    if _missing:
        st.caption("⚠️ no reference for: " + ", ".join(f"`{m}`" for m in _missing))
elif cand.get("tool_desc"):
    st.caption(f"📖 tool: _{cand['tool_desc']}_")

vals: dict = {}
vals["prompt"] = st.text_area("prompt — natural operator question, exactly ONE correct tool",
                              cur.get("prompt", ""), height=80, key=f"prompt_{cand['id']}")
cc = st.columns(2)
vals["gold_tool"] = cc[0].text_input("gold_tool", cur.get("gold_tool", ""), key=f"gt_{cand['id']}")
fam_default = cur.get("family", "other")
vals["family"] = cc[1].selectbox("family", FAMILIES,
                                 index=FAMILIES.index(fam_default) if fam_default in FAMILIES else 5,
                                 key=f"fam_{cand['id']}")

ga_raw = st.text_area("gold_args (JSON)", json.dumps(cur.get("gold_args", {}), indent=1),
                      height=80, key=f"ga_{cand['id']}")
ga_ok, ga_val = True, {}
try:
    ga_val = json.loads(ga_raw or "{}")
except json.JSONDecodeError as e:
    ga_ok = False
    st.error(f"gold_args is not valid JSON: {e}")
vals["gold_args"] = ga_val

vals["result_assertion"] = st.text_area(
    "result_assertion — Python predicate over `result`, true iff the task was answered "
    "(e.g. len(result.get('results',[]))>=1)",
    cur.get("result_assertion", ""), height=70, key=f"ra_{cand['id']}")
with st.expander("assertion hints by tool type"):
    st.markdown(
        "- **query rows** (`execute_infra_iql`): `len(result.get('results',[]))>=1`\n"
        "- **schema/list tools** (`list_infra_node_schemas`, `list_finops_vendors`, "
        "`list_finops_cost_measures`): `len(result)>=1` (a non-empty list of type defs / metadata)\n"
        "- **neighbor/path** (`get_infra_one_hop_neighbors`): fill `gold_args` with a real "
        "node id (discover one via a live `tools/call`), then `len(result.get('neighbors',[]))>=1`\n"
        "- **Resilience** (`list_resilience_applications`, `list_resilience_compliance_risks`): non-empty record "
        "list; (`get_resilience_cve_details`): `result.get('cve_id')` present\n"
        "- generic success: `result.get('status')=='ok'` or `result.get('status')!='error'`")

m = st.columns(3)
dl = cur.get("difficulty", "")
vals["difficulty"] = m[0].selectbox("difficulty", DIFFICULTY,
                                    index=DIFFICULTY.index(dl) if dl in DIFFICULTY else 0,
                                    key=f"diff_{cand['id']}")
vals["hops"] = m[1].number_input("hops", 1, 10, int(cur.get("hops", 1) or 1), key=f"hops_{cand['id']}")
rt = cur.get("reasoning_type", "")
vals["reasoning_type"] = m[2].selectbox("reasoning_type", REASONING,
                                        index=REASONING.index(rt) if rt in REASONING else 0,
                                        key=f"rt_{cand['id']}")
vals["notes"] = st.text_input("notes (optional)", cur.get("notes", ""), key=f"nt_{cand['id']}")

# live preview of the record that will be written
with st.expander("Preview record to be saved", expanded=False):
    st.json({**cand, **vals})

st.divider()
# Accept is blocked ONLY by a real correctness error (invalid gold_args JSON).
# result_assertion is encouraged but OPTIONAL — a tool-level gold row is useful,
# so accept now and upgrade to correctness-gradable later.
accept_disabled = not ga_ok
has_assertion = bool(vals["result_assertion"].strip()) and "<EDIT" not in vals["result_assertion"]
b = st.columns([1, 1, 1, 1, 1])
if b[0].button("✅ Accept & next", type="primary", disabled=accept_disabled, use_container_width=True):
    record("accept", vals); advance(); st.rerun()
if b[1].button("🗑️ Reject", use_container_width=True):
    record("reject", vals); advance(); st.rerun()
if b[2].button("⏭️ Skip", use_container_width=True):
    record("skip", vals)
    st.session_state.idx = min(idx + 1, n - 1); st.rerun()
if b[3].button("◀ Prev", use_container_width=True):
    st.session_state.idx = max(0, idx - 1); st.rerun()
if b[4].button("Next ▶", use_container_width=True):
    st.session_state.idx = min(n - 1, idx + 1); st.rerun()

if accept_disabled:
    st.error("Accept is disabled because **gold_args is not valid JSON** — fix it above.")
elif has_assertion:
    st.caption("✅ This row will be saved as **tool+assertion** gold (correctness-gradable).")
else:
    st.caption("➡️ No `result_assertion` yet — Accept saves this as **tool-level gold** "
               "(`gold_level=tool`). Add a predicate over `result` to upgrade it, or accept now "
               "and fill assertions in a later pass.")
