#!/usr/bin/env python3
"""Generate the COMRADES result figures (vector PDF) for the ICSE paper.

Requires matplotlib + pandas. Run from a venv that has them, e.g.:

    agents/dashboard_agent/.venv/bin/python evaluation/comrades/plots.py

Reads the result CSVs under evaluation/comrades/results/ and writes vector PDFs
(plus PNG previews) into paper/figures/. Charts are styled to match the IEEE
two-column paper: serif font, colorblind-safe palette, greyscale-legible line
styles/markers, 95% normal-approx proportion CIs (same as aggregate.py).

aggregate.py stays the stdlib-only LaTeX-table path; this is the chart path.
"""
from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(REPO_ROOT, "paper", "figures")

# Okabe-Ito colorblind-safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}


def _style() -> None:
    """Publication style: serif ~8pt, thin axes, light grid, tight."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


PREVIEW = "/tmp/comrades_figs"  # PNG previews for visual QA; PDFs go to the paper


def _save(fig, name: str) -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(PREVIEW, exist_ok=True)
    pdf = os.path.join(FIGDIR, name + ".pdf")
    # Pass bbox/pad explicitly: the rcParams default was being ignored, clipping the
    # axis labels at the figure edges.
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.08)  # QA only, not in repo
    plt.close(fig)
    print(f"wrote {os.path.relpath(pdf, REPO_ROOT)}  (preview in {PREVIEW})")


def _ci95_prop(p: float, n: int) -> float:
    """95% normal-approx CI half-width on a proportion (matches aggregate.py)."""
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


# --- RQ1: failure -> communication -----------------------------------------
def rq1_robustness() -> bool:
    path = os.path.join(RESULTS, os.getenv("COMRADES_RQ1_CSV", "rq1_full.csv"))
    if not os.path.exists(path):
        print("skip rq1_robustness: no rq1_full.csv")
        return False
    df = pd.read_csv(path)
    df["success"] = df["data_obtained"].astype(str).str.lower().eq("true")
    g = df.groupby("fault_rate")
    rates = sorted(df["fault_rate"].unique())
    succ = g["success"].mean().reindex(rates)
    n = g["success"].size().reindex(rates)
    ci = [_ci95_prop(succ[r], int(n[r])) for r in rates]
    msgs = g["messages"].mean().reindex(rates)
    red = g["redundant_frac"].mean().reindex(rates)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(3.4, 3.1),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    # top: task success with CI band
    ax1.fill_between(rates, succ.values - ci, succ.values + ci,
                     color=OI["blue"], alpha=0.18, linewidth=0)
    ax1.plot(rates, succ.values, color=OI["blue"], marker="o", label="Task success")
    ax1.set_ylabel("Task success")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower left", frameon=False)

    # bottom: communication stays flat; redundancy near zero (twin axis)
    ax2.plot(rates, msgs.values, color=OI["vermillion"], marker="s",
             linestyle="-", label="Mean messages")
    ax2.set_ylabel("Mean messages", color=OI["vermillion"])
    ax2.tick_params(axis="y", labelcolor=OI["vermillion"])
    ax2.set_ylim(0, max(msgs.values) * 1.3)
    ax2.grid(True, axis="y")

    ax2b = ax2.twinx()
    ax2b.spines["top"].set_visible(False)
    ax2b.plot(rates, red.values, color=OI["green"], marker="^",
              linestyle="--", label="Redundant frac.")
    ax2b.set_ylabel("Redundant frac.", color=OI["green"])
    ax2b.tick_params(axis="y", labelcolor=OI["green"])
    ax2b.set_ylim(0, max(0.05, max(red.values) * 1.5))

    ax2.set_xlabel("Backend fault rate")
    ax2.set_xticks(rates)
    # merged legend for bottom panel
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=False)

    _save(fig, "rq1_robustness")
    return True


def rq1_by_family() -> bool:
    path = os.path.join(RESULTS, "rq1_full.csv")
    if not os.path.exists(path) or "family" not in pd.read_csv(path, nrows=1).columns:
        print("skip rq1_by_family: no family column")
        return False
    df = pd.read_csv(path)
    df["success"] = df["data_obtained"].astype(str).str.lower().eq("true")
    rates = sorted(df["fault_rate"].unique())
    markers = ["o", "s", "^", "D", "v"]
    colors = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"], OI["purple"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, (fam, sub) in enumerate(sorted(df.groupby("family"))):
        s = sub.groupby("fault_rate")["success"].mean().reindex(rates)
        ax.plot(rates, s.values, marker=markers[i % 5], color=colors[i % 5],
                linestyle=["-", "--", "-.", ":", "-"][i % 5], label=fam)
    ax.set_xlabel("Backend fault rate")
    ax.set_ylabel("Task success")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rates)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    _save(fig, "rq1_by_family")
    return True


# --- RQ2: fail-fast vs retry (inter-agent) ---------------------------------
def rq2_amplification() -> bool:
    path = os.path.join(RESULTS, "multiagent.csv")
    if not os.path.exists(path):
        print("skip rq2_amplification: no multiagent.csv")
        return False
    df = pd.read_csv(path)
    rates = sorted(df["fault_rate"].unique())
    policies = sorted(df["retries"].unique())  # e.g. [0, 2]
    labels = {p: ("fail-fast" if p == 0 else f"retry (r={p})") for p in policies}
    colors = {policies[0]: OI["skyblue"], policies[-1]: OI["vermillion"]}
    hatches = {policies[0]: "", policies[-1]: "//"}

    width = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = range(len(rates))
    for j, pol in enumerate(policies):
        calls, succ = [], []
        for r in rates:
            cell = df[(df["fault_rate"] == r) & (df["retries"] == pol)]
            calls.append(cell["a2a_calls"].mean())
            succ.append(cell["success"].mean())
        offs = [xi + (j - (len(policies) - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, calls, width, label=labels[pol], color=colors[pol],
                      edgecolor="black", linewidth=0.5, hatch=hatches[pol])
        for b, s in zip(bars, succ):
            ax.annotate(f"{s*100:.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=6.5, xytext=(0, 1),
                        textcoords="offset points")
    ax.set_xlabel("Backend fault rate")
    ax.set_ylabel("A2A calls / request")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(r) for r in rates])
    ax.set_ylim(0, max(3.2, df["a2a_calls"].max() * 1.25))
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", frameon=False, title="(% = task success)",
              title_fontsize=6.5)
    _save(fig, "rq2_amplification")
    return True


# --- RQ2 by backend family: who drives under/over-communication ------------
def rq2_by_family() -> bool:
    """Per-backend-family fail-fast vs retry at peak stress (reviewer Q2).

    Two stacked panels sharing x = backend family, at the highest fault rate:
    top = communication (A2A calls/request) — shows where retry over-communicates;
    bottom = task success with 95% CIs — shows what that communication buys. A
    family with high retry comms but no success gain (e.g. ITSM) is over-
    communication waste; one where retry lifts success (infra/resilience) is
    recovery. fail-fast holds comms flat at ~1.0 everywhere (under-communication).
    """
    path = os.path.join(RESULTS, "multiagent.csv")
    if not os.path.exists(path):
        print("skip rq2_by_family: no multiagent.csv")
        return False
    df = pd.read_csv(path)
    rate = max(df["fault_rate"].unique())  # peak stress
    sub = df[df["fault_rate"] == rate]
    fams = sorted(sub["family"].unique())
    pols = sorted(sub["retries"].unique())  # [0, 2]
    labels = {pols[0]: "fail-fast", pols[-1]: f"retry (r={pols[-1]})"}
    colors = {pols[0]: OI["skyblue"], pols[-1]: OI["vermillion"]}
    hatches = {pols[0]: "", pols[-1]: "//"}

    def cell(fam, pol, col):
        c = sub[(sub["family"] == fam) & (sub["retries"] == pol)]
        return c[col].mean(), len(c)

    width = 0.38
    x = range(len(fams))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(3.4, 2.5),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    for j, pol in enumerate(pols):
        offs = [xi + (j - (len(pols) - 1) / 2) * width for xi in x]
        calls = [cell(f, pol, "a2a_calls")[0] for f in fams]
        ax1.bar(offs, calls, width, label=labels[pol], color=colors[pol],
                edgecolor="black", linewidth=0.5, hatch=hatches[pol])
        succ, ns = zip(*[cell(f, pol, "success") for f in fams])
        ci = [_ci95_prop(s, n) for s, n in zip(succ, ns)]
        ax2.bar(offs, succ, width, color=colors[pol], edgecolor="black",
                linewidth=0.5, hatch=hatches[pol],
                yerr=ci, error_kw={"elinewidth": 0.6, "capsize": 1.5})

    ax1.set_ylabel("A2A calls / req")
    ax1.set_ylim(0, max(3.4, sub["a2a_calls"].max() * 1.15))
    ax1.axhline(1.0, color=OI["skyblue"], lw=0.6, ls=":", zorder=0)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper left", frameon=False, ncol=2)
    ax2.set_ylabel("Task success")
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, axis="y")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(fams)
    ax2.set_xlabel(f"Backend family (fault rate {rate})")
    _save(fig, "rq2_by_family")
    return True


# --- stubs: fill automatically when the data lands -------------------------
def rq3_arch_model() -> bool:
    # Grouped bars: amplification/comms by architecture (in-process/MCP/A2A) x model.
    # Needs the RQ3 sweep output (task #138). No-op until then.
    print("skip rq3_arch_model: RQ3 sweep data not present yet")
    return False


def rq5_harness() -> bool:
    # Per-harness comms vs fault rate. Needs RQ5 runs (task #135, on hold). No-op.
    print("skip rq5_harness: RQ5 harness data not present yet")
    return False


def main() -> None:
    _style()
    rq1_robustness()
    rq2_amplification()
    rq2_by_family()
    rq1_by_family()
    rq3_arch_model()
    rq5_harness()


if __name__ == "__main__":
    main()
