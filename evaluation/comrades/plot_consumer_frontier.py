"""Page-1 figure: how each data consumer's calls and success move under faults.

Every row is one consumer of the COMRADES harness. Panel (a) shows calls per task
to the data layer with no faults (hollow) and at a 0.6 fault rate (filled), and
panel (b) the success each consumer reaches at those two rates, each scored on its
own task family. Run from the repo root:

    lab/.venv/bin/python evaluation/comrades/plot_consumer_frontier.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "evaluation/comrades/results"
SM = ROOT / "evaluation/comrades/schema_map/results"
HAR = ROOT / "evaluation/comrades/harness/results"
OUT = ROOT / "paper/figures/consumer_frontier.pdf"

FEWER, MORE, FLAT = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"


def _rq1(path: Path) -> tuple[float, float, float, float]:
    d = pd.read_csv(path)
    g = d.groupby("fault_rate")
    # harness retries are invisible to the agent, so count them when recorded
    calls = g.attempts.mean() if "attempts" in d.columns else g.tool_steps.mean()
    return (
        calls[0.0],
        calls[0.6],
        g.data_obtained.mean()[0.0],
        g.data_obtained.mean()[0.6],
    )


def rows() -> list[tuple[str, str, float, float, float, float]]:
    """Return (group, label, calls0, calls6, succ0, succ6) for every consumer."""
    out = []
    out.append(("Agent", "ReAct, fail-fast", *_rq1(RES / "rq1_recorded_seeds.csv")))
    for strat, label in (
        ("react_retry", "ReAct, retry"),
        ("cot", "Chain of thought"),
        ("reflect", "Self-reflection"),
    ):
        p = RES / "strategies" / f"rq1_{strat}.csv"
        if p.exists():
            out.append(("Agent", label, *_rq1(p)))

    mcp = pd.read_csv(SM / "control_wide_mcp.csv")
    a2a = pd.read_csv(SM / "control_wide_a2a.csv")
    for df, mech, label in (
        (mcp, "wf", "Workflow, MCP"),
        (a2a, "wf", "Workflow, A2A"),
        (mcp, "skill", "Skill, MCP"),
        (mcp, "free", "Free loop, MCP"),
    ):
        g = df[df.mechanism == mech].groupby("fault_rate")
        out.append(
            ("Schema mapping", label, g.proxy_calls.mean()[0.0], g.proxy_calls.mean()[0.6],
             g.f1.mean()[0.0], g.f1.mean()[0.6])
        )

    for h, label in (("claude", "Claude Code"), ("codex", "Codex"), ("antigravity", "Gemini CLI")):
        d = pd.read_csv(HAR / f"rq5_{h}_v2.csv")
        g = d.groupby("fault_rate")
        out.append(
            ("Coding assistant", label, g.tool_calls.mean()[0.0], g.tool_calls.mean()[0.6],
             g.gold_obtained.mean()[0.0], g.gold_obtained.mean()[0.6])
        )
    return out


def colour(c0: float, c6: float) -> str:
    ratio = c6 / c0
    if ratio < 0.9:
        return FEWER
    if ratio > 1.1:
        return MORE
    return FLAT


def main() -> None:
    data = rows()
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
         "mathtext.fontset": "stix", "font.size": 7, "axes.linewidth": 0.5,
         "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42}
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(3.15, 0.19 * len(data) + 0.75), sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.08},
    )
    ys, labels, prev, y = [], [], None, 0.0
    for grp, label, *_ in data:
        if prev is not None and grp != prev:
            y += 0.55
        ys.append(y)
        labels.append(label)
        prev = grp
        y += 1.0

    for yy, (grp, label, c0, c6, s0, s6) in zip(ys, data):
        col = colour(c0, c6)
        for ax, a, b in ((ax1, c0, c6), (ax2, s0, s6)):
            ax.annotate("", xy=(b, yy), xytext=(a, yy),
                        arrowprops={"arrowstyle": "-", "color": col, "lw": 1.6, "alpha": 0.55})
            ax.plot([a], [yy], "o", ms=4.2, mfc="white", mec=col, mew=1.0, zorder=3)
            ax.plot([b], [yy], "o", ms=4.2, mfc=col, mec="white", mew=0.6, zorder=4)

    ax1.set_xscale("log")
    ax1.set_xlim(0.9, 40)
    ax1.set_xticks([1, 2, 5, 10, 20])
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_xticklabels(["1", "2", "5", "10", "20"])
    ax1.set_xlabel("(a) calls per task", color=INK)
    ax2.set_xlim(-0.03, 1.05)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["0", "0.5", "1"])
    ax2.set_xlabel("(b) success", color=INK)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(labels, color=INK)
    ax1.invert_yaxis()

    # group names, set in the left margin above each block
    prev = None
    for yy, (grp, *_rest) in zip(ys, data):
        if grp != prev:
            ax1.annotate(grp, xy=(0, yy - 0.62), xycoords=("axes fraction", "data"),
                         xytext=(-2, 0), textcoords="offset points", ha="right",
                         fontsize=6.3, style="italic", color=MUTED)
            prev = grp

    for ax in (ax1, ax2):
        ax.grid(axis="x", color="#e4e3df", lw=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED, length=2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4, label="no faults"),
        plt.Line2D([], [], marker="o", ls="", mfc=MUTED, mec="white", ms=4, label="fault rate 0.6"),
        plt.Line2D([], [], color=FEWER, lw=1.6, label="fewer calls"),
        plt.Line2D([], [], color=MORE, lw=1.6, label="more calls"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=0.8)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.99, bottom=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    pd.DataFrame(data, columns=["group", "consumer", "calls_0", "calls_06", "success_0",
                                "success_06"]).round(3).to_csv(OUT.with_suffix(".csv"), index=False)
    print(pd.read_csv(OUT.with_suffix(".csv")).to_string())


if __name__ == "__main__":
    main()
