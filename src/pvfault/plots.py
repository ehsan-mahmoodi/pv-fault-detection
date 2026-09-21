"""Figures. Dark and light variants so they sit on either theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DARK = {
    "bg": "#0d1117", "panel": "#161b22", "ink": "#e6edf3", "mut": "#8b949e",
    "acc": "#58a6ff", "grn": "#3fb950", "amb": "#d29922", "red": "#f85149",
    "grid": "#30363d",
}
LIGHT = {
    "bg": "#ffffff", "panel": "#f6f8fa", "ink": "#24292f", "mut": "#57606a",
    "acc": "#0969da", "grn": "#1a7f37", "amb": "#9a6700", "red": "#cf222e",
    "grid": "#d0d7de",
}


def _style(ax, c, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(c["panel"])
    for spine in ax.spines.values():
        spine.set_color(c["grid"])
    ax.tick_params(colors=c["mut"], labelsize=8)
    ax.grid(True, color=c["grid"], alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=c["ink"], fontsize=11, fontweight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=c["mut"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=c["mut"], fontsize=9)


def fleet_overview(
    ratio: pd.DataFrame, report: pd.DataFrame, truth: pd.DataFrame,
    out: Path, theme: str = "dark",
) -> Path:
    """Three panels: the fleet band, the flagged strings, and the loss ranking."""
    c = DARK if theme == "dark" else LIGHT
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor=c["bg"])

    flagged = set(report["string"]) if not report.empty else set()
    days = ratio.index.to_numpy()

    # Panel 1 - the healthy fleet as a band, with faulted strings over it.
    ax = axes[0]
    healthy = ratio[[col for col in ratio.columns if col not in flagged]]
    lo = healthy.quantile(0.02, axis=1)
    hi = healthy.quantile(0.98, axis=1)
    med = healthy.median(axis=1)
    ax.fill_between(days, lo, hi, color=c["acc"], alpha=0.18,
                    label="healthy fleet, 2nd-98th pct")
    ax.plot(days, med, color=c["acc"], lw=1.4, label="fleet median")
    for col in list(flagged)[:6]:
        ax.plot(days, ratio[col], lw=1.0, alpha=0.9, color=c["red"])
    ax.set_ylim(0, 1.25)
    _style(ax, c, "Daily output vs fleet median", "day of year", "ratio to peer median")
    leg = ax.legend(fontsize=7, facecolor=c["panel"], edgecolor=c["grid"], loc="lower left")
    for t in leg.get_texts():
        t.set_color(c["mut"])

    # Panel 2 - distribution of mean ratio, threshold marked.
    ax = axes[1]
    mean_ratio = ratio.mean(axis=0)
    ax.hist(mean_ratio, bins=60, color=c["acc"], alpha=0.75)
    if flagged:
        ax.hist(mean_ratio[list(flagged)], bins=60, color=c["red"], alpha=0.95,
                label="flagged")
        leg = ax.legend(fontsize=7, facecolor=c["panel"], edgecolor=c["grid"])
        for t in leg.get_texts():
            t.set_color(c["mut"])
    ax.set_yscale("log")
    _style(ax, c, "Where the fleet sits", "mean ratio to peer median", "strings (log)")

    # Panel 3 - money.
    ax = axes[2]
    if not report.empty:
        top = report.head(12).iloc[::-1]
        labels = [s.replace("string_", "S") for s in top["string"]]
        colours = [
            c["red"] if "zero_output" in d else c["amb"] if "windowed_trend" in d
            else c["acc"] for d in top["detector"]
        ]
        ax.barh(labels, top["annual_lost_eur"], color=colours)
        for i, v in enumerate(top["annual_lost_eur"]):
            ax.text(v, i, f" {v:,.0f}", va="center", color=c["mut"], fontsize=7)
        ax.set_xlim(0, float(top["annual_lost_eur"].max()) * 1.22)
    _style(ax, c, "Annualised loss by string", "EUR per year", None)

    fig.suptitle(
        "Nir PV plant - 1,584 strings, 120 days of 30-minute SCADA",
        color=c["ink"], fontsize=12, fontweight="bold", x=0.007, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["bg"])
    plt.close(fig)
    return out


def fault_signatures(
    ratio: pd.DataFrame, truth: pd.DataFrame, out: Path, theme: str = "dark"
) -> Path:
    """One panel per fault type, showing what the detector is actually looking at."""
    c = DARK if theme == "dark" else LIGHT
    kinds = list(dict.fromkeys(truth["kind"]))
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.0 * len(kinds), 3.0),
                             facecolor=c["bg"], squeeze=False)
    days = ratio.index.to_numpy()
    med = ratio.median(axis=1)

    for ax, kind in zip(axes[0], kinds):
        ids = truth.loc[truth["kind"] == kind, "string_id"].astype(int).tolist()
        ax.plot(days, med, color=c["mut"], lw=1.0, ls="--", label="fleet median")
        for sid in ids[:3]:
            col = f"string_{sid:04d}"
            if col in ratio:
                ax.plot(days, ratio[col], lw=1.2, color=c["red"])
        ax.set_ylim(0, 1.2)
        _style(ax, c, kind, "day", "ratio")

    fig.suptitle("Fault signatures in the peer ratio", color=c["ink"],
                 fontsize=11, fontweight="bold", x=0.007, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=c["bg"])
    plt.close(fig)
    return out
