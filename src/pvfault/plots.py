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


def detection_replay(
    ratio: pd.DataFrame, truth: pd.DataFrame, first_flag: dict[str, int],
    out: Path, step: int = 2, fps: int = 10, hold_s: float = 2.5,
) -> Path:
    """Animated GIF: the record plays out day by day and strings turn red when caught.

    Dark only - a GIF cannot follow the viewer's theme, and the README already
    leads with the dark figures.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    c = DARK
    days = np.arange(1, ratio.shape[0] + 1)
    kind_of = {f"string_{int(s):04d}": k for s, k in zip(truth["string_id"], truth["kind"])}
    tracked = list(dict.fromkeys([*kind_of, *first_flag]))  # faults + any false alarm
    lo = ratio.quantile(0.02, axis=1).to_numpy()
    hi = ratio.quantile(0.98, axis=1).to_numpy()
    med = ratio.median(axis=1).to_numpy()

    fig = plt.figure(figsize=(8.0, 4.2), facecolor=c["bg"])
    ax = fig.add_axes((0.07, 0.12, 0.58, 0.72))
    side = fig.add_axes((0.68, 0.05, 0.31, 0.79))
    side.axis("off")
    _style(ax, c, None, "day of record", "ratio to peer median")
    ax.set_xlim(1, days[-1])
    ax.set_ylim(0, 1.25)

    band = [ax.fill_between(days[:1], lo[:1], hi[:1], color=c["acc"], alpha=0.18)]
    (med_line,) = ax.plot([], [], color=c["acc"], lw=1.3, label="fleet median")
    lines = {col: ax.plot([], [], lw=1.0, color=c["amb"], alpha=0.8)[0] for col in tracked}
    ax.plot([], [], lw=1.0, color=c["amb"], label="injected fault, not yet caught")
    ax.plot([], [], lw=1.4, color=c["red"], label="flagged by detector")
    leg = ax.legend(fontsize=7, facecolor=c["panel"], edgecolor=c["grid"], loc="center left")
    for t in leg.get_texts():
        t.set_color(c["mut"])

    fig.text(0.007, 0.95, "Nir PV plant - detectors replayed on 1,584 strings",
             color=c["ink"], fontsize=11, fontweight="bold")
    clock = fig.text(0.07, 0.87, "", color=c["mut"], fontsize=9)
    tally = side.text(0, 1, "", color=c["ink"], fontsize=10, fontweight="bold",
                      va="top", family="monospace")
    log = side.text(0, 0.80, "", color=c["mut"], fontsize=7.5, va="top",
                    family="monospace", linespacing=1.5)

    ends = sorted({*range(step, days[-1] + 1, step), int(days[-1])})
    frames = ends + [ends[-1]] * int(hold_s * fps)

    def draw(n):
        band[0].remove()
        band[0] = ax.fill_between(days[:n], lo[:n], hi[:n], color=c["acc"], alpha=0.18)
        med_line.set_data(days[:n], med[:n])
        caught = {col: d for col, d in first_flag.items() if d <= n}
        for col, line in lines.items():
            line.set_data(days[:n], ratio[col].to_numpy()[:n])
            if col in caught:
                line.set(color=c["red"], alpha=1.0, lw=1.4)
        hits = sum(col in kind_of for col in caught)
        clock.set_text(f"day {n} of {days[-1]}")
        tally.set_text(f"found  {hits:>2} / {len(kind_of)}\n"
                       f"false  {len(caught) - hits:>2}")
        rows = sorted(caught.items(), key=lambda kv: kv[1])
        log.set_text("\n".join(
            f"day {d:>3}  S{col[-4:]}  {kind_of.get(col, 'FALSE ALARM')}"
            for col, d in rows
        ))
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100,
              savefig_kwargs={"facecolor": c["bg"]})
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
