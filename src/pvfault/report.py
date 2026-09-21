"""Turning alarms into a work order list.

A detector that emits "string_0412 has a robust z of -11.3" has not finished the
job. A maintenance planner needs to know which fault to attend to first, and the
only ranking they can act on is money. So every alarm is converted into lost energy
and then into lost revenue at the tariff, and the list is sorted by that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detect import string_columns
from .plant import PlantSpec

# Iranian PPA tariffs for utility PV were set in the region of EUR 0.06-0.10/kWh
# over the plant's life. 0.08 is used here and is a single, visible assumption.
DEFAULT_TARIFF_EUR_PER_KWH = 0.08


def estimate_losses(
    scada: pd.DataFrame,
    ratio: pd.DataFrame,
    alarms: pd.DataFrame,
    spec: PlantSpec,
    tariff_eur_per_kwh: float = DEFAULT_TARIFF_EUR_PER_KWH,
    interval_minutes: int = 30,
) -> pd.DataFrame:
    """Attach observed and annualised loss to each alarming string."""
    if alarms.empty:
        return alarms.assign(
            lost_kwh=[], annual_lost_kwh=[], annual_lost_eur=[], inverter=[]
        )

    cols = string_columns(scada)
    hours_per_step = interval_minutes / 60.0
    daily_kwh = (
        scada[scada["poa_wm2"] > 0].groupby("day_of_year")[cols].sum() * hours_per_step
    )
    peer_median_kwh = daily_kwh.median(axis=1)
    window_days = daily_kwh.shape[0]

    # One row per string: worst severity wins, detectors are concatenated.
    grouped = (
        alarms.sort_values("severity", ascending=False)
        .groupby("string", as_index=False)
        .agg(
            detector=("detector", lambda s: " + ".join(sorted(set(s)))),
            severity=("severity", "max"),
            first_day=("first_day", "min"),
            days_affected=("days_affected", "max"),
            evidence=("evidence", lambda s: "; ".join(s)),
        )
    )

    lost, annual, inverter = [], [], []
    for _, row in grouped.iterrows():
        col = row["string"]
        shortfall = (peer_median_kwh - daily_kwh[col]).clip(lower=0.0)
        observed = float(shortfall.sum())
        lost.append(observed)
        annual.append(observed / window_days * 365.0)
        idx = int(col.split("_")[1])
        inverter.append(spec.inverter_of_string(idx) + 1)

    grouped["lost_kwh"] = np.round(lost, 1)
    grouped["annual_lost_kwh"] = np.round(annual, 1)
    grouped["annual_lost_eur"] = np.round(np.array(annual) * tariff_eur_per_kwh, 2)
    grouped["inverter"] = inverter
    return grouped.sort_values("annual_lost_eur", ascending=False).reset_index(drop=True)


def score(report: pd.DataFrame, truth: pd.DataFrame, n_strings: int) -> dict:
    """Score the detector against known ground truth."""
    if truth.empty:
        return {}
    flagged = {int(s.split("_")[1]) for s in report["string"]} if not report.empty else set()
    actual = set(truth["string_id"].astype(int))

    tp = len(flagged & actual)
    fp = len(flagged - actual)
    fn = len(actual - flagged)
    tn = n_strings - len(flagged | actual)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_kind = {}
    for kind in sorted(truth["kind"].unique()):
        ids = set(truth.loc[truth["kind"] == kind, "string_id"].astype(int))
        by_kind[kind] = f"{len(ids & flagged)}/{len(ids)}"

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "recall_by_fault_type": by_kind,
        "missed_strings": sorted(actual - flagged),
    }


def summarise(report: pd.DataFrame, spec: PlantSpec) -> str:
    """A short operator-facing summary."""
    if report.empty:
        return "No strings flagged."
    total_eur = float(report["annual_lost_eur"].sum())
    total_kwh = float(report["annual_lost_kwh"].sum())
    expected_annual_kwh = spec.dc_capacity_kw * 2006.0  # design specific yield
    pct = total_kwh / expected_annual_kwh * 100.0
    lines = [
        f"{len(report)} of {spec.string_count} strings flagged "
        f"({len(report) / spec.string_count:.1%} of the fleet).",
        f"Estimated annualised loss {total_kwh:,.0f} kWh "
        f"= EUR {total_eur:,.0f} at tariff, {pct:.2f}% of design yield.",
        f"Worst inverter: {int(report.groupby('inverter')['annual_lost_eur'].sum().idxmax())}.",
    ]
    return "\n".join(lines)
