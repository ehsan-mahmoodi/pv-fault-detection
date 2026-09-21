"""Fault detection on string-level SCADA.

The governing idea is peer comparison. Absolute thresholds on performance ratio are
brittle because they move with season, soiling and temperature; but 1,584 strings on
one site all see very nearly the same weather, so a string that drifts away from its
peers is suspicious regardless of the absolute number.

Three detectors, each aimed at a different failure signature:

    zero_output     a string producing nothing while its peers produce      -> disconnect
    peer_ratio      a persistent gap to the fleet median, via robust z-score -> underperformance
    trend           a downward slope in that gap over time, via Theil-Sen    -> soiling / degradation

They are combined into one ranked table, because an operator wants a work order
list, not three dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .plant import PlantSpec

MIN_POA = 300.0  # W/m2; below this the signal-to-noise is not worth trusting


@dataclass(frozen=True)
class DetectorConfig:
    min_poa_wm2: float = MIN_POA
    zero_output_frac: float = 0.02   # below this share of peer median = offline
    zero_output_days: int = 2        # consecutive days before raising
    z_threshold: float = -4.0        # robust z below which a string is flagged
    trend_threshold: float = -0.0015  # ratio lost per day, whole window
    min_days: int = 10
    window_days: int = 30            # rolling window for the soiling detector
    window_stride: int = 5
    window_trend_threshold: float = -0.0022  # ratio lost per day within a window


def string_columns(scada: pd.DataFrame) -> list[str]:
    return [c for c in scada.columns if c.startswith("string_")]


def daily_ratio(scada: pd.DataFrame, cfg: DetectorConfig | None = None) -> pd.DataFrame:
    """Daily energy per string divided by the fleet median for that day.

    A healthy string sits at ~1.0 every day. Weather, season and temperature affect
    numerator and denominator together and so largely cancel.
    """
    cfg = cfg or DetectorConfig()
    cols = string_columns(scada)
    usable = scada[scada["poa_wm2"] >= cfg.min_poa_wm2]
    if usable.empty:
        raise ValueError("no timestamps above the irradiance threshold")

    daily = usable.groupby("day_of_year")[cols].sum()
    median = daily.median(axis=1)
    ratio = daily.div(median, axis=0)
    return ratio


def detect_zero_output(
    ratio: pd.DataFrame, cfg: DetectorConfig | None = None
) -> pd.DataFrame:
    """Strings that stop producing while the rest of the plant carries on."""
    cfg = cfg or DetectorConfig()
    offline = ratio < cfg.zero_output_frac
    rows = []
    for col in ratio.columns:
        flags = offline[col].to_numpy()
        run, best, best_end = 0, 0, None
        for i, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run > best:
                best, best_end = run, i
        if best >= cfg.zero_output_days:
            first = best_end - best + 1
            rows.append(
                {
                    "string": col,
                    "detector": "zero_output",
                    "severity": 1.0,
                    "first_day": int(ratio.index[first]),
                    "days_affected": int(best),
                    "evidence": f"{best} consecutive days below "
                                f"{cfg.zero_output_frac:.0%} of peer median",
                }
            )
    return pd.DataFrame(rows)


def robust_z(ratio: pd.DataFrame) -> pd.Series:
    """Median-absolute-deviation z-score of each string's mean daily ratio."""
    mean_ratio = ratio.mean(axis=0)
    med = float(np.median(mean_ratio))
    mad = float(np.median(np.abs(mean_ratio - med)))
    scale = 1.4826 * mad
    if scale <= 0:
        return pd.Series(0.0, index=mean_ratio.index)
    return (mean_ratio - med) / scale


def detect_peer_ratio(
    ratio: pd.DataFrame, cfg: DetectorConfig | None = None
) -> pd.DataFrame:
    """Persistent underperformance against the fleet."""
    cfg = cfg or DetectorConfig()
    z = robust_z(ratio)
    mean_ratio = ratio.mean(axis=0)
    hits = z[z <= cfg.z_threshold]
    rows = [
        {
            "string": col,
            "detector": "peer_ratio",
            "severity": float(min(1.0, abs(z[col]) / 20.0)),
            "first_day": int(ratio.index[0]),
            "days_affected": int(ratio.shape[0]),
            "evidence": f"mean ratio {mean_ratio[col]:.3f}, robust z {z[col]:.1f}",
        }
        for col in hits.index
    ]
    return pd.DataFrame(rows)


def theil_sen_slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Median of pairwise slopes. Resistant to the outliers a cloudy day creates."""
    n = y.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float) if x is None else x.astype(float)
    # Sub-sample pairs when the series is long, to keep this O(k) not O(n^2).
    if n > 120:
        rng = np.random.default_rng(0)
        i = rng.integers(0, n, size=8000)
        j = rng.integers(0, n, size=8000)
        keep = i != j
        i, j = i[keep], j[keep]
    else:
        i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    slopes = (y[j][ok] - y[i][ok]) / dx[ok]
    return float(np.median(slopes)) if slopes.size else 0.0


def detect_trend(
    ratio: pd.DataFrame, cfg: DetectorConfig | None = None
) -> pd.DataFrame:
    """Strings whose peer ratio is sliding downwards over time."""
    cfg = cfg or DetectorConfig()
    if ratio.shape[0] < cfg.min_days:
        return pd.DataFrame()
    rows = []
    days = ratio.index.to_numpy(dtype=float)
    for col in ratio.columns:
        slope = theil_sen_slope(ratio[col].to_numpy(dtype=float), days)
        if slope <= cfg.trend_threshold:
            total = slope * (days[-1] - days[0])
            rows.append(
                {
                    "string": col,
                    "detector": "trend",
                    "severity": float(min(1.0, abs(total))),
                    "first_day": int(days[0]),
                    "days_affected": int(ratio.shape[0]),
                    "evidence": f"{slope * 100:.3f}% of peer median lost per day "
                                f"({total:.1%} over the window)",
                }
            )
    return pd.DataFrame(rows)


def detect_windowed_trend(
    ratio: pd.DataFrame, cfg: DetectorConfig | None = None
) -> pd.DataFrame:
    """Soiling that is washed off by rain hides from a whole-window slope.

    Accumulation runs for weeks and is then largely reset, so the net gradient
    across the record is close to flat even though the string lost real energy.
    Scanning shorter overlapping windows and keeping the worst one recovers it.
    """
    cfg = cfg or DetectorConfig()
    n_days = ratio.shape[0]
    if n_days < cfg.window_days:
        return pd.DataFrame()

    days = ratio.index.to_numpy(dtype=float)
    starts = range(0, n_days - cfg.window_days + 1, max(1, cfg.window_stride))
    rows = []
    for col in ratio.columns:
        series = ratio[col].to_numpy(dtype=float)
        worst_slope, worst_start = 0.0, 0
        for s in starts:
            seg = series[s : s + cfg.window_days]
            slope = theil_sen_slope(seg, days[s : s + cfg.window_days])
            if slope < worst_slope:
                worst_slope, worst_start = slope, s
        if worst_slope <= cfg.window_trend_threshold:
            drop = worst_slope * cfg.window_days
            rows.append(
                {
                    "string": col,
                    "detector": "windowed_trend",
                    "severity": float(min(1.0, abs(drop))),
                    "first_day": int(days[worst_start]),
                    "days_affected": int(cfg.window_days),
                    "evidence": f"{drop:.1%} lost over a {cfg.window_days}-day window "
                                f"from day {int(days[worst_start])}",
                }
            )
    return pd.DataFrame(rows)


def run_all(
    scada: pd.DataFrame, cfg: DetectorConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every detector and return (alarms, daily ratio matrix)."""
    cfg = cfg or DetectorConfig()
    ratio = daily_ratio(scada, cfg)
    parts = [
        detect_zero_output(ratio, cfg),
        detect_peer_ratio(ratio, cfg),
        detect_trend(ratio, cfg),
        detect_windowed_trend(ratio, cfg),
    ]
    parts = [p for p in parts if not p.empty]
    alarms = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(
            columns=["string", "detector", "severity", "first_day",
                     "days_affected", "evidence"]
        )
    )
    return alarms, ratio
