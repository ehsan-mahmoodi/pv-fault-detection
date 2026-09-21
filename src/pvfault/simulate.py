"""Synthetic SCADA generation with injected faults.

Produces string-level power at a configurable interval for a whole plant, then
injects a known set of faults. Because the ground truth is known, the detector can
be scored honestly rather than eyeballed - which is the entire point of building a
simulator instead of hand-waving at a real export.

Fault types implemented:

    disconnect    a string drops offline at a moment and stays there
    soiling       gradual loss accumulating over weeks, partially reset by rain
    shading       a repeatable daily loss over a fixed solar-time window
    degradation   an abrupt step down in output that then persists
    underperf     a persistent low-grade offset, the hardest of the five to see
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .irradiance import (
    cell_temperature,
    cos_zenith,
    haurwitz_ghi,
    plane_of_array,
)
from .plant import PlantSpec

FaultKind = Literal["disconnect", "soiling", "shading", "degradation", "underperf"]


@dataclass(frozen=True)
class Fault:
    string_id: int
    kind: FaultKind
    start_day: int
    magnitude: float  # final fractional loss, 0-1
    detail: str = ""


def _ambient_profile(
    day_of_year: np.ndarray, clock_hour: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A plausible desert ambient temperature: seasonal swing plus a daily cycle."""
    seasonal = 20.0 - 13.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.0)
    daily = 9.0 * np.sin(2 * np.pi * (clock_hour - 9.0) / 24.0)
    noise = rng.normal(0.0, 1.2, size=day_of_year.shape)
    return seasonal + daily + noise


def _clearness(
    n_steps: int, day_of_year: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cloud modulation. Yazd is arid, so most days are near-clear."""
    days = np.unique(day_of_year)
    per_day = {}
    for d in days:
        roll = rng.random()
        if roll < 0.78:
            per_day[d] = rng.uniform(0.95, 1.0)      # clear
        elif roll < 0.93:
            per_day[d] = rng.uniform(0.70, 0.95)     # hazy
        else:
            per_day[d] = rng.uniform(0.30, 0.70)     # overcast
    base = np.array([per_day[d] for d in day_of_year])
    jitter = np.clip(rng.normal(1.0, 0.04, size=n_steps), 0.6, 1.1)
    return np.clip(base * jitter, 0.05, 1.0)


def generate(
    spec: PlantSpec,
    days: int = 120,
    start_day_of_year: int = 60,
    interval_minutes: int = 30,
    faults: list[Fault] | None = None,
    seed: int = 20260921,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate string-level SCADA.

    Returns
    -------
    scada : DataFrame
        Columns timestamp, day_of_year, hour, poa_wm2, cell_temp_c, plus one
        column per string holding AC-side power in kW.
    truth : DataFrame
        The injected faults, for scoring.
    """
    spec.validate()
    rng = np.random.default_rng(seed)
    faults = faults or []

    steps_per_day = int(24 * 60 / interval_minutes)
    hours = np.arange(0, 24, interval_minutes / 60.0)
    doy = np.repeat(np.arange(start_day_of_year, start_day_of_year + days), steps_per_day)
    clock = np.tile(hours, days)
    n = doy.size

    cz = cos_zenith(spec.latitude, doy, clock, spec.longitude, spec.timezone_offset_h)
    ghi_clear = haurwitz_ghi(cz, spec.altitude_m)
    clearness = _clearness(n, doy, rng)
    ghi = ghi_clear * clearness

    poa = plane_of_array(
        ghi, cz, doy, spec.latitude, clock, spec.longitude,
        spec.timezone_offset_h, spec.tilt_deg, spec.azimuth_deg,
    )
    ambient = _ambient_profile(doy, clock, rng)
    tcell = cell_temperature(poa, ambient, spec.noct_c)

    # Healthy string power. target_pr already folds in soiling, mismatch, wiring
    # and inverter losses at reference conditions; the temperature term is applied
    # explicitly on top so hot afternoons show the dip an operator would expect.
    temp_factor = 1.0 + spec.temp_coeff_pmax * (tcell - 25.0)
    base_kw = (
        spec.string_capacity_kw * (poa / 1000.0) * spec.target_pr * temp_factor
    )
    base_kw = np.maximum(base_kw, 0.0)

    # Per-string manufacturing spread: real fleets are not identical.
    spread = rng.normal(1.0, 0.015, size=spec.string_count)

    day_index = doy - start_day_of_year
    power = np.empty((n, spec.string_count), dtype=np.float32)
    for s in range(spec.string_count):
        series = base_kw * spread[s]
        noise = rng.normal(1.0, 0.012, size=n)
        power[:, s] = series * noise

    for f in faults:
        power[:, f.string_id] = _apply_fault(
            power[:, f.string_id], f, day_index, clock, rng
        )

    power = np.maximum(power, 0.0)
    power[poa <= 1.0] = 0.0

    timestamps = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        (doy - 1) * 24 + clock, unit="h"
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "day_of_year": doy,
            "hour": clock,
            "poa_wm2": poa,
            "cell_temp_c": tcell,
        }
    )
    cols = pd.DataFrame(
        power, columns=[f"string_{i:04d}" for i in range(spec.string_count)]
    )
    scada = pd.concat([frame, cols], axis=1)

    truth = pd.DataFrame(
        [
            {
                "string_id": f.string_id,
                "kind": f.kind,
                "start_day": f.start_day,
                "magnitude": f.magnitude,
                "detail": f.detail,
            }
            for f in faults
        ]
    )
    return scada, truth


def _apply_fault(
    series: np.ndarray,
    fault: Fault,
    day_index: np.ndarray,
    clock: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = series.copy()
    active = day_index >= fault.start_day

    if fault.kind == "disconnect":
        out[active] = 0.0

    elif fault.kind == "soiling":
        # Linear accumulation to the stated magnitude over 45 days, with a rain
        # event near day 30 clearing most of it.
        elapsed = np.clip(day_index - fault.start_day, 0, None)
        loss = np.clip(elapsed / 45.0, 0.0, 1.0) * fault.magnitude
        rained = day_index >= (fault.start_day + 30)
        loss = np.where(rained, loss * 0.25, loss)
        out = out * (1.0 - loss)

    elif fault.kind == "shading":
        # A structure shades the string for a fixed window each morning.
        window = (clock >= 7.0) & (clock <= 10.0)
        out = np.where(active & window, out * (1.0 - fault.magnitude), out)

    elif fault.kind == "degradation":
        out[active] = out[active] * (1.0 - fault.magnitude)

    elif fault.kind == "underperf":
        drift = rng.normal(0.0, 0.004, size=out.size)
        out[active] = out[active] * (1.0 - fault.magnitude + drift[active])

    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown fault kind: {fault.kind}")

    return out


def default_fault_set(spec: PlantSpec, seed: int = 7) -> list[Fault]:
    """A representative fault set spread across inverters and severities."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(spec.string_count, size=14, replace=False)
    return [
        Fault(int(picks[0]), "disconnect", 30, 1.00, "combiner fuse"),
        Fault(int(picks[1]), "disconnect", 72, 1.00, "connector failure"),
        Fault(int(picks[2]), "soiling", 12, 0.22, "dust, no wash cycle"),
        Fault(int(picks[3]), "soiling", 20, 0.16, "dust, partial"),
        Fault(int(picks[4]), "soiling", 8, 0.28, "near access track"),
        Fault(int(picks[5]), "shading", 0, 0.45, "new switchgear cabinet"),
        Fault(int(picks[6]), "shading", 0, 0.30, "vegetation"),
        Fault(int(picks[7]), "degradation", 55, 0.18, "PID onset"),
        Fault(int(picks[8]), "degradation", 40, 0.12, "cell cracking"),
        Fault(int(picks[9]), "degradation", 90, 0.25, "bypass diode"),
        Fault(int(picks[10]), "underperf", 0, 0.07, "module mismatch"),
        Fault(int(picks[11]), "underperf", 0, 0.05, "undersized cable"),
        Fault(int(picks[12]), "underperf", 25, 0.09, "loose termination"),
        Fault(int(picks[13]), "soiling", 35, 0.19, "bird fouling"),
    ]
