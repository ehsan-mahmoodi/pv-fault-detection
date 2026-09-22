"""Tests. Run with:  PYTHONPATH=src python -m pytest -q   (or python tests/test_pvfault.py)"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvfault import (  # noqa: E402
    NIR, DetectorConfig, Fault, PlantSpec, default_fault_set,
    estimate_losses, generate, run_all, score,
)
from pvfault.detect import daily_ratio, replay, robust_z, theil_sen_slope  # noqa: E402
from pvfault.irradiance import cos_zenith, haurwitz_ghi  # noqa: E402


# ---------------------------------------------------------------- plant model

def test_plant_geometry_is_consistent():
    NIR.validate()
    assert NIR.modules_per_string == 23
    assert NIR.strings_per_inverter == 176
    assert abs(NIR.dc_capacity_kw - 10200.96) < 1.0
    assert abs(NIR.dc_ac_ratio - 1.133) < 0.01


def test_inverter_mapping_covers_every_string():
    seen = {NIR.inverter_of_string(i) for i in range(NIR.string_count)}
    assert seen == set(range(NIR.inverter_count))


def test_validate_rejects_bad_geometry():
    bad = PlantSpec(**{**NIR.__dict__, "string_count": 1000})
    try:
        bad.validate()
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible string count")


# ---------------------------------------------------------------- irradiance

def test_sun_is_below_horizon_at_midnight():
    doy = np.array([172])
    cz = cos_zenith(NIR.latitude, doy, np.array([0.0]), NIR.longitude, NIR.timezone_offset_h)
    assert cz[0] == 0.0


def test_summer_noon_is_brighter_than_winter_noon():
    noon = np.array([12.0])
    summer = cos_zenith(NIR.latitude, np.array([172]), noon, NIR.longitude, NIR.timezone_offset_h)
    winter = cos_zenith(NIR.latitude, np.array([355]), noon, NIR.longitude, NIR.timezone_offset_h)
    assert summer[0] > winter[0]


def test_clear_sky_ghi_is_physically_plausible():
    cz = np.linspace(0.05, 1.0, 40)
    ghi = haurwitz_ghi(cz, NIR.altitude_m)
    assert ghi.max() < 1200.0
    assert np.all(np.diff(ghi) > 0)  # monotonic in sun elevation


# ---------------------------------------------------------------- simulation

def test_generate_shape_and_night_is_zero():
    spec = NIR
    scada, truth = generate(spec, days=6, interval_minutes=60, faults=[], seed=1)
    assert len(scada) == 6 * 24
    assert truth.empty
    night = scada[scada["poa_wm2"] <= 1.0]
    cols = [c for c in scada.columns if c.startswith("string_")]
    assert float(night[cols].to_numpy().max()) == 0.0


def test_healthy_plant_lands_near_design_performance_ratio():
    scada, _ = generate(NIR, days=60, interval_minutes=30, faults=[], seed=5)
    cols = [c for c in scada.columns if c.startswith("string_")]
    energy_kwh = scada[cols].to_numpy().sum() * 0.5
    poa_kwh_m2 = scada["poa_wm2"].sum() * 0.5 / 1000.0
    pr = energy_kwh / (NIR.dc_capacity_kw * poa_kwh_m2)
    # Temperature losses sit on top of the design PR, so expect a little below it.
    assert 0.78 < pr < NIR.target_pr + 0.01


def test_disconnect_actually_zeroes_the_string():
    f = Fault(3, "disconnect", start_day=2, magnitude=1.0)
    scada, _ = generate(NIR, days=6, interval_minutes=60, faults=[f], seed=2)
    after = scada[scada["day_of_year"] >= scada["day_of_year"].min() + 2]
    assert float(after["string_0003"].max()) == 0.0
    assert float(scada["string_0004"].max()) > 0.0


# ---------------------------------------------------------------- detectors

def test_theil_sen_recovers_a_known_slope():
    x = np.arange(50, dtype=float)
    y = 1.0 - 0.003 * x
    assert abs(theil_sen_slope(y, x) - (-0.003)) < 1e-9


def test_theil_sen_survives_outliers():
    x = np.arange(60, dtype=float)
    y = 1.0 - 0.002 * x
    y[[5, 17, 41]] = 0.1  # three cloudy days
    assert abs(theil_sen_slope(y, x) - (-0.002)) < 5e-4


def test_robust_z_is_zero_for_a_uniform_fleet():
    ratio = pd.DataFrame(np.ones((10, 8)), columns=[f"string_{i:04d}" for i in range(8)])
    assert float(np.abs(robust_z(ratio)).max()) == 0.0


def test_healthy_plant_raises_no_alarms():
    scada, _ = generate(NIR, days=40, interval_minutes=60, faults=[], seed=11)
    alarms, _ = run_all(scada)
    assert alarms.empty, f"false positives on a healthy plant: {alarms['string'].tolist()}"


def test_daily_ratio_centres_on_one():
    scada, _ = generate(NIR, days=20, interval_minutes=60, faults=[], seed=3)
    ratio = daily_ratio(scada)
    assert abs(float(ratio.to_numpy().mean()) - 1.0) < 0.01


# ---------------------------------------------------------------- end to end

def test_end_to_end_precision_and_recall():
    spec = NIR
    faults = default_fault_set(spec)
    scada, truth = generate(spec, days=120, interval_minutes=30, faults=faults, seed=20260921)
    alarms, ratio = run_all(scada)
    report = estimate_losses(scada, ratio, alarms, spec)
    result = score(report, truth, spec.string_count)

    assert result["precision"] == 1.0, "a clean plant should not be flagged"
    assert result["recall"] >= 0.85, f"recall regressed to {result['recall']}"
    assert result["recall_by_fault_type"]["disconnect"] == "2/2"
    assert result["recall_by_fault_type"]["soiling"] == "4/4"


def test_replay_catches_disconnect_after_it_happens():
    spec = NIR
    fault = Fault(5, "disconnect", 20, 1.0)
    scada, _ = generate(spec, days=40, interval_minutes=60, faults=[fault], seed=3)
    first = replay(daily_ratio(scada))
    day = first["string_0005"]
    assert 21 <= day <= 21 + DetectorConfig().zero_output_days, day


def test_losses_are_ranked_and_positive():
    spec = NIR
    scada, truth = generate(spec, days=90, interval_minutes=60,
                            faults=default_fault_set(spec), seed=4)
    alarms, ratio = run_all(scada)
    report = estimate_losses(scada, ratio, alarms, spec)
    assert not report.empty
    assert (report["annual_lost_eur"] >= 0).all()
    assert report["annual_lost_eur"].is_monotonic_decreasing
    assert report["inverter"].between(1, spec.inverter_count).all()


def test_tighter_threshold_flags_fewer_strings():
    spec = NIR
    scada, _ = generate(spec, days=90, interval_minutes=60,
                        faults=default_fault_set(spec), seed=6)
    loose, _ = run_all(scada, DetectorConfig(z_threshold=-3.0))
    tight, _ = run_all(scada, DetectorConfig(z_threshold=-8.0))
    assert len(set(tight["string"])) <= len(set(loose["string"]))


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
