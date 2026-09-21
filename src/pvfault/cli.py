"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .detect import DetectorConfig, run_all
from .plant import NIR, PlantSpec
from .report import estimate_losses, score, summarise
from .simulate import default_fault_set, generate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvfault",
        description="Detect underperforming PV strings from SCADA data.",
    )
    p.add_argument("--version", action="version", version=f"pvfault {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="generate synthetic SCADA with known faults")
    sim.add_argument("--days", type=int, default=120)
    sim.add_argument("--interval", type=int, default=30, help="minutes")
    sim.add_argument("--seed", type=int, default=20260921)
    sim.add_argument("--out", type=Path, default=Path("outputs"))
    sim.add_argument("--plant", type=Path, help="plant spec JSON; defaults to Nir")

    det = sub.add_parser("detect", help="run detectors over a SCADA csv")
    det.add_argument("--scada", type=Path, required=True)
    det.add_argument("--truth", type=Path, help="ground truth csv, enables scoring")
    det.add_argument("--out", type=Path, default=Path("outputs"))
    det.add_argument("--tariff", type=float, default=0.08, help="EUR per kWh")
    det.add_argument("--z-threshold", type=float, default=-4.0)
    det.add_argument("--plant", type=Path, help="plant spec JSON; defaults to Nir")

    demo = sub.add_parser("demo", help="simulate then detect, end to end")
    demo.add_argument("--days", type=int, default=120)
    demo.add_argument("--interval", type=int, default=30)
    demo.add_argument("--seed", type=int, default=20260921)
    demo.add_argument("--out", type=Path, default=Path("outputs"))
    demo.add_argument("--tariff", type=float, default=0.08)
    return p


def _load_plant(path: Path | None) -> PlantSpec:
    return PlantSpec.from_json(path) if path else NIR


def cmd_simulate(args) -> int:
    spec = _load_plant(args.plant)
    scada, truth = generate(
        spec, days=args.days, interval_minutes=args.interval,
        faults=default_fault_set(spec), seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    scada.to_csv(args.out / "scada.csv", index=False)
    truth.to_csv(args.out / "truth.csv", index=False)
    print(f"wrote {args.out/'scada.csv'}  ({len(scada):,} rows x {spec.string_count} strings)")
    print(f"wrote {args.out/'truth.csv'}  ({len(truth)} injected faults)")
    return 0


def cmd_detect(args) -> int:
    spec = _load_plant(args.plant)
    scada = pd.read_csv(args.scada, parse_dates=["timestamp"])
    cfg = DetectorConfig(z_threshold=args.z_threshold)
    alarms, ratio = run_all(scada, cfg)
    report = estimate_losses(scada, ratio, alarms, spec, tariff_eur_per_kwh=args.tariff)

    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "work_orders.csv", index=False)
    print(summarise(report, spec))
    print()
    if not report.empty:
        cols = ["string", "inverter", "detector", "annual_lost_kwh",
                "annual_lost_eur", "evidence"]
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(report[cols].head(20).to_string(index=False))

    if args.truth and args.truth.exists():
        truth = pd.read_csv(args.truth)
        result = score(report, truth, spec.string_count)
        (args.out / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\nscored against ground truth:")
        print(json.dumps(result, indent=2))
    return 0


def cmd_demo(args) -> int:
    spec = NIR
    print(f"{spec.name}: {spec.dc_capacity_kw/1000:.2f} MWp DC, "
          f"{spec.string_count} strings, {spec.inverter_count} inverters, "
          f"DC/AC {spec.dc_ac_ratio:.2f}")
    scada, truth = generate(
        spec, days=args.days, interval_minutes=args.interval,
        faults=default_fault_set(spec), seed=args.seed,
    )
    print(f"simulated {len(scada):,} intervals over {args.days} days "
          f"with {len(truth)} injected faults")

    alarms, ratio = run_all(scada)
    report = estimate_losses(scada, ratio, alarms, spec, tariff_eur_per_kwh=args.tariff)
    result = score(report, truth, spec.string_count)

    args.out.mkdir(parents=True, exist_ok=True)
    scada.to_csv(args.out / "scada.csv", index=False)
    truth.to_csv(args.out / "truth.csv", index=False)
    report.to_csv(args.out / "work_orders.csv", index=False)
    (args.out / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(summarise(report, spec))
    print()
    cols = ["string", "inverter", "detector", "annual_lost_kwh", "annual_lost_eur"]
    with pd.option_context("display.width", 200):
        print(report[cols].to_string(index=False))
    print("\nscore:", json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"simulate": cmd_simulate, "detect": cmd_detect, "demo": cmd_demo}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
