"""Regenerate every figure in outputs/ from a single deterministic run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvfault import NIR, default_fault_set, estimate_losses, generate, run_all, score
from pvfault.detect import replay
from pvfault.plots import detection_replay, fault_signatures, fleet_overview

OUT = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    scada, truth = generate(NIR, days=120, faults=default_fault_set(NIR), seed=20260921)
    alarms, ratio = run_all(scada)
    report = estimate_losses(scada, ratio, alarms, NIR)
    result = score(report, truth, NIR.string_count)
    print("precision %.2f  recall %.2f  f1 %.2f"
          % (result["precision"], result["recall"], result["f1"]))

    for theme in ("dark", "light"):
        suffix = "" if theme == "dark" else "-light"
        fleet_overview(ratio, report, truth, OUT / f"fleet-overview{suffix}.png", theme)
        fault_signatures(ratio, truth, OUT / f"fault-signatures{suffix}.png", theme)
    detection_replay(ratio, truth, replay(ratio, step=2), OUT / "detection-replay.gif")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
