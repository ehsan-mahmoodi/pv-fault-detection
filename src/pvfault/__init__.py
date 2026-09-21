"""String-level fault detection for utility-scale photovoltaic plants."""

__version__ = "0.1.0"

from .plant import NIR, PlantSpec
from .simulate import Fault, default_fault_set, generate
from .detect import DetectorConfig, run_all
from .report import estimate_losses, score, summarise

__all__ = [
    "NIR", "PlantSpec", "Fault", "default_fault_set", "generate",
    "DetectorConfig", "run_all", "estimate_losses", "score", "summarise",
]
