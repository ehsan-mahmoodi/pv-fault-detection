"""Plant model.

The reference plant is the 10 MW Nir photovoltaic plant in Yazd province, whose
design figures come from the project profile I authored: 36,432 modules of 280 Wp
arranged in 1,584 strings, nine 1 MW central inverters, and an expected first-year
yield of 20,465 MWh at a performance ratio of 87%.

Those numbers fix the geometry of everything downstream, so they live here rather
than being scattered through the simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class PlantSpec:
    """Design specification of a utility-scale PV plant."""

    name: str
    latitude: float
    longitude: float
    altitude_m: float
    timezone_offset_h: float

    module_count: int
    module_wp: float
    string_count: int
    inverter_count: int
    inverter_ac_kw: float

    tilt_deg: float
    azimuth_deg: float  # 180 = due south

    temp_coeff_pmax: float  # fraction per degree C, negative
    noct_c: float
    target_pr: float

    @property
    def modules_per_string(self) -> int:
        return self.module_count // self.string_count

    @property
    def strings_per_inverter(self) -> int:
        return self.string_count // self.inverter_count

    @property
    def dc_capacity_kw(self) -> float:
        return self.module_count * self.module_wp / 1000.0

    @property
    def ac_capacity_kw(self) -> float:
        return self.inverter_count * self.inverter_ac_kw

    @property
    def string_capacity_kw(self) -> float:
        return self.modules_per_string * self.module_wp / 1000.0

    @property
    def dc_ac_ratio(self) -> float:
        return self.dc_capacity_kw / self.ac_capacity_kw

    def inverter_of_string(self, string_index: int) -> int:
        """Which inverter a given string is wired to."""
        return string_index // self.strings_per_inverter

    def validate(self) -> None:
        if self.module_count % self.string_count:
            raise ValueError(
                f"{self.module_count} modules do not divide evenly into "
                f"{self.string_count} strings"
            )
        if self.string_count % self.inverter_count:
            raise ValueError(
                f"{self.string_count} strings do not divide evenly across "
                f"{self.inverter_count} inverters"
            )
        if not 0.0 < self.target_pr <= 1.0:
            raise ValueError("target_pr must be a fraction in (0, 1]")
        if self.temp_coeff_pmax > 0:
            raise ValueError("temp_coeff_pmax is expected to be negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "PlantSpec":
        with open(path, encoding="utf-8") as fh:
            spec = cls(**json.load(fh))
        spec.validate()
        return spec

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)


NIR = PlantSpec(
    name="Nir PV Plant",
    latitude=32.03,
    longitude=54.35,
    altitude_m=2250.0,
    timezone_offset_h=3.5,
    module_count=36432,
    module_wp=280.0,
    string_count=1584,
    inverter_count=9,
    inverter_ac_kw=1000.0,
    tilt_deg=30.0,
    azimuth_deg=180.0,
    temp_coeff_pmax=-0.0041,
    noct_c=45.0,
    target_pr=0.87,
)
