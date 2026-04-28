from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StarMeasurement:
    star_id: int
    ra: float
    dec: float
    x_pixel: float
    y_pixel: float
    instr_mag: float
    catalog_mag: float = 0.0
    cal_mag: float = 0.0
    residual: float = 0.0


@dataclass
class PhotometryResult:
    stars: list[StarMeasurement] = field(default_factory=list)
    r_squared: float = 0.0
    n_matched: int = 0
