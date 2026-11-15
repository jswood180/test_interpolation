"""Multi-frame reconstruction methods (Phase 2B).

Outputs are on the target grid defined in :mod:`mlx_sr.types`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .types import Reconstruction, SensorConfig, Subpage

__all__ = [
    "reconstruct",
    "baseline_bicubic",
    "baseline_average",
    "normalized_convolution",
    "ibp",
    "btv_map",
]


def reconstruct(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, method: str,
                scale: int = 2, **kw) -> Reconstruction:
    """Dispatch to a method: ``bicubic``, ``average``, ``nc``, ``ibp``, ``btv``."""
    raise NotImplementedError("Phase 2B")


def baseline_bicubic(subpages: list[Subpage], sensor: SensorConfig, scale: int = 2,
                     mode: Literal["merged", "single"] = "merged") -> Reconstruction:
    raise NotImplementedError("Phase 2B")


def baseline_average(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig,
                     scale: int = 2) -> Reconstruction:
    raise NotImplementedError("Phase 2B")


def normalized_convolution(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig,
                           scale: int = 2, sigma: float | None = None) -> Reconstruction:
    raise NotImplementedError("Phase 2B")


def ibp(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int = 2,
        n_iter: int = 20) -> Reconstruction:
    raise NotImplementedError("Phase 2B")


def btv_map(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int = 2,
            lam: float = 0.01, alpha: float = 0.7, p: int = 2, eps_charbonnier: float = 1e-2,
            init: str = "nc", estimate_fpn: bool = False, n_outer: int = 3,
            psf_sigma: float | None = None) -> Reconstruction:
    raise NotImplementedError("Phase 2B")
