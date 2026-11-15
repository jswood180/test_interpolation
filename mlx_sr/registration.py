"""Sub-pixel registration of subpages (Phase 2A).

All shifts follow the convention in :mod:`mlx_sr.types`: ``(dx, dy)`` in LR
pixels, (x, y) order.  ``skimage.registration.phase_cross_correlation``
returns ``(row, col)``; swap it.  Estimated shifts are relative to
``ref_index``.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .types import Capture, GyroTrace, Registration, SensorConfig, Subpage

__all__ = [
    "register",
    "fill_subpage",
    "merged_frames",
    "phase_correlation",
    "gauss_newton_translation",
    "integrate_gyro",
    "estimate_time_offset",
    "refine_against_hr",
]


def register(capture: Capture, method: Literal["phase", "gn", "gyro", "hybrid"] = "phase",
             ref_index: int = 0, **kw) -> Registration:
    """Estimate per-subpage shifts ``(K, 2)`` relative to subpage ``ref_index``."""
    raise NotImplementedError("Phase 2A")


def fill_subpage(sp: Subpage) -> np.ndarray:
    """Fill unread/dead pixels by normalized convolution of their neighbors."""
    raise NotImplementedError("Phase 2A")


def merged_frames(capture: Capture) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merged ``(values, mask)`` per full frame."""
    raise NotImplementedError("Phase 2A")


def phase_correlation(ref: np.ndarray, img: np.ndarray, upsample_factor: int = 20) -> np.ndarray:
    """Upsampled phase correlation on Hann-windowed images -> ``(dx, dy)``."""
    raise NotImplementedError("Phase 2A")


def gauss_newton_translation(ref_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], img: np.ndarray,
                             mask: np.ndarray, init: np.ndarray, iters: int = 20, tol: float = 1e-4
                             ) -> tuple[np.ndarray, float]:
    """Masked Gauss-Newton translation refinement -> ``(shift, min Hessian eigenvalue)``."""
    raise NotImplementedError("Phase 2A")


def integrate_gyro(gyro: GyroTrace, sensor: SensorConfig, subpage_windows: np.ndarray,
                   time_offset: float = 0.0) -> np.ndarray:
    """Exposure-averaged gyro-predicted shifts ``(K, 2)`` relative to ``t = 0``."""
    raise NotImplementedError("Phase 2A")


def estimate_time_offset(image_shifts: np.ndarray, sub_t: np.ndarray, gyro: GyroTrace,
                         search: float = 0.05) -> float:
    """Camera-gyro offset tau (s) by cross-correlating image and gyro rates."""
    raise NotImplementedError("Phase 2A")


def refine_against_hr(capture: Capture, hr_estimate: np.ndarray, **kw) -> Registration:
    """Optional: re-register subpages against an HR estimate."""
    raise NotImplementedError("Phase 2A")
