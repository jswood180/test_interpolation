"""Image-quality metrics (Phase 2C).

``psnr`` is a minimal working version for the Phase 1 tests; Phase 2C may
replace the implementation but must keep the signature.  ROI arguments are
:class:`mlx_sr.types.Roi` in LR scene coords; ``scale`` and ``shift`` describe
the image's target grid (see ``Roi.to_scale``).
"""

from __future__ import annotations

import numpy as np

from .types import BarGroup, Roi

__all__ = [
    "psnr",
    "ssim",
    "slanted_edge_mtf",
    "bar_contrast",
    "hotspot_error",
    "flat_sigma",
    "registration_rms",
]


def _crop(a: np.ndarray, border: int) -> np.ndarray:
    return a[border:-border, border:-border] if border > 0 else a


def psnr(img: np.ndarray, ref: np.ndarray, *, data_range: float | None = None, border: int = 0) -> float:
    """PSNR in dB.  ``data_range`` defaults to ``ref.max() - ref.min()`` (after
    cropping ``border`` pixels).  Identical images give ``inf``."""
    a = _crop(np.asarray(img, float), border)
    b = _crop(np.asarray(ref, float), border)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    dr = float(b.max() - b.min()) if data_range is None else float(data_range)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(dr * dr / mse))


def ssim(img: np.ndarray, ref: np.ndarray, *, data_range: float | None = None, border: int = 0) -> float:
    raise NotImplementedError("Phase 2C")


def slanted_edge_mtf(img: np.ndarray, roi: Roi, scale: int, shift=(0.0, 0.0)) -> dict:
    """ISO-12233-style MTF.  Returns at least ``{"freq", "mtf", "mtf50", "mtf_at"}``
    (``mtf_at`` a callable); frequencies in cycles per LR pixel."""
    raise NotImplementedError("Phase 2C")


def bar_contrast(img: np.ndarray, bar_specs: list[BarGroup], scale: int = 1, shift=(0.0, 0.0)) -> dict:
    raise NotImplementedError("Phase 2C")


def hotspot_error(img: np.ndarray, truth_peak: float, roi: Roi, scale: int = 1, shift=(0.0, 0.0)) -> float:
    raise NotImplementedError("Phase 2C")


def flat_sigma(img: np.ndarray, roi: Roi, scale: int = 1, shift=(0.0, 0.0)) -> float:
    raise NotImplementedError("Phase 2C")


def registration_rms(est: np.ndarray, true: np.ndarray) -> float:
    """RMS shift error after removing the mean offset between ``est`` and ``true``."""
    raise NotImplementedError("Phase 2C")
