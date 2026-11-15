"""Sparse forward operator ``A_k`` for reconstruction (Phase 2B).

Independent implementation from the simulator (inverse-crime separation):
separable 1-D erf-integrated box (x) Gaussian PSF weights on the target grid,
not the simulator's 8x integral image.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from .types import SensorConfig, Subpage

__all__ = ["build_operator"]


def build_operator(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int,
                   psf_sigma: float | None = None) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Return ``(A, y)``: rows are the valid (masked) samples of all subpages,
    columns the ``(24 s) * (32 s)`` target pixels (row-major)."""
    raise NotImplementedError("Phase 2B")
