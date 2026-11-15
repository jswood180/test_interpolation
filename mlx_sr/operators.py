"""Sparse forward operator ``A`` for multi-frame reconstruction (Phase 2B).

Independent implementation from the simulator (inverse-crime separation):
separable 1-D erf-integrated box (x) Gaussian PSF weights on the 2x/3x
target grid, not the simulator's 8x integral image.  This module must not
import :mod:`mlx_sr.sensor` or :mod:`mlx_sr.scene` (a test enforces it).

Model
-----
The unknown ``x`` is the scene, piecewise constant on the ``(24 s, 32 s)``
target grid whose cell ``(u, v)`` covers LR scene coords
``[v/s, (v+1)/s) x [u/s, (u+1)/s)`` (the grid of the reference pose, shift
``(0, 0)``; see :mod:`mlx_sr.types`).  Under shift ``(dx, dy)`` the LR pixel
``(i, j)`` reports the mean over its footprint ``[j+dx, j+1+dx) x
[i+dy, i+1+dy)`` of the scene blurred by an isotropic Gaussian PSF of
``psf_sigma`` LR px.  Per axis the sampling kernel is ``K = box (x) G``, whose
cumulative mass is (with ``a`` the footprint's left edge)

``C(x) = sigma * [F((x - a) / sigma) - F((x - a - 1) / sigma)]``,
``F(t) = t * Phi(t) + phi(t)``

(the antiderivative of the normal CDF ``Phi``), so the weight of target cell
``v`` is ``C((v+1)/s) - C(v/s)``.  The 2-D weight is the outer product of
the x and y weights.  Weights are truncated to ``[a - 3 sigma, a + 1 + 3 sigma]``
(the box plus +/-3 sigma), mass falling outside the target grid is folded into
the border cell (replicate boundary), and each 1-D weight vector is
renormalized to sum to 1, so every row of ``A`` sums to exactly 1.

Motion blur within an exposure is not modeled (the exposure-averaged shift is
used), nor is roll.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from scipy.special import ndtr

from .types import SensorConfig, Subpage

__all__ = ["build_operator", "valid_samples", "axis_weights", "PSF_SUPPORT_SIGMAS"]

PSF_SUPPORT_SIGMAS: float = 3.0
"""Half-width of the Gaussian part of the kernel support, in sigmas."""

_SIGMA_MIN: float = 1e-6  # below this the PSF is treated as a delta (pure box)


def _phi_antideriv(t: np.ndarray) -> np.ndarray:
    """``F(t) = t Phi(t) + phi(t)`` (antiderivative of the standard normal CDF)."""
    return t * ndtr(t) + np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)


def axis_weights(left: np.ndarray, n_cells: int, scale: int, sigma: float) -> np.ndarray:
    """1-D sampling weights, shape ``(len(left), n_cells)``.

    Row ``n`` holds the weights of the footprint ``[left[n], left[n] + 1)``
    (LR px) blurred by a Gaussian of ``sigma`` LR px over the target cells
    ``[v/scale, (v+1)/scale)``.  Out-of-grid mass is folded into the border
    cells, the support is truncated to the box +/- ``PSF_SUPPORT_SIGMAS``
    sigma, and rows are renormalized to sum to 1.
    """
    a = np.asarray(left, dtype=float).reshape(-1, 1)
    edges = np.arange(n_cells + 1, dtype=float) / scale  # cell edges, LR px
    if sigma < _SIGMA_MIN:
        C = np.clip(edges[None, :] - a, 0.0, 1.0)
        support = 0.0
    else:
        C = sigma * (_phi_antideriv((edges[None, :] - a) / sigma)
                     - _phi_antideriv((edges[None, :] - a - 1.0) / sigma))
        support = PSF_SUPPORT_SIGMAS * sigma
    C[:, 0] = 0.0  # fold mass left of the grid into cell 0
    C[:, -1] = 1.0  # ... and right of the grid into the last cell
    W = np.diff(C, axis=1)
    # Truncate: keep cells that intersect [a - support, a + 1 + support].
    lo = edges[None, :-1]
    hi = edges[None, 1:]
    keep = (hi > a - support) & (lo < a + 1.0 + support)
    keep[:, 0] |= (a[:, 0] - support) < 0.0  # folded border cells stay
    keep[:, -1] |= (a[:, 0] + 1.0 + support) > n_cells / scale
    W = np.where(keep, np.clip(W, 0.0, None), 0.0)
    W[W < 1e-14] = 0.0
    W /= W.sum(axis=1, keepdims=True)
    return W


def valid_samples(subpages: list[Subpage]) -> list[np.ndarray]:
    """Per subpage, the flat LR pixel indices (row-major, ascending) used as
    rows of ``A``: read, not dead (``mask``) and finite."""
    out = []
    for sp in subpages:
        ok = np.asarray(sp.mask, bool) & np.isfinite(sp.values)
        out.append(np.flatnonzero(ok))
    return out


def build_operator(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int,
                   psf_sigma: float | None = None) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Return ``(A, y)``.

    Rows of ``A`` are the valid samples (``mask`` True and finite) of all
    subpages, in subpage order and row-major pixel order within a subpage
    (the same order as :func:`valid_samples`); columns are the
    ``(24 s) * (32 s)`` target pixels, row-major.  ``y`` holds the sample
    values.  ``shifts`` is ``(K, 2)`` ``(dx, dy)`` in LR px relative to the
    target grid.  ``psf_sigma`` defaults to ``sensor.psf_sigma_px``.
    """
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    if len(subpages) != shifts.shape[0]:
        raise ValueError(f"{len(subpages)} subpages but {shifts.shape[0]} shifts")
    if int(scale) != scale or scale < 1:
        raise ValueError("scale must be a positive integer")
    scale = int(scale)
    sigma = float(sensor.psf_sigma_px if psf_sigma is None else psf_sigma)
    if sigma < 0:
        raise ValueError("psf_sigma must be >= 0")
    rows, cols = sensor.shape
    H, W = rows * scale, cols * scale

    rr, cc, vv, ys = [], [], [], []
    row0 = 0
    for sp, (dx, dy), idx in zip(subpages, shifts, valid_samples(subpages)):
        if idx.size == 0:
            continue
        wx = axis_weights(np.arange(cols) + dx, W, scale, sigma)  # (cols, W)
        wy = axis_weights(np.arange(rows) + dy, H, scale, sigma)  # (rows, H)
        # Pad each pixel's nonzero cells to a common length (zero weights).
        cx, vx = _pad_nonzero(wx)
        cy, vy = _pad_nonzero(wy)
        i, j = np.divmod(idx, cols)
        col = cy[i][:, :, None] * W + cx[j][:, None, :]  # (P, Ly, Lx)
        val = vy[i][:, :, None] * vx[j][:, None, :]
        r = np.broadcast_to((row0 + np.arange(idx.size))[:, None, None], col.shape)
        nz = val != 0.0
        rr.append(r[nz])
        cc.append(col[nz])
        vv.append(val[nz])
        ys.append(np.asarray(sp.values, float).ravel()[idx])
        row0 += idx.size

    if row0 == 0:
        return scipy.sparse.csr_matrix((0, H * W)), np.zeros(0)
    A = scipy.sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                                shape=(row0, H * W))
    A.sum_duplicates()
    return A, np.concatenate(ys)


def _pad_nonzero(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column indices and values of each row's nonzeros, padded with zeros."""
    nnz = (w != 0).sum(axis=1)
    L = int(nnz.max())
    order = np.argsort(w == 0, axis=1, kind="stable")[:, :L]  # nonzeros first
    vals = np.take_along_axis(w, order, axis=1)
    return order, vals
