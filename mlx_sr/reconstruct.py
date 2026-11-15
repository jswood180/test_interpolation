"""Multi-frame reconstruction methods (Phase 2B).

All methods return a :class:`~mlx_sr.types.Reconstruction` whose image is on
the ``(24 s, 32 s)`` target grid defined in :mod:`mlx_sr.types`: target pixel
``(u, v)`` covers LR scene coords ``[v/s, (v+1)/s) x [u/s, (u+1)/s)`` of the
*reference pose*, i.e. the pose for which ``shift == (0, 0)``.  With absolute
true shifts that is the ``t = 0`` grid (``truth_at_scale(capture, s)``); with
registration output relative to subpage ``r`` it is subpage ``r``'s grid
(``truth_at_scale(capture, s, ref_index=r)``).

Methods (``reconstruct(..., method=...)``):

``"bicubic"``
    Single-frame baseline: one subpage pair merged Adafruit-style (or one
    subpage with its chess holes filled), dead pixels filled, then Keys
    bicubic (a = -0.5) upsampling.  Ignores ``shifts``.
``"average"``
    Registered average: every subpage is warped onto the reference LR grid
    and the warped subpages are averaged, then bicubic upsampling.  This is
    the "denoising only, no super-resolution" baseline.
``"nc"``
    Normalized convolution ("drizzle"): Gaussian splat of every sample at its
    registered position on the target grid, divided by the certainty map.
``"ibp"``
    Irani-Peleg iterative back-projection with the exact forward operator
    ``A`` (:func:`mlx_sr.operators.build_operator`), normalized by the column
    sums of ``A``.
``"btv"``
    Robust MAP (Farsiu et al. 2004): Charbonnier-smoothed L1 data term plus a
    bilateral total variation prior, minimized with L-BFGS-B using analytic
    gradients through ``A`` / ``A^T``; optional alternating estimation of
    per-pixel FPN offsets.

Dead pixels (``DEAD_VALUE``) and unread pixels (NaN) are excluded everywhere
through ``Subpage.mask``.  This module must not import :mod:`mlx_sr.sensor`
or :mod:`mlx_sr.scene` (inverse-crime separation).
"""

from __future__ import annotations

import time
import warnings
from typing import Literal

import numpy as np
from scipy import ndimage, optimize

from .operators import build_operator, valid_samples
from .types import Reconstruction, SensorConfig, Subpage

__all__ = [
    "reconstruct",
    "baseline_bicubic",
    "baseline_average",
    "normalized_convolution",
    "ibp",
    "btv_map",
    "btv_value_and_grad",
    "map_objective",
    "default_btv_params",
    "upsample_bicubic",
    "fill_missing",
]

METHODS = ("bicubic", "average", "nc", "ibp", "btv")
_ALIASES = {"btv_map": "btv", "map": "btv", "normalized_convolution": "nc", "drizzle": "nc",
            "baseline_bicubic": "bicubic", "baseline_average": "average"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keys_kernel(t: np.ndarray, a: float = -0.5) -> np.ndarray:
    t = np.abs(t)
    out = np.where(t <= 1, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1, 0.0)
    return np.where((t > 1) & (t < 2), a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a, out)


def _bicubic_matrix(n: int, scale: int) -> np.ndarray:
    """``(n * scale, n)`` Keys cubic-convolution matrix, replicate boundary.

    Target pixel ``u`` has its center at LR coordinate ``(u + 0.5)/scale``,
    i.e. at fractional LR index ``(u + 0.5)/scale - 0.5`` (LR pixel ``i`` is
    centered at ``i + 0.5``).
    """
    t = (np.arange(n * scale) + 0.5) / scale - 0.5
    base = np.floor(t).astype(int)
    M = np.zeros((n * scale, n))
    rows = np.arange(n * scale)
    for o in (-1, 0, 1, 2):
        idx = base + o
        w = _keys_kernel(t - idx)
        np.add.at(M, (rows, np.clip(idx, 0, n - 1)), w)
    return M


def upsample_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    """Keys bicubic (a = -0.5) upsampling of a complete LR image onto the
    target grid (pixel centers aligned as in :mod:`mlx_sr.types`)."""
    img = np.asarray(img, float)
    if scale == 1:
        return img.copy()
    Cy = _bicubic_matrix(img.shape[0], scale)
    Cx = _bicubic_matrix(img.shape[1], scale)
    return Cy @ img @ Cx.T


def fill_missing(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill pixels where ``mask`` is False by repeated normalized convolution
    with the 4-neighbor (cross) kernel; known pixels are returned unchanged.
    For a chess subpage this is the 4-neighbor mean."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("no valid pixels to fill from")
    out = np.where(mask, values, 0.0).astype(float)
    known = mask.copy()
    k = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    while not known.all():
        num = ndimage.convolve(np.where(known, out, 0.0), k, mode="constant")
        den = ndimage.convolve(known.astype(float), k, mode="constant")
        new = (~known) & (den > 0)
        if not new.any():  # isolated region: fall back to diagonal neighbors
            k = np.ones((3, 3))
            continue
        out[new] = num[new] / den[new]
        known = known | new
    return out


def _sample_positions(n: int, shift: float, scale: int) -> np.ndarray:
    """Fractional target-grid index of the centers of LR pixels ``0..n-1``."""
    return (np.arange(n) + 0.5 + shift) * scale - 0.5


def _gauss_splat_weights(n_lr: int, n_hr: int, shift: float, scale: int, sigma_hr: float,
                         truncate: float = 3.0) -> np.ndarray:
    """``(n_lr, n_hr)`` Gaussian weights between LR sample centers and target pixel centers."""
    pos = _sample_positions(n_lr, shift, scale)[:, None]
    d = np.arange(n_hr)[None, :] - pos
    w = np.exp(-0.5 * (d / sigma_hr) ** 2)
    w[np.abs(d) > truncate * sigma_hr] = 0.0
    return w


def _check_inputs(subpages: list[Subpage], shifts: np.ndarray | None) -> np.ndarray | None:
    if len(subpages) == 0:
        raise ValueError("need at least one subpage")
    if shifts is None:
        return None
    shifts = np.asarray(shifts, float).reshape(-1, 2)
    if shifts.shape[0] != len(subpages):
        raise ValueError(f"{len(subpages)} subpages but {shifts.shape[0]} shifts")
    return shifts


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def reconstruct(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, method: str,
                scale: int = 2, **kw) -> Reconstruction:
    """Dispatch to a method: ``bicubic``, ``average``, ``nc``, ``ibp``, ``btv``.

    Extra keyword arguments go to the method.  ``info["runtime_s"]`` holds the
    wall time.
    """
    m = _ALIASES.get(method, method)
    t0 = time.perf_counter()
    if m == "bicubic":
        rec = baseline_bicubic(subpages, sensor, scale, **kw)
    elif m == "average":
        rec = baseline_average(subpages, shifts, sensor, scale, **kw)
    elif m == "nc":
        rec = normalized_convolution(subpages, shifts, sensor, scale, **kw)
    elif m == "ibp":
        rec = ibp(subpages, shifts, sensor, scale, **kw)
    elif m == "btv":
        rec = btv_map(subpages, shifts, sensor, scale, **kw)
    else:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    rec.info["runtime_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_bicubic(subpages: list[Subpage], sensor: SensorConfig, scale: int = 2,
                     mode: Literal["merged", "single"] = "merged", ref_index: int = 0) -> Reconstruction:
    """Single-frame bicubic baseline.

    ``mode="merged"``: the full frame containing subpage ``ref_index``
    (subpages ``2f`` and ``2f + 1`` with ``f = ref_index // 2``, or the
    nearest available partner) merged Adafruit-style, i.e. ignoring that the
    two halves were exposed at different times and poses.  ``mode="single"``:
    subpage ``ref_index`` alone with its unread chess pixels filled by the
    4-neighbor mean.  Dead pixels are filled from their neighbors in both
    modes.  Then Keys bicubic upsampling.
    """
    _check_inputs(subpages, None)
    if mode == "merged":
        a = 2 * (ref_index // 2)
        pair = [a, a + 1] if a + 1 < len(subpages) else [a - 1, a] if a >= 1 else [a]
        values = np.full(sensor.shape, np.nan)
        mask = np.zeros(sensor.shape, bool)
        for k in pair:
            sp = subpages[k]
            m = np.asarray(sp.mask, bool) & np.isfinite(sp.values) & ~mask
            values[m] = sp.values[m]
            mask |= m
    elif mode == "single":
        sp = subpages[ref_index]
        mask = np.asarray(sp.mask, bool) & np.isfinite(sp.values)
        values = np.where(mask, sp.values, np.nan)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    lr = fill_missing(values, mask)
    img = upsample_bicubic(lr, scale)
    return Reconstruction(image=img, certainty=None, scale=scale, method="bicubic",
                          info={"mode": mode, "lr": lr})


def baseline_average(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig,
                     scale: int = 2, sigma: float = 0.2) -> Reconstruction:
    """Registered average, then bicubic upsampling (denoising-only baseline).

    Every valid sample is placed at its registered position on the reference
    LR grid and the samples are averaged per LR pixel with a Gaussian weight
    of ``sigma`` LR px (normalized convolution at ``scale=1``), which is
    shift-and-add at the native resolution.  The averaged LR image is then
    upsampled with Keys bicubic, so the result has the noise reduction of
    multi-frame averaging but no resolution beyond the LR grid.  (A
    fill-then-cubic-warp variant was measured about 0.8 dB worse on the chart,
    so this is the stronger, fairer baseline.)  ``sigma=0.2`` LR px: smaller
    values gain ~0.1 dB but leave more pixels uncovered at small K; with
    asymmetric shifts the Gaussian weighting can bias positions by up to
    ~0.1 LR px.  LR pixels with no sample within 3 sigma are filled from
    their neighbors.
    """
    shifts = _check_inputs(subpages, shifts)
    lr_rec = normalized_convolution(subpages, shifts, sensor, 1, sigma=sigma)
    lr = lr_rec.image
    img = upsample_bicubic(lr, scale)
    return Reconstruction(image=img, certainty=None, scale=scale, method="average",
                          info={"lr": lr, "lr_certainty": lr_rec.certainty, "sigma_lr": float(sigma)})


# ---------------------------------------------------------------------------
# Normalized convolution
# ---------------------------------------------------------------------------


def default_nc_sigma(n_subpages: int) -> float:
    """Default splat sigma in LR px: about the mean sample spacing,
    ``0.9 / sqrt(K / 2)`` clipped to ``[0.3, 0.6]`` (K/2 samples per LR px^2)."""
    density = max(n_subpages, 1) / 2.0
    return float(np.clip(0.9 / np.sqrt(density), 0.3, 0.6))


def normalized_convolution(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig,
                           scale: int = 2, sigma: float | None = None) -> Reconstruction:
    """Gaussian-splat normalized convolution ("drizzle").

    Every valid sample is splatted at its registered center
    ``((j + 0.5 + dx) s - 0.5, (i + 0.5 + dy) s - 0.5)`` (target-grid index)
    with an isotropic Gaussian of ``sigma`` LR px (truncated at 3 sigma);
    the image is the weighted sum divided by the certainty (sum of weights).
    ``certainty`` is that raw weight sum.  Pixels with zero certainty (only
    possible for very few, clustered samples) are filled from their
    neighbors.  ``sigma`` defaults to :func:`default_nc_sigma`.
    """
    shifts = _check_inputs(subpages, shifts)
    rows, cols = sensor.shape
    H, W = rows * scale, cols * scale
    if sigma is None:
        sigma = default_nc_sigma(len(subpages))
    sig_hr = float(sigma) * scale
    num = np.zeros((H, W))
    cert = np.zeros((H, W))
    for sp, (dx, dy), idx in zip(subpages, shifts, valid_samples(subpages)):
        if idx.size == 0:
            continue
        m = np.zeros(rows * cols)
        m[idx] = 1.0
        m = m.reshape(sensor.shape)
        v = np.where(m > 0, np.nan_to_num(sp.values), 0.0)
        gx = _gauss_splat_weights(cols, W, dx, scale, sig_hr)
        gy = _gauss_splat_weights(rows, H, dy, scale, sig_hr)
        num += gy.T @ v @ gx
        cert += gy.T @ m @ gx
    good = cert > 1e-12
    img = np.zeros((H, W))
    img[good] = num[good] / cert[good]
    if not good.all():
        img = fill_missing(img, good)
    return Reconstruction(image=img, certainty=cert, scale=scale, method="nc",
                          info={"sigma_lr": float(sigma)})


# ---------------------------------------------------------------------------
# Iterative back-projection
# ---------------------------------------------------------------------------


def _initial_estimate(init, subpages, shifts, sensor, scale) -> np.ndarray:
    H, W = sensor.shape[0] * scale, sensor.shape[1] * scale
    if isinstance(init, np.ndarray):
        if init.shape != (H, W):
            raise ValueError(f"init shape {init.shape} != {(H, W)}")
        return init.astype(float).copy()
    if init == "nc":
        return normalized_convolution(subpages, shifts, sensor, scale).image
    if init == "average":
        return baseline_average(subpages, shifts, sensor, scale).image
    if init == "bicubic":
        return baseline_bicubic(subpages, sensor, scale).image
    raise ValueError(f"unknown init {init!r}")


def ibp(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int = 2,
        n_iter: int = 20, step: float = 1.0, init: str | np.ndarray = "nc",
        psf_sigma: float | None = None) -> Reconstruction:
    """Irani-Peleg iterative back-projection.

    ``x <- x + step * A^T (y - A x) / (A^T 1)``.  Because every row of ``A``
    sums to 1 and ``A >= 0``, ``A D^-1 A^T`` (``D = diag(A^T 1)``) has
    eigenvalues in ``[0, 1]``, so for ``0 < step <= 1`` the residual norm
    ``||y - A x||`` is non-increasing.  ``info["residual_norms"]`` holds the
    ``n_iter + 1`` residual norms (initial estimate first).  No
    regularization: IBP amplifies noise with many iterations.
    """
    shifts = _check_inputs(subpages, shifts)
    A, y = build_operator(subpages, shifts, sensor, scale, psf_sigma)
    H, W = sensor.shape[0] * scale, sensor.shape[1] * scale
    x = _initial_estimate(init, subpages, shifts, sensor, scale).ravel()
    colsum = np.asarray(A.sum(axis=0)).ravel()
    inv = np.where(colsum > 1e-12, 1.0 / np.maximum(colsum, 1e-12), 0.0)
    AT = A.T.tocsr()
    r = y - A @ x
    norms = [float(np.linalg.norm(r))]
    for _ in range(n_iter):
        x = x + step * inv * (AT @ r)
        r = y - A @ x
        norms.append(float(np.linalg.norm(r)))
    return Reconstruction(image=x.reshape(H, W), certainty=colsum.reshape(H, W), scale=scale,
                          method="ibp", info={"residual_norms": np.asarray(norms), "n_iter": n_iter})


# ---------------------------------------------------------------------------
# BTV-MAP
# ---------------------------------------------------------------------------


def btv_offsets(p: int, alpha: float) -> list[tuple[int, int, float]]:
    """BTV shift set ``(dy, dx, alpha^(|dy|+|dx|))``: every unordered pair of
    neighbors in the ``(2p+1)^2`` window exactly once (``dy in [0, p]``,
    ``dx in [-p, p]``, excluding ``(0, dx <= 0)``).  This is the full
    half-window, slightly more isotropic than Farsiu's ``l + m >= 0`` index
    set (which omits e.g. ``(1, -2)``); the decay ``alpha^(|l|+|m|)`` is his."""
    out = []
    for dy in range(0, p + 1):
        for dx in range(-p, p + 1):
            if dy == 0 and dx <= 0:
                continue
            out.append((dy, dx, float(alpha) ** (abs(dy) + abs(dx))))
    return out


def btv_value_and_grad(x: np.ndarray, p: int = 2, alpha: float = 0.7, eps: float = 0.05
                       ) -> tuple[float, np.ndarray]:
    """Charbonnier-smoothed bilateral TV of image ``x`` and its gradient.

    ``BTV(x) = sum_{(dy,dx)} alpha^(|dy|+|dx|) sum_{u,v} rho(x[u+dy, v+dx] - x[u, v])``
    with ``rho(d) = sqrt(d^2 + eps^2) - eps``, summed over pixel pairs that
    both lie inside the image (no wrap-around).
    """
    x = np.asarray(x, float)
    H, W = x.shape
    g = np.zeros_like(x)
    f = 0.0
    for dy, dx, w in btv_offsets(p, alpha):
        if dy >= H or abs(dx) >= W:
            continue
        # Pairs (u, v) -> (u + dy, v + dx), both inside.
        r0, r1 = 0, H - dy
        c0, c1 = max(0, -dx), W - max(0, dx)
        a = x[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
        b = x[r0:r1, c0:c1]
        d = a - b
        s = np.sqrt(d * d + eps * eps)
        f += w * float(np.sum(s - eps))
        gd = w * d / s
        g[r0 + dy:r1 + dy, c0 + dx:c1 + dx] += gd
        g[r0:r1, c0:c1] -= gd
    return f, g


FPN_SHRINK: bool = True
"""Default for ``btv_map(fpn_shrink=...)``: MAP (shrunk) rather than plain-mean FPN update."""


def _charbonnier(r: np.ndarray, eps: float) -> tuple[float, np.ndarray]:
    s = np.sqrt(r * r + eps * eps)
    return float(np.sum(s - eps)), r / s


def map_objective(x: np.ndarray, A, y: np.ndarray, shape: tuple[int, int], lam: float,
                  alpha: float = 0.7, p: int = 2, eps_data: float = 0.3, eps_btv: float = 0.2,
                  AT=None) -> tuple[float, np.ndarray]:
    """BTV-MAP objective and its analytic gradient for flat ``x``.

    ``f(x) = sum rho_d(A x - y) + lam * BTV(x)`` and
    ``grad f = A^T rho_d'(A x - y) + lam * grad BTV(x)``, with Charbonnier
    ``rho_d`` of ``eps_data`` and BTV as in :func:`btv_value_and_grad` with
    ``eps_btv``.  ``AT`` may pass a precomputed ``A.T`` (CSR) for speed.
    """
    fd, gr = _charbonnier(A @ x - y, eps_data)
    fb, gb = btv_value_and_grad(x.reshape(shape), p, alpha, eps_btv)
    gd = (A.T if AT is None else AT) @ gr
    return fd + lam * fb, gd + lam * gb.ravel()


DEFAULT_LAM_PER_K: float = 0.042
"""Default BTV weight per kelvin of noise sigma (see :func:`default_btv_params`)."""

DEFAULT_EPS_DATA_PER_K: float = 0.4
"""Default data-term Charbonnier epsilon per kelvin of noise sigma."""

DEFAULT_EPS_BTV_K: float = 0.2
"""Default BTV Charbonnier epsilon in kelvin (scene-difference scale)."""

NOISE_FLOOR_K: float = 0.25
"""Lower bound of :func:`noise_scale`, so the default ``lam`` never vanishes."""


def noise_scale(sensor: SensorConfig) -> float:
    """Per-sample error scale ``max(sqrt(noise_sigma_k^2 + fpn_sigma_k^2), NOISE_FLOOR_K)`` K."""
    return max(float(np.hypot(sensor.noise_sigma_k, sensor.fpn_sigma_k)), NOISE_FLOOR_K)


def default_btv_params(sensor: SensorConfig, scale: int = 2) -> dict:
    """Default ``lam``, ``eps_charbonnier`` and ``eps_btv`` for :func:`btv_map`.

    ``lam = 0.042 sigma (2 / scale)^2``, ``eps_charbonnier = 0.4 sigma`` and
    ``eps_btv = 0.2`` K, with ``sigma`` = :func:`noise_scale`.

    - Tuned on the BAB chart (PSF 0.4 px, 2x, K = 16, 8 Hz, sigma = 0.71 K:
      lam = 0.030, eps = 0.29 K) as a balance between PSNR, which is flat
      within about 0.1 dB over lam = 0.03-0.06, and contrast above native
      Nyquist, which drops quickly with lam.
    - Scaling with sigma kept BTV ahead of both baselines on the chart from
      2 Hz (sigma 0.37 K) to 32 Hz (sigma 1.42 K).
    - The ``NOISE_FLOOR_K = 0.25`` floor keeps the part of the null space of
      ``A`` that the data cannot see (e.g. target-Nyquist stripes) regularized
      for (nearly) noiseless input: with a 0.05 K floor a noiseless PSF-0
      textured scene came out worse than bicubic.
    - The ``(2 / scale)^2`` factor: at 3x, ``lam / 2.25`` gave +0.4 dB and
      3x the bar contrast at 0.6 cyc/LR px on the chart.
    - Not universal: on the low-contrast (2 K) textured scene at 8 Hz BTV
      only ties the registered average at best; tune ``lam`` per scene class
      when that matters.
    """
    sig = noise_scale(sensor)
    return dict(lam=DEFAULT_LAM_PER_K * sig * (2.0 / scale) ** 2,
                eps_charbonnier=DEFAULT_EPS_DATA_PER_K * sig, eps_btv=DEFAULT_EPS_BTV_K)


def btv_map(subpages: list[Subpage], shifts: np.ndarray, sensor: SensorConfig, scale: int = 2,
            lam: float | None = None, alpha: float = 0.7, p: int = 2,
            eps_charbonnier: float | None = None, init: str | np.ndarray = "nc",
            estimate_fpn: bool = False, n_outer: int = 3, psf_sigma: float | None = None,
            maxiter: int = 400, eps_btv: float | None = None, fpn_shrink: bool = FPN_SHRINK,
            fpn_sigma_prior: float | None = None) -> Reconstruction:
    """Robust MAP super-resolution (Farsiu et al. 2004).

    Minimizes ``sum_r rho_d(A x + b[pix(r)] - y)_r + lam * BTV(x)`` with
    ``rho(t) = sqrt(t^2 + eps^2) - eps`` (Charbonnier, a smoothed L1; see
    :func:`btv_value_and_grad` for BTV) using L-BFGS-B with analytic gradients
    ``A^T rho_d'(r) + lam * grad BTV``.  ``b = 0`` unless ``estimate_fpn``.

    Defaults: ``lam``, ``eps_charbonnier`` and ``eps_btv`` of None come from
    :func:`default_btv_params` (scaled by the sensor noise); ``alpha=0.7`` and
    ``p=2`` are Farsiu's decay and window; ``init="nc"``; ``maxiter=400``
    L-BFGS-B iterations per solve (it typically converges in 100-300).

    ``estimate_fpn=True`` alternates ``n_outer`` solves for ``x`` with an
    update of the per-LR-pixel offsets ``b``: the masked mean residual of each
    pixel, minus the mean over all read pixels (zero-mean constraint, which
    removes the offset ambiguity between ``b`` and ``x``; ``A``'s rows sum
    to 1).  With
    ``fpn_shrink=True`` (default) the mean is the MAP estimate under a
    Gaussian prior ``b ~ N(0, sf^2)``:
    ``b = sum_k r_k / (n + (noise_sigma_k / sf)^2)`` where ``sf`` is
    ``fpn_sigma_prior`` (default ``sensor.fpn_sigma_k``, i.e. the assumed
    residual FPN after calibration).  With ``sf == 0`` the MAP estimate is
    ``b = 0``, so FPN estimation is skipped (with a warning).  This is a
    deliberate refinement of the plain masked mean: with only
    ``n = K/2`` reads per pixel the plain mean (``fpn_shrink=False``) mostly
    estimates temporal noise: on the BAB chart at
    8 Hz, K = 16, FPN 0.2 K its offset error was 0.27-0.37 K (worse than not
    estimating) and grew with ``n_outer``, while the shrunk update gave
    0.17 K and lowered the flat-region sigma for every tested seed.  The offsets
    used in the final solve are in ``info["fpn"]`` (``(24, 32)``, NaN where
    never read).
    """
    shifts = _check_inputs(subpages, shifts)
    A, y = build_operator(subpages, shifts, sensor, scale, psf_sigma)
    AT = A.T.tocsr()
    H, W = sensor.shape[0] * scale, sensor.shape[1] * scale
    dflt = default_btv_params(sensor, scale)
    lam = dflt["lam"] if lam is None else float(lam)
    eps_d = dflt["eps_charbonnier"] if eps_charbonnier is None else float(eps_charbonnier)
    eps_b = dflt["eps_btv"] if eps_btv is None else float(eps_btv)
    if min(lam, eps_d, eps_b) < 0 or eps_d == 0 or eps_b == 0:
        raise ValueError("lam must be >= 0 and the Charbonnier epsilons > 0")
    x = _initial_estimate(init, subpages, shifts, sensor, scale).ravel()

    pix = np.concatenate(valid_samples(subpages))
    npix = sensor.shape[0] * sensor.shape[1]
    counts = np.bincount(pix, minlength=npix).astype(float)
    seen = counts > 0
    b = np.zeros(npix)
    shrink = 0.0
    fpn_requested = estimate_fpn
    if estimate_fpn and fpn_shrink:
        sf = float(sensor.fpn_sigma_k if fpn_sigma_prior is None else fpn_sigma_prior)
        if sf <= 0.0:
            warnings.warn("estimate_fpn with a zero FPN prior: the MAP offsets are 0, skipping "
                          "FPN estimation", RuntimeWarning, stacklevel=2)
            estimate_fpn = False
        else:
            shrink = (sensor.noise_sigma_k / sf) ** 2

    def fun(xv: np.ndarray, yc: np.ndarray) -> tuple[float, np.ndarray]:
        return map_objective(xv, A, yc, (H, W), lam, alpha, p, eps_d, eps_b, AT=AT)

    n_solve = max(int(n_outer), 1) if estimate_fpn else 1
    history = []
    for it in range(n_solve):
        if it > 0:  # FPN step
            r = y - A @ x
            b = np.zeros(npix)
            b[seen] = np.bincount(pix, weights=r, minlength=npix)[seen] / (counts[seen] + shrink)
            b[seen] -= b[seen].mean()
        res = optimize.minimize(fun, x, args=(y - b[pix],), jac=True, method="L-BFGS-B",
                                options=dict(maxiter=maxiter, maxcor=10))
        x = res.x
        history.append(dict(fun=float(res.fun), nit=int(res.nit), status=int(res.status)))
    info = dict(lam=lam, alpha=alpha, p=p, eps_charbonnier=eps_d, eps_btv=eps_b,
                psf_sigma=float(sensor.psf_sigma_px if psf_sigma is None else psf_sigma),
                history=history, estimate_fpn=estimate_fpn, init=init if isinstance(init, str) else "array")
    if fpn_requested:
        info["fpn"] = np.where(seen, b, np.nan).reshape(sensor.shape)
    colsum = np.asarray(A.sum(axis=0)).ravel().reshape(H, W)
    return Reconstruction(image=x.reshape(H, W), certainty=colsum, scale=scale, method="btv", info=info)
