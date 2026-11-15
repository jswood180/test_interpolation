"""Sub-pixel registration of subpages (Phase 2A).

All shifts follow the convention in :mod:`mlx_sr.types`: ``(dx, dy)`` in LR
pixels, (x, y) order.  ``skimage.registration.phase_cross_correlation``
returns ``(row, col)``; it is swapped here.  Estimated shifts are relative to
subpage ``ref_index`` (whose own entry is exactly zero).

Methods of :func:`register`
---------------------------
``"phase"``
    Upsampled cross-correlation (``phase_cross_correlation`` with
    ``normalization=None``) of Hann-windowed, hole-filled images.  Per
    subpage by default; ``merged=True`` registers merged full frames and
    assigns each frame's shift to both of its subpages (only valid when the
    shift is constant within a frame).
``"gn"``
    Masked Gauss-Newton translation (plus a brightness offset) of each raw
    chess subpage against a cubic-spline interpolant of a reference,
    initialized by phase correlation.  Only the pixels a subpage actually
    read enter the cost.  Pass 1 uses the filled, slightly blurred
    reference subpage; pass 2 (default) re-registers every subpage against
    a leave-one-out multi-frame reference (normalized convolution of all
    other subpages at their pass-1 shifts) and re-anchors on the reference
    subpage.
``"gyro"``
    Gyro integration only (``time_offset`` and ``bias`` are taken from the
    keyword arguments, default 0 / none); no image data is used.
``"hybrid"``
    Gyro seed -> Gauss-Newton refinement -> gyro-bias fit -> multi-frame
    re-registration -> per-frame accept (fused with the gyro) or fallback
    to the gyro (see :func:`register`).  When no ``time_offset`` is given
    it is estimated from image-only shifts with
    :func:`estimate_time_offset` and used only if it is well determined.

Confidence
----------
``Registration.confidence`` is a precision in px^-2 for every method:
roughly ``1 / sigma^2`` of the worst-direction 1-sigma shift error.  For
image methods it is the minimum eigenvalue of the noise-normalized
Gauss-Newton Hessian ``(J^T J - E[J_n^T J_n]) / sigma_r^2`` evaluated at the
estimate, where ``J`` holds the reference gradients (offset parameter
profiled out), ``E[J_n^T J_n]`` is the expected contribution of the
reference's own noise to ``J^T J`` (without this correction pure noise
looks like texture and gets a high confidence) and ``sigma_r`` is the RMS
residual.  In simulation the realized errors of single subpages are about
1-2x the predicted sigma (interpolation and fill errors are not in the
model).  For gyro shifts it is ``1 / var`` of the predicted angle-random-
walk, bias-drift, time-offset and floor error; fused hybrid frames add the
two precisions.  The reference subpage gets ``CONF_REF``.
``info["sigma_px"] = 1 / sqrt(confidence)``.

Failure modes (measured in simulation)
--------------------------------------
- Fixed-pattern locking: FPN is sensor-locked, so it pulls every subpage
  toward zero shift relative to the reference.  On low-contrast scenes it
  biases the image-derived gyro-bias fit; the hybrid's FPN-estimation pass
  removes part of it (low-contrast 0.15 K-noise case: hybrid RMS
  0.13-0.19 px without it, 0.04-0.12 px with it).
- Weak images: at 0.3 K contrast with the default 8 Hz BAB noise the image
  shifts are garbage (2-3.5 px RMS, phase peaks anywhere) and every
  confidence is 0; the hybrid then returns the uncorrected gyro, whose
  error is the unknown bias drift (0.2-1 px over 2 s for a 0.05-1 deg/s
  bias).  At intermediate SNR (images good to about 0.3-0.5 px but
  confidence still 0) this raw gyro can be worse than image-only ``"gn"``.
- Time offset: only observable when the motion has enough rate content
  (wobble, strong tremor at high refresh); with natural motion at 8 Hz the
  estimate is rejected and the prior (0) is used.

Inverse-crime guard: only ``capture.subpages``, ``capture.gyro`` (stamps
and rates, not its truth fields ``bias`` / ``time_offset_s``) and
``capture.sensor`` (datasheet-like configuration) are used here, never
``true_shift``, ``motion``, ``fpn``, ``scene_hr`` or ``meta``.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable, Literal

import numpy as np
from scipy import ndimage

from .types import Capture, GyroTrace, Registration, SensorConfig, Subpage

__all__ = [
    "register",
    "fill_masked",
    "fill_subpage",
    "merged_frames",
    "spline_interpolant",
    "phase_correlation",
    "gauss_newton_translation",
    "integrate_gyro",
    "estimate_time_offset",
    "refine_against_hr",
    "CONF_REF",
    "NOISE_HESSIAN_MARGIN",
]

CONF_REF: float = 1e6
"""Confidence (px^-2) given to the reference subpage, whose shift is zero by definition."""

NOISE_HESSIAN_MARGIN: float = 1.5
"""Factor on the expected reference-noise Hessian subtracted in the image
confidence.  The noise term fluctuates by tens of percent from frame to
frame (the gradient samples are strongly correlated), so subtracting only
its mean lets pure-noise frames reach sigma ~ 0.5 px; the margin requires
the scene's gradient energy to exceed the noise's by 50 % before any
confidence is reported."""

_DEG = math.pi / 180.0
_CONF_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def fill_masked(values: np.ndarray, mask: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """Fill pixels where ``mask`` is False by normalized convolution.

    ``out = conv(v * m) / conv(m)`` with a Gaussian of ``sigma`` px at the
    holes; valid pixels keep their values.  For a chess subpage and the
    default ``sigma`` this is essentially the mean of the four read
    neighbors.  Holes without valid neighbors are filled with a doubled
    sigma, and finally with the mean of the valid pixels.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not mask.any():
        raise ValueError("cannot fill an image without valid pixels")
    v = np.where(mask, values, 0.0)
    m = mask.astype(float)
    out = np.where(mask, values, np.nan)
    holes = ~mask
    s = float(sigma)
    while holes.any() and s <= 2.0 * max(values.shape):
        num = ndimage.gaussian_filter(v, s, mode="mirror", truncate=3.0)
        den = ndimage.gaussian_filter(m, s, mode="mirror", truncate=3.0)
        ok = holes & (den > 1e-6)
        out[ok] = num[ok] / den[ok]
        holes &= ~ok
        s *= 2.0
    if holes.any():
        out[holes] = values[mask].mean()
    return out


def fill_subpage(sp: Subpage, sigma: float = 0.7) -> np.ndarray:
    """Fill unread/dead pixels of a subpage by normalized convolution of their neighbors."""
    return fill_masked(sp.values, sp.mask, sigma)


def merged_frames(capture: Capture) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merged ``(values, mask)`` per full frame (ordered by ``frame_index``).

    Each pixel comes from whichever subpage of the frame read it (ignoring
    the time difference between the subpages); ``values`` is NaN where
    ``mask`` is False (unread or dead).
    """
    frames: dict[int, list[Subpage]] = {}
    for sp in capture.subpages:
        frames.setdefault(sp.frame_index, []).append(sp)
    out = []
    for f in sorted(frames):
        vals = np.full(capture.sensor.shape, np.nan)
        mask = np.zeros(capture.sensor.shape, dtype=bool)
        for sp in frames[f]:
            take = sp.mask & ~mask
            vals[take] = sp.values[take]
            mask |= sp.mask
        out.append((vals, mask))
    return out


def spline_interpolant(img: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Cubic-spline interpolant ``fn(x, y)`` of a finite image.

    ``x`` is the column coordinate and ``y`` the row coordinate in pixel-
    index units (``fn(j, i) == img[i, j]``); mirror boundary.
    """
    img = np.asarray(img, dtype=float)
    if not np.all(np.isfinite(img)):
        raise ValueError("spline_interpolant needs a finite image (fill it first)")
    coeffs = ndimage.spline_filter(img, order=3, mode="mirror")

    def fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(coeffs, [np.asarray(y, float), np.asarray(x, float)],
                                       order=3, mode="mirror", prefilter=False)

    return fn


def phase_correlation(ref: np.ndarray, img: np.ndarray, upsample_factor: int = 20, *,
                      iters: int = 2) -> np.ndarray:
    """Upsampled phase correlation on Hann-windowed images -> ``(dx, dy)``.

    ``ref`` and ``img`` must be finite (fill them first).  Uses
    ``skimage.registration.phase_cross_correlation(normalization=None)`` on
    mean-removed, Hann-windowed images.  The window is fixed to the array,
    not to the content, which biases the estimate toward zero; with
    ``iters > 1`` the image is resampled by the current estimate (cubic
    spline) and the residual shift is added, which removes most of that
    bias.  Returns the shift of ``img`` relative to ``ref`` in the
    :mod:`mlx_sr.types` convention (``img[i, j]`` samples the scene at
    ``ref``'s coordinate ``(j + dx, i + dy)``).
    """
    from skimage.filters import window
    from skimage.registration import phase_cross_correlation

    ref = np.asarray(ref, dtype=float)
    img = np.asarray(img, dtype=float)
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(img))):
        raise ValueError("phase_correlation needs finite images (fill them first)")
    w = window("hann", ref.shape)
    a = (ref - ref.mean()) * w
    total = np.zeros(2)
    cur = img
    fn = spline_interpolant(img) if iters > 1 else None
    rows, cols = img.shape
    jj, ii = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    for it in range(max(1, iters)):
        b = (cur - cur.mean()) * w
        sh, _, _ = phase_cross_correlation(a, b, upsample_factor=upsample_factor, normalization=None)
        # skimage returns the (row, col) shift that registers img onto ref;
        # for img(x) = ref(x + d) that is exactly d.
        total += np.array([sh[1], sh[0]], dtype=float)
        if it + 1 < iters:
            # Resample img so that it is aligned with ref under the current estimate.
            cur = fn(jj - total[0], ii - total[1]).reshape(rows, cols)
    return total


# ---------------------------------------------------------------------------
# Gauss-Newton translation
# ---------------------------------------------------------------------------


def gauss_newton_translation(ref_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], img: np.ndarray,
                             mask: np.ndarray, init: np.ndarray, iters: int = 20, tol: float = 1e-4,
                             *, border: float = 1.0, noise_hessian: np.ndarray | None = None,
                             max_step: float = 1.0, fd_step: float = 1e-2,
                             return_info: bool = False):
    """Masked Gauss-Newton translation refinement -> ``(shift, confidence)``.

    Model: ``img[i, j] = ref_fn(j + dx, i + dy) + c`` over the pixels where
    ``mask`` is True (``c`` is a brightness offset, solved jointly and
    profiled out of the confidence).  ``ref_fn`` takes column/row
    coordinates in the reference's pixel-index units (see
    :func:`spline_interpolant`); gradients are central differences of
    ``ref_fn`` with step ``fd_step``.  Pixels whose sample point falls
    within ``border`` px of the reference edge are excluded at each
    iteration.  Steps longer than ``max_step`` are clipped and a step that
    increases the mean squared residual is halved (up to 6 times).

    ``noise_hessian``: optional ``(2, 2)`` expected per-pixel contribution
    of the reference noise to ``J^T J`` (K^2/px^2); it is subtracted
    (times the number of pixels) before the eigenvalue is taken.

    Returns ``(shift (2,), confidence)`` where confidence is the minimum
    eigenvalue of the noise-normalized translation Hessian (px^-2, see the
    module docstring).  With ``return_info=True`` a third element is a dict
    with ``offset``, ``sigma_resid``, ``n_used``, ``n_iter``,
    ``converged`` and ``lambda_min_raw`` (min eigenvalue of ``J^T J``).
    """
    img = np.asarray(img, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(img)
    rows, cols = img.shape
    ii, jj = np.nonzero(mask)
    v_all = img[ii, jj]
    h = float(fd_step)

    def evaluate(s: np.ndarray, grad: bool = True):
        x = jj + s[0]
        y = ii + s[1]
        ok = (x >= border) & (x <= cols - 1 - border) & (y >= border) & (y <= rows - 1 - border)
        x, y, v = x[ok], y[ok], v_all[ok]
        if v.size == 0:
            e = np.zeros(0)
            return e, e, e, e
        R = ref_fn(x, y)
        if not grad:
            return R, None, None, v
        gx = (ref_fn(x + h, y) - ref_fn(x - h, y)) / (2 * h)
        gy = (ref_fn(x, y + h) - ref_fn(x, y - h)) / (2 * h)
        return R, gx, gy, v

    def cost(s: np.ndarray) -> float:
        R, _, _, v = evaluate(s, grad=False)
        if v.size < 8:
            return np.inf
        r = R - v
        return float(np.mean((r - r.mean()) ** 2))

    s = np.asarray(init, dtype=float).reshape(2).copy()
    converged = False
    n_iter = 0
    cur_cost = cost(s)
    for n_iter in range(1, iters + 1):
        R, gx, gy, v = evaluate(s)
        if v.size < 8:
            break
        c = float(np.mean(v - R))
        r = R + c - v
        J = np.column_stack([gx, gy, np.ones_like(gx)])
        H = J.T @ J
        g = J.T @ r
        H[np.diag_indices(3)] += 1e-9 * (np.trace(H) + 1e-12)
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = d[:2]
        nrm = float(np.hypot(*step))
        if nrm > max_step:
            step = step * (max_step / nrm)
            nrm = max_step
        accepted = False
        for _ in range(7):
            new = s + step
            new_cost = cost(new)
            if new_cost <= cur_cost * (1 + 1e-9) or nrm < tol:
                accepted = True
                break
            step = step * 0.5
            nrm *= 0.5
        if not accepted:
            converged = True  # no descent direction left
            break
        s = new
        cur_cost = new_cost
        if nrm < tol:
            converged = True
            break

    R, gx, gy, v = evaluate(s)
    n = int(v.size)
    info = dict(offset=np.nan, sigma_resid=np.inf, n_used=n, n_iter=n_iter, converged=converged,
                lambda_min_raw=0.0)
    if n < 8:
        out = (s, 0.0)
        return (*out, info) if return_info else out
    c = float(np.mean(v - R))
    r = R + c - v
    sig2 = max(float(np.sum(r * r)) / max(n - 3, 1), _CONF_FLOOR)
    G = np.column_stack([gx - gx.mean(), gy - gy.mean()])
    Hs = G.T @ G
    lam_raw = float(np.linalg.eigvalsh(Hs)[0])
    if noise_hessian is not None:
        Hs = Hs - n * np.asarray(noise_hessian, dtype=float)
    lam = float(np.linalg.eigvalsh(Hs)[0])
    conf = max(lam, 0.0) / sig2
    info.update(offset=c, sigma_resid=math.sqrt(sig2), lambda_min_raw=lam_raw)
    return (s, conf, info) if return_info else (s, conf)


def _grad_cov(fn: Callable, x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    gx = (fn(x + h, y) - fn(x - h, y)) / (2 * h)
    gy = (fn(x, y + h) - fn(x, y - h)) / (2 * h)
    return np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])


def _mc_noise_hessian(build: Callable[[np.random.Generator], np.ndarray], shape: tuple[int, int],
                      noise_var: float, n_mc: int = 12, fd_step: float = 1e-2) -> np.ndarray:
    """Expected ``E[grad n grad n^T]`` (2x2, K^2/px^2) of a reference's noise.

    ``build(rng)`` must run the (linear) reference construction on unit
    white noise.  Monte Carlo with a fixed internal seed (this calibrates a
    linear operator; it is not a draw from the data), averaged over interior
    points at several sub-pixel phases, scaled by ``noise_var``.
    """
    if noise_var <= 0:
        return np.zeros((2, 2))
    rng = np.random.default_rng(0)
    rows, cols = shape
    yy, xx = np.mgrid[1:rows - 1, 1:cols - 1].astype(float)
    phases = ((0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25))
    x = np.concatenate([xx.ravel() + ox for ox, _ in phases])
    y = np.concatenate([yy.ravel() + oy for _, oy in phases])
    acc = np.zeros((2, 2))
    for _ in range(n_mc):
        acc += _grad_cov(spline_interpolant(build(rng)), x, y, fd_step)
    return acc / n_mc * noise_var


def _noise_var(sensor: SensorConfig, noise_sigma_k: float | None) -> tuple[float, float]:
    """Per-pixel ``(temporal, fixed-pattern)`` variances of the non-scene part of a reading.

    Temporal: ``noise_sigma_k`` (default the sensor's NETD-derived sigma)
    squared plus quantization ``lsb^2 / 12``; fixed pattern: ``fpn_sigma_k^2``.
    """
    sig = sensor.noise_sigma_k if noise_sigma_k is None else float(noise_sigma_k)
    return sig ** 2 + sensor.adc_lsb_k ** 2 / 12.0, sensor.fpn_sigma_k ** 2


class _Reference:
    """Finite reference image, its spline interpolant and its noise Hessian
    (scaled by ``NOISE_HESSIAN_MARGIN``)."""

    def __init__(self, img: np.ndarray, noise_hessian: np.ndarray):
        self.filled = img
        self.fn = spline_interpolant(img)
        self.noise_hessian = NOISE_HESSIAN_MARGIN * noise_hessian


def _single_image(values: np.ndarray, mask: np.ndarray, fill_sigma: float, blur: float) -> np.ndarray:
    f = fill_masked(values, mask, fill_sigma)
    return ndimage.gaussian_filter(f, blur, mode="mirror", truncate=3.0) if blur > 0 else f


def _single_reference(values: np.ndarray, mask: np.ndarray, noise_var: tuple[float, float],
                      fill_sigma: float, blur: float) -> _Reference:
    """Reference from one (sub)frame: filled, then Gaussian-blurred by ``blur`` px.

    The blur makes the reference noise roughly stationary.  Without it the
    interpolant's noise variance is about 4x lower at filled (unread) chess
    positions than at read ones, and the least-squares cost then pulls
    same-parity subpages toward a half-diagonal shift (errors of about
    0.6 px at 8 Hz BAB noise).
    """
    img = _single_image(values, mask, fill_sigma, blur)
    Hn = _mc_noise_hessian(lambda rng: _single_image(rng.standard_normal(mask.shape), mask, fill_sigma, blur),
                           mask.shape, float(sum(noise_var)))
    return _Reference(img, Hn)


def _splat(values: np.ndarray, mask: np.ndarray, shift: np.ndarray, w: float,
           shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear splat of a subpage's valid samples onto the reference grid.

    Pixel ``(i, j)`` under shift ``s`` sits at reference grid coordinate
    ``(j + dx, i + dy)``; returns the ``(sum w v, sum w)`` accumulators.
    """
    rows, cols = shape
    m = mask & np.isfinite(values)
    ii, jj = np.nonzero(m)
    v = values[ii, jj]
    x = jj + float(shift[0])
    y = ii + float(shift[1])
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx, fy = x - x0, y - y0
    num = np.zeros(rows * cols)
    den = np.zeros(rows * cols)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < cols) & (yi >= 0) & (yi < rows)
        idx = (yi[ok] * cols + xi[ok]).astype(np.int64)
        num += np.bincount(idx, weights=w * ww[ok] * v[ok], minlength=rows * cols)
        den += np.bincount(idx, weights=w * ww[ok], minlength=rows * cols)
    return num.reshape(shape), den.reshape(shape)


def _normalize_stack(num: np.ndarray, den: np.ndarray, sigma: float) -> np.ndarray:
    gn = ndimage.gaussian_filter(num, sigma, mode="mirror", truncate=3.0) if sigma > 0 else num
    gd = ndimage.gaussian_filter(den, sigma, mode="mirror", truncate=3.0) if sigma > 0 else den
    ok = gd > 1e-3 * max(float(gd.max()), 1e-300)
    out = np.where(ok, gn / np.where(ok, gd, 1.0), np.nan)
    if not ok.all():
        out = fill_masked(out, ok, 0.7)
    return out


class _StackReference:
    """Multi-frame reference: normalized convolution of all registered subpages.

    Averaging many subpages cuts the reference noise (and the chess-fill
    error) roughly by the number of frames used.  ``image(exclude=k)``
    leaves subpage ``k`` out so its own noise does not pull its estimate.
    """

    def __init__(self, subpages: list[Subpage], shifts: np.ndarray, weights: np.ndarray,
                 shape: tuple[int, int], sigma: float):
        self.sigma = sigma
        self.shape = shape
        self.subpages = subpages
        self.shifts = np.asarray(shifts, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.parts: list[tuple[np.ndarray, np.ndarray] | None] = []
        self.num = np.zeros(shape)
        self.den = np.zeros(shape)
        for sp, s, w in zip(subpages, self.shifts, self.weights):
            if w <= 0:
                self.parts.append(None)
                continue
            p = _splat(sp.values, sp.mask, s, float(w), shape)
            self.parts.append(p)
            self.num += p[0]
            self.den += p[1]

    def image(self, exclude: int | None = None) -> np.ndarray:
        num, den = self.num, self.den
        if exclude is not None and self.parts[exclude] is not None:
            num = num - self.parts[exclude][0]
            den = den - self.parts[exclude][1]
        return _normalize_stack(num, den, self.sigma)

    def noise_hessian(self, noise_var: tuple[float, float]) -> np.ndarray:
        """Temporal noise averages down over the stack, but the fixed pattern
        is the same in every subpage (sensor-locked, so it pulls toward zero
        relative shift); the Monte Carlo draws it once per realization."""
        shape = self.shape
        t_sd, f_sd = math.sqrt(noise_var[0]), math.sqrt(noise_var[1])
        if t_sd == 0 and f_sd == 0:
            return np.zeros((2, 2))

        def build(rng: np.random.Generator) -> np.ndarray:
            num = np.zeros(shape)
            den = np.zeros(shape)
            fpn = rng.standard_normal(shape) * f_sd
            for sp, s, w in zip(self.subpages, self.shifts, self.weights):
                if w <= 0:
                    continue
                n_, d_ = _splat(rng.standard_normal(shape) * t_sd + fpn, sp.mask, s, float(w), shape)
                num += n_
                den += d_
            return _normalize_stack(num, den, self.sigma)

        return _mc_noise_hessian(build, shape, 1.0, n_mc=6)


def _clip_norm(s: np.ndarray, max_shift: float) -> np.ndarray:
    n = float(np.hypot(*s))
    return s * (max_shift / n) if n > max_shift else s


# ---------------------------------------------------------------------------
# Gyro
# ---------------------------------------------------------------------------


class _GyroIntegrator:
    """Cumulative trapezoid angles of a gyro trace (bias removed)."""

    def __init__(self, gyro: GyroTrace, bias: np.ndarray | None = None):
        t = np.asarray(gyro.t, dtype=float)
        w = np.asarray(gyro.omega, dtype=float)
        if bias is not None:
            w = w - np.asarray(bias, dtype=float).reshape(1, -1)
        A = np.zeros_like(w)
        A[1:] = np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t)[:, None], axis=0)
        self.t = t
        self.A = A  # (N, 3) pitch, yaw, roll angles (rad) vs gyro stamp
        self.rate_hz = float(gyro.rate_hz)

    def angles(self, T: np.ndarray, tau: float) -> np.ndarray:
        """``(..., 2)`` (yaw, pitch) at camera times ``T`` relative to camera ``t = 0``."""
        T = np.asarray(T, dtype=float)
        st = T - tau
        a0 = np.array([np.interp(-tau, self.t, self.A[:, k]) for k in (1, 0)])
        out = np.stack([np.interp(st, self.t, self.A[:, 1]), np.interp(st, self.t, self.A[:, 0])], axis=-1)
        return out - a0

    def coverage_ok(self, T: np.ndarray, tau: float) -> bool:
        st = np.asarray(T, dtype=float) - tau
        return bool(st.min() >= self.t[0] - 1e-12 and st.max() <= self.t[-1] + 1e-12)

    def window_samples(self, windows: np.ndarray, n: int | None = None) -> np.ndarray:
        windows = np.asarray(windows, dtype=float).reshape(-1, 2)
        if n is None:
            T = float(np.max(windows[:, 1] - windows[:, 0]))
            n = max(8, int(math.ceil(2.0 * T * self.rate_hz)))
        frac = (np.arange(n) + 0.5) / n
        return windows[:, :1] + frac[None, :] * (windows[:, 1:2] - windows[:, :1])

    def shifts(self, windows: np.ndarray, tau: float, scale: tuple[float, float]) -> np.ndarray:
        ts = self.window_samples(windows)
        ang = self.angles(ts, tau)  # (K, n, 2)
        sh = np.stack([scale[0] * np.tan(ang[..., 0]), scale[1] * np.tan(ang[..., 1])], axis=-1)
        return sh.mean(axis=1)


def integrate_gyro(gyro: GyroTrace, sensor: SensorConfig, subpage_windows: np.ndarray,
                   time_offset: float = 0.0, *, bias: np.ndarray | None = None) -> np.ndarray:
    """Exposure-averaged gyro-predicted shifts ``(K, 2)`` relative to camera ``t = 0``.

    Trapezoid-integrates ``omega - bias`` over the gyro stamps, maps camera
    time ``T`` to stamp ``T - time_offset`` (a gyro sample stamped ``t``
    measured camera time ``t + time_offset``), converts (yaw, pitch) with
    ``f_px * tan`` and averages over each exposure window with a midpoint
    rule (at least 8 samples, about two per gyro period).  ``bias`` is in
    rad/s in ``omega`` axis order (pitch, yaw, roll).
    """
    integ = _GyroIntegrator(gyro, bias)
    windows = np.asarray(subpage_windows, dtype=float).reshape(-1, 2)
    ts = integ.window_samples(windows)
    if not integ.coverage_ok(np.concatenate([ts.ravel(), [0.0]]), time_offset):
        warnings.warn("gyro stamps do not cover the exposure windows for this time offset; "
                      "angles are clamped at the ends", RuntimeWarning, stacklevel=2)
    return integ.shifts(windows, time_offset, sensor.f_px)


def _windows_from_times(sub_t: np.ndarray) -> np.ndarray:
    sub_t = np.asarray(sub_t, dtype=float)
    if sub_t.ndim == 2:
        return sub_t
    if sub_t.size < 2:
        raise ValueError("need at least two subpage times")
    T = float(np.median(np.diff(sub_t)))
    return np.column_stack([sub_t - 0.5 * T, sub_t + 0.5 * T])


def _weighted_detrend(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove the weighted least-squares fit of ``a + b t`` from each column of ``x``."""
    X = np.column_stack([np.ones_like(t), t])
    A = (X * w[:, None]).T @ X
    coef = np.linalg.lstsq(A, (X * w[:, None]).T @ x, rcond=None)[0]
    return x - X @ coef


def estimate_time_offset(image_shifts: np.ndarray, sub_t: np.ndarray, gyro: GyroTrace,
                         search: float = 0.05, *, sensor: SensorConfig | None = None,
                         weights: np.ndarray | None = None, step: float = 5e-4,
                         domain: Literal["levels", "rates"] = "levels", return_info: bool = False):
    """Camera-gyro offset tau (s) by cross-correlating image- and gyro-derived motion.

    ``image_shifts``: ``(K, 2)`` image-derived shifts (any common offset);
    ``sub_t``: ``(K,)`` exposure mid-times (contiguous windows of the median
    spacing are assumed) or ``(K, 2)`` exposure windows.  For each candidate
    ``tau`` in ``[-search, search]`` (grid ``step``) the gyro is integrated
    and exposure-averaged exactly like the image.  The score is the pooled
    weighted correlation of the x and y signals; its peak is refined with a
    parabola through the neighbors.

    ``domain="rates"`` correlates the rates (first differences of
    consecutive subpages, mean removed, so a constant gyro bias drops out).
    ``domain="levels"`` (default) correlates the shifts themselves after
    removing a weighted linear trend per axis, which absorbs the gyro bias
    drift and any offset.  The two are equivalent for noiseless data, but
    differencing acts as a high-pass filter that amplifies the white
    registration noise relative to slow motion (for a 1.5 Hz wobble at a
    32 Hz subpage rate the tau information drops by roughly
    ``(2 pi f T)^2 / 2``, about 20x), so the level form is several times
    more precise for wobble and drift and about as precise for tremor.

    ``weights`` (``(K,)``, e.g. registration confidences; differences use
    the smaller of their two endpoints).  ``sensor`` supplies ``f_px``
    (default: unit scale, which only changes the x/y pooling).  Returns
    tau, or ``(tau, info)`` when ``return_info``, with ``info = {"peak":
    score at the peak, "sd": predicted standard error of tau (s), "taus":
    grid, "scores": scores, "at_edge": peak on the search boundary}``.
    ``sd`` is ``sqrt(max(chi2_red, 1) / sum_k w_k |beta d g_k / d tau|^2)``
    with the gyro shifts' tau-derivative prepared like the signals, ``beta``
    the fitted gain and ``chi2_red`` the weighted reduced chi-square of the
    residual ``image - beta g(tau)`` (so optimistic or unit weights are
    rescaled by the actual scatter).  Weights at or above ``CONF_REF`` (the
    reference subpage) are replaced by the largest other weight.  A high
    correlation peak alone does not mean tau is observable: slow drift
    correlates well at any tau.
    """
    s = np.asarray(image_shifts, dtype=float).reshape(-1, 2)
    windows = _windows_from_times(sub_t)
    if windows.shape[0] != s.shape[0]:
        raise ValueError("image_shifts and sub_t lengths differ")
    if domain not in ("levels", "rates"):
        raise ValueError(f"unknown domain {domain!r}")
    K = s.shape[0]
    w = np.ones(K) if weights is None else np.clip(np.asarray(weights, float), 0.0, None)
    big = w >= CONF_REF  # the reference's nominal confidence is not a measurement precision
    if big.any():
        w = np.where(big, w[~big].max() if (~big).any() else 1.0, w)
    mid = windows.mean(axis=1)
    scale = sensor.f_px if sensor is not None else (1.0, 1.0)
    integ = _GyroIntegrator(gyro)
    taus = np.arange(-search, search + 0.5 * step, step)
    scores = np.full(taus.size, -np.inf)

    if domain == "rates":
        T = float(np.median(windows[:, 1] - windows[:, 0]))
        pair = np.diff(mid) <= 1.5 * T  # consecutive subpages only
        ww = np.minimum(w[:-1], w[1:]) * pair

        def prep(x: np.ndarray) -> np.ndarray:
            d = np.diff(x, axis=0)
            return d - (ww / ww.sum()) @ d
    else:
        ww = w

        def prep(x: np.ndarray) -> np.ndarray:
            return _weighted_detrend(x, mid - mid.mean(), ww)

    if ww.sum() <= 0 or (ww > 0).sum() < 3:
        info = dict(peak=float("nan"), sd=float("inf"), taus=taus, scores=scores, at_edge=True)
        return (0.0, info) if return_info else 0.0
    wn = ww / ww.sum()
    si = prep(s)
    var_i = float(np.sum(wn[:, None] * si * si))
    for n, tau in enumerate(taus):
        sg = prep(integ.shifts(windows, float(tau), scale))
        var_g = float(np.sum(wn[:, None] * sg * sg))
        if var_g <= 0 or var_i <= 0:
            continue
        scores[n] = float(np.sum(wn[:, None] * si * sg)) / math.sqrt(var_i * var_g)
    k = int(np.argmax(scores))
    tau = float(taus[k])
    if 0 < k < taus.size - 1 and np.all(np.isfinite(scores[k - 1:k + 2])):
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            tau += float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) * step
    # Standard error from the weighted least-squares view: Fisher information
    # of the fitted model beta * g(tau), inflated by the reduced chi-square of
    # the residuals when the weights are optimistic.
    sg = prep(integ.shifts(windows, tau, scale))
    den_b = float(np.sum(ww[:, None] * sg * sg))
    beta = float(np.sum(ww[:, None] * si * sg)) / den_b if den_b > 0 else 1.0
    resid = si - beta * sg
    n_used = int((ww > 0).sum())
    dof = max(2 * n_used - (5 if domain == "levels" else 3), 1)
    chi2 = float(np.sum(ww[:, None] * resid * resid)) / dof
    d = 1e-3
    dg = beta * (prep(integ.shifts(windows, tau + d, scale)) - prep(integ.shifts(windows, tau - d, scale))) / (2 * d)
    fisher = float(np.sum(ww[:, None] * dg * dg))
    sd = math.sqrt(max(chi2, 1.0) / fisher) if fisher > 0 else float("inf")
    info = dict(peak=float(scores[k]), sd=sd, taus=taus, scores=scores,
                at_edge=bool(k == 0 or k == taus.size - 1))
    return (tau, info) if return_info else tau


# ---------------------------------------------------------------------------
# Registration methods
# ---------------------------------------------------------------------------


def _subpage_times(capture: Capture) -> np.ndarray:
    return capture.subpage_windows.mean(axis=1)


def _sigma_from_conf(conf: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.where(conf > 0, 1.0 / np.sqrt(np.maximum(conf, _CONF_FLOOR)), np.inf)


def _gn_one(R: _Reference, sp: Subpage, s0: np.ndarray, iters: int, tol: float, max_shift: float):
    s0 = _clip_norm(np.asarray(s0, dtype=float), max_shift)
    s, c = gauss_newton_translation(R.fn, sp.values, sp.mask, s0, iters=iters, tol=tol,
                                    noise_hessian=R.noise_hessian)
    if float(np.hypot(*s)) > max_shift + 1.0:  # diverged: keep the init, no confidence
        return s0, 0.0
    return s, c


def _single_pass(capture: Capture, ref_index: int, init: np.ndarray | None, *, fill_sigma: float,
                 ref_blur: float, iters: int, tol: float, max_shift: float,
                 noise_var: tuple[float, float]):
    """Gauss-Newton of every subpage against the filled, blurred reference subpage."""
    sps = capture.subpages
    K = len(sps)
    ref = sps[ref_index]
    R = _single_reference(ref.values, ref.mask, noise_var, fill_sigma, ref_blur)
    ref_filled = fill_subpage(ref, fill_sigma)
    shifts = np.zeros((K, 2))
    conf = np.zeros(K)
    inits = np.zeros((K, 2))
    for k, sp in enumerate(sps):
        if k == ref_index:
            conf[k] = CONF_REF
            continue
        if init is None:
            s0 = phase_correlation(ref_filled, fill_subpage(sp, fill_sigma), upsample_factor=10, iters=2)
        else:
            s0 = np.asarray(init[k], dtype=float)
        inits[k] = _clip_norm(s0, max_shift)
        shifts[k], conf[k] = _gn_one(R, sp, s0, iters, tol, max_shift)
    return shifts, conf, inits


def _stack_pass(capture: Capture, ref_index: int, stack_shifts: np.ndarray, stack_weights: np.ndarray,
                init: np.ndarray, *, stack_sigma: float, iters: int, tol: float, max_shift: float,
                noise_var: tuple[float, float], subpages: list[Subpage] | None = None):
    """Gauss-Newton of every subpage (reference included) against a leave-one-out
    multi-frame reference, then re-anchored so the reference subpage is zero.
    ``subpages`` overrides ``capture.subpages`` (e.g. FPN-corrected copies)."""
    sps = capture.subpages if subpages is None else subpages
    K = len(sps)
    st = _StackReference(sps, stack_shifts, stack_weights, capture.sensor.shape, stack_sigma)
    Hn = st.noise_hessian(noise_var)
    shifts = np.zeros((K, 2))
    conf = np.zeros(K)
    for k, sp in enumerate(sps):
        R = _Reference(st.image(exclude=k), Hn)
        shifts[k], conf[k] = _gn_one(R, sp, init[k], iters, tol, max_shift)
    shifts = shifts - shifts[ref_index]
    conf[ref_index] = CONF_REF
    return shifts, conf


def _estimate_fpn(subpages: list[Subpage], shifts: np.ndarray, shape: tuple[int, int], stack_sigma: float,
                  noise_var: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Wiener-shrunk fixed-pattern estimate from residuals against the multi-frame reference.

    For each subpage the leave-one-out reference is sampled at its shift and
    the mean-free residual is averaged per sensor pixel; the average is
    shrunk by ``s_f^2 / (s_f^2 + s_t^2 / n_reads)`` and made zero-mean.
    Returns ``(fpn_hat, residual_fpn_var)`` where the residual variance is the
    Wiener MMSE ``s_f^2 (1 - shrink)`` averaged over the array.  With small
    motion the scene and the pattern are only partly separable, so this is a
    partial correction.
    """
    t_var, f_var = noise_var
    if f_var <= 0:
        return np.zeros(shape), 0.0
    st = _StackReference(subpages, shifts, np.ones(len(subpages)), shape, stack_sigma)
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    for k, sp in enumerate(subpages):
        m = sp.mask & np.isfinite(sp.values)
        ii, jj = np.nonzero(m)
        fn = spline_interpolant(st.image(exclude=k))
        r = sp.values[ii, jj] - fn(jj + shifts[k][0], ii + shifts[k][1])
        acc[ii, jj] += r - r.mean()
        cnt[ii, jj] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        shrink = np.where(cnt > 0, f_var / (f_var + t_var / np.maximum(cnt, 1)), 0.0)
    fpn = shrink * raw
    fpn -= fpn[cnt > 0].mean() if (cnt > 0).any() else 0.0
    return fpn, float(f_var * np.mean(1.0 - shrink))


def _image_gn(capture: Capture, ref_index: int, init: np.ndarray | None, *, fill_sigma: float = 0.7,
              ref_blur: float = 0.6, passes: int = 2, stack_sigma: float = 0.5,
              stack_min_conf: float = 25.0, iters: int = 20, tol: float = 1e-4, max_shift: float = 4.0,
              noise_sigma_k: float | None = None, noise_correction: bool = True):
    """Image-only Gauss-Newton: a single-reference pass, then ``passes - 1``
    multi-frame-reference passes built from the frames with confidence of at
    least ``stack_min_conf`` (all frames when fewer than a quarter qualify)."""
    nv = _noise_var(capture.sensor, noise_sigma_k) if noise_correction else (0.0, 0.0)
    K = capture.n_subpages
    shifts, conf, inits = _single_pass(capture, ref_index, init, fill_sigma=fill_sigma, ref_blur=ref_blur,
                                       iters=iters, tol=tol, max_shift=max_shift, noise_var=nv)
    for _ in range(max(0, passes - 1)):
        w = (conf >= stack_min_conf).astype(float)
        if w.sum() < max(3, K // 4):
            w = np.ones(K)
        shifts, conf = _stack_pass(capture, ref_index, shifts, w, shifts, stack_sigma=stack_sigma,
                                   iters=iters, tol=tol, max_shift=max_shift, noise_var=nv)
    return shifts, conf, dict(init=inits)


def _phase_result(R: _Reference, values: np.ndarray, mask: np.ndarray, s: np.ndarray,
                  max_shift: float) -> tuple[np.ndarray, float]:
    """Confidence of a phase-correlation shift (GN Hessian at ``s``, no update);
    a peak beyond ``max_shift`` (typical of pure noise) is clipped with zero confidence."""
    if float(np.hypot(*s)) > max_shift:
        return _clip_norm(s, max_shift), 0.0
    _, c = gauss_newton_translation(R.fn, values, mask, s, iters=0, noise_hessian=R.noise_hessian)
    return s, c


def _register_phase(capture: Capture, ref_index: int, *, merged: bool = False,
                    upsample_factor: int = 20, iters: int = 3, fill_sigma: float = 0.7,
                    ref_blur: float = 0.6, max_shift: float = 4.0,
                    noise_sigma_k: float | None = None) -> Registration:
    sps = capture.subpages
    K = len(sps)
    shifts = np.zeros((K, 2))
    conf = np.zeros(K)
    nv = _noise_var(capture.sensor, noise_sigma_k)
    if merged:
        frames = merged_frames(capture)
        order = sorted({sp.frame_index for sp in sps})
        pos = {f: n for n, f in enumerate(order)}
        rf = pos[sps[ref_index].frame_index]
        ref_filled = fill_masked(frames[rf][0], frames[rf][1], fill_sigma)
        R = _single_reference(frames[rf][0], frames[rf][1], nv, fill_sigma, ref_blur)
        fshift = np.zeros((len(frames), 2))
        fconf = np.zeros(len(frames))
        for n, (vals, mask) in enumerate(frames):
            if n == rf:
                fconf[n] = CONF_REF
                continue
            s = phase_correlation(ref_filled, fill_masked(vals, mask, fill_sigma), upsample_factor, iters=iters)
            fshift[n], fconf[n] = _phase_result(R, vals, mask, s, max_shift)
        for k, sp in enumerate(sps):
            shifts[k] = fshift[pos[sp.frame_index]]
            conf[k] = fconf[pos[sp.frame_index]]
    else:
        ref = sps[ref_index]
        ref_filled = fill_subpage(ref, fill_sigma)
        R = _single_reference(ref.values, ref.mask, nv, fill_sigma, ref_blur)
        for k, sp in enumerate(sps):
            if k == ref_index:
                conf[k] = CONF_REF
                continue
            s = phase_correlation(ref_filled, fill_subpage(sp, fill_sigma), upsample_factor, iters=iters)
            shifts[k], conf[k] = _phase_result(R, sp.values, sp.mask, s, max_shift)
    shifts[ref_index] = 0.0
    conf[ref_index] = CONF_REF
    return Registration(shifts=shifts, confidence=conf, method="phase",
                        info=dict(merged=merged, sigma_px=_sigma_from_conf(conf)))


def _register_gn(capture: Capture, ref_index: int, *, init: np.ndarray | None = None, **kw) -> Registration:
    if init is not None:
        init = np.asarray(init, dtype=float).reshape(-1, 2)
    shifts, conf, info = _image_gn(capture, ref_index, init, **kw)
    info["sigma_px"] = _sigma_from_conf(conf)
    return Registration(shifts=shifts, confidence=conf, method="gn", info=info)


def _gyro_variance_px(capture: Capture, ref_index: int, gyro_noise_dps_rthz: float,
                      bias_sigma_rad: np.ndarray | float, offset_sigma_px: float = 0.0) -> np.ndarray:
    """Predicted (K, 2) variance (px^2) of gyro-relative shifts: ARW + bias drift + offset."""
    t = _subpage_times(capture)
    dt = np.abs(t - t[ref_index])
    f = np.asarray(capture.sensor.f_px)
    nd = gyro_noise_dps_rthz * _DEG
    bs = np.broadcast_to(np.asarray(bias_sigma_rad, dtype=float), (2,))
    var = (f[None, :] ** 2) * (nd ** 2 * dt[:, None] + (bs[None, :] * dt[:, None]) ** 2)
    return var + offset_sigma_px ** 2 + 1e-8


def _register_gyro(capture: Capture, ref_index: int, *, time_offset: float = 0.0,
                   bias: np.ndarray | None = None, gyro_noise_dps_rthz: float = 0.007,
                   bias_prior_dps: float = 1.0) -> Registration:
    gyro = capture.gyro
    if gyro is None:
        raise ValueError("capture has no gyro")
    g = integrate_gyro(gyro, capture.sensor, capture.subpage_windows, time_offset, bias=bias)
    shifts = g - g[ref_index]
    bsig = 0.0 if bias is not None else bias_prior_dps * _DEG
    var = _gyro_variance_px(capture, ref_index, gyro_noise_dps_rthz, bsig)
    conf = 1.0 / var.max(axis=1)
    conf[ref_index] = CONF_REF
    return Registration(shifts=shifts, confidence=conf, method="gyro", time_offset_s=float(time_offset),
                        gyro_bias=None if bias is None else np.asarray(bias, float),
                        info=dict(sigma_px=_sigma_from_conf(conf)))


def _fit_bias(resid: np.ndarray, t_rel: np.ndarray, w: np.ndarray, use: np.ndarray,
              slope_prior_sd: np.ndarray, offset_prior_sd: float):
    """Per-axis ridge fit ``resid = a + c t_rel`` -> ``(a (2,), c (2,), [cov_x, cov_y])``."""
    a = np.zeros(2)
    c = np.zeros(2)
    cov = []
    X = np.column_stack([np.ones_like(t_rel), t_rel])
    for ax in range(2):
        ww = np.where(use, w[:, ax], 0.0)
        A = (X * ww[:, None]).T @ X + np.diag([1.0 / offset_prior_sd ** 2, 1.0 / slope_prior_sd[ax] ** 2])
        Ainv = np.linalg.inv(A)
        a[ax], c[ax] = Ainv @ ((X * ww[:, None]).T @ resid[:, ax])
        cov.append(Ainv)
    return a, c, cov


def _estimate_bias(gn_s: np.ndarray, gn_c: np.ndarray, g0: np.ndarray, t_rel: np.ndarray, ref_index: int,
                   arw_var: np.ndarray, f: np.ndarray, *, min_conf: float, sigma_cap: float,
                   bias_prior_dps: float, offset_prior_px: float, n_iter: int, signif: float):
    """Robust weighted ridge fit of ``image - gyro_seed = a + c t`` per axis.

    Weights come from the image confidences with sigma capped at
    ``sigma_cap`` (so frames with zero reported confidence still count a
    little), then are rescaled per axis by the robust (MAD) scatter of the
    normalized residuals whenever that exceeds 1, which self-calibrates the
    weights when the confidences are optimistic or zero; residuals beyond 3
    scaled sigma are clipped and the fit repeated.  The slope has a ridge
    prior (``bias_prior_dps``) and is set to 0 unless it is at least
    ``signif`` posterior standard deviations from 0.  Returns ``(bias (3,)
    rad/s in omega order, a (2,) px, bias_sd_xy (2,) rad/s in (yaw, pitch)
    = (x, y) order, offset_sd px, frames used (K,) bool)``.
    """
    sig = np.minimum(_sigma_from_conf(gn_c), sigma_cap)
    w0 = 1.0 / (sig[:, None] ** 2 + arw_var)
    resid = gn_s - g0
    slope_prior = f * bias_prior_dps * _DEG
    cand = gn_c >= min_conf
    cand[ref_index] = False
    use = cand.copy()
    scale = np.ones(2)
    a, c = np.zeros(2), np.zeros(2)
    cov = [np.diag([offset_prior_px ** 2, sp ** 2]) for sp in slope_prior]
    for _ in range(max(1, n_iter)):
        if use.sum() < 3:
            a, c = np.zeros(2), np.zeros(2)
            cov = [np.diag([offset_prior_px ** 2, sp ** 2]) for sp in slope_prior]
            break
        a, c, cov = _fit_bias(resid, t_rel, w0 / scale[None, :] ** 2, use, slope_prior, offset_prior_px)
        z = (resid - (a[None, :] + c[None, :] * t_rel[:, None])) * np.sqrt(w0)
        new_scale = np.maximum(1.0, 1.4826 * np.median(np.abs(z[use]), axis=0))
        new_use = cand & np.all(np.abs(z) <= 3.0 * new_scale[None, :], axis=1)
        if np.array_equal(new_use, use) and np.allclose(new_scale, scale, rtol=1e-3):
            break
        use, scale = new_use, new_scale
    c_sd = np.array([math.sqrt(cov[0][1, 1]), math.sqrt(cov[1][1, 1])])
    c = np.where(np.abs(c) >= signif * c_sd, c, 0.0)
    bias = np.array([-c[1] / f[1], -c[0] / f[0], 0.0])  # (pitch, yaw, roll) rad/s
    bias_sd_xy = c_sd / f
    offset_sd = float(math.sqrt(max(cov[0][0, 0], cov[1][0, 0])))
    return bias, a, bias_sd_xy, offset_sd, use


def _register_hybrid(capture: Capture, ref_index: int, *, time_offset: float | None = None,
                     bias: np.ndarray | None = None, min_confidence: float = 100.0,
                     bias_min_confidence: float = 4.0, bias_sigma_cap_px: float = 1.0,
                     bias_signif: float = 0.0, max_gyro_dev: float = 0.5,
                     tau_search: float = 0.05, tau_min_corr: float = 0.5, time_offset_prior: float = 0.0,
                     time_offset_prior_sd: float = 0.015, tau_max_sd: float = 0.003,
                     time_offset_est_sd_floor: float = 5e-4,
                     fuse: bool = True, gyro_floor_px: float = 0.02, fpn_correction: bool = True,
                     bias_prior_dps: float = 1.0, offset_prior_px: float = 0.5,
                     gyro_noise_dps_rthz: float = 0.007, n_bias_iters: int = 6, fill_sigma: float = 0.7,
                     ref_blur: float = 0.6, stack_sigma: float = 0.5, iters: int = 20, tol: float = 1e-4,
                     passes: int = 2, stack_min_conf: float = 25.0, noise_correction: bool = True,
                     max_shift: float = 4.0, noise_sigma_k: float | None = None) -> Registration:
    gyro = capture.gyro
    if gyro is None:
        raise ValueError("hybrid registration needs a gyro")
    sensor = capture.sensor
    windows = capture.subpage_windows
    t_mid = windows.mean(axis=1)
    t_rel = t_mid - t_mid[ref_index]
    K = capture.n_subpages
    f = np.asarray(sensor.f_px)
    nv = _noise_var(sensor, noise_sigma_k) if noise_correction else (0.0, 0.0)
    common = dict(iters=iters, tol=tol, max_shift=max_shift, noise_var=nv)
    info: dict = {}

    # 1. Time offset from image-only shifts (when not given).
    tau_source = "given"
    if time_offset is None:
        img_s, img_c, _ = _image_gn(capture, ref_index, None, fill_sigma=fill_sigma, ref_blur=ref_blur,
                                    passes=passes, stack_sigma=stack_sigma, stack_min_conf=stack_min_conf,
                                    iters=iters, tol=tol, max_shift=max_shift, noise_sigma_k=noise_sigma_k,
                                    noise_correction=noise_correction)
        # Inverse-variance weights; the cap keeps the reference and a few
        # very confident frames from dominating.
        tau, tinfo = estimate_time_offset(img_s, windows, gyro, tau_search, sensor=sensor,
                                          weights=np.minimum(img_c, 1e4), return_info=True)
        info.update(tau_peak_corr=tinfo["peak"], tau_raw=tau, tau_raw_sd=tinfo["sd"])
        if ((img_c > 0).sum() >= 4 and np.isfinite(tinfo["peak"]) and tinfo["peak"] >= tau_min_corr
                and tinfo["sd"] <= tau_max_sd and not tinfo["at_edge"]):
            time_offset, tau_source = tau, "estimated"
            tau_sd = max(tinfo["sd"], time_offset_est_sd_floor)
        else:
            time_offset, tau_source, tau_sd = float(time_offset_prior), "prior", time_offset_prior_sd
    else:
        tau_sd = 0.0
    info["tau_source"] = tau_source
    tau = float(time_offset)

    # 2. Gyro seed relative to the reference subpage.
    g0 = integrate_gyro(gyro, sensor, windows, tau, bias=bias)
    g0 = g0 - g0[ref_index]
    arw_var = _gyro_variance_px(capture, ref_index, gyro_noise_dps_rthz, 0.0)
    bias_kw = dict(min_conf=bias_min_confidence, sigma_cap=bias_sigma_cap_px, bias_prior_dps=bias_prior_dps,
                   offset_prior_px=offset_prior_px, n_iter=n_bias_iters, signif=bias_signif)

    def corrected_gyro(gn_s: np.ndarray, gn_c: np.ndarray):
        """Bias-corrected gyro shifts and the image anchor offset ``a``.

        The gyro is exact relative to the reference at ``t_rel = 0``, so the
        fitted intercept ``a`` is an error of the image shifts (the noise of
        the reference subpage's own anchoring) and is removed from them.
        """
        b, a, bsd, osd, use = _estimate_bias(gn_s, gn_c, g0, t_rel, ref_index, arw_var, f, **bias_kw)
        if bias is not None:
            return g0, np.asarray(bias, float), a, np.zeros(2), osd, use
        g = integrate_gyro(gyro, sensor, windows, tau, bias=b)
        return g - g[ref_index], b, a, bsd, osd, use

    def decide(img_s: np.ndarray, gn_c: np.ndarray, g: np.ndarray) -> np.ndarray:
        acc = (gn_c >= min_confidence) & (np.hypot(*(img_s - g).T) <= max_gyro_dev)
        acc[ref_index] = False
        return acc

    # 3. Gauss-Newton from the gyro seed against the reference subpage.
    s1, c1, _ = _single_pass(capture, ref_index, g0, fill_sigma=fill_sigma, ref_blur=ref_blur, **common)
    g1, _, a1, _, _, _ = corrected_gyro(s1, c1)
    s1 = s1 - a1[None, :]
    acc1 = decide(s1, c1, g1)
    combined = np.where(acc1[:, None], s1, g1)
    combined[ref_index] = 0.0
    # 4. Second pass against a multi-frame reference built with the combined shifts.
    gn_s, gn_c = _stack_pass(capture, ref_index, combined, np.ones(K), combined, stack_sigma=stack_sigma,
                             **common)
    if fpn_correction and nv[1] > 0:
        # 4b. The fixed pattern is sensor-locked and pulls every subpage toward
        # the reference's alignment; estimate it, remove it and re-register.
        g4, _, a4, _, _, _ = corrected_gyro(gn_s, gn_c)
        s4 = gn_s - a4[None, :]
        cur = np.where(decide(s4, gn_c, g4)[:, None], s4, g4)
        cur[ref_index] = 0.0
        fpn_hat, f_res = _estimate_fpn(capture.subpages, cur, sensor.shape, stack_sigma, nv)
        corrected = [dataclasses.replace(sp, values=sp.values - fpn_hat) for sp in capture.subpages]
        gn_s, gn_c = _stack_pass(capture, ref_index, cur, np.ones(K), cur, stack_sigma=stack_sigma,
                                 **{**common, "noise_var": (nv[0], f_res)}, subpages=corrected)
        info["fpn_estimate"] = fpn_hat
    # 5. Bias fit, then accept the image estimate or fall back to the corrected gyro.
    g, gyro_bias, a, bias_sd_xy, offset_sd, use = corrected_gyro(gn_s, gn_c)
    img_s = gn_s - a[None, :]
    with np.errstate(divide="ignore"):
        img_c = np.where(gn_c > 0, 1.0 / (1.0 / np.maximum(gn_c, _CONF_FLOOR) + offset_sd ** 2), 0.0)
    accept = decide(img_s, gn_c, g)
    g_var = _gyro_variance_px(capture, ref_index, gyro_noise_dps_rthz, bias_sd_xy) + gyro_floor_px ** 2
    if tau_sd > 0:  # time-offset uncertainty times the shift's sensitivity to tau
        d = 1e-3
        gp = integrate_gyro(gyro, sensor, windows, tau + d, bias=gyro_bias)
        gm = integrate_gyro(gyro, sensor, windows, tau - d, bias=gyro_bias)
        dg = ((gp - gp[ref_index]) - (gm - gm[ref_index])) / (2 * d)
        g_var = g_var + (tau_sd * dg) ** 2
    g_conf = 1.0 / g_var.max(axis=1)
    if fuse:
        # Inverse-variance fusion of accepted image shifts with the gyro.
        wi = img_c[:, None]
        wg = 1.0 / g_var
        fused = (wi * img_s + wg * g) / (wi + wg)
        shifts = np.where(accept[:, None], fused, g)
        conf = np.where(accept, img_c + g_conf, g_conf)
    else:
        shifts = np.where(accept[:, None], img_s, g)
        conf = np.where(accept, img_c, g_conf)
    shifts[ref_index] = 0.0
    conf[ref_index] = CONF_REF
    fallback = ~accept
    fallback[ref_index] = False
    info.update(accepted=accept, fallback=fallback, gn_shifts=gn_s, gn_confidence=gn_c, gyro_shifts=g,
                gyro_seed=g0, gyro_sigma_px=np.sqrt(g_var), bias_fit_frames=use, bias_sd=bias_sd_xy[::-1],
                image_offset_px=a, time_offset_sd=tau_sd,
                sigma_px=_sigma_from_conf(conf))
    return Registration(shifts=shifts, confidence=conf, method="hybrid", time_offset_s=tau,
                        gyro_bias=gyro_bias, info=info)


def register(capture: Capture, method: Literal["phase", "gn", "gyro", "hybrid"] = "phase",
             ref_index: int = 0, **kw) -> Registration:
    """Estimate per-subpage shifts ``(K, 2)`` relative to subpage ``ref_index``.

    Keyword arguments by method (all optional):

    - ``"phase"``: ``merged=False``, ``upsample_factor=20``, ``iters=3``
      (passes of the window-bias correction), ``fill_sigma=0.7``.
    - ``"gn"``: ``init`` (``(K, 2)``; default per-subpage phase
      correlation), ``passes=2`` (1 = single reference subpage only; each
      further pass re-registers against a leave-one-out multi-frame
      reference built from frames with confidence >= ``stack_min_conf=25``),
      ``ref_blur=0.6`` (px, Gaussian blur of the single reference),
      ``stack_sigma=0.5``, ``iters=20``, ``tol=1e-4``, ``max_shift=4.0``
      (px, init clip and divergence limit), ``noise_correction=True``.
    - ``"gyro"``: ``time_offset=0.0`` (s), ``bias=None`` (rad/s, omega
      order), ``gyro_noise_dps_rthz=0.007``, ``bias_prior_dps=1.0``.
    - ``"hybrid"``: ``time_offset=None`` (estimate), ``bias=None``
      (estimate), ``min_confidence=100`` (px^-2, i.e. sigma <= 0.1 px, to
      accept an image shift), ``bias_min_confidence=4`` (sigma <= 0.5 px,
      to use a frame in the bias fit), ``bias_signif=0`` (zero the fitted
      slope unless it is this many posterior sd from 0), ``max_gyro_dev=0.5``
      (px), ``fuse=True`` (accepted frames are the inverse-variance mean of
      image and gyro; ``False`` uses the image shift alone, as in the plan),
      ``fpn_correction=True`` (estimate and remove the fixed pattern before
      the final multi-frame pass), ``tau_search=0.05``, ``tau_min_corr=0.5``
      and ``tau_max_sd=0.003`` (the estimated tau is used only if the
      correlation peak and its predicted standard error pass both limits;
      otherwise ``time_offset_prior=0.0`` with
      ``time_offset_prior_sd=0.015``), ``bias_prior_dps=1.0`` (Gaussian
      prior sd of the bias; shrinks the fit when the images are weak),
      ``offset_prior_px=0.5``, ``gyro_noise_dps_rthz=0.007`` and
      ``gyro_floor_px=0.02`` (gyro error model), and the image options
      ``fill_sigma``, ``ref_blur``, ``passes`` and ``stack_min_conf`` (for
      the image-only tau step), ``stack_sigma``, ``iters``, ``tol``,
      ``max_shift``, ``noise_correction``.
    - ``"phase"`` also takes ``max_shift=4.0``: peaks farther away are
      clipped and get zero confidence.

    Image methods also take ``noise_sigma_k`` (default: the sensor's
    NETD-derived temporal sigma; quantization and ``fpn_sigma_k`` are added
    from the sensor config) for the confidence's reference-noise correction.

    Hybrid steps: (1) if no ``time_offset`` is given, estimate it with
    :func:`estimate_time_offset` from image-only ``"gn"`` shifts weighted
    by their confidence; (2) integrate the gyro as the seed; (3) refine
    each subpage by Gauss-Newton from the seed against the reference
    subpage; (4) fit ``image - gyro = a + c t`` per axis over the frames
    with confidence >= ``bias_min_confidence`` (inverse-variance weights
    rescaled by the robust residual scatter, ridge prior on ``c``,
    3-sigma clipping), giving the bias ``-c / f`` and the image anchor
    offset ``a`` (the gyro is exact relative to the reference, so ``a`` is
    removed from the image shifts); (5) build a multi-frame reference from
    the accepted image shifts and the corrected gyro shifts and re-register
    every subpage against it; (5b) estimate the fixed pattern from the
    residuals, remove it and re-register once more; (6) refit the bias and
    accept a frame's image shift when its confidence is at least
    ``min_confidence`` and it lies within ``max_gyro_dev`` of the
    bias-corrected gyro, otherwise fall back to the gyro.  The roll bias is
    not observable from translations and is reported as 0.
    """
    K = capture.n_subpages
    if not 0 <= ref_index < K:
        raise IndexError(f"ref_index {ref_index} out of range for {K} subpages")
    if method == "phase":
        return _register_phase(capture, ref_index, **kw)
    if method == "gn":
        return _register_gn(capture, ref_index, **kw)
    if method == "gyro":
        return _register_gyro(capture, ref_index, **kw)
    if method == "hybrid":
        return _register_hybrid(capture, ref_index, **kw)
    raise ValueError(f"unknown registration method {method!r}")


# ---------------------------------------------------------------------------
# Alternating re-registration against an HR estimate
# ---------------------------------------------------------------------------


def _box_kernel(scale: int) -> np.ndarray:
    """Integer-offset weights of a centered box of width ``scale`` target px."""
    half = scale / 2.0
    n = int(math.ceil(half + 0.5))
    t = np.arange(-n, n + 1, dtype=float)
    w = np.clip(np.minimum(t + 0.5, half) - np.maximum(t - 0.5, -half), 0.0, None)
    w = w[w > 0]
    return w / w.sum()


def refine_against_hr(capture: Capture, hr_estimate: np.ndarray, *, init: np.ndarray | Registration | None = None,
                      psf_sigma_px: float | None = None, iters: int = 20, tol: float = 1e-4,
                      max_shift: float = 4.0) -> Registration:
    """Re-register every subpage against an HR estimate (alternating SR).

    ``hr_estimate`` is ``(24 s, 32 s)`` on a target grid (see
    :mod:`mlx_sr.types`); the returned shifts are relative to that grid's
    reference pose (for a reconstruction built from shifts relative to
    ``ref_index`` that is the reference subpage, but its entry is not
    forced to zero).  The predicted LR reading is the HR estimate blurred by
    the Gaussian PSF (``psf_sigma_px``, default the sensor's) and a box of
    one LR pixel, sampled with a cubic spline at the footprint center.
    ``init``: ``(K, 2)`` shifts or a :class:`Registration` (default zeros).
    Confidence is the noise-normalized Hessian eigenvalue without the
    reference-noise correction (the HR estimate is treated as noiseless).
    """
    hr = np.asarray(hr_estimate, dtype=float)
    rows, cols = capture.sensor.shape
    scale = hr.shape[0] // rows
    if scale < 1 or hr.shape != (rows * scale, cols * scale):
        raise ValueError(f"hr_estimate shape {hr.shape} is not an integer multiple of {(rows, cols)}")
    psf = capture.sensor.psf_sigma_px if psf_sigma_px is None else float(psf_sigma_px)
    img = ndimage.gaussian_filter(hr, psf * scale, mode="nearest") if psf > 0 else hr
    k = _box_kernel(scale)
    img = ndimage.convolve1d(img, k, axis=0, mode="nearest")
    img = ndimage.convolve1d(img, k, axis=1, mode="nearest")
    hr_fn = spline_interpolant(img)

    def ref_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return hr_fn((np.asarray(x) + 0.5) * scale - 0.5, (np.asarray(y) + 0.5) * scale - 0.5)

    K = capture.n_subpages
    if isinstance(init, Registration):
        init = init.shifts
    s0 = np.zeros((K, 2)) if init is None else np.asarray(init, dtype=float).reshape(K, 2)
    shifts = np.zeros((K, 2))
    conf = np.zeros(K)
    for n, sp in enumerate(capture.subpages):
        s, c = gauss_newton_translation(ref_fn, sp.values, sp.mask, _clip_norm(s0[n], max_shift),
                                        iters=iters, tol=tol)
        if float(np.hypot(*s)) > max_shift + 1.0:
            s, c = s0[n], 0.0
        shifts[n], conf[n] = s, c
    return Registration(shifts=shifts, confidence=conf, method="hr_refine",
                        info=dict(scale=scale, sigma_px=_sigma_from_conf(conf)))
