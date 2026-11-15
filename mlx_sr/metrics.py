"""Image-quality metrics (Phase 2C).

Conventions
-----------
Images
    2-D arrays in degrees C on a target grid of scale ``s`` (``s = 1`` is the
    LR sensor grid).  Target pixel ``(u, v)`` covers LR scene coords
    ``[v/s, (v+1)/s) x [u/s, (u+1)/s)`` plus the grid's reference ``shift``
    (see :mod:`mlx_sr.types`).  NaN (and other non-finite) pixels are ignored
    by every metric.

ROIs
    :class:`mlx_sr.types.Roi` in LR scene coords.  ``Roi.to_scale(scale, shift)``
    selects the target pixels lying fully inside the ROI; the selection is
    clipped to the image, so a ROI that runs off the image edge uses only the
    pixels that exist, and an empty selection gives NaN results instead of an
    exception.

Units
    Distances are in LR pixels and spatial frequencies in **cycles per LR
    pixel** whatever the image scale, so the native (sensor) Nyquist is 0.5
    and the Nyquist of a scale-``s`` grid is ``0.5 s``.  Temperatures and
    temperature errors are in K (degrees C differences).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import ndimage

from .types import BarGroup, Roi

__all__ = [
    "psnr",
    "ssim",
    "slanted_edge_mtf",
    "bar_contrast",
    "hotspot_error",
    "flat_sigma",
    "registration_rms",
    "MTF_OVERSAMPLE",
    "BAR_CONTRAST_THRESHOLD",
]

MTF_OVERSAMPLE: int = 4
"""ESF bins per target-grid pixel pitch (ISO 12233 uses 4x)."""

BAR_CONTRAST_THRESHOLD: float = 0.1
"""Minimum bar contrast (``(I_max - I_min) / (high - low)``) for a group to count as resolved."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crop(a: np.ndarray, border: int) -> np.ndarray:
    return a[border:-border, border:-border] if border > 0 else a


def _roi_patch(img: np.ndarray, roi: Roi, scale: float, shift) -> tuple[np.ndarray, int, int]:
    """Pixels of ``img`` fully inside ``roi`` (clipped to the image).

    Returns ``(patch, row0, col0)``; ``patch`` may be empty.
    """
    a = np.asarray(img, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {a.shape}")
    sh = (0.0, 0.0) if shift is None else shift
    rs, cs = roi.to_scale(scale, sh)
    r0, r1 = min(rs.start, a.shape[0]), min(rs.stop, a.shape[0])
    c0, c1 = min(cs.start, a.shape[1]), min(cs.stop, a.shape[1])
    return a[r0:r1, c0:c1], r0, c0


def _shift_xy(shift) -> tuple[float, float]:
    if shift is None:
        return 0.0, 0.0
    return float(shift[0]), float(shift[1])


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------


def _check_nan_policy(nan_policy: str) -> None:
    if nan_policy not in ("omit", "propagate"):
        raise ValueError(f"nan_policy must be 'omit' or 'propagate', not {nan_policy!r}")


def psnr(img: np.ndarray, ref: np.ndarray, *, data_range: float | None = None, border: int = 0,
         nan_policy: str = "omit") -> float:
    """PSNR in dB.  ``data_range`` defaults to ``ref.max() - ref.min()`` (after
    cropping ``border`` pixels).  Identical images give ``inf``.

    ``nan_policy="omit"`` (default) excludes pixels where either image is
    non-finite (NaN if none remain); ``"propagate"`` returns NaN if any pixel
    is non-finite.  "omit" does not penalize holes, so when comparing methods
    check that reconstructions are finite (or use "propagate").  Pass an
    explicit ``data_range`` when comparing methods so the score does not
    depend on the reference's extremes.
    """
    _check_nan_policy(nan_policy)
    a = _crop(np.asarray(img, float), border)
    b = _crop(np.asarray(ref, float), border)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any() or (nan_policy == "propagate" and not ok.all()):
        return float("nan")
    a, b = a[ok], b[ok]
    dr = float(b.max() - b.min()) if data_range is None else float(data_range)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    if dr <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(dr * dr / mse))


def ssim(img: np.ndarray, ref: np.ndarray, *, data_range: float | None = None, border: int = 0,
         win_size: int | None = None, nan_policy: str = "omit") -> float:
    """Mean SSIM (scikit-image, uniform window, default constants).

    ``data_range`` defaults to the cropped reference's range (pass it
    explicitly when comparing methods).  ``win_size`` defaults to 7 and is
    rounded down to an odd size and reduced to fit tiny images (minimum 3).
    With ``nan_policy="omit"`` (default) windows touching a non-finite pixel
    in either image are excluded from the mean (NaN if no window is valid);
    ``"propagate"`` returns NaN if any pixel is non-finite.  Identical finite
    images give 1.
    """
    _check_nan_policy(nan_policy)
    from skimage.metrics import structural_similarity

    a = _crop(np.asarray(img, float), border)
    b = _crop(np.asarray(ref, float), border)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected 2-D images, got shape {a.shape}")
    bad = ~(np.isfinite(a) & np.isfinite(b))
    if bad.all() or (nan_policy == "propagate" and bad.any()):
        return float("nan")
    dr = float(np.max(b[~bad]) - np.min(b[~bad])) if data_range is None else float(data_range)
    if dr <= 0.0:
        return 1.0 if np.array_equal(a[~bad], b[~bad]) else float("nan")
    w = 7 if win_size is None else int(win_size)
    w = min(w, min(a.shape))
    w -= int(w % 2 == 0)
    if w < 3:
        return float("nan")
    a = np.where(bad, 0.0, a)
    b = np.where(bad, 0.0, b)
    with warnings.catch_warnings():
        # skimage 0.27 flags the v2 signature change; data_range is always explicit here.
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        _, smap = structural_similarity(a, b, data_range=dr, win_size=w, full=True)
    # Exclude windows that touch an invalid pixel, and (like skimage) the
    # half-window border where the window is incomplete.
    invalid = ndimage.binary_dilation(bad, structure=np.ones((w, w), bool)) if bad.any() else bad
    pad = (w - 1) // 2
    keep = ~invalid
    keep[:pad, :] = keep[-pad:, :] = False
    keep[:, :pad] = keep[:, -pad:] = False
    if not keep.any():
        return float("nan")
    return float(np.mean(smap[keep]))


# ---------------------------------------------------------------------------
# Slanted-edge MTF
# ---------------------------------------------------------------------------


def _nan_mtf(reason: str, scale: float) -> dict:
    f = np.zeros(0)

    def mtf_at(freq):
        out = np.full(np.shape(freq), np.nan)
        return float(out) if out.ndim == 0 else out

    return {"freq": f, "freqs": f, "mtf": np.zeros(0), "mtf50": float("nan"), "mtf_at": mtf_at,
            "nyquist": 0.5 * scale, "angle_deg": float("nan"), "esf": np.zeros(0),
            "esf_x": np.zeros(0), "lsf": np.zeros(0), "n_lines": 0, "phase_cycles": float("nan"),
            "transposed": False, "lsf_rms": float("nan"), "truncated": False,
            "valid": False, "reason": reason}


def _first_crossing(f: np.ndarray, m: np.ndarray, level: float) -> float:
    """Frequency where ``m`` first falls to ``level`` (linear interpolation)."""
    below = np.nonzero(m <= level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    k = below[0]
    f0, f1, m0, m1 = f[k - 1], f[k], m[k - 1], m[k]
    if m0 == m1:
        return float(f1)
    return float(f0 + (m0 - level) * (f1 - f0) / (m0 - m1))


def _row_centroids(z: np.ndarray, x: np.ndarray, center: np.ndarray | None, half_win: float
                   ) -> np.ndarray:
    """Per-row centroid (in the coordinate ``x``) of the absolute first difference.

    ``z`` (rows, cols) with NaN rows already handled; ``x`` the column
    coordinates.  If ``center`` is given, a Hann window of half-width
    ``half_win`` around it suppresses noise far from the edge.
    """
    d = np.abs(np.diff(z, axis=1))
    xm = 0.5 * (x[1:] + x[:-1])
    if center is not None:
        u = (xm[None, :] - center[:, None]) / half_win
        d = d * np.where(np.abs(u) < 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
    w = d.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (d * xm[None, :]).sum(axis=1) / w
    c[~(w > 0)] = np.nan
    return c


def slanted_edge_mtf(img: np.ndarray, roi: Roi, scale: int = 1, shift=(0.0, 0.0), *,
                     oversample: int = MTF_OVERSAMPLE, f_max: float | None = None) -> dict:
    """ISO-12233-style slanted-edge MTF of the edge inside ``roi``.

    Steps: crop the ROI (target pixels fully inside it); orient the edge as
    near-vertical (transpose if near-horizontal); find the edge position on
    each line as the centroid of the absolute first difference (two passes,
    the second Hann-windowed around the first fit); least-squares line fit;
    project every pixel center onto the edge normal; bin the ESF at
    ``1 / (oversample * scale)`` LR px; differentiate (central difference),
    apply a window centered on the edge that is flat over the inner half of
    each side of the occupied ESF and Hann-tapered over the outer half; FFT; normalize to 1 at DC.  The MTF is divided by
    the transfer functions of the central difference (``sinc(2 f h)``) and of
    the bin averaging (``sinc(f h)``), ``h`` being the bin width, so a
    perfect estimator returns the image's MTF.

    The measured MTF includes the target-grid pixel aperture and the
    reconstruction itself, i.e. it is the MTF of the image as delivered.

    Returns a dict with ``freq`` (alias ``freqs``) in cycles per LR pixel
    from 0 to ``f_max`` (default ``scale``, twice the grid Nyquist), ``mtf``,
    ``mtf50`` (first crossing of 0.5, NaN if it does not fall that low
    within ``f_max``), ``mtf_at(f)`` (linear interpolation, NaN outside the
    returned range), ``nyquist`` (``0.5 * scale``), ``angle_deg`` (from the
    line's axis, positive = leaning right going down for a near-vertical
    edge), ``esf_x``/``esf``/``lsf`` (binned, LR px), ``n_lines``,
    ``phase_cycles`` (edge drift across the lines in target pixels; below
    about 1 the oversampled ESF has gaps, filled by interpolation, and the
    MTF is less reliable), ``transposed``, ``lsf_rms`` (RMS half-width of the
    windowed LSF, LR px), ``truncated`` (the ESF on the shorter side of the
    edge spans under 3 ``lsf_rms``: the MTF may be biased high; use a wider
    ROI) and ``valid``/``reason``.  Use a
    slant of a few degrees (the chart uses 7).  ``f_max`` is capped at
    ``scale * oversample / 4``.  Degenerate input (too small, no edge, all NaN)
    returns ``valid=False`` with NaN results instead of raising.
    """
    s = float(scale)
    patch, r0, c0 = _roi_patch(img, roi, s, shift)
    if patch.shape[0] < 3 or patch.shape[1] < 4:
        return _nan_mtf("roi too small", s)
    dx, dy = _shift_xy(shift)
    # Pixel-center coordinates in LR scene units.
    xs = (c0 + np.arange(patch.shape[1]) + 0.5) / s + dx
    ys = (r0 + np.arange(patch.shape[0]) + 0.5) / s + dy

    finite = np.isfinite(patch)
    if finite.sum() < 8:
        return _nan_mtf("too few finite pixels", s)
    # Orientation from the mean absolute gradient.
    z0 = np.where(finite, patch, float(np.mean(patch[finite])))
    gx = np.nanmean(np.abs(np.diff(z0, axis=1)))
    gy = np.nanmean(np.abs(np.diff(z0, axis=0)))
    transposed = bool(gy > gx)
    if transposed:
        patch, finite = patch.T, finite.T
        xs, ys = ys, xs
    if patch.shape[0] < 3 or patch.shape[1] < 4:
        return _nan_mtf("roi too small across the edge", s)

    # Lines (rows of the oriented patch) without NaN are used for the edge fit.
    full = finite.all(axis=1)
    if full.sum() < 3:
        return _nan_mtf("too few complete lines", s)
    z = patch[full]
    yl = ys[full]
    step = float(np.max(patch[finite]) - np.min(patch[finite]))
    if not step > 0:
        return _nan_mtf("no edge contrast", s)

    c = _row_centroids(z, xs, None, 0.0)
    ok = np.isfinite(c)
    if ok.sum() < 3:
        return _nan_mtf("edge not found", s)
    b, a = np.polyfit(yl[ok], c[ok], 1)
    # Second pass: Hann window around the first fit to reject far-field noise.
    width = xs[-1] - xs[0]
    half = max(0.5 * width, 2.0 / s)
    c2 = _row_centroids(z, xs, a + b * yl, half)
    ok2 = np.isfinite(c2)
    if ok2.sum() >= 3:
        b, a = np.polyfit(yl[ok2], c2[ok2], 1)
        # One round of outlier rejection.
        res = c2 - (a + b * yl)
        mad = np.nanmedian(np.abs(res[ok2] - np.nanmedian(res[ok2])))
        keep = ok2 & (np.abs(res) <= max(5.0 * 1.4826 * mad, 0.05 / s))
        if keep.sum() >= 3 and keep.sum() < ok2.sum():
            b, a = np.polyfit(yl[keep], c2[keep], 1)
        n_lines = int(keep.sum()) if keep.sum() >= 3 else int(ok2.sum())
    else:
        n_lines = int(ok.sum())

    # Signed distance of every finite pixel center from the edge line.
    X, Y = np.meshgrid(xs, ys)
    cos_t = 1.0 / math.sqrt(1.0 + b * b)
    dist = ((X - (a + b * Y)) * cos_t)[finite]
    vals = patch[finite]

    h = 1.0 / (oversample * s)  # bin width, LR px
    lo = math.floor(dist.min() / h)
    hi = math.ceil(dist.max() / h)
    idx = np.floor(dist / h).astype(int) - lo
    nb = hi - lo
    cnt = np.bincount(idx, minlength=nb)[:nb]
    tot = np.bincount(idx, weights=vals, minlength=nb)[:nb]
    centers = (np.arange(nb) + lo + 0.5) * h
    filled = cnt > 0
    if filled.sum() < 8:
        return _nan_mtf("too few ESF bins", s)
    esf = np.interp(centers, centers[filled], tot[filled] / np.maximum(cnt[filled], 1))
    # Trim to the occupied range.
    first, last = np.nonzero(filled)[0][[0, -1]]
    centers, esf = centers[first:last + 1], esf[first:last + 1]
    left, right = -centers[0], centers[-1]
    if min(left, right) < 4.0 * h:
        return _nan_mtf("edge too close to the roi boundary", s)

    lsf = np.gradient(esf)  # central difference over 2 bins
    # Window: flat over the inner half of each side, Hann tapers over the
    # outer half (an asymmetric Tukey window centered on the edge).  A full
    # Hann over a short ESF narrows the LSF and inflates the MTF.
    win = np.ones_like(centers)
    for side, ext in ((centers < 0, left), (centers > 0, right)):
        u = (np.abs(centers[side]) - 0.5 * ext) / (0.5 * ext + h)
        win[side] = np.where(u > 0, 0.5 * (1.0 + np.cos(np.pi * np.clip(u, 0.0, 1.0))), 1.0)
    lsfw = lsf * win
    n = lsf.size
    # RMS half-width of the windowed LSF, to flag ROIs too short for it.
    wl = np.abs(lsfw)
    lsf_rms = float(np.sqrt(np.sum(wl * centers ** 2) / np.sum(wl))) if wl.sum() > 0 else float("nan")
    truncated = bool(not np.isfinite(lsf_rms) or min(left, right) < 3.0 * lsf_rms)
    nfft = int(2 ** math.ceil(math.log2(max(n, 1) * 4)))
    nfft = max(nfft, 1024)
    spec = np.abs(np.fft.rfft(lsfw, nfft))
    freq = np.fft.rfftfreq(nfft, d=h)
    if not spec[0] > 0 or abs(lsfw.sum()) < 1e-6 * step:
        return _nan_mtf("no edge contrast", s)
    mtf = spec / spec[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.sinc(2.0 * freq * h) * np.sinc(freq * h)
    fm = s if f_max is None else float(f_max)
    fm = min(fm, 0.25 / h)  # keep the sinc corrections well away from their zeros
    sel = freq <= fm + 1e-12
    freq, mtf, corr = freq[sel], mtf[sel], corr[sel]
    mtf = mtf / corr

    angle = math.degrees(math.atan(b))
    # Edge drift across the fitted lines, in target pixels: the number of
    # sampling-phase cycles the ESF oversampling relies on (ISO wants >= 1).
    phase_cycles = float(abs(b) * (yl.max() - yl.min()) * s)
    mtf50 = _first_crossing(freq, mtf, 0.5)

    def mtf_at(f):
        fa = np.asarray(f, dtype=float)
        out = np.interp(fa, freq, mtf, left=np.nan, right=np.nan)
        return float(out) if out.ndim == 0 else out

    return {"freq": freq, "freqs": freq, "mtf": mtf, "mtf50": mtf50, "mtf_at": mtf_at,
            "nyquist": 0.5 * s, "angle_deg": angle, "esf_x": centers, "esf": esf, "lsf": lsf,
            "n_lines": n_lines, "phase_cycles": phase_cycles, "transposed": transposed,
            "lsf_rms": lsf_rms, "truncated": truncated, "valid": True, "reason": ""}


# ---------------------------------------------------------------------------
# Bar groups
# ---------------------------------------------------------------------------


def _bar_profile(img: np.ndarray, g: BarGroup, scale: float, shift) -> tuple[np.ndarray, np.ndarray]:
    """Bar-averaged profile ``(pos, value)`` across the bars of group ``g``.

    The ROI is inset along the bar length (by ``min(1, length/4)`` LR px at
    each end) so the bar ends' blur is excluded, then averaged along the bars.
    ``pos`` are pixel-center coordinates (LR px) along the modulation axis.
    """
    roi = g.roi
    dx, dy = _shift_xy(shift)
    if g.orientation == "x":
        ins = min(1.0, 0.25 * roi.height)
        sub = Roi(roi.x0, roi.y0 + ins, roi.x1, roi.y1 - ins)
    else:
        ins = min(1.0, 0.25 * roi.width)
        sub = Roi(roi.x0 + ins, roi.y0, roi.x1 - ins, roi.y1)
    patch, r0, c0 = _roi_patch(img, sub, scale, shift)
    if patch.size == 0:
        return np.zeros(0), np.zeros(0)
    if g.orientation == "x":
        with np.errstate(invalid="ignore"):
            prof = np.nanmean(patch, axis=0) if np.isfinite(patch).any() else np.full(patch.shape[1], np.nan)
        pos = (c0 + np.arange(patch.shape[1]) + 0.5) / scale + dx
    else:
        with np.errstate(invalid="ignore"):
            prof = np.nanmean(patch, axis=1) if np.isfinite(patch).any() else np.full(patch.shape[0], np.nan)
        pos = (r0 + np.arange(patch.shape[0]) + 0.5) / scale + dy
    ok = np.isfinite(prof)
    return pos[ok], prof[ok]


def _bar_group_contrast(img: np.ndarray, g: BarGroup, scale: float, shift,
                        threshold: float) -> dict:
    nan = float("nan")
    pos, prof = _bar_profile(img, g, scale, shift)
    span = g.high_c - g.low_c
    out = {"freq": g.freq_cyc_per_lr, "period": g.period_lr, "contrast": nan, "michelson": nan,
           "modulation": nan, "i_max": nan, "i_min": nan, "ordered": False, "min_dip_contrast": nan,
           "above_nyquist": g.freq_cyc_per_lr > 0.5 * scale, "resolvable": False,
           "n_samples": int(pos.size)}
    if pos.size < 2 or span == 0:
        return out
    p = g.period_lr
    start = g.roi.x0 if g.orientation == "x" else g.roi.y0
    n = (prof - g.low_c) / span  # 0 = low level, 1 = high level

    def bar_stat(k: int, hot: bool) -> float:
        lo = start + k * p + (0.0 if hot else 0.5 * p)
        hi = lo + 0.5 * p
        m = (pos >= lo) & (pos < hi)
        if m.any():
            return float(n[m].max() if hot else n[m].min())
        c = 0.5 * (lo + hi)  # no sample inside: nearest sample to the bar center
        j = int(np.argmin(np.abs(pos - c)))
        return float(n[j]) if abs(pos[j] - c) <= 0.5 / scale + 0.25 * p else nan

    hot = np.array([bar_stat(k, True) for k in range(g.n_cycles)])
    # Cold bars between hot bars only: the last cold bar merges with the background.
    cold = np.array([bar_stat(k, False) for k in range(g.n_cycles - 1)])
    if not (np.isfinite(hot).any() and np.isfinite(cold).any()):
        return out
    nmax, nmin = float(np.nanmean(hot)), float(np.nanmean(cold))
    # Primary contrast: (I_max - I_min) / (high - low), offset-invariant.
    contrast = nmax - nmin
    den = nmax + nmin
    michelson = (nmax - nmin) / den if den > 0 else nan
    # Every interior cold bar must sit below both neighboring hot bars.
    ordered = bool(all(np.isfinite(cold[k]) and np.isfinite(hot[k]) and np.isfinite(hot[k + 1])
                       and cold[k] < hot[k] and cold[k] < hot[k + 1] for k in range(g.n_cycles - 1)))
    # Fundamental amplitude at the bar frequency over the group extent, as a
    # fraction of an ideal square wave's fundamental (4/pi * span/2).  A
    # quadratic envelope is fitted jointly so the group's overall bump (a
    # heavily blurred group still sits above the background) does not leak in.
    m = (pos >= start) & (pos < start + g.n_cycles * p)
    modulation = nan
    if m.sum() >= 6:
        modulation = _sine_amplitude(pos[m] - start, n[m], g.freq_cyc_per_lr,
                                     g.n_cycles * p) / (2.0 / np.pi)
    # Local contrast of each interior dip against its weaker neighbor.
    dips = np.full(g.n_cycles - 1, np.nan)
    for k in range(g.n_cycles - 1):
        dips[k] = min(hot[k], hot[k + 1]) - cold[k]
    min_dip = float(np.min(dips)) if dips.size and np.isfinite(dips).all() else nan
    out.update(contrast=float(contrast), michelson=float(michelson), modulation=float(modulation),
               min_dip_contrast=min_dip, i_max=g.low_c + nmax * span, i_min=g.low_c + nmin * span,
               ordered=ordered)
    out["resolvable"] = bool(np.isfinite(contrast) and contrast >= threshold and ordered
                             and np.isfinite(min_dip) and min_dip >= threshold
                             and not out["above_nyquist"])
    return out


def _sine_amplitude(x: np.ndarray, y: np.ndarray, f: float, length: float) -> float:
    """Least-squares amplitude of ``b cos(2 pi f x) + c sin(2 pi f x)`` fitted
    jointly with a quadratic envelope in ``x / length``."""
    ph = 2.0 * np.pi * f * x
    u = x / length - 0.5
    A = np.column_stack([np.ones(x.size), np.cos(ph), np.sin(ph), u, u * u])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(coef[1], coef[2]))


def bar_contrast(img: np.ndarray, bar_specs: list[BarGroup], scale: int = 1, shift=(0.0, 0.0), *,
                 threshold: float = BAR_CONTRAST_THRESHOLD) -> dict:
    """Bar-group contrast and the limiting resolvable group.

    For each group the image is averaged along the bars (inset by
    ``min(1, length/4)`` LR px from the bar ends) and normalized so the
    group's true low/high levels map to 0/1.  ``I_max`` is the mean over hot
    bars of each bar's maximum and ``I_min`` the mean over the interior cold
    bars (between hot bars; the last cold bar merges with the background) of
    each bar's minimum.

    ``contrast`` (the primary value, used for resolvability) is
    ``I_max - I_min`` on that scale, i.e. ``(I_max - I_min) / (high - low)``:
    the contrast transfer, 1 for perfect bars and 0 for uniform gray.  It
    equals the Michelson contrast ``(I_max - I_min) / (I_max + I_min)`` on the
    normalized scale whenever the bar mean is preserved (``I_max + I_min =
    1``), and unlike Michelson it does not change when the image has a DC
    offset (FPN, bias), which would otherwise move the denominator.  The
    Michelson value is reported as ``michelson`` (Michelson of raw degrees C
    is meaningless because it depends on the temperature zero).

    A group is ``resolvable`` when ``contrast >= threshold`` and every bar is
    individually distinguishable: each interior cold bar's dip below its
    weaker neighboring hot bar (``min_dip_contrast``, same units) is also
    ``>= threshold`` (which implies ``ordered``), and the bar frequency does
    not exceed the image grid's Nyquist (``0.5 * scale``), where any
    "contrast" would be aliasing.  The limiting group is the
    highest-frequency group reached, in order of increasing frequency,
    before the first unresolvable one (so spurious contrast above the limit
    does not count).  Interpolators leak spectral images: a bicubic upscale
    of LR data carries some correctly phased energy above the native Nyquist
    (the image of the alias), so on noisy baselines groups just above
    0.5 cyc/LR px could occasionally pass; average over seeds.

    Returns a dict: ``groups`` (per-group dicts in input order with
    ``freq``, ``period``, ``contrast``, ``michelson``, ``modulation``
    (fundamental amplitude relative to an ideal square wave, fitted with a
    quadratic envelope; NaN with fewer than 6 samples), ``i_max``/``i_min``
    (deg C), ``ordered``, ``min_dip_contrast``, ``above_nyquist``,
    ``resolvable``, ``n_samples``), arrays ``freq``, ``contrast``,
    ``michelson``, ``modulation``, ``resolvable`` (input order),
    ``limit_index`` (index into ``bar_specs`` or None), ``limit_freq`` and
    ``limit_period`` (NaN if none) and ``threshold``.
    """
    s = float(scale)
    groups = [_bar_group_contrast(img, g, s, shift, threshold) for g in bar_specs]
    freq = np.array([g["freq"] for g in groups], dtype=float)
    res = np.array([g["resolvable"] for g in groups], dtype=bool)
    limit = None
    for i in np.argsort(freq, kind="stable"):
        if not res[i]:
            break
        limit = int(i)
    return {
        "groups": groups,
        "freq": freq,
        "contrast": np.array([g["contrast"] for g in groups], dtype=float),
        "michelson": np.array([g["michelson"] for g in groups], dtype=float),
        "modulation": np.array([g["modulation"] for g in groups], dtype=float),
        "resolvable": res,
        "limit_index": limit,
        "limit_freq": float(freq[limit]) if limit is not None else float("nan"),
        "limit_period": float(1.0 / freq[limit]) if limit is not None else float("nan"),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# Hot spots, flat regions, registration
# ---------------------------------------------------------------------------


def hotspot_error(img: np.ndarray, truth_peak: float, roi: Roi, scale: int = 1, shift=(0.0, 0.0)) -> float:
    """Signed peak error in K: the maximum of the finite pixels inside ``roi``
    minus ``truth_peak`` (negative = the hot spot reads too cold, the usual
    effect of blur on a small target).  This is the "max temperature in a
    box" reading a thermal camera user sees, so noise biases it upward.
    NaN if the ROI holds no finite pixel."""
    patch, _, _ = _roi_patch(img, roi, float(scale), shift)
    v = patch[np.isfinite(patch)]
    if v.size == 0:
        return float("nan")
    return float(v.max() - float(truth_peak))


def flat_sigma(img: np.ndarray, roi: Roi, scale: int = 1, shift=(0.0, 0.0)) -> float:
    """Sample standard deviation (ddof=1, K) of the finite pixels inside a
    flat ``roi``: residual noise + FPN + reconstruction artifacts.  NaN with
    fewer than 2 finite pixels.  Note that pixels on a scale-``s`` grid are
    correlated, so this is not directly comparable to per-LR-pixel NETD."""
    patch, _, _ = _roi_patch(img, roi, float(scale), shift)
    v = patch[np.isfinite(patch)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1))


def registration_rms(est: np.ndarray, true: np.ndarray, *, remove: str = "mean",
                     ref_index: int = 0) -> float:
    """RMS shift error in LR px after removing the global offset between
    ``est`` and ``true``.

    ``est`` and ``true`` are ``(K, 2)`` ``(dx, dy)`` shifts (or ``(K,)`` for
    one axis).  The error per subpage is the Euclidean norm of
    ``est - true - offset``; the result is ``sqrt(mean(|e|^2))``.
    ``remove="mean"`` (default) takes the offset as the mean difference over
    the valid subpages; ``remove="ref"`` takes it as the difference at
    ``ref_index`` (so the reference has zero error, as in a registration
    that is relative to the reference subpage); ``remove="none"`` keeps it.
    Rows with a non-finite value in either input are ignored; NaN if none
    remain.
    """
    e = np.asarray(est, dtype=float)
    t = np.asarray(true, dtype=float)
    if e.shape != t.shape:
        raise ValueError(f"shape mismatch {e.shape} vs {t.shape}")
    if e.ndim == 1:
        e, t = e[:, None], t[:, None]
    d = e - t
    ok = np.isfinite(d).all(axis=1)
    if remove == "mean":
        if not ok.any():
            return float("nan")
        d = d - d[ok].mean(axis=0)
    elif remove == "ref":
        if not ok[ref_index]:
            return float("nan")
        d = d - d[ref_index]
    elif remove != "none":
        raise ValueError(f"remove must be 'mean', 'ref' or 'none', not {remove!r}")
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(d[ok] ** 2, axis=1))))
