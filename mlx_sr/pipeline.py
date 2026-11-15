"""End-to-end simulate -> register -> reconstruct -> evaluate (Phase 3).

:func:`run_pipeline` takes a simulated :class:`~mlx_sr.types.Capture` and a
:class:`~mlx_sr.types.Registration` and reconstructs the scene with the
requested method *and* every baseline, then scores all of them against the
same ground truth.  :func:`run_case` wraps simulation and registration
around it (with timings).

Reconstructions compared (keys of ``result["metrics"]``)
--------------------------------------------------------
``"bicubic"``
    Single full frame (the frame containing the reference subpage) merged
    Adafruit-style, then bicubic.  The "what you get today" baseline.  Under
    fast motion the two halves of the frame come from different poses,
    which this baseline (like real firmware) ignores.
``"bicubic_single"``
    The reference subpage alone, chess holes filled, then bicubic (immune to
    the in-frame motion, but half the samples).
``"average"``
    Registered average at the native grid, then bicubic: multi-frame
    *denoising* with no resolution beyond the LR grid.  The gap between the
    method and this baseline is *not* all super-resolution: it also holds
    the method's prior and PSF deconvolution (see the still twin below).
``"nc"``, ``"ibp"``, ``"btv"``
    Multi-frame reconstructions on the estimated shifts.
``"<name>@true"``
    The same reconstruction on the *true* relative shifts (oracle
    registration).  The gap to ``"<name>"`` is the cost of registration
    error.
``"<method>@still"`` (``run_case(still_control=True)`` only)
    The method run on a *still twin*: the same scene, seed, temporal noise
    realization and fixed pattern, but no motion at all (so no sub-pixel
    diversity; zero shifts).  Whatever the method gains on the still twin
    comes from its prior, deconvolution and noise averaging, not from
    super-resolution.  The gap ``method - method@still`` is the total effect
    of the motion (within-exposure blur or jitter plus sub-pixel spread, and
    fixed-pattern decorrelation).  This control matters for BTV: its
    edge-preserving prior sharpens the chart's step edges (raising the
    slanted-edge MTF50 well above the bicubic baseline) even with no motion.
``"<method>@nospread"`` (``run_case(nospread_control=True)`` only)
    The method on a *no-spread twin*: same seed, noise and fixed pattern,
    and the same motion *within* each exposure, but with each subpage's
    mean pose removed (:func:`nospread_trace`), so there is no sub-pixel
    spread between subpages.  ``method - method@nospread`` isolates the
    effect of the spread, the only part that can be super-resolution.  PSNR
    alone can still rise for reasons other than SR (e.g. blur that happens
    to suit the prior), so read it together with the high-band fidelity
    difference.

Grids
-----
Registration shifts are relative to subpage ``ref_index``, so every
reconstruction here (estimated or oracle) is built on subpage
``ref_index``'s grid: the oracle uses ``capture.relative_true_shifts(r)``.
All of them are scored against ``truth_at_scale(capture, s, ref_index=r)``
and scene ROIs are mapped with ``shift = capture.ref_shift(r)``.  The
bicubic baseline is the frame containing subpage ``r``, which is on that
grid up to the in-frame motion (part of its handicap, as on real hardware).

Metrics (per reconstruction)
----------------------------
``psnr`` / ``ssim`` against the box-averaged truth with an explicit
``data_range`` (the truth's range) and a border of ``2 s`` target px;
``rmse_k``; ``hf_fidelity`` (:func:`mlx_sr.metrics.highband_fidelity`: 1 = the
detail above native Nyquist recovered exactly, 0 = none, negative = wrong
detail; a prior that suits the scene already scores above 0 with no motion,
so it is meaningful only as a difference between reconstructions, e.g.
against the still or no-spread twin; NaN when the truth has less than
``HF_MIN_RMS_K`` of such detail); ``mtf50`` and ``mtf_at_0.6`` / ``mtf_at_0.75`` (slanted edge,
cycles per LR px; native Nyquist 0.5) and ``mtf_valid``; ``bar_limit_freq``
(highest resolvable bar group, cycles per LR px; NaN when none) and
``bar_contrast_<f>`` per group; ``hotspot_err_k`` per hot spot (signed peak
error); ``flat_sigma_k``; ``finite`` (all pixels finite); ``runtime_s``.
Scene features that are absent give NaN.

Summary values
--------------
``effective_gain`` = ``mtf50[method] / mtf50[bicubic]`` (NaN if either MTF is
invalid), ``effective_gain_vs_average`` likewise against the registered
average, ``bar_ratio`` = ``bar_limit_freq[method] / bar_limit_freq[bicubic]``,
``psnr_gain_db`` / ``psnr_gain_vs_average_db``, ``hf_gain`` =
``hf_fidelity[method] - hf_fidelity[bicubic]`` and ``oracle_gap_db`` =
``psnr[method@true] - psnr[method]``.  With the still control:
``motion_gain_db`` = ``psnr[method] - psnr[method@still]``,
``motion_mtf_ratio`` = ``mtf50[method] / mtf50[method@still]`` and
``motion_bar_ratio`` likewise for the bar limit and ``motion_hf_gain`` =
``hf_fidelity[method] - hf_fidelity[method@still]``.  With the no-spread
control: ``spread_gain_db`` = ``psnr[method] - psnr[method@nospread]``,
``spread_hf_gain`` likewise for ``hf_fidelity``, ``nospread_residual_px``
(the twin's remaining spread) and, with both controls,
``within_exposure_db`` / ``within_exposure_hf`` = no-spread minus still.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np

from . import metrics as M
from .motion import angles_to_shift
from .reconstruct import reconstruct
from .registration import CONF_REF, register
from .sensor import simulate_capture, truth_at_scale
from .types import Capture, GyroConfig, MotionConfig, MotionTrace, Registration, Scene, SensorConfig

__all__ = ["run_pipeline", "run_case", "evaluate_image", "infer_ref_index", "flatten_result",
           "truth_highband_rms", "motion_diagnostics", "nospread_trace", "DEFAULT_BASELINES", "MTF_FREQS", "HF_MIN_RMS_K"]

DEFAULT_BASELINES: tuple[str, ...] = ("bicubic", "average", "nc")
"""Baselines computed alongside the method (``"ibp"`` is available but slower)."""

MTF_FREQS: tuple[float, ...] = (0.6, 0.75)
"""Frequencies (cycles per LR px) at which the MTF is reported, above native Nyquist."""

HF_MIN_RMS_K: float = 0.2
"""``hf_fidelity`` is reported (else NaN) only when the truth's RMS above the
native Nyquist exceeds this many K: below it (e.g. the smooth ``textured``
scene, about 0.07 K) there is essentially no detail to super-resolve and the
ratio is dominated by noise."""

_VARIANTS = {"bicubic_single": ("bicubic", {"mode": "single"})}
"""Baseline names that map to a method plus fixed keyword arguments."""

_SHIFT_INDEPENDENT = {"bicubic"}


def infer_ref_index(registration: Registration) -> int:
    """Reference subpage of a registration: the entry with zero shift and the
    reference confidence (``CONF_REF``); falls back to the first zero shift."""
    s = np.asarray(registration.shifts, float)
    c = np.asarray(registration.confidence, float)
    zero = np.all(s == 0.0, axis=1)
    cand = np.flatnonzero(zero & (c >= CONF_REF))
    if cand.size:
        return int(cand[0])
    cand = np.flatnonzero(zero)
    if cand.size:
        return int(cand[0])
    raise ValueError("cannot infer the reference subpage (no zero shift); pass ref_index")


def truth_highband_rms(truth: np.ndarray, scale: int, border: int, f_cut: float = 0.5) -> float:
    """RMS (K) of the Hann-windowed truth above ``f_cut`` cycles per LR px."""
    if scale < 2:
        return float("nan")
    t = np.asarray(truth, float)[border:-border or None, border:-border or None]
    w = np.outer(np.hanning(t.shape[0]), np.hanning(t.shape[1]))
    R = np.abs(np.fft.fft2((t - np.sum(t * w) / np.sum(w)) * w)) ** 2
    fy = np.fft.fftfreq(t.shape[0]) * scale
    fx = np.fft.fftfreq(t.shape[1]) * scale
    hi = np.maximum(np.abs(fy)[:, None], np.abs(fx)[None, :]) > f_cut
    return float(np.sqrt(R[hi].sum() / np.sum(w ** 2) / t.size))


def evaluate_image(img: np.ndarray, capture: Capture, truth: np.ndarray, scale: int,
                   shift: np.ndarray, *, data_range: float, border: int,
                   keep_curve: bool = False, report_hf: bool = True) -> dict[str, Any]:
    """All metrics of one reconstruction (see the module docstring)."""
    img = np.asarray(img, float)
    out: dict[str, Any] = {
        "finite": bool(np.isfinite(img).all()),
        "psnr": M.psnr(img, truth, data_range=data_range, border=border),
        "ssim": M.ssim(img, truth, data_range=data_range, border=border),
    }
    b = img[border:-border, border:-border] - truth[border:-border, border:-border]
    out["rmse_k"] = float(np.sqrt(np.nanmean(b ** 2)))
    out["hf_fidelity"] = (M.highband_fidelity(img, truth, scale, border=border)
                          if scale >= 2 and report_hf else float("nan"))
    meta = capture.scene_meta
    sh = (float(shift[0]), float(shift[1]))
    out["mtf50"] = float("nan")
    out["mtf_valid"] = False
    for f in MTF_FREQS:
        out[f"mtf_at_{f:g}"] = float("nan")
    out["bar_limit_freq"] = float("nan")
    out["flat_sigma_k"] = float("nan")
    if meta is None:
        return out
    if meta.edge is not None:
        mtf = M.slanted_edge_mtf(img, meta.edge.roi, scale, sh)
        out["mtf_valid"] = bool(mtf["valid"])
        if mtf["valid"]:
            out["mtf50"] = float(mtf["mtf50"])
            for f in MTF_FREQS:
                out[f"mtf_at_{f:g}"] = float(mtf["mtf_at"](f))
            if keep_curve:
                out["mtf_curve"] = (np.asarray(mtf["freq"]), np.asarray(mtf["mtf"]))
    if meta.bars:
        bc = M.bar_contrast(img, meta.bars, scale, sh)
        out["bar_limit_freq"] = float(bc["limit_freq"])
        for f, c in zip(bc["freq"], bc["contrast"]):
            out[f"bar_contrast_{f:.2f}"] = float(c)
    for n, hs in enumerate(meta.hotspots):
        out[f"hotspot_err_k_r{hs.radius_lr:g}"] = M.hotspot_error(img, hs.peak_c, hs.roi, scale, sh)
    if meta.flat:
        out["flat_sigma_k"] = M.flat_sigma(img, meta.flat[0].roi, scale, sh)
    return out


def motion_diagnostics(capture: Capture, ref_index: int = 0, n_samples: int = 33) -> dict[str, float]:
    """Truth-side motion summary: ``shift_spread_px`` (RMS distance of the
    exposure-averaged subpage shifts from their mean, i.e. the sub-pixel
    diversity available) and ``intra_blur_px`` (RMS of the shift within each
    exposure about its mean, averaged over subpages: the motion blur the
    reconstruction operator does not model)."""
    s = capture.relative_true_shifts(ref_index)
    spread = float(np.sqrt(np.mean(np.sum((s - s.mean(axis=0)) ** 2, axis=1))))
    blur = []
    for sp in capture.subpages:
        t = sp.t_start + (np.arange(n_samples) + 0.5) / n_samples * (sp.t_end - sp.t_start)
        p = angles_to_shift(capture.motion.theta_at(t), capture.sensor)
        blur.append(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))
    return {"shift_spread_px": spread, "intra_blur_px": float(np.mean(blur))}


def _ratio(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float("nan")
    return float(a / b)


def run_pipeline(capture: Capture, registration: Registration | None, method: str = "btv",
                 scale: int = 2, *, ref_index: int | None = None,
                 baselines: Iterable[str] = DEFAULT_BASELINES, oracle: bool | Iterable[str] = True,
                 recon_kw: dict[str, dict] | None = None, keep_images: bool = False,
                 keep_curves: bool = False) -> dict[str, Any]:
    """Reconstruct with ``method`` and the baselines, and score everything.

    Parameters
    ----------
    capture, registration
        The simulated capture and a registration of it.  ``registration=None``
        runs the oracle reconstructions only (``oracle`` must be truthy).
    method
        ``"btv"`` (default), ``"nc"``, ``"ibp"`` or ``"average"``.
    ref_index
        Reference subpage of ``registration``; inferred if None.
    baselines
        Extra reconstructions on the estimated shifts (``"bicubic"`` is
        always included because the gains are defined against it).
    oracle
        ``True`` adds ``method@true`` and ``average@true``; an iterable names
        the methods to run on the true shifts; ``False`` disables.
    recon_kw
        Per-method keyword arguments for :func:`mlx_sr.reconstruct.reconstruct`,
        e.g. ``{"btv": {"estimate_fpn": True}}``.
    keep_images / keep_curves
        Store the reconstructions (``result["reconstructions"]``) / the MTF
        curves (``metrics[name]["mtf_curve"]``).

    Returns
    -------
    dict with ``method``, ``scale``, ``ref_index``, ``reconstruction`` (the
    method's, estimated shifts; oracle if ``registration`` is None),
    ``shifts`` and ``true_shifts`` (relative to ``ref_index``),
    ``registration`` (``method``, ``rms_px``, ``max_err_px``,
    ``time_offset_s``, ``n_fallback``), ``metrics`` (per reconstruction, see
    the module docstring), the summary gains, ``timings`` (seconds per
    reconstruction) and ``truth``.
    """
    recon_kw = recon_kw or {}
    if registration is None:
        if not oracle:
            raise ValueError("registration=None needs oracle reconstructions")
        r = 0 if ref_index is None else int(ref_index)
        est = None
    else:
        r = infer_ref_index(registration) if ref_index is None else int(ref_index)
        est = np.asarray(registration.shifts, float)
    true_rel = capture.relative_true_shifts(r)
    shift_ref = capture.ref_shift(r)
    truth = truth_at_scale(capture, scale, ref_index=r)
    data_range = float(truth.max() - truth.min()) or 1.0
    border = 2 * scale
    hf_rms = truth_highband_rms(truth, scale, border)
    report_hf = bool(np.isfinite(hf_rms) and hf_rms >= HF_MIN_RMS_K)

    jobs: list[tuple[str, str, np.ndarray | None]] = [("bicubic", "bicubic", None)]
    if est is not None:
        for m in [method, *baselines]:
            if m != "bicubic" and all(j[0] != m for j in jobs):
                jobs.append((m, m, est))
    if oracle:
        names = [method, "average"] if oracle is True else list(oracle)
        for m in names:
            if m != "bicubic" and all(j[0] != f"{m}@true" for j in jobs):
                jobs.append((f"{m}@true", m, true_rel))

    metrics: dict[str, dict] = {}
    recons = {}
    timings: dict[str, float] = {}
    for name, m, shifts in jobs:
        kw = dict(recon_kw.get(m, {}))
        if m in _VARIANTS:
            m, extra = _VARIANTS[m]
            kw = {**recon_kw.get(m, {}), **extra}
        if m == "bicubic":
            kw.setdefault("ref_index", r)
        sh = shifts if shifts is not None else np.zeros((capture.n_subpages, 2))
        rec = reconstruct(capture.subpages, sh, capture.sensor, m, scale=scale, **kw)
        timings[name] = float(rec.info.get("runtime_s", np.nan))
        metrics[name] = evaluate_image(rec.image, capture, truth, scale, shift_ref,
                                       data_range=data_range, border=border, keep_curve=keep_curves,
                                       report_hf=report_hf)
        metrics[name]["runtime_s"] = timings[name]
        recons[name] = rec

    main = method if method in metrics else f"{method}@true"
    mb, mbi = metrics[main], metrics["bicubic"]
    ma = metrics.get("average", metrics.get("average@true"))
    out: dict[str, Any] = {
        "method": method,
        "scale": scale,
        "ref_index": r,
        "reconstruction": recons[main],
        "shifts": est,
        "true_shifts": true_rel,
        "metrics": metrics,
        "timings": timings,
        "truth": truth,
        "data_range": data_range,
        "truth_hf_rms_k": hf_rms,
        **motion_diagnostics(capture, r),
        "effective_gain": _ratio(mb["mtf50"], mbi["mtf50"]),
        "bar_ratio": _ratio(mb["bar_limit_freq"], mbi["bar_limit_freq"]),
        "psnr_gain_db": float(mb["psnr"] - mbi["psnr"]),
        "hf_gain": float(mb["hf_fidelity"] - mbi["hf_fidelity"]),
        "effective_gain_vs_average": _ratio(mb["mtf50"], ma["mtf50"]) if ma else float("nan"),
        "psnr_gain_vs_average_db": float(mb["psnr"] - ma["psnr"]) if ma else float("nan"),
        "oracle_gap_db": (float(metrics[f"{method}@true"]["psnr"] - metrics[method]["psnr"])
                          if method in metrics and f"{method}@true" in metrics else float("nan")),
    }
    if registration is not None:
        err = est - true_rel
        out["registration"] = {
            "method": registration.method,
            "rms_px": M.registration_rms(est, true_rel, remove="ref", ref_index=r),
            "max_err_px": float(np.max(np.hypot(err[:, 0], err[:, 1]))),
            "time_offset_s": registration.time_offset_s,
            "n_fallback": int(np.sum(registration.info.get("fallback", np.zeros(0, bool)))),
        }
    if keep_images:
        out["reconstructions"] = recons
    return out


def run_case(scene: Scene, sensor: SensorConfig, motion_cfg: MotionConfig | None,
             gyro_cfg: GyroConfig | None, n_frames: int, seed: int, *,
             reg_method: str | None = None, method: str = "btv", scale: int = 2,
             ref_index: int = 0, reg_kw: dict | None = None, still_control: bool | Iterable[str] = False,
             nospread_control: bool | Iterable[str] = False, **pipeline_kw) -> dict[str, Any]:
    """Simulate a capture, register it and run :func:`run_pipeline`.

    ``reg_method`` defaults to ``"hybrid"`` when a gyro is simulated and
    ``"gn"`` otherwise.  Adds ``capture``, ``reg_method`` and the timings
    ``simulate_s`` and ``register_s`` to the result.  ``still_control=True``
    also reconstructs a still twin (see the module docstring) and adds
    ``metrics["<method>@still"]`` and the ``motion_*`` summary values; an
    iterable of method names runs each of them on the twin.
    ``nospread_control`` likewise adds a no-spread twin
    (:func:`nospread_trace`: the same within-exposure motion with the
    subpage-to-subpage motion removed), ``metrics["<method>@nospread"]``
    and the ``spread_*`` / ``within_exposure_*`` summary values.
    """
    t0 = time.perf_counter()
    cap = simulate_capture(scene, sensor, motion_cfg, gyro_cfg, n_frames, seed)
    t1 = time.perf_counter()
    if reg_method is None:
        reg_method = "hybrid" if cap.gyro is not None else "gn"
    reg = register(cap, reg_method, ref_index=ref_index, **(reg_kw or {}))
    t2 = time.perf_counter()
    res = run_pipeline(cap, reg, method, scale, ref_index=ref_index, **pipeline_kw)
    res["capture"] = cap
    res["reg_method"] = reg_method
    res["timings"]["simulate_s"] = t1 - t0
    res["timings"]["register_s"] = t2 - t1
    if still_control:
        names = [method] if still_control is True else list(still_control)
        if method not in names:
            names.insert(0, method)
        twin = simulate_capture(scene, sensor, MotionConfig.still(), None, n_frames, seed)
        ctl = run_pipeline(twin, None, method, scale, ref_index=ref_index, baselines=(),
                           oracle=names, recon_kw=pipeline_kw.get("recon_kw"),
                           keep_curves=pipeline_kw.get("keep_curves", False),
                           keep_images=pipeline_kw.get("keep_images", False))
        for nm in names:
            res["metrics"][f"{nm}@still"] = ctl["metrics"][f"{nm}@true"]
            if "reconstructions" in res:
                res["reconstructions"][f"{nm}@still"] = ctl["reconstructions"][f"{nm}@true"]
            res["timings"][f"{nm}@still"] = ctl["metrics"][f"{nm}@true"]["runtime_s"]
        m, m_still = res["metrics"][method], res["metrics"][f"{method}@still"]
        res["motion_gain_db"] = float(m["psnr"] - m_still["psnr"])
        res["motion_mtf_ratio"] = _ratio(m["mtf50"], m_still["mtf50"])
        res["motion_bar_ratio"] = _ratio(m["bar_limit_freq"], m_still["bar_limit_freq"])
        res["motion_hf_gain"] = float(m["hf_fidelity"] - m_still["hf_fidelity"])
    if nospread_control:
        names = [method] if nospread_control is True else list(nospread_control)
        if method not in names:
            names.insert(0, method)
        twin = simulate_capture(scene, sensor, nospread_trace(cap), None, n_frames, seed)
        ctl = run_pipeline(twin, None, method, scale, ref_index=ref_index, baselines=(),
                           oracle=names, recon_kw=pipeline_kw.get("recon_kw"),
                           keep_images=pipeline_kw.get("keep_images", False))
        for nm in names:
            res["metrics"][f"{nm}@nospread"] = ctl["metrics"][f"{nm}@true"]
            res["timings"][f"{nm}@nospread"] = ctl["metrics"][f"{nm}@true"]["runtime_s"]
            if "reconstructions" in res:
                res["reconstructions"][f"{nm}@nospread"] = ctl["reconstructions"][f"{nm}@true"]
        m, m_ns = res["metrics"][method], res["metrics"][f"{method}@nospread"]
        res["nospread_residual_px"] = ctl["shift_spread_px"]
        res["spread_gain_db"] = float(m["psnr"] - m_ns["psnr"])
        res["spread_hf_gain"] = float(m["hf_fidelity"] - m_ns["hf_fidelity"])
        if f"{method}@still" in res["metrics"]:
            m_st = res["metrics"][f"{method}@still"]
            res["within_exposure_db"] = float(m_ns["psnr"] - m_st["psnr"])
            res["within_exposure_hf"] = float(m_ns["hf_fidelity"] - m_st["hf_fidelity"])
    return res


def nospread_trace(capture: Capture, samples_per_subpage: int = 500) -> MotionTrace:
    """Motion of ``capture`` with the subpage-to-subpage motion removed.

    Within every subpage window the true angles (and roll) are kept but
    their mean over the window is subtracted, so a capture simulated with
    this trace has the same within-exposure motion (blur under the
    ``"full"`` exposure model, per-pixel jitter under ``"staggered"``) but
    essentially zero sub-pixel spread between subpages (residual: the
    second-order ``tan`` nonlinearity).  It separates the effect of
    within-exposure motion from the effect of inter-subpage spread.
    """
    ts, th, rl = [], [], []
    has_roll = capture.motion.roll is not None
    eps = 1e-9
    for sp in capture.subpages:
        mid = sp.t_start + (np.arange(samples_per_subpage) + 0.5) / samples_per_subpage * (sp.t_end - sp.t_start)
        t = np.concatenate([[sp.t_start + eps], mid, [sp.t_end - eps]])
        theta = capture.motion.theta_at(t)
        th.append(theta - theta[1:-1].mean(axis=0))
        if has_roll:
            r = capture.motion.roll_at(t)
            rl.append(r - r[1:-1].mean())
        ts.append(t)
    return MotionTrace(t=np.concatenate(ts), theta=np.concatenate(th),
                       roll=np.concatenate(rl) if has_roll else None)


_SUMMARY_KEYS = ("effective_gain", "effective_gain_vs_average", "bar_ratio", "psnr_gain_db",
                 "psnr_gain_vs_average_db", "hf_gain", "oracle_gap_db", "motion_gain_db",
                 "motion_mtf_ratio", "motion_bar_ratio", "motion_hf_gain", "spread_gain_db",
                 "spread_hf_gain", "within_exposure_db", "within_exposure_hf", "nospread_residual_px")


def flatten_result(res: dict[str, Any]) -> dict[str, Any]:
    """One flat ``{column: scalar}`` row per pipeline result (for CSV output):
    ``<recon>.<metric>`` for every reconstruction plus the summary values,
    ``reg.*`` and ``time.*``."""
    row: dict[str, Any] = {k: res.get(k, np.nan) for k in _SUMMARY_KEYS}
    for k in ("truth_hf_rms_k", "shift_spread_px", "intra_blur_px"):
        row[k] = res.get(k, np.nan)
    for name, mets in res["metrics"].items():
        for k, v in mets.items():
            if k == "mtf_curve":
                continue
            row[f"{name}.{k}"] = float(v) if isinstance(v, (bool, np.bool_)) else v
    for k, v in res.get("registration", {}).items():
        if k != "method":
            row[f"reg.{k}"] = np.nan if v is None else v
    for k, v in res["timings"].items():
        row[f"time.{k}"] = v
    return row
