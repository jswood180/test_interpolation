"""Reconstruction tests S1-S4 (plan section 5) plus baseline and API checks.

Fixtures use known shifts: ``simulate_capture(..., trace_from_shifts(s), ...)``
gives ``true_shift == s`` exactly, so reconstructions from ``s`` are scored
against the ``t = 0`` truth grid ``truth_at_scale(capture, 2)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlx_sr.metrics import highband_fidelity, psnr
from mlx_sr.motion import trace_from_shifts
from mlx_sr.reconstruct import (
    baseline_bicubic,
    btv_map,
    ibp,
    normalized_convolution,
    reconstruct,
    upsample_bicubic,
)
from mlx_sr.scene import make_scene
from mlx_sr.sensor import simulate_capture, truth_at_scale
from mlx_sr.types import BarGroup, SensorConfig

SCALE = 2
BORDER = 4  # target px cropped for PSNR (replicate-boundary effects)


def bar_mtf(img: np.ndarray, group: BarGroup, scale: int) -> float:
    """MTF estimate at a bar group's frequency (self-contained helper).

    The rows of the group's ROI are averaged into a profile, the fundamental
    (sinusoid at the bar frequency, phase free, plus an offset) is fitted by
    least squares, and its amplitude is divided by the fundamental amplitude
    of the ideal square wave, ``(2 / pi) * (high - low)``.  1 = full contrast,
    0 = bars not resolved.  The truth at 2x gives about 0.87 at 0.6 cyc/LR px
    (the 2x box sampling's own MTF).
    """
    rs, cs = group.roi.to_scale(scale)
    prof = img[rs, cs].mean(axis=0)
    x = (np.arange(cs.start, cs.stop) + 0.5) / scale
    ph = 2 * np.pi * group.freq_cyc_per_lr * (x - group.roi.x0)
    M = np.stack([np.ones_like(x), np.cos(ph), np.sin(ph)], axis=1)
    coef, *_ = np.linalg.lstsq(M, prof, rcond=None)
    return float(np.hypot(coef[1], coef[2]) / ((2 / np.pi) * (group.high_c - group.low_c)))


def _shifts(seed: int, K: int = 16) -> np.ndarray:
    """Uniform shifts in [-0.75, 0.75] LR px; the first frame (subpages 0 and 1)
    is at the t = 0 pose so the single-frame bicubic baseline is aligned with
    the truth grid and not penalized by an Adafruit merge of two poses."""
    s = np.random.default_rng(seed).uniform(-0.75, 0.75, size=(K, 2))
    s[:2] = 0.0
    return s


def _capture(scene, sensor, s, seed):
    return simulate_capture(scene, sensor, trace_from_shifts(s, sensor), None, len(s) // 2, seed)


@pytest.fixture(scope="module")
def chart_scene():
    return make_scene("chart")


@pytest.fixture(scope="module")
def s1_case(chart_scene):
    """BAB, K = 16, PSF 0.4, 8 Hz (NETD 0.25 K at 1 Hz -> 0.71 K per subpage),
    default FPN 0.1 K."""
    sensor = SensorConfig.bab(psf_sigma_px=0.4, refresh_hz=8.0)
    s = _shifts(0)
    cap = _capture(chart_scene, sensor, s, seed=100)
    return cap, s, sensor, truth_at_scale(cap, SCALE)


def _bar(chart_scene, f):
    return next(g for g in chart_scene.meta.bars if abs(g.freq_cyc_per_lr - f) < 1e-6)


# S1 ------------------------------------------------------------------------


def test_s1_btv_beats_baselines(s1_case, chart_scene):
    """S1: BTV-MAP at 2x beats bicubic by >= 1 dB, the registered average by
    >= 0.5 dB, and both on MTF at 0.6 cyc/LR px.  (Measured: about +4.7 dB,
    +4.8 dB; MTF 0.28 vs 0.08 / 0.08.  On this chart the registered average
    is about as good as bicubic in PSNR, because edge blur, not noise,
    dominates the error.)

    This comparison includes what BTV's prior and PSF deconvolution give
    without any sub-pixel information (BTV on a zero-shift twin also beats
    bicubic by several dB); the super-resolution part is tested separately
    by :func:`test_s1_gain_requires_subpixel_shifts`."""
    cap, s, sensor, truth = s1_case
    assert np.ptp(s[:, 0]) >= 0.5 and np.ptp(s[:, 1]) >= 0.5
    g06 = _bar(chart_scene, 0.6)
    res = {m: reconstruct(cap.subpages, s, sensor, m, SCALE) for m in ("bicubic", "average", "btv")}
    p = {m: psnr(r.image, truth, border=BORDER) for m, r in res.items()}
    mtf = {m: bar_mtf(r.image, g06, SCALE) for m, r in res.items()}
    assert p["btv"] >= p["bicubic"] + 1.0, p
    assert p["btv"] >= p["average"] + 0.5, p
    assert mtf["btv"] > max(mtf["bicubic"], mtf["average"]), mtf
    assert all(r.image.shape == (48, 64) for r in res.values())
    assert res["btv"].info["runtime_s"] > 0.0


def test_s1_btv_beats_baselines_second_seed(chart_scene):
    """S1 robustness: a second shift draw and noise seed."""
    sensor = SensorConfig.bab(psf_sigma_px=0.4, refresh_hz=8.0)
    s = _shifts(2)
    cap = _capture(chart_scene, sensor, s, seed=102)
    truth = truth_at_scale(cap, SCALE)
    p = {m: psnr(reconstruct(cap.subpages, s, sensor, m, SCALE).image, truth, border=BORDER)
         for m in ("bicubic", "average", "btv")}
    assert p["btv"] >= p["bicubic"] + 1.0 and p["btv"] >= p["average"] + 0.5, p


def _btv_and_zero_shift_twin(chart_scene, sensor, s, seed, **kw):
    """BTV on shifts ``s`` and on a twin with the same noise seed but zero
    shifts; returns ``(psnr, hf)`` for both."""
    out = []
    for shifts in (s, np.zeros_like(s)):
        cap = _capture(chart_scene, sensor, shifts, seed)
        truth = truth_at_scale(cap, SCALE)
        img = reconstruct(cap.subpages, shifts, sensor, "btv", SCALE, **kw).image
        out.append((psnr(img, truth, border=BORDER), highband_fidelity(img, truth, SCALE, border=BORDER)))
    return out


PAIRED_SEEDS = range(8)
"""Shift draws ``k`` with noise seeds ``100 + k`` for the paired S1/S4 controls."""


def _paired_differences(chart_scene, sensor, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``(PSNR, HF)`` differences BTV(shifts) - BTV(zero-shift twin)."""
    dp, dh = [], []
    for k in PAIRED_SEEDS:
        (p, hf), (p0, hf0) = _btv_and_zero_shift_twin(chart_scene, sensor, _shifts(k), 100 + k, **kw)
        dp.append(p - p0)
        dh.append(hf - hf0)
    return np.asarray(dp), np.asarray(dh)


def test_s1_gain_requires_subpixel_shifts(chart_scene):
    """S1, paired control: the same capture with zero shifts (same noise and
    fixed pattern) isolates the super-resolution part of the S1 gain.
    Single pairs vary a lot (measured over shift draws k = 0-7 with noise
    seeds 100 + k: PSNR difference -0.07 to +1.93 dB, high-band fidelity
    difference +0.04 to +0.25), so the assertion is on the mean over the 8
    pairs (measured: +1.40 dB, sd 0.70; HF +0.146, sd 0.078), with
    thresholds about 2.7 standard errors below the measured means.  Most
    pairs must also gain HF."""
    sensor = SensorConfig.bab(psf_sigma_px=0.4, refresh_hz=8.0)
    dp, dh = _paired_differences(chart_scene, sensor)
    assert dp.mean() >= 0.7, dp
    assert dh.mean() >= 0.07, dh
    assert np.sum(dh > 0) >= 7, dh


def test_nc_and_ibp_improve_on_bicubic(s1_case, chart_scene):
    cap, s, sensor, truth = s1_case
    p_bic = psnr(reconstruct(cap.subpages, s, sensor, "bicubic").image, truth, border=BORDER)
    p_nc = psnr(reconstruct(cap.subpages, s, sensor, "nc").image, truth, border=BORDER)
    p_ibp = psnr(reconstruct(cap.subpages, s, sensor, "ibp", n_iter=10).image, truth, border=BORDER)
    assert p_nc > p_bic and p_ibp > p_bic


# S2 ------------------------------------------------------------------------


@pytest.mark.parametrize("K", [8, 16])
def test_s2_nc_certainty_positive(chart_scene, K):
    """S2a: normalized-convolution certainty > 0 everywhere for K >= 8."""
    sensor = SensorConfig.bab(n_dead=4)
    s = _shifts(5, K)
    cap = _capture(chart_scene, sensor, s, seed=5)
    rec = normalized_convolution(cap.subpages, s, sensor, SCALE)
    assert rec.certainty.shape == (48, 64)
    assert np.all(rec.certainty > 0)
    assert np.all(np.isfinite(rec.image))
    assert rec.image.min() > 20.0  # dead pixels (-273.15) did not leak in


def test_s2_ibp_residual_decreases(s1_case):
    """S2b: the IBP residual ||y - A x|| decreases monotonically over the
    first 10 iterations."""
    cap, s, sensor, _ = s1_case
    rec = ibp(cap.subpages, s, sensor, SCALE, n_iter=10)
    r = rec.info["residual_norms"]
    assert r.shape == (11,)
    assert np.all(np.diff(r) < 0), r


# S3 ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 4])
def test_s3_fpn_estimation_reduces_flat_sigma(chart_scene, seed):
    """S3: with FPN 0.2 K, estimate_fpn=True lowers the flat-region sigma.

    Also checks that the offset estimate is informative: its RMS error against
    the (zero-mean) true FPN is below the RMS of the FPN itself, i.e. the
    correction removes more pattern than it adds noise.  (Measured flat sigma
    0.230 -> 0.188 K and 0.293 -> 0.277 K; offset error about 0.17 K vs
    0.2 K FPN.)  With K/2 = 8 reads per pixel and 0.71 K temporal noise the
    per-pixel mean alone cannot resolve a 0.2 K pattern well, which is why
    the update uses the MAP shrinkage (see ``btv_map``).
    """
    sensor = SensorConfig.bab(psf_sigma_px=0.4, refresh_hz=8.0, fpn_sigma_k=0.2)
    s = _shifts(seed)
    cap = _capture(chart_scene, sensor, s, seed=100 + seed)
    rs, cs = chart_scene.meta.flat[0].roi.to_scale(SCALE)
    off = btv_map(cap.subpages, s, sensor, SCALE, estimate_fpn=False)
    on = btv_map(cap.subpages, s, sensor, SCALE, estimate_fpn=True)
    assert on.image[rs, cs].std() < off.image[rs, cs].std()
    true = cap.fpn - cap.fpn.mean()
    est = on.info["fpn"]
    assert est.shape == (24, 32) and abs(np.nanmean(est)) < 1e-9
    assert np.sqrt(np.nanmean((est - true) ** 2)) < np.sqrt(np.mean(true ** 2))
    truth = truth_at_scale(cap, SCALE)
    assert psnr(on.image, truth, border=BORDER) > psnr(off.image, truth, border=BORDER) - 0.2


# S4 ------------------------------------------------------------------------


def test_s4_psf_mismatch_still_beats_bicubic(chart_scene):
    """S4: simulated PSF 0.6 px, assumed 0.4 px: BTV still beats bicubic and
    the registered average on PSNR (inverse-crime robustness).  (Measured
    about +2.8 dB over bicubic.)  Like S1 this includes the prior's gain;
    see :func:`test_s4_psf_mismatch_paired_control` for what remains of the
    sub-pixel gain."""
    sensor = SensorConfig.bab(psf_sigma_px=0.6, refresh_hz=8.0)
    s = _shifts(1)
    cap = _capture(chart_scene, sensor, s, seed=101)
    truth = truth_at_scale(cap, SCALE)
    p_bic = psnr(reconstruct(cap.subpages, s, sensor, "bicubic").image, truth, border=BORDER)
    p_avg = psnr(reconstruct(cap.subpages, s, sensor, "average").image, truth, border=BORDER)
    rec = reconstruct(cap.subpages, s, sensor, "btv", psf_sigma=0.4)
    assert rec.info["psf_sigma"] == 0.4
    p_btv = psnr(rec.image, truth, border=BORDER)
    assert p_btv > p_bic + 1.0 and p_btv > p_avg, (p_btv, p_bic, p_avg)


def test_s4_psf_mismatch_paired_control(chart_scene):
    """S4, paired control: with the PSF mismatched (0.6 simulated, 0.4
    assumed) the shifts still help PSNR over the zero-shift twin, but the
    high-band fidelity gain shrinks to a fraction of S1's.  Over the same 8
    pairs as S1 (measured: PSNR +0.39 dB, sd 0.18, per pair +0.16 to
    +0.62; HF +0.030, sd 0.046, per pair -0.02 to +0.12) the mean PSNR
    gain stays positive and the mean HF gain stays well below S1's.  A wrong
    PSF assumption erases most of the super-resolution, which the README
    reports; this test pins that behavior so a change in either direction is
    noticed."""
    sensor = SensorConfig.bab(psf_sigma_px=0.6, refresh_hz=8.0)
    dp, dh = _paired_differences(chart_scene, sensor, psf_sigma=0.4)
    assert dp.mean() > 0.15, dp
    assert abs(dh.mean()) < 0.09, dh


# Baselines, alignment and API ------------------------------------------------


def test_upsample_bicubic_alignment_on_ramp():
    """Target pixel centers are at LR coordinate (u + 0.5)/s: a linear ramp
    sampled at LR centers is reproduced exactly (away from the border)."""
    for s in (2, 3):
        lr = 0.7 * (np.arange(32) + 0.5)[None, :] + 0.3 * (np.arange(24) + 0.5)[:, None]
        hr = upsample_bicubic(lr, s)
        ref = 0.7 * ((np.arange(32 * s) + 0.5) / s)[None, :] + 0.3 * ((np.arange(24 * s) + 0.5) / s)[:, None]
        np.testing.assert_allclose(hr[2 * s:-2 * s, 2 * s:-2 * s], ref[2 * s:-2 * s, 2 * s:-2 * s], atol=1e-12)


def test_all_methods_on_static_flat_scene_are_unbiased():
    """Every method returns the flat temperature on a noiseless flat scene,
    with dead pixels present (they must go through the mask)."""
    scene = make_scene("flat")
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, adc_lsb_k=0.0, n_dead=4)
    s = _shifts(3, 8)
    cap = _capture(scene, sensor, s, seed=3)
    value = scene.meta.flat[0].value_c
    for m in ("bicubic", "average", "nc", "ibp", "btv"):
        rec = reconstruct(cap.subpages, s, sensor, m, SCALE)
        assert rec.image.shape == (48, 64) and rec.scale == SCALE and rec.method == m
        np.testing.assert_allclose(rec.image, value, atol=1e-3, err_msg=m)


def test_bicubic_modes(s1_case):
    cap, s, sensor, truth = s1_case
    merged = baseline_bicubic(cap.subpages, sensor, SCALE, mode="merged")
    single = baseline_bicubic(cap.subpages, sensor, SCALE, mode="single")
    assert merged.image.shape == single.image.shape == (48, 64)
    # The merged pair has twice the samples of a chess-filled single subpage.
    assert psnr(merged.image, truth, border=BORDER) > psnr(single.image, truth, border=BORDER)


def test_btv_scale3_runs(s1_case):
    cap, s, sensor, _ = s1_case
    rec = reconstruct(cap.subpages, s, sensor, "btv", scale=3)
    assert rec.image.shape == (72, 96) and np.all(np.isfinite(rec.image))


def test_reconstruct_argument_errors(s1_case):
    cap, s, sensor, _ = s1_case
    with pytest.raises(ValueError):
        reconstruct(cap.subpages, s, sensor, "nope")
    with pytest.raises(ValueError):
        reconstruct(cap.subpages, s[:-1], sensor, "btv")


def test_btv_fpn_zero_prior_skips_estimation(s1_case):
    """With a zero FPN prior the MAP offsets are 0: estimation is skipped
    (warned) instead of falling back to the noisy plain mean."""
    cap, s, _, _ = s1_case
    sensor = SensorConfig.bab(psf_sigma_px=0.4, refresh_hz=8.0, fpn_sigma_k=0.0)
    with pytest.warns(RuntimeWarning):
        rec = btv_map(cap.subpages, s, sensor, SCALE, estimate_fpn=True)
    ref = btv_map(cap.subpages, s, sensor, SCALE)
    np.testing.assert_array_equal(rec.image, ref.image)
    assert np.all(np.nan_to_num(rec.info["fpn"]) == 0.0)


def test_btv_noiseless_beats_bicubic():
    """The default lam has a noise floor: noiseless, PSF-0 input still gets
    enough regularization for the part of the null space the data cannot see
    (target-Nyquist stripes) and beats bicubic."""
    scene = make_scene("textured", contrast_k=2.0, seed=7)
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, adc_lsb_k=0.0, psf_sigma_px=0.0)
    s = _shifts(0)
    cap = _capture(scene, sensor, s, seed=100)
    truth = truth_at_scale(cap, SCALE)
    p_bic = psnr(reconstruct(cap.subpages, s, sensor, "bicubic").image, truth, border=BORDER)
    p_btv = psnr(reconstruct(cap.subpages, s, sensor, "btv").image, truth, border=BORDER)
    assert p_btv > p_bic + 1.0, (p_btv, p_bic)
