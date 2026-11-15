"""Registration tests (Phase 2A): R1-R5 from the plan plus unit tests.

Error metric: ``_rms`` is the RMS of the 2-D error vector ``|est - true|``
over all subpages except the reference, against
``capture.relative_true_shifts(ref_index)`` with no global offset removed
(stricter than a per-axis RMS or an offset-free RMS).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from mlx_sr.motion import trace_from_shifts
from mlx_sr.registration import (
    CONF_REF,
    estimate_time_offset,
    fill_masked,
    fill_subpage,
    gauss_newton_translation,
    integrate_gyro,
    merged_frames,
    phase_correlation,
    refine_against_hr,
    register,
    spline_interpolant,
)
from mlx_sr.scene import make_scene
from mlx_sr.sensor import simulate_capture, truth_at_scale
from mlx_sr.types import GyroConfig, GyroTrace, MotionConfig, SensorConfig

DEG = math.pi / 180.0


def _rms(shifts: np.ndarray, capture, ref_index: int = 0) -> float:
    err = np.asarray(shifts) - capture.relative_true_shifts(ref_index)
    keep = np.ones(len(err), dtype=bool)
    keep[ref_index] = False
    return float(np.sqrt(np.mean(np.sum(err[keep] ** 2, axis=1))))


def _sensor_with_noise(sigma_k: float, **kw) -> SensorConfig:
    """BAB sensor whose per-subpage temporal noise is ``sigma_k`` (no FPN)."""
    refresh = kw.pop("refresh_hz", 8.0)
    return SensorConfig.bab(refresh_hz=refresh, netd_1hz_k=sigma_k / math.sqrt(refresh), fpn_sigma_k=0.0, **kw)


@pytest.fixture(scope="module")
def chart_scene():
    return make_scene("chart")


@pytest.fixture(scope="module")
def low_contrast_scene():
    return make_scene("low_contrast")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_fill_keeps_valid_pixels_and_fills_holes(textured):
    sen = SensorConfig.bab(n_dead=3)
    cap = simulate_capture(textured, sen, None, None, 1, seed=3)
    sp = cap.subpages[0]
    out = fill_subpage(sp)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[sp.mask], sp.values[sp.mask])
    # Dead pixels are filled, not left at -273.15.
    assert out.min() > 0.0
    # A linear ramp is reproduced at interior chess holes (symmetric neighbors).
    i, j = np.mgrid[0:24, 0:32]
    ramp = 0.3 * j - 0.2 * i + 5.0
    mask = (i + j) % 2 == 0
    filled = fill_masked(np.where(mask, ramp, np.nan), mask)
    np.testing.assert_allclose(filled[2:-2, 2:-2], ramp[2:-2, 2:-2], atol=1e-6)


def test_merged_frames_union(textured):
    cap = simulate_capture(textured, SensorConfig.bab(), None, None, 2, seed=1)
    frames = merged_frames(cap)
    assert len(frames) == 2
    for vals, mask in frames:
        assert mask.all()
        assert np.all(np.isfinite(vals))


def test_phase_correlation_sign_convention():
    """img[i, j] = ref at (j + dx, i + dy): an integer roll gives the exact shift."""
    rng = np.random.default_rng(0)
    from scipy import ndimage

    ref = ndimage.gaussian_filter(rng.standard_normal((48, 64)), 2.0)
    img = np.roll(ref, shift=(-2, -1), axis=(0, 1))  # img[i, j] = ref[i + 2, j + 1]
    s = phase_correlation(ref[8:-8, 8:-8], img[8:-8, 8:-8])
    np.testing.assert_allclose(s, [1.0, 2.0], atol=0.06)


def test_register_sign_convention_integer_shifts(textured):
    """Noise-free subpages shifted by known integer amounts are recovered by every image method."""
    sen = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, adc_lsb_k=0.0)
    shifts = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [0, -1], [0, -1], [-1, 1], [-1, 1]], float)
    cap = simulate_capture(textured, sen, trace_from_shifts(shifts, sen), None, 4, seed=0)
    for method, kw in (("phase", dict(merged=True)), ("phase", {}), ("gn", {})):
        reg = register(cap, method, **kw)
        np.testing.assert_allclose(reg.shifts, shifts, atol=0.05, err_msg=f"{method} {kw}")


def test_gauss_newton_exact_on_synthetic_translation():
    rng = np.random.default_rng(1)
    from scipy import ndimage

    ref = ndimage.gaussian_filter(rng.standard_normal((24, 32)), 1.5) * 5
    fn = spline_interpolant(ref)
    true = np.array([0.37, -0.61])
    i, j = np.mgrid[0:24, 0:32]
    img = fn((j + true[0]).ravel(), (i + true[1]).ravel()).reshape(24, 32) + 0.25
    mask = (i + j) % 2 == 1
    s, conf, info = gauss_newton_translation(fn, np.where(mask, img, np.nan), mask, np.zeros(2),
                                             return_info=True)
    np.testing.assert_allclose(s, true, atol=2e-3)
    assert info["converged"] and abs(info["offset"] - 0.25) < 1e-2
    assert conf > 1e4  # noiseless: tiny residual, high precision


def test_confidence_low_for_noise_only_scene(flat):
    """On a flat scene the images carry no shift information: the noise-corrected
    confidence must stay low (sigma well above 0.1 px) for most subpages."""
    cap = simulate_capture(flat, SensorConfig.bab(), MotionConfig.natural(), None, 8, seed=2)
    reg = register(cap, "gn")
    others = np.delete(reg.confidence, 0)
    assert np.median(others) < 25.0  # sigma > 0.2 px


def test_integrate_gyro_matches_truth(textured):
    sen = SensorConfig.bab()
    cap = simulate_capture(textured, sen, MotionConfig.wobble(), GyroConfig.ideal(), 8, seed=4)
    g = integrate_gyro(cap.gyro, sen, cap.subpage_windows, 0.0)
    np.testing.assert_allclose(g, cap.true_shifts, atol=2e-3)
    # With an offset and a bias, passing the true values reproduces the truth.
    gc = GyroConfig(noise_density_dps_rthz=0.0, bias_dps=(0.4, -0.7, 0.1), time_offset_s=0.02)
    cap = simulate_capture(textured, sen, MotionConfig.wobble(), gc, 8, seed=4)
    g = integrate_gyro(cap.gyro, sen, cap.subpage_windows, 0.02, bias=np.array([0.4, -0.7, 0.1]) * DEG)
    np.testing.assert_allclose(g, cap.true_shifts, atol=2e-3)
    g_wrong = integrate_gyro(cap.gyro, sen, cap.subpage_windows, 0.0)
    assert np.abs(g_wrong - cap.true_shifts).max() > 0.1


@pytest.mark.parametrize("domain", ["levels", "rates"])
def test_estimate_time_offset_from_true_shifts(textured, domain):
    sen = SensorConfig.bab(refresh_hz=32.0)
    cap = simulate_capture(textured, sen, MotionConfig.wobble(),
                           GyroConfig(time_offset_s=-0.012, bias_dps=(0.3, 0.5, 0.0)), 16, seed=5)
    tau = estimate_time_offset(cap.relative_true_shifts(0), cap.subpage_windows, cap.gyro,
                               sensor=sen, domain=domain)
    assert abs(tau - (-0.012)) < 5e-4


def test_registration_ignores_truth_fields(textured):
    """Inverse-crime guard: wiping every truth field leaves the result unchanged."""
    sen = SensorConfig.bab()
    cap = simulate_capture(textured, sen, MotionConfig.natural(), GyroConfig(), 4, seed=6)
    blind_subpages = [dataclasses.replace(sp, true_shift=None) for sp in cap.subpages]
    blind_gyro = GyroTrace(t=cap.gyro.t, omega=cap.gyro.omega, rate_hz=cap.gyro.rate_hz)
    blind = dataclasses.replace(cap, subpages=blind_subpages, gyro=blind_gyro, scene_hr=None,
                                fpn=None, motion=None, meta={})
    for method in ("phase", "gn", "hybrid"):
        a = register(cap, method)
        b = register(blind, method)
        np.testing.assert_array_equal(a.shifts, b.shifts)


def test_ref_index_and_dead_pixels(textured):
    sen = _sensor_with_noise(0.1, n_dead=4)
    rng = np.random.default_rng(8)
    shifts = rng.uniform(-1.0, 1.0, (8, 2))
    cap = simulate_capture(textured, sen, trace_from_shifts(shifts, sen), None, 4, seed=8)
    reg = register(cap, "gn", ref_index=3)
    np.testing.assert_array_equal(reg.shifts[3], [0.0, 0.0])
    assert reg.confidence[3] == CONF_REF
    assert np.all(np.isfinite(reg.shifts))
    assert _rms(reg.shifts, cap, ref_index=3) < 0.1
    with pytest.raises(IndexError):
        register(cap, "gn", ref_index=8)
    with pytest.raises(ValueError):
        register(cap, "nope")


def test_hybrid_falls_back_to_gyro_on_flat_scene(flat):
    """No image information: every frame falls back to the gyro."""
    cap = simulate_capture(flat, SensorConfig.bab(), MotionConfig.natural(),
                           GyroConfig(bias_dps=(0.0, 0.0, 0.0), time_offset_s=0.0), 8, seed=9)
    reg = register(cap, "hybrid")
    assert not reg.info["accepted"].any()
    assert reg.info["tau_source"] == "prior"
    assert _rms(reg.shifts, cap) < 0.05  # unbiased, synchronized gyro is accurate


def test_hybrid_with_given_offset_and_bias(textured):
    sen = SensorConfig.bab()
    gc = GyroConfig(bias_dps=(0.2, -0.3, 0.0), time_offset_s=0.01)
    cap = simulate_capture(textured, sen, MotionConfig.natural(), gc, 8, seed=11)
    bias = np.array([0.2, -0.3, 0.0]) * DEG
    reg = register(cap, "hybrid", time_offset=0.01, bias=bias)
    assert reg.info["tau_source"] == "given" and reg.time_offset_s == 0.01
    np.testing.assert_array_equal(reg.gyro_bias, bias)
    assert _rms(reg.shifts, cap) < 0.05
    hard = register(cap, "hybrid", time_offset=0.01, bias=bias, fuse=False)
    acc = hard.info["accepted"]
    assert acc.any()
    np.testing.assert_allclose(hard.shifts[acc], hard.info["gn_shifts"][acc] - hard.info["image_offset_px"])
    gyro_only = register(cap, "gyro", time_offset=0.01, bias=bias)
    assert _rms(gyro_only.shifts, cap) < 0.05


def test_refine_against_hr_improves_shifts(chart_scene):
    sen = SensorConfig.bab()
    rng = np.random.default_rng(10)
    shifts = rng.uniform(-1.5, 1.5, (16, 2))
    cap = simulate_capture(chart_scene, sen, trace_from_shifts(shifts, sen), None, 8, seed=10)
    noisy = cap.relative_true_shifts(0) + rng.normal(0.0, 0.15, (16, 2))
    hr = truth_at_scale(cap, 4, ref_index=0)  # test-only use of truth as the "HR estimate"
    reg = refine_against_hr(cap, hr, init=noisy)
    err = np.sqrt(np.mean(np.sum((reg.shifts - cap.relative_true_shifts(0)) ** 2, axis=1)))
    assert err < 0.05
    assert reg.method == "hr_refine"


# ---------------------------------------------------------------------------
# R1-R5 (plan section 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_R1_phase_correlation_merged_frames(textured, seed):
    """R1: phase correlation on merged frames, textured scene, 0.1 K noise,
    shifts uniform in [-1.5, 1.5] px (constant within each frame): RMS < 0.1 px."""
    sen = _sensor_with_noise(0.1)
    rng = np.random.default_rng(100 + seed)
    frame_shifts = rng.uniform(-1.5, 1.5, (8, 2))
    cap = simulate_capture(textured, sen, trace_from_shifts(np.repeat(frame_shifts, 2, axis=0), sen),
                           None, 8, seed)
    reg = register(cap, "phase", merged=True)
    assert _rms(reg.shifts, cap) < 0.1


@pytest.mark.parametrize("seed", [0, 1])
def test_R2_gauss_newton_single_subpages(textured, seed):
    """R2: masked Gauss-Newton on single chess subpages, same conditions
    (every subpage has its own shift): RMS < 0.1 px."""
    sen = _sensor_with_noise(0.1)
    rng = np.random.default_rng(200 + seed)
    shifts = rng.uniform(-1.5, 1.5, (16, 2))
    cap = simulate_capture(textured, sen, trace_from_shifts(shifts, sen), None, 8, seed)
    reg = register(cap, "gn")
    assert _rms(reg.shifts, cap) < 0.1


@pytest.mark.parametrize("seed", [0, 1])
def test_R3_time_offset(chart_scene, seed):
    """R3: tau estimate within 2 ms of the true 15 ms (wobble, 32 Hz refresh,
    400 Hz gyro with default noise and a bias, 2 s).  The chart scene is used
    as a calibration target: on the 2 K textured scene the 32 Hz noise
    (1.4 K per subpage) leaves 1-3 ms errors."""
    sen = SensorConfig.bab(refresh_hz=32.0)
    gc = GyroConfig(rate_hz=400.0, time_offset_s=0.015, bias_dps=(0.3, -0.5, 0.2))
    cap = simulate_capture(chart_scene, sen, MotionConfig.wobble(), gc, 32, seed)
    assert abs(cap.meta["duration_s"] - 2.0) < 1e-9
    img = register(cap, "gn")
    tau = estimate_time_offset(img.shifts, cap.subpage_windows, cap.gyro, sensor=sen,
                               weights=img.confidence)
    assert abs(tau - 0.015) < 0.002
    hyb = register(cap, "hybrid")
    assert hyb.info["tau_source"] == "estimated"
    assert abs(hyb.time_offset_s - 0.015) < 0.002


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_R4_hybrid_beats_image_only_low_contrast(low_contrast_scene, seed):
    """R4: on the 0.3 K low-contrast scene with natural tremor (default BAB at
    8 Hz, 0.7 K noise per subpage; default gyro with tau = 10 ms and a random
    0.05-1 deg/s bias), hybrid RMS <= 0.5 x the better image-only method.

    K = 16 subpages (2 s), the plan's test size.  At this SNR no frame passes
    the confidence gates, so the hybrid is the uncorrected gyro and its error
    is the bias drift: measured 0.24-0.96 px against 2.4-3.5 px image-only
    (ratios 0.10-0.30 over six seeds).  With K = 32 (4 s) a 1 deg/s bias
    drifts about twice as far and the ratio reaches 0.7 for the worst seeds.
    """
    cap = simulate_capture(low_contrast_scene, SensorConfig.bab(), MotionConfig.natural(),
                           GyroConfig(), 8, seed)
    image_only = min(_rms(register(cap, m).shifts, cap) for m in ("phase", "gn"))
    hybrid = _rms(register(cap, "hybrid").shifts, cap)
    assert hybrid <= 0.5 * image_only


@pytest.mark.parametrize("seed", [0, 1])
def test_R5_hybrid_recovers_gyro_bias(textured, seed):
    """R5: hybrid recovers the injected gyro bias (pitch and yaw; roll is not
    observable from translations) to within 20 % per axis.  Textured 2 K
    scene, natural motion, default BAB at 8 Hz, 4 s, tau estimated or prior."""
    bias_dps = np.array([0.6, -0.8, 0.3])
    cap = simulate_capture(textured, SensorConfig.bab(), MotionConfig.natural(),
                           GyroConfig(bias_dps=tuple(bias_dps)), 16, seed)
    reg = register(cap, "hybrid")
    est = reg.gyro_bias[:2] / DEG
    np.testing.assert_array_less(np.abs(est - bias_dps[:2]), 0.2 * np.abs(bias_dps[:2]))
