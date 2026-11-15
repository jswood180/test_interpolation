"""Metric tests M1-M4 (plan section 5) plus real-scene and robustness checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage
from scipy.special import ndtr

from mlx_sr import metrics
from mlx_sr.metrics import (
    bar_contrast,
    flat_sigma,
    hotspot_error,
    psnr,
    registration_rms,
    slanted_edge_mtf,
    ssim,
)
from mlx_sr.scene import make_scene
from mlx_sr.sensor import box_sample, merge_subpages, simulate_capture, truth_at_scale
from mlx_sr.types import Roi, SensorConfig


def _synthetic_edge(rows: int, cols: int, scale: int, sigma: float, angle_deg: float, *,
                    supersample: int = 1, hot_right: bool = True, low: float = 25.0,
                    high: float = 35.0, center: tuple[float, float] | None = None) -> np.ndarray:
    """Analytic Gaussian-blurred straight edge on a scale-``scale`` grid.

    The edge line is ``x = cx + (y - cy) tan(angle)`` in LR px; the profile
    across it is ``Phi(d / sigma)`` (erf), ``d`` the signed normal distance.
    ``supersample=1`` point-samples at pixel centers (no aperture);
    ``supersample=n`` averages an n x n grid per pixel (a box pixel aperture).
    """
    cx, cy = (cols / scale / 2.0 + 0.13, rows / scale / 2.0) if center is None else center
    th = math.radians(angle_deg)
    off = (np.arange(supersample) + 0.5) / supersample
    ys = (np.arange(rows)[:, None] + off[None, :]).ravel() / scale
    xs = (np.arange(cols)[:, None] + off[None, :]).ravel() / scale
    X, Y = np.meshgrid(xs, ys)
    d = (X - (cx + (Y - cy) * math.tan(th))) * math.cos(th)
    if not hot_right:
        d = -d
    v = ndtr(d / sigma)
    v = v.reshape(rows, supersample, cols, supersample).mean(axis=(1, 3))
    return low + (high - low) * v


def _expected_mtf(f: np.ndarray, sigma: float, angle_deg: float, pitch: float | None) -> np.ndarray:
    """Gaussian MTF, times the box-pixel aperture projected on the edge normal
    (a trapezoid: two boxes of widths ``p cos`` and ``p sin``) when ``pitch``."""
    m = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * f ** 2)
    if pitch is not None:
        th = math.radians(angle_deg)
        m = m * np.sinc(pitch * f * math.cos(th)) * np.sinc(pitch * f * math.sin(th))
    return m


@pytest.fixture(scope="module")
def chart_capture():
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, n_dead=0, psf_sigma_px=0.0, adc_lsb_k=0.0)
    return simulate_capture(make_scene("chart"), sensor, None, None, n_frames=1, seed=0)


# M1 ------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("aperture", ["point", "pixel"])
@pytest.mark.parametrize("sigma", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("angle", [5.0, 7.0])
def test_m1_slanted_edge_matches_gaussian_mtf(scale, aperture, sigma, angle):
    rows = cols = 24 * scale
    ss = 1 if aperture == "point" else 16
    img = _synthetic_edge(rows, cols, scale, sigma, angle, supersample=ss)
    r = slanted_edge_mtf(img, Roi(0.0, 0.0, cols / scale, rows / scale), scale)
    assert r["valid"]
    assert r["angle_deg"] == pytest.approx(angle, abs=0.1)
    f = r["freq"]
    sel = f <= 0.5 + 1e-12
    assert f[sel].max() == pytest.approx(0.5, abs=0.01)
    expected = _expected_mtf(f[sel], sigma, angle, None if aperture == "point" else 1.0 / scale)
    err = np.abs(r["mtf"][sel] - expected).max()
    assert err < 0.03, f"max |MTF - expected| = {err:.4f}"
    # mtf_at agrees with the curve; mtf50 agrees with the analytic crossing where it exists.
    assert r["mtf_at"](0.25) == pytest.approx(float(np.interp(0.25, f, r["mtf"])))
    if aperture == "point":
        f50 = math.sqrt(math.log(2.0) / (2.0 * math.pi ** 2 * sigma ** 2))
        if f50 < 0.5:
            assert r["mtf50"] == pytest.approx(f50, abs=0.02)


@pytest.mark.parametrize("variant", ["horizontal", "hot_left"])
def test_m1_orientation_and_polarity(variant):
    sigma, angle, scale = 0.5, 6.0, 2
    img = _synthetic_edge(48, 48, scale, sigma, angle, hot_right=(variant != "hot_left"))
    if variant == "horizontal":
        img = img.T
    r = slanted_edge_mtf(img, Roi(0.0, 0.0, 24.0, 24.0), scale)
    assert r["valid"] and r["transposed"] == (variant == "horizontal")
    f = r["freq"]
    sel = f <= 0.5
    assert np.abs(r["mtf"][sel] - _expected_mtf(f[sel], sigma, angle, None)).max() < 0.03


@pytest.mark.parametrize("scale", [1, 2])
@pytest.mark.parametrize("aperture", ["point", "pixel"])
@pytest.mark.parametrize("sigma", [0.5, 0.8])
def test_m1_on_chart_edge_roi_geometry(scale, aperture, sigma):
    """M1 on the chart's own (short) edge ROI, where windowing matters most."""
    edge = make_scene("chart").meta.edge
    img = _synthetic_edge(24 * scale, 32 * scale, scale, sigma, edge.angle_deg,
                          supersample=1 if aperture == "point" else 16, center=edge.center)
    r = slanted_edge_mtf(img, edge.roi, scale)
    assert r["valid"] and not r["truncated"]
    f = r["freq"]
    sel = f <= 0.5 + 1e-12
    expected = _expected_mtf(f[sel], sigma, edge.angle_deg, None if aperture == "point" else 1.0 / scale)
    err = np.abs(r["mtf"][sel] - expected).max()
    assert err < 0.03, f"max |MTF - expected| = {err:.4f}"


def test_m1_short_side_is_flagged():
    """An edge 2 LR px from the ROI boundary with a wide LSF is flagged ``truncated``."""
    edge = make_scene("chart").meta.edge
    img = _synthetic_edge(48, 64, 2, 1.2, edge.angle_deg, center=(4.0, edge.center[1]))
    r = slanted_edge_mtf(img, edge.roi, 2)
    assert r["valid"] and r["truncated"]


def test_m1_infinite_pixel_is_ignored():
    sigma, angle, scale = 0.5, 7.0, 2
    img = _synthetic_edge(48, 48, scale, sigma, angle)
    ref = slanted_edge_mtf(img, Roi(0.0, 0.0, 24.0, 24.0), scale)
    for bad in (np.inf, -np.inf):
        im = img.copy()
        im[10, 30] = bad
        r = slanted_edge_mtf(im, Roi(0.0, 0.0, 24.0, 24.0), scale)
        assert r["valid"]
        assert r["mtf_at"](0.5) == pytest.approx(ref["mtf_at"](0.5), abs=0.01)


def test_m1_noisy_edge_stays_close(rng):
    """Moderate noise (SNR 100 on the step) keeps the MTF within 0.05."""
    sigma, angle, scale = 0.5, 7.0, 2
    img = _synthetic_edge(48, 48, scale, sigma, angle) + rng.normal(0.0, 0.1, (48, 48))
    r = slanted_edge_mtf(img, Roi(0.0, 0.0, 24.0, 24.0), scale)
    f = r["freq"]
    sel = f <= 0.5
    assert np.abs(r["mtf"][sel] - _expected_mtf(f[sel], sigma, angle, None)).max() < 0.05


# M2 ------------------------------------------------------------------------


def test_m2_identical_images(chart_capture):
    t = truth_at_scale(chart_capture, 2)
    assert psnr(t, t) == float("inf")
    assert psnr(t, t, data_range=10.0, border=2) == float("inf")
    assert ssim(t, t) == pytest.approx(1.0)
    assert ssim(t, t, data_range=10.0, border=2) == pytest.approx(1.0)


def test_m2_ssim_ordering_and_data_range(chart_capture, rng):
    t = truth_at_scale(chart_capture, 2)
    noisy_a = t + rng.normal(0.0, 0.2, t.shape)
    noisy_b = t + rng.normal(0.0, 1.0, t.shape)
    sa, sb = ssim(noisy_a, t, data_range=35.0), ssim(noisy_b, t, data_range=35.0)
    assert 1.0 > sa > sb > 0.0
    assert psnr(noisy_a, t, data_range=35.0) > psnr(noisy_b, t, data_range=35.0)
    # Larger data_range makes the constants larger, so SSIM rises.
    assert ssim(noisy_b, t, data_range=100.0) > sb


@pytest.mark.filterwarnings("ignore:.*structural_similarity")
def test_m2_ssim_matches_skimage_on_finite_input(chart_capture, rng):
    from skimage.metrics import structural_similarity

    t = truth_at_scale(chart_capture, 2)
    n = t + rng.normal(0.0, 0.5, t.shape)
    assert ssim(n, t, data_range=35.0) == pytest.approx(structural_similarity(n, t, data_range=35.0), abs=1e-12)
    assert ssim(n, t, data_range=35.0, border=3) == pytest.approx(
        structural_similarity(n[3:-3, 3:-3], t[3:-3, 3:-3], data_range=35.0), abs=1e-12)


def test_m2_nan_and_tiny_inputs(rng):
    a = rng.normal(25.0, 1.0, (24, 32))
    b = a + rng.normal(0.0, 0.1, a.shape)
    ref_psnr, ref_ssim = psnr(b, a, data_range=10.0), ssim(b, a, data_range=10.0)
    bn = b.copy()
    bn[5, 7] = np.nan
    assert np.isfinite(psnr(bn, a, data_range=10.0))
    assert psnr(bn, a, data_range=10.0) == pytest.approx(ref_psnr, abs=0.1)
    s = ssim(bn, a, data_range=10.0)
    assert np.isfinite(s) and s == pytest.approx(ref_ssim, abs=0.02)
    # Tiny images shrink the window instead of raising; degenerate ones give NaN.
    assert np.isfinite(ssim(b[:5, :5], a[:5, :5], data_range=10.0))
    assert math.isnan(ssim(b[:2, :2], a[:2, :2], data_range=10.0))
    assert math.isnan(psnr(np.full((4, 4), np.nan), a[:4, :4]))
    with pytest.raises(ValueError):
        ssim(a, a[:-1])
    # "propagate" makes holes count instead of silently dropping them.
    assert math.isnan(psnr(bn, a, data_range=10.0, nan_policy="propagate"))
    assert math.isnan(ssim(bn, a, data_range=10.0, nan_policy="propagate"))
    assert psnr(b, a, data_range=10.0, nan_policy="propagate") == pytest.approx(ref_psnr)
    with pytest.raises(ValueError):
        psnr(b, a, nan_policy="raise")
    # Even window sizes round down to odd instead of raising.
    assert ssim(b, a, data_range=10.0, win_size=6) == pytest.approx(ssim(b, a, data_range=10.0, win_size=5))


# M3 ------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("blur_lr", [0.3, 0.5, 0.6])
def test_m3_bar_contrast_falls_with_frequency(scale, blur_lr):
    scene = make_scene("bars")
    hr = ndimage.gaussian_filter(scene.hr, blur_lr * scene.hr_factor, mode="nearest")
    img = box_sample(hr, scene.hr_factor, scene.margin_lr, (24, 32), scale=scale)
    res = bar_contrast(img, scene.meta.bars, scale)
    order = np.argsort(res["freq"])
    c = res["contrast"][order]
    m = res["modulation"][order]
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(m))
    assert np.all(np.diff(c) < 0), f"contrast not monotone: {np.round(c, 3)}"
    assert np.all(np.diff(m) < 0), f"modulation not monotone: {np.round(m, 3)}"
    assert c[0] > (0.8 if blur_lr <= 0.5 else 0.7) and c[-1] < c[0]
    # A DC offset (FPN, bias) does not change the contrast; Michelson is reported too.
    shifted = bar_contrast(img - 2.0, scene.meta.bars, scale)
    np.testing.assert_allclose(shifted["contrast"], res["contrast"], atol=1e-12)
    assert np.all(np.isfinite(res["michelson"]))


def test_m3_heavy_blur_limits_resolution():
    scene = make_scene("bars")
    hr = ndimage.gaussian_filter(scene.hr, 0.9 * scene.hr_factor, mode="nearest")
    img = box_sample(hr, scene.hr_factor, scene.margin_lr, (24, 32), scale=2)
    res = bar_contrast(img, scene.meta.bars, 2)
    # Heavy blur washes out the finest groups, so the limit is a coarser group.
    assert res["limit_index"] is not None
    assert res["limit_freq"] < max(res["freq"])
    assert not res["resolvable"][int(np.argmax(res["freq"]))]
    for g in res["groups"]:
        if g["freq"] <= res["limit_freq"]:
            assert g["resolvable"]


def test_m3_bars_above_grid_nyquist_are_not_resolvable(chart_capture):
    lr = truth_at_scale(chart_capture, 1)
    bars = chart_capture.scene_meta.bars
    res = bar_contrast(lr, bars, 1)
    for g in res["groups"]:
        if g["freq"] > 0.5:
            assert g["above_nyquist"] and not g["resolvable"]
    assert res["limit_freq"] <= 0.5


def test_m3_bicubic_baseline_does_not_resolve_above_native_nyquist():
    """Aliased bars (0.6-0.9 cyc/LR px seen by the 1 px LR grid) upscaled by
    bicubic must not count as resolved, even though the 2x grid could hold them."""
    scene = make_scene("chart")
    for seed in range(8):
        cap = simulate_capture(scene, SensorConfig.bab(psf_sigma_px=0.0), None, None, n_frames=1, seed=seed)
        lr, _ = merge_subpages(cap.subpages[0], cap.subpages[1])
        up = ndimage.zoom(lr, 2, order=3, grid_mode=True, mode="grid-mirror")
        res = bar_contrast(up, scene.meta.bars, 2)
        assert res["limit_freq"] <= 0.5, (seed, res["limit_freq"])


# M4 ------------------------------------------------------------------------


def test_m4_registration_rms_offset_invariant(rng):
    true = rng.uniform(-1.5, 1.5, (16, 2))
    err = rng.normal(0.0, 0.05, (16, 2))
    est = true + err
    base = registration_rms(est, true)
    for off in [(0.0, 0.0), (3.7, -2.2), (-0.4, 0.9)]:
        assert registration_rms(est + np.array(off), true) == pytest.approx(base, abs=1e-12)
        assert registration_rms(est, true + np.array(off)) == pytest.approx(base, abs=1e-12)
    centered = err - err.mean(axis=0)
    assert base == pytest.approx(math.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    # Reference-offset removal: shifting by a constant does not change it either.
    ref = registration_rms(est, true, remove="ref", ref_index=3)
    assert registration_rms(est + 5.0, true, remove="ref", ref_index=3) == pytest.approx(ref, abs=1e-12)
    d = err - err[3]
    assert ref == pytest.approx(math.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    # Without removal the offset counts.
    assert registration_rms(est + 1.0, true, remove="none") > 1.0


def test_m4_registration_rms_nan_rows_and_1d():
    true = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, -0.3], [0.2, 0.9]])
    est = true + 2.0
    est[2] = np.nan
    assert registration_rms(est, true) == pytest.approx(0.0, abs=1e-12)
    # 1-D: differences (1, 1, 1.5) -> centered (-1/6, -1/6, 1/3).
    assert registration_rms(np.array([1.0, 2.0, 3.5]), np.array([0.0, 1.0, 2.0])) == pytest.approx(
        math.sqrt(np.mean(np.array([-1.0, -1.0, 2.0]) ** 2) / 36.0))
    assert math.isnan(registration_rms(np.full((3, 2), np.nan), np.zeros((3, 2))))
    with pytest.raises(ValueError):
        registration_rms(np.zeros((3, 2)), np.zeros((4, 2)))


# Real scene ------------------------------------------------------------------


def test_all_metrics_on_chart_truth_at_scale_2(chart_capture):
    meta = chart_capture.scene_meta
    t = truth_at_scale(chart_capture, 2)

    # Sharp edge box-averaged on 0.5 LR px cells (and anti-aliased on 1/8 LR
    # px HR cells): MTF = aperture sincs projected on the edge normal.
    e = meta.edge
    r = slanted_edge_mtf(t, e.roi, 2)
    assert r["valid"] and r["angle_deg"] == pytest.approx(e.angle_deg, abs=0.3)
    th = math.radians(e.angle_deg)
    f = r["freq"][r["freq"] <= 0.5]
    exp_ = np.ones_like(f)
    for p in (0.5, 1.0 / 8.0):
        exp_ *= np.sinc(p * f * math.cos(th)) * np.sinc(p * f * math.sin(th))
    assert np.abs(r["mtf"][: f.size] - exp_).max() < 0.03
    assert r["mtf50"] > 0.5  # sharper than native Nyquist on a 2x truth

    bars = bar_contrast(t, meta.bars, 2)
    assert np.all(bars["resolvable"]) and bars["limit_freq"] == pytest.approx(0.9)
    assert np.all(bars["contrast"] > 0.5)

    errs = [hotspot_error(t, h.peak_c, h.roi, 2) for h in meta.hotspots]
    radii = [h.radius_lr for h in meta.hotspots]
    assert radii == sorted(radii)
    # The 0.3 LR px disc never covers a whole 0.5 px cell, so it reads cold;
    # the 0.6 and 1.2 LR px discs each fully contain at least one cell.
    assert errs[0] < -0.2 * (meta.hotspots[0].peak_c - meta.hotspots[0].background_c)
    assert errs[0] < errs[1] <= errs[2]
    assert errs[1] == pytest.approx(0.0, abs=1e-9) and errs[2] == pytest.approx(0.0, abs=1e-9)

    assert flat_sigma(t, meta.flat[0].roi, 2) == pytest.approx(0.0, abs=1e-9)


def test_shifted_grid_roi_convention(chart_capture):
    """A reconstruction on a shifted grid measured with the same ``shift``
    gives the same result as the unshifted one."""
    meta = chart_capture.scene_meta
    sh = (0.25, -0.5)
    t0 = truth_at_scale(chart_capture, 2)
    ts = truth_at_scale(chart_capture, 2, shift=sh)
    r0 = slanted_edge_mtf(t0, meta.edge.roi, 2)
    rs = slanted_edge_mtf(ts, meta.edge.roi, 2, shift=sh)
    assert rs["mtf_at"](0.5) == pytest.approx(r0["mtf_at"](0.5), abs=0.02)
    h = meta.hotspots[-1]
    assert hotspot_error(ts, h.peak_c, h.roi, 2, shift=sh) == pytest.approx(0.0, abs=1e-9)


# Robustness ------------------------------------------------------------------


def test_metrics_on_tiny_bicubic_baseline(chart_capture):
    """Every metric runs on a 32 x 24 noisy LR frame and on its bicubic 2x upscale."""
    sensor = SensorConfig.bab(n_dead=2)
    cap = simulate_capture(make_scene("chart"), sensor, None, None, n_frames=1, seed=3)
    lr, mask = merge_subpages(cap.subpages[0], cap.subpages[1])
    lr = np.where(mask, lr, np.nan)  # dead pixels -> NaN
    filled = np.where(np.isnan(lr), np.nanmean(lr), lr)
    up = ndimage.zoom(filled, 2, order=3, grid_mode=True, mode="grid-mirror")
    meta = cap.scene_meta
    for img, s in [(lr, 1), (up, 2)]:
        r = slanted_edge_mtf(img, meta.edge.roi, s)
        assert r["valid"] and np.isfinite(r["mtf_at"](0.25))
        assert 0.0 < r["mtf_at"](0.25) < 1.2
        b = bar_contrast(img, meta.bars, s)
        assert np.all(np.isfinite(b["contrast"]))
        for h in meta.hotspots:
            assert np.isfinite(hotspot_error(img, h.peak_c, h.roi, s))
        assert np.isfinite(flat_sigma(img, meta.flat[0].roi, s))
        t = truth_at_scale(cap, s)
        assert np.isfinite(psnr(img, t, data_range=35.0))
        assert np.isfinite(ssim(img, t, data_range=35.0))
    # The bicubic upscale cannot add detail: its MTF50 stays below native Nyquist.
    assert slanted_edge_mtf(up, meta.edge.roi, 2)["mtf50"] < 0.5


def test_edge_of_image_and_degenerate_rois():
    sigma, angle, scale = 0.5, 7.0, 2
    img = _synthetic_edge(48, 48, scale, sigma, angle)
    # ROI running off the image is clipped to the existing pixels.
    r = slanted_edge_mtf(img, Roi(-5.0, -5.0, 30.0, 30.0), scale)
    assert r["valid"]
    sel = r["freq"] <= 0.5
    assert np.abs(r["mtf"][sel] - _expected_mtf(r["freq"][sel], sigma, angle, None)).max() < 0.03
    # Entirely outside, too small, or flat: NaN results, no exception.
    for roi, im in [(Roi(40.0, 40.0, 50.0, 50.0), img), (Roi(0.0, 0.0, 0.5, 0.5), img),
                    (Roi(0.0, 0.0, 24.0, 24.0), np.full_like(img, 25.0)),
                    (Roi(0.0, 0.0, 24.0, 24.0), np.full_like(img, np.nan))]:
        bad = slanted_edge_mtf(im, roi, scale)
        assert not bad["valid"] and math.isnan(bad["mtf50"]) and math.isnan(bad["mtf_at"](0.3))
    out = Roi(40.0, 40.0, 50.0, 50.0)
    assert math.isnan(hotspot_error(img, 30.0, out, scale))
    assert math.isnan(flat_sigma(img, out, scale))


def test_nan_pixels_inside_edge_roi(rng):
    sigma, angle, scale = 0.5, 7.0, 2
    img = _synthetic_edge(48, 48, scale, sigma, angle)
    holes = rng.random(img.shape) < 0.03
    img[holes] = np.nan
    r = slanted_edge_mtf(img, Roi(0.0, 0.0, 24.0, 24.0), scale)
    assert r["valid"]
    sel = r["freq"] <= 0.5
    assert np.abs(r["mtf"][sel] - _expected_mtf(r["freq"][sel], sigma, angle, None)).max() < 0.03
    flat = np.full((48, 48), 25.0) + rng.normal(0.0, 0.2, (48, 48))
    flat[holes] = np.nan
    assert flat_sigma(flat, Roi(0.0, 0.0, 24.0, 24.0), scale) == pytest.approx(0.2, rel=0.1)


def test_public_api():
    for name in metrics.__all__:
        assert hasattr(metrics, name)
