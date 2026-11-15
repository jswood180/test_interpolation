"""Forward-model tests F1-F4 and F7 (plan section 5)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mlx_sr.motion import trace_from_shifts
from mlx_sr.sensor import merge_subpages, simulate_capture, truth_at_scale
from mlx_sr.types import DEAD_VALUE, MotionConfig, MotionTrace, Scene, SensorConfig


def _block_mean(scene: Scene) -> np.ndarray:
    m, h = scene.margin_lr, scene.hr_factor
    crop = scene.hr[m * h:-m * h, m * h:-m * h]
    return crop.reshape(24, h, 32, h).mean(axis=(1, 3))


def _merged(capture, f: int = 0) -> np.ndarray:
    values, _ = merge_subpages(capture.subpages[2 * f], capture.subpages[2 * f + 1])
    return values


# F1 ------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["chart", "textured"])
def test_f1_static_ideal_equals_block_mean(kind, chart, textured, clean_bab):
    scene = chart if kind == "chart" else textured
    cap = simulate_capture(scene, clean_bab, MotionConfig.still(), None, n_frames=2, seed=0)
    expected = _block_mean(scene)
    for f in range(2):
        np.testing.assert_allclose(_merged(cap, f), expected, atol=1e-9, rtol=0)
    assert np.allclose(cap.true_shifts, 0.0)


def test_f1_raw_array_input_matches_scene(chart, clean_bab):
    a = simulate_capture(chart, clean_bab, None, None, 1, seed=0)
    b = simulate_capture(chart.hr, clean_bab, None, None, 1, seed=0)
    np.testing.assert_array_equal(_merged(a), _merged(b))
    assert a.meta["scene"] is chart.meta and b.meta["scene"] is None


def test_truth_at_scale_nests(chart, clean_bab):
    cap = simulate_capture(chart, clean_bab, None, None, 1, seed=0)
    t1 = truth_at_scale(cap, 1)
    np.testing.assert_allclose(t1, _block_mean(chart), atol=1e-9)
    for s in (2, 3):
        ts = truth_at_scale(cap, s)
        assert ts.shape == (24 * s, 32 * s)
        np.testing.assert_allclose(ts.reshape(24, s, 32, s).mean(axis=(1, 3)), t1, atol=1e-9)
    # A shifted truth grid by one LR px equals a roll.
    t1s = truth_at_scale(cap, 1, shift=(1.0, 0.0))
    np.testing.assert_allclose(t1s[:, :-1], t1[:, 1:], atol=1e-9)


def test_truth_ref_index_matches_static_capture(textured, clean_bab):
    """With constant motion, the ref-subpage truth equals what the sensor saw."""
    s = np.array([[0.3, -0.6]] * 2)
    cap = simulate_capture(textured, clean_bab, trace_from_shifts(s, clean_bab), None, 1, seed=0)
    np.testing.assert_allclose(cap.ref_shift(0), s[0])
    t = truth_at_scale(cap, 1, ref_index=0)
    np.testing.assert_allclose(t, truth_at_scale(cap, 1, shift=s[0]), atol=0)
    np.testing.assert_allclose(_merged(cap), t, atol=1e-9)
    with pytest.raises(ValueError):
        truth_at_scale(cap, 1, shift=(0, 0), ref_index=0)


def test_scene_shape_mismatch_raises(textured, clean_bab):
    bad = Scene(hr=textured.hr[:-8], meta=textured.meta)
    with pytest.raises(ValueError):
        simulate_capture(bad, clean_bab, None, None, 1, seed=0)


# F2 ------------------------------------------------------------------------


@pytest.mark.parametrize("psf", [0.0, 0.4])
def test_f2_constant_shift_is_roll(textured, psf):
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, psf_sigma_px=psf, adc_lsb_k=0.0)
    ref = _merged(simulate_capture(textured, sensor, None, None, 1, seed=0))
    # +1 LR px in x: pixel j now samples what pixel j+1 saw.
    cap = simulate_capture(textured, sensor, trace_from_shifts([[1.0, 0.0]] * 2, sensor), None, 1, seed=0)
    np.testing.assert_allclose(cap.true_shifts, [[1.0, 0.0]] * 2, atol=1e-12)
    out = _merged(cap)
    np.testing.assert_allclose(out[:, :-1], np.roll(ref, -1, axis=1)[:, :-1], atol=1e-9)
    # +1 LR px in y.
    cap = simulate_capture(textured, sensor, trace_from_shifts([[0.0, 1.0]] * 2, sensor), None, 1, seed=0)
    np.testing.assert_allclose(_merged(cap)[:-1], np.roll(ref, -1, axis=0)[:-1], atol=1e-9)


def test_f2_scene_moved_right_is_positive_roll(textured, clean_bab):
    ref = _merged(simulate_capture(textured, clean_bab, None, None, 1, seed=0))
    moved = Scene(hr=np.roll(textured.hr, textured.hr_factor, axis=1), meta=textured.meta)
    out = _merged(simulate_capture(moved, clean_bab, None, None, 1, seed=0))
    np.testing.assert_allclose(out[:, 1:], np.roll(ref, 1, axis=1)[:, 1:], atol=1e-9)


def test_f2_positive_yaw_gives_positive_dx(clean_bab, textured):
    fx, fy = clean_bab.f_px
    yaw, pitch = np.radians(0.5), np.radians(-0.3)
    t = np.array([0.0, 10.0])
    trace = MotionTrace(t=t, theta=np.array([[yaw, pitch], [yaw, pitch]]))
    cap = simulate_capture(textured, clean_bab, trace, None, 1, seed=0)
    np.testing.assert_allclose(cap.true_shifts[0], [fx * np.tan(yaw), fy * np.tan(pitch)], rtol=1e-12)
    assert cap.true_shifts[0, 0] > 0 and cap.true_shifts[0, 1] < 0


# F3 ------------------------------------------------------------------------


@pytest.mark.parametrize("readout", ["chess", "interleaved"])
@pytest.mark.parametrize("rate", [2.0, 8.0, 32.0])
def test_f3_masks_and_timestamps(flat, rate, readout):
    sensor = SensorConfig.bab(refresh_hz=rate, readout=readout, n_dead=0)
    cap = simulate_capture(flat, sensor, None, None, n_frames=3, seed=1)
    m0, m1 = sensor.subpage_mask(0), sensor.subpage_mask(1)
    assert not np.any(m0 & m1) and np.all(m0 | m1)
    if readout == "chess":
        assert m0[0, 0] and m0[1, 1] and not m0[0, 1]
    assert [sp.subpage_id for sp in cap.subpages] == [0, 1, 0, 1, 0, 1]
    assert [sp.frame_index for sp in cap.subpages] == [0, 0, 1, 1, 2, 2]
    t0 = np.array([sp.t_start for sp in cap.subpages])
    np.testing.assert_allclose(np.diff(t0), 1.0 / rate, rtol=1e-12)
    np.testing.assert_allclose([sp.t_end - sp.t_start for sp in cap.subpages], 1.0 / rate, rtol=1e-12)
    assert t0[0] == 0.0
    for sp in cap.subpages:
        read = sensor.subpage_mask(sp.subpage_id)
        np.testing.assert_array_equal(sp.mask, read)
        assert np.all(np.isnan(sp.values[~read])) and not np.any(np.isnan(sp.values[read]))


# F4 ------------------------------------------------------------------------


def test_f4_px_per_degree():
    bh, bv = SensorConfig.bab().px_per_deg
    ah, av = SensorConfig.baa().px_per_deg
    assert abs(bh - 0.54) <= 0.02 and abs(bv - 0.66) <= 0.02
    assert abs(ah - 0.20) <= 0.02 and abs(av - 0.27) <= 0.02
    assert SensorConfig.bab().netd_1hz_k == 0.25 and SensorConfig.baa().netd_1hz_k == 0.14
    assert SensorConfig.bab(netd_1hz_k=0.3).netd_1hz_k == 0.3
    fx, fy = SensorConfig.bab().f_px
    assert fx == pytest.approx(16 / np.tan(np.radians(27.5)))
    assert fy == pytest.approx(12 / np.tan(np.radians(17.5)))


# F7 ------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [2.0, 8.0, 32.0])
def test_f7_temporal_noise_scales_with_sqrt_rate(flat, rate):
    sensor = SensorConfig.bab(refresh_hz=rate, netd_1hz_k=0.1, fpn_sigma_k=0.0, n_dead=0, adc_lsb_k=0.01)
    cap = simulate_capture(flat, sensor, MotionConfig.still(), None, n_frames=8, seed=3)
    res = np.concatenate([sp.values[sp.mask] - 25.0 for sp in cap.subpages])
    expected = 0.1 * np.sqrt(rate)
    assert cap.meta["noise_sigma_k"] == pytest.approx(expected)
    assert abs(res.std() / expected - 1.0) < 0.10
    assert abs(res.mean()) < 4 * expected / np.sqrt(res.size)


def test_f7_dead_pixels(flat):
    sensor = SensorConfig.bab(n_dead=4)
    cap = simulate_capture(flat, sensor, None, None, n_frames=2, seed=5)
    assert cap.dead_mask.sum() == 4
    for sp in cap.subpages:
        read = sensor.subpage_mask(sp.subpage_id)
        dead_here = read & cap.dead_mask
        assert np.all(sp.values[dead_here] == DEAD_VALUE)
        assert not np.any(sp.mask[cap.dead_mask])
        np.testing.assert_array_equal(sp.mask, read & ~cap.dead_mask)
    merged, mask = merge_subpages(cap.subpages[0], cap.subpages[1])
    assert np.all(merged[cap.dead_mask] == DEAD_VALUE) and not np.any(mask[cap.dead_mask])


def test_f7_fpn_static_across_frames(flat):
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.2, n_dead=0, adc_lsb_k=0.0)
    cap = simulate_capture(flat, sensor, MotionConfig.natural(0.3), None, n_frames=4, seed=9)
    for sp in cap.subpages:
        np.testing.assert_allclose(sp.values[sp.mask] - 25.0, cap.fpn[sp.mask], atol=1e-9)
    assert abs(cap.fpn.std() / 0.2 - 1.0) < 0.15


def test_quantization_step(flat):
    sensor = SensorConfig.bab(adc_lsb_k=0.05, fpn_sigma_k=0.1)
    cap = simulate_capture(flat, sensor, None, None, 1, seed=2)
    v = cap.subpages[0].values[cap.subpages[0].mask] / 0.05
    np.testing.assert_allclose(v, np.round(v), atol=1e-6)


# Misc ----------------------------------------------------------------------


def test_deterministic_and_streams_independent(chart):
    sensor = SensorConfig.bab()
    a = simulate_capture(chart, sensor, MotionConfig.natural(), None, 2, seed=11)
    b = simulate_capture(chart, sensor, MotionConfig.natural(), None, 2, seed=11)
    c = simulate_capture(chart, sensor, MotionConfig.natural(), None, 2, seed=12)
    for x, y in zip(a.subpages, b.subpages):
        np.testing.assert_array_equal(x.values, y.values)
    assert not np.allclose(np.nan_to_num(a.subpages[0].values), np.nan_to_num(c.subpages[0].values))
    from mlx_sr.types import GyroConfig
    d = simulate_capture(chart, sensor, MotionConfig.natural(), GyroConfig(), 2, seed=11)
    np.testing.assert_array_equal(a.fpn, d.fpn)
    np.testing.assert_array_equal(a.true_shifts, d.true_shifts)


def test_motion_blur_averages_footprints(textured, clean_bab):
    """Two exposure halves at shifts 0 and 1 average to the mean of both static frames."""
    T = clean_bab.subpage_period_s
    eps = 1e-12
    fx = clean_bab.f_px[0]
    yaw1 = np.arctan(1.0 / fx)
    trace = MotionTrace(t=np.array([0.0, T / 2 - eps, T / 2, 2 * T]),
                        theta=np.array([[0, 0], [0, 0], [yaw1, 0], [yaw1, 0]], dtype=float))
    cap = simulate_capture(textured, clean_bab, trace, None, 1, seed=0)
    s0 = simulate_capture(textured, clean_bab, None, None, 1, seed=0).subpages[0]
    s1 = simulate_capture(textured, clean_bab, trace_from_shifts([[1.0, 0.0]] * 2, clean_bab), None, 1,
                          seed=0).subpages[0]
    m = s0.mask
    np.testing.assert_allclose(cap.subpages[0].values[m], 0.5 * (s0.values[m] + s1.values[m]), atol=1e-9)
    assert cap.subpages[0].true_shift[0] == pytest.approx(0.5)


def test_roll_changes_corners_not_center(textured, clean_bab):
    cfg = dataclasses.replace(MotionConfig.still(), roll_rms_deg=1.0)
    cap = simulate_capture(textured, clean_bab, cfg, None, 1, seed=4)
    assert cap.motion.roll is not None
    ref = simulate_capture(textured, clean_bab, None, None, 1, seed=4)
    d = np.abs(np.nan_to_num(cap.subpages[0].values - ref.subpages[0].values))
    assert d[11:13, 15:17].max() < d[[0, 0, -1, -1], [0, -1, 0, -1]].max()


def test_out_of_bounds_warns(textured, clean_bab):
    with pytest.warns(RuntimeWarning, match="outside the HR canvas"):
        cap = simulate_capture(textured, clean_bab, trace_from_shifts([[5.0, 0.0]] * 2, clean_bab), None, 1, 0)
    assert cap.meta["oob_fraction"] > 0


# Staggered exposure model -------------------------------------------------


def test_staggered_exposure_equals_full_without_motion(chart):
    """With no motion the exposure-timing model cannot matter."""
    full = simulate_capture(chart, SensorConfig.bab(n_dead=2), None, None, 2, seed=3)
    stag = simulate_capture(chart, SensorConfig.bab(n_dead=2, exposure="staggered"), None, None, 2, seed=3)
    for a, b in zip(full.subpages, stag.subpages):
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_equal(a.values, b.values)


def test_staggered_exposure_samples_each_pixel_at_its_readout_time(textured, clean_bab):
    """Constant pan (dx = v t): under ``"staggered"`` each read pixel equals the
    box sample of the scene at the shift of its own conversion time
    (row-major readout order, window T / N), and ``true_shift`` is the mean
    of those shifts.  Under ``"full"`` the same pixel is the exposure average."""
    sensor = dataclasses.replace(clean_bab, exposure="staggered")
    v = 2.0  # LR px per s
    t = np.linspace(0.0, 1.0, 2001)
    yaw = np.arctan(v * t / sensor.f_px[0])
    trace = MotionTrace(t=t, theta=np.stack([yaw, np.zeros_like(t)], axis=1))
    cap = simulate_capture(textured, sensor, trace, None, 1, seed=0)
    T = sensor.subpage_period_s
    for k, sp in enumerate(cap.subpages):
        idx = np.flatnonzero(sp.mask.ravel())
        N = idx.size
        times = sp.t_start + (np.arange(N) + 0.5) * T / N
        dx = sensor.f_px[0] * np.tan(np.interp(times, t, yaw))
        for n in (0, N // 3, N - 1):
            i, j = np.unravel_index(idx[n], sensor.shape)
            ref = truth_at_scale(cap, 1, shift=(dx[n], 0.0))[i, j]
            assert sp.values[i, j] == pytest.approx(ref, abs=1e-6)
        assert sp.true_shift[0] == pytest.approx(dx.mean(), abs=1e-6)
        assert abs(sp.true_shift[1]) < 1e-12
    full = simulate_capture(textured, clean_bab, trace, None, 1, seed=0)
    assert np.nanmax(np.abs(full.subpages[0].values - cap.subpages[0].values)) > 1e-3


def test_unknown_exposure_rejected():
    with pytest.raises(ValueError):
        SensorConfig.bab(exposure="rolling")
