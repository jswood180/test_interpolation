"""Motion and gyro tests F5, F6 (plan section 5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch

from mlx_sr.motion import angles_to_shift, build_motion, shift_to_angles, simulate_gyro, trace_from_shifts
from mlx_sr.sensor import simulate_capture
from mlx_sr.types import GyroConfig, MotionConfig, SensorConfig


# F5 ------------------------------------------------------------------------


@pytest.mark.parametrize("rms_deg", [0.05, 0.3, 1.0])
def test_f5_tremor_psd_peak_and_rms(rms_deg, rng):
    cfg = MotionConfig(tremor_rms_deg=rms_deg, drift_rms_deg=0.0)
    duration = 20.0
    model = build_motion(cfg, duration, rng)
    fs = 1000.0
    t = np.arange(int(duration * fs)) / fs + 0.37e-3  # not the normalization grid
    ang = model.angles_at(t)
    for a in range(2):
        x = np.degrees(ang[:, a])
        f, p = welch(x - x.mean(), fs=fs, nperseg=4096)
        assert 8.0 <= f[np.argmax(p)] <= 12.0
        assert abs(x.std() / rms_deg - 1.0) < 0.10
        band = (f >= 7.5) & (f <= 12.5)
        assert p[band].sum() / p.sum() > 0.95
    assert np.allclose(model.angles_at(0.0), 0.0)
    assert abs(np.corrcoef(ang[:, 0], ang[:, 1])[0, 1]) < 0.2  # independent axes


def test_f5_long_capture_ensemble_rms(rng):
    """On a long capture the window normalization barely changes the spectrum:
    the ensemble RMS sqrt(sum a^2 / 2) of the synthesized sinusoids is within 10 %."""
    model = build_motion(MotionConfig(tremor_rms_deg=0.2, drift_rms_deg=0.0), 30.0, rng)
    for a in range(2):
        amp = model.components[a][0]
        assert np.degrees(np.sqrt(np.sum(amp ** 2) / 2)) == pytest.approx(0.2, rel=0.10)


def test_drift_band_and_wobble_amplitude(rng):
    drift = build_motion(MotionConfig(tremor_rms_deg=0.0, drift_rms_deg=0.5), 30.0, rng)
    t = np.arange(30000) / 1000.0
    x = np.degrees(drift.angles_at(t)[:, 0])
    f, p = welch(x - x.mean(), fs=1000.0, nperseg=8192)
    assert p[f <= 2.2].sum() / p.sum() > 0.95
    assert abs(x.std() / 0.5 - 1.0) < 0.10

    A = 2.0
    wob = build_motion(MotionConfig(tremor_rms_deg=0.0, drift_rms_deg=0.0, wobble_amp_deg=A, wobble_hz=1.5,
                                    wobble_phase_rad=0.3), 4.0, rng)
    ang = np.degrees(wob.angles_at(np.arange(4000) / 1000.0))
    assert np.ptp(ang[:, 0]) == pytest.approx(2 * A, rel=1e-3)
    assert np.ptp(ang[:, 1]) == pytest.approx(2 * A, rel=1e-3)
    r = np.hypot(ang[:, 0] - ang[:, 0].mean(), ang[:, 1] - ang[:, 1].mean())
    assert np.allclose(r, A, rtol=1e-3)  # circle

    line = build_motion(MotionConfig(tremor_rms_deg=0.0, drift_rms_deg=0.0, wobble_amp_deg=A,
                                     wobble_shape="line"), 4.0, rng)
    assert np.allclose(line.angles_at(np.linspace(0, 4, 100))[:, 1], 0.0)


def test_drift_rate_and_roll(rng):
    m = build_motion(MotionConfig(tremor_rms_deg=0.0, drift_rms_deg=0.0, drift_rate_dps=(1.0, -0.5),
                                  roll_rms_deg=0.4), 2.0, rng)
    t = np.linspace(0, 2, 2001)
    ang = np.degrees(m.angles_at(t))
    np.testing.assert_allclose(ang[:, 0], 1.0 * t, atol=1e-9)
    np.testing.assert_allclose(ang[:, 1], -0.5 * t, atol=1e-9)
    assert abs(ang[:, 2].std() / 0.4 - 1.0) < 0.1
    assert m.trace(t).roll is not None


def test_angle_shift_roundtrip():
    s = SensorConfig.baa()
    th = np.radians([[1.0, -2.0], [0.0, 0.5]])
    np.testing.assert_allclose(shift_to_angles(angles_to_shift(th, s), s), th, atol=1e-15)


def test_trace_from_shifts_exact(textured):
    s = SensorConfig.bab()
    shifts = np.random.default_rng(0).uniform(-1.5, 1.5, (8, 2))
    cap = simulate_capture(textured, s, trace_from_shifts(shifts, s), None, 4, seed=0)
    np.testing.assert_allclose(cap.true_shifts, shifts, atol=1e-12)


# F6 ------------------------------------------------------------------------


def _integrate(gyro):
    """Trapezoid-integrate (pitch, yaw) rates -> (yaw, pitch) angles, zero at stamp t = 0."""
    yaw = cumulative_trapezoid(gyro.omega[:, 1], gyro.t, initial=0.0)
    pitch = cumulative_trapezoid(gyro.omega[:, 0], gyro.t, initial=0.0)
    ang = np.column_stack([yaw, pitch])
    i0 = int(np.flatnonzero(gyro.t == 0.0)[0])
    return ang - ang[i0]


def test_f6_noiseless_gyro_integrates_to_theta(rng):
    cfg = MotionConfig.wobble(amp_deg=2.0, hz=1.5, tremor_rms_deg=0.3)
    duration = 2.0
    model = build_motion(cfg, duration, rng)
    gyro = simulate_gyro(model, GyroConfig.ideal(400.0), duration, rng)
    assert gyro.rate_hz == 400.0 and gyro.omega.shape == (gyro.t.size, 3)
    assert gyro.t[0] <= -0.05 and gyro.t[-1] >= duration + 0.05 and 0.0 in gyro.t
    np.testing.assert_allclose(np.diff(gyro.t), 1 / 400.0, rtol=1e-9)
    np.testing.assert_allclose(_integrate(gyro), model.angles_at(gyro.t)[:, :2], atol=1e-4)


def test_f6_via_capture_truth_trace(textured):
    cap = simulate_capture(textured, SensorConfig.bab(), MotionConfig.wobble(), GyroConfig.ideal(), 8, seed=3)
    g = cap.gyro
    inside = (g.t >= 0) & (g.t <= cap.meta["duration_s"])
    np.testing.assert_allclose(_integrate(g)[inside], cap.motion.theta_at(g.t[inside]), atol=1e-4)


def test_f6_bias_gives_linear_drift(rng):
    bias_dps = (0.5, -0.3, 0.2)
    model = build_motion(MotionConfig.natural(), 4.0, rng)
    gyro = simulate_gyro(model, GyroConfig(noise_density_dps_rthz=0.0, bias_dps=bias_dps, time_offset_s=0.0),
                         4.0, rng)
    np.testing.assert_allclose(gyro.bias, np.radians(bias_dps))
    err = _integrate(gyro) - model.angles_at(gyro.t)[:, :2]
    slope_yaw = np.polyfit(gyro.t, err[:, 0], 1)[0]
    slope_pitch = np.polyfit(gyro.t, err[:, 1], 1)[0]
    assert slope_yaw == pytest.approx(np.radians(bias_dps[1]), rel=1e-3)
    assert slope_pitch == pytest.approx(np.radians(bias_dps[0]), rel=1e-3)


def test_gyro_time_offset_and_noise(rng):
    tau = 0.015
    model = build_motion(MotionConfig.wobble(), 3.0, rng)
    g = simulate_gyro(model, GyroConfig(noise_density_dps_rthz=0.0, bias_dps=(0, 0, 0), time_offset_s=tau),
                      2.0, rng)
    assert g.time_offset_s == tau
    assert g.t[0] <= -(0.05 + tau) + 1e-12
    # Independent of rates_at: central differences of the angles at camera time t + tau.
    h = 1e-5
    fd = (model.angles_at(g.t + tau + h) - model.angles_at(g.t + tau - h)) / (2 * h)
    np.testing.assert_allclose(g.omega[:, 1], fd[:, 0], atol=1e-6)  # yaw rate
    np.testing.assert_allclose(g.omega[:, 0], fd[:, 1], atol=1e-6)  # pitch rate
    # Integrated gyro (stamp clock) equals theta shifted by tau.
    np.testing.assert_allclose(_integrate(g), model.angles_at(g.t + tau)[:, :2] - model.angles_at(tau)[:, :2],
                               atol=1e-4)

    nd = 0.007
    g = simulate_gyro(build_motion(MotionConfig.still(), 10.0, rng),
                      GyroConfig(noise_density_dps_rthz=nd, bias_dps=(0, 0, 0)), 10.0, rng)
    sd = np.degrees(g.omega.std(axis=0))
    np.testing.assert_allclose(sd, nd * np.sqrt(400.0), rtol=0.1)


def test_random_bias_and_offset_in_range(rng):
    g = simulate_gyro(build_motion(MotionConfig.still(), 1.0, rng), GyroConfig(), 1.0, rng)
    mag = np.degrees(np.abs(g.bias))
    assert np.all((mag >= 0.05) & (mag <= 1.0))
    taus = [simulate_gyro(build_motion(MotionConfig.still(), 1.0, rng), GyroConfig(time_offset_s=None), 1.0,
                          rng).time_offset_s for _ in range(20)]
    assert all(0.0 <= x <= 0.03 for x in taus) and np.ptp(taus) > 0.005
