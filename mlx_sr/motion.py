"""Camera motion synthesis (tremor, drift, wobble, roll), angle-to-pixel
conversion and gyro simulation.

Motion is synthesized as a sum of sinusoids per axis (random frequencies in
each band, random phases), so angles *and* exact angular rates can be
evaluated at any time.  Tremor and drift amplitudes are normalized so their
realized standard deviation over ``[0, duration]`` equals the requested RMS.
All angles are re-referenced so the pose at ``t = 0`` is zero (see the shift
convention in :mod:`mlx_sr.types`).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import GyroConfig, GyroTrace, MotionConfig, MotionTrace, SensorConfig

__all__ = [
    "MotionModel",
    "MotionSource",
    "TraceSource",
    "build_motion",
    "as_motion_source",
    "angles_to_shift",
    "shift_to_angles",
    "trace_from_shifts",
    "simulate_gyro",
    "TRACE_RATE_HZ",
]

TRACE_RATE_HZ: float = 1000.0
"""Sampling rate of the stored truth ``MotionTrace``."""

_DEG = np.pi / 180.0


class MotionSource(Protocol):
    """Anything that gives angles ``(N, 3)`` = (yaw, pitch, roll) and their rates."""

    def angles_at(self, t: np.ndarray) -> np.ndarray: ...

    def rates_at(self, t: np.ndarray) -> np.ndarray: ...


class MotionModel:
    """Analytic motion: ``angle_a(t) = sum_k A sin(2 pi f t + phi) + rate_a t - angle_a(0)``.

    ``components[a]`` is a tuple ``(amp, freq, phase)`` of 1-D arrays for axis
    ``a`` in ``(yaw, pitch, roll)``; ``linear_rate`` is ``(3,)`` rad/s.
    """

    def __init__(self, components: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                 linear_rate: np.ndarray | None = None):
        if len(components) != 3:
            raise ValueError("components must have 3 axes (yaw, pitch, roll)")
        self.components = [tuple(np.asarray(c, dtype=float) for c in comp) for comp in components]
        self.linear_rate = np.zeros(3) if linear_rate is None else np.asarray(linear_rate, float)
        self._offset = np.zeros(3)
        self._offset = self._raw(np.zeros(1))[0]

    def _raw(self, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((t.size, 3))
        for a, (amp, f, ph) in enumerate(self.components):
            if amp.size:
                out[:, a] = np.sin(2 * np.pi * np.outer(t, f) + ph) @ amp
            else:
                out[:, a] = 0.0
        out += t[:, None] * self.linear_rate
        return out

    def angles_at(self, t: np.ndarray | float) -> np.ndarray:
        """``(N, 3)`` yaw, pitch, roll in rad, zero at ``t = 0``."""
        return self._raw(t) - self._offset

    def rates_at(self, t: np.ndarray | float) -> np.ndarray:
        """``(N, 3)`` exact angular rates in rad/s."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((t.size, 3))
        for a, (amp, f, ph) in enumerate(self.components):
            if amp.size:
                out[:, a] = np.cos(2 * np.pi * np.outer(t, f) + ph) @ (amp * 2 * np.pi * f)
            else:
                out[:, a] = 0.0
        out += self.linear_rate
        return out

    def trace(self, t: np.ndarray) -> MotionTrace:
        ang = self.angles_at(t)
        roll = ang[:, 2] if np.any(ang[:, 2] != 0) else None
        return MotionTrace(t=np.asarray(t, float), theta=ang[:, :2].copy(), roll=roll)


class TraceSource:
    """Wrap a sampled :class:`MotionTrace` (linear interpolation; rates by
    finite differences of the samples)."""

    def __init__(self, trace: MotionTrace):
        self.trace = trace
        roll = trace.roll if trace.roll is not None else np.zeros_like(trace.t)
        self._ang = np.column_stack([trace.theta, roll])
        if trace.t.size > 1:
            self._rate = np.column_stack([np.gradient(self._ang[:, a], trace.t) for a in range(3)])
        else:
            self._rate = np.zeros_like(self._ang)

    def angles_at(self, t: np.ndarray | float) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return np.column_stack([np.interp(tt, self.trace.t, self._ang[:, a]) for a in range(3)])

    def rates_at(self, t: np.ndarray | float) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return np.column_stack([np.interp(tt, self.trace.t, self._rate[:, a]) for a in range(3)])


def as_motion_source(motion: MotionModel | MotionTrace | TraceSource) -> MotionSource:
    if isinstance(motion, MotionTrace):
        return TraceSource(motion)
    return motion


def _band_components(rng: np.random.Generator, band: tuple[float, float], n: int,
                     spectrum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = band
    if spectrum == "flat":
        f = rng.uniform(lo, hi, n)
        amp = rng.rayleigh(1.0, n)
    elif spectrum == "pink":  # 1/f amplitude, log-uniform frequencies
        f = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        amp = rng.rayleigh(1.0, n) / f
    else:
        raise ValueError(spectrum)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    return amp, f, ph


def _normalize(comp: tuple[np.ndarray, np.ndarray, np.ndarray], rms: float,
               t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp, f, ph = comp
    if rms <= 0 or amp.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    x = np.sin(2 * np.pi * np.outer(t_grid, f) + ph) @ amp
    sd = x.std()
    return amp * (rms / sd), f, ph


def build_motion(cfg: MotionConfig, duration_s: float, rng: np.random.Generator) -> MotionModel:
    """Synthesize a :class:`MotionModel` for a capture of length ``duration_s``.

    Tremor (flat spectrum in ``tremor_band_hz``) and drift (1/f amplitude in
    ``drift_band_hz``) are drawn independently for yaw and pitch, each
    normalized to its requested RMS over ``[0, duration_s]`` (evaluated on a
    1 kHz grid).  Wobble and the linear drift rate are added deterministically.
    """
    duration_s = max(float(duration_s), 1e-3)
    n_grid = max(int(np.ceil(duration_s * 1000.0)) + 1, 2000)
    t_grid = np.linspace(0.0, duration_s, n_grid)
    n = cfg.n_components
    comps: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = [[], [], []]
    for a in range(2):  # yaw, pitch
        tr = _band_components(rng, cfg.tremor_band_hz, n, "flat")
        dr = _band_components(rng, cfg.drift_band_hz, n, "pink")
        comps[a].append(_normalize(tr, cfg.tremor_rms_deg * _DEG, t_grid))
        comps[a].append(_normalize(dr, cfg.drift_rms_deg * _DEG, t_grid))
    # Roll: tremor + drift spectrum, normalized overall.
    rt = _band_components(rng, cfg.tremor_band_hz, n, "flat")
    rd = _band_components(rng, cfg.drift_band_hz, n, "pink")
    roll = tuple(np.concatenate([x, y]) for x, y in zip(rt, rd))
    comps[2].append(_normalize(roll, cfg.roll_rms_deg * _DEG, t_grid))

    phase = rng.uniform(0.0, 2 * np.pi) if cfg.wobble_phase_rad is None else cfg.wobble_phase_rad
    if cfg.wobble_amp_deg > 0:
        A, f = cfg.wobble_amp_deg * _DEG, cfg.wobble_hz
        comps[0].append((np.array([A]), np.array([f]), np.array([phase])))
        if cfg.wobble_shape == "circle":
            comps[1].append((np.array([A]), np.array([f]), np.array([phase + np.pi / 2])))
        elif cfg.wobble_shape == "lissajous":
            comps[1].append((np.array([A]), np.array([1.5 * f]), np.array([phase + np.pi / 2])))
        elif cfg.wobble_shape != "line":
            raise ValueError(f"unknown wobble_shape {cfg.wobble_shape!r}")

    merged = [tuple(np.concatenate([c[i] for c in axis]) if axis else np.zeros(0) for i in range(3))
              for axis in comps]
    rate = np.array([cfg.drift_rate_dps[0] * _DEG, cfg.drift_rate_dps[1] * _DEG, 0.0])
    return MotionModel(merged, linear_rate=rate)


def angles_to_shift(theta: np.ndarray, sensor: SensorConfig) -> np.ndarray:
    """``(..., 2)`` (yaw, pitch) rad -> ``(..., 2)`` (dx, dy) LR px via ``f_px * tan``."""
    theta = np.asarray(theta, dtype=float)
    fx, fy = sensor.f_px
    return np.stack([fx * np.tan(theta[..., 0]), fy * np.tan(theta[..., 1])], axis=-1)


def shift_to_angles(shift: np.ndarray, sensor: SensorConfig) -> np.ndarray:
    """Inverse of :func:`angles_to_shift`."""
    shift = np.asarray(shift, dtype=float)
    fx, fy = sensor.f_px
    return np.stack([np.arctan(shift[..., 0] / fx), np.arctan(shift[..., 1] / fy)], axis=-1)


def trace_from_shifts(shifts_px: np.ndarray, sensor: SensorConfig) -> MotionTrace:
    """Piecewise-constant motion giving exactly ``shifts_px[k]`` during subpage ``k``.

    Useful for tests with known shifts: ``simulate_capture(scene, sensor,
    trace_from_shifts(s, sensor), None, n_frames=len(s) // 2, seed)`` yields
    ``true_shift == s[k]``.  The pose jumps at the subpage boundaries, so a gyro
    simulated from this trace is meaningless (pass ``gyro_cfg=None``).
    """
    shifts_px = np.asarray(shifts_px, dtype=float).reshape(-1, 2)
    T = sensor.subpage_period_s
    eps = 1e-9 * T
    t, th = [], []
    ang = shift_to_angles(shifts_px, sensor)
    for k, a in enumerate(ang):
        t += [k * T, (k + 1) * T - eps]
        th += [a, a]
    return MotionTrace(t=np.asarray(t), theta=np.asarray(th), roll=None)


def simulate_gyro(motion: MotionModel | MotionTrace | TraceSource, cfg: GyroConfig,
                  duration_s: float, rng: np.random.Generator) -> GyroTrace:
    """Sample a gyro at ``cfg.rate_hz`` over stamps covering
    ``[-(pad + |tau|), duration_s + pad + |tau|]`` (``t = 0`` is a sample).

    ``omega[n] = rate(t[n] + tau) + bias + N(0, nd * sqrt(rate_hz))`` with axes
    ``(pitch, yaw, roll)`` in rad/s.  Random draws (from ``rng``, in order):
    tau if ``time_offset_s is None``, bias if ``bias_dps is None``, noise.
    """
    src = as_motion_source(motion)
    if cfg.time_offset_s is None:
        tau = float(rng.uniform(*cfg.time_offset_range_s))
    else:
        tau = float(cfg.time_offset_s)
    pad = cfg.pad_s + abs(tau)
    n0 = int(np.ceil(pad * cfg.rate_hz - 1e-9))
    n1 = int(np.ceil((duration_s + pad) * cfg.rate_hz - 1e-9))
    t = np.arange(-n0, n1 + 1) / cfg.rate_hz
    n = t.size
    rates = src.rates_at(t + tau)  # (yaw, pitch, roll)
    omega = rates[:, [1, 0, 2]]
    if cfg.bias_dps is None:
        lo, hi = cfg.bias_range_dps
        bias = rng.uniform(lo, hi, 3) * rng.choice([-1.0, 1.0], 3) * _DEG
    else:
        bias = np.asarray(cfg.bias_dps, dtype=float) * _DEG
    sigma = cfg.noise_density_dps_rthz * np.sqrt(cfg.rate_hz) * _DEG
    noise = rng.standard_normal((n, 3)) * sigma
    return GyroTrace(t=t, omega=omega + bias + noise, rate_hz=cfg.rate_hz, bias=bias,
                     time_offset_s=tau)
