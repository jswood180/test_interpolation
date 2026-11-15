"""Shared dataclasses and conventions for the MLX90640 super-resolution prototype.

This module is the frozen interface between the simulator (``scene``, ``motion``,
``sensor``) and the consumers (``registration``, ``operators``, ``reconstruct``,
``metrics``, ``pipeline``).  Do not change it without going through the
orchestrator.

Conventions
-----------
Arrays
    Images are row-major numpy arrays: ``img[i, j]`` with row ``i`` = y (down)
    and column ``j`` = x (right).  The sensor array is ``(24, 32)``.

Scene coordinates ("LR scene coords")
    Continuous ``(x, y)`` coordinates in units of LR pixels, with origin at the
    top-left corner of the *central crop* of the HR scene (i.e. after removing
    the margin).  The LR pixel ``(i, j)`` at zero shift covers
    ``x in [j, j+1)``, ``y in [i, i+1)`` and its center is ``(j+0.5, i+0.5)``.
    HR canvas pixel ``(I, J)`` covers
    ``x in [J/hr - margin, (J+1)/hr - margin)`` where ``hr = hr_factor``.

Shift convention
    A shift ``s = (dx, dy)`` is given in LR pixels, in **(x, y) order** (note:
    skimage returns ``(row, col)`` = ``(dy, dx)``).  Under shift ``s`` the LR
    pixel ``(i, j)`` samples scene coordinates ``(j + 0.5 + dx, i + 0.5 + dy)``,
    i.e. its footprint is ``[j+dx, j+1+dx) x [i+dy, i+1+dy)``.  Shifts are
    relative to the reference pose at ``t = 0``.  Consequently a constant shift
    of ``(+1, 0)`` makes the output equal ``np.roll(out0, -1, axis=1)`` in the
    interior, and moving the *scene content* right by one LR pixel makes it
    equal ``np.roll(out0, +1, axis=1)``.

Angles
    ``MotionTrace.theta[:, 0]`` is yaw and ``theta[:, 1]`` is pitch (radians,
    relative to the pose at ``t = 0``).  ``dx = fx * tan(yaw)`` and
    ``dy = fy * tan(pitch)``; positive yaw gives positive ``dx`` and positive
    pitch gives positive ``dy``.  The shift is applied uniformly to every
    pixel (the small-angle shift at the array center); the true pinhole shift
    grows toward the edges (by about ``sec^2`` of the field angle, roughly 3x
    at the BAA corners), so BAA results are optimistic.  This is a known model
    limitation, like roll.  Roll (optional) rotates the sampling
    footprint centers about the array center ``(16, 12)``:
    ``x' = cx + cos(r)(x-cx) - sin(r)(y-cy)``, ``y' = cy + sin(r)(x-cx) + cos(r)(y-cy)``.
    Only translation is assumed by reconstruction; roll is a model error.

Gyro axes
    ``GyroTrace.omega`` columns are ``(x, y, z) = (pitch rate, yaw rate,
    roll rate)`` in rad/s.  Note the order differs from ``theta``
    (yaw, pitch).  Angles are treated as small independent rotations, so
    ``omega = d(theta)/dt`` per axis.

Gyro time offset
    A gyro sample stamped ``t`` measured the motion at camera time
    ``t + time_offset_s``:  ``omega[n] = true_rate(t[n] + tau) + bias + noise``.
    To get the rate at camera time ``T`` use the gyro sample stamped
    ``T - tau``.  Gyro stamps extend beyond the capture on both sides by
    ``GyroConfig.pad_s + |tau|`` (so ``GyroTrace.t`` starts negative and
    ``t == 0`` is always a sample), which covers the reference subpage and
    a time-offset search window.

Explicit motion traces
    A :class:`MotionTrace` passed to ``simulate_capture`` is used as given and
    is *not* re-referenced to ``t = 0`` (e.g. ``trace_from_shifts`` yields
    absolute shifts).  Synthesized motion (``MotionConfig``) is always zero
    at ``t = 0``.

Reference frames for evaluation
    Registration estimates shifts relative to subpage ``ref_index``, whose own
    true shift is generally *not* zero (exposure averaging, motion after
    ``t = 0``).  Compare a reconstruction built from registered shifts with
    ``truth_at_scale(capture, s, ref_index=r)`` (equivalently
    ``shift=capture.ref_shift(r)``); use the default (``t = 0`` grid) only for
    reconstructions built from absolute ``capture.true_shifts``.

Target (super-resolved) grids
    At scale ``s`` the output is ``(24 s, 32 s)``.  Target pixel ``(u, v)``
    covers LR scene coords ``[v/s, (v+1)/s) x [u/s, (u+1)/s)`` plus the grid's
    reference shift.  LR pixel ``(i, j)`` therefore spans target pixels
    ``s*i .. s*i+s-1`` by ``s*j .. s*j+s-1`` at zero shift.

Subpage timing
    ``SensorConfig.refresh_hz`` is the *subpage* rate.  Subpage ``k`` (0-based)
    has ``subpage_id = k % 2``, ``frame_index = k // 2`` and is exposed over
    ``[k / refresh_hz, (k + 1) / refresh_hz)``.  ``n_frames`` in
    ``simulate_capture`` counts full frames, so ``K = 2 * n_frames`` subpages.

Temperatures
    All image values are degrees Celsius.  Dead pixels read ``DEAD_VALUE``
    (-273.15) and are ``False`` in ``Subpage.mask``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "DEAD_VALUE",
    "LR_SHAPE",
    "SensorConfig",
    "MotionConfig",
    "GyroConfig",
    "Subpage",
    "GyroTrace",
    "MotionTrace",
    "Capture",
    "Registration",
    "Reconstruction",
    "Roi",
    "EdgeSpec",
    "BarGroup",
    "HotSpot",
    "FlatRegion",
    "SceneMeta",
    "Scene",
]

DEAD_VALUE: float = -273.15
"""Value a dead pixel reports (degrees C)."""

LR_SHAPE: tuple[int, int] = (24, 32)
"""MLX90640 array shape (rows, cols)."""


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorConfig:
    """MLX90640 sensor model parameters.

    Defaults marked *assumption* are not taken from a datasheet and should be
    swept in experiments.
    """

    variant: Literal["BAA", "BAB"] = "BAB"
    fov_deg: tuple[float, float] = (55.0, 35.0)  # (horizontal, vertical)
    shape: tuple[int, int] = LR_SHAPE  # (rows, cols)
    refresh_hz: float = 8.0  # subpage rate, 0.5-64 Hz
    netd_1hz_k: float = 0.1  # RMS noise at 1 Hz refresh (datasheet BAB figure)
    fpn_sigma_k: float = 0.1  # residual per-pixel offset after calibration (assumption)
    n_dead: int = 0  # datasheet allows up to 4 defective pixels
    psf_sigma_px: float = 0.4  # Gaussian optics blur in LR px (unpublished; assumption)
    adc_lsb_k: float = 0.01  # output quantization step (assumption); 0 disables
    readout: Literal["chess", "interleaved"] = "chess"

    def __post_init__(self) -> None:
        if self.variant not in ("BAA", "BAB"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if not (0.5 <= self.refresh_hz <= 64.0):
            raise ValueError("refresh_hz must be within 0.5-64 Hz")
        if self.readout not in ("chess", "interleaved"):
            raise ValueError(f"unknown readout {self.readout!r}")
        if min(self.netd_1hz_k, self.fpn_sigma_k, self.psf_sigma_px, self.adc_lsb_k) < 0:
            raise ValueError("noise, FPN, PSF and LSB parameters must be >= 0")

    @classmethod
    def bab(cls, **overrides: Any) -> "SensorConfig":
        """MLX90640ESF-BAB: 55 x 35 deg FOV."""
        kw: dict[str, Any] = dict(variant="BAB", fov_deg=(55.0, 35.0))
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def baa(cls, **overrides: Any) -> "SensorConfig":
        """MLX90640ESF-BAA: 110 x 75 deg FOV."""
        kw: dict[str, Any] = dict(variant="BAA", fov_deg=(110.0, 75.0))
        kw.update(overrides)
        return cls(**kw)

    @property
    def f_px(self) -> tuple[float, float]:
        """Pinhole focal length in LR pixels, ``(fx, fy) = (N/2) / tan(FOV/2)``."""
        rows, cols = self.shape
        fx = (cols / 2.0) / math.tan(math.radians(self.fov_deg[0]) / 2.0)
        fy = (rows / 2.0) / math.tan(math.radians(self.fov_deg[1]) / 2.0)
        return fx, fy

    @property
    def px_per_deg(self) -> tuple[float, float]:
        """Small-angle image shift per degree at the array center, ``(x, y)``."""
        fx, fy = self.f_px
        return fx * math.pi / 180.0, fy * math.pi / 180.0

    @property
    def subpage_period_s(self) -> float:
        return 1.0 / self.refresh_hz

    @property
    def noise_sigma_k(self) -> float:
        """Per-subpage temporal noise sigma: ``NETD_1Hz * sqrt(refresh_hz)`` (assumed scaling)."""
        return self.netd_1hz_k * math.sqrt(self.refresh_hz)

    def subpage_mask(self, subpage_id: int) -> np.ndarray:
        """Boolean ``(rows, cols)`` mask of the pixels read in a subpage.

        Chess: ``(i + j) % 2 == subpage_id``.  Interleaved: ``i % 2 == subpage_id``.
        """
        rows, cols = self.shape
        i, j = np.mgrid[0:rows, 0:cols]
        if self.readout == "chess":
            return ((i + j) % 2) == subpage_id
        return (i % 2) == subpage_id


@dataclass(frozen=True)
class MotionConfig:
    """Camera angular motion model (all angles per axis, degrees).

    The total yaw/pitch is ``tremor + drift + drift_rate * t + wobble``, then
    re-referenced so the pose at ``t = 0`` is zero.

    - ``tremor_rms_deg``: per-axis RMS of physiological tremor, band-limited to
      ``tremor_band_hz`` (default 8-12 Hz).  The realized RMS over the capture
      duration is normalized to exactly this value (so sweeps get the
      requested tremor in every capture; for short captures the long-run RMS
      of the same realization can differ by tens of percent).
    - ``drift_rms_deg``: per-axis RMS of slow sway (1/f amplitude spectrum in
      ``drift_band_hz``, below 2 Hz), normalized like tremor.  *Assumption.*
    - ``drift_rate_dps``: constant pan rate ``(yaw, pitch)`` in deg/s.
    - ``wobble_amp_deg`` / ``wobble_hz``: deliberate sweep amplitude (peak,
      i.e. +/- amp) and frequency.  ``wobble_shape``: ``"circle"`` (yaw = A sin,
      pitch = A cos), ``"line"`` (yaw only) or ``"lissajous"`` (pitch at
      1.5 x frequency).  ``wobble_phase_rad=None`` draws a random phase.
    - ``roll_rms_deg``: optional roll (tremor + drift spectrum), normalized RMS.
    - ``n_components``: sinusoids per band per axis in the spectral synthesis.
    """

    tremor_rms_deg: float = 0.1
    tremor_band_hz: tuple[float, float] = (8.0, 12.0)
    drift_rms_deg: float = 0.2
    drift_band_hz: tuple[float, float] = (0.1, 2.0)
    drift_rate_dps: tuple[float, float] = (0.0, 0.0)
    wobble_amp_deg: float = 0.0
    wobble_hz: float = 1.5
    wobble_shape: Literal["circle", "line", "lissajous"] = "circle"
    wobble_phase_rad: float | None = None
    roll_rms_deg: float = 0.0
    n_components: int = 64

    @classmethod
    def still(cls) -> "MotionConfig":
        """No motion at all."""
        return cls(tremor_rms_deg=0.0, drift_rms_deg=0.0)

    @classmethod
    def natural(cls, tremor_rms_deg: float = 0.1, **overrides: Any) -> "MotionConfig":
        """Handheld: tremor plus default drift."""
        return cls(tremor_rms_deg=tremor_rms_deg, **overrides)

    @classmethod
    def wobble(cls, amp_deg: float = 2.0, hz: float = 1.5, **overrides: Any) -> "MotionConfig":
        """Deliberate wobble on top of natural tremor and drift."""
        return cls(wobble_amp_deg=amp_deg, wobble_hz=hz, **overrides)


@dataclass(frozen=True)
class GyroConfig:
    """Simulated MEMS gyro.  Defaults are *assumptions* (typical consumer IMU).

    - ``rate_hz``: output data rate.
    - ``noise_density_dps_rthz``: white rate noise density (deg/s/sqrt(Hz));
      per-sample sigma is ``noise_density * sqrt(rate_hz)`` so the integrated
      angle random walk equals the noise density.
    - ``bias_dps``: constant bias per axis ``(x, y, z)`` in deg/s; ``None``
      draws each axis with magnitude uniform in ``bias_range_dps`` and a random
      sign.
    - ``time_offset_s``: camera-gyro offset tau (see module docstring);
      ``None`` draws it uniformly from ``time_offset_range_s``.
    - ``pad_s``: extra gyro coverage before ``t = 0`` and after the capture
      (in addition to ``|tau|``).
    """

    rate_hz: float = 400.0
    noise_density_dps_rthz: float = 0.007
    bias_dps: tuple[float, float, float] | None = None
    bias_range_dps: tuple[float, float] = (0.05, 1.0)
    time_offset_s: float | None = 0.01
    time_offset_range_s: tuple[float, float] = (0.0, 0.03)
    pad_s: float = 0.05

    @classmethod
    def ideal(cls, rate_hz: float = 400.0) -> "GyroConfig":
        """Noiseless, unbiased, perfectly synchronized gyro."""
        return cls(rate_hz=rate_hz, noise_density_dps_rthz=0.0,
                   bias_dps=(0.0, 0.0, 0.0), time_offset_s=0.0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Subpage:
    """One MLX90640 subpage readout.

    ``values``: ``(24, 32)`` degrees C, NaN where not read, ``DEAD_VALUE`` at
    dead pixels of this subpage.  ``mask``: read and not dead.
    ``true_shift``: exposure-averaged true shift ``(dx, dy)`` (truth, may be None).
    """

    values: np.ndarray
    mask: np.ndarray
    subpage_id: int
    t_start: float
    t_end: float
    frame_index: int
    true_shift: np.ndarray | None = None

    @property
    def t_mid(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


@dataclass
class GyroTrace:
    """Gyro samples: ``t (N,)`` stamps in s, ``omega (N, 3)`` rad/s (pitch, yaw, roll).

    Stamps lie on the grid ``n / rate_hz`` and extend past the capture on
    both sides (``t[0] < 0``; see the module docstring).  ``bias`` (rad/s,
    ``(3,)``) and ``time_offset_s`` are truth, for evaluation only.
    """

    t: np.ndarray
    omega: np.ndarray
    rate_hz: float
    bias: np.ndarray | None = None
    time_offset_s: float | None = None


@dataclass
class MotionTrace:
    """True camera motion: ``t (N,)`` s, ``theta (N, 2)`` (yaw, pitch) rad.

    ``roll (N,)`` rad is optional (None = no roll).  Angles are relative to the
    pose at ``t = 0``.
    """

    t: np.ndarray
    theta: np.ndarray
    roll: np.ndarray | None = None

    def theta_at(self, t: np.ndarray | float) -> np.ndarray:
        """Linearly interpolated ``(len(t), 2)`` yaw/pitch (clamped at the ends)."""
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return np.stack([np.interp(tt, self.t, self.theta[:, a]) for a in range(2)], axis=-1)

    def roll_at(self, t: np.ndarray | float) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        if self.roll is None:
            return np.zeros_like(tt)
        return np.interp(tt, self.t, self.roll)


@dataclass(kw_only=True)
class Capture:
    """A simulated capture plus all ground truth.

    ``scene_hr`` is the unblurred HR scene (degrees C) including the margin;
    ``dead_mask`` (24, 32) bool; ``fpn`` (24, 32) static offsets in K.
    ``meta`` holds ``"scene"`` (:class:`SceneMeta` or None), ``"seed"``,
    ``"n_frames"``, ``"duration_s"``, ``"noise_sigma_k"``, ``"motion_cfg"``,
    ``"gyro_cfg"`` and ``"oob_fraction"`` (fraction of sample footprints that
    left the HR canvas; should be 0).
    """

    subpages: list[Subpage]
    gyro: GyroTrace | None
    sensor: SensorConfig
    scene_hr: np.ndarray
    hr_factor: int = 8
    margin_lr: int = 4
    dead_mask: np.ndarray
    fpn: np.ndarray
    motion: MotionTrace
    meta: dict = field(default_factory=dict)

    @property
    def n_subpages(self) -> int:
        return len(self.subpages)

    @property
    def true_shifts(self) -> np.ndarray:
        """``(K, 2)`` exposure-averaged true shifts relative to the ``t = 0`` pose."""
        return np.array([sp.true_shift for sp in self.subpages], dtype=float)

    def ref_shift(self, ref_index: int = 0) -> np.ndarray:
        """True shift ``(2,)`` of subpage ``ref_index``: the reference shift of
        the grid that registration-based reconstructions live on."""
        return np.asarray(self.subpages[ref_index].true_shift, dtype=float).copy()

    def relative_true_shifts(self, ref_index: int = 0) -> np.ndarray:
        """True shifts relative to subpage ``ref_index`` (what registration estimates)."""
        s = self.true_shifts
        return s - s[ref_index]

    @property
    def subpage_windows(self) -> np.ndarray:
        """``(K, 2)`` exposure windows ``(t_start, t_end)``."""
        return np.array([(sp.t_start, sp.t_end) for sp in self.subpages], dtype=float)

    @property
    def scene_meta(self) -> "SceneMeta | None":
        return self.meta.get("scene")


@dataclass
class Registration:
    """Per-subpage shift estimates ``(K, 2)`` in (dx, dy) relative to ``ref_index``.

    ``gyro_bias``: estimated gyro bias (rad/s, same axis order as
    ``GyroTrace.omega``) when the method estimates it, else None.
    """

    shifts: np.ndarray
    confidence: np.ndarray
    method: str
    time_offset_s: float | None = None
    gyro_bias: np.ndarray | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Reconstruction:
    """Reconstructed image ``(24 s, 32 s)`` on the target grid (see module docstring)."""

    image: np.ndarray
    certainty: np.ndarray | None
    scale: int
    method: str
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scene metadata (ROIs for metrics)
# ---------------------------------------------------------------------------

_EPS = 1e-9


@dataclass(frozen=True)
class Roi:
    """Axis-aligned region in LR scene coords, half-open ``[x0, x1) x [y0, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)

    def to_scale(self, scale: float, shift: tuple[float, float] | np.ndarray = (0.0, 0.0)
                 ) -> tuple[slice, slice]:
        """``(rows, cols)`` slices of target-grid pixels lying fully inside the ROI.

        ``scale=1`` gives LR pixel indices.  ``shift`` is the reference shift
        of the target grid (``(0, 0)`` for the ``t = 0`` pose).
        """
        dx, dy = float(shift[0]), float(shift[1])
        c0 = math.ceil((self.x0 - dx) * scale - _EPS)
        c1 = math.floor((self.x1 - dx) * scale + _EPS)
        r0 = math.ceil((self.y0 - dy) * scale - _EPS)
        r1 = math.floor((self.y1 - dy) * scale + _EPS)
        return slice(max(r0, 0), max(r1, 0)), slice(max(c0, 0), max(c1, 0))

    def to_hr(self, hr_factor: int = 8, margin_lr: int = 4) -> tuple[slice, slice]:
        """``(rows, cols)`` slices of HR canvas pixels fully inside the ROI."""
        s = self.to_scale(hr_factor, shift=(-margin_lr, -margin_lr))
        return s


@dataclass(frozen=True)
class EdgeSpec:
    """Slanted edge.  The edge line is ``x_edge(y) = cx + (y - cy) * tan(angle)``
    (angle measured from vertical, positive = edge leans right going down).
    Pixels with ``x > x_edge(y)`` (``hot_side="right"``) are ``high_c``.
    ``roi`` contains only the edge and the two flat sides (no other features).
    """

    roi: Roi
    angle_deg: float
    center: tuple[float, float]
    low_c: float
    high_c: float
    hot_side: Literal["right", "left"] = "right"


@dataclass(frozen=True)
class BarGroup:
    """Square-wave bar group (duty 0.5).  ``orientation="x"``: intensity varies
    along x (vertical bars).  Bars start (hot) at ``roi.x0``; the group spans
    ``n_cycles * period_lr``.  ``roi`` is the exact extent of the bars.
    """

    roi: Roi
    period_lr: float
    n_cycles: int
    low_c: float
    high_c: float
    orientation: Literal["x", "y"] = "x"

    @property
    def freq_cyc_per_lr(self) -> float:
        return 1.0 / self.period_lr


@dataclass(frozen=True)
class HotSpot:
    """Small uniform hot disc of true temperature ``peak_c`` on ``background_c``.
    ``roi`` is a square around the disc with clearance for blur.
    """

    center: tuple[float, float]
    radius_lr: float
    peak_c: float
    background_c: float
    roi: Roi


@dataclass(frozen=True)
class FlatRegion:
    """Uniform region of temperature ``value_c``; ``roi`` keeps >= 2 LR px from
    any feature so blur and motion do not reach it.
    """

    roi: Roi
    value_c: float


@dataclass
class SceneMeta:
    """Metadata a scene generator attaches for metrics.

    All geometry is in LR scene coords (see module docstring).  Convert with
    ``Roi.to_scale(scale, shift)`` for target grids or ``Roi.to_hr`` for the HR
    canvas.  Lists are empty when a scene has no such feature.
    """

    kind: str
    shape_lr: tuple[int, int]
    hr_factor: int
    margin_lr: int
    background_c: float
    edge: EdgeSpec | None = None
    bars: list[BarGroup] = field(default_factory=list)
    hotspots: list[HotSpot] = field(default_factory=list)
    flat: list[FlatRegion] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Scene:
    """HR scene ``hr`` (degrees C) of shape
    ``((rows + 2 margin) * hr_factor, (cols + 2 margin) * hr_factor)`` plus metadata.
    """

    hr: np.ndarray
    meta: SceneMeta
    hr_factor: int = 8
    margin_lr: int = 4
