"""MLX90640 forward model: HR scene + motion -> chess subpages (+ gyro).

Simulator pipeline per capture (deliberately *different* from the
reconstruction operator in :mod:`mlx_sr.operators`, to avoid the inverse
crime):

1. Optics: blur the 8x HR canvas with a discrete Gaussian of
   ``psf_sigma_px * hr_factor`` HR px (``scipy.ndimage.gaussian_filter``).
2. Pixel footprint: exact box integration of the (blurred, piecewise-constant)
   HR canvas.  An integral image gives the footprint mean ``B`` at every
   integer HR corner; since the footprint is exactly ``hr`` HR px wide, the
   mean at a fractional corner is the bilinear interpolation of ``B``
   (exact, not an approximation).  Without roll all pixels share one shift,
   so a subpage is ``B`` gathered through a bilinear-splat kernel of the
   exposure's shift samples.
3. Motion blur: each subpage averages the footprint integral over its exposure
   window ``[k/rate, (k+1)/rate)`` with a midpoint rule whose sample count
   keeps the footprint step below ``EXPOSURE_STEP_PX`` (4 to
   ``EXPOSURE_SAMPLES_HZ * T`` samples).  ``true_shift`` is the
   average shift over the same samples.
4. Readout: chess (or interleaved) subpage masks, ``subpage_id = k % 2``.
5. Noise: static per-pixel FPN offsets ``N(0, fpn_sigma_k)``, temporal noise
   ``N(0, netd_1hz_k * sqrt(refresh_hz))``, quantization to ``adc_lsb_k``,
   dead pixels reading ``DEAD_VALUE``; unread pixels are NaN.

Random streams: ``SeedSequence(seed).spawn(4)`` gives independent generators
for motion, gyro, fixed-pattern defects (FPN, dead pixels) and temporal noise,
so changing e.g. the gyro config does not change the image noise.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

from . import motion as _motion
from .types import (
    DEAD_VALUE,
    Capture,
    GyroConfig,
    MotionConfig,
    MotionTrace,
    Scene,
    SensorConfig,
    Subpage,
)

__all__ = [
    "simulate_capture",
    "truth_at_scale",
    "merge_subpages",
    "box_sample",
    "EXPOSURE_SAMPLES_HZ",
    "EXPOSURE_STEP_PX",
]

EXPOSURE_SAMPLES_HZ: float = 2000.0
"""Maximum temporal sampling density used to integrate motion blur within a subpage."""

EXPOSURE_STEP_PX: float = 0.01
"""Target footprint displacement between consecutive motion-blur samples (LR px)."""


class _IntegralImage:
    """Exact box means of a piecewise-constant image at fractional coordinates."""

    def __init__(self, img: np.ndarray):
        self.shape = img.shape
        self.mean = float(img.mean())  # subtract for numerical accuracy
        S = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(img - self.mean, axis=0), axis=1)
        self.S = S

    def _at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        H, W = self.shape
        ix = np.clip(np.floor(x).astype(np.int64), 0, W - 1)
        iy = np.clip(np.floor(y).astype(np.int64), 0, H - 1)
        fx = x - ix
        fy = y - iy
        S = self.S
        return (S[iy, ix] * (1 - fx) * (1 - fy) + S[iy, ix + 1] * fx * (1 - fy)
                + S[iy + 1, ix] * (1 - fx) * fy + S[iy + 1, ix + 1] * fx * fy)

    def box_mean(self, x0, x1, y0, y1) -> tuple[np.ndarray, float]:
        """Mean over ``[x0, x1) x [y0, y1)`` (HR px); boxes are clipped to the
        canvas (edge region then averages only the inside part).  Returns
        ``(mean, fraction_of_boxes_clipped)``."""
        H, W = self.shape
        cx0, cx1 = np.clip(x0, 0, W), np.clip(x1, 0, W)
        cy0, cy1 = np.clip(y0, 0, H), np.clip(y1, 0, H)
        clipped = (cx0 != x0) | (cx1 != x1) | (cy0 != y0) | (cy1 != y1)
        area = (cx1 - cx0) * (cy1 - cy0)
        tot = self._at(cx1, cy1) - self._at(cx0, cy1) - self._at(cx1, cy0) + self._at(cx0, cy0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(area > 0, tot / np.where(area > 0, area, 1.0), 0.0) + self.mean
        return m, float(np.mean(clipped)) if clipped.size else 0.0


def box_sample(img_hr: np.ndarray, hr_factor: int, margin_lr: float, shape: tuple[int, int],
               scale: float = 1.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Box-average ``img_hr`` onto a ``(rows*scale, cols*scale)`` grid whose pixel
    ``(u, v)`` covers LR scene coords ``[v/s + dx, (v+1)/s + dx) x [u/s + dy, ...)``.
    """
    rows, cols = int(round(shape[0] * scale)), int(round(shape[1] * scale))
    ii = _IntegralImage(img_hr)
    v = np.arange(cols)
    u = np.arange(rows)
    x0 = (v / scale + shift[0] + margin_lr) * hr_factor
    x1 = ((v + 1) / scale + shift[0] + margin_lr) * hr_factor
    y0 = (u / scale + shift[1] + margin_lr) * hr_factor
    y1 = ((u + 1) / scale + shift[1] + margin_lr) * hr_factor
    X0, Y0 = np.meshgrid(x0, y0)
    X1, Y1 = np.meshgrid(x1, y1)
    m, _ = ii.box_mean(X0, X1, Y0, Y1)
    return m


def _footprint_means(img: np.ndarray, hr: int) -> np.ndarray:
    """``B[Y, X] = mean(img[Y:Y+hr, X:X+hr])`` for all integer corners, shape ``(H-hr+1, W-hr+1)``."""
    mean = float(img.mean())
    S = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(img - mean, axis=0), axis=1)
    return (S[hr:, hr:] - S[:-hr, hr:] - S[hr:, :-hr] + S[:-hr, :-hr]) / (hr * hr) + mean


def _bilinear(B: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation of ``B`` at ``(X, Y)`` (clamped); also returns the out-of-range mask."""
    H, W = B.shape
    out = (X < 0) | (Y < 0) | (X > W - 1) | (Y > H - 1)
    X = np.clip(X, 0, W - 1)
    Y = np.clip(Y, 0, H - 1)
    ix = np.minimum(np.floor(X).astype(np.int64), W - 2)
    iy = np.minimum(np.floor(Y).astype(np.int64), H - 2)
    fx, fy = X - ix, Y - iy
    v = (B[iy, ix] * (1 - fx) * (1 - fy) + B[iy, ix + 1] * fx * (1 - fy)
         + B[iy + 1, ix] * (1 - fx) * fy + B[iy + 1, ix + 1] * fx * fy)
    return v, out


def _motion_kernel(off_hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear splat of ``(n, 2)`` fractional HR offsets (x, y) onto integer
    offsets.  Returns ``(offsets (m, 2) int, weights (m,))`` with weights
    summing to 1; averaging ``B`` over the samples equals ``sum w B[base + off]``."""
    n = off_hr.shape[0]
    f = np.floor(off_hr)
    a = off_hr - f
    f = f.astype(np.int64)
    ox = np.concatenate([f[:, 0], f[:, 0] + 1, f[:, 0], f[:, 0] + 1])
    oy = np.concatenate([f[:, 1], f[:, 1], f[:, 1] + 1, f[:, 1] + 1])
    w = np.concatenate([(1 - a[:, 0]) * (1 - a[:, 1]), a[:, 0] * (1 - a[:, 1]),
                        (1 - a[:, 0]) * a[:, 1], a[:, 0] * a[:, 1]]) / n
    keys, inv = np.unique(np.stack([ox, oy], axis=1), axis=0, return_inverse=True)
    wk = np.bincount(inv.ravel(), weights=w, minlength=keys.shape[0])
    keep = wk != 0
    return keys[keep], wk[keep]


def _n_exposure_samples(source, sensor: SensorConfig, t0: float, T: float) -> int:
    """Midpoint-rule sample count for one exposure window.

    The footprint path (translation plus roll displacement at the array
    corner) is probed at ``EXPOSURE_SAMPLES_HZ``; the count is chosen so that
    consecutive samples move at most ``EXPOSURE_STEP_PX`` (min 4 samples, max
    the probe count).
    """
    n_probe = max(4, int(np.ceil(T * EXPOSURE_SAMPLES_HZ)))
    tp = t0 + np.linspace(0.0, T, n_probe, endpoint=False)
    ang = source.angles_at(tp)
    sh = _motion.angles_to_shift(ang[:, :2], sensor)
    corner = 0.5 * float(np.hypot(*sensor.shape))
    step = np.hypot(np.diff(sh[:, 0]), np.diff(sh[:, 1])) + np.abs(np.diff(ang[:, 2])) * corner
    n = int(np.ceil(step.sum() / EXPOSURE_STEP_PX))
    return int(np.clip(n, 4, n_probe))


def _scene_geometry(scene_hr, sensor: SensorConfig) -> tuple[np.ndarray, int, int, object]:
    rows, cols = sensor.shape
    if isinstance(scene_hr, Scene):
        img = np.asarray(scene_hr.hr, float)
        hr, m = scene_hr.hr_factor, scene_hr.margin_lr
        if img.shape != ((rows + 2 * m) * hr, (cols + 2 * m) * hr):
            raise ValueError(f"Scene shape {img.shape} does not match sensor {sensor.shape} "
                             f"with hr_factor={hr}, margin_lr={m}")
        return img, hr, m, scene_hr.meta
    img = np.asarray(scene_hr, dtype=float)
    hr = 8
    H, W = img.shape
    if H % hr or W % hr:
        raise ValueError("scene_hr shape must be a multiple of hr_factor=8")
    mr, mc = (H // hr - rows), (W // hr - cols)
    if mr != mc or mr < 0 or mr % 2:
        raise ValueError(f"scene_hr shape {img.shape} does not match a symmetric margin around {sensor.shape}")
    return img, hr, mr // 2, None


def simulate_capture(scene_hr: np.ndarray | Scene, sensor: SensorConfig,
                     motion_cfg: MotionConfig | MotionTrace | None, gyro_cfg: GyroConfig | None,
                     n_frames: int, seed: int) -> Capture:
    """Simulate ``n_frames`` full frames (``K = 2 n_frames`` subpages).

    ``scene_hr``: a :class:`Scene` (metadata is copied to ``capture.meta["scene"]``)
    or a raw HR array at ``hr_factor=8`` with a symmetric margin.
    ``motion_cfg``: a :class:`MotionConfig` (synthesized with the motion
    stream), an explicit :class:`MotionTrace` (e.g. from
    :func:`mlx_sr.motion.trace_from_shifts`), or None for no motion.
    ``gyro_cfg``: None for no gyro.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    img, hr, margin, scene_meta = _scene_geometry(scene_hr, sensor)
    rows, cols = sensor.shape
    K = 2 * n_frames
    T = sensor.subpage_period_s
    duration = K * T

    ss = np.random.SeedSequence(seed)
    rng_motion, rng_gyro, rng_fixed, rng_noise = (np.random.default_rng(s) for s in ss.spawn(4))

    if motion_cfg is None:
        motion_cfg = MotionConfig.still()
    if isinstance(motion_cfg, MotionTrace):
        source = _motion.TraceSource(motion_cfg)
        truth_trace = motion_cfg  # stored as given
    else:
        source = _motion.build_motion(motion_cfg, duration, rng_motion)
        trace_t = np.arange(int(np.ceil(duration * _motion.TRACE_RATE_HZ)) + 1) / _motion.TRACE_RATE_HZ
        truth_trace = source.trace(trace_t)

    # Optics blur on the HR canvas, then the footprint-mean image B[Y, X] =
    # mean(blurred[Y:Y+hr, X:X+hr]).  Because the footprint is an integer
    # number of HR px wide, the footprint mean as a function of a continuous
    # top-left corner is exactly bilinear between integer corners, so
    # bilinear interpolation of B is exact box integration.
    sigma_hr = sensor.psf_sigma_px * hr
    blurred = ndimage.gaussian_filter(img, sigma_hr, mode="nearest", truncate=4.0) if sigma_hr > 0 else img
    B = _footprint_means(blurred, hr)
    # Fixed pattern: FPN and dead pixels.
    fpn = rng_fixed.standard_normal((rows, cols)) * sensor.fpn_sigma_k
    dead_mask = np.zeros((rows, cols), dtype=bool)
    if sensor.n_dead > 0:
        idx = rng_fixed.choice(rows * cols, size=sensor.n_dead, replace=False)
        dead_mask.flat[idx] = True

    jj, iii = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    base_x = (np.arange(cols) + margin) * hr
    base_y = (np.arange(rows) + margin) * hr
    cx, cy = cols / 2.0, rows / 2.0
    sigma_n = sensor.noise_sigma_k
    masks = [sensor.subpage_mask(0), sensor.subpage_mask(1)]

    subpages: list[Subpage] = []
    oob = []
    for k in range(K):
        t0, t1 = k * T, (k + 1) * T
        n_sub = _n_exposure_samples(source, sensor, t0, T)
        ts = t0 + (np.arange(n_sub) + 0.5) / n_sub * T
        ang = source.angles_at(ts)  # (n_sub, 3)
        shifts = _motion.angles_to_shift(ang[:, :2], sensor)  # (n_sub, 2)
        roll = ang[:, 2]
        if np.any(roll != 0.0):
            # Per-pixel corners differ under roll: evaluate every sample.
            c, s_ = np.cos(roll)[:, None, None], np.sin(roll)[:, None, None]
            xc = cx + c * (jj + 0.5 - cx) - s_ * (iii + 0.5 - cy)
            yc = cy + s_ * (jj + 0.5 - cx) + c * (iii + 0.5 - cy)
            X0 = (xc - 0.5 + shifts[:, 0, None, None] + margin) * hr
            Y0 = (yc - 0.5 + shifts[:, 1, None, None] + margin) * hr
            m, out = _bilinear(B, X0, Y0)
            signal = m.mean(axis=0)
            oob.append(np.mean(np.any(out, axis=(1, 2))))
        else:
            # Same shift for all pixels: splat the shift path into a small
            # bilinear kernel over integer HR offsets, then gather.
            offs, w = _motion_kernel(shifts * hr)
            Y = base_y[:, None, None] + offs[None, None, :, 1]
            X = base_x[None, :, None] + offs[None, None, :, 0]
            Yc = np.clip(Y, 0, B.shape[0] - 1)
            Xc = np.clip(X, 0, B.shape[1] - 1)
            signal = B[Yc, Xc] @ w
            bad = (np.abs(shifts[:, 0]) > margin) | (np.abs(shifts[:, 1]) > margin)
            oob.append(float(np.mean(bad)))

        sid = k % 2
        read = masks[sid]
        noise = rng_noise.standard_normal((rows, cols)) * sigma_n
        vals = signal + fpn + noise
        if sensor.adc_lsb_k > 0:
            vals = np.round(vals / sensor.adc_lsb_k) * sensor.adc_lsb_k
        vals = np.where(read, vals, np.nan)
        vals[read & dead_mask] = DEAD_VALUE
        subpages.append(Subpage(values=vals, mask=read & ~dead_mask, subpage_id=sid,
                                t_start=t0, t_end=t1, frame_index=k // 2,
                                true_shift=shifts.mean(axis=0)))

    gyro = None
    if gyro_cfg is not None:
        gyro = _motion.simulate_gyro(source, gyro_cfg, duration, rng_gyro)

    oob_fraction = float(np.mean(oob)) if oob else 0.0
    if oob_fraction > 0:
        warnings.warn(f"{oob_fraction:.1%} of exposure samples had footprints outside the HR canvas; "
                      "increase margin_lr or reduce motion", RuntimeWarning, stacklevel=2)

    meta = dict(scene=scene_meta, seed=seed, n_frames=n_frames, duration_s=duration,
                noise_sigma_k=sigma_n, motion_cfg=motion_cfg if not isinstance(motion_cfg, MotionTrace) else None,
                gyro_cfg=gyro_cfg, oob_fraction=oob_fraction)
    return Capture(subpages=subpages, gyro=gyro, sensor=sensor, scene_hr=img, hr_factor=hr,
                   margin_lr=margin, dead_mask=dead_mask, fpn=fpn, motion=truth_trace, meta=meta)


def truth_at_scale(capture: Capture, scale: int, shift: tuple[float, float] | np.ndarray | None = None,
                   *, ref_index: int | None = None) -> np.ndarray:
    """Ground truth box-averaged onto the ``(24 s, 32 s)`` target grid (no PSF).

    The grid reference shift is, in order of precedence: ``shift`` if given;
    ``capture.ref_shift(ref_index)`` if ``ref_index`` is given; else ``(0, 0)``
    (the ``t = 0`` pose).  Use ``ref_index=r`` for reconstructions built from
    registration output relative to subpage ``r``; use the default for
    reconstructions built from absolute ``capture.true_shifts``.
    """
    if shift is not None and ref_index is not None:
        raise ValueError("pass either shift or ref_index, not both")
    if shift is None:
        shift = capture.ref_shift(ref_index) if ref_index is not None else (0.0, 0.0)
    return box_sample(capture.scene_hr, capture.hr_factor, capture.margin_lr, capture.sensor.shape,
                      scale=scale, shift=(float(shift[0]), float(shift[1])))


def merge_subpages(a: Subpage, b: Subpage) -> tuple[np.ndarray, np.ndarray]:
    """Naive (Adafruit-style) merge of two complementary subpages.

    Returns ``(values, mask)``: each pixel comes from whichever subpage read it
    (ignoring that they were exposed at different times); dead pixels are
    ``False`` in ``mask``.
    """
    values = np.where(a.mask, a.values, np.where(b.mask, b.values, np.nan))
    read_a = ~np.isnan(a.values)
    read_b = ~np.isnan(b.values)
    values = np.where(read_a & ~a.mask, a.values, values)  # keep DEAD_VALUE visible
    values = np.where(read_b & ~b.mask, b.values, values)
    return values, a.mask | b.mask
