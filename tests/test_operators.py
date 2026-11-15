"""Forward-operator tests O1-O3 (plan section 5) plus grid-alignment checks."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr

from mlx_sr.motion import trace_from_shifts
from mlx_sr.operators import axis_weights, build_operator, valid_samples
from mlx_sr.reconstruct import btv_value_and_grad, map_objective
from mlx_sr.sensor import simulate_capture, truth_at_scale
from mlx_sr.types import DEAD_VALUE, SensorConfig


def _capture(scene, sensor, shifts, seed=0):
    s = np.asarray(shifts, float)
    return simulate_capture(scene, sensor, trace_from_shifts(s, sensor), None, len(s) // 2, seed)


def _with_dead(subpages, rng, n=6):
    """Copies of ``subpages`` with ``n`` extra dead pixels per subpage."""
    out = []
    for sp in subpages:
        read = np.flatnonzero(sp.mask)
        dead = rng.choice(read, size=n, replace=False)
        v, m = sp.values.copy(), sp.mask.copy()
        v.flat[dead] = DEAD_VALUE
        m.flat[dead] = False
        out.append(dataclasses.replace(sp, values=v, mask=m))
    return out


def _matrix_free(subpages, shifts, sensor, scale, sigma):
    """Independent separable application of the operator and its adjoint:
    ``A_k x = mask_k . (Wy_k X Wx_k^T)`` and ``A_k^T r = Wy_k^T R_k Wx_k``."""
    rows, cols = sensor.shape
    H, W = rows * scale, cols * scale
    idx = valid_samples(subpages)
    Wx = [axis_weights(np.arange(cols) + dx, W, scale, sigma) for dx, _ in shifts]
    Wy = [axis_weights(np.arange(rows) + dy, H, scale, sigma) for _, dy in shifts]

    def fwd(x):
        X = x.reshape(H, W)
        return np.concatenate([(wy @ X @ wx.T).ravel()[i] for wx, wy, i in zip(Wx, Wy, idx)])

    def adj(r):
        out = np.zeros((H, W))
        n0 = 0
        for wx, wy, i in zip(Wx, Wy, idx):
            R = np.zeros(rows * cols)
            R[i] = r[n0:n0 + i.size]
            n0 += i.size
            out += wy.T @ R.reshape(rows, cols) @ wx
        return out.ravel()

    return fwd, adj


SHIFTS = np.array([[0.0, 0.0], [0.0, 0.0], [0.37, -0.81], [-1.2, 0.45], [0.9, 1.3], [-0.55, -0.25]])


# O1 ------------------------------------------------------------------------


@pytest.mark.parametrize("scale,sigma", [(2, 0.4), (3, 0.4), (2, 0.0), (3, 0.75)])
def test_o1_adjoint_and_matrix_free_agreement(textured, scale, sigma):
    """O1: |<Ax, y> - <x, A^T y>| < 1e-10 * scale.  The sparse A^T is compared
    with an independent matrix-free adjoint (separable weights), so this is
    not a trivial transpose identity; dead pixels are included."""
    rng = np.random.default_rng(1)
    sensor = SensorConfig.bab(psf_sigma_px=sigma)
    cap = _capture(textured, sensor, SHIFTS)
    subs = _with_dead(cap.subpages, rng)
    A, y = build_operator(subs, SHIFTS, sensor, scale)
    H, W = 24 * scale, 32 * scale
    assert A.shape == (sum(int(sp.mask.sum()) for sp in subs), H * W)
    assert np.all(np.isfinite(y)) and not np.any(y == DEAD_VALUE)
    fwd, adj = _matrix_free(subs, SHIFTS, sensor, scale, sigma)
    for _ in range(3):
        x = rng.standard_normal(H * W)
        r = rng.standard_normal(A.shape[0])
        Ax = A @ x
        ATr = adj(r)
        np.testing.assert_allclose(Ax, fwd(x), atol=1e-12)
        np.testing.assert_allclose(A.T @ r, ATr, atol=1e-12)
        assert abs(Ax @ r - x @ ATr) < 1e-10 * scale


# O2 ------------------------------------------------------------------------


@pytest.mark.parametrize("scale,sigma", [(2, 0.4), (3, 0.6), (2, 0.0)])
def test_o2_rows_sum_to_one(textured, scale, sigma):
    """O2: every row of A sums to 1, including border samples whose footprint
    leaves the target grid (large shifts); weights are non-negative."""
    sensor = SensorConfig.bab(psf_sigma_px=sigma)
    cap = _capture(textured, sensor, SHIFTS)
    A, _ = build_operator(cap.subpages, SHIFTS, sensor, scale)
    np.testing.assert_allclose(np.asarray(A.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    assert A.data.min() >= 0.0


def test_axis_weights_match_numerical_box_gauss_integral():
    """Interior 1-D weights equal the numerically integrated box (x) Gaussian
    kernel over each target cell (checks the erf antiderivative formula)."""
    sigma, scale, n = 0.4, 3, 66
    left = np.array([10.27])
    w = axis_weights(left, n, scale, sigma)[0]

    def kernel(x):  # box [a, a+1) convolved with N(0, sigma^2)
        return ndtr((x - left[0]) / sigma) - ndtr((x - left[0] - 1.0) / sigma)

    ref = np.array([integrate.quad(kernel, v / scale, (v + 1) / scale, epsabs=1e-13)[0]
                    for v in range(n)])
    # Same truncation rule as the operator: keep cells intersecting the box +/- 3 sigma.
    lo, hi = np.arange(n) / scale, (np.arange(n) + 1) / scale
    ref[~((hi > left[0] - 3 * sigma) & (lo < left[0] + 1 + 3 * sigma))] = 0.0
    ref /= ref.sum()
    np.testing.assert_allclose(w, ref, atol=1e-10)
    assert np.isclose(w.sum(), 1.0)
    # Centroid of the weights sits at the footprint center.
    centers = (np.arange(n) + 0.5) / scale
    assert abs(w @ centers - (left[0] + 0.5)) < 1e-3


def test_grid_alignment_exact_with_half_pixel_shifts(chart, clean_bab):
    """Grid alignment and sign convention: with PSF 0 and shifts that are
    multiples of 1/s, each LR footprint is exactly a union of target cells, so
    A @ truth_at_scale(capture, s) reproduces the noiseless samples.  Only
    samples whose footprint lies inside the target grid are compared (the
    operator folds off-grid mass into border cells, the simulator samples the
    real margin)."""
    s = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, -0.5], [0.5, -0.5], [-1.5, 1.0], [-1.5, 1.0]])
    cap = _capture(chart, clean_bab, s)
    truth = truth_at_scale(cap, 2).ravel()
    inside = []
    for (dx, dy), idx in zip(s, valid_samples(cap.subpages)):
        i, j = np.divmod(idx, 32)
        inside.append((j + dx >= 0) & (j + 1 + dx <= 32) & (i + dy >= 0) & (i + 1 + dy <= 24))
    inside = np.concatenate(inside)
    assert inside.mean() > 0.85
    A, y = build_operator(cap.subpages, s, clean_bab, 2, psf_sigma=0.0)
    np.testing.assert_allclose((A @ truth)[inside], y[inside], atol=1e-9)
    # A sign error in the shift would break it.
    A_bad, _ = build_operator(cap.subpages, -s, clean_bab, 2, psf_sigma=0.0)
    assert np.max(np.abs((A_bad @ truth - y)[inside])) > 1.0


def test_operator_predicts_blurred_capture(chart):
    """With PSF 0.4, A applied to the 6x truth predicts the (simulator's)
    noiseless samples to within the piecewise-constant discretization error."""
    sensor = SensorConfig.bab(netd_1hz_k=0.0, fpn_sigma_k=0.0, adc_lsb_k=0.0, psf_sigma_px=0.4)
    s = np.array([[0.0, 0.0], [0.0, 0.0], [0.31, -0.62], [-0.44, 0.18]])
    cap = _capture(chart, sensor, s)
    A, y = build_operator(cap.subpages, s, sensor, 6)
    err = A @ truth_at_scale(cap, 6).ravel() - y
    assert np.sqrt(np.mean(err ** 2)) < 0.02  # K, on a 10-35 K contrast chart
    assert np.max(np.abs(err)) < 0.3


# O3 ------------------------------------------------------------------------


@pytest.mark.parametrize("p,alpha,eps", [(2, 0.7, 0.2), (1, 0.5, 0.05), (3, 0.8, 1.0)])
def test_o3_btv_gradient_finite_difference(p, alpha, eps):
    """O3: the analytic BTV gradient matches central finite differences."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((9, 11)) * 2.0
    f0, g = btv_value_and_grad(x, p, alpha, eps)
    h = 1e-6
    fd = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = h
        fd[idx] = (btv_value_and_grad(x + e, p, alpha, eps)[0] - btv_value_and_grad(x - e, p, alpha, eps)[0]) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    assert f0 >= 0.0


def test_o3_full_map_objective_gradient(textured):
    """O3 (data + prior): the BTV-MAP objective gradient through A / A^T
    matches finite differences along random directions and per coordinate."""
    rng = np.random.default_rng(4)
    sensor = SensorConfig.bab()
    cap = _capture(textured, sensor, SHIFTS)
    A, y = build_operator(cap.subpages, SHIFTS, sensor, 2)
    shape = (48, 64)
    x = 25.0 + rng.standard_normal(48 * 64)
    kw = dict(lam=0.05, alpha=0.7, p=2, eps_data=0.3, eps_btv=0.2)
    f0, g = map_objective(x, A, y, shape, **kw)
    h = 1e-5
    for _ in range(5):
        d = rng.standard_normal(x.size)
        fp = map_objective(x + h * d, A, y, shape, **kw)[0]
        fm = map_objective(x - h * d, A, y, shape, **kw)[0]
        assert abs((fp - fm) / (2 * h) - g @ d) < 1e-5 * max(1.0, abs(g @ d))
    for j in rng.choice(x.size, size=20, replace=False):
        e = np.zeros_like(x)
        e[j] = h
        fd = (map_objective(x + e, A, y, shape, **kw)[0] - map_objective(x - e, A, y, shape, **kw)[0]) / (2 * h)
        assert abs(fd - g[j]) < 1e-5 * max(1.0, abs(g[j]))
