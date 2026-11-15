"""Pipeline tests P1-P3 (plan section 5) plus the high-band fidelity metric."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx_sr.metrics import highband_fidelity
from mlx_sr.pipeline import flatten_result, infer_ref_index, run_case, run_pipeline
from mlx_sr.registration import register
from mlx_sr.reconstruct import upsample_bicubic
from mlx_sr.scene import make_scene
from mlx_sr.sensor import simulate_capture, truth_at_scale
from mlx_sr.types import GyroConfig, MotionConfig, SensorConfig

ROOT = Path(__file__).resolve().parents[1]

METRIC_KEYS = {"finite", "psnr", "ssim", "rmse_k", "hf_fidelity", "mtf50", "mtf_valid", "mtf_at_0.6",
               "mtf_at_0.75", "bar_limit_freq", "flat_sigma_k", "runtime_s"}
SUMMARY_KEYS = {"effective_gain", "effective_gain_vs_average", "bar_ratio", "psnr_gain_db",
                "psnr_gain_vs_average_db", "hf_gain", "oracle_gap_db"}


@pytest.fixture(scope="module")
def chart():
    return make_scene("chart", margin_lr=6)


def test_p1_tiny_tremor_report_is_complete(chart):
    """P1: tiny tremor (0.05 deg) runs end to end and the report has every
    field.  No gain is asserted in either direction (the benchmark reports
    the outcome)."""
    res = run_case(chart, SensorConfig.bab(), MotionConfig.natural(0.05), GyroConfig(), 8, seed=3,
                   baselines=("bicubic", "bicubic_single", "average", "nc", "ibp"),
                   still_control=True)
    expected = {"bicubic", "bicubic_single", "average", "nc", "ibp", "btv", "btv@true",
                "average@true", "btv@still"}
    assert set(res["metrics"]) == expected
    for name, m in res["metrics"].items():
        assert METRIC_KEYS <= set(m), name
        assert m["finite"], name
        assert math.isfinite(m["psnr"]) and math.isfinite(m["ssim"]), name
        assert math.isfinite(m["hf_fidelity"]), name  # the chart has detail above Nyquist
        assert m["mtf_valid"], name
    for k in SUMMARY_KEYS | {"motion_gain_db", "motion_mtf_ratio", "motion_hf_gain"}:
        assert k in res
    assert all(math.isfinite(res[k]) for k in ("psnr_gain_db", "hf_gain", "oracle_gap_db",
                                                "motion_gain_db"))
    assert res["reconstruction"].image.shape == (48, 64)
    assert res["shifts"].shape == (16, 2) and res["true_shifts"].shape == (16, 2)
    assert res["ref_index"] == 0 and res["reg_method"] == "hybrid"
    reg = res["registration"]
    assert reg["method"] == "hybrid" and math.isfinite(reg["rms_px"])
    assert res["shift_spread_px"] >= 0 and res["intra_blur_px"] >= 0
    assert {"register_s", "simulate_s", "btv", "bicubic"} <= set(res["timings"])
    row = flatten_result(res)
    assert row["btv.psnr"] == res["metrics"]["btv"]["psnr"]
    assert all(np.isscalar(v) for v in row.values())


def test_p2_deliberate_wobble_beats_bicubic(chart):
    """P2: deliberate wobble (+/-2 deg, BAB, gyro-hybrid registration): BTV
    beats the bicubic baseline on PSNR.  This includes BTV's prior gain (it
    also beats bicubic on a still capture); the super-resolution part is
    tested by :func:`test_p2_wobble_gain_requires_motion`."""
    for seed in (11, 12):
        res = run_case(chart, SensorConfig.bab(), MotionConfig.wobble(2.0, 1.0), GyroConfig(), 8,
                       seed=seed, reg_method="hybrid", baselines=("bicubic", "average"), oracle=False)
        m = res["metrics"]
        assert m["btv"]["psnr"] > m["bicubic"]["psnr"] + 1.0, seed
        assert res["psnr_gain_db"] > 1.0
        assert res["registration"]["rms_px"] < 0.2


def test_p2_wobble_gain_requires_motion(chart):
    """P2, paired controls: BAA, +/-2 deg wobble at 1 Hz.  BTV beats its
    still twin (same noise, no motion) on PSNR and on high-band fidelity,
    and the HF gain survives against the no-spread twin (same within-
    exposure motion, no subpage-to-subpage spread), so it comes from the
    sub-pixel spread.  (Measured for seeds 11-12: +2.1 to +2.7 dB, HF
    +0.36 to +0.46.)"""
    for seed in (11, 12):
        res = run_case(chart, SensorConfig.baa(), MotionConfig.wobble(2.0, 1.0), GyroConfig(), 8,
                       seed=seed, baselines=("bicubic",), oracle=False, still_control=True,
                       nospread_control=True)
        assert res["motion_gain_db"] > 1.0, (seed, res["motion_gain_db"])
        assert res["motion_hf_gain"] > 0.15, (seed, res["motion_hf_gain"])
        assert res["spread_hf_gain"] > 0.15, (seed, res["spread_hf_gain"])
        assert res["nospread_residual_px"] < 0.01


def test_nospread_twin_keeps_within_exposure_motion(chart):
    """The no-spread trace keeps the motion inside every exposure (same
    blur) and removes the subpage-to-subpage motion."""
    from mlx_sr.pipeline import motion_diagnostics, nospread_trace

    for exposure in ("full", "staggered"):
        sensor = SensorConfig.bab(exposure=exposure)
        cap = simulate_capture(chart, sensor, MotionConfig.wobble(2.0, 1.0), None, 4, seed=4)
        twin = simulate_capture(chart, sensor, nospread_trace(cap), None, 4, seed=4)
        d, d0 = motion_diagnostics(cap), motion_diagnostics(twin)
        assert d["shift_spread_px"] > 0.5
        assert d0["shift_spread_px"] < 0.01
        assert d0["intra_blur_px"] == pytest.approx(d["intra_blur_px"], rel=0.02)
        np.testing.assert_array_equal(twin.fpn, cap.fpn)


def test_pipeline_grids_and_oracle_consistency(chart):
    """Oracle reconstructions use the true relative shifts on the reference
    grid, so with zero noise and no motion the estimated and oracle runs
    agree and the reference shift is honored for ref_index != 0."""
    sensor = SensorConfig.bab()
    cap = simulate_capture(chart, sensor, MotionConfig.wobble(1.0, 1.0), GyroConfig(), 4, seed=5)
    r = 3
    reg = register(cap, "hybrid", ref_index=r)
    assert infer_ref_index(reg) == r
    res = run_pipeline(cap, reg, "nc", 2, baselines=(), oracle=["nc"])
    assert res["ref_index"] == r
    np.testing.assert_allclose(res["truth"], truth_at_scale(cap, 2, ref_index=r))
    np.testing.assert_allclose(res["true_shifts"][r], 0.0)
    # Registration is accurate here, so the oracle gap is small.
    assert abs(res["oracle_gap_db"]) < 1.0
    # registration=None runs the oracle only.
    res0 = run_pipeline(cap, None, "nc", 2, ref_index=r, baselines=(), oracle=["nc"])
    assert set(res0["metrics"]) == {"bicubic", "nc@true"}
    assert res0["metrics"]["nc@true"]["psnr"] == pytest.approx(res["metrics"]["nc@true"]["psnr"])


def test_still_control_shares_noise_and_has_no_motion(chart):
    res = run_case(chart, SensorConfig.baa(), MotionConfig.natural(0.3), GyroConfig(), 4, seed=8,
                   baselines=(), oracle=False, still_control=["btv", "average"])
    assert {"btv@still", "average@still"} <= set(res["metrics"])
    twin = simulate_capture(chart, SensorConfig.baa(), MotionConfig.still(), None, 4, seed=8)
    np.testing.assert_allclose(twin.true_shifts, 0.0)
    np.testing.assert_array_equal(twin.fpn, res["capture"].fpn)  # same fixed pattern
    assert res["motion_gain_db"] == pytest.approx(
        res["metrics"]["btv"]["psnr"] - res["metrics"]["btv@still"]["psnr"])


def test_highband_fidelity_reference_points(chart):
    """1 for the truth itself, about 0 for an interpolation of LR data (no
    detail above native Nyquist), negative for noise or misplaced detail."""
    cap = simulate_capture(chart, SensorConfig.bab(), None, None, 1, seed=0)
    t2 = truth_at_scale(cap, 2)
    assert highband_fidelity(t2, t2, 2, border=4) == pytest.approx(1.0)
    lr = truth_at_scale(cap, 1)
    f_interp = highband_fidelity(upsample_bicubic(lr, 2), t2, 2, border=4)
    assert -0.1 < f_interp < 0.1
    rng = np.random.default_rng(0)
    assert highband_fidelity(t2 + rng.normal(0, 3.0, t2.shape), t2, 2, border=4) < 0
    # Misplaced detail (shifted by half an LR pixel) is penalized, not rewarded.
    assert highband_fidelity(np.roll(t2, 1, axis=1), t2, 2, border=4) < 0
    with pytest.raises(ValueError):
        highband_fidelity(lr, lr, 1)
    bad = t2.copy()
    bad[10, 10] = np.nan
    assert math.isnan(highband_fidelity(bad, t2, 2, border=4))


@pytest.mark.slow
def test_p3_benchmark_quick_smoke(tmp_path):
    """P3: ``run_benchmark.py --quick`` writes results.md, results.csv and the PNGs."""
    out = tmp_path / "results"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_benchmark.py"), "--quick",
                           "--out", str(out), "--jobs", "2"],
                          capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-2000:]
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "Central sweep" in md and "BTV still" in md
    assert (out / "results.csv").stat().st_size > 0
    pngs = {p.name for p in out.glob("*.png")}
    for name in ("psnr_vs_motion_chart.png", "mtf_curves.png", "gain_vs_k.png", "reg_rms_vs_contrast.png"):
        assert name in pngs
