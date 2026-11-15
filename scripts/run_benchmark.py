"""Benchmark sweeps for the MLX90640 tremor super-resolution prototype.

Writes ``results.csv`` (one row per case), ``results.md`` (mean +/- std
tables) and PNG plots to ``docs/results/`` (or ``--out``).

Usage::

    python scripts/run_benchmark.py            # full sweep (all cores)
    python scripts/run_benchmark.py --quick    # smoke run, well under 2 minutes
    python scripts/run_benchmark.py --jobs 4 --seeds 3 --out /tmp/res

Sweeps (see ``build_cases``):

``central``
    Scenes chart / people / textured, BAB and BAA, motion: natural tremor
    RMS 0.05, 0.1, 0.3, 1 and 3 deg (each with the default 0.2 deg drift),
    tremor 0.1 deg without drift, deliberate circular wobble +/-2 deg at
    1 Hz and 2 Hz, and a slow wobble +/-1 deg at 0.5 Hz.  K = 16 subpages
    at 8 Hz (2 s), PSF 0.4 px, 2x, gyro-hybrid registration, plus an
    image-only ("gn") registration of the same capture, oracle (true-shift)
    reconstructions and a still-twin control.
``k``
    K in {4, 8, 16, 32} subpages at 8 Hz (chart, BAB/BAA, tremor 0.3 deg
    and wobble +/-2 deg at 1 Hz).
``refresh``
    Subpage rate in {4, 8, 16, 32} Hz at a fixed 2 s capture (K = 8 ... 64).
``psf``
    True PSF sigma in {0.2, 0.4, 0.6, 0.8} px while the reconstruction keeps
    assuming 0.4 px (the MLX90640 PSF is unpublished, so a real system has
    to guess).
``scale``
    2x versus 3x output.
``reg``
    Registration RMS of gyro / phase / gn / hybrid versus scene contrast
    (textured scene, 0.3-10 K RMS).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

MARGIN_LR = 6  # LR px of scene margin: keeps 3 deg tremor on BAB inside the canvas

# name -> (label, MotionConfig kwargs).  "natural" = tremor + default 0.2 deg drift.
MOTIONS: dict[str, tuple[str, dict]] = {
    "t0.05": ("tremor 0.05 deg", dict(tremor_rms_deg=0.05)),
    "t0.1": ("tremor 0.1 deg", dict(tremor_rms_deg=0.1)),
    "t0.1_nodrift": ("tremor 0.1 deg, no drift", dict(tremor_rms_deg=0.1, drift_rms_deg=0.0)),
    "t0.3": ("tremor 0.3 deg", dict(tremor_rms_deg=0.3)),
    "t1": ("tremor 1 deg", dict(tremor_rms_deg=1.0)),
    "t3": ("tremor 3 deg", dict(tremor_rms_deg=3.0)),
    "wob2_1hz": ("wobble +/-2 deg 1 Hz", dict(tremor_rms_deg=0.1, wobble_amp_deg=2.0, wobble_hz=1.0)),
    "wob2_2hz": ("wobble +/-2 deg 2 Hz", dict(tremor_rms_deg=0.1, wobble_amp_deg=2.0, wobble_hz=2.0)),
    "wob1_0.5hz": ("wobble +/-1 deg 0.5 Hz", dict(tremor_rms_deg=0.1, wobble_amp_deg=1.0, wobble_hz=0.5)),
}
CENTRAL_MOTIONS = list(MOTIONS)
TREMOR_ORDER = ["t0.05", "t0.1", "t0.3", "t1", "t3"]
WOBBLE_ORDER = ["wob1_0.5hz", "wob2_1hz", "wob2_2hz"]

BASELINES = ("bicubic", "bicubic_single", "average", "nc", "ibp")
ORACLE = ("btv", "average", "ibp")
STILL = ("btv", "ibp", "average")


def build_cases(quick: bool, seeds: int | None) -> list[dict]:
    """Every case as a plain dict (picklable for the worker processes)."""
    n_seed = seeds if seeds is not None else (1 if quick else 5)
    n_seed_side = seeds if seeds is not None else (1 if quick else 5)
    cases: list[dict] = []

    def add(sweep, n, **kw):
        base = dict(sweep=sweep, scene="chart", variant="BAB", motion="t0.3", refresh_hz=8.0,
                    n_frames=8, psf_sigma_px=0.4, psf_assumed=None, scale=2, contrast_k=None,
                    image_only=False)
        base.update(kw)
        for seed in range(n):
            cases.append(dict(base, seed=1000 + seed))

    variants = ("BAB", "BAA")
    if quick:
        for v in variants:
            for m in ("t0.1", "t1", "wob2_1hz"):
                add("central", n_seed, variant=v, motion=m, image_only=True)
        add("central", n_seed, scene="people", variant="BAB", motion="wob2_1hz", image_only=True)
        for k in (8, 16):
            add("k", n_seed_side, variant="BAB", motion="wob2_1hz", n_frames=k // 2)
        for hz in (8.0, 16.0):
            add("refresh", n_seed_side, variant="BAB", motion="wob2_1hz", refresh_hz=hz,
                n_frames=int(hz))
        for ps in (0.4, 0.8):
            add("psf", n_seed_side, variant="BAB", motion="wob2_1hz", psf_sigma_px=ps, psf_assumed=0.4)
        for c in (1.0, 10.0):
            add("reg", n_seed_side, scene="textured", variant="BAB", motion="wob2_1hz", contrast_k=c)
        return cases

    for scene in ("chart", "people", "textured"):
        for v in variants:
            for m in CENTRAL_MOTIONS:
                add("central", n_seed, scene=scene, variant=v, motion=m, image_only=True)
    for v in variants:
        for m in ("t0.3", "wob2_1hz"):
            for k in (4, 8, 16, 32):
                add("k", n_seed_side, variant=v, motion=m, n_frames=k // 2)
            for hz in (4.0, 8.0, 16.0, 32.0):
                add("refresh", n_seed_side, variant=v, motion=m, refresh_hz=hz, n_frames=int(hz))
            for ps in (0.2, 0.4, 0.6, 0.8):
                add("psf", n_seed_side, variant=v, motion=m, psf_sigma_px=ps, psf_assumed=0.4)
            for sc in (2, 3):
                add("scale", n_seed_side, variant=v, motion=m, scale=sc)
    for m in ("t0.3", "wob2_1hz"):
        for c in (0.3, 1.0, 3.0, 10.0):
            add("reg", n_seed, scene="textured", variant="BAB", motion=m, contrast_k=c)
    return cases


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _scene(kind: str, contrast_k: float | None):
    from mlx_sr.scene import make_scene

    kw = dict(margin_lr=MARGIN_LR)
    if contrast_k is not None:
        kw["contrast_k"] = contrast_k
    return make_scene(kind, **kw)


def _sensor(case: dict):
    from mlx_sr.types import SensorConfig

    make = SensorConfig.bab if case["variant"] == "BAB" else SensorConfig.baa
    return make(refresh_hz=case["refresh_hz"], psf_sigma_px=case["psf_sigma_px"])


def run_one(case: dict) -> dict:
    """Run one case and return one flat CSV row."""
    from mlx_sr import metrics as M
    from mlx_sr.pipeline import flatten_result, run_case, run_pipeline
    from mlx_sr.registration import register
    from mlx_sr.sensor import simulate_capture
    from mlx_sr.types import GyroConfig, MotionConfig

    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.perf_counter()
    scene = _scene(case["scene"], case["contrast_k"])
    sensor = _sensor(case)
    motion = MotionConfig(**MOTIONS[case["motion"]][1])
    row = {k: v for k, v in case.items()}
    row["motion_label"] = MOTIONS[case["motion"]][0]
    row["K"] = 2 * case["n_frames"]

    if case["sweep"] == "reg":
        cap = simulate_capture(scene, sensor, motion, GyroConfig(), case["n_frames"], case["seed"])
        true_rel = cap.relative_true_shifts(0)
        for meth in ("gyro", "phase", "gn", "hybrid"):
            ts = time.perf_counter()
            reg = register(cap, meth, ref_index=0)
            row[f"reg_{meth}.rms_px"] = M.registration_rms(reg.shifts, true_rel, remove="ref")
            row[f"reg_{meth}.time_s"] = time.perf_counter() - ts
        row["reg_hybrid.n_fallback"] = int(np.sum(reg.info.get("fallback", 0)))
        row["shift_spread_px"] = float(np.sqrt(np.mean(np.sum((true_rel - true_rel.mean(0)) ** 2, 1))))
        row["time.total_s"] = time.perf_counter() - t0
        return row

    recon_kw = {}
    if case["psf_assumed"] is not None:
        recon_kw = {m: {"psf_sigma": case["psf_assumed"]} for m in ("btv", "ibp")}
    central = case["sweep"] == "central"
    res = run_case(scene, sensor, motion, GyroConfig(), case["n_frames"], case["seed"],
                   reg_method="hybrid", method="btv", scale=case["scale"],
                   baselines=BASELINES if central else ("bicubic", "average"),
                   oracle=ORACLE if central else ("btv",),
                   still_control=STILL if central else ("btv",),
                   recon_kw=recon_kw, keep_curves=central)
    row.update(flatten_result(res))
    if central:
        for name in ("bicubic", "average", "btv", "btv@still"):
            curve = res["metrics"][name].get("mtf_curve")
            if curve is not None:
                f = np.linspace(0.0, 1.0, 41)
                row.update({f"curve.{name}.{x:.3f}": float(np.interp(x, *curve)) for x in f})
    if case["image_only"]:
        cap = res["capture"]
        ts = time.perf_counter()
        reg = register(cap, "gn", ref_index=0)
        row["time.register_gn_s"] = time.perf_counter() - ts
        alt = run_pipeline(cap, reg, "btv", case["scale"], ref_index=0, baselines=(), oracle=False,
                           recon_kw=recon_kw)
        for k, v in alt["metrics"]["btv"].items():
            if k != "mtf_curve":
                row[f"btv@gn.{k}"] = float(v) if isinstance(v, (bool, np.bool_)) else v
        row["reg_gn.rms_px"] = alt["registration"]["rms_px"]
        row["gn_minus_hybrid_db"] = row["btv@gn.psnr"] - row["btv.psnr"]
    row["btv_minus_average_db"] = row["btv.psnr"] - row["average.psnr"]
    row["time.total_s"] = time.perf_counter() - t0
    return row


def _init_worker() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


# ---------------------------------------------------------------------------
# Aggregation and tables
# ---------------------------------------------------------------------------


def _fmt(mu: float, sd: float, nd: int = 2) -> str:
    if not np.isfinite(mu):
        return "n/a"
    if not np.isfinite(sd):
        return f"{mu:.{nd}f}"
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"


def _stat(df, col):
    if col not in df.columns:
        return float("nan"), float("nan")
    v = df[col].astype(float).to_numpy()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")


def _cell(df, col, nd=2):
    return _fmt(*_stat(df, col), nd=nd)


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


class Frame:
    """Tiny column store (avoids a pandas dependency)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        cols: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    cols.append(k)
        self.columns = cols

    def __getitem__(self, col):
        return _Col([r.get(col, np.nan) for r in self.rows])

    def where(self, **eq) -> "Frame":
        return Frame([r for r in self.rows if all(r.get(k) == v for k, v in eq.items())])

    def unique(self, col) -> list:
        out = []
        for r in self.rows:
            v = r.get(col)
            if v not in out:
                out.append(v)
        return out

    def __len__(self):
        return len(self.rows)


class _Col(list):
    def astype(self, t):
        return _Col([t(x) if x is not None and x != "" else float("nan") for x in self])

    def to_numpy(self):
        return np.asarray(self, dtype=float)


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    fr = Frame(rows)
    cols = [c for c in fr.columns if not c.startswith("curve.")]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()
                        if not k.startswith("curve.")})


SUMMARY_MOTIONS = ["t0.1", "t0.1_nodrift", "t0.3", "t1", "wob1_0.5hz", "wob2_1hz", "wob2_2hz"]


def summary_table(fr: Frame) -> list[str]:
    """Headline table: every scene and variant, the key motion cases."""
    c = fr.where(sweep="central")
    if not len(c):
        return []
    out = ["## Summary\n",
           "Key columns of the central sweep (K = 16 at 8 Hz, 2x, hybrid registration).  "
           "`BTV - avg` = PSNR(BTV) - PSNR(registered average); `motion dB` = PSNR(BTV) - "
           "PSNR(BTV still), the part of the BTV result the motion itself buys; `oracle gap` = "
           "PSNR(BTV, true shifts) - PSNR(BTV); `gn - hybrid` = PSNR(BTV, image-only registration) "
           "- PSNR(BTV, hybrid); `HF` = high-band fidelity (n/a: the scene has no detail above "
           "native Nyquist); `gain` = MTF50(BTV) / MTF50(bicubic) (chart only).\n"]
    rows = []
    for scene in c.unique("scene"):
        for v in c.where(scene=scene).unique("variant"):
            for m in SUMMARY_MOTIONS:
                d = c.where(scene=scene, variant=v, motion=m)
                if not len(d):
                    continue
                rows.append([scene, v, MOTIONS[m][0], _cell(d, "shift_spread_px"), _cell(d, "intra_blur_px"),
                             _cell(d, "bicubic.psnr"), _cell(d, "average.psnr"), _cell(d, "btv@still.psnr"),
                             _cell(d, "btv.psnr"), _cell(d, "btv_minus_average_db"), _cell(d, "motion_gain_db"),
                             _cell(d, "oracle_gap_db"), _cell(d, "gn_minus_hybrid_db"),
                             _cell(d, "btv@still.hf_fidelity"), _cell(d, "btv.hf_fidelity"),
                             _cell(d, "effective_gain") if scene == "chart" else "n/a"])
    out.append(_table(["scene", "variant", "motion", "spread px", "blur px", "PSNR bicubic", "PSNR average",
                       "PSNR BTV still", "PSNR BTV", "BTV - avg", "motion dB", "oracle gap", "gn - hybrid",
                       "HF BTV still", "HF BTV", "gain"], rows))
    out.append("")
    return out


def central_tables(fr: Frame) -> list[str]:
    out = []
    c = fr.where(sweep="central")
    for scene in c.unique("scene"):
        for v in c.where(scene=scene).unique("variant"):
            sub = c.where(scene=scene, variant=v)
            n = len(sub.where(motion=sub.unique("motion")[0]))
            out.append(f"### {scene}, {v} (n = {n} seeds per row)\n")
            out.append("PSNR (dB) against the box-averaged truth on the reference grid. "
                       "`BTV still` is the paired still-twin control (same noise, no motion); "
                       "`BTV true` uses the true shifts; `BTV gn` uses image-only registration.\n")
            rows = []
            for m in [m for m in CENTRAL_MOTIONS if m in sub.unique("motion")]:
                d = sub.where(motion=m)
                rows.append([MOTIONS[m][0], _cell(d, "shift_spread_px"), _cell(d, "intra_blur_px"),
                             _cell(d, "bicubic.psnr"), _cell(d, "average.psnr"), _cell(d, "btv@still.psnr"),
                             _cell(d, "btv.psnr"), _cell(d, "btv@true.psnr"), _cell(d, "btv@gn.psnr"),
                             _cell(d, "ibp.psnr")])
            out.append(_table(["motion", "spread px", "blur px", "bicubic", "average", "BTV still",
                               "BTV", "BTV true", "BTV gn", "IBP"], rows))
            out.append("")
            out.append("Super-resolution measures.  `gain` = MTF50(BTV) / MTF50(bicubic) "
                       "(`vs avg`: / MTF50(average)); MTF50 in cycles per LR px (native Nyquist 0.5); "
                       "`MTF@0.6/0.75` of BTV; "
                       "`bar ratio` = bar limit (BTV / bicubic); `HF` = high-band fidelity "
                       "above native Nyquist (1 = exact, 0 = none, < 0 = wrong detail; n/a "
                       "when the scene has no such detail); `motion dB` = PSNR(BTV) - "
                       "PSNR(BTV still).\n")
            rows = []
            for m in [m for m in CENTRAL_MOTIONS if m in sub.unique("motion")]:
                d = sub.where(motion=m)
                rows.append([MOTIONS[m][0], _cell(d, "effective_gain"), _cell(d, "effective_gain_vs_average"),
                             _cell(d, "bicubic.mtf50"), _cell(d, "btv.mtf50"), _cell(d, "btv@still.mtf50"),
                             _cell(d, "btv.mtf_at_0.6"), _cell(d, "btv.mtf_at_0.75"), _cell(d, "bar_ratio"),
                             _cell(d, "average.hf_fidelity"), _cell(d, "btv@still.hf_fidelity"),
                             _cell(d, "btv.hf_fidelity"), _cell(d, "ibp.hf_fidelity"),
                             _cell(d, "ibp@still.hf_fidelity"), _cell(d, "motion_gain_db")])
            out.append(_table(["motion", "gain", "vs avg", "MTF50 bicubic", "MTF50 BTV", "MTF50 BTV still",
                               "MTF@0.6", "MTF@0.75",
                               "bar ratio", "HF avg", "HF BTV still", "HF BTV", "HF IBP",
                               "HF IBP still", "motion dB"], rows))
            out.append("")
            out.append("Registration RMS error (LR px, relative to the reference subpage) and the "
                       "hybrid's gyro fallbacks (of K - 1 subpages); flat-region noise sigma (K).\n")
            rows = []
            for m in [m for m in CENTRAL_MOTIONS if m in sub.unique("motion")]:
                d = sub.where(motion=m)
                rows.append([MOTIONS[m][0], _cell(d, "reg.rms_px", 3), _cell(d, "reg_gn.rms_px", 3),
                             _cell(d, "reg.n_fallback", 1), _cell(d, "oracle_gap_db"),
                             _cell(d, "bicubic.flat_sigma_k", 3), _cell(d, "average.flat_sigma_k", 3),
                             _cell(d, "btv.flat_sigma_k", 3)])
            out.append(_table(["motion", "hybrid RMS", "gn RMS", "fallbacks", "oracle gap dB",
                               "flat σ bicubic", "flat σ average", "flat σ BTV"], rows))
            out.append("")
    return out


def side_tables(fr: Frame) -> list[str]:
    out = []
    specs = [
        ("k", "K", "K (subpages)", "Frame count K at 8 Hz (duration K / 8 s)."),
        ("refresh", "refresh_hz", "subpage rate Hz", "Subpage rate at a fixed 2 s capture (K = 2 x rate)."),
        ("psf", "psf_sigma_px", "true PSF σ px", "True PSF sigma; the reconstruction always assumes 0.4 px."),
        ("scale", "scale", "scale", "Output scale (3x uses lam / 2.25)."),
    ]
    for sweep, col, label, desc in specs:
        s = fr.where(sweep=sweep)
        if not len(s):
            continue
        out.append(f"### Sweep: {sweep}\n")
        out.append(desc + " Chart scene, hybrid registration.  `HF` = high-band fidelity of BTV, "
                   "`motion dB` = PSNR(BTV) - PSNR(BTV still).\n")
        rows = []
        for v in s.unique("variant"):
            for m in s.where(variant=v).unique("motion"):
                for x in sorted(s.where(variant=v, motion=m).unique(col)):
                    d = s.where(variant=v, motion=m, **{col: x})
                    rows.append([v, MOTIONS[m][0], f"{x:g}", str(len(d)), _cell(d, "bicubic.psnr"),
                                 _cell(d, "average.psnr"), _cell(d, "btv@still.psnr"), _cell(d, "btv.psnr"),
                                 _cell(d, "btv@true.psnr"), _cell(d, "motion_gain_db"),
                                 _cell(d, "effective_gain"), _cell(d, "btv.hf_fidelity"),
                                 _cell(d, "reg.rms_px", 3), _cell(d, "intra_blur_px"),
                                 _cell(d, "time.btv", 2)])
        out.append(_table(["variant", "motion", label, "n", "PSNR bicubic", "PSNR average", "PSNR BTV still",
                           "PSNR BTV", "PSNR BTV true", "motion dB", "gain", "HF", "reg RMS px",
                           "blur px", "BTV s"], rows))
        out.append("")
    s = fr.where(sweep="reg")
    if len(s):
        out.append("### Sweep: registration versus scene contrast\n")
        out.append("Textured scene (RMS contrast in K), BAB, 8 Hz, K = 16.  RMS error in LR px "
                   "relative to the reference subpage.\n")
        rows = []
        for m in s.unique("motion"):
            for c in sorted(s.where(motion=m).unique("contrast_k")):
                d = s.where(motion=m, contrast_k=c)
                rows.append([MOTIONS[m][0], f"{c:g}", str(len(d)), _cell(d, "shift_spread_px", 3),
                             _cell(d, "reg_gyro.rms_px", 3), _cell(d, "reg_phase.rms_px", 3),
                             _cell(d, "reg_gn.rms_px", 3), _cell(d, "reg_hybrid.rms_px", 3),
                             _cell(d, "reg_hybrid.n_fallback", 1)])
        out.append(_table(["motion", "contrast K", "n", "spread px", "gyro", "phase", "gn", "hybrid",
                           "hybrid fallbacks"], rows))
        out.append("")
    return out


def write_markdown(fr: Frame, path: Path, meta: dict) -> None:
    lines = [
        "# Benchmark results",
        "",
        "Generated by `scripts/run_benchmark.py`"
        + (" with `--quick` (smoke run: 1 seed, reduced grid; not for conclusions)" if meta["quick"] else "")
        + f".  {meta['n_cases']} cases, {meta['runtime_s']:.0f} s wall time on {meta['jobs']} worker processes.",
        "All values are mean ± sample std over seeds.  Simulation only (see the README for the model "
        "and its limitations).",
        "",
        "Common settings unless a sweep varies them: K = 16 subpages at 8 Hz (2 s), PSF σ 0.4 px, "
        "NETD per datasheet scaled by √rate (BAB 0.71 K, BAA 0.40 K per subpage), FPN 0.1 K, "
        "2x output, simulated 400 Hz gyro (0.007 °/s/√Hz, random 0.05-1 °/s bias, 10 ms offset), "
        "hybrid registration.  'tremor X deg' includes the default 0.2° RMS drift; wobble cases add "
        "a circular sweep on top of 0.1° tremor and the drift.",
        "",
        "Column guide: `spread px` = RMS sub-pixel diversity of the true subpage shifts; `blur px` = "
        "RMS motion within one exposure (not modeled by the reconstruction); `bicubic` = one frame "
        "merged Adafruit-style; `average` = registered multi-frame average at the native grid "
        "(denoising only); `BTV still` = the same BTV on a still twin of the capture (same noise and "
        "fixed pattern, no motion: shows what the prior and deconvolution give without any "
        "super-resolution); `BTV true` = BTV on the true shifts (oracle registration).",
        "",
    ]
    lines += summary_table(fr)
    if any(r.get("sweep") == "central" for r in fr.rows):
        lines += ["## Central sweep: tremor amplitude and deliberate wobble", ""]
        lines += central_tables(fr)
    lines += ["## Side sweeps", ""]
    lines += side_tables(fr)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Reference categorical palette (light mode), fixed slot per method.
COLORS = {
    "btv": "#2a78d6", "average": "#eb6834", "bicubic": "#1baf7a", "btv@still": "#eda100",
    "ibp": "#e87ba4", "nc": "#008300", "btv@true": "#6250d6", "btv@gn": "#e34948",
    "gyro": "#1baf7a", "phase": "#eda100", "gn": "#eb6834", "hybrid": "#2a78d6",
}
LABELS = {"btv": "BTV-MAP", "average": "registered average", "bicubic": "bicubic (1 frame)",
          "btv@still": "BTV on still twin", "btv@true": "BTV, true shifts", "btv@gn": "BTV, image-only reg.",
          "ibp": "IBP", "nc": "normalized conv."}
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })
    return plt


def _series(ax, x, df_by_x, col, key, ls="-", marker="o"):
    mu = np.array([_stat(d, col)[0] for d in df_by_x])
    sd = np.array([_stat(d, col)[1] for d in df_by_x])
    sd = np.where(np.isfinite(sd), sd, 0.0)
    ax.errorbar(x, mu, yerr=sd, color=COLORS[key], ls=ls, marker=marker, capsize=2, elinewidth=1,
                label=LABELS.get(key, key), markeredgecolor=SURFACE, markeredgewidth=1.5)


def plot_psnr_vs_motion(fr: Frame, out: Path, scene: str = "chart") -> Path | None:
    c = fr.where(sweep="central", scene=scene)
    if not len(c):
        return None
    plt = _style()
    variants = c.unique("variant")
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 3.8), squeeze=False,
                             sharey=True)
    order = [m for m in TREMOR_ORDER + WOBBLE_ORDER if m in c.unique("motion")]
    for ax, v in zip(axes[0], variants):
        d = c.where(variant=v)
        x = np.arange(len(order))
        groups = [d.where(motion=m) for m in order]
        for key in ("bicubic", "average", "btv@still", "btv", "btv@true"):
            _series(ax, x, groups, f"{key}.psnr", key, ls="--" if key == "btv@true" else "-")
        n_t = sum(m in TREMOR_ORDER for m in order)
        if 0 < n_t < len(order):
            ax.axvline(n_t - 0.5, color=TEXT2, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([MOTIONS[m][0].replace("wobble ", "wobble\n").replace("tremor ", "tremor\n")
                            for m in order], fontsize=7)
        ax.set_title(f"{scene} scene, MLX90640 {v}")
        ax.set_ylabel("PSNR vs truth (dB)")
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    p = out / f"psnr_vs_motion_{scene}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_mtf_curves(fr: Frame, out: Path) -> Path | None:
    c = fr.where(sweep="central", scene="chart")
    if not len(c):
        return None
    plt = _style()
    motions = [m for m in ("t0.1", "wob2_1hz") if m in c.unique("motion")]
    variants = c.unique("variant")
    fig, axes = plt.subplots(len(motions), len(variants), figsize=(5.0 * len(variants), 3.3 * len(motions)),
                             squeeze=False, sharex=True, sharey=True)
    f = np.linspace(0.0, 1.0, 41)
    for i, m in enumerate(motions):
        for j, v in enumerate(variants):
            ax = axes[i][j]
            d = c.where(variant=v, motion=m)
            for key in ("bicubic", "average", "btv@still", "btv"):
                cols = [f"curve.{key}.{x:.3f}" for x in f]
                arr = np.array([[r.get(k, np.nan) for k in cols] for r in d.rows], float)
                if not np.isfinite(arr).any():
                    continue
                ax.plot(f, np.nanmean(arr, axis=0), color=COLORS[key], label=LABELS[key])
            ax.axvline(0.5, color=TEXT2, lw=0.8, ls=":")
            ax.text(0.51, 1.02, "native Nyquist", fontsize=7, color=TEXT2, va="bottom")
            ax.set_title(f"{v}, {MOTIONS[m][0]}")
            ax.set_ylim(0, 1.25)
            if i == len(motions) - 1:
                ax.set_xlabel("spatial frequency (cycles per LR pixel)")
            if j == 0:
                ax.set_ylabel("slanted-edge MTF")
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    p = out / "mtf_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_side(fr: Frame, out: Path, sweep: str, col: str, xlabel: str, logx: bool = False) -> Path | None:
    s = fr.where(sweep=sweep)
    if not len(s):
        return None
    plt = _style()
    variants = s.unique("variant")
    motions = s.unique("motion")
    fig, axes = plt.subplots(len(motions), len(variants), figsize=(4.6 * len(variants), 3.2 * len(motions)),
                             squeeze=False, sharex=True)
    for i, m in enumerate(motions):
        for j, v in enumerate(variants):
            ax = axes[i][j]
            d = s.where(variant=v, motion=m)
            xs = sorted(d.unique(col))
            groups = [d.where(**{col: x}) for x in xs]
            for key in ("bicubic", "average", "btv@still", "btv"):
                _series(ax, np.asarray(xs, float), groups, f"{key}.psnr", key)
            if logx:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_title(f"{v}, {MOTIONS[m][0]}")
            if i == len(motions) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel("PSNR vs truth (dB)")
    axes[0][0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    p = out / f"psnr_vs_{sweep}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_gain_vs_k(fr: Frame, out: Path) -> Path | None:
    s = fr.where(sweep="k")
    if not len(s):
        return None
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    styles = {"BAB": "-", "BAA": "--"}
    pal = ["#2a78d6", "#eb6834"]
    for v in s.unique("variant"):
        for mi, m in enumerate(s.where(variant=v).unique("motion")):
            d = s.where(variant=v, motion=m)
            ks = sorted(d.unique("K"))
            mu = [_stat(d.where(K=k), "motion_gain_db")[0] for k in ks]
            sd = [_stat(d.where(K=k), "motion_gain_db")[1] for k in ks]
            sd = np.where(np.isfinite(sd), sd, 0.0)
            ax.errorbar(ks, mu, yerr=sd, color=pal[mi % 2], ls=styles.get(v, "-"), marker="o", capsize=2,
                        elinewidth=1, label=f"{v}, {MOTIONS[m][0]}", markeredgecolor=SURFACE,
                        markeredgewidth=1.5)
    ax.axhline(0.0, color=TEXT2, lw=0.8)
    ax.set_xscale("log", base=2)
    ks = sorted(s.unique("K"))
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("subpages K (8 Hz)")
    ax.set_ylabel("PSNR(BTV) - PSNR(BTV still) (dB)")
    ax.set_title("Gain the motion buys versus frame count")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "gain_vs_k.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_reg_vs_contrast(fr: Frame, out: Path) -> Path | None:
    s = fr.where(sweep="reg")
    if not len(s):
        return None
    plt = _style()
    motions = s.unique("motion")
    fig, axes = plt.subplots(1, len(motions), figsize=(4.8 * len(motions), 3.6), squeeze=False, sharey=True)
    for ax, m in zip(axes[0], motions):
        d = s.where(motion=m)
        cs = sorted(d.unique("contrast_k"))
        groups = [d.where(contrast_k=c) for c in cs]
        for meth in ("gyro", "phase", "gn", "hybrid"):
            mu = np.array([_stat(g, f"reg_{meth}.rms_px")[0] for g in groups])
            sd = np.array([_stat(g, f"reg_{meth}.rms_px")[1] for g in groups])
            sd = np.where(np.isfinite(sd), sd, 0.0)
            ax.errorbar(cs, mu, yerr=np.minimum(sd, mu * 0.95), color=COLORS[meth], marker="o", capsize=2,
                        elinewidth=1, label={"gyro": "gyro only", "phase": "phase correlation",
                                             "gn": "Gauss-Newton (image)", "hybrid": "gyro + image hybrid"}[meth],
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(cs)
        ax.set_xticklabels([f"{c:g}" for c in cs])
        ax.set_xlabel("scene RMS contrast (K)")
        ax.set_title(f"BAB, {MOTIONS[m][0]}")
    axes[0][0].set_ylabel("registration RMS error (LR px)")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    p = out / "reg_rms_vs_contrast.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="smoke run (1 seed, reduced grid)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results", help="output directory")
    ap.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: CPU count - 1)")
    ap.add_argument("--sweeps", default=None, help="comma-separated subset: central,k,refresh,psf,scale,reg")
    args = ap.parse_args(argv)

    cases = build_cases(args.quick, args.seeds)
    if args.sweeps:
        keep = set(args.sweeps.split(","))
        cases = [c for c in cases if c["sweep"] in keep]
    jobs = args.jobs or max(1, min(len(cases), (os.cpu_count() or 2) - 1))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases on {jobs} worker(s) -> {args.out}", flush=True)

    t0 = time.perf_counter()
    rows: list[dict] = []
    if jobs == 1:
        _init_worker()
        for i, c in enumerate(cases):
            rows.append(run_one(c))
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
            for i, r in enumerate(ex.map(run_one, cases, chunksize=1)):
                rows.append(r)
                if (i + 1) % max(1, len(cases) // 10) == 0:
                    print(f"  {i + 1}/{len(cases)} cases, {time.perf_counter() - t0:.0f} s", flush=True)
    runtime = time.perf_counter() - t0

    fr = Frame(rows)
    write_csv(rows, args.out / "results.csv")
    write_markdown(fr, args.out / "results.md",
                   dict(quick=args.quick, n_cases=len(cases), runtime_s=runtime, jobs=jobs))
    plots = [
        plot_psnr_vs_motion(fr, args.out, "chart"),
        plot_psnr_vs_motion(fr, args.out, "people"),
        plot_psnr_vs_motion(fr, args.out, "textured"),
        plot_mtf_curves(fr, args.out),
        plot_gain_vs_k(fr, args.out),
        plot_side(fr, args.out, "refresh", "refresh_hz", "subpage rate (Hz), 2 s capture", logx=True),
        plot_side(fr, args.out, "psf", "psf_sigma_px", "true PSF σ (LR px); reconstruction assumes 0.4"),
        plot_reg_vs_contrast(fr, args.out),
    ]
    written = [p.name for p in plots if p is not None]
    print(f"done in {runtime:.0f} s (cases) / {time.perf_counter() - t0:.0f} s (total); wrote results.csv, "
          f"results.md, {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
