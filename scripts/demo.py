"""Side-by-side demo montage: truth / bicubic / average / NC / BTV-MAP.

Simulates one scene twice with the same seed (natural 0.1 deg tremor, and
a deliberate +/-2 deg wobble at 1 Hz), registers with the gyro hybrid,
reconstructs at 2x and writes one PNG.  A last column shows BTV on the
still twin (same noise, no motion) so the effect of the prior alone is
visible.  Usage::

    python scripts/demo.py                       # docs/results/demo_montage.png
    python scripts/demo.py --scene people --variant BAA --out demo.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mlx_sr.pipeline import run_case  # noqa: E402
from mlx_sr.scene import make_scene  # noqa: E402
from mlx_sr.types import GyroConfig, MotionConfig, SensorConfig  # noqa: E402

COLUMNS = [("truth", "truth (2x grid)"), ("bicubic", "bicubic, 1 frame"), ("average", "registered average"),
           ("nc", "normalized conv."), ("btv", "BTV-MAP"), ("btv@still", "BTV, still twin")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", default="chart", help="scene kind (chart, people, text, ...)")
    ap.add_argument("--variant", default="BAB", choices=("BAB", "BAA"))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "results" / "demo_montage.png")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.simplefilter("ignore", RuntimeWarning)
    scene = make_scene(args.scene, margin_lr=6)
    sensor = SensorConfig.bab() if args.variant == "BAB" else SensorConfig.baa()
    cases = [("natural tremor 0.1 deg", MotionConfig.natural(0.1)),
             ("wobble +/-2 deg at 1 Hz", MotionConfig.wobble(2.0, 1.0))]
    results = []
    for label, motion in cases:
        res = run_case(scene, sensor, motion, GyroConfig(), 8, args.seed, reg_method="hybrid",
                       baselines=("bicubic", "average", "nc"), oracle=False, still_control=True,
                       keep_images=True)
        results.append((label, res))

    fig, axes = plt.subplots(len(results), len(COLUMNS), figsize=(2.6 * len(COLUMNS), 2.75 * len(results)),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, (label, res) in enumerate(results):
        truth = res["truth"]
        # Color range from the truth, ignoring the few hottest pixels (the small
        # hot spots) so the low-contrast structure stays visible.
        lo, hi = np.percentile(truth, [1.0, 98.0])
        hi = max(hi, lo + 1e-3)
        for j, (key, title) in enumerate(COLUMNS):
            ax = axes[i][j]
            if key == "truth":
                img, psnr = truth, None
            else:
                img = res["reconstructions"][key].image
                psnr = res["metrics"][key]["psnr"]
            ax.imshow(img, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            sub = "" if psnr is None else f"\n{psnr:.1f} dB"
            ax.set_title(title + sub, fontsize=8, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(f"{args.variant}\n{label}", fontsize=8, color="#52514e")
    fig.suptitle(f"MLX90640 {args.variant}, {args.scene} scene, K = 16 subpages at 8 Hz, 2x "
                 "(PSNR against the truth)", fontsize=9, color="#0b0b0b")
    fig.tight_layout(h_pad=1.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for label, res in results:
        m = res["metrics"]
        print(f"  {label}: " + ", ".join(f"{k} {m[k]['psnr']:.2f} dB" for k, _ in COLUMNS[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
