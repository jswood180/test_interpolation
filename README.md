# test_interpolation: tremor-driven super-resolution for the MLX90640 (simulation)

A Python simulation prototype that asks one question: **can small motion of a
handheld MLX90640 (32 x 24 thermopile array) be used for multi-frame
super-resolution, the "move the sensor while shooting" trick behind FLIR
UltraMax?** It simulates the sensor, the hand motion and a gyro, registers the
frames, reconstructs a 2x image and measures how much of any improvement is real
super-resolution rather than noise averaging or a sharpening prior.

Simulation only: no hardware I/O, no deep learning, no OpenCV. The research
behind the design, with sources, is in [docs/RESEARCH.md](docs/RESEARCH.md). All
numbers below come from [docs/results/results.md](docs/results/results.md).

![Demo montage](docs/results/demo_montage.png)

## How the technique works

1. **Sub-pixel diversity.** If the camera moves by a fraction of a pixel between
   frames, each frame samples the scene at slightly different positions. Many
   such frames together sample the scene more densely than one frame.
2. **Registration.** The shift of every frame must be known to a small fraction
   of a pixel. Here it comes from the images (upsampled phase correlation, then
   masked Gauss-Newton), from an integrated gyro, or from a hybrid of both, which
   also estimates the gyro bias and the camera-gyro time offset.
3. **Reconstruction.** A forward model `y_k = D_k B S_k x + n` (shift, optics
   blur and pixel box, then the MLX90640 chess-pattern subpage mask) is inverted
   on a 2x grid. The main method is robust MAP with a bilateral-total-variation
   prior (Farsiu et al. 2004) solved with L-BFGS-B. Baselines: single-frame
   bicubic, the registered multi-frame average (denoising only), normalized
   convolution and iterative back-projection.
4. **The catch.** The MLX90640 has huge pixels: about 0.54 px/deg (BAB, 55 deg)
   or 0.20 px/deg (BAA, 110 deg) horizontally. Physiological tremor is about
   0.1 deg at 8-12 Hz, and each subpage integrates for a full subpage period
   (125 ms at 8 Hz), so the tremor mostly averages out inside each exposure.

### What is simulated

- **Scenes** on an 8x canvas in deg C: a resolution chart (slanted edge, bar
  groups at 0.4-0.9 cycles per LR pixel, hot spots), two people with a mug, a
  smooth random texture and text.
- **Motion**: tremor (8-12 Hz) plus slow drift (default 0.2 deg RMS), optional
  deliberate circular wobble, converted to pixel shift with `f_px tan(theta)`.
- **Sensor**: exact box integration plus a Gaussian PSF (sigma 0.4 px by
  default; the real PSF is unpublished), motion blur within each exposure, chess
  subpages exposed at alternating times, NETD from the datasheet scaled by the
  square root of the refresh rate, fixed-pattern offsets, dead pixels and
  quantization.
- **Gyro**: 400 Hz, 0.007 deg/s/sqrt(Hz) white noise, a random 0.05-1 deg/s bias
  and a 10 ms camera-gyro offset (typical consumer-IMU assumptions).
- **Inverse-crime guard**: the simulator (8x canvas, integral image) and the
  reconstruction operator (separable erf weights on the 2x grid) are separate
  implementations, and the PSF sweep runs with a deliberately wrong assumed PSF.

### How the gain is measured

Every reconstruction is scored against the box-averaged ground truth on the same
grid. Besides PSNR, SSIM, slanted-edge MTF and bar-group resolvability, two
controls separate the sources of improvement:

- **Registered average**: multi-frame averaging at the native resolution. The
  step from bicubic to this baseline is pure denoising.
- **Still twin**: the same BTV reconstruction run on a twin capture with the
  same seed, noise and fixed pattern but **no motion at all**. With no motion
  there is no sub-pixel information, so whatever BTV gains here comes from its
  prior and from deconvolving the PSF. The step from the still twin to the real
  capture is the part the motion buys (`motion dB` in the tables).
- **High-band fidelity** (`HF`): `1 - |H(x - truth)|^2 / |H truth|^2` where `H`
  keeps only frequencies above the sensor's Nyquist. 1 means the detail the
  sensor grid cannot carry was recovered exactly, 0 means none was, and a
  negative value means wrong detail. Unlike MTF50 it cannot be raised by a prior
  that merely sharpens edges.

## Results and conclusions

From [docs/results/results.md](docs/results/results.md): 590 simulated cases,
5 seeds each, K = 16 subpages at 8 Hz (2 s), 2x output, gyro-hybrid
registration unless stated otherwise. Values are mean ± std over seeds.

Chart scene (the only scene with real detail above the sensor's Nyquist that
all metrics can see):

| variant | motion | spread px | blur px | PSNR average | PSNR BTV still | PSNR BTV | motion dB | HF BTV still | HF BTV |
|---|---|---|---|---|---|---|---|---|---|
| BAB | tremor 0.1 deg, no drift | 0.01 | 0.08 | 23.50 | 26.15 | 26.31 | 0.16 ± 0.19 | 0.13 | 0.09 |
| BAB | tremor 0.1 deg (+ drift) | 0.17 | 0.08 | 23.33 | 26.15 | 26.86 | 0.71 ± 0.46 | 0.13 | 0.18 |
| BAB | tremor 1 deg | 0.22 | 0.80 | 22.13 | 26.15 | 23.74 | -2.41 ± 0.38 | 0.13 | 0.02 |
| BAB | wobble ±1 deg 0.5 Hz | 0.64 | 0.11 | 23.32 | 26.15 | 28.24 | 2.09 ± 0.90 | 0.13 | 0.24 |
| BAB | wobble ±2 deg 1 Hz | 1.19 | 0.28 | 23.16 | 26.15 | 27.54 | 1.39 ± 0.42 | 0.13 | 0.12 |
| BAA | tremor 0.1 deg (+ drift) | 0.07 | 0.03 | 23.40 | 26.25 | 26.55 | 0.30 ± 0.22 | 0.14 | 0.16 |
| BAA | wobble ±2 deg 1 Hz | 0.47 | 0.11 | 23.43 | 26.25 | 28.86 | 2.61 ± 1.07 | 0.14 | 0.60 |

`spread px` is the RMS sub-pixel diversity of the true subpage shifts and
`blur px` the RMS motion within one exposure. Plots:
[PSNR vs motion](docs/results/psnr_vs_motion_chart.png),
[MTF curves](docs/results/mtf_curves.png),
[gain vs K](docs/results/gain_vs_k.png),
[PSNR vs refresh rate](docs/results/psnr_vs_refresh.png),
[PSNR vs PSF](docs/results/psnr_vs_psf.png),
[registration vs contrast](docs/results/reg_rms_vs_contrast.png).

**Does natural tremor alone help on the MLX90640? Barely, and only on
high-contrast targets.** Tremor by itself (no drift) produces 0.01 px of
sub-pixel spread, because each 125 ms exposure averages the 8-12 Hz motion; its
motion gain on the chart is 0.16 ± 0.19 dB (BAB) and 0.18 ± 0.15 dB (BAA),
within the seed scatter. The slow drift of a hand (0.2 deg RMS here, an
assumption) adds 0.17 px of spread on the BAB and gives 0.71 ± 0.46 dB on the
chart (0.30 ± 0.22 dB on the BAA). On the people and smooth-texture scenes the
same natural motion gives no gain (between -0.68 and +0.29 dB; no positive
value exceeds its seed scatter). Larger tremor hurts: at 1 deg on the BAB the motion blur
within an exposure (0.80 px) costs 2.41 dB.

**How much does deliberate wobble help? Clearly on the chart, but only if the
motion is slow relative to the exposure.** A ±2 deg circular wobble at 1 Hz on
the BAA gives 2.61 ± 1.07 dB over the still twin and raises the high-band
fidelity from 0.14 to 0.60, the clearest real super-resolution in the benchmark.
On the BAB the same wobble moves 1.19 px but blurs 0.28 px per exposure: PSNR
still gains 1.39 ± 0.42 dB, but HF stays at the still-twin level (0.12 vs
0.13). A slower ±1 deg wobble at 0.5 Hz does better on the BAB (2.09 ± 0.90 dB,
HF 0.24), and ±2 deg at 2 Hz is worse than no motion (-0.57 ± 0.42 dB, 0.53 px
blur). Raising the BAB to a 16 Hz subpage rate (same 2 s) lifts the 1 Hz wobble
to 2.41 ± 0.73 dB with HF 0.27, despite the higher per-frame noise. More frames
help too: on the BAB with 0.3 deg tremor the motion gain grows from 0.77 dB at
K = 4 to 2.06 dB at K = 32. **None of this carries over to the people or
smooth-texture scenes**, where wobble gains nothing over the still twin (people
BAB +0.04 ± 0.62 dB, BAA -1.91 ± 2.28 dB; texture BAB -0.01 ± 0.28 dB, BAA
-1.42 ± 0.32 dB).

**How much of the gain is denoising versus super-resolution? Most of the
headline gain is neither motion nor SR.** On the chart with natural tremor
(BAB):
- bicubic to registered average (denoising): PSNR 23.28 to 23.33 dB, while the
  flat-region noise falls from 0.589 to 0.254 K. Averaging cleans the noise but
  does not sharpen.
- average to BTV on the still twin (prior plus PSF deconvolution, no motion):
  23.33 to 26.15 dB.
- still twin to the real capture (what the motion buys): 26.15 to 26.86 dB.

The slanted-edge "effective gain" (MTF50 of BTV over bicubic) is 3.35 ± 0.31
for this case and 3.3-4.0 for most chart cases, but **it is not evidence of
super-resolution**: BTV on the still twin, with no sub-pixel information at all,
already reaches MTF50 = 1.03 cycles per LR pixel against 0.33 for bicubic. The
edge-preserving prior turns the chart's step edges into steps whatever the data
say. The same prior also helps the piecewise-constant people scene (still twin
34.54 dB against 28.25 dB for the average), and hurts the smooth texture, where
the registered average is the best method (27.87 against 25.81 dB for BTV on
the BAB with natural tremor). That scene has almost no detail above the sensor's
Nyquist, so there is nothing to super-resolve, and the chart-tuned
regularization is too weak for it.

**Oracle versus estimated shifts: registration is not the bottleneck.** The
hybrid registration error on the chart is 0.014-0.062 px RMS for every motion
except 3 deg tremor, and the gap to BTV on the true shifts is between -0.07 and
0.34 dB (most chart cases within 0.11 dB). Image-only Gauss-Newton is
almost as good on the chart (`gn - hybrid` between -0.88 and +0.18 dB). The gyro
matters at low contrast: on a 1 K texture the hybrid reaches 0.101 px against
0.243 px image-only, and at 0.3 K image-only registration fails (2.73 px) while
the hybrid falls back to the bias-limited gyro (0.526 px). At 3 deg of tremor on
the BAB the hybrid falls back to the gyro for 15 of 15 subpages (0.299 px).

**BAA versus BAB.** The wide BAA turns the same hand motion into less shift
(0.07 vs 0.17 px of spread for natural motion), so natural motion helps it
less (0.30 vs 0.71 dB). The same property means less motion blur, and with
deliberate wobble the BAA is the clear winner (HF 0.60 vs 0.12 at ±2 deg 1 Hz).
It also has the lower noise (0.40 vs 0.71 K per subpage at 8 Hz). Its lens
distortion is not modeled (the shift grows toward the corners), so BAA results
are optimistic.

**Other findings.**
- **The PSF assumption is critical.** Assuming 0.4 px when the true PSF is
  0.2 px makes BTV worse than bicubic even without motion (21.99 against
  24.35 dB on the BAB); with a wider true PSF (0.6-0.8 px) the motion gain
  shrinks to 0.08-0.90 dB.
- **3x output adds nothing over 2x**: the high-band fidelity at 3x is lower
  (0.52 vs 0.60 for the BAA wobble).
- **Expected gain**: the research estimate of 1.3-1.8x effective linear
  resolution (docs/RESEARCH.md, unverified) cannot be confirmed or refuted from
  MTF50 here because of the prior effect above; the high-band fidelity says that
  real recovery of sub-Nyquist detail happens only with deliberate slow wobble,
  mostly on the BAA, and only on scenes that contain such detail.

## Limitations

- **Simulation only.** Nothing here has been validated on hardware; every
  conclusion depends on the sensor and motion models.
- **Translation-only model.** Reconstruction assumes a uniform shift per
  subpage. Roll and the BAA lens distortion (shift growing toward the corners)
  are not modeled, and motion blur within an exposure is simulated but not
  modeled by the reconstruction.
- **Unpublished PSF.** The 0.4 px Gaussian PSF is an assumption; the PSF sweep
  shows how sensitive the results are to it.
- **sqrt(rate) noise scaling.** Per-subpage noise is the datasheet 1 Hz NETD
  times the square root of the refresh rate, which is an assumption, as is the
  flat noise across the array.
- **Synthetic scenes and tuning.** The BTV parameters were tuned on the chart.
  Piecewise-constant synthetic scenes flatter edge-preserving priors; a stronger
  regularization would likely suit smooth scenes better (not benchmarked).
- **Assumed gyro and drift parameters** (noise, bias, time offset, 0.2 deg
  drift) are typical values, not measurements.
- **Static scene.** Moving objects are not modeled.
- **Image borders.** The outermost LR pixel of the reconstruction can show
  artifacts where shifted samples do not cover the grid; the metrics crop a
  2 LR px border.

## Next steps for hardware

1. **Add a gyro.** A 6-axis MEMS IMU with a gyro data rate of at least 400 Hz,
   a FIFO and a data-ready interrupt (for example the TDK ICM-42688-P, ST
   LSM6DSO or Bosch BMI270 class). Check its noise density against the
   0.007 deg/s/sqrt(Hz) assumed here, and calibrate its bias from a still period
   before each burst. Use the gyro, not the accelerometer: shake is rotation.
2. **Log per-subpage timestamps.** Record each subpage with its subpage ID and
   the host time its data-ready bit was seen, and log the gyro alongside, so the
   camera-gyro offset can be estimated (the hybrid does this by
   cross-correlation).
3. **Avoid Adafruit `getFrame()` merging.** It merges two subpages exposed at
   different times and poses into one frame; under motion that frame is
   inconsistent (on the BAB chart with wobble the merged-bicubic baseline is
   1.5-2.6 dB below the registered average). Keep each subpage as its own sparse
   observation, using the Melexis C library or raw logging.
4. **Measure the PSF** (for example from a slanted-edge target with the sensor
   stepped on a turntable) and calibrate px/deg the same way; the PSF sweep
   shows a wrong PSF can erase the gain.
5. **Design the motion.** The simulation favors a slow, deliberate, circular
   wobble (about ±1-2 deg at 0.5-1 Hz) with the fastest subpage rate the noise
   allows (16 Hz did better than 8 Hz for the BAB), rather than relying on
   natural tremor.

## Install

Python 3.12 or newer.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Run

```bash
pytest -q                        # fast suite (slow tests are deselected), under a minute
pytest -q -m slow                # benchmark smoke test (run_benchmark.py --quick)
python scripts/demo.py           # writes docs/results/demo_montage.png
python scripts/run_benchmark.py --quick --out /tmp/quick   # smoke run, seconds to tens of seconds
python scripts/run_benchmark.py  # full sweep: docs/results/results.{md,csv} and PNG plots
```

The full benchmark (590 cases) took 76 s on 31 worker processes; use `--jobs N`
to limit the processes and `--sweeps central,k,refresh,psf,scale,reg` to run a
subset. It runs all sweeps from the plan with 5 seeds each, with these
deliberate trims: the frame-count, refresh, PSF and scale sweeps use the chart
scene with two motions (0.3 deg tremor and ±2 deg 1 Hz wobble) only, and the
registration-vs-contrast sweep uses the BAB only.

## Architecture

```
simulate (scene, motion, sensor, gyro) -> register -> reconstruct -> evaluate
```

| Module | Role |
|---|---|
| `mlx_sr/types.py` | Shared dataclasses and conventions (shift `(dx, dy)` in LR px, grids, timing, gyro axes) |
| `mlx_sr/scene.py` | 8x HR scenes with metric metadata (chart, people, text, textured, flat) |
| `mlx_sr/motion.py` | Tremor, drift and wobble synthesis, angle-to-shift, gyro simulation |
| `mlx_sr/sensor.py` | MLX90640 forward model producing a `Capture` (subpages plus truth), `truth_at_scale` |
| `mlx_sr/registration.py` | Phase correlation, masked Gauss-Newton, gyro integration, time-offset estimate, hybrid |
| `mlx_sr/operators.py` | Sparse forward operator `A` (shift, erf-integrated box and Gaussian PSF, chess mask) |
| `mlx_sr/reconstruct.py` | Bicubic, registered average, normalized convolution, IBP, BTV-MAP with optional FPN estimation |
| `mlx_sr/metrics.py` | PSNR/SSIM, slanted-edge MTF, bar contrast, hot-spot error, flat sigma, registration RMS, high-band fidelity |
| `mlx_sr/pipeline.py` | `run_pipeline` / `run_case`: method plus all baselines, oracle and still-twin controls, gains |
| `scripts/run_benchmark.py` | Sweeps, `results.csv`, `results.md`, plots |
| `scripts/demo.py` | Demo montage |
| `tests/` | Forward model, registration, operators, reconstruction, metrics and pipeline tests |

## License and status

Research prototype. See [docs/RESEARCH.md](docs/RESEARCH.md) for the
literature, the sensor facts and the full list of sources.
