# Multi-Frame Sub-Pixel Super-Resolution for the MLX90640

Research notes for a Python prototype that improves the resolution of the MLX90640 (32x24 thermopile array) by exploiting small motion of the sensor between frames, optionally assisted by an IMU.

**Conventions.** Every claim links its source inline, and a full reference list is at the end.
- **[my calc]** marks a figure I derived from cited numbers.
- **[unverified]** marks a claim I could not confirm from a primary source.
- **[recommendation]** marks a design suggestion from this research, not something established by a source.

---

## 1. Summary

- Multi-frame super-resolution (SR) combines frames that are offset by sub-pixel amounts into one image on a finer grid. Commercial thermal cameras do this with natural hand motion: [FLIR UltraMax](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf) and [Testo SuperResolution](https://www.testo.com/en-SE/testo-superresolution/p/0554-7806).
- Vendor claims are 2x linear resolution (4x pixels) from FLIR and 1.6x geometric resolution from Testo. Independent verification is weak ([arXiv 2608.02404](https://arxiv.org/pdf/2608.02404)). Hobbyist tests show modest gains ([EEVblog thread](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/)).
- **The biggest MLX90640-specific risk:** its instantaneous field of view (IFOV) is very coarse, about 1.7–3.4°/px [my calc from the [datasheet](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf)]. Natural hand tremor alone may move the image by only a few hundredths of a pixel. Plan for deliberate wobble ([section 7](#7-pitfalls)).
- **Chess-mode readout** samples the two halves of each frame at different times. The SR model must treat each subpage as its own observation ([section 2.2](#22-chess-subpages-and-the-adafruit-driver)).
- **Recommended pipeline [recommendation]:**
  - Hybrid registration: a gyro predicts each shift and image-based ECC or phase correlation refines it.
  - Reconstruction: a normalized-convolution initial estimate, refined by a robust L1 + bilateral-TV MAP solver.
  - Output on a 2x grid.
  - Evaluate against a registered multi-frame average, not only against single-frame bicubic.

---

## 2. Sensor facts: MLX90640

Unless noted, the source is the [Melexis datasheet, Rev 13 (2026-10-21)](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf), cross-checked against the [Rev 11 copy](https://cdn.sparkfun.com/assets/3/1/c/6/f/MLX90640-Datasheet.pdf).

### 2.1 Specifications

| Item | Value |
|---|---|
| Array | 32x24 = 768 pixels |
| FOV variants | **BAA** 110°x75°, **BAB** 55°x35°. FOV is defined at the 50% sensitivity points of a rotated point source, along the 32-pixel direction. Central pointing error max 5° (BAA) / 3° (BAB) |
| Average IFOV [my calc] | BAA ≈ 3.44°/px (H), 3.13°/px (V). BAB ≈ 1.72°/px (H), 1.46°/px (V) |
| NETD at 1 Hz (Table 14) | BAA average **0.14 K** (min 0.1, std 0.05). BAB average **0.25 K** (min 0.2, std 0.05). The headline "0.1 K @ 1 Hz" is the BAA minimum |
| Noise vs refresh rate | Given only as a graph (Fig. 19). Corner pixels are noisier because of the integrated lens. Scaling noise by √rate is an assumption **[unverified]** |
| Refresh rate | 0.5–64 Hz, set in Control register 0x800D. **The rate applies per subpage**: a full frame takes two subpage periods |
| ADC resolution | 16–19 bit. The default (and calibration) setting is 18 bit |
| I2C | FM+ up to 1 MHz. EEPROM operations max 400 kHz |
| Bandwidth [my calc] | One subpage read is 832 words ≈ 15 kbit. That is ≈ 37 ms at 400 kHz (16 Hz feasible, 32 Hz not) and ≈ 15 ms at 1 MHz (32 Hz feasible) |
| Defective pixels | Up to 4 (broken or outlier), flagged in EEPROM. Replace them by interpolating neighbors |
| Warm-up | About 4 minutes after power-on before absolute temperature measurement (Fig. 7) |
| PSF / MTF | **Not published** in the datasheet or anywhere I found. Measure it before drawing conclusions |

One hobbyist reports that 16 Hz works at 800 kHz I2C and 32 Hz fails on an ESP32 ([Reddit, via mirror](https://redlib.hbubli.cc/r/esp32/comments/1ftkfn5/esp32s3_thermal_camera_with_mlx90640/lr1z0yb)). This is anecdotal.

### 2.2 Chess subpages and the Adafruit driver

- **The readout.** The array is read in two subpages. In chess mode (factory default) they form a checkerboard; in TV-interleaved mode they are alternate rows. The sensor is calibrated in chess mode, and Melexis recommends chess mode for better fixed-pattern noise ([datasheet](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf)). Each subpage refreshes 384 pixels, and ambient temperature (Ta) needs data from both.
- **The Adafruit driver.** In the [Adafruit CircuitPython driver](https://github.com/adafruit/Adafruit_CircuitPython_MLX90640) (`adafruit_mlx90640.py`), `getFrame()` reads two subpages back to back. For each one it writes only the pixels whose pattern matches that subpage.
  - The returned 768-value frame therefore **mixes two time instants**.
  - Broken and outlier pixels are written as **−273.15**, not interpolated.
  - The driver is pure Python. Its example uses an 800 kHz I2C clock and warns about "Too many retries" at high refresh rates.
- **Implication [recommendation].** Keep each subpage as its own sparse observation: samples, mask, timestamp and shift. Mask defective pixels. For throughput, consider the [Melexis C library](https://github.com/melexis/mlx90640-library), the [Melexis Python driver](https://github.com/melexis-fir/mlx9064x-driver-py), or logging raw data for offline processing.

---

## 3. Prior art and evidence

### 3.1 EEVblog thread: improving resolution by moving the sensor

Two pages, June 2020: [page 1](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/), [page 2](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/25/).

- **What was tried.** The original poster took about 16 hand-held Seek XR frames, aligned them by hand in Photoshop and averaged them. Text that was unreadable in one frame became readable. A tripod stack showed per-pixel sensitivity differences (fixed-pattern noise); hand-held motion gave smoother results.
- **Averaging vs. real SR.** One participant (zrq) argued that averaging registered frames gives "a low noise but blured image" with no resolution gain. He said panorama and stitching tools cannot exceed Nyquist, and recommended true multi-frame SR (OpenCV Bilateral TV-L1, FRVSR, EDVR).
- **A measured comparison.** Using a Therm-App Pro, one participant (Ultrapurple) found:
  - tripod plus SR looked **worse than a bicubic resize**,
  - hand-held SR was only slightly better than tripod SR,
  - the results were "closer than expected".
- **Commercial history:**
  - Microscan: 4 germanium prisms in a chopper wheel shift the image.
  - Piezo actuators shift the sensor or optics.
  - Testo's SR depends on hand shake and is "not possible" on a tripod.
  - FLIR One Pro LT claims 160x120 after processing from an 80x60 Lepton core.
- **IMU.** Phones use the IMU as a rough motion prior for alignment, and the gain is not uniform across the image.
- **Page 2.** Test at hyperfocal distance, because focus blur ruins comparisons. The open problem is radiometrically accurate output. One suggestion was to put known-temperature references in the scene.
- **Scope.** The MLX90640 is not discussed.

### 3.2 FLIR UltraMax

Sources: [product page](https://www.flir.com/discover/rd-science/ultramax--the-ultimate-resolution/), [technical note](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf).

- **How it works.** It captures **16 frames in under 1 s**, stored as one JPG and merged in FLIR software. It uses "the natural movement of the human body" to offset the frames.
- **Claims:**
  - **2x resolution and 4x pixels** (320x240 → 640x480, 640x480 → 1280x960).
  - **"50% less noise."**
  - Radiometric data is retained.
  - Smaller measurement spot size. In FLIR's example a 120°F target reads 98°F natively and 118°F with UltraMax.
- **Stated limitations:**
  - Too much user or target motion means the frames can't be aligned.
  - A tripod gives too little offset.
  - Uniformly low-contrast or out-of-focus scenes impede it.
  - FLIR recommends holding the camera steady with both hands.
- **Not published:** algorithm details and MTF data.

### 3.3 Testo SuperResolution and independent checks

- Testo claims usable geometric resolution improved "by a factor of 1.6" with four times as many measuring values ([Testo](https://www.testo.com/en-SE/testo-superresolution/p/0554-7806)).
- A 2026 preprint found only a "marginal" improvement from UltraMax. It says UltraMax has never been independently verified in the peer-reviewed literature, and its loosened-tripod capture may not reproduce real use ([arXiv 2608.02404](https://arxiv.org/pdf/2608.02404); I saw it only via a search summary).
- A figure of "~1.4–2x effective resolution" for UltraMax is sometimes quoted. I found no source for it **[unverified]**.

---

## 4. Motion: tremor, pixel shift and the IMU

### 4.1 Tremor characteristics

- Physiological hand tremor is concentrated in **8–12 Hz**. Linear RMS amplitude at the hand is about 50–200 µm ([Comparison of Baseline Tremor..., PMC3989364](https://pmc.ncbi.nlm.nih.gov/articles/PMC3989364)).
- Camera shake is dominated by **yaw and pitch**: people are "about ten times worse" at controlling those than linear motion. Shake consists of many tiny movements punctuated by a few large ones ([Lensrentals](https://www.lensrentals.com/blog/2013/07/good-vibrations-designing-a-better-stabilization-test-part-i/)).
- Angular amplitude is not well sourced. Lensrentals commenters suggest ≈ 0.057° within a 1/30 s interval and angular rates of 0.01–0.03 rad/s **[unverified]**.

### 4.2 Converting rotation to pixel shift

For a pinhole model, shift ≈ f_px · tan(Δθ), with f_px = (N/2) / tan(FOV/2) [my calc]:

| Variant | f_px (H) | px/° (H) | f_px (V) | px/° (V) |
|---|---|---|---|---|
| BAA 110°x75° | ≈ 11.2 px/rad | ≈ 0.20 | ≈ 15.6 px/rad | ≈ 0.27 |
| BAB 55°x35° | ≈ 30.7 px/rad | ≈ 0.54 | ≈ 38.1 px/rad | ≈ 0.66 |

Caveats:
- The horizontal and vertical pinhole focal lengths disagree.
- FOV is defined at the 50% sensitivity points.
- The wide BAA lens likely has distortion that isn't documented **[unverified]**.

**Calibrate px/° empirically [recommendation]**: rotate the sensor on a turntable while logging gyro rate and image shift. Roll rotates the image about the optical axis, so use a Euclidean model rather than pure translation.

### 4.3 Gyro vs. accelerometer

- **Use the gyro.** Rotation dominates hand shake ([Lensrentals](https://www.lensrentals.com/blog/2013/07/good-vibrations-designing-a-better-stabilization-test-part-i/)). Rotation-induced shift does not depend on scene depth, and a gyro measures rotation rate directly.
- **The accelerometer is a poor fit.** It measures linear acceleration plus gravity, so translation needs double integration, which drifts. Translation-induced shift (f_px · t / Z) is tiny anyway: 1 mm at 1 m on the BAB is ≈ 0.03 px [my calc; engineering reasoning].
- **Getting a shift from the gyro.** Integrate gyro rate between subpage timestamps to get Δθ, then convert with calibrated px/°. Estimate gyro bias from a still period before each burst [recommendation].
- **Prior art.** Gyro-only video stabilization with automatic gyro-camera calibration, including drift, has been demonstrated ([Karpenko et al. 2011](https://graphics.stanford.edu/papers/stabilization/)).

### 4.4 Camera–IMU time synchronization

- Timestamp each subpage when its "new data available" bit is seen, and log the gyro at 200–1000 Hz [recommendation].
- Estimate the fixed time offset by cross-correlating image-derived angular rate with gyro angular rate. This is how [Kalibr](https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration) initializes its temporal calibration.
- The full joint spatial-temporal estimator is in [Furgale et al., IROS 2013](https://furgalep.github.io/bib/furgale_iros13.pdf), which estimated time offsets to within a fraction of the smallest measurement period.

---

## 5. Recommended algorithm pipeline [recommendation]

### 5.1 Acquisition

1. Log raw data per subpage: subpage ID, host timestamp, plus gyro samples. Start at 16 Hz with 1 MHz I2C. The rate is a trade-off between noise and frame count ([section 2.1](#21-specifications)).
2. Mask EEPROM-flagged pixels ([section 2.2](#22-chess-subpages-and-the-adafruit-driver)).
3. Correct fixed-pattern noise with a per-pixel offset flat-field against a uniform target. When the sensor moves, fixed-pattern noise becomes a moving artifact relative to the scene; the [EEVblog thread](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/) observed it in tripod stacks.

### 5.2 Registration

**Hybrid registration is recommended.** The gyro predicts each subpage's shift ([section 4](#4-motion-tremor-pixel-shift-and-the-imu)), an image method refines it, and the gyro is the fallback in low-contrast scenes.

- **Phase correlation.** [`skimage.registration.phase_cross_correlation`](https://scikit-image.org/docs/stable/api/skimage.registration) with `upsample_factor=N` registers to 1/N px, using Guizar-Sicairos et al. (2008) matrix-multiply DFT upsampling.
  - The docs say phase normalization is "not very robust to noise"; use `normalization=None` when noise is high.
  - On 32x24 frames, apply a Hann window or padding [recommendation].
- **ECC.** [`cv2.findTransformECC`](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html) supports translation, Euclidean, affine and homography models.
  - `findTransformECCWithMask` accepts validity masks, which suits chess subpages and dead pixels.
  - It takes an initial warp, so the gyro prediction can seed it.
  - It throws an exception if it does not converge.
- **Joint refinement.** Optionally re-register each subpage against the current SR estimate (reprojected) and iterate.

Achievable registration accuracy at 32x24 has no published figure **[unverified]**. Measure it in simulation as RMS shift error versus SNR and contrast. FLIR notes that low-contrast scenes defeat alignment ([technical note](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf)).

### 5.3 Reconstruction

1. **Initial estimate: normalized convolution / kernel regression.** Splat every sample onto the high-resolution (HR) grid with Gaussian weights, and keep a certainty map.
   - This handles irregular and missing samples natively, which matches chess subpages ([Pham, van Vliet & Schutte 2006](https://link.springer.com/article/10.1155/ASP/2006/83268); [Takeda, Farsiu & Milanfar 2007](https://people.duke.edu/~sf59/KernelRegression_Final.pdf)).
   - Drizzle is the closely related astronomy method. It keeps the flexibility of shift-and-add while preserving most of the resolution of interlacing, at the cost of correlated noise ([Fruchter & Hook 2002](https://arxiv.org/abs/astro-ph/9808087)).
2. **Refinement: robust MAP with an L1 data term and a bilateral-TV (BTV) prior** ([Farsiu, Robinson, Elad & Milanfar 2004](https://people.duke.edu/~sf59/SRfinal.pdf)).
   - Forward model: shift → PSF (pixel box convolved with a Gaussian) → decimate with the subpage mask.
   - The L1 term rejects outliers such as registration errors, dead pixels and small scene motion. BTV controls noise.
   - At 32x24 → 2–4x, gradient descent costs very little.
3. **Baselines:**
   - Registered multi-frame average + bicubic.
   - [Irani–Peleg iterative back-projection](https://www.weizmann.ac.il/math/irani/sites/math.irani/files/uploads/cvgip-sr-1991_.pdf): simple, but it amplifies noise without a prior.
4. **Output grid: 2x.**
   - Reconstruction constraints lose value as magnification grows ([Baker & Kanade 2002](https://ri.cmu.edu/publications/limits-on-super-resolution-and-how-to-break-them)).
   - Lin & Shum derive explicit effective magnification limits ([2004](https://www.microsoft.com/en-us/research/?p=151767)). An often-quoted practical limit of ~1.6 is **[unverified]**.
   - Beyond 3x adds little.

**Deep-learning SR is not recommended here.** There is no training data, and ML upscalers such as waifu2x lose radiometric data ([EEVblog thread](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/)).

For the kernel-regression merge, see the unofficial Python implementation of [Wronski et al. 2019](https://arxiv.org/abs/1905.03277) at [Jamy-L/Handheld-Multi-Frame-Super-Resolution](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution).

---

## 6. Simulation and evaluation [recommendation]

### 6.1 Synthetic data

- **Ground truth.** HR scenes at 8x (256x192) in °C: background 20–25°C, people 30–34°C, small hot spots 50–80°C, plus slanted edges, bar groups and text.
- **Optics and sampling.** Integrate over each pixel's footprint at its sub-pixel offset (box aperture), convolved with a Gaussian blur. Sweep σ from 0.2 to 0.8 px, because the PSF is unpublished ([section 2.1](#21-specifications)). Add mild radial distortion for the BAA.
- **Motion.**
  - Yaw and pitch only.
  - Tremor: band-limited noise at 8–12 Hz ([PMC3989364](https://pmc.ncbi.nlm.nih.gov/articles/PMC3989364)).
  - Drift: a < 2 Hz random walk.
  - Sweep RMS amplitude from 0.05° to 3° and convert with px/° ([section 4.2](#42-converting-rotation-to-pixel-shift)).
  - Average the motion over each subpage interval to model motion blur. The sensor's real integration behavior is **[unverified]**.
- **Readout.** Chess subpages sampled at alternating timestamps, 1/rate apart.
- **Noise:**
  - Temporal noise σ(1 Hz) = 0.14 K (BAA) or 0.25 K (BAB) ([datasheet Table 14](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf)), scaled by √rate as an assumption **[unverified]**, and higher in the corners.
  - Per-pixel fixed-pattern offsets of 0.1–0.3 K **[unverified]**.
  - Up to 4 dead pixels, plus quantization.
- **Gyro.** Bias, white noise and a time offset.

### 6.2 Metrics

- **PSNR / SSIM** against ground truth on the output grid, compared with (a) single-frame bicubic and (b) registered multi-frame average + bicubic. Baseline (b) separates denoising from real resolution gain, which was zrq's point in the [EEVblog thread](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/).
- **Resolution:**
  - Slanted-edge MTF: MTF50, and contrast above native Nyquist (0.5 cycles/px).
  - Smallest resolvable bar group.
- **Radiometry:**
  - Peak-temperature error on small hot spots, as in FLIR's 98°F vs 118°F example ([technical note](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf)).
  - Noise standard deviation in flat regions.
- **Registration:** RMS shift error against the true shifts.
- **Sweeps:** frame count, tremor amplitude, refresh rate and PSF σ.
  - Rule of thumb: at least r² well-spread shifts for upscale factor r, and more for random shifts **[unverified]**.
  - With chess readout, count subpages, not frames.
- **Hardware validation:** a heated plate behind a slanted edge or bar mask, shot at hyperfocal distance ([EEVblog thread, page 2](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/25/)).

---

## 7. Pitfalls

1. **Tremor too small for the IFOV.** If tremor is about 0.1° **[unverified]**, the shift is only ≈ 0.03 px on the BAA and ≈ 0.06 px on the BAB [my calc].
   - Wronski et al. found natural hand motion random enough in the sub-pixel domain for phone cameras ([arXiv 1905.03277](https://arxiv.org/abs/1905.03277)), but phones have a far finer IFOV.
   - **Mitigation [recommendation]:** deliberate wobble of about ±1–3° (hand sweep, vibration motor or pan/tilt servo). Check the histogram of fractional shifts.
   - Too much motion breaks alignment ([FLIR](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf)).
2. **Mixing subpages.** Fusing chess subpages into one frame while moving breaks the model. Mask the −273.15 bad-pixel values from the Adafruit driver ([section 2.2](#22-chess-subpages-and-the-adafruit-driver)).
3. **Interleaved mode** is not the calibration mode and has worse fixed-pattern noise ([datasheet](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf)).
4. **Noise vs. frame count.** Higher refresh rates give more frames but noisier ones. Pure-Python reads can also drop frames and add timestamp jitter ([Adafruit driver](https://github.com/adafruit/Adafruit_CircuitPython_MLX90640)).
5. **Fixed-pattern noise is attached to the sensor**, not the scene ([EEVblog thread](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/)). Flat-field it or estimate it jointly.
6. **Scene problems:**
   - Parallax and moving objects (use robust weights).
   - Low-contrast scenes and defocus ([FLIR](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf)).
7. **Thermal drift** during warm-up: wait about 4 minutes ([datasheet](https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf)).
8. **BAA lens distortion** makes shifts non-uniform across the field **[unverified]**. Use a calibrated or per-region motion model.
9. **Over-regularization.** BTV can flatten small hot-spot peaks. Always report radiometric error.
10. **No aliasing, no gain.** If the optics blur beyond the pixel pitch, SR reduces to denoising. Measure the PSF first.

---

## 8. Expected gain

- **Estimate [unverified, my synthesis]:** about **1.3–1.8x effective linear resolution in the best case**. That assumes aliased optics, good registration, a static scene and enough sub-pixel diversity. Expect less if the PSF is wide.
- **Evidence behind it:**
  - Vendor claims: 2x ([FLIR](https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf)) and 1.6x ([Testo](https://www.testo.com/en-SE/testo-superresolution/p/0554-7806)).
  - Weak independent results: [arXiv 2608.02404](https://arxiv.org/pdf/2608.02404) and the [EEVblog tests](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/).
  - Theoretical limits on magnification ([Baker & Kanade 2002](https://ri.cmu.edu/publications/limits-on-super-resolution-and-how-to-break-them)).
- **Success criterion [recommendation]:** a measurable MTF50 or bar-target gain over a *registered multi-frame average + bicubic*, not merely over single-frame bicubic.

---

## 9. Existing open-source code

- **No existing project** applies multi-frame SR to the MLX90640, AMG8833 or Lepton, based on GitHub and web searches in November 2026. Existing MLX90640 projects interpolate only, for example [makerportal/raspi-thermal-cam](https://github.com/makerportal/raspi-thermal-cam).
- **MLX90640 drivers:**
  - [Adafruit CircuitPython](https://github.com/adafruit/Adafruit_CircuitPython_MLX90640)
  - [Melexis C library](https://github.com/melexis/mlx90640-library)
  - [Melexis Python driver](https://github.com/melexis-fir/mlx9064x-driver-py)
  - Others: [leswright1977/mlx90640_python](https://github.com/leswright1977/mlx90640_python), [RevKarl/MLX90640-python](https://github.com/RevKarl/MLX90640-python), [Seeed_Python_MLX9064x](https://github.com/Seeed-Studio/Seeed_Python_MLX9064x)
- **Hand-held SR implementations (visible light):**
  - [Jamy-L/Handheld-Multi-Frame-Super-Resolution](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution)
  - [zach00700/py-handheld-super-res](https://github.com/zach00700/py-handheld-super-res)
  - [kunzmi/ImageStackAlignator](https://github.com/kunzmi/ImageStackAlignator)
- **Thermal-adjacent (untested):**
  - [posystorage/FLIR-Lepton-PocketThermal](https://github.com/posystorage/FLIR-Lepton-PocketThermal) advertises "real-time super-resolution"; its method is **[unverified]**.
  - [meister1378/BiaSRNet](https://github.com/meister1378/BiaSRNet) is a learned burst thermal SR model.
- **OpenCV BTV-L1:** the `superres` module lives in opencv_contrib. Whether its Python bindings are usable is **[unverified]**.

---

## References

**Primary sources reviewed**
1. EEVblog Forum, "Improving thermal sensor resolution by vibrating/moving sensor while shooting," 2020. [Page 1](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/), [Page 2](https://www.eevblog.com/forum/thermal-imaging/improving-thermal-sensor-resolution-by-vibratingmoving-sensor-while-shooting/25/)
2. FLIR, "UltraMax – the Ultimate Resolution" (product page). <https://www.flir.com/discover/rd-science/ultramax--the-ultimate-resolution/>
3. FLIR, "UltraMax – The Ultimate Resolution" technical note. <https://www.transcat.ca/media/pdf/FLIR-ULTRAMAX-thermal-imager-application-note.pdf>

**Sensor and drivers**

4. Melexis, MLX90640 datasheet, Rev 13 (2026-10-21). <https://media.melexis.com/-/media/files/documents/datasheets/MLX90640-datasheet-melexis.pdf>
5. Melexis, MLX90640 datasheet, Rev 11 (2018), SparkFun copy. <https://cdn.sparkfun.com/assets/3/1/c/6/f/MLX90640-Datasheet.pdf>
6. Adafruit, Adafruit_CircuitPython_MLX90640. <https://github.com/adafruit/Adafruit_CircuitPython_MLX90640>
7. Melexis, mlx90640-library. <https://github.com/melexis/mlx90640-library>
8. Melexis, mlx9064x-driver-py. <https://github.com/melexis-fir/mlx9064x-driver-py>
9. Reddit r/esp32 discussion (mirror), I2C clock vs refresh rate. <https://redlib.hbubli.cc/r/esp32/comments/1ftkfn5/esp32s3_thermal_camera_with_mlx90640/lr1z0yb>

**Commercial claims and verification**

10. Testo, SuperResolution. <https://www.testo.com/en-SE/testo-superresolution/p/0554-7806>
11. "Loggia dei Lanzi: AI Thermography Enhancement Comparisons through 3D Photogrammetry," arXiv 2608.02404. <https://arxiv.org/pdf/2608.02404>

**Super-resolution algorithms**

12. S. Farsiu, M. D. Robinson, M. Elad, P. Milanfar, "Fast and Robust Multiframe Super Resolution," IEEE TIP 13(10), 2004. <https://people.duke.edu/~sf59/SRfinal.pdf>
13. M. Irani, S. Peleg, "Improving Resolution by Image Registration," CVGIP: Graphical Models and Image Processing 53(3), 1991. <https://www.weizmann.ac.il/math/irani/sites/math.irani/files/uploads/cvgip-sr-1991_.pdf>
14. A. S. Fruchter, R. N. Hook, "Drizzle: A Method for the Linear Reconstruction of Undersampled Images," PASP 114, 2002. <https://arxiv.org/abs/astro-ph/9808087>
15. H. Takeda, S. Farsiu, P. Milanfar, "Kernel Regression for Image Processing and Reconstruction," IEEE TIP 16(2), 2007. <https://people.duke.edu/~sf59/KernelRegression_Final.pdf>
16. T. Q. Pham, L. J. van Vliet, K. Schutte, "Robust Fusion of Irregularly Sampled Data Using Adaptive Normalized Convolution," EURASIP JASP, 2006. <https://link.springer.com/article/10.1155/ASP/2006/83268>
17. S. Baker, T. Kanade, "Limits on Super-Resolution and How to Break Them," IEEE TPAMI 24(9), 2002. <https://ri.cmu.edu/publications/limits-on-super-resolution-and-how-to-break-them>
18. Z. Lin, H.-Y. Shum, "Fundamental Limits of Reconstruction-Based Superresolution Algorithms under Local Translation," IEEE TPAMI 26(1), 2004. <https://www.microsoft.com/en-us/research/?p=151767>
19. B. Wronski et al., "Handheld Multi-Frame Super-Resolution," SIGGRAPH 2019. <https://arxiv.org/abs/1905.03277>

**Registration**

20. scikit-image, `skimage.registration` (phase_cross_correlation). <https://scikit-image.org/docs/stable/api/skimage.registration>
21. OpenCV, Video tracking module (findTransformECC). <https://docs.opencv.org/3.4/dc/d6b/group__video__track.html>

**Motion, IMU, calibration**

22. "Comparison of Baseline Tremor Under Various Microsurgical Conditions," PMC3989364. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3989364>
23. Lensrentals, "Good Vibrations: Designing a Better Stabilization Test, Part I," 2013. <https://www.lensrentals.com/blog/2013/07/good-vibrations-designing-a-better-stabilization-test-part-i/>
24. A. Karpenko, D. Jacobs, J. Baek, M. Levoy, "Digital Video Stabilization and Rolling Shutter Correction using Gyroscopes," Stanford CSTR 2011-03. <https://graphics.stanford.edu/papers/stabilization/>
25. ETH ASL, Kalibr camera-IMU calibration wiki. <https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration>
26. P. Furgale, J. Rehder, R. Siegwart, "Unified Temporal and Spatial Calibration for Multi-Sensor Systems," IROS 2013. <https://furgalep.github.io/bib/furgale_iros13.pdf>

**Open-source implementations**

27. Jamy-L, Handheld-Multi-Frame-Super-Resolution. <https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution>
28. zach00700, py-handheld-super-res. <https://github.com/zach00700/py-handheld-super-res>
29. kunzmi, ImageStackAlignator. <https://github.com/kunzmi/ImageStackAlignator>
30. makerportal, raspi-thermal-cam. <https://github.com/makerportal/raspi-thermal-cam>
31. posystorage, FLIR-Lepton-PocketThermal. <https://github.com/posystorage/FLIR-Lepton-PocketThermal>
32. meister1378, BiaSRNet. <https://github.com/meister1378/BiaSRNet>
33. leswright1977, mlx90640_python. <https://github.com/leswright1977/mlx90640_python>
34. RevKarl, MLX90640-python. <https://github.com/RevKarl/MLX90640-python>
35. Seeed Studio, Seeed_Python_MLX9064x. <https://github.com/Seeed-Studio/Seeed_Python_MLX9064x>
