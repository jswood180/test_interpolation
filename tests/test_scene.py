"""Scene generator and metadata tests."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_sr import metrics
from mlx_sr.scene import SCENE_KINDS, make_scene
from mlx_sr.types import Roi, Scene


@pytest.mark.parametrize("kind", sorted(SCENE_KINDS))
def test_scene_shapes_and_meta(kind):
    sc = make_scene(kind)
    assert isinstance(sc, Scene)
    assert sc.hr.shape == ((24 + 8) * 8, (32 + 8) * 8)
    assert np.all(np.isfinite(sc.hr))
    meta = sc.meta
    assert meta.shape_lr == (24, 32) and meta.hr_factor == 8 and meta.margin_lr == 4
    rois = [f.roi for f in meta.flat] + [b.roi for b in meta.bars] + [h.roi for h in meta.hotspots]
    if meta.edge is not None:
        rois.append(meta.edge.roi)
    for r in rois:
        assert 0 <= r.x0 < r.x1 <= 32 and 0 <= r.y0 < r.y1 <= 24, (kind, r)


def test_scene_deterministic():
    a = make_scene("textured", seed=3)
    b = make_scene("textured", seed=3)
    c = make_scene("textured", seed=4)
    np.testing.assert_array_equal(a.hr, b.hr)
    assert not np.allclose(a.hr, c.hr)
    assert a.hr.std() == pytest.approx(2.0)
    assert make_scene("low_contrast").hr.std() == pytest.approx(0.3)


def test_chart_metadata_consistent_with_pixels():
    sc = make_scene("chart")
    m = sc.meta
    hr = sc.hr
    # Flat region is exactly flat (with a 2 LR px clearance it stays flat after blur).
    for f in m.flat:
        rs, cs = f.roi.to_hr(8, 4)
        assert np.all(hr[rs, cs] == f.value_c)
        big = Roi(f.roi.x0 - 2, f.roi.y0 - 2, f.roi.x1 + 2, f.roi.y1 + 2)
        rs, cs = big.to_hr(8, 4)
        assert np.all(hr[rs, cs] == f.value_c)
    # Edge ROI contains only the two edge levels (and mixtures).
    e = m.edge
    rs, cs = e.roi.to_hr(8, 4)
    patch = hr[rs, cs]
    assert patch.min() == pytest.approx(e.low_c) and patch.max() == pytest.approx(e.high_c)
    # Left column cold, right column hot.
    assert np.all(patch[:, 0] == e.low_c) and np.all(patch[:, -1] == e.high_c)
    # Edge position per HR row follows the specified line.
    ys = (np.arange(rs.start, rs.stop) + 0.5) / 8 - 4
    frac_hot = (patch - e.low_c) / (e.high_c - e.low_c)
    x_edge = e.roi.x0 + (patch.shape[1] - frac_hot.sum(axis=1)) / 8
    np.testing.assert_allclose(x_edge, e.center[0] + (ys - e.center[1]) * np.tan(np.radians(e.angle_deg)),
                               atol=0.02)
    # Hot spots: the HR max inside the ROI is the true peak for discs >= one HR cell.
    for h in m.hotspots:
        rs, cs = h.roi.to_hr(8, 4)
        assert hr[rs, cs].max() == pytest.approx(h.peak_c, abs=1e-9 if h.radius_lr >= 0.25 else 5)
        area = (hr[rs, cs] - h.background_c).sum() / (h.peak_c - h.background_c) / 64
        assert area == pytest.approx(np.pi * h.radius_lr ** 2, rel=0.02)
    # Bars: column profile has the stated period.
    for g in m.bars:
        rs, cs = g.roi.to_hr(8, 4)
        prof = hr[rs, cs].mean(axis=0)
        frac = (prof - g.low_c) / (g.high_c - g.low_c)
        assert frac.mean() == pytest.approx(0.5, abs=0.06)
        spec = np.abs(np.fft.rfft(frac - frac.mean()))
        freqs = np.fft.rfftfreq(frac.size, d=1 / 8)
        assert freqs[np.argmax(spec)] == pytest.approx(g.freq_cyc_per_lr, rel=0.15)
    freqs = [g.freq_cyc_per_lr for g in m.bars]
    assert min(freqs) < 0.5 < max(freqs)


def test_roi_to_scale():
    r = Roi(2.0, 3.0, 6.0, 5.0)
    assert r.to_scale(1) == (slice(3, 5), slice(2, 6))
    assert r.to_scale(2) == (slice(6, 10), slice(4, 12))
    assert r.to_scale(2, shift=(0.25, 0.0)) == (slice(6, 10), slice(4, 11))
    assert r.to_hr(8, 4) == (slice(56, 72), slice(48, 80))
    assert r.width == 4 and r.height == 2 and r.center == (4.0, 4.0)


def test_people_and_text_meta():
    p = make_scene("people")
    assert p.meta.hotspots and p.meta.flat and len(p.meta.extra["faces"]) == 2
    rs, cs = p.meta.flat[0].roi.to_hr(8, 4)
    assert np.ptp(p.hr[rs, cs]) == 0
    t = make_scene("text", text="AB")
    rs, cs = t.meta.extra["text_roi"].to_hr(8, 4)
    assert t.hr[rs, cs].max() > 30 and t.hr[rs, cs].min() == 25


def test_text_counters_are_not_filled():
    t = make_scene("text", text="O", contrast_k=10.0)
    r = t.meta.extra["text_roi"]
    cx, cy = r.center
    rs, cs = Roi(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5).to_hr(8, 4)
    assert np.all(t.hr[rs, cs] == 25.0)  # hole of the O is background
    rs, cs = r.to_hr(8, 4)
    assert t.hr[rs, cs].max() == pytest.approx(35.0)


def test_minimal_psnr():
    a = make_scene("textured").hr
    assert metrics.psnr(a, a) == float("inf")
    b = a + 0.1
    expected = 10 * np.log10(np.ptp(a) ** 2 / 0.01)
    assert metrics.psnr(b, a) == pytest.approx(expected)
    assert metrics.psnr(b, a, data_range=1.0) == pytest.approx(20.0)
    assert metrics.psnr(b, a, border=4) == pytest.approx(10 * np.log10(np.ptp(a[4:-4, 4:-4]) ** 2 / 0.01))
