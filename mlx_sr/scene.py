"""High-resolution (8x) synthetic thermal scenes with metric metadata.

Every generator returns a :class:`mlx_sr.types.Scene`: ``scene.hr`` is a
float64 array in degrees C of shape
``((24 + 2 margin) * hr, (32 + 2 margin) * hr)`` (default ``(256, 320)``), and
``scene.meta`` is a :class:`mlx_sr.types.SceneMeta` that lists the slanted
edge, the bar groups, the hot spots and the flat regions in LR scene
coordinates (x right, y down, LR-pixel units, origin at the top-left of the
central crop).  Use ``Roi.to_scale(scale, shift)`` to index a reconstruction
and ``Roi.to_hr(hr, margin)`` to index ``scene.hr``.

Feature boundaries are anti-aliased by supersampling each HR cell, so the HR
image is the area-average of the continuous scene over HR cells.

Kinds (``make_scene(kind)``):

- ``"chart"``: slanted edge, five bar groups (0.4-0.9 cyc/LR px), three
  sub-pixel to pixel-sized hot discs and a flat background region.
- ``"edge"``: one large slanted edge (for MTF work).
- ``"bars"``: bar groups only, larger.
- ``"people"``: two warm human silhouettes, a hot mug and a wall gradient.
- ``"text"``: rendered text (matplotlib TextPath) on a flat background.
- ``"textured"``: smooth multi-scale random texture with RMS ``contrast_k``.
- ``"flat"``: uniform background.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

from .types import (
    LR_SHAPE,
    BarGroup,
    EdgeSpec,
    FlatRegion,
    HotSpot,
    Roi,
    Scene,
    SceneMeta,
)

__all__ = [
    "make_scene",
    "chart_scene",
    "edge_scene",
    "bars_scene",
    "people_scene",
    "text_scene",
    "textured_scene",
    "flat_scene",
    "SCENE_KINDS",
]

Indicator = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _Canvas:
    """HR canvas with anti-aliased painting in LR scene coordinates."""

    def __init__(self, shape_lr: tuple[int, int], hr_factor: int, margin_lr: int,
                 background_c: float, supersample: int = 4):
        self.shape_lr = shape_lr
        self.hr = hr_factor
        self.margin = margin_lr
        rows = (shape_lr[0] + 2 * margin_lr) * hr_factor
        cols = (shape_lr[1] + 2 * margin_lr) * hr_factor
        self.img = np.full((rows, cols), float(background_c))
        self.ss = supersample
        # Supersample point coordinates (LR scene coords), shape (rows*ss, cols*ss).
        step = 1.0 / (hr_factor * supersample)
        xs = (np.arange(cols * supersample) + 0.5) * step - margin_lr
        ys = (np.arange(rows * supersample) + 0.5) * step - margin_lr
        self._xs, self._ys = xs, ys

    def coverage(self, inside: Indicator, bbox: Roi | None = None) -> np.ndarray:
        """Fraction of each HR cell where ``inside(x, y)`` holds."""
        cov = np.zeros_like(self.img)
        ss = self.ss
        if bbox is None:
            r0, r1, c0, c1 = 0, self.img.shape[0], 0, self.img.shape[1]
        else:
            rs, cs = Roi(bbox.x0 - 1, bbox.y0 - 1, bbox.x1 + 1, bbox.y1 + 1).to_hr(self.hr, self.margin)
            r0, r1 = max(rs.start, 0), min(rs.stop + 1, self.img.shape[0])
            c0, c1 = max(cs.start, 0), min(cs.stop + 1, self.img.shape[1])
            if r1 <= r0 or c1 <= c0:
                return cov
        X, Y = np.meshgrid(self._xs[c0 * ss:c1 * ss], self._ys[r0 * ss:r1 * ss])
        m = inside(X, Y).astype(float)
        m = m.reshape(r1 - r0, ss, c1 - c0, ss).mean(axis=(1, 3))
        cov[r0:r1, c0:c1] = m
        return cov

    def paint(self, inside: Indicator, value: float, bbox: Roi | None = None) -> None:
        """Alpha-composite ``value`` over the canvas where ``inside`` holds."""
        a = self.coverage(inside, bbox)
        self.img = self.img * (1.0 - a) + value * a

    def meta(self, kind: str, background_c: float, **kw) -> SceneMeta:
        return SceneMeta(kind=kind, shape_lr=self.shape_lr, hr_factor=self.hr,
                         margin_lr=self.margin, background_c=background_c, **kw)

    def scene(self, meta: SceneMeta) -> Scene:
        return Scene(hr=self.img, meta=meta, hr_factor=self.hr, margin_lr=self.margin)


# ---------------------------------------------------------------------------
# Primitive shapes
# ---------------------------------------------------------------------------


def _rect(roi: Roi) -> Indicator:
    return lambda x, y: (x >= roi.x0) & (x < roi.x1) & (y >= roi.y0) & (y < roi.y1)


def _disc(cx: float, cy: float, r: float) -> Indicator:
    return lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < r * r


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> Indicator:
    return lambda x, y: ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1.0


def _edge_halfplane(cx: float, cy: float, angle_deg: float, box: Roi) -> Indicator:
    t = np.tan(np.radians(angle_deg))
    rect = _rect(box)
    return lambda x, y: (x > cx + (y - cy) * t) & rect(x, y)


def _bars(group: BarGroup) -> Indicator:
    roi, p = group.roi, group.period_lr
    rect = _rect(roi)
    if group.orientation == "x":
        return lambda x, y: rect(x, y) & (np.mod(x - roi.x0, p) < 0.5 * p)
    return lambda x, y: rect(x, y) & (np.mod(y - roi.y0, p) < 0.5 * p)


def _pad(roi: Roi, d: float) -> Roi:
    return Roi(roi.x0 - d, roi.y0 - d, roi.x1 + d, roi.y1 + d)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def chart_scene(*, contrast_k: float = 10.0, background_c: float = 25.0,
                hr_factor: int = 8, margin_lr: int = 4,
                shape_lr: tuple[int, int] = LR_SHAPE, hotspot_peak_c: float | None = None,
                edge_angle_deg: float = 7.0) -> Scene:
    """Resolution test chart laid out for the 32 x 24 array.

    Layout (LR scene coords): slanted edge box ``[0.5, 11.5) x [0.5, 14.5)`` with ROI
    ``[2, 10) x [2, 13)``; bar groups at periods 2.5, 2.0, 1.667, 1.333 and
    1.111 LR px (0.4, 0.5, 0.6, 0.75 and 0.9 cyc/LR px) in two rows at
    ``x >= 13``; hot discs of radius 0.3, 0.6 and 1.2 LR px along ``y ~ 19``;
    flat ROI ``[2, 10) x [17, 22)``.
    """
    if shape_lr != LR_SHAPE:
        raise ValueError("chart_scene layout is designed for the 24 x 32 array")
    hot = background_c + contrast_k
    peak = background_c + 3.5 * contrast_k if hotspot_peak_c is None else hotspot_peak_c
    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)

    # Slanted edge: hot to the right of the line inside the box.
    box = Roi(0.5, 0.5, 11.5, 14.5)
    center = (6.0, 7.5)
    cv.paint(_edge_halfplane(center[0], center[1], edge_angle_deg, box), hot, box)
    edge = EdgeSpec(roi=Roi(2.0, 2.0, 10.0, 13.0), angle_deg=edge_angle_deg, center=center,
                    low_c=background_c, high_c=hot, hot_side="right")

    # Bar groups, three cycles each, 5 LR px tall.
    bars: list[BarGroup] = []
    layout = [  # (x0, y0, period)
        (13.3, 1.0, 2.5), (22.6, 1.0, 2.0),
        (13.3, 7.5, 5.0 / 3.0), (19.7, 7.5, 4.0 / 3.0), (25.4, 7.5, 10.0 / 9.0),
    ]
    for x0, y0, p in layout:
        g = BarGroup(roi=Roi(x0, y0, x0 + 3 * p, y0 + 5.0), period_lr=p, n_cycles=3,
                     low_c=background_c, high_c=hot, orientation="x")
        cv.paint(_bars(g), hot, g.roi)
        bars.append(g)

    # Hot spots (uniform discs), different sizes relative to the LR pixel.
    spots: list[HotSpot] = []
    for (cx, cy, r) in [(15.3, 18.6, 0.3), (21.1, 19.2, 0.6), (27.4, 18.9, 1.2)]:
        cv.paint(_disc(cx, cy, r), peak, Roi(cx - r, cy - r, cx + r, cy + r))
        d = r + 2.0
        spots.append(HotSpot(center=(cx, cy), radius_lr=r, peak_c=peak, background_c=background_c,
                             roi=Roi(cx - d, cy - d, cx + d, cy + d)))

    flat = [FlatRegion(roi=Roi(2.0, 17.0, 10.0, 22.0), value_c=background_c)]
    meta = cv.meta("chart", background_c, edge=edge, bars=bars, hotspots=spots, flat=flat,
                   extra={"contrast_k": contrast_k})
    return cv.scene(meta)


def edge_scene(*, contrast_k: float = 10.0, background_c: float = 25.0, angle_deg: float = 5.0,
               hr_factor: int = 8, margin_lr: int = 4,
               shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """One near-vertical slanted edge through the array center, hot on the right.

    The half-plane extends into the margins, so the ROI can be large:
    ``[4, cols-4) x [2, rows-2)``.
    """
    rows, cols = shape_lr
    hot = background_c + contrast_k
    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)
    center = (cols / 2.0 + 0.3, rows / 2.0)
    big = Roi(-margin_lr, -margin_lr, cols + margin_lr, rows + margin_lr)
    cv.paint(_edge_halfplane(center[0], center[1], angle_deg, big), hot)
    edge = EdgeSpec(roi=Roi(4.0, 2.0, cols - 4.0, rows - 2.0), angle_deg=angle_deg, center=center,
                    low_c=background_c, high_c=hot, hot_side="right")
    return cv.scene(cv.meta("edge", background_c, edge=edge, extra={"contrast_k": contrast_k}))


def bars_scene(*, contrast_k: float = 10.0, background_c: float = 25.0,
               periods_lr: tuple[float, ...] = (4.0, 2.5, 2.0, 5.0 / 3.0, 4.0 / 3.0, 10.0 / 9.0),
               hr_factor: int = 8, margin_lr: int = 4,
               shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """Bar groups (three cycles, vertical bars) stacked in rows of up to three.

    The background between groups is flat; the flat ROI is the strip under the
    last row when there is room.
    """
    rows, cols = shape_lr
    hot = background_c + contrast_k
    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)
    bars: list[BarGroup] = []
    x, y, h = 1.0, 1.0, 5.0
    for p in periods_lr:
        w = 3 * p
        if x + w > cols - 1.0:
            x, y = 1.0, y + h + 2.0
        if y + h > rows - 1.0:
            raise ValueError("bar groups do not fit in the array")
        g = BarGroup(roi=Roi(x, y, x + w, y + h), period_lr=p, n_cycles=3,
                     low_c=background_c, high_c=hot, orientation="x")
        cv.paint(_bars(g), hot, g.roi)
        bars.append(g)
        x += w + 1.5
    flat = []
    y_free = max(b.roi.y1 for b in bars) + 2.0
    if y_free + 2.0 <= rows - 2.0:
        flat.append(FlatRegion(roi=Roi(2.0, y_free, cols - 2.0, rows - 2.0), value_c=background_c))
    return cv.scene(cv.meta("bars", background_c, bars=bars, flat=flat,
                            extra={"contrast_k": contrast_k}))


def people_scene(*, background_c: float = 20.0, skin_c: float = 34.0, clothes_c: float = 29.0,
                 mug_c: float = 60.0, hr_factor: int = 8, margin_lr: int = 4,
                 shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """Two people (head + torso) in a room with a slight wall gradient and a hot mug.

    Metadata: the mug is a hot spot; the faces are listed in ``extra["faces"]``
    as ``(cx, cy, rx, ry)``; a flat ROI lies on the floor strip, which has no
    gradient.
    """
    rows, cols = shape_lr
    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)
    # Wall gradient above the floor line (y < floor_y): +1.5 K from left to right.
    ys_hr = (np.arange(cv.img.shape[0]) + 0.5) / hr_factor - margin_lr
    xs_hr = (np.arange(cv.img.shape[1]) + 0.5) / hr_factor - margin_lr
    X, Y = np.meshgrid(xs_hr, ys_hr)
    floor_y = 18.5
    wall = np.where(Y < floor_y, 1.5 * np.clip(X / cols, 0.0, 1.0), 0.0)
    cv.img = cv.img + wall

    faces = []
    for (cx, top, scale) in [(9.0, 3.0, 1.0), (21.5, 5.0, 0.85)]:
        rx, ry = 1.9 * scale, 2.4 * scale
        hy = top + ry
        torso = Roi(cx - 3.6 * scale, hy + ry + 0.4, cx + 3.6 * scale, rows + margin_lr)
        cv.paint(_rect(torso), clothes_c, torso)
        cv.paint(_ellipse(cx, hy + ry + 0.9, 1.0 * scale, 0.9 * scale), skin_c - 1.0,
                 Roi(cx - 1.5, hy + ry, cx + 1.5, hy + ry + 2.0))  # neck
        cv.paint(_ellipse(cx, hy, rx, ry), skin_c, Roi(cx - rx, hy - ry, cx + rx, hy + ry))
        faces.append((cx, hy, rx, ry))

    mug_c_xy, mug_r = (28.3, 16.4), 0.7
    cv.paint(_disc(*mug_c_xy, mug_r), mug_c, Roi(27.0, 15.0, 29.6, 17.8))
    wall_at_mug = background_c + 1.5 * min(mug_c_xy[0] / cols, 1.0)
    spots = [HotSpot(center=mug_c_xy, radius_lr=mug_r, peak_c=mug_c, background_c=wall_at_mug,
                     roi=Roi(mug_c_xy[0] - 2.5, mug_c_xy[1] - 2.5, mug_c_xy[0] + 2.5, mug_c_xy[1] + 2.5))]
    flat = [FlatRegion(roi=Roi(27.0, floor_y + 2.0, 31.0, rows - 0.5), value_c=background_c)]
    meta = cv.meta("people", background_c, hotspots=spots, flat=flat, extra={"faces": faces})
    return cv.scene(meta)


def text_scene(text: str = "HOT 42", *, contrast_k: float = 10.0, background_c: float = 25.0,
               height_lr: float = 9.0, hr_factor: int = 8, margin_lr: int = 4,
               shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """Hot text rendered with matplotlib's bundled DejaVu Sans via ``TextPath``.

    The text is centered horizontally in the top half; the flat ROI is the
    bottom strip.  ``extra["text_roi"]`` is the text bounding box.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath

    rows, cols = shape_lr
    hot = background_c + contrast_k
    tp = TextPath((0, 0), text, size=1.0, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    verts = tp.vertices.copy()
    (x0, y0), (x1, y1) = verts.min(axis=0), verts.max(axis=0)
    s = height_lr / (y1 - y0)
    w = (x1 - x0) * s
    if w > cols - 2:
        s *= (cols - 2) / w
        w = cols - 2
    h = (y1 - y0) * s
    ox, oy = (cols - w) / 2.0, 1.5
    verts[:, 0] = (verts[:, 0] - x0) * s + ox
    verts[:, 1] = oy + (y1 - verts[:, 1]) * s  # flip y (image rows go down)
    path = Path(verts, tp.codes)
    polys = [Path(poly) for poly in path.to_polygons()]
    bbox = Roi(ox, oy, ox + w, oy + h)

    def inside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Even-odd rule over the glyph contours, so counters (holes in O, 4,
        # A, ...) stay at the background temperature.
        pts = np.column_stack([x.ravel(), y.ravel()])
        acc = np.zeros(pts.shape[0], dtype=bool)
        for poly in polys:
            acc ^= poly.contains_points(pts)
        return acc.reshape(x.shape)

    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)
    cv.paint(inside, hot, bbox)
    flat = [FlatRegion(roi=Roi(2.0, bbox.y1 + 2.5, cols - 2.0, rows - 1.5), value_c=background_c)]
    meta = cv.meta("text", background_c, flat=flat,
                   extra={"text": text, "text_roi": bbox, "contrast_k": contrast_k})
    return cv.scene(meta)


def textured_scene(*, contrast_k: float = 2.0, background_c: float = 25.0, seed: int = 0,
                   scales_lr: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
                   hr_factor: int = 8, margin_lr: int = 4,
                   shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """Smooth random texture: sum of Gaussian-filtered white noise at several
    scales (sigma in LR px), normalized so the HR image has standard deviation
    ``contrast_k`` (RMS contrast) and mean ``background_c``.
    """
    rng = np.random.default_rng(seed)
    rows = (shape_lr[0] + 2 * margin_lr) * hr_factor
    cols = (shape_lr[1] + 2 * margin_lr) * hr_factor
    acc = np.zeros((rows, cols))
    for s in scales_lr:
        layer = ndimage.gaussian_filter(rng.standard_normal((rows, cols)), s * hr_factor, mode="reflect")
        acc += layer / layer.std() * np.sqrt(s)  # more weight on coarse scales
    acc = (acc - acc.mean()) / acc.std()
    hr = background_c + contrast_k * acc
    meta = SceneMeta(kind="textured", shape_lr=shape_lr, hr_factor=hr_factor, margin_lr=margin_lr,
                     background_c=background_c, extra={"contrast_k": contrast_k, "seed": seed})
    return Scene(hr=hr, meta=meta, hr_factor=hr_factor, margin_lr=margin_lr)


def flat_scene(*, background_c: float = 25.0, hr_factor: int = 8, margin_lr: int = 4,
               shape_lr: tuple[int, int] = LR_SHAPE) -> Scene:
    """Uniform scene; the flat ROI is the whole array."""
    rows, cols = shape_lr
    cv = _Canvas(shape_lr, hr_factor, margin_lr, background_c)
    flat = [FlatRegion(roi=Roi(0.0, 0.0, float(cols), float(rows)), value_c=background_c)]
    return cv.scene(cv.meta("flat", background_c, flat=flat))


SCENE_KINDS: dict[str, Callable[..., Scene]] = {
    "chart": chart_scene,
    "edge": edge_scene,
    "bars": bars_scene,
    "people": people_scene,
    "text": text_scene,
    "textured": textured_scene,
    "low_contrast": lambda **kw: textured_scene(**{"contrast_k": 0.3, **kw}),
    "flat": flat_scene,
}


def make_scene(kind: str = "chart", **kwargs) -> Scene:
    """Build a scene by name (see module docstring); kwargs go to the generator.

    ``"low_contrast"`` is ``textured`` with ``contrast_k=0.3`` K RMS.
    """
    try:
        gen = SCENE_KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown scene kind {kind!r}; choose from {sorted(SCENE_KINDS)}") from None
    return gen(**kwargs)
