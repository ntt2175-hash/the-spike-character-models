"""Sara: uniform prints, painted in cylindrical (angle, height) space.

u = 0.5 + atan2(x, -(y - Y0)) / 2pi  (0.5 = front center, u > 0.5 = her left side)
v = (z - Z0) / (Z1 - Z0)
"""
from __future__ import annotations

import math

import numpy as np

from spike_pipeline.modeling import paint

from common import log

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
JERSEY_Y0, JERSEY_Z0, JERSEY_Z1 = 0.0, 0.86, 1.43
SHORTS_Y0, SHORTS_Z0, SHORTS_Z1 = 0.005, 0.74, 1.02
_REF_JERSEY_MM = 570.0        # print layout below is authored for a 570 mm jersey panel


def configure(bones):
    """Place the print canvases from the body landmarks (the prints follow the proportions)."""
    global JERSEY_Z0, JERSEY_Z1, SHORTS_Z0, SHORTS_Z1
    import body
    lm = body.landmarks(bones)
    JERSEY_Z0, JERSEY_Z1 = lm["hem_z"] - 0.07, lm["neck_z"] + 0.083
    SHORTS_Z0, SHORTS_Z1 = lm["hip_z"] - 0.14, lm["hip_z"] + 0.14


def cyl_uv(p, y0, z0, z1):
    ang = math.atan2(p[0], -(p[1] - y0))
    return 0.5 + ang / (2 * math.pi), (p[2] - z0) / (z1 - z0)


def paint_jersey(path, pal, px=2048):
    """Canvas units: u * 1000 horizontally, height in mm vertically (570 mm)."""
    H = (JERSEY_Z1 - JERSEY_Z0) * 1000
    k = H / _REF_JERSEY_MM
    c = paint.Canvas(1000, H, px, (0, 0), supersample=2, background=(*paint.hex_rgb(pal["jersey_front_top"]), 1.0))
    # Front/back body: vertical gradient, dark at the chest to pale at the hem (front brighter than back).
    grad_front = c.vertical_gradient(430 * k, 60 * k, pal["jersey_front_top"], pal["jersey_front_bottom"])
    grad_back = c.vertical_gradient(430 * k, 60 * k, "#1c3050", "#5f8aa8")
    X, Y = c.grid_mm()
    front = ((X > 300) & (X < 700)).astype(np.float32)
    c.paint(np.ones_like(front), grad_back, 1.0)
    c.paint(front, grad_front, 1.0)
    # White side panels: armpit to hem, flaring slightly toward the hem; royal-blue piping on the front seam.
    for side in (-1, 1):
        def edge(y, inner):
            base = 500 + side * (205 if inner else 305)
            yy = (430 * k - y) / k
            return base + side * (0.00018 * yy ** 2 if inner else -0.00008 * yy ** 2)
        ys = np.linspace(-5, H + 5, 40)
        inner = [(edge(y, True), y) for y in ys]
        outer = [(edge(y, False), y) for y in ys[::-1]]
        c.paint(c.mask_polygon(inner + outer, blur_mm=0.4), pal["jersey_side"], 1.0)
        c.paint(c.mask_stroke(inner[::3], [4.5] * len(inner[::3]), blur_mm=0.3), pal["jersey_piping"], 1.0)
    # Wind-swirl print on the lower front: long streams sweeping up toward her left side, curling
    # into a couple of open spirals (canvas x units are ~0.8 mm, y units are mm).
    rng = np.random.default_rng(12)
    polys = []
    for j in range(7):
        y0 = (75 + j * 15 + rng.uniform(-6, 6)) * k    # above the hem
        pts = []
        for i in range(24):
            t = i / 23.0
            x = 300 + 420 * t
            y = y0 + (150 * t ** 1.6 + 14 * math.sin(t * 5.0 + j)) * k
            pts.append((x, y))
        w = rng.uniform(2.2, 4.8)
        polys.append(c.stroke_polygon(pts[::2], [0.4, w, w, w * 0.8, 0.4], samples=10))
    for cx, cy, r0 in ((470, 175, 40), (585, 225, 28), (395, 125, 24)):
        cy *= k
        spiral = [(cx + (r0 * (1 - 0.7 * t)) * 1.25 * math.cos(6.0 * t + 0.6), cy + r0 * k * (1 - 0.7 * t) * math.sin(6.0 * t + 0.6))
                  for t in np.linspace(0.0, 1.0, 30)]
        polys.append(c.stroke_polygon(spiral[::2], [3.8, 3.2, 2.2, 0.4], samples=10))
    swirl = c.mask_polygons(polys, blur_mm=0.35) * front
    c.paint(swirl, pal["jersey_print"], 0.75)
    # Numbers: front (her right of center, chest) and back (centered, larger).
    c.text("2", 452, 330 * k, 155 * k, pal["jersey_print"], FONT, stroke_mm=6, stroke_color="#1a2a48")
    c.text("2", 1000 * 0.02, 260 * k, 190 * k, pal["jersey_print"], FONT, stroke_mm=6, stroke_color="#1a2a48")
    c.text("2", 1000 * 0.98 + 40, 260 * k, 190 * k, pal["jersey_print"], FONT, stroke_mm=6, stroke_color="#1a2a48")
    c.save(path, alpha=False)
    log("painted", path)
    return path


def paint_shorts(path, pal, px=1024):
    H = (SHORTS_Z1 - SHORTS_Z0) * 1000
    c = paint.Canvas(1000, H, px, (0, 0), supersample=2, background=(*paint.hex_rgb(pal["shorts_base"]), 1.0))
    c.paint(np.ones((c.H, c.W), np.float32), c.vertical_gradient(200, 20, pal["shorts_base"], pal["shorts_gradient"]), 0.8)
    for side in (-1, 1):
        cx = 500 + side * 250
        ys = np.linspace(-5, H + 5, 30)
        w = lambda y: 20 + 0.22 * max(H - y, 0.0)
        left = [(cx - w(y), y) for y in ys]
        right = [(cx + w(y), y) for y in ys[::-1]]
        c.paint(c.mask_polygon(left + right, blur_mm=0.4), pal["shorts_side"], 1.0)
        front_edge = left if side > 0 else right[::-1]
        c.paint(c.mask_stroke(front_edge[::3], [4.0] * len(front_edge[::3]), blur_mm=0.3), pal["jersey_piping"], 1.0)
    # Small "2" on the right-leg side panel.
    c.text("2", 500 - 250, 70, 45, pal["jersey_print"], FONT)
    c.save(path, alpha=False)
    log("painted", path)
    return path
