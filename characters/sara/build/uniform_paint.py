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
SCRIPT_FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"   # team wordmark
JERSEY_Y0, JERSEY_Z0, JERSEY_Z1 = 0.0, 0.86, 1.43
CHEST_Z = 1.2
_LM = {}
SHORTS_Y0, SHORTS_Z0, SHORTS_Z1 = 0.005, 0.74, 1.02
_REF_JERSEY_MM = 570.0        # print layout below is authored for a 570 mm jersey panel


def configure(bones):
    """Place the print canvases from the body landmarks (the prints follow the proportions)."""
    global JERSEY_Z0, JERSEY_Z1, SHORTS_Z0, SHORTS_Z1, CHEST_Z
    import body
    import garments as G
    lm = body.landmarks(bones)
    _LM.clear()
    _LM.update(lm)
    JERSEY_Z0, JERSEY_Z1 = G.jersey_canvas(lm)
    CHEST_Z = body.master(bones).bust_apex()[0]
    SHORTS_Z0, SHORTS_Z1 = lm["hip_z"] - 0.14, lm["hip_z"] + 0.14


def cyl_uv(p, y0, z0, z1):
    ang = math.atan2(p[0], -(p[1] - y0))
    return 0.5 + ang / (2 * math.pi), (p[2] - z0) / (z1 - z0)


def paint_jersey(path, pal, px=2048):
    """Sara's jersey print (front reference sara_jersey_front.png). Canvas: x = u * 1000 around the torso
    (500 = front center, > 500 = her left), y = mm above JERSEY_Z0. Everything is placed from the body's
    own landmarks (bust apex, waist, hem), so the print sits on the forms as in the art:
      navy upper front -> gradient to pale cyan below the bust | weasels wordmark over the upper bust |
      a large 2 just below the bust apex | round crest at her left strap | white sweeping arcs, sparkles
      and a halftone on the lower front | white-to-light-blue side panels (seam = side_seam_u, the same
      curve as the modeled piping) | back number between the shoulder blades | a fine mesh weave."""
    import garments as G
    H = (JERSEY_Z1 - JERSEY_Z0) * 1000
    k = H / _REF_JERSEY_MM
    Y = lambda z: (z - JERSEY_Z0) * 1000.0  # noqa: E731
    c = paint.Canvas(1000, H, px, (0, 0), supersample=2, background=(*paint.hex_rgb(pal["jersey_front_top"]), 1.0))
    X, Yg = c.grid_mm()
    zc = JERSEY_Z0 + Yg / 1000.0
    seam = G.side_seam_u(zc, _LM)
    front = (np.abs(X - 500) < seam).astype(np.float32)
    # Base colors: back (darker) everywhere, front gradient navy -> mid blue -> pale cyan -> near white.
    c.paint(np.ones_like(front), c.vertical_gradient(Y(_LM["chest_z"]), Y(_LM["hem_z"]), "#1a2e55", "#4c7aa3"), 1.0)
    top, mid, low, hem = pal["jersey_front_top"], "#3f7fb0", pal["jersey_front_bottom"], "#d4ecf7"
    g1 = c.vertical_gradient(Y(CHEST_Z - 0.018), Y(_LM["waist_z"] + 0.005), top, mid)
    g2 = c.vertical_gradient(Y(_LM["waist_z"] + 0.005), Y(_LM["hem_z"] + 0.09), mid, low)
    g3 = c.vertical_gradient(Y(_LM["hem_z"] + 0.09), Y(_LM["hem_z"]), low, hem)
    zsplit1, zsplit2 = _LM["waist_z"] + 0.005, _LM["hem_z"] + 0.09
    grad = np.where((zc > zsplit1)[..., None], g1, np.where((zc > zsplit2)[..., None], g2, g3))
    c.paint(front, grad, 1.0)
    # Side panels (torso sides, armpit down): white at the top to light blue at the hem, feathered at the
    # armpit (above it the strap band is a 3D attribute along the armhole). Back seam at 305 from center.
    armpit = _LM["shoulder_z"] - 0.07
    panel = ((np.abs(X - 500) >= seam) & (np.abs(X - 500) < 305 - 0.00008 * ((430 * k - Yg) / k) ** 2)).astype(np.float32)
    panel *= np.clip((Y(armpit) - Yg) / 12.0, 0.0, 1.0)
    c.paint(panel, c.vertical_gradient(Y(armpit), Y(_LM["hem_z"]), pal["jersey_side"], "#b9d6ea"), 1.0)
    # Lower-front graphic: thin white arcs sweeping from her lower right up toward her left side.
    arcs = [([(330, 0.975), (430, 1.02), (540, 1.075), (640, 1.14), (700, 1.19)], 3.0),
            ([(340, 0.955), (450, 0.992), (560, 1.042), (660, 1.098), (706, 1.132)], 2.0),
            ([(430, 0.932), (520, 0.957), (610, 0.998), (692, 1.052)], 3.4),
            ([(298, 1.035), (360, 1.0), (420, 0.965), (480, 0.94)], 2.4),
            ([(560, 0.93), (640, 0.966), (702, 1.012)], 1.8)]
    zr = _LM["hem_z"] + 0.02, _LM["waist_z"] + 0.1                    # authored for hem..waist of the reference body
    polys = []
    for pts, w in arcs:
        q = [(x, Y(zr[0] + (z - 0.936) / (1.2 - 0.936) * (zr[1] - zr[0]))) for x, z in pts]
        polys.append(c.stroke_polygon(q, [0.2, w, w, w * 0.8, 0.2], samples=12))
    c.paint(c.mask_polygons(polys, blur_mm=0.3) * front, "#f4fbff", 0.9)
    # Halftone toward the hem: dots growing downward (white on the pale cyan).
    zt = _LM["hem_z"] + 0.12
    t = np.clip((Y(zt) - Yg) / (Y(zt) - Y(_LM["hem_z"])), 0.0, 1.0)
    gx, gy = (X % 7.0) - 3.5, (Yg % 6.0) - 3.0
    dots = (np.sqrt((gx * 0.85) ** 2 + gy ** 2) < 0.4 + 1.5 * t).astype(np.float32) * (t > 0.02)
    c.paint(dots * front, "#ffffff", 0.32)
    # Sparkles (4-point stars) and a thin ring around the big one, on her left of the stomach.
    zs = _LM["waist_z"] + 0.012
    for sx_, dz_, r_ in ((561, 0.0, 17.0), (490, 0.018, 10.0), (620, 0.05, 6.0)):
        cy = Y(zs + dz_)
        arms = []
        for ang in (0.0, math.pi / 2, math.pi, 1.5 * math.pi):
            tip = (sx_ + 1.25 * r_ * math.cos(ang), cy + r_ * math.sin(ang))
            arms.append(c.stroke_polygon([(sx_, cy), tip], [r_ * 0.2, 0.02], samples=8))
        c.paint(c.mask_polygons(arms, blur_mm=0.25) * front, "#ffffff", 0.97)
    ring = [(561 + 1.25 * 33 * math.cos(a_), Y(zs) + 33 * math.sin(a_)) for a_ in np.linspace(0, 2 * math.pi, 72)]
    c.paint(c.mask_stroke(ring, [1.4] * len(ring), blur_mm=0.25) * front, "#ffffff", 0.85)
    # Chest: wordmark over the upper bust, the number just below the apex (front reference hierarchy).
    wz, nz_ = Y(CHEST_Z + 0.052), Y(CHEST_Z - 0.024)
    c.text("weasels", 500, wz, 54, "#f4f8fc", SCRIPT_FONT, stroke_mm=4.2, stroke_color="#9ccbec")
    c.text("weasels", 500, wz, 54, "#f7fbff", SCRIPT_FONT, stroke_mm=1.5, stroke_color="#13254a")
    c.text("2", 506, nz_ - 4, 84, "#7fb4de", FONT)                                       # soft blue offset shadow
    c.text("2", 500, nz_, 84, pal["jersey_print"], FONT, stroke_mm=1.6, stroke_color="#13254a")
    # Round crest near her left strap.
    cx_, cy_ = 586, Y(_LM["neck_z"] - 0.02)
    c.paint(c.mask_ellipse(cx_, cy_, 20, 14, blur_mm=0.3), "#7fb0dc", 1.0)
    c.paint(c.mask_ellipse(cx_, cy_, 17, 12, blur_mm=0.3), "#dcecf8", 1.0)
    star = [(cx_ + 9 * (1.0 if i % 2 == 0 else 0.42) * math.cos(math.pi / 2 + i * math.pi / 5),
             cy_ + 6.5 * (1.0 if i % 2 == 0 else 0.42) * math.sin(math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    c.paint(c.mask_polygon(star, blur_mm=0.2), "#3f78b8", 1.0)
    # Back number between the shoulder blades (the back seam is at u = 0 / 1).
    bz = Y(_LM["chest_z"] + 0.005)
    for bx in (0.0, 1000.0):
        c.text("2", bx, bz, 175, pal["jersey_print"], FONT, stroke_mm=2.0, stroke_color="#13254a")
    # Fine sports-mesh weave everywhere (reads up close, disappears at game distance).
    weave = ((np.sin(X * 2.6) * np.sin(Yg * 3.1)) > 0.55).astype(np.float32)
    c.paint(weave, "#000000", 0.035)
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
    # Small "2" low on the outside front of her left leg (front reference).
    c.text("2", 500 + 175, 48, 42, pal["jersey_print"], FONT)
    c.save(path, alpha=False)
    log("painted", path)
    return path
