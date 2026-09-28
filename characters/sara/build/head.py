"""Sara: head, face and eyes.

Identity notes driving the numbers (characters/sara/character.json face.design_notes):
soft oval with a small pointed chin, cheeks fuller than the jaw suggests,
eye line at ~51% of head height, large but vertically shallow eyes under a
heavy upper lid, thick upper lash line with a slight outer wing, thin broken
lower lash line, thin straight brows angled toward the nose, minimal nose
with a pink tip, small mouth parted in an exhale.
"""
from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
import numpy as np

from spike_pipeline.modeling import head as headlib
from spike_pipeline.modeling import paint

from common import CHAR, log

HS = 1.12                               # head scale from the art-direction pass (7.9 heads with hair)
EYE_X, EYE_Z = 0.039, 1.5461            # painted eye centers (eye spacing 78 mm); LeftEye bone height
CHIN_Z = EYE_Z - 0.078 * HS             # acorn face: eye line to chin ~1.1x the eye spacing
CROWN_Z = EYE_Z + 0.089 * HS            # top of the skull (hair adds ~2 cm)
NOSE_T, MOUTH_T = 0.299, 0.168          # nose tip and lip line, fraction of skull height from the chin
AXIS_Y = 0.012
EYE_SCALE = 1.30 * HS                   # painted eye design mm -> face mm
EYE_UV_SIZE = 0.05 * EYE_SCALE          # the eye decal texture covers the 50 x 50 design-mm canvas
FACE_TEX_ORIGIN = (-0.11, CHIN_Z - 0.01)  # face projection: x -0.11..0.11, z chin-1cm .. +27cm
FACE_TEX_SIZE = (0.22, 0.27)


def profile():
    """Acorn-shaped head: round, wide skull and cheeks, quick taper to a small pointed chin."""
    t = [0.00, 0.05, 0.10, 0.16, 0.22, 0.28, 0.34, 0.42, 0.50, 0.60, 0.70, 0.80, 0.88, 0.94, 1.00]
    Y = lambda vals: [AXIS_Y + (v - AXIS_Y) * HS for v in vals]
    W = lambda vals: [v * HS for v in vals]
    return headlib.HeadProfile(
        CHIN_Z, CROWN_Z, t,
        y_front=Y([-0.064, -0.071, -0.077, -0.080, -0.082, -0.0835, -0.0845, -0.085, -0.085, -0.084, -0.080, -0.072, -0.060, -0.043, -0.010]),
        y_back=Y([-0.046, -0.032, -0.014, 0.006, 0.028, 0.052, 0.075, 0.093, 0.103, 0.108, 0.107, 0.100, 0.089, 0.070, 0.027]),
        half_width=W([0.011, 0.027, 0.041, 0.053, 0.062, 0.069, 0.0745, 0.0785, 0.0805, 0.0815, 0.0815, 0.079, 0.072, 0.058, 0.022]),
        y_wide=Y([-0.054, -0.050, -0.043, -0.033, -0.021, -0.009, 0.0, 0.006, 0.009, 0.011, 0.012, 0.012, 0.012, 0.012, 0.010]),
        n_front=[2.0, 2.1, 2.3, 2.5, 2.7, 2.85, 2.9, 2.9, 2.85, 2.65, 2.45, 2.25, 2.1, 2.0, 2.0],
        n_back=[2.0] * 15,
    )


def features(p):
    te = p.t_of(EYE_Z)
    k = HS
    return [
        headlib.Feature(0.0, NOSE_T, 0.0046 * k, 0.0042 * k, 0.034),       # nose tip (tiny)
        headlib.Feature(0.0, NOSE_T + 0.07, 0.0016 * k, 0.0035 * k, 0.07),  # nose bridge
        headlib.Feature(0.0, NOSE_T - 0.035, -0.0011 * k, 0.006 * k, 0.013),  # under-nose recess
        headlib.Feature(0.0, MOUTH_T + 0.022, 0.001 * k, 0.011 * k, 0.013),  # upper lip
        headlib.Feature(0.0, MOUTH_T, -0.0008 * k, 0.008 * k, 0.009),       # lip line
        headlib.Feature(0.0, MOUTH_T - 0.025, 0.0007 * k, 0.009 * k, 0.013),  # lower lip
        headlib.Feature(0.0, 0.06, 0.0012 * k, 0.01 * k, 0.035),            # chin point
        headlib.Feature(EYE_X, te, -0.0014 * k, 0.014 * k, 0.055, mirror=True),     # soft socket
        headlib.Feature(0.054, 0.33, 0.0018 * k, 0.018 * k, 0.08, mirror=True),     # cheek fullness
        headlib.Feature(0.03, te + 0.09, 0.0008 * k, 0.02 * k, 0.035, mirror=True),  # brow plane
    ]


def build_head_mesh(coll_name="sara_GAME_LOD0"):
    p = profile()
    verts, quads, tris, grid, front = headlib.loft(p, features(p), rings=96, segments=128)
    bm = bmesh.new()
    bv = [bm.verts.new(v) for v in verts]
    for q in quads:
        bm.faces.new([bv[i] for i in q])
    for t in tris:
        bm.faces.new([bv[i] for i in t])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        for loop in f.loops:
            co = loop.vert.co
            loop[uv].uv = ((co.x - FACE_TEX_ORIGIN[0]) / FACE_TEX_SIZE[0], (co.z - FACE_TEX_ORIGIN[1]) / FACE_TEX_SIZE[1])
    me = bpy.data.meshes.new("sara_face_LOD0")
    bm.to_mesh(me)
    bm.free()
    for poly in me.polygons:
        poly.use_smooth = True
    obj = bpy.data.objects.new("sara_face_LOD0", me)
    _link(obj, coll_name)
    # Back of the head must never show face features: store a front mask attribute for the shader.
    attr = me.attributes.new("front", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.asarray(front, dtype=np.float32))
    return obj, p, grid


def build_eye_decals(face_obj, p, coll_name="sara_GAME_LOD0"):
    """Per-eye surface patches cloned from the face, offset forward, with a local eye UV."""
    decals = {}
    for side, sx in (("L", 1.0), ("R", -1.0)):
        for layer, offset in (("eye", 0.00035), ("lash", 0.0007)):
            bm = bmesh.new()
            bm.from_mesh(face_obj.data)
            keep = []
            for f in bm.faces:
                c = f.calc_center_median()
                if (sx * c.x > 0.004 and abs(c.x - sx * EYE_X) < 0.036 and abs(c.z - EYE_Z) < 0.03 and c.y < -0.045):
                    keep.append(f)
            kill = [f for f in bm.faces if f not in set(keep)]
            bmesh.ops.delete(bm, geom=kill, context="FACES")
            bm.normal_update()
            for v in bm.verts:
                v.co += v.normal * offset
            uv_face = bm.loops.layers.uv.get("UVMap")
            uv_eye = bm.loops.layers.uv.new("eye_uv")
            for f in bm.faces:
                for loop in f.loops:
                    co = loop.vert.co
                    loop[uv_eye].uv = (0.5 + (sx * co.x - EYE_X) / EYE_UV_SIZE, 0.5 + (co.z - EYE_Z) / EYE_UV_SIZE)
            name = f"sara_{layer}_{side}_LOD0"
            me = bpy.data.meshes.new(name)
            bm.to_mesh(me)
            bm.free()
            for poly in me.polygons:
                poly.use_smooth = True
            me.uv_layers.active = me.uv_layers["eye_uv"]
            obj = bpy.data.objects.new(name, me)
            _link(obj, coll_name)
            decals[(layer, side)] = obj
    return decals


def build_ears(coll_name="sara_GAME_LOD0"):
    objs = []
    for sx in (1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, radius=1.0)
        o = bpy.context.active_object
        o.name = f"sara_ear_{'L' if sx > 0 else 'R'}_LOD0"
        o.scale = (0.0055 * HS, 0.012 * HS, 0.019 * HS)
        o.location = (sx * 0.0755 * HS, 0.012 + 0.005 * HS, EYE_Z - 0.019)
        o.rotation_euler = (math.radians(-12), math.radians(sx * -8), math.radians(sx * 10))
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        for poly in o.data.polygons:
            poly.use_smooth = True
        _link(o, coll_name)
        objs.append(o)
    return objs


def _link(obj, coll_name):
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)
    for c in obj.users_collection:
        c.objects.unlink(obj)
    coll.objects.link(obj)


# ---------------------------------------------------------------------------
# Painting (units: mm)
# ---------------------------------------------------------------------------
def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def paint_face(path, px=2048):
    ox, oz = FACE_TEX_ORIGIN[0] * 1000, FACE_TEX_ORIGIN[1] * 1000
    c = paint.Canvas(FACE_TEX_SIZE[0] * 1000, FACE_TEX_SIZE[1] * 1000, px, (ox, oz), supersample=2,
                     background=(*paint.hex_rgb("#fbe6e8"), 1.0))
    ez = EYE_Z * 1000
    k = EYE_SCALE
    for sx in (1, -1):
        ex = sx * EYE_X * 1000
        # Blush: soft oval + short hatch strokes (anime blush).
        c.paint(c.mask_gauss(sx * 46.0 * HS, ez - 24.0 * HS, 11.0 * HS, 4.8 * HS, angle_deg=sx * -8), "#f19cae", 0.5)
        polys = [c.stroke_polygon([(sx * (39.0 + i * 3.8) * HS, ez - 21.5 * HS), (sx * (37.5 + i * 3.8) * HS, ez - 26.0 * HS)],
                                  [0.55, 0.22]) for i in range(4)]
        c.paint(c.mask_polygons(polys, blur_mm=0.08), "#e8889b", 0.38)
        # Pink flush around the eye: under the lower lid and at the outer corner, like the reference close-up.
        c.paint(c.mask_gauss(ex + sx * 2.0, ez - 10.2 * k, 12.5 * k, 3.0 * k), "#f3b3c2", 0.55)
        c.paint(c.mask_gauss(ex + sx * 15.0 * k, ez + 1.0 * k, 3.5 * k, 4.0 * k), "#f0a9ba", 0.35)
        c.paint(c.mask_gauss(ex + sx * 1.0, ez + 11.5 * k, 11.0 * k, 2.2 * k), "#f5c6cf", 0.35)   # upper lid tint
        # Brows: thin, dark, straight, inner end lower (focus).
        brow = [(ex - sx * 13.0 * k, ez + 14.6 * k), (ex - sx * 2.5 * k, ez + 16.8 * k), (ex + sx * 8.5 * k, ez + 17.4 * k),
                (ex + sx * 15.5 * k, ez + 15.4 * k)]
        c.paint(c.mask_stroke(brow, [1.3, 1.75, 1.35, 0.3], blur_mm=0.06), "#2a1d24", 0.97)
    # Nose: tiny shadow tick + a soft pink bridge flush (the reference's nose is almost only shadow).
    nz = (CHIN_Z + NOSE_T * (CROWN_Z - CHIN_Z)) * 1000
    c.paint(c.mask_gauss(0.0, nz + 9.0, 3.0, 9.0), "#f4bcc7", 0.35)
    c.paint(c.mask_gauss(0.3, nz + 0.4, 1.8, 1.2), "#eea3b1", 0.55)
    c.paint(c.mask_stroke([(-1.9, nz + 1.4), (-1.2, nz - 0.2), (0.4, nz - 0.6)], [0.2, 0.5, 0.15], blur_mm=0.08), "#b87a88", 0.7)
    # Mouth: small, parted exhale with a hint of upper teeth.
    mz = (CHIN_Z + MOUTH_T * (CROWN_Z - CHIN_Z)) * 1000
    mouth = [(x * 0.95 * HS, mz + dy * HS) for x, dy in ((-4.4, 0.4), (-2.2, 1.5), (0.0, 1.7), (2.3, 1.45), (4.3, 0.3),
                                                     (2.3, -1.9), (0.0, -2.4), (-2.3, -1.85))]
    mouth_poly = paint.catmull(mouth, 12, closed=True)
    m_mask = c.mask_polygon(mouth_poly, blur_mm=0.07)
    c.paint(m_mask, "#6e3040", 0.96)
    c.paint(c.mask_gauss(0.0, mz - 1.3, 2.4, 0.9) * m_mask, "#b0566a", 0.75)            # tongue
    teeth = c.mask_polygon(paint.catmull([(-3.0, mz + 0.95), (0.0, mz + 1.45), (3.0, mz + 0.85), (2.3, mz + 0.1),
                                          (0.0, mz + 0.3), (-2.3, mz + 0.12)], 10, closed=True), blur_mm=0.05)
    c.paint(teeth * m_mask, "#fbf6f7", 0.95)
    c.paint(c.mask_stroke([(-4.3, mz + 0.5), (-1.8, mz + 1.75), (1.8, mz + 1.7), (4.2, mz + 0.4)], [0.15, 0.36, 0.36, 0.15],
                          blur_mm=0.1), "#94505f", 0.55)                                   # upper lip line
    c.paint(c.mask_gauss(0.0, mz - 4.6, 3.4, 1.1), "#f2b6c0", 0.45)                     # lower lip light
    c.save(path, alpha=False)
    log("painted", path)
    return path


# Eye-local design coordinates: origin at the eye center, +x toward the OUTER corner, +y up (design mm,
# scaled by EYE_SCALE on the face).
INNER = (-15.2, 0.4)
OUTER = (16.6, 2.2)
UPPER_LID = [INNER, (-12.0, 6.4), (-5.0, 10.2), (2.5, 10.8), (10.0, 9.1), (15.0, 5.8), OUTER]
LOWER_LID = [INNER, (-11.5, -4.6), (-4.0, -8.7), (3.5, -9.2), (10.5, -6.8), OUTER]
CORNER_Y = 1.3
IRIS_C = (0.6, -0.4)
IRIS_R = (8.9, 10.0)


def _opening_polygon():
    return paint.catmull(UPPER_LID, 14) + paint.catmull(LOWER_LID[::-1], 14)


def paint_eye_white(path, px=2048):
    """RGB = sclera shading; A = the eye opening mask."""
    c = paint.Canvas(50, 50, px, (-25, -25), supersample=2, background=(*paint.hex_rgb("#f7f6fc"), 0.0))
    opening = c.mask_polygon(_opening_polygon(), blur_mm=0.1)
    c.paint(np.ones_like(opening), c.vertical_gradient(10.0, 0.5, "#9aa4c4", "#f7f6fc"), 1.0)   # lid shadow
    c.paint(c.mask_gauss(-14.0, 0.2, 2.2, 1.8), "#f0a9b8", 0.8)       # inner corner
    c.paint(c.mask_gauss(15.2, 1.8, 2.4, 1.6), "#efb9c5", 0.55)       # outer corner
    # Pink waterline along the lower lid.
    lower = paint.catmull(LOWER_LID, 20)
    wl = c.mask_stroke([(x, y + 0.55) for x, y in lower[4:-4:2]], [0.3, 0.9, 1.0, 0.9, 0.3], blur_mm=0.25)
    c.paint(wl, "#f2b4c3", 0.8)
    c.rgba[..., 3] = opening
    c.save(path)
    log("painted", path)
    return path


def paint_iris(path, px=2048, palette=None, mirror_highlights=False):
    """Anime iris after the reference close-up: cobalt outer ring with a jagged inner edge, bright cyan
    field brightening to pale aqua at the bottom, a dark cog ring around the pupil, radial streaks, lid
    shadow, twin white pupil dots, a warm reflection, and the character's catchlights."""
    pal = palette or {}
    c = paint.Canvas(50, 50, px, (-25, -25), supersample=2, background=(0, 0, 0, 0))
    cx, cy = IRIS_C
    rx, ry = IRIS_R
    r, th = c.polar(cx, cy, rx, ry)                    # r = 1 on the iris edge
    iris = _smoothstep(1.0, 0.985, r)
    _, Y = c.grid_mm()
    tt = np.clip((cy + ry - Y) / (2 * ry), 0, 1)[..., None]      # 0 top, 1 bottom
    top, mid, bot = paint.hex_rgb(pal.get("iris_top", "#1d4aa6")), paint.hex_rgb(pal.get("iris_mid", "#43c3e4")), \
        paint.hex_rgb(pal.get("iris_bottom", "#bdf7f7"))
    field = np.where(tt < 0.5, top * (1 - tt / 0.5) + mid * (tt / 0.5), mid * (1 - (tt - 0.5) / 0.5) + bot * ((tt - 0.5) / 0.5))
    c.paint(iris, field, 1.0)
    rng = np.random.default_rng(11)
    phases = rng.uniform(0, 2 * np.pi, 3)
    jag = lambda f, a: np.sin(th * f + phases[0]) * a + np.sin(th * (f * 1.7) + phases[1]) * a * 0.5
    # Radial light streaks through the cyan field.
    streak = np.clip(np.sin(th * 46 + np.sin(th * 7) * 0.8) * 0.5 + 0.5, 0, 1) ** 6
    band = _smoothstep(0.45, 0.55, r) * _smoothstep(0.92, 0.8, r)
    c.paint(streak * band * iris, "#dcfdff", 0.35)
    # Outer cobalt ring, jagged inner edge.
    outer_edge = 0.87 + jag(26, 0.025)
    c.paint(_smoothstep(outer_edge - 0.015, outer_edge + 0.015, r) * iris, pal.get("iris_ring", "#2552bd"), 0.95)
    # Dark cog ring around the pupil.
    cog_outer = 0.41 + np.clip(jag(18, 0.06), -0.02, 0.08)
    cog = _smoothstep(cog_outer + 0.015, cog_outer - 0.015, r) * _smoothstep(0.25, 0.29, r)
    c.paint(cog * iris, pal.get("iris_cog", "#1c3a92"), 0.72)
    # Bright lower glow and lid shadow over the top.
    glow = c.mask_ellipse(cx, cy - ry * 0.42, rx * 0.62, ry * 0.36, blur_mm=1.2)
    c.paint(glow * iris * _smoothstep(0.35, 0.5, r), "#e6ffff", 0.7)
    shadow = np.clip((Y - (cy + ry * 0.1)) / (ry * 0.8), 0, 1)
    c.paint(shadow * iris, "#1a3f7a", 0.45)
    # Limbal edge line.
    c.paint(_smoothstep(0.95, 0.985, r) * iris, "#16285c", 0.95)
    # Pupil: slightly tall, soft edge.
    pr, _ = c.polar(cx, cy + 0.3, rx * 0.22, ry * 0.32)
    c.paint(_smoothstep(1.0, 0.94, pr), pal.get("pupil", "#141b2b"), 1.0)
    hs = -1.0 if mirror_highlights else 1.0
    # Twin white dots at the top of the pupil and a warm reflection beside it.
    for dx, dy, rr, a in ((0.5, 2.9, 0.62, 1.0), (1.75, 3.05, 0.42, 0.95)):
        c.paint(c.mask_ellipse(cx + hs * dx, cy + dy, rr, rr * 0.9, blur_mm=0.05), "#ffffff", a)
    c.paint(c.mask_ellipse(cx - hs * 2.3, cy + 0.8, 0.75, 0.62, blur_mm=0.15), "#ff9fb0", 0.8)
    # Character catchlights (iris-relative 0..1, v from the top).
    for cl in (pal.get("catchlights") or []):
        u, v = cl["pos"]
        if mirror_highlights:
            u = 1.0 - u
        hx = cx + (u - 0.5) * 2 * rx
        hy = cy + (0.5 - v) * 2 * ry
        rr = cl["size"] * rx * 1.4
        c.paint(c.mask_ellipse(hx, hy, rr * 0.7, rr * 0.55, angle_deg=-25 * hs, blur_mm=0.35), "#ffffff",
                0.55 * min(1.0, cl["intensity"]))
    c.save(path)
    log("painted", path)
    return path


def paint_lash(path, px=2048):
    """Upper lash line with individual lashes, crease, lower lashes. Alpha texture (upper half = upper lid)."""
    c = paint.Canvas(50, 50, px, (-25, -25), supersample=2, background=(*paint.hex_rgb("#211a20"), 0.0))
    col = "#1b1418"
    upper = paint.catmull(UPPER_LID, 20)
    n = len(upper)
    # Main line: fine at the inner corner, heavy and sharp over the outer third, outward wing.
    line_pts = [(INNER[0] + 0.3, INNER[1] + 0.5)] + UPPER_LID[1:-1] + [(OUTER[0] + 0.3, OUTER[1] + 0.5), (20.8, 2.9)]
    polys = [c.stroke_polygon(line_pts, [0.45, 1.0, 1.45, 1.85, 2.5, 2.9, 2.3, 0.25])]
    # Upper lashes: clumped, long toward the outer corner, sweeping outward.
    rng = np.random.default_rng(5)
    for i in range(14):
        u = 0.32 + 0.66 * (i / 13.0) ** 0.75
        j = min(int(u * (n - 1)), n - 2)
        x, y = upper[j]
        x2, y2 = upper[j + 1]
        tx, ty = x2 - x, y2 - y
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / ln, tx / ln
        length = 1.8 + 4.2 * u ** 1.6 + rng.uniform(-0.3, 0.3)
        lean = 0.45 + 1.1 * u
        p1 = (x + nx * length * 0.5 + tx / ln * lean * length * 0.3, y + ny * length * 0.5 + 0.25)
        p2 = (x + nx * length * 0.9 + tx / ln * lean * length * 0.8, y + ny * length * 0.75 + 0.1)
        polys.append(c.stroke_polygon([(x + nx * 0.4, y + ny * 0.4), p1, p2], [0.8, 0.42, 0.04], samples=8))
    for (x0, y0), (x1, y1), w in (((16.4, 3.6), (21.8, 4.6), 0.75), ((17.2, 2.8), (22.4, 2.2), 0.6), ((15.2, 5.2), (19.6, 7.6), 0.7)):
        polys.append(c.stroke_polygon([(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2 + 0.5), (x1, y1)], [w, w * 0.6, 0.05], samples=10))
    c.paint(c.mask_polygons(polys, blur_mm=0.035), col, 1.0)
    # Crease (double eyelid) over the middle/outer part.
    crease = [(-4.0, 13.6), (2.5, 14.4), (9.5, 12.9), (15.0, 9.8)]
    c.paint(c.mask_stroke(crease, [0.12, 0.38, 0.34, 0.06], blur_mm=0.08), "#7d4c5a", 0.85)
    # Lower lid line and lower lashes: continuous over the outer 70%, darker toward the corner.
    lower = paint.catmull(LOWER_LID, 20)
    m = len(lower)
    polys = [c.stroke_polygon([(x, y - 0.25) for x, y in lower[int(0.3 * m):int(0.97 * m):2]], [0.12, 0.35, 0.5, 0.62, 0.3],
                              samples=6)]
    for i in range(12):
        u = 0.45 + 0.5 * i / 11.0
        x, y = lower[int(u * (m - 1))]
        polys.append(c.stroke_polygon([(x, y - 0.3), (x + 0.6 + 0.9 * u, y - 1.2 - 1.3 * u)], [0.28, 0.03], samples=6))
    c.paint(c.mask_polygons(polys, blur_mm=0.04), "#3a2a33", 0.92)
    c.save(path)
    log("painted", path)
    return path
