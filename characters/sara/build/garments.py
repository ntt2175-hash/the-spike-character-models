"""Sara: uniform, legwear and shoes as real garments (shells with thickness), not paint.

Jersey : sleeveless cap-shoulder cut, deep armholes, crew neck with a slight front scoop, hem at the
         upper hip. Follows the slender body with light ease; never a wide rectangular block.
Shorts : fitted, short inseam, leg openings slightly higher on the outer side.
Pads   : sleeve knee pads, thick padded front, blue elastic bands at both edges.
Legwear: LEFT black knee-high (top hidden under the pad); RIGHT low ankle sock.
Every garment is placed from the body master's landmarks (body.landmarks) and sized by the same
proportion parameters, so it follows the body when proportions change.
Shoes  : low-mid volleyball shoes: outsole, midsole, mesh upper, heel counter, side stripe,
         tongue and criss-cross laces, as separate material zones.
"""
from __future__ import annotations

import math

import numpy as np

from spike_pipeline.modeling import sdf

import body as bodymod


def _v(*a):
    return np.array(a, dtype=np.float64)


def _bone(bones, n):
    return _v(*bones[n]["head"]), _v(*bones[n]["tail"])


# ---------------------------------------------------------------------------
# cut shapes (also used to paint trims by proximity)
# ---------------------------------------------------------------------------
def jersey_cuts(bones):
    lm = bodymod.landmarks(bones)
    sc, kt, nz = lm["s"], lm["kt"], lm["neck_z"]
    ax, az = lm["shoulder_x"] + 0.021 * sc, lm["shoulder_z"] - 0.082 * sc * kt
    return {
        # Crew neck that dips at the front: one tilted opening, so the edge is a single clean line.
        "neck": [sdf.Ellipsoid(_v(0.0, -0.012 * sc, nz + 0.058 * sc), _v(0.058, 0.085 * lm["torso_half_depth"] / (0.086 * sc), 0.07) * sc,
                               sdf.rotation_to(_v(0.0, 0.42, 1.0)))],
        "arm_L": [sdf.Ellipsoid(_v(ax, 0.004, az), _v(0.07, 0.1, 0.1 * kt) * sc, sdf.rotation_to(_v(0.25, 0.0, 1.0)))],
        "arm_R": [sdf.Ellipsoid(_v(-ax, 0.004, az), _v(0.07, 0.1, 0.1 * kt) * sc, sdf.rotation_to(_v(-0.25, 0.0, 1.0)))],
        # Hem at the upper hip (key art): the shorts show below it; a touch lower at the back.
        "hem": [sdf.HalfSpace(_v(0.0, 0.0, lm["hem_z"]), _v(0.0, 0.1, -1.0))],
    }


def jersey_field(bones, voxel=0.0022):
    """Closed solid of the jersey's outer surface; openings are cut afterwards (jersey_keep).

    Built from Sara's own torso sections: close over the shoulders and upper chest, then the fabric
    HANGS - the front falls from the chest (not following the waist back in), the back from the
    shoulder blades, the sides from the rib cage with a slight flare at the hem. Never a rectangular block.
    """
    m = bodymod.master(bones)
    lm = m.lm
    sc = lm["s"]
    rows = m.torso_rows()
    z_apex, y_apex = m.bust_apex()
    z_scap = m.tz(0.72)
    hem = lm["hem_z"]
    i_chest = int(np.argmin(np.abs(rows[:, 0] - m.tz(0.75))))
    w_chest = rows[i_chest, 1]
    drape = []
    zs = np.linspace(hem - 0.03 * sc, m.tz(0.93), 22)
    for z in zs:
        w, yf, yb, nf, nb = [float(np.interp(z, rows[:, 0], rows[:, j])) for j in range(1, 6)]
        below = max(0.0, z_apex - z)
        hang_f = y_apex - 0.004 * sc + 0.07 * below                 # front falls from the chest, easing back a little
        yf_j = min(yf - 0.007 * sc, hang_f) if z < z_apex else yf - 0.007 * sc
        yb_j = max(yb + 0.007 * sc, float(np.interp(z_scap, rows[:, 0], rows[:, 3])) + 0.004 * sc - 0.06 * max(0.0, z_scap - z))
        k_hem = 1.0 - min(1.0, (z - hem) / (0.12 * sc)) if z > hem else 1.0
        w_j = max(w + 0.007 * sc, (w_chest - 0.006 * sc + 0.008 * sc * k_hem) if z < m.tz(0.75) else w + 0.007 * sc)
        drape.append((z, w_j, yf_j, yb_j, 2.1, 2.2))
    f = sdf.Field(_v(-0.2, -0.18, hem - 0.09), _v(0.2, 0.16, lm["neck_z"] + 0.1), voxel)
    for prim, k in bodymod.torso_parts(bones):
        f.union(prim, k)
    f.offset(0.007)
    f.union(sdf.ZLoft(drape), 0.03 * sc)
    shorts_outer = f.copy()
    shorts_outer.d[:] = sdf.BIG
    shorts_outer.union(bodymod.torso_parts(bones)[0][0], 0.0)          # the torso loft, below the waistband
    shorts_outer.intersect(sdf.HalfSpace(_v(0.0, 0.0, lm["waistband_z"] + 0.02 * sc), _v(0.0, 0.0, 1.0)), 0.02 * sc)
    shorts_outer.offset(0.012)
    f.union_field(shorts_outer, 0.02 * sc)
    f.drape(0.003 * sc, grow=0.0006 * sc)         # fabric bridges small creases (armpit, sternum, spine)
    return f


def jersey_keep(bones):
    """Signed 'keep' distances (keep where < 0) for the jersey openings."""
    cuts = jersey_cuts(bones)
    return ([("remove", p) for p in cuts["neck"] + cuts["arm_L"] + cuts["arm_R"]] + [("keep", p) for p in cuts["hem"]])


def shorts_field(bones, voxel=0.002):
    """The full body field (torso + legs) grown by the fit allowance, so the kept part of the shorts
    encloses the skin everywhere (no poke-through at the glute/thigh creases)."""
    m = bodymod.master(bones)
    lm = m.lm
    sc = lm["s"]
    f = bodymod.body_field(bones, voxel=voxel, include=("torso", "legs"),
                           bmin=(-0.2, -0.14, lm["hip_z"] - 0.16 * sc), bmax=(0.2, 0.15, lm["waistband_z"] + 0.03 * sc))
    tt = m.p["thigh_thickness"]
    for side in ("Left", "Right"):
        hip, knee = _bone(bones, f"{side}UpperLeg")
        f.union(sdf.RoundCone(hip + _v(0, 0, 0.01), hip + (knee - hip) * 0.4, 0.088 * tt * sc, 0.074 * tt * sc), 0.05 * sc)
    f.offset(0.004)
    f.drape(0.004 * sc, grow=0.001 * sc)          # fabric spans the gluteal cleft and creases
    return f


def shorts_keep(bones):
    """Keep below the waistband and above each leg opening (the opening plane is higher on the outer side)."""
    out = [("keep", sdf.HalfSpace(_v(0.0, 0.0, bodymod.landmarks(bones)["waistband_z"]), _v(0.0, 0.0, 1.0)))]
    openings = []
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        hip, knee = _bone(bones, f"{side}UpperLeg")
        ax = (knee - hip) / np.linalg.norm(knee - hip)
        n = ax + _v(sx * 0.2, 0.0, 0.0)
        p = hip + (knee - hip) * 0.2
        # This leg's side of the body (x on her left for the left leg): below the opening plane is removed.
        side_region = sdf.HalfSpace(_v(0.0, 0.0, 0.0), _v(-sx, 0.0, 0.0))
        out.append(("remove", _Intersect(_Neg(sdf.HalfSpace(p, n)), side_region)))
        openings.append((p, n))
    return out, openings


class _Neg:
    def __init__(self, prim):
        self.prim = prim

    def bbox(self):
        return self.prim.bbox()

    def eval(self, P):
        return -self.prim.eval(P)


class _Intersect:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def bbox(self):
        return self.b.bbox()

    def eval(self, P):
        return np.maximum(self.a.eval(P), self.b.eval(P))


def leg_band(bones, side, t0, t1, grow, voxel=0.0018, extra=()):
    """(solid field, keep cuts) around the leg between two fractions of knee->ankle (t<0 = above the knee)."""
    hip, knee = _bone(bones, f"{side}UpperLeg")
    _, ankle = _bone(bones, f"{side}LowerLeg")
    shin = ankle - knee
    lo = knee + shin * t1
    hi = knee + shin * t0 if t0 >= 0 else knee + (hip - knee) * (-t0)
    bmin = np.minimum(lo, hi) - 0.09
    bmax = np.maximum(lo, hi) + 0.09
    f = sdf.Field(bmin, bmax, voxel)
    for prim, k in bodymod.leg_parts(bones, side):
        f.union(prim, k)
    for prim, k in extra:
        f.union(prim, k)
    f.offset(grow)
    up = (hi - lo) / np.linalg.norm(hi - lo)
    return f, [("keep", sdf.HalfSpace(hi, up)), ("keep", sdf.HalfSpace(lo, -up))]


def kneepad_field(bones, side):
    """Sara's knee protection (key art): a THIN sleeve that follows the knee, with a flat, low front plate
    over the patella. Never a foam knee pad: the leg line stays readable through it."""
    m = bodymod.master(bones)
    kn = m.p["knee_size"] * m.s
    knee = _bone(bones, f"{side}LowerLeg")[0]
    plate = [(sdf.Ellipsoid(knee + _v(0.0, -0.03, -0.002) * kn, _v(0.03, 0.012, 0.044) * kn), 0.02 * m.s)]
    return leg_band(bones, side, -0.15, 0.14, 0.003, extra=plate)


def kneepad_plate(bones, side):
    """The white front plate over the patella, a hair proud of the sleeve (the sleeve itself is pale blue)."""
    m = bodymod.master(bones)
    kn = m.p["knee_size"] * m.s
    knee = _bone(bones, f"{side}LowerLeg")[0]
    plate = [(sdf.Ellipsoid(knee + _v(0.0, -0.03, -0.002) * kn, _v(0.03, 0.012, 0.044) * kn), 0.02 * m.s)]
    f, keeps = leg_band(bones, side, -0.12, 0.11, 0.0042, extra=plate)
    f.intersect(sdf.HalfSpace(knee + _v(0.0, -0.014, 0.0), _v(0.0, 1.0, 0.0)), 0.004)       # front only
    for sx in (1.0, -1.0):
        f.intersect(sdf.HalfSpace(knee + _v(sx * 0.032 * kn / m.s, 0.0, 0.0), _v(sx, 0.0, 0.0)), 0.004)
    return f, keeps


def kneepad_bands(bones, side):
    """Slim sky-blue elastic bands at the sleeve's top and bottom edges."""
    return [leg_band(bones, side, -0.15, -0.125, 0.0042), leg_band(bones, side, 0.115, 0.14, 0.0042)]


def sock_field(bones, side, high=True):
    if high:
        return leg_band(bones, side, 0.12, 1.3, 0.0034, voxel=0.0018)      # continues down into the shoe
    return leg_band(bones, side, 0.86, 1.3, 0.0034, voxel=0.0016)


# ---------------------------------------------------------------------------
# shoes: constructed footwear around the foot anatomy
# ---------------------------------------------------------------------------
class _Plane:
    """Keep-side half space helper: solid where (p - point) . normal < 0."""

    def __init__(self, point, normal):
        self.h = sdf.HalfSpace(point, normal)

    def bbox(self):
        return None

    def eval(self, P):
        return self.h.eval(P)


def _intersect_all(field, prims, k=0.0):
    for pr in prims:
        field.intersect(pr, k)
    return field


def shoe_frame(bones, side):
    m = bodymod.master(bones)
    sc = m.s
    foot = m._foot(side)
    ankle, _ = _bone(bones, f"{side}Foot")
    heel, tip = foot.a.copy(), foot.b.copy()
    fwd = _v(tip[0] - heel[0], tip[1] - heel[1], 0.0)
    fwd /= np.linalg.norm(fwd)
    back = heel - fwd * 0.026 * sc
    front = tip + fwd * 0.002 * sc
    L = float(np.linalg.norm((front - back)[:2]))
    lat = np.cross(fwd, _v(0, 0, 1))
    return {"m": m, "s": sc, "foot": foot, "ankle": ankle, "back": _v(back[0], back[1], 0.0), "front": front,
            "fwd": fwd, "lat": lat, "L": L, "fw": m.p["foot_width"] * sc, "ak": m.p["ankle"] * sc}


def _at(fr, along, z=0.0, lateral=0.0):
    return fr["back"] + fr["fwd"] * (along * fr["L"]) + fr["lat"] * lateral + _v(0, 0, z)


def shoe_parts(bones, side, voxel=0.001):
    """Sara's low volleyball shoe as separate constructed parts (key art): white upper with a rounded toe
    box and a firm heel, black heel counter, black side stripe, black eyestay and laces, a light tongue,
    a thin white midsole with toe spring and a dark outsole line. Returns [(name, field, color, remesh)]."""
    fr = shoe_frame(bones, side)
    sc, fw, fwd, lat, L = fr["s"], fr["fw"], fr["fwd"], fr["lat"], fr["L"]
    R = np.stack([lat, fwd, _v(0, 0, 1)], axis=1)
    lo = np.minimum(fr["back"], fr["front"]) - _v(0.07, 0.07, 0.0)
    hi = np.maximum(fr["back"], fr["front"]) + _v(0.07, 0.07, 0.13)
    lo[2] = -0.003

    def field():
        return sdf.Field(lo, hi, voxel)

    def last():
        """The shoe last: foot + rounded toe box + firm heel."""
        f = field()
        f.union(fr["foot"], 0.0)
        # Close to the foot: a low, rounded toe box, a vamp that follows the instep, a narrow firm heel.
        f.union(sdf.Ellipsoid(_at(fr, 0.8, 0.021 * sc), _v(0.031 * fw / 0.9, 0.034 * sc, 0.015 * sc), R), 0.016 * sc)  # toe box
        f.union(sdf.Ellipsoid(_at(fr, 0.47, 0.04 * sc), _v(0.027 * fw / 0.9, 0.056 * sc, 0.018 * sc), R), 0.02 * sc)   # vamp
        f.union(sdf.Ellipsoid(_at(fr, 0.13, 0.033 * sc), _v(0.026 * fw / 0.9, 0.034 * sc, 0.031 * sc), R), 0.016 * sc)  # heel
        return f

    # Sole: thin, following the last's outline, thicker at the heel, toe spring at the front.
    def sole_top(along):
        return float(np.interp(along, [0.0, 0.5, 0.75, 1.0], [0.0135, 0.0105, 0.0088, 0.01])) * sc

    sole = last()
    sole.offset(0.0032 * sc)
    top_heel, top_toe = sole_top(0.0), sole_top(0.75)
    slope = (top_heel - top_toe) / (0.75 * L)
    sole.intersect(sdf.HalfSpace(_at(fr, 0.0, top_heel), _v(fwd[0] * slope, fwd[1] * slope, 1.0)), 0.003 * sc)
    sole.intersect(sdf.HalfSpace(_v(0, 0, 0.0), _v(0, 0, -1.0)), 0.0)
    spring = 0.1                                               # toe spring: the sole lifts ahead of the ball
    sole.intersect(sdf.HalfSpace(_at(fr, 0.8, 0.0), _v(fwd[0] * spring, fwd[1] * spring, -1.0)), 0.004 * sc)
    outsole = sole.copy()
    outsole.offset(0.0005 * sc)
    outsole.intersect(sdf.HalfSpace(_v(0, 0, 0.003 * sc), _v(0, 0, 1.0)), 0.0)
    # Upper: close over the last (thin leather, not a padded block); clean collar below the ankle bones.
    upper = last()
    upper.offset(0.0028 * sc)
    upper.intersect(sdf.HalfSpace(_at(fr, 0.0, top_heel - 0.004 * sc), _v(0, 0, -1.0)), 0.0)
    collar_n = _v(-fwd[0] * 0.14, -fwd[1] * 0.14, 1.0)
    collar_p = _v(*fr["ankle"][:2], 0.063 * sc)
    upper.intersect(sdf.HalfSpace(collar_p, collar_n), 0.004 * sc)
    upper.subtract(sdf.Capsule(_v(fr["ankle"][0], fr["ankle"][1] + 0.006 * sc, 0.048 * sc),
                               _v(fr["ankle"][0], fr["ankle"][1] + 0.012 * sc, 0.3), 0.028 * fr["ak"] / sc * sc), 0.005 * sc)

    def overlay(regions, grow=0.0022):
        f = upper.copy()
        f.offset(grow * sc)
        return _intersect_all(f, regions, 0.0015 * sc)

    # Heel counter: a curved panel whose front edge sweeps from the collar down and forward to the sole.
    c0, c1 = _at(fr, 0.13, 0.066 * sc), _at(fr, 0.3, 0.012 * sc)
    dc = (c1 - c0) / np.linalg.norm(c1 - c0)
    nc = np.cross(lat, dc)
    nc = nc if nc @ fwd > 0 else -nc
    heel_counter = overlay([sdf.HalfSpace(c0, nc / np.linalg.norm(nc)), sdf.HalfSpace(_v(0, 0, 0.05 * sc), _v(0, 0, 1.0))])
    # Side stripe: a swept band from the heel counter forward and down toward the midsole.
    p0, p1 = _at(fr, 0.2, 0.05 * sc), _at(fr, 0.66, 0.021 * sc)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.cross(d, lat)
    n /= np.linalg.norm(n)
    hw = 0.0058 * sc
    stripe = overlay([sdf.HalfSpace(p0 + n * hw, n), sdf.HalfSpace(p0 - n * hw, -n),
                      sdf.HalfSpace(_at(fr, 0.68), fwd), sdf.HalfSpace(_at(fr, 0.18), -fwd)], grow=0.0024)
    # Lace area: a light tongue down the instep between two black eyestay strips.
    ew = 0.0165 * fw / 0.9
    lace_box = [sdf.HalfSpace(_at(fr, 0.71), fwd), sdf.HalfSpace(_at(fr, 0.34), -fwd), sdf.HalfSpace(_v(0, 0, 0.03 * sc), _v(0, 0, -1.0))]
    eyestay = field()
    for sgn in (1.0, -1.0):
        e = overlay([sdf.HalfSpace(_at(fr, 0.0, 0.0, sgn * ew), sgn * lat),
                     sdf.HalfSpace(_at(fr, 0.0, 0.0, sgn * (ew - 0.0052 * sc)), -sgn * lat)] + lace_box)
        eyestay.union_field(e, 0.0)
    tongue = overlay([sdf.HalfSpace(_at(fr, 0.0, 0.0, ew - 0.0052 * sc), lat),
                      sdf.HalfSpace(_at(fr, 0.0, 0.0, -(ew - 0.0052 * sc)), -lat)] + lace_box, grow=0.0016)
    # Laces seated on the eyestay surface (sampled from the upper).
    laces = field()
    for i in range(4):
        a_ = 0.41 + i * 0.075
        pts = []
        for lt in (-ew * 0.85, ew * 0.85):
            q = _at(fr, a_, 0.0, lt)
            zs = np.linspace(0.11 * sc, 0.0, 220)
            dd = upper.sample_at(np.stack([np.full_like(zs, q[0]), np.full_like(zs, q[1]), zs], axis=1))
            iz = int(np.argmax(dd < 0))
            pts.append(_v(q[0], q[1], zs[iz] + 0.0036 * sc))
        a, b = pts
        laces.union(sdf.Capsule(a - fwd * 0.005 * sc, b + fwd * 0.005 * sc, 0.0023 * sc), 0.0)
        laces.union(sdf.Capsule(b - fwd * 0.005 * sc, a + fwd * 0.005 * sc, 0.0023 * sc), 0.0)
    return [("upper", upper, "@shoe_upper", 9000), ("midsole", sole, "@shoe_sole", 5000),
            ("outsole", outsole, "@shoe_outsole", None), ("heel", heel_counter, "@shoe_accent", None),
            ("stripe", stripe, "@shoe_accent", None), ("eyestay", eyestay, "@shoe_accent", None),
            ("tongue", tongue, "@shoe_tongue", None), ("laces", laces, "@shoe_lace", None)]


def _n(v):
    return v / np.linalg.norm(v)
