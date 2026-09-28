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
    """Closed solid of the jersey's outer surface; openings are cut afterwards (jersey_keep)."""
    m = bodymod.master(bones)
    lm, p = m.lm, m.p
    sc, kt = lm["s"], lm["kt"]
    f = sdf.Field(_v(-0.2, -0.18, lm["hem_z"] - 0.09) * _v(1, 1, 1), _v(0.2, 0.16, lm["neck_z"] + 0.1), voxel)
    for prim, k in bodymod.torso_parts(bones):
        f.union(prim, k)
    f.offset(0.008)
    # Drape: falls from the chest with light ease instead of hugging the waist, and stays outside the
    # shorts; sized from the same torso/pelvis parameters as the body.
    f.union(sdf.Ellipsoid(_v(0.0, -0.004, m.tz(0.364)), _v(0.126 * p["torso_width"], 0.095 * p["torso_depth"], 0.22 * kt) * sc), 0.05 * sc)
    f.union(sdf.Ellipsoid(_v(0.0, -0.006, m.tz(0.161)), _v(0.132 * p["pelvis_width"], 0.104 * p["pelvis_depth"], 0.07 * kt) * sc), 0.05 * sc)
    shorts_outer = f.copy()
    shorts_outer.d[:] = sdf.BIG
    shorts_outer.union(bodymod.torso_parts(bones)[0][0], 0.0)          # the torso loft, below the waistband
    shorts_outer.intersect(sdf.HalfSpace(_v(0.0, 0.0, lm["waistband_z"] + 0.02 * sc), _v(0.0, 0.0, 1.0)), 0.02 * sc)
    shorts_outer.offset(0.012)
    f.union_field(shorts_outer, 0.02 * sc)
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
    """Sleeve pad with a controlled front pad close to the knee, so the lower leg never reads bulky."""
    m = bodymod.master(bones)
    kn = m.p["knee_size"] * m.s
    knee = _bone(bones, f"{side}LowerLeg")[0]
    pad = [(sdf.Ellipsoid(knee + _v(0.0, -0.027, -0.004) * kn, _v(0.033, 0.021, 0.05) * kn), 0.014 * m.s)]
    return leg_band(bones, side, -0.13, 0.16, 0.005, extra=pad)


def kneepad_bands(bones, side):
    return [leg_band(bones, side, -0.13, -0.1, 0.0068), leg_band(bones, side, 0.13, 0.16, 0.0068)]


def sock_field(bones, side, high=True):
    if high:
        return leg_band(bones, side, 0.12, 1.02, 0.0034, voxel=0.0018)
    return leg_band(bones, side, 0.86, 1.02, 0.0034, voxel=0.0016)


# ---------------------------------------------------------------------------
# shoes
# ---------------------------------------------------------------------------
def shoe_field(bones, side, voxel=0.0016):
    """Low volleyball shoe sized from the foot bones and the body's foot/ankle parameters: slim last,
    thin sole, trim toe box, so the shoe supports the leg line instead of dominating it."""
    m = bodymod.master(bones)
    sc = m.s
    fw = m.p["foot_width"] * sc
    ak = m.p["ankle"] * sc
    ankle, toe_h = _bone(bones, f"{side}Foot")
    _, toe_t = _bone(bones, f"{side}Toes")
    heel = _v(ankle[0], ankle[1] + 0.048 * sc, 0.0)
    tip = _v(toe_t[0], toe_t[1] - 0.01 * sc, 0.0)
    fwd = tip - heel
    L = np.linalg.norm(fwd)
    fwd /= L
    lat = np.cross(fwd, _v(0, 0, 1))
    mid = (heel + tip) * 0.5
    R = np.stack([lat, fwd, _v(0, 0, 1)], axis=1)
    h = 0.88 * sc                                     # upper height scale
    z_out, z_mid = 0.0062 * sc, 0.019 * sc            # outsole top, midsole top
    f = sdf.Field(np.minimum(heel, tip) - 0.08, np.maximum(heel, tip) + _v(0.08, 0.08, 0.16), voxel)
    # Sole slab (outsole + midsole), slight toe spring.
    f.union(sdf.RoundBox(mid + _v(0, 0, 0.0115 * sc), _v(0.03 * fw, L * 0.5 - 0.008, 0.0062 * sc), 0.0075 * sc, R), 0.0)
    # Upper: heel counter, midfoot, toe box, blended; opening at the collar.
    f.union(sdf.Ellipsoid(heel + fwd * 0.046 * sc + _v(0, 0, 0.056 * h), _v(0.037 * fw, 0.048 * sc, 0.05 * h), R), 0.018 * sc)
    f.union(sdf.Ellipsoid(mid + _v(0, 0, 0.049 * h), _v(0.04 * fw, 0.084 * sc, 0.043 * h), R), 0.022 * sc)
    f.union(sdf.Ellipsoid(tip - fwd * 0.053 * sc + _v(0, 0, 0.031 * h), _v(0.038 * fw, 0.058 * sc, 0.028 * h), R), 0.022 * sc)
    f.union(sdf.Ellipsoid(mid + fwd * 0.035 * sc + _v(0, 0, 0.078 * h), _v(0.025 * fw, 0.058 * sc, 0.019 * h), R), 0.014 * sc)  # tongue
    f.subtract(sdf.Capsule(_v(ankle[0], ankle[1] + 0.008 * sc, 0.098 * h), _v(ankle[0], ankle[1] + 0.02 * sc, 0.24), 0.034 * ak), 0.006 * sc)
    return f, {"heel": heel, "tip": tip, "fwd": fwd, "lat": lat, "L": L, "mid": mid, "h": h, "fw": fw,
               "z_out": z_out, "z_mid": z_mid}


def shoe_zone(p, info, side_sign):
    """Material zone for a shoe vertex: outsole / midsole / heel / stripe / upper."""
    rel = p - info["heel"]
    along = float(rel @ info["fwd"]) / info["L"]      # 0 heel .. 1 toe
    lateral = float(rel @ info["lat"])
    z, h = p[2], info["h"]
    if z < info["z_out"]:
        return "outsole"
    if z < info["z_mid"]:
        return "midsole"
    if along < 0.2 and z < 0.1 * h:
        return "heel"
    band_z = (0.072 - 0.054 * (along - 0.15) / 0.55) * h
    if abs(lateral) > 0.026 * info["fw"] and 0.15 < along < 0.7 and abs(z - band_z) < 0.008 * h:
        return "stripe"
    return "upper"


def laces(info, pairs=5):
    """Criss-cross lace capsules across the top of the vamp."""
    out = []
    mid, fwd, lat, h, fw = info["mid"], info["fwd"], info["lat"], info["h"], info["fw"]
    for i in range(pairs):
        a = 0.03 + i * 0.021
        z = (0.094 - i * 0.0068) * h
        c0 = mid + fwd * (a - 0.06) + _v(0, 0, z)
        w = 0.0145 * fw / 0.85
        out.append(sdf.Capsule(c0 - lat * w - fwd * 0.004, c0 + lat * w + fwd * 0.004, 0.0021))
        out.append(sdf.Capsule(c0 + lat * w - fwd * 0.004, c0 - lat * w + fwd * 0.004, 0.0021))
    return out


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def shoe_masks(P, info):
    """Smooth zone masks per vertex: white (midsole) and black (outsole, heel counter, side stripe, laces)."""
    rel = P - info["heel"]
    along = (rel @ info["fwd"]) / info["L"]
    lateral = rel @ info["lat"]
    z, h, zo, zm, fw = P[:, 2], info["h"], info["z_out"], info["z_mid"], info["fw"] / 0.85
    outsole = _ss(zo + 0.001, zo - 0.001, z)
    midsole = _ss(zo - 0.001, zo + 0.001, z) * _ss(zm + 0.001, zm - 0.001, z)
    upper = _ss(zm - 0.001, zm + 0.001, z)
    heel = _ss(0.215, 0.185, along) * _ss(0.103 * h, 0.097 * h, z) * upper
    band_z = (0.072 - 0.054 * (along - 0.15) / 0.55) * h
    stripe = _ss(0.0095 * h, 0.0077 * h, np.abs(z - band_z)) * _ss(0.022 * fw, 0.027 * fw, np.abs(lateral)) * \
        _ss(0.12, 0.17, along) * _ss(0.72, 0.66, along) * upper
    laces = _ss(0.068 * h, 0.072 * h, z) * _ss(0.019 * fw, 0.015 * fw, np.abs(lateral)) * _ss(0.3, 0.36, along) * \
        _ss(0.78, 0.72, along)
    black = np.clip(np.maximum.reduce([outsole, heel, stripe, laces]), 0, 1)
    return midsole, black
