"""Sara: uniform, legwear and shoes as real garments (shells with thickness), not paint.

Jersey : sleeveless cap-shoulder cut, deep armholes, crew neck with a slight front scoop, straight
         loose hem ~6 cm below the waistband. Hangs from the chest (does not follow the waist).
Shorts : fitted, short inseam, leg openings slightly higher on the outer side.
Pads   : sleeve knee pads, thick padded front, blue elastic bands at both edges.
Legwear: LEFT black knee-high (top hidden under the pad); RIGHT low ankle sock.
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
def jersey_cuts():
    return {
        # Crew neck that dips at the front: one tilted opening, so the edge is a single clean line.
        "neck": [sdf.Ellipsoid(_v(0.0, -0.012, 1.405), _v(0.062, 0.1, 0.072), sdf.rotation_to(_v(0.0, 0.42, 1.0)))],
        "arm_L": [sdf.Ellipsoid(_v(0.182, 0.004, 1.226), _v(0.072, 0.1, 0.1), sdf.rotation_to(_v(0.25, 0.0, 1.0)))],
        "arm_R": [sdf.Ellipsoid(_v(-0.182, 0.004, 1.226), _v(0.072, 0.1, 0.1), sdf.rotation_to(_v(-0.25, 0.0, 1.0)))],
        "hem": [sdf.HalfSpace(_v(0.0, 0.0, 0.878), _v(0.0, 0.0, -1.0))],
    }


def jersey_field(bones, voxel=0.0022):
    """Closed solid of the jersey's outer surface; openings are cut afterwards (jersey_keep)."""
    f = sdf.Field(_v(-0.2, -0.18, 0.84), _v(0.2, 0.16, 1.45), voxel)
    for prim, k in bodymod.torso_parts(bones):
        f.union(prim, k)
    f.offset(0.009)
    # Drape: hangs straight from the chest (not following the waist) and stays outside the shorts.
    f.union(sdf.Ellipsoid(_v(0.0, -0.004, 1.03), _v(0.14, 0.106, 0.235)), 0.05)
    f.union(sdf.Ellipsoid(_v(0.0, -0.006, 0.925), _v(0.16, 0.122, 0.08)), 0.05)
    shorts_outer = f.copy()
    shorts_outer.d[:] = sdf.BIG
    for prim, k in bodymod.torso_parts(bones)[:5]:
        shorts_outer.union(prim, k)
    shorts_outer.offset(0.016)
    f.union_field(shorts_outer, 0.02)
    return f


def jersey_keep():
    """Signed 'keep' distances (keep where < 0) for the jersey openings."""
    cuts = jersey_cuts()
    return ([("remove", p) for p in cuts["neck"] + cuts["arm_L"] + cuts["arm_R"]] + [("keep", p) for p in cuts["hem"]])


def shorts_field(bones, voxel=0.002):
    """The full body field (torso + legs) grown by the fit allowance, so the kept part of the shorts
    encloses the skin everywhere (no poke-through at the glute/thigh creases)."""
    f = bodymod.body_field(bones, voxel=voxel, include=("torso", "legs"), bmin=(-0.2, -0.14, 0.72), bmax=(0.2, 0.15, 1.03))
    for side in ("Left", "Right"):
        hip, knee = _bone(bones, f"{side}UpperLeg")
        f.union(sdf.RoundCone(hip + _v(0, 0, 0.01), hip + (knee - hip) * 0.4, 0.088, 0.074), 0.05)
    f.offset(0.0045)
    return f


def shorts_keep(bones):
    """Keep below the waistband and above each leg opening (the opening plane is higher on the outer side)."""
    out = [("keep", sdf.HalfSpace(_v(0.0, 0.0, 1.0), _v(0.0, 0.0, 1.0)))]
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
    knee = _bone(bones, f"{side}LowerLeg")[0]
    pad = [(sdf.Ellipsoid(knee + _v(0.0, -0.034, -0.004), _v(0.042, 0.03, 0.058)), 0.025)]
    return leg_band(bones, side, -0.13, 0.16, 0.009, extra=pad)


def kneepad_bands(bones, side):
    return [leg_band(bones, side, -0.13, -0.1, 0.0115), leg_band(bones, side, 0.13, 0.16, 0.0115)]


def sock_field(bones, side, high=True):
    if high:
        return leg_band(bones, side, 0.12, 1.02, 0.0045, voxel=0.0018)
    return leg_band(bones, side, 0.86, 1.02, 0.0045, voxel=0.0016)


# ---------------------------------------------------------------------------
# shoes
# ---------------------------------------------------------------------------
def shoe_field(bones, side, voxel=0.0016):
    ankle, toe_h = _bone(bones, f"{side}Foot")
    _, toe_t = _bone(bones, f"{side}Toes")
    heel = _v(ankle[0], ankle[1] + 0.052, 0.0)
    tip = _v(toe_t[0], toe_t[1] - 0.012, 0.0)
    fwd = tip - heel
    L = np.linalg.norm(fwd)
    fwd /= L
    lat = np.cross(fwd, _v(0, 0, 1))
    mid = (heel + tip) * 0.5
    R = np.stack([lat, fwd, _v(0, 0, 1)], axis=1)
    f = sdf.Field(np.minimum(heel, tip) - 0.08, np.maximum(heel, tip) + _v(0.08, 0.08, 0.16), voxel)
    # Sole slab (outsole + midsole), slight toe spring.
    f.union(sdf.RoundBox(mid + _v(0, 0, 0.014), _v(0.036, L * 0.5 - 0.008, 0.008), 0.009, R), 0.0)
    # Upper: heel counter, midfoot, toe box, blended; opening at the collar.
    f.union(sdf.Ellipsoid(heel + fwd * 0.048 + _v(0, 0, 0.062), _v(0.043, 0.052, 0.056), R), 0.02)
    f.union(sdf.Ellipsoid(mid + fwd * 0.0 + _v(0, 0, 0.056), _v(0.046, 0.088, 0.05), R), 0.025)
    f.union(sdf.Ellipsoid(tip - fwd * 0.055 + _v(0, 0, 0.036), _v(0.044, 0.062, 0.034), R), 0.025)
    f.union(sdf.Ellipsoid(mid + fwd * 0.035 + _v(0, 0, 0.088), _v(0.029, 0.062, 0.022), R), 0.015)      # tongue ridge
    f.subtract(sdf.Capsule(_v(ankle[0], ankle[1] + 0.008, 0.108), _v(ankle[0], ankle[1] + 0.02, 0.24), 0.031), 0.006)
    return f, {"heel": heel, "tip": tip, "fwd": fwd, "lat": lat, "L": L, "mid": mid}


def shoe_zone(p, info, side_sign):
    """Material zone for a shoe vertex: outsole / midsole / heel / stripe / upper."""
    rel = p - info["heel"]
    along = float(rel @ info["fwd"]) / info["L"]      # 0 heel .. 1 toe
    lateral = float(rel @ info["lat"])
    z = p[2]
    if z < 0.0075:
        return "outsole"
    if z < 0.024:
        return "midsole"
    if along < 0.2 and z < 0.11:
        return "heel"
    # Side stripe: a sweeping band from the heel collar down to the forefoot sole line.
    band_z = 0.08 - 0.06 * (along - 0.15) / 0.55
    if abs(lateral) > 0.03 and 0.15 < along < 0.7 and abs(z - band_z) < 0.009:
        return "stripe"
    return "upper"


def laces(info, pairs=5):
    """Criss-cross lace capsules across the top of the vamp."""
    out = []
    mid, fwd, lat = info["mid"], info["fwd"], info["lat"]
    for i in range(pairs):
        a = 0.03 + i * 0.022
        z = 0.094 - i * 0.0068
        c0 = mid + fwd * (a - 0.06) + _v(0, 0, z)
        out.append(sdf.Capsule(c0 - lat * 0.016 - fwd * 0.004, c0 + lat * 0.016 + fwd * 0.004, 0.0024))
        out.append(sdf.Capsule(c0 + lat * 0.016 - fwd * 0.004, c0 - lat * 0.016 + fwd * 0.004, 0.0024))
    return out


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def shoe_masks(P, info):
    """Smooth zone masks per vertex: white (midsole) and black (outsole, heel counter, side stripe, laces)."""
    rel = P - info["heel"]
    along = (rel @ info["fwd"]) / info["L"]
    lateral = rel @ info["lat"]
    z = P[:, 2]
    outsole = _ss(0.0085, 0.0065, z)
    midsole = _ss(0.0065, 0.0085, z) * _ss(0.0255, 0.0235, z)
    upper = _ss(0.0235, 0.0255, z)
    heel = _ss(0.215, 0.185, along) * _ss(0.113, 0.107, z) * upper
    band_z = 0.08 - 0.06 * (along - 0.15) / 0.55
    stripe = _ss(0.0105, 0.0085, np.abs(z - band_z)) * _ss(0.026, 0.031, np.abs(lateral)) * _ss(0.12, 0.17, along) * \
        _ss(0.72, 0.66, along) * upper
    laces = _ss(0.076, 0.08, z) * _ss(0.022, 0.018, np.abs(lateral)) * _ss(0.3, 0.36, along) * _ss(0.78, 0.72, along)
    black = np.clip(np.maximum.reduce([outsole, heel, stripe, laces]), 0, 1)
    return midsole, black
