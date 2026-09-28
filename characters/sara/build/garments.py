"""Sara: uniform, legwear and shoes as real garments (shells with thickness), not paint.

Jersey : (front reference sara_jersey_front.png) sleeveless, narrow straps, a V-neck with a modeled white
         trim and navy piping, navy armhole bindings, white side panels with modeled navy seam piping,
         hem just below the hip joints. Follows the body (bust, a slight waist) and hangs as a clean A-line
         over the hips; never a wide rectangular block.
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
def jersey_canvas(lm):
    """Vertical extent (z0, z1) of the jersey print canvas (uniform_paint) from the body landmarks."""
    return lm["hem_z"] - 0.07, lm["neck_z"] + 0.083


def side_seam_u(z, lm):
    """Front-panel / side-panel seam as a print-canvas offset from the front center (1000 = one full turn)
    at height z: the seam follows the body, in at the waist and out toward the hem and the armpit. Shared
    by the painted panels and the modeled seam piping, so paint and geometry always agree."""
    z0, z1 = jersey_canvas(lm)
    k = (z1 - z0) * 1000.0 / 570.0
    yy = (430.0 * k - (np.asarray(z, dtype=np.float64) - z0) * 1000.0) / k
    return 205.0 + 0.00018 * yy ** 2


class VNeck:
    """V-neck opening (front reference): straight edges from the shoulder-neck junction down to a point just
    below the collarbones, joined to a round back / side neckline. Removal region < 0."""

    def __init__(self, zb, x1, z1, y_front, back):
        self.zb, self.m = zb, x1 / (z1 - zb)
        self.norm = math.sqrt(1.0 + 1.0 / self.m ** 2)
        self.y_front, self.back = y_front, back

    def bbox(self):
        return None

    def eval(self, P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        dv = np.maximum((self.zb + np.abs(x) / self.m - z) / self.norm, y - self.y_front)
        return np.minimum(dv, self.back.eval(P))


class _NeckCylinder:
    """Everything within r of the neck axis above a flat base: the back and side neckline."""

    def __init__(self, a, b, r):
        self.cap = sdf.Capsule(a, b, r)
        self.z0 = float(a[2])

    def bbox(self):
        return None

    def eval(self, P):
        return np.maximum(self.cap.eval(P), self.z0 - P[..., 2])


class Offset:
    """prim grown by d (eval - d): 'within d of the prim'."""

    def __init__(self, prim, d):
        self.prim, self.d = prim, d

    def bbox(self):
        return None

    def eval(self, P):
        return self.prim.eval(P) - self.d


class SeamBand:
    """A narrow band around the torso-side seam on one side (sx), between two heights."""

    def __init__(self, lm, sx, half_w, z0, z1, y0=0.0):
        self.lm, self.sx, self.hw, self.z0, self.z1, self.y0 = lm, sx, half_w, z0, z1, y0

    def bbox(self):
        return None

    def eval(self, P):
        x, y, z = P[..., 0], P[..., 1] - self.y0, P[..., 2]
        ang = np.arctan2(x, -y)
        target = self.sx * side_seam_u(z, self.lm) / 1000.0 * 2.0 * math.pi
        r = np.sqrt(x * x + y * y)
        d = np.abs(ang - target) * r - self.hw
        return np.maximum(d, np.maximum(self.z0 - z, z - self.z1))


def jersey_cuts(bones):
    lm = bodymod.landmarks(bones)
    sc, kt, nz = lm["s"], lm["kt"], lm["neck_z"]
    # Narrow straps (front reference): the armholes reach further in than a tank top's.
    ax, az = lm["shoulder_x"] + 0.016 * sc, lm["shoulder_z"] - 0.082 * sc * kt
    # Back / side neckline: a clean line at the base of the neck (never a collar riding up the neck).
    back_neck = _NeckCylinder(_v(0.0, 0.004 * sc, nz + 0.005 * sc), _v(0.0, 0.022 * sc, nz + 0.25 * sc), 0.06 * sc)
    return {
        # V-neck: the point sits just below the collarbones; the back neckline stays round and higher.
        "neck": [VNeck(nz - 0.05 * sc, 0.052 * sc, nz + 0.03 * sc, -0.015 * sc, back_neck)],
        "arm_L": [sdf.Ellipsoid(_v(ax, 0.004, az), _v(0.076, 0.1, 0.1 * kt) * sc, sdf.rotation_to(_v(0.25, 0.0, 1.0)))],
        "arm_R": [sdf.Ellipsoid(_v(-ax, 0.004, az), _v(0.076, 0.1, 0.1 * kt) * sc, sdf.rotation_to(_v(-0.25, 0.0, 1.0)))],
        # Hem just below the hip joints (front reference): the shorts show below it; a touch lower at the back.
        "hem": [sdf.HalfSpace(_v(0.0, 0.0, lm["hem_z"]), _v(0.0, 0.1, -1.0))],
    }


def jersey_trims(bones, jersey_thickness=0.0022):
    """The jersey's real garment edges and seams (front reference), as thin shells on its outer surface:
    [(name, keep cuts, grow, thickness, material key)]. The solid is jersey_field grown by `grow`."""
    lm = bodymod.landmarks(bones)
    cuts = jersey_cuts(bones)
    neck, hem = cuts["neck"][0], cuts["hem"][0]
    on = jersey_thickness + 0.0003
    out = [
        # V-neck trim: a white band with a navy piping line along its outer edge.
        ("neck_trim", [("remove", neck), ("keep", Offset(neck, 0.0085))], on, 0.0015, "trim_white"),
        ("neck_piping", [("remove", Offset(neck, 0.0085)), ("keep", Offset(neck, 0.011))], on + 0.0002, 0.0012, "piping"),
    ]
    for side in ("L", "R"):
        arm = cuts[f"arm_{side}"][0]
        # Armhole binding: a slim navy edge (the strap stays light).
        out.append((f"arm_binding_{side}", [("remove", arm), ("keep", Offset(arm, 0.0055)), ("remove", neck)], on, 0.0014, "piping"))
        # Side-panel seam piping: from the hem up into the armhole.
        sx = 1.0 if side == "L" else -1.0
        band = SeamBand(lm, sx, 0.0021, lm["hem_z"] - 0.02, lm["shoulder_z"])
        out.append((f"seam_piping_{side}", [("keep", band), ("remove", arm), ("keep", hem)], on, 0.0011, "piping"))
    return out


def _hip_extents(bones, z0, z1, voxel=0.004):
    """Per-height outer extents of what the jersey hangs over below the waist - the SHORTS (roomier than
    the legs at the thigh tops), else the body: rows (z, max |x|, min y, max y)."""
    f = shorts_field(bones, voxel=voxel)
    solid = f.d < 0
    out = []
    xs = f.origin[0] + np.arange(f.shape[0]) * f.voxel
    ys = f.origin[1] + np.arange(f.shape[1]) * f.voxel
    for k in range(f.shape[2]):
        z = f.origin[2] + k * f.voxel
        sl = solid[:, :, k]
        if z < z0 or z > z1 or not sl.any():
            continue
        ix, iy = np.nonzero(sl)
        out.append((z, float(np.abs(xs[ix]).max()), float(ys[iy].min()), float(ys[iy].max())))
    return np.asarray(out)


def jersey_field(bones, voxel=0.0022):
    """Closed solid of the jersey's outer surface; openings are cut afterwards (jersey_keep).

    Built from Sara's own torso sections (front reference: a clean jersey that follows the body): close
    over the shoulders and the chest, then the fabric HANGS - the front falls from the bust (not following
    the waist back in), the back from the shoulder blades, the sides come in a little at the waist and flare
    over the hips into the hem. Below the waist it clears the measured hip / thigh extents (plus the shorts
    under it), so nothing ever pokes through. Never a rectangular block.
    """
    m = bodymod.master(bones)
    lm = m.lm
    sc = lm["s"]
    rows = m.torso_rows()
    z_apex, y_apex = m.bust_apex()
    z_scap = m.tz(0.72)
    hem = lm["hem_z"]
    waist = lm["waist_z"]
    i_chest = int(np.argmin(np.abs(rows[:, 0] - m.tz(0.75))))
    w_chest = rows[i_chest, 1]
    ext = _hip_extents(bones, hem - 0.05 * sc, waist)
    ease = 0.009 * sc                                  # shorts thickness + the jersey's own ease
    drape = []
    zs = np.linspace(hem - 0.03 * sc, m.tz(0.93), 26)
    for z in zs:
        w, yf, yb, nf, nb = [float(np.interp(z, rows[:, 0], rows[:, j])) for j in range(1, 6)]
        below = max(0.0, z_apex - z)
        hang_f = y_apex - 0.004 * sc + 0.07 * below                 # front falls from the chest, easing back a little
        yf_j = min(yf - 0.007 * sc, hang_f) if z < z_apex else yf - 0.007 * sc
        yb_j = max(yb + 0.007 * sc, float(np.interp(z_scap, rows[:, 0], rows[:, 3])) + 0.004 * sc - 0.06 * max(0.0, z_scap - z))
        k_hem = 1.0 - min(1.0, (z - hem) / (0.12 * sc)) if z > hem else 1.0
        w_side = w_chest - 0.006 * sc + 0.014 * sc * k_hem                     # A-line flare toward the hem
        w_side -= 0.007 * sc * math.exp(-((z - waist) / (0.05 * sc)) ** 2)      # slight waist shaping
        w_j = max(w + 0.007 * sc, w_side) if z < m.tz(0.75) else w + 0.007 * sc
        n_f = 2.1
        if z < waist and len(ext):
            ex, ey0, ey1 = (float(np.interp(z, ext[:, 0], ext[:, j])) for j in (1, 2, 3))
            w_j = max(w_j, ex + ease)
            yf_j = min(yf_j, ey0 - ease)
            yb_j = max(yb_j, ey1 + ease)
            n_f = 2.1 + 0.5 * min(1.0, (waist - z) / (0.1 * sc))            # squarer over the hips: covers the thighs
        drape.append((z, w_j, yf_j, yb_j, n_f, 2.2 + 0.4 * (n_f - 2.1)))
    f = sdf.Field(_v(-0.22, -0.18, hem - 0.09), _v(0.22, 0.17, lm["neck_z"] + 0.1), voxel)
    for prim, k in bodymod.torso_parts(bones):
        f.union(prim, k)
    f.offset(0.007)
    f.union(sdf.ZLoft(drape), 0.03 * sc)
    # Below the waistband the jersey also clears the shorts' real shape (thigh fronts, the roomy seat):
    # a superellipse section alone cuts their corners.
    sf = shorts_field(bones, voxel=0.003)
    P = f.points(tuple(slice(0, n) for n in f.shape)).reshape(-1, 3)
    ds = sf.sample_at(P) - (0.002 + 0.007) * sc                       # shorts thickness + ease
    ds = np.where(P[:, 2] > lm["waistband_z"] + 0.01 * sc, sdf.BIG, ds).reshape(f.shape)
    f.d = sdf.smin(f.d, ds.astype(np.float32), 0.012 * sc).astype(np.float32)
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
        p = hip + (knee - hip) * 0.3            # a little longer and looser (front reference)
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
