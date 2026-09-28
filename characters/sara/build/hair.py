"""Sara: hairstyle, ribbon and clip, designed from the primary masses down.

Primary masses (silhouette first)
  skull volume : the whole top and back are pulled up into a HIGH tie behind her left crown, so the
                 head reads round and full with a visible gathered ridge, never a helmet
  fringe       : one mass parted left of center, cut into pointed locks of varied length that sweep
                 away from the part (toward the eyes and sideways), long locks over the upper lids and
                 between the eyes, outer locks framing the eye corners; broad clumps layered on top
                 give it depth (varied, asymmetric, directional: never a comb or a helmet edge)
  side framing : a wide hooked lock that bows out off her left cheek and curls in under it, a long
                 mass behind her left ear falling in front of the shoulder to the chest, and a longer,
                 straighter lock on her right to the collarbone (the asymmetry of the art)
  ponytail     : a long, thick S-curve that lifts out of the tie in a root puff and splits into three
                 sub-masses (A/B/C) after the first quarter; tips separate and drift
Secondary: two thin crown braids running into the tie, layering, nape wisps.
Tertiary : ahoge curl, temple flyaways (LOD0 only). Every fringe lock grows out of the mass: no strays mid-forehead.
Ornaments: a thin sky-blue satin ribbon (tall wire-like loop that bends over the top, a long tail laid
           over the head that falls behind her right ear, a short tail), a cross-shaped clip on her right.

The secondary-motion chains are DERIVED from this design (chain_definitions), never the reverse, so
the bones always sit inside the masses they move.
"""
from __future__ import annotations

import math

import numpy as np

from spike_pipeline.modeling import lock as L

import head as headmod

HS = headmod.HS
EYE_Z = headmod.EYE_Z
C = np.array([0.0, headmod.AXIS_Y, 0.5 * (headmod.CHIN_Z + headmod.CROWN_Z) + 0.008])   # scalp projection center


def _v(*a):
    return np.array(a, dtype=np.float64)


def _n(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Scalp:
    """The real head (head master distance field): the hairline and every lock follow the skull."""

    def __init__(self, head):
        self.f = head.field()

    def inside(self, q, margin=0.0):
        return float(self.f.sample_at(np.asarray(q, dtype=np.float64)[None, :])[0]) < margin

    def surface(self, direction, lift=0.0):
        d = _n(direction)
        lo, hi = 0.0, 0.25
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.inside(C + d * mid):
                lo = mid
            else:
                hi = mid
        return C + d * (lo + lift)

    def project(self, q, lift):
        return self.surface(np.asarray(q, float) - C, lift)

    def keep_out(self, q, min_lift):
        q = np.asarray(q, float)
        if self.inside(q, margin=min_lift):
            return self.project(q, min_lift)
        return q

    def outward(self, q):
        return _n(np.asarray(q, float) - C)


def _dir(alpha_deg, elev_deg):
    """alpha 0 = front center, +90 = her left, 180 = back; elev 90 = straight up."""
    a, e = math.radians(alpha_deg), math.radians(elev_deg)
    return _v(math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))


# Hairline height relative to the eye line (x HS). Front high (the fringe covers it), low behind the ears.
_HL_A = [0, 30, 60, 80, 95, 110, 125, 150, 180]
_HL_DZ = [0.062, 0.058, 0.046, 0.034, 0.018, -0.006, -0.032, -0.058, -0.066]


def hairline_z(alpha_deg):
    return EYE_Z + HS * float(np.interp(abs(((alpha_deg + 180) % 360) - 180), _HL_A, _HL_DZ))


def hairline_point(sc, alpha_deg, lift=0.0):
    """Scalp point on the hairline (bisection on elevation so the point sits at hairline_z)."""
    z = hairline_z(alpha_deg)
    lo, hi = -70.0, 88.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if sc.surface(_dir(alpha_deg, mid))[2] < z:
            lo = mid
        else:
            hi = mid
    return sc.surface(_dir(alpha_deg, lo), lift), lo


def cap(sc, lift=0.0028, n_alpha=80, n_up=16):
    """Dark under-layer covering the scalp inside the hairline, so gaps between locks never show skin."""
    verts, faces = [], []
    grid = np.zeros((n_alpha, n_up), dtype=int)
    for i in range(n_alpha):
        alpha = -180 + 360 * i / n_alpha
        _, e0 = hairline_point(sc, alpha)
        for j in range(n_up):
            e = e0 + (89.0 - e0) * (j / n_up) ** 0.9
            grid[i, j] = len(verts)
            verts.append(sc.surface(_dir(alpha, e), lift))
    top = len(verts)
    verts.append(sc.surface(_v(0.0, 0.0, 1.0), lift))
    for i in range(n_alpha):
        ii = (i + 1) % n_alpha
        for j in range(n_up - 1):
            faces.append((grid[i, j], grid[ii, j], grid[ii, j + 1], grid[i, j + 1]))
        faces.append((grid[i, n_up - 1], grid[ii, n_up - 1], top))
    return np.asarray(verts), faces


def _slerp(a, b, t):
    a, b = _n(a), _n(b)
    w = math.acos(max(-1.0, min(1.0, float(a @ b))))
    if w < 1e-6:
        return a
    return (math.sin((1 - t) * w) * a + math.sin(t * w) * b) / math.sin(w)


def _resample(points, n):
    """n points evenly spaced by arc length along the Catmull-Rom curve through points."""
    path = L.catmull_rom(points, 12)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]
    u = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(u, s, path[:, k]) for k in range(3)], axis=1)


def _along_at(points, index):
    """Arc-length fraction (the lock's 'along') of control point `index`."""
    path = L.catmull_rom(points, 10)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return float(s[min(index * 10, len(s) - 1)] / s[-1])


def _frames(pts):
    """Parallel-transported (tangent, side, up) frames along a polyline."""
    T = np.gradient(pts, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    side = np.cross(T[0], _v(0, 0, 1))
    if np.linalg.norm(side) < 1e-6:
        side = _v(1, 0, 0)
    side = _n(side)
    S, U = [], []
    for t in T:
        side = _n(side - t * (side @ t))
        S.append(side)
        U.append(np.cross(side, t))
    return T, np.asarray(S), np.asarray(U)


# ---------------------------------------------------------------------------
# tie and ponytail design
# ---------------------------------------------------------------------------
def tie_point(sc):
    return sc.surface(_dir(118, 52), 0.02)


# Spine offsets from the tie: (x toward her left, y toward her back, z). A rise out of the tie, then a
# long S: out behind the head, back in toward the shoulder blades, a last kick out at the tips.
_PONY = [(0.0, 0.0, 0.0), (0.018, 0.045, 0.006), (0.035, 0.085, -0.07), (0.032, 0.1, -0.19),
         (0.014, 0.088, -0.31), (-0.008, 0.074, -0.42), (-0.016, 0.07, -0.52)]
# Sub-masses: (spread at the tips, length fraction). Spread grows after the first quarter.
_SUB = {"A": (_v(0.0, 0.0, 0.0), 1.0), "B": (_v(0.056, 0.03, 0.014), 0.9), "C": (_v(-0.056, 0.02, 0.022), 0.8)}
_PONY_R = ([0.0, 0.06, 0.15, 0.35, 0.6, 0.85, 1.0], [0.009, 0.022, 0.036, 0.046, 0.044, 0.034, 0.016])


def ponytail_spine(tie, n=41):
    return _resample([tie + _v(*o) for o in _PONY], n)


def sub_spines(tie, n=41):
    base = ponytail_spine(tie, n)
    out = {}
    for k, (spread, length) in _SUB.items():
        pts = [p + spread * L._smooth(0.22, 1.0, i / (n - 1)) for i, p in enumerate(base) if i / (n - 1) <= length + 1e-6]
        out[k] = np.asarray(pts)
    return out


def _pony_out(spine):
    """Outward (flat axis) for a ponytail lock: radial from the nearest spine point."""
    def f(q):
        d = np.linalg.norm(spine - q, axis=1)
        i = int(np.argmin(d))
        v = q - spine[i]
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            return _v(0.0, 1.0, 0.0)
        return v / nv
    return f


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build(bones, batch_hair: L.LockBatch, batch_ribbon: L.LockBatch):
    sc = Scalp(headmod.HEAD)
    T = tie_point(sc)
    tie_n = sc.outward(T)
    subs = sub_spines(T)
    base = ponytail_spine(T)
    paths = {}                         # (chain id, side) -> (control points, bones)
    free = {"hair": [], "ribbon": []}  # per lock: 'along' where it leaves the head (weights blend in there)

    def add(points, width, thickness, tag, batch=batch_hair, free_at=None, **kw):
        kw.setdefault("outward", sc.outward)
        points = [np.asarray(p, dtype=np.float64) for p in points]
        v, f, a = L.lock(points, width, thickness, **kw)
        batch.add(v, f, a, tag)
        free["hair" if batch is batch_hair else "ribbon"].append(0.0 if free_at is None else _along_at(points, free_at))
        return points

    def front_y(x, z, gap):
        y = headmod.front_y(x, z)
        return (y if y is not None else headmod.AXIS_Y - 0.03) - gap

    # --- 1. skull volume: every lock from the hairline over the scalp into the tie ----------------------
    def to_tie(start, width, lifts, thick=0.0075, tag="top"):
        d0, d1 = start - C, T - C
        pts = [start] + [sc.surface(_slerp(d0, d1, t), lift) for t, lift in lifts]
        pts.append(T - tie_n * 0.004)
        add(pts, width, thick, tag, tip_start=0.86, tip_power=1.0, root_min=0.9, crescent=0.14,
            thickness_fn=lambda u: 0.22 + 0.78 * L._smooth(0.0, 0.12, u))   # hair grows out of the scalp: no ledge

    for alpha in np.linspace(-112, 104, 17):
        start, _ = hairline_point(sc, alpha, 0.0032)
        to_tie(start, 0.046 * HS * (1.0 if abs(alpha) < 80 else 0.85),
               ((0.2, 0.009), (0.45, 0.013), (0.7, 0.015), (0.9, 0.014)))
    for alpha in np.linspace(118, 242, 12):
        start, _ = hairline_point(sc, alpha, 0.0032)
        to_tie(start, 0.05 * HS, ((0.3, 0.008), (0.6, 0.012), (0.85, 0.014)), tag="back")

    # --- 2. braids: a three-strand braid along each side of the head, from the temple back into the tie ---
    def surface_path(dirs, lift, n=30):
        """Points on the scalp through a list of directions (piecewise slerp), evenly resampled."""
        pts = []
        for d0, d1 in zip(dirs[:-1], dirs[1:]):
            for t in np.linspace(0.0, 1.0, 12, endpoint=False):
                pts.append(sc.surface(_slerp(d0, d1, t), lift))
        pts.append(sc.surface(dirs[-1], lift))
        return _resample(pts, n)

    braid_routes = {
        "L": [hairline_point(sc, 62, 0.0)[0] - C, _dir(92, 46), T - C],
        "R": [hairline_point(sc, -58, 0.0)[0] - C, _dir(-100, 44), _dir(-150, 50), _dir(170, 54), T - C],
    }
    for route in braid_routes.values():
        centre = surface_path(route, 0.018, 40)
        length = float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))
        turns = length / 0.026                       # a crossing every ~1.3 cm (three per turn)
        Tn, _, _ = _frames(centre)
        for k in range(3):
            ph = 2 * math.pi * k / 3
            pts = []
            for i, c in enumerate(centre):
                u = i / (len(centre) - 1)
                nrm = sc.outward(c)
                side = _n(np.cross(Tn[i], nrm))
                a = 2 * math.pi * turns * u + ph
                taper = 1.0 - 0.3 * u
                pts.append(c + side * (0.0042 * math.sin(a) * taper) + nrm * (0.0019 * math.sin(2 * a) * taper))
            add(pts, 0.0094, 0.0046, "braid", samples=4, tip_start=0.94, root_min=0.6, crescent=0.05)

    # --- 3. fringe: ONE mass from high on the head, cut at the edge into pointed locks of varied length ---
    # Tips: (x, dz above the eye line (x HS), sweep, tag). The part sits left of center (PART_X); locks
    # right of it sweep toward her right, locks left of it toward her left, the long ones reach down over
    # the upper lids and between the eyes, the outer ones frame the eye corners and lead into the side
    # locks. Irregular spacing and lengths (never a comb), but every lock grows out of the same mass: no
    # stray strand stands off the forehead.
    PART_X = 0.022
    tips = [(-0.074, -0.006, -0.008, "bang_C"), (-0.058, 0.012, -0.009, "bang_C"), (-0.041, 0.001, -0.011, "bang_C"),
            (-0.024, 0.017, -0.008, "bang_B"), (-0.007, -0.002, -0.009, "bang_B"), (0.009, 0.019, -0.004, "bang_B"),
            (0.027, 0.012, 0.003, "bang_A"), (0.045, 0.0, 0.007, "bang_A"), (0.061, 0.014, 0.007, "bang_A"),
            (0.077, -0.003, 0.006, "bang_A")]
    cx = np.array([t[0] for t in tips])
    cz = np.array([EYE_Z + t[1] * HS for t in tips])
    csw = np.array([t[2] for t in tips])
    x_lo, x_hi = -0.084, 0.088

    def cap_z(x):
        alpha = math.degrees(math.asin(max(-0.95, min(0.95, x / 0.088))))
        return min(EYE_Z + 0.044 * HS, hairline_z(alpha) - 0.012)

    def edge_z(x):
        """Jagged lower edge: pointed tips at the lock centers, V notches between them."""
        if x <= cx[0]:
            t = (cx[0] - x) / (cx[0] - x_lo)
            return cz[0] + (cap_z(x) - cz[0]) * t ** 0.7
        if x >= cx[-1]:
            t = (x - cx[-1]) / (x_hi - cx[-1])
            return cz[-1] + (cap_z(x) - cz[-1]) * t ** 0.7
        k = int(np.searchsorted(cx, x) - 1)
        t = (x - cx[k]) / (cx[k + 1] - cx[k])
        zn = min(cap_z(x), max(cz[k], cz[k + 1]) + 0.017 * HS)
        if t < 0.5:
            return cz[k] + (zn - cz[k]) * (2 * t) ** 0.7
        return cz[k + 1] + (zn - cz[k + 1]) * (2 * (1 - t)) ** 0.7

    def side_k(x):
        """1 inside the fringe, falling to 0 at its ends so the mass melts into the head at the temples."""
        return L._smooth(0.0, 0.016, x - x_lo) * L._smooth(0.0, 0.016, x_hi - x)

    def column(x, tz, nv=30, lift=0.0, sw=None):
        alpha = math.degrees(math.asin(max(-0.95, min(0.95, x / 0.088))))
        ar = 12.0 + (alpha - 12.0) * 0.5              # roots gather toward the part, left of center
        k = 0.3 + 0.7 * side_k(x)
        if sw is None:                                # directional sweep away from the part
            sw = float(np.interp(x, cx, csw))
        root = sc.surface(_dir(ar, 80), 0.004 + 0.5 * lift)
        rise = sc.surface(_dir(0.8 * ar + 0.2 * alpha, 70), 0.0145 * k + lift)
        p1, _ = hairline_point(sc, alpha, 0.0155 * k + lift)
        z2 = max(hairline_z(alpha) - 0.012 * HS, tz + 0.012)
        x2 = p1[0] + (x - p1[0]) * 0.35 + 0.1 * sw
        p2 = _v(x2, front_y(x2, z2, 0.0125 * k + lift), z2)
        z3 = 0.5 * (z2 + tz)
        x3 = p1[0] + (x - p1[0]) * 0.72 + 0.35 * sw
        p3 = _v(x3, front_y(x3, z3, 0.009 * k + lift), z3)
        tip = _v(x + sw, front_y(x + sw, tz, 0.0052 * k + 0.7 * lift), tz)
        return _resample([root, rise, p1, p2, p3, tip], nv), [root, rise, p1, p2, p3, tip]

    xs = np.linspace(x_lo, x_hi, 150)
    cols = [column(x, edge_z(x))[0] for x in xs]
    owner = np.array([int(np.argmin(np.abs(cx - x))) for x in xs])

    def ridge(i, v):
        k = owner[i]
        half = 0.5 * (cx[min(k + 1, len(cx) - 1)] - cx[max(k - 1, 0)]) / (2 if 0 < k < len(cx) - 1 else 1)
        t = min(abs(xs[i] - cx[k]) / max(half, 1e-4), 1.0)
        return (0.0009 + 0.0014 * L._smooth(0.3, 0.7, v)) * (0.5 + 0.5 * math.cos(math.pi * t))

    thick = lambda v: 0.0056 * (1.0 - 0.8 * L._smooth(0.72, 1.0, v)) * (0.35 + 0.65 * L._smooth(0.0, 0.12, v))  # noqa: E731
    side_thin = np.array([0.15 + 0.85 * side_k(x) for x in xs])
    # One segment per lock (split at the notches) so each lock follows its own bang chain.
    bounds = [0] + [int(np.argmin(np.abs(xs - 0.5 * (cx[k] + cx[k + 1])))) for k in range(len(cx) - 1)] + [len(xs) - 1]
    for k in range(len(cx)):
        i0, i1 = bounds[k], bounds[k + 1]
        seg = cols[i0:i1 + 1]
        v, f, a = L.sheet(seg, lambda vv: thick(vv), ridge_fn=lambda i, vv, o=i0: ridge(i + o, vv), center=C,
                          col_scale=side_thin[i0:i1 + 1])
        batch_hair.add(v, f, a, tips[k][3])
        free["hair"].append(_along_at(column(cx[k], cz[k])[1], 2))
    # Layering: a few broad clumps lying ON the fringe (same flow, a little in front of it), ending at
    # different lengths, so the fringe reads as overlapping hair with depth instead of one cut edge.
    for x, dz, sw, w, tag in ((-0.047, 0.007, -0.013, 0.026, "bang_C"), (-0.016, 0.009, -0.011, 0.022, "bang_B"),
                              (0.036, 0.006, 0.006, 0.024, "bang_A"), (0.068, 0.004, 0.008, 0.02, "bang_A")):
        _, ctrl = column(x, EYE_Z + dz * HS, lift=0.0028, sw=sw)
        add(ctrl[1:], w * HS, 0.0034, tag, free_at=1, tip_start=0.5, tip_power=0.8, root_min=0.35, crescent=0.25,
            thickness_fn=lambda u: 0.3 + 0.7 * L._smooth(0.0, 0.2, u))
    for tag, x in (("bang_C", -0.032), ("bang_B", 0.0), ("bang_A", 0.036)):
        k = int(np.argmin(np.abs(cx - x)))
        paths[("hair_" + tag, None)] = (column(cx[k], cz[k])[1][2:], 3)

    # --- 4. side framing --------------------------------------------------------------------------------
    def face_side(x, z, gap):
        """Point just outside the face/skull at (x, z), on the front-side surface."""
        return sc.keep_out(_v(x, front_y(x, z, gap) if abs(x) < 0.086 * HS else -0.02, z), gap)

    K = HS / 1.12          # these locks were placed on the head at HS 1.12: keep them relative to the head

    def hp(x, y, dz):
        """Head-relative point: x, y (at HS 1.12) and height above the eye line, scaled with the head."""
        return _v(x * K, headmod.AXIS_Y + (y - headmod.AXIS_Y) * K, EYE_Z + dz * K)

    # Hooked lock on her left side: hangs free in front of the ear, beside (not on) the cheek, and only
    # its tip curls forward and up toward the jaw.
    top, _ = hairline_point(sc, 76, 0.008)
    # Wide, bowed outward off the cheek (face-framing width in the front silhouette), then curling in.
    hook = [top,
            sc.keep_out(hp(0.097, -0.024, 0.004), 0.013),
            sc.keep_out(hp(0.105, -0.032, -0.034), 0.016),
            hp(0.1, -0.044, -0.066),
            hp(0.086, -0.058, -0.082),
            hp(0.072, -0.065, -0.07)]
    add(hook, 0.038 * HS, 0.0062, "side_L", free_at=1, tip_start=0.55, tip_power=0.7, root_min=0.85, crescent=0.26,
        outward=_v(0.45, -1.0, 0.0))
    paths[("hair_side_A", "L")] = (hook[1:], 4)
    add([top + _v(0.0, 0.012, -0.004)] + [p + _v(0.004, 0.013, 0.004) for p in hook[1:-2]] + [hook[-2] + _v(0.006, 0.02, 0.012)],
        0.02 * HS, 0.0045, "side_L", free_at=1, tip_start=0.45, root_min=0.85, crescent=0.15, outward=_v(0.6, -1.0, 0.0))
    # Long mass behind her left ear, falling in front of the shoulder to the chest.
    top, _ = hairline_point(sc, 106, 0.005)
    mass = [top,
            sc.keep_out(hp(0.1, 0.036, -0.03), 0.012),
            _v(0.112, 0.004, 1.43),
            _v(0.124, -0.032, 1.36),
            _v(0.128, -0.058, 1.29),
            _v(0.121, -0.076, 1.23)]
    add(mass, 0.044 * HS, 0.0085, "sideB_L", free_at=1, tip_start=0.55, tip_power=0.7, root_min=0.8, crescent=0.22,
        twist_deg=-20)
    add([p + _v(-0.008, 0.012, 0.004) for p in mass[:-1]] + [mass[-1] + _v(-0.004, 0.014, 0.03)], 0.03 * HS, 0.007,
        "sideB_L", free_at=1, tip_start=0.5, root_min=0.8, crescent=0.2, twist_deg=15)
    paths[("hair_side_B", "L")] = (mass[1:], 5)
    # Her right side: two long locks hanging free in front of the ear (one to the collarbone, one to the
    # jaw) and a fine strand, so the side reads as separated hair, not a band.
    top, _ = hairline_point(sc, -78, 0.008)
    # Longer and straighter than the hook (asymmetry), bowing out past the jaw and in toward the collarbone.
    right = [top,
             sc.keep_out(hp(-0.095, -0.028, -0.004), 0.013),
             sc.keep_out(hp(-0.103, -0.033, -0.052), 0.015),
             _v(-0.102, -0.036, 1.43),
             _v(-0.094, -0.047, 1.37),
             _v(-0.083, -0.06, 1.322)]
    add(right, 0.034 * HS, 0.0066, "side_R", free_at=1, tip_start=0.55, tip_power=0.7, root_min=0.85, crescent=0.26,
        twist_deg=14, outward=_v(-0.45, -1.0, 0.0))
    paths[("hair_side_A", "R")] = (right[1:], 5)
    top2, _ = hairline_point(sc, -90, 0.008)
    add([top2] + [p + _v(-0.004, 0.017, 0.002) for p in right[1:3]] + [right[3] + _v(0.002, 0.02, 0.03)], 0.02 * HS, 0.0052,
        "side_R", free_at=1, tip_start=0.5, root_min=0.85, crescent=0.2, twist_deg=-10, outward=_v(-0.6, -1.0, 0.0))
    add([top + _v(0.003, -0.004, 0.0)] + [p + _v(0.006, -0.008, -0.004) for p in right[1:4]] + [right[4] + _v(0.014, -0.004, 0.012)],
        0.0045, 0.0017, "side_R", free_at=1, tip_start=0.2, root_min=1.0, crescent=0.0)

    # --- 5. ponytail: locks bundled around the A/B/C sub-spines, fanning and drifting toward the tips ------
    rng = np.random.default_rng(7)
    plan = [("A", 7, 0.078, 0.013, 1.0), ("B", 4, 0.066, 0.012, 0.8), ("C", 4, 0.062, 0.011, 0.76)]   # few broad masses
    for key, count, w, th, rk in plan:
        spine = subs[key]
        Tn, S, U = _frames(spine)
        n = len(spine)
        # Core: a solid rounded mass inside each sub-mass so gaps between locks never show through.
        core = [T] + [spine[i] for i in range(3, int(n * 0.86), 3)]
        add(core, 0.056 * rk, 0.042 * rk, "ponytail_" + key, ring=12, tip_start=0.55, tip_power=0.8, root_min=0.3,
            crescent=0.0, outward=lambda q, U0=U[0]: U0, thickness_tip=0.6)
        u_all = np.linspace(0.0, 1.0, n) * (n - 1) / (len(base) - 1)
        clumps = np.array([0.4, 2.5, 4.4]) + rng.uniform(-0.3, 0.3)
        for k in range(count):
            phi = 2 * math.pi * k / count + rng.uniform(-0.35, 0.35)
            length = rng.uniform(0.86, 1.0)
            pts = []
            for i in range(0, n, 3):
                u = u_all[i]
                if i / (n - 1) > length:
                    break
                r = float(np.interp(u, *_PONY_R)) * rk * rng.uniform(0.85, 1.0)
                g = 0.86 * L._smooth(0.5, 1.0, u)             # tips gather into three soft clumps
                ph = phi + (clumps[int(np.argmin(np.abs(np.angle(np.exp(1j * (clumps - phi))))))] - phi) * g
                ring = S[i] * math.cos(ph) + U[i] * math.sin(ph)
                drift = (S[i] * 0.007 * u ** 1.5 * math.sin(3.1 * u + 1.3 * k) +
                         U[i] * 0.005 * u ** 1.5 * math.cos(2.3 * u + 1.7 * k))
                pts.append(spine[i] + ring * r + drift)
            pts[0] = T + (S[0] * math.cos(phi) + U[0] * math.sin(phi)) * 0.004
            if len(pts) >= 4:
                add(pts, w * rng.uniform(0.9, 1.1), th, "ponytail_" + key, tip_start=0.5, tip_power=0.5, root_min=0.55,
                    crescent=0.2, twist_deg=rng.uniform(-35, 35), outward=_pony_out(spine))
        bones_n = {"A": 8, "B": 6, "C": 6}[key]
        paths[("hair_ponytail_" + key, None)] = (list(spine), bones_n)
    # Root puff: short locks arching up and out of the tie before falling with the tail.
    Tn, S, U = _frames(base)
    for k in range(5):
        phi = math.pi * (0.1 + 0.8 * k / 4)
        ring = lambda i: S[i] * math.cos(phi) + U[i] * math.sin(phi)
        pts = [T + tie_n * 0.002, base[2] + ring(2) * 0.018 + _v(0, 0, 0.016), base[5] + ring(5) * 0.03,
               base[9] + ring(9) * 0.033, base[13] + ring(13) * 0.03]
        add(pts, 0.056, 0.012, "ponytail_A", tip_start=0.55, root_min=0.7, crescent=0.2, outward=_pony_out(base))
    # Nape wisps: short locks below the back mass.
    for alpha in (164, 180, 196):
        top, _ = hairline_point(sc, alpha, 0.004)
        o = sc.outward(top)
        add([top + _v(0, 0, 0.018), top, top + o * 0.006 + _v(0, 0, -0.018), top + o * 0.004 + _v(0, 0.002, -0.034)],
            0.016, 0.0042, "nape", tip_start=0.4, root_min=0.9, crescent=0.1)

    # --- 6. tertiary: ahoge curl and temple flyaways -----------------------------------------------------
    r0 = sc.surface(_dir(12, 80), 0.016)
    curl = [r0, r0 + _v(0.001, -0.004, 0.013), r0 + _v(0.004, -0.015, 0.019), r0 + _v(0.007, -0.025, 0.013),
            r0 + _v(0.008, -0.028, 0.004)]
    add(curl, 0.005, 0.0019, "ahoge_A", tip_start=0.3, root_min=1.0, crescent=0.0, outward=_v(1.0, 0.0, 0.0))
    add([p + _v(-0.003, 0.002, -0.0015 * i) for i, p in enumerate(curl[:-1])], 0.0038, 0.0015, "ahoge_A",
        tip_start=0.3, root_min=1.0, crescent=0.0, outward=_v(1.0, 0.0, 0.0))
    paths[("hair_ahoge_A", None)] = (curl[:4], 2)
    for fid, alpha, drift in (("flyaway_A", 66, _v(0.02, -0.01, 0.0)), ("flyaway_B", -64, _v(-0.018, -0.012, 0.0)),
                              ("flyaway_C", 150, _v(0.03, 0.035, 0.0))):
        top, _ = hairline_point(sc, alpha, 0.008)
        o = sc.outward(top)
        pts = [top, top + o * 0.012 + _v(0, 0, -0.02) + drift * 0.3, top + o * 0.018 + _v(0, 0, -0.045) + drift * 0.7,
               top + o * 0.016 + _v(0, 0, -0.066) + drift]
        pts = [pts[0]] + [sc.keep_out(p, 0.006) for p in pts[1:]]
        add(pts, 0.0036, 0.0013, fid, free_at=1, tip_start=0.15, root_min=1.0, crescent=0.0)
        paths[("hair_" + fid, None)] = (pts, 3)

    # --- 7. ribbon --------------------------------------------------------------------------------------
    K = T + tie_n * 0.013 + _v(0.0, 0.0, 0.004)
    # Loop: tall and narrow ("wire-like"), rising from the knot and bending forward over the top.
    up = _n(_v(0.1, 0.12, 1.0))
    bend = _n(_v(-0.55, -0.6, 0.0))
    lat = _n(np.cross(up, bend))
    h = 0.1 * HS
    loop, centre_line = [], []
    for s_ in np.linspace(0.0, 1.0, 36):
        yy = h * math.sin(math.pi * s_) ** 0.35
        k = yy / h
        half = 0.0035 + 0.0085 * k ** 3
        xx = half * math.cos(math.pi * s_)
        q = K + up * yy + bend * (0.042 * k ** 2.2 - 0.012 * k ** 6) + lat * xx
        loop.append(q)
    for k in np.linspace(0.0, 1.0, 4):
        centre_line.append(K + up * h * k + bend * (0.042 * k ** 2.2 - 0.012 * k ** 6))
    add(loop, 0.0105, 0.0011, "ribbon_loop_A", batch=batch_ribbon, crescent=0.06, thickness_tip=1.0, twist_deg=160,
        width_fn=lambda u: 1.0 - 0.18 * math.sin(math.pi * u), outward=lambda q: lat)
    paths[("ribbon_loop_A", None)] = (centre_line, 3)
    # Tails hang from the knot beside the ponytail (the S+ wind makes them stream): a long one on its
    # right, a short one on its left, each with a soft S and a twist so both satin faces catch light.
    def tail_path(offset, u_end, wave):
        pts = []
        for u in np.linspace(0.0, u_end, 7):
            j = int(round(u * (len(base) - 1)))
            k_ = L._smooth(0.0, 0.12, u)
            w_ = wave * math.sin(u * 14.0) * (u / u_end)
            pts.append(base[j] + offset * k_ + _v(w_, 0.35 * w_, 0.0) + (K - T) * (1.0 - k_))
        return pts
    tail = tail_path(_v(-0.05, 0.022, 0.0), 0.44, 0.012)
    add(tail, 0.0125, 0.0011, "ribbon_tail_A", batch=batch_ribbon, crescent=0.08, thickness_tip=1.0, twist_deg=110,
        width_fn=lambda u: (1.0 - 0.08 * math.sin(math.pi * u)) * (1.0 - 0.5 * L._smooth(0.94, 1.0, u)),
        outward=_pony_out(base))
    paths[("ribbon_tail_A", None)] = (tail, 4)
    short = tail_path(_v(0.046, 0.016, 0.0), 0.2, -0.008)
    add(short, 0.0125, 0.0011, "ribbon_tail_B", batch=batch_ribbon, crescent=0.08, thickness_tip=1.0, twist_deg=-70,
        width_fn=lambda u: 1.0 - 0.5 * L._smooth(0.9, 1.0, u), outward=_pony_out(base))
    paths[("ribbon_tail_B", None)] = (short, 3)

    cv, cf = cap(sc)
    tie_axis = _n(base[2] - base[0])
    return {"tie": T, "tie_axis": tie_axis, "knot": K, "cap": (cv, cf), "clip": hair_clip(sc), "paths": paths,
            "free": free}


def hair_clip(sc):
    """Cross-shaped clip on her right side above the fringe (in-game close-up): a sky-blue bar along
    the head and a white bar crossing it near its outer end. Returns box specs."""
    c = sc.surface(_dir(-40, 40), 0.016)
    n = sc.outward(c)
    along = _n(np.cross(_v(0.0, 0.0, 1.0), n))
    vert = np.cross(n, along)
    tilt = math.radians(-12)
    along, vert = along * math.cos(tilt) + vert * math.sin(tilt), -along * math.sin(tilt) + vert * math.cos(tilt)
    # Two parallel sky-blue bars along the hair flow, crossed by one white bar (in-game close-up).
    return [
        {"center": c + vert * 0.0046, "axes": (along, vert, n), "half": (0.015, 0.0012, 0.0011), "color": "#5d9fe0",
         "name": "barA"},
        {"center": c - vert * 0.0046, "axes": (along, vert, n), "half": (0.016, 0.0012, 0.0011), "color": "#5d9fe0",
         "name": "barB"},
        {"center": c + along * 0.002 + n * 0.0011, "axes": (along, vert, n), "half": (0.0011, 0.0135, 0.0011),
         "color": "#eef1f6", "name": "cross"},
    ]


# ---------------------------------------------------------------------------
# chains derived from the design
# ---------------------------------------------------------------------------
_SPRING_DEFAULTS = {
    "hair_ponytail": {"stiffness": 0.9, "drag": 0.35, "gravity": 0.6, "radius": 0.022, "wind_response": 1.0},
    "hair_bang": {"stiffness": 3.2, "drag": 0.6, "gravity": 0.1, "radius": 0.008, "wind_response": 0.3},
    "hair_side": {"stiffness": 1.6, "drag": 0.45, "gravity": 0.4, "radius": 0.01, "wind_response": 0.6},
    "hair_ahoge": {"stiffness": 3.5, "drag": 0.5, "gravity": 0.0, "radius": 0.004, "wind_response": 0.4},
    "hair_flyaway": {"stiffness": 0.4, "drag": 0.2, "gravity": 0.2, "radius": 0.004, "wind_response": 1.4},
    "ribbon_loop": {"stiffness": 3.8, "drag": 0.7, "gravity": 0.0, "radius": 0.006, "wind_response": 0.2},
    "ribbon_tail": {"stiffness": 0.5, "drag": 0.25, "gravity": 0.4, "radius": 0.005, "wind_response": 1.3},
}
_LOD = {"hair_ponytail_A": (True, True), "hair_ponytail_B": (True, False), "hair_bang": (True, False),
        "hair_side": (True, False), "ribbon_loop": (True, False), "ribbon_tail_A": (True, False)}


def chain_definitions(height, existing=()):
    """character.json secondary chains for the hair and ribbon, at reference height 1.0.

    Each chain's bones follow the designed mass exactly ("points"); spring settings are kept from an
    existing entry with the same id/side, else taken from the family defaults."""
    info = build(None, L.LockBatch(), L.LockBatch())
    old = {(c["id"], c.get("side")): c for c in existing}
    out = []
    for (cid, side), (pts, nb) in info["paths"].items():
        P = _resample(pts, nb + 1) / height
        family = cid.rsplit("_", 1)[0]
        prev = old.get((cid, side), {})
        ch = {"id": cid, "parent": "Head", "bones": nb}
        if side:
            ch["side"] = side
        length = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        d = P[1] - P[0]
        ch.update({"root": [round(float(x), 4) for x in P[0]], "direction": [round(float(x), 3) for x in d / np.linalg.norm(d)],
                   "length": round(length, 4), "curve_deg": 0,
                   "points": [[round(float(x), 4) for x in p] for p in P]})
        for k, v in _SPRING_DEFAULTS[family].items():
            ch[k] = prev.get(k, v)
        lod = _LOD.get(cid, _LOD.get(family, (False, False)))
        ch["keep_at_lod1"], ch["keep_at_lod2"] = prev.get("keep_at_lod1", lod[0]), prev.get("keep_at_lod2", lod[1])
        out.append(ch)
    return out
