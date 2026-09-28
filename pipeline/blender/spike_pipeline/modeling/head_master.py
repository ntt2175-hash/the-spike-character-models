"""Head master: a stylized anime head sculpted from primary forms, for every character.

Construction order (never details before the primary forms are right):
  primary mass (one smooth loft: cranium, forehead, face plane, cheeks, jaw, chin in a single silhouette;
  midface cheek fullness is part of its sections) -> chin -> eye sockets and brow -> nose (bridge, tip, wings, underside plane)
  -> lips and mouth -> ears (helix, lobe, concha)

The primary mass is a vertical loft of horizontal superellipse sections (PROFILE), so the head reads
as one continuous form from every angle; features are small volumes placed ON that surface (their
protrusion / depth is measured from the loft), so they integrate instead of floating.

Everything is relative to the eye-line center E and scaled by head_scale (1.0 = the reference anime
head, 0.167 m chin to crown). Character design comes from the `head_shape` multipliers in
character.json: the system is shared, each character's head is their own. Painted eyes, brows and
blush are decals / textures on this form.

Units: meters, Blender space (Z up, faces -Y, +X = her left).
"""
from __future__ import annotations

import numpy as np

from . import sdf

DEFAULTS = {
    "width": 1.0,          # overall head width
    "depth": 1.0,          # skull depth behind the face
    "cranium": 1.0,        # eye line -> crown
    "face_length": 1.0,    # eye line -> chin
    "cheek": 1.0,          # midface cheek volume (under the eyes, beside the nose)
    "jaw": 1.0,            # lower-face width at the jaw
    "jaw_round": 1.0,      # jaw contour: < 1 sharper / more V-shaped, > 1 softer / rounder
    "chin": 1.0,           # chin volume
    "nose": 1.0,           # nose size / projection
    "lips": 1.0,           # lip volume
    "socket": 1.0,         # eye socket depth
    "brow": 1.0,           # brow ridge volume above the soft brow plane (1 = none, 2 = pronounced)
    "ear": 1.0,            # ear size
    "asym": 0.0,           # authored asymmetry: her right midface contour this much fuller (0.015 = 1.5 %)
}

LOWER, UPPER = 0.078, 0.089          # reference head: eye line -> chin, eye line -> crown (m)
NOSE_DZ, MOUTH_DZ = -0.0345, -0.053  # nose tip / mouth line below the eye line: ~45 % and ~68 % of eye -> chin
                                     # (a high, tiny nose reads as a doll; the front reference sits lower)
EYE_T = LOWER / (LOWER + UPPER)

# Reference primary mass (head_scale 1). Rows: t (0 chin .. 1 crown), half width, y front, y back,
# y at the widest point (splits the front / back halves), front squareness (2 = elliptic, higher = a
# flatter face plane). y is measured from the head axis, the face looks toward -Y.
# Soft oval with a small chin: the jaw line is gently convex (it widens a little faster near the chin
# and slower toward the cheekbone: no straight V, no round jowl), the widest point of the lower face
# sits back toward the ear, the face plane is broad and flat across the eyes.
PROFILE = np.array([
    # t     half_w   y_front  y_back   y_wide   n_front
    [0.000, 0.0008, -0.0585, -0.0545, -0.0565, 2.0],    # chin tip: a soft point, not a ball
    [0.006, 0.0060, -0.0620, -0.0495, -0.0555, 2.0],
    [0.015, 0.0120, -0.0660, -0.0430, -0.0505, 2.1],
    [0.050, 0.0252, -0.0715, -0.0320, -0.0450, 2.2],
    [0.100, 0.0368, -0.0772, -0.0140, -0.0380, 2.25],
    [0.160, 0.0478, -0.0800, 0.0060, -0.0280, 2.35],
    [0.220, 0.0570, -0.0820, 0.0280, -0.0170, 2.45],
    [0.280, 0.0655, -0.0835, 0.0520, -0.0070, 2.55],
    [0.340, 0.0725, -0.0845, 0.0750, 0.0010, 2.6],
    [0.420, 0.0788, -0.0850, 0.0930, 0.0060, 2.6],
    [0.500, 0.0810, -0.0850, 0.1030, 0.0090, 2.6],
    [0.600, 0.0812, -0.0835, 0.1075, 0.0110, 2.5],    # the vault: a round dome (superellipse n 2.5)
    [0.700, 0.0782, -0.0797, 0.1050, 0.0120, 2.35],
    [0.800, 0.0716, -0.0719, 0.0975, 0.0120, 2.2],
    [0.880, 0.0615, -0.0601, 0.0860, 0.0120, 2.1],
    [0.940, 0.0485, -0.0448, 0.0695, 0.0120, 2.0],
    [0.975, 0.0350, -0.0290, 0.0530, 0.0120, 2.0],
    [0.990, 0.0245, -0.0168, 0.0400, 0.0120, 2.0],
    [0.997, 0.0152, -0.0059, 0.0293, 0.0120, 2.0],
    [1.000, 0.0005, 0.0115, 0.0125, 0.0120, 2.0],      # crown: sections close to a point (no end cap)
])


def _v(*a):
    return np.array(a, dtype=np.float64)


def _bump(t, t0, t1, t2, t3):
    """0 below t0, rising to 1 over [t0, t1], 1 until t2, falling to 0 at t3."""
    up = np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1)
    dn = np.clip((t3 - t) / max(t3 - t2, 1e-9), 0, 1)
    w = np.minimum(up, dn)
    return w * w * (3 - 2 * w)


class Head:
    def __init__(self, eye_center, eye_x, head_scale=1.0, shape=None):
        self.E = np.asarray(eye_center, dtype=np.float64)
        self.ex = float(eye_x)
        self.s = float(head_scale)
        self.p = dict(DEFAULTS)
        for key, val in (shape or {}).items():
            if key in DEFAULTS:
                self.p[key] = float(val)
        p, s = self.p, self.s
        self.chin_z = self.E[2] - LOWER * s * p["face_length"]
        self.crown_z = self.E[2] + UPPER * s * p["cranium"]
        self.base = sdf.ZLoft(self._stations())
        self.loft = _SideScale(self.base, self.p["asym"], self.E[2], s) if self.p["asym"] else self.base
        self._field = None

    # ------------------------------------------------------------------ primary mass
    def z_of(self, t):
        """Station height: t = 0 chin .. EYE_T eye line .. 1 crown."""
        dz = (t - EYE_T) * (LOWER + UPPER) * self.s
        return self.E[2] + dz * (self.p["face_length"] if t < EYE_T else self.p["cranium"])

    def _stations(self):
        p, s = self.p, self.s
        rows = []
        for t, hw, yf, yb, yw, nf in PROFILE:
            jaw = _bump(t, 0.0, 0.12, 0.26, 0.40)                 # the lower face below the cheekbones
            chin = _bump(t, -1.0, 0.0, 0.04, 0.2)
            w = hw * p["width"] * (1.0 + (p["jaw"] - 1.0) * jaw)
            w *= 1.0 + 0.12 * (p["jaw_round"] - 1.0) * chin           # a fuller (rounder) or narrower chin
            # Midface cheek volume lives in the section shape (fuller front corners under the eyes), never in
            # separate bumps: an ellipsoid wide enough to read as a cheek pokes out of the jaw silhouette.
            cheek = _bump(t, 0.14, 0.24, 0.34, 0.44)
            n_front = nf + 0.6 * (p["jaw_round"] - 1.0) * jaw + 0.35 * (p["cheek"] - 1.0) * cheek
            z = self.z_of(t)
            y0 = self.E[1]
            rows.append([z, w * s, y0 + yf * s, y0 + yb * s * p["depth"], n_front, 2.0, y0 + yw * s])
        return rows

    def surface_y(self, x, z):
        """Front surface y of the primary mass at (x, z) (analytic, before features)."""
        zc = np.clip(z, self.base.z0, self.base.z1)
        w, yf, _, nf, _, yw = sdf._interp_cols(zc, self.base.Z, self.base.V)
        k = max(0.0, 1.0 - abs(x / w) ** nf)
        return float(yw - (yw - yf) * k ** (1.0 / nf))

    # ------------------------------------------------------------------ placement helpers
    def dz(self, dz):
        """Eye-relative height (reference units, - = below the eyes) -> world z."""
        return self.E[2] + dz * self.s * (self.p["face_length"] if dz < 0 else self.p["cranium"])

    def on(self, x, dz, out=0.0):
        """Point on the face surface at (x, dz) (reference units), `out` in front of it (m)."""
        X, Z = x * self.s * self.p["width"], self.dz(dz)
        return _v(X, self.surface_y(X, Z) - out, Z)

    def bump(self, x, dz, out, r):
        """Ellipsoid whose front-most point stands `out` (ref units) in front of the surface."""
        r = np.asarray(r, dtype=np.float64) * self.s
        c = self.on(x, dz, out * self.s)
        c[1] += r[1]
        return sdf.Ellipsoid(c, r)

    def dent(self, x, dz, depth, r):
        """Ellipsoid that, subtracted, recesses the surface by `depth` (ref units) at its center."""
        r = np.asarray(r, dtype=np.float64) * self.s
        c = self.on(x, dz, 0.0)
        c[1] += depth * self.s - r[1]
        return sdf.Ellipsoid(c, r)

    # ------------------------------------------------------------------ the sculpt
    def ops(self):
        """[(mode, primitive, blend)] applied in order; mode 'add' or 'sub'."""
        p, s = self.p, self.s
        ops = [("add", self.loft, 0.0)]
        add = lambda prim, b: ops.append(("add", prim, b * s))  # noqa: E731
        sub = lambda prim, b: ops.append(("sub", prim, b * s))  # noqa: E731
        ex = self.ex / (s * p["width"])                         # eye center x in reference units
        # (midface cheek volume: in the loft sections, see _stations)
        # 2. chin: small, present, rounded
        add(self.bump(0.0, -LOWER + 0.012, 0.0008 * p["chin"], (0.012, 0.008, 0.009)), 0.014)
        # 3. eye sockets: shallow orbital recesses so the eyes sit IN the face; a soft brow above
        for sx in (1.0, -1.0):
            # (a separate brow ridge reads as a heavy, sleepy crease in clay: the brow stays a soft plane)
            sub(self.dent(sx * ex, -0.001, 0.0005 * p["socket"], (0.022, 0.012, 0.013)), 0.02)
            if p["brow"] > 1.0:
                add(self.bump(sx * (ex - 0.004), 0.024, 0.0002 * (p["brow"] - 1.0), (0.026, 0.01, 0.009)), 0.018)
        # 4. nose: subtle bridge rising out of the face plane, small tip, soft wings (no underside cut: in toon
        #    shading it reads as a gray patch over the lip)
        n = p["nose"]
        tip_dz = NOSE_DZ
        # A narrow bridge (a clear side plane in 3/4) that rises out of the face plane between the eyes.
        bridge_top = self.on(0.0, -0.008, -0.0016 * s)
        bridge_top[1] += 0.0016 * s
        bridge_low = self.on(0.0, tip_dz + 0.0055, 0.0034 * s * n)
        bridge_low[1] += 0.0027 * s * n
        add(sdf.RoundCone(bridge_top, bridge_low, 0.0016 * s, 0.0027 * s * n), 0.007)
        add(self.bump(0.0, tip_dz, 0.0052 * n, (0.0047 * n, 0.0042 * n, 0.0040 * n)), 0.006)
        for sx in (1.0, -1.0):
            add(self.bump(sx * 0.0043 * n, tip_dz - 0.0018, 0.0022 * n, (0.0026, 0.0024, 0.0022)), 0.004)
        # 5. lips and mouth: small, with volume, set into the lower face (not stuck on)
        m_dz = MOUTH_DZ
        L = p["lips"]
        add(self.bump(0.0, m_dz + 0.0032, 0.0006 * L, (0.0105, 0.0036, 0.0028 * L)), 0.008)
        add(self.bump(0.0, m_dz - 0.0042, 0.0008 * L, (0.0088, 0.0038, 0.0032 * L)), 0.008)
        sub(self.dent(0.0, m_dz, 0.0005, (0.0082, 0.0045, 0.0022)), 0.004)
        # the soft crease under the lower lip: the mouth sits IN the lower face, the chin pad below it
        sub(self.dent(0.0, m_dz - 0.0098, 0.0004, (0.011, 0.005, 0.0032)), 0.006)
        # 6. ears: helix body tilted back, lobe, concha; between the eye line and the nose base
        e = p["ear"]
        R = sdf.rotation_to(_v(0.0, 0.2, 0.98))
        for sx in (1.0, -1.0):
            c = self.E + _v(sx * 0.0752 * p["width"], 0.0055, -0.019 * p["face_length"]) * s
            add(sdf.Ellipsoid(c, _v(0.0056, 0.0118, 0.0185) * s * e, R), 0.005)
            add(sdf.Ellipsoid(c + _v(sx * 0.0012, 0.0015, -0.0145) * s * e, _v(0.0042, 0.0058, 0.0062) * s * e), 0.004)
            sub(sdf.Ellipsoid(c + _v(sx * 0.0052, -0.0008, -0.0015) * s * e, _v(0.0036, 0.0068, 0.0098) * s * e, R), 0.0025)
        return ops

    def bounds(self, pad=0.012):
        s, p = self.s, self.p
        lo = _v(-0.1 * s * p["width"], self.E[1] - 0.11 * s, self.chin_z - 0.015 * s) - pad
        hi = _v(0.1 * s * p["width"], self.E[1] + 0.125 * s * p["depth"], self.crown_z + 0.01 * s) + pad
        return lo, hi

    def field(self, voxel=0.0011, extra=()):
        """The head as a distance field (cached when no extra parts are given). `extra` = [(prim, blend)]
        unioned first (e.g. the neck, so the jaw -> neck transition is one surface)."""
        if self._field is not None and not extra:
            return self._field
        lo, hi = self.bounds()
        f = sdf.Field(lo, hi, voxel)
        for prim, b in extra:
            f.union(prim, b)
        for mode, prim, b in self.ops():
            if mode == "add":
                f.union(prim, b)
            else:
                f.subtract(prim, b)
        if not extra:
            self._field = f
        return f

    def front_y(self, x, z):
        """Y of the head's front surface at (x, z), features included; None if the ray misses."""
        f = self.field()
        ys = np.linspace(self.E[1] - 0.12 * self.s, self.E[1] + 0.06 * self.s, 480)
        P = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], axis=1)
        d = f.sample_at(P)
        hit = np.nonzero(d < 0)[0]
        return float(ys[hit[0]]) if len(hit) else None

    # ------------------------------------------------------------------ landmarks for painting / rigging
    def landmarks(self):
        return {"nose_tip_z": self.dz(NOSE_DZ), "mouth_z": self.dz(MOUTH_DZ), "chin_z": self.chin_z,
                "crown_z": self.crown_z, "eye_z": float(self.E[2])}


class _SideScale:
    """Authored asymmetry for the primary mass: her right side (x < 0) is widened by `a` over the midface
    (cheekbone to jaw), fading out above the eyes and at the chin. Small values only (a few percent)."""

    def __init__(self, prim, a, eye_z, s):
        self.prim, self.a, self.ez, self.s = prim, float(a), float(eye_z), float(s)
        self.z0, self.z1 = prim.z0, prim.z1

    def bbox(self):
        lo, hi = self.prim.bbox()
        k = 1.0 + abs(self.a)
        return np.array([lo[0] * k, lo[1], lo[2]]), np.array([hi[0] * k, hi[1], hi[2]])

    def eval(self, P):
        dz = (P[..., 2] - self.ez) / self.s
        w = _bump(dz, -0.07, -0.05, -0.012, 0.004)
        k = 1.0 + self.a * w * (P[..., 0] < 0)
        Q = np.array(P, dtype=np.float64, copy=True)
        Q[..., 0] = Q[..., 0] / k
        return self.prim.eval(Q)
