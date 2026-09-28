"""Signed-distance sculpting.

Forms are built the way a sculptor blocks them: shaped primitives (tapered
capsules for limbs, ellipsoids for muscle masses) blended with explicit
smooth-union radii, so anatomical transitions (neck to trapezius, deltoid to
arm, glute to thigh, vastus to knee) are continuous surfaces rather than
intersecting parts. Garments are offset shells of the body field with real
thickness, cut by planes and tubes (hem, neckline, armholes, leg openings).

Distances are in meters. Fields are evaluated on a regular grid (numpy) and
meshed with marching cubes (scikit-image), then cleaned in Blender.
"""
from __future__ import annotations

import numpy as np

BIG = 1.0


def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def smin(a, b, k):
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------------------------------------------------------------------------
# primitives: eval(P) with P shaped (..., 3); bbox() -> (min, max)
# ---------------------------------------------------------------------------
class RoundCone:
    """Tapered capsule from a (radius ra) to b (radius rb). Exact SDF."""

    def __init__(self, a, b, ra, rb):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.ra, self.rb = float(ra), float(rb)

    def bbox(self):
        r = max(self.ra, self.rb)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def eval(self, P):
        a, b, r1, r2 = self.a, self.b, self.ra, self.rb
        ba = b - a
        l2 = float(ba @ ba)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        pa = P - a
        y = pa @ ba
        z = y - l2
        q = pa * l2 - y[..., None] * ba
        x2 = _dot(q, q)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d_tail = np.sqrt(x2 + z2) * il2 - r2
        d_head = np.sqrt(x2 + y2) * il2 - r1
        d_body = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
        out = np.where(np.sign(y) * a2 * y2 < k, d_head, d_body)
        return np.where(np.sign(z) * a2 * z2 > k, d_tail, out)


def Capsule(a, b, r):
    return RoundCone(a, b, r, r)


class Ellipsoid:
    """Ellipsoid with optional rotation (3x3, local->world). Good approximate SDF."""

    def __init__(self, center, radii, rot=None):
        self.c = np.asarray(center, dtype=np.float64)
        self.r = np.asarray(radii, dtype=np.float64)
        self.R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)

    def bbox(self):
        ext = np.abs(self.R) @ self.r
        return self.c - ext, self.c + ext

    def eval(self, P):
        p = (P - self.c) @ self.R  # world -> local (R orthonormal)
        k0 = np.sqrt(_dot(p / self.r, p / self.r))
        k1 = np.sqrt(_dot(p / (self.r * self.r), p / (self.r * self.r)))
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


class RoundBox:
    def __init__(self, center, half, radius, rot=None):
        self.c = np.asarray(center, dtype=np.float64)
        self.h = np.asarray(half, dtype=np.float64)
        self.rad = float(radius)
        self.R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)

    def bbox(self):
        ext = np.abs(self.R) @ (self.h + self.rad)
        return self.c - ext, self.c + ext

    def eval(self, P):
        p = (P - self.c) @ self.R
        q = np.abs(p) - self.h
        outside = np.sqrt(_dot(np.maximum(q, 0.0), np.maximum(q, 0.0)))
        inside = np.minimum(np.maximum(q[..., 0], np.maximum(q[..., 1], q[..., 2])), 0.0)
        return outside + inside - self.rad


class HalfSpace:
    """Solid on the side OPPOSITE the normal (d < 0 where (p - point) . n < 0)."""

    def __init__(self, point, normal):
        self.p = np.asarray(point, dtype=np.float64)
        n = np.asarray(normal, dtype=np.float64)
        self.n = n / np.linalg.norm(n)

    def bbox(self):
        return None

    def eval(self, P):
        return (P - self.p) @ self.n


class Transformed:
    """Wrap a primitive with a per-point warp: eval(P) = prim.eval(warp(P)) * scale."""

    def __init__(self, prim, warp, scale=1.0, bbox=None):
        self.prim, self.warp, self.scale, self._bbox = prim, warp, scale, bbox

    def bbox(self):
        return self._bbox if self._bbox is not None else self.prim.bbox()

    def eval(self, P):
        return self.prim.eval(self.warp(P)) * self.scale


# ---------------------------------------------------------------------------
# lofts: smooth, continuous anatomical forms (the body is sculpted as profiles, not blobs)
# ---------------------------------------------------------------------------
def smooth_table(ts, vals, n=256):
    """Dense, C1-smooth resampling of station values (Catmull-Rom through the stations).
    ts: (k,) increasing; vals: (k, m). Returns (t_dense (n,), v_dense (n, m))."""
    ts = np.asarray(ts, dtype=np.float64)
    V = np.asarray(vals, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    P = np.concatenate([ts[:, None], V], axis=1)
    P = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    per = max(4, n // (len(ts) - 1))
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(per):
            t = k / per
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    out = np.asarray(out)
    order = np.argsort(out[:, 0], kind="stable")
    return out[order, 0], out[order, 1:]


def _interp_cols(t, T, V):
    return [np.interp(t, T, V[:, j]) for j in range(V.shape[1])]


def _superellipse_distance(x, y, rx, ry, n):
    """First-order distance to |x/rx|^n + |y/ry|^n = 1 (exact zero set, well-behaved near it)."""
    ax, ay = np.abs(x) / rx, np.abs(y) / ry
    s = ax ** n + ay ** n
    k = s ** (1.0 / n)
    kk = np.maximum(k, 1e-9)
    gx = kk ** (1.0 - n) * np.maximum(ax, 1e-12) ** (n - 1.0) / rx
    gy = kk ** (1.0 - n) * np.maximum(ay, 1e-12) ** (n - 1.0) / ry
    g = np.sqrt(gx * gx + gy * gy)
    d = (k - 1.0) / np.maximum(g, 1e-9)
    return np.maximum(d, -np.minimum(rx, ry))


class Loft:
    """A limb segment from a to b whose cross-section is an asymmetric ellipse that varies smoothly
    along the segment: stations rows (t, r_out, r_in, r_front, r_back), t in [0, 1]. u_hint points to
    the 'out' side (away from the body midline), v_hint to the 'front'. Beyond the ends the section
    is held for ext[i] and then closed with a rounded cap, so segments meeting at a joint with matched
    radii read as one continuous limb."""

    def __init__(self, a, b, stations, u_hint, v_hint=(0.0, -1.0, 0.0), ext=(0.0, 0.0), n=2.0, cap=(1.0, 1.0)):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        ax = self.b - self.a
        self.L = float(np.linalg.norm(ax))
        self.ax = ax / self.L
        u = np.asarray(u_hint, dtype=np.float64)
        u = u - self.ax * (u @ self.ax)
        self.u = u / np.linalg.norm(u)
        v = np.cross(self.ax, self.u)
        self.v = v if v @ np.asarray(v_hint, dtype=np.float64) >= 0 else -v
        st = np.asarray(stations, dtype=np.float64)
        self.T, self.R = smooth_table(st[:, 0], st[:, 1:5])
        self.ext = ext
        self.n = n
        self.cap = cap           # cap length as a fraction of the end radius (1 = hemisphere, <1 = flattened)

    def bbox(self):
        r = float(self.R.max()) * max(1.0, *self.cap) + max(self.ext)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def eval(self, P):
        q = P - self.a
        t = (q @ self.ax) / self.L
        x, y = q @ self.u, q @ self.v
        tc = np.clip(t, 0.0, 1.0)
        ro, ri, rf, rb = _interp_cols(tc, self.T, self.R)
        rx = np.where(x >= 0, ro, ri)
        ry = np.where(y >= 0, rf, rb)
        e = np.where(t < 0, np.maximum(-t * self.L - self.ext[0], 0.0), np.maximum((t - 1.0) * self.L - self.ext[1], 0.0))
        rc = np.minimum(rx, ry) * np.where(t < 0.5, self.cap[0], self.cap[1])
        sc = np.sqrt(np.clip(1.0 - (e / rc) ** 2, 0.0, 1.0))
        d = _superellipse_distance(x, y, np.maximum(rx * sc, 1e-5), np.maximum(ry * sc, 1e-5), self.n)
        beyond = e >= rc
        return np.where(beyond, np.sqrt(x * x + y * y + (e - rc) ** 2), np.where(e > 0, np.maximum(d, e - rc), d))


class ZLoft:
    """Vertical loft (torso, head): horizontal superellipse sections |x/w|^n + |(y - yc)/d|^n = 1 with separate
    front and back depth and exponent, varying smoothly with height. stations rows
    (z, half_width, y_front, y_back, n_front, n_back[, y_wide]); y_wide (default: midway) is where the
    section is widest, splitting the front and back halves. Rounded caps close the ends."""

    def __init__(self, stations, x0=0.0):
        st = np.asarray(stations, dtype=np.float64)
        if st.shape[1] == 6:
            st = np.hstack([st, 0.5 * (st[:, 2:3] + st[:, 3:4])])
        self.Z, self.V = smooth_table(st[:, 0], st[:, 1:7])
        self.z0, self.z1 = float(st[0, 0]), float(st[-1, 0])
        self.x0 = x0

    def bbox(self):
        w = float(self.V[:, 0].max())
        return (np.array([self.x0 - w, float(self.V[:, 1].min()), self.z0 - w]),
                np.array([self.x0 + w, float(self.V[:, 2].max()), self.z1 + w]))

    def eval(self, P):
        z = P[..., 2]
        zc = np.clip(z, self.z0, self.z1)
        w, yf, yb, nf, nb, yc = _interp_cols(zc, self.Z, self.V)
        dy = P[..., 1] - yc
        ry = np.where(dy < 0, yc - yf, yb - yc)
        n = np.where(dy < 0, nf, nb)
        e = np.where(z < self.z0, self.z0 - z, np.where(z > self.z1, z - self.z1, 0.0))
        rc = np.minimum(w, ry)
        sc = np.sqrt(np.clip(1.0 - (e / rc) ** 2, 0.0, 1.0))
        x = P[..., 0] - self.x0
        d = _superellipse_distance(x, dy, np.maximum(w * sc, 1e-5), np.maximum(ry * sc, 1e-5), n)
        beyond = e >= rc
        return np.where(beyond, np.sqrt(x * x + dy * dy + (e - rc) ** 2), np.where(e > 0, np.maximum(d, e - rc), d))


# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------
class Field:
    def __init__(self, bmin, bmax, voxel):
        self.voxel = float(voxel)
        self.origin = np.asarray(bmin, dtype=np.float64)
        size = np.asarray(bmax, dtype=np.float64) - self.origin
        self.shape = tuple(int(np.ceil(s / voxel)) + 1 for s in size)
        self.d = np.full(self.shape, BIG, dtype=np.float32)

    def copy(self):
        f = Field.__new__(Field)
        f.voxel, f.origin, f.shape = self.voxel, self.origin.copy(), self.shape
        f.d = self.d.copy()
        return f

    def _window(self, prim, pad):
        bb = prim.bbox()
        if bb is None:
            return tuple(slice(0, n) for n in self.shape)
        lo = np.floor((bb[0] - pad - self.origin) / self.voxel).astype(int)
        hi = np.ceil((bb[1] + pad - self.origin) / self.voxel).astype(int) + 1
        lo = np.clip(lo, 0, np.array(self.shape))
        hi = np.clip(hi, 0, np.array(self.shape))
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    def points(self, win):
        axes = [self.origin[i] + self.voxel * np.arange(win[i].start, win[i].stop) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return np.stack([X, Y, Z], axis=-1)

    def sample(self, prim, win=None):
        win = win or tuple(slice(0, n) for n in self.shape)
        return prim.eval(self.points(win)).astype(np.float32)

    def union(self, prim, k=0.0):
        win = self._window(prim, k + 3 * self.voxel)
        if any(s.stop <= s.start for s in win):
            return self
        self.d[win] = smin(self.d[win], self.sample(prim, win), k)
        return self

    def subtract(self, prim, k=0.0):
        win = self._window(prim, k + 3 * self.voxel)
        if any(s.stop <= s.start for s in win):
            return self
        self.d[win] = smax(self.d[win], -self.sample(prim, win), k)
        return self

    def intersect(self, prim, k=0.0):
        self.d = smax(self.d, self.sample(prim), k).astype(np.float32)
        return self

    def union_field(self, other: "Field", k=0.0):
        self.d = smin(self.d, other.d, k).astype(np.float32)
        return self

    def sample_at(self, P):
        """Trilinear sample of the field at arbitrary points (N,3); outside the grid returns BIG."""
        P = np.asarray(P, dtype=np.float64)
        g = (P - self.origin) / self.voxel
        i0 = np.floor(g).astype(int)
        f = g - i0
        shape = np.array(self.shape)
        inside = np.all((i0 >= 0) & (i0 < shape - 1), axis=1)
        out = np.full(len(P), BIG, dtype=np.float64)
        if not inside.any():
            return out
        i0, f = i0[inside], f[inside]
        d = self.d
        acc = np.zeros(len(i0))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                    acc += w * d[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        out[inside] = acc
        return out

    def drape(self, sigma, grow=0.0):
        """Fabric bridging: Gaussian-smooth the distance field (sigma in m) so cloth spans small creases
        and concavities instead of following them, then grow by `grow` to compensate shrinkage."""
        from scipy import ndimage
        self.d = ndimage.gaussian_filter(self.d, sigma / self.voxel, mode="nearest").astype(np.float32) - np.float32(grow)
        return self

    def offset(self, amount):
        """Grow (positive) or shrink the solid."""
        self.d = self.d - np.float32(amount)
        return self

    def shell(self, thickness, outward=True):
        """Turn the solid into a thin shell of the given thickness around its surface."""
        half = thickness * 0.5
        base = self.d - np.float32(half if outward else -half)
        self.d = (np.abs(base) - np.float32(half)).astype(np.float32)
        return self

    def mesh(self, level=0.0):
        from skimage import measure

        verts, faces, _normals, _vals = measure.marching_cubes(self.d, level=level, spacing=(self.voxel,) * 3,
                                                                gradient_direction="ascent")
        # skimage winds these triangles clockwise seen from outside; reverse them so every surface has
        # outward (right-handed) normals: decals, inverted-hull outlines and back-face culling rely on it.
        return verts + self.origin, faces[:, ::-1].copy()


# ---------------------------------------------------------------------------
# Blender glue
# ---------------------------------------------------------------------------
def to_blender_mesh(name, verts, faces):
    import bpy

    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    me.loops.add(len(faces) * 3)
    me.loops.foreach_set("vertex_index", np.asarray(faces, dtype=np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()
    return me


def rotation_to(direction, up=(0.0, 0.0, 1.0)):
    """3x3 local->world rotation whose local Z points along direction."""
    z = np.asarray(direction, dtype=np.float64)
    z = z / np.linalg.norm(z)
    u = np.asarray(up, dtype=np.float64)
    if abs(float(z @ u)) > 0.98:
        u = np.array([0.0, 1.0, 0.0])
    x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
