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
        return verts + self.origin, faces


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
