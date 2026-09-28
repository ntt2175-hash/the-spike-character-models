"""Hair locks.

An anime hairstyle is a set of designed locks, not a helmet and not random
spikes. A lock is swept along a Catmull-Rom path with a lens-shaped cross
section that lies flat on the head (thin axis = away from the scalp), a
crescent curvature that wraps it around the skull, a root ease, a long body
and a sharp tapered point, optional twist and a curl at the tip.

Returns plain arrays so recipes can place hundreds of locks quickly; the
per-vertex 'along' parameter (0 root .. 1 tip) drives highlights and weights.
"""
from __future__ import annotations

import math

import numpy as np


def catmull_rom(points, samples_per_segment=10):
    P = [np.asarray(p, dtype=np.float64) for p in points]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(samples_per_segment):
            t = k / samples_per_segment
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.asarray(out)


def _smooth(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def default_width(u, tip_start=0.45, root=0.12, root_min=0.7, tip_power=0.75):
    root_k = root_min + (1.0 - root_min) * _smooth(0.0, root, u)
    tip_k = 1.0 - _smooth(tip_start, 1.0, u) ** tip_power
    return max(root_k * tip_k, 0.0)


def lock(points, width, thickness, *, outward, ring=10, samples=10, tip_start=0.45, tip_power=0.75, root_min=0.7,
         crescent=0.18, twist_deg=0.0, thickness_tip=0.25, width_fn=None, thickness_fn=None):
    """Sweep one lock.

    points   : control points (root first)
    width    : max width (m)
    thickness: max thickness (m)
    outward  : callable(p) -> unit vector pointing away from the scalp at p (flat axis), or a fixed vector
    thickness_fn: optional callable(u) -> thickness multiplier (e.g. thin roots that grow out of the scalp)
    Returns (verts (N,3), faces (list of index tuples), along (N,))
    """
    path = catmull_rom(points, samples)
    n = len(path)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = s / max(s[-1], 1e-9)
    verts, along = [], []
    prev_w = None
    for i in range(n):
        p = path[i]
        t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t) or 1.0
        out = np.asarray(outward(p) if callable(outward) else outward, dtype=np.float64)
        f = out - t * (out @ t)
        if np.linalg.norm(f) < 1e-6:
            f = np.cross(t, [1.0, 0.0, 0.0])
        f /= np.linalg.norm(f)
        w_axis = np.cross(t, f)
        if prev_w is not None and w_axis @ prev_w < 0:
            w_axis = -w_axis
            f = -f
        prev_w = w_axis
        tw = math.radians(twist_deg) * u[i]
        if tw:
            ca, sa = math.cos(tw), math.sin(tw)
            w_axis, f = w_axis * ca + f * sa, -w_axis * sa + f * ca
        wk = width_fn(u[i]) if width_fn else default_width(u[i], tip_start, 0.12, root_min, tip_power)
        hw = 0.5 * width * wk
        th = 0.5 * thickness * (thickness_tip + (1.0 - thickness_tip) * wk) * (thickness_fn(u[i]) if thickness_fn else 1.0)
        for j in range(ring):
            a = 2.0 * math.pi * j / ring
            x = math.cos(a)
            y = math.sin(a)
            # Lens: thin at the side edges; crescent: edges curl toward the scalp.
            y = y * th * (0.35 + 0.65 * abs(y) ** 0.5)
            y -= crescent * hw * (x * x)
            verts.append(p + w_axis * (x * hw) + f * y)
            along.append(u[i])
    faces = []
    for i in range(n - 1):
        for j in range(ring):
            jn = (j + 1) % ring
            a, b = i * ring + j, i * ring + jn
            c, d = (i + 1) * ring + jn, (i + 1) * ring + j
            faces.append((a, b, c, d))
    # Caps: root fan and a tip point.
    root_c = len(verts)
    verts.append(path[0])
    along.append(0.0)
    for j in range(ring):
        faces.append((root_c, (j + 1) % ring, j))
    tip_c = len(verts)
    verts.append(path[-1] + (path[-1] - path[-2]) * 0.6)
    along.append(1.0)
    base = (n - 1) * ring
    for j in range(ring):
        faces.append((base + j, base + (j + 1) % ring, tip_c))
    return np.asarray(verts), faces, np.asarray(along)


class LockBatch:
    """Accumulates many locks into one mesh (with per-vertex along / lock id)."""

    def __init__(self):
        self.verts, self.faces, self.along, self.lock_id, self.tags = [], [], [], [], []
        self._n = 0

    def add(self, verts, faces, along, tag=""):
        lid = len(self.tags)
        self.tags.append(tag)
        self.verts.append(verts)
        self.faces.extend([tuple(i + self._n for i in f) for f in faces])
        self.along.append(along)
        self.lock_id.append(np.full(len(verts), lid))
        self._n += len(verts)
        return lid

    def arrays(self):
        return (np.concatenate(self.verts), self.faces, np.concatenate(self.along), np.concatenate(self.lock_id))


def sheet(columns, thickness_fn, ridge_fn=None, center=None, col_scale=None):
    """A closed shell swept from a list of column paths (same point count each, root first).

    Used for masses that must read as one surface and only split at the edge (a fringe cut into
    points): the front surface follows the columns, the back is offset toward `center` by
    thickness_fn(v) (times col_scale per column), and all borders are closed with rims. ridge_fn(i, v) pushes the front surface
    outward to model shallow lock ridges. Returns (verts, faces, along).
    """
    F = np.asarray(columns, dtype=np.float64)            # (n_cols, n_v, 3)
    nc, nv, _ = F.shape
    dI = np.gradient(F, axis=0)
    dV = np.gradient(F, axis=1)
    N = np.cross(dI, dV)
    N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-12)
    c = np.asarray(center if center is not None else F.reshape(-1, 3).mean(axis=0))
    flip = np.sum(N * (F - c), axis=2) < 0
    N[flip] *= -1.0
    v = np.linspace(0.0, 1.0, nv)
    if ridge_fn is not None:
        R = np.array([[ridge_fn(i, vv) for vv in v] for i in range(nc)])
        F = F + N * R[..., None]
    T = np.array([thickness_fn(vv) for vv in v])[None, :, None]
    if col_scale is not None:
        T = T * np.asarray(col_scale, dtype=np.float64)[:, None, None]
    B = F - N * T
    verts = np.concatenate([F.reshape(-1, 3), B.reshape(-1, 3)])
    off = nc * nv
    idx = lambda i, j: i * nv + j  # noqa: E731
    faces = []
    for i in range(nc - 1):
        for j in range(nv - 1):
            a, b, cc, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces.append((a, b, cc, d))
            faces.append((d + off, cc + off, b + off, a + off))
    for i in range(nc - 1):                               # root and edge rims
        for j in (0, nv - 1):
            a, b = idx(i, j), idx(i + 1, j)
            faces.append((a, b, b + off, a + off) if j else (b, a, a + off, b + off))
    for j in range(nv - 1):                               # side rims
        for i in (0, nc - 1):
            a, b = idx(i, j), idx(i, j + 1)
            faces.append((a, b, b + off, a + off) if i == 0 else (b, a, a + off, b + off))
    along = np.concatenate([np.tile(v, nc), np.tile(v, nc)])
    return verts, faces, along
