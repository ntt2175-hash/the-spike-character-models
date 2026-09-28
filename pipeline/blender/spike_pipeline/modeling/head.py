"""Anime head loft.

A head is lofted from horizontal slices, the way a modeler blocks a head from
front and side orthographic references: for each height t (0 = chin, 1 =
crown) the recipe gives the front-most and back-most extents (side profile),
the half width and the Y of the widest point (front profile), and
superellipse exponents that flatten the face plane while keeping the skull
round. Small features (nose, lips, sockets, cheeks) are Gaussian
displacements, because an anime face is mostly plane changes, not detail.

The mesh is a clean ring/segment quad grid (deformation-friendly loops
around the face), so regions such as the eye area can be addressed by index
for decals and weights.
"""
from __future__ import annotations

import math

import numpy as np


def _interp(ts, vals, t):
    return float(np.interp(t, ts, vals))


class HeadProfile:
    def __init__(self, chin_z, crown_z, t, y_front, y_back, half_width, y_wide, n_front, n_back):
        self.z0, self.z1 = chin_z, crown_z
        self.t = np.asarray(t, dtype=np.float64)
        self.yf = np.asarray(y_front, dtype=np.float64)
        self.yb = np.asarray(y_back, dtype=np.float64)
        self.w = np.asarray(half_width, dtype=np.float64)
        self.yw = np.asarray(y_wide, dtype=np.float64)
        self.nf = np.asarray(n_front, dtype=np.float64)
        self.nb = np.asarray(n_back, dtype=np.float64)

    def z(self, t):
        return self.z0 + (self.z1 - self.z0) * t

    def t_of(self, z):
        return (z - self.z0) / (self.z1 - self.z0)

    def section(self, t, angles):
        w = _interp(self.t, self.w, t)
        yw = _interp(self.t, self.yw, t)
        f = max(yw - _interp(self.t, self.yf, t), 1e-4)
        b = max(_interp(self.t, self.yb, t) - yw, 1e-4)
        nf, nb = _interp(self.t, self.nf, t), _interp(self.t, self.nb, t)
        c, s = np.cos(angles), np.sin(angles)
        front = s >= 0.0  # angle pi/2 faces -Y (front)
        n = np.where(front, nf, nb)
        x = w * np.sign(c) * np.abs(c) ** (2.0 / n)
        y = np.where(front, yw - f * np.abs(s) ** (2.0 / nf), yw + b * np.abs(s) ** (2.0 / nb))
        return x, y

    def front_y(self, x, t):
        """Face surface Y at (x, t) on the front half."""
        w = _interp(self.t, self.w, t)
        yw = _interp(self.t, self.yw, t)
        f = yw - _interp(self.t, self.yf, t)
        nf = _interp(self.t, self.nf, t)
        k = max(0.0, 1.0 - abs(x / w) ** nf)
        return yw - f * k ** (1.0 / nf)


class Feature:
    """Gaussian displacement toward -Y (front) in meters. Negative = recess."""

    def __init__(self, x, t, amount, sx, st, mirror=False):
        self.x, self.t, self.a, self.sx, self.st, self.mirror = x, t, amount, sx, st, mirror

    def value(self, X, T):
        v = self.a * np.exp(-((X - self.x) ** 2) / (2 * self.sx ** 2) - ((T - self.t) ** 2) / (2 * self.st ** 2))
        if self.mirror:
            v = v + self.a * np.exp(-((X + self.x) ** 2) / (2 * self.sx ** 2) - ((T - self.t) ** 2) / (2 * self.st ** 2))
        return v


def loft(profile: HeadProfile, features=(), rings=84, segments=112):
    """Return (verts (N,3), quads list, tris list, grid index array [ring, seg], front mask per vertex)."""
    ts = np.linspace(0.0, 1.0, rings + 1) ** 1.0
    ts = 0.5 - 0.5 * np.cos(np.pi * ts)  # denser rings at chin and crown
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    verts = []
    grid = np.zeros((rings + 1, segments), dtype=np.int64)
    for i, t in enumerate(ts[1:-1], start=1):
        x, y = profile.section(t, angles)
        z = np.full_like(x, profile.z(t))
        for j in range(segments):
            grid[i, j] = len(verts)
            verts.append([x[j], y[j], z[j]])
    verts = np.asarray(verts, dtype=np.float64)
    # Features displace along the horizontal outward normal, weighted toward the front.
    T = profile.t_of(verts[:, 2])
    disp = np.zeros(len(verts))
    for f in features:
        disp += f.value(verts[:, 0], T)
    yw = np.interp(T, profile.t, profile.yw)
    frontness = np.clip((yw - verts[:, 1]) / 0.04, 0.0, 1.0)
    verts[:, 1] -= disp * frontness
    # Poles: chin tip and crown.
    chin_x, chin_y = profile.section(ts[0] + 1e-4, np.array([np.pi / 2]))
    chin = [0.0, float(_interp(profile.t, profile.yf, 0.0) * 0.5 + _interp(profile.t, profile.yb, 0.0) * 0.5), profile.z0]
    crown = [0.0, float(_interp(profile.t, profile.yw, 1.0)), profile.z1]
    ci = len(verts)
    verts = np.vstack([verts, [chin, crown]])
    quads, tris = [], []
    for i in range(1, rings - 1):
        for j in range(segments):
            jn = (j + 1) % segments
            quads.append([grid[i, j], grid[i, jn], grid[i + 1, jn], grid[i + 1, j]])
    for j in range(segments):
        jn = (j + 1) % segments
        tris.append([ci, grid[1, jn], grid[1, j]])
        tris.append([ci + 1, grid[rings - 1, j], grid[rings - 1, jn]])
    front = np.concatenate([frontness, [1.0, 0.0]])
    return verts, quads, tris, grid, front


def ring_seg_of(profile: HeadProfile, grid_shape, t, angle):
    rings = grid_shape[0] - 1
    segments = grid_shape[1]
    u = math.acos(1.0 - 2.0 * t) / math.pi
    return int(round(u * rings)), int(round(angle / (2 * math.pi) * segments)) % segments
