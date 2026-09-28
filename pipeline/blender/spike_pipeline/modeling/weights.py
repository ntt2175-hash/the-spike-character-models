"""Skin weights from bone segments.

Distance-to-segment weights with a smooth falloff, limited to the four
strongest influences and normalized, then relaxed over the mesh graph so
joints bend with a gradient instead of a crease. Recipes restrict which bones
a part may use (a jersey hem never follows the thigh; a bang only follows its
own chain), which is most of what makes procedural weights behave.
"""
from __future__ import annotations

import numpy as np


def segment_distance(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(P - closest, axis=1), t


def compute(P, segments, power=4.0, bias=None, max_influences=4):
    """P: (N,3). segments: list of (bone_name, head, tail). Returns (names, W (N, B))."""
    names = [s[0] for s in segments]
    D = np.stack([segment_distance(P, np.asarray(h, float), np.asarray(t, float))[0] for _, h, t in segments], axis=1)
    if bias is not None:
        D = D * np.asarray([bias.get(n, 1.0) for n in names])[None, :]
    W = 1.0 / np.maximum(D, 1e-4) ** power
    if max_influences and W.shape[1] > max_influences:
        idx = np.argsort(-W, axis=1)[:, max_influences:]
        np.put_along_axis(W, idx, 0.0, axis=1)
    W /= W.sum(axis=1, keepdims=True)
    return names, W


def relax(W, faces, n_verts, iterations=3, factor=0.5):
    """Laplacian smoothing of weights over mesh edges."""
    edges = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edges.add((min(a, b), max(a, b)))
    e = np.asarray(sorted(edges))
    for _ in range(iterations):
        acc = np.zeros_like(W)
        cnt = np.zeros(n_verts)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        np.add.at(acc, e[:, 1], W[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1)
        np.add.at(cnt, e[:, 1], 1)
        avg = acc / np.maximum(cnt, 1)[:, None]
        W = W * (1 - factor) + avg * factor
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    return W


def limit(W, max_influences=4):
    if W.shape[1] > max_influences:
        idx = np.argsort(-W, axis=1)[:, max_influences:]
        W = W.copy()
        np.put_along_axis(W, idx, 0.0, axis=1)
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)


def apply_to_object(obj, names, W, threshold=1e-3):
    """Write weights as vertex groups (replacing existing groups with the same names)."""
    for n in names:
        g = obj.vertex_groups.get(n)
        if g is not None:
            obj.vertex_groups.remove(g)
    groups = [obj.vertex_groups.new(name=n) for n in names]
    for b, g in enumerate(groups):
        col = W[:, b]
        idx = np.nonzero(col > threshold)[0]
        # Group by identical weight values is not worth it; add per vertex.
        for i in idx:
            g.add([int(i)], float(col[i]), "REPLACE")
