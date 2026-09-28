#!/usr/bin/env python3
"""Instant gray-clay head preview without Blender: sphere-traces a character's head-master SDF from the
front, 3/4, side and a slight low angle, so primary facial forms can be judged in seconds.

    python pipeline/tools/head_preview.py <character_id> <out.png> [--voxel 0.0012] [--size 360]

Judge forms here (cranium, face mass, cheeks, jaw, chin, sockets, nose, lips, ears); judge the painted
face and the final look in Blender renders.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender")]
import spike_specs as ss  # noqa: E402
from spike_pipeline.modeling import head_master as HM  # noqa: E402

VIEWS = [("front", 0, 0), ("3/4", 35, 0), ("side", 90, 0), ("low 3/4", 30, -14), ("3/4 left", -35, 3)]


def head_for(char_id: str, eye_x: float | None = None) -> HM.Head:
    c = ss.load_character(char_id)
    bones = {b["name"]: b for b in ss.build_character_skeleton(ss.load_spec("skeleton"), c)}
    hs = float(c["proportions"].get("head_scale", 1.12))
    ez = float(bones["LeftEye"]["head"][2])
    ex = eye_x if eye_x is not None else 0.0415 * hs / 1.12
    return HM.Head((0.0, 0.012, ez), ex, hs, c.get("head_shape", {}))


def render(field, center, yaw, pitch, size, half):
    """Orthographic sphere trace; the camera orbits the head (yaw from the front toward her left)."""
    a, b = math.radians(yaw), math.radians(pitch)
    fwd = np.array([-math.sin(a) * math.cos(b), math.cos(a) * math.cos(b), -math.sin(b)])   # view dir (into the face)
    right = np.array([math.cos(a), math.sin(a), 0.0])
    up = np.cross(right, fwd)
    u = np.linspace(-half, half, size)
    U, V = np.meshgrid(u, -u)
    O = center[None] + U.reshape(-1, 1) * right + V.reshape(-1, 1) * up - fwd * 0.25
    t = np.zeros(len(O))
    hit = np.zeros(len(O), bool)
    alive = np.ones(len(O), bool)
    for _ in range(400):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        P = O[idx] + t[idx, None] * fwd
        d = field.sample_at(P)
        done = d < 2e-4
        hit[idx[done]] = True
        t[idx] += np.clip(d, 2e-4, 0.003) * 0.9
        alive[idx[done | (t[idx] > 0.5)]] = False
    P = O + t[:, None] * fwd
    e = field.voxel
    n = np.stack([field.sample_at(P + [e, 0, 0]) - field.sample_at(P - [e, 0, 0]),
                  field.sample_at(P + [0, e, 0]) - field.sample_at(P - [0, e, 0]),
                  field.sample_at(P + [0, 0, e]) - field.sample_at(P - [0, 0, e])], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    key = np.array([-0.45, -0.7, 0.55]); key /= np.linalg.norm(key)
    fill = np.array([0.6, -0.4, 0.1]); fill /= np.linalg.norm(fill)
    rim = np.array([0.2, 0.9, 0.3]); rim /= np.linalg.norm(rim)
    lum = 0.2 + 0.62 * np.clip(n @ key, 0, 1) + 0.18 * np.clip(n @ fill, 0, 1) + 0.12 * np.clip(n @ rim, 0, 1)
    img = np.where(hit, np.clip(lum, 0, 1) * 225 + 12, 58)
    return img.reshape(size, size).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("character")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.0012)
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--face", action="store_true", help="close-up on the face (eyes to chin)")
    a = ap.parse_args()
    h = head_for(a.character)
    f = h.field(voxel=a.voxel)
    center = np.array([0.0, h.E[1] + 0.012, 0.5 * (h.chin_z + h.crown_z)])
    half = 0.66 * (h.crown_z - h.chin_z)
    if a.face:
        center = np.array([0.0, h.E[1] - 0.04 * h.s, h.E[2] - 0.03 * h.s])
        half = 0.36 * (h.crown_z - h.chin_z)
    tiles = [render(f, center, yaw, pitch, a.size, half) for _, yaw, pitch in VIEWS]
    sheet = Image.new("L", (len(tiles) * (a.size + 8) + 8, a.size + 30), 40)
    for i, (tile, (label, _, _)) in enumerate(zip(tiles, VIEWS)):
        x = 8 + i * (a.size + 8)
        sheet.paste(Image.fromarray(tile), (x, 24))
        ImageDraw.Draw(sheet).text((x + 4, 6), label, fill=230)
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
