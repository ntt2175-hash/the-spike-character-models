#!/usr/bin/env python3
"""Instant body-proportion preview without Blender: projects a character's body-master SDF to front
and side silhouettes (plus a depth-shaded front view) so contour changes can be judged in seconds.

    python pipeline/tools/sdf_preview.py <character_id> <out.png> [--voxel 0.004]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender")]
import spike_specs as ss  # noqa: E402
from spike_pipeline.modeling import body_master as BM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("character")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.004)
    a = ap.parse_args()
    c = ss.load_character(a.character)
    bones = {b["name"]: b for b in ss.build_character_skeleton(ss.load_spec("skeleton"), c)}
    body = BM.Body(bones, c.get("body_shape", {}), c["proportions"]["height_m"])
    f = body.body_field(voxel=a.voxel)
    solid = f.d < 0                                   # (x, y, z)
    front = solid.any(axis=1).T[::-1]                  # rows = z (top first), cols = x
    side = solid.any(axis=0).T[::-1]                   # rows = z, cols = y (front on the left)
    # depth-shaded front: first solid voxel along +y
    idx = np.where(solid.any(axis=1), solid.argmax(axis=1), -1).T[::-1]
    shade = np.where(idx >= 0, 60 + 170 * (1 - idx / max(idx.max(), 1)), 238).astype(np.uint8)
    h = front.shape[0]
    panels = [Image.fromarray(np.where(front, 20, 238).astype(np.uint8)), Image.fromarray(shade),
              Image.fromarray(np.where(side, 20, 238).astype(np.uint8))]
    W = sum(p.width for p in panels) + 40
    sheet = Image.new("L", (W, h + 30), 250)
    x = 10
    for p, label in zip(panels, ("front", "front (depth)", "side")):
        sheet.paste(p, (x, 25))
        ImageDraw.Draw(sheet).text((x, 5), label, fill=0)
        x += p.width + 10
    scale = 3 if a.voxel >= 0.004 else 2
    sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST)
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
