"""Render QC / presentation tiles from a built Sara .blend.

    blender -b --python characters/sara/build/render_sara.py -- <sara_v02.blend> <out_dir> [pose]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bpy  # noqa: E402

import dev_render  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, out = Path(args[0]), Path(args[1])
tiles_kind = args[2] if len(args) > 2 else "turnaround"
bpy.ops.wm.open_mainfile(filepath=str(blend))
dev_render.setup(samples=12)
_, character = common.bones_dict()
rig = bpy.data.objects["sara_rig"]
import poses  # noqa: E402
pose = args[3] if len(args) > 3 else "stand"
getattr(poses, pose)(rig)
face = {"anchor": "eyes", "anchor_offset": [0.0, 0.0, 0.0], "yaw_deg": 0, "pitch_deg": 2, "roll_deg": 0,
        "focal_length_mm": 70, "frame_height_m": 0.4, "screen_offset": [0, 0], "move": {"type": "static"}}
if tiles_kind == "turnaround":
    tiles = [("qc_front", "lookdev_neutral", (700, 1000), "front"), ("qc_side", "lookdev_neutral", (700, 1000), "side"),
             ("qc_rear", "lookdev_neutral", (700, 1000), "rear"), ("qc_34", "lookdev_neutral", (700, 1000), "3/4"),
             ("cu_face", "cinematic_soft", (900, 800), "face 3/4"), (face, "cinematic_soft", (800, 800), "face front"),
             ("hair_detail", "cinematic_soft", (900, 800), "hair back 3/4"), ("ms_waist", "cinematic_soft", (900, 900), "waist-up"),
             ("low_hero", "sp_hero", (900, 1000), "S+ low hero"), ("qc_gameplay", "gameplay_readable", (900, 600), "gameplay distance")]
elif tiles_kind == "clay":
    # Test B (volume) on the full costume; the nude proportion study is study_body.py.
    dev_render.clay()
    tiles = dev_render.clay_tiles("clay")
elif tiles_kind == "silhouette":
    # Transparent film: the alpha channel IS the silhouette (pipeline/tools/silhouette_compare.py).
    bpy.context.scene.render.film_transparent = True
    tiles = [("qc_front", "lookdev_neutral", (700, 1000), "front"), ("qc_side", "lookdev_neutral", (700, 1000), "side"),
             ("qc_rear", "lookdev_neutral", (700, 1000), "rear"), ("qc_34", "lookdev_neutral", (700, 1000), "3/4")]
    dev_render.render_shots(rig, character, out / "stand", tiles)
    poses.keyart(rig)
    tiles = [("keyart_match", "lookdev_neutral", (594, 1346), "key-art match")]
    out = out / "keyart"
elif tiles_kind == "keyart":
    tiles = [("keyart_match", "lookdev_neutral", (594, 1346), "key-art match"),
             ("keyart_match", "sp_hero", (594, 1346), "key-art match (S+ light)"),
             ("low_hero", "sp_hero", (900, 1100), "S+ low hero"), ("cu_face_low", "sp_hero", (900, 900), "S+ face")]
else:
    tiles = json.loads(tiles_kind)
dev_render.render_shots(rig, character, out, tiles)
