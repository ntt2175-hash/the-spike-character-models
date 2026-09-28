"""Nude proportion study (Test B, volume): the complete, uncut body + head + hands from the proportion
master, skinned to the rig, posed, rendered in gray clay (and optionally as silhouettes).

    blender -b --python characters/sara/build/study_body.py -- <out_dir> [pose] [--silhouette]

Garments never define anatomy: this study must read as a professional sculpt on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from spike_pipeline import rig as rigmod  # noqa: E402
from spike_pipeline.modeling import weights as W  # noqa: E402

import body  # noqa: E402
import build_sara as BS  # noqa: E402
import dev_render  # noqa: E402
import head  # noqa: E402
import poses  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = Path(argv[0])
pose = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "stand"
bpy.ops.wm.read_factory_settings(use_empty=True)
dev_render.setup(samples=16)
bones, character = common.bones_dict()
rig = rigmod.build_rig("sara")
rig.hide_render = True
primary = [n for n in bones if bones[n]["kind"] in ("primary", "twist") and bones[n]["deform"]]
body_bones = [n for n in primary if n not in ("Jaw",) and not any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
body_obj = BS.field_object("study_body", body.body_field(bones), decimate=0.35, smooth=(0.5, 4), qf=15000)
hands = {s: BS.field_object(f"study_hand_{s[0]}", body.hand_field(bones, s), decimate=0.4, smooth=(0.5, 2), qf=2800)
         for s in ("Left", "Right")}
face_obj, prof, grid = head.build_head_mesh(BS.LOD0)
ears = head.build_ears(BS.LOD0)
BS.skin_body(body_obj, bones)
for side, h in hands.items():
    fingers = [n for n in bones if n.startswith(side) and any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
    BS.skin(h, bones, [f"{side}Hand", f"{side}LowerArmTwist", f"{side}LowerArm"] + fingers, power=6.0, relax=1)
for obj in [face_obj] + list(ears):
    W.apply_to_object(obj, ["Head"], np.ones((len(obj.data.vertices), 1)))
for obj in [body_obj, face_obj] + list(hands.values()) + list(ears):
    common.link(obj, BS.LOD0)
    BS.bind(obj, rig)
getattr(poses, pose)(rig)
if "--silhouette" in argv:
    bpy.context.scene.render.film_transparent = True
    dev_render.render_shots(rig, character, out, [t for t in dev_render.clay_tiles("nude study")[:5]])
else:
    dev_render.clay()
    dev_render.render_shots(rig, character, out, dev_render.clay_tiles("nude study"))
