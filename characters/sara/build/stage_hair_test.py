"""Dev check: head + face + hair + ribbon."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bpy  # noqa: E402

from spike_pipeline import rig as rigmod, toon  # noqa: E402

import assemble  # noqa: E402
import body  # noqa: E402
import dev_render  # noqa: E402
import head  # noqa: E402
import materials_sara  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = Path(args[0])
tex_src = Path(args[1]) if len(args) > 1 else None
tex = out / "tex"
tex.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
dev_render.setup(samples=12)
bones, character = common.bones_dict()
pal = dict(character["palette"])
rig = rigmod.build_rig("sara")
rig.hide_render = True

f = body.body_field(bones, include=("torso", "arms"))
b = common.sdf_to_object("sara_body_LOD0", f, "sara_GAME_LOD0")
common.decimate(b, 0.35)
common.smooth(b, 0.5, 4)
common.quadriflow(b, 12000)
common.shade_smooth(b)

face_obj, prof, grid = head.build_head_mesh()
decals = head.build_eye_decals(face_obj, prof)
ears = head.build_ears()
names = ["T_sara_face_base.png", "T_sara_eye_white.png", "T_sara_iris_L.png", "T_sara_iris_R.png", "T_sara_lash.png"]
if tex_src and all((tex_src / n).exists() for n in names[1:]) and "--repaint" not in args:
    for n in names[1:]:
        shutil.copy(tex_src / n, tex / n)
    head.paint_face(tex / names[0])  # face layout depends on the head proportions: always repaint
else:
    eye_pal = dict(character["face"]["eye_shader"])
    for k in ("iris_top", "iris_mid", "iris_bottom", "iris_ring", "iris_cog", "pupil"):
        eye_pal[k] = pal[k]
    head.paint_face(tex / names[0])
    head.paint_eye_white(tex / names[1])
    head.paint_iris(tex / names[2], palette=eye_pal)
    head.paint_iris(tex / names[3], palette=eye_pal, mirror_highlights=True)
    head.paint_lash(tex / names[4])

hair_obj, ribbon_obj, hair_arrays, ribbon_arrays, info = assemble.build_hair(bones)

skin = toon.toon_material("M_sara_skin", "@skin_base", "@skin_shadow", pal, threshold=0.3, softness=0.05)
face_mat = toon.toon_textured("M_sara_face", tex / names[0], shade_mul=(0.9, 0.74, 0.8), threshold=0.17, softness=0.07,
                              mask_attribute="front", fallback="#fae3e5")
outline = toon.outline_material("M_sara_outline", "#3a2530")
hair_outline = toon.outline_material("M_sara_hair_outline", "#15111a")
face_obj.data.materials.append(face_mat)
toon.add_outline(face_obj, outline, 0.0018)
for e in ears:
    e.data.materials.append(skin)
    toon.add_outline(e, outline, 0.0012)
b.data.materials.append(skin)
toon.add_outline(b, outline, 0.0022)
for (layer, side), obj in decals.items():
    m = (toon.anime_eye_material(f"M_sara_eye_{side}", tex / names[1], tex / f"T_sara_iris_{side}.png") if layer == "eye"
         else toon.anime_lash_material(f"M_sara_lash_{side}", tex / names[4]))
    m.node_tree.nodes["lid_heavy"].outputs[0].default_value = 0.2
    obj.data.materials.append(m)
hair_obj.data.materials.append(materials_sara.hair_material(pal))
toon.add_outline(hair_obj, hair_outline, 0.0011)
info["cap_obj"].data.materials.append(materials_sara.hair_material(pal))
ribbon_obj.data.materials.append(materials_sara.satin_material(pal))
toon.add_outline(ribbon_obj, hair_outline, 0.0012)

front = {"anchor": "eyes", "anchor_offset": [0.0, 0.01, 0.0], "yaw_deg": 0, "pitch_deg": 2, "roll_deg": 0,
         "focal_length_mm": 70, "frame_height_m": 0.42, "screen_offset": [0, 0], "move": {"type": "static"}}
tiles = [(front, "cinematic_soft", (800, 900), "front"),
         (dict(front, yaw_deg=35), "cinematic_soft", (800, 900), "3/4"),
         (dict(front, yaw_deg=90), "cinematic_soft", (800, 900), "side (her left)"),
         (dict(front, yaw_deg=-90), "cinematic_soft", (800, 900), "side (her right)"),
         (dict(front, yaw_deg=160, frame_height_m=0.6, anchor_offset=[0, -0.08, 0]), "cinematic_soft", (800, 900), "back 3/4"),
         ("ecu_eyes", "sp_hero", (1200, 520), "ECU eyes (S+)"),
         (dict(front, yaw_deg=25, pitch_deg=-15, frame_height_m=0.5), "sp_hero", (800, 900), "low 3/4 (S+)")]
dev_render.render_shots(rig, character, out, tiles)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "hair_test.blend"))
