"""Dev check: head loft + painted face + layered eyes, close-up renders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bpy  # noqa: E402

from spike_pipeline import rig as rigmod, toon  # noqa: E402

import body  # noqa: E402
import dev_render  # noqa: E402
import head  # noqa: E402

out = Path(sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "/tmp/sara_head")
tex = out / "tex"
tex.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
dev_render.setup(samples=12)
bones, character = common.bones_dict()
pal = dict(character["palette"])
eye_pal = dict(character["face"]["eye_shader"])
for k in ("iris_top", "iris_mid", "iris_bottom", "iris_ring", "iris_cog"):
    eye_pal[k] = pal[k]
eye_pal["pupil"] = pal["pupil"]
rig = rigmod.build_rig("sara")
rig.hide_render = True

skip_body = "--no-body" in sys.argv
if not skip_body:
    f = body.body_field(bones, include=("torso", "arms"))
    b = common.sdf_to_object("sara_body_LOD0", f, "sara_GAME_LOD0")
    common.decimate(b, 0.35)
    common.smooth(b, 0.5, 4)
    common.quadriflow(b, 12000)
    common.shade_smooth(b)

face_obj, prof, grid = head.build_head_mesh()
decals = head.build_eye_decals(face_obj, prof)
ears = head.build_ears()

head.paint_face(tex / "T_sara_face_base.png")
head.paint_eye_white(tex / "T_sara_eye_white.png")
head.paint_iris(tex / "T_sara_iris_L.png", palette=eye_pal)
head.paint_iris(tex / "T_sara_iris_R.png", palette=eye_pal, mirror_highlights=True)
head.paint_lash(tex / "T_sara_lash.png")

skin = toon.toon_material("M_sara_skin", "@skin_base", "@skin_shadow", pal, threshold=0.3, softness=0.05)
face_mat = toon.toon_textured("M_sara_face", tex / "T_sara_face_base.png", shade_mul=(0.9, 0.74, 0.8), threshold=0.26,
                              softness=0.06, mask_attribute="front", fallback="#fae3e5")
outline = toon.outline_material("M_sara_outline", "#3a2530")
face_obj.data.materials.append(face_mat)
toon.add_outline(face_obj, outline, 0.0018)
for e in ears:
    e.data.materials.append(skin)
    toon.add_outline(e, outline, 0.0012)
if not skip_body:
    b.data.materials.append(skin)
    toon.add_outline(b, outline, 0.0022)
for (layer, side), obj in decals.items():
    if layer == "eye":
        m = toon.anime_eye_material(f"M_sara_eye_{side}", tex / "T_sara_eye_white.png", tex / f"T_sara_iris_{side}.png")
    else:
        m = toon.anime_lash_material(f"M_sara_lash_{side}", tex / "T_sara_lash.png")
    obj.data.materials.append(m)
    if "--heavy" in sys.argv:
        m.node_tree.nodes["lid_heavy"].outputs[0].default_value = 0.35

front_cu = {"anchor": "eyes", "anchor_offset": [0.0, -0.02, 0.0], "yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0,
            "focal_length_mm": 85, "frame_height_m": 0.3, "screen_offset": [0, 0], "move": {"type": "static"}}
side_cu = dict(front_cu, yaw_deg=90)
low34 = dict(front_cu, yaw_deg=35, pitch_deg=-12, frame_height_m=0.34)
tiles = [(front_cu, "cinematic_soft", (800, 800), "front"),
         ("cu_face", "cinematic_soft", (900, 800), "3/4 close-up"),
         (side_cu, "cinematic_soft", (800, 800), "side"),
         ("ecu_eyes", "sp_hero", (1200, 520), "ECU eyes (S+)"),
         (low34, "cinematic_dramatic", (800, 800), "low 3/4")]
dev_render.render_shots(rig, character, out, tiles)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "head_test.blend"))
