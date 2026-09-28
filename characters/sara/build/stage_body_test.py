"""Dev check: sculpt the body field, clean it, render clay turnaround."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets sys.path)

import bpy  # noqa: E402

from spike_pipeline import rig as rigmod, toon  # noqa: E402

import body  # noqa: E402
import dev_render  # noqa: E402

out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "/tmp/sara_body"
bpy.ops.wm.read_factory_settings(use_empty=True)
dev_render.setup()
bones, character = common.bones_dict()
rig = rigmod.build_rig("sara")
rig.hide_render = True

f = body.body_field(bones)
obj = common.sdf_to_object("sara_body_hi", f, "sara_MASTER")
common.decimate(obj, 0.35)
common.smooth(obj, 0.5, 4)
common.quadriflow(obj, 16000)
common.shade_smooth(obj)
for side in ("Left", "Right"):
    hf = body.hand_field(bones, side)
    h = common.sdf_to_object(f"sara_hand_{side}", hf, "sara_MASTER")
    common.decimate(h, 0.4)
    common.smooth(h, 0.5, 2)
    common.quadriflow(h, 2600)
    common.shade_smooth(h)
clay = bpy.data.materials.new("clay")
clay.use_nodes = True
bsdf = clay.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.62, 0.6, 0.6, 1)
bsdf.inputs["Roughness"].default_value = 0.55
outline = toon.outline_material("clay_outline", "#3a3340")
for o in bpy.data.collections["sara_MASTER"].objects:
    o.data.materials.clear()
    o.data.materials.append(clay)
    toon.add_outline(o, outline, 0.0022)
tiles = [("qc_front", "lookdev_neutral", (600, 900), "front"), ("qc_side", "lookdev_neutral", (600, 900), "side"),
         ("qc_rear", "lookdev_neutral", (600, 900), "rear"), ("qc_34", "lookdev_neutral", (600, 900), "3/4"),
         ("ms_waist", "cinematic_soft", (900, 900), "waist-up"), ("cu_face", "cinematic_soft", (900, 700), "shoulders")]
dev_render.render_shots(rig, character, out, tiles)
bpy.ops.wm.save_as_mainfile(filepath=str(Path(out) / "body_test.blend"))
