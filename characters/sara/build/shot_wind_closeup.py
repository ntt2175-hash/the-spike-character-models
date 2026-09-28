"""Recreate the windy cut-in close-up with the 3D model: a forward-lean ready stance, the face looking up
into a high camera, and the hair / ribbon SIMULATED in gusty wind (spike_pipeline.hair_sim, the same
chain data the runtime spring bones use). Renders a still and a short clip of the hair flowing.

    blender -b --python characters/sara/build/shot_wind_closeup.py -- <sara.blend> <out_dir> [frames]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bpy  # noqa: E402

import spike_specs as ss  # noqa: E402
from spike_pipeline import camera, hair_sim, lighting  # noqa: E402

import dev_render  # noqa: E402
import poses  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, out = Path(args[0]), Path(args[1])
n_frames = int(args[2]) if len(args) > 2 else 36
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(blend))
scene = dev_render.setup(samples=16)
_, character = common.bones_dict()
rig = bpy.data.objects["sara_rig"]
poses.lean_closeup(rig)
for o in bpy.data.objects:                      # a fuller, wider-open gaze for the shot
    if o.type == "MESH" and o.name.startswith("sara_eye_"):
        for m in o.data.materials:
            if m and m.node_tree and "lid_heavy" in m.node_tree.nodes:
                m.node_tree.nodes["lid_heavy"].outputs[0].default_value = 0.0

# Camera: high and close, looking down into the upturned face; the face right of center, hair flowing
# into the left of the frame (reference composition, 2.18:1).
shot = {"anchor": "eyes", "anchor_offset": [0.0, 0.0, -0.01], "yaw_deg": -8, "pitch_deg": 24, "roll_deg": -4,
        "focal_length_mm": 55, "frame_height_m": 0.3, "screen_offset": [0.24, 0.04], "move": {"type": "static"}}
res = (1320, 606)
scene.render.resolution_x, scene.render.resolution_y = res
cam = bpy.data.objects.get("dev_cam") or camera.new_camera("dev_cam")
scene.camera = cam
ev = camera.evaluate_shot(rig, character, shot, 0.0, aspect=res[0] / res[1])
camera.apply_to_camera(cam, ev, use_dof=False)
gym = {"key": {"azimuth_deg": -20, "elevation_deg": 34, "color": "#fff3e6", "energy": 1.05, "shadows": True,
               "shadow_softness": 0.18, "kicker": True},
       "fill": {"color": "#d8a078", "energy": 0.34},
       "rim": {"azimuth_deg": 150, "color": "#ffc27a", "strength": 0.95, "width": 0.3},
       "environment": {"background": "#7a4118", "env_light_scale": 0.3, "exposure": 1.0, "glow": 0.45, "saturation": 1.05},
       "face_shadow_soft": 0.03, "outline_scale": 1.0}
lighting.apply_profile(gym, character["palette"], cam)
# Backdrop: a warm, out-of-focus gym - a soft glow behind her head falling off to deep amber at the edges
# (camera-facing background only; the ambient light stays the profile's).
nt = scene.world.node_tree
bg = nt.nodes["spike_bg"]
tc = nt.nodes.new("ShaderNodeTexCoord")
grad = nt.nodes.new("ShaderNodeTexGradient")
grad.gradient_type = "SPHERICAL"
mapn = nt.nodes.new("ShaderNodeMapping")
mapn.inputs["Location"].default_value = (-0.15, 0.1, 0.0)
mapn.inputs["Scale"].default_value = (0.9, 1.7, 1.0)
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.16, 0.07, 0.02, 1.0)
ramp.color_ramp.elements[1].color = (0.78, 0.44, 0.16, 1.0)
ramp.color_ramp.elements[1].position = 0.85
nt.links.new(tc.outputs["Window"], mapn.inputs["Vector"])
nt.links.new(mapn.outputs["Vector"], grad.inputs["Vector"])
nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])

# Hair physics: settle into the wind, then keep simulating for the clip.
# The fringe keeps its sculpted shape (short, stiff, sheltered by the head); everything else flies.
sim = hair_sim.ChainSim(rig, character, prefixes=("hair_side", "hair_ponytail", "hair_flyaway", "hair_ahoge", "ribbon_"))
sim.wind = hair_sim.Wind(direction=(-1.0, 0.25, 0.3), speed=5.5, gust=0.45, turbulence=0.75, seed=7)
sim.run(2.5)
sim.apply()
scene.render.filepath = str(out / "still.png")
bpy.ops.render.render(write_still=True)
common.log("still rendered")
frames = out / "frames"
frames.mkdir(exist_ok=True)
scene.eevee.taa_render_samples = 8
scene.render.resolution_x, scene.render.resolution_y = 990, 454
for f in range(n_frames):
    sim.run(1.0 / 24.0)
    sim.apply()
    scene.render.filepath = str(frames / f"{f:04d}.png")
    bpy.ops.render.render(write_still=True)
common.log(f"rendered {n_frames} frames to {frames}")
