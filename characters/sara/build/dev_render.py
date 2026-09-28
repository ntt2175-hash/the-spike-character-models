"""Quick look-dev renders during the build (EEVEE toon, shared shot presets)."""
from __future__ import annotations

import json
from pathlib import Path

import bpy

import spike_specs as ss
from spike_pipeline import camera, lighting, toon

from common import CHAR, log


def setup(scene=None, samples=8):
    scene = scene or bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    e = scene.eevee
    for k, v in dict(taa_render_samples=samples, use_shadows=True, shadow_ray_count=2, shadow_step_count=6,
                     use_raytracing=False, shadow_resolution_scale=1.0).items():
        if hasattr(e, k):
            setattr(e, k, v)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    toon.ensure_scene_props(scene)
    return scene


def render_shots(rig, character, out_dir, tiles, prefix=""):
    """tiles: [(shot_id, profile_id, (w, h), label)]"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    specs = ss.load_all_specs()
    shots = ss.merged_shots(specs["shots"], character)
    cam = bpy.data.objects.get("dev_cam") or camera.new_camera("dev_cam")
    scene.camera = cam
    manifest = []
    for i, (shot_id, prof, res, label) in enumerate(tiles):
        scene.render.resolution_x, scene.render.resolution_y = res
        shot = shots[shot_id] if isinstance(shot_id, str) else shot_id
        ev = camera.evaluate_shot(rig, character, shot, 0.0, aspect=res[0] / res[1])
        camera.apply_to_camera(cam, ev, use_dof=False)
        bpy.context.view_layer.update()
        lighting.apply_profile(specs["lighting"]["profiles"][prof], character["palette"], cam)
        name = f"{prefix}{i:02d}.png"
        scene.render.filepath = str(out / name)
        bpy.ops.render.render(write_still=True)
        manifest.append({"file": name, "label": label or (shot_id if isinstance(shot_id, str) else "custom")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("rendered", len(manifest), "to", out)
    return out


# ---------------------------------------------------------------------------
# Test B (volume): gray clay. The model must read as a professional sculpt before hair, textures and
# toon lighting. Views: front, side, 3/4 (primary), rear 3/4, rear, low angle, torso close-up.
# ---------------------------------------------------------------------------
def _free(yaw, pitch, fl, fh, anchor="body_center"):
    return {"anchor": anchor, "anchor_offset": [0, 0, 0], "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": 0,
            "focal_length_mm": fl, "frame_height_rel": fh, "screen_offset": [0, 0], "move": {"type": "static"}}


def clay_tiles(label="clay"):
    return [("qc_front", "lookdev_neutral", (600, 1000), f"{label} front"),
            ("qc_side", "lookdev_neutral", (600, 1000), f"{label} side"),
            (_free(35, 5, 50, 1.15), "lookdev_neutral", (600, 1000), f"{label} 3/4 (primary test)"),
            (_free(145, 4, 50, 1.15), "lookdev_neutral", (600, 1000), f"{label} rear 3/4"),
            ("qc_rear", "lookdev_neutral", (600, 1000), f"{label} rear"),
            (_free(30, -18, 35, 1.2), "lookdev_neutral", (600, 1000), f"{label} low angle"),
            (_free(35, 2, 60, 0.42, "Chest"), "lookdev_neutral", (800, 900), f"{label} torso close-up 3/4")]


def clay(hide=()):
    """Neutral clay on every mesh; outlines and painted eye decals off; objects matching `hide` hidden."""
    mat = bpy.data.materials.new("M_qc_clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.62, 0.6, 0.58, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        n = o.name.lower()
        if "_eye_" in n or "_lash_" in n or any(h in n for h in hide):
            o.hide_render = True
            continue
        for m in o.modifiers:
            if m.type == "SOLIDIFY" and m.name == "spike_outline":
                m.show_render = False
        o.data.materials.clear()
        o.data.materials.append(mat)
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("qc_world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.32, 0.33, 0.36, 1.0)
    bg.inputs["Strength"].default_value = 0.6
    return mat
