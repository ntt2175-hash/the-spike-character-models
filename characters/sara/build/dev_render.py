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
