"""Rig + stand-in turnaround demo.

Builds the character's standard rig from the specs, dresses it with the
identity-colored stand-in mannequin, and renders turnaround / close-up /
gameplay-distance / silhouette tiles using the shared shot presets and
lighting profiles.

    blender -b --python pipeline/blender/demos/turnaround_demo.py -- --character sara --out <dir>
(headless Linux needs a GL context for EEVEE: wrap with xvfb-run)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

import bpy  # noqa: E402

import spike_specs as ss  # noqa: E402
from spike_pipeline import camera, common, lighting, proxy, rig  # noqa: E402

TILES = [
    ("qc_front", "beauty", "lookdev_neutral", (720, 960)),
    ("qc_side", "beauty", "lookdev_neutral", (720, 960)),
    ("qc_rear", "beauty", "lookdev_neutral", (720, 960)),
    ("qc_34", "beauty", "lookdev_neutral", (720, 960)),
    ("qc_front", "silhouette", None, (360, 480)),
    ("qc_side", "silhouette", None, (360, 480)),
    ("qc_rear", "silhouette", None, (360, 480)),
    ("qc_34", "silhouette", None, (360, 480)),
    ("cu_face", "beauty", "cinematic_soft", (960, 720)),
    ("hair_detail", "beauty", "cinematic_soft", (960, 720)),
    ("low_hero", "beauty", "sp_hero", (960, 720)),
    ("qc_gameplay", "beauty", "gameplay_readable", (960, 720)),
]


def setup_render(scene):
    scene.render.engine = "BLENDER_EEVEE"
    e = scene.eevee
    for k, v in dict(taa_render_samples=8, use_shadows=True, shadow_ray_count=2, shadow_step_count=4,
                     use_raytracing=False).items():
        if hasattr(e, k):
            setattr(e, k, v)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.film_transparent = False


def silhouette_material():
    mat = bpy.data.materials.get("QC_silhouette") or bpy.data.materials.new("QC_silhouette")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0, 0, 0, 1)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return mat


def floor(char_light_prop=True):
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=1.4, fill_type="NGON", location=(0, 0, 0))
    f = bpy.context.active_object
    f.name = "lookdev_floor"
    mat = bpy.data.materials.new("lookdev_floor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.32, 0.33, 0.36, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    f.data.materials.append(mat)
    return f


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="sara")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    setup_render(scene)
    ctx = common.load_context(args.character)
    character = ctx["character"]
    shots = ss.merged_shots(ctx["specs"]["shots"], character)
    profiles = ctx["specs"]["lighting"]["profiles"]
    rig_obj = rig.build_rig(args.character, ctx)
    proxy_coll = proxy.build_proxy(rig_obj, character)
    fl = floor()
    rig_obj.hide_render = True
    cam = camera.new_camera("qc_cam")
    scene.camera = cam
    sil = silhouette_material()
    manifest = []
    for i, (shot_id, mode, profile_id, res) in enumerate(TILES):
        scene.render.resolution_x, scene.render.resolution_y = res
        aspect = res[0] / res[1]
        ev = camera.evaluate_shot(rig_obj, character, shots[shot_id], 0.0, aspect=aspect)
        camera.apply_to_camera(cam, ev, use_dof=True)
        bpy.context.view_layer.update()
        if mode == "silhouette":
            bpy.context.view_layer.material_override = sil
            fl.hide_render = True
            lighting.ensure_world(scene)
            scene.world.node_tree.nodes["spike_bg"].inputs["Color"].default_value = (1, 1, 1, 1)
        else:
            bpy.context.view_layer.material_override = None
            fl.hide_render = shot_id in ("low_hero",)
            lighting.apply_profile(profiles[profile_id], character["palette"], cam,
                                   rim_bias_deg=character.get("lighting", {}).get("rim_azimuth_bias_deg", 0.0))
        name = f"{i:02d}_{shot_id}_{mode}.png"
        scene.render.filepath = str(out / name)
        bpy.ops.render.render(write_still=True)
        manifest.append({"file": name, "shot": shot_id, "mode": mode, "lighting": profile_id,
                         "label": shots[shot_id].get("label", shot_id)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("TURNAROUND_DONE", out)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
