"""Lighting profiles (pipeline/specs/lighting.json) -> Blender.

Directions are camera-relative (azimuth 0 = from behind the camera, +90 =
from screen right, 180 = backlight), exactly as in Godot's LightingDirector.
The toon preview materials read the rest of the profile from scene custom
properties (see toon.py), so a profile change is a handful of keyframes.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

import spike_specs as ss

from . import toon

KEY_NAME = "spike_key"
SUN_STRENGTH_PER_ENERGY = 2.2


def _lin(hex_color: str, palette: dict):
    return tuple(ss.srgb_to_linear(c) for c in ss.hex_to_rgb(ss.resolve_color(hex_color, palette)))


def camera_relative_direction(cam_obj, azimuth_deg: float, elevation_deg: float) -> Vector:
    """Unit vector pointing FROM the subject TOWARD the light."""
    m = cam_obj.matrix_world.to_3x3()
    fwd = m @ Vector((0.0, 0.0, -1.0))
    fwd.z = 0.0
    if fwd.length < 1e-6:
        fwd = m @ Vector((0.0, 1.0, 0.0))
        fwd.z = 0.0
    fwd.normalize()
    right = fwd.cross(Vector((0.0, 0.0, 1.0))).normalized()
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    horizontal = -fwd * math.cos(az) + right * math.sin(az)
    return (horizontal * math.cos(el) + Vector((0.0, 0.0, 1.0)) * math.sin(el)).normalized()


def ensure_key_light() -> bpy.types.Object:
    obj = bpy.data.objects.get(KEY_NAME)
    if obj is None:
        light = bpy.data.lights.new(KEY_NAME, "SUN")
        obj = bpy.data.objects.new(KEY_NAME, light)
        bpy.context.scene.collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    return obj


def ensure_world(scene=None):
    scene = scene or bpy.context.scene
    world = scene.world or bpy.data.worlds.new("spike_world")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    if "spike_bg" not in nt.nodes:
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputWorld")
        path = nt.nodes.new("ShaderNodeLightPath")
        cam_bg = nt.nodes.new("ShaderNodeBackground"); cam_bg.name = "spike_bg"
        light_bg = nt.nodes.new("ShaderNodeBackground"); light_bg.name = "spike_ambient"
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(light_bg.outputs[0], mix.inputs[1])
        nt.links.new(cam_bg.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return world


def apply_profile(profile: dict, palette: dict, cam_obj, frame: int | None = None, rim_bias_deg: float = 0.0,
                  scene=None):
    scene = scene or bpy.context.scene
    toon.ensure_scene_props(scene)
    key = ensure_key_light()
    k = profile["key"]
    direction = camera_relative_direction(cam_obj, k["azimuth_deg"], k["elevation_deg"])
    key.rotation_quaternion = (-direction).to_track_quat("-Z", "Y")
    key.data.color = _lin(k["color"], palette)
    key.data.energy = SUN_STRENGTH_PER_ENERGY * k["energy"]
    key.data.use_shadow = bool(k.get("shadows", True))
    key.data.angle = math.radians(1.0 + 6.0 * k.get("shadow_softness", 0.3))

    world = ensure_world(scene)
    bg = world.node_tree.nodes["spike_bg"]
    amb = world.node_tree.nodes["spike_ambient"]
    env = profile["environment"]
    bg.inputs["Color"].default_value = (*_lin(env["background"], palette), 1.0)
    bg.inputs["Strength"].default_value = 1.0
    amb.inputs["Color"].default_value = (*_lin(profile["fill"]["color"], palette), 1.0)
    amb.inputs["Strength"].default_value = 0.05

    scene["spike_char_light"] = max(0.03, min(1.25, k["energy"]))
    scene["spike_fill_color"] = _lin(profile["fill"]["color"], palette)
    scene["spike_fill_mix"] = max(0.0, min(1.0, profile["fill"]["energy"] * 0.8))
    rim = profile["rim"]
    scene["spike_rim_color"] = _lin(rim["color"], palette)
    scene["spike_rim_strength"] = rim["strength"]
    scene["spike_rim_width"] = rim["width"]
    scene["spike_env_light"] = env.get("env_light_scale", 1.0)

    if frame is not None:
        key.keyframe_insert("rotation_quaternion", frame=frame)
        key.data.keyframe_insert("color", frame=frame)
        key.data.keyframe_insert("energy", frame=frame)
        bg.inputs["Color"].keyframe_insert("default_value", frame=frame)
        amb.inputs["Color"].keyframe_insert("default_value", frame=frame)
        for prop in ("spike_char_light", "spike_fill_color", "spike_fill_mix", "spike_rim_color",
                     "spike_rim_strength", "spike_rim_width", "spike_env_light"):
            scene.keyframe_insert(f'["{prop}"]', frame=frame)
    return key


def set_constant_interpolation(id_data):
    ad = getattr(id_data, "animation_data", None)
    if not ad or not ad.action:
        return
    for fc in _fcurves(ad.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def _fcurves(action):
    """F-curves of an action on Blender 4.2 (flat) and 4.4+ (layered/slotted)."""
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out.extend(bag.fcurves)
    return out
