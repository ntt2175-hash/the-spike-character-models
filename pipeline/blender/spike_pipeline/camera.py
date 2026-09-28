"""Shot presets (pipeline/specs/shots.json) -> Blender cameras.

The math matches addons/spike_character/camera/shot_math.gd so a shot framed
in a Blender QC render is the same shot in Godot.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Quaternion, Vector

import spike_specs as ss

from . import rig as rigmod

EASE = {
    "linear": lambda t: t,
    "sine_in_out": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
    "quad_out": lambda t: 1.0 - (1.0 - t) ** 2,
    "cubic_out": lambda t: 1.0 - (1.0 - t) ** 3,
    "expo_out": lambda t: 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t),
    "quad_in": lambda t: t * t,
    "back_out": lambda t: 1.0 + 2.70158 * (t - 1.0) ** 3 + 1.70158 * (t - 1.0) ** 2,
}


def ease(name: str, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return EASE.get(name or "linear", EASE["linear"])(t)


def character_basis(rig):
    """World-space character axes: left (+X char), up, forward (+Z char)."""
    rot = rig.matrix_world.to_3x3()
    fwd = rot @ Vector((0.0, -1.0, 0.0))
    fwd.z = 0.0
    fwd.normalize()
    up = Vector((0.0, 0.0, 1.0))
    left = up.cross(fwd).normalized()  # facing -Y, the character's left is +X
    return left, up, fwd


def char_to_world(rig, v):
    left, up, fwd = character_basis(rig)
    return left * v[0] + up * v[1] + fwd * v[2]


def _noise(t: float, seed: float) -> float:
    return (math.sin(t * 7.3 + seed) + 0.6 * math.sin(t * 13.1 + seed * 2.1) + 0.3 * math.sin(t * 23.7 + seed * 3.7)) / 1.9


def evaluate_shot(rig, character: dict, shot: dict, progress: float, anchor_world: Vector | None = None,
                  time_s: float = 0.0, aspect: float = 16 / 9):
    """Camera transform + lens for a shot at move progress 0..1.

    Returns dict(location, rotation (Quaternion), lens_mm, ortho_scale|None, shift, focus_distance, target).
    """
    H = character["proportions"]["height_m"]
    shot = dict(shot)
    move = shot.get("move", {"type": "static"})
    e = ease(move.get("ease", "linear"), progress)
    amount = move.get("amount", 0.0)
    mtype = move.get("type", "static")
    if mtype == "orbit":
        shot["yaw_deg"] = shot.get("yaw_deg", 0.0) + amount * e
    anchor = anchor_world if anchor_world is not None else rigmod.resolve_anchor(rig, shot["anchor"], character)
    target = anchor + char_to_world(rig, shot.get("anchor_offset", [0, 0, 0]))
    offset = ss.shot_camera_offset(shot, H)
    scale = 1.0
    if mtype == "push_in":
        scale = 1.0 - amount * e
    elif mtype == "pull_out":
        scale = 1.0 + amount * e
    cam_off = char_to_world(rig, offset) * scale
    rise = Vector((0.0, 0.0, amount * e)) if mtype == "rise" else Vector()
    location = target + cam_off + rise
    target = target + rise
    view = (target - location).normalized()
    right = view.cross(Vector((0.0, 0.0, 1.0))).normalized()
    if mtype == "track":
        location += right * (amount * e)
        target += right * (amount * e)
    if mtype == "drift":
        location += right * (amount * math.sin(math.pi * e)) + Vector((0, 0, amount * 0.4 * math.sin(2 * math.pi * e)))
    n = move.get("noise", 0.0)
    if n:
        up_cam = right.cross(view)
        location += right * (n * _noise(time_s, 1.3)) + up_cam * (n * _noise(time_s, 4.1))
    view = (target - location).normalized()
    rot = view.to_track_quat("-Z", "Y")
    rot = rot @ Quaternion((0.0, 0.0, 1.0), math.radians(shot.get("roll_deg", 0.0)))
    so = shot.get("screen_offset", [0.0, 0.0])
    big = max(aspect, 1.0)
    shift = (-so[0] * 0.5 * (aspect / big), -so[1] * 0.5 * (1.0 / big))
    ortho = None
    if shot.get("projection") == "orthographic":
        fh = ss.shot_frame_height(shot, H)
        ortho = fh * (aspect if aspect > 1.0 else 1.0)
        location = target - view * 8.0
    return {"location": location, "rotation": rot, "lens_mm": shot.get("focal_length_mm", 50.0), "ortho_scale": ortho,
            "shift": shift, "focus_distance": (target - location).length, "target": target,
            "dof": shot.get("dof", {"enabled": False})}


def apply_to_camera(cam_obj, ev: dict, frame: int | None = None, use_dof: bool = True):
    cam = cam_obj.data
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.location = ev["location"]
    cam_obj.rotation_quaternion = ev["rotation"]
    cam.sensor_fit = "VERTICAL"
    cam.sensor_height = ss.SENSOR_HEIGHT_MM
    cam.clip_start = 0.02
    cam.clip_end = 200.0
    if ev["ortho_scale"]:
        cam.type = "ORTHO"
        cam.ortho_scale = ev["ortho_scale"]
    else:
        cam.type = "PERSP"
        cam.lens = ev["lens_mm"]
    cam.shift_x, cam.shift_y = ev["shift"]
    dof = ev.get("dof") or {}
    cam.dof.use_dof = bool(use_dof and dof.get("enabled"))
    if cam.dof.use_dof:
        cam.dof.focus_distance = ev["focus_distance"]
        cam.dof.aperture_fstop = max(0.9, min(16.0, 0.22 / max(dof.get("amount", 0.05), 1e-3)))
    if frame is not None:
        cam_obj.keyframe_insert("location", frame=frame)
        cam_obj.keyframe_insert("rotation_quaternion", frame=frame)
        cam.keyframe_insert("lens", frame=frame)
        cam.keyframe_insert("shift_x", frame=frame)
        cam.keyframe_insert("shift_y", frame=frame)
        if cam.dof.use_dof:
            cam.dof.keyframe_insert("focus_distance", frame=frame)
            cam.dof.keyframe_insert("aperture_fstop", frame=frame)
        cam.dof.keyframe_insert("use_dof", frame=frame)


def new_camera(name: str) -> bpy.types.Object:
    cam = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(obj)
    return obj
