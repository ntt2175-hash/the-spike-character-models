"""Build the standard armature for a character.

The rig builder turns pipeline/specs/skeleton.json plus the character's
proportions and secondary chains (characters/<id>/character.json) into a
correctly named, parented, A-posed armature. Every character starts from
this, so the names, hierarchy, deform flags and bone collections are right
from the first day of blocking. Artists then refine joint placement against
the sculpt; the validator checks that the conventions survive.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

import spike_specs as ss

from . import common

BONE_COLLECTIONS = {
    "primary": "Body",
    "finger": "Fingers",
    "twist": "Twist (driven)",
    "socket": "Sockets",
    "secondary": "Secondary",
}


def build_rig(char: str, ctx: dict | None = None, replace: bool = True) -> bpy.types.Object:
    ctx = ctx or common.load_context(char)
    common.ensure_object_mode()
    layout = common.ensure_scene_layout(char)
    bones = ss.build_character_skeleton(ctx["specs"]["skeleton"], ctx["character"])

    name = common.rig_name(char)
    existing = bpy.data.objects.get(name)
    if existing is not None:
        if not replace:
            return existing
        arm_data = existing.data
        bpy.data.objects.remove(existing, do_unlink=True)
        if arm_data.users == 0:
            bpy.data.armatures.remove(arm_data)

    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    layout["game"].objects.link(obj)
    arm.display_type = "OCTAHEDRAL"
    obj.show_in_front = True

    common.set_active(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    colls = {kind: arm.collections.new(label) for kind, label in BONE_COLLECTIONS.items()}
    ebones = {}
    for b in bones:
        eb = arm.edit_bones.new(b["name"])
        eb.head = Vector(b["head"])
        eb.tail = Vector(b["tail"])
        eb.use_deform = bool(b["deform"])
        colls[b["kind"]].assign(eb)
        ebones[b["name"]] = eb
    for b in bones:
        if b["parent"]:
            eb = ebones[b["name"]]
            eb.parent = ebones[b["parent"]]
            # Connect only when the child starts exactly at the parent's tail.
            eb.use_connect = (eb.head - eb.parent.tail).length < 1e-5 and b["kind"] in ("primary", "finger", "secondary")
    _orient_rolls(ebones)
    bpy.ops.object.mode_set(mode="OBJECT")

    _add_twist_constraints(obj, ctx["specs"]["skeleton"])
    for kind in ("socket", "twist"):
        colls[kind].is_visible = False
    obj["spike_character"] = char
    obj["spike_spec_version"] = ctx["specs"]["skeleton"]["version"]
    return obj


def _orient_rolls(ebones: dict):
    """Consistent rolls: limbs bend on local X, spine/neck/head roll 0.

    Blender computes a sane default roll for vertical bones; limb bones get a
    roll so that their local Z points forward (-Y world), which keeps elbows
    and knees bending on the same local axis on both sides.
    """
    forward = Vector((0.0, -1.0, 0.0))
    for name, eb in ebones.items():
        axis = (eb.tail - eb.head).normalized()
        if abs(axis.dot(Vector((0.0, 0.0, 1.0)))) > 0.98:
            eb.roll = 0.0
            continue
        eb.align_roll(forward - axis * axis.dot(forward))


def _add_twist_constraints(obj, skeleton: dict):
    """Twist bones copy (or counter) part of the Y (twist) rotation of their driver bone.

    Positive influence copies, negative counter-rotates. The motion is baked into
    every exported clip, so Godot never evaluates these constraints.
    """
    for t in skeleton.get("twist_bones", []):
        influence = float(t["influence"])
        for twist, drv in ((t["name"], t["driver_bone"]), (ss.mirror_name(t["name"]), ss.mirror_name(t["driver_bone"]))):
            pb = obj.pose.bones.get(twist)
            if pb is None:
                continue
            con = pb.constraints.new("COPY_ROTATION")
            con.name = "spike_twist"
            con.target = obj
            con.subtarget = drv
            con.use_x = False
            con.use_z = False
            con.use_y = True
            con.invert_y = influence < 0
            con.mix_mode = "ADD"
            con.target_space = "LOCAL"
            con.owner_space = "LOCAL"
            con.influence = abs(influence)


def pose_bone_world_head(obj, bone_name: str) -> Vector:
    pb = obj.pose.bones[bone_name]
    return obj.matrix_world @ pb.head


def resolve_anchor(obj, anchor: str, character: dict) -> Vector:
    """World position for a shot/VFX anchor (bone name or alias)."""
    handed = character.get("identity", {}).get("handedness", "right")
    if anchor == "eyes":
        return (pose_bone_world_head(obj, "LeftEye") + pose_bone_world_head(obj, "RightEye")) * 0.5
    if anchor == "feet":
        p = (pose_bone_world_head(obj, "LeftToes") + pose_bone_world_head(obj, "RightToes")) * 0.5
        p.z = obj.matrix_world.translation.z
        return p
    if anchor == "body_center":
        # Half the character height above the floor, under the hips: full-body framing that does not
        # depend on the character's leg/torso proportions.
        p = pose_bone_world_head(obj, "Hips")
        p.z = obj.matrix_world.translation.z + 0.5 * character["proportions"]["height_m"]
        return p
    if anchor in ("hit_hand", "aim_hand"):
        hit = "socket_ball_R" if handed == "right" else "socket_ball_L"
        aim = "socket_ball_L" if handed == "right" else "socket_ball_R"
        return pose_bone_world_head(obj, hit if anchor == "hit_hand" else aim)
    return pose_bone_world_head(obj, anchor)


def arm_angle_below_horizontal(obj, side: str = "Left") -> float:
    bone = obj.data.bones[f"{side}UpperArm"]
    v = bone.tail_local - bone.head_local
    horizontal = Vector((v.x, v.y, 0.0))
    return math.degrees(v.angle(horizontal)) if horizontal.length > 1e-6 else 90.0
