"""Presentation poses for Sara, written as limb directions (aim) plus twist.

aim(bone, direction, twist) points the bone's Y axis along a world direction with the smallest
rotation from its current orientation, then twists it around its own axis. This keeps pose data
readable ("left forearm reaches forward-up") and rig-convention independent.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Quaternion, Vector


def aim(rig, name, direction, twist_deg=0.0):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    m = pb.matrix.copy()
    head = m.translation.copy()
    r = m.to_3x3()
    cur = (r @ Vector((0, 1, 0))).normalized()
    d = Vector(direction).normalized()
    q = cur.rotation_difference(d)
    r2 = q.to_matrix() @ r
    if twist_deg:
        r2 = Matrix.Rotation(math.radians(twist_deg), 3, d) @ r2
    pb.matrix = Matrix.Translation(head) @ r2.to_4x4()
    bpy.context.view_layer.update()


def reset(rig):
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def curl_fingers(rig, side, deg, thumb_deg=None):
    for pb in rig.pose.bones:
        n = pb.name
        if n.startswith(side) and any(f in n for f in ("Index", "Middle", "Ring", "Little", "Thumb")):
            a = thumb_deg if ("Thumb" in n and thumb_deg is not None) else deg
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (math.radians(a), 0.0, 0.0)
    bpy.context.view_layer.update()


def stand(rig):
    """Relaxed standing pose: weight on the left leg, arms down, a small head tilt."""
    reset(rig)
    hips = rig.pose.bones["Hips"]
    hips.location = (0.0, -0.012, 0.0)
    aim(rig, "Spine", (0.0, 0.01, 1.0))
    aim(rig, "LeftUpperLeg", (0.02, -0.01, -1.0))
    aim(rig, "LeftLowerLeg", (0.0, 0.04, -1.0))
    aim(rig, "RightUpperLeg", (-0.07, -0.1, -1.0))
    aim(rig, "RightLowerLeg", (-0.02, 0.12, -1.0))
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        aim(rig, f"{side}UpperArm", (s * 0.2, 0.04, -1.0))
        aim(rig, f"{side}LowerArm", (s * 0.12, -0.2, -1.0))
        aim(rig, f"{side}Hand", (s * 0.03, -0.1, -1.0))
        curl_fingers(rig, side, 16, thumb_deg=8)
    aim(rig, "Head", (0.05, -0.05, 1.0), twist_deg=-6)


def keyart(rig):
    """The identity pose: airborne aim-and-load. Left hand reaches forward-up with spread fingers,
    right hand cocked behind the head, right knee tucked, left leg trailing down, head tilted toward
    the reach."""
    reset(rig)
    rig.pose.bones["Hips"].location = (0.0, 0.45, 0.0)  # bone-local Y = up for the Hips bone
    aim(rig, "Spine", (0.0, 0.12, 1.0), twist_deg=-12)
    aim(rig, "Chest", (0.02, 0.16, 1.0), twist_deg=-10)
    aim(rig, "UpperChest", (0.05, 0.1, 1.0), twist_deg=-8)
    aim(rig, "LeftUpperArm", (0.5, -0.72, 0.48))
    aim(rig, "LeftLowerArm", (0.42, -0.8, 0.44))
    aim(rig, "LeftHand", (0.36, -0.72, 0.6), twist_deg=-60)
    aim(rig, "RightShoulder", (-0.95, 0.05, 0.3))
    aim(rig, "RightUpperArm", (-0.5, 0.3, 0.8))
    aim(rig, "RightLowerArm", (0.45, 0.05, 0.9))
    aim(rig, "RightHand", (0.35, 0.25, 0.9), twist_deg=40)
    curl_fingers(rig, "Left", -6, thumb_deg=-10)
    curl_fingers(rig, "Right", 10, thumb_deg=4)
    aim(rig, "Neck", (0.12, -0.04, 1.0))
    aim(rig, "Head", (0.32, -0.12, 0.94), twist_deg=18)
    aim(rig, "LeftUpperLeg", (0.1, -0.28, -0.95))
    aim(rig, "LeftLowerLeg", (0.08, 0.3, -0.95))
    aim(rig, "LeftFoot", (0.05, -0.2, -0.98))
    aim(rig, "RightUpperLeg", (-0.06, -0.62, -0.78))
    aim(rig, "RightLowerLeg", (-0.04, 0.88, -0.48))
    aim(rig, "RightFoot", (-0.02, 0.5, -0.87))
