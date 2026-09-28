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


def ground(rig, standing="Left", ankle_z=None):
    """Move the hips up/down so the standing leg's ankle sits at its rest height (feet on the floor)."""
    bpy.context.view_layer.update()
    rest = rig.data.bones[f"{standing}Foot"].head_local
    target = rest.z if ankle_z is None else ankle_z
    z = (rig.matrix_world @ rig.pose.bones[f"{standing}Foot"].head).z
    rig.pose.bones["Hips"].location.y += target - z   # Hips bone-local Y = world up
    bpy.context.view_layer.update()


def stand(rig):
    """Natural neutral stance (contrapposto): weight on her left leg with the hip shifted over it, the
    right knee soft and the foot turned out a little, shoulders countering the hips, elbows and
    fingers relaxed, a small head tilt. Never a mannequin A-pose."""
    reset(rig)
    hips = rig.pose.bones["Hips"]
    hips.rotation_mode = "XYZ"
    hips.location = (0.016, 0.0, 0.0)                 # shift over the standing leg
    hips.rotation_euler = (0.0, math.radians(-3.0), math.radians(4.0))
    bpy.context.view_layer.update()
    aim(rig, "Spine", (-0.035, 0.01, 1.0))
    aim(rig, "Chest", (-0.03, 0.0, 1.0))
    aim(rig, "UpperChest", (-0.015, -0.01, 1.0), twist_deg=-3)
    # Standing leg under the body; free leg relaxed forward-out with a soft knee.
    # Thighs angle in toward the knees (never parallel tubes): a long tapering leg line.
    aim(rig, "LeftUpperLeg", (-0.085, -0.01, -1.0))
    aim(rig, "LeftLowerLeg", (0.01, 0.035, -1.0))
    aim(rig, "LeftFoot", (0.1, -0.72, -0.3))
    aim(rig, "RightUpperLeg", (0.05, -0.14, -1.0))
    aim(rig, "RightLowerLeg", (-0.05, 0.16, -1.0))
    aim(rig, "RightFoot", (-0.25, -0.7, -0.26))
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        aim(rig, f"{side}UpperArm", (s * 0.16, 0.05, -1.0))
        aim(rig, f"{side}LowerArm", (s * 0.06, -0.32, -1.0))
        aim(rig, f"{side}Hand", (s * -0.02, -0.26, -1.0), twist_deg=s * 10)
        curl_fingers(rig, side, 24, thumb_deg=10)
    aim(rig, "Neck", (0.02, -0.03, 1.0))
    aim(rig, "Head", (0.06, -0.05, 1.0), twist_deg=-6)
    ground(rig, "Left")


def stream_chain(rig, prefix, directions):
    """Pose a secondary chain (hair / ribbon) along a list of world directions, root to tip, so a still
    frame shows the flow the spring bones would produce in motion."""
    names = sorted(n for n in rig.pose.bones.keys() if n.startswith(prefix + "_") and n[len(prefix) + 1:len(prefix) + 3].isdigit())
    for i, n in enumerate(names):
        d = directions[min(int(i * len(directions) / len(names)), len(directions) - 1)]
        aim(rig, n, d)


def keyart(rig):
    """The identity pose (key art): airborne reach-and-load. Her left arm reaches long toward the
    viewer's right with an open palm; the right arm is cocked up beside the head; the torso leans back
    and toward her right; the head tilts toward the reach; the right knee is tucked with the foot kicked
    behind; the left leg extends long and down with the toes pointed; hair and ribbon stream behind."""
    reset(rig)
    rig.pose.bones["Hips"].location = (0.0, 0.45, 0.0)  # bone-local Y = up for the Hips bone
    hips = rig.pose.bones["Hips"]
    hips.rotation_mode = "XYZ"
    hips.rotation_euler = (math.radians(-8.0), math.radians(10.0), math.radians(-6.0))
    bpy.context.view_layer.update()
    aim(rig, "Spine", (-0.18, 0.12, 1.0), twist_deg=-8)
    aim(rig, "Chest", (-0.22, 0.15, 1.0), twist_deg=-8)
    aim(rig, "UpperChest", (-0.16, 0.1, 1.0), twist_deg=-6)
    # Reaching arm: long and nearly straight, out to her left and a little forward and up.
    aim(rig, "LeftShoulder", (0.95, -0.05, 0.1))
    aim(rig, "LeftUpperArm", (0.9, -0.38, 0.05))
    aim(rig, "LeftLowerArm", (0.9, -0.4, 0.12))
    aim(rig, "LeftHand", (0.85, -0.42, 0.3), twist_deg=-75)
    # Loading arm: cocked up beside the head.
    aim(rig, "RightShoulder", (-0.93, 0.08, 0.3))
    aim(rig, "RightUpperArm", (-0.72, 0.28, 0.6))
    aim(rig, "RightLowerArm", (0.3, 0.35, 0.88))
    aim(rig, "RightHand", (0.4, 0.2, 0.9), twist_deg=40)
    curl_fingers(rig, "Left", -4, thumb_deg=-8)
    curl_fingers(rig, "Right", 12, thumb_deg=6)
    aim(rig, "Neck", (0.08, -0.06, 1.0))
    aim(rig, "Head", (0.4, -0.12, 0.9), twist_deg=16)
    # Legs: right knee tucked forward with the foot kicked behind; left leg long and pointed.
    aim(rig, "RightUpperLeg", (-0.18, -0.55, -0.82))
    aim(rig, "RightLowerLeg", (0.0, 0.95, 0.1))
    aim(rig, "RightFoot", (-0.05, 0.75, 0.6))
    aim(rig, "LeftUpperLeg", (0.1, 0.22, -0.97))
    aim(rig, "LeftLowerLeg", (0.16, 0.3, -0.94))
    aim(rig, "LeftFoot", (0.08, -0.25, -0.96))
    # Hair and ribbon stream back and to her right (the viewer's left), as in the art.
    stream_chain(rig, "hair_ponytail_A", [(-0.3, 0.7, 0.3), (-0.7, 0.6, 0.1), (-0.9, 0.35, -0.1), (-0.95, 0.2, 0.05)])
    stream_chain(rig, "hair_ponytail_B", [(-0.2, 0.75, 0.4), (-0.6, 0.65, 0.2), (-0.85, 0.4, 0.1), (-0.9, 0.2, 0.2)])
    stream_chain(rig, "hair_ponytail_C", [(-0.3, 0.65, 0.2), (-0.75, 0.5, -0.05), (-0.95, 0.25, -0.2)])
    stream_chain(rig, "ribbon_tail_A", [(-0.4, 0.5, 0.6), (-0.8, 0.3, 0.5), (-0.9, 0.1, 0.4)])
    stream_chain(rig, "ribbon_loop_A", [(-0.2, 0.1, 0.97)])
