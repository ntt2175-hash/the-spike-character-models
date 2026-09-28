"""Previs posing controls on top of the standard deform rig.

Adds world-space IK targets (hands, feet) with pole targets, plus rotation
targets for the hands and feet, so key poses can be written as "where the
hands and feet are" instead of joint angles. This is a previs/blocking
control layer; production animation uses the animator's control rig, then
bakes to the deform skeleton.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector


class PoseRig:
    def __init__(self, rig: bpy.types.Object, prefix: str | None = None):
        self.rig = rig
        self.prefix = prefix or rig.name
        self.targets: dict[str, bpy.types.Object] = {}
        self.rest_rot: dict[str, Matrix] = {}
        self.rot_cons: dict[str, bpy.types.Constraint] = {}
        self._build()

    # -- setup ---------------------------------------------------------------
    def _empty(self, name, loc, rot=None, size=0.05):
        e = bpy.data.objects.new(f"{self.prefix}_{name}", None)
        e.empty_display_size = size
        e.empty_display_type = "SPHERE"
        bpy.context.scene.collection.objects.link(e)
        e.location = loc
        e.rotation_mode = "QUATERNION"
        if rot is not None:
            e.rotation_quaternion = rot.to_quaternion()
        e.hide_render = True
        return e

    def _build(self):
        rig = self.rig
        mw = rig.matrix_world
        for side in ("Left", "Right"):
            sx = 1.0 if side == "Left" else -1.0
            for limb, end, mid, pole_off in (("arm", f"{side}Hand", f"{side}LowerArm", Vector((sx * 0.1, 0.35, 0.0))),
                                             ("leg", f"{side}Foot", f"{side}LowerLeg", Vector((0.0, -0.6, 0.0)))):
                end_b = rig.data.bones[end]
                mid_b = rig.data.bones[mid]
                rest = (mw @ end_b.matrix_local)
                tgt = self._empty(f"{side}_{limb}_ik", rest.translation, rest.to_3x3())
                pole = self._empty(f"{side}_{limb}_pole", (mw @ mid_b.head_local) + pole_off, size=0.03)
                self.targets[f"{side}_{limb}"] = tgt
                self.targets[f"{side}_{limb}_pole"] = pole
                self.rest_rot[f"{side}_{limb}"] = rest.to_3x3()
                pb = rig.pose.bones[mid]
                ik = pb.constraints.new("IK")
                ik.name = "previs_ik"
                ik.target = tgt
                ik.pole_target = pole
                ik.chain_count = 2
                ik.use_tail = False
                cr = rig.pose.bones[end].constraints.new("COPY_ROTATION")
                cr.name = "previs_rot"
                cr.target = tgt
                cr.target_space = "WORLD"
                cr.owner_space = "WORLD"
                cr.influence = 1.0 if limb == "leg" else 0.0
                self.rot_cons[f"{side}_{limb}"] = cr
        self._calibrate_poles()

    def _calibrate_poles(self):
        """Pick each IK pole angle so the elbow/knee points at its pole target."""
        rig = self.rig
        for side in ("Left", "Right"):
            for limb, root, mid in (("arm", f"{side}UpperArm", f"{side}LowerArm"), ("leg", f"{side}UpperLeg", f"{side}LowerLeg")):
                tgt = self.targets[f"{side}_{limb}"]
                pole = self.targets[f"{side}_{limb}_pole"]
                ik = rig.pose.bones[mid].constraints["previs_ik"]
                home = tgt.location.copy()
                shoulder = rig.matrix_world @ rig.data.bones[root].head_local
                tgt.location = shoulder.lerp(home, 0.7)  # force a bend
                best, best_d = 0.0, 1e9
                for deg in range(-180, 180, 5):
                    ik.pole_angle = math.radians(deg)
                    bpy.context.view_layer.update()
                    elbow = rig.matrix_world @ rig.pose.bones[mid].head
                    line_p = shoulder
                    line_d = (tgt.location - shoulder).normalized()
                    foot = line_p + line_d * (elbow - line_p).dot(line_d)
                    bend = (elbow - foot)
                    want = pole.location - (line_p + line_d * (pole.location - line_p).dot(line_d))
                    d = -bend.normalized().dot(want.normalized()) if bend.length > 1e-6 else 1.0
                    if d < best_d:
                        best, best_d = deg, d
                ik.pole_angle = math.radians(best)
                tgt.location = home
        bpy.context.view_layer.update()

    # -- helpers -------------------------------------------------------------
    def bone_local_delta(self, bone: str, world_delta: Vector) -> Vector:
        b = self.rig.data.bones[bone]
        return (self.rig.matrix_world @ b.matrix_local).to_3x3().inverted() @ world_delta

    def oriented(self, key: str, finger_dir: Vector, palm_dir: Vector) -> Quaternion:
        """World rotation for a hand/foot target so the bone points along finger_dir with its palm/sole toward palm_dir."""
        rest = self.rest_rot[key]
        y_rest = (rest @ Vector((0, 1, 0))).normalized()
        side = 1.0 if key.startswith("Left") else -1.0
        inward = Vector((-side, 0.0, 0.0)) if key.endswith("arm") else Vector((0.0, 0.0, -1.0))
        palm_rest = (inward - y_rest * inward.dot(y_rest)).normalized()
        a = Matrix((y_rest, palm_rest, y_rest.cross(palm_rest))).transposed()
        f = finger_dir.normalized()
        p = (palm_dir - f * palm_dir.dot(f)).normalized()
        b = Matrix((f, p, f.cross(p))).transposed()
        return (b @ a.inverted() @ rest).to_quaternion()

    # -- keying ---------------------------------------------------------------
    def key(self, frame: int, *, hips=None, hips_rot=None, spine=None, chest=None, upper_chest=None, neck=None,
            head=None, hands=None, hand_rot=None, hand_follow=None, feet=None, foot_pitch=None, poles=None,
            fingers=None):
        """Key a pose. Angles in degrees as (pitch_fwd, twist_left, bend_right) in each bone's local frame.

        hips        : world position of the Hips bone head
        hands/feet  : {"Left": Vector, "Right": Vector} world IK target positions
        hand_rot    : {"Left": Quaternion}  world rotation (see oriented())
        hand_follow : {"Left": 0..1} 1 = hand uses hand_rot, 0 = hand follows the forearm
        foot_pitch  : {"Left": deg} toe-down pitch applied to the rest foot orientation
        poles       : {"Left_arm": Vector, ...} world pole positions
        fingers     : {"Left": curl_deg}
        """
        rig = self.rig
        if hips is not None:
            pb = rig.pose.bones["Hips"]
            rest_head = rig.matrix_world @ rig.data.bones["Hips"].head_local
            pb.location = self.bone_local_delta("Hips", Vector(hips) - rest_head)
            pb.keyframe_insert("location", frame=frame)
        for bone, ang in (("Hips", hips_rot), ("Spine", spine), ("Chest", chest), ("UpperChest", upper_chest),
                          ("Neck", neck), ("Head", head)):
            if ang is None:
                continue
            pb = rig.pose.bones[bone]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((math.radians(ang[0]), math.radians(ang[1]), math.radians(-ang[2])), "XYZ")
            pb.keyframe_insert("rotation_euler", frame=frame)
        for side, pos in (hands or {}).items():
            t = self.targets[f"{side}_arm"]
            t.location = pos
            t.keyframe_insert("location", frame=frame)
        for side, q in (hand_rot or {}).items():
            t = self.targets[f"{side}_arm"]
            t.rotation_quaternion = q
            t.keyframe_insert("rotation_quaternion", frame=frame)
        for side, w in (hand_follow or {}).items():
            con = self.rot_cons[f"{side}_arm"]
            con.influence = w
            con.keyframe_insert("influence", frame=frame)
        for side, pos in (feet or {}).items():
            t = self.targets[f"{side}_leg"]
            t.location = pos
            t.keyframe_insert("location", frame=frame)
        for side, deg in (foot_pitch or {}).items():
            t = self.targets[f"{side}_leg"]
            rot = Matrix.Rotation(math.radians(deg), 3, "X") @ self.rest_rot[f"{side}_leg"]
            t.rotation_quaternion = rot.to_quaternion()
            t.keyframe_insert("rotation_quaternion", frame=frame)
        for key_, pos in (poles or {}).items():
            t = self.targets[f"{key_}_pole"]
            t.location = pos
            t.keyframe_insert("location", frame=frame)
        for side, curl in (fingers or {}).items():
            for b in rig.pose.bones:
                if b.name.startswith(side) and any(f in b.name for f in ("Index", "Middle", "Ring", "Little")):
                    b.rotation_mode = "XYZ"
                    b.rotation_euler = Euler((math.radians(curl), 0.0, 0.0), "XYZ")
                    b.keyframe_insert("rotation_euler", frame=frame)
