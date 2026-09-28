"""Identity-colored STAND-IN mannequin, skinned to the standard rig.

This is NOT a character model and must never be presented as one. It exists
so that layout, camera, lighting and animation work (previs, S+ sequence
timing, QC framing) can start before the production model exists, using the
real rig, the real proportions and the character's color blocking.

Limbs, torso, hair strands, ribbons and hem flaps are single smooth tubes
skinned to their bone chains (so elbows, knees and ponytails bend); hands,
shoes and face features are rigid pieces. Everything lives in the
{char}_PROXY collection (never exported, invisible to the validator) and
carries the custom property spike_proxy = True.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import spike_specs as ss

from . import toon

RING = 20


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bone_frame(rig, bone_name):
    b = rig.data.bones[bone_name]
    m = rig.matrix_world @ b.matrix_local
    r = m.to_3x3()
    return (m.translation.copy(), (r @ Vector((1, 0, 0))).normalized(), (r @ Vector((0, 1, 0))).normalized(),
            (r @ Vector((0, 0, 1))).normalized(), b.length)


def _head(rig, bone):
    return rig.matrix_world @ rig.data.bones[bone].head_local


def _tail(rig, bone):
    return rig.matrix_world @ rig.data.bones[bone].tail_local


def _link(name, me, coll, mats):
    for p in me.polygons:
        p.use_smooth = True
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    for m in mats:
        me.materials.append(m)
    obj["spike_proxy"] = True
    return obj


def _parent_to_bone(obj, rig, bone_name):
    bpy.context.view_layer.update()
    mw = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    bpy.context.view_layer.update()
    obj.matrix_world = mw


def _interp_profile(profile, u):
    if u <= profile[0][0]:
        return profile[0][1:]
    for (u0, *a), (u1, *b) in zip(profile, profile[1:]):
        if u <= u1:
            t = (u - u0) / max(u1 - u0, 1e-9)
            return tuple(x + (y - x) * t for x, y in zip(a, b))
    return profile[-1][1:]


def skinned_tube(rig, name, coll, mats, bones, profile, *, start=None, end=None, extend_start=0.0,
                 extend_end=0.0, step=0.012, side_threshold=None, flat_axis_from=None, closed_caps=True):
    """A smooth tube along a chain of bones, weighted to those bones.

    bones   : chain of bone names; the path runs head(bones[0]) -> ... -> tail(bones[-1])
    profile : [(u, rx, rz), ...] with u in 0..1 over the WHOLE path including extensions,
              rx across the bone's X axis, rz across its Z axis (meters)
    side_threshold : if set, faces whose ring angle has |cos| above it use mats[1] (side panels)
    """
    pts = [start if start is not None else _head(rig, bones[0])]
    for b in bones:
        pts.append(_tail(rig, b))
    if end is not None:
        pts[-1] = end
    frames = [_bone_frame(rig, b) for b in bones]
    seg_len = [(pts[i + 1] - pts[i]).length for i in range(len(bones))]
    total = extend_start + sum(seg_len) + extend_end

    # Sample stations: (position, x_axis, z_axis, {bone: weight}, u)
    stations = []

    def frame_at(k, f):
        _, x0, _, z0, _ = frames[k]
        d = (pts[k + 1] - pts[k]).normalized()
        blend = None
        if k > 0 and f < 0.2:
            blend = (k - 1, 0.5 - 0.5 * (f / 0.2))
        elif k < len(bones) - 1 and f > 0.8:
            blend = (k + 1, 0.5 - 0.5 * ((1.0 - f) / 0.2))
        x, z = x0.copy(), z0.copy()
        if blend:
            _, x1, _, z1, _ = frames[blend[0]]
            x = x.lerp(x1, blend[1])
            z = z.lerp(z1, blend[1])
        x = (x - d * x.dot(d)).normalized()
        z = d.cross(x).normalized() if z.dot(d.cross(x)) >= 0 else -d.cross(x).normalized()
        return x, z

    def weights_at(k, f):
        w = {bones[k]: 1.0}
        if k > 0 and f < 0.2:
            t = 0.5 + 0.5 * (f / 0.2)
            w = {bones[k]: t, bones[k - 1]: 1.0 - t}
        elif k < len(bones) - 1 and f > 0.8:
            t = 0.5 + 0.5 * ((1.0 - f) / 0.2)
            w = {bones[k]: t, bones[k + 1]: 1.0 - t}
        return w

    n_ext0 = max(0, int(math.ceil(extend_start / step)))
    d0 = (pts[1] - pts[0]).normalized()
    for i in range(n_ext0, 0, -1):
        s = i * extend_start / n_ext0
        x, z = frame_at(0, 0.0)
        stations.append((pts[0] - d0 * s, x, z, {bones[0]: 1.0}, (extend_start - s) / total))
    acc = extend_start
    for k in range(len(bones)):
        n = max(2, int(math.ceil(seg_len[k] / step)))
        for i in range(n + (1 if k == len(bones) - 1 else 0)):
            f = i / n
            p = pts[k].lerp(pts[k + 1], f)
            x, z = frame_at(k, f)
            stations.append((p, x, z, weights_at(k, f), (acc + f * seg_len[k]) / total))
        acc += seg_len[k]
    n_ext1 = max(0, int(math.ceil(extend_end / step)))
    d1 = (pts[-1] - pts[-2]).normalized()
    for i in range(1, n_ext1 + 1):
        s = i * extend_end / n_ext1
        x, z = frame_at(len(bones) - 1, 1.0)
        stations.append((pts[-1] + d1 * s, x, z, {bones[-1]: 1.0}, (acc + s) / total))

    bm = bmesh.new()
    vweights = []
    rings = []
    for p, x, z, w, u in stations:
        rx, rz = _interp_profile(profile, u)
        ring = []
        for j in range(RING):
            a = 2.0 * math.pi * j / RING
            ring.append(bm.verts.new(p + x * (math.cos(a) * rx) + z * (math.sin(a) * rz)))
            vweights.append(w)
        rings.append(ring)
    face_mats = []
    for r0, r1 in zip(rings, rings[1:]):
        for j in range(RING):
            jn = (j + 1) % RING
            bm.faces.new((r0[j], r0[jn], r1[jn], r1[j]))
            a = 2.0 * math.pi * (j + 0.5) / RING
            face_mats.append(1 if side_threshold is not None and abs(math.cos(a)) > side_threshold else 0)
    if closed_caps:
        for ring, (p, x, z, w, u), flip in ((rings[0], stations[0], True), (rings[-1], stations[-1], False)):
            c = bm.verts.new(p)
            vweights.append(w)
            for j in range(RING):
                jn = (j + 1) % RING
                bm.faces.new((c, ring[jn], ring[j]) if flip else (c, ring[j], ring[jn]))
                face_mats.append(0)
    bm.verts.index_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for poly, mi in zip(me.polygons, face_mats):
        poly.material_index = mi if mi < len(mats) else 0
    obj = _link(name, me, coll, mats)
    groups = {}
    for vi, w in enumerate(vweights):
        for bone, val in w.items():
            if bone not in groups:
                groups[bone] = obj.vertex_groups.new(name=bone)
            groups[bone].add([vi], val, "REPLACE")
    obj.parent = rig
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    return obj


def ellipsoid(rig, bone, center, radii, coll, mat, name, rot=None, segments=24, rings=14):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=1.0)
    s = Matrix.Diagonal((radii[0], radii[1], radii[2], 1.0))
    r = (rot or Matrix.Identity(3)).to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ r @ s, verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = _link(name, me, coll, [mat])
    _parent_to_bone(obj, rig, bone)
    return obj


def box(rig, bone, center, size, coll, mat, name, rot=None, bevel=0.3):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.5 * bevel, segments=3, affect="EDGES", profile=0.5)
    s = Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    r = (rot or Matrix.Identity(3)).to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ r @ s, verts=bm.verts)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = _link(name, me, coll, [mat])
    _parent_to_bone(obj, rig, bone)
    return obj


def _front_y(center, radii, x, z):
    k = 1.0 - ((x - center.x) / radii[0]) ** 2 - ((z - center.z) / radii[2]) ** 2
    return center.y - radii[1] * math.sqrt(max(0.0, k))


def _bone_rot(rig, bone):
    _, x, y, z, _ = _bone_frame(rig, bone)
    return Matrix((x, y, z)).transposed()


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
def build_proxy(rig, character: dict, coll_name: str | None = None, outline: bool = True,
                palette_override: dict | None = None) -> bpy.types.Collection:
    char = character["id"]
    H = character["proportions"]["height_m"]
    pal = dict(character["palette"])
    if palette_override:
        pal.update(palette_override)
    coll = bpy.data.collections.get(coll_name or f"{char}_PROXY")
    if coll is None:
        coll = bpy.data.collections.new(coll_name or f"{char}_PROXY")
        bpy.context.scene.collection.children.link(coll)
    toon.ensure_scene_props()
    tag = coll.name

    def M(slot, base, shade=None, **kw):
        return toon.toon_material(f"PX_{tag}_{slot}", base, shade, pal, **kw)

    skin = M("skin", "@skin_base", "@skin_shadow", threshold=0.3, softness=0.05)
    face = M("face", "@skin_light", "@skin_base", threshold=0.24, softness=0.07)
    hair = M("hair", "@hair_base", "@hair_shadow", threshold=0.36, softness=0.03)
    jersey_top = M("jersey_top", "@jersey_front_top", "#16263d",
                   gradient=("@jersey_front_bottom", "#6793ad", 0.05, 0.95))
    jersey_mid = M("jersey_mid", "#4f7d9a", "#2e5270")
    jersey_low = M("jersey_low", "@jersey_front_bottom", "#6793ad")
    jersey_side = M("jersey_side", "@jersey_side", "#b9c3d6")
    shorts = M("shorts", "@shorts_base", "#0c1a33")
    shorts_side = M("shorts_side", "@shorts_side", "#7497b0")
    pad = M("kneepad", "@kneepad_pad", "#b4bccb")
    pad_band = M("kneepad_band", "@kneepad_band", "#5f84a0")
    sock = M("sock", "@sock_black", "#0c0c12", rim_mask=1.4)
    ankle = M("ankle_sock", "@ankle_sock", "#aab0bf")
    shoe = M("shoe_upper", "@shoe_upper", "#8a90a3")
    sole = M("shoe_sole", "@shoe_sole", "#b9bcc6")
    shoe_acc = M("shoe_accent", "@shoe_accent", "#050506")
    ribbon = M("ribbon", "@ribbon", "#3f6fb0", threshold=0.25)
    tie = M("hair_tie", "@hair_tie", "#1c2536")
    white = M("print", "@jersey_print", "#b8cad6", rim_mask=0.3)
    sclera = M("sclera", "@sclera", "#c9c8d8", threshold=0.1, rim_mask=0.0)
    iris_mid = M("iris", "@iris_mid", "@iris_top", threshold=0.1, rim_mask=0.0, emission=0.25)
    iris_low = M("iris_low", "@iris_bottom", "@iris_mid", threshold=0.1, rim_mask=0.0, emission=0.35)
    pupil = M("pupil", "@pupil", "#05080c", threshold=0.1, rim_mask=0.0)
    lash = M("lash", "#1a1418", "#0c090b", rim_mask=0.0)
    catch = M("catchlight", "#ffffff", "#ffffff", rim_mask=0.0, emission=1.2)
    mouth = M("mouth", "#8a4452", "#5a2733", rim_mask=0.0)
    blush = M("blush", "@skin_blush", "@skin_blush", rim_mask=0.0)
    outline_mat = toon.outline_material(f"PX_{tag}_outline", "#2a1f2c")
    outlined, plain = [], []

    # --- torso: pelvis -> neck as one skinned form -------------------------
    torso_bones = ["Hips", "Spine", "Chest", "UpperChest", "Neck"]
    body = skinned_tube(rig, f"proxy_{tag}_torso", coll, [skin], torso_bones, [
        (0.00, 0.128, 0.086), (0.10, 0.152, 0.100), (0.20, 0.150, 0.096), (0.30, 0.128, 0.085),
        (0.40, 0.116, 0.079), (0.52, 0.128, 0.086), (0.64, 0.142, 0.094), (0.74, 0.148, 0.094),
        (0.80, 0.128, 0.078), (0.85, 0.060, 0.052), (0.90, 0.043, 0.041), (1.00, 0.040, 0.039)],
        extend_start=0.085)
    outlined.append(body)
    # Shorts (pelvis + leg openings) and jersey as garment shells over the body.
    outlined.append(skinned_tube(rig, f"proxy_{tag}_shorts", coll, [shorts, shorts_side], ["Hips", "Spine"], [
        (0.0, 0.134, 0.091), (0.25, 0.158, 0.105), (0.55, 0.156, 0.101), (0.8, 0.136, 0.09), (1.0, 0.132, 0.088)],
        end=_head(rig, "Spine").lerp(_tail(rig, "Spine"), 0.2), extend_start=0.09, side_threshold=0.9))
    outlined.append(skinned_tube(rig, f"proxy_{tag}_jersey", coll, [jersey_top, jersey_side],
                                 ["Spine", "Chest", "UpperChest"], [
        (0.0, 0.136, 0.092), (0.2, 0.124, 0.085), (0.45, 0.136, 0.092), (0.62, 0.15, 0.1),
        (0.8, 0.156, 0.1), (0.92, 0.14, 0.085), (1.0, 0.085, 0.06)], side_threshold=0.82))
    # Hem: loose flared skirt, pale end of the gradient.
    outlined.append(skinned_tube(rig, f"proxy_{tag}_jersey_hem", coll, [jersey_low, jersey_side], ["Hips", "Spine"], [
        (0.0, 0.168, 0.118), (0.45, 0.156, 0.107), (0.75, 0.14, 0.096), (1.0, 0.137, 0.094)],
        start=_head(rig, "Hips") + Vector((0, 0, -0.035)), end=_head(rig, "Spine").lerp(_tail(rig, "Spine"), 0.5),
        side_threshold=0.86, closed_caps=False))
    # Number on the chest.
    curve = bpy.data.curves.new(f"proxy_{tag}_number", "FONT")
    curve.body = str(character.get("identity", {}).get("jersey_number", ""))
    curve.size = 0.1
    curve.extrude = 0.003
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    num = bpy.data.objects.new(f"proxy_{tag}_number", curve)
    coll.objects.link(num)
    num.data.materials.append(white)
    up_head = _head(rig, "UpperChest")
    num.location = up_head + Vector((0.018, -0.104, 0.03))
    num.rotation_euler = (math.radians(90), 0, math.radians(-4))
    num["spike_proxy"] = True
    _parent_to_bone(num, rig, "UpperChest")
    plain.append(num)

    # --- limbs ---------------------------------------------------------------
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        sh = _head(rig, f"{side}UpperArm")
        outlined.append(skinned_tube(rig, f"proxy_{tag}_arm_{side}", coll, [skin],
                                     [f"{side}Shoulder", f"{side}UpperArm", f"{side}LowerArm"], [
            (0.0, 0.03, 0.03), (0.2, 0.046, 0.044), (0.3, 0.043, 0.041), (0.55, 0.035, 0.034),
            (0.62, 0.031, 0.029), (0.75, 0.033, 0.03), (1.0, 0.021, 0.018)],
            start=_head(rig, f"{side}Shoulder").lerp(sh, 0.45)))
        outlined.append(skinned_tube(rig, f"proxy_{tag}_leg_{side}", coll, [skin], [f"{side}UpperLeg", f"{side}LowerLeg"], [
            (0.0, 0.08, 0.082), (0.1, 0.082, 0.084), (0.3, 0.069, 0.07), (0.47, 0.052, 0.054),
            (0.53, 0.047, 0.05), (0.66, 0.05, 0.056), (0.88, 0.032, 0.034), (1.0, 0.026, 0.028)],
            extend_start=0.03))
        outlined.append(skinned_tube(rig, f"proxy_{tag}_shorts_leg_{side}", coll, [shorts], [f"{side}UpperLeg"], [
            (0.0, 0.087, 0.089), (1.0, 0.083, 0.085)], end=_head(rig, f"{side}UpperLeg").lerp(_tail(rig, f"{side}UpperLeg"), 0.2),
            extend_start=0.03, closed_caps=False))
        knee = _tail(rig, f"{side}UpperLeg")
        lower = f"{side}LowerLeg"
        outlined.append(skinned_tube(rig, f"proxy_{tag}_kneepad_{side}", coll, [pad], [f"{side}UpperLeg", lower], [
            (0.0, 0.054, 0.058), (0.5, 0.058, 0.066), (1.0, 0.052, 0.058)],
            start=_head(rig, f"{side}UpperLeg").lerp(knee, 0.9), end=knee.lerp(_tail(rig, lower), 0.13)))
        for f0 in (0.9, 1.1):
            pt = knee.lerp(_tail(rig, lower), f0 - 1.0) if f0 > 1.0 else _head(rig, f"{side}UpperLeg").lerp(knee, f0)
            bone = lower if f0 > 1.0 else f"{side}UpperLeg"
            d = (_tail(rig, bone) - _head(rig, bone)).normalized()
            plain.append(skinned_tube(rig, f"proxy_{tag}_kneeband_{side}_{f0}", coll, [pad_band], [bone], [
                (0.0, 0.056, 0.061), (1.0, 0.056, 0.061)], start=pt - d * 0.009, end=pt + d * 0.009))
        # Hand: palm + fingers (rigid pieces; fingers are rigid per phalanx).
        hd, x, y, z, ln = _bone_frame(rig, f"{side}Hand")
        plain.append(box(rig, f"{side}Hand", hd + y * 0.036, (0.052, 0.07, 0.022), coll, skin, f"proxy_{tag}_palm_{side}",
                         rot=_bone_rot(rig, f"{side}Hand"), bevel=0.5))
        for fname in ("Thumb", "Index", "Middle", "Ring", "Little"):
            names = [b.name for b in rig.data.bones if b.name.startswith(f"{side}{fname}")]
            chain_r = 0.0072 if fname == "Thumb" else 0.0062
            plain.append(skinned_tube(rig, f"proxy_{tag}_finger_{side}{fname}", coll, [skin], names, [
                (0.0, chain_r, chain_r * 0.9), (1.0, chain_r * 0.72, chain_r * 0.66)], step=0.006, extend_end=0.003))
        # Shoes.
        fhd = _head(rig, f"{side}Foot")
        toe_tail = _tail(rig, f"{side}Toes")
        heel = fhd + Vector((0, 0.045, 0.0))
        mid = (heel + toe_tail) * 0.5
        L = Vector((toe_tail.x - heel.x, toe_tail.y - heel.y, 0)).length
        ground = rig.matrix_world.translation.z
        plain.append(box(rig, f"{side}Foot", Vector((mid.x, mid.y, ground + 0.05)), (0.092, L + 0.02, 0.085),
                         coll, shoe, f"proxy_{tag}_shoe_{side}", bevel=0.85))
        plain.append(box(rig, f"{side}Foot", Vector((mid.x, mid.y, ground + 0.012)), (0.098, L + 0.035, 0.024),
                         coll, sole, f"proxy_{tag}_sole_{side}", bevel=0.5))
        plain.append(box(rig, f"{side}Foot", Vector((mid.x + sx * 0.047, mid.y + 0.01, ground + 0.055)), (0.006, L * 0.55, 0.034),
                         coll, shoe_acc, f"proxy_{tag}_shoe_stripe_{side}", bevel=0.3))
        for o in plain[-3:]:
            outlined.append(o)
    # Asymmetric legwear: LEFT black knee-high, RIGHT ankle sock.
    kl = _tail(rig, "LeftUpperLeg")
    outlined.append(skinned_tube(rig, f"proxy_{tag}_sock_L", coll, [sock], ["LeftLowerLeg"], [
        (0.0, 0.053, 0.059), (0.2, 0.054, 0.06), (0.65, 0.035, 0.037), (1.0, 0.03, 0.032)],
        start=kl.lerp(_tail(rig, "LeftLowerLeg"), 0.12), closed_caps=False))
    outlined.append(skinned_tube(rig, f"proxy_{tag}_anklesock_R", coll, [ankle], ["RightLowerLeg"], [
        (0.0, 0.03, 0.032), (1.0, 0.03, 0.032)],
        start=_tail(rig, "RightUpperLeg").lerp(_tail(rig, "RightLowerLeg"), 0.9), closed_caps=False))

    # --- head ------------------------------------------------------------------
    crown = _tail(rig, "Head").z
    head_h = H / character["proportions"]["head_units"]
    cranium_c = Vector((0.0, 0.006, crown - 0.094))
    cr_r = (0.081, 0.09, 0.094)
    jaw_c = Vector((0.0, -0.016, crown - head_h + 0.07))
    jaw_r = (0.063, 0.069, 0.072)
    plain.append(ellipsoid(rig, "Head", cranium_c, cr_r, coll, face, f"proxy_{tag}_cranium", segments=32, rings=20))
    outlined.append(plain[-1])
    plain.append(ellipsoid(rig, "Head", jaw_c, jaw_r, coll, face, f"proxy_{tag}_face_lower", segments=32, rings=20))
    outlined.append(plain[-1])
    chin = Vector((0.0, -0.05, crown - head_h + 0.018))
    outlined.append(ellipsoid(rig, "Head", chin, (0.024, 0.024, 0.02), coll, face, f"proxy_{tag}_chin"))
    for sx in (1, -1):
        outlined.append(ellipsoid(rig, "Head", Vector((sx * 0.079, 0.012, crown - 0.118)), (0.012, 0.018, 0.024), coll, face,
                                  f"proxy_{tag}_ear_{sx}"))
    for side, sx in (("Left", 1), ("Right", -1)):
        eye_c = _head(rig, f"{side}Eye")
        base = Vector((eye_c.x, _front_y(cranium_c, cr_r, eye_c.x, eye_c.z) - 0.001, eye_c.z))
        face_rot = Matrix.Rotation(math.radians(-10 * sx), 3, "Z")
        plain.append(ellipsoid(rig, "Head", base + Vector((0, -0.001, 0)), (0.0235, 0.0035, 0.0165), coll, sclera,
                               f"proxy_{tag}_sclera_{side}", rot=face_rot))
        eb = f"{side}Eye"
        plain.append(ellipsoid(rig, eb, base + Vector((sx * 0.002, -0.0028, -0.001)), (0.0135, 0.0025, 0.0158), coll, iris_mid,
                               f"proxy_{tag}_iris_{side}", rot=face_rot))
        plain.append(ellipsoid(rig, eb, base + Vector((sx * 0.002, -0.0036, -0.0075)), (0.0105, 0.002, 0.0062), coll, iris_low,
                               f"proxy_{tag}_iris_low_{side}", rot=face_rot))
        plain.append(ellipsoid(rig, eb, base + Vector((sx * 0.002, -0.0042, 0.0005)), (0.0052, 0.0016, 0.0068), coll, pupil,
                               f"proxy_{tag}_pupil_{side}", rot=face_rot))
        plain.append(ellipsoid(rig, "Head", base + Vector((sx * 0.006, -0.0052, 0.006)), (0.0038, 0.0012, 0.0038), coll,
                               catch, f"proxy_{tag}_catch_{side}", rot=face_rot))
        lid = ellipsoid(rig, "Head", base + Vector((0, -0.0035, 0.0118)), (0.0265, 0.0035, 0.0105), coll, face,
                        f"proxy_{tag}_lid_{side}", rot=face_rot)
        lid["spike_lid"] = side
        plain.append(lid)
        plain.append(ellipsoid(rig, "Head", base + Vector((sx * 0.002, -0.0066, 0.0045)), (0.026, 0.0015, 0.0032), coll, lash,
                               f"proxy_{tag}_lash_{side}", rot=face_rot @ Matrix.Rotation(math.radians(-6 * sx), 3, "Y")))
        plain.append(ellipsoid(rig, "Head", base + Vector((sx * 0.004, -0.003, 0.031)), (0.019, 0.0025, 0.0022), coll, lash,
                               f"proxy_{tag}_brow_{side}", rot=face_rot @ Matrix.Rotation(math.radians(9 * sx), 3, "Y")))
        cz = eye_c.z - 0.03
        plain.append(ellipsoid(rig, "Head", Vector((sx * 0.046, _front_y(jaw_c, jaw_r, sx * 0.046, cz) - 0.004, cz)),
                               (0.016, 0.004, 0.008), coll, blush, f"proxy_{tag}_blush_{side}",
                               rot=Matrix.Rotation(math.radians(-24 * sx), 3, "Z")))
    mz, nz = crown - head_h + 0.048, crown - head_h + 0.084
    plain.append(ellipsoid(rig, "Jaw", Vector((0.0, _front_y(jaw_c, jaw_r, 0.0, mz) - 0.001, mz)), (0.0085, 0.003, 0.0055),
                           coll, mouth, f"proxy_{tag}_mouth"))
    plain.append(ellipsoid(rig, "Head", Vector((0.0, _front_y(jaw_c, jaw_r, 0.0, nz) - 0.002, nz)), (0.0035, 0.004, 0.0035),
                           coll, blush, f"proxy_{tag}_nose_tip"))

    # --- hair ------------------------------------------------------------------
    cap = ellipsoid(rig, "Head", cranium_c + Vector((0, 0.007, 0.008)), (0.088, 0.098, 0.101), coll, hair,
                    f"proxy_{tag}_hair_cap", segments=40, rings=24)
    _cut_face_opening(cap, cranium_c, crown)
    outlined.append(cap)
    outlined.append(ellipsoid(rig, "Head", cranium_c + Vector((0, 0.045, -0.05)), (0.074, 0.06, 0.07), coll, hair,
                              f"proxy_{tag}_hair_nape"))
    chains = character.get("secondary_chains", {}).get("chains", [])
    for ch in chains:
        names = ss.chain_bone_names(ch)
        cid = ch["id"] + (f"_{ch['side']}" if ch.get("side") else "")
        if ch["id"].startswith("cloth"):
            continue  # the hem shell already reads; the hem bones still drive it in the production mesh
        if "ponytail" in ch["id"]:
            r0 = {"hair_ponytail_A": 0.034, "hair_ponytail_B": 0.026, "hair_ponytail_C": 0.022}.get(ch["id"], 0.02)
            prof = [(0.0, r0 * 0.75, r0 * 0.6), (0.12, r0, r0 * 0.75), (0.5, r0 * 0.85, r0 * 0.6), (1.0, 0.0025, 0.0015)]
        elif "bang" in ch["id"]:
            prof = [(0.0, 0.017, 0.005), (0.4, 0.015, 0.0045), (1.0, 0.0014, 0.001)]
        elif "side" in ch["id"]:
            prof = [(0.0, 0.017, 0.007), (0.5, 0.015, 0.006), (1.0, 0.0016, 0.001)]
        elif "ahoge" in ch["id"] or "flyaway" in ch["id"]:
            prof = [(0.0, 0.0035, 0.0018), (1.0, 0.0006, 0.0005)]
        elif ch["id"].startswith("ribbon_loop"):
            prof = [(0.0, 0.004, 0.0012), (0.5, 0.011, 0.0012), (1.0, 0.004, 0.0012)]
        elif ch["id"].startswith("ribbon"):
            prof = [(0.0, 0.0065, 0.0012), (1.0, 0.0065, 0.0012)]
        else:
            continue
        mat = ribbon if ch["id"].startswith("ribbon") else hair
        o = skinned_tube(rig, f"proxy_{tag}_{cid}", coll, [mat], names, prof, step=0.008,
                         extend_start=0.012 if ch["id"].startswith("hair_bang") else 0.0)
        (outlined if not ch["id"].startswith(("hair_flyaway", "hair_ahoge")) else plain).append(o)
    root = next((c for c in chains if c["id"] == "hair_ponytail_A"), None)
    if root:
        p = _head(rig, ss.chain_bone_names(root)[0])
        outlined.append(ellipsoid(rig, "Head", p, (0.022, 0.022, 0.014), coll, tie, f"proxy_{tag}_hair_tie",
                                  rot=Matrix.Rotation(math.radians(-55), 3, "X")))

    if outline:
        for o in outlined:
            if o.type == "MESH":
                toon.add_outline(o, outline_mat, thickness=0.0026)
    return coll


def _cut_face_opening(cap, cranium_c, crown):
    """Remove the front-lower part of the hair cap so the face shows (hairline at the brow)."""
    me = cap.data
    bm = bmesh.new()
    bm.from_mesh(me)
    mw = cap.matrix_world
    kill = []
    for v in bm.verts:
        p = mw @ v.co
        rel = p - cranium_c
        if rel.y < -0.02 and p.z < crown - 0.05 and abs(rel.x) < 0.076:
            kill.append(v)
        elif rel.y < 0.0 and p.z < crown - 0.135:
            kill.append(v)
    bmesh.ops.delete(bm, geom=kill, context="VERTS")
    bm.to_mesh(me)
    bm.free()


def set_lids(coll, closed: float, frame: int | None = None):
    """0 = the character's default lid, 1 = fully closed. Keys the lid objects when a frame is given."""
    for o in coll.objects:
        if "spike_lid" not in o:
            continue
        o.scale = (1.0, 1.0, 1.0 + 1.6 * closed)
        o.delta_location = (0.0, 0.0, -0.0095 * closed)
        if frame is not None:
            o.keyframe_insert("scale", frame=frame)
            o.keyframe_insert("delta_location", frame=frame)
