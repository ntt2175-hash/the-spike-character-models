"""Build Sara v0.2 (procedural master) into a .blend: sculpted body, lofted head, painted face and
layered eyes, lock hair, ribbon, garments, shoes, preview materials, skin weights, bound to the
standard rig.

    blender -b --python characters/sara/build/build_sara.py -- <out_dir> [--tex-cache <dir>]
(headless Linux: wrap with xvfb-run; needs numpy, Pillow, scikit-image in Blender's Python)
"""
from __future__ import annotations

import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from spike_pipeline import rig as rigmod, toon  # noqa: E402
from spike_pipeline.modeling import sdf, weights as W  # noqa: E402

import assemble  # noqa: E402
import body  # noqa: E402
import dev_render  # noqa: E402
import garments as G  # noqa: E402
import head  # noqa: E402
import materials_sara  # noqa: E402
import uniform_paint as UP  # noqa: E402

LOD0 = "sara_GAME_LOD0"
log = common.log


# ---------------------------------------------------------------------------
def verts_of(obj):
    co = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def faces_of(obj):
    return [tuple(p.vertices) for p in obj.data.polygons]


def segs(bones, names):
    return [(n, bones[n]["head"], bones[n]["tail"]) for n in names if n in bones]


def skin(obj, bones, names, power=4.0, relax=3, bias=None, post=None):
    P = verts_of(obj)
    names_, w = W.compute(P, segs(bones, names), power=power, bias=bias)
    if post:
        w = post(P, names_, w)
    if relax:
        w = W.relax(w, faces_of(obj), len(P), iterations=relax, factor=0.5)
    w = W.limit(w, 4)
    W.apply_to_object(obj, names_, w)


def skin_body(obj, bones, power=4.0, relax=3):
    """Part-aware skinning: membership in torso / arm / leg parts first (from the proportion master's
    distance fields), then distance-to-bone weights inside each part."""
    P = verts_of(obj)
    m = body.master(bones)
    gnames, M = m.part_weights(P)
    groups = m.part_groups()
    acc = {}
    for j, g in enumerate(gnames):
        sel = M[:, j] > 1e-3
        if not sel.any():
            continue
        bnames = [b for b in groups[g][1] if b in bones]
        names_, w = W.compute(P[sel], segs(bones, bnames), power=power)
        for k, n in enumerate(names_):
            acc.setdefault(n, np.zeros(len(P)))[sel] += w[:, k] * M[sel, j]
    names_ = list(acc)
    w = np.stack([acc[n] for n in names_], axis=1)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    if relax:
        w = W.relax(w, faces_of(obj), len(P), iterations=relax, factor=0.5)
    w = W.limit(w, 4)
    W.apply_to_object(obj, names_, w)


def bind(obj, rig):
    obj.parent = rig
    mods = list(obj.modifiers)
    arm = obj.modifiers.new("Armature", "ARMATURE")
    arm.object = rig
    # Armature must evaluate first (before the outline hull).
    idx = list(obj.modifiers).index(arm)
    if idx:
        obj.modifiers.move(idx, 0)
    return obj


def cylindrical_uv(obj, y0, z0, z1):
    me = obj.data
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        us = []
        for li in poly.loop_indices:
            p = me.vertices[me.loops[li].vertex_index].co
            us.append(UP.cyl_uv(p, y0, z0, z1))
        umax = max(u for u, _ in us)
        for li, (u, v) in zip(poly.loop_indices, us):
            if umax - u > 0.5:
                u += 1.0
            uv.data[li].uv = (u, v)


def field_object(name, field, decimate=None, smooth=None, qf=None):
    obj = common.sdf_to_object(name, field, LOD0)
    if decimate:
        common.decimate(obj, decimate)
    if smooth:
        common.smooth(obj, smooth[0], smooth[1])
    if qf:
        common.quadriflow(obj, qf)
    common.shade_smooth(obj)
    return obj


def keep_distance(P, keeps):
    """Combined signed distance of a garment's keep region (< 0 = keep)."""
    d = np.full(len(P), -1.0)
    for mode, prim in keeps:
        v = prim.eval(P)
        d = np.maximum(d, v if mode == "keep" else -v)
    return d


def cut_surface(obj, keeps, snap_iters=4):
    """Delete faces outside the keep region and snap the new boundary exactly onto the cut surface."""
    P = verts_of(obj)
    d = keep_distance(P, keeps)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    kill = [f for f in bm.faces if np.mean([d[v.index] for v in f.verts]) > 0.0]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    eye = np.eye(3) * 1e-4
    for it in range(snap_iters):
        bverts = [v for v in bm.verts if v.is_boundary]
        if not bverts:
            break
        if it > 0:  # relax along the boundary loop, then re-snap
            new = {}
            for v in bverts:
                nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
                if len(nb) == 2:
                    new[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
            for v, c in new.items():
                v.co = c
        Q = np.array([tuple(v.co) for v in bverts])
        dq = keep_distance(Q, keeps)
        g = np.stack([(keep_distance(Q + e, keeps) - keep_distance(Q - e, keeps)) / 2e-4 for e in eye], axis=1)
        gl = np.linalg.norm(g, axis=1, keepdims=True)
        g /= np.maximum(gl, 1e-9)
        step = np.clip(dq, -0.01, 0.01)[:, None]
        ok = (np.abs(dq) < 0.02) & (gl[:, 0] > 0.3)   # only vertices that really sit on this cut
        Q = np.where(ok[:, None], Q - g * step, Q)
        for v, q in zip(bverts, Q):
            v.co = q
    # Snapping can collapse sliver faces at the cut; zero-area faces give undefined normals, which the
    # thickness pass would turn into long spikes.
    bmesh.ops.dissolve_degenerate(bm, dist=2e-5, edges=bm.edges[:])
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()


def thicken(obj, thickness):
    m = obj.modifiers.new("thickness", "SOLIDIFY")
    m.thickness = thickness
    # The cut surface has outward normals: grow the shell outward (away from the skin) and flip, so the
    # original layer faces the body and the new outer layer faces out.
    m.offset = 1.0
    m.use_flip_normals = True
    m.use_rim = True
    # Plain offset: every vertex moves exactly `thickness` along its normal. Even offset divides by the
    # angle between face normals and exploded into spikes at sharp cut corners.
    m.use_even_offset = False
    m.use_quality_normals = True
    common.apply_modifiers(obj)
    common.shade_smooth(obj)


def garment(name, field, keeps, quads, thickness, pre_decimate=0.3):
    obj = common.sdf_to_object(name, field, LOD0)
    if pre_decimate:
        common.decimate(obj, pre_decimate)
    common.smooth(obj, 0.4, 3)
    common.quadriflow(obj, quads)
    cut_surface(obj, keeps)
    thicken(obj, thickness)
    return obj


def trim_garment(name, solid, keeps, grow, thickness, quads=4000, margin=0.006, shell=0.03):
    """A thin band (trim, binding, piping) lying on a garment's outer surface. Only a slab near the band
    is meshed (a closed region a few cm deep, so remeshing stays robust), then the mesh is cut exactly to
    the band, every face that is not on the outer surface is dropped, and the strip is thickened."""
    f = solid.copy()
    f.offset(grow)
    surface = f.copy()
    pts = f.points(tuple(slice(0, n) for n in f.shape)).reshape(-1, 3)
    near = keep_distance(pts, keeps).reshape(f.shape)
    f.d = np.maximum(f.d, (near - margin).astype(np.float32))
    f.d = np.maximum(f.d, -f.d - shell)
    obj = common.sdf_to_object(name, f, LOD0)
    n0 = len(obj.data.polygons)
    if n0 > quads * 4:
        common.decimate(obj, max(0.05, quads * 4 / n0))
    common.smooth(obj, 0.3, 2)
    common.quadriflow(obj, quads)
    cut_surface(obj, keeps)
    # Keep the band on the outer surface only (the slab's inner wall also lies inside the band's cut).
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    C = np.array([tuple(fc.calc_center_median()) for fc in bm.faces]) if bm.faces else np.zeros((0, 3))
    if len(C):
        off = np.abs(surface.sample_at(C)) > 0.004
        bmesh.ops.delete(bm, geom=[fc for fc, o in zip(bm.faces, off) if o], context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    thicken(obj, thickness)
    return obj


def cull_covered_skin(body_obj, jersey_solid, shorts_solid, bones, margin=0.012):
    """Delete body faces fully hidden under tight garments (standard game practice: no z-fighting,
    fewer triangles). A margin is kept at every opening so lifted hems never reveal a gap."""
    P = verts_of(body_obj)
    hem_z = body.landmarks(bones)["hem_z"]
    in_jersey = (jersey_solid.sample_at(P) < -0.004) & (keep_distance(P, G.jersey_keep(bones)) < -margin) & (P[:, 2] > hem_z)
    shorts_keeps, _ = G.shorts_keep(bones)
    in_shorts = (shorts_solid.sample_at(P) < -0.002) & (keep_distance(P, shorts_keeps) < -margin)
    hidden = in_jersey | in_shorts
    me = body_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    kill = [f for f in bm.faces if all(hidden[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(me)
    bm.free()
    log(f"culled {len(kill)} hidden skin faces under the uniform")


def strap_attribute(obj, bones, width=0.017):
    """'strap' = the light band along each armhole over the shoulder strap (front reference): the side
    panel continues up the outer half of the strap; the inner half by the neckline stays navy."""
    P = verts_of(obj)
    cuts = G.jersey_cuts(bones)
    lm = body.landmarks(bones)
    d = np.minimum(np.abs(cuts["arm_L"][0].eval(P)), np.abs(cuts["arm_R"][0].eval(P)))
    band = np.clip(1.0 - (d - width * 0.75) / (width * 0.25), 0.0, 1.0)
    band *= np.clip((P[:, 2] - (lm["shoulder_z"] - 0.075)) / 0.02, 0.0, 1.0)      # above the armpit only
    a = obj.data.attributes.new("strap", "FLOAT", "POINT")
    a.data.foreach_set("value", band.astype(np.float32))


def trim_attribute(obj, cuts, width=0.009):
    P = verts_of(obj)
    d = np.full(len(P), 1.0)
    for prim in cuts:
        d = np.minimum(d, np.abs(prim.eval(P)))
    trim = np.clip(1.0 - (d - width * 0.4) / (width * 0.6), 0, 1)
    a = obj.data.attributes.new("trim", "FLOAT", "POINT")
    a.data.foreach_set("value", trim.astype(np.float32))


# ---------------------------------------------------------------------------
def main(out_dir: Path, tex_cache: Path | None):
    t0 = time.time()
    tex = out_dir / "textures"
    tex.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    dev_render.setup(samples=12)
    bones, character = common.bones_dict()
    pal = dict(character["palette"])
    rig = rigmod.build_rig("sara")
    rig.hide_render = True
    primary = [n for n in bones if bones[n]["kind"] in ("primary", "twist") and bones[n]["deform"]]
    body_bones = [n for n in primary if n not in ("Jaw",) and not any(f in n for f in
                  ("Thumb", "Index", "Middle", "Ring", "Little"))]

    # --- body + hands -------------------------------------------------------------------
    body_obj = field_object("sara_body_LOD0", body.body_field(bones), decimate=0.35, smooth=(0.5, 4), qf=15000)
    hands = {}
    for side in ("Left", "Right"):
        hands[side] = field_object(f"sara_hand_{side[0]}_LOD0", body.hand_field(bones, side), decimate=0.4, smooth=(0.5, 2),
                                   qf=2800)
    # --- head -----------------------------------------------------------------------------
    face_obj, prof, grid = head.build_head_mesh(LOD0)
    decals = head.build_eye_decals(face_obj, prof, LOD0)
    ears = head.build_ears(LOD0)
    # --- hair -----------------------------------------------------------------------------
    hair_obj, ribbon_obj, hair_arrays, ribbon_arrays, info = assemble.build_hair(bones, LOD0)
    cap_obj = info["cap_obj"]
    bpy.ops.mesh.primitive_torus_add(major_radius=0.016, minor_radius=0.0055, major_segments=32, minor_segments=10,
                                     location=tuple(info["tie"]))
    tie_obj = bpy.context.active_object
    tie_obj.name = "sara_hairtie_LOD0"
    ax = Vector(tuple(info["tie_axis"]))
    tie_obj.rotation_mode = "QUATERNION"
    tie_obj.rotation_quaternion = ax.to_track_quat("Z", "Y")
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    common.link(tie_obj, LOD0)
    clip_objs = []
    for part in info["clip"]:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.35, segments=3, affect="EDGES", profile=0.5)
        ax = [Vector(a) for a in part["axes"]]
        R = Matrix((ax[0], ax[1], ax[2])).transposed().to_4x4()
        S = Matrix.Diagonal((*part["half"], 1.0))
        bmesh.ops.transform(bm, matrix=Matrix.Translation(Vector(part["center"])) @ R @ S, verts=bm.verts)
        me = bpy.data.meshes.new(f"sara_hairclip_{part['name']}_LOD0")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(o)
        common.link(o, LOD0)
        o["clip_color"] = part["color"]
        clip_objs.append(o)
    # --- garments ---------------------------------------------------------------------------
    jersey_solid = G.jersey_field(bones)
    jersey = garment("sara_jersey_LOD0", jersey_solid, G.jersey_keep(bones), 11000, 0.0022)
    # The jersey's real garment edges and seams: V-neck trim + piping, armhole bindings, side-seam piping.
    trims = []
    for tname, tkeeps, grow, tthick, tmat in G.jersey_trims(bones, 0.0022):
        o = trim_garment(f"sara_jersey_{tname}_LOD0", jersey_solid, tkeeps, grow, tthick,
                         quads=5000 if "neck" in tname else 3500)
        o["trim_mat"] = tmat
        trims.append(o)
    shorts_solid = G.shorts_field(bones)
    shorts_keeps, _ = G.shorts_keep(bones)
    shorts = garment("sara_shorts_LOD0", shorts_solid, shorts_keeps, 5000, 0.002)
    cull_covered_skin(body_obj, jersey_solid, shorts_solid, bones)
    pads, bands, socks, plates = {}, [], {}, {}
    for side in ("Left", "Right"):
        f, keeps = G.kneepad_field(bones, side)
        pads[side] = garment(f"sara_kneepad_{side[0]}_LOD0", f, keeps, 1800, 0.0024, pre_decimate=None)
        f, keeps = G.kneepad_plate(bones, side)
        plates[side] = garment(f"sara_kneeplate_{side[0]}_LOD0", f, keeps, 900, 0.0015, pre_decimate=None)
        for i, (bf, bkeeps) in enumerate(G.kneepad_bands(bones, side)):
            bands.append((side, garment(f"sara_kneeband_{side[0]}{i}_LOD0", bf, bkeeps, 900, 0.0018, pre_decimate=None)))
    f, keeps = G.sock_field(bones, "Left", high=True)
    socks["Left"] = garment("sara_sock_L_LOD0", f, keeps, 2600, 0.002, pre_decimate=None)
    f, keeps = G.sock_field(bones, "Right", high=False)
    socks["Right"] = garment("sara_sock_R_LOD0", f, keeps, 900, 0.002, pre_decimate=None)
    shoes = {}
    for side in ("Left", "Right"):
        parts = []
        for pname, f, color, qf in G.shoe_parts(bones, side):
            o = field_object(f"sara_shoe_{side[0]}_{pname}_LOD0", f, decimate=0.3 if qf else 0.5, smooth=(0.3, 2), qf=qf)
            o["shoe_color"] = color
            parts.append(o)
        shoes[side] = parts
    log(f"geometry built in {time.time() - t0:.0f}s")

    # --- textures -----------------------------------------------------------------------------
    names = {"face": "T_sara_face_base.png", "whiteL": "T_sara_eye_white_L.png", "whiteR": "T_sara_eye_white_R.png",
             "irisL": "T_sara_iris_L.png", "irisR": "T_sara_iris_R.png", "lashL": "T_sara_lash_L.png",
             "lashR": "T_sara_lash_R.png", "jersey": "T_sara_jersey.png",
             "shorts": "T_sara_shorts.png"}
    eye_pal = dict(character["face"]["eye_shader"])
    for k in ("iris_top", "iris_mid", "iris_bottom", "iris_ring", "iris_cog", "pupil"):
        eye_pal[k] = pal[k]
    UP.configure(bones)
    painters = {
        "face": lambda p: head.paint_face(p),
        "whiteL": lambda p: head.paint_eye_white(p, open_k=head.EYE_OPEN["L"]),
        "whiteR": lambda p: head.paint_eye_white(p, open_k=head.EYE_OPEN["R"]),
        "irisL": lambda p: head.paint_iris(p, palette=eye_pal),
        "irisR": lambda p: head.paint_iris(p, palette=eye_pal, mirror_highlights=True),
        "lashL": lambda p: head.paint_lash(p, open_k=head.EYE_OPEN["L"]),
        "lashR": lambda p: head.paint_lash(p, open_k=head.EYE_OPEN["R"]),
        "jersey": lambda p: UP.paint_jersey(p, pal),
        "shorts": lambda p: UP.paint_shorts(p, pal),
    }
    for key, fname in names.items():
        if tex_cache and (tex_cache / fname).exists():
            shutil.copy(tex_cache / fname, tex / fname)
        else:
            painters[key](tex / fname)

    # --- UVs / attributes -----------------------------------------------------------------------
    cylindrical_uv(jersey, UP.JERSEY_Y0, UP.JERSEY_Z0, UP.JERSEY_Z1)
    cylindrical_uv(shorts, UP.SHORTS_Y0, UP.SHORTS_Z0, UP.SHORTS_Z1)
    cuts = G.jersey_cuts(bones)
    trim_attribute(jersey, cuts["neck"] + cuts["arm_L"] + cuts["arm_R"] + cuts["hem"], width=0.004)
    strap_attribute(jersey, bones)

    # --- materials -------------------------------------------------------------------------------
    M = lambda n, b, s=None, **kw: toon.toon_material(n, b, s, pal, **kw)
    mats = {
        "skin": M("M_sara_skin", "@skin_base", "@skin_shadow", threshold=0.3, softness=0.05),
        "face": toon.toon_textured("M_sara_face", tex / names["face"], shade_mul=(0.9, 0.74, 0.8), threshold=0.17,
                                   softness=0.07, mask_attribute="front", fallback="#fbe6e8"),
        "hair": materials_sara.hair_material(pal),
        "ribbon": materials_sara.satin_material(pal),
        "tie": M("M_sara_hairtie", "@hair_tie", "#18202e"),
        "jersey": toon.toon_textured("M_sara_jersey", tex / names["jersey"], shade_mul=(0.66, 0.7, 0.86), threshold=0.3,
                                     softness=0.04, overlay_attribute="trim", overlay_color="#22325a",
                                     overlays=[("strap", "#e6eef6")]),
        "shorts": toon.toon_textured("M_sara_shorts", tex / names["shorts"], shade_mul=(0.62, 0.66, 0.84), threshold=0.3,
                                     softness=0.04),
        "trim_white": M("M_sara_jersey_trim", "@jersey_side", "#b8c3d6", threshold=0.3, softness=0.04),
        "piping": M("M_sara_jersey_piping", "@jersey_piping", "#12214a", threshold=0.3, softness=0.04),
        "pad": M("M_sara_kneepad", "@kneepad_sleeve", "#8ea6c4", threshold=0.28, softness=0.06),
        "plate": M("M_sara_kneeplate", "@kneepad_pad", "#b7c3d6", threshold=0.28, softness=0.06),
        "band": M("M_sara_kneeband", "@kneepad_band", "#4f86b8"),
        "sock": M("M_sara_sock", "@sock_black", "#0b0b10", rim_mask=1.5),
        "ankle": M("M_sara_anklesock", "@ankle_sock", "#a4aabb"),
        "upper": M("M_sara_shoe_upper", "@shoe_upper", "#868c9f", threshold=0.3),
        "midsole": M("M_sara_shoe_sole", "@shoe_sole", "#b6bac6"),
        "black": M("M_sara_shoe_trim", "@shoe_accent", "#050507", rim_mask=0.6),
    }
    outline = toon.outline_material("M_sara_outline", "#3a2530")
    hair_outline = toon.outline_material("M_sara_hair_outline", "#2a1f26")
    cloth_outline = toon.outline_material("M_sara_cloth_outline", "#16192a")

    def dress(obj, mat, ol=outline, width=0.002):
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if ol is not None:
            toon.add_outline(obj, ol, width)

    dress(body_obj, mats["skin"], width=0.0022)
    for h in hands.values():
        dress(h, mats["skin"], width=0.0012)
    dress(face_obj, mats["face"], width=0.0018)
    for e in ears:
        dress(e, mats["skin"], width=0.0012)
    for (layer, side), obj in decals.items():
        m = (toon.anime_eye_material(f"M_sara_eye_{side}", tex / names[f"white{side}"], tex / names[f"iris{side}"])
             if layer == "eye" else toon.anime_lash_material(f"M_sara_lash_{side}", tex / names[f"lash{side}"]))
        m.node_tree.nodes["lid_heavy"].outputs[0].default_value = 0.2
        dress(obj, m, ol=None)
    dress(hair_obj, mats["hair"], hair_outline, 0.0008)
    dress(cap_obj, mats["hair"], None)
    dress(ribbon_obj, mats["ribbon"], toon.outline_material("M_sara_ribbon_outline", "#3a5f96"), 0.0006)
    dress(tie_obj, mats["tie"], hair_outline, 0.001)
    for o in clip_objs:
        dress(o, M(f"M_sara_clip_{o.name.split('_')[2]}", o["clip_color"], None, threshold=0.25, rim_mask=0.6), hair_outline, 0.0007)
    dress(jersey, mats["jersey"], cloth_outline, 0.0018)
    for o in trims:
        dress(o, mats[o["trim_mat"]], cloth_outline, 0.0005)
    dress(shorts, mats["shorts"], cloth_outline, 0.0016)
    for side in pads:
        dress(pads[side], mats["pad"], cloth_outline, 0.0012)
        dress(plates[side], mats["plate"], cloth_outline, 0.0006)
    for side, b in bands:
        dress(b, mats["band"], cloth_outline, 0.001)
    dress(socks["Left"], mats["sock"], cloth_outline, 0.0014)
    dress(socks["Right"], mats["ankle"], cloth_outline, 0.001)
    shoe_shade = {"@shoe_upper": "#a9aebd", "@shoe_sole": "#c3c6d0", "@shoe_outsole": "#101014", "@shoe_accent": "#050507",
                  "@shoe_tongue": "#9ea3b3", "@shoe_lace": "#0b0b0e"}
    for side, parts in shoes.items():
        for o in parts:
            key = o["shoe_color"]
            mat = bpy.data.materials.get(f"M_sara_{key[1:]}") or M(f"M_sara_{key[1:]}", key, shoe_shade[key], threshold=0.3,
                                                                    softness=0.04, rim_mask=0.8)
            dress(o, mat, cloth_outline, 0.0009 if key in ("@shoe_upper", "@shoe_sole") else 0.0005)

    # --- weights + binding ----------------------------------------------------------------------
    t1 = time.time()
    skin_body(body_obj, bones)
    for side, h in hands.items():
        fingers = [n for n in bones if n.startswith(side) and any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
        skin(h, bones, [f"{side}Hand", f"{side}LowerArmTwist", f"{side}LowerArm"] + fingers, power=6.0, relax=1)
    for obj in [face_obj, cap_obj, tie_obj] + clip_objs + list(ears) + list(decals.values()):
        W.apply_to_object(obj, ["Head"], np.ones((len(obj.data.vertices), 1)))
    assemble.hair_weights(hair_obj, hair_arrays, bones, info["free"]["hair"])
    assemble.hair_weights(ribbon_obj, ribbon_arrays, bones, info["free"]["ribbon"])
    hem_chains = [n for n in bones if n.startswith("cloth_jersey")]
    hem_z = body.landmarks(bones)["hem_z"]

    def hem_fade(P, names_, w):
        fade = np.clip((hem_z + 0.07 - P[:, 2]) / 0.08, 0.0, 1.0)   # hem chains act only near the hem
        for j, n in enumerate(names_):
            if n.startswith("cloth_"):
                w[:, j] *= fade
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)

    skin(jersey, bones, ["Hips", "Spine", "Chest", "UpperChest", "Neck", "LeftShoulder", "RightShoulder",
                         "LeftUpperArm", "RightUpperArm"] + hem_chains,
         bias={"LeftUpperArm": 1.8, "RightUpperArm": 1.8}, post=hem_fade)
    for o in trims:                    # trims ride with the jersey
        skin(o, bones, ["Hips", "Spine", "Chest", "UpperChest", "Neck", "LeftShoulder", "RightShoulder",
                        "LeftUpperArm", "RightUpperArm"] + hem_chains,
             bias={"LeftUpperArm": 1.8, "RightUpperArm": 1.8}, post=hem_fade)
    skin(shorts, bones, ["Hips", "Spine", "LeftUpperLeg", "RightUpperLeg", "LeftUpperLegTwist", "RightUpperLegTwist"])
    for side in ("Left", "Right"):
        skin(pads[side], bones, [f"{side}UpperLeg", f"{side}LowerLeg"], relax=4)
        skin(plates[side], bones, [f"{side}UpperLeg", f"{side}LowerLeg"], relax=4)
    for side, b in bands:
        skin(b, bones, [f"{side}UpperLeg", f"{side}LowerLeg"], relax=2)
    for side in ("Left", "Right"):
        skin(socks[side], bones, [f"{side}LowerLeg", f"{side}Foot"], relax=2)
        for o in shoes[side]:
            skin(o, bones, [f"{side}Foot", f"{side}Toes"], power=6.0, relax=2)
    log(f"weights in {time.time() - t1:.0f}s")
    for obj in bpy.data.collections[LOD0].objects:
        if obj.type == "MESH":
            bind(obj, rig)
    rig["spike_model"] = "sara v0.2 procedural master"
    out_dir.mkdir(parents=True, exist_ok=True)
    blend = out_dir / "sara_v02.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    tri = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.collections[LOD0].objects if o.type == "MESH")
    log(f"saved {blend}  ({tri} triangles before outline hulls, {time.time() - t0:.0f}s total)")
    return blend


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = Path(argv[0])
    cache = Path(argv[argv.index("--tex-cache") + 1]) if "--tex-cache" in argv else None
    main(out, cache)
