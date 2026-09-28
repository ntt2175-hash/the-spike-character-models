"""Turn recipe output (arrays) into Blender objects with attributes, highlights and weights."""
from __future__ import annotations

import bmesh
import bpy
import numpy as np

from spike_pipeline.modeling import lock as L
from spike_pipeline.modeling import weights as W

import hair as hairmod
from common import link, log


def mesh_from_arrays(name, verts, faces, coll, attrs=None, smooth=True):
    bm = bmesh.new()
    bv = [bm.verts.new(v) for v in verts]
    for f in faces:
        try:
            bm.faces.new([bv[i] for i in f])
        except ValueError:
            pass  # duplicate face at a degenerate tip
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = smooth
    for key, values in (attrs or {}).items():
        a = me.attributes.new(key, "FLOAT", "POINT")
        a.data.foreach_set("value", np.asarray(values, dtype=np.float32))
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    link(obj, coll)
    return obj


def angel_ring(verts, normals, lock_id, along, tags, seed=4):
    """Stylized angel ring: a band around the crown, lower at the front, with a jagged lower edge and a
    few breaks where locks separate; plus a band high on the ponytail. Outward-facing surfaces only."""
    rng = np.random.default_rng(seed)
    c = np.array([0.0, 0.012, 1.55])
    radial = verts - c
    radial[:, 2] *= 0.6
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    facing = np.clip((normals * radial).sum(axis=1), 0, 1)
    ang = np.arctan2(verts[:, 0], -(verts[:, 1] - c[1]))
    z_c = 1.629 + 0.012 * (1.0 - np.cos(ang)) * 0.5
    jag = 0.0035 * np.sin(ang * 23.0) + 0.0022 * np.sin(ang * 41.0 + 1.3)
    dz = verts[:, 2] - z_c
    band = np.where(dz > 0, np.exp(-(dz ** 2) / (2 * 0.0048 ** 2)), np.exp(-((dz - jag) ** 2) / (2 * 0.0036 ** 2)))
    breaks = (np.cos(ang * 9.0 + 0.7) > -0.82).astype(float)
    is_head = np.array([tags[i] in ("top", "bang_A", "bang_B", "bang_C") for i in lock_id])
    hl = band * facing ** 1.5 * breaks * is_head
    is_tail = np.array([tags[i].startswith("ponytail") for i in lock_id])
    tail_band = np.exp(-((along - 0.15) ** 2) / (2 * 0.03 ** 2))
    hl += tail_band * facing ** 2 * is_tail * (rng.uniform(0, 1, len(verts)) > 0.25)
    return np.clip(hl * 1.6, 0, 1)


def build_hair(bones, coll="sara_GAME_LOD0"):
    hb, rb = L.LockBatch(), L.LockBatch()
    info = hairmod.build(bones, hb, rb)
    hv, hf, ha, hid = hb.arrays()
    hair = mesh_from_arrays("sara_hair_LOD0", hv, hf, coll, {"along": ha})
    hair.data.update()
    normals = np.array([v.normal for v in hair.data.vertices])
    hl = angel_ring(hv, normals, hid, ha, hb.tags)
    a = hair.data.attributes.new("hl", "FLOAT", "POINT")
    a.data.foreach_set("value", hl.astype(np.float32))
    cv, cf = info["cap"]
    cap_obj = mesh_from_arrays("sara_hair_cap_LOD0", cv, cf, coll, {"along": np.zeros(len(cv)), "hl": np.zeros(len(cv))})
    info["cap_obj"] = cap_obj
    rv, rf, ra, rid = rb.arrays()
    ribbon = mesh_from_arrays("sara_ribbon_LOD0", rv, rf, coll, {"along": ra})
    log(f"hair: {len(hb.tags)} locks, {len(hair.data.polygons)} faces; ribbon {len(ribbon.data.polygons)} faces")
    return hair, ribbon, (hv, hf, ha, hid, hb.tags), (rv, rf, ra, rid, rb.tags), info


def hair_weights(obj, arrays, bones):
    """Each lock follows its own chain (or the Head), blended along the chain."""
    verts, faces, along, lock_id, tags = arrays
    names_all = []
    Wall = np.zeros((len(verts), 0))
    chain_for = {
        "ponytail_A": "hair_ponytail_A", "ponytail_B": "hair_ponytail_B", "ponytail_C": "hair_ponytail_C",
        "bang_A": "hair_bang_A", "bang_B": "hair_bang_B", "bang_C": "hair_bang_C",
        "side_L": "hair_side_A", "side_R": "hair_side_A", "ahoge_A": "hair_ahoge_A",
        "flyaway_A": "hair_flyaway_A", "flyaway_B": "hair_flyaway_B", "flyaway_C": "hair_flyaway_C",
        "ribbon_loop_A": "ribbon_loop_A", "ribbon_tail_A": "ribbon_tail_A",
    }
    groups = {}
    for lid, tag in enumerate(tags):
        idx = np.nonzero(lock_id == lid)[0]
        chain = chain_for.get(tag)
        segs = [("Head", bones["Head"]["head"], bones["Head"]["tail"])]
        if chain:
            suffix = "_L" if tag == "side_L" else ("_R" if tag == "side_R" else "")
            cnames = sorted(n for n in bones if n.startswith(chain + "_") and n.endswith(suffix) and
                            (suffix or not n.endswith(("_L", "_R"))))
            segs += [(n, bones[n]["head"], bones[n]["tail"]) for n in cnames]
        names, w = W.compute(verts[idx], segs, power=3.0)
        if chain:
            # Roots stay on the head: blend by 'along' so the scalp end never swings.
            root_k = np.clip(along[idx] / 0.18, 0, 1)[:, None] if not tag.startswith("ponytail") else 1.0
            w = w * root_k
            w[:, 0] += 1.0 - w.sum(axis=1)
        for j, n in enumerate(names):
            groups.setdefault(n, np.zeros(len(verts)))[idx] += w[:, j]
    names = list(groups)
    Wm = np.stack([groups[n] for n in names], axis=1)
    Wm = W.limit(Wm, 4)
    W.apply_to_object(obj, names, Wm)
