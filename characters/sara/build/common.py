"""Shared helpers for Sara's build stages (mesh cleanup, render checks)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for p in (REPO / "pipeline", REPO / "pipeline" / "blender", HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bpy  # noqa: E402

import spike_specs as ss  # noqa: E402

CHAR = "sara"


def log(*a):
    print(f"[sara-build {time.strftime('%H:%M:%S')}]", *a, flush=True)


def bones_dict():
    sk = ss.load_spec("skeleton")
    c = ss.load_character(CHAR)
    return {b["name"]: b for b in ss.build_character_skeleton(sk, c)}, c


def apply_modifiers(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return obj


def link(obj, coll_name):
    coll = bpy.data.collections.get(coll_name)
    if coll is None:
        coll = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(coll)
    for c in obj.users_collection:
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


def new_mesh_object(name, me, coll_name):
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return link(obj, coll_name)


def smooth(obj, factor=0.5, repeat=5):
    m = obj.modifiers.new("smooth", "SMOOTH")
    m.factor = factor
    m.iterations = repeat
    return apply_modifiers(obj)


def decimate(obj, ratio):
    m = obj.modifiers.new("decimate", "DECIMATE")
    m.ratio = ratio
    return apply_modifiers(obj)


def quadriflow(obj, faces):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    t = time.time()
    res = bpy.ops.object.quadriflow_remesh(target_faces=faces, use_mesh_symmetry=False, use_preserve_sharp=False,
                                           use_preserve_boundary=False, smooth_normals=False, seed=3)
    log(f"quadriflow {obj.name}: {res} -> {len(obj.data.polygons)} faces in {time.time() - t:.1f}s")
    return obj


def shade_smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def sdf_to_object(name, field, coll_name, level=0.0):
    from spike_pipeline.modeling import sdf

    t = time.time()
    verts, faces = field.mesh(level)
    me = sdf.to_blender_mesh(name, verts, faces)
    log(f"marching cubes {name}: {len(verts)} verts, {len(faces)} tris ({time.time() - t:.1f}s)")
    obj = new_mesh_object(name, me, coll_name)
    shade_smooth(obj)
    return obj
