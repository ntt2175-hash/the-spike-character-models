"""Scene conventions shared by every Blender tool.

Collections
    {char}_MASTER          high-resolution sculpts, source meshes, bake cages. Never exported.
    {char}_GAME            everything that ships
      {char}_rig           the armature (object lives directly in {char}_GAME)
      {char}_LOD0..2       game meshes, one collection per LOD

Objects
    {char}_{part}_LOD{n}   game mesh, part is lowercase (body, face, eyes, hair, jersey ...)
Materials
    M_{char}_{slot}[_{variant}]   slot from pipeline/specs/materials.json
"""
from __future__ import annotations

import re

import bpy

import spike_specs as ss

LODS = ("LOD0", "LOD1", "LOD2")
_MESH_RE = re.compile(r"^(?P<char>[a-z0-9]+)_(?P<part>[a-z][a-z0-9_]*)_(?P<lod>LOD[0-2])$")


def master_collection_name(char: str) -> str:
    return f"{char}_MASTER"


def game_collection_name(char: str) -> str:
    return f"{char}_GAME"


def lod_collection_name(char: str, lod: str) -> str:
    return f"{char}_{lod}"


def rig_name(char: str) -> str:
    return f"{char}_rig"


def parse_mesh_name(name: str):
    """'sara_hair_LOD0' -> ('sara', 'hair', 'LOD0') or None."""
    m = _MESH_RE.match(name)
    return (m["char"], m["part"], m["lod"]) if m else None


def parse_material_name(char: str, name: str, slots) -> str | None:
    """Return the slot for 'M_{char}_{slot}[_{variant}]', longest slot wins."""
    prefix = f"M_{char}_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    for slot in sorted(slots, key=len, reverse=True):
        if rest == slot or rest.startswith(slot + "_"):
            return slot
    return None


def ensure_collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if coll.name not in parent.children:
        parent.children.link(coll)
    return coll


def ensure_scene_layout(char: str) -> dict:
    """Create the standard collection layout (idempotent)."""
    master = ensure_collection(master_collection_name(char))
    game = ensure_collection(game_collection_name(char))
    lods = {lod: ensure_collection(lod_collection_name(char, lod), game) for lod in LODS}
    return {"master": master, "game": game, "lods": lods}


def game_meshes(char: str) -> dict[str, list[bpy.types.Object]]:
    """Mesh objects per LOD collection."""
    out = {}
    for lod in LODS:
        coll = bpy.data.collections.get(lod_collection_name(char, lod))
        out[lod] = [o for o in coll.all_objects if o.type == "MESH"] if coll else []
    return out


def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def set_active(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def load_context(char: str) -> dict:
    """All specs + the character, loaded once per tool run."""
    specs = ss.load_all_specs()
    return {"specs": specs, "character": ss.load_character(char), "char": char}
