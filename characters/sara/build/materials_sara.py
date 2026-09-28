"""Sara's Blender preview materials (EEVEE toon). Godot uses the shaders in addons/spike_character."""
from __future__ import annotations

import bpy

from spike_pipeline import toon
from spike_pipeline.toon import _math, _node, _rgba


def hair_material(pal, name="M_sara_hair"):
    """Toon hair: base -> tip gradient along the lock, painted-in angel ring from the 'hl' attribute."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 700, 0)
    grp = _node(nt, "ShaderNodeGroup", 450, 0)
    grp.node_tree = toon.toon_group()
    along = _node(nt, "ShaderNodeAttribute", -700, 100, attribute_type="GEOMETRY", attribute_name="along")
    hl = _node(nt, "ShaderNodeAttribute", -700, -150, attribute_type="GEOMETRY", attribute_name="hl")
    tipk = _node(nt, "ShaderNodeMapRange", -500, 100, interpolation_type="SMOOTHSTEP")
    tipk.inputs["From Min"].default_value = 0.62
    tipk.inputs["From Max"].default_value = 1.0
    nt.links.new(along.outputs["Fac"], tipk.inputs["Value"])
    base = _node(nt, "ShaderNodeMix", -300, 150, data_type="RGBA")
    base.inputs["A"].default_value = _rgba(pal["hair_base"])
    base.inputs["B"].default_value = _rgba("#7d7686")
    nt.links.new(tipk.outputs[0], base.inputs["Factor"])
    shine = _node(nt, "ShaderNodeMix", -100, 150, data_type="RGBA")
    shine.inputs["B"].default_value = _rgba(pal["hair_highlight"])
    nt.links.new(hl.outputs["Fac"], shine.inputs["Factor"])
    nt.links.new(base.outputs["Result"], shine.inputs["A"])
    shade = _node(nt, "ShaderNodeMix", -100, -100, data_type="RGBA")
    shade.inputs["A"].default_value = _rgba("#322c3a")
    shade.inputs["B"].default_value = _rgba("#4a4456")
    nt.links.new(tipk.outputs[0], shade.inputs["Factor"])
    nt.links.new(shine.outputs["Result"], grp.inputs["Base"])
    nt.links.new(shade.outputs["Result"], grp.inputs["Shade"])
    grp.inputs["Threshold"].default_value = 0.34
    grp.inputs["Softness"].default_value = 0.09
    grp.inputs["Rim Mask"].default_value = 1.4
    nt.links.new(grp.outputs[0], out.inputs["Surface"])
    return mat


def satin_material(pal, name="M_sara_ribbon"):
    """Soft satin: pale sky-blue with a gentle sheen toward grazing angles and a soft ramp."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 700, 0)
    grp = _node(nt, "ShaderNodeGroup", 450, 0)
    grp.node_tree = toon.toon_group()
    lw = _node(nt, "ShaderNodeLayerWeight", -600, 0)
    lw.inputs["Blend"].default_value = 0.35
    sheen = _node(nt, "ShaderNodeMix", -300, 100, data_type="RGBA")
    sheen.inputs["A"].default_value = _rgba("#86bdf0")
    sheen.inputs["B"].default_value = _rgba("#d9eeff")
    nt.links.new(lw.outputs["Facing"], sheen.inputs["Factor"])
    nt.links.new(sheen.outputs["Result"], grp.inputs["Base"])
    grp.inputs["Shade"].default_value = _rgba("#5f93d2")
    grp.inputs["Threshold"].default_value = 0.2
    grp.inputs["Softness"].default_value = 0.12
    grp.inputs["Rim Mask"].default_value = 0.8
    nt.links.new(grp.outputs[0], out.inputs["Surface"])
    mat.use_backface_culling = False
    return mat


def shoe_material(pal, name="M_sara_shoe"):
    """One material, zones from smooth vertex masks (shoe_white, shoe_black) so stripes stay clean."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 700, 0)
    grp = _node(nt, "ShaderNodeGroup", 450, 0)
    grp.node_tree = toon.toon_group()
    wm = _node(nt, "ShaderNodeAttribute", -700, 100, attribute_type="GEOMETRY", attribute_name="shoe_white")
    bm_ = _node(nt, "ShaderNodeAttribute", -700, -150, attribute_type="GEOMETRY", attribute_name="shoe_black")
    for label, y, cols in (("base", 150, (pal["shoe_upper"], pal["shoe_sole"], pal["shoe_accent"])),
                           ("shade", -100, ("#868c9f", "#b6bac6", "#050507"))):
        m1 = _node(nt, "ShaderNodeMix", -400, y, data_type="RGBA")
        m1.inputs["A"].default_value = _rgba(cols[0])
        m1.inputs["B"].default_value = _rgba(cols[1])
        nt.links.new(wm.outputs["Fac"], m1.inputs["Factor"])
        m2 = _node(nt, "ShaderNodeMix", -150, y, data_type="RGBA")
        m2.inputs["B"].default_value = _rgba(cols[2])
        nt.links.new(bm_.outputs["Fac"], m2.inputs["Factor"])
        nt.links.new(m1.outputs["Result"], m2.inputs["A"])
        nt.links.new(m2.outputs["Result"], grp.inputs["Base" if label == "base" else "Shade"])
    grp.inputs["Threshold"].default_value = 0.3
    grp.inputs["Softness"].default_value = 0.04
    grp.inputs["Rim Mask"].default_value = 0.8
    nt.links.new(grp.outputs[0], out.inputs["Surface"])
    return mat
