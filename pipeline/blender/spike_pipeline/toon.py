"""EEVEE toon preview materials for Blender renders (previs, QC beauty passes).

This is a Blender-side approximation of the Godot shaders so previs and QC
renders read like the game: a lit/shade two-tone ramp, a hand-picked shade
color (never a darkened base), a fresnel rim, and an inverted-hull outline.

Lighting context comes from SCENE custom properties so a single keyframe on
the scene relights every character material at once:
    spike_char_light   overall character light level (1.0 = key on, ~0 = blackout)
    spike_fill_color   RGB tint pushed into the shade color (lighting profile fill)
    spike_fill_mix     0..1 how much of the fill tint enters the shade
    spike_rim_color    RGB rim color
    spike_rim_strength rim intensity
    spike_rim_width    0..1 rim width
"""
from __future__ import annotations

import bpy

import spike_specs as ss

GROUP_NAME = "SpikeToon"

SCENE_DEFAULTS = {
    "spike_char_light": 1.0,
    "spike_fill_color": (0.75, 0.8, 0.9),
    "spike_fill_mix": 0.25,
    "spike_rim_color": (1.0, 1.0, 1.0),
    "spike_rim_strength": 0.4,
    "spike_rim_width": 0.3,
}


def ensure_scene_props(scene=None):
    scene = scene or bpy.context.scene
    for k, v in SCENE_DEFAULTS.items():
        if k not in scene:
            scene[k] = v
    return scene


def _socket(group, in_out, name, socket_type, default=None):
    sock = group.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
    if default is not None and hasattr(sock, "default_value"):
        sock.default_value = default
    return sock


def _attr(nodes, name, x, y):
    n = nodes.new("ShaderNodeAttribute")
    n.attribute_type = "VIEW_LAYER"
    n.attribute_name = name
    n.location = (x, y)
    return n


def toon_group() -> bpy.types.ShaderNodeTree:
    grp = bpy.data.node_groups.get(GROUP_NAME)
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(GROUP_NAME, "ShaderNodeTree")
    _socket(grp, "INPUT", "Base", "NodeSocketColor", (0.8, 0.8, 0.8, 1.0))
    _socket(grp, "INPUT", "Shade", "NodeSocketColor", (0.5, 0.5, 0.6, 1.0))
    _socket(grp, "INPUT", "Threshold", "NodeSocketFloat", 0.32)
    _socket(grp, "INPUT", "Softness", "NodeSocketFloat", 0.04)
    _socket(grp, "INPUT", "Rim Mask", "NodeSocketFloat", 1.0)
    _socket(grp, "INPUT", "Emission", "NodeSocketFloat", 0.0)
    _socket(grp, "OUTPUT", "Shader", "NodeSocketShader")
    n, l = grp.nodes, grp.links
    gi = n.new("NodeGroupInput"); gi.location = (-1200, 0)
    go = n.new("NodeGroupOutput"); go.location = (900, 0)

    diffuse = n.new("ShaderNodeBsdfDiffuse"); diffuse.location = (-1000, 250)
    to_rgb = n.new("ShaderNodeShaderToRGB"); to_rgb.location = (-800, 250)
    bw = n.new("ShaderNodeRGBToBW"); bw.location = (-620, 250)
    l.new(diffuse.outputs[0], to_rgb.inputs[0])
    l.new(to_rgb.outputs["Color"], bw.inputs[0])

    # Stepped ramp: smoothstep(threshold - soft, threshold + soft, luminance).
    lo = n.new("ShaderNodeMath"); lo.operation = "SUBTRACT"; lo.location = (-620, 60)
    hi = n.new("ShaderNodeMath"); hi.operation = "ADD"; hi.location = (-620, -80)
    l.new(gi.outputs["Threshold"], lo.inputs[0]); l.new(gi.outputs["Softness"], lo.inputs[1])
    l.new(gi.outputs["Threshold"], hi.inputs[0]); l.new(gi.outputs["Softness"], hi.inputs[1])
    ramp = n.new("ShaderNodeMapRange"); ramp.location = (-420, 200)
    ramp.interpolation_type = "SMOOTHSTEP"
    l.new(bw.outputs[0], ramp.inputs["Value"])
    l.new(lo.outputs[0], ramp.inputs["From Min"])
    l.new(hi.outputs[0], ramp.inputs["From Max"])

    fill = _attr(n, "spike_fill_color", -620, -260)
    fill_mix = _attr(n, "spike_fill_mix", -620, -420)
    shade_tint = n.new("ShaderNodeMix"); shade_tint.data_type = "RGBA"; shade_tint.blend_type = "MULTIPLY"
    shade_tint.location = (-420, -200)
    l.new(fill_mix.outputs["Fac"], shade_tint.inputs["Factor"])
    l.new(gi.outputs["Shade"], shade_tint.inputs["A"])
    l.new(fill.outputs["Color"], shade_tint.inputs["B"])

    two_tone = n.new("ShaderNodeMix"); two_tone.data_type = "RGBA"; two_tone.location = (-200, 100)
    l.new(ramp.outputs[0], two_tone.inputs["Factor"])
    l.new(shade_tint.outputs["Result"], two_tone.inputs["A"])
    l.new(gi.outputs["Base"], two_tone.inputs["B"])

    light = _attr(n, "spike_char_light", -200, -120)
    lit = n.new("ShaderNodeVectorMath"); lit.operation = "SCALE"; lit.location = (0, 60)
    l.new(two_tone.outputs["Result"], lit.inputs[0])
    l.new(light.outputs["Fac"], lit.inputs["Scale"])

    # Rim: fresnel facing term, width-controlled, masked, tinted.
    lw = n.new("ShaderNodeLayerWeight"); lw.location = (-420, -560)
    lw.inputs["Blend"].default_value = 0.5
    width = _attr(n, "spike_rim_width", -620, -620)
    inv = n.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.location = (-420, -720)
    inv.inputs[0].default_value = 1.0
    l.new(width.outputs["Fac"], inv.inputs[1])
    rim_ramp = n.new("ShaderNodeMapRange"); rim_ramp.location = (-200, -560)
    rim_ramp.interpolation_type = "SMOOTHSTEP"
    l.new(lw.outputs["Facing"], rim_ramp.inputs["Value"])
    l.new(inv.outputs[0], rim_ramp.inputs["From Min"])
    rim_ramp.inputs["From Max"].default_value = 1.0
    rim_str = _attr(n, "spike_rim_strength", -200, -760)
    rim_col = _attr(n, "spike_rim_color", -200, -900)
    m1 = n.new("ShaderNodeMath"); m1.operation = "MULTIPLY"; m1.location = (0, -560)
    l.new(rim_ramp.outputs[0], m1.inputs[0]); l.new(rim_str.outputs["Fac"], m1.inputs[1])
    m2 = n.new("ShaderNodeMath"); m2.operation = "MULTIPLY"; m2.location = (160, -560)
    l.new(m1.outputs[0], m2.inputs[0]); l.new(gi.outputs["Rim Mask"], m2.inputs[1])
    rim = n.new("ShaderNodeVectorMath"); rim.operation = "SCALE"; rim.location = (320, -560)
    l.new(rim_col.outputs["Color"], rim.inputs[0]); l.new(m2.outputs[0], rim.inputs["Scale"])

    # Self-emission (eyes, VFX-lit accents) ignores the light level.
    glow = n.new("ShaderNodeVectorMath"); glow.operation = "SCALE"; glow.location = (320, -200)
    l.new(gi.outputs["Base"], glow.inputs[0]); l.new(gi.outputs["Emission"], glow.inputs["Scale"])

    add1 = n.new("ShaderNodeVectorMath"); add1.operation = "ADD"; add1.location = (500, 0)
    l.new(lit.outputs[0], add1.inputs[0]); l.new(rim.outputs[0], add1.inputs[1])
    add2 = n.new("ShaderNodeVectorMath"); add2.operation = "ADD"; add2.location = (650, 0)
    l.new(add1.outputs[0], add2.inputs[0]); l.new(glow.outputs[0], add2.inputs[1])
    emit = n.new("ShaderNodeEmission"); emit.location = (780, 0)
    l.new(add2.outputs[0], emit.inputs["Color"])
    l.new(emit.outputs[0], go.inputs["Shader"])
    return grp


def _rgba(hex_color: str, palette: dict | None = None):
    if palette is not None:
        hex_color = ss.resolve_color(hex_color, palette)
    r, g, b = (ss.srgb_to_linear(c) for c in ss.hex_to_rgb(hex_color))
    return (r, g, b, 1.0)


def toon_material(name: str, base: str, shade: str | None = None, palette: dict | None = None,
                  threshold: float = 0.32, softness: float = 0.04, rim_mask: float = 1.0,
                  emission: float = 0.0, gradient: tuple | None = None) -> bpy.types.Material:
    """gradient = (bottom_base, bottom_shade, z_from, z_to): vertical gradient over the object's
    generated (undeformed bounding-box) Z, so it stays stable under skinning."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (300, 0)
    grp = nt.nodes.new("ShaderNodeGroup"); grp.node_tree = toon_group(); grp.location = (0, 0)
    base_rgba = _rgba(base, palette)
    if shade is None:
        # Default shade: a cooler, more saturated, darker version of base (never plain black).
        r, g, b, _ = base_rgba
        shade_rgba = (r * 0.62, g * 0.6, min(1.0, b * 0.78 + 0.02), 1.0)
    else:
        shade_rgba = _rgba(shade, palette)
    grp.inputs["Base"].default_value = base_rgba
    grp.inputs["Shade"].default_value = shade_rgba
    if gradient is not None:
        tc = nt.nodes.new("ShaderNodeTexCoord"); tc.location = (-700, 0)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-520, 0)
        rng = nt.nodes.new("ShaderNodeMapRange"); rng.location = (-360, 0)
        rng.inputs["From Min"].default_value = gradient[3]
        rng.inputs["From Max"].default_value = gradient[2]
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], rng.inputs["Value"])
        for sock_name, top, bottom in (("Base", base_rgba, _rgba(gradient[0], palette)),
                                       ("Shade", shade_rgba, _rgba(gradient[1], palette))):
            mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.location = (-180, 120 if sock_name == "Base" else -60)
            mix.inputs["A"].default_value = top
            mix.inputs["B"].default_value = bottom
            nt.links.new(rng.outputs[0], mix.inputs["Factor"])
            nt.links.new(mix.outputs["Result"], grp.inputs[sock_name])
    grp.inputs["Threshold"].default_value = threshold
    grp.inputs["Softness"].default_value = softness
    grp.inputs["Rim Mask"].default_value = rim_mask
    grp.inputs["Emission"].default_value = emission
    nt.links.new(grp.outputs[0], out.inputs["Surface"])
    mat.diffuse_color = base_rgba
    return mat


def outline_material(name: str, color: str = "#241c28", palette: dict | None = None) -> bpy.types.Material:
    """Inverted-hull outline: emission, backface culled, used through a Solidify modifier."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    light = nt.nodes.new("ShaderNodeAttribute")
    light.attribute_type = "VIEW_LAYER"
    light.attribute_name = "spike_char_light"
    scale = nt.nodes.new("ShaderNodeVectorMath"); scale.operation = "SCALE"
    scale.inputs[0].default_value = _rgba(color, palette)[:3]
    nt.links.new(light.outputs["Fac"], scale.inputs["Scale"])
    nt.links.new(scale.outputs[0], emit.inputs["Color"])
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    mat.use_backface_culling = True
    if hasattr(mat, "use_backface_culling_shadow"):
        mat.use_backface_culling_shadow = True
    return mat


def add_outline(obj, outline_mat, thickness: float = 0.0025):
    """Attach the inverted hull. The outline material goes last; Solidify offsets to it."""
    if any(m.type == "SOLIDIFY" and m.name == "spike_outline" for m in obj.modifiers):
        return
    obj.data.materials.append(outline_mat)
    mod = obj.modifiers.new("spike_outline", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.material_offset = len(obj.data.materials) - 1
    mod.material_offset_rim = len(obj.data.materials) - 1


# ---------------------------------------------------------------------------
# Textured and layered-eye materials
# ---------------------------------------------------------------------------
def _img(path, colorspace="sRGB", extension="REPEAT"):
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = colorspace
    return img


def _node(nt, kind, x, y, **props):
    n = nt.nodes.new(kind)
    n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _math(nt, op, a, b=None, x=0, y=0, clamp=False):
    n = _node(nt, "ShaderNodeMath", x, y, operation=op, use_clamp=clamp)
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def toon_textured(name, image_path, shade_mul=(0.8, 0.72, 0.8), threshold=0.3, softness=0.05, rim_mask=1.0,
                  uv_map="UVMap", mask_attribute=None, fallback="#fae3e5", palette=None, overlay_attribute=None,
                  overlay_color="#22325a", overlays=()):
    """Toon material whose base color is a painted texture; shade = base x shade_mul.

    mask_attribute: a point attribute (0..1) that blends the texture toward the fallback color
    (e.g. keep face features off the back of the head).
    overlay_attribute / overlays: point attributes (0..1) that blend in a flat color on top, in order
    (garment trims and panels defined in 3D where a texture projection cannot place them).
    """
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 500, 0)
    grp = _node(nt, "ShaderNodeGroup", 250, 0)
    grp.node_tree = toon_group()
    uv = _node(nt, "ShaderNodeUVMap", -700, 0, uv_map=uv_map)
    tex = _node(nt, "ShaderNodeTexImage", -500, 0)
    tex.image = _img(image_path)
    tex.interpolation = "Cubic"
    nt.links.new(uv.outputs[0], tex.inputs["Vector"])
    color = tex.outputs["Color"]
    if mask_attribute:
        attr = _node(nt, "ShaderNodeAttribute", -500, -250, attribute_type="GEOMETRY", attribute_name=mask_attribute)
        mix = _node(nt, "ShaderNodeMix", -250, 0, data_type="RGBA")
        mix.inputs["A"].default_value = _rgba(fallback, palette)
        nt.links.new(attr.outputs["Fac"], mix.inputs["Factor"])
        nt.links.new(color, mix.inputs["B"])
        color = mix.outputs["Result"]
    if overlay_attribute:
        attr2 = _node(nt, "ShaderNodeAttribute", -500, -450, attribute_type="GEOMETRY", attribute_name=overlay_attribute)
        mix2 = _node(nt, "ShaderNodeMix", -250, -250, data_type="RGBA")
        mix2.inputs["B"].default_value = _rgba(overlay_color, palette)
        nt.links.new(attr2.outputs["Fac"], mix2.inputs["Factor"])
        nt.links.new(color, mix2.inputs["A"])
        color = mix2.outputs["Result"]
    for i, (attr_name, col) in enumerate(overlays):
        a3 = _node(nt, "ShaderNodeAttribute", -500, -650 - 200 * i, attribute_type="GEOMETRY", attribute_name=attr_name)
        m3 = _node(nt, "ShaderNodeMix", -250, -450 - 200 * i, data_type="RGBA")
        m3.inputs["B"].default_value = _rgba(col, palette)
        nt.links.new(a3.outputs["Fac"], m3.inputs["Factor"])
        nt.links.new(color, m3.inputs["A"])
        color = m3.outputs["Result"]
    shade = _node(nt, "ShaderNodeMix", -50, -120, data_type="RGBA", blend_type="MULTIPLY")
    shade.inputs["Factor"].default_value = 1.0
    shade.inputs["B"].default_value = (*shade_mul, 1.0)
    nt.links.new(color, shade.inputs["A"])
    nt.links.new(color, grp.inputs["Base"])
    nt.links.new(shade.outputs["Result"], grp.inputs["Shade"])
    grp.inputs["Threshold"].default_value = threshold
    grp.inputs["Softness"].default_value = softness
    grp.inputs["Rim Mask"].default_value = rim_mask
    nt.links.new(grp.outputs[0], out.inputs["Surface"])
    return mat


def _value(nt, label, default, x, y):
    v = _node(nt, "ShaderNodeValue", x, y)
    v.name = label
    v.label = label
    v.outputs[0].default_value = default
    return v.outputs[0]


def anime_eye_material(name, white_path, iris_path, corner_v=0.52, glow_color=(0.6, 0.95, 1.0)):
    """Layered anime eye: sclera + opening mask (blink) and a gaze-offset iris.

    Keyframe-able Value nodes: blink (0 open .. 1 closed), lid_heavy (0 open .. 1 heavy upper lid),
    gaze_x / gaze_y (eye-UV units), eye_glow (S+ close-up iris lift).
    """
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 900, 0)
    uv = _node(nt, "ShaderNodeUVMap", -1300, 0, uv_map="eye_uv")
    sep = _node(nt, "ShaderNodeSeparateXYZ", -1100, 0)
    nt.links.new(uv.outputs[0], sep.inputs[0])
    blink = _value(nt, "blink", 0.0, -1300, -300)
    gx = _value(nt, "gaze_x", 0.0, -1300, -400)
    gy = _value(nt, "gaze_y", 0.0, -1300, -500)
    glow = _value(nt, "eye_glow", 0.0, -1300, -600)
    heavy = _value(nt, "lid_heavy", 0.0, -1300, -700)
    # Opening: scale V around the corner line. Lower lid: 1 / max(1 - blink, 0.02);
    # upper lid additionally lowered by lid_heavy.
    open_lo = _math(nt, "MAXIMUM", _math(nt, "SUBTRACT", 1.0, blink, -1100, -300), 0.02, -950, -300)
    heavy_k = _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", heavy, 0.45, -1100, -700), -950, -700)
    open_up = _math(nt, "MAXIMUM", _math(nt, "MULTIPLY", open_lo, heavy_k, -800, -650), 0.02, -650, -650)
    dv = _math(nt, "SUBTRACT", sep.outputs["Y"], corner_v, -950, -100)
    is_up = _math(nt, "GREATER_THAN", dv, 0.0, -800, 0)
    scale = _node(nt, "ShaderNodeMix", -650, -300, data_type="FLOAT")
    nt.links.new(is_up, scale.inputs["Factor"])
    nt.links.new(open_lo, scale.inputs["A"])
    nt.links.new(open_up, scale.inputs["B"])
    vb = _math(nt, "ADD", _math(nt, "DIVIDE", dv, scale.outputs["Result"], -800, -150), corner_v, -650, -150)
    comb_w = _node(nt, "ShaderNodeCombineXYZ", -500, 0)
    nt.links.new(sep.outputs["X"], comb_w.inputs["X"])
    nt.links.new(vb, comb_w.inputs["Y"])
    white = _node(nt, "ShaderNodeTexImage", -300, 100, extension="CLIP", interpolation="Cubic")
    white.image = _img(white_path)
    nt.links.new(comb_w.outputs[0], white.inputs["Vector"])
    # Iris: offset by gaze.
    ix = _math(nt, "SUBTRACT", sep.outputs["X"], gx, -800, -450)
    iy = _math(nt, "SUBTRACT", sep.outputs["Y"], gy, -800, -550)
    comb_i = _node(nt, "ShaderNodeCombineXYZ", -500, -450)
    nt.links.new(ix, comb_i.inputs["X"])
    nt.links.new(iy, comb_i.inputs["Y"])
    iris = _node(nt, "ShaderNodeTexImage", -300, -400, extension="CLIP", interpolation="Cubic")
    iris.image = _img(iris_path)
    nt.links.new(comb_i.outputs[0], iris.inputs["Vector"])
    mix = _node(nt, "ShaderNodeMix", 0, 0, data_type="RGBA")
    nt.links.new(iris.outputs["Alpha"], mix.inputs["Factor"])
    nt.links.new(white.outputs["Color"], mix.inputs["A"])
    nt.links.new(iris.outputs["Color"], mix.inputs["B"])
    light = _node(nt, "ShaderNodeAttribute", 0, -250, attribute_type="VIEW_LAYER", attribute_name="spike_char_light")
    glow_amt = _math(nt, "MULTIPLY", glow, iris.outputs["Alpha"], 150, -400)
    lit = _node(nt, "ShaderNodeVectorMath", 250, 0, operation="SCALE")
    nt.links.new(mix.outputs["Result"], lit.inputs[0])
    nt.links.new(_math(nt, "MAXIMUM", light.outputs["Fac"], 0.0, 150, -250), lit.inputs["Scale"])
    glow_vec = _node(nt, "ShaderNodeVectorMath", 250, -300, operation="SCALE")
    glow_vec.inputs[0].default_value = glow_color
    nt.links.new(glow_amt, glow_vec.inputs["Scale"])
    add = _node(nt, "ShaderNodeVectorMath", 420, -100, operation="ADD")
    nt.links.new(lit.outputs[0], add.inputs[0])
    nt.links.new(glow_vec.outputs[0], add.inputs[1])
    emit = _node(nt, "ShaderNodeEmission", 580, -100)
    nt.links.new(add.outputs[0], emit.inputs["Color"])
    transp = _node(nt, "ShaderNodeBsdfTransparent", 580, 100)
    shader = _node(nt, "ShaderNodeMixShader", 760, 0)
    nt.links.new(white.outputs["Alpha"], shader.inputs["Fac"])
    nt.links.new(transp.outputs[0], shader.inputs[1])
    nt.links.new(emit.outputs[0], shader.inputs[2])
    nt.links.new(shader.outputs[0], out.inputs["Surface"])
    return mat


def anime_lash_material(name, lash_path, corner_v=0.52, color="#211a20"):
    """Upper lash line that closes into a downward arc on blink; lower lash fades on blink."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = _node(nt, "ShaderNodeOutputMaterial", 900, 0)
    uv = _node(nt, "ShaderNodeUVMap", -1300, 0, uv_map="eye_uv")
    sep = _node(nt, "ShaderNodeSeparateXYZ", -1100, 0)
    nt.links.new(uv.outputs[0], sep.inputs[0])
    blink = _value(nt, "blink", 0.0, -1300, -300)
    # s = 1 - 1.6 blink, kept away from zero so the stroke never vanishes.
    heavy = _value(nt, "lid_heavy", 0.0, -1300, -700)
    heavy_k = _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", heavy, 0.45, -1100, -700), -950, -700)
    s0 = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", blink, 1.6, -1100, -300), -950, -300),
               heavy_k, -800, -250)
    s_abs = _math(nt, "MAXIMUM", _math(nt, "ABSOLUTE", s0, None, -800, -350), 0.22, -650, -350)
    s = _math(nt, "MULTIPLY", s_abs, _math(nt, "SIGN", s0, None, -800, -450), -500, -400)
    dv = _math(nt, "SUBTRACT", sep.outputs["Y"], corner_v, -950, -100)
    vu = _math(nt, "ADD", _math(nt, "DIVIDE", dv, s, -350, -150), corner_v, -200, -150)
    comb = _node(nt, "ShaderNodeCombineXYZ", -50, 0)
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(vu, comb.inputs["Y"])
    up = _node(nt, "ShaderNodeTexImage", 100, 100, extension="CLIP", interpolation="Cubic")
    up.image = _img(lash_path)
    nt.links.new(comb.outputs[0], up.inputs["Vector"])
    low = _node(nt, "ShaderNodeTexImage", 100, -250, extension="CLIP", interpolation="Cubic")
    low.image = up.image
    nt.links.new(uv.outputs[0], low.inputs["Vector"])
    upper_only = _math(nt, "GREATER_THAN", vu, corner_v - 0.012, 250, 250)
    lower_only = _math(nt, "LESS_THAN", sep.outputs["Y"], corner_v - 0.012, 250, -450)
    a_up = _math(nt, "MULTIPLY", up.outputs["Alpha"], upper_only, 420, 150)
    fade = _math(nt, "SUBTRACT", 1.0, blink, 250, -550)
    a_low = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", low.outputs["Alpha"], lower_only, 420, -300), fade, 560, -300)
    alpha = _math(nt, "MAXIMUM", a_up, a_low, 700, 0)
    light = _node(nt, "ShaderNodeAttribute", 420, -650, attribute_type="VIEW_LAYER", attribute_name="spike_char_light")
    col = _node(nt, "ShaderNodeVectorMath", 600, -650, operation="SCALE")
    col.inputs[0].default_value = _rgba(color)[:3]
    nt.links.new(light.outputs["Fac"], col.inputs["Scale"])
    emit = _node(nt, "ShaderNodeEmission", 760, -300)
    nt.links.new(col.outputs[0], emit.inputs["Color"])
    transp = _node(nt, "ShaderNodeBsdfTransparent", 760, 150)
    shader = _node(nt, "ShaderNodeMixShader", 880, 0)
    nt.links.new(alpha, shader.inputs["Fac"])
    nt.links.new(transp.outputs[0], shader.inputs[1])
    nt.links.new(emit.outputs[0], shader.inputs[2])
    nt.links.new(shader.outputs[0], out.inputs["Surface"])
    return mat
