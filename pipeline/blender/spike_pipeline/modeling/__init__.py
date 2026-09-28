"""Procedural modeling toolkit used by character build recipes.

    sdf      signed-distance sculpting: primitives, smooth blends, meshing
    head     anime head loft from front/side profiles
    lock     hair locks swept along splines (lens cross-section, taper, twist)
    paint    texture painting (face features, iris, garment prints) with Pillow
    weights  skin weights from bone segments, smoothing and transfer

These are TOOLS. The design decisions live in each character's recipe
(characters/<id>/build/build_<id>.py) so characters share technology, not
design. Requires numpy, Pillow and scikit-image in Blender's Python (or the
bpy module environment).
"""
