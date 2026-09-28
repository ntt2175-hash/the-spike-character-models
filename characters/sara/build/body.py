"""Sara: body from the shared proportion master.

Sara is the first validated configuration of spike_pipeline.modeling.body_master: her skeleton
proportions (character.json "proportions") and volume multipliers ("body_shape") describe a
slender, long-limbed anime volleyball athlete. Her athleticism reads from length, posture and
motion, not muscle volume. No body numbers live here; change the character data instead.

Units: meters, Blender space (Z up, faces -Y, +X = her left).
"""
from __future__ import annotations

import spike_specs as ss
from spike_pipeline.modeling import body_master

_CACHE = {}


def master(bones) -> body_master.Body:
    key = id(bones)
    if key not in _CACHE:
        c = ss.load_character("sara")
        _CACHE[key] = body_master.Body(bones, c.get("body_shape", {}), c["proportions"]["height_m"])
    return _CACHE[key]


def torso_parts(bones):
    return master(bones).torso_parts()


def arm_parts(bones, side):
    return master(bones).arm_parts(side)


def leg_parts(bones, side):
    return master(bones).leg_parts(side)


def body_field(bones, voxel=0.0035, include=("torso", "arms", "legs"), bmin=None, bmax=None):
    return master(bones).body_field(voxel, include, bmin, bmax)


def hand_field(bones, side, voxel=0.0011):
    return master(bones).hand_field(side, voxel)


def landmarks(bones):
    return master(bones).lm
