"""Sara: body sculpt recipe (signed-distance primitives).

Design intent (from the key art): a tall, long-legged jumper. Full, strong
thighs with a visible vastus teardrop above the knee, a high calf peak,
narrow waist, athletic but not muscular upper body, slim neck, long slim
forearms. Shoulders a touch narrower than the hips' outer line.

All positions are derived from the rig's rest pose so the sculpt and the
skeleton cannot disagree. Units: meters, Blender space (Z up, faces -Y,
+X = her left).
"""
from __future__ import annotations

import numpy as np

from spike_pipeline.modeling import sdf


def _v(*a):
    return np.array(a, dtype=np.float64)


def _lerp(a, b, t):
    return a + (b - a) * t


def _bone(bones, name):
    b = bones[name]
    return _v(*b["head"]), _v(*b["tail"])


def torso_parts(bones):
    """(primitive, blend_k) list for the torso, neck included."""
    hips_h, _ = _bone(bones, "Hips")
    parts = [
        # Pelvis mass and hips' outer line.
        (sdf.Ellipsoid(_v(0.0, 0.006, 0.9), _v(0.131, 0.09, 0.1)), 0.0),
        (sdf.Ellipsoid(_v(0.0, -0.024, 0.95), _v(0.118, 0.07, 0.07)), 0.05),       # lower abdomen
        # Glutes: two masses, blended soft into the pelvis and the thighs.
        (sdf.Ellipsoid(_v(0.056, 0.056, 0.858), _v(0.064, 0.058, 0.074)), 0.05),
        (sdf.Ellipsoid(_v(-0.056, 0.056, 0.858), _v(0.064, 0.058, 0.074)), 0.05),
        # Waist (narrowest), rib cage, upper chest.
        (sdf.Ellipsoid(_v(0.0, 0.006, 1.056), _v(0.091, 0.067, 0.08)), 0.065),
        (sdf.Ellipsoid(_v(0.0, 0.012, 1.17), _v(0.118, 0.086, 0.112)), 0.06),
        (sdf.Ellipsoid(_v(0.0, 0.004, 1.25), _v(0.128, 0.076, 0.075)), 0.05),
        (sdf.Ellipsoid(_v(0.0, 0.044, 1.2), _v(0.114, 0.05, 0.11)), 0.05),        # back / lats
        # Chest.
        (sdf.Ellipsoid(_v(0.05, -0.052, 1.19), _v(0.052, 0.044, 0.048)), 0.035),
        (sdf.Ellipsoid(_v(-0.05, -0.052, 1.19), _v(0.052, 0.044, 0.048)), 0.035),
        # Shoulder girdle and trapezius slope into the neck.
        (sdf.RoundCone(_v(0.02, 0.012, 1.3), _v(0.156, 0.014, 1.318), 0.045, 0.042), 0.04),
        (sdf.RoundCone(_v(-0.02, 0.012, 1.3), _v(-0.156, 0.014, 1.318), 0.045, 0.042), 0.04),
        (sdf.RoundCone(_v(0.012, 0.03, 1.362), _v(0.126, 0.022, 1.324), 0.03, 0.03), 0.045),
        (sdf.RoundCone(_v(-0.012, 0.03, 1.362), _v(-0.126, 0.022, 1.324), 0.03, 0.03), 0.045),
        # Neck: slim, slight forward lean, sternocleidomastoid hint.
        (sdf.RoundCone(_v(0.0, 0.02, 1.325), _v(0.0, 0.012, 1.48), 0.047, 0.041), 0.035),
        (sdf.RoundCone(_v(0.028, 0.0, 1.44), _v(0.012, -0.03, 1.345), 0.012, 0.011), 0.02),
        (sdf.RoundCone(_v(-0.028, 0.0, 1.44), _v(-0.012, -0.03, 1.345), 0.012, 0.011), 0.02),
    ]
    return parts


def arm_parts(bones, side):
    s = 1.0 if side == "Left" else -1.0
    sh, el = _bone(bones, f"{side}UpperArm")
    _, wr = _bone(bones, f"{side}LowerArm")
    up = el - sh
    fo = wr - el
    parts = [
        # Deltoid cap, oriented down the arm.
        (sdf.Ellipsoid(sh + up * 0.12 + _v(s * 0.006, 0.0, 0.004), _v(0.047, 0.047, 0.066), sdf.rotation_to(up)), 0.03),
        (sdf.RoundCone(sh, el, 0.047, 0.036), 0.03),
        # Biceps (front) and triceps (back) masses.
        (sdf.Ellipsoid(_lerp(sh, el, 0.5) + _v(0.0, -0.012, 0.004), _v(0.032, 0.033, 0.062), sdf.rotation_to(up)), 0.025),
        (sdf.Ellipsoid(_lerp(sh, el, 0.42) + _v(0.0, 0.014, 0.004), _v(0.029, 0.03, 0.07), sdf.rotation_to(up)), 0.025),
        (sdf.RoundCone(el, wr, 0.036, 0.024), 0.03),
        # Forearm belly near the elbow (brachioradialis / flexors).
        (sdf.Ellipsoid(_lerp(el, wr, 0.27) + _v(0.0, -0.004, 0.006), _v(0.039, 0.035, 0.072), sdf.rotation_to(fo)), 0.03),
        (sdf.Ellipsoid(wr + _v(0.0, 0.0, 0.0), _v(0.024, 0.018, 0.021), sdf.rotation_to(fo)), 0.015),  # wrist
    ]
    return parts


def leg_parts(bones, side):
    s = 1.0 if side == "Left" else -1.0
    hip, knee = _bone(bones, f"{side}UpperLeg")
    _, ankle = _bone(bones, f"{side}LowerLeg")
    th = knee - hip
    sh = ankle - knee
    parts = [
        (sdf.RoundCone(hip + _v(0.0, 0.0, 0.01), knee, 0.093, 0.046), 0.05),
        # Vastus lateralis (outer sweep), rectus femoris (front), hamstrings (back).
        (sdf.Ellipsoid(_lerp(hip, knee, 0.4) + _v(s * 0.032, -0.004, 0.0), _v(0.056, 0.06, 0.155), sdf.rotation_to(th)), 0.04),
        (sdf.Ellipsoid(_lerp(hip, knee, 0.35) + _v(0.0, -0.03, 0.0), _v(0.052, 0.048, 0.16), sdf.rotation_to(th)), 0.04),
        (sdf.Ellipsoid(_lerp(hip, knee, 0.4) + _v(-s * 0.004, 0.03, 0.0), _v(0.055, 0.048, 0.15), sdf.rotation_to(th)), 0.04),
        # Inner-thigh mass high, vastus medialis teardrop low.
        (sdf.Ellipsoid(_lerp(hip, knee, 0.2) + _v(-s * 0.03, 0.004, 0.0), _v(0.05, 0.055, 0.1), sdf.rotation_to(th)), 0.04),
        (sdf.Ellipsoid(_lerp(hip, knee, 0.82) + _v(-s * 0.026, -0.014, 0.0), _v(0.032, 0.034, 0.055), sdf.rotation_to(th)), 0.025),
        # Knee: patella and the side condyles.
        (sdf.Ellipsoid(knee + _v(0.0, -0.022, 0.008), _v(0.024, 0.016, 0.028)), 0.04),
        (sdf.Ellipsoid(knee + _v(0.0, 0.002, 0.0), _v(0.042, 0.039, 0.04)), 0.05),
        (sdf.RoundCone(knee, ankle, 0.044, 0.024), 0.035),
        # Calf: two gastrocnemius heads, high peak; lateral head a bit lower.
        (sdf.Ellipsoid(_lerp(knee, ankle, 0.27) + _v(-s * 0.012, 0.025, 0.0), _v(0.037, 0.041, 0.11), sdf.rotation_to(sh)), 0.035),
        (sdf.Ellipsoid(_lerp(knee, ankle, 0.31) + _v(s * 0.014, 0.021, 0.0), _v(0.033, 0.037, 0.1), sdf.rotation_to(sh)), 0.035),
        # Shin front edge, ankle bones.
        (sdf.Ellipsoid(_lerp(knee, ankle, 0.45) + _v(0.0, -0.012, 0.0), _v(0.028, 0.026, 0.16), sdf.rotation_to(sh)), 0.03),
        (sdf.Ellipsoid(ankle + _v(0.0, 0.004, 0.004), _v(0.027, 0.026, 0.024)), 0.02),
    ]
    foot_h, foot_t = _bone(bones, f"{side}Foot")
    _, toe_t = _bone(bones, f"{side}Toes")
    parts += [
        (sdf.Ellipsoid(foot_h + _v(0.0, 0.02, -0.035), _v(0.03, 0.04, 0.035)), 0.02),              # heel
        (sdf.RoundCone(foot_h + _v(0.0, -0.01, -0.03), toe_t + _v(0.0, 0.01, 0.0), 0.034, 0.024), 0.02),
    ]
    return parts


def body_field(bones, voxel=0.0035, include=("torso", "arms", "legs"), bmin=(-0.66, -0.25, 0.0), bmax=(0.66, 0.2, 1.53)):
    field = sdf.Field(_v(*bmin), _v(*bmax), voxel)
    parts = []
    if "torso" in include:
        parts += torso_parts(bones)
    if "arms" in include:
        parts += arm_parts(bones, "Left") + arm_parts(bones, "Right")
    if "legs" in include:
        parts += leg_parts(bones, "Left") + leg_parts(bones, "Right")
    for prim, k in parts:
        field.union(prim, k)
    return field


def hand_field(bones, side, voxel=0.0011):
    """Fingers need a finer grid than the body: separate field per hand."""
    h, t = _bone(bones, f"{side}Hand")
    names = [n for n in bones if n.startswith(side) and any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
    pts = [h, t] + [_v(*bones[n]["tail"]) for n in names]
    lo = np.min(pts, axis=0) - 0.04
    hi = np.max(pts, axis=0) + 0.04
    field = sdf.Field(lo, hi, voxel)
    ax = (t - h) / np.linalg.norm(t - h)
    R = sdf.rotation_to(ax)
    # Palm: a flattened rounded slab from the wrist to the knuckles; thenar/hypothenar pads.
    idx_base = _v(*bones[f"{side}IndexProximal"]["head"])
    lit_base = _v(*bones[f"{side}LittleProximal"]["head"])
    palm_c = (h + (idx_base + lit_base) * 0.5) * 0.5
    across = idx_base - lit_base
    across /= np.linalg.norm(across)
    normal = np.cross(ax, across)
    Rp = np.stack([across, normal, ax], axis=1)
    field.union(sdf.RoundBox(palm_c, _v(0.031, 0.006, 0.034), 0.009, Rp), 0.0)
    field.union(sdf.Capsule(h - ax * 0.012, h + ax * 0.012, 0.02), 0.012)            # wrist blend
    thumb_mc = _v(*bones[f"{side}ThumbMetacarpal"]["head"])
    field.union(sdf.Ellipsoid(_lerp(h, thumb_mc, 0.7) + normal * 0.004, _v(0.016, 0.011, 0.022), Rp), 0.01)
    for f in ("Thumb", "Index", "Middle", "Ring", "Little"):
        chain = [n for n in names if n.startswith(f"{side}{f}")]
        base_r = 0.0082 if f == "Thumb" else {"Index": 0.0068, "Middle": 0.007, "Ring": 0.0066, "Little": 0.0057}[f]
        for i, n in enumerate(chain):
            a, b = _v(*bones[n]["head"]), _v(*bones[n]["tail"])
            r0 = base_r * (1.0 - 0.1 * i)
            r1 = base_r * (1.0 - 0.1 * (i + 1)) * (0.92 if i == len(chain) - 1 else 1.0)
            if i == len(chain) - 1:
                b = b + (b - a) * 0.12  # fingertip pad past the joint
            field.union(sdf.RoundCone(a, b, r0, r1), 0.004)
            if i < len(chain) - 1:
                field.union(sdf.Ellipsoid(b, _v(r1 * 1.08, r1 * 1.05, r1 * 1.1)), 0.003)  # knuckle
    return field
