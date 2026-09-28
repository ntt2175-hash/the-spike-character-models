"""Body proportion master: one parametric body for every character.

A character's body is defined by two blocks in its character.json:

  proportions  (skeleton; see spike_specs.apply_proportions)
      height_m, head_units, leg_ratio (hip joint height / height = leg length), neck_length,
      shoulder_width, hip_width, arm_length, hand_scale, foot_scale
      -> torso length follows from them: neck base - hip joint

  body_shape   (volumes; this module, ranges in pipeline/specs/body.json)
      multipliers on the reference body (1.0): torso_width, torso_depth, bust, waist_width,
      waist_depth, pelvis_width, pelvis_depth, glute, neck_width, shoulder_mass, deltoid,
      arm_thickness, forearm_thickness, wrist, thigh_thickness, thigh_outer, thigh_inner,
      knee_size, calf_thickness, ankle, foot_width, hand_width, finger_thickness

Every form is placed relative to the skeleton (vertical torso positions as fractions of the
hip-to-neck distance, limb forms along their bones) and sized by the multipliers, so changing a
proportion changes the relationships between segments, never a uniform scale. The torso and each limb
segment are LOFTS: one continuous surface whose cross-section follows smooth profiles (tables below),
so contours read as a human body, not a union of muscle blobs. The reference body is a realistic
slender-athletic 1.66 m woman; stylized characters are expressed as departures from it.
Garments and shoes read the same landmarks (Body.lm) so they follow the body by construction.
"""
from __future__ import annotations

import numpy as np

from . import sdf

REF_HEIGHT = 1.66
REF_TORSO = 0.4671          # hip joint -> neck base of the reference body at 1.66 m

DEFAULTS = {
    "torso_width": 1.0, "torso_depth": 1.0, "bust": 1.0, "waist_width": 1.0, "waist_depth": 1.0,
    "pelvis_width": 1.0, "pelvis_depth": 1.0, "glute": 1.0, "neck_width": 1.0,
    "shoulder_mass": 1.0, "deltoid": 1.0, "arm_thickness": 1.0, "forearm_thickness": 1.0, "wrist": 1.0,
    "thigh_thickness": 1.0, "thigh_outer": 1.0, "thigh_inner": 1.0, "knee_size": 1.0,
    "calf_thickness": 1.0, "ankle": 1.0, "foot_width": 1.0, "hand_width": 1.0, "finger_thickness": 1.0,
}


# Reference body profiles (realistic slender-athletic woman at 1.66 m). Designed as smooth, continuous
# human contours: every value is a cross-section radius in meters; the proportion parameters scale
# them zone by zone. Torso rows: (fraction of hip joint -> neck base, half width, y front, y back,
# front exponent, back exponent). Limb rows: (t along the bone, r_out, r_in, r_front, r_back).
TORSO = np.array([                           # rounded sections (exponent ~2): a ribcage barrel, not an extrusion
    (-0.135, 0.004, -0.022, 0.050, 2.0, 2.0),  # crotch: a rounded arch between the thighs
    (-0.105, 0.030, -0.036, 0.068, 2.0, 2.0),
    (-0.075, 0.075, -0.054, 0.086, 2.0, 2.1),  # gluteal fold (the glute lobes add the rear volume)
    (-0.04, 0.118, -0.066, 0.098, 2.1, 2.2),
    (0.00, 0.134, -0.073, 0.103, 2.2, 2.3),    # hip joint: one pelvis volume, depth in proportion to width
    (0.06, 0.134, -0.077, 0.099, 2.2, 2.3),
    (0.13, 0.129, -0.079, 0.092, 2.2, 2.3),    # iliac crest, subtle lower belly
    (0.21, 0.121, -0.078, 0.087, 2.1, 2.2),
    (0.29, 0.114, -0.075, 0.085, 2.0, 2.2),
    (0.37, 0.110, -0.073, 0.086, 2.0, 2.2),    # natural waist: a gradual inflection, a mild lumbar curve
    (0.45, 0.111, -0.075, 0.091, 2.0, 2.2),    # upper abdomen, just under the ribcage
    (0.52, 0.114, -0.079, 0.097, 2.0, 2.3),
    (0.59, 0.118, -0.083, 0.102, 2.0, 2.3),    # lower ribs: the barrel has real depth
    (0.66, 0.122, -0.084, 0.105, 2.0, 2.4),    # chest line / sternum (the chest forms sit on it)
    (0.73, 0.125, -0.084, 0.105, 2.1, 2.4),
    (0.80, 0.125, -0.082, 0.102, 2.1, 2.4),
    (0.86, 0.116, -0.074, 0.095, 2.1, 2.3),    # armpit / upper chest: fills smoothly into the chest
    (0.93, 0.088, -0.052, 0.079, 2.0, 2.2),    # shoulder slope
    (1.00, 0.055, -0.034, 0.060, 2.0, 2.0),    # neck base
])
THIGH = np.array([                           # forms peak at different heights: quads front, hamstrings back, outer sweep
    (0.00, 0.072, 0.072, 0.080, 0.084),
    (0.10, 0.075, 0.075, 0.083, 0.086),        # trochanter: the hip line stays controlled
    (0.25, 0.077, 0.074, 0.084, 0.084),        # outer sweep peaks lower, hamstring mass behind
    (0.40, 0.075, 0.068, 0.084, 0.078),        # rectus femoris: the front line stays full longer
    (0.55, 0.069, 0.062, 0.078, 0.070),
    (0.70, 0.062, 0.057, 0.068, 0.061),
    (0.84, 0.055, 0.056, 0.057, 0.053),        # vastus medialis keeps the inner line full above the knee
    (1.00, 0.050, 0.050, 0.051, 0.048),        # knee
])
SHIN = np.array([
    (0.00, 0.050, 0.050, 0.051, 0.048),
    (0.08, 0.047, 0.048, 0.047, 0.055),
    (0.18, 0.048, 0.048, 0.044, 0.064),
    (0.30, 0.048, 0.047, 0.043, 0.066),        # calf peak, high
    (0.42, 0.044, 0.043, 0.042, 0.059),
    (0.55, 0.038, 0.037, 0.040, 0.047),
    (0.70, 0.031, 0.031, 0.035, 0.033),
    (0.85, 0.026, 0.026, 0.029, 0.027),
    (1.00, 0.027, 0.027, 0.028, 0.030),        # ankle bones
])
UPPER_ARM = np.array([                         # segmented: deltoid cap, insertion, biceps/triceps, elbow
    (0.00, 0.050, 0.046, 0.049, 0.050),
    (0.10, 0.055, 0.046, 0.051, 0.051),        # deltoid cap
    (0.22, 0.052, 0.044, 0.049, 0.050),
    (0.36, 0.043, 0.041, 0.045, 0.047),        # deltoid insertion: the line steps in
    (0.52, 0.042, 0.040, 0.047, 0.047),        # biceps (front) / triceps (back)
    (0.70, 0.039, 0.038, 0.043, 0.042),
    (0.86, 0.034, 0.034, 0.036, 0.036),        # narrowing above the elbow
    (1.00, 0.036, 0.036, 0.035, 0.037),        # elbow condyles
])
FOREARM = np.array([                           # out/in = back of hand / palm; front/back = thumb / little side
    (0.00, 0.036, 0.036, 0.035, 0.037),
    (0.16, 0.041, 0.038, 0.044, 0.040),        # forearm belly (brachioradialis / flexors), high
    (0.34, 0.037, 0.034, 0.040, 0.037),
    (0.55, 0.030, 0.028, 0.033, 0.031),
    (0.75, 0.024, 0.022, 0.028, 0.027),
    (0.90, 0.020, 0.019, 0.026, 0.025),        # wrist: flatter than wide
    (1.00, 0.020, 0.019, 0.026, 0.025),
])
FOOT = np.array([                              # heel -> toe tip; out, in, top (dorsal), bottom (plantar)
    (0.00, 0.019, 0.019, 0.023, 0.023),        # narrow, compact rounded heel
    (0.14, 0.024, 0.024, 0.044, 0.024),        # heel / ankle: the top rises steeply into the ankle
    (0.32, 0.028, 0.031, 0.046, 0.017),        # arch: the sole lifts off the ground, high instep
    (0.52, 0.033, 0.036, 0.033, 0.015),        # instep slopes down to the metatarsals
    (0.72, 0.038, 0.040, 0.021, 0.016),        # ball of the foot (widest, on the ground)
    (0.88, 0.032, 0.034, 0.015, 0.012),        # toes: a compact forefoot
    (1.00, 0.023, 0.025, 0.010, 0.009),        # rounded toe tip
])


def _v(*a):
    return np.array(a, dtype=np.float64)


def _lerp(a, b, t):
    return a + (b - a) * t


class Body:
    def __init__(self, bones, shape=None, height=REF_HEIGHT):
        self.bones = bones
        self.p = dict(DEFAULTS)
        for k, v in (shape or {}).items():
            if k in DEFAULTS:
                self.p[k] = float(v)
        self.s = height / REF_HEIGHT
        hip_l, _ = self.bone("LeftUpperLeg")
        hip_r, _ = self.bone("RightUpperLeg")
        neck, _ = self.bone("Neck")
        self.hip_z = 0.5 * (hip_l[2] + hip_r[2])
        self.neck_z = neck[2]
        self.torso_len = self.neck_z - self.hip_z
        self.kt = self.torso_len / (REF_TORSO * self.s)          # vertical torso compression vs reference
        self.lm = self._landmarks()

    # ------------------------------------------------------------------ helpers
    def bone(self, n):
        b = self.bones[n]
        return _v(*b["head"]), _v(*b["tail"])

    def tz(self, frac):
        """Height at a fraction of the hip-joint -> neck-base distance."""
        return self.hip_z + frac * self.torso_len

    def _landmarks(self):
        p, s = self.p, self.s
        ua_l, _ = self.bone("LeftUpperArm")
        return {
            "hip_z": self.hip_z, "neck_z": self.neck_z, "torso_len": self.torso_len, "s": s, "kt": self.kt,
            "waist_z": self.tz(0.377), "chest_z": self.tz(0.664), "waistband_z": self.hip_z + 0.12 * s * self.kt,
            "hem_z": self.hip_z + 0.05 * s * self.kt, "shoulder_x": abs(ua_l[0]), "shoulder_z": ua_l[2],
            "torso_half_width": 0.118 * s * p["torso_width"], "torso_half_depth": 0.086 * s * p["torso_depth"],
            "pelvis_half_width": 0.131 * s * p["pelvis_width"],
        }

    def E(self, center, radii, rot=None):
        return sdf.Ellipsoid(center, radii, rot)

    # ------------------------------------------------------------------ torso
    def _torso_loft(self):
        p, s = self.p, self.s
        F = TORSO[:, 0]
        wm = np.interp(F, [-1.0, 0.1, 0.37, 0.62, 0.95, 1.0, 2.0],
                       [p["pelvis_width"], p["pelvis_width"], p["waist_width"], p["torso_width"], p["torso_width"],
                        p["neck_width"], p["neck_width"]])
        dm = np.interp(F, [-1.0, 0.1, 0.37, 0.62, 2.0], [p["pelvis_depth"], p["pelvis_depth"], p["waist_depth"],
                                                         p["torso_depth"], p["torso_depth"]])
        gm = np.interp(F, [-1.0, 0.06, 0.2, 2.0], [p["glute"], p["glute"], 1.0, 1.0])
        rows = []
        for (f, w, yf, yb, nf, nb), a_, b_, g_ in zip(TORSO, wm, dm, gm):
            rows.append((self.tz(f), w * s * a_, yf * s * b_, yb * s * b_ * g_, nf, nb))
        self._torso_rows = np.asarray(rows)
        return sdf.ZLoft(rows)

    def torso_rows(self):
        """Torso sections (z, half width, y front, y back, n front, n back) in meters (for garments)."""
        if not hasattr(self, "_torso_rows"):
            self._torso_loft()
        return self._torso_rows

    def bust_apex(self):
        """(z, y) of the chest's most forward point (garments hang from it)."""
        p, s = self.p, self.s
        c = self._bust_center()
        return c[2] - 0.01 * s, c[1] + 0.004 * s - 0.04 * s * p["bust"]

    def _bust_center(self):
        p, s = self.p, self.s
        return _v(0.052 * s * p["torso_width"], -0.057 * s * p["torso_depth"], self.tz(0.655))

    def _thigh_rows(self):
        p = self.p
        th = self._scaled(THIGH, [0.0, 0.75, 1.0], [p["thigh_thickness"], p["thigh_thickness"], p["knee_size"]],
                          out_in=p["thigh_outer"])
        th[:, 2] *= np.interp(THIGH[:, 0], [0.0, 0.6, 1.0], [p["thigh_inner"], 1.0, 1.0])   # inner fullness (high)
        return th

    def torso_parts(self):
        """The torso is ONE smooth loft (pelvis -> waist -> rib cage -> upper chest); the chest, shoulder
        girdle, trapezius and neck blend into it with wide, soft transitions."""
        p, s = self.p, self.s
        tw, td, bu = p["torso_width"], p["torso_depth"], p["bust"]
        sh_l, _ = self.bone("LeftShoulder")
        ua_l, _ = self.bone("LeftUpperArm")
        sh_r, _ = self.bone("RightShoulder")
        ua_r, _ = self.bone("RightUpperArm")
        nz = self.neck_z
        parts = [(self._torso_loft(), 0.0)]
        pw, pd, gl = p["pelvis_width"], p["pelvis_depth"], p["glute"]
        for sx in (1.0, -1.0):   # glute lobes: real rear volume with a soft cleft and a curved lower line
            parts.append((self.E(_v(sx * 0.052 * s * pw, 0.058 * s * pd * gl, self.tz(-0.04)),
                                 _v(0.064 * pw, 0.05 * pd * gl, 0.078 * self.kt) * s), 0.045 * s))
        bc = self._bust_center()
        for sx in (1.0, -1.0):
            # Chest: one teardrop per side - a tapered slope from the upper chest into a rounder lower mass,
            # angled slightly outward, with a natural valley between. Smooth, never a disc or a shelf.
            top = _v(sx * 0.047 * s * tw, bc[1] + 0.016 * s, self.tz(0.75))
            low = _v(sx * (bc[0] + 0.004 * s), bc[1] + 0.004 * s, bc[2] - 0.01 * s)
            parts.append((sdf.RoundCone(top, low, 0.022 * s * bu, 0.04 * s * bu), 0.045 * s))
        sm = p["shoulder_mass"] * s
        nw = p["neck_width"] * s
        for sh, ua, sx in ((sh_l, ua_l, 1.0), (sh_r, ua_r, -1.0)):
            # Shoulder: neck -> clavicle -> deltoid. A slim girdle mass under the acromion, the clavicle ridge
            # in front, and the trapezius rising high up the side of the neck.
            parts.append((sdf.RoundCone(sh + _v(0, 0.022, -0.006) * s, ua + _v(-sx * 0.008, 0.008, 0.012) * s,
                                        0.036 * sm, 0.034 * sm), 0.045 * s))
            # (a subtle ridge riding just on the upper-chest surface, rising slightly toward the shoulder)
            parts.append((sdf.RoundCone(_v(sx * 0.02 * nw, -0.036 * s * td, nz - 0.014 * s),
                                        ua + _v(-sx * 0.024, -0.014, 0.02) * s, 0.0042 * s, 0.005 * s), 0.026 * s))
            # (starts high on the neck and is fuller at the base: a short visible neck and a soft neck-to-
            # shoulder transition, never a thin mannequin neck)
            parts.append((sdf.RoundCone(_v(sx * 0.016 * nw, 0.03 * s, nz + 0.058 * s), _lerp(sh, ua, 0.72) + _v(0, 0.0137, 0.0214) * s,
                                        0.033 * sm, 0.03 * sm), 0.055 * s))
        # Neck: slender but not thin, slight forward lean, sternocleidomastoid hint.
        parts.append((sdf.RoundCone(_v(0, 0.02 * s, nz - 0.022 * s), _v(0, 0.012 * s, nz + 0.1331 * s), 0.052 * nw, 0.044 * nw), 0.04 * s))
        for sx in (1.0, -1.0):
            parts.append((sdf.RoundCone(_v(sx * 0.028 * nw, 0.0, nz + 0.093 * s), _v(sx * 0.012 * nw, -0.03 * s, nz - 0.002 * s),
                                        0.012 * nw, 0.011 * nw), 0.02 * s))
        return parts

    # ------------------------------------------------------------------ arms
    def arm_parts(self, side):
        p, s = self.p, self.s
        sh, el = self.bone(f"{side}UpperArm")
        _, wr = self.bone(f"{side}LowerArm")
        at, ft, wk, dk = p["arm_thickness"], p["forearm_thickness"], p["wrist"], p["deltoid"]
        ef = 0.5 * (at + ft)
        up = self._scaled(UPPER_ARM, [0.0, 0.18, 0.34, 1.0], [dk, dk, at, ef])
        fo = self._scaled(FOREARM, [0.0, 0.12, 0.55, 0.85, 1.0], [ef, ft, ft, wk, wk])
        # Forearm frame: 'front/back' = across the wrist (thumb side), the wide axis; 'out/in' = the
        # back of the hand / the palm, the thin axis.
        idx = _v(*self.bones[f"{side}IndexProximal"]["head"])
        lit = _v(*self.bones[f"{side}LittleProximal"]["head"])
        across = idx - lit
        palm_n = np.cross(wr - el, across)
        ua_dir = (el - sh) / np.linalg.norm(el - sh)
        back = np.cross(_v(0, 0, 1), ua_dir)                 # behind the elbow (olecranon side)
        back = back if back @ _v(0, 1, 0) > 0 else -back
        back /= np.linalg.norm(back)
        return [
            (sdf.Loft(sh, el, up, u_hint=_v(0, 0, 1), ext=(0.012 * s, 0.0)), 0.03 * s),
            (sdf.Loft(el, wr, fo, u_hint=palm_n, v_hint=across, ext=(0.0, 0.004 * s)), 0.02 * s),
            # Elbow point and wrist bones: small, structural accents that segment the arm.
            (self.E(el + back * 0.024 * ef * s, _v(0.014, 0.012, 0.016) * ef * s, sdf.rotation_to(ua_dir)), 0.012 * s),
            (self.E(wr - (wr - el) / np.linalg.norm(wr - el) * 0.012 * s, _v(0.021, 0.027, 0.014) * wk * s,
                    np.stack([_n(palm_n), _n(across), _n(wr - el)], axis=1)), 0.012 * s),
        ]

    def _scaled(self, table, knots, mults, out_in=None):
        t = table[:, 0]
        m = np.interp(t, knots, mults) * self.s
        rows = table.copy()
        rows[:, 1:] = table[:, 1:] * m[:, None]
        if out_in is not None:
            rows[:, 1] = rows[:, 2] + (rows[:, 1] - rows[:, 2]) * out_in
        return rows

    # ------------------------------------------------------------------ legs
    def leg_parts(self, side):
        p, s = self.p, self.s
        sx = 1.0 if side == "Left" else -1.0
        hip, knee = self.bone(f"{side}UpperLeg")
        _, ankle = self.bone(f"{side}LowerLeg")
        tt, to, ti = p["thigh_thickness"], p["thigh_outer"], p["thigh_inner"]
        kn, ct, ak, fw = p["knee_size"], p["calf_thickness"], p["ankle"], p["foot_width"] * s
        th = self._thigh_rows()
        shn = self._scaled(SHIN, [0.0, 0.12, 0.25, 0.7, 0.88, 1.0], [kn, ct, ct, ct, ak, ak])
        axis_t = (knee - hip) / np.linalg.norm(knee - hip)
        axis_s = (ankle - knee) / np.linalg.norm(ankle - knee)
        front = _v(0, -1, 0)
        inner = _v(-sx, 0, 0)
        parts = [
            # One long, clean line: thigh -> small knee -> slender calf -> narrow ankle.
            # The thigh's top continues up inside the hip as a long dome: its outer side IS the hip curve
            # (trochanter -> iliac crest -> waist), so the pelvis never forms a ledge or a blend bulge.
            (sdf.Loft(hip, knee, th, u_hint=_v(sx, 0, 0), ext=(0.0, 0.018 * s), cap=(1.9, 1.0)), 0.035 * s),
            # (the thigh and shin sections overlap through the knee, so the joint never pinches into a ring)
            (sdf.Loft(knee, ankle, shn, u_hint=_v(sx, 0, 0), ext=(0.018 * s, 0.0)), 0.02 * s),
            # Knee structure, small but readable: patella, vastus medialis teardrop, tibial tuberosity.
            (self.E(knee + front * 0.035 * kn * s + axis_t * -0.004 * s, _v(0.017, 0.011, 0.021) * kn * s,
                    sdf.rotation_to(axis_t)), 0.012 * s),
            (self.E(_lerp(hip, knee, 0.87) + inner * 0.03 * tt * s + front * 0.012 * s, _v(0.022, 0.022, 0.042) * tt * s,
                    sdf.rotation_to(axis_t)), 0.02 * s),
            (self.E(knee + axis_s * 0.05 * s + front * 0.036 * kn * s, _v(0.012, 0.008, 0.018) * kn * s,
                    sdf.rotation_to(axis_s)), 0.012 * s),
        ]
        parts.append((self._foot(side), 0.022 * s))
        return parts

    def _foot(self, side):
        """Compact foot with a raised arch: heel -> arch -> ball -> rounded toes (the shoe sits around it)."""
        s = self.s
        sx = 1.0 if side == "Left" else -1.0
        fw = self.p["foot_width"]
        ankle, _ = self.bone(f"{side}Foot")
        _, toe_t = self.bone(f"{side}Toes")
        fs = float(np.linalg.norm(toe_t[:2] - ankle[:2])) / 0.187           # foot length vs reference
        heel = _v(ankle[0], ankle[1] + 0.04 * s * fs, 0.024 * s)
        tip = _v(toe_t[0], toe_t[1], 0.011 * s)
        rows = FOOT.copy()
        rows[:, 1:3] *= fw * s
        rows[:, 3:5] *= s
        return sdf.Loft(heel, tip, rows, u_hint=_v(sx, 0, 0), v_hint=_v(0, 0, 1))

    # ------------------------------------------------------------------ skinning
    def part_groups(self):
        """Body parts and the bones that may move each: skin weights are assigned per PART first (by the
        part's own distance field), then per bone inside the part. The chest stays with the ribcage when
        the arm moves, the glutes with the pelvis when the thigh moves: clean deformation by construction."""
        groups = {"torso": (self.torso_parts(), ["Hips", "Spine", "Chest", "UpperChest", "Neck", "LeftShoulder", "RightShoulder"])}
        for side in ("Left", "Right"):
            groups[f"arm_{side}"] = (self.arm_parts(side), [f"{side}Shoulder", f"{side}UpperArm", f"{side}UpperArmTwist",
                                                             f"{side}LowerArm", f"{side}LowerArmTwist", f"{side}Hand"])
            groups[f"leg_{side}"] = (self.leg_parts(side), ["Hips", f"{side}UpperLeg", f"{side}UpperLegTwist",
                                                             f"{side}LowerLeg", f"{side}Foot", f"{side}Toes"])
        return groups

    def part_weights(self, P, softness=0.01):
        """(group names, membership (N, G)) from the parts' distance fields: a smooth partition whose
        transition width is `softness` (m) around where two parts meet."""
        names, D = [], []
        for g, (parts, _bones) in self.part_groups().items():
            d = np.full(len(P), 1.0)
            for prim, k in parts:
                d = sdf.smin(d, prim.eval(P), k)
            names.append(g)
            D.append(d)
        D = np.stack(D, axis=1)
        M = np.exp(-(D - D.min(axis=1, keepdims=True)) / (softness * self.s))
        return names, M / M.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ fields
    def body_field(self, voxel=0.0035, include=("torso", "arms", "legs"), bmin=None, bmax=None):
        s = self.s
        bmin = _v(-0.66, -0.25, 0.0) * s if bmin is None else _v(*bmin)
        bmax = _v(0.66, 0.2, self.neck_z + 0.183 * s) if bmax is None else _v(*bmax)
        field = sdf.Field(bmin, bmax, voxel)
        parts = []
        if "torso" in include:
            parts += self.torso_parts()
        if "arms" in include:
            parts += self.arm_parts("Left") + self.arm_parts("Right")
        if "legs" in include:
            parts += self.leg_parts("Left") + self.leg_parts("Right")
        for prim, k in parts:
            field.union(prim, k)
        return field

    def hand_field(self, side, voxel=0.0011):
        """Fingers need a finer grid than the body: separate field per hand."""
        p, s = self.p, self.s
        bones = self.bones
        h, t = self.bone(f"{side}Hand")
        names = [n for n in bones if n.startswith(side) and any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Little"))]
        pts = [h, t] + [_v(*bones[n]["tail"]) for n in names]
        field = sdf.Field(np.min(pts, axis=0) - 0.04 * s, np.max(pts, axis=0) + 0.04 * s, voxel)
        ax = (t - h) / np.linalg.norm(t - h)
        idx_base = _v(*bones[f"{side}IndexProximal"]["head"])
        lit_base = _v(*bones[f"{side}LittleProximal"]["head"])
        palm_c = (h + (idx_base + lit_base) * 0.5) * 0.5
        across = idx_base - lit_base
        across /= np.linalg.norm(across)
        normal = np.cross(ax, across)
        Rp = np.stack([across, normal, ax], axis=1)
        hw, fk, wk = p["hand_width"], p["finger_thickness"], p["wrist"]
        palm_half_len = 0.5 * float(np.linalg.norm((idx_base + lit_base) * 0.5 - h))
        # Palm: a narrow flattened slab from the wrist to the knuckles; thenar/hypothenar pads.
        field.union(sdf.RoundBox(palm_c, _v(0.031 * hw * s, 0.0055 * s, palm_half_len * 0.76), 0.0085 * s * _mix(hw, 1.0), Rp), 0.0)
        field.union(sdf.Capsule(h - ax * 0.012 * s, h + ax * 0.012 * s, 0.02 * wk * s), 0.012 * s)
        thumb_mc = _v(*bones[f"{side}ThumbMetacarpal"]["head"])
        field.union(self.E(_lerp(h, thumb_mc, 0.7) + normal * 0.004 * s, _v(0.016 * hw, 0.011, 0.022) * s, Rp), 0.01 * s)
        for f in ("Thumb", "Index", "Middle", "Ring", "Little"):
            chain = [n for n in names if n.startswith(f"{side}{f}")]
            # Slender, tapering fingers (the key art's reaching hand): long, fine at the tips.
            base_r = (0.0076 if f == "Thumb" else {"Index": 0.0062, "Middle": 0.0064, "Ring": 0.006, "Little": 0.0052}[f]) * fk * s
            for i, n in enumerate(chain):
                a, b = _v(*bones[n]["head"]), _v(*bones[n]["tail"])
                r0 = base_r * (1.0 - 0.13 * i)                 # elegant taper toward the tip
                r1 = base_r * (1.0 - 0.13 * (i + 1)) * (0.88 if i == len(chain) - 1 else 1.0)
                if i == len(chain) - 1:
                    b = b + (b - a) * 0.12                     # fingertip pad past the joint
                field.union(sdf.RoundCone(a, b, r0, r1), 0.004 * s)
                if i < len(chain) - 1:
                    field.union(self.E(b, _v(r1 * 1.06, r1 * 1.03, r1 * 1.08)), 0.003 * s)  # knuckle
        return field


def _mix(a, b):
    return 0.5 * (a + b)


def _n(v):
    return v / np.linalg.norm(v)
