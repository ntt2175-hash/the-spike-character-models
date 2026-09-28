#!/usr/bin/env python3
"""Proportion audit: the character's proportion relationships, measured from its data.

    python pipeline/tools/proportion_audit.py <character_id> [--json out.json] [--compare other.json]

Measures (from the skeleton, the body master's cross-sections and the modeled head size) the
relationships a character artist checks, in the order they should be fixed: head-to-body, neck,
shoulders, torso length, ribcage depth, waist, pelvis, legs, arms, hands/feet. Measurements are a
tool, not the authority: judge them together with silhouettes (Test A) and clay (Test B).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "blender")]
import spike_specs as ss  # noqa: E402
from spike_pipeline.modeling import body_master as BM  # noqa: E402
from spike_pipeline.modeling import head_master as HM  # noqa: E402

HAIR_TOP = 0.022                        # hair volume above the skull


def measure(char_id: str) -> dict:
    c = ss.load_character(char_id)
    pr = c["proportions"]
    H = pr["height_m"]
    bones = {b["name"]: b for b in ss.build_character_skeleton(ss.load_spec("skeleton"), c)}
    body = BM.Body(bones, c.get("body_shape", {}), H)
    p, s = body.p, body.s
    hs = float(pr.get("head_scale", 1.12))
    eye = bones["LeftEye"]["head"][2]
    head = HM.Head((0.0, 0.012, eye), 0.0415 * hs / 1.12, hs, c.get("head_shape", {}))   # the modeled head
    chin = head.chin_z
    head_h = head.crown_z - head.chin_z
    head_w = 2 * float(HM.PROFILE[:, 1].max()) * hs * head.p["width"]
    head_total = head_h + HAIR_TOP
    rows = body.torso_rows()
    tz = lambda f: body.tz(f)  # noqa: E731

    def row(f):
        return {k: float(np.interp(tz(f), rows[:, 0], rows[:, j])) for j, k in enumerate(("z", "w", "yf", "yb"))}

    chest, waist, pelvis = row(0.73), row(0.37), row(0.03)
    th = body._thigh_rows()
    hip_x = abs(bones["LeftUpperLeg"]["head"][0])
    hip_line = hip_x + th[1, 1]
    sh_x = abs(bones["LeftUpperArm"]["head"][0])
    deltoid = sh_x + BM.UPPER_ARM[1, 1] * p["deltoid"] * s
    knee_z = bones["LeftLowerLeg"]["head"][2]
    ankle_z = bones["LeftLowerLeg"]["tail"][2]
    shin = body._scaled(BM.SHIN, [0.0, 0.12, 0.25, 0.7, 0.88, 1.0],
                        [p["knee_size"], p["calf_thickness"], p["calf_thickness"], p["calf_thickness"], p["ankle"], p["ankle"]])
    arm = np.linalg.norm(np.subtract(bones["LeftUpperArm"]["tail"], bones["LeftUpperArm"]["head"])) + \
        np.linalg.norm(np.subtract(bones["LeftLowerArm"]["tail"], bones["LeftLowerArm"]["head"]))
    hand = np.linalg.norm(np.subtract(bones["LeftMiddleDistal"]["tail"], bones["LeftHand"]["head"]))
    foot = np.linalg.norm(np.subtract(bones["LeftToes"]["tail"][:2], bones["LeftFoot"]["head"][:2])) + 0.04 * s
    m = {
        "1_heads_tall (height / head incl. hair)": H / head_total,
        "1_head_width / shoulder_span": head_w / (2 * deltoid),
        "2_visible_neck (chin -> neck base) / head_height": (chin - body.neck_z) / head_h,
        "3_shoulder_span / chest_width": (2 * deltoid) / (2 * chest["w"]),
        "3_shoulder_span (m)": 2 * deltoid,
        "4_torso (neck base -> hip joint) / leg (hip joint -> floor)": body.torso_len / body.hip_z,
        "4_torso / head_height": body.torso_len / head_h,
        "5_ribcage depth / width": (chest["yb"] - chest["yf"]) / (2 * chest["w"]),
        "6_waist_width / chest_width": waist["w"] / chest["w"],
        "6_waist depth / width": (waist["yb"] - waist["yf"]) / (2 * waist["w"]),
        "7_hip_line / chest_width": hip_line / chest["w"],
        "7_pelvis depth / hip width": (pelvis["yb"] - pelvis["yf"]) / (2 * hip_line),
        "8_leg_ratio (hip joint / height)": body.hip_z / H,
        "8_thigh (hip->knee) / shin (knee->ankle)": (body.hip_z - knee_z) / (knee_z - ankle_z),
        "8_mid-thigh width / knee width": (th[4, 1] + th[4, 2]) / (th[7, 1] + th[7, 2]),
        "8_calf peak depth / ankle depth": (shin[3, 3] + shin[3, 4]) / (shin[8, 3] + shin[8, 4]),
        "9_arm (shoulder->wrist) / torso": arm / body.torso_len,
        "9_fingertip height / height": (bones["LeftMiddleDistal"]["tail"][2]) / H,
        "10_hand length / head_height": hand / head_h,
        "10_foot length / height": foot / H,
    }
    return {k: round(float(v), 3) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("character")
    ap.add_argument("--json")
    ap.add_argument("--compare")
    a = ap.parse_args()
    m = measure(a.character)
    old = json.loads(Path(a.compare).read_text()) if a.compare else {}
    w = max(len(k) for k in m)
    for k, v in m.items():
        extra = f"   (was {old[k]})" if k in old and old[k] != v else ""
        print(f"{k:<{w}}  {v:7.3f}{extra}")
    if a.json:
        Path(a.json).write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
