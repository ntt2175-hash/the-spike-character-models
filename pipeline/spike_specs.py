"""Shared, dependency-free access to the pipeline specs.

Used by the spec checker, the pure-Python tools and the Blender scripts
(Blender 4.2-5.2 ship Python 3.11-3.13; nothing here needs third-party
packages). Everything that turns spec data into numbers (skeleton expansion,
proportion mapping, lens math) lives here once, so Blender and the tests
cannot drift apart. The Godot runtime mirrors the lens math in
addons/spike_character/camera/shot_math.gd.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "pipeline" / "specs"
CHARACTERS_DIR = REPO_ROOT / "characters"

TIERS = ("normal", "s", "splus")
SENSOR_HEIGHT_MM = 24.0

# Godot 4 SkeletonProfileHumanoid bone names. Primary bones must match these
# exactly so shared clips retarget with a BoneMap and no renaming.
GODOT_HUMANOID_BONES = (
    "Root", "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftEye", "RightEye", "Jaw",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "LeftThumbMetacarpal", "LeftThumbProximal", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "RightThumbMetacarpal", "RightThumbProximal", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
)

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_spec(name: str) -> dict:
    """Load pipeline/specs/<name>.json."""
    return load_json(SPECS_DIR / f"{name}.json")


def load_all_specs() -> dict:
    return {p.stem: load_json(p) for p in sorted(SPECS_DIR.glob("*.json"))}


def character_ids() -> list[str]:
    return sorted(
        p.parent.name for p in CHARACTERS_DIR.glob("*/character.json")
        if not p.parent.name.startswith("_")
    )


def load_character(char_id: str) -> dict:
    return load_json(CHARACTERS_DIR / char_id / "character.json")


# --------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------
def is_hex_color(value) -> bool:
    return isinstance(value, str) and bool(_HEX.match(value))


def resolve_color(value: str, palette: dict, _depth: int = 0) -> str:
    """Resolve '@token' references against a character palette."""
    if isinstance(value, str) and value.startswith("@"):
        if _depth > 4:
            raise ValueError(f"palette reference loop at {value}")
        key = value[1:]
        if key not in palette:
            raise KeyError(f"palette has no '{key}'")
        return resolve_color(palette[key], palette, _depth + 1)
    if not is_hex_color(value):
        raise ValueError(f"not a #rrggbb color: {value!r}")
    return value


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# --------------------------------------------------------------------------
# Vectors (tiny helpers so Blender and the tests share one implementation)
# --------------------------------------------------------------------------
def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _norm(a):
    n = _len(a)
    return [a[0] / n, a[1] / n, a[2] / n] if n > 1e-12 else [0.0, 0.0, 0.0]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _rotate(v, axis, angle_rad):
    """Rodrigues rotation of v around a unit axis."""
    k = _norm(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    kv = _cross(k, v)
    kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2]
    return [v[i] * c + kv[i] * s + k[i] * kd * (1.0 - c) for i in range(3)]


def angle_between_deg(a, b) -> float:
    na, nb = _norm(a), _norm(b)
    d = max(-1.0, min(1.0, na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]))
    return math.degrees(math.acos(d))


# --------------------------------------------------------------------------
# Skeleton expansion
# --------------------------------------------------------------------------
def mirror_name(name: str) -> str:
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if name.endswith("_R"):
        return name[:-2] + "_L"
    return name


def _mirror_point(p):
    return [-p[0], p[1], p[2]]


def expand_skeleton(skeleton: dict) -> list[dict]:
    """Return the full standard bone list for a height-1.0 reference character.

    Each bone: {name, parent, head, tail, deform, kind}. Order is parent-first.
    """
    bones: list[dict] = []
    by_name: dict[str, dict] = {}

    def add(bone):
        if bone["name"] in by_name:
            raise ValueError(f"duplicate bone {bone['name']}")
        bones.append(bone)
        by_name[bone["name"]] = bone

    for b in skeleton["primary_bones"]:
        add({"name": b["name"], "parent": b["parent"], "head": list(b["head"]),
             "tail": list(b["tail"]), "deform": b.get("deform", True), "kind": "primary"})
    # Mirror every Left* primary bone.
    for b in list(bones):
        if b["name"].startswith("Left"):
            parent = b["parent"]
            add({"name": mirror_name(b["name"]),
                 "parent": mirror_name(parent) if parent and parent.startswith("Left") else parent,
                 "head": _mirror_point(b["head"]), "tail": _mirror_point(b["tail"]),
                 "deform": b["deform"], "kind": "primary"})

    # Fingers, expanded along each hand's axis.
    for side in ("Left", "Right"):
        hand = by_name[f"{side}Hand"]
        axis = _norm(_sub(hand["tail"], hand["head"]))
        lane_dir = [0.0, -1.0, 0.0]  # forward (-Y) = toward the thumb in the A-pose
        # Make lane_dir orthogonal to the hand axis.
        d = sum(lane_dir[i] * axis[i] for i in range(3))
        lane_dir = _norm(_sub(lane_dir, _mul(axis, d)))
        palm_normal = _norm(_cross(axis, lane_dir))
        if side == "Right":
            palm_normal = _mul(palm_normal, -1.0)
        for finger, f in skeleton["fingers"].items():
            if not isinstance(f, dict):
                continue
            start = _add(_add(hand["head"], _mul(axis, f["base"])), _mul(lane_dir, f["lane"]))
            start = _add(start, _mul(palm_normal, f.get("drop", 0.0)))
            # Splay around the palm normal; the thumb also angles forward.
            direction = _rotate(axis, palm_normal, math.radians(-f["splay_deg"]))
            if finger == "thumb":
                direction = _norm(_add(_mul(direction, 0.7), _mul(lane_dir, 0.7)))
            parent = hand["name"]
            head = start
            for bone_suffix, length in zip(f["bones"], f["lengths"]):
                tail = _add(head, _mul(direction, length))
                name = f"{side}{bone_suffix}"
                add({"name": name, "parent": parent, "head": head, "tail": tail,
                     "deform": True, "kind": "finger"})
                parent, head = name, tail

    # Twist bones (mid-segment, parented to the segment they twist).
    for t in skeleton.get("twist_bones", []):
        for side_name in (t["name"], mirror_name(t["name"])):
            parent_name = t["parent"] if side_name == t["name"] else mirror_name(t["parent"])
            p = by_name[parent_name]
            seg = _sub(p["tail"], p["head"])
            head = _add(p["head"], _mul(seg, t["at"]))
            tail = _add(head, _mul(seg, 0.25))
            add({"name": side_name, "parent": parent_name, "head": head, "tail": tail,
                 "deform": t.get("deform", True), "kind": "twist"})

    # Sockets (non-deform attachment points).
    for s in skeleton.get("sockets", []):
        variants = [(s["name"], s["parent"])]
        if s["parent"].startswith("Left"):
            variants.append((mirror_name(s["name"]), mirror_name(s["parent"])))
        for name, parent_name in variants:
            p = by_name[parent_name]
            seg = _sub(p["tail"], p["head"])
            head = _add(p["head"], _mul(seg, s.get("offset_along", 0.5)))
            head = _add(head, [0.0, s.get("offset_forward", 0.0), 0.0])
            tail = _add(head, [0.0, 0.0, 0.02])
            add({"name": name, "parent": parent_name, "head": head, "tail": tail,
                 "deform": False, "kind": "socket"})
    return bones


# --------------------------------------------------------------------------
# Secondary chains
# --------------------------------------------------------------------------
def chain_bone_names(chain: dict) -> list[str]:
    side = f"_{chain['side']}" if chain.get("side") else ""
    return [f"{chain['id']}_{i:02d}{side}" for i in range(1, chain["bones"] + 1)]


def expand_chain(chain: dict) -> list[dict]:
    """Bones for one secondary chain at reference height 1.0 (Blender space).

    The chain bends by curve_deg per bone around the axis perpendicular to its
    direction and world up, giving a natural hanging arc that artists refine.
    """
    names = chain_bone_names(chain)
    if chain.get("points"):
        # Designed chains: the bones follow the modeled mass exactly (bones + 1 joint positions).
        pts = [list(p) for p in chain["points"]]
        out, parent = [], chain["parent"]
        for i, name in enumerate(names):
            out.append({"name": name, "parent": parent, "head": pts[i], "tail": pts[i + 1],
                        "deform": True, "kind": "secondary"})
            parent = name
        return out
    seg_len = chain["length"] / chain["bones"]
    direction = _norm(chain["direction"])
    bend_axis = _cross(direction, [0.0, 0.0, 1.0])
    if _len(bend_axis) < 1e-6:
        bend_axis = [1.0, 0.0, 0.0]
    out = []
    head = list(chain["root"])
    parent = chain["parent"]
    for name in names:
        tail = _add(head, _mul(direction, seg_len))
        out.append({"name": name, "parent": parent, "head": head, "tail": tail,
                    "deform": True, "kind": "secondary"})
        parent, head = name, tail
        direction = _norm(_rotate(direction, bend_axis, math.radians(chain.get("curve_deg", 0.0))))
    return out


# --------------------------------------------------------------------------
# Proportions: reference skeleton -> character skeleton
# --------------------------------------------------------------------------
REF_BREAKS = {"floor": 0.0, "hip_joint": 0.525, "neck_base": 0.812, "head_pivot": 0.895, "crown": 1.0}
_ARM_ROOTS = ("LeftUpperArm", "RightUpperArm")
_LEG_ROOTS = ("LeftUpperLeg", "RightUpperLeg")


def _piecewise(z: float, src: list[float], dst: list[float]) -> float:
    if z <= src[0]:
        return dst[0] + (z - src[0])
    for i in range(len(src) - 1):
        if z <= src[i + 1]:
            t = (z - src[i]) / (src[i + 1] - src[i])
            return dst[i] + t * (dst[i + 1] - dst[i])
    return dst[-1] + (z - src[-1])


def target_breaks(proportions: dict) -> list[float]:
    """Vertical breakpoints (normalized to height 1) for a character."""
    head_units = proportions.get("head_units", 7.2)
    leg = proportions.get("leg_ratio", REF_BREAKS["hip_joint"])
    ref_neck = REF_BREAKS["head_pivot"] - REF_BREAKS["neck_base"]
    neck = ref_neck * proportions.get("neck_length", 1.0)
    # The head pivot (skull base) sits ~73% of a head height below the crown.
    head_above_pivot = 0.73 / head_units
    pivot = 1.0 - head_above_pivot
    neck_base = pivot - neck
    if not (leg < neck_base < pivot < 1.0):
        raise ValueError(f"impossible proportions: leg={leg} neck_base={neck_base} pivot={pivot}")
    return [0.0, leg, neck_base, pivot, 1.0]


def apply_proportions(bones: list[dict], proportions: dict) -> list[dict]:
    """Scale a height-1.0 bone list to a character. Returns new bone dicts in meters."""
    height = proportions["height_m"]
    src = [REF_BREAKS[k] for k in ("floor", "hip_joint", "neck_base", "head_pivot", "crown")]
    dst = target_breaks(proportions)
    shoulder_w = proportions.get("shoulder_width", 1.0)
    hip_w = proportions.get("hip_width", 1.0)
    arm_len = proportions.get("arm_length", 1.0)
    hand_s = proportions.get("hand_scale", 1.0)
    foot_s = proportions.get("foot_scale", 1.0)
    by_name = {b["name"]: b for b in bones}

    def chain_root(name):
        """Return which limb root (if any) this bone hangs from."""
        b = by_name.get(name)
        while b is not None:
            if b["name"] in _ARM_ROOTS or b["name"] in _LEG_ROOTS:
                return b["name"]
            b = by_name.get(b["parent"]) if b["parent"] else None
        return None

    def map_torso(p, x_scale=1.0):
        return [p[0] * x_scale, p[1], _piecewise(p[2], src, dst)]

    out = []
    mapped: dict[str, dict] = {}
    for b in bones:
        root = chain_root(b["name"])
        if root in _ARM_ROOTS:
            r = by_name[root]
            anchor = map_torso(r["head"], shoulder_w)

            def f(p, r=r, anchor=anchor):
                rel = _sub(p, r["head"])
                return _add(anchor, _mul(rel, arm_len))
            head, tail = f(b["head"]), f(b["tail"])
            if b["kind"] == "finger" or b["name"].endswith("Hand"):
                hand = by_name["LeftHand" if b["name"].startswith("Left") else "RightHand"]
                hand_anchor = f(hand["head"])
                head = _add(hand_anchor, _mul(_sub(head, hand_anchor), hand_s))
                tail = _add(hand_anchor, _mul(_sub(tail, hand_anchor), hand_s))
        elif root in _LEG_ROOTS:
            r = by_name[root]
            leg_ratio = dst[1] / src[1]

            ankle_z = by_name["LeftLowerLeg"]["tail"][2]
            hip_src = r["head"][2]
            leg_k = (hip_src * leg_ratio - ankle_z) / (hip_src - ankle_z)

            def g(p, r=r, ankle_z=ankle_z, leg_k=leg_k):
                # Longer legs lengthen the thigh and shin; the ankle and foot keep their anatomical height.
                z = ankle_z + (p[2] - ankle_z) * leg_k if p[2] >= ankle_z else p[2]
                return [r["head"][0] * hip_w + (p[0] - r["head"][0]), p[1], z]
            head, tail = g(b["head"]), g(b["tail"])
            if b["name"].endswith(("Foot", "Toes")) or b["name"].startswith("socket_foot"):
                foot = by_name["LeftFoot" if "Left" in b["name"] or b["name"].endswith("_L") else "RightFoot"]
                ankle = g(foot["head"])
                head = _add(ankle, _mul(_sub(head, ankle), foot_s))
                tail = _add(ankle, _mul(_sub(tail, ankle), foot_s))
        else:
            x_scale = shoulder_w if "Shoulder" in b["name"] else 1.0
            head, tail = map_torso(b["head"], x_scale), map_torso(b["tail"], x_scale)
        nb = dict(b)
        nb["head"] = _mul(head, height)
        nb["tail"] = _mul(tail, height)
        out.append(nb)
        mapped[nb["name"]] = nb
    return out


def build_character_skeleton(skeleton: dict, character: dict) -> list[dict]:
    """Standard skeleton + the character's secondary chains, in meters."""
    base = expand_skeleton(skeleton)
    scaled = apply_proportions(base, character["proportions"])
    height = character["proportions"]["height_m"]
    for chain in character.get("secondary_chains", {}).get("chains", []):
        for b in expand_chain(chain):
            nb = dict(b)
            nb["head"] = _mul(b["head"], height)
            nb["tail"] = _mul(b["tail"], height)
            scaled.append(nb)
    return scaled


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------
def expand_expression(preset: dict, shape_names: set[str]) -> dict[str, float]:
    """Expand side-less shorthand ('eye_narrow' -> eye_narrow_L + eye_narrow_R)."""
    out: dict[str, float] = {}
    for key, weight in preset.items():
        if key in shape_names:
            out[key] = float(weight)
        elif f"{key}_L" in shape_names and f"{key}_R" in shape_names:
            out[f"{key}_L"] = float(weight)
            out[f"{key}_R"] = float(weight)
        else:
            raise KeyError(f"expression references unknown shape '{key}'")
    return out


# --------------------------------------------------------------------------
# Lens / shot math (mirrored in addons/spike_character/camera/shot_math.gd)
# --------------------------------------------------------------------------
def vertical_fov_deg(focal_length_mm: float) -> float:
    return math.degrees(2.0 * math.atan(SENSOR_HEIGHT_MM * 0.5 / focal_length_mm))


def shot_frame_height(shot: dict, character_height_m: float) -> float:
    if "frame_height_m" in shot:
        return float(shot["frame_height_m"])
    return float(shot["frame_height_rel"]) * character_height_m


def shot_distance(shot: dict, character_height_m: float) -> float:
    """Distance from the anchor at which the frame height is filled."""
    half = shot_frame_height(shot, character_height_m) * 0.5
    return half / math.tan(math.radians(vertical_fov_deg(shot["focal_length_mm"])) * 0.5)


def shot_camera_offset(shot: dict, character_height_m: float) -> list[float]:
    """Camera position relative to the anchor, in CHARACTER space (+Y up, +Z forward, +X her left)."""
    d = shot_distance(shot, character_height_m)
    yaw = math.radians(shot.get("yaw_deg", 0.0))
    pitch = math.radians(shot.get("pitch_deg", 0.0))
    horizontal = d * math.cos(pitch)
    return [horizontal * math.sin(yaw), d * math.sin(pitch), horizontal * math.cos(yaw)]


def character_to_blender(v) -> list[float]:
    """Character space (+Y up, +Z forward, +X left) -> Blender (Z up, forward -Y, left +X)."""
    return [v[0], -v[2], v[1]]


def merged_shots(shots_spec: dict, character: dict | None = None) -> dict:
    """Global presets + the character's own shots, with character overrides applied."""
    shots = {k: dict(v) for k, v in shots_spec["presets"].items()}
    if character:
        cam = character.get("camera", {})
        for k, v in cam.get("shots", {}).items():
            shots[k] = dict(v)
        for k, override in cam.get("shot_overrides", {}).items():
            if k in shots:
                shots[k].update(override)
    return shots
