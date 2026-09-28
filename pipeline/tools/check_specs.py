#!/usr/bin/env python3
"""Validate every pipeline spec and every characters/*/character.json.

The specs are the contract between Blender, Godot and the art team, so a
broken reference (a shot that names a missing lighting profile, an expression
that drives a shape that does not exist, a chain that exceeds the bone budget)
is caught here, in CI, before anyone opens an engine.

Usage:
    python pipeline/tools/check_specs.py            # all specs + all characters
    python pipeline/tools/check_specs.py --strict   # warnings fail too
Exit code 1 on errors.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spike_specs as ss  # noqa: E402

MOVE_TYPES = {"static", "push_in", "pull_out", "orbit", "rise", "track", "drift"}
SEQUENCE_PHASES = {"cinematic_transition", "character_reveal", "character_motion", "hero_shot",
                   "action_transition", "return_to_gameplay"}
SEQUENCE_EVENTS = {"gameplay_freeze", "action_resolved", "gameplay_resume"}
CAMERA_MODES = {"gameplay", "gameplay_push", "gameplay_follow"}
ANCHOR_ALIASES = {"eyes", "feet", "hit_hand", "aim_hand", "body_center"}
GAZE_TARGETS = {"camera", "ball", "forward", "none"}
SPRING_RANGES = {"stiffness": (0.0, 4.0), "drag": (0.0, 1.0), "gravity": (0.0, 2.0), "radius": (0.0, 0.1),
                 "wind_response": (0.0, 2.0)}


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str):
        self.errors.append(f"[ERROR] {where}: {msg}")

    def warn(self, where: str, msg: str):
        self.warnings.append(f"[warn]  {where}: {msg}")


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# --------------------------------------------------------------------------
def check_skeleton(spec: dict, r: Report) -> list[dict]:
    where = "skeleton.json"
    try:
        bones = ss.expand_skeleton(spec)
    except Exception as exc:  # noqa: BLE001
        r.err(where, f"expansion failed: {exc}")
        return []
    names = {b["name"] for b in bones}
    for b in bones:
        if b["parent"] and b["parent"] not in names:
            r.err(where, f"{b['name']} has unknown parent {b['parent']}")
    missing = [n for n in ss.GODOT_HUMANOID_BONES if n not in names]
    if missing:
        r.err(where, f"missing Godot humanoid bones: {missing}")
    re.compile(spec["conventions"]["secondary_bone_pattern"])
    re.compile(spec["conventions"]["socket_bone_pattern"])
    sock_re = re.compile(spec["conventions"]["socket_bone_pattern"])
    for b in bones:
        if b["kind"] == "socket" and not sock_re.match(b["name"]):
            r.err(where, f"socket {b['name']} does not match socket_bone_pattern")
        seg = ss._len(ss._sub(b["tail"], b["head"]))
        if seg < 1e-4:
            r.err(where, f"{b['name']} has zero length")
    # A-pose sanity: upper arm angle below horizontal.
    by = {b["name"]: b for b in bones}
    ua = by["LeftUpperArm"]
    v = ss._sub(ua["tail"], ua["head"])
    angle = ss.angle_between_deg(v, [1.0, 0.0, 0.0])
    target = spec["conventions"]["arm_angle_deg"]
    if abs(angle - target) > spec["conventions"]["arm_angle_tolerance_deg"]:
        r.err(where, f"LeftUpperArm is {angle:.1f} deg below horizontal, convention is {target}")
    return bones


def check_shapes(spec: dict, bone_names: set[str], r: Report) -> set[str]:
    where = "shapes.json"
    names = [s["name"] for s in spec["facial"]["shapes"]]
    if len(names) != len(set(names)):
        r.err(where, "duplicate facial shape names")
    for s in spec["facial"]["shapes"]:
        if s["tier"] not in ss.TIERS:
            r.err(where, f"{s['name']} has invalid tier {s['tier']}")
    shape_set = set(names)
    for pid, preset in spec["expression_presets"].items():
        if pid == "note":
            continue
        try:
            ss.expand_expression(preset, shape_set)
        except KeyError as exc:
            r.err(where, f"expression '{pid}': {exc}")
    for c in spec["correctives"]["shapes"]:
        for seg in ("segment_a", "segment_b"):
            for bone in c[seg]:
                if bone not in bone_names:
                    r.err(where, f"corrective {c['name']} references unknown bone {bone}")
        if not c["from_deg"] < c["to_deg"]:
            r.err(where, f"corrective {c['name']} from_deg must be < to_deg")
    return shape_set


def check_materials(spec: dict, r: Report):
    where = "materials.json"
    shader_dir = ss.REPO_ROOT / "addons" / "spike_character" / "shaders"
    for fam_id, fam in spec["families"].items():
        if not (shader_dir / fam["shader"]).exists():
            r.err(where, f"family '{fam_id}' shader {fam['shader']} not found in {shader_dir.relative_to(ss.REPO_ROOT)}")
    for slot, info in spec["slots"].items():
        if info["family"] not in spec["families"]:
            r.err(where, f"slot '{slot}' uses unknown family {info['family']}")
    re.compile(spec["naming"]["material_pattern"])


def check_lighting(spec: dict, r: Report) -> set[str]:
    where = "lighting.json"
    tokens = set(spec["palette_tokens"]["tokens"])
    for pid, p in spec["profiles"].items():
        for part in ("key", "fill", "rim"):
            if part not in p:
                r.err(where, f"profile {pid} missing '{part}'")
                continue
            col = p[part]["color"]
            if not (ss.is_hex_color(col) or col in tokens):
                r.err(where, f"profile {pid}.{part}.color invalid: {col}")
        bg = p["environment"]["background"]
        if not (ss.is_hex_color(bg) or bg in tokens):
            r.err(where, f"profile {pid}.environment.background invalid: {bg}")
        key_e, fill_e = p["key"]["energy"], p["fill"]["energy"]
        if key_e > 0.2 and fill_e > 0.45 * key_e + 1e-6:
            r.err(where, f"profile {pid}: fill energy {fill_e} exceeds 0.45 x key ({key_e}); shadows will flatten")
    return set(spec["profiles"].keys())


def _check_shot(where: str, sid: str, shot: dict, anchors: set[str], easing: set[str], r: Report):
    for field in ("anchor", "yaw_deg", "pitch_deg", "focal_length_mm"):
        if field not in shot:
            r.err(where, f"shot {sid} missing {field}")
    if ("frame_height_m" in shot) == ("frame_height_rel" in shot):
        r.err(where, f"shot {sid} needs exactly one of frame_height_m / frame_height_rel")
    if shot.get("anchor") and shot["anchor"] not in anchors:
        r.err(where, f"shot {sid} anchor '{shot['anchor']}' is not a bone or alias")
    move = shot.get("move", {"type": "static"})
    if move.get("type") not in MOVE_TYPES:
        r.err(where, f"shot {sid} move type {move.get('type')} invalid")
    if move.get("ease") and move["ease"] not in easing:
        r.err(where, f"shot {sid} ease {move['ease']} invalid")
    so = shot.get("screen_offset", [0, 0])
    if any(abs(v) > 1.0 for v in so):
        r.err(where, f"shot {sid} screen_offset outside -1..1")
    if not (10.0 <= shot.get("focal_length_mm", 50) <= 300.0):
        r.err(where, f"shot {sid} focal length out of range")


def check_shots(spec: dict, anchors: set[str], lighting_ids: set[str], pose_ids: set[str], r: Report):
    where = "shots.json"
    easing = set(spec["easing"])
    for sid, shot in spec["presets"].items():
        _check_shot(where, sid, shot, anchors, easing, r)
    for t in spec["transitions"].values():
        if t["ease"] not in easing:
            r.err(where, f"transition ease {t['ease']} invalid")
    ids = [q["id"] for q in spec["qc_render_set"]]
    if len(ids) != len(set(ids)):
        r.err(where, "duplicate qc_render_set ids")
    for q in spec["qc_render_set"]:
        if q["shot"] not in spec["presets"]:
            r.warn(where, f"qc {q['id']} uses shot '{q['shot']}' that each character must define")
        if q.get("lighting") and q["lighting"] not in lighting_ids:
            r.err(where, f"qc {q['id']} lighting '{q['lighting']}' unknown")
        if q["pose"] not in pose_ids:
            r.err(where, f"qc {q['id']} pose '{q['pose']}' unknown")
    gc = spec["gameplay_camera"]
    if not gc["min_distance_m"] <= gc["base_distance_m"] <= gc["max_distance_m"]:
        r.err(where, "gameplay_camera base distance outside min/max")
    return easing


def check_animations(spec: dict, r: Report) -> tuple[set[str], set[str]]:
    where = "animations.json"
    clip_re = re.compile(spec["naming"]["clip_pattern"])
    events = set(spec["events"].keys())
    ids = [c["id"] for c in spec["clips"]]
    if len(ids) != len(set(ids)):
        r.err(where, "duplicate clip ids")
    for c in spec["clips"]:
        if not clip_re.match(c["id"]):
            r.err(where, f"clip id {c['id']} breaks clip_pattern")
        if c["tier"] not in ss.TIERS:
            r.err(where, f"clip {c['id']} invalid tier")
        _check_events(where, c["id"], c["events"], events, r)
    for pid, p in spec["personality_params"].items():
        if pid == "note":
            continue
        if not p["min"] <= p["default"] <= p["max"]:
            r.err(where, f"personality {pid} default outside range")
    return set(ids), set(k for k in spec["poses"] if k != "note")


def _check_events(where, clip_id, ev: dict, known: set[str], r: Report):
    order = ["windup", "commit", "contact", "recover"]
    for name, t in ev.items():
        base = re.sub(r"_\d+$", "", name)
        if base not in known:
            r.err(where, f"clip {clip_id} event '{name}' is not a standard event")
        if not 0.0 <= float(t) <= 1.0:
            r.err(where, f"clip {clip_id} event '{name}' time {t} outside 0..1 (normalized)")
    seq = [ev[k] for k in order if k in ev]
    if seq != sorted(seq):
        r.err(where, f"clip {clip_id} events out of order (windup < commit < contact < recover)")


def check_budgets(spec: dict, r: Report):
    where = "budgets.json"
    lods = spec["lods"]
    for tier in ss.TIERS:
        seq = [lods[l]["triangles"][tier] for l in ("LOD0", "LOD1", "LOD2")]
        if seq != sorted(seq, reverse=True):
            r.err(where, f"triangle budgets must decrease LOD0 > LOD1 > LOD2 for tier {tier}")
    for lod_id, lod in lods.items():
        for k, px in lod["texture_px"].items():
            if not _is_pow2(px):
                r.err(where, f"{lod_id} texture {k} = {px} is not a power of two")
    th = [lods[l]["min_screen_height_px"] for l in ("LOD0", "LOD1", "LOD2")]
    if th != sorted(th, reverse=True):
        r.err(where, "LOD switch heights must decrease")


# --------------------------------------------------------------------------
def check_character(char_id: str, specs: dict, ctx: dict, r: Report):
    where = f"characters/{char_id}/character.json"
    c = ss.load_character(char_id)
    if c.get("id") != char_id:
        r.err(where, f"id '{c.get('id')}' does not match folder '{char_id}'")
    if c.get("tier") not in ss.TIERS:
        r.err(where, f"invalid tier {c.get('tier')}")
    tier = c.get("tier", "normal")
    palette = c.get("palette", {})
    for k, v in palette.items():
        if k == "note":
            continue
        try:
            ss.resolve_color(v, palette)
        except (KeyError, ValueError) as exc:
            r.err(where, f"palette.{k}: {exc}")
    for token in specs["lighting"]["palette_tokens"]["tokens"]:
        if token[1:] not in palette:
            r.err(where, f"palette must define '{token[1:]}' for lighting token {token}")

    # Skeleton + chains
    try:
        bones = ss.build_character_skeleton(specs["skeleton"], c)
    except Exception as exc:  # noqa: BLE001
        r.err(where, f"skeleton build failed: {exc}")
        bones = []
    names = [b["name"] for b in bones]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})
        r.err(where, f"duplicate bones after adding chains: {dup}")
    bone_set = set(names)
    sec_re = re.compile(specs["skeleton"]["conventions"]["secondary_bone_pattern"])
    families = specs["skeleton"]["secondary_chain_standard"]["families"]
    chains = c.get("secondary_chains", {}).get("chains", [])
    for ch in chains:
        fam = ch["id"].split("_")[0]
        if fam not in families:
            r.err(where, f"chain {ch['id']} family '{fam}' unknown")
            continue
        if ch["bones"] > families[fam]["max_bones_per_chain"]:
            r.err(where, f"chain {ch['id']} has {ch['bones']} bones, max {families[fam]['max_bones_per_chain']}")
        for bn in ss.chain_bone_names(ch):
            if not sec_re.match(bn):
                r.err(where, f"chain bone {bn} breaks secondary_bone_pattern")
        if ch["parent"] not in ctx["bone_names"]:
            r.err(where, f"chain {ch['id']} parent {ch['parent']} unknown")
        for k, (lo, hi) in SPRING_RANGES.items():
            if k in ch and not lo <= ch[k] <= hi:
                r.err(where, f"chain {ch['id']} {k}={ch[k]} outside [{lo}, {hi}]")
        if ch.get("points") and len(ch["points"]) != ch["bones"] + 1:
            r.err(where, f"chain {ch['id']} has {len(ch['points'])} points for {ch['bones']} bones (needs bones + 1)")
        if ch.get("keep_at_lod2") and not ch.get("keep_at_lod1", True):
            r.err(where, f"chain {ch['id']} kept at LOD2 but not LOD1")
    for col in c.get("secondary_chains", {}).get("colliders", []):
        if col["bone"] not in bone_set:
            r.err(where, f"collider bone {col['bone']} unknown")
    deform = sum(1 for b in bones if b["deform"])
    max_def = specs["budgets"]["lods"]["LOD0"]["max_deform_bones"]
    if deform > max_def:
        r.err(where, f"{deform} deform bones exceeds LOD0 budget {max_def}")
    lod2_def = sum(1 for b in bones if b["deform"] and b["kind"] != "secondary") + sum(
        ch["bones"] for ch in chains if ch.get("keep_at_lod2"))
    if lod2_def > specs["budgets"]["lods"]["LOD2"]["max_deform_bones"]:
        r.err(where, f"LOD2 active deform bones {lod2_def} exceeds budget")
    ctx.setdefault("report", {})[char_id] = {"deform_bones": deform, "lod2_deform_bones": lod2_def,
                                               "total_bones": len(bones)}

    # Proportions
    p = c["proportions"]
    try:
        ss.target_breaks(p)
    except ValueError as exc:
        r.err(where, f"proportions: {exc}")
    if not 1.3 <= p["height_m"] <= 2.1:
        r.err(where, "height_m outside 1.3..2.1 m")
    body_spec = specs.get("body", {})
    for k, v in p.items():
        rng = body_spec.get("skeleton_params", {}).get(k, {}).get("range")
        if rng and isinstance(v, (int, float)) and not rng[0] <= v <= rng[1]:
            r.err(where, f"proportions.{k}={v} outside {rng}")
    shape_params = body_spec.get("shape_params", {})
    for k, v in c.get("body_shape", {}).items():
        if k == "note":
            continue
        if k not in shape_params:
            r.err(where, f"body_shape.{k} is not a body master parameter")
        elif not shape_params[k]["range"][0] <= v <= shape_params[k]["range"][1]:
            r.err(where, f"body_shape.{k}={v} outside {shape_params[k]['range']}")

    # Face / expressions
    shape_set = ctx["shapes"]
    required = [s["name"] for s in specs["shapes"]["facial"]["shapes"]
                if ss.TIERS.index(s["tier"]) <= ss.TIERS.index(tier)]
    ctx["report"][char_id]["required_facial_shapes"] = len(required)
    expressions = dict((k, v) for k, v in specs["shapes"]["expression_presets"].items() if k != "note")
    for eid, preset in c.get("expressions", {}).items():
        try:
            ss.expand_expression(preset, shape_set)
        except KeyError as exc:
            r.err(where, f"expression {eid}: {exc}")
        expressions[eid] = preset
    face = c.get("face", {})
    if face.get("default_expression") and face["default_expression"] not in expressions:
        r.err(where, f"default_expression {face['default_expression']} unknown")
    for k, v in face.get("eye_shader", {}).items():
        if isinstance(v, str):
            try:
                ss.resolve_color(v, palette)
            except (KeyError, ValueError) as exc:
                r.err(where, f"face.eye_shader.{k}: {exc}")

    # Personality + animation
    pp = specs["animations"]["personality_params"]
    for k, v in c.get("personality", {}).get("params", {}).items():
        if k not in pp:
            r.err(where, f"personality param {k} unknown")
        elif not pp[k]["min"] <= v <= pp[k]["max"]:
            r.err(where, f"personality {k}={v} outside [{pp[k]['min']}, {pp[k]['max']}]")
    clip_re = re.compile(specs["animations"]["naming"]["clip_pattern"])
    anim = c.get("animation", {})
    for std, actual in anim.get("overrides", {}).items():
        if std not in ctx["clip_ids"]:
            r.err(where, f"override for unknown clip {std}")
        if not clip_re.match(actual) or not actual.startswith(std + "__"):
            r.err(where, f"override name {actual} must be '{std}__{char_id}'")
    for clip, ev in anim.get("event_overrides", {}).items():
        _check_events(where, clip, ev, ctx["event_names"], r)

    # Cameras
    shots = ss.merged_shots(specs["shots"], c)
    for sid, shot in c.get("camera", {}).get("shots", {}).items():
        _check_shot(where, sid, shot, ctx["anchors"] | bone_set, ctx["easing"], r)
    for sid in c.get("camera", {}).get("shot_overrides", {}):
        if sid not in specs["shots"]["presets"]:
            r.err(where, f"shot_override for unknown preset {sid}")
    for q in specs["shots"]["qc_render_set"]:
        if q["shot"] not in shots:
            r.err(where, f"qc render {q['id']} needs shot '{q['shot']}' (define it in camera.shots)")

    # VFX / sound
    vfx = c.get("vfx", {}).get("cues", {})
    for cue_id, cue in vfx.items():
        path = cue["scene"].replace("res://", "")
        if not (ss.REPO_ROOT / path).exists():
            r.warn(where, f"vfx cue {cue_id}: scene {cue['scene']} not authored yet")
    sounds = set(c.get("sound", {}).get("cues", []))

    # Special sequence
    if "special" in c:
        _check_special(where, c, shots, expressions, sounds, vfx, ctx, bone_set, r)
    elif tier == "splus":
        r.err(where, "S+ characters must define a special sequence")


def _check_special(where, c, shots, expressions, sounds, vfx, ctx, bone_set, r: Report):
    sp = c["special"]
    shot_ids = [s["id"] for s in sp["shots"]]
    if len(shot_ids) != len(set(shot_ids)):
        r.err(where, "special: duplicate shot ids")
    by_id = {s["id"]: s for s in sp["shots"]}
    shader_params = {p["name"] for p in ctx["shapes_spec"]["shader_driven"]["params"]}
    anchors = ctx["anchors"] | bone_set | {"ball"}
    for variant, seq in sp["variants"].items():
        for sid in seq:
            if sid not in by_id:
                r.err(where, f"special variant {variant} references unknown shot {sid}")
    for s in sp["shots"]:
        w = f"{where} special.{s['id']}"
        if s.get("phase") not in SEQUENCE_PHASES:
            r.err(w, f"phase {s.get('phase')} invalid")
        cam = s.get("camera", {})
        if "shot" in cam and cam["shot"] not in shots:
            r.err(w, f"camera shot {cam['shot']} unknown")
        if "mode" in cam and cam["mode"] not in CAMERA_MODES:
            r.err(w, f"camera mode {cam['mode']} invalid")
        if "then" in cam and cam["then"] not in CAMERA_MODES:
            r.err(w, f"camera 'then' {cam['then']} invalid")
        light = s.get("lighting", {})
        if light and light["profile"] not in ctx["lighting_ids"]:
            r.err(w, f"lighting profile {light['profile']} unknown")
        dur = s["duration_s"]
        for track in ("face", "gaze", "shader", "hair", "vfx", "sound", "time_scale", "events", "shake", "overlay"):
            for cue in s.get(track, []):
                if not 0.0 <= cue["t"] <= dur + 1e-6:
                    r.err(w, f"{track} cue at t={cue['t']} outside shot duration {dur}")
        for cue in s.get("face", []):
            if cue["expression"] not in expressions:
                r.err(w, f"expression {cue['expression']} unknown")
        for cue in s.get("gaze", []):
            if cue["target"] not in GAZE_TARGETS and cue["target"] not in bone_set:
                r.err(w, f"gaze target {cue['target']} invalid")
        for cue in s.get("shader", []):
            if cue["param"] not in shader_params:
                r.err(w, f"shader param {cue['param']} unknown")
        for cue in s.get("vfx", []):
            if cue["cue"] not in vfx:
                r.err(w, f"vfx cue {cue['cue']} not declared in vfx.cues")
            if cue.get("anchor") and cue["anchor"] not in anchors:
                r.err(w, f"vfx anchor {cue['anchor']} invalid")
        for cue in s.get("sound", []):
            if cue["cue"] not in sounds:
                r.err(w, f"sound cue {cue['cue']} not declared in sound.cues")
        for cue in s.get("events", []):
            if cue["name"] not in SEQUENCE_EVENTS:
                r.err(w, f"event {cue['name']} invalid")
        for cue in s.get("time_scale", []):
            if not 0.05 <= cue["value"] <= 1.0:
                r.err(w, f"time_scale {cue['value']} outside 0.05..1")
        body = s.get("body")
        if body and not (body["clip"].startswith("sp_") or body["clip"] in ctx["clip_ids"]):
            r.err(w, f"body clip {body['clip']} is neither a standard clip nor sp_*")

    def total(seq, short):
        t = 0.0
        for sid in seq:
            shot = by_id[sid]
            t += shot["duration_s"] - (shot.get("short", {}).get("start_s", 0.0) if short else 0.0)
        return t
    rules = sp.get("rules", {})
    full = total(sp["variants"].get("full", []), False)
    short = total(sp["variants"].get("short", []), True)
    ctx["report"][c["id"]]["special_full_s"] = round(full, 3)
    ctx["report"][c["id"]]["special_short_s"] = round(short, 3)
    if full > rules.get("max_full_s", 99):
        r.err(where, f"special full variant {full:.2f}s exceeds max_full_s {rules['max_full_s']}")
    if short > rules.get("max_short_s", 99):
        r.err(where, f"special short variant {short:.2f}s exceeds max_short_s {rules['max_short_s']}")
    fired = {e["name"] for s in sp["shots"] for e in s.get("events", [])}
    for required in SEQUENCE_EVENTS:
        if required not in fired:
            r.err(where, f"special never fires '{required}'")
    if sp["skip"]["jump_to"] not in by_id:
        r.err(where, "special skip.jump_to unknown")
    # The skip target must still resolve the action.
    jt = by_id.get(sp["skip"]["jump_to"], {})
    if "action_resolved" not in {e["name"] for e in jt.get("events", [])}:
        r.err(where, "skip.jump_to shot must fire action_resolved so skipping never changes the outcome")


# --------------------------------------------------------------------------
def run(strict: bool = False, quiet: bool = False) -> Report:
    r = Report()
    specs = ss.load_all_specs()
    for name, spec in specs.items():
        if spec.get("spec") != f"spike.{name}":
            r.err(f"{name}.json", f"'spec' field must be 'spike.{name}'")
    bones = check_skeleton(specs["skeleton"], r)
    bone_names = {b["name"] for b in bones}
    shape_set = check_shapes(specs["shapes"], bone_names, r)
    check_materials(specs["materials"], r)
    lighting_ids = check_lighting(specs["lighting"], r)
    clip_ids, pose_ids = check_animations(specs["animations"], r)
    anchors = bone_names | ANCHOR_ALIASES
    easing = check_shots(specs["shots"], anchors, lighting_ids, pose_ids, r)
    check_budgets(specs["budgets"], r)
    ctx = {"bone_names": bone_names, "shapes": shape_set, "shapes_spec": specs["shapes"],
           "lighting_ids": lighting_ids, "clip_ids": clip_ids, "event_names": set(specs["animations"]["events"]),
           "anchors": anchors, "easing": easing}
    for char_id in ss.character_ids():
        check_character(char_id, specs, ctx, r)
    if not quiet:
        for line in r.errors + r.warnings:
            print(line)
        for cid, info in ctx.get("report", {}).items():
            print(f"[info]  {cid}: " + ", ".join(f"{k}={v}" for k, v in info.items()))
        print(f"{len(r.errors)} error(s), {len(r.warnings)} warning(s)")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()
    r = run(args.strict)
    return 1 if r.errors or (args.strict and r.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
