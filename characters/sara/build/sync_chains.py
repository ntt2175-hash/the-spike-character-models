"""Regenerate Sara's hair and ribbon chains in character.json from the hair design.

The hairstyle is the source of truth: every chain's joints are resampled from the mass it moves
(hair.chain_definitions). Spring settings of chains that already exist are preserved. Run after any
change to hair.py, before build_sara.py:

    <blender python> characters/sara/build/sync_chains.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402,F401  (sets up sys.path and bpy)

import hair  # noqa: E402

path = common.REPO / "characters" / common.CHAR / "character.json"
text = path.read_text()
data = json.loads(text)
height = data["proportions"]["height_m"]
chains = data["secondary_chains"]["chains"]
designed = hair.chain_definitions(height, [c for c in chains if c["id"].startswith(("hair_", "ribbon_"))])

# Rewrite only the chain lines, one chain per line, keeping the file's hand-kept layout elsewhere.
lines = text.splitlines()
start = next(i for i, ln in enumerate(lines) if ln.strip() == '"chains": [')
end = next(i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("]"))
indent = " " * (len(lines[start + 1]) - len(lines[start + 1].lstrip()))
others = [c for c in chains if not c["id"].startswith(("hair_", "ribbon_"))]
body = [indent + json.dumps(c, separators=(", ", ": ")) for c in designed + others]
body = [ln + "," for ln in body[:-1]] + [body[-1]]
out = "\n".join(lines[:start + 1] + body + lines[end:]) + "\n"
json.loads(out)
path.write_text(out)
total = sum(c["bones"] for c in designed)
common.log(f"synced {len(designed)} hair/ribbon chains ({total} bones) into {path.name}")
for c in designed:
    common.log(f"  {c['id']}{'_' + c['side'] if c.get('side') else ''}: {c['bones']} bones, {c['length'] * height:.3f} m")
