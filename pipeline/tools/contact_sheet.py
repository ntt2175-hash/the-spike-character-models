#!/usr/bin/env python3
"""Compose QC / demo renders into one labeled contact sheet.

    python pipeline/tools/contact_sheet.py <render_dir> --out sheet.png \
        [--reference characters/sara/reference/sara_keyart.jpg] [--title "..."] [--note "..."]

Reads <render_dir>/manifest.json (written by the Blender presentation and demo
scripts): a list of {file, label, mode, ...}. Requires Pillow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (22, 22, 26)
FG = (236, 236, 240)
MUTED = (150, 152, 162)
ACCENT = (95, 208, 242)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose(render_dir: Path, out: Path, reference: Path | None = None, title: str = "", note: str = "",
            columns: int = 4, tile_h: int = 420):
    manifest = json.loads((render_dir / "manifest.json").read_text())
    tiles = []
    if reference is not None and reference.exists():
        tiles.append(("2D key art", Image.open(reference).convert("RGB")))
    for item in manifest:
        img = Image.open(render_dir / item["file"]).convert("RGB")
        label = item.get("label", item["file"])
        if item.get("mode") and item["mode"] != "beauty":
            label = f"{label} [{item['mode']}]"
        tiles.append((label, img))
    scaled = []
    for label, img in tiles:
        w = int(img.width * tile_h / img.height)
        scaled.append((label, img.resize((w, tile_h), Image.LANCZOS)))
    rows, row, row_w = [], [], 0
    max_w = max(sum(im.width for _, im in scaled[i:i + columns]) for i in range(0, len(scaled), columns))
    for label, img in scaled:
        if row and (len(row) >= columns or row_w + img.width > max_w):
            rows.append(row)
            row, row_w = [], 0
        row.append((label, img))
        row_w += img.width
    if row:
        rows.append(row)
    pad, label_h, head_h, foot_h = 12, 30, 70 if title else 0, 44 if note else 0
    width = max(sum(im.width for _, im in r) + pad * (len(r) + 1) for r in rows)
    height = head_h + sum(tile_h + label_h + pad for _ in rows) + pad + foot_h
    sheet = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((pad, 18), title, fill=FG, font=_font(30))
    y = head_h + pad
    for r in rows:
        x = pad
        for label, img in r:
            sheet.paste(img, (x, y))
            d.text((x + 4, y + tile_h + 6), label, fill=MUTED, font=_font(17))
            x += img.width + pad
        y += tile_h + label_h + pad
    if note:
        d.rectangle([0, height - foot_h, width, height], fill=(34, 30, 20))
        d.text((pad, height - foot_h + 12), note, fill=(255, 214, 120), font=_font(19))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--tile-height", type=int, default=420)
    a = ap.parse_args()
    print(compose(a.render_dir, a.out, a.reference, a.title, a.note, a.columns, a.tile_height))


if __name__ == "__main__":
    main()
