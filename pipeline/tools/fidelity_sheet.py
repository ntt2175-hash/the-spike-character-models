#!/usr/bin/env python3
"""Character fidelity test sheet: the 2D reference next to the 3D render, one row per category.

    python pipeline/tools/fidelity_sheet.py --out sheet.png --title "Sara v20" \
        --row "FACE|ref.png|300,120,560,380|render_face.png|note text" \
        --row "JERSEY|ref.png|130,330,597,1000|render_waist.png,render_side.png|note text"

Each row: CATEGORY | reference image | reference crop box (x0,y0,x1,y1, or 'full') | render image(s)
(comma separated) | a short note. Tiles are scaled to a common row height. The point is to judge the
same character, category by category, not just the same palette.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, MUTED = (22, 22, 26), (236, 236, 240), (150, 152, 162)


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load(path, box=None):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    im = bg.convert("RGB")
    if box and box != "full":
        im = im.crop(tuple(int(v) for v in box.split(",")))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Character fidelity test")
    ap.add_argument("--row", action="append", default=[])
    ap.add_argument("--height", type=int, default=420)
    a = ap.parse_args()
    rows = []
    for spec in a.row:
        cat, ref, box, renders, note = (spec.split("|") + [""] * 5)[:5]
        tiles = [("2D", _load(ref, box))] + [("3D", _load(r)) for r in renders.split(",") if r]
        scaled = [(lbl, im.resize((max(1, int(im.width * a.height / im.height)), a.height), Image.LANCZOS)) for lbl, im in tiles]
        rows.append((cat, scaled, note))
    pad, head, label_w = 12, 56, 170
    W = max(label_w + sum(im.width + pad for _, im in r) + pad for _, r, _ in rows)
    H = head + len(rows) * (a.height + 40 + pad)
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 14), a.title, fill=FG, font=_font(26))
    y = head
    for cat, tiles, note in rows:
        d.text((pad, y + 6), cat, fill=FG, font=_font(22))
        x = label_w
        for lbl, im in tiles:
            sheet.paste(im, (x, y))
            d.text((x + 6, y + 6), lbl, fill=(255, 255, 255), font=_font(16))
            x += im.width + pad
        d.text((label_w, y + a.height + 8), note, fill=MUTED, font=_font(16))
        y += a.height + 40 + pad
    sheet.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
