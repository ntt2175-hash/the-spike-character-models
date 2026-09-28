#!/usr/bin/env python3
"""Black-silhouette comparison: 2D reference art vs 3D renders.

    python pipeline/tools/silhouette_compare.py --reference characters/sara/reference/sara_keyart.jpg \
        --views <render_dir>/stand --match <render_dir>/keyart --out sheet.png [--title "..."]

--views : render dir (manifest.json + transparent PNGs from render_sara.py 'silhouette') whose alpha
          channels become silhouettes (front / side / rear / 3/4 ...).
--match : render dir of the key-art-match camera + pose; drawn next to the reference and overlaid on
          it (reference black, 3D red, overlap dark red) after fitting both to the same bounding box.
The reference silhouette is everything that is not paper white. All panels are scaled to the same
figure height so proportions (not pixel sizes) are compared. Each panel also prints width profiles:
the figure's width at fixed fractions of its height, relative to its height.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (236, 236, 240)
FG = (18, 18, 22)
PANEL_H = 900
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def mask_from_render(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return np.asarray(im)[..., 3] > 40


def mask_from_art(path: Path, white=232) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    m = a.min(axis=2) < white
    # Close pinholes inside the figure (highlights close to white) with a small dilate/erode.
    from PIL import ImageFilter
    im = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return np.asarray(im) > 127


def crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_panel(mask: np.ndarray, height=PANEL_H, color=FG) -> Image.Image:
    m = crop(mask)
    scale = height / m.shape[0]
    w = max(1, int(round(m.shape[1] * scale)))
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((w, height), Image.LANCZOS)
    out = Image.new("RGB", (w, height), BG)
    out.paste(Image.new("RGB", (w, height), color), (0, 0), im)
    return out


def width_profile(mask: np.ndarray, fracs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Filled width (count of silhouette pixels per row, all runs) / figure height at height fractions
    measured from the top."""
    m = crop(mask)
    h = m.shape[0]
    return {f: float(m[min(int(f * h), h - 1)].sum()) / h for f in fracs}


def overlay(ref: np.ndarray, mod: np.ndarray, height=PANEL_H) -> Image.Image:
    a, b = crop(ref), crop(mod)
    wa = int(round(a.shape[1] * height / a.shape[0]))
    wb = int(round(b.shape[1] * height / b.shape[0]))
    w = max(wa, wb)
    A = np.zeros((height, w), bool)
    B = np.zeros((height, w), bool)
    A[:, (w - wa) // 2:(w - wa) // 2 + wa] = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((wa, height))) > 127
    B[:, (w - wb) // 2:(w - wb) // 2 + wb] = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((wb, height))) > 127
    out = np.full((height, w, 3), BG, np.uint8)
    out[A & ~B] = (30, 30, 34)
    out[B & ~A] = (220, 50, 50)
    out[A & B] = (120, 20, 26)
    iou = (A & B).sum() / max((A | B).sum(), 1)
    return Image.fromarray(out), float(iou)


def load_dir(d: Path):
    man = json.loads((d / "manifest.json").read_text())
    return [(e["label"], mask_from_render(d / e["file"])) for e in man]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--views")
    ap.add_argument("--match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Silhouette comparison: 2D reference vs 3D")
    a = ap.parse_args()
    ref = mask_from_art(Path(a.reference))
    panels = [("2D reference", to_panel(ref), width_profile(ref))]
    iou = None
    if a.match:
        for label, m in load_dir(Path(a.match)):
            panels.append((f"3D {label}", to_panel(m), width_profile(m)))
            ov, iou = overlay(ref, m)
            panels.append((f"overlay (IoU {iou:.2f})", ov, None))
    if a.views:
        for label, m in load_dir(Path(a.views)):
            panels.append((f"3D {label}", to_panel(m), width_profile(m)))
    pad, top, bottom = 24, 70, 150
    W = sum(p.width for _, p, _ in panels) + pad * (len(panels) + 1)
    sheet = Image.new("RGB", (W, top + PANEL_H + bottom), (250, 250, 252))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 18), a.title, fill=(20, 20, 24), font=_font(30))
    x = pad
    for label, im, prof in panels:
        sheet.paste(im, (x, top))
        d.text((x, top + PANEL_H + 10), label, fill=(20, 20, 24), font=_font(20))
        if prof:
            txt = "  ".join(f"{int(f * 100)}%:{v:.3f}" for f, v in list(prof.items())[::2])
            d.text((x, top + PANEL_H + 40), "width/height at", fill=(110, 110, 120), font=_font(14))
            for i, (f, v) in enumerate(prof.items()):
                d.text((x + (i % 4) * 70, top + PANEL_H + 60 + (i // 4) * 20), f"{int(f * 100)}% {v:.3f}", fill=(80, 80, 90),
                       font=_font(14))
        x += im.width + pad
    sheet.save(a.out)
    print(a.out, f"IoU={iou:.3f}" if iou is not None else "")


if __name__ == "__main__":
    main()
