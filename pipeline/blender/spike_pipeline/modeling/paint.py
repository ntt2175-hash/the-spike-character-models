"""Texture painting in physical units (millimeters), for anime features and prints.

A Canvas maps a rectangle of the model (in mm) to pixels, supersamples for
anti-aliasing, and composites layers with alpha. Shapes are authored as
curves (Catmull-Rom through points) and variable-width strokes, which is how
anime eye lines, lashes and brows are drawn: tapered, confident strokes, not
blurry blobs.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def catmull(points, samples=24, closed=False):
    """Catmull-Rom spline through points -> dense polyline (list of (x, y))."""
    P = [np.asarray(p, dtype=np.float64) for p in points]
    if len(P) < 2:
        return [tuple(p) for p in P]
    if closed:
        P = [P[-1]] + P + [P[0], P[1]]
    else:
        P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(samples):
            t = k / samples
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(q[0]), float(q[1])))
    if not closed:
        out.append((float(P[-2][0]), float(P[-2][1])))
    return out


class Canvas:
    def __init__(self, width_mm, height_mm, px, origin_mm=(0.0, 0.0), supersample=2, background=(0, 0, 0, 0)):
        """px = output width in pixels. origin_mm = mm coordinate of the canvas' bottom-left corner."""
        self.w_mm, self.h_mm = width_mm, height_mm
        self.out_w = int(px)
        self.out_h = int(round(px * height_mm / width_mm))
        self.ss = supersample
        self.W, self.H = self.out_w * supersample, self.out_h * supersample
        self.ox, self.oy = origin_mm
        self.scale = self.W / width_mm
        self.rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
        self.rgba[..., :] = np.array(background, dtype=np.float32)

    # coordinates ------------------------------------------------------------
    def px(self, x, y):
        return ((x - self.ox) * self.scale, (self.h_mm - (y - self.oy)) * self.scale)

    def grid_mm(self):
        xs = self.ox + (np.arange(self.W) + 0.5) / self.scale
        ys = self.oy + self.h_mm - (np.arange(self.H) + 0.5) / self.scale
        return np.meshgrid(xs, ys)

    # masks ------------------------------------------------------------------
    def mask_polygon(self, pts_mm, blur_mm=0.0):
        m = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(m).polygon([self.px(*p) for p in pts_mm], fill=255)
        if blur_mm > 0:
            m = m.filter(ImageFilter.GaussianBlur(blur_mm * self.scale))
        return np.asarray(m, dtype=np.float32) / 255.0

    def mask_ellipse(self, cx, cy, rx, ry, angle_deg=0.0, blur_mm=0.0, n=96):
        a = math.radians(angle_deg)
        pts = []
        for k in range(n):
            t = 2 * math.pi * k / n
            x, y = rx * math.cos(t), ry * math.sin(t)
            pts.append((cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)))
        return self.mask_polygon(pts, blur_mm)

    def mask_stroke(self, pts_mm, widths_mm, blur_mm=0.0, samples=16, cap=True):
        """Variable-width stroke along a Catmull-Rom curve. widths: per control point or a callable(u)."""
        line = catmull(pts_mm, samples)
        n = len(line)
        if callable(widths_mm):
            ws = [widths_mm(i / (n - 1)) for i in range(n)]
        else:
            ws = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(widths_mm)), widths_mm)
        left, right = [], []
        for i, (x, y) in enumerate(line):
            x0, y0 = line[max(i - 1, 0)]
            x1, y1 = line[min(i + 1, n - 1)]
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln, dx / ln
            h = ws[i] * 0.5
            left.append((x + nx * h, y + ny * h))
            right.append((x - nx * h, y - ny * h))
        poly = left + right[::-1]
        m = self.mask_polygon(poly, blur_mm)
        return m

    def stroke_polygon(self, pts_mm, widths_mm, samples=16):
        """Polygon (mm) of a variable-width stroke, for batching many strokes into one mask."""
        line = catmull(pts_mm, samples)
        n = len(line)
        ws = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(widths_mm)), widths_mm)
        left, right = [], []
        for i, (x, y) in enumerate(line):
            x0, y0 = line[max(i - 1, 0)]
            x1, y1 = line[min(i + 1, n - 1)]
            dx, dy = x1 - x0, y1 - y0
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln, dx / ln
            h = ws[i] * 0.5
            left.append((x + nx * h, y + ny * h))
            right.append((x - nx * h, y - ny * h))
        return left + right[::-1]

    def mask_polygons(self, polys_mm, blur_mm=0.0):
        m = Image.new("L", (self.W, self.H), 0)
        d = ImageDraw.Draw(m)
        for poly in polys_mm:
            d.polygon([self.px(*p) for p in poly], fill=255)
        if blur_mm > 0:
            m = m.filter(ImageFilter.GaussianBlur(blur_mm * self.scale))
        return np.asarray(m, dtype=np.float32) / 255.0

    def polar(self, cx, cy, sx=1.0, sy=1.0):
        """(r, theta) of every pixel around (cx, cy), with anisotropic scaling (ellipse -> circle)."""
        X, Y = self.grid_mm()
        dx, dy = (X - cx) / sx, (Y - cy) / sy
        return np.sqrt(dx * dx + dy * dy), np.arctan2(dy, dx)

    def mask_gauss(self, cx, cy, sx, sy, angle_deg=0.0):
        X, Y = self.grid_mm()
        a = math.radians(angle_deg)
        dx, dy = X - cx, Y - cy
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        return np.exp(-(u * u) / (2 * sx * sx) - (v * v) / (2 * sy * sy)).astype(np.float32)

    # painting ---------------------------------------------------------------
    def paint(self, mask, color, alpha=1.0, mode="normal"):
        """Composite a color (hex or (H,W,3) array) through a mask (H,W) 0..1."""
        col = hex_rgb(color) if isinstance(color, str) else np.asarray(color, dtype=np.float32)
        a = np.clip(mask * alpha, 0.0, 1.0)[..., None]
        base = self.rgba[..., :3]
        if mode == "multiply":
            src = base * col
        elif mode == "screen":
            src = 1.0 - (1.0 - base) * (1.0 - col)
        else:
            src = np.broadcast_to(col, base.shape)
        self.rgba[..., :3] = base * (1.0 - a) + src * a
        self.rgba[..., 3:4] = self.rgba[..., 3:4] * (1.0 - a) + a
        return self

    def paint_alpha_only(self, mask, alpha=1.0):
        self.rgba[..., 3] = np.maximum(self.rgba[..., 3], np.clip(mask * alpha, 0, 1))
        return self

    def erase(self, mask, amount=1.0):
        self.rgba[..., 3] *= (1.0 - np.clip(mask * amount, 0, 1))
        return self

    def vertical_gradient(self, y0, y1, c0, c1):
        """(H,W,3) color array blending c0 at y0 to c1 at y1 (mm)."""
        _, Y = self.grid_mm()
        t = np.clip((Y - y0) / (y1 - y0), 0.0, 1.0)[..., None]
        return hex_rgb(c0) * (1 - t) + hex_rgb(c1) * t

    def text(self, s, x, y, size_mm, color, font_path, anchor="mm", stroke_mm=0.0, stroke_color="#000000", alpha=1.0):
        m = Image.new("L", (self.W, self.H), 0)
        font = ImageFont.truetype(font_path, int(size_mm * self.scale))
        d = ImageDraw.Draw(m)
        d.text(self.px(x, y), s, font=font, fill=255, anchor=anchor, stroke_width=int(stroke_mm * self.scale),
               stroke_fill=255)
        mask = np.asarray(m, dtype=np.float32) / 255.0
        if stroke_mm > 0:
            self.paint(mask, stroke_color, alpha)
            m2 = Image.new("L", (self.W, self.H), 0)
            ImageDraw.Draw(m2).text(self.px(x, y), s, font=font, fill=255, anchor=anchor)
            mask = np.asarray(m2, dtype=np.float32) / 255.0
        self.paint(mask, color, alpha)
        return mask

    def save(self, path, alpha=True):
        img = (np.clip(self.rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
        im = Image.fromarray(img, "RGBA")
        if self.ss > 1:
            im = im.resize((self.out_w, self.out_h), Image.LANCZOS)
        if not alpha:
            im = im.convert("RGB")
        im.save(path)
        return path
