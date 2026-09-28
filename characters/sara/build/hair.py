"""Sara: hairstyle, ribbon and tie.

Construction (character.json hair.construction):
  primary   : pulled-back top and nape masses converging on a high tie off-center to her left;
              layered fringe with a long lock falling between the eyes; asymmetric side locks
              (left curls under the jaw, right hangs long); a long, thick ponytail
  secondary : ponytail sub-masses that fan toward the tips; thin fringe strands in front
  tertiary  : ahoge, flyaways (LOD0 only)
  ribbon    : thin satin band over a navy elastic tie, one standing loop and one long tail
Every lock is placed against the head loft and the rig's hair chains, so the geometry and the
secondary-motion bones agree by construction.
"""
from __future__ import annotations

import math

import numpy as np

from spike_pipeline.modeling import lock as L

import head as headmod

C = np.array([0.0, 0.012, 1.563])      # head center used for scalp projection
EYE_Z = headmod.EYE_Z


def Z(z):
    """Heights below were designed on the earlier, longer head (eye line 1.5528); map them onto the
    acorn head so the hairstyle keeps its placement relative to the eyes."""
    return EYE_Z + (z - 1.5528) * 0.85


def _v(*a):
    return np.array(a, dtype=np.float64)


class Scalp:
    def __init__(self, profile):
        self.p = profile

    def inside(self, q, margin=0.0):
        p = self.p
        t = p.t_of(q[2])
        if t <= 0.0 or t >= 1.0:
            return False
        w = np.interp(t, p.t, p.w) + margin
        yw = np.interp(t, p.t, p.yw)
        if q[1] < yw:
            d = yw - np.interp(t, p.t, p.yf) + margin
            n = np.interp(t, p.t, p.nf)
        else:
            d = np.interp(t, p.t, p.yb) - yw + margin
            n = np.interp(t, p.t, p.nb)
        return abs(q[0] / w) ** n + abs((q[1] - yw) / d) ** n <= 1.0

    def surface(self, direction, lift=0.0):
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        lo, hi = 0.0, 0.2
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.inside(C + d * mid):
                lo = mid
            else:
                hi = mid
        return C + d * (lo + lift)

    def project(self, q, lift):
        return self.surface(np.asarray(q, float) - C, lift)

    def keep_out(self, q, min_lift):
        q = np.asarray(q, float)
        if self.inside(q, margin=min_lift):
            return self.project(q, min_lift)
        return q

    def outward(self, q):
        v = np.asarray(q, float) - C
        return v / np.linalg.norm(v)


_HL_A = [0, 30, 60, 80, 95, 108, 120, 150, 180]
_HL_Z = [Z(1.618), Z(1.614), Z(1.598), Z(1.585), Z(1.566), Z(1.535), Z(1.512), Z(1.482), Z(1.472)]


def hairline_z(alpha_deg):
    """Hairline height around the head. alpha 0 = front center, +-180 = nape."""
    return float(np.interp(abs(((alpha_deg + 180) % 360) - 180), _HL_A, _HL_Z))


def _dir(alpha_deg, elev_deg):
    a, e = math.radians(alpha_deg), math.radians(elev_deg)
    return _v(math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))


def hairline_point(sc, alpha_deg, lift=0.0):
    """Scalp point on the hairline (bisection on elevation so the point sits at hairline_z)."""
    z = hairline_z(alpha_deg)
    lo, hi = -70.0, 88.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if sc.surface(_dir(alpha_deg, mid)).__getitem__(2) < z:
            lo = mid
        else:
            hi = mid
    return sc.surface(_dir(alpha_deg, lo), lift), lo


def cap(sc, lift=0.0028, n_alpha=72, n_up=14):
    """Dark under-layer covering the scalp inside the hairline, so gaps between locks never show skin."""
    verts, faces = [], []
    grid = np.zeros((n_alpha, n_up), dtype=int)
    for i in range(n_alpha):
        alpha = -180 + 360 * i / n_alpha
        _, e0 = hairline_point(sc, alpha)
        for j in range(n_up):
            e = e0 + (89.0 - e0) * (j / n_up) ** 0.9
            grid[i, j] = len(verts)
            verts.append(sc.surface(_dir(alpha, e), lift))
    top = len(verts)
    verts.append(sc.surface(_v(0.0, 0.0, 1.0), lift))
    for i in range(n_alpha):
        ii = (i + 1) % n_alpha
        for j in range(n_up - 1):
            faces.append((grid[i, j], grid[ii, j], grid[ii, j + 1], grid[i, j + 1]))
        faces.append((grid[i, n_up - 1], grid[ii, n_up - 1], top))
    return np.asarray(verts), faces


def _slerp(a, b, t):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    w = math.acos(max(-1.0, min(1.0, float(a @ b))))
    if w < 1e-6:
        return a
    return (math.sin((1 - t) * w) * a + math.sin(t * w) * b) / math.sin(w)


def build(bones, batch_hair: L.LockBatch, batch_ribbon: L.LockBatch):
    prof = headmod.profile()
    sc = Scalp(prof)
    tie = _v(*bones["hair_ponytail_A_01"]["head"])
    tags = []

    def add(points, width, thickness, tag, batch=batch_hair, **kw):
        kw.setdefault("outward", sc.outward)
        v, f, a = L.lock(points, width, thickness, **kw)
        batch.add(v, f, a, tag)

    # --- A/B. pulled-back locks: hairline -> over the scalp -> tie ------------------------------
    def pulled_back(alpha, width, lifts):
        start, _ = hairline_point(sc, alpha, 0.002)
        d0, d1 = start - C, tie - C
        pts = [start] + [sc.surface(_slerp(d0, d1, t), lift) for t, lift in lifts]
        pts.append(tie + _v(0.0, 0.005, -0.002))
        add(pts, width, 0.0068, "top", tip_start=0.84, root_min=0.85, crescent=0.12)

    for alpha in np.linspace(-100, 100, 15):
        pulled_back(alpha, 0.038 if abs(alpha) < 75 else 0.032, ((0.22, 0.007), (0.45, 0.009), (0.7, 0.01), (0.88, 0.009)))
    for alpha in (104, 116, 128, -104, -116, -128):
        pulled_back(alpha, 0.036, ((0.25, 0.007), (0.5, 0.009), (0.78, 0.009)))
    for alpha in np.linspace(140, 220, 9):
        pulled_back(alpha, 0.04, ((0.3, 0.007), (0.6, 0.009), (0.85, 0.009)))

    # --- C. fringe: top-front -> over the forehead -> points at the eyes -------------------------
    def front_y(x, z, gap):
        return prof.front_y(x, prof.t_of(z)) - gap

    fringe = [  # (root x, mid x, tip x, tip z, width, tag)
        (0.004, 0.005, 0.007, Z(1.536), 0.024, "bang_B"),
        (0.02, 0.024, 0.029, Z(1.563), 0.025, "bang_A"),
        (0.036, 0.042, 0.049, Z(1.554), 0.023, "bang_A"),
        (0.052, 0.061, 0.069, Z(1.543), 0.019, "bang_A"),
        (-0.012, -0.014, -0.017, Z(1.556), 0.024, "bang_B"),
        (-0.028, -0.032, -0.037, Z(1.567), 0.023, "bang_C"),
        (-0.044, -0.051, -0.059, Z(1.552), 0.021, "bang_C"),
        (-0.058, -0.066, -0.073, Z(1.538), 0.017, "bang_C"),
    ]
    for i, (rx, mx, tx, tz, w, tag) in enumerate(fringe):
        wave = 0.0035 * math.sin(i * 2.1 + 0.5)
        mx, tx = mx + wave, tx - wave * 1.4
        root = sc.project(_v(rx * 0.8, -0.045, Z(1.654)), 0.004)
        p1 = sc.project(_v(rx, -0.082, Z(1.628)), 0.008)
        p2 = _v(mx, front_y(mx, Z(1.598), 0.009), Z(1.598))
        p3 = _v((mx + tx) / 2, front_y((mx + tx) / 2, (Z(1.598) + tz) / 2, 0.008), (Z(1.598) + tz) / 2)
        tip = _v(tx, front_y(tx, tz, 0.006), tz)
        add([root, p1, p2, p3, tip], w * 1.45, 0.0052, tag, tip_start=0.4, tip_power=0.65, root_min=0.9, crescent=0.2,
            outward=_v(0.0, -1.0, 0.25))
    # Back layer between the front locks: fills the fringe so the forehead never shows in stripes.
    for i in range(len(fringe) - 1):
        a, b = fringe[i], fringe[i + 1]
        if a[0] * b[0] < 0 and abs(a[0]) > 0.01:
            continue
        rx, tx = 0.5 * (a[0] + b[0]), 0.5 * (a[2] + b[2])
        tz = max(a[3], b[3]) + 0.008
        root = sc.project(_v(rx * 0.8, -0.05, Z(1.652)), 0.003)
        p1 = sc.project(_v(rx, -0.08, Z(1.625)), 0.006)
        p2 = _v(rx, front_y(rx, Z(1.598), 0.006), Z(1.598))
        tip = _v(tx, front_y(tx, tz, 0.004), tz)
        add([root, p1, p2, tip], 0.028, 0.0045, a[5], tip_start=0.5, root_min=0.9, crescent=0.2, outward=_v(0.0, -1.0, 0.25))
    for rx, tx, tz in ((0.011, 0.016, Z(1.548)), (-0.036, -0.047, Z(1.557))):
        root = sc.project(_v(rx, -0.07, Z(1.645)), 0.006)
        mid = _v((rx + tx) / 2, front_y((rx + tx) / 2, Z(1.596), 0.011), Z(1.596))
        tip = _v(tx, front_y(tx, tz, 0.009), tz)
        add([root, mid, tip], 0.0085, 0.0028, "bang_B" if abs(tx) < 0.02 else ("bang_A" if tx > 0 else "bang_C"),
            tip_start=0.3, root_min=0.8, crescent=0.1, outward=_v(0.0, -1.0, 0.2))

    # --- D. side locks ---------------------------------------------------------------------------
    left = [_v(0.071, -0.032, Z(1.604)), _v(0.08, -0.042, Z(1.556)), _v(0.075, -0.052, Z(1.500)), _v(0.062, -0.066, Z(1.458)),
            _v(0.046, -0.074, Z(1.442))]
    left = [sc.keep_out(p, 0.006) for p in left]
    add(left, 0.024, 0.0055, "side_L", tip_start=0.5, root_min=0.85, crescent=0.2)
    add([sc.keep_out(p + _v(0.004, 0.012, 0.004), 0.006) for p in left[:-1]], 0.016, 0.004, "side_L", tip_start=0.4)
    right = [_v(-0.071, -0.03, Z(1.606)), _v(-0.083, -0.036, Z(1.540)), _v(-0.085, -0.03, Z(1.470)), _v(-0.082, -0.02, Z(1.410)),
             _v(-0.078, -0.012, Z(1.358))]
    right = [sc.keep_out(p, 0.006) for p in right]
    add(right, 0.026, 0.006, "side_R", tip_start=0.55, root_min=0.85, crescent=0.2)
    add([sc.keep_out(p + _v(-0.003, 0.014, 0.0), 0.006) for p in right[:-1]] + [right[-1] + _v(0.004, 0.02, 0.03)],
        0.018, 0.0045, "side_R", tip_start=0.45)

    # --- E. ponytail: locks bundled along the A/B/C chains, fanning toward the tips ---------------
    def chain_points(cid):
        names = [n for n in bones if n.startswith(cid + "_")]
        names.sort()
        return [np.asarray(bones[n]["head"]) for n in names] + [np.asarray(bones[names[-1]]["tail"])]

    rng = np.random.default_rng(3)
    plan = [("hair_ponytail_A", 10, 0.044, 0.011), ("hair_ponytail_B", 6, 0.037, 0.01), ("hair_ponytail_C", 6, 0.034, 0.009)]
    for cid, count, w, th in plan:
        pts0 = chain_points(cid)
        axis0 = pts0[1] - pts0[0]
        axis0 /= np.linalg.norm(axis0)
        side = np.cross(axis0, _v(0, 0, 1))
        side /= np.linalg.norm(side)
        up = np.cross(side, axis0)
        for k in range(count):
            phi = 2 * math.pi * k / count + rng.uniform(-0.3, 0.3)
            length = rng.uniform(0.78, 1.0)
            fan = rng.uniform(0.012, 0.028)
            pts = []
            n = len(pts0)
            for i, p in enumerate(pts0):
                u = i / (n - 1)
                if u > length:
                    break
                r = 0.006 + fan * u ** 1.2
                off = side * math.cos(phi) * r + up * math.sin(phi) * r * 0.7
                wobble = side * 0.013 * u * math.sin(u * 3.4 + k * 0.9) + up * 0.006 * u * math.cos(u * 2.6 + k)
                pts.append(p + off + wobble)
            pts[0] = tie + (side * math.cos(phi) + up * math.sin(phi)) * 0.004
            if len(pts) >= 3:
                add(pts, w * rng.uniform(0.85, 1.1), th, cid.replace("hair_", ""), tip_start=0.35, tip_power=0.65,
                    root_min=0.75, crescent=0.12, twist_deg=rng.uniform(-40, 40),
                    outward=lambda q, s=side, u_=up, ph=phi: s * math.cos(ph) + u_ * math.sin(ph))
    # Root puff: short locks arching up out of the tie before falling.
    for k in range(4):
        phi = -0.9 + 0.6 * k
        a0 = tie
        ax = chain_points("hair_ponytail_A")
        p1 = tie + _v(0.012 * math.cos(phi), 0.03, 0.022 + 0.004 * k)
        p2 = ax[2] + _v(0.02 * math.cos(phi), 0.022, 0.0)
        p3 = ax[3] + _v(0.022 * math.cos(phi), 0.02, 0.0)
        add([a0, p1, p2, p3], 0.03, 0.008, "ponytail_A", tip_start=0.5, root_min=0.8, crescent=0.15,
            outward=lambda q: _v(0.0, 1.0, 0.3))

    # --- F/G/H: ahoge, flyaways, nape strands -----------------------------------------------------
    for chain_id, w in (("hair_ahoge_A", 0.0045),):
        pts = chain_points(chain_id)
        tip = pts[-1] + (pts[-1] - pts[-2]) * 0.8 + _v(0.0, -0.012, -0.008)
        add(pts + [tip], w, 0.0018, "ahoge_A", tip_start=0.2, root_min=1.0, crescent=0.0, outward=_v(1.0, 0.0, 0.0))
        add([p + _v(-0.006, 0.004, -0.002) for p in pts] + [tip + _v(-0.012, 0.006, -0.012)], w * 0.8, 0.0015, "ahoge_A",
            tip_start=0.2, root_min=1.0, crescent=0.0, outward=_v(1.0, 0.0, 0.0))
    for fid in ("hair_flyaway_C",):  # temple flyaways (A, B) are wind-only accents added by the S+ VFX pass
        pts = [sc.keep_out(p, 0.004) for p in chain_points(fid)]
        curl = (pts[-1] - pts[-2]) * 0.5 + _v(0.0, 0.0, 0.012)
        add(pts + [pts[-1] + curl], 0.0034, 0.0012, fid.replace("hair_", ""), tip_start=0.15, root_min=1.0, crescent=0.0)
    for x in (0.028, -0.004, -0.034):
        top = sc.surface(_v(x * 8, 1.0, -0.62), 0.004)
        add([top, top + _v(x * 0.2, 0.012, -0.03), top + _v(x * 0.35, 0.008, -0.055)], 0.014, 0.004, "nape",
            tip_start=0.35, root_min=0.9)

    # --- ribbon: soft satin. A relaxed standing loop and two flowing tails with S-waves and twist ------
    loop_pts = chain_points("ribbon_loop_A")
    k0 = loop_pts[0]
    up = loop_pts[-1] - k0
    h = np.linalg.norm(up)
    up /= h
    lat = np.cross(up, _v(0.0, -1.0, 0.0))
    lat /= np.linalg.norm(lat)
    normal = np.cross(lat, up)
    path = []
    for s_ in np.linspace(0.0, 1.0, 30):
        a = 2 * math.pi * s_
        yy = h * (0.5 - 0.5 * math.cos(a)) ** 0.8
        xx = 0.0095 * math.sin(a) * (1.0 - 0.2 * (yy / h)) + 0.008 * math.sin(math.pi * yy / h) ** 2
        zz = 0.005 * math.sin(a) + 0.004 * math.sin(math.pi * yy / h)          # a soft twist in depth
        path.append(k0 + up * yy * 1.05 + lat * xx + normal * zz)
    add(path, 0.0135, 0.0011, "ribbon_loop_A", batch=batch_ribbon, crescent=0.08, width_fn=lambda u: 1.0 - 0.15 * math.sin(math.pi * u),
        outward=lambda q, n_=normal: n_, thickness_tip=1.0, twist_deg=25)
    tail = chain_points("ribbon_tail_A")
    n_t = len(tail)
    flowing = []
    for i, p in enumerate(tail):
        u = i / (n_t - 1)
        flowing.append(p + lat * (0.022 * u * math.sin(u * 5.2)) + normal * (0.012 * u * math.cos(u * 4.0)))
    add(flowing, 0.0145, 0.0011, "ribbon_tail_A", batch=batch_ribbon, crescent=0.1, twist_deg=70, thickness_tip=1.0,
        width_fn=lambda u: (1.0 - 0.1 * math.sin(math.pi * u)) * (1.0 - 0.45 * L._smooth(0.93, 1.0, u)), outward=sc.outward)
    short = [k0 + (p - tail[0]) * 0.55 + lat * (-0.01 * (i / (n_t - 1))) + _v(0.0, 0.0, -0.012 * (i / (n_t - 1)))
             for i, p in enumerate(tail)]
    add(short, 0.0135, 0.0011, "ribbon_tail_A", batch=batch_ribbon, crescent=0.1, twist_deg=-50, thickness_tip=1.0,
        width_fn=lambda u: 1.0 - 0.45 * L._smooth(0.9, 1.0, u), outward=sc.outward)
    cv, cf = cap(sc)
    return {"tie": tie, "knot": k0, "loop_normal": normal, "loop_up": up, "cap": (cv, cf), "clip": hair_clip(sc)}


def hair_clip(sc):
    """Cross-shaped hair clip on her right side above the fringe (in-game close-up): a sky-blue bar
    along the head and a white bar crossing it near its outer end. Returns box specs."""
    a, e = math.radians(-38), math.radians(34)
    d = _v(math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))
    c = sc.surface(d, 0.0105)
    n = sc.outward(c)
    along = np.cross(_v(0.0, 0.0, 1.0), n)
    along /= np.linalg.norm(along)
    vert = np.cross(n, along)
    tilt = math.radians(-12)
    along, vert = along * math.cos(tilt) + vert * math.sin(tilt), -along * math.sin(tilt) + vert * math.cos(tilt)
    return [
        {"center": c, "axes": (along, vert, n), "half": (0.024, 0.0026, 0.0012), "color": "#6fb0e8", "name": "bar"},
        {"center": c + along * -0.012 + n * 0.0009, "axes": (along, vert, n), "half": (0.0022, 0.02, 0.0012),
         "color": "#f3f5fa", "name": "cross"},
    ]
