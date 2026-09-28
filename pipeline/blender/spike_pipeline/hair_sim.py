"""Secondary-motion simulation in Blender: hair / ribbon / cloth chains under gravity, wind and collisions.

This is the offline twin of the runtime spring bones (Godot SpringBoneSimulator3D): it reads the SAME data
- the chains the rig builder made from character.json `secondary_chains` (stiffness, drag, gravity,
  radius, wind_response per chain) and the `colliders` list - so a Blender still or clip previews how the
  game will move the hair, and presentation shots (key art, cut-ins) get physically plausible flow
  instead of hand-posed strands.

Model (per chain, Verlet particles at the joints, root pinned to its parent bone):
  inertia with drag  +  gravity  +  aerodynamic wind (the strand relaxes toward the air velocity; gusts
  and travelling turbulence make it flutter)  +  a spring toward the chain's rest shape carried by its
  parent (stiffness)  ->  length constraints (follow-the-leader, so strands never stretch)  ->  collisions
  with the body (spheres / capsules pushed out by the chain radius).

    sim = ChainSim(rig, character, prefixes=("hair_", "ribbon_"))
    sim.wind = Wind(direction=(-1, 0.2, 0.35), speed=6.0, gust=0.35, turbulence=0.45)
    sim.run(2.0)            # settle into the wind
    sim.apply()             # pose the bones from the simulated joints
    for f in range(n): sim.run(1 / 24); sim.apply(); render ...
"""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Matrix, Vector


class Wind:
    """Air velocity field (m/s): a base flow with slow gusts and travelling turbulence (sum of waves)."""

    def __init__(self, direction=(-1.0, 0.0, 0.2), speed=5.0, gust=0.3, turbulence=0.4, seed=3):
        d = np.asarray(direction, dtype=np.float64)
        self.dir = d / np.linalg.norm(d)
        self.speed, self.gust, self.turb = float(speed), float(gust), float(turbulence)
        rng = np.random.default_rng(seed)
        self.waves = []
        for octave in range(4):
            k = rng.normal(size=3)
            k = k / np.linalg.norm(k) * (6.0 + 9.0 * octave)            # spatial frequency (1/m)
            amp = rng.normal(size=3)
            amp = amp / np.linalg.norm(amp) / (1.0 + octave)
            self.waves.append((k, amp, 2.0 + 2.5 * octave, rng.uniform(0, 2 * math.pi)))
        self.gust_phase = rng.uniform(0, 2 * math.pi, 3)

    def __call__(self, P, t):
        g = 1.0 + self.gust * (0.6 * math.sin(1.3 * t + self.gust_phase[0]) + 0.4 * math.sin(2.9 * t + self.gust_phase[1]))
        v = np.tile(self.dir * self.speed * g, (len(P), 1))
        for k, amp, w, ph in self.waves:
            s = np.sin(P @ k - w * t + ph)[:, None]
            v += amp[None, :] * s * self.speed * self.turb
        return v


# How much of the free-stream wind reaches each kind of strand: the head shelters the fringe (it flutters,
# it does not flip up), long free masses take the full flow, fine flyaways even more.
WIND_EXPOSURE = {"hair_bang": 0.05, "hair_ahoge": 0.35, "hair_side": 0.85, "hair_ponytail": 1.0, "hair_flyaway": 1.2,
                 "ribbon": 1.0, "cloth": 0.5}


class _Chain:
    def __init__(self, names, params, exposure, offset):
        self.names = names
        self.p = params
        self.exposure = exposure
        self.offset = offset          # where this strand samples the turbulence: strands separate, never march in step


class ChainSim:
    def __init__(self, rig, character, prefixes=("hair_", "ribbon_"), substeps_per_second=240, seed=11):
        self.rig = rig
        self.h = float(character["proportions"]["height_m"])
        self.dt = 1.0 / substeps_per_second
        self.t = 0.0
        self.wind = Wind()
        specs = {}
        for c in character["secondary_chains"]["chains"]:
            specs[(c["id"], c.get("side"))] = c
        bones = rig.data.bones
        self.chains = []
        rng = np.random.default_rng(seed)
        for (cid, side), c in specs.items():
            if not cid.startswith(tuple(prefixes)):
                continue
            suffix = f"_{side}" if side else ""
            names = [f"{cid}_{i:02d}{suffix}" for i in range(1, int(c["bones"]) + 1)]
            if all(n in bones for n in names):
                exposure = next((v for k, v in WIND_EXPOSURE.items() if cid.startswith(k)), 1.0)
                self.chains.append(_Chain(names, c, exposure, rng.normal(size=3) * 0.35))
        self.colliders = self._colliders(character)
        bpy.context.view_layer.update()
        self._init_particles()

    # ------------------------------------------------------------------ setup
    def _colliders(self, character):
        out = []
        for c in character["secondary_chains"].get("colliders", []):
            if c["bone"] not in self.rig.data.bones:
                continue
            out.append((c["bone"], c["shape"], float(c.get("along", 0.5)), float(c["radius_rel"]) * self.h,
                        float(c.get("height_rel", 0.0)) * self.h))
        return out

    def _bone_world(self, name):
        pb = self.rig.pose.bones[name]
        return self.rig.matrix_world @ pb.matrix

    def _init_particles(self):
        for ch in self.chains:
            b0 = self.rig.data.bones[ch.names[0]]
            parent = b0.parent
            P_rest = [b0.head_local.copy()] + [self.rig.data.bones[n].tail_local.copy() for n in ch.names]
            M_par_rest = parent.matrix_local if parent else Matrix.Identity(4)
            inv = M_par_rest.inverted()
            ch.parent = parent.name if parent else None
            ch.rest_local = [inv @ p for p in P_rest]                      # joints in the parent bone's rest frame
            ch.lengths = [(P_rest[i + 1] - P_rest[i]).length for i in range(len(ch.names))]
            X = np.array([list(self._rest_world(ch, i)) for i in range(len(P_rest))])
            ch.x = X.copy()
            ch.x_prev = X.copy()

    def _rest_world(self, ch, i):
        M = self._bone_world(ch.parent) if ch.parent else self.rig.matrix_world
        return M @ ch.rest_local[i]

    def _collider_shapes(self):
        shapes = []
        for bone, shape, along, r, hgt in self.colliders:
            M = self._bone_world(bone)
            pb = self.rig.pose.bones[bone]
            head, tail = M @ Vector((0, 0, 0)), M @ Vector((0, pb.length, 0))
            c = np.array(head.lerp(tail, along))
            if shape == "capsule":
                ax = np.array(tail - head)
                ax /= max(np.linalg.norm(ax), 1e-9)
                shapes.append(("capsule", c - ax * hgt * 0.5, c + ax * hgt * 0.5, r))
            else:
                shapes.append(("sphere", c, None, r))
        return shapes

    # ------------------------------------------------------------------ simulation
    def run(self, seconds):
        steps = max(1, int(round(seconds / self.dt)))
        shapes = self._collider_shapes()
        # The body holds its pose within one run() call: the rest shapes carried by the parents are fixed.
        rests = [np.array([list(self._rest_world(ch, i)) for i in range(len(ch.x))]) for ch in self.chains]
        for _ in range(steps):
            self._step(shapes, rests)
            self.t += self.dt

    def _step(self, shapes, rests):
        dt = self.dt
        for ch, rest in zip(self.chains, rests):
            p = ch.p
            n = len(ch.x)
            x, xp = ch.x, ch.x_prev
            vel = (x - xp) / dt
            damp = math.exp(-float(p.get("drag", 0.4)) * 4.0 * dt)
            air = 2.5 + 7.0 * float(p.get("wind_response", 0.5))          # 1/s: how fast the strand follows the air
            g = np.array([0.0, 0.0, -9.81 * float(p.get("gravity", 0.5)) * 1.6])
            k_rest = float(p.get("stiffness", 1.0)) * 18.0                 # 1/s^2 spring toward the rest shape
            acc = g[None, :] + air * (ch.exposure * self.wind(x + ch.offset, self.t) - vel)
            # Stiffness: each joint is pulled toward where the rest shape would put it from its parent joint.
            target = np.empty_like(x)
            target[0] = rest[0]
            target[1:] = x[:-1] + (rest[1:] - rest[:-1])
            acc += k_rest * (target - x)
            new = x + (x - xp) * damp + acc * dt * dt
            new[0] = rest[0]                                               # the root rides on its parent bone
            # Length constraints (follow the leader) and collisions.
            rad = float(p.get("radius", 0.01))
            for i in range(1, n):
                d = new[i] - new[i - 1]
                L = np.linalg.norm(d)
                new[i] = new[i - 1] + d * (ch.lengths[i - 1] / max(L, 1e-9))
                new[i] = self._collide(new[i], shapes, rad)
                d = new[i] - new[i - 1]
                new[i] = new[i - 1] + d * (ch.lengths[i - 1] / max(np.linalg.norm(d), 1e-9))
            ch.x_prev, ch.x = x, new

    @staticmethod
    def _collide(q, shapes, rad):
        for kind, a, b, r in shapes:
            if kind == "sphere":
                c = a
            else:
                ab = b - a
                t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
                c = a + ab * t
            d = q - c
            L = np.linalg.norm(d)
            R = r + rad
            if L < R:
                q = c + d * (R / max(L, 1e-9))
        return q

    # ------------------------------------------------------------------ output
    def apply(self):
        """Pose every chain bone along its simulated joints (root to tip, minimal rotation)."""
        for ch in self.chains:
            for i, name in enumerate(ch.names):
                bpy.context.view_layer.update()
                pb = self.rig.pose.bones[name]
                m = pb.matrix.copy()
                inv = self.rig.matrix_world.inverted()
                head = m.translation.copy()
                r = m.to_3x3()
                cur = (r @ Vector((0, 1, 0))).normalized()
                goal = (inv @ Vector(ch.x[i + 1])) - head
                if goal.length < 1e-6:
                    continue
                q = cur.rotation_difference(goal.normalized())
                pb.matrix = Matrix.Translation(head) @ (q.to_matrix() @ r).to_4x4()
        bpy.context.view_layer.update()
