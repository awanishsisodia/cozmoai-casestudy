"""Synthetic apartment + ray-cast "LiDAR" captures in Stray Scanner format.

This is test scaffolding: it lets every stage of the pipeline (and the drift
ablation) be exercised with exact ground truth before any phone capture
exists.  Synthetic numbers are never reported as benchmark results.

Layout (metres, z-up)::

    y
    7.5 +-----------+  +----+
        |  kitchen  |==|hall|
    4.12+-----------+  |    |  +---------+
    4.0 +-----------+  |    |  |         |
        |  living   |==|    |==| bedroom |
        |           |  |    |  |         |
    0   +-----------+  +----+  +---------+
        0          5.0 5.12 6.32 6.44    10.0   x
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .frames import rot_z
from .io_stray import A_WORLD, C_CV2ARKIT

WALL_TOP = 2.9


@dataclass
class Box:
    lo: tuple
    hi: tuple


ROOMS = {
    "living":  dict(x=(0.0, 5.0), y=(0.0, 4.0), h=2.60),
    "kitchen": dict(x=(0.0, 5.0), y=(4.12, 7.5), h=2.58),
    "hallway": dict(x=(5.12, 6.32), y=(0.0, 7.5), h=2.40),
    "bedroom": dict(x=(6.44, 10.0), y=(0.0, 3.5), h=2.55),
}
# doors: (room_a, room_b, wall x-range, y-range, width) -- all in walls normal to x
DOORS = [
    dict(a="living", b="hallway", x=(5.0, 5.12), y=(1.50, 2.40), top=2.03),
    dict(a="kitchen", b="hallway", x=(5.0, 5.12), y=(5.50, 6.32), top=2.03),
    dict(a="bedroom", b="hallway", x=(6.32, 6.44), y=(2.00, 2.86), top=2.03),
]
WINDOWS = [dict(room="living", x=(-0.2, 0.0), y=(1.0, 2.2), z=(0.9, 2.1))]


def _boxes() -> list[Box]:
    T = 0.12
    b: list[Box] = []

    def wall_x(x0, x1, y0, y1, holes):
        """Wall slab spanning y0..y1 at x0..x1, with rectangular holes (y-range, z-range)."""
        ys = sorted({y0, y1, *[h[0][0] for h in holes], *[h[0][1] for h in holes]})
        for ya, yb in zip(ys[:-1], ys[1:]):
            mid = (ya + yb) / 2
            hole = next((h for h in holes if h[0][0] <= mid <= h[0][1]), None)
            if hole is None:
                b.append(Box((x0, ya, 0), (x1, yb, WALL_TOP)))
            else:
                (_, (z0, z1)) = hole
                if z0 > 0:
                    b.append(Box((x0, ya, 0), (x1, yb, z0)))
                b.append(Box((x0, ya, z1), (x1, yb, WALL_TOP)))

    def wall_y(x0, x1, y0, y1):
        b.append(Box((x0, y0, 0), (x1, y1, WALL_TOP)))

    # exterior + interior walls
    wall_x(-0.2, 0.0, -0.2, 7.7, [((w["y"]), w["z"]) for w in WINDOWS])
    wall_x(5.0, 5.12, -0.2, 7.7, [((d["y"]), (0.0, d["top"])) for d in DOORS if d["x"] == (5.0, 5.12)])
    wall_x(6.32, 6.44, -0.2, 7.7, [((d["y"]), (0.0, d["top"])) for d in DOORS if d["x"] == (6.32, 6.44)])
    wall_x(10.0, 10.2, -0.2, 3.7, [])
    wall_y(-0.2, 10.2, -0.2, 0.0)
    wall_y(-0.2, 5.12, 4.0, 4.12)       # living / kitchen
    wall_y(6.32, 10.2, 3.5, 3.7)        # bedroom north
    wall_y(-0.2, 6.44, 7.5, 7.7)        # north
    # floor & per-room ceilings
    b.append(Box((-0.2, -0.2, -0.2), (10.2, 7.7, 0.0)))
    for r in ROOMS.values():
        b.append(Box((r["x"][0], r["y"][0], r["h"]), (r["x"][1], r["y"][1], r["h"] + 0.2)))
    # furniture
    b.append(Box((1.0, 3.05, 0.0), (3.0, 3.95, 0.80)))   # sofa against living north wall
    b.append(Box((7.5, 0.6, 0.0), (9.0, 1.6, 0.75)))     # bedroom table
    b.append(Box((0.0, 6.9, 0.0), (3.0, 7.5, 0.90)))     # kitchen counter
    return b


def raycast(origin: np.ndarray, dirs: np.ndarray, boxes: list[Box], max_t: float = 6.0) -> np.ndarray:
    """Distance along unit rays to the nearest box (slab method); inf if none."""
    t_best = np.full(len(dirs), np.inf)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    for bx in boxes:
        lo, hi = np.array(bx.lo), np.array(bx.hi)
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-6)
        t_best = np.where(hit & (tmin < t_best), tmin, t_best)
    t_best[t_best > max_t] = np.inf
    return t_best


def _trajectory(rng: np.random.Generator, subset: list[str] | None) -> list[tuple[np.ndarray, float]]:
    """Positions + yaw: spin in each room, walk through doors between rooms."""
    order = ["living", "hallway", "bedroom", "hallway", "kitchen", "hallway", "living"]
    if subset:
        order = [r for r in order if r in subset]
        order = [r for i, r in enumerate(order) if i == 0 or r != order[i - 1]]
    door_pt = {}
    for d in DOORS:
        c = np.array([np.mean(d["x"]), np.mean(d["y"])])
        door_pt[(d["a"], d["b"])] = door_pt[(d["b"], d["a"])] = c
    poses = []
    pos = None
    for i, room in enumerate(order):
        r = ROOMS[room]
        c = np.array([np.mean(r["x"]), np.mean(r["y"])])
        if pos is not None:
            prev = order[i - 1]
            dp = door_pt.get((prev, room))
            for target in ([dp] if dp is not None else []) + [c]:
                n = max(2, int(np.linalg.norm(target - pos) / 0.25))
                for a in np.linspace(0, 1, n, endpoint=False)[1:]:
                    p = pos + a * (target - pos)
                    yaw = np.arctan2(*(target - pos)[::-1])
                    poses.append((p, yaw))
                pos = target
        pos = c
        # look around: spin at a few points inside the room
        ext = np.array([r["x"][1] - r["x"][0], r["y"][1] - r["y"][0]])
        pts = [c] if room == "hallway" else [c, c + 0.22 * ext * np.array([1, 1]), c + 0.22 * ext * np.array([-1, -1])]
        if room == "hallway":
            pts = [np.array([c[0], r["y"][0] + 0.8]), c, np.array([c[0], r["y"][1] - 0.8])]
        for p in pts:
            yaw0 = rng.uniform(0, 2 * np.pi)
            for yaw in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                poses.append((p + rng.normal(0, 0.02, 2), yaw0 + yaw))
        pos = pts[-1]
    return poses


def _look(p_xy: np.ndarray, yaw: float, pitch: float, height: float) -> np.ndarray:
    """Camera-to-world for an OpenCV camera looking along yaw with pitch (radians, + = up)."""
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, fwd], axis=1)
    T[:3, 3] = [p_xy[0], p_xy[1], height]
    return T


def _R_to_quat(R: np.ndarray):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return (*q, (R[k, j] - R[j, k]) / s)


def make_stray_capture(out: str | Path, seed: int = 0, rooms: list[str] | None = None,
                       yaw_drift_deg_per_m: float = 0.0, z_drift_per_m: float = 0.0,
                       noise: float = 0.004) -> Path:
    """Write a synthetic Stray Scanner folder + ground_truth.json."""
    out = Path(out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    boxes = _boxes()
    W, H = 256, 192
    K_rgb = np.array([[1440.0, 0, 960], [0, 1440.0, 720], [0, 0, 1]])
    K = K_rgb.copy()
    K[:2] *= W / 1920
    v, u = np.mgrid[0:H, 0:W]
    rays_c = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    zc = 1.0 / np.linalg.norm(rays_c, axis=1)          # depth = t * zc
    rays_c = rays_c * zc[:, None]

    traj = _trajectory(rng, rooms)
    lines = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    travelled, prev, drift_T = 0.0, None, np.eye(4)
    for i, (p, yaw) in enumerate(traj):
        pitch = np.radians(rng.uniform(-5, 18))
        T_true = _look(p, yaw, pitch, 1.45 + rng.normal(0, 0.03))
        dirs = rays_c @ T_true[:3, :3].T
        t = raycast(T_true[:3, 3], dirs, boxes)
        depth = np.where(np.isfinite(t), t * zc, 0.0)
        depth = np.where(depth > 0, depth + rng.normal(0, noise + 0.002 * depth), 0.0)
        Image.fromarray(np.clip(depth * 1000, 0, 65535).astype(np.uint16).reshape(H, W)).save(out / "depth" / f"{i:06d}.png")
        Image.fromarray(np.full((H, W), 2, np.uint8)).save(out / "confidence" / f"{i:06d}.png")
        # accumulated odometry drift: yaw rotation about the start + vertical creep
        if prev is not None:
            travelled += np.linalg.norm(p - prev)
        prev = p
        D = np.eye(4)
        D[:3, :3] = rot_z(np.radians(yaw_drift_deg_per_m * travelled))
        D[2, 3] = z_drift_per_m * travelled
        T_rep = D @ T_true
        R_ark = A_WORLD.T @ T_rep[:3, :3] @ C_CV2ARKIT
        t_ark = A_WORLD.T @ T_rep[:3, 3]
        qx, qy, qz, qw = _R_to_quat(R_ark)
        lines.append(f"{i / 6.0:.4f}, {i},{t_ark[0]:.6f}, {t_ark[1]:.6f}, {t_ark[2]:.6f}, {qx:.8f}, {qy:.8f}, {qz:.8f}, {qw:.8f}")
    (out / "odometry.csv").write_text("\n".join(lines) + "\n")
    np.savetxt(out / "camera_matrix.csv", K_rgb, delimiter=",", fmt="%.6f")
    (out / "ground_truth.json").write_text(json.dumps(ground_truth(rooms), indent=2))
    return out


def ground_truth(rooms: list[str] | None = None) -> dict:
    sel = rooms or list(ROOMS)
    gt_rooms = []
    for name in sel:
        r = ROOMS[name]
        w, d = r["x"][1] - r["x"][0], r["y"][1] - r["y"][0]
        ops = []
        for dd in DOORS:
            if name in (dd["a"], dd["b"]):
                other = dd["b"] if dd["a"] == name else dd["a"]
                if other in sel:
                    ops.append({"type": "door", "width_m": round(dd["y"][1] - dd["y"][0], 3), "to": other})
        for wd in WINDOWS:
            if wd["room"] == name:
                ops.append({"type": "window", "width_m": round(wd["y"][1] - wd["y"][0], 3)})
        gt_rooms.append({"name": name, "ceiling_height_m": r["h"], "floor_area_m2": round(w * d, 4),
                         "walls_m": [round(w, 3), round(d, 3), round(w, 3), round(d, 3)], "openings": ops})
    adj = sorted({tuple(sorted((d["a"], d["b"]))) for d in DOORS if d["a"] in sel and d["b"] in sel})
    return {"source": "synthetic", "rooms": gt_rooms, "adjacency": [list(a) for a in adj],
            "footprint_m2": round(sum(r["floor_area_m2"] for r in gt_rooms), 4)}
