#!/usr/bin/env python3
"""scan2plan - iPhone LiDAR scan (Stray Scanner export) -> dimensioned floor plan.

    python scan2plan.py data/<capture_folder> [-o out/<name>]

Writes <out>/result.json (rooms, walls, openings, heights, areas, each with a
90 % interval) and <out>/plan.png.

Method (all measurements come from thousands of LiDAR points, never from a
coarse grid):
  1. load depth + pose every ~0.17 s, keep high-confidence depth < 4 m
  2. fuse into a 1 cm point cloud with surface normals
  3. rotate so walls are axis-aligned (dominant wall direction)
  4. floor / ceiling = height histograms of up / down-facing surfaces
  5. rooms = floor map cut at doorways (morphological opening), rooms the
     camera actually walked in
  6. walls = 1 cm histogram peaks of wall points; polygon = cells between
     wall lines that are mostly floor
  7. openings = gaps in a wall's points at door height through which the
     scanner saw things behind the wall (door), or above a solid sill (window)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = 0.05            # 2-D map resolution (m) - only used to find rooms, not to measure
# 90 % half-widths for the LiDAR tier (sensor prior: ~1 cm depth noise)
CI = {"wall": (0.015, 0.003), "height": (0.012, 0.0), "opening": (0.02, 0.0), "area": (0.05, 0.01)}


def measure(v, kind):
    a, r = CI[kind]
    hw = a + r * abs(v)
    return {"value": round(float(v), 3), "ci90": [round(float(v - hw), 3), round(float(v + hw), 3)]}


# ----------------------------------------------------------------------------- 1. load

def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_frames(folder: Path, hz: float = 6.0):
    """Yield (depth[m], confidence-mask, K, T_world_cam) with world z-up and OpenCV camera axes."""
    rows = []
    for line in (folder / "odometry.csv").read_text().splitlines()[1:]:
        p = [s.strip() for s in line.split(",")]
        rows.append([float(s) if s else np.nan for s in p[:13]] + [np.nan] * (13 - len(p[:13])))
    odo = np.array(rows)
    K_rgb = np.loadtxt(folder / "camera_matrix.csv", delimiter=",")
    step = max(1, int(round(1 / hz / np.median(np.diff(odo[:, 0])))))
    A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])        # ARKit y-up -> z-up
    for r in odo[::step]:
        f = int(r[1])
        depth = np.asarray(Image.open(folder / "depth" / f"{f:06d}.png")).astype(np.float32) / 1000
        cpath = folder / "confidence" / f"{f:06d}.png"
        conf = np.asarray(Image.open(cpath)) >= 2 if cpath.exists() else np.ones_like(depth, bool)
        K = K_rgb.copy() if np.isnan(r[9:13]).any() else np.array([[r[9], 0, r[11]], [0, r[10], r[12]], [0, 0, 1]])
        K[:2] *= depth.shape[1] / (2 * K_rgb[0, 2])           # RGB intrinsics -> depth resolution
        T = np.eye(4)
        T[:3, :3] = A @ quat_to_R(*r[5:9])   # Stray poses already use OpenCV camera axes (verified: floor 1.4 m below phone)
        T[:3, 3] = A @ r[2:5]
        yield depth, conf, K, T


# ----------------------------------------------------------------------------- 2. cloud

def frame_points(depth, conf, K, T, stride=2, max_range=4.0):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    P = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    dx, dy = np.zeros_like(P), np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    N = np.cross(dx, dy)
    smooth = np.zeros_like(depth, bool)                        # no normals across depth edges
    d = depth
    smooth[1:-1, 1:-1] = (np.abs(d[1:-1, 2:] - d[1:-1, :-2]) < 0.05 * d[1:-1, 1:-1]) & \
                         (np.abs(d[2:, 1:-1] - d[:-2, 1:-1]) < 0.05 * d[1:-1, 1:-1])
    ok = (d > 0.2) & (d < max_range) & conf & smooth & (u % stride == 0) & (v % stride == 0)
    P, N = P[ok], N[ok]
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    N[(N * P).sum(1) > 0] *= -1                                # normals face the camera
    return P @ T[:3, :3].T + T[:3, 3], N @ T[:3, :3].T


def build_cloud(folder: Path, voxel=0.01):
    Ps, Ns, cams = [], [], []
    for depth, conf, K, T in load_frames(folder):
        P, N = frame_points(depth, conf, K, T)
        Ps.append(P), Ns.append(N), cams.append(T[:3, 3])
    P, N = np.concatenate(Ps), np.concatenate(Ns)
    _, inv, cnt = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    P = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    N = np.stack([np.bincount(inv, N[:, k]) for k in range(3)], 1)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return P, N, np.array(cams), len(cams)


# ----------------------------------------------------------------------------- 3-4. align, heights

def manhattan_align(P, N, cams):
    vert = np.abs(N[:, 2]) < 0.15
    a = np.arctan2(N[vert, 1], N[vert, 0])
    th = np.angle(np.exp(4j * a).mean()) / 4
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return P @ R.T, N @ R.T, cams @ R.T


def height_peak(z, pick):
    """Median height of the chosen histogram peak (1 cm bins)."""
    h, e = np.histogram(z, np.arange(z.min() - 0.02, z.max() + 0.03, 0.01))
    h = ndimage.uniform_filter1d(h.astype(float), 3)
    pk = np.nonzero((h >= np.roll(h, 1)) & (h >= np.roll(h, -1)) & (h >= 0.2 * h.max()))[0]
    z0 = e[pick(pk, h)] + 0.005
    return float(np.median(z[np.abs(z - z0) < 0.02]))


def floor_ceiling(P, N):
    up, dn = N[:, 2] > 0.9, N[:, 2] < -0.9
    floor = height_peak(P[up, 2], lambda pk, h: pk.min())                 # lowest strong level
    zc = P[dn & (P[:, 2] > floor + 1.9), 2]
    ceil = height_peak(zc, lambda pk, h: pk[np.argmax(h[pk])]) if len(zc) > 500 else None
    return floor, ceil


# ----------------------------------------------------------------------------- 5. rooms

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def find_rooms(P, N, cams, floor, ceil):
    lo = P[:, :2].min(0) - 0.5
    shape = tuple(np.ceil((P[:, :2].max(0) + 0.5 - lo) / RES).astype(int) + 1)
    idx = lambda xy: tuple(np.floor((xy - lo) / RES).astype(int).T)  # noqa: E731
    occ = np.zeros(shape, bool)
    flat = (N[:, 2] > 0.9) & (np.abs(P[:, 2] - floor) < 0.04)
    if ceil is not None:
        flat |= (N[:, 2] < -0.9) & (np.abs(P[:, 2] - ceil) < 0.05)
    occ[idx(P[flat, :2])] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, disk(2)))
    if ceil is not None:
        # door headers hang below the ceiling, so the ceiling map is cut at every doorway
        cm = np.zeros(shape, bool)
        cm[idx(P[(N[:, 2] < -0.9) & (P[:, 2] > floor + 2.15), :2])] = True
        core, n = ndimage.label(ndimage.binary_erosion(ndimage.binary_fill_holes(cm), disk(4)))
    else:
        core, n = ndimage.label(ndimage.binary_opening(occ, disk(9)))   # no ceiling: cut floor map at doorways
    walked = core[idx(cams[:, :2])]
    keep = [k for k in range(1, n + 1) if (walked == k).sum() >= 3 and (core == k).sum() * RES ** 2 > 1.5]
    lab = np.zeros_like(core)
    for i, k in enumerate(keep, 1):
        lab[core == k] = i
    dist, (ii, jj) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    grown = np.where(occ & (dist * RES < 0.6), lab[ii, jj], 0)       # grow back to the walls
    return [(grown == k) for k in range(1, len(keep) + 1)], lo


def wall_lines(P, N, axis, min_area=0.5):
    """Positions of wall planes perpendicular to `axis` (signed normals keep the two faces of a wall apart)."""
    lines = []
    for sgn in (1, -1):
        m = (N[:, axis] * sgn > 0.9) & (np.abs(N[:, 2]) < 0.15)
        x = P[m, axis]
        if len(x) < 50:
            continue
        e = np.arange(x.min() - 0.02, x.max() + 0.03, 0.01)
        h = ndimage.uniform_filter1d(np.histogram(x, e)[0].astype(float), 3)
        for i in np.argsort(-h):
            if h[i] == 0 or not (h[i] >= h[max(i - 1, 0)] and h[i] >= h[min(i + 1, len(h) - 1)]):
                continue
            if any(abs(e[i] - l) < 0.05 for l, _ in lines):
                continue
            sel = np.abs(x - e[i] - 0.005) < 0.02
            cells = np.unique(np.floor(np.stack([P[m][sel, 1 - axis], P[m][sel, 2]], 1) / 0.05).astype(int), axis=0)
            if len(cells) * 0.0025 >= min_area:                         # >= 0.3 m^2 of wall surface seen
                lines.append((float(np.median(x[sel])), len(cells)))
    return sorted(l for l, _ in lines)


def room_polygon(mask, lo, P, N):
    near = ndimage.binary_dilation(mask, disk(6))
    ij = np.floor((P[:, :2] - lo) / RES).astype(int)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < mask.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < mask.shape[1])
    sel = inb.copy()
    sel[inb] = near[ij[inb, 0], ij[inb, 1]]
    ii, jj = np.nonzero(mask)
    ext = [(lo[0] + ii.min() * RES, lo[0] + (ii.max() + 1) * RES), (lo[1] + jj.min() * RES, lo[1] + (jj.max() + 1) * RES)]
    grid = []
    for ax in (0, 1):
        L = wall_lines(P[sel], N[sel], ax)
        L = [l for l in L if ext[ax][0] - 0.3 < l < ext[ax][1] + 0.3]
        L += [b for b in ext[ax] if not L or min(abs(b - l) for l in L) > 0.15]
        grid.append(sorted(L))
    X, Y = grid
    cells = []
    for xa, xb in zip(X[:-1], X[1:]):
        for ya, yb in zip(Y[:-1], Y[1:]):
            gx, gy = np.meshgrid(np.linspace(xa, xb, 6)[1:-1], np.linspace(ya, yb, 6)[1:-1])
            i = np.clip(((gx - lo[0]) / RES).astype(int), 0, mask.shape[0] - 1)
            j = np.clip(((gy - lo[1]) / RES).astype(int), 0, mask.shape[1] - 1)
            if mask[i, j].mean() >= 0.5:
                cells.append(box(xa, ya, xb, yb))
    if not cells:
        return None
    poly = unary_union(cells).buffer(-0.25, join_style=2).buffer(0.25, join_style=2)   # drop steps < 50 cm (furniture)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return None if poly.is_empty else orient(Polygon(poly.exterior).simplify(0.005), 1.0)


# ----------------------------------------------------------------------------- 7. openings

def runs(mask):
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def find_openings(a, b, P, N, floor, poly):
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    n = np.array([-u[1], u[0]])                                 # inward normal (CCW polygon)
    rel = P[:, :2] - a
    s, dist, z = rel @ u, rel @ n, P[:, 2] - floor
    on_wall = (np.abs(dist) < 0.06) & (N[:, :2] @ n > 0.8) & (s >= 0) & (s <= L)
    behind = (dist < -0.25) & (dist > -4) & (s >= 0) & (s <= L) & (z > 0.1) & (z < 2.0)
    nb = int(np.ceil(L / 0.01))
    hist = lambda m: np.bincount(np.clip((s[m] / 0.01).astype(int), 0, nb - 1), minlength=nb) > 0  # noqa: E731
    door_cov = ndimage.binary_closing(hist(on_wall & (z > 0.3) & (z < 1.8)), np.ones(3))
    low_cov = hist(on_wall & (z > 0.15) & (z < 0.7))
    mid_cov = hist(on_wall & (z > 1.0) & (z < 1.5))
    seen_behind = np.bincount(np.clip((s[behind] / 0.01).astype(int), 0, nb - 1), minlength=nb)
    out = []
    for s0, s1 in runs(~door_cov):
        w = (s1 - s0) * 0.01
        if 0.55 <= w <= 1.8 and seen_behind[s0:s1].sum() > 30:
            out.append(("door", s0, s1))
    for s0, s1 in runs(~mid_cov & ndimage.binary_dilation(low_cov, np.ones(15))):
        w = (s1 - s0) * 0.01
        if 0.35 <= w <= 3.0 and not any(o[1] <= (s0 + s1) / 2 <= o[2] for o in out):
            out.append(("window", s0, s1))
    return [{"type": t, "width_m": (s1 - s0) * 0.01, "offset_m": s0 * 0.01,
             "center": (a + u * (s0 + s1) / 2 * 0.01).round(3).tolist(),
             "_probe": a + u * (s0 + s1) / 2 * 0.01 - n * 0.5} for t, s0, s1 in out]


# ----------------------------------------------------------------------------- main

def scan2plan(folder: Path, out: Path):
    P, N, cams, n_frames = build_cloud(folder)
    P, N, cams = manhattan_align(P, N, cams)
    floor, ceil = floor_ceiling(P, N)
    masks, lo = find_rooms(P, N, cams, floor, ceil)
    rooms = []
    for k, mask in enumerate(masks, 1):
        poly = room_polygon(mask, lo, P, N)
        if poly is None or poly.area < 1.0:
            continue
        inside = np.array([poly.buffer(-0.2).contains(Point(p)) for p in P[::50, :2]]) if poly.area > 2 else None
        rc = ceil
        if ceil is not None and inside is not None and inside.sum() > 200:   # per-room ceiling height
            sub = slice(None, None, 50)
            Pi, Ni = P[sub][inside], N[sub][inside]
            try:
                rc = floor_ceiling(Pi, Ni)[1] or ceil
            except ValueError:
                pass
        pts = np.array(poly.exterior.coords)
        walls, openings = [], []
        for i in range(len(pts) - 1):
            wid = f"r{k}_w{i + 1}"
            walls.append({"id": wid, "start": pts[i].round(3).tolist(), "end": pts[i + 1].round(3).tolist(),
                          "length_m": measure(np.linalg.norm(pts[i + 1] - pts[i]), "wall")})
            for o in find_openings(pts[i], pts[i + 1], P, N, floor, poly):
                o["wall_id"] = wid
                openings.append(o)
        rooms.append({"id": f"r{k}", "name": f"Room {k}", "polygon": pts[:-1].round(3).tolist(), "_poly": poly,
                      "floor_area_m2": measure(poly.area, "area"), "perimeter_m": round(poly.length, 3),
                      "ceiling_height_m": measure(rc - floor, "height") if rc else None,
                      "walls": walls, "openings": openings})
    adjacency = []
    for r in rooms:
        for o in r["openings"]:
            o["width_m"] = measure(o["width_m"], "opening")
            other = next((x["id"] for x in rooms if x is not r and x["_poly"].contains(Point(o["_probe"]))), None)
            o["connects_to"] = other
            del o["_probe"]
            if other and sorted([r["id"], other]) not in adjacency:
                adjacency.append(sorted([r["id"], other]))
    for r in rooms:
        del r["_poly"]
    result = {"capture": folder.name, "tier": "lidar", "frames_used": n_frames, "units": "metres",
              "ceiling_observed": ceil is not None, "rooms": rooms, "adjacency": adjacency,
              "total_floor_area_m2": measure(sum(r["floor_area_m2"]["value"] for r in rooms), "area")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    render(result, out / "plan.png")
    return result


def render(res, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    colors = ["#dbe9f6", "#e3f1df", "#fbeedd", "#efe2f3", "#f6e0e0"]
    for k, r in enumerate(res["rooms"]):
        pts = np.array(r["polygon"])
        ax.fill(*pts.T, color=colors[k % len(colors)])
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            ax.plot(*zip(a, b), color="#1a202c", lw=3)
            d = (b - a) / np.linalg.norm(b - a)
            mid = (a + b) / 2 + np.array([-d[1], d[0]]) * 0.18
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ax.text(*mid, f"{w['length_m']['value']:.2f}", rotation=ang if -90 < ang <= 90 else ang - 180,
                    ha="center", va="center", fontsize=7, color="#4a5568")
        for o in r["openings"]:
            w = next(x for x in r["walls"] if x["id"] == o["wall_id"])
            a, b = np.array(w["start"]), np.array(w["end"])
            d = (b - a) / np.linalg.norm(b - a)
            p0 = a + d * o["offset_m"]
            p1 = p0 + d * o["width_m"]["value"]
            ax.plot(*zip(p0, p1), color="white" if o["type"] == "door" else "#3182ce", lw=4)
            c = (p0 + p1) / 2 + np.array([-d[1], d[0]]) * 0.35
            ax.text(*c, f"{o['type']} {o['width_m']['value']:.2f}", ha="center", fontsize=6.5, color="#2c5282")
        c = Polygon(pts).representative_point()
        h = r["ceiling_height_m"]
        ax.text(c.x, c.y, f"{r['name']}\n{r['floor_area_m2']['value']:.2f} m²" + (f"\nceiling {h['value']:.2f} m" if h else ""),
                ha="center", va="center", fontsize=9, weight="bold")
    ax.set_title(f"{res['capture']}  -  total {res['total_floor_area_m2']['value']:.2f} m²")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    res = scan2plan(a.capture, a.out or Path("out") / a.capture.name)
    for r in res["rooms"]:
        h = r["ceiling_height_m"]
        print(f"{r['name']}: area {r['floor_area_m2']['value']:.2f} m2, ceiling {h['value'] if h else 'n/a'} m, "
              f"walls {[w['length_m']['value'] for w in r['walls']]}, "
              f"openings {[(o['type'], o['width_m']['value'], o['connects_to']) for o in r['openings']]}")
    print(f"total {res['total_floor_area_m2']['value']:.2f} m2 -> {a.out or Path('out') / a.capture.name}")
