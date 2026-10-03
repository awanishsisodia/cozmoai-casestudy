"""Geometry back-end: posed depth frames -> rooms (polygons, heights, walls).

Pipeline (same for every tier):

1. fuse frames into a voxelised cloud with normals
2. level (photo/video only; LiDAR has gravity from ARKit) and rotate the
   world so walls are axis-aligned ("Manhattan frame")
3. floor / ceiling heights from horizontal-surface histograms
4. 2-D maps: free space carved by camera rays, ceiling coverage
5. room segmentation (continuous captures): connected components of the
   ceiling map -- door headers sit below the ceiling, so they cut it
6. per-room polygon: candidate wall lines from wall-point histograms (1 cm),
   grid cells between lines kept if mostly interior, unioned, slivers removed

Wall positions come from thousands of wall points (median), never from the
coarse grid, which is why the 5 cm map resolution does not limit accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .frames import Capture, Frame, frame_cloud, make_T, rot_z

MAP_RES = 0.05


@dataclass
class Cloud:
    P: np.ndarray
    N: np.ndarray
    good: np.ndarray

    def subset(self, m: np.ndarray) -> "Cloud":
        return Cloud(self.P[m], self.N[m], self.good[m])

    @property
    def vertical(self) -> np.ndarray:
        return self.good & (np.abs(self.N[:, 2]) < 0.2)


def build_cloud(frames: list[Frame], stride: int = 2, voxel: float = 0.015, max_range: float = 5.0) -> Cloud:
    Ps, Ns, Gs = [], [], []
    for f in frames:
        P, N, g = frame_cloud(f, stride=stride, max_range=max_range)
        Ps.append(P), Ns.append(N), Gs.append(g)
    P = np.concatenate(Ps) if Ps else np.zeros((0, 3))
    N = np.concatenate(Ns) if Ns else np.zeros((0, 3))
    G = np.concatenate(Gs) if Gs else np.zeros(0, bool)
    if len(P) == 0:
        return Cloud(P, N, G)
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Pm = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    gw = G.astype(float)
    Nm = np.stack([np.bincount(inv, N[:, k] * gw) for k in range(3)], 1)
    gc = np.bincount(inv, gw)
    nn = np.linalg.norm(Nm, axis=1)
    good = (gc > 0) & (nn > 0.5 * np.maximum(gc, 1e-9))   # consistent normals within the voxel
    Nm = np.divide(Nm, nn[:, None], out=np.zeros_like(Nm), where=nn[:, None] > 0)
    return Cloud(Pm, Nm, good)


def transform_frames(frames: list[Frame], T: np.ndarray) -> None:
    for f in frames:
        f.T_wc = T @ f.T_wc


# --------------------------------------------------------------------------- levelling & Manhattan

def estimate_up(cloud: Cloud, prior: np.ndarray) -> np.ndarray:
    """Refine the gravity direction from horizontal surfaces (floor, ceiling, tables)."""
    u = prior / np.linalg.norm(prior)
    N = cloud.N[cloud.good]
    for ang in (25, 12, 6):
        c = N @ u
        m = np.abs(c) > np.cos(np.radians(ang))
        if m.sum() < 50:
            break
        v = (N[m] * np.sign(c[m])[:, None]).sum(0)
        # walls should be perpendicular to up: blend in their constraint via SVD
        w = np.abs(c) < np.sin(np.radians(ang))
        if w.sum() > 50:
            Nw = N[w]
            _, _, Vt = np.linalg.svd(Nw.T @ Nw)
            nwall = Vt[-1] * np.sign(Vt[-1] @ u)
            v = v / np.linalg.norm(v) + nwall
        u = v / np.linalg.norm(v)
    return u


def level_capture(cap: Capture, cloud: Cloud | None = None) -> None:
    """Rotate world so that gravity is +z.  Prior: phones are held roughly upright (-y_cam is up)."""
    if cloud is None:
        cloud = build_cloud(cap.frames, stride=4)
    prior = np.mean([-f.T_wc[:3, 1] for f in cap.frames], axis=0)
    u = estimate_up(cloud, prior)
    z = np.array([0, 0, 1.0])
    v = np.cross(u, z)
    s, c = np.linalg.norm(v), u @ z
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    transform_frames(cap.frames, make_T(R, np.zeros(3)))


def manhattan_yaw(cloud: Cloud) -> tuple[float, float]:
    """Dominant wall direction (radians, in [-pi/4, pi/4)) and its concentration in [0, 1]."""
    m = cloud.vertical
    if m.sum() < 30:
        return 0.0, 0.0
    a = np.arctan2(cloud.N[m, 1], cloud.N[m, 0])
    z = np.exp(4j * a).mean()
    theta = np.angle(z) / 4
    # refine on inliers
    d = np.angle(np.exp(4j * (a - theta))) / 4
    inl = np.abs(d) < np.radians(5)
    if inl.sum() > 30:
        theta = theta + np.mean(d[inl])
    return float(np.angle(np.exp(4j * theta)) / 4), float(np.abs(z))


def align_manhattan(cap: Capture, cloud: Cloud) -> float:
    theta, _ = manhattan_yaw(cloud)
    transform_frames(cap.frames, make_T(rot_z(-theta), np.zeros(3)))
    return theta


# --------------------------------------------------------------------------- heights

def _peaks_1d(z: np.ndarray, lo: float, hi: float, res: float = 0.01, smooth: int = 2):
    if len(z) == 0 or hi <= lo:
        return np.array([]), np.array([]), lo
    bins = np.arange(lo, hi + res, res)
    h, _ = np.histogram(z, bins)
    h = ndimage.uniform_filter1d(h.astype(float), 2 * smooth + 1)
    is_pk = (h >= np.roll(h, 1)) & (h > np.roll(h, -1)) & (h > 0)
    idx = np.nonzero(is_pk)[0]
    return idx, h[idx], lo


def floor_ceiling(cloud: Cloud) -> tuple[float | None, float | None, dict]:
    """Floor and ceiling elevation (m) from horizontal surfaces + support statistics."""
    up = cloud.good & (cloud.N[:, 2] > 0.9)
    dn = cloud.good & (cloud.N[:, 2] < -0.9)
    zu, zd = cloud.P[up, 2], cloud.P[dn, 2]
    info = {"n_floor": int(up.sum()), "n_ceiling": int(dn.sum())}
    floor = None
    if len(zu) > 20:
        lo, hi = np.percentile(zu, 0.5) - 0.05, np.percentile(zu, 99.5) + 0.05
        idx, cnt, base = _peaks_1d(zu, lo, hi)
        if len(idx):
            ok = idx[cnt >= 0.15 * cnt.max()]
            z0 = base + (ok.min() + 0.5) * 0.01
            near = zu[np.abs(zu - z0) < 0.025]
            floor = float(np.median(near))
            info["floor_mad"] = float(np.median(np.abs(near - floor)))
            info["n_floor"] = int(len(near))
    ceil = None
    if len(zd) > 20:
        zmin = (floor if floor is not None else np.min(zd) - 2.5) + 2.0
        zd2 = zd[zd > zmin]
        if len(zd2) > 20:
            idx, cnt, base = _peaks_1d(zd2, zd2.min() - 0.05, zd2.max() + 0.05)
            if len(idx):
                z0 = base + (idx[np.argmax(cnt)] + 0.5) * 0.01
                near = zd2[np.abs(zd2 - z0) < 0.025]
                ceil = float(np.median(near))
                info["ceiling_mad"] = float(np.median(np.abs(near - ceil)))
                info["n_ceiling"] = int(len(near))
    return floor, ceil, info


# --------------------------------------------------------------------------- 2-D maps

@dataclass
class Grid:
    x0: float
    y0: float
    nx: int
    ny: int
    res: float = MAP_RES

    @classmethod
    def around(cls, xy: np.ndarray, pad: float = 0.5, res: float = MAP_RES) -> "Grid":
        lo = xy.min(0) - pad
        hi = xy.max(0) + pad
        n = np.ceil((hi - lo) / res).astype(int) + 1
        return cls(float(lo[0]), float(lo[1]), int(n[0]), int(n[1]), res)

    def idx(self, xy: np.ndarray):
        i = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        j = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return i, j, ok

    def count(self, xy: np.ndarray) -> np.ndarray:
        i, j, ok = self.idx(xy)
        m = np.zeros((self.nx, self.ny), np.int32)
        np.add.at(m, (i[ok], j[ok]), 1)
        return m

    def centers(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return xs, ys

    def sample(self, m: np.ndarray, xy: np.ndarray) -> np.ndarray:
        i, j, ok = self.idx(xy)
        out = np.zeros(len(xy), m.dtype)
        out[ok] = m[i[ok], j[ok]]
        return out


def carve_free_space(frames: list[Frame], grid: Grid, floor: float, ceil: float,
                     stride: int = 4, step: float = 0.04, max_frames: int = 400) -> np.ndarray:
    """2-D count of camera rays passing through each cell between floor and ceiling."""
    m = np.zeros((grid.nx, grid.ny), np.int32)
    sel = frames if len(frames) <= max_frames else [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    for f in sel:
        d = f.depth[::stride, ::stride]
        h, w = d.shape
        v, u = np.mgrid[0:h, 0:w]
        u = u * stride + (stride - 1) / 2
        v = v * stride + (stride - 1) / 2
        rays = np.stack([(u - f.K[0, 2]) / f.K[0, 0], (v - f.K[1, 2]) / f.K[1, 1], np.ones_like(u, float)], -1)
        ok = (d > 0.2) & (d < 5.0)
        if f.conf is not None:
            ok &= f.conf[::stride, ::stride]
        rays, d = rays[ok], d[ok]
        endpoints = (rays * d[:, None]) @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]
        c = f.T_wc[:3, 3]
        vec = endpoints - c
        L = np.linalg.norm(vec, axis=1)
        nmax = int(np.ceil(L.max() / step)) if len(L) else 0
        for k in range(1, nmax):
            t = k * step
            sel_r = L - 0.10 > t
            if not sel_r.any():
                break
            p = c + vec[sel_r] * (t / L[sel_r])[:, None]
            pz = (p[:, 2] > floor + 0.05) & (p[:, 2] < ceil - 0.05)
            i, j, okk = grid.idx(p[pz, :2])
            np.add.at(m, (i[okk], j[okk]), 1)
    return m


def disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def segment_rooms(cloud: Cloud, grid: Grid, floor: float, interior: np.ndarray,
                  min_area: float = 1.2) -> np.ndarray:
    """Label map of rooms from ceiling coverage (door headers interrupt the ceiling)."""
    dn = cloud.good & (cloud.N[:, 2] < -0.9) & (cloud.P[:, 2] > floor + 2.12)
    ceil = grid.count(cloud.P[dn, :2]) > 0
    ceil = ndimage.binary_fill_holes(ceil)
    # label the *eroded* map so thin bridges (a wall a cell or two thick) cannot merge rooms,
    # then grow the labels back over the original ceiling coverage
    core = ndimage.binary_erosion(ceil, structure=disk(4))
    lab, n = ndimage.label(core)
    keep = [k for k in range(1, n + 1) if (lab == k).sum() * grid.res ** 2 >= min_area / 2]
    out = np.zeros_like(lab)
    for new, k in enumerate(keep, 1):
        out[lab == k] = new
    if out.max() > 0:
        _, (ii, jj) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        grown = out[ii, jj]
        grown[~ceil] = 0
        out = grown
    if out.max() == 0:  # no ceiling seen: single room = interior
        return interior.astype(np.int32)
    # assign interior cells to the nearest ceiling component (within 1 m)
    dist, (ii, jj) = ndimage.distance_transform_edt(out == 0, return_indices=True)
    full = out[ii, jj]
    full[(dist * grid.res > 1.0) | ~interior] = 0
    full[out > 0] = out[out > 0]
    return full


# --------------------------------------------------------------------------- polygons

@dataclass
class WallLine:
    axis: int          # 0: line x = pos (wall normal along x); 1: line y = pos
    pos: float
    support_m2: float
    inferred: bool = False


@dataclass
class Room:
    id: str
    polygon: Polygon
    floor_z: float | None
    ceiling_z: float | None
    height_info: dict = field(default_factory=dict)
    walls: list = field(default_factory=list)       # list[dict]
    openings: list = field(default_factory=list)    # list[dict]
    name: str = ""
    group: str = ""
    frame_ids: list = field(default_factory=list)
    kind: str = "room"

    @property
    def ceiling_height(self):
        if self.floor_z is None or self.ceiling_z is None:
            return None
        return self.ceiling_z - self.floor_z


def wall_lines(cloud: Cloud, axis: int, min_support: float = 0.12) -> list[WallLine]:
    m = cloud.good & (np.abs(cloud.N[:, axis]) > 0.92) & (np.abs(cloud.N[:, 2]) < 0.2)
    if m.sum() < 20:
        return []
    P = cloud.P[m]
    x = P[:, axis]
    lines: list[WallLine] = []
    bins = np.arange(x.min() - 0.02, x.max() + 0.03, 0.01)
    h, _ = np.histogram(x, bins)
    hs = ndimage.uniform_filter1d(h.astype(float), 3)
    # only true local maxima are candidates; noise tails of a strong wall are shoulders, not peaks
    is_pk = (hs >= np.roll(hs, 1)) & (hs >= np.roll(hs, -1)) & (hs > 0)
    order = [i for i in np.argsort(-hs) if is_pk[i]]
    used = np.zeros(len(hs), bool)
    other = 1 - axis
    for i in order:
        if used[max(0, i - 5):i + 6].any():
            continue
        used[i] = True
        c = bins[i] + 0.005
        sel = np.abs(x - c) < 0.025
        if sel.sum() < 10:
            continue
        cells = np.unique(np.floor(np.stack([P[sel, other], P[sel, 2]], 1) / 0.05).astype(int), axis=0)
        sup = len(cells) * 0.0025
        if sup < min_support:
            continue
        lines.append(WallLine(axis, float(np.median(x[sel])), float(sup)))
    return lines


def polygon_from_lines(mask: np.ndarray, grid: Grid, xl: list[WallLine], yl: list[WallLine],
                       frac: float = 0.5) -> Polygon | None:
    if not mask.any():
        return None
    ii, jj = np.nonzero(mask)
    xs, ys = grid.centers()
    ext = (xs[ii].min() - grid.res / 2, xs[ii].max() + grid.res / 2, ys[jj].min() - grid.res / 2, ys[jj].max() + grid.res / 2)

    def merge(lines, lo, hi):
        pos = sorted(l.pos for l in lines)
        for b in (lo, hi):  # mask extents as fallback boundaries
            if not pos or min(abs(p - b) for p in pos) > 0.12:
                pos.append(b)
        return sorted(pos)

    X = np.array(merge(xl, ext[0], ext[1]))
    Y = np.array(merge(yl, ext[2], ext[3]))
    wx, wy = np.diff(X), np.diff(Y)
    sel = np.zeros((len(wx), len(wy)), bool)
    for a in range(len(wx)):
        for b in range(len(wy)):
            xa, xb, ya, yb = X[a], X[a + 1], Y[b], Y[b + 1]
            sx = np.arange(xa + min(0.01, wx[a] / 4), xb, max(min(grid.res, wx[a] / 3), 1e-3))
            sy = np.arange(ya + min(0.01, wy[b] / 4), yb, max(min(grid.res, wy[b] / 3), 1e-3))
            gx, gy = np.meshgrid(sx, sy)
            sel[a, b] = grid.sample(mask, np.stack([gx.ravel(), gy.ravel()], 1)).mean() >= frac
    # Thin boundary strips (< 16 cm, outside on one side) are door reveals / wall thickness, not room.
    thin = 0.16
    for _ in range(3):
        pad = np.pad(sel, 1)
        out_x = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1]
        out_y = ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
        drop = sel & (((wx < thin)[:, None] & out_x) | ((wy < thin)[None, :] & out_y))
        if not drop.any():
            break
        sel &= ~drop
    cells = [box(X[a], Y[b], X[a + 1], Y[b + 1]) for a, b in zip(*np.nonzero(sel))]
    if not cells:
        return None
    poly = unary_union(cells)
    poly = poly.buffer(-0.09, join_style=2).buffer(0.09, join_style=2)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty:
        return None
    poly = Polygon(poly.exterior).simplify(0.004)
    return orient(poly, 1.0)


def snap_polygon(poly: Polygon, lines_x: list[WallLine], lines_y: list[WallLine], tol: float = 0.04) -> Polygon:
    """Snap axis-aligned polygon edges onto measured wall lines (sub-centimetre positions)."""
    pts = np.array(poly.exterior.coords)[:-1]
    for axis, lines in ((0, lines_x), (1, lines_y)):
        if not lines:
            continue
        pos = np.array([l.pos for l in lines])
        for k in range(len(pts)):
            d = np.abs(pos - pts[k, axis])
            if d.min() < tol:
                pts[k, axis] = pos[np.argmin(d)]
    return orient(Polygon(pts), 1.0)


def describe_walls(room: Room, cloud: Cloud) -> list[dict]:
    """Per-edge wall records: length, normal, and how much of it was actually observed."""
    pts = np.array(room.polygon.exterior.coords)
    vert = cloud.vertical
    P = cloud.P[vert]
    N = cloud.N[vert]
    fz = room.floor_z if room.floor_z is not None else P[:, 2].min()
    cz = room.ceiling_z if room.ceiling_z is not None else fz + 2.5
    band = (P[:, 2] > fz + 0.1) & (P[:, 2] < cz - 0.1)
    walls = []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        L = float(np.linalg.norm(b - a))
        u = (b - a) / max(L, 1e-9)
        n_in = np.array([-u[1], u[0]])  # CCW polygon -> left normal points inside
        sel = band & (np.abs(N[:, :2] @ n_in) > 0.85)
        rel = P[sel, :2] - a
        dist = rel @ n_in
        s = rel @ u
        near = (np.abs(dist) < 0.05) & (s > -0.02) & (s < L + 0.02)
        if near.sum():
            covered = np.unique(np.floor(s[near] / 0.05).astype(int))
            obs = float(np.clip(len(covered) * 0.05 / max(L, 0.05), 0, 1))
            spread = float(np.std(dist[near]))
        else:
            obs, spread = 0.0, None
        walls.append({"id": f"{room.id}_w{k + 1}", "start": a.round(4).tolist(), "end": b.round(4).tolist(),
                      "length_m": L, "normal_in": n_in.round(4).tolist(), "observed_fraction": obs,
                      "plane_spread_m": spread, "n_points": int(near.sum())})
    return walls


def extract_room(room_id: str, mask: np.ndarray, grid: Grid, cloud: Cloud) -> Room | None:
    """Polygon, heights and walls for one room given its interior mask."""
    ii, jj = np.nonzero(mask)
    if len(ii) == 0:
        return None
    xs, ys = grid.centers()
    dil = ndimage.binary_dilation(mask, structure=disk(int(round(0.3 / grid.res))))
    near = grid.sample(dil, cloud.P[:, :2]).astype(bool)
    local = cloud.subset(near)
    fz, cz, hinfo = floor_ceiling(local)
    xl = wall_lines(local, 0)
    yl = wall_lines(local, 1)
    poly = polygon_from_lines(mask, grid, xl, yl)
    if poly is None or poly.area < 0.8:
        return None
    poly = snap_polygon(poly, xl, yl)
    # heights restricted to the room proper (avoid neighbouring ceilings through doors)
    inner = poly.buffer(-0.15)
    if not inner.is_empty:
        from shapely import contains_xy
        inside = contains_xy(inner, local.P[:, 0], local.P[:, 1])
        if inside.sum() > 100:
            f2, c2, h2 = floor_ceiling(local.subset(inside))
            fz, cz, hinfo = (f2 if f2 is not None else fz), (c2 if c2 is not None else cz), h2
    room = Room(id=room_id, polygon=poly, floor_z=fz, ceiling_z=cz, height_info=hinfo)
    room.walls = describe_walls(room, local)
    return room
