"""Opening (door / window / passage) detection by ray see-through voting.

For every wall we look at the plane 3 cm *behind* its face (inside the wall
thickness, where only the jambs bound an opening).  Every camera ray that
crosses that plane votes:

* ``see``     - the ray's depth sample lies beyond the plane (by more than the
                sensor noise): the ray crossed the plane, so that point of the
                plane is empty -- even if the ray then hits the jamb.
* ``block``   - the sample lies on the wall face, i.e. before the plane.
* ``nodata``  - the ray returned no depth (glass, sky, very dark, > range).
* rays ending well in front of the wall (furniture) carry no information.

Votes are accumulated on a 1 cm (along wall) x 5 cm (height) grid.  Columns
that are see-through from near the floor to door height are doors; columns
that are see-through/no-data at window height with a solid sill are windows.
Width is the length of the qualifying run, so its resolution is 1 cm.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from .frames import Frame

DS = 0.01    # along-wall resolution
DZ = 0.05    # vertical resolution
BEHIND = 0.04       # test plane depth behind the wall face
# sample must lie this far beyond the plane to count as "see": a + b * range.
# LiDAR noise is ~1 cm so a few mm suffice; monocular depth needs a range-proportional margin.
SEE_MARGIN = {"lidar": (0.005, 0.0), "video": (0.03, 0.04), "photo": (0.03, 0.04)}


def _rays(f: Frame, stride: int):
    d = f.depth[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride + (stride - 1) / 2
    v = v * stride + (stride - 1) / 2
    r = np.stack([(u - f.K[0, 2]) / f.K[0, 0], (v - f.K[1, 2]) / f.K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d.reshape(-1)
    valid = d > 0
    if f.conf is not None:
        valid &= f.conf[::stride, ::stride].reshape(-1)
    rng = d * np.linalg.norm(r, axis=1)          # depth -> range along ray
    dirs = r / np.linalg.norm(r, axis=1, keepdims=True)
    return dirs @ f.T_wc[:3, :3].T, rng, valid


def vote_walls(frames: list[Frame], walls: list[dict], floor_z: float, ceil_z: float,
               stride: int = 3, max_frames: int = 350, tier: str = "lidar") -> list[dict]:
    """Accumulate see/block/nodata votes for each wall; returns per-wall vote grids."""
    H = max(ceil_z - floor_z, 1.0)
    m_a, m_b = SEE_MARGIN.get(tier, SEE_MARGIN["photo"])
    grids = []
    for w in walls:
        L = w["length_m"]
        ns, nz = int(np.ceil(L / DS)) + 1, int(np.ceil(H / DZ)) + 1
        grids.append({k: np.zeros((ns, nz), np.int32) for k in ("see", "block", "nodata")})
    sel = frames if len(frames) <= max_frames else [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    for f in sel:
        dirs, rng, valid = _rays(f, stride)
        c = f.center
        for w, g in zip(walls, grids):
            a = np.array(w["start"])
            b = np.array(w["end"])
            L = w["length_m"]
            u = (b - a) / max(L, 1e-9)
            n = np.array(w["normal_in"])
            h_cam = (c[:2] - a) @ n
            if h_cam < 0.2:                       # camera must be inside, in front of the wall
                continue
            dn = dirs[:, :2] @ n
            ok = dn < -0.05
            t_w = np.full(len(dirs), np.inf)
            t_w[ok] = (-BEHIND - h_cam) / dn[ok]
            q = c + dirs * t_w[:, None]
            s = (q[:, :2] - a) @ u
            z = q[:, 2] - floor_z
            ok &= (s >= 0) & (s < L) & (z >= 0) & (z < H) & (t_w < 6.0)
            if not ok.any():
                continue
            si = (s[ok] / DS).astype(int)
            zi = (z[ok] / DZ).astype(int)
            tv, rv, vv, dv = t_w[ok], rng[ok], valid[ok], dirs[ok]
            margin = m_a + m_b * tv
            see = vv & (rv > tv + margin)
            nod = ~vv
            np.add.at(g["see"], (si[see], zi[see]), 1)
            np.add.at(g["nodata"], (si[nod], zi[nod]), 1)
            # "block" is credited where the ray actually hit the wall face, not where it would
            # have crossed the test plane: an oblique ray stopped by the face just outside an
            # opening must not vote "solid" for a cell inside it.
            hit = c + dv * np.where(vv, rv, 0)[:, None]
            h_hit = (hit[:, :2] - a) @ n
            block = vv & ~see & (h_hit < 0.03 + margin) & (h_hit > -BEHIND - 0.10 - margin)
            if block.any():
                sb = (hit[block, :2] - a) @ u
                zb = hit[block, 2] - floor_z
                okb = (sb >= 0) & (sb < L) & (zb >= 0) & (zb < H)
                np.add.at(g["block"], ((sb[okb] / DS).astype(int), (zb[okb] / DZ).astype(int)), 1)
    return grids


def _runs(mask: np.ndarray, max_gap: int):
    runs, start, gap = [], None, 0
    for i, v in enumerate(np.append(mask, False)):
        if v:
            if start is None:
                start = i
            gap = 0
            end = i
        elif start is not None:
            gap += 1
            if gap > max_gap or i == len(mask):
                runs.append((start, end + 1))
                start, gap = None, 0
    return runs


def detect_openings(grid: dict, wall: dict, height: float, tier: str = "lidar") -> list[dict]:
    see, block, nod = grid["see"], grid["block"], grid["nodata"]
    informative = see + block
    # per-cell majority; sparse cells (few rays near the floor) count when uncontested
    open_c = (see >= 1) & (see >= 0.5 * block)
    solid_c = (block >= 1) & ~open_c
    win_c = open_c | ((nod >= 3) & (nod >= block))
    zc = (np.arange(see.shape[1]) + 0.5) * DZ

    def band(lo, hi):
        return (zc >= lo) & (zc <= hi)

    door_band, top_band = band(0.15, 1.80), band(1.85, height - 0.1)
    sill_band, glass_band = band(0.15, 0.55), band(1.0, 1.55)
    nb = door_band.sum()
    inf_d = (informative[:, door_band] > 0).sum(1)
    door_frac = open_c[:, door_band].sum(1) / np.maximum(inf_d, 1)
    door_cov = inf_d / max(nb, 1)
    is_door = (door_frac >= 0.75) & (door_cov >= 0.4)
    is_win = (win_c[:, glass_band].mean(1) >= 0.75) & (solid_c[:, sill_band].mean(1) >= 0.5) & ~is_door

    out = []
    a, b = np.array(wall["start"]), np.array(wall["end"])
    u = (b - a) / max(wall["length_m"], 1e-9)
    for kind, m, min_w in (("door", is_door, 0.45), ("window", is_win, 0.30)):
        for s0, s1 in _runs(m, max_gap=3):
            width = (s1 - s0) * DS
            if width < min_w:
                continue
            cols = slice(s0, s1)
            prof = open_c[cols].mean(0) if kind == "door" else win_c[cols].mean(0)
            openz = zc[prof >= 0.5]
            top = float(openz.max() + DZ / 2) if len(openz) else None
            bottom = float(openz.min() - DZ / 2) if len(openz) else None
            k = kind
            if kind == "door" and top is not None and top >= height - 0.12:
                k = "passage"        # open to the ceiling: cased opening / open plan
            if kind == "door" and width > 1.6:
                k = "passage"
            mid = (s0 + s1) / 2 * DS
            out.append({
                "type": k, "wall_id": wall["id"], "width_m": float(width),
                "offset_m": float(s0 * DS), "center": (a + u * mid).round(4).tolist(),
                "sill_m": (None if kind == "door" else bottom), "head_m": top,
                "evidence": {"see_votes": int(see[cols].sum()), "block_votes": int(block[cols].sum()),
                             "nodata_votes": int(nod[cols].sum()),
                             "edge_sharpness": _edge_sharpness(m, s0, s1)},
            })
    return out


def _edge_sharpness(m: np.ndarray, s0: int, s1: int) -> float:
    """Fraction of the 3 cm on each side of the run that is clearly solid (1 = crisp jamb edges)."""
    left = m[max(0, s0 - 3):s0]
    right = m[s1:s1 + 3]
    side = np.concatenate([left, right])
    return float(1 - side.mean()) if len(side) else 0.0


def link_openings(rooms) -> list[tuple[str, str, str]]:
    """Connect doors/passages to the room on the other side; merge the two sides of one opening."""
    edges = []
    for r in rooms:
        for op in r.openings:
            if op["type"] == "window":
                continue
            wall = next(w for w in r.walls if w["id"] == op["wall_id"])
            n = np.array(wall["normal_in"])
            probe = np.array(op["center"]) - n * 0.40
            other = next((o for o in rooms if o is not r and o.polygon.buffer(0.05).contains(Point(probe))), None)
            op["connects_to"] = other.id if other else None
    # merge duplicate observations of the same physical opening
    nid = 0
    for r in rooms:
        for op in r.openings:
            if "opening_id" in op:
                continue
            nid += 1
            op["opening_id"] = f"op{nid}"
            o_id = op.get("connects_to")
            if not o_id:
                continue
            other = next(o for o in rooms if o.id == o_id)
            twin = None
            for op2 in other.openings:
                if op2.get("connects_to") == r.id and "opening_id" not in op2 and \
                        np.linalg.norm(np.array(op2["center"]) - np.array(op["center"])) < 0.5:
                    twin = op2
                    break
            if twin:
                twin["opening_id"] = op["opening_id"]
                w1, w2 = op["evidence"]["see_votes"], twin["evidence"]["see_votes"]
                width = (op["width_m"] * w1 + twin["width_m"] * w2) / max(w1 + w2, 1)
                op["width_m_single_side"], twin["width_m_single_side"] = op["width_m"], twin["width_m"]
                op["width_m"] = twin["width_m"] = float(width)
                op["seen_from_both_sides"] = twin["seen_from_both_sides"] = True
            edges.append((r.id, o_id, op["opening_id"]))
    return edges
