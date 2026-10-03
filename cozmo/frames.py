"""Common capture representation shared by every tier.

Every input tier (photos, video, LiDAR) is converted into a list of posed
RGB-D frames.  From there on, one geometry back-end does the work, so the
tiers differ only in how good their depth and poses are -- which is exactly
what the per-tier calibration of the confidence intervals captures.

Conventions
-----------
* World frame: metres, z points up (gravity), x/y horizontal.
* Camera frame: OpenCV (x right, y down, z forward).
* ``T_wc``: 4x4 camera-to-world transform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Frame:
    depth: np.ndarray                 # (H, W) float32 metres, 0 = invalid
    K: np.ndarray                     # (3, 3) intrinsics at depth resolution
    T_wc: np.ndarray                  # (4, 4) camera-to-world
    idx: int = 0
    timestamp: float = 0.0
    group: str = "capture"            # room folder name for the photo tier
    conf: Optional[np.ndarray] = None  # (H, W) bool, True = trustworthy depth
    rgb_loader: Optional[Callable[[], np.ndarray]] = None  # lazy (h, w, 3) uint8
    rgb_path: Optional[str] = None

    def rgb(self) -> Optional[np.ndarray]:
        return self.rgb_loader() if self.rgb_loader else None

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]


@dataclass
class Capture:
    tier: str                          # "photo" | "video" | "lidar"
    frames: list[Frame]
    name: str = "capture"
    continuous: bool = True            # one trajectory (video/lidar) vs per-room folders
    meta: dict = field(default_factory=dict)

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for f in self.frames:
            if f.group not in seen:
                seen.append(f.group)
        return seen


# --------------------------------------------------------------------------- geometry helpers

def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def camera_points(depth: np.ndarray, K: np.ndarray, stride: int = 1):
    """Back-project a depth map to camera-frame points.  Returns (pts (h,w,3), valid (h,w))."""
    d = depth[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride + (stride - 1) / 2.0
    v = v * stride + (stride - 1) / 2.0
    x = (u - K[0, 2]) / K[0, 0] * d
    y = (v - K[1, 2]) / K[1, 1] * d
    pts = np.stack([x, y, d], axis=-1)
    return pts, d > 0


def camera_normals(pts: np.ndarray) -> np.ndarray:
    """Normals from an organised point grid via central differences, oriented toward the camera."""
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-12)
    # orient toward camera (camera at origin): n . p < 0
    flip = np.sum(n * pts, axis=-1) > 0
    n[flip] *= -1
    return n


def frame_cloud(f: Frame, stride: int = 2, max_range: float = 5.0, min_range: float = 0.2):
    """World-frame points, normals and per-point ray info for one frame.

    Normals at depth discontinuities are unreliable; they are rejected by
    requiring neighbouring depths to agree within a relative tolerance.
    """
    pts, valid = camera_points(f.depth, f.K, stride)
    if f.conf is not None:
        valid &= f.conf[::stride, ::stride]
    d = pts[..., 2]
    valid &= (d > min_range) & (d < max_range)
    # smoothness test for normals
    smooth = np.ones_like(valid)
    dd = np.abs(np.diff(d, axis=1))
    smooth[:, 1:] &= dd < 0.04 * d[:, 1:] + 0.01
    smooth[:, :-1] &= dd < 0.04 * d[:, 1:] + 0.01
    dd = np.abs(np.diff(d, axis=0))
    smooth[1:, :] &= dd < 0.04 * d[1:, :] + 0.01
    smooth[:-1, :] &= dd < 0.04 * d[1:, :] + 0.01
    n = camera_normals(pts)
    R, t = f.T_wc[:3, :3], f.T_wc[:3, 3]
    P = pts[valid] @ R.T + t
    N = n[valid] @ R.T
    good_n = smooth[valid] & (np.linalg.norm(N, axis=1) > 0.5)
    return P, N, good_n
