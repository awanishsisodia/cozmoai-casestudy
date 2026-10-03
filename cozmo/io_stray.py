"""Loader for Stray Scanner exports (LiDAR tier).

A Stray Scanner dataset folder contains::

    rgb.mp4               colour video (1920x1440)
    depth/000000.png      uint16 depth in millimetres (256x192)
    confidence/000000.png ARKit confidence 0/1/2
    odometry.csv          timestamp, frame, x, y, z, qx, qy, qz, qw  (ARKit camera-to-world)
    camera_matrix.csv     3x3 intrinsics at RGB resolution
    imu.csv               (unused)

ARKit uses a y-up world and a camera looking down -z.  We convert to our
z-up world and OpenCV camera here, once, so nothing downstream cares.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .frames import Capture, Frame, quat_to_R
from .io_media import extract_video_frames, load_image

# ARKit world (x, y-up, z) -> our world (x, -z, y)  [z-up]
A_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
# OpenCV camera -> ARKit camera axes
C_CV2ARKIT = np.diag([1.0, -1.0, -1.0])


def is_stray(path: Path) -> bool:
    return path.is_dir() and (path / "odometry.csv").exists() and (path / "depth").is_dir()


def read_odometry(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0] or parts[0][0].isalpha():
            continue
        rows.append([float(p) for p in parts[:9]])
    return np.array(rows)


def arkit_to_T(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    R_ark = quat_to_R(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = A_WORLD @ R_ark @ C_CV2ARKIT
    T[:3, 3] = A_WORLD @ np.array([x, y, z])
    return T


def load_stray(path: str | Path, target_hz: float = 6.0, min_motion: float = 0.03,
               min_conf: int = 2, rgb_cache: Path | None = None) -> Capture:
    path = Path(path)
    odo = read_odometry(path / "odometry.csv")
    K_rgb = np.loadtxt(path / "camera_matrix.csv", delimiter=",")
    depth_files = sorted((path / "depth").glob("*.png"))
    if not depth_files:
        raise FileNotFoundError(f"no depth frames in {path}")
    d0 = np.asarray(Image.open(depth_files[0]))
    dh, dw = d0.shape
    rgb_w = 2 * K_rgb[0, 2]  # principal point is ~centre
    s = dw / rgb_w
    K = K_rgb.copy()
    K[:2] *= s

    # frame sub-sampling: time-based plus a motion gate
    ts = odo[:, 0]
    step = max(1, int(round((1.0 / target_hz) / max(np.median(np.diff(ts)), 1e-3))))
    chosen = []
    last_T = None
    for i in range(0, len(odo), step):
        T = arkit_to_T(*odo[i, 2:9])
        if last_T is not None:
            dt = np.linalg.norm(T[:3, 3] - last_T[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ last_T[:3, :3]) - 1) / 2, -1, 1)))
            if dt < min_motion and dr < 3.0:
                continue
        chosen.append((i, T))
        last_T = T

    rgb_frames: list[Path] = []
    if (path / "rgb.mp4").exists():
        cache = rgb_cache or (path / ".cozmo_rgb")
        # decode only the chosen frames lazily via a full extraction at native fps index
        rgb_frames = extract_video_frames(path / "rgb.mp4", cache, fps=60.0 / step if step else 6.0)

    frames = []
    for k, (i, T) in enumerate(chosen):
        fidx = int(odo[i, 1])
        dfile = path / "depth" / f"{fidx:06d}.png"
        if not dfile.exists():
            continue
        depth = np.asarray(Image.open(dfile)).astype(np.float32) / 1000.0
        cfile = path / "confidence" / f"{fidx:06d}.png"
        conf = (np.asarray(Image.open(cfile)) >= min_conf) if cfile.exists() else None
        rgb_path = None
        if rgb_frames:
            j = min(len(rgb_frames) - 1, i // step)
            rgb_path = str(rgb_frames[j])
        frames.append(Frame(depth=depth, K=K, T_wc=T, idx=fidx, timestamp=float(odo[i, 0]),
                            conf=conf, rgb_path=rgb_path,
                            rgb_loader=(lambda p=rgb_path: load_image(p)) if rgb_path else None))
    return Capture(tier="lidar", frames=frames, name=path.name, continuous=True,
                   meta={"source": "stray_scanner", "n_raw_frames": int(len(odo)),
                         "n_frames_used": len(frames), "depth_res": [dw, dh]})
