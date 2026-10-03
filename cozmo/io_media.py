"""Image / video decoding helpers (HEIC photos, MOV/MP4 video) with an ffmpeg fallback."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos default to HEIC
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:  # pragma: no cover
    pass

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ffmpeg not found: `brew install ffmpeg` or `pip install imageio-ffmpeg`") from e


def load_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR)
    return np.asarray(im)


def focal_35mm(path: str | Path) -> float | None:
    """35 mm-equivalent focal length from EXIF, if present."""
    try:
        exif = Image.open(path).getexif()
        sub = exif.get_ifd(0x8769)
        f35 = sub.get(0xA405) or exif.get(0xA405)
        return float(f35) if f35 else None
    except Exception:
        return None


def intrinsics_from_35mm(f35: float, w: int, h: int) -> np.ndarray:
    """35 mm-equivalent focal is defined on the 43.27 mm diagonal."""
    f = f35 * np.hypot(w, h) / 43.27
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def intrinsics_from_hfov(hfov_deg: float, w: int, h: int) -> np.ndarray:
    f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def extract_video_frames(video: str | Path, out_dir: str | Path, fps: float, max_side: int = 960) -> list[Path]:
    """Decode a clip to JPEG frames at ``fps`` (rotation metadata is applied by ffmpeg)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("f_*.jpg"))
    stamp = out_dir / "params.json"
    params = {"fps": fps, "max_side": max_side, "src": str(Path(video).resolve())}
    if existing and stamp.exists() and json.loads(stamp.read_text()) == params:
        return existing
    for p in existing:
        p.unlink()
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    cmd = [ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(video),
           "-vf", f"fps={fps},{scale}", "-q:v", "2", str(out_dir / "f_%05d.jpg")]
    subprocess.run(cmd, check=True)
    stamp.write_text(json.dumps(params))
    return sorted(out_dir.glob("f_*.jpg"))


def sharpness(img: np.ndarray) -> float:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
