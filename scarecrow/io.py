from pathlib import Path

import cv2
import numpy as np


def image_paths(path: Path) -> list[Path]:
    """Resolve a file or directory to a list of image paths."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if path.is_file():
        return [path]
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)


def load(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to decode: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(img: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write: {p}")


def save_pattern(pattern: np.ndarray, path: str | Path) -> None:
    """Save grayscale pattern (float [0,1]) as PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.round(pattern * 255).clip(0, 255).astype(np.uint8)
    if not cv2.imwrite(str(p), arr):
        raise OSError(f"Failed to write: {p}")


def load_pattern(path: str | Path) -> np.ndarray:
    """Load pattern PNG as float32 array in [0, 1]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pattern not found: {p}")
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Failed to decode pattern: {p}")
    return m.astype(np.float32) / 255.0
