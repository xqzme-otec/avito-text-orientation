"""Real labelled text crops: TextOCR word crops (upright by construction).

The test boxes are horizontal text lines (width/height >= 1.5), so we keep only
crops of the same shape and drop tiny / degenerate ones.
"""
import io

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .common import DATA_DIR

TEXTOCR_DIR = DATA_DIR / "external" / "textocr"


def _decode(b: bytes) -> np.ndarray | None:
    # PIL instead of cv2.imdecode: some PNGs carry broken ICC profiles and
    # libpng floods stderr with warnings.
    try:
        return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))[:, :, ::-1].copy()
    except Exception:
        return None


def load_textocr(split: str = "test", max_n: int | None = None, min_ar: float = 1.5,
                 min_h: int = 10, min_len: int = 2, seed: int = 0):
    """Returns (list of BGR images, list of texts)."""
    files = sorted(TEXTOCR_DIR.glob(f"{split}-*.parquet"))
    imgs, texts = [], []
    for f in files:
        t = pq.read_table(f, columns=["image", "text"]).to_pydict()
        for im, tx in zip(t["image"], t["text"]):
            if tx is None or len(tx.strip()) < min_len or tx.strip() == ".":
                continue
            img = _decode(im["bytes"])
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < min_h or w / h < min_ar:
                continue
            imgs.append(img)
            texts.append(tx)
    if max_n is not None and len(imgs) > max_n:
        idx = np.random.RandomState(seed).choice(len(imgs), max_n, replace=False)
        imgs = [imgs[i] for i in idx]
        texts = [texts[i] for i in idx]
    return imgs, texts
