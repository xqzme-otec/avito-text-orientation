"""Torch datasets: on-the-fly synthetic crops + real TextOCR crops, and fixed
validation sets."""
import io
import pickle
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import ARTIFACTS_DIR, rot180, seed_everything
from .data_real import TEXTOCR_DIR, load_textocr
from .model import preprocess
from .synth import Renderer, TextSampler, degrade, list_fonts


def load_textocr_bytes(split: str = "train", min_ar: float = 1.5, min_h: int = 10):
    """Encoded TextOCR crops (kept compressed in RAM, decoded in workers)."""
    out, latin = [], []
    for f in sorted(TEXTOCR_DIR.glob(f"{split}-*.parquet")):
        t = pq.read_table(f, columns=["image", "text"]).to_pydict()
        for im, tx in zip(t["image"], t["text"]):
            if tx is None or len(tx.strip()) < 2:
                continue
            b = im["bytes"]
            try:
                w, h = Image.open(io.BytesIO(b)).size  # header only
            except Exception:
                continue
            if h >= min_h and w / h >= min_ar:
                out.append(b)
                latin.append(tx)
    return out, latin


class BytesStore:
    """Encoded images concatenated in one file + offsets; opened with np.memmap
    lazily, so DataLoader workers don't receive a pickled copy of ~1.4 GB."""

    def __init__(self, path):
        self.path = str(path)
        self.offsets = np.load(self.path + ".idx.npy")
        self._mm = None

    @staticmethod
    def build(path, blobs):
        path = str(path)
        offsets = np.zeros(len(blobs) + 1, np.int64)
        offsets[1:] = np.cumsum([len(b) for b in blobs])
        with open(path, "wb") as f:
            for b in blobs:
                f.write(b)
        np.save(path + ".idx.npy", offsets)
        return BytesStore(path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.memmap(self.path, np.uint8, "r")
        return self._mm[self.offsets[i]:self.offsets[i + 1]].tobytes()

    def __getstate__(self):
        return {"path": self.path, "offsets": self.offsets, "_mm": None}


def textocr_store(split: str = "train"):
    """(BytesStore of filtered TextOCR crops, their texts) cached in artifacts/."""
    path = ARTIFACTS_DIR / f"textocr_{split}.bin"
    txt = ARTIFACTS_DIR / f"textocr_{split}.txt"
    if not path.exists():
        blobs, texts = load_textocr_bytes(split)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        BytesStore.build(path, blobs)
        txt.write_text("\n".join(t.replace("\n", " ") for t in texts))
    return BytesStore(path), txt.read_text().split("\n")


class TrainDataset(Dataset):
    """Mix of synthetic (prob p_synth) and real crops, each flipped w.p. 0.5."""

    def __init__(self, real_bytes, length: int, p_synth: float = 0.6, seed: int = 0,
                 extra_latin=None):
        # every index is a distinct deterministic sample -> iterate once over
        # range(total_steps * batch) instead of re-spawning workers per epoch
        self.real = real_bytes
        self.length = length
        self.p_synth = p_synth
        self.seed = seed
        # built once in the main process (reading Wikipedia in every worker is
        # slow and memory hungry); workers get a copy of the ~30 MB phrase list
        self._renderer = Renderer(list_fonts("train"), TextSampler("train", extra_latin=extra_latin))

    def renderer(self):
        return self._renderer

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        r = random.Random(self.seed * 1_000_000_007 + i)
        if r.random() < self.p_synth or not len(self.real):
            img = self.renderer().render(r)
        else:
            b = self.real[r.randrange(len(self.real))]
            img = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))[:, :, ::-1]
            if r.random() < 0.5:
                img = degrade(r, np.ascontiguousarray(img))
        y = r.random() < 0.5
        if y:
            img = rot180(img)
        x, m = preprocess(img)
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(float(y))


class ArrayDataset(Dataset):
    def __init__(self, imgs, ys=None):
        self.imgs, self.ys = imgs, ys

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x, m = preprocess(self.imgs[i])
        y = 0.0 if self.ys is None else float(self.ys[i])
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(y)


def build_val_sets(n_real: int = 3000, n_syn: int = 4000, seed: int = 123):
    """Fixed labelled validation sets (cached in artifacts/val_sets.pkl):
    * real_en: TextOCR *test* split crops (real photos, English)
    * syn_ru : synthetic crops with held-out fonts and held-out Wikipedia articles
    """
    path = ARTIFACTS_DIR / "val_sets.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    seed_everything(seed)
    rs = np.random.RandomState(seed)
    imgs, _ = load_textocr("test", max_n=n_real, seed=seed)
    y = rs.rand(len(imgs)) < 0.5
    real = ([rot180(im) if f else im for im, f in zip(imgs, y)], y.astype(np.float32))

    r = random.Random(seed)
    rend = Renderer(list_fonts("val"), TextSampler("val", n_docs=5000, seed=seed))
    s_imgs = [rend.render(r) for _ in range(n_syn)]
    ys = rs.rand(n_syn) < 0.5
    syn = ([rot180(im) if f else im for im, f in zip(s_imgs, ys)], ys.astype(np.float32))
    sets = {"real_en": real, "syn_ru": syn}
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    path.write_bytes(pickle.dumps(sets))
    return sets
