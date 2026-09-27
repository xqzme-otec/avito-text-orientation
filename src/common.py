"""Shared helpers: paths, seeding, image I/O and metrics."""
import os
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test" / "test" / "images"
SAMPLE_SUB = ROOT / "test" / "sample_submission.csv"
MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def read_bgr(path) -> np.ndarray:
    """Read an image as uint8 BGR (PaddleOCR models were trained on BGR)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def rot180(img: np.ndarray) -> np.ndarray:
    return img[::-1, ::-1].copy()


def brier(p, y) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def report(name: str, p, y) -> dict:
    """Print 1-Brier, accuracy and log-loss for a vector of p_180 predictions."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    res = {
        "1-brier": 1 - brier(p, y),
        "acc": float(np.mean((p > 0.5) == (y > 0.5))),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "n": int(len(y)),
    }
    print(f"{name:40s} 1-Brier={res['1-brier']:.4f} acc={res['acc']:.4f} "
          f"logloss={res['logloss']:.4f} n={res['n']}")
    return res


def preprocess_batch(imgs):
    """List of BGR crops -> (N,3,H,W) tensor and (N,W) mask tensor."""
    import torch

    from .model import preprocess

    xs, ms = zip(*(preprocess(im) for im in imgs))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))


def fit_temperature(logits, y) -> float:
    """Temperature T minimising the Brier score of sigmoid(logit / T).
    The model is antisymmetric (p(rot x) = 1 - p(x)), so no bias term is needed."""
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y, np.float64)
    grid = np.exp(np.linspace(np.log(0.2), np.log(5.0), 801))
    scores = [brier(1 / (1 + np.exp(-logits / t)), y) for t in grid]
    return float(grid[int(np.argmin(scores))])
