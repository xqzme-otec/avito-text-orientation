"""Inference pipeline shared by the scripts and the notebook:

    BGR crops -> CNN logit + PaddleOCR recognizer features -> stacker -> p_180
"""
from pathlib import Path

import numpy as np
import torch

from .common import MODELS_DIR, preprocess_batch
from .model import OrientNet
from .pretrained import RecScorer
from .stacking import REC_KEYS, AntisymLR, build_features

CNN_DIR = MODELS_DIR / "w24_v1"          # final CNN (weights shipped with the repo)
STACKER_PATH = MODELS_DIR / "stacker.json"
REC_MODELS = {"rec_eslav": "eslav_PP-OCRv5_rec_mobile.onnx",
              "rec_cyr": "cyrillic_PP-OCRv5_rec_mobile.onnx"}
assert tuple(REC_MODELS) == REC_KEYS


def load_cnn(model_dir: Path = CNN_DIR) -> OrientNet:
    ck = torch.load(Path(model_dir) / "best.pt", map_location="cpu")
    model = OrientNet(ck["width"])
    model.load_state_dict(ck["state_dict"])
    return model.eval()


@torch.no_grad()
def cnn_logits(model: OrientNet, imgs, bs: int = 512) -> np.ndarray:
    """Uncalibrated logit of P(rotated by 180) for each crop (CPU)."""
    out = []
    for i in range(0, len(imgs), bs):
        x, m = preprocess_batch(imgs[i:i + bs])
        out.append(model(x, m).numpy())
    return np.concatenate(out)


def compute_features(imgs, cnn: OrientNet, threads: int = 0, verbose: bool = True) -> dict:
    """All inputs of the stacker for a list of BGR crops."""
    feats = {"cnn": cnn_logits(cnn, imgs)}
    for key, name in REC_MODELS.items():
        feats[key] = RecScorer(name, threads=threads).features(imgs)
        if verbose:
            print(f"  {key}: done", flush=True)
    return feats


def predict_proba(feats: dict, stacker: AntisymLR) -> np.ndarray:
    # clip: a single confident mistake costs at most (1 - 1e-4)^2 in Brier
    return np.clip(stacker.predict(build_features(feats)), 1e-4, 1 - 1e-4)
