"""Write submission.csv for the test crops.

    python scripts/make_submission.py                 # final: CNN + recognizers
    python scripts/make_submission.py --cnn_only      # CNN alone (1st submission)

Needs only: test images, models/w24_v1 (shipped), models/stacker.json (shipped)
and the two PaddleOCR recognizers (`python scripts/download_data.py --minimal`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import SAMPLE_SUB, TEST_DIR, read_bgr  # noqa: E402
from src.pipeline import (CNN_DIR, STACKER_PATH, cnn_logits, compute_features,  # noqa: E402
                          load_cnn, predict_proba)
from src.stacking import AntisymLR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default=str(TEST_DIR))
    ap.add_argument("--sample", default=str(SAMPLE_SUB))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cnn_only", action="store_true")
    args = ap.parse_args()

    sub = pd.read_csv(args.sample)
    t0 = time.time()
    imgs = [read_bgr(Path(args.test_dir) / f"{i}.png") for i in sub["image_id"]]
    print(f"loaded {len(imgs)} crops in {time.time() - t0:.0f}s")

    cnn = load_cnn()
    t0 = time.time()
    if args.cnn_only:
        t = json.loads((CNN_DIR / "calibration.json").read_text())["temperature"]
        p = np.clip(1 / (1 + np.exp(-cnn_logits(cnn, imgs) / t)), 1e-4, 1 - 1e-4)
    else:
        p = predict_proba(compute_features(imgs, cnn), AntisymLR.load(STACKER_PATH))
    print(f"inference: {time.time() - t0:.0f}s")

    sub["p_180"] = p
    sub[["image_id", "p_180"]].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sub)} rows, share p_180>0.5 = {(p > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
