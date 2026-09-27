"""Fit the stacker (CNN + PaddleOCR recognizers) on the labelled validation sets.

    python scripts/fit_stacker.py

1. computes features for both validation sets (cached in artifacts/features/);
2. reports 1-Brier of the CNN alone and of the stack (5-fold CV and
   cross-domain: fit on one validation set, evaluate on the other);
3. fits the final stacker on both sets and saves it to models/stacker.json.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ARTIFACTS_DIR, report  # noqa: E402
from src.dataset import build_val_sets  # noqa: E402
from src.pipeline import STACKER_PATH, compute_features, load_cnn  # noqa: E402
from src.stacking import AntisymLR, build_features  # noqa: E402


def val_features() -> dict:
    out_dir = ARTIFACTS_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    cnn, res = load_cnn(), {}
    for name, (imgs, y) in build_val_sets().items():
        path = out_dir / f"{name}.npz"
        if not path.exists():
            print(f"features for {name} ({len(imgs)} crops)")
            np.savez(path, y=np.asarray(y), **compute_features(imgs, cnn))
        res[name] = dict(np.load(path))
    return res


def main():
    D = val_features()
    X = {k: build_features(d) for k, d in D.items()}
    y = {k: d["y"] for k, d in D.items()}

    print("\n--- CNN alone (sigmoid of raw logit)")
    for k in D:
        report(k, 1 / (1 + np.exp(-D[k]["cnn"])), y[k])

    print("\n--- stack, cross-domain (fit on one set, evaluate on the other)")
    for a, b in [("real_en", "syn_ru"), ("syn_ru", "real_en")]:
        report(f"{a} -> {b}", AntisymLR().fit(X[a], y[a]).predict(X[b]), y[b])

    print("\n--- stack, 5-fold CV on the union")
    Xu = np.concatenate(list(X.values()))
    yu = np.concatenate(list(y.values()))
    grp = np.concatenate([[k] * len(y[k]) for k in y])
    p = np.zeros(len(yu))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xu, yu):
        p[te] = AntisymLR().fit(Xu[tr], yu[tr]).predict(Xu[te])
    for k in y:
        report(k, p[grp == k], yu[grp == k])

    final = AntisymLR().fit(Xu, yu)
    final.save(STACKER_PATH)
    print("\nsaved", STACKER_PATH)


if __name__ == "__main__":
    main()
