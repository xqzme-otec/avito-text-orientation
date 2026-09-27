"""Stacking of the CNN logit with PaddleOCR recognizer evidence.

Features (7 per crop), all *antisymmetric* - rotating the crop by 180 degrees
negates them:
  * CNN logit                                   (1)
  * for each recognizer (eslav, cyrillic):      (3 + 3)
      feature(crop) - feature(rot180(crop)) for
      - mean log-prob of the decoded characters,
      - mean log max-prob over all CTC frames,
      - share of frames that decode to a character.
A logistic regression *without intercept* on antisymmetric features is itself
antisymmetric: p(rot x) = 1 - p(x), exactly like the CNN.
"""
import json

import numpy as np

REC_KEYS = ("rec_eslav", "rec_cyr")


def build_features(d: dict) -> np.ndarray:
    """d: {'cnn': (N,), 'rec_eslav': (N, 6), 'rec_cyr': (N, 6)} -> (N, 7)."""
    cols = [d["cnn"][:, None]]
    for k in REC_KEYS:
        f = d[k]
        cols.append(f[:, :3] - f[:, 3:])  # as-is minus rotated
    return np.concatenate(cols, 1)


class AntisymLR:
    """Logistic regression without intercept (antisymmetric by construction)."""

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=C, fit_intercept=False, max_iter=2000)
        self.mu = None

    def fit(self, X, y):
        # scale without centering (centering would break antisymmetry)
        self.mu = np.sqrt((X ** 2).mean(0)) + 1e-9
        self.lr.fit(X / self.mu, y)
        return self

    def decision(self, X):
        return (X / self.mu) @ self.lr.coef_[0]

    def predict(self, X):
        return 1 / (1 + np.exp(-self.decision(X)))

    # the fitted stacker is 2 x 7 numbers; it is saved to JSON so that the
    # submission can be reproduced exactly without re-generating validation data
    def save(self, path):
        json.dump({"coef": self.lr.coef_[0].tolist(), "scale": self.mu.tolist()},
                  open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls()
        m.lr.coef_ = np.array([d["coef"]])
        m.mu = np.array(d["scale"])
        return m
