"""Open-source PaddleOCR models (ONNX exports from RapidOCR) used as orientation scorers.

* TextlineOriCls  - PP-LCNet text line orientation classifiers (0 / 180).
* RecScorer       - CTC text recognizer; the orientation signal is how "readable"
                    the crop is as-is vs. rotated by 180 degrees (a language-aware
                    signal, since upside-down Cyrillic/Latin decodes to garbage).
"""
import math

import cv2
import numpy as np
import onnxruntime as ort

from .common import PRETRAINED_DIR, rot180


def _session(name: str, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads  # 0 = all cores
    return ort.InferenceSession(str(PRETRAINED_DIR / name), so,
                                providers=["CPUExecutionProvider"])


def _resize_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """PaddleOCR-style keep-ratio resize to height h, right zero-pad to width w."""
    ih, iw = img.shape[:2]
    rw = min(w, int(math.ceil(h * iw / float(ih))))
    x = cv2.resize(img, (rw, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = (x.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    out = np.zeros((3, h, w), dtype=np.float32)
    out[:, :, :rw] = x
    return out


class TextlineOriCls:
    SHAPES = {
        "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx": (80, 160),
        "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx": (80, 160),
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (48, 192),
    }

    def __init__(self, name: str):
        self.sess = _session(name)
        self.h, self.w = self.SHAPES[name]
        self.inp = self.sess.get_inputs()[0].name

    def logit(self, imgs, batch: int = 64) -> np.ndarray:
        """Returns log(p180/p0) for each BGR image."""
        out = []
        for i in range(0, len(imgs), batch):
            x = np.stack([_resize_pad(im, self.h, self.w) for im in imgs[i:i + batch]])
            p = self.sess.run(None, {self.inp: x})[0].astype(np.float64)
            p = np.clip(p, 1e-7, 1)
            out.append(np.log(p[:, 1]) - np.log(p[:, 0]))
        return np.concatenate(out)


class RecScorer:
    """CTC recognizer. For each crop returns features computed on the crop as-is
    and on its 180-rotated copy; their difference is the orientation evidence."""

    def __init__(self, name: str, h: int = 48, max_w: int = 960, threads: int = 0):
        self.sess = _session(name, threads)
        self.inp = self.sess.get_inputs()[0].name
        self.h, self.max_w = h, max_w

    def _prep(self, img):
        ih, iw = img.shape[:2]
        w = int(min(self.max_w, max(16, math.ceil(self.h * iw / ih))))
        return _resize_pad(img, self.h, w)

    def _feats(self, probs: np.ndarray, valid_t: int) -> np.ndarray:
        """Features of one CTC posterior (T x C): mean log max-prob over
        non-blank frames, fraction of non-blank frames, sum of log max-prob."""
        p = probs[:valid_t]
        idx = p.argmax(1)
        mx = np.log(np.clip(p.max(1), 1e-9, 1))
        nb = idx != 0
        # collapse repeats -> decoded characters
        keep = nb & np.concatenate([[True], idx[1:] != idx[:-1]])
        n_chars = keep.sum()
        mean_nb = mx[keep].mean() if n_chars else 0.0
        return np.array([mean_nb, mx.mean(), n_chars / max(1, valid_t)], dtype=np.float64)

    def features(self, imgs, batch: int = 32) -> np.ndarray:
        """Returns (N, 6): features for the image as-is and rotated by 180."""
        res = np.zeros((len(imgs), 6))
        # sort by width so that batches are padded as little as possible
        order = np.argsort([im.shape[1] / im.shape[0] for im in imgs])
        for s in range(0, len(order), batch):
            ids = order[s:s + batch]
            for k, flip in enumerate((False, True)):
                xs = [self._prep(rot180(imgs[i]) if flip else imgs[i]) for i in ids]
                W = max(x.shape[2] for x in xs)
                xb = np.zeros((len(xs), 3, self.h, W), dtype=np.float32)
                for j, x in enumerate(xs):
                    xb[j, :, :, :x.shape[2]] = x
                probs = self.sess.run(None, {self.inp: xb})[0]
                T = probs.shape[1]
                for j, (i, x) in enumerate(zip(ids, xs)):
                    valid_t = max(1, int(round(T * x.shape[2] / W)))
                    res[i, 3 * k:3 * k + 3] = self._feats(probs[j], valid_t)
        return res
