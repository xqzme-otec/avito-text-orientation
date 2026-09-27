"""Compact orientation classifier.

Input: a text crop resized to height H (keep aspect), squeezed if wider than W,
then *centre*-padded to (H, W). Centre padding makes rot180 of the padded tensor
equal to the padded rot180 of the crop, so the 180-degree counterpart is just
`x.flip(2, 3)`.

The head is antisymmetric by construction:
    logit(x) = f(x) - f(rot180(x))
which guarantees p(rot180(x)) = 1 - p(x) — the two orientations are always
assessed consistently, and the network only has to learn a "how upright does
this look" score f.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 32, 256


def preprocess(img: np.ndarray, h: int = H, w: int = W) -> tuple[np.ndarray, np.ndarray]:
    """BGR uint8 crop -> (3, h, w) float32 in [-1, 1] and (w,) validity mask."""
    ih, iw = img.shape[:2]
    nw = int(round(iw * h / ih))
    nw = _even_width(max(4, min(w, nw)))
    x = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA if ih > h else cv2.INTER_LINEAR)
    x = x[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    out = np.zeros((3, h, w), np.float32)
    mask = np.zeros((w,), np.float32)
    # nw and w are both even -> padding is exactly symmetric, and the flipped
    # tensor is the padded rot180 crop (no 1px shift)
    l = (w - nw) // 2
    out[:, :, l:l + nw] = x
    mask[l:l + nw] = 1
    return out, mask


def _even_width(nw: int) -> int:
    return nw - (nw % 2)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class OrientNet(nn.Module):
    """Plain VGG-style CNN of dense 3x3 convs (fast on CPU/ONNX and on MPS,
    unlike depthwise convs). Width c=16 -> ~0.2M parameters."""

    def __init__(self, width: int = 16):
        super().__init__()
        c = width
        self.features = nn.Sequential(
            ConvBNAct(3, c, 3, 2),                    # 16 x 128
            ConvBNAct(c, 2 * c, 3, 2),                # 8 x 64
            ConvBNAct(2 * c, 2 * c),
            ConvBNAct(2 * c, 4 * c, 3, 2),            # 4 x 32
            ConvBNAct(4 * c, 4 * c),
            ConvBNAct(4 * c, 6 * c, 3, (2, 1)),       # 2 x 32
            ConvBNAct(6 * c, 6 * c),
        )
        # per-column score, averaged over valid columns -> f(x)
        self.head = nn.Conv2d(6 * c, 1, (2, 1))

    def score(self, x, mask):
        s = self.head(self.features(x)).squeeze(1).squeeze(1)       # (B, W')
        m = F.adaptive_max_pool1d(mask[:, None], s.shape[1]).squeeze(1)  # (B, W')
        return (s * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, x, mask):
        """Returns the logit of P(image is rotated by 180)."""
        return self.score(x.flip(2, 3), mask.flip(1)) - self.score(x, mask)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
