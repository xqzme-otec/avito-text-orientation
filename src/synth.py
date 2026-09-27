"""Synthetic text-line crops (Russian + Latin) rendered with open fonts.

The test set consists of crops produced by a text detector on user photos,
screenshots and banners from Avito. We imitate that distribution:
  * text: Russian Wikipedia phrases, English words, Avito-like strings
    (prices, phone numbers, part codes...), with random casing;
  * look: ~150 OFL Google Fonts (incl. variable weights), strokes, shadows,
    solid / gradient / noisy backgrounds;
  * box geometry: tight detector-like box with random margins, occasional
    clipping and pieces of neighbouring lines;
  * degradations: perspective, blur, low resolution, noise, JPEG.
All rendered crops are upright; the 180-degree label is applied by the caller.
"""
import random
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .common import DATA_DIR

FONTS_DIR = DATA_DIR / "external" / "fonts"
WIKI_PATH = DATA_DIR / "external" / "corpus" / "ruwiki-00002.parquet"

_RU_WORD = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё\-]*")
_LAT_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# ----------------------------------------------------------------------------- text
class TextSampler:
    def __init__(self, part: str = "train", n_docs: int = 30000,
                 extra_latin: list[str] | None = None, seed: int = 0):
        """part='train' uses the first 80% of Wikipedia articles, 'val' the rest,
        so validation phrases never appear in training."""
        rng = random.Random(seed)
        col = pq.read_table(WIKI_PATH, columns=["text"]).column("text")
        cut = int(len(col) * 0.8)
        pool = range(0, cut) if part == "train" else range(cut, len(col))
        idx = rng.sample(pool, min(n_docs, len(pool)))
        sents, lat = [], set()
        for i in idx:
            for line in col[i].as_py().split("\n"):
                line = line.strip()
                if len(line) < 20:
                    continue
                for s in re.split(r"(?<=[.!?])\s+", line):
                    if 8 <= len(s) <= 300:
                        sents.append(s)
                lat.update(w for w in _LAT_WORD.findall(line) if len(w) >= 2)
        self.sents = sents
        self.latin = sorted(lat | set(extra_latin or []))

    # -- generators of Avito-like strings
    @staticmethod
    def _price(r):
        v = r.choice([r.randint(1, 999), r.randint(1000, 99999), r.randint(100000, 9999999)])
        s = f"{v:,}".replace(",", r.choice([" ", ".", "", " "]))
        return r.choice(["{} ₽", "{} руб.", "{}₽", "от {} ₽", "{} р.", "{}", "Цена: {} ₽", "{} $"]).format(s)

    @staticmethod
    def _phone(r):
        d = [str(r.randint(0, 9)) for _ in range(10)]
        return r.choice(["+7 ({}{}{}) {}{}{}-{}{}-{}{}", "8 {}{}{} {}{}{} {}{} {}{}", "+7{}{}{}{}{}{}{}{}{}{}",
                         "8-{}{}{}-{}{}{}-{}{}-{}{}"]).format(*d)

    @staticmethod
    def _code(r):
        al = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
        parts = ["".join(r.choice(al) for _ in range(r.randint(2, 8))) for _ in range(r.randint(1, 4))]
        return r.choice(["-", " ", "", "/"]).join(parts)

    @staticmethod
    def _misc(r):
        return r.choice([
            f"{r.randint(1, 31):02d}.{r.randint(1, 12):02d}.{r.randint(1990, 2026)}",
            f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}",
            f"{r.randint(1, 999)} {r.choice(['км', 'м²', 'кг', 'шт', 'мл', 'г', 'см', 'ГБ', 'л.с.'])}",
            f"www.{r.choice(['avito', 'shop', 'market', 'auto', 'dom'])}{r.randint(1, 99)}.ru",
            f"@{r.choice(['shop', 'user', 'store'])}_{r.randint(10, 9999)}",
            f"{r.randint(1, 99)}%",
            f"№ {r.randint(1, 99999)}",
            str(r.randint(0, 10 ** r.randint(1, 9))),
        ])

    def _ru_phrase(self, r, max_words):
        words = self.sents[r.randrange(len(self.sents))].split()
        n = min(len(words), r.choice([1, 1, 2, 2, 3, 3, 4, 5, 6, 8, max_words]))
        s = r.randrange(0, len(words) - n + 1)
        return " ".join(words[s:s + n]).strip(" ,;:«»()\"")

    def sample(self, r: random.Random) -> str:
        t = r.random()
        if t < 0.55:
            s = self._ru_phrase(r, 10)
        elif t < 0.75:
            s = " ".join(r.choice(self.latin) for _ in range(r.choice([1, 1, 1, 2, 2, 3])))
        elif t < 0.83:
            s = self._price(r)
        elif t < 0.87:
            s = self._phone(r)
        elif t < 0.91:
            s = self._code(r)
        elif t < 0.96:
            s = self._misc(r)
        else:  # mixed ru + latin / number
            s = f"{self._ru_phrase(r, 3)} {r.choice([r.choice(self.latin), self._misc(r), self._price(r)])}"
        s = s.strip() or "Авито"
        c = r.random()
        if c < 0.35:
            s = s.upper()
        elif c < 0.5:
            s = s.title()
        elif c < 0.55:
            s = s.lower()
        return s


# ----------------------------------------------------------------------------- fonts
def list_fonts(part: str = "all", fonts_dir: Path = FONTS_DIR) -> list[Path]:
    """part='train' / 'val': deterministic split of font families (every 7th
    family is held out for validation)."""
    fonts = sorted(p for p in fonts_dir.glob("*.ttf"))
    if part == "all":
        return fonts
    fams = sorted({p.name.split("-")[0].split("[")[0] for p in fonts})
    val = set(fams[3::7])
    return [p for p in fonts if (p.name.split("-")[0].split("[")[0] in val) == (part == "val")]


@lru_cache(maxsize=4096)
def _font(path: str, size: int, var_idx: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if var_idx >= 0:
        try:
            axes = f.get_variation_axes()
            vals = []
            for ax in axes:
                lo, hi = ax["minimum"], ax["maximum"]
                name = ax.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
                if "eight" in name:  # weight: pick one of 4 weights
                    vals.append(lo + (hi - lo) * var_idx / 3)
                else:
                    vals.append(ax.get("default", lo))
            f.set_variation_by_axes(vals)
        except Exception:
            pass
    return f


@lru_cache(maxsize=None)
def _cmap(path: str) -> frozenset:
    return frozenset((TTFont(path, lazy=True).getBestCmap() or {}).keys())


def _supported(path: str, text: str) -> bool:
    cm = _cmap(path)
    return all(ord(c) in cm for c in text if not c.isspace())


# ----------------------------------------------------------------------------- rendering
def _rand_color(r):
    return tuple(r.randint(0, 255) for _ in range(3))


def _contrast_pair(r):
    """Foreground/background colors. Most real crops are dark-on-light or
    light-on-dark with low saturation; the rest are arbitrary colour pairs."""
    t = r.random()
    if t < 0.45:
        g = r.randint(180, 255)
        bg = tuple(int(np.clip(g + r.randint(-15, 15), 0, 255)) for _ in range(3))
        v = r.randint(0, 90)
        fg = tuple(int(np.clip(v + r.randint(-20, 20), 0, 255)) for _ in range(3))
        return fg, bg
    if t < 0.6:
        g = r.randint(0, 80)
        bg = tuple(int(np.clip(g + r.randint(-15, 15), 0, 255)) for _ in range(3))
        v = r.randint(170, 255)
        fg = tuple(int(np.clip(v + r.randint(-20, 20), 0, 255)) for _ in range(3))
        return fg, bg
    while True:
        a, b = _rand_color(r), _rand_color(r)
        la = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
        lb = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2]
        if abs(la - lb) > r.choice([50, 80, 110]):
            return a, b


def _background(r, h, w, color):
    t = r.random()
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = color
    if t < 0.6:
        pass
    elif t < 0.8:  # linear gradient
        c2 = np.array(color) + np.array([r.randint(-60, 60) for _ in range(3)])
        g = np.linspace(0, 1, w if r.random() < 0.5 else h, dtype=np.float32)
        g = g[None, :, None] if len(g) == w else g[:, None, None]
        bg = bg * (1 - g) + c2[None, None, :] * g
    else:  # smooth noise texture
        sh, sw = max(2, h // r.randint(2, 12)), max(2, w // r.randint(2, 12))
        n = np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(4, 25), (sh, sw, 3))
        bg = bg + cv2.resize(n.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 0, 255)


def _text_mask(r, text, font, stroke):
    """Render text into a float mask (and optional stroke mask)."""
    l, t, rt, b = font.getbbox(text, stroke_width=stroke)
    W, H = rt - l + 8, b - t + 8
    fill = Image.new("L", (W, H), 0)
    ImageDraw.Draw(fill).text((4 - l, 4 - t), text, font=font, fill=255)
    out = Image.new("L", (W, H), 0)
    if stroke:
        ImageDraw.Draw(out).text((4 - l, 4 - t), text, font=font, fill=255, stroke_width=stroke,
                                 stroke_fill=255)
    return np.asarray(fill, np.float32) / 255, np.asarray(out, np.float32) / 255


class Renderer:
    def __init__(self, fonts: list[Path], texts: TextSampler):
        self.fonts = [str(f) for f in fonts]
        self.texts = texts

    def _pick_font(self, r, text, size):
        for _ in range(6):
            path = r.choice(self.fonts)
            if _supported(path, text):
                return _font(path, size, r.randint(0, 3) if "[" in path else -1)
        return _font(self.fonts[0], size, -1)

    def render(self, r: random.Random) -> np.ndarray:
        """One upright BGR uint8 crop."""
        text = self.texts.sample(r)
        size = r.choice([20, 24, 28, 32, 40, 48])
        font = self._pick_font(r, text, size)
        stroke = r.choice([0] * 8 + [1, 2, 3])
        fill, outline = _text_mask(r, text, font, stroke)
        ys, xs = np.nonzero(np.maximum(fill, outline) > 0.2)
        if len(ys) == 0:
            return self.render(r)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        th = y1 - y0
        # line height of the font (box of the detector covers ascenders/descenders
        # of the line, not just the ink of this particular string)
        asc, desc = font.getmetrics()
        lh = max(th, int((asc + desc) * r.uniform(0.75, 1.0)))
        # detector-like margins (occasionally negative = clipped text)
        mt = int(lh * r.uniform(-0.05, 0.3)) + (lh - th) // 2
        mb = int(lh * r.uniform(-0.05, 0.3)) + (lh - th) - (lh - th) // 2
        ml = int(lh * r.uniform(-0.1, 0.5))
        mr = int(lh * r.uniform(-0.1, 0.5))
        H = th + mt + mb
        W = (x1 - x0) + ml + mr
        if H < 6 or W < 8:
            return self.render(r)

        fg, bgc = _contrast_pair(r)
        canvas = _background(r, H, W, bgc)

        def paste(mask, color, dy, dx):
            # paste mask (in its own coords) with offset so that mask[y0,x0] -> (mt+dy, ml+dx)
            oy, ox = mt + dy - y0, ml + dx - x0
            sy0, sx0 = max(0, -oy), max(0, -ox)
            ty0, tx0 = max(0, oy), max(0, ox)
            hh = min(mask.shape[0] - sy0, H - ty0)
            ww = min(mask.shape[1] - sx0, W - tx0)
            if hh <= 0 or ww <= 0:
                return
            m = mask[sy0:sy0 + hh, sx0:sx0 + ww, None]
            reg = canvas[ty0:ty0 + hh, tx0:tx0 + ww]
            reg[:] = reg * (1 - m) + np.array(color, np.float32) * m

        if r.random() < 0.12:  # drop shadow
            d = max(1, lh // r.randint(8, 20))
            paste(fill, tuple(c * 0.3 for c in fg) if r.random() < 0.5 else (0, 0, 0), d, d)
        if stroke:
            paste(outline, _rand_color(r), 0, 0)
        paste(fill, fg, 0, 0)
        if r.random() < 0.05:  # underline
            yy = min(H - 1, mt + th + max(1, lh // 12))
            canvas[yy:yy + max(1, lh // 16), ml:W - mr] = fg
        if r.random() < 0.1:  # piece of a neighbouring line above or below
            other = self.texts.sample(r)
            of = self._pick_font(r, other, size)
            m2, _ = _text_mask(r, other, of, 0)
            gap = int(lh * r.uniform(0.1, 0.5))
            if r.random() < 0.5:
                paste(m2, fg, -(m2.shape[0] - 8) - gap, r.randint(-m2.shape[1] // 2, W // 2))
            else:
                paste(m2, fg, th + gap, r.randint(-m2.shape[1] // 2, W // 2))
        img = canvas.astype(np.uint8)
        if r.random() < 0.1:
            img = 255 - img
        return degrade(r, img)


# ----------------------------------------------------------------------------- degradations
def degrade(r: random.Random, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if r.random() < 0.5:  # small rotation / shear / perspective
        a = np.deg2rad(r.uniform(-4, 4))
        sh = r.uniform(-0.25, 0.25) if r.random() < 0.3 else 0.0
        M = np.array([[np.cos(a), -np.sin(a) + sh, 0], [np.sin(a), np.cos(a), 0]], np.float32)
        c = np.array([w / 2, h / 2])
        M[:, 2] = c - M[:, :2] @ c
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if r.random() < 0.25:
        d = 0.08
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[r.uniform(-d, d) * w, r.uniform(-d, d) * h] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    # resolution: the crops' height in the test set ranges 10..400 px
    th = int(np.clip(np.exp(r.uniform(np.log(9), np.log(90))), 8, 120))
    tw = max(8, int(w * th / h * r.uniform(0.85, 1.15)))
    img = cv2.resize(img, (tw, th), interpolation=r.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]))
    if r.random() < 0.25:
        k = r.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), r.uniform(0.3, 1.2))
    if r.random() < 0.3:
        img = np.clip(img + np.random.default_rng(r.randrange(1 << 30)).normal(
            0, r.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    if r.random() < 0.4:
        q = r.randint(20, 90)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    if r.random() < 0.3:  # color / contrast jitter
        img = np.clip(img.astype(np.float32) * r.uniform(0.6, 1.3) + r.uniform(-40, 40), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)
