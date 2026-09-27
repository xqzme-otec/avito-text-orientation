"""Download all external (open-source) resources used by the solution.

    python scripts/download_data.py            # everything (for training / validation)
    python scripts/download_data.py --minimal  # only the 2 recognizers (to reproduce submission)

* PaddleOCR ONNX models (RapidOCR mirror, Apache-2.0)       -> models/pretrained/
* TextOCR word crops (MiXaiLL76/TextOCR_OCR on HF, CC-BY)   -> data/external/textocr/
* Russian Wikipedia shard (wikimedia/wikipedia, CC-BY-SA)   -> data/external/corpus/
* Google Fonts with Cyrillic glyphs (OFL), pinned commit    -> data/external/fonts/
"""
import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DATA_DIR, PRETRAINED_DIR, ROOT  # noqa: E402

RAPIDOCR = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/"
PRETRAINED = [
    "PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
    "PP-OCRv5/cls/ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx",
    "PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "PP-OCRv5/rec/eslav_PP-OCRv5_rec_mobile.onnx",
    "PP-OCRv5/rec/cyrillic_PP-OCRv5_rec_mobile.onnx",
    "PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx",
]
HF = "https://huggingface.co/datasets/"
TEXTOCR = [f"data/train-0000{i}-of-00005.parquet" for i in range(5)] + [
    "data/test-00000-of-00001.parquet"]
WIKI = "wikimedia/wikipedia/resolve/main/20231101.ru/train-00002-of-00021.parquet"


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    print("ok", dst.relative_to(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minimal", action="store_true")
    args = ap.parse_args()
    jobs = [(RAPIDOCR + p, PRETRAINED_DIR / Path(p).name) for p in PRETRAINED]
    if args.minimal:
        jobs = [j for j in jobs if "_rec_" in j[0] and "PP-OCRv5" in j[0]]
        for j in jobs:
            fetch(*j)
        return
    jobs += [(HF + "MiXaiLL76/TextOCR_OCR/resolve/main/" + p,
              DATA_DIR / "external" / "textocr" / Path(p).name) for p in TEXTOCR]
    jobs += [(HF + WIKI, DATA_DIR / "external" / "corpus" / "ruwiki-00002.parquet")]
    commit = (ROOT / "configs" / "google_fonts_commit.txt").read_text().strip()
    for p in (ROOT / "configs" / "fonts.txt").read_text().split():
        url = f"https://raw.githubusercontent.com/google/fonts/{commit}/{p}"
        jobs.append((url.replace("[", "%5B").replace("]", "%5D").replace(",", "%2C"),
                     DATA_DIR / "external" / "fonts" / Path(p).name))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))


if __name__ == "__main__":
    main()
