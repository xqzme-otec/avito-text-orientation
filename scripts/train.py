"""Train the compact orientation classifier.

    python scripts/train.py --width 16 --steps 12000 --name orient_w16

Every sample is generated deterministically from its index (fixed seed), so the
run is reproducible given the same hardware/library versions.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import MODELS_DIR, fit_temperature, preprocess_batch, report, seed_everything  # noqa: E402
from src.dataset import TrainDataset, build_val_sets, textocr_store  # noqa: E402
from src.model import OrientNet, n_params  # noqa: E402


@torch.no_grad()
def predict_logits(model, xs, ms, device, bs=1024):
    model.eval()
    out = [model(xs[i:i + bs].to(device), ms[i:i + bs].to(device)).float().cpu()
           for i in range(0, len(xs), bs)]
    model.train()
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="orient")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--p_synth", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    val = {k: (*preprocess_batch(imgs), y) for k, (imgs, y) in build_val_sets().items()}
    real, latin = textocr_store("train")
    print(f"real train crops: {len(real)}  device: {device}", flush=True)

    ds = TrainDataset(real, args.steps * args.bs, args.p_synth, args.seed,
                      extra_latin=sorted(set(latin)))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    drop_last=True, prefetch_factor=4)

    model = OrientNet(args.width).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["state_dict"])
    print(f"params: {n_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    out_dir = MODELS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    def ckpt():
        return {"width": args.width, "state_dict": model.state_dict(), "args": vars(args)}

    best, hist, tot, t0 = -1.0, [], 0.0, time.time()
    for step, (x, m, y) in enumerate(dl, 1):
        x, m, y = x.to(device), m.to(device), y.to(device)
        loss = F.binary_cross_entropy_with_logits(model(x, m), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            res = {"step": step, "loss": tot / args.eval_every,
                   "samples_per_s": args.eval_every * args.bs / (time.time() - t0)}
            for k, (xs, ms, yv) in val.items():
                p = 1 / (1 + np.exp(-predict_logits(model, xs, ms, device)))
                res[k] = report(f"step{step} {k}", p, yv)["1-brier"]
            score = float(np.mean([res[k] for k in val]))
            print(json.dumps(res), flush=True)
            hist.append(res)
            if score > best:
                best = score
                torch.save(ckpt(), out_dir / "best.pt")
            torch.save(ckpt(), out_dir / "last.pt")
            (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
            tot, t0 = 0.0, time.time()
    print("best mean 1-Brier:", best)

    # --- calibration: one temperature fitted on both validation sets together
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu")["state_dict"])
    lg = np.concatenate([predict_logits(model, xs, ms, device) for xs, ms, _ in val.values()])
    yy = np.concatenate([yv for _, _, yv in val.values()])
    t = fit_temperature(lg, yy)
    calib = {"temperature": t}
    for k, (xs, ms, yv) in val.items():
        p = 1 / (1 + np.exp(-predict_logits(model, xs, ms, device) / t))
        calib[k] = report(f"calibrated T={t:.3f} {k}", p, yv)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
