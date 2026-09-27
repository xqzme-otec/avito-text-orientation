"""Export the trained model to ONNX (temperature baked in: output = calibrated logit)
and benchmark CPU speed with onnxruntime.

    python scripts/export_onnx.py --model models/orient_w16
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import H, W, OrientNet, n_params  # noqa: E402


class Calibrated(torch.nn.Module):
    def __init__(self, model, t):
        super().__init__()
        self.model, self.t = model, t

    def forward(self, x, mask):
        return self.model(x, mask) / self.t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    mdir = Path(args.model)
    ck = torch.load(mdir / "best.pt", map_location="cpu")
    model = OrientNet(ck["width"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    calib = mdir / "calibration.json"
    t = json.loads(calib.read_text())["temperature"] if calib.exists() else 1.0
    net = Calibrated(model, t).eval()

    out = mdir / "orient.onnx"
    x, m = torch.randn(8, 3, H, W), torch.ones(8, W)
    torch.onnx.export(net, (x, m), str(out), input_names=["x", "mask"], output_names=["logit"],
                      dynamic_axes={"x": {0: "n"}, "mask": {0: "n"}, "logit": {0: "n"}},
                      opset_version=17, dynamo=False)
    print(f"params={n_params(model)}  onnx size={out.stat().st_size / 1e6:.2f} MB")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    ref = net(x, m).detach().numpy()
    got = sess.run(None, {"x": x.numpy(), "mask": m.numpy()})[0]
    print("max |torch - onnx| =", float(np.abs(ref - got).max()))
    xb, mb = np.random.randn(64, 3, H, W).astype(np.float32), np.ones((64, W), np.float32)
    sess.run(None, {"x": xb, "mask": mb})
    t0 = time.time()
    for _ in range(10):
        sess.run(None, {"x": xb, "mask": mb})
    print(f"CPU 1 thread: {(time.time() - t0) / 640 * 1000:.3f} ms / crop (batch 64)")


if __name__ == "__main__":
    main()
