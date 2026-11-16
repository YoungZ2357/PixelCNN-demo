"""Check the browser engine (static/pixelcnn_demo.js) against PyTorch. Needs node on PATH.

    uv run python scripts/check_js.py                  # runs/gated
    uv run python scripts/check_js.py --run runs/smoke

1. logits: JS incremental evaluation vs a PyTorch forward pass, same float16-rounded weights.
2. sampling: the same uniform draws through JS (cached, one step per pixel) and PyTorch (one full
   forward pass per pixel) must give the same image.
3. receptive fields: receptiveField() in JS vs the nonzero input gradient of PyTorch models with
   random weights, for both the original PixelCNN and the gated one.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pixelcnn_demo.data import get_test_set  # noqa: E402
from pixelcnn_demo.export import build_bundle, load_run, round_weights  # noqa: E402
from pixelcnn_demo.model import GatedPixelCNN, OriginalPixelCNN  # noqa: E402


def write_bundle(model, out, dtype):
    manifest, buffers = build_bundle(model, dtype=dtype, per_class=0)
    for path, data in buffers.items():
        (out / path).write_bytes(data)
    (out / "manifest.json").write_text(json.dumps(manifest))


@torch.no_grad()
def naive_sample(model, y, u):
    x = torch.zeros(1, 1, 28, 28, dtype=torch.float64)
    for pos in range(784):
        r, c = divmod(pos, 28)
        p = torch.sigmoid(model(x, torch.tensor([y]))[0, 0, r, c])
        x[0, 0, r, c] = float(u[pos] < p)
    return x.flatten()


def receptive_fixtures(n=16):
    torch.manual_seed(0)
    cases = []
    for kind in ("original", "gated"):
        for layers in (1, 2, 4, 7):
            m = (OriginalPixelCNN(channels=4, layers=layers) if kind == "original"
                 else GatedPixelCNN(channels=4, layers=layers, k_first=3, k=3, head=4)).double()
            for r, c in ((0, 0), (5, 9), (10, 8), (15, 15), (8, 0), (3, 15)):
                x = torch.rand(1, 1, n, n, dtype=torch.float64, requires_grad=True)
                m.features(x, torch.tensor([0]))[0, :, r, c].sum().backward()
                g = (x.grad[0, 0] != 0).flatten().int().tolist()
                cases.append({"kind": kind, "layers": layers, "r": r, "c": c, "n": n, "set": g})
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "runs" / "gated"))
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--data", default=str(ROOT / "data"))
    args = p.parse_args()
    if not shutil.which("node"):
        raise SystemExit("node is not on PATH")

    model, _, _ = load_run(args.run)
    model = round_weights(model, args.dtype).double().eval()
    x, labels = get_test_set(args.data)
    images = [(x[0].flatten(), int(labels[0])), (x[1].flatten(), model.null_class),
              (torch.bernoulli(torch.full((784,), 0.3), generator=torch.Generator().manual_seed(1)), 4)]
    fixtures = {"logits": [], "samples": [], "receptive": receptive_fixtures()}
    with torch.no_grad():
        for img, y in images:
            lg = model(img.reshape(1, 1, 28, 28).double(), torch.tensor([y])).flatten()
            fixtures["logits"].append({"x": img.int().tolist(), "y": y, "logits": lg.tolist()})
    g = torch.Generator().manual_seed(2)
    for y in (3, model.null_class):
        u = torch.rand(784, generator=g, dtype=torch.float64)
        fixtures["samples"].append({"y": y, "u": u.tolist(), "x": naive_sample(model, y, u).int().tolist()})

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_bundle(model, tmp, args.dtype)
        (tmp / "fixtures.json").write_text(json.dumps(fixtures))
        shutil.copy(ROOT / "pixelcnn_demo" / "static" / "pixelcnn_demo.js", tmp / "engine.mjs")
        r = subprocess.run(["node", str(ROOT / "scripts" / "check_js.mjs"), str(tmp)])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
