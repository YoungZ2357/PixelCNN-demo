"""Export a trained model as a framework-free bundle for static/pixelcnn_demo.js, used both by
the notebook widget (in memory) and by the static site (files).

    manifest.json
    weights.bin            every tensor, concatenated, float16 or float32 little-endian
    examples/images.bin    uint8 [n, 784] binarized test images (0/1), grouped by digit
    examples/labels.bin    uint8 [n]

manifest:
    {"version": 1, "image": [28, 28], "config": {...}, "null_class": 10, "dtype": "float16",
     "weights": "weights.bin",
     "layers": [{"first": bool,
                 "v": conv, "h": conv,                  # CausalConv, see model.py
                 "v2h": conv1x1, "h_out": conv1x1,
                 "v_emb": emb, "h_emb": emb}],
     "head": [conv1x1, conv1x1],                       # ReLU before each
     "metrics": {"epoch", "test", "test_uncond"} | null,   # nats / image
     "history": [...] | null,
     "examples": {"images", "labels", "n", "per_class"} | null}
    conv     = {"cin", "cout", "kh", "kw", "dy", "dx", "w", "b"}   # w: [cout, cin, kh, kw] row-major
    conv1x1  = {"cin", "cout", "w", "b"}                            # w: [cout, cin]
    emb      = {"n", "dim", "w"}                                    # w: [n, dim]
    w, b: element offsets into weights.bin.
"""
import json
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


class _Blob:
    def __init__(self, dtype):
        self.dtype, self.parts, self.off = np.dtype(dtype), [], 0

    def add(self, t):
        a = t.detach().cpu().float().numpy().astype(self.dtype).ravel()
        off = self.off
        self.parts.append(a)
        self.off += a.size
        return off

    def bytes(self):
        return np.concatenate(self.parts).astype(self.dtype.newbyteorder("<")).tobytes()


def _conv(blob, m):
    c = m.conv
    return {"cin": c.in_channels, "cout": c.out_channels, "kh": m.kh, "kw": m.kw,
            "dy": m.dy, "dx": m.dx, "w": blob.add(c.weight), "b": blob.add(c.bias)}


def _conv1x1(blob, c):
    return {"cin": c.in_channels, "cout": c.out_channels, "w": blob.add(c.weight), "b": blob.add(c.bias)}


def _emb(blob, e):
    return {"n": e.num_embeddings, "dim": e.embedding_dim, "w": blob.add(e.weight)}


def example_indices(labels, per_class):
    labels = np.asarray(labels)
    return np.concatenate([np.flatnonzero(labels == d)[:per_class] for d in range(10)])


def build_bundle(model, metrics=None, history=None, dtype="float16", per_class=3, data_dir="./data"):
    """Returns (manifest dict, {path: bytes}). per_class: test images per digit (0 to skip)."""
    blob = _Blob(dtype)
    layers = [{"first": L.first, "v": _conv(blob, L.v), "h": _conv(blob, L.h),
               "v2h": _conv1x1(blob, L.v2h), "h_out": _conv1x1(blob, L.h_out),
               "v_emb": _emb(blob, L.v_emb), "h_emb": _emb(blob, L.h_emb)} for L in model.layers]
    head = [_conv1x1(blob, model.head[1]), _conv1x1(blob, model.head[3])]
    buffers = {"weights.bin": blob.bytes()}
    examples = None
    if per_class:
        from .data import get_test_set
        x, y = get_test_set(data_dir)
        idx = example_indices(y.numpy(), per_class)
        buffers["examples/images.bin"] = x[idx].reshape(len(idx), -1).to(torch.uint8).numpy().tobytes()
        buffers["examples/labels.bin"] = y[idx].numpy().astype(np.uint8).tobytes()
        examples = {"images": "examples/images.bin", "labels": "examples/labels.bin",
                    "n": len(idx), "per_class": per_class}
    manifest = {"version": FORMAT_VERSION, "image": [28, 28], "config": model.config,
                "null_class": model.null_class, "dtype": dtype, "weights": "weights.bin",
                "layers": layers, "head": head, "metrics": metrics, "history": history,
                "examples": examples}
    return manifest, buffers


def load_run(run_dir, device="cpu"):
    """(model, metrics, history) from a training run directory."""
    from .train import load_checkpoint
    run_dir = Path(run_dir)
    model, ckpt = load_checkpoint(run_dir / "model.pt", device)
    model.eval()
    hist_path = run_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else None
    metrics = None
    if history:
        last = history[-1]
        metrics = {"epoch": last["epoch"], "test": last["test"], "test_uncond": last["test_uncond"]}
    return model, metrics, history


def export_bundle(run_dir, out_dir, **kw):
    """Write the bundle of a run directory to out_dir. kw: see build_bundle."""
    model, metrics, history = load_run(run_dir)
    manifest, buffers = build_bundle(model, metrics, history, **kw)
    out_dir = Path(out_dir)
    for path, data in buffers.items():
        p = out_dir / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir / "manifest.json"


def round_weights(model, dtype):
    """Copy of `model` whose parameters went through `dtype`, as the browser sees them."""
    import copy
    m = copy.deepcopy(model).float()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(getattr(torch, dtype)).float())
    return m
