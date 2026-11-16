"""Train the class-conditional Gated PixelCNN on binarized MNIST.

    uv run python -m pixelcnn_demo.train                       # defaults -> runs/gated
    uv run python -m pixelcnn_demo.train --epochs 40 --channels 64 --layers 12
    uv run python -m pixelcnn_demo.train --resume              # continue runs/gated
    uv run python -m pixelcnn_demo.train --max-steps 200 --channels 32 --layers 6 --out runs/smoke

A run directory holds:
    model.pt      {state_dict, config, epoch, optimizer, scheduler, args}
    history.json  per epoch: train NLL, test NLL with the true class and with no class (nats/image)
    samples.png   one row of samples per class (with --figures, and at the end of training)

The checkpoint is written after every epoch, so an interrupted run loses at most one epoch.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import get_device, get_loaders
from .model import build, nll

BITS = 1 / (784 * math.log(2))  # nats per image -> bits per dimension


@torch.no_grad()
def evaluate(model, loader, device, amp):
    """Mean test NLL (nats/image), conditioned on the true class and on the null class."""
    model.eval()
    tot = {"cond": 0.0, "uncond": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            for key, yy in (("cond", y), ("uncond", torch.full_like(y, model.null_class))):
                tot[key] += nll(model(x, yy).float(), x).sum().item()
        n += len(x)
    model.train()
    return {k: v / n for k, v in tot.items()}


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def save_samples(model, path, temperature=1.0, per_class=10, seed=0):
    import matplotlib.pyplot as plt
    dev = next(model.parameters()).device
    y = torch.arange(model.n_classes + 1).repeat_interleave(per_class)
    g = torch.Generator(device=dev).manual_seed(seed)
    model.eval()
    x = model.sample(y, temperature, generator=g).cpu()
    model.train()
    rows = model.n_classes + 1
    grid = x.reshape(rows, per_class, 28, 28).permute(0, 2, 1, 3).reshape(rows * 28, per_class * 28)
    fig, ax = plt.subplots(figsize=(per_class * 0.6, rows * 0.6))
    ax.imshow(grid, cmap="gray", interpolation="nearest")
    ax.set_yticks([28 * i + 14 for i in range(rows)],
                  [str(i) for i in range(model.n_classes)] + ["none"])
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    amp = args.amp and device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = get_loaders(args.data, args.batch, args.workers,
                                            pin_memory=device.type == "cuda")
    config = dict(channels=args.channels, layers=args.layers, k_first=args.k_first, k=args.k,
                  head=args.head, n_classes=10)
    model = build(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    history, start = [], 1
    ckpt_path, hist_path = out / "model.pt", out / "history.json"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt["config"] != config:
            raise SystemExit(f"--resume: {ckpt_path} has config {ckpt['config']}, not {config}")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []
        print(f"resuming {ckpt_path} after epoch {ckpt['epoch']}")
    elif ckpt_path.exists() and not args.force:
        raise SystemExit(f"{ckpt_path} exists; pass --resume to continue or --force to overwrite")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters on {device}, amp={amp}, config {config}")
    step = 0
    for epoch in range(start, args.epochs + 1):
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            drop = torch.rand(len(y), device=device) < args.p_uncond
            y = torch.where(drop, torch.full_like(y, model.null_class), y)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x, y)
            loss = nll(logits.float(), x).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(x)
            seen += len(x)
            step += 1
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}: {run_loss / seen:.2f} nats", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        test = evaluate(model, test_loader, device, amp)
        row = {"epoch": epoch, "train": run_loss / seen, "test": test["cond"],
               "test_uncond": test["uncond"], "lr": sched.get_last_lr()[0],
               "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch}/{args.epochs}  train {row['train']:.2f}  test {test['cond']:.2f} "
              f"({test['cond'] * BITS:.4f} bits/dim)  test, no class {test['uncond']:.2f} nats  "
              f"[{row['seconds']:.0f}s]", flush=True)
        torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch,
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "args": vars(args)}, ckpt_path)
        hist_path.write_text(json.dumps(history, indent=1))
        if args.figures:
            save_samples(model, out / "samples.png")
        if args.max_steps and step >= args.max_steps:
            break
    if not args.figures:
        save_samples(model, out / "samples.png")
    print(f"saved {ckpt_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/gated")
    p.add_argument("--data", default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--layers", type=int, default=12, help="gated layers, the first included")
    p.add_argument("--k-first", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--head", type=int, default=256)
    p.add_argument("--p-uncond", type=float, default=0.1, help="probability of training with no class")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable bfloat16 autocast")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    p.add_argument("--figures", action="store_true", help="write samples.png after every epoch")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing run")
    train(p.parse_args(argv))


if __name__ == "__main__":
    main()
