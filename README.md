# PixelCNN-demo

A class-conditional **Gated PixelCNN** (van den Oord et al., *Conditional Image Generation with
PixelCNN Decoders*, NeurIPS 2016) trained on binarized MNIST. The demo runs the model in the
browser and has three tabs, one for each idea in the paper:

1. **Blind spot.** This tab is computed from the architecture alone and needs no model. Hover a
   pixel to see its receptive field as layers are added. The original PixelCNN never reaches a
   wedge of earlier pixels. The vertical and horizontal stacks of the Gated PixelCNN reach all of them.
2. **Sampling.** Pick a class y and watch the image fill in, one network evaluation per pixel (784
   in all). You can also adjust the temperature and sample one of every digit.
3. **Inpainting.** Keep the top half of a test digit and the model completes the bottom half. Keep
   the bottom half instead and the model fails, because it can only condition on the past. The
   −log p of the kept half shows the difference.

## Setup

```bash
uv sync
```

This installs the CUDA 12.8 build of PyTorch on Windows/Linux (see `[tool.uv.sources]` in
`pyproject.toml`).

## 1. Train (run it yourself, in a terminal)

```bash
uv run python -m pixelcnn_demo.train
```

The defaults are 12 gated layers × 64 channels, a 7×7 first layer and 3×3 after it, 30 epochs,
Adam at 1e-3 with cosine decay, and bf16 autocast on CUDA. With probability 0.1 a training image is
given the "no class" label, so the same network also samples unconditionally. The run is written to
`runs/gated/`: `model.pt`, `history.json` and `samples.png`. The checkpoint is saved after every
epoch, so you can stop the run and continue later with `--resume`. Useful flags:

```bash
uv run python -m pixelcnn_demo.train --epochs 40 --figures          # samples.png after every epoch
uv run python -m pixelcnn_demo.train --resume                        # continue an interrupted run
uv run python -m pixelcnn_demo.train --out runs/smoke --channels 32 --layers 8 --head 64 --epochs 1 --max-steps 300 --force
```

The last command is a smoke test of about 30 s. Each epoch prints the test NLL in nats per image
and in bits/dim, both with the true class and with no class.

## 2. Check the browser engine against PyTorch

```bash
uv run python scripts/check_js.py
```

This needs `node` on PATH and runs three checks:

- **Logits:** the JS logits match a PyTorch forward pass (same float16 weights, tolerance 2e-3).
- **Sampling:** JS sampling (incremental, one step per pixel) gives the same image as PyTorch's
  naive sampler (784 full forward passes) when both use the same random numbers.
- **Receptive fields:** the receptive fields drawn in tab 1 match the nonzero input gradients of
  real PyTorch models, for both architectures.

## 3. Build and preview the site

```bash
uv run python scripts/build_site.py
```

```bash
uv run python -m http.server -d docs
```

Then open http://localhost:8000. You can link straight to a tab with `?tab=blind`, `?tab=sample`
or `?tab=inpaint`. The bundle is about 2 MB of float16 weights plus 30 test images. To publish,
commit `docs/` and set **Settings → Pages → Source** to `main` / `/docs`.

## Notebook

```bash
uv run jupyter lab main.ipynb
```

The notebook contains the same demo as an anywidget (`PixelCNNDemo.from_run("runs/gated")`), the
training curves, samples drawn in PyTorch, and the blind spot computed from gradients. The widget
carries the model weights, so clear the outputs before you commit the notebook.

## How the browser runs it

`pixelcnn_demo/static/pixelcnn_demo.js` is plain JS with no framework and no build step. It
evaluates the network incrementally in raster order:

- **Vertical stack:** row r depends only on rows ≤ r, so it is computed once, as soon as that row
  is finished.
- **Horizontal stack:** at (r, c) it depends only on earlier pixels and on features to its left that
  are already cached, so each pixel needs one step through every layer.

The result is exactly what 784 full forward passes would give (check 2 above verifies this). A full
sample takes about 1 s on a laptop.

## Differences from the paper

- **Binary pixels:** each conditional is one Bernoulli logit, not a 256-way softmax.
- **One channel:** there is no R→G→B ordering inside a pixel.
- **Class conditioning:** a one-hot vector, i.e. one learned bias per class and gate (eq. 4).
  An extra "no class" index makes the model unconditional.

## Presenting it (about 3.5 minutes)

| Time | Tab | Do | Say |
| --- | --- | --- | --- |
| 0:00–1:00 | Blind spot | *PixelCNN*: hover a mid pixel and drag layers 1 → 12. Then switch to *Gated PixelCNN*. | Red pixels come earlier but never reach the output, at any depth. Splitting into a vertical stack (above) and a horizontal stack (left) removes the blind spot. |
| 1:00–2:20 | Sampling | Sample a 7 at *slow*, then *All digits*, then *none*. Move the temperature. | Each pixel needs its own pass, 784 one after another. Training needs one pass, because the real image supplies every x<ᵢ at once. y enters every gate as a bias. |
| 2:20–3:30 | Inpainting | Keep the top half and complete it two or three times. Then *keep bottom half*. | The model can only condition on the past. With the top missing, it generates the top blind and the halves do not fit. The −log p of the kept half jumps. |

## Layout

| Path | What |
| --- | --- |
| `pixelcnn_demo/model.py` | `CausalConv`, `GatedLayer`, `GatedPixelCNN` (plus `OriginalPixelCNN` for the blind spot) |
| `pixelcnn_demo/data.py` | binarized MNIST (train resampled, test one fixed draw) |
| `pixelcnn_demo/train.py` | training CLI |
| `pixelcnn_demo/export.py` | model bundle (`manifest.json` + `weights.bin` + test images) |
| `pixelcnn_demo/widget.py`, `pixelcnn_demo/static/` | anywidget and the single JS/CSS front end |
| `site/index.html`, `scripts/build_site.py` | standalone page and its builder (→ `docs/`) |
| `scripts/check_js.py`, `scripts/check_js.mjs` | JS-vs-PyTorch checks |
