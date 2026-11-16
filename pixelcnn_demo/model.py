"""Class-conditional Gated PixelCNN (van den Oord et al., 2016) for binarized 28x28 images.

Every convolution is a `CausalConv`: a plain rectangular kernel placed at a fixed offset from the
output pixel, so causality is a property of the shape, not of a weight mask:

    out[r, c] = sum_{i<kh, j<kw} W[:, :, i, j] . x[r + dy + i, c + dx + j]     (zeros outside)

Vertical stack   kernel (k//2 + 1) x k, rows r-k//2 .. r, cols c-k//2 .. c+k//2.
                 It sees the current row, so it is shifted down one row before it feeds the
                 horizontal stack: the horizontal stack at row r only gets rows < r from it.
Horizontal stack kernel 1 x (k//2 + 1), cols c-k//2 .. c (first layer: 1 x k//2, cols < c).
                 Residual connection from layer 2 on.

Each layer (paper eq. 2-4, Fig. 2), with y the class and h_y a learned per-class bias:

    v' = conv_v(v) + V_v h_y                    v_next = tanh(v'_a) * sigmoid(v'_b)
    h' = conv_h(h) + 1x1(shift_down(v')) + V_h h_y
    h_next = h + 1x1(tanh(h'_a) * sigmoid(h'_b))

Class index `n_classes` is "no class": trained with probability p_uncond, so the same model also
samples unconditionally. Output: one Bernoulli logit per pixel.

`OriginalPixelCNN` (single masked stack, 3x3 masks A then B) exists only to show the blind spot.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    def __init__(self, cin, cout, kh, kw, dy, dx):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (kh, kw))
        self.kh, self.kw, self.dy, self.dx = kh, kw, dy, dx
        # F.pad order: left, right, top, bottom. Negative values crop.
        self.pad = (-dx, kw - 1 + dx, -dy, kh - 1 + dy)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


def shift_down(x):
    return F.pad(x, (0, 0, 1, -1))


def gate(x):
    a, b = x.chunk(2, dim=1)
    return torch.tanh(a) * torch.sigmoid(b)


class GatedLayer(nn.Module):
    def __init__(self, cin, channels, k, first, n_cond):
        super().__init__()
        r = k // 2
        c2 = 2 * channels
        self.first = first
        self.v = CausalConv(cin, c2, r + 1, k, -r, -r)
        self.h = CausalConv(cin, c2, 1, r, 0, -r) if first else CausalConv(cin, c2, 1, r + 1, 0, -r)
        self.v2h = nn.Conv2d(c2, c2, 1)
        self.h_out = nn.Conv2d(channels, channels, 1)
        self.v_emb = nn.Embedding(n_cond, c2)
        self.h_emb = nn.Embedding(n_cond, c2)

    def forward(self, v, h, y):
        vp = self.v(v) + self.v_emb(y)[:, :, None, None]
        hp = self.h(h) + self.v2h(shift_down(vp)) + self.h_emb(y)[:, :, None, None]
        h_new = self.h_out(gate(hp))
        if not self.first:
            h_new = h_new + h
        return gate(vp), h_new


class GatedPixelCNN(nn.Module):
    def __init__(self, channels=64, layers=12, k_first=7, k=3, head=256, n_classes=10):
        super().__init__()
        self.config = dict(channels=channels, layers=layers, k_first=k_first, k=k, head=head,
                           n_classes=n_classes)
        self.n_classes = n_classes
        self.layers = nn.ModuleList(
            [GatedLayer(1, channels, k_first, True, n_classes + 1)] +
            [GatedLayer(channels, channels, k, False, n_classes + 1) for _ in range(layers - 1)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, head, 1), nn.ReLU(),
                                  nn.Conv2d(head, 1, 1))

    @property
    def null_class(self):
        return self.n_classes

    def features(self, x, y):
        """Last horizontal-stack features [B, C, H, W]."""
        v = h = x
        for layer in self.layers:
            v, h = layer(v, h, y)
        return h

    def forward(self, x, y):
        """x [B, 1, H, W] in {0, 1}; y [B] class index (n_classes = none). Returns logits [B, 1, H, W]."""
        return self.head(self.features(x, y))

    @torch.no_grad()
    def sample(self, y, temperature=1.0, x=None, known=None, generator=None):
        """Pixel-by-pixel sampling with a full forward pass per pixel (the naive way).
        y [B]; x, known [B, 1, H, W]: pixels where known is True are copied from x, not sampled."""
        b = len(y)
        dev = next(self.parameters()).device
        out = torch.zeros(b, 1, 28, 28, device=dev) if x is None else x.clone().float().to(dev)
        known = torch.zeros_like(out, dtype=torch.bool) if known is None else known.to(dev)
        y = y.to(dev)
        for r in range(28):
            for c in range(28):
                if known[:, :, r, c].all():
                    continue
                p = torch.sigmoid(self(out, y)[:, :, r, c] / temperature)
                s = torch.bernoulli(p, generator=generator)
                out[:, :, r, c] = torch.where(known[:, :, r, c], out[:, :, r, c], s)
        return out


def nll(logits, x):
    """Per-image negative log-likelihood in nats: [B]."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)


class MaskedConv(nn.Conv2d):
    """k x k convolution with the PixelCNN raster mask. Mask A hides the centre, B shows it."""

    def __init__(self, cin, cout, k, mask):
        super().__init__(cin, cout, k, padding=k // 2)
        m = torch.ones_like(self.weight)
        m[:, :, k // 2, k // 2 + (mask == "B"):] = 0
        m[:, :, k // 2 + 1:] = 0
        self.register_buffer("mask", m)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)


class OriginalPixelCNN(nn.Module):
    """Single stack of masked convolutions (PixelCNN, 2016a). Only used to show the blind spot."""

    def __init__(self, channels=8, layers=6, k=3):
        super().__init__()
        self.convs = nn.ModuleList([MaskedConv(1, channels, k, "A")] +
                                   [MaskedConv(channels, channels, k, "B") for _ in range(layers - 1)])

    def features(self, x, y=None):
        for conv in self.convs:
            x = torch.tanh(conv(x))
        return x


def build(config):
    return GatedPixelCNN(**config)
