"""Binarized MNIST: pixels in {0, 1}, so a per-pixel Bernoulli is an exact likelihood.

Train images are resampled x ~ Bernoulli(intensity) every time they are read ("dynamic");
the test set is one seeded draw, the same for every evaluation.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class BinarizedMNIST(Dataset):
    def __init__(self, base, mode, seed=0):
        self.p = base.data.float().div(255).unsqueeze(1)  # [N, 1, 28, 28]
        self.targets = base.targets.clone()
        self.mode = mode
        if mode == "fixed":
            self.x = torch.bernoulli(self.p, generator=torch.Generator().manual_seed(seed))
        elif mode != "dynamic":
            raise ValueError(f"unknown mode {mode!r}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = torch.bernoulli(self.p[i]) if self.mode == "dynamic" else self.x[i]
        return x, int(self.targets[i])


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_test_set(data_dir="./data", seed=0):
    """The fixed binarized test set: (x [N, 1, 28, 28] float {0, 1}, labels [N])."""
    ds = BinarizedMNIST(datasets.MNIST(data_dir, train=False, download=True), "fixed", seed)
    return ds.x, ds.targets


def get_loaders(data_dir="./data", batch_size=128, num_workers=2, pin_memory=False, seed=0):
    train = BinarizedMNIST(datasets.MNIST(data_dir, train=True, download=True), "dynamic")
    test = BinarizedMNIST(datasets.MNIST(data_dir, train=False, download=True), "fixed", seed)
    kw = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(test, batch_size=500, shuffle=False, **kw))
