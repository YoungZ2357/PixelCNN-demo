from .data import get_device, get_loaders, get_test_set
from .export import build_bundle, export_bundle, load_run
from .model import GatedPixelCNN, OriginalPixelCNN, nll


def __getattr__(name):  # the widget pulls in anywidget; import it only when asked for
    if name == "PixelCNNDemo":
        from .widget import PixelCNNDemo
        return PixelCNNDemo
    raise AttributeError(name)


__all__ = ["GatedPixelCNN", "OriginalPixelCNN", "PixelCNNDemo", "build_bundle", "export_bundle",
           "get_device", "get_loaders", "get_test_set", "load_run", "nll"]
