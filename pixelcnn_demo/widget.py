from pathlib import Path

import anywidget
import traitlets

from .export import build_bundle, load_run

_STATIC = Path(__file__).parent / "static"


class PixelCNNDemo(anywidget.AnyWidget):
    """The three-tab demo (blind spot, sampling, inpainting), running the model in the browser.

        demo = PixelCNNDemo.from_run("runs/gated")
        demo            # display
        demo.tab = "sample"   # "blind" | "sample" | "inpaint", synced both ways
    """
    _esm = _STATIC / "pixelcnn_demo.js"
    _css = _STATIC / "pixelcnn_demo.css"

    manifest = traitlets.Dict().tag(sync=True)
    buffers = traitlets.Dict().tag(sync=True)  # path -> bytes (sent as binary buffers)
    tab = traitlets.Unicode("blind").tag(sync=True)

    @classmethod
    def from_run(cls, run_dir, dtype="float16", per_class=3, data_dir="./data", **kw):
        model, metrics, history = load_run(run_dir)
        manifest, buffers = build_bundle(model, metrics, history, dtype=dtype, per_class=per_class,
                                         data_dir=data_dir)
        return cls(manifest=manifest, buffers=buffers, **kw)
