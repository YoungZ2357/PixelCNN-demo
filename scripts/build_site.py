"""Build the static site (GitHub Pages) from a trained run.

    uv run python scripts/build_site.py                    # runs/gated -> docs/
    uv run python scripts/build_site.py --run runs/smoke --out _site

Then preview with `uv run python -m http.server -d docs` and open http://localhost:8000.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pixelcnn_demo.export import export_bundle  # noqa: E402

STATIC = ROOT / "pixelcnn_demo" / "static"


def build_site(run, out, dtype="float16", per_class=3, data_dir=ROOT / "data"):
    out = Path(out)
    if (out / "bundle").exists():
        shutil.rmtree(out / "bundle")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "site" / "index.html", out / "index.html")
    for f in ("pixelcnn_demo.js", "pixelcnn_demo.css"):
        shutil.copy(STATIC / f, out / f)
    (out / ".nojekyll").touch()  # serve files as-is
    export_bundle(run, out / "bundle", dtype=dtype, per_class=per_class, data_dir=data_dir)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"site written to {out.resolve()}  ({size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "runs" / "gated"))
    p.add_argument("--out", default=str(ROOT / "docs"))
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--per-class", type=int, default=3, help="test images per digit for inpainting")
    args = p.parse_args()
    build_site(args.run, args.out, args.dtype, args.per_class)


if __name__ == "__main__":
    main()
