"""CLI entry point for the synthetic data generator.

Example::

    python -m plate_redactor.generator.generate \\
        --n 5000 --seed 42 --backgrounds data/backgrounds --out data/synthetic

Run with no ``--backgrounds`` to use synthesised solid-colour fallbacks (handy
for a quick smoke run; real photos make a far better detector).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .augment import AugmentConfig, augment_image
from .backgrounds import BackgroundPool
from .compositor import composite
from .plate import render_plate
from .writer import prepare_dirs, write_data_yaml, write_sample

try:  # progress bar is nice-to-have; degrade gracefully if missing
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


def _split_assignment(n: int, val_split: float, seed: int) -> list[str]:
    """Deterministically assign each of ``n`` indices to 'train' or 'val'."""
    n_val = int(round(n * val_split))
    order = np.random.default_rng(seed).permutation(n)
    is_val = set(order[:n_val].tolist())
    return ["val" if i in is_val else "train" for i in range(n)]


def run(n: int = 5000, seed: int = 42, out: str | Path = "data/synthetic",
        backgrounds: str | Path | None = None, val_split: float = 0.1,
        font: str | None = None, cfg: AugmentConfig | None = None,
        show_progress: bool = True) -> dict:
    """Generate a synthetic dataset. Returns a summary dict.

    Each image is produced from an independent, seed-derived RNG
    (``default_rng([seed, i])``), so the dataset is fully reproducible and
    order-independent.
    """
    if n <= 0:
        raise ValueError("--n must be positive")
    cfg = cfg or AugmentConfig()
    pool = BackgroundPool(backgrounds)
    root = prepare_dirs(out)
    write_data_yaml(out)

    splits = _split_assignment(n, val_split, seed)
    counts = {"train": 0, "val": 0}
    width = len(str(n - 1))

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc="generating", unit="img")

    for i in iterator:
        rng = np.random.default_rng([seed, i])
        plate, _text = render_plate(rng=rng, font_path=font)
        bg = pool.sample(rng)
        image, bbox = composite(plate, bg, rng, cfg)
        image = augment_image(image, bbox, rng, cfg)
        split = splits[i]
        write_sample(root, split, f"img_{i:0{width}d}", image, bbox)
        counts[split] += 1

    return {
        "images": n,
        "train": counts["train"],
        "val": counts["val"],
        "out": str(root),
        "backgrounds": "fallback (synthetic)" if pool.using_fallback else str(backgrounds),
        "data_yaml": str(Path(root) / "data.yaml"),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m plate_redactor.generator.generate",
        description="Generate synthetic DE licence-plate detection data (YOLO format).",
    )
    p.add_argument("--n", type=int, default=5000, help="number of images to generate")
    p.add_argument("--seed", type=int, default=42, help="global random seed")
    p.add_argument("--backgrounds", default=None,
                   help="directory of background images (omit → solid fallbacks)")
    p.add_argument("--out", default="data/synthetic", help="output root directory")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="fraction of images assigned to val (default 0.1)")
    p.add_argument("--font", default=None,
                   help="path to a plate font (.ttf/.otf); see generator/fonts.py")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(
        n=args.n, seed=args.seed, out=args.out,
        backgrounds=args.backgrounds, val_split=args.val_split, font=args.font,
    )
    print(
        f"\nDone. Wrote {summary['images']} images "
        f"({summary['train']} train / {summary['val']} val) to {summary['out']}\n"
        f"  backgrounds: {summary['backgrounds']}\n"
        f"  descriptor:  {summary['data_yaml']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
