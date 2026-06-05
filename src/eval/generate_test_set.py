"""Phase 3 — generate the dedicated hard-case test set (``data/test_hard/``).

This is **not** the Phase-1 train/val data: a different seed (default 999) and
the most challenging augmentation settings, so the numbers reflect real-world
difficulty rather than the saturated synthetic val split. Six stress subsets
(100 images each) plus a portrait/landscape orientation check (50 each):

    images/<subset>/img_000.jpg   labels/<subset>/img_000.txt   (YOLO format)

Each subset forces the augmentation it is named for (probability 1.0) so every
image is genuinely hard, rather than relying on the Phase-1 random toggles.

Usage (from the repo root)::

    python src/eval/generate_test_set.py                       # fallback backgrounds
    python src/eval/generate_test_set.py --backgrounds data/backgrounds
    python src/eval/generate_test_set.py --per-subset 100 --seed 999
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from plate_redactor.generator.augment import AugmentConfig, augment_image
from plate_redactor.generator.backgrounds import BackgroundPool
from plate_redactor.generator.compositor import composite
from plate_redactor.generator.plate import render_plate

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports
from common import HARD_SUBSETS, ORIENT_SUBSETS  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it

# Fallback canvas sizes per orientation (mirrors generator.backgrounds style).
_ORIENT_SIZES = {
    "landscape": ((1024, 768), (1280, 720), (800, 600)),
    "portrait": ((768, 1024), (720, 1280), (600, 800)),
}


def subset_config(name: str) -> AugmentConfig:
    """The augmentation profile for one hard-case subset (forced, prob high)."""
    off = dict(perspective=False, brightness_contrast=False, shadow=False,
               blur=False, dirt=False, occlusion=False)
    if name == "tilt":
        return AugmentConfig(**{**off, "perspective": True}, prob=1.0,
                             max_rotation_deg=20.0, max_keystone=0.18,
                             min_area_frac=0.05, max_area_frac=0.30)
    if name == "dirty":
        return AugmentConfig(**{**off, "brightness_contrast": True, "dirt": True,
                                "blur": True}, prob=1.0,
                             min_area_frac=0.05, max_area_frac=0.25)
    if name == "occluded":
        return AugmentConfig(**{**off, "occlusion": True}, prob=1.0,
                             occlusion_min_frac=0.20, occlusion_max_frac=0.40,
                             min_area_frac=0.06, max_area_frac=0.30)
    if name == "shadow":
        return AugmentConfig(**{**off, "shadow": True}, prob=1.0,
                             min_area_frac=0.05, max_area_frac=0.30)
    if name == "small":
        # No appearance augs — the difficulty is purely the tiny plate (<5 %).
        return AugmentConfig(**off, prob=0.0,
                             min_area_frac=0.01, max_area_frac=0.05)
    if name == "mixed":
        return AugmentConfig(perspective=True, brightness_contrast=True,
                             shadow=True, blur=True, dirt=True, occlusion=True,
                             prob=0.7, occlusion_min_frac=0.15,
                             occlusion_max_frac=0.40,
                             min_area_frac=0.03, max_area_frac=0.25)
    raise ValueError(f"unknown subset: {name}")


def _synth_oriented_bg(rng: np.random.Generator, orientation: str) -> Image.Image:
    """A solid-gradient fallback background of the requested orientation."""
    sizes = _ORIENT_SIZES[orientation]
    w, h = sizes[int(rng.integers(0, len(sizes)))]
    base = rng.integers(40, 216, size=3)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :] = base
    grad = np.linspace(-25, 25, h).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grad[:, None, None], 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _split_pool_by_orientation(pool: BackgroundPool) -> dict[str, list[Path]]:
    """Bucket real background paths into portrait / landscape by aspect ratio."""
    buckets: dict[str, list[Path]] = {"portrait": [], "landscape": []}
    for p in pool.paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except OSError:
            continue
        buckets["portrait" if h > w else "landscape"].append(p)
    return buckets


def _prepare_dirs(root: Path, subsets: list[str]) -> None:
    for sub in ("images", "labels"):
        for s in subsets:
            (root / sub / s).mkdir(parents=True, exist_ok=True)


def _write(root: Path, subset: str, name: str, image: Image.Image,
           bbox_xywh: list[float]) -> None:
    """Write one image + YOLO label into ``<root>/{images,labels}/<subset>/``."""
    image.convert("RGB").save(root / "images" / subset / f"{name}.jpg",
                              "JPEG", quality=92)
    x, y, w, h = bbox_xywh
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731 — YOLO coords in [0,1]
    cx, cy = x + w / 2.0, y + h / 2.0
    line = (f"0 {clamp(cx):.6f} {clamp(cy):.6f} "
            f"{clamp(w):.6f} {clamp(h):.6f}\n")
    (root / "labels" / subset / f"{name}.txt").write_text(line, encoding="utf-8")


def _gen_subset(root: Path, subset: str, n: int, cfg: AugmentConfig,
                pool: BackgroundPool, base_seed: int, idx: int,
                font: str | None, orientation: str | None,
                orient_paths: dict[str, list[Path]] | None) -> None:
    width = len(str(max(0, n - 1)))
    desc = subset if orientation is None else f"{subset}"
    for i in tqdm(range(n), desc=desc, unit="img"):
        rng = np.random.default_rng([base_seed, idx, i])
        plate, _ = render_plate(rng=rng, font_path=font)
        if orientation is not None:
            candidates = (orient_paths or {}).get(orientation, [])
            if candidates:
                with Image.open(candidates[int(rng.integers(0, len(candidates)))]) as im:
                    bg = im.convert("RGB")
            else:
                bg = _synth_oriented_bg(rng, orientation)
        else:
            bg = pool.sample(rng)
        image, bbox = composite(plate, bg, rng, cfg)
        image = augment_image(image, bbox, rng, cfg)
        _write(root, subset, f"img_{i:0{width}d}", image, bbox)


def run(out: str | Path = "data/test_hard", seed: int = 999,
        backgrounds: str | Path | None = None, per_subset: int = 100,
        per_orientation: int = 50, font: str | None = None) -> dict:
    root = Path(out)
    subsets = list(HARD_SUBSETS) + list(ORIENT_SUBSETS)
    _prepare_dirs(root, subsets)
    pool = BackgroundPool(backgrounds)
    orient_paths = None if pool.using_fallback else _split_pool_by_orientation(pool)

    counts: dict[str, int] = {}
    for idx, subset in enumerate(HARD_SUBSETS):
        _gen_subset(root, subset, per_subset, subset_config(subset), pool,
                    seed, idx, font, None, None)
        counts[subset] = per_subset

    for j, orient in enumerate(ORIENT_SUBSETS):
        _gen_subset(root, orient, per_orientation, AugmentConfig(), pool,
                    seed, len(HARD_SUBSETS) + j, font, orient, orient_paths)
        counts[orient] = per_orientation

    return {
        "out": str(root),
        "counts": counts,
        "total": sum(counts.values()),
        "backgrounds": "fallback (synthetic)" if pool.using_fallback else str(backgrounds),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python src/eval/generate_test_set.py",
        description="Generate the Phase-3 hard-case test set (YOLO format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="data/test_hard", help="output root (gitignored)")
    p.add_argument("--seed", type=int, default=999,
                   help="seed (must differ from any training seed)")
    p.add_argument("--backgrounds", default=None,
                   help="directory of background images (omit -> solid fallbacks)")
    p.add_argument("--per-subset", type=int, default=100,
                   help="images per hard subset (tilt/dirty/occluded/shadow/small/mixed)")
    p.add_argument("--per-orientation", type=int, default=50,
                   help="images per orientation (portrait/landscape)")
    p.add_argument("--font", default=None, help="path to a plate font (.ttf/.otf)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(out=args.out, seed=args.seed, backgrounds=args.backgrounds,
                  per_subset=args.per_subset, per_orientation=args.per_orientation,
                  font=args.font)
    rows = "  ".join(f"{k}={v}" for k, v in summary["counts"].items())
    print(f"\nDone. Wrote {summary['total']} test images to {summary['out']}")
    print(f"  subsets: {rows}")
    print(f"  backgrounds: {summary['backgrounds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
