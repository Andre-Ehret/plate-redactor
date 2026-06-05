"""Phase 5 — generate the committed contract-test fixtures (``tests/fixtures/``).

These are the example stills the contract suite (``tests/test_contract.py``) runs
the exported model against. Unlike the Phase-3 hard-case set they are **few,
named and committed** so the suite is hermetic — but they are produced from the
same Phase-1 generator building blocks with a fixed seed, so they remain fully
synthetic (no real plate photo ever enters the repo) and reproducible.

Layout (flat, companion ``.txt`` next to each ``.jpg``)::

    tests/fixtures/landscape_clean.jpg   tests/fixtures/landscape_clean.txt
    tests/fixtures/portrait_clean.jpg    tests/fixtures/portrait_clean.txt
    ...

Each ``.txt`` is a one-line YOLO label ``0 <cx> <cy> <w> <h>`` (class 0 = plate,
normalised centre form) — the same format the generator's writer emits.

Regenerate with::

    python tests/fixtures/generate_fixtures.py            # --seed 1234 (default)

The committed fixtures use the **solid-colour fallback backgrounds**, matching the
released checkpoint (which trained on fallback backgrounds, see WORKING_STATE.md),
so the detector reliably fires on the orientation fixtures (contract test T6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Repo-root import of the generator package + the Phase-3 subset configs.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from plate_redactor.generator.augment import AugmentConfig, augment_image  # noqa: E402
from plate_redactor.generator.backgrounds import BackgroundPool  # noqa: E402
from plate_redactor.generator.compositor import composite  # noqa: E402
from plate_redactor.generator.plate import render_plate  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))
from generate_test_set import _synth_oriented_bg, subset_config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# A clean, well-sized plate so the orientation fixtures detect reliably (T6).
_CLEAN = AugmentConfig.disabled()
_CLEAN.min_area_frac = 0.10
_CLEAN.max_area_frac = 0.22

# (name, augment-config, orientation|None). Orientation forces a portrait/
# landscape fallback canvas; None uses a random fallback size.
_FIXTURES: list[tuple[str, AugmentConfig, str | None]] = [
    ("landscape_clean", _CLEAN, "landscape"),
    ("portrait_clean", _CLEAN, "portrait"),
    ("tilt", subset_config("tilt"), "landscape"),
    ("occluded", subset_config("occluded"), "landscape"),
    ("dirty", subset_config("dirty"), "portrait"),
    ("shadow", subset_config("shadow"), "landscape"),
]


def _write(name: str, image: Image.Image, bbox_xywh: list[float]) -> None:
    image.convert("RGB").save(OUT_DIR / f"{name}.jpg", "JPEG", quality=92)
    x, y, w, h = bbox_xywh
    clamp = lambda v: min(1.0, max(0.0, v))  # noqa: E731
    cx, cy = x + w / 2.0, y + h / 2.0
    line = f"0 {clamp(cx):.6f} {clamp(cy):.6f} {clamp(w):.6f} {clamp(h):.6f}\n"
    (OUT_DIR / f"{name}.txt").write_text(line, encoding="utf-8")


def generate(seed: int = 1234) -> list[str]:
    pool = BackgroundPool(None)  # solid-colour fallbacks — matches the checkpoint
    written: list[str] = []
    for idx, (name, cfg, orient) in enumerate(_FIXTURES):
        rng = np.random.default_rng([seed, idx])
        plate, _ = render_plate(rng=rng)
        bg = _synth_oriented_bg(rng, orient) if orient else pool.sample(rng)
        image, bbox = composite(plate, bg, rng, cfg)
        image = augment_image(image, bbox, rng, cfg)
        _write(name, image, bbox)
        written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate Phase-5 contract-test fixtures.")
    p.add_argument("--seed", type=int, default=1234, help="generation seed (default 1234)")
    args = p.parse_args(argv)
    names = generate(args.seed)
    print(f"Wrote {len(names)} fixtures (seed {args.seed}) to {OUT_DIR}:")
    for n in names:
        print(f"  {n}.jpg + {n}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
