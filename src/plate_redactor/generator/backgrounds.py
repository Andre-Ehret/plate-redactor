"""Background image pool for compositing plates onto.

Real backgrounds (cars, streets, parking lots) are supplied by the user via a
directory and are **never committed** (``data/`` is git-ignored — see the README
for where to source freely-licensed images, e.g. Unsplash / Pexels / CC0 Flickr).

When no directory is given, the pool synthesises solid-colour backgrounds in both
portrait and landscape orientations. These keep the generator (and the smoke
test) runnable with zero setup and exercise the portrait/landscape code paths,
but real photos give far better detectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fallback canvas sizes (w, h) — mix of landscape and portrait.
_FALLBACK_SIZES = (
    (1024, 768), (1280, 720), (800, 600),   # landscape
    (768, 1024), (720, 1280), (600, 800),   # portrait
)


class BackgroundPool:
    """Samples background images, either from disk or synthesised fallbacks."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.paths: list[Path] = []
        if directory is not None:
            d = Path(directory)
            if not d.is_dir():
                raise FileNotFoundError(f"Background directory not found: {d}")
            self.paths = sorted(
                p for p in d.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not self.paths:
                raise ValueError(f"No background images found under: {d}")

    @property
    def using_fallback(self) -> bool:
        return not self.paths

    def _fallback(self, rng: np.random.Generator) -> Image.Image:
        w, h = _FALLBACK_SIZES[int(rng.integers(0, len(_FALLBACK_SIZES)))]
        base = rng.integers(40, 216, size=3)
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[:, :] = base
        # a mild vertical gradient so it is not perfectly flat
        grad = np.linspace(-25, 25, h).astype(np.int16)
        img = np.clip(img.astype(np.int16) + grad[:, None, None], 0, 255).astype(np.uint8)
        return Image.fromarray(img, "RGB")

    def sample(self, rng: np.random.Generator) -> Image.Image:
        """Return one background as an RGB :class:`PIL.Image.Image`."""
        if not self.paths:
            return self._fallback(rng)
        path = self.paths[int(rng.integers(0, len(self.paths)))]
        with Image.open(path) as im:
            return im.convert("RGB")
