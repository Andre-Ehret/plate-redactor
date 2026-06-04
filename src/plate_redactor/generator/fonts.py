"""Font resolution for the synthetic plate renderer.

We deliberately **do not commit a licence-plate font binary** to this public
repo: the freely-circulating FE-Schrift digitisations (dafont, Fontasy, …) have
murky redistribution terms, which clashes with the project's clean-provenance
stance. Because this is a *detector* (it only learns *where* a plate is, not how
to read it), exact glyph fidelity is irrelevant — any bold, roughly monospaced
font produces a perfectly good training target.

Resolution order (first hit wins):

1. an explicit path passed to ``--font`` / ``load_font(explicit=...)``
2. the ``PLATE_REDACTOR_FONT`` environment variable
3. any ``*.ttf`` / ``*.otf`` dropped into ``generator/assets/fonts/``
   (git-ignored — this is where you put a real FE-Schrift if you have one)
4. a common system font (DejaVu / Liberation / Arial / Helvetica), runtime-only
5. Pillow's built-in scalable default font (always available, permissively
   licensed) — guarantees the generator runs anywhere with no setup

For *reproducible* high-quality datasets, supply an explicit font; the system
fallback (step 4) can differ between machines.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import ImageFont

# Curated, license-clean / commonly-installed fonts. Bold or monospace first so
# the plate looks plausibly DIN-1451-ish. Used at runtime only — never committed.
_SYSTEM_FONT_CANDIDATES = (
    # Linux — DejaVu / Liberation (permissive, ship with most distros)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\consolab.ttf",
)


def _bundled_font_candidates() -> list[str]:
    assets = Path(__file__).resolve().parent / "assets" / "fonts"
    if not assets.is_dir():
        return []
    fonts: list[str] = []
    for pattern in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
        fonts.extend(str(p) for p in sorted(assets.glob(pattern)))
    return fonts


def resolve_font_path(explicit: str | None = None) -> str | None:
    """Return the path of the first usable font, or ``None`` to use PIL's default."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("PLATE_REDACTOR_FONT")
    if env:
        candidates.append(env)
    candidates.extend(_bundled_font_candidates())
    candidates.extend(_SYSTEM_FONT_CANDIDATES)

    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def load_font(size: int, explicit: str | None = None) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at ``size`` px, falling back to PIL's default font."""
    path = resolve_font_path(explicit)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    # Pillow >= 10.1 ships a scalable default font; always available.
    return ImageFont.load_default(size=size)
