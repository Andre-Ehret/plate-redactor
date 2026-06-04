"""Render plausible German (DE-format) licence plates as RGBA images.

A DE plate is: ``<city> <letters> <digits>`` — e.g. ``B AB 1234`` — drawn as a
white rounded rectangle with a black border and a blue EU strip (gold stars +
country code ``D``) on the left. Only the *appearance* matters here; the text is
random and never read back (this is a detector, not OCR).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .fonts import load_font

# DE plates use Latin capitals (umlauts exist but are rare; we skip them).
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGITS = "0123456789"
_NONZERO = "123456789"

# Colours
_WHITE = (255, 255, 255, 255)
_BLACK = (0, 0, 0, 255)
_EU_BLUE = (0, 51, 153, 255)
_EU_GOLD = (255, 204, 0, 255)

# Plate proportions (a real DE plate is 520 mm x 110 mm ≈ 4.73:1).
_PLATE_H = 130
_PLATE_W = int(_PLATE_H * 4.7)


def _pick(rng: np.random.Generator, alphabet: str, n: int) -> str:
    idx = rng.integers(0, len(alphabet), size=n)
    return "".join(alphabet[i] for i in idx)


def random_plate_text(rng: np.random.Generator | None = None) -> str:
    """Generate a random DE-format plate string, e.g. ``"M AB 1234"``.

    City code is 1–3 letters, then 1–2 letters, then a 1–4 digit number with no
    leading zero — matching the real German registration grammar.
    """
    if rng is None:
        rng = np.random.default_rng()
    city = _pick(rng, _LETTERS, int(rng.integers(1, 4)))
    letters = _pick(rng, _LETTERS, int(rng.integers(1, 3)))
    n_digits = int(rng.integers(1, 5))
    digits = _pick(rng, _NONZERO, 1)
    if n_digits > 1:
        digits += _pick(rng, _DIGITS, n_digits - 1)
    return f"{city} {letters} {digits}"


def _draw_eu_strip(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int,
                   font_path: str | None) -> None:
    """Blue EU strip with a ring of gold stars and the country code ``D``."""
    draw.rectangle([x0, y0, x1, y1], fill=_EU_BLUE)
    cx = (x0 + x1) / 2.0
    strip_w = x1 - x0
    # ring of 12 stars in the upper portion
    ring_cy = y0 + (y1 - y0) * 0.32
    ring_r = strip_w * 0.30
    star_r = max(1.0, strip_w * 0.07)
    for k in range(12):
        ang = 2.0 * np.pi * k / 12.0
        sx = cx + ring_r * np.sin(ang)
        sy = ring_cy - ring_r * np.cos(ang)
        draw.ellipse([sx - star_r, sy - star_r, sx + star_r, sy + star_r], fill=_EU_GOLD)
    # country code "D" below the ring
    d_font = load_font(int(strip_w * 0.55), font_path)
    db = draw.textbbox((0, 0), "D", font=d_font)
    dw, dh = db[2] - db[0], db[3] - db[1]
    draw.text((cx - dw / 2 - db[0], y0 + (y1 - y0) * 0.62 - db[1]),
              "D", font=d_font, fill=_WHITE)


def render_plate(text: str | None = None,
                 rng: np.random.Generator | None = None,
                 font_path: str | None = None) -> tuple[Image.Image, str]:
    """Render a DE-format plate.

    Args:
        text: plate text to render. ``None`` → a random plausible plate.
        rng: numpy Generator for the random text (ignored if ``text`` given).
        font_path: explicit font path (see :mod:`.fonts` for resolution order).

    Returns:
        ``(image, text)`` — an RGBA :class:`PIL.Image.Image` (opaque plate on a
        transparent background, so it can be composited and warped) and the
        string that was rendered.
    """
    if rng is None:
        rng = np.random.default_rng()
    if text is None:
        text = random_plate_text(rng)
    text = text.upper()

    w, h = _PLATE_W, _PLATE_H
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    radius = int(h * 0.09)
    border = max(2, int(h * 0.045))
    # white body
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=_WHITE)
    # black border
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                           outline=_BLACK, width=border)

    # EU strip on the left
    strip_x0 = border
    strip_x1 = border + int(w * 0.085)
    _draw_eu_strip(draw, strip_x0, border, strip_x1, h - 1 - border, font_path)

    # registration text in the remaining area
    avail_x0 = strip_x1 + int(w * 0.03)
    avail_x1 = w - border - int(w * 0.02)
    avail_w = avail_x1 - avail_x0
    avail_h = h - 2 * border

    size = int(h * 0.62)
    font = load_font(size, font_path)
    while size > 8:
        tb = draw.textbbox((0, 0), text, font=font)
        if (tb[2] - tb[0]) <= avail_w and (tb[3] - tb[1]) <= avail_h:
            break
        size -= 2
        font = load_font(size, font_path)

    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = avail_x0 + (avail_w - tw) / 2 - tb[0]
    ty = border + (avail_h - th) / 2 - tb[1]
    draw.text((tx, ty), text, font=font, fill=_BLACK)

    return img, text
