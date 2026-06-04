"""Composite a rendered plate onto a background at a random scale and position.

Returns the composited RGB image plus the plate's bounding box in **normalised
``[x, y, w, h]`` coordinates** (top-left origin) — the same coordinate space as
the §2 interface contract in the README.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .augment import AugmentConfig, augment_plate_geometry


def composite(plate: Image.Image, background: Image.Image,
              rng: np.random.Generator, cfg: AugmentConfig
              ) -> tuple[Image.Image, list[float]]:
    """Paste ``plate`` (RGBA) onto ``background`` (RGB).

    Applies geometry augmentation to the plate, scales it so it covers a random
    2 %–30 % of the image area (``cfg.min_area_frac`` / ``max_area_frac``), and
    places it fully within the frame.

    Returns ``(image, [x, y, w, h])`` with the box normalised to ``[0, 1]``.
    """
    bg = background.convert("RGB")
    bg_w, bg_h = bg.size

    plate = augment_plate_geometry(plate, rng, cfg).convert("RGBA")
    pw, ph = plate.size

    # scale so the plate covers a target fraction of the image area
    frac = float(rng.uniform(cfg.min_area_frac, cfg.max_area_frac))
    scale = math.sqrt((frac * bg_w * bg_h) / (pw * ph))
    new_w = max(1, int(round(pw * scale)))
    new_h = max(1, int(round(ph * scale)))

    # never exceed the frame (cap at 95 % of each side)
    fit = min(1.0, 0.95 * bg_w / new_w, 0.95 * bg_h / new_h)
    if fit < 1.0:
        new_w = max(1, int(new_w * fit))
        new_h = max(1, int(new_h * fit))

    plate = plate.resize((new_w, new_h), Image.LANCZOS)

    max_x = bg_w - new_w
    max_y = bg_h - new_h
    px = int(rng.integers(0, max_x + 1))
    py = int(rng.integers(0, max_y + 1))

    canvas = bg.copy()
    canvas.paste(plate, (px, py), plate)

    # tight box from the (possibly warped) alpha channel
    alpha_box = plate.getchannel("A").getbbox()
    if alpha_box is None:  # fully transparent — should not happen
        alpha_box = (0, 0, new_w, new_h)
    ax0, ay0, ax1, ay1 = alpha_box
    bx, by = px + ax0, py + ay0
    bw, bh = ax1 - ax0, ay1 - ay0

    bbox = [bx / bg_w, by / bg_h, bw / bg_w, bh / bg_h]
    bbox = [min(1.0, max(0.0, v)) for v in bbox]
    return canvas, bbox
