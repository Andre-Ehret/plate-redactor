"""Augmentation pipeline — the hard cases that determine real-world recall.

Augmentations split into two scopes:

* **Geometry** (:func:`augment_plate_geometry`) runs on the *plate alone* before
  compositing — perspective/rotation change the plate's outline, so the axis-
  aligned bounding box is recomputed from the warped alpha mask afterwards.
* **Appearance** (:func:`augment_image`) runs on the *composited image* and
  never changes the box: brightness/contrast, shadow, blur, dirt and partial
  occlusion. Occlusion deliberately leaves the box intact — the detector must
  still fire on a partly-covered plate (recall over precision).

Scale variation (plate occupies 2 %–30 % of the image) is handled by the
compositor, driven by :attr:`AugmentConfig.min_area_frac` / ``max_area_frac``.

Every augmentation is independently toggled and applied with probability
:attr:`AugmentConfig.prob`, all driven by a single seeded
:class:`numpy.random.Generator` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class AugmentConfig:
    """Per-augmentation toggles and ranges. Each is applied with prob ``prob``."""

    perspective: bool = True
    brightness_contrast: bool = True
    shadow: bool = True
    blur: bool = True
    dirt: bool = True
    occlusion: bool = True

    prob: float = 0.6

    # geometry
    max_rotation_deg: float = 20.0
    max_keystone: float = 0.12

    # scale (used by the compositor)
    min_area_frac: float = 0.02
    max_area_frac: float = 0.30

    # appearance
    occlusion_max_frac: float = 0.40  # max share of the plate that may be hidden

    @classmethod
    def disabled(cls) -> "AugmentConfig":
        """All augmentations off — useful for debugging / clean baselines."""
        return cls(perspective=False, brightness_contrast=False, shadow=False,
                   blur=False, dirt=False, occlusion=False)


def _hit(rng: np.random.Generator, cfg: AugmentConfig, enabled: bool) -> bool:
    return enabled and rng.random() < cfg.prob


# --------------------------------------------------------------------------- #
# Geometry (plate-only, pre-composite)
# --------------------------------------------------------------------------- #

def augment_plate_geometry(plate: Image.Image, rng: np.random.Generator,
                           cfg: AugmentConfig) -> Image.Image:
    """Apply perspective tilt + rotation to an RGBA plate, cropped tight to alpha."""
    if not _hit(rng, cfg, cfg.perspective):
        return plate

    rgba = np.array(plate.convert("RGBA"))
    h, w = rgba.shape[:2]
    pad = int(max(h, w) * 0.6)
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = rgba
    ch, cw = canvas.shape[:2]

    src = np.float32([[pad, pad], [pad + w, pad],
                      [pad + w, pad + h], [pad, pad + h]])

    # rotation about the canvas centre
    ang = np.deg2rad(rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg))
    cx, cy = cw / 2.0, ch / 2.0
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    rotated = np.empty_like(src)
    for i, (px, py) in enumerate(src):
        dx, dy = px - cx, py - cy
        rotated[i] = [cx + dx * cos_a - dy * sin_a,
                      cy + dx * sin_a + dy * cos_a]

    # mild keystone: jitter each corner by up to max_keystone of the plate size
    jitter = rng.uniform(-cfg.max_keystone, cfg.max_keystone, size=(4, 2))
    jitter[:, 0] *= w
    jitter[:, 1] *= h
    dst = (rotated + jitter).astype(np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        canvas, m, (cw, ch), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    ys, xs = np.nonzero(warped[:, :, 3])
    if ys.size == 0:
        return plate
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Image.fromarray(warped[y0:y1, x0:x1])


# --------------------------------------------------------------------------- #
# Appearance (composited image, box-preserving)
# --------------------------------------------------------------------------- #

def _brightness_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = float(rng.uniform(0.6, 1.4))   # contrast
    beta = float(rng.uniform(-55, 55))     # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _shadow(img: np.ndarray, box_px: tuple[int, int, int, int],
            rng: np.random.Generator) -> np.ndarray:
    """Soft, partial shadow falling across (and around) the plate."""
    h, w = img.shape[:2]
    bx, by, bw, bh = box_px
    mask = np.zeros((h, w), dtype=np.float32)
    # a random quadrilateral roughly over the plate region
    cx = bx + rng.uniform(0.1, 0.9) * bw
    cy = by + rng.uniform(0.1, 0.9) * bh
    half_w = max(8.0, bw * float(rng.uniform(0.4, 1.0)))
    half_h = max(8.0, bh * float(rng.uniform(0.4, 1.2)))
    pts = np.array([
        [cx - half_w, cy - half_h], [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h], [cx - half_w, cy + half_h],
    ], dtype=np.float32)
    pts += rng.uniform(-0.2, 0.2, size=pts.shape) * np.array([bw, bh])
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 1.0)
    k = max(3, (int(min(bw, bh) * 0.3) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    factor = float(rng.uniform(0.35, 0.65))
    mask = mask[:, :, None] * (1.0 - factor)
    return np.clip(img.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(1, 5)) * 2 + 1  # odd kernel 3..9
    return cv2.GaussianBlur(img, (k, k), 0)


def _dirt(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Salt-and-pepper speckle simulating dirt / sensor noise."""
    h, w = img.shape[:2]
    amount = float(rng.uniform(0.005, 0.03))
    n = int(amount * h * w)
    if n <= 0:
        return img
    out = img.copy()
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    salt = rng.random(n) < 0.5
    out[ys[salt], xs[salt]] = 255
    out[ys[~salt], xs[~salt]] = 0
    return out


def _occlusion(img: np.ndarray, box_px: tuple[int, int, int, int],
               rng: np.random.Generator, max_frac: float) -> np.ndarray:
    """Cover <= ``max_frac`` of the plate with a solid random rectangle."""
    bx, by, bw, bh = box_px
    if bw <= 1 or bh <= 1:
        return img
    frac = float(rng.uniform(0.1, max_frac))
    # pick width fraction, derive height fraction so wf*hf == frac (clamped)
    wf = float(rng.uniform(0.2, 1.0))
    hf = min(1.0, frac / wf)
    ow = max(1, int(bw * wf))
    oh = max(1, int(bh * hf))
    ox = bx + int(rng.integers(0, max(1, bw - ow + 1)))
    oy = by + int(rng.integers(0, max(1, bh - oh + 1)))
    colour = tuple(int(c) for c in rng.integers(0, 256, size=3))
    out = img.copy()
    cv2.rectangle(out, (ox, oy), (ox + ow, oy + oh), colour, thickness=-1)
    return out


def augment_image(image: Image.Image, bbox_norm: list[float],
                  rng: np.random.Generator, cfg: AugmentConfig) -> Image.Image:
    """Apply box-preserving appearance augmentations to a composited RGB image.

    ``bbox_norm`` is ``[x, y, w, h]`` normalised 0–1 (top-left origin) — used to
    target shadow/occlusion at the plate. The box itself is never modified.
    """
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_norm
    box_px = (int(x * w), int(y * h), max(1, int(bw * w)), max(1, int(bh * h)))

    if _hit(rng, cfg, cfg.brightness_contrast):
        img = _brightness_contrast(img, rng)
    if _hit(rng, cfg, cfg.shadow):
        img = _shadow(img, box_px, rng)
    if _hit(rng, cfg, cfg.blur):
        img = _blur(img, rng)
    if _hit(rng, cfg, cfg.dirt):
        img = _dirt(img, rng)
    if _hit(rng, cfg, cfg.occlusion):
        img = _occlusion(img, box_px, rng, cfg.occlusion_max_frac)

    return Image.fromarray(img, "RGB")
