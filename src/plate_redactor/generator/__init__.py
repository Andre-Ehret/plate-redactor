"""Synthetic licence-plate data generator (Phase 1).

Composites artificial German (DE-format) licence plates onto background images
and writes image + bounding-box label pairs in YOLO format. No real plate photos
are ever used — see the project README for the data-provenance (GDPR) rationale.

Public API:

    from plate_redactor.generator import render_plate, generate

    plate_img, text = render_plate()          # one synthetic plate
    summary = generate.run(n=100, seed=42)    # a whole dataset

CLI:

    python -m plate_redactor.generator.generate --n 5000 --seed 42 \
        --backgrounds <dir> --out data/synthetic
"""

from .augment import AugmentConfig
from .plate import random_plate_text, render_plate

__all__ = ["AugmentConfig", "render_plate", "random_plate_text"]
