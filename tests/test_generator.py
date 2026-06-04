"""Smoke test for the synthetic data generator (Phase 1).

Generates a small dataset with fallback (no real backgrounds) and checks the
output files exist, the YOLO labels are well-formed, and every coordinate is
within [0, 1]. Must run in well under 30 s on CPU.
"""

from __future__ import annotations

from pathlib import Path

from plate_redactor.generator import generate, render_plate
from plate_redactor.generator.augment import AugmentConfig

N = 10


def _yolo_lines(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(v) for v in line.split()])
    return rows


def test_render_plate_returns_image_and_text():
    img, text = render_plate()
    assert img.mode == "RGBA"
    assert img.width > img.height  # plates are wide
    assert text and text == text.upper()


def test_generate_smoke(tmp_path):
    out = tmp_path / "synthetic"
    summary = generate.run(n=N, seed=42, out=out, backgrounds=None,
                           val_split=0.2, show_progress=False)

    assert summary["images"] == N
    assert summary["train"] + summary["val"] == N
    assert summary["val"] == 2  # 20 % of 10

    # data.yaml present
    assert (out / "data.yaml").is_file()
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "plate" in yaml_text and "train" in yaml_text and "val" in yaml_text

    # exactly N image/label pairs across both splits
    images = list((out / "images").rglob("*.jpg"))
    labels = list((out / "labels").rglob("*.txt"))
    assert len(images) == N
    assert len(labels) == N

    # every image has a matching label, and every label is valid YOLO
    for img_path in images:
        split = img_path.parent.name
        label = out / "labels" / split / f"{img_path.stem}.txt"
        assert label.is_file(), f"missing label for {img_path}"
        rows = _yolo_lines(label)
        assert len(rows) == 1, "expected exactly one plate per image"
        cls, cx, cy, w, h = rows[0]
        assert cls == 0.0
        for v in (cx, cy, w, h):
            assert 0.0 <= v <= 1.0, f"coord out of range in {label}: {v}"
        assert w > 0.0 and h > 0.0


def test_generate_is_reproducible(tmp_path):
    a = generate.run(n=4, seed=7, out=tmp_path / "a", backgrounds=None,
                     show_progress=False)
    b = generate.run(n=4, seed=7, out=tmp_path / "b", backgrounds=None,
                     show_progress=False)
    la = ((tmp_path / "a" / "labels").rglob("*.txt"))
    for label_a in la:
        rel = label_a.relative_to(tmp_path / "a")
        label_b = (tmp_path / "b") / rel
        assert label_b.read_text() == label_a.read_text()
    assert a["images"] == b["images"] == 4


def test_disabled_augmentation_still_writes(tmp_path):
    out = tmp_path / "plain"
    summary = generate.run(n=3, seed=1, out=out, backgrounds=None,
                           cfg=AugmentConfig.disabled(), show_progress=False)
    assert summary["images"] == 3
    assert len(list((out / "images").rglob("*.jpg"))) == 3
