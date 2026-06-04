"""Phase 2 — local CPU sanity check for a trained checkpoint.

Loads ``models/best.pt``, runs inference on a few random validation images, and
prints/saves the detections so you can eyeball that the model actually fires on
plates before moving to Phase 3 (proper evaluation) or Phase 4 (export).

No GPU required — a handful of images on CPU is fast.

Usage (from the repo root, after downloading the checkpoint)::

    python src/training/sanity_check.py
    python src/training/sanity_check.py --weights models/best.pt --n 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU sanity check for a trained plate detector (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models" / "best.pt"),
        help="Path to the trained checkpoint.",
    )
    p.add_argument(
        "--val-dir",
        default=str(REPO_ROOT / "data" / "synthetic" / "images" / "val"),
        help="Directory of validation images to sample from.",
    )
    p.add_argument("--n", type=int, default=5, help="Number of random images to test.")
    p.add_argument("--imgsz", type=int, default=320, help="Inference input size (match training).")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (recall bias).")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random image sample.")
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "models" / "sanity_output"),
        help="Directory for annotated output images (gitignored).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed. Install the training extras:\n"
            '    pip install -e ".[train]"'
        )

    weights = Path(args.weights)
    if not weights.is_file():
        sys.exit(f"Checkpoint not found: {weights}\nTrain first or pass --weights.")

    val_dir = Path(args.val_dir)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    if not images:
        sys.exit(f"No images found under {val_dir}. Run Phase 1 to populate the val split.")

    rng = random.Random(args.seed)
    sample = rng.sample(images, k=min(args.n, len(images)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    print(f"Loaded {weights} — testing {len(sample)} image(s) at imgsz={args.imgsz}, "
          f"conf={args.conf}, iou={args.iou}\n")

    images_with_dets = 0
    for img_path in sample:
        result = model(str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        boxes = result.boxes
        n_det = 0 if boxes is None else len(boxes)
        if n_det:
            images_with_dets += 1
        print(f"{img_path.name}: {n_det} detection(s)")
        for i in range(n_det):
            conf = float(boxes.conf[i])
            x, y, w, h = (float(v) for v in boxes.xywhn[i])  # normalised centre form
            print(f"    score={conf:.3f}  box(cx,cy,w,h)=({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")

        annotated = result.plot()  # BGR ndarray with boxes drawn
        out_path = out_dir / f"{img_path.stem}_annotated.jpg"
        _save_bgr(annotated, out_path)

    print(f"\nAnnotated images -> {out_dir}")
    print(f"Detections on {images_with_dets}/{len(sample)} images "
          f"(DoD: >= 4/5 expected on a trained model).")
    return 0


def _save_bgr(bgr_array, out_path: Path) -> None:
    """Save an OpenCV-style BGR ndarray as RGB without a hard cv2 dependency."""
    from PIL import Image

    rgb = bgr_array[..., ::-1]  # BGR -> RGB
    Image.fromarray(rgb).save(out_path, "JPEG", quality=92)


if __name__ == "__main__":
    raise SystemExit(main())
