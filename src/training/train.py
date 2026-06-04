"""Phase 2 — train the compact single-class licence-plate detector.

Architecture: **YOLOv8n** (nano) from Ultralytics.

Why YOLOv8n:
  * Single-class detection with a small footprint (~6 MB fp32 weights), which
    keeps the eventual on-device `.tflite` small (Phase 4).
  * Native TFLite export with NMS baked into the graph — directly relevant to
    the Phase 4 export and the README §2 interface contract.
  * Trains on a free Kaggle P100 / Colab T4 in reasonable time on ~5 k images.
  * Apache-2.0 licensed — matches this repo's licence.

This script is a thin, reproducible wrapper around ``ultralytics`` so the same
command runs locally (smoke), on Kaggle, and on Colab. It:

  1. loads the Phase-1 ``data.yaml`` (resolving its ``path:`` to absolute so
     Ultralytics finds the images regardless of CWD / datasets_dir),
  2. trains YOLOv8n with documented defaults (see ``src/training/README.md``),
  3. copies the best checkpoint to ``models/best.pt`` for Phases 3–4,
  4. runs a final validation pass and prints recall / precision / mAP.

Ultralytics writes loss curves, per-epoch metrics and plots to the run dir
(``<project>/<name>/``: ``results.csv``, ``results.png``, ``*_curve.png``).

Usage (run from the repo root)::

    python src/training/train.py --data data/synthetic/data.yaml --epochs 100

Install the training deps first::

    pip install -e ".[train]"      # pulls ultralytics (+ torch)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the YOLOv8n single-class plate detector (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=str(REPO_ROOT / "data" / "synthetic" / "data.yaml"),
        help="Path to the Phase-1 YOLO data.yaml.",
    )
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Square input side length. 320 matches the §2 contract candidate; "
        "try 416 and compare (see README).",
    )
    p.add_argument("--batch", type=int, default=16, help="Batch size.")
    p.add_argument(
        "--project",
        default=str(REPO_ROOT / "models" / "runs"),
        help="Output directory for Ultralytics runs (gitignored).",
    )
    p.add_argument("--name", default="plate_yolov8n", help="Run name (sub-dir of --project).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Base weights to fine-tune (COCO-pretrained YOLOv8n by default).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Ultralytics device string (e.g. '0', 'cpu'). Default: auto-select.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience (epochs without val improvement).",
    )
    p.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Checkpoint every N epochs (Colab sessions can drop — keep this low).",
    )
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "models" / "best.pt"),
        help="Where to copy the best checkpoint after training.",
    )
    # Inference thresholds — recall-biased per the §2 contract (over-redact).
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the final validation pass (recall bias).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for the final validation pass.",
    )
    return p.parse_args(argv)


def resolve_data_yaml(data_path: str | Path) -> Path:
    """Return a data.yaml whose ``path:`` is absolute.

    The Phase-1 writer records ``path:`` as given on the command line, which is
    often relative. Ultralytics resolves a relative ``path:`` against its global
    ``datasets_dir`` setting (``~/datasets`` by default), not the CWD — a classic
    "dataset not found" trap. We rewrite a sibling ``*.resolved.yaml`` with an
    absolute ``path:`` and hand that to the trainer instead.
    """
    import yaml  # provided transitively by ultralytics

    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        sys.exit(
            f"data.yaml not found: {data_path}\n"
            "Run Phase 1 first, e.g.:\n"
            "  python -m plate_redactor.generator.generate "
            "--n 5000 --seed 42 --backgrounds data/backgrounds --out data/synthetic"
        )

    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    # The Phase-1 generator always writes data.yaml at the dataset root, right
    # next to images/ and labels/. The recorded `path:` is whatever was passed
    # to --out (often relative to the *generation* CWD), which Ultralytics then
    # mis-joins (e.g. .../data/synthetic/data/synthetic/images/val). The only
    # always-correct base is the yaml's own directory, so use that and ignore
    # the recorded value.
    cfg["path"] = str(data_path.parent)

    resolved = data_path.with_suffix(".resolved.yaml")
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed. Install the training extras:\n"
            '    pip install -e ".[train]"'
        )

    data_yaml = resolve_data_yaml(args.data)
    print(f"[train] data: {data_yaml}")
    print(f"[train] imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} seed={args.seed}")

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        save_period=args.save_period,
        exist_ok=True,
    )

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        sys.exit(f"Expected best checkpoint not found: {best}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"[train] best checkpoint -> {out}")
    print(f"[train] run artefacts (results.csv, *_curve.png) -> {run_dir}")

    # Final recall-biased validation pass — recall is the headline metric (§2).
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    box = metrics.box
    print("\n[val] recall-biased metrics (conf={:.2f}, iou={:.2f}):".format(args.conf, args.iou))
    print(f"  recall    : {box.mr:.4f}   <-- DoD target >= 0.80")
    print(f"  precision : {box.mp:.4f}")
    print(f"  mAP@0.5   : {box.map50:.4f}")
    print(f"  mAP@.5:.95: {box.map:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
