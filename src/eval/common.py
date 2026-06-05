"""Phase 3 — shared dataset + inference helpers for the eval scripts.

Kept separate from :mod:`metrics` (which is torch-free and unit-tested) because
this module lazily pulls in ``ultralytics``. The eval scripts are run by path
(``python src/eval/evaluate.py``), so ``src/eval`` is on ``sys.path`` and these
are imported as plain sibling modules — mirroring ``src/training``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import metrics

# Subsets that count toward the headline 600-image OVERALL recall gate.
HARD_SUBSETS = ("tilt", "dirty", "occluded", "shadow", "small", "mixed")
# Orientation check — reported separately, each has its own 0.90 recall gate.
ORIENT_SUBSETS = ("portrait", "landscape")
ALL_SUBSETS = HARD_SUBSETS + ORIENT_SUBSETS

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ImageSample:
    """One test image: its path and ground-truth boxes (normalised xyxy)."""

    subset: str
    image_path: Path
    label_path: Path
    gt_boxes: list[metrics.Box] = field(default_factory=list)
    # All detections at the inference conf floor: (score, xyxy). Filled lazily.
    preds: list[metrics.Pred] = field(default_factory=list)


def load_gt(label_path: Path) -> list[metrics.Box]:
    """Read a YOLO label file into a list of normalised ``(x1,y1,x2,y2)`` boxes."""
    if not label_path.is_file():
        return []
    boxes: list[metrics.Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, cx, cy, w, h = parts
        boxes.append(metrics.yolo_to_xyxy(float(cx), float(cy), float(w), float(h)))
    return boxes


def discover_subsets(data_root: Path) -> list[str]:
    """Return the subset names that actually exist under ``<root>/images/``."""
    img_root = data_root / "images"
    present = [s for s in ALL_SUBSETS if (img_root / s).is_dir()]
    return present


def iter_samples(data_root: Path, subset: str) -> list[ImageSample]:
    """Build :class:`ImageSample` objects (GT loaded) for one subset."""
    img_dir = data_root / "images" / subset
    lbl_dir = data_root / "labels" / subset
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in _IMG_EXTS)
    samples: list[ImageSample] = []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        samples.append(ImageSample(subset, img, lbl, gt_boxes=load_gt(lbl)))
    return samples


def load_model(weights: str | Path):
    """Load an Ultralytics YOLO checkpoint (lazy import so --help works bare)."""
    weights = Path(weights)
    if not weights.is_file():
        raise SystemExit(
            f"Checkpoint not found: {weights}\n"
            "Phase 3 needs the Phase 2 checkpoint. Download the release asset to "
            "models/best.pt, or pass --model."
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is not installed. Install the eval extras:\n"
            '    pip install -e ".[eval]"'
        )
    return YOLO(str(weights))


def aggregate_at_conf(samples: list[ImageSample], conf: float,
                      iou_match: float = 0.5, with_ap: bool = False) -> dict:
    """Aggregate detection metrics over ``samples`` at one operating ``conf``.

    Predictions below ``conf`` are dropped (recall-biased operating point); AP, if
    requested, is threshold-independent and uses every detection.
    """
    tp = fp = fn = total_gt = 0
    scored_all: list[tuple[float, bool]] = []
    fn_samples: list[ImageSample] = []
    for s in samples:
        kept = [p for p in s.preds if p[0] >= conf]
        t, f, n = metrics.counts(s.gt_boxes, kept, iou_match)
        tp += t
        fp += f
        fn += n
        total_gt += len(s.gt_boxes)
        if n > 0:
            fn_samples.append(s)
        if with_ap:
            scored, _ = metrics.greedy_match(s.gt_boxes, s.preds, iou_match)
            scored_all.extend(scored)
    precision, recall, f1 = metrics.precision_recall_f1(tp, fp, fn)
    out = {
        "images": len(samples), "gt": total_gt,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": recall, "precision": precision, "f1": f1,
        "fn_samples": fn_samples,
    }
    if with_ap:
        out["map50"] = metrics.average_precision(scored_all, total_gt)
    return out


def run_inference(model, samples: list[ImageSample], imgsz: int,
                  conf_floor: float, iou: float) -> None:
    """Populate ``sample.preds`` for each sample (in place).

    Inference runs at a low ``conf_floor`` so a single pass yields every detection
    needed for both the operating-point table *and* the threshold sweep / AP —
    callers re-threshold in Python rather than re-running the model.
    """
    for s in samples:
        result = model(str(s.image_path), imgsz=imgsz, conf=conf_floor,
                       iou=iou, verbose=False)[0]
        boxes = result.boxes
        preds: list[metrics.Pred] = []
        if boxes is not None and len(boxes):
            xyxyn = boxes.xyxyn.tolist()   # normalised corner form
            confs = boxes.conf.tolist()
            for (x1, y1, x2, y2), sc in zip(xyxyn, confs):
                preds.append((float(sc), (float(x1), float(y1),
                                          float(x2), float(y2))))
        s.preds = preds
