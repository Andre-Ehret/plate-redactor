"""Phase 3 — detection metrics (pure Python, no torch / ultralytics).

Everything here is deliberately framework-free so it is fast to unit-test and
carries the matching/AP logic that the acceptance gates depend on. Boxes are
``(x1, y1, x2, y2)`` in **normalised** ``[0, 1]`` coordinates throughout (IoU is
scale-invariant, so as long as predictions and ground truth share the same space
the numbers are correct).

Matching follows the usual single-class object-detection convention:

  * predictions are greedily assigned to ground-truth boxes in descending score
    order; a prediction is a **true positive** if it overlaps an as-yet-unmatched
    GT box at IoU >= ``iou_thr``, otherwise a **false positive**;
  * any GT box left unmatched is a **false negative** (a missed plate — the
    failure mode this whole project is calibrated against).
"""

from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)
Pred = tuple[float, Box]                  # (score, box)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    """Convert a YOLO centre box ``(cx, cy, w, h)`` to corner form ``(x1,y1,x2,y2)``."""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection-over-union of two corner-form boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def greedy_match(gt_boxes: Sequence[Box], preds: Iterable[Pred],
                 iou_thr: float = 0.5) -> tuple[list[tuple[float, bool]], set[int]]:
    """Greedily match predictions to ground-truth boxes (descending score).

    Returns ``(scored, matched_gt)`` where ``scored`` is a list of
    ``(score, is_true_positive)`` in descending-score order (suitable for AP) and
    ``matched_gt`` is the set of GT indices that got a match.
    """
    preds_sorted = sorted(preds, key=lambda p: p[0], reverse=True)
    matched: set[int] = set()
    scored: list[tuple[float, bool]] = []
    for score, box in preds_sorted:
        best_iou, best_j = iou_thr, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou_xyxy(box, gt)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched.add(best_j)
            scored.append((score, True))
        else:
            scored.append((score, False))
    return scored, matched


def counts(gt_boxes: Sequence[Box], preds: Sequence[Pred],
           iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` for one image at a fixed (already-applied) conf."""
    scored, matched = greedy_match(gt_boxes, preds, iou_thr)
    tp = sum(1 for _, is_tp in scored if is_tp)
    fp = len(scored) - tp
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Standard precision / recall / F1 from aggregate counts."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def average_precision(scored: Sequence[tuple[float, bool]], total_gt: int) -> float:
    """All-points (VOC2010+/COCO-style) AP from scored detections.

    ``scored`` is ``(score, is_tp)`` over *all* detections (no conf filter) for
    the class; ``total_gt`` is the number of ground-truth boxes. Returns AP at the
    IoU threshold the matching used (we only evaluate AP@0.5 here).
    """
    if total_gt == 0:
        return 0.0
    order = sorted(scored, key=lambda x: x[0], reverse=True)
    tp = fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for _, is_tp in order:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / total_gt)
        precisions.append(tp / (tp + fp))

    # Monotonically decreasing precision envelope, then integrate over recall.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap
