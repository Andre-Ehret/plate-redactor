"""Phase 5 — canonical output post-processing for ``plate-detector.tflite``.

This is the **reference implementation** the app-side TypeScript
(``redaction.ts``) should mirror, turning the raw TFLite output tensor into the
§2 contract form::

    [{ "box": [x, y, w, h], "score": float }, ...]   # x,y top-left, all 0–1

NMS location for the v0.1.0 export
----------------------------------
**NMS is baked into the model graph** (Ultralytics exported with ``nms=True``),
so the app does *not* need to run suppression itself. The model emits a single
output tensor:

    shape  [1, 300, 6]   dtype float32
    row    [x1, y1, x2, y2, score, class]

  * ``300`` = the fixed maximum number of detections; unused slots are padded
    with (near-)zero rows, filtered out by the confidence threshold.
  * ``x1, y1, x2, y2`` = box corners in **xyxy** form, normalised ``0–1`` to the
    320×320 input. (Values may sit a hair outside ``[0, 1]`` when a plate touches
    the frame edge — :func:`parse_output` clamps them.)
  * ``score`` = confidence ``0–1``; ``class`` is always ``0`` (single class
    ``plate``).

:func:`parse_output` is therefore the only function the app strictly needs. The
standalone :func:`nms` below is kept as the canonical reference **in case a
future build ships raw predictions** (``export.py --no-nms``); it is a no-op for
the current in-model-NMS artefact.
"""

from __future__ import annotations


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two boxes in ``[x, y, w, h]`` (top-left) form."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[list[float]], scores: list[float],
        iou_threshold: float = 0.45) -> list[int]:
    """Greedy non-max suppression. Returns the kept indices, highest score first.

    Boxes are ``[x, y, w, h]`` (top-left, normalised). This is the canonical
    reference for the app side; **not needed for the v0.1.0 export** (NMS is
    in-model) but used if a ``--no-nms`` build ships raw predictions.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) <= iou_threshold]
    return keep


def parse_output(raw, conf_threshold: float = 0.25) -> list[dict]:
    """Parse the raw ``[1, 300, 6]`` output tensor into the §2 contract form.

    ``raw`` is the model's single output tensor (numpy array or nested list) of
    shape ``[1, 300, 6]`` (or ``[300, 6]``), each row ``[x1, y1, x2, y2, score,
    class]`` with xyxy coords normalised ``0–1``. Returns a list of
    ``{"box": [x, y, w, h], "score": float}`` — top-left ``x, y`` plus width and
    height, all clamped to ``[0, 1]`` — sorted by descending score, keeping only
    detections with ``score >= conf_threshold``.

    NMS is already applied in-model, so no suppression is done here.
    """
    rows = _as_rows(raw)
    dets: list[dict] = []
    for row in rows:
        x1, y1, x2, y2, score = (float(row[0]), float(row[1]), float(row[2]),
                                 float(row[3]), float(row[4]))
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        dets.append({"box": [x1, y1, w, h], "score": score})
    dets.sort(key=lambda d: d["score"], reverse=True)
    return dets


def _as_rows(raw):
    """Normalise ``raw`` ([1,300,6] / [300,6], numpy or list) to a list of rows."""
    try:  # numpy fast path
        import numpy as np
        arr = np.asarray(raw, dtype="float32")
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr.tolist()
    except ImportError:  # pragma: no cover — pure-python fallback
        if raw and isinstance(raw[0], (list, tuple)) and raw and not isinstance(raw[0][0], (list, tuple)):
            return list(raw)
        # assume [1, N, 6]
        return list(raw[0])
