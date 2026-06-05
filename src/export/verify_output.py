"""Phase 4 — verify the exported ``.tflite`` against the §2 interface contract.

Loads ``models/plate-detector.tflite`` with the raw TFLite runtime (exactly what
``react-native-fast-tflite`` uses — no Ultralytics), runs one forward pass on a
blank image, and reports the real I/O signature:

  * input tensor: shape, dtype, expected value range (``[0,1]`` = ``pixel/255``);
  * output tensor(s): shape, dtype, value ranges;
  * **whether NMS is baked into the graph or must be applied in post-processing**
    — derived from the output signature and stated explicitly in stdout;
  * a best-effort parse of the output into ``[{box:[x,y,w,h], score}]`` with a
    check that the box coordinates fall in ``[0, 1]``.

It then writes/updates ``models/export_report.md`` — the committed Phase-4
deliverable — folding in ``export_meta.json`` (sizes/settings),
``benchmark_results.json`` (latency) and ``eval_results.json`` (recall) when
those have been produced.

Usage (after ``export.py``)::

    python src/export/verify_output.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import
import tflite_io  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the exported TFLite I/O against the §2 contract (Phase 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "plate-detector.tflite"),
                   help="Exported TFLite model to verify.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence floor when listing example detections.")
    p.add_argument("--meta", default=str(REPO_ROOT / "models" / "export_meta.json"),
                   help="export.py metadata (sizes/settings) to fold into the report.")
    p.add_argument("--benchmark", default=str(REPO_ROOT / "models" / "benchmark_results.json"),
                   help="benchmark.py results to fold into the report (if present).")
    p.add_argument("--eval", default=str(REPO_ROOT / "models" / "eval_results.json"),
                   help="evaluate.py results to fold into the report (if present).")
    p.add_argument("--report", default=str(REPO_ROOT / "models" / "export_report.md"),
                   help="Where to write the Phase-4 export report.")
    return p.parse_args(argv)


def _shape(shape) -> list[int]:
    """Tensor shape as plain ints (runtimes may return numpy int dtypes)."""
    return [int(d) for d in shape]


def _squeeze_dims(shape) -> list[int]:
    return [int(d) for d in shape if int(d) != 1]


def classify_nms(output_details: list[dict], outputs: list[np.ndarray]) -> tuple[str, str]:
    """Return ``(location, explanation)`` — ``"in-model"`` / ``"post-processing"`` / ``"unknown"``.

    Heuristics for a YOLOv8 single-class detector at imgsz 320:
      * ≥ 4 output tensors → TFLite_Detection_PostProcess style (boxes / classes /
        scores / count) → NMS baked in;
      * one ``(N, 6)`` tensor with small ``N`` (≤ ~300) → ``[x1,y1,x2,y2,conf,cls]``
        post-NMS detections → NMS baked in;
      * one ``(4+nc, anchors)`` tensor with a large anchor dim (≈ 2100 for 320)
        → raw predictions → NMS in post-processing.
    """
    if len(output_details) >= 4:
        return ("in-model",
                "Multiple output tensors (TFLite_Detection_PostProcess-style: "
                "boxes / classes / scores / count) — NMS is baked into the graph.")
    dims = _squeeze_dims(outputs[0].shape)
    if len(dims) == 2:
        big, small = max(dims), min(dims)
        if small in (5, 6) and big >= 1000:
            return ("post-processing",
                    f"Single raw tensor {list(outputs[0].shape)} (~{big} anchors × "
                    f"{small} [xywh + class score(s)]) — NMS must run in "
                    "app-side post-processing.")
        if small == 6 and big <= 300:
            return ("in-model",
                    f"Single tensor {list(outputs[0].shape)} ({big} detections × 6 "
                    "[x1,y1,x2,y2,conf,cls]) — NMS is baked into the graph.")
    return ("unknown",
            f"Unrecognised output signature {[list(o.shape) for o in outputs]} — "
            "inspect manually and set the NMS note by hand.")


def _example_detections(outputs, output_details, nms_loc, conf):
    """Best-effort parse to ``[{box:[x,y,w,h], score}]`` (top few) for a sanity print.

    Returns ``(detections, coords_in_unit_range)``. Boxes are top-left ``[x,y,w,h]``
    normalised — the §2 output form.
    """
    out = outputs[0]
    dims = _squeeze_dims(out.shape)
    rows: list[tuple[list[float], float]] = []

    if nms_loc == "in-model" and len(output_details) == 1 and len(dims) == 2:
        arr = out.reshape(dims)
        if arr.shape[0] == 6:  # (6, N) → (N, 6)
            arr = arr.T
        for x1, y1, x2, y2, score, _cls in arr:
            if score >= conf:
                rows.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             float(score)))
    elif nms_loc == "in-model" and len(output_details) >= 4:
        # TFLite_Detection_PostProcess: boxes [1,N,4] (ymin,xmin,ymax,xmax) + scores [1,N]
        boxes = next((o for o in outputs if _squeeze_dims(o.shape)[-1:] == [4]), None)
        scores = next((o for o in outputs
                       if len(_squeeze_dims(o.shape)) == 1 and o.dtype == np.float32), None)
        if boxes is not None and scores is not None:
            b = boxes.reshape(-1, 4)
            s = scores.reshape(-1)
            for (ymin, xmin, ymax, xmax), score in zip(b, s):
                if score >= conf:
                    rows.append(([float(xmin), float(ymin),
                                  float(xmax - xmin), float(ymax - ymin)], float(score)))
    else:  # raw predictions: (4+nc, anchors) or (anchors, 4+nc); single class → 5
        arr = out.reshape(dims)
        if arr.shape[0] in (5, 6):  # (channels, anchors) → (anchors, channels)
            arr = arr.T
        for row in arr:
            cx, cy, w, h = row[:4]
            score = float(row[4:].max())
            if score >= conf:
                rows.append(([float(cx - w / 2), float(cy - h / 2),
                              float(w), float(h)], score))

    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:5]
    in_unit = all(0.0 <= v <= 1.0 for box, _ in rows for v in box) if rows else None
    return [{"box": box, "score": round(sc, 4)} for box, sc in rows], in_unit


def _load_json(path: str) -> dict | None:
    p = Path(path)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _write_report(path: Path, *, model: str, backend: str, in_det: dict,
                  out_details: list[dict], nms_loc: str, nms_expl: str,
                  meta: dict | None, bench: dict | None, ev: dict | None) -> None:
    q = (meta or {}).get("quantisation", "fp16")
    fp32 = (meta or {}).get("fp32_size_mb")
    exp = (meta or {}).get("exported_size_mb")
    if exp is None:  # no export_meta.json — read the actual artefact size
        mp = Path(model)
        if mp.is_file():
            exp = round(mp.stat().st_size / 1e6, 3)
    in_shape = _shape(in_det["shape"])
    nms_md = ("baked **into the model graph** — the app consumes detections directly"
              if nms_loc == "in-model" else
              "applied in **app-side post-processing** — the model emits raw "
              "predictions the app must NMS" if nms_loc == "post-processing" else
              "**undetermined from the signature — set by hand after inspection**")

    def latency_line() -> str:
        if not bench:
            return "_Run `benchmark.py` to fill this in._"
        return (f"mean **{bench['mean_ms']:.1f} ms**, median {bench['median_ms']:.1f} ms, "
                f"p95 **{bench['p95_ms']:.1f} ms** over {bench['n_images']} images "
                f"on CPU ({bench.get('backend', '?')}, {bench.get('num_threads', '?')} threads) "
                f"— target p95 ≤ 500 ms: **{'PASS' if bench['p95_ms'] <= 500 else 'FAIL'}**.")

    def recall_line() -> str:
        if not ev:
            return ("_Run `evaluate.py --model models/plate-detector.tflite` and paste "
                    "the exported-model recall here, alongside the Phase-3 fp32 baseline._")
        r = ev.get("overall", {}).get("recall")
        passed = ev.get("gates", {}).get("passed")
        return (f"exported-model overall recall **{r:.3f}** (hard gate ≥ 0.90: "
                f"**{'PASS' if passed else 'FAIL'}**). Fill in the Phase-3 fp32 baseline "
                "recall and the delta (must not drop > 2 pp).")

    lines = [
        "# Phase 4 — TFLite export report",
        "",
        "Auto-drafted by `verify_output.py`. **Confirm the NMS note and paste the "
        "fp32-baseline recall by hand before committing.**",
        "",
        "## Quantisation",
        "",
        f"- **Format chosen:** `{q}`. fp16 is the primary choice — it halves the "
        "model size with negligible detection-accuracy loss and needs no "
        "calibration data. int8 is the documented fallback (used only if fp16 "
        "misses the 8 MB target or recall drops > 2 pp).",
        "",
        "## File size",
        "",
        f"- fp32 baseline (`best.pt`): **{fp32 if fp32 is not None else '?'} MB**",
        f"- exported (`{q}` `.tflite`): **{exp if exp is not None else '?'} MB** "
        f"(target ≤ 8 MB: {'PASS' if (exp or 0) <= 8 and exp is not None else '?'})",
        "",
        "## I/O signature (raw interpreter)",
        "",
        f"- Backend used: `{backend}`",
        f"- **Input:** shape `{in_shape}` (NHWC), dtype `{np.dtype(in_det['dtype']).name}`, "
        "value range `[0, 1]` (pixel / 255).",
        "- **Output tensor(s):**",
    ]
    for d in out_details:
        sc, zp = d["quantization"]
        qnote = f", quant (scale={sc}, zero={zp})" if sc else ""
        lines.append(f"  - `{d['name']}` shape `{_shape(d['shape'])}`, "
                     f"dtype `{np.dtype(d['dtype']).name}`{qnote}")
    lines += [
        "",
        "## NMS location",
        "",
        f"- NMS is {nms_md}.",
        f"- Detector: {nms_expl}",
        "",
        "## Recall before/after quantisation",
        "",
        f"- {recall_line()}",
        "",
        "## CPU latency (single still)",
        "",
        f"- {latency_line()}",
        "",
        "## Contract notes",
        "",
        "- **Input normalisation:** pixel values divided by 255 → `[0, 1]`.",
        "- **Recommended `conf` threshold:** 0.25 (recall-biased operating point "
        "from the Phase-3 sweep — confirm against `eval_results.json` "
        "`threshold_sweep.recommended_conf`).",
        f"- **Versioned artefact:** `plate-detector-v{(meta or {}).get('version', '0.1.0')}"
        ".tflite` — upload as a GitHub Release asset (not committed).",
        f"- Verified model: `{model}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interp, backend = tflite_io.load_interpreter(args.model)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    imgsz = tflite_io.input_size(in_det)

    print(f"[verify] model={args.model}  backend={backend}")
    print(f"\n[input]  name={in_det['name']!r}")
    print(f"         shape={_shape(in_det['shape'])} dtype={np.dtype(in_det['dtype']).name} "
          f"quant={in_det['quantization']}")
    print(f"         expected value range: [0, 1] (pixel / 255), imgsz={imgsz}")

    interp.set_tensor(in_det["index"], tflite_io.blank_input(in_det, imgsz))
    interp.invoke()
    outputs = [tflite_io.dequantize_output(interp.get_tensor(d["index"]), d)
               for d in out_details]

    print(f"\n[output] {len(out_details)} tensor(s):")
    for d, o in zip(out_details, outputs):
        print(f"         name={d['name']!r} shape={_shape(d['shape'])} "
              f"dtype={np.dtype(d['dtype']).name} quant={d['quantization']}")
        print(f"           dequantised range: [{float(o.min()):.4f}, {float(o.max()):.4f}]")

    nms_loc, nms_expl = classify_nms(out_details, outputs)
    print(f"\n[nms]    {nms_loc.upper()}: {nms_expl}")

    dets, in_unit = _example_detections(outputs, out_details, nms_loc, args.conf)
    print(f"\n[parse]  example detections (conf ≥ {args.conf}, blank image so likely "
          f"none): {dets}")
    if in_unit is None:
        print("         (no detections on the blank image — run benchmark/eval on real "
              "images to exercise the parse)")
    else:
        print(f"         box coords in [0, 1]: {'YES' if in_unit else 'NO — investigate'}")
    print("\n[parse]  contract form: [{ 'box': [x, y, w, h], 'score': s }], "
          "x/y top-left, all normalised 0–1.")

    _write_report(
        Path(args.report), model=args.model, backend=backend, in_det=in_det,
        out_details=out_details, nms_loc=nms_loc, nms_expl=nms_expl,
        meta=_load_json(args.meta), bench=_load_json(args.benchmark),
        ev=_load_json(args.eval))
    print(f"\n[verify] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
