# Phase 4 — TFLite export report

Auto-drafted by `verify_output.py`. **Confirm the NMS note and paste the fp32-baseline recall by hand before committing.**

## Quantisation

- **Format chosen:** `fp16`. fp16 is the primary choice — it halves the model size with negligible detection-accuracy loss and needs no calibration data. int8 is the documented fallback (used only if fp16 misses the 8 MB target or recall drops > 2 pp).

## File size

- fp32 baseline (`best.pt`): **≈ 6.2 MB** (Phase-2 YOLOv8n checkpoint; see `WORKING_STATE.md`)
- exported (`fp16` `.tflite`): **6.166 MB** (target ≤ 8 MB: PASS)

## I/O signature (raw interpreter)

- Backend used: `ai_edge_litert`
- **Input:** shape `[1, 320, 320, 3]` (NHWC), dtype `float32`, value range `[0, 1]` (pixel / 255).
- **Output tensor(s):**
  - `Identity` shape `[1, 300, 6]`, dtype `float32`

## NMS location

- NMS is baked **into the model graph** — the app consumes detections directly.
- Detector: Single tensor [1, 300, 6] (300 detections × 6 [x1,y1,x2,y2,conf,cls]) — NMS is baked into the graph.

## Recall before/after quantisation

- _Run `evaluate.py --model models/plate-detector.tflite` and paste the exported-model recall here, alongside the Phase-3 fp32 baseline._

## CPU latency (single still)

- _Run `benchmark.py` to fill this in._

## Contract notes

- **Input normalisation:** pixel values divided by 255 → `[0, 1]`.
- **Recommended `conf` threshold:** 0.25 (recall-biased operating point from the Phase-3 sweep — confirm against `eval_results.json` `threshold_sweep.recommended_conf`).
- **Versioned artefact:** `plate-detector-v0.1.0.tflite` — upload as a GitHub Release asset (not committed).
- Verified model: `models/plate-detector-v0.1.0.tflite`
