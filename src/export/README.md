# Phase 4 — TFLite export & quantisation

Exports the Phase-2 checkpoint (`models/best.pt`) to a quantised `.tflite` that
satisfies the README §2 interface contract and the size/latency targets below,
then verifies the I/O signature, benchmarks CPU latency, and re-checks recall so
quantisation hasn't regressed the detector.

## Targets

| Target | Value | Notes |
|---|---|---|
| File size | ≤ 8 MB | Comfortable bundled app asset |
| CPU inference (single still) | p95 ≤ 500 ms | One captured image, not live frames |
| Quantisation | **fp16** primary; int8 fallback | fp16 halves size, no calibration, negligible accuracy loss |

**Quantisation decision — fp16 first.** int8 needs a representative calibration
set and risks a recall regression; only fall back to it if fp16 exceeds 8 MB or
recall drops > 2 pp vs. the fp32 baseline. Document the outcome either way in
`models/export_report.md`.

## Files

| File | Purpose |
|---|---|
| `export.py` | `best.pt` → `plate-detector.tflite` (fp16/int8) + versioned release artefact; writes `export_meta.json` |
| `verify_output.py` | Raw-interpreter I/O check vs. §2; determines **NMS location**; drafts `export_report.md` |
| `benchmark.py` | CPU latency (mean / median / **p95**) on N val images → `benchmark_results.json` |
| `tflite_io.py` | Runtime-agnostic interpreter loader + dtype-aware pre/post-processing (shared) |
| `../../notebooks/export.ipynb` | Self-contained Kaggle/Colab notebook wrapping all of the above |

## Install

```bash
pip install -e ".[export]"     # ultralytics (export) + tensorflow (TFLite runtime)
```

`verify_output.py` / `benchmark.py` use the raw TFLite runtime — exactly what the
app's `react-native-fast-tflite` sees. They try `tflite-runtime`, then
`ai-edge-litert`, then `tensorflow.lite`, so any one of those satisfies them.

## Run (from the repo root, with `models/best.pt` present)

```bash
# 1. export (fp16, imgsz 320, NMS baked into the graph by default)
python src/export/export.py

# 2. verify the I/O signature + NMS location → drafts models/export_report.md
python src/export/verify_output.py

# 3. CPU latency benchmark → models/benchmark_results.json
python src/export/benchmark.py

# 4. recall regression — run the Phase-3 eval against the exported .tflite
python src/eval/evaluate.py --model models/plate-detector.tflite --data data/test_hard
```

The recall check reuses `src/eval/evaluate.py` unchanged: Ultralytics loads a
`.tflite` through its TFLite runtime, giving the **same** pre/post-processing as
the Phase-3 fp32 run for an apples-to-apples recall delta. The exported model
must still clear the **overall recall ≥ 0.90** hard gate; if it drops > 2 pp,
switch to int8 (`--quantisation int8 --data data/synthetic/data.yaml`) or revisit
the export settings.

## NMS location

`export.py` defaults to baking NMS into the graph (`--nms`); pass `--no-nms` for
raw predictions. `verify_output.py` inspects the **actual** exported signature and
records the authoritative note in `models/export_report.md` and the root README
§2 — the contract follows what the model really emits, not the request flag.

## Outputs

`models/export_report.md` is the one committed deliverable (everything else under
`models/` stays gitignored). The versioned `plate-detector-v0.1.0.tflite` ships as
a **GitHub Release asset** — never committed (no `.tflite` in Git).
