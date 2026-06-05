# Working state

Living snapshot of where the project is. Update this after each unit of work.

_Last updated: 2026-06-05 — Phase 4 export scaffolding complete (export / verify /
benchmark scripts + shared TFLite-runtime helper + notebook + doc/contract
updates); awaiting a GPU/checkpoint run to produce the real `.tflite`, sizes,
latency and recall-after-quantisation numbers._

## Current phase

**Phase 4 — TFLite export + quantisation → SCAFFOLDING DONE; run pending.** The
export script (`export.py`, fp16 primary + int8 fallback, NMS baked in by default,
versioned artefact), the raw-interpreter signature/NMS verifier (`verify_output.py`,
auto-drafts `export_report.md`), the CPU latency benchmark (`benchmark.py`), the
shared runtime-agnostic helper (`tflite_io.py`) and `notebooks/export.ipynb` are
in place; all four scripts compile and their `--help`/argparse paths work. The
recall regression reuses `src/eval/evaluate.py` unchanged (Ultralytics loads the
`.tflite` through its TFLite runtime → apples-to-apples with the Phase-3 fp32
run). **Next:** run `notebooks/export.ipynb` where `models/best.pt` lives
(Kaggle/Colab) to produce `plate-detector-v0.1.0.tflite`, then record real file
size (≤ 8 MB), CPU p95 (≤ 500 ms), recall-after-quantisation (≥ 0.90) and the
confirmed NMS location here and in `models/export_report.md`. **Gate caveat:** the
Phase-3 ≥ 0.90 recall gate has not yet been cleared on a real run (Phase 3 was
also scaffolding) — clear it before treating the export as final.

> **Caveat — synthetic val is saturated.** Val recall/precision came out at
> 1.00 / 1.00 (mAP@0.5 0.995). That's a red flag, not a victory: the synthetic
> val split is too close to the train distribution, so these numbers overstate
> real-world performance. Phase 3 (real, consenting photos) is where we find the
> true recall; if the gap is large we likely need to harden the generator
> (more variance/difficulty) and/or a small real fine-tune set (see Open
> decisions: synthetic-vs-real share).

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Repo setup (structure, license, README contract) | ✅ Done |
| 1 | Synthetic data generator (DE-plate compositing + augmentation, image + bbox labels) | ✅ Done |
| 2 | Train compact single-class detector | ✅ Done (Kaggle T4; recall 1.00 on synthetic val — see caveat) |
| 3 | Evaluation (recall-focused; skewed/dirty/occluded/shadow) | 🟡 Scaffolding done; GPU/checkpoint run pending |
| 4 | TFLite export + quantisation | 🟡 Scaffolding done; GPU/checkpoint run pending |
| 5 | Integration contract test | ⬜ Not started |

## Done in Phase 0

- Project layout: `src/plate_redactor/`, `tests/`, git-ignored `data/` & `models/`.
- `pyproject.toml` — src-layout package, `requires-python >=3.10,<3.14`, baseline
  deps `numpy` / `pillow` / `opencv-python`, `[dev]` extra (`pytest`).
- `.gitignore` — `data/`, `models/`, weights/checkpoints, real-plate dirs, venv,
  caches, editor/OS files.
- `LICENSE` — Apache 2.0 (copyright line filled).
- `README.md` — full interface contract + Open-decisions (TBD) table + build phases.
- `CLAUDE.md` — repo guidance and principles.
- Smoke test passed in a fresh `python3.12` venv: `pip install -e .` clean,
  `import plate_redactor` works (numpy 2.4.6, cv2 4.13.0, PIL 12.2.0).

## Done in Phase 1

- `plate_redactor.generator` package: `plate.py` (DE-plate RGBA renderer),
  `backgrounds.py` (disk pool + synthetic portrait/landscape fallback),
  `compositor.py` (geometry-augment + scale + paste, records normalised bbox),
  `augment.py` (`AugmentConfig`: perspective/rotation, brightness/contrast,
  shadow, blur, dirt, occlusion ≤40 %, scale 2–30 %; each toggled + seeded),
  `writer.py` (YOLO labels + `images|labels/train|val` layout + `data.yaml`),
  `generate.py` (CLI + `run()` API, tqdm progress), `fonts.py` (font resolution).
- CLI: `python -m plate_redactor.generator.generate --n … --seed … --backgrounds
  … --out …` (also console-script `plate-redactor-generate`). Reproducible via
  per-image `default_rng([seed, i])`.
- Smoke test `tests/test_generator.py` (4 tests, ~2 s): files exist, YOLO valid,
  coords in [0,1], reproducibility, disabled-augmentation path.
- Visual spot-check (20 samples): plates legible, augmentations realistic, boxes
  align tightly incl. rotated case; portrait + landscape both covered; plate area
  0.02–0.29.
- Deps: added `tqdm`; bumped Pillow to `>=10.1` (`load_default(size=)`). Added
  `NOTICE` (font-licensing rationale — no font binary committed).

### Decisions made in Phase 1

- **Label format: YOLO** (`0 cx cy w h`, normalised, class always 0) +
  `data.yaml` descriptor. Compositor records top-left `[x,y,w,h]` (matching the
  README §2 contract); writer converts to YOLO centre form.
- **Package path:** generator lives at `src/plate_redactor/generator/` (project
  `src`-layout), so CLI is `python -m plate_redactor.generator.generate` — the
  issue's literal `generator.generate` adapted to the package per CLAUDE.md.
- **Font:** none bundled (licence-clean stance); runtime resolution with a
  Pillow-default fallback. FE-Schrift is opt-in via `--font` / `assets/fonts/`.

## Done in Phase 2 (scaffolding)

- `src/training/train.py` — YOLOv8n trainer wrapping `ultralytics`. CLI flags:
  `--data --epochs(100) --imgsz(320) --batch(16) --project(models/runs) --name
  --seed --weights --device --patience(50) --save-period(10) --out --conf(0.25)
  --iou(0.45)`. Lazy ultralytics import (so `--help` works without it), resolves
  the data.yaml `path:` to absolute (avoids Ultralytics' `~/datasets` trap),
  copies best checkpoint to `models/best.pt`, then prints a recall-biased val
  pass (recall/precision/mAP).
- `src/training/sanity_check.py` — CPU inference on N random val images; prints
  detection counts + scores + boxes, saves annotated images to
  `models/sanity_output/`.
- `src/training/README.md` — architecture rationale + hyperparameter table.
- `notebooks/train.ipynb` — self-contained Kaggle/Colab notebook (installs deps,
  clones repo, regenerates or attaches dataset, trains, inspects curves,
  persists `best.pt` to Kaggle output / Drive). Includes the Kaggle↔Colab
  rotation guidance (§4).
- `pyproject.toml` — added `[train]` extra (`ultralytics>=8.1`); kept out of base
  install (pulls torch). `.gitignore` — added `.ipynb_checkpoints/`.
- **GPU run (released checkpoint):** Kaggle, **Tesla T4**, 100 epochs in 0.68 h,
  imgsz 320, batch 16, seed 0, 5 000 synthetic images (fallback backgrounds).
  Result: **recall 1.00, precision 1.00, mAP@0.5 0.995, mAP@.5:.95 0.995** on the
  synthetic val split (500 imgs). `best.pt` = 6.2 MB, 3.0 M params, 8.1 GFLOPs.
  Sanity check: 5/5 images, confidence 0.96–0.98. **Treat the perfect metrics as
  saturation, not real-world performance** (see caveat at top).
- Bug found + fixed during the run: `resolve_data_yaml()` doubled the dataset path
  for relative `path:` (Ultralytics "images not found"); now uses the yaml's own
  dir. Notebook clone cell hardened to force-update to latest `main`. Both pushed.
- Note: the released run used **synthetic fallback backgrounds** (solid colours),
  not real vehicle photos — another reason the val is easy. For a stronger model,
  re-run with real backgrounds (`--backgrounds` / attached Kaggle dataset).

### Decisions made in Phase 2

- **Architecture: YOLOv8n** (Ultralytics, Apache-2.0) — small footprint, native
  TFLite export with NMS baked in (eases Phase 4), trains on free P100/T4.
- **Input resolution: 320** as the working choice (matches the smaller §2
  contract candidate); compare 416 in Phase 3 if small-plate recall lags.
- README "Open decisions" updated for both.

## Done in Phase 3 (scaffolding)

- `src/eval/metrics.py` — framework-free detection metrics: IoU, YOLO→xyxy,
  greedy score-ordered matching, precision/recall/F1, all-points AP@0.5. No
  torch/ultralytics, so it is fast to unit-test (the gate logic lives here).
- `src/eval/common.py` — subset constants (`HARD_SUBSETS` = tilt/dirty/occluded/
  shadow/small/mixed → the 600-image OVERALL gate; `ORIENT_SUBSETS` =
  portrait/landscape), dataset iteration + GT loading, lazy-ultralytics model
  loading + single low-conf inference pass, and `aggregate_at_conf()` (re-applies
  the operating conf in Python so one inference pass feeds both the table and the
  sweep).
- `src/eval/generate_test_set.py` — builds `data/test_hard/` (gitignored): six
  hard subsets (100 each, augmentation **forced** at prob 1.0 per subset) + 50
  portrait / 50 landscape with orientation-forced backgrounds. Seed 999 (never a
  training seed). Reuses the Phase-1 generator building blocks.
- `src/eval/evaluate.py` — per-subset + overall + orientation table, hard gates
  (overall/portrait/landscape recall ≥ 0.90), `models/eval_results.json`,
  annotated false-negative images (`models/eval_failures/`, GT red / pred green),
  and an auto-drafted `models/eval_report.md` (heuristic FN categorisation).
  `--strict` exits non-zero on a failed gate.
- `src/eval/threshold_sweep.py` — conf 0.05→0.50, plots recall/precision/F1,
  recommends the highest-recall conf with precision ≥ 0.60, saves
  `models/threshold_sweep.png` and merges the recommended conf + curve into
  `eval_results.json`.
- `notebooks/evaluate.ipynb` — one-click Kaggle/Colab run (install → clone →
  attach `best.pt` → generate test set → evaluate → sweep → inspect).
- `tests/test_eval_metrics.py` — 13 unit tests for metrics + aggregation (IoU,
  greedy matching incl. the missed-plate FN case, PRF, AP edge cases, conf
  thresholding). Full suite: **17 passed**.
- `pyproject.toml` — added `[eval]` extra (`ultralytics` + `matplotlib`).
  `.gitignore` — un-ignore `models/eval_report.md` (the only committed eval
  output). Small backward-compatible generator change: `AugmentConfig.
  occlusion_min_frac` (default 0.10) so the `occluded` subset can force 20–40 %.

### Decisions made in Phase 3

- **Acceptance metric / recall threshold** (was TBD): overall recall **≥ 0.90**
  at IoU-match 0.5, operating conf 0.25; portrait & landscape recall ≥ 0.90 are
  also hard gates; per-subset recall ≥ 0.85 is soft (documented if missed);
  precision ≥ 0.60 is informational. Recorded in README "Open decisions".
- **Single low-conf inference pass + Python re-thresholding** for both the
  operating-point table and the sweep (no per-threshold re-inference); mAP@0.5 is
  threshold-independent (uses all detections).

## Done in Phase 4 (scaffolding)

- `src/export/export.py` — Ultralytics TFLite export. CLI: `--model --imgsz(320)
  --quantisation(fp16|int8) --data --out --version(0.1.0) --nms/--no-nms --meta`.
  fp16 by default (`half=True`); int8 calibrates on `data.yaml`. Tries in-graph
  NMS (`nms=True`), auto-falls back to a raw export if the installed Ultralytics
  rejects it (loud warning). Copies to `models/plate-detector.tflite` **and**
  `plate-detector-v0.1.0.tflite` (release artefact), prints size vs the 8 MB
  target, writes `models/export_meta.json` (fp32-vs-exported sizes + settings).
- `src/export/tflite_io.py` — runtime-agnostic interpreter loader (tries
  `tflite-runtime` → `ai-edge-litert` → `tensorflow.lite`) + dtype-aware
  pre/post-processing (handles fp16 float I/O and int8 quantised I/O). Torch/
  ultralytics-free: this is the same raw surface the app's react-native-fast-tflite
  uses. Imported as a sibling module (like `eval/common.py`).
- `src/export/verify_output.py` — loads the `.tflite` raw, prints input (shape/
  dtype/range `[0,1]`) and output tensors, classifies **NMS location** (in-model
  vs post-processing) from the output signature, best-effort parses to
  `[{box:[x,y,w,h], score}]` and checks coords ∈ `[0,1]`, then auto-drafts
  `models/export_report.md` (folds in `export_meta.json` + `benchmark_results.json`
  + `eval_results.json` when present).
- `src/export/benchmark.py` — times `invoke()` only over N (default 20) random val
  images on **CPU** (warm-up discarded), reports mean/median/**p95** vs the 500 ms
  target, writes `models/benchmark_results.json`.
- `notebooks/export.ipynb` — one-click Kaggle/Colab run (install → clone → fetch
  `best.pt` → export → verify → benchmark → recall regression → report → save
  artefact). `src/export/README.md` — workflow + quantisation/NMS notes.
- `pyproject.toml` — added `[export]` extra (`ultralytics` + `tensorflow`).
  `.gitignore` — un-ignore `models/export_report.md` (the only committed Phase-4
  output; `.tflite`, `export_meta.json`, `benchmark_results.json` stay ignored).
  README §2 contract + "Open decisions" updated (fp16, in-model NMS, ≤ 8 MB /
  p95 ≤ 500 ms).
- **Recall regression reuses `evaluate.py` as-is** — `python src/eval/evaluate.py
  --model models/plate-detector.tflite` works because Ultralytics loads a `.tflite`
  through its TFLite runtime, giving the same pre/post-processing as the fp32 run.
  No change to the eval code was needed.

### Decisions made in Phase 4

- **Quantisation: fp16** (was TBD) — halves size, no calibration, negligible
  accuracy loss; int8 is the documented fallback if fp16 misses the 8 MB target or
  drops recall > 2 pp. Recorded in README "Open decisions".
- **NMS location: in-model** (was TBD) — baked into the graph at export (`--nms`)
  so the app needs no suppression step; `verify_output.py` confirms the actual
  per-build signature and the report records it. Recorded in README "Open decisions".
- **Size/latency targets** (were TBD): file ≤ 8 MB, CPU p95 ≤ 500 ms on a single
  still. Recorded in README "Open decisions".

## Open decisions (still TBD)

See README → "Open decisions". Resolved: detector architecture (YOLOv8n, Ph2),
input resolution (320, provisional, Ph2), recall threshold/acceptance metric
(overall recall ≥ 0.90, Ph3), quantisation (fp16, Ph4), NMS location (in-model,
Ph4), target file size & latency (≤ 8 MB / p95 ≤ 500 ms, Ph4). Outstanding: final
repo name, synthetic-vs-real fine-tuning share.

## Notes for the next session

- **Run the actual Phase 4 export** (after the Phase-3 recall gate is cleared on a
  real run). Open `notebooks/export.ipynb` where `models/best.pt` lives
  (Kaggle/Colab): it exports fp16, verifies the I/O + NMS, benchmarks CPU latency,
  and re-runs `evaluate.py` against the `.tflite`. **Record here and in
  `models/export_report.md`:** exported file size (≤ 8 MB), CPU p95 (≤ 500 ms),
  recall-after-quantisation (≥ 0.90, and the delta vs the fp32 baseline — must not
  drop > 2 pp), and the **confirmed NMS location** (verify_output prints it). If
  fp16 misses size or recall, re-export `--quantisation int8 --data
  data/synthetic/data.yaml` and document the int8 deltas. Upload
  `plate-detector-v0.1.0.tflite` as a GitHub Release asset (never commit it);
  commit only `models/export_report.md`.
- Local dry-run without a GPU/checkpoint: the scripts compile and `--help` works
  bare; full export needs `pip install -e ".[export]"` and `models/best.pt`.
  `verify_output.py` / `benchmark.py` only need a `.tflite` + a TFLite runtime, so
  you can smoke them against any exported model.
- **Run the actual Phase 3 evaluation.** Open `notebooks/evaluate.ipynb` on the
  platform where `models/best.pt` lives (Kaggle/Colab). It generates
  `data/test_hard/` (seed 999), runs `evaluate.py` + `threshold_sweep.py`, and
  shows the table, plot and failure images. Record the real recall here and the
  recommended operating conf. **If overall (or portrait/landscape) recall < 0.90,
  open a retrain issue** with targeted augmentation for the failing subset before
  Phase 4 — and remember the released checkpoint trained on *fallback* (solid)
  backgrounds, so a gap is expected. For a fair test, generate `test_hard` with
  the same `--backgrounds` source the model was trained on.
- Commit `models/eval_report.md` (the only un-ignored eval output); everything
  else under `models/` (json, png, failure images) stays gitignored.
- Local dry-run without a GPU/checkpoint: `python src/eval/generate_test_set.py
  --per-subset 5 --per-orientation 2 --out /tmp/th` then `pytest
  tests/test_eval_metrics.py`. Full inference needs `pip install -e ".[eval]"`.

### Phase 2 notes (kept for reference)

- **Run the actual Phase 2 training on GPU.** Open `notebooks/train.ipynb` on
  Kaggle (P100) or Colab (T4). First generate a real dataset with downloaded
  freely-licensed backgrounds (`--backgrounds <dir>`) for a usable detector — the
  synthetic-fallback backgrounds are only for smoke runs. Then run `train.py`
  (defaults: YOLOv8n, imgsz 320, 100 epochs, batch 16).
- After the run: confirm val **recall ≥ 0.80**, inspect `models/runs/<name>/
  results.png` for plateau/divergence, run `sanity_check.py` (≥ 4/5 images), and
  **record which platform produced the checkpoint** above. Ship `best.pt` as a
  release asset — never commit it.
- Local smoke option before burning GPU time: `pip install -e ".[train]"`,
  generate ~50 images, `python src/training/train.py --epochs 2 --batch 4` to
  exercise the full path on CPU.
- For authentic plates during dataset generation, drop a licensed FE-Schrift into
  `src/plate_redactor/generator/assets/fonts/` or pass `--font` (see `NOTICE`).
