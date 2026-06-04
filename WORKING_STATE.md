# Working state

Living snapshot of where the project is. Update this after each unit of work.

_Last updated: 2026-06-04 — Phase 2 scaffolding complete (training not yet run on GPU)._

## Current phase

**Phase 2 — Train detector → scripts + notebook ready; GPU run pending.** The
training pipeline is written and locally validated (argument parsing, notebook
JSON, byte-compile). The actual ~100-epoch train must run on Kaggle/Colab GPU to
produce `models/best.pt`; record which platform produced the released checkpoint
here once done. Next up after that: Phase 3 (recall-focused evaluation).

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Repo setup (structure, license, README contract) | ✅ Done |
| 1 | Synthetic data generator (DE-plate compositing + augmentation, image + bbox labels) | ✅ Done |
| 2 | Train compact single-class detector | 🟡 Scaffolded (GPU run pending) |
| 3 | Evaluation (recall-focused; skewed/dirty/occluded/shadow) | ⬜ Not started |
| 4 | TFLite export + quantisation | ⬜ Not started |
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
- Local validation: both scripts byte-compile and `--help` cleanly; notebook is
  valid JSON. **Not yet run on GPU** — `models/best.pt`, recall ≥ 0.80, and the
  loss-curve / sanity-check DoD items require a Kaggle/Colab run.

### Decisions made in Phase 2

- **Architecture: YOLOv8n** (Ultralytics, Apache-2.0) — small footprint, native
  TFLite export with NMS baked in (eases Phase 4), trains on free P100/T4.
- **Input resolution: 320** as the working choice (matches the smaller §2
  contract candidate); compare 416 in Phase 3 if small-plate recall lags.
- README "Open decisions" updated for both.

## Open decisions (still TBD)

See README → "Open decisions". Resolved in Ph2: detector architecture (YOLOv8n)
and input resolution (320, provisional). Outstanding: final repo name,
quantisation int8 vs fp16 (Ph4), NMS in-model vs post-processing (Ph4), recall
threshold/metric (Ph3), target file size & latency, synthetic-vs-real fine-tuning
share.

## Notes for the next session

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
