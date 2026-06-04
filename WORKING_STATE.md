# Working state

Living snapshot of where the project is. Update this after each unit of work.

_Last updated: 2026-06-04 — Phase 1 complete._

## Current phase

**Phase 1 — Synthetic data generator → DONE.** Next up: Phase 2 (train the
compact single-class detector on Kaggle/Colab).

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Repo setup (structure, license, README contract) | ✅ Done |
| 1 | Synthetic data generator (DE-plate compositing + augmentation, image + bbox labels) | ✅ Done |
| 2 | Train compact single-class detector | ⬜ Not started |
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

## Open decisions (still TBD)

See README → "Open decisions". Outstanding: final repo name, detector
architecture (Ph2), input resolution 320 vs 416, quantisation int8 vs fp16 (Ph4),
NMS in-model vs post-processing (Ph4), recall threshold/metric (Ph3), target file
size & latency, synthetic-vs-real fine-tuning share.

## Notes for the next session

- Phase 2 = train a compact single-class detector. Generate a real dataset first
  (`python -m plate_redactor.generator.generate --n <N> --seed 42 --backgrounds
  <dir>`) with downloaded freely-licensed backgrounds; data is YOLO-format with
  `data.yaml`, so a YOLO-family model (e.g. a small YOLO) trains directly on it.
- Train on Kaggle/Colab; don't commit checkpoints (release assets / external).
- For authentic plates during dataset generation, drop a licensed FE-Schrift into
  `src/plate_redactor/generator/assets/fonts/` or pass `--font` (see `NOTICE`).
- Still TBD and relevant soon: detector architecture (Ph2) and input resolution
  (320 vs 416) — pick when wiring up training.
