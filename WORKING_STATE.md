# Working state

Living snapshot of where the project is. Update this after each unit of work.

_Last updated: 2026-06-04 — Phase 0 complete._

## Current phase

**Phase 0 — Repo setup → DONE.** Next up: Phase 1 (synthetic data generator).

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Repo setup (structure, license, README contract) | ✅ Done |
| 1 | Synthetic data generator (DE-plate compositing + augmentation, image + bbox labels) | ⬜ Not started |
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

## Open decisions (still TBD)

See README → "Open decisions". Outstanding: final repo name, detector
architecture (Ph2), input resolution 320 vs 416, quantisation int8 vs fp16 (Ph4),
NMS in-model vs post-processing (Ph4), recall threshold/metric (Ph3), target file
size & latency, synthetic-vs-real fine-tuning share.

## Notes for the next session

- Phase 1 = synthetic generator: composite artificial DE plates onto vehicle
  backgrounds; augment (angle/perspective, lighting/shadow, blur, dirt, partial
  occlusion, scale); write images **and** bounding-box labels in the training
  format. No real plate photos in the repo.
- Decide the bbox label format in Phase 1 and document it (the README output
  contract uses normalised `[x, y, w, h]`).
