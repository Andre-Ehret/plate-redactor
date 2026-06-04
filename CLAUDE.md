# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`plate-redactor` (working title) is an **open-source, on-device licence-plate
detector** shipped as a `.tflite` model. A consuming app (clatchr) runs it on a
captured photo **before upload** to black out licence plates — closing a GDPR
gap that neither Android ML Kit nor iOS Vision covers.

It is a **detector, not OCR**: for redaction only *where* a plate is matters,
not *what it says*. The plate text is never read or stored.

Read `README.md` for the full **interface contract** (input/output shape,
quantisation, versioning, data provenance) and `WORKING_STATE.md` for current
progress. The original German work order lives in the project knowledge as
`plate-redactor_Arbeitsauftrag.md`.

## Non-negotiable principles

1. **Recall over Precision.** A missed plate is a data leak; an over-redacted
   region is harmless. When in doubt, over-redact — tune toward recall.
2. **Synthetic-first / GDPR.** The data generator is public; **real plate photos
   are never committed** to this repo. Training uses synthetic data so no
   personal data sits in the training set.
3. **Stable contract.** The model is a swappable internal. Keep the `.tflite`
   signature and the app-side `redactImage` contract in `README.md` stable;
   change internals freely, the contract rarely.
4. **No large artefacts in Git.** Checkpoints and `.tflite` files ship as
   versioned release assets, never committed. `data/` and `models/` are
   git-ignored.

## Project layout

```
src/plate_redactor/   # generator, training, export (Phases 1–4)
data/                 # gitignored — synthetic images & labels
models/               # gitignored — checkpoints, .tflite artefacts
tests/                # contract tests (Phase 5)
```

## Environment

- Python **3.10–3.13** (3.12 recommended; TensorFlow does not yet support 3.14,
  which Phases 2/4 need). Local interpreter: `python3.12`.
- Setup: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
- Baseline deps (`numpy`, `pillow`, `opencv-python`) cover the generator;
  training/TFLite deps are added in their phases to keep the base install light.
- Tests: `pytest`.

## Working style

- **One phase per work order.** Phases are sequenced 0→5 (see `README.md` →
  "Build phases"). Do the requested phase only; don't pull work forward.
- After finishing a unit of work, **update `WORKING_STATE.md`** (and the relevant
  README "Open decisions" entries when a TBD gets resolved).
- Commit/push only when asked. Branch off `main` for new work.
- When a deferred decision (resolution, quantisation, NMS location, recall
  threshold, file-size/latency targets) is made, record it in the README "Open
  decisions" table instead of leaving it implicit in code.
