# plate-redactor

> **Working title — final repo name is TBD** (see [Open decisions](#open-decisions)).

An open-source, **on-device licence-plate detector** shipped as a `.tflite`
model. A consuming app (e.g. clatchr) runs it on a captured photo **before
upload** to black out licence plates, closing a GDPR gap that neither Android's
ML Kit nor iOS's Vision framework covers out of the box.

This is a **detector, not OCR** — for redaction only *where* a plate is matters,
not *what it says*. Detection is also more robust on skewed, dirty or partly
occluded plates where OCR fails, and the plate text is never read or stored.

**Status: Phase 0 (repo setup).** Nothing is trained yet. This repo currently
only defines the project structure and the interface contract below.

---

## Interface contract

This is the stable contract between this repo and the consuming app. Either side
can evolve its internals independently as long as this contract holds.

### Model artefact

- **File:** `plate-detector.tflite`
- **Task:** single-class object detection — one class, `plate`
- **Output primitive:** bounding boxes

### Input

- **Format:** RGB image
- **Shape:** fixed **square** side length — **`320 × 320` or `416 × 416`, TBD**
  (see [Open decisions](#open-decisions))
- **Normalisation:** pixel values scaled to **`[0, 1]`** (i.e. `pixel / 255.0`).
  Final value re-confirmed at export once quantisation is fixed.
- The app is responsible for resizing/letterboxing the captured still to the
  model input size and for mapping boxes back to original image coordinates.

### Output

A list of detections, each:

```jsonc
{
  "box":   [x, y, w, h],  // normalised to [0, 1] relative to the input image
  "score": 0.0            // confidence in [0, 1]
}
```

- `x, y` = top-left corner; `w, h` = width/height. All normalised `0–1`.
- **NMS (non-max suppression): TBD** — to be decided whether it runs *inside*
  the model graph or in app-side post-processing. This README will state the
  final choice explicitly once Phase 4 (export) fixes it. Until then, assume
  post-processing NMS may be required.

### Quantisation

- **int8 or fp16 — TBD** (see [Open decisions](#open-decisions)). Goal: small
  file size + fast single-image CPU inference. Marked TBD until Phase 4.

### Evaluation bias — Recall over Precision

**A missed plate is a data leak; an over-redacted region is harmless.**
The model and its acceptance thresholds are deliberately calibrated toward
**over-redaction**. When in doubt, redact more, not less.

### Versioning

- The model ships as a **versioned release artefact**; the app **pins a specific
  version** (committed file or release asset).
- **No large artefacts (checkpoints, `.tflite`) are committed to Git** — they
  live in releases / external storage. A stable contract means the model's
  internals can be swapped without changing app logic.

### Data provenance (GDPR)

- The **synthetic data generator is public**; **real plate photos are never
  committed** to this repo. Training is **synthetic-first** (artificial plates
  composited onto vehicle backgrounds), so no personal data sits in the training
  set — making the result GDPR-clean and safely publishable.
- Not legal advice — for commercial use, have a DPO/lawyer review.

### App-side signature (reference — lives in the app repo, not here)

```ts
redactImage(uri: string): Promise<{ uri: string; facesFound: number; platesFound: number }>
```

Runs on a **single captured still before upload**, not on live camera frames.
Latency and model size are therefore non-critical — a slightly heavier model is
acceptable.

---

## Open decisions

Deferred design choices, tracked explicitly. All marked **TBD** until resolved in
the noted phase.

| Decision | Status |
|---|---|
| Final repo / model name | **TBD** |
| Detector architecture (compact single-class) | **TBD** (Phase 2) |
| Input resolution — `320` vs `416` | **TBD** |
| Quantisation — `int8` vs `fp16` | **TBD** (Phase 4) |
| NMS location — in-model vs post-processing | **TBD** (Phase 4) |
| Recall threshold / acceptance metric | **TBD** (Phase 3) |
| Target file size & inference latency | **TBD** |
| Share of synthetic vs (consenting) real data for fine-tuning | **TBD** |

---

## Project structure

```
src/        # generator, training, export (Phases 1–4)
data/       # gitignored — synthetic images & labels
models/     # gitignored — checkpoints, .tflite artefacts
tests/      # contract tests (Phase 5)
README.md
LICENSE     # Apache 2.0
pyproject.toml
.gitignore
```

`data/` and `models/` are git-ignored (only a `.gitkeep` is tracked). Real plate
photos and large artefacts must never be committed.

---

## Getting started

Requires **Python 3.10–3.13** (3.12 recommended; TensorFlow does not yet support
3.14, which later phases will need).

```bash
# create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install the package (editable) with baseline generator deps
pip install -e .

# optional: dev tooling (pytest)
pip install -e ".[dev]"
```

Smoke test:

```bash
python -c "import plate_redactor; print(plate_redactor.__version__)"
```

Baseline dependencies (`numpy`, `pillow`, `opencv-python`) cover the synthetic
data generator (Phase 1). Training and TFLite-export dependencies are added in
later phases to keep the base install light.

---

## Build phases

One phase per work order; test briefly after each.

- **Phase 0 — Repo setup** *(this phase)*: structure, Apache-2.0 license, README
  with the interface contract.
- **Phase 1 — Synthetic data generator**: DE-plate compositing + augmentation;
  writes images **and** bounding-box labels.
- **Phase 2 — Train the detector**: compact single-class architecture.
- **Phase 3 — Evaluation**: recall-focused; skewed / dirty / occluded / shadow
  test cases; over-redaction preferred.
- **Phase 4 — TFLite export + quantisation**: export, quantise, check file size
  and single-image inference time.
- **Phase 5 — Integration contract test**: run sample stills through the
  exported model and verify the output format matches this contract.

---

## License

[Apache License 2.0](LICENSE). Applies to code, the generator scripts, and future
model weights.
