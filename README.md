# plate-redactor

> **Working title — final repo name is TBD** (see [Open decisions](#open-decisions)).

An open-source, **on-device licence-plate detector** shipped as a `.tflite`
model. A consuming app (e.g. clatchr) runs it on a captured photo **before
upload** to black out licence plates, closing a GDPR gap that neither Android's
ML Kit nor iOS's Vision framework covers out of the box.

This is a **detector, not OCR** — for redaction only *where* a plate is matters,
not *what it says*. Detection is also more robust on skewed, dirty or partly
occluded plates where OCR fails, and the plate text is never read or stored.

**Status: Phase 1 (synthetic data generator).** Nothing is trained yet, but the
generator that produces the training data is in place — see
[Synthetic data generator](#synthetic-data-generator) below. This repo defines
the project structure, the interface contract, and the data pipeline.

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
| Detector architecture (compact single-class) | **YOLOv8n** (Ultralytics, Apache-2.0) — Phase 2 |
| Input resolution — `320` vs `416` | **320** (working choice, Phase 2; revisit 416 in Phase 3 if recall lags) |
| Quantisation — `int8` vs `fp16` | **TBD** (Phase 4) |
| NMS location — in-model vs post-processing | **TBD** (Phase 4) |
| Recall threshold / acceptance metric | **Overall recall ≥ 0.90** @ IoU-match 0.5, operating conf 0.25 (hard gate; also portrait & landscape ≥ 0.90). Per-subset recall ≥ 0.85 soft; precision ≥ 0.60 informational — Phase 3 |
| Target file size & inference latency | **TBD** |
| Share of synthetic vs (consenting) real data for fine-tuning | **TBD** |

---

## Project structure

```
src/plate_redactor/
  generator/        # Phase 1 — synthetic data generator
    plate.py        #   render a DE-format plate (RGBA)
    backgrounds.py  #   background pool (disk or synthetic fallback)
    compositor.py   #   paste plate onto background, record bbox
    augment.py      #   augmentation pipeline (AugmentConfig)
    writer.py       #   YOLO label + dataset-layout writer
    generate.py     #   CLI entry point
    fonts.py        #   font resolution (no font binary committed)
src/training/       # Phase 2 — train / sanity-check the detector
  train.py          #   train YOLOv8n -> models/best.pt
  sanity_check.py   #   CPU inference on a few val images
  README.md         #   architecture decision + hyperparameter notes
src/eval/           # Phase 3 — recall-focused evaluation
  metrics.py        #   IoU / matching / precision-recall / AP (torch-free)
  common.py         #   subsets, dataset iteration, inference, aggregation
  generate_test_set.py  # build the hard-case test set (data/test_hard/)
  evaluate.py       #   results table + eval_results.json + failures + report
  threshold_sweep.py    # conf sweep -> threshold_sweep.png + recommended conf
notebooks/          # Kaggle/Colab notebooks
  train.ipynb       #   Phase 2 — training
  evaluate.ipynb    #   Phase 3 — evaluation
data/               # gitignored — synthetic images & labels, backgrounds
models/             # gitignored — checkpoints, .tflite artefacts
tests/              # generator smoke test (Phase 1); contract tests (Phase 5)
README.md
LICENSE             # Apache 2.0
NOTICE              # font licensing rationale
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

Baseline dependencies (`numpy`, `pillow`, `opencv-python`, `tqdm`) cover the
synthetic data generator (Phase 1). Training and TFLite-export dependencies are
added in later phases to keep the base install light.

---

## Synthetic data generator

Composites artificial **German (DE-format)** licence plates onto background
images and writes image + bounding-box label pairs ready for training. **No real
plate photos are ever used** (see [Data provenance](#data-provenance-gdpr)).

```bash
# 5000 images from your own backgrounds, reproducible via --seed
python -m plate_redactor.generator.generate \
    --n 5000 --seed 42 --backgrounds data/backgrounds --out data/synthetic

# installed console-script alias (same thing)
plate-redactor-generate --n 5000 --seed 42 --backgrounds data/backgrounds

# no backgrounds → synthesised solid-colour fallbacks (quick smoke run)
python -m plate_redactor.generator.generate --n 50 --seed 42
```

> The generator lives in the `plate_redactor.generator` package (the project's
> `src/`-layout package), so the module path is
> `python -m plate_redactor.generator.generate`.

**Flags:** `--n` (image count), `--seed` (global seed), `--backgrounds` (image
dir; omit for fallbacks), `--out` (default `data/synthetic`), `--val-split`
(default `0.1`), `--font` (path to a plate font — see [Fonts](#fonts)).

### Output layout (YOLO format)

```
data/synthetic/
  images/train/  images/val/
  labels/train/  labels/val/
  data.yaml
```

Each image has a one-line label `0 <cx> <cy> <w> <h>` — class `0` (`plate`),
**centre** coordinates normalised `0–1`. (The runtime interface contract above
uses top-left `[x, y, w, h]`; the compositor records that form and the label
writer converts to YOLO centre form.) `data.yaml` is the standard YOLO dataset
descriptor.

### Augmentations

Each is independently toggled and seeded for reproducibility (see
`AugmentConfig`): perspective/rotation (±20° tilt + mild keystone), brightness/
contrast, partial shadow, Gaussian blur, dirt/salt-and-pepper noise, partial
occlusion (≤40 % of the plate — the box is kept so the detector must still fire),
and scale variation (plate occupies 2 %–30 % of the image area).

### Backgrounds

Supply a directory via `--backgrounds`; subfolders are searched. Backgrounds are
**never committed** (`data/` is git-ignored). Source freely-licensed vehicle /
street / parking-lot photos from e.g. [Unsplash](https://unsplash.com),
[Pexels](https://pexels.com), or CC0 Flickr (queries like `"car street"`,
`"parking lot"`). Mix portrait and landscape. With no directory, the generator
synthesises solid-colour backgrounds in both orientations so it runs with zero
setup (real photos make a far better detector).

### Fonts

To keep the public repo clean of murky-licensed binaries, **no plate font is
bundled**. The renderer resolves a font at runtime (explicit `--font` →
`PLATE_REDACTOR_FONT` → `generator/assets/fonts/*.ttf` → a common system font →
Pillow's built-in default). Glyph fidelity is irrelevant to a *detector*; for
authentic-looking plates drop a licensed FE-Schrift into
`src/plate_redactor/generator/assets/fonts/` (git-ignored) or pass `--font`. See
[`NOTICE`](NOTICE) for the full rationale and licensing notes.

---

## Build phases

One phase per work order; test briefly after each.

- **Phase 0 — Repo setup**: structure, Apache-2.0 license, README with the
  interface contract.
- **Phase 1 — Synthetic data generator**: DE-plate compositing +
  augmentation; writes images **and** bounding-box labels. See
  [Synthetic data generator](#synthetic-data-generator).
- **Phase 2 — Train the detector**: compact single-class architecture (YOLOv8n).
- **Phase 3 — Evaluation**: recall-focused; a dedicated hard-case test set
  (tilt / dirt / occlusion / shadow / tiny / mixed + portrait/landscape) with
  over-redaction preferred. Build it and evaluate with:

  ```bash
  pip install -e ".[eval]"
  python src/eval/generate_test_set.py            # -> data/test_hard/
  python src/eval/evaluate.py --model models/best.pt
  python src/eval/threshold_sweep.py --model models/best.pt
  ```

  Hard gate: overall (and portrait/landscape) recall **≥ 0.90**. See
  `notebooks/evaluate.ipynb` for a one-click Kaggle/Colab run.
- **Phase 4 — TFLite export + quantisation**: export, quantise, check file size
  and single-image inference time.
- **Phase 5 — Integration contract test**: run sample stills through the
  exported model and verify the output format matches this contract.

---

## License

[Apache License 2.0](LICENSE). Applies to code, the generator scripts, and future
model weights.
