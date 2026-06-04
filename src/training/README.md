# Phase 2 — Training the detector

Trains the compact single-class licence-plate detector on the Phase-1 synthetic
dataset and produces `models/best.pt` for evaluation (Phase 3) and TFLite export
(Phase 4).

## Architecture decision — YOLOv8n

We use **YOLOv8n** (nano) from [Ultralytics](https://github.com/ultralytics/ultralytics).

| Reason | Detail |
|---|---|
| Small footprint | ~6 MB fp32 weights → keeps the eventual on-device `.tflite` small (Phase 4) |
| Export path | Native TFLite export with **NMS baked into the graph** — directly relevant to the README §2 contract and Phase 4 |
| Compute | Trains on a free Kaggle P100 / Colab T4 in reasonable time on ~5 k images |
| Single class | One class (`plate`); YOLO single-class detection is well-trodden |
| Licence | Apache-2.0 — matches this repo |

This decision is also recorded in the root `README.md` → *Open decisions* and in
the docstring of `train.py`.

## Files

| File | Purpose |
|---|---|
| `train.py` | Train YOLOv8n; copy best checkpoint to `models/best.pt`; print val recall/precision/mAP |
| `sanity_check.py` | CPU inference on a few val images; prints + saves annotated outputs |
| `../../notebooks/train.ipynb` | Self-contained Kaggle/Colab notebook wrapping the above |

## Install

Training deps are an opt-in extra so the base install stays light:

```bash
pip install -e ".[train]"     # pulls ultralytics (+ torch)
```

## Run

From the **repo root** (so relative dataset paths resolve):

```bash
# 1. generate data (Phase 1) if not already present
python -m plate_redactor.generator.generate \
    --n 5000 --seed 42 --backgrounds data/backgrounds --out data/synthetic

# 2. train
python src/training/train.py --data data/synthetic/data.yaml --epochs 100

# 3. sanity-check the result on CPU
python src/training/sanity_check.py
```

`train.py` copies the best checkpoint to `models/best.pt` and writes all run
artefacts (loss curves, per-epoch metrics) to `models/runs/<name>/`.

### data.yaml path resolution

Ultralytics resolves a **relative** `path:` in `data.yaml` against its global
`datasets_dir` setting (`~/datasets`), *not* the current directory — a common
"dataset not found" trap. `train.py` sidesteps this by rewriting a sibling
`data.resolved.yaml` with an **absolute** `path:` and training on that. So the
generator's default relative `path:` is fine; just run from the repo root.

## Hyperparameters

Defaults chosen for synthetic-only data with a recall bias (§2: *over-redact
rather than miss*). All are CLI flags on `train.py`.

| Param | Default | Notes |
|---|---|---|
| `imgsz` | **320** | Matches the §2 contract candidate; re-evaluate at 416 (see below) |
| `epochs` | 100 | Sufficient for synthetic-only; early-stop `patience=50` guards plateaus |
| `batch` | 16 | Fits P100/T4 VRAM with YOLOv8n at 320 px |
| `seed` | 0 | Reproducibility |
| `conf` | 0.25 | Low → recall bias (final val pass + sanity check). Inference only, not a train param |
| `iou` (NMS) | 0.45 | Standard; tighten in Phase 3 if duplicate boxes appear |
| `patience` | 50 | Early-stop if val mAP stalls for 50 epochs |
| `save_period` | 10 | Checkpoint every 10 epochs (Colab sessions can drop) |

### Input resolution — chosen: **320**

We default to **320×320** for the released checkpoint: it matches the smaller §2
contract candidate, keeps the model fast on a single CPU still, and is plenty for
plates that occupy 2–30 % of the frame (per the Phase-1 generator). 416 is worth
a comparison run if Phase 3 shows recall lagging on small/distant plates — flip
`--imgsz 416` and compare val recall. The root README *Open decisions* table is
updated to reflect 320 as the working choice (revisit in Phase 3/4).

## What "done" looks like (Phase 2 DoD)

- `python src/training/train.py` completes on Kaggle/Colab with no error.
- `models/best.pt` exists and loads cleanly with `ultralytics`.
- Val **recall ≥ 0.80** on the synthetic val split (modest by design — hard eval
  is Phase 3).
- Loss curves in `models/runs/<name>/results.png` show no divergence/early plateau.
- `sanity_check.py` shows detections on **≥ 4 of 5** val images.
- `models/` and notebook outputs stay gitignored — only scripts are committed.

## Compute / rotating between Kaggle and Colab

See `../../notebooks/train.ipynb` (Markdown cell) and §4 of the work order:

- **Kaggle**: 30 GPU-hrs/week free (P100); add the dataset as a Kaggle Dataset
  input; pull `best.pt` from `/kaggle/working/`.
- **Colab Free**: T4, max 12-hr sessions, no guarantee; mount Drive and rely on
  `save_period=10` so a dropped session loses ≤ 10 epochs.
- Rotate between the two when one rate-limits. **Record which platform produced
  the released checkpoint** in `WORKING_STATE.md`.
