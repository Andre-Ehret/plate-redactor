# `plate-detector.tflite` — app integration contract

The single source of truth the **clatchr app team** references when implementing
`redactImage` in `src/lib/redaction.ts`. It mirrors the README §2 contract,
pinned to the verified **v0.1.0** export. The model is a swappable internal: this
document is the stable surface between the two repos.

> Verified against the real exported artefact by `tests/test_contract.py`
> (8 checks, CPU only). Output layout below is the actual `[1, 300, 6]` signature
> read off `plate-detector-v0.1.0.tflite`.

---

## Model artefact

- **Filename:** `plate-detector.tflite` (working copy) /
  `plate-detector-v0.1.0.tflite` (versioned release asset).
- **Task:** single-class object detection — one class, `plate` (class index `0`).
- **Download (GitHub Releases):**
  ```
  https://github.com/Andre-Ehret/plate-redactor/releases/download/v0.1.0/plate-detector-v0.1.0.tflite
  ```
  (repo name is still provisional — see README "Open decisions"). The app **pins
  a specific version** (bundled file or release asset). Do not track `latest`.
- **Size:** ≈ 6.2 MB (≤ 8 MB target). **Quantisation:** fp16.

## Input

| Property | Value |
|---|---|
| Tensor name | `images` |
| Shape | `[1, 320, 320, 3]` (NHWC) |
| dtype | `float32` |
| Colour | RGB |
| Normalisation | **`pixel / 255.0`** → `[0.0, 1.0]` |

The app is responsible for: resizing/letterboxing the captured still to
`320 × 320`, RGB ordering, the `/255` scale, and **mapping output boxes back to
the original image coordinates** (the model works in its own normalised
`0–1` space).

## Output

**NMS is baked into the model graph** — the app does **not** run suppression.

Single output tensor:

| Property | Value |
|---|---|
| Tensor name | `Identity` |
| Shape | `[1, 300, 6]` |
| dtype | `float32` |
| Row layout | `[x1, y1, x2, y2, score, class]` |

- `300` = fixed max detections; unused slots are (near-)zero rows → drop them by
  the confidence threshold.
- `x1, y1, x2, y2` = box corners in **xyxy**, normalised `0–1` to the 320×320
  input. Values may sit a hair outside `[0, 1]` at frame edges — **clamp to
  `[0, 1]`**.
- `score` ∈ `[0, 1]`; `class` is always `0`.

### Parsing to the §2 form

Convert each kept row to the contract form `{ box: [x, y, w, h], score }` with
**top-left** `x, y` and width/height, all normalised `0–1`:

```
x = clamp01(x1);  y = clamp01(y1)
w = clamp01(x2) - x;  h = clamp01(y2) - y
```

The canonical reference is `src/export/postprocess.py` (`parse_output`) — the
TypeScript should mirror it exactly.

### NMS

- **Not required** for the v0.1.0 export (already applied in-model).
- A reference greedy NMS (`nms(boxes, scores, iou_threshold=0.45)`) lives in
  `src/export/postprocess.py` for the **fallback** case only — if a future build
  ships raw predictions (`export.py --no-nms`). IoU threshold to mirror then:
  **0.45**.

## Recommended confidence threshold

- **`conf = 0.25`** — the recall-biased operating point. **Recall over
  precision:** a missed plate is a data leak; an over-redacted region is
  harmless. When tuning, lower `conf` rather than raise it.

## App-side function signature (lives in the app repo)

```ts
redactImage(uri: string): Promise<{
  uri: string;
  facesFound: number;
  platesFound: number;
}>
```

Runs on a **single captured still before upload**, not on live camera frames,
**alongside** the existing ML Kit face detection. Latency/model size are
non-critical for a single still (CPU p95 ≤ 500 ms target). `platesFound` is the
number of detections kept after thresholding; each is blacked out before the
image leaves the device. **No unredacted byte may leave the device.**

## Data provenance & licence

The detector is trained **entirely on synthetic data** — artificial DE-format
plates composited onto vehicle/solid backgrounds by the public generator in this
repo. **No real plate photos are used in training or committed anywhere**, so the
model carries no personal data and is GDPR-clean and safely publishable. Code,
generator and model weights are **Apache 2.0**. (Not legal advice — have a
DPO/lawyer review for commercial use.)
