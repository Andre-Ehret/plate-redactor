"""Phase 4 — CPU latency benchmark for the exported ``.tflite``.

Runs inference on N random validation images **on CPU** (the TFLite interpreter
is CPU-only here — no GPU delegate) and reports mean / median / p95 latency. This
mirrors the app's single-still use case: one captured photo redacted before
upload, not a live frame stream.

Only the ``invoke()`` call is timed (pre/post-processing excluded) so the number
reflects the model, not Pillow. A few warm-up passes are discarded first.

**Informational on dev hardware** — real-device latency is verified in Phase 5.
Target: p95 ≤ 500 ms.

Usage (after ``export.py``)::

    python src/export/benchmark.py
    python src/export/benchmark.py --n 20 --num-threads 1
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import
import tflite_io  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU latency benchmark for the exported TFLite model (Phase 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "plate-detector.tflite"),
                   help="Exported TFLite model to benchmark.")
    p.add_argument("--images-dir",
                   default=str(REPO_ROOT / "data" / "synthetic" / "images" / "val"),
                   help="Directory of images to sample from.")
    p.add_argument("--n", type=int, default=20, help="Number of images to time.")
    p.add_argument("--warmup", type=int, default=3, help="Warm-up passes to discard.")
    p.add_argument("--num-threads", type=int, default=1,
                   help="Interpreter CPU threads (1 ≈ a conservative single-still target).")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--out", default=str(REPO_ROOT / "models" / "benchmark_results.json"),
                   help="Where to write the latency JSON.")
    return p.parse_args(argv)


def _sample_images(images_dir: Path, n: int, seed: int) -> list[Path]:
    if not images_dir.is_dir():
        sys.exit(f"Images directory not found: {images_dir}\n"
                 "Generate a dataset (Phase 1) or pass --images-dir.")
    imgs = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in _IMG_EXTS)
    if not imgs:
        sys.exit(f"No images under {images_dir}.")
    rng = random.Random(seed)
    if len(imgs) <= n:
        return imgs
    return rng.sample(imgs, n)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interp, backend = tflite_io.load_interpreter(args.model, num_threads=args.num_threads)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    imgsz = tflite_io.input_size(in_det)

    images = _sample_images(Path(args.images_dir), args.n, args.seed)
    # Pre-process up front so only invoke() is timed.
    tensors = [tflite_io.preprocess(p, in_det, imgsz) for p in images]

    print(f"[bench] model={args.model} backend={backend} imgsz={imgsz} "
          f"threads={args.num_threads}")
    print(f"[bench] {len(images)} images, {args.warmup} warm-up passes (CPU only)")

    for t in tensors[: max(1, args.warmup)]:
        interp.set_tensor(in_det["index"], t)
        interp.invoke()

    times_ms: list[float] = []
    for t in tensors:
        interp.set_tensor(in_det["index"], t)
        start = time.perf_counter()
        interp.invoke()
        for d in out_details:  # touch outputs so lazy work isn't excluded
            interp.get_tensor(d["index"])
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms.sort()
    mean = statistics.fmean(times_ms)
    median = statistics.median(times_ms)
    p95 = times_ms[min(len(times_ms) - 1, int(round(0.95 * (len(times_ms) - 1))))]
    passed = p95 <= 500.0

    print(f"\n[bench] mean   {mean:7.1f} ms")
    print(f"[bench] median {median:7.1f} ms")
    print(f"[bench] p95    {p95:7.1f} ms   (target ≤ 500 ms: "
          f"{'PASS' if passed else 'FAIL'})")
    print("[bench] note: informational on dev hardware; real-device latency is "
          "verified in Phase 5.")

    results = {
        "model": str(args.model),
        "backend": backend,
        "num_threads": args.num_threads,
        "imgsz": imgsz,
        "n_images": len(times_ms),
        "warmup": args.warmup,
        "mean_ms": round(mean, 2),
        "median_ms": round(median, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(times_ms[0], 2),
        "max_ms": round(times_ms[-1], 2),
        "target_p95_ms": 500,
        "p95_ok": passed,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[bench] results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
