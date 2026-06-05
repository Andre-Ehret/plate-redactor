"""Phase 4 — export the Phase-2 checkpoint to a quantised ``.tflite``.

Wraps Ultralytics' TFLite export so the same command runs locally, on Kaggle and
on Colab. It:

  1. loads ``models/best.pt`` (the YOLOv8n checkpoint),
  2. exports to TFLite at the locked input size (``imgsz=320``), fp16 by default
     (``half=True``) — int8 is available as a fallback (``--quantisation int8``,
     needs a calibration dataset),
  3. tries to **bake NMS into the graph** (``nms=True``) so the app receives
     ready-made detections; falls back to a raw export (post-processing NMS) if
     the installed Ultralytics doesn't support in-graph NMS for TFLite,
  4. copies the result to ``models/plate-detector.tflite`` and to the versioned
     release name ``models/plate-detector-v<version>.tflite``,
  5. writes ``models/export_meta.json`` (sizes / settings) for the report, and
     prints the exported file size in MB.

**Whether NMS actually ended up in the graph is confirmed by**
``verify_output.py`` — it inspects the real output signature and writes the
authoritative note into ``models/export_report.md`` / the README.

Quantisation decision (see the Phase-4 issue): **fp16 first** — it halves the
model size with negligible detection-accuracy loss and needs no calibration data.
Only fall back to int8 if fp16 misses the 8 MB target or recall drops > 2 pp.

Usage (from the repo root, with ``models/best.pt`` present)::

    pip install -e ".[export]"
    python src/export/export.py                       # fp16, imgsz 320, NMS in-graph
    python src/export/export.py --quantisation int8 --data data/synthetic/data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION = "0.1.0"  # semver tag for the release artefact (plate-detector-v0.1.0.tflite)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export the YOLOv8n plate detector to quantised TFLite (Phase 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "best.pt"),
                   help="Phase-2 checkpoint to export.")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Square input side length — must match Phase 2 (locked at 320).")
    p.add_argument("--quantisation", choices=("fp16", "int8"), default="fp16",
                   help="fp16 (primary) or int8 (fallback; needs --data for calibration).")
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "synthetic" / "data.yaml"),
                   help="data.yaml used to calibrate int8 quantisation (int8 only).")
    p.add_argument("--out", default=str(REPO_ROOT / "models" / "plate-detector.tflite"),
                   help="Working copy of the exported model.")
    p.add_argument("--version", default=DEFAULT_VERSION,
                   help="Semver tag for the versioned release artefact name.")
    nms = p.add_mutually_exclusive_group()
    nms.add_argument("--nms", dest="nms", action="store_true", default=True,
                     help="Bake NMS into the graph (default; app gets ready detections).")
    nms.add_argument("--no-nms", dest="nms", action="store_false",
                     help="Raw predictions — NMS done in app-side post-processing.")
    p.add_argument("--meta", default=str(REPO_ROOT / "models" / "export_meta.json"),
                   help="Where to record export settings + sizes (for the report).")
    return p.parse_args(argv)


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def _export(model, *, imgsz: int, quant: str, data: str, nms: bool):
    """Run Ultralytics export, retrying without NMS if the version rejects it.

    Returns ``(exported_path, nms_requested_effective)``.
    """
    kwargs = dict(format="tflite", imgsz=imgsz)
    if quant == "fp16":
        kwargs["half"] = True
    else:  # int8 — Ultralytics calibrates on the dataset referenced by data.yaml
        kwargs["int8"] = True
        kwargs["data"] = data

    try:
        out = model.export(nms=nms, **kwargs)
        return Path(out), nms
    except TypeError as e:
        # Older Ultralytics: `nms` not accepted for TFLite export → raw graph.
        if nms:
            print(f"[export] in-graph NMS unavailable ({e}); exporting raw "
                  "predictions — NMS will run in app-side post-processing.")
            out = model.export(**kwargs)
            return Path(out), False
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.model)
    if not src.is_file():
        sys.exit(
            f"Checkpoint not found: {src}\n"
            "Phase 4 needs the Phase-2 checkpoint. Download the release asset to "
            "models/best.pt, or pass --model."
        )
    if args.quantisation == "int8" and not Path(args.data).is_file():
        sys.exit(
            f"int8 calibration needs a dataset: data.yaml not found at {args.data}.\n"
            "Generate one (Phase 1) or pass --data."
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('ultralytics is not installed. Install the export extras:\n'
                 '    pip install -e ".[export]"')

    fp32_mb = _mb(src)
    print(f"[export] model={src} ({fp32_mb:.2f} MB fp32)")
    print(f"[export] imgsz={args.imgsz} quantisation={args.quantisation} "
          f"nms={'in-graph' if args.nms else 'post-processing'}")

    model = YOLO(str(src))
    exported, nms_effective = _export(
        model, imgsz=args.imgsz, quant=args.quantisation, data=args.data, nms=args.nms)
    if not exported.is_file():
        sys.exit(f"Export reported success but file is missing: {exported}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    versioned = out.parent / f"plate-detector-v{args.version}.tflite"
    shutil.copy2(exported, versioned)

    exported_mb = _mb(out)
    print(f"[export] exported -> {exported} ({exported_mb:.2f} MB)")
    print(f"[export] working copy -> {out}")
    print(f"[export] release artefact -> {versioned}")
    print(f"[export] size: {exported_mb:.2f} MB (target ≤ 8 MB) — "
          f"{'OK' if exported_mb <= 8 else 'OVER TARGET, consider int8'}")

    meta = {
        "source_model": str(src),
        "fp32_size_mb": round(fp32_mb, 3),
        "exported_model": str(out),
        "versioned_artefact": str(versioned),
        "exported_size_mb": round(exported_mb, 3),
        "size_target_mb": 8,
        "size_ok": exported_mb <= 8,
        "imgsz": args.imgsz,
        "quantisation": args.quantisation,
        "nms_requested_in_graph": args.nms,
        "nms_export_effective": nms_effective,
        "version": args.version,
        "raw_export_path": str(exported),
    }
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[export] meta -> {meta_path}")
    print("\n[export] next: verify the signature + NMS location:\n"
          "    python src/export/verify_output.py\n"
          "    python src/export/benchmark.py\n"
          "    python src/eval/evaluate.py --model models/plate-detector.tflite "
          "--data data/test_hard   # recall regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
