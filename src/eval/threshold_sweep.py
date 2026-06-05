"""Phase 3 — confidence-threshold sweep.

Runs the detector once over the full hard-case test set, then re-thresholds in
Python across conf 0.05 → 0.50 to trace recall / precision / F1. The recommended
operating point is the **highest-recall threshold that still holds precision
>= 0.60** — recall-biased per README §2 — and is recorded back into
``models/eval_results.json`` alongside a plot.

Usage (from the repo root, with ``models/best.pt`` present)::

    pip install -e ".[eval]"
    python src/eval/threshold_sweep.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports
import common  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_PRECISION = 0.60  # operating-point constraint (informational gate, README §2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confidence-threshold sweep for the plate detector (Phase 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "best.pt"))
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "test_hard"))
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--conf-floor", type=float, default=0.001,
                   help="Low conf for the single inference pass.")
    p.add_argument("--start", type=float, default=0.05)
    p.add_argument("--stop", type=float, default=0.50)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--plot", default=str(REPO_ROOT / "models" / "threshold_sweep.png"))
    p.add_argument("--results", default=str(REPO_ROOT / "models" / "eval_results.json"),
                   help="eval_results.json to merge the sweep into (created if absent).")
    return p.parse_args(argv)


def _frange(start: float, stop: float, step: float) -> list[float]:
    vals, v = [], start
    while v <= stop + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def recommend(curve: list[dict], min_precision: float) -> float | None:
    """Highest-recall conf whose precision >= ``min_precision`` (lowest conf wins)."""
    eligible = [c for c in curve if c["precision"] >= min_precision]
    if not eligible:
        return None
    best = max(eligible, key=lambda c: (c["recall"], -c["conf"]))
    return best["conf"]


def _plot(curve: list[dict], recommended: float | None, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confs = [c["conf"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(confs, [c["recall"] for c in curve], "o-", label="Recall", color="#d62728")
    ax.plot(confs, [c["precision"] for c in curve], "s-", label="Precision", color="#1f77b4")
    ax.plot(confs, [c["f1"] for c in curve], "^-", label="F1", color="#2ca02c")
    ax.axhline(MIN_PRECISION, ls=":", color="gray", lw=1,
               label=f"precision floor {MIN_PRECISION:.2f}")
    if recommended is not None:
        ax.axvline(recommended, ls="--", color="black", lw=1.2,
                   label=f"recommended conf {recommended:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Recall / Precision / F1 vs confidence (hard-case test set)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data)
    model = common.load_model(args.model)

    # Collect every hard-subset sample and run inference once at the floor.
    samples = []
    for subset in common.discover_subsets(data_root):
        if subset in common.HARD_SUBSETS:
            samples.extend(common.iter_samples(data_root, subset))
    if not samples:
        sys.exit(f"No hard-case samples under {data_root}. "
                 "Run generate_test_set.py first.")
    print(f"[sweep] {len(samples)} images; inferring once at conf>={args.conf_floor}")
    common.run_inference(model, samples, args.imgsz, args.conf_floor, args.iou)

    curve = []
    print(f"\n{'conf':>5}  {'recall':>6}  {'precision':>9}  {'f1':>6}")
    for conf in _frange(args.start, args.stop, args.step):
        m = common.aggregate_at_conf(samples, conf, args.iou_match)
        row = {"conf": conf, "recall": m["recall"],
               "precision": m["precision"], "f1": m["f1"]}
        curve.append(row)
        print(f"{conf:>5.2f}  {m['recall']:>6.3f}  {m['precision']:>9.3f}  {m['f1']:>6.3f}")

    rec = recommend(curve, MIN_PRECISION)
    print(f"\n[sweep] recommended operating conf: "
          f"{rec if rec is not None else 'none meets precision floor'}"
          f"  (highest recall with precision >= {MIN_PRECISION:.2f})")

    _plot(curve, rec, Path(args.plot))
    print(f"[sweep] plot -> {args.plot}")

    # Merge into eval_results.json (preserve existing keys if present).
    results_path = Path(args.results)
    data = {}
    if results_path.is_file():
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data["threshold_sweep"] = {
        "min_precision": MIN_PRECISION,
        "recommended_conf": rec,
        "curve": curve,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[sweep] recommended conf + curve -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
