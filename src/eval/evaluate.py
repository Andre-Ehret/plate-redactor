"""Phase 3 — recall-focused evaluation of the trained detector.

Loads the Phase-2 checkpoint, runs it over the hard-case test set
(``data/test_hard/`` — produced by ``generate_test_set.py``), matches predictions
to ground truth at IoU >= 0.5, and reports per-subset + overall recall /
precision / F1 / mAP@0.5. **Recall is the headline metric and the hard gate** —
a missed plate is a data leak (see README §2).

Outputs (all gitignored except scripts and the report):
  * a results table on stdout;
  * ``models/eval_results.json`` — full metrics + gate verdicts + config;
  * ``models/eval_failures/<subset>/`` — annotated false-negative images;
  * ``models/eval_report.md`` — a short auto-drafted failure summary.

Usage (from the repo root, with ``models/best.pt`` present)::

    pip install -e ".[eval]"
    python src/eval/evaluate.py
    python src/eval/evaluate.py --conf 0.25 --model models/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports
import common  # noqa: E402
import metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Acceptance gates (README §2 / Phase-3 issue). Recall is the only hard gate.
OVERALL_RECALL_GATE = 0.90
ORIENT_RECALL_GATE = 0.90
SUBSET_RECALL_SOFT = 0.85
PRECISION_INFO = 0.60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recall-focused evaluation of the plate detector (Phase 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "best.pt"),
                   help="Trained checkpoint to evaluate.")
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "test_hard"),
                   help="Hard-case test-set root (from generate_test_set.py).")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Inference input size (match training).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Operating confidence threshold (recall bias, set low).")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold for inference.")
    p.add_argument("--iou-match", type=float, default=0.5,
                   help="IoU threshold for matching a prediction to a GT box.")
    p.add_argument("--conf-floor", type=float, default=0.001,
                   help="Low conf for the single inference pass (re-thresholded "
                        "in Python for the table / AP).")
    p.add_argument("--out", default=str(REPO_ROOT / "models" / "eval_results.json"),
                   help="Where to write the full results JSON.")
    p.add_argument("--failures", default=str(REPO_ROOT / "models" / "eval_failures"),
                   help="Directory for annotated false-negative images.")
    p.add_argument("--report", default=str(REPO_ROOT / "models" / "eval_report.md"),
                   help="Where to write the failure-summary markdown.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if a hard recall gate is not met.")
    return p.parse_args(argv)


def _fmt_row(name: str, m: dict) -> str:
    return (f"{name:<12} {m['images']:>6} {m['gt']:>5} {m['tp']:>4} {m['fn']:>4} "
            f"{m['fp']:>4}  {m['recall']:>6.3f}   {m['precision']:>8.3f}  "
            f"{m['f1']:>6.3f}  {m.get('map50', float('nan')):>7.3f}")


def _print_table(per_subset: dict, overall: dict, orient: dict) -> None:
    header = (f"{'Subset':<12} {'Images':>6} {'GT':>5} {'TP':>4} {'FN':>4} "
              f"{'FP':>4}  {'Recall':>6}   {'Precision':>8}  {'F1':>6}  {'mAP@.5':>7}")
    print("\n" + header)
    print("-" * len(header))
    for name in common.HARD_SUBSETS:
        if name in per_subset:
            print(_fmt_row(name, per_subset[name]))
    print("-" * len(header))
    print(_fmt_row("OVERALL", overall))
    if orient:
        print("-" * len(header))
        for name in common.ORIENT_SUBSETS:
            if name in orient:
                print(_fmt_row(name, orient[name]))


def _annotate_failures(fn_samples, out_dir: Path, conf: float) -> int:
    """Draw GT (red) + kept predictions (green) for FN images. Returns count."""
    from PIL import Image, ImageDraw

    n = 0
    for s in fn_samples:
        with Image.open(s.image_path) as im:
            img = im.convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2) in s.gt_boxes:
            draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 0), width=3)
        for score, (x1, y1, x2, y2) in s.preds:
            if score >= conf:
                draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h],
                               outline=(0, 255, 0), width=2)
        sub_dir = out_dir / s.subset
        sub_dir.mkdir(parents=True, exist_ok=True)
        img.save(sub_dir / f"{s.image_path.stem}_FN.jpg", "JPEG", quality=90)
        n += 1
    return n


def _categorise_fn(fn_samples) -> dict[str, int]:
    """Heuristic bucket for each missed plate, to seed the failure report."""
    cats = {"too_small": 0, "extreme_aspect": 0, "other": 0}
    for s in fn_samples:
        if not s.gt_boxes:
            cats["other"] += 1
            continue
        x1, y1, x2, y2 = s.gt_boxes[0]
        w, h = x2 - x1, y2 - y1
        area = w * h
        if area < 0.02:
            cats["too_small"] += 1
        elif h > 0 and (w / h > 6.0 or w / h < 2.0):
            cats["extreme_aspect"] += 1
        else:
            cats["other"] += 1
    return cats


def _write_report(path: Path, overall: dict, per_subset: dict, orient: dict,
                  cats: dict, conf: float, gates: dict) -> None:
    weak = [n for n in common.HARD_SUBSETS
            if n in per_subset and per_subset[n]["recall"] < SUBSET_RECALL_SOFT]
    n_fn_imgs = sum(cats.values())  # all FN images, incl. orientation subsets
    lines = [
        "# Phase 3 — evaluation failure summary",
        "",
        f"Auto-drafted by `evaluate.py` at conf={conf:.2f}, IoU-match=0.50. "
        "**Inspect the annotated FN images under `models/eval_failures/` and "
        "refine this paragraph by hand.**",
        "",
        f"Overall recall **{overall['recall']:.3f}** "
        f"({overall['tp']}/{overall['gt']} plates found, {overall['fn']} missed) "
        f"on {overall['images']} hard-case images — hard gate "
        f"(≥ {OVERALL_RECALL_GATE:.2f}) **{'PASSED' if gates['overall'] else 'NOT MET'}**. "
        f"Precision {overall['precision']:.3f}, mAP@0.5 {overall.get('map50', float('nan')):.3f}. "
        + (f"Weakest subset(s) below the {SUBSET_RECALL_SOFT:.2f} soft floor: "
           f"{', '.join(weak)}. " if weak else "All subsets clear the soft recall floor. ")
        + (f"Across the {n_fn_imgs} false-negative image(s) (all subsets), the "
           f"heuristic split is {cats['too_small']} tiny-plate, "
           f"{cats['extreme_aspect']} extreme-aspect, {cats['other']} other "
           f"(occlusion / shadow / angle — confirm visually)."
           if n_fn_imgs else "No false negatives at this operating point."),
    ]
    if orient:
        po = orient.get("portrait", {}).get("recall", float("nan"))
        lo = orient.get("landscape", {}).get("recall", float("nan"))
        lines += ["", f"Orientation check: portrait recall {po:.3f}, "
                      f"landscape recall {lo:.3f} (both gate ≥ "
                      f"{ORIENT_RECALL_GATE:.2f})."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data)
    if not (data_root / "images").is_dir():
        sys.exit(f"Test set not found at {data_root}.\n"
                 "Generate it first:  python src/eval/generate_test_set.py")

    subsets = common.discover_subsets(data_root)
    if not subsets:
        sys.exit(f"No known subsets under {data_root}/images/.")

    model = common.load_model(args.model)
    print(f"[eval] model={args.model}  data={data_root}")
    print(f"[eval] imgsz={args.imgsz} conf={args.conf} iou(nms)={args.iou} "
          f"iou(match)={args.iou_match}")

    per_subset: dict[str, dict] = {}
    orient: dict[str, dict] = {}
    all_fn = []
    agg_overall = {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0}
    scored_overall: list[tuple[float, bool]] = []

    for subset in subsets:
        samples = common.iter_samples(data_root, subset)
        common.run_inference(model, samples, args.imgsz, args.conf_floor, args.iou)
        m = common.aggregate_at_conf(samples, args.conf, args.iou_match, with_ap=True)
        target = orient if subset in common.ORIENT_SUBSETS else per_subset
        target[subset] = m
        if subset in common.HARD_SUBSETS:
            for k in ("images", "gt", "tp", "fp", "fn"):
                agg_overall[k] += m[k]
            for s in samples:
                sc, _ = metrics.greedy_match(s.gt_boxes, s.preds, args.iou_match)
                scored_overall.extend(sc)
        all_fn.extend(m["fn_samples"])

    pr, rc, f1 = metrics.precision_recall_f1(
        agg_overall["tp"], agg_overall["fp"], agg_overall["fn"])
    overall = {**agg_overall, "recall": rc, "precision": pr, "f1": f1,
               "map50": metrics.average_precision(scored_overall, agg_overall["gt"])}

    _print_table(per_subset, overall, orient)

    # Hard gates.
    gates = {
        "overall": overall["recall"] >= OVERALL_RECALL_GATE,
        "portrait": orient.get("portrait", {}).get("recall", 1.0) >= ORIENT_RECALL_GATE,
        "landscape": orient.get("landscape", {}).get("recall", 1.0) >= ORIENT_RECALL_GATE,
    }
    all_pass = all(gates.values())
    print("\n[gates]")
    print(f"  overall recall    {overall['recall']:.3f} >= {OVERALL_RECALL_GATE:.2f}  "
          f"{'PASS' if gates['overall'] else 'FAIL'}")
    for o in common.ORIENT_SUBSETS:
        if o in orient:
            print(f"  {o:<9} recall   {orient[o]['recall']:.3f} >= {ORIENT_RECALL_GATE:.2f}  "
                  f"{'PASS' if gates[o] else 'FAIL'}")
    print(f"\n  ==> Phase-3 acceptance: {'PASSED — proceed to Phase 4' if all_pass else 'NOT MET — retrain before export'}")

    # Annotated failures + report + JSON.
    n_fail_imgs = _annotate_failures(all_fn, Path(args.failures), args.conf)
    cats = _categorise_fn(all_fn)
    _write_report(Path(args.report), overall, per_subset, orient, cats, args.conf, gates)

    def _clean(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "fn_samples"}

    results = {
        "config": {
            "model": str(args.model), "data": str(data_root),
            "imgsz": args.imgsz, "conf": args.conf, "iou_nms": args.iou,
            "iou_match": args.iou_match,
        },
        "gates": {
            "overall_recall_min": OVERALL_RECALL_GATE,
            "orient_recall_min": ORIENT_RECALL_GATE,
            "subset_recall_soft_min": SUBSET_RECALL_SOFT,
            "precision_info_min": PRECISION_INFO,
            "results": gates, "passed": all_pass,
        },
        "overall": overall,
        "subsets": {k: _clean(v) for k, v in per_subset.items()},
        "orientation": {k: _clean(v) for k, v in orient.items()},
        "failures": {"annotated_images": n_fail_imgs, "categories": cats},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[eval] results  -> {out}")
    print(f"[eval] failures -> {args.failures} ({n_fail_imgs} images)")
    print(f"[eval] report   -> {args.report}")

    if args.strict and not all_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
