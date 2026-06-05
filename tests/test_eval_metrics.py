"""Phase 3 — unit tests for the framework-free eval metrics + aggregation.

These cover the matching / AP logic the acceptance gates depend on, without
needing ultralytics or a checkpoint. The eval scripts are run by path, so we put
``src/eval`` on the path to import the sibling modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval"))

import common  # noqa: E402
import metrics  # noqa: E402


def test_yolo_to_xyxy_centre_conversion():
    assert metrics.yolo_to_xyxy(0.5, 0.5, 0.2, 0.4) == pytest.approx((0.4, 0.3, 0.6, 0.7))


def test_iou_identical_and_disjoint():
    box = (0.0, 0.0, 1.0, 1.0)
    assert metrics.iou_xyxy(box, box) == pytest.approx(1.0)
    assert metrics.iou_xyxy((0.0, 0.0, 0.4, 0.4), (0.6, 0.6, 1.0, 1.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.1, 0.0, 0.3, 0.2)  # shifted by half its width
    # intersection 0.1*0.2=0.02; union 0.04+0.04-0.02=0.06
    assert metrics.iou_xyxy(a, b) == pytest.approx(0.02 / 0.06)


def test_counts_perfect_detection():
    gt = [(0.1, 0.1, 0.3, 0.3)]
    preds = [(0.9, (0.1, 0.1, 0.3, 0.3))]
    assert metrics.counts(gt, preds, 0.5) == (1, 0, 0)


def test_counts_missed_plate_is_false_negative():
    gt = [(0.1, 0.1, 0.3, 0.3)]
    assert metrics.counts(gt, [], 0.5) == (0, 0, 1)  # the failure mode we guard


def test_counts_low_iou_is_fp_and_fn():
    gt = [(0.0, 0.0, 0.2, 0.2)]
    preds = [(0.8, (0.15, 0.15, 0.35, 0.35))]  # IoU well below 0.5
    assert metrics.counts(gt, preds, 0.5) == (0, 1, 1)


def test_greedy_match_one_gt_two_preds_extra_is_fp():
    gt = [(0.1, 0.1, 0.3, 0.3)]
    preds = [(0.9, (0.1, 0.1, 0.3, 0.3)), (0.5, (0.1, 0.1, 0.3, 0.3))]
    tp, fp, fn = metrics.counts(gt, preds, 0.5)
    assert (tp, fp, fn) == (1, 1, 0)  # highest-score pred wins, the other is FP


def test_precision_recall_f1_basic():
    p, r, f1 = metrics.precision_recall_f1(tp=9, fp=1, fn=1)
    assert r == pytest.approx(0.9)
    assert p == pytest.approx(0.9)
    assert f1 == pytest.approx(0.9)


def test_average_precision_perfect_is_one():
    scored = [(0.9, True), (0.8, True), (0.7, True)]
    assert metrics.average_precision(scored, total_gt=3) == pytest.approx(1.0)


def test_average_precision_with_a_false_positive_below_one():
    scored = [(0.9, True), (0.8, False), (0.7, True)]
    ap = metrics.average_precision(scored, total_gt=2)
    assert 0.0 < ap <= 1.0


def test_average_precision_no_gt_is_zero():
    assert metrics.average_precision([(0.9, True)], total_gt=0) == 0.0


def _sample(subset, gt, preds):
    s = common.ImageSample(subset, Path("x.jpg"), Path("x.txt"), gt_boxes=gt)
    s.preds = preds
    return s


def test_aggregate_applies_conf_threshold():
    gt = [(0.1, 0.1, 0.3, 0.3)]
    # one correct pred at score 0.3, one stray low-score FP at 0.1
    samples = [_sample("tilt", gt, [(0.3, (0.1, 0.1, 0.3, 0.3)),
                                     (0.1, (0.6, 0.6, 0.8, 0.8))])]
    hi = common.aggregate_at_conf(samples, conf=0.25, iou_match=0.5)
    assert (hi["tp"], hi["fp"], hi["fn"]) == (1, 0, 0)
    assert hi["recall"] == pytest.approx(1.0)

    lo = common.aggregate_at_conf(samples, conf=0.05, iou_match=0.5)
    assert lo["fp"] == 1  # the stray detection now survives the threshold


def test_aggregate_collects_fn_samples_and_ap():
    gt = [(0.1, 0.1, 0.3, 0.3)]
    samples = [
        _sample("small", gt, []),                                   # missed -> FN
        _sample("small", gt, [(0.9, (0.1, 0.1, 0.3, 0.3))]),        # found
    ]
    m = common.aggregate_at_conf(samples, conf=0.25, iou_match=0.5, with_ap=True)
    assert m["fn"] == 1 and m["tp"] == 1
    assert m["recall"] == pytest.approx(0.5)
    assert len(m["fn_samples"]) == 1
    assert "map50" in m
