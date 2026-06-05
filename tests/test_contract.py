"""Phase 5 — integration contract test for ``plate-detector.tflite``.

Verifies the exported model conforms **exactly** to the §2 interface contract
(README) and can be loaded + consumed the way the app will load it: through the
raw TFLite runtime (``tflite-runtime`` / ``ai-edge-litert`` / ``tensorflow.lite``)
with no Ultralytics in the loop. All eight checks (T1–T8) run on CPU, no GPU.

Model resolution order:
  1. ``$PLATE_DETECTOR_PATH``
  2. ``models/plate-detector.tflite``
  3. ``models/plate-detector-v0.1.0.tflite`` (the committed release artefact)

If no model and no TFLite runtime are available the suite skips (so a bare
checkout without the release asset doesn't fail CI), but it runs fully wherever
the artefact + a runtime are present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "export"))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SIZE_LIMIT_MB = 8.0
CONF = 0.25  # recall-biased operating point (Phase-3 sweep)


def _resolve_model() -> Path | None:
    env = os.environ.get("PLATE_DETECTOR_PATH")
    candidates = [Path(env)] if env else []
    candidates += [
        REPO_ROOT / "models" / "plate-detector.tflite",
        REPO_ROOT / "models" / "plate-detector-v0.1.0.tflite",
    ]
    return next((c for c in candidates if c.is_file()), None)


MODEL_PATH = _resolve_model()


@pytest.fixture(scope="module")
def runtime():
    """Load the model once; skip the whole module if model/runtime is missing."""
    if MODEL_PATH is None:
        pytest.skip("No plate-detector .tflite found (set PLATE_DETECTOR_PATH).")
    try:
        import tflite_io  # noqa: F401 — sibling module from src/export
    except ImportError:  # pragma: no cover
        pytest.skip("src/export not importable")
    import tflite_io
    import postprocess
    try:
        interp, backend = tflite_io.load_interpreter(str(MODEL_PATH))
    except SystemExit as e:  # no runtime installed
        pytest.skip(f"No TFLite runtime available: {e}")
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    imgsz = tflite_io.input_size(in_det)
    return {
        "interp": interp, "backend": backend, "in_det": in_det,
        "out_details": out_details, "imgsz": imgsz,
        "tflite_io": tflite_io, "postprocess": postprocess,
    }


def _infer(rt, tensor):
    """Set input, invoke, return the dequantised first output tensor."""
    interp, in_det, out_details = rt["interp"], rt["in_det"], rt["out_details"]
    interp.set_tensor(in_det["index"], tensor)
    interp.invoke()
    return rt["tflite_io"].dequantize_output(
        interp.get_tensor(out_details[0]["index"]), out_details[0])


def _fixtures(*names: str) -> list[Path]:
    if names:
        return [FIXTURES / f"{n}.jpg" for n in names]
    return sorted(FIXTURES.glob("*.jpg"))


# --------------------------------------------------------------------------- #
# T1 — Input shape
# --------------------------------------------------------------------------- #
def test_t1_input_shape(runtime):
    in_det = runtime["in_det"]
    assert list(in_det["shape"]) == [1, 320, 320, 3], (
        f"input shape {list(in_det['shape'])} != [1, 320, 320, 3]")
    assert np.dtype(in_det["dtype"]) == np.float32, (
        f"input dtype {np.dtype(in_det['dtype'])} != float32")


# --------------------------------------------------------------------------- #
# T2 — Input range: zeros and ones must not crash; scores stay in [0, 1]
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_t2_input_range(runtime, fill):
    imgsz, in_det = runtime["imgsz"], runtime["in_det"]
    arr = np.full((1, imgsz, imgsz, 3), fill, dtype=np.float32)
    tensor = runtime["tflite_io"]._quantize_input(arr, in_det)
    out = _infer(runtime, tensor)  # must not raise
    scores = out.reshape(-1, out.shape[-1])[:, 4]
    assert np.all(scores >= -1e-6) and np.all(scores <= 1.0 + 1e-6), (
        f"scores out of [0,1]: min={scores.min()} max={scores.max()}")


# --------------------------------------------------------------------------- #
# T3 — Output parseable into list of {box, score} for several synthetic images
# --------------------------------------------------------------------------- #
def test_t3_output_parseable(runtime):
    paths = _fixtures()
    assert len(paths) >= 5, f"need ≥5 fixtures, found {len(paths)}"
    pp, io = runtime["postprocess"], runtime["tflite_io"]
    for p in paths[:5] if len(paths) > 5 else paths:
        tensor = io.preprocess(p, runtime["in_det"], runtime["imgsz"])
        out = _infer(runtime, tensor)
        dets = pp.parse_output(out, conf_threshold=CONF)
        assert isinstance(dets, list)
        for d in dets:
            assert set(d) == {"box", "score"}, f"unexpected keys: {d.keys()}"
            assert len(d["box"]) == 4


# --------------------------------------------------------------------------- #
# T4 — Box coordinates normalised to [0, 1]
# --------------------------------------------------------------------------- #
def test_t4_boxes_normalised(runtime):
    pp, io = runtime["postprocess"], runtime["tflite_io"]
    for p in _fixtures():
        out = _infer(runtime, io.preprocess(p, runtime["in_det"], runtime["imgsz"]))
        for d in pp.parse_output(out, conf_threshold=CONF):
            for v in d["box"]:
                assert 0.0 <= v <= 1.0, f"{p.name}: coord {v} not in [0,1]"


# --------------------------------------------------------------------------- #
# T5 — Scores in [0, 1]
# --------------------------------------------------------------------------- #
def test_t5_score_range(runtime):
    pp, io = runtime["postprocess"], runtime["tflite_io"]
    for p in _fixtures():
        out = _infer(runtime, io.preprocess(p, runtime["in_det"], runtime["imgsz"]))
        for d in pp.parse_output(out, conf_threshold=CONF):
            assert 0.0 <= d["score"] <= 1.0, f"{p.name}: score {d['score']}"


# --------------------------------------------------------------------------- #
# T6 — Orientation: portrait and landscape each yield ≥1 detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["portrait_clean", "landscape_clean"])
def test_t6_orientation(runtime, name):
    pp, io = runtime["postprocess"], runtime["tflite_io"]
    p = FIXTURES / f"{name}.jpg"
    assert p.is_file(), f"missing orientation fixture {p}"
    out = _infer(runtime, io.preprocess(p, runtime["in_det"], runtime["imgsz"]))
    dets = pp.parse_output(out, conf_threshold=CONF)
    assert len(dets) >= 1, f"{name}: expected ≥1 detection, got {len(dets)}"


# --------------------------------------------------------------------------- #
# T7 — No crash on a blank solid-grey image (zero detections acceptable)
# --------------------------------------------------------------------------- #
def test_t7_blank_image(runtime):
    imgsz, in_det = runtime["imgsz"], runtime["in_det"]
    arr = np.full((1, imgsz, imgsz, 3), 0.5, dtype=np.float32)  # mid grey
    tensor = runtime["tflite_io"]._quantize_input(arr, in_det)
    out = _infer(runtime, tensor)  # must not raise
    dets = runtime["postprocess"].parse_output(out, conf_threshold=CONF)
    assert isinstance(dets, list)  # zero detections is fine


# --------------------------------------------------------------------------- #
# T8 — File size ≤ 8 MB
# --------------------------------------------------------------------------- #
def test_t8_file_size():
    if MODEL_PATH is None:
        pytest.skip("No plate-detector .tflite found (set PLATE_DETECTOR_PATH).")
    size_mb = MODEL_PATH.stat().st_size / 1e6
    assert size_mb <= SIZE_LIMIT_MB, f"{MODEL_PATH.name} is {size_mb:.2f} MB > {SIZE_LIMIT_MB} MB"
