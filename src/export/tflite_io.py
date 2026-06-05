"""Phase 4 — TFLite interpreter helpers shared by the export scripts.

Loads a ``.tflite`` model with whatever runtime is available
(``tflite-runtime`` → ``ai-edge-litert`` → ``tensorflow.lite``) and provides
**dtype-aware** pre/post-processing so one code path handles both an fp16 model
(float32 I/O) and an int8 model (quantised I/O).

Deliberately torch/ultralytics-free: the raw interpreter is exactly what the
consuming app (``react-native-fast-tflite``) sees, so exercising it here tests
the real §2 contract surface — not Ultralytics' Python conveniences.

These scripts are run by path (``python src/export/verify_output.py``), so
``src/export`` is on ``sys.path`` and this is imported as a plain sibling module
— mirroring ``src/eval/common.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_interpreter(model_path: str | Path, num_threads: int | None = None):
    """Return ``(interpreter, backend_name)`` using the first runtime available.

    Preference order matches the issue (``tflite-runtime`` first — the lightweight
    runtime the app shares), then its maintained successor ``ai-edge-litert``,
    then the heavyweight ``tensorflow.lite`` fallback that ships on Kaggle/Colab.
    """
    model_path = str(model_path)
    if not Path(model_path).is_file():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Run the export first:  python src/export/export.py"
        )
    errors: list[str] = []

    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=model_path, num_threads=num_threads), "tflite_runtime"
    except Exception as e:  # noqa: BLE001 — any import/runtime failure → next backend
        errors.append(f"tflite_runtime: {e}")

    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=model_path, num_threads=num_threads), "ai_edge_litert"
    except Exception as e:  # noqa: BLE001
        errors.append(f"ai_edge_litert: {e}")

    try:
        import tensorflow as tf  # type: ignore

        return (
            tf.lite.Interpreter(model_path=model_path, num_threads=num_threads),
            "tensorflow.lite",
        )
    except Exception as e:  # noqa: BLE001
        errors.append(f"tensorflow.lite: {e}")

    raise SystemExit(
        "No TFLite runtime available. Install one of:\n"
        "    pip install tflite-runtime        # lightweight, matches the app\n"
        "    pip install ai-edge-litert        # maintained successor\n"
        "    pip install tensorflow            # heavy fallback (Kaggle/Colab default)\n"
        "\nTried:\n  " + "\n  ".join(errors)
    )


def input_size(input_detail: dict) -> int:
    """Square side length the model expects, read from its input tensor shape."""
    shape = list(input_detail["shape"])
    # NHWC: [1, H, W, 3]; assume square per the §2 contract.
    return int(shape[1]) if len(shape) == 4 else 320


def _quantize_input(arr01: np.ndarray, input_detail: dict) -> np.ndarray:
    """Cast a ``[0,1]`` float image to the input tensor's dtype, quantising if int."""
    dtype = input_detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero = input_detail["quantization"]
        scale = scale or 1.0
        q = np.round(arr01 / scale + zero)
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)
    return arr01.astype(dtype)


def preprocess(image_path: str | Path, input_detail: dict, imgsz: int) -> np.ndarray:
    """Load → RGB → resize → ``/255`` ([0,1]) → NHWC, in the input tensor's dtype.

    Normalisation matches the §2 contract (``pixel / 255``). A plain square resize
    (no letterbox) is enough for a representative latency measurement.
    """
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = (np.asarray(im, dtype=np.float32) / 255.0)[None, ...]  # 1,H,W,3
    return _quantize_input(arr, input_detail)


def blank_input(input_detail: dict, imgsz: int) -> np.ndarray:
    """An all-zero ([0,1]) input in the correct dtype — for a smoke forward pass."""
    arr = np.zeros((1, imgsz, imgsz, 3), dtype=np.float32)
    return _quantize_input(arr, input_detail)


def dequantize_output(arr: np.ndarray, output_detail: dict) -> np.ndarray:
    """Dequantise an int8/uint8 output tensor back to float using its scale/zero."""
    scale, zero = output_detail["quantization"]
    if np.issubdtype(output_detail["dtype"], np.integer) and scale:
        return (arr.astype(np.float32) - zero) * scale
    return arr.astype(np.float32)
