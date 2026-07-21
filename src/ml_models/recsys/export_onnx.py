"""
Export Two-Tower item tower to ONNX format for faster inference.

Why ONNX:
  - ONNX Runtime skips Python/PyTorch overhead for embedding lookup + MLP
  - Typical speedup: 2-4x on CPU for batch inference
  - Enables deployment without PyTorch dependency (lighter Docker image)

What is exported:
  - item_tower only (user tower runs less frequently — only on request, not precompute)
  - Item embeddings are precomputed once at startup, ONNX speeds up the precompute step

After export, run benchmark to verify correctness and measure speedup.

Usage:
    python -m src.ml_models.recsys.export_onnx
    python -m src.ml_models.recsys.export_onnx --benchmark --n-runs 100
"""

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/artifacts/recsys_models"))
_ONNX_PATH = _MODEL_DIR / "item_tower.onnx"
_OPSET = 17


def _load_item_tower():
    """Load item tower from saved checkpoint."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2]))  # ensure src/ is in path
    import joblib

    from src.serving.recsys_api.model import TwoTowerModel

    encoders_dir = _MODEL_DIR / "encoders"
    user_enc = joblib.load(encoders_dir / "user_encoder.pkl")
    item_enc = joblib.load(encoders_dir / "item_encoder.pkl")
    cat_enc = joblib.load(encoders_dir / "cat_encoder.pkl")

    model = TwoTowerModel(
        num_users=len(user_enc.classes_) + 1,
        num_items=len(item_enc.classes_) + 1,
        num_categories=len(cat_enc.classes_) + 1,
        embedding_dim=32,
    )

    weights_path = _MODEL_DIR / "model/best_two_tower.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model.item_tower, len(item_enc.classes_), len(cat_enc.classes_)


def export_item_tower(item_tower, n_items: int, n_categories: int) -> str:
    """Export item tower to ONNX. Returns path to .onnx file."""
    _ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Dummy inputs matching item_tower.forward(item_id, item_category, item_num)
    dummy_item_id = torch.zeros(1, dtype=torch.long)
    dummy_cat_id = torch.zeros(1, dtype=torch.long)
    dummy_item_num = torch.zeros(1, 2, dtype=torch.float32)

    torch.onnx.export(
        item_tower,
        (dummy_item_id, dummy_cat_id, dummy_item_num),
        str(_ONNX_PATH),
        opset_version=_OPSET,
        input_names=["item_id", "item_category", "item_num"],
        output_names=["item_embedding"],
        dynamic_axes={
            "item_id": {0: "batch_size"},
            "item_category": {0: "batch_size"},
            "item_num": {0: "batch_size"},
            "item_embedding": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,
    )
    logger.info("ONNX export complete: %s (opset=%d)", _ONNX_PATH, _OPSET)
    return str(_ONNX_PATH)


def benchmark(
    item_tower, onnx_path: str, batch_sizes: list[int] = [1, 64, 256], n_runs: int = 50
) -> dict:
    """
    Compare latency and verify output correctness between PyTorch and ONNX Runtime.

    Returns dict with latency_ms per batch_size for both backends.
    """
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    item_tower.eval()
    results = {}

    for bs in batch_sizes:
        item_ids = torch.randint(0, 100, (bs,))
        cat_ids = torch.randint(0, 50, (bs,))
        item_nums = torch.rand(bs, 2)

        # PyTorch timing
        pt_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                pt_out = item_tower(item_ids, cat_ids, item_nums)
                pt_times.append((time.perf_counter() - t0) * 1000)

        # ONNX Runtime timing
        ort_times = []
        ort_inputs = {
            "item_id": item_ids.numpy(),
            "item_category": cat_ids.numpy(),
            "item_num": item_nums.numpy(),
        }
        for _ in range(n_runs):
            t0 = time.perf_counter()
            ort_out = ort_session.run(None, ort_inputs)[0]
            ort_times.append((time.perf_counter() - t0) * 1000)

        # Correctness check
        max_diff = float(np.abs(pt_out.numpy() - ort_out).max())
        speedup = np.median(pt_times) / np.median(ort_times)

        results[f"batch_{bs}"] = {
            "pytorch_median_ms": round(np.median(pt_times), 3),
            "onnx_median_ms": round(np.median(ort_times), 3),
            "speedup_x": round(speedup, 2),
            "max_diff": round(max_diff, 8),
            "correct": max_diff < 1e-4,
        }
        logger.info(
            "bs=%d | PyTorch=%.2fms | ONNX=%.2fms | speedup=%.2fx | max_diff=%.2e | correct=%s",
            bs,
            np.median(pt_times),
            np.median(ort_times),
            speedup,
            max_diff,
            max_diff < 1e-4,
        )

    return results


def main(run_benchmark: bool = False, n_runs: int = 50) -> None:
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("onnx_export")

    item_tower, n_items, n_cats = _load_item_tower()
    onnx_path = export_item_tower(item_tower, n_items, n_cats)

    with mlflow.start_run(run_name="onnx_export"):
        mlflow.log_param("opset", _OPSET)
        mlflow.log_param("backend", "item_tower")
        mlflow.log_artifact(onnx_path, "onnx_model")

        if run_benchmark:
            bench = benchmark(item_tower, onnx_path, n_runs=n_runs)
            for config, metrics in bench.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{config}_{k}", v)
            print("\n=== ONNX Benchmark ===")
            for config, metrics in bench.items():
                print(f"\n{config}:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Run latency benchmark after export"
    )
    parser.add_argument(
        "--n-runs", type=int, default=50, help="Number of runs for benchmark"
    )
    args = parser.parse_args()
    main(run_benchmark=args.benchmark, n_runs=args.n_runs)
