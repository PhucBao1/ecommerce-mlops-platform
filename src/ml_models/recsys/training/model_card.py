"""
Model Card Generator — automated fairness analysis per user segment.

Called at the end of each training run. Computes NDCG@10 per RFM segment
(Champion/Loyal/At Risk/New) and flags bias when the gap between best and
worst segment exceeds BIAS_THRESHOLD.

Output:
  - model_card.md logged as MLflow artifact
  - per-segment metrics logged to MLflow (for Grafana dashboards)
  - bias_flag metric: 1 if biased, 0 if fair

Usage:
    python -m src.ml_models.recsys.training.model_card --run-id <mlflow_run_id>
    # Or called from train_model.py at end of training
"""

import argparse
import logging
import os
import textwrap
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_BIAS_THRESHOLD = float(os.getenv("MODEL_CARD_BIAS_THRESHOLD", "0.8"))

# RFM segment label — column expected in the holdout data
_SEGMENT_COL = "rfm_segment"
_KNOWN_SEGMENTS = ["Champion", "Loyal", "At Risk", "New"]


# ---------------------------------------------------------------------------
# Segment-level evaluation
# ---------------------------------------------------------------------------


def _ndcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    """Compute NDCG@K for a single user."""
    from src.ml_models.recsys.evaluation.metrics import ndcg_at_k

    return ndcg_at_k(actual, predicted, k)


def evaluate_per_segment(
    model,
    valid_df: pd.DataFrame,
    device: torch.device,
    k: int = 10,
) -> dict[str, float]:
    """
    Compute NDCG@K per RFM segment.

    Args:
        model: Trained TwoTowerModel.
        valid_df: Validation DataFrame with columns [customer_id, product_id, label, rfm_segment].
        device: torch device.
        k: Cutoff for NDCG.

    Returns:
        Dict mapping segment name → mean NDCG@K.
    """
    model.eval()
    results: dict[str, list[float]] = {seg: [] for seg in _KNOWN_SEGMENTS}

    if _SEGMENT_COL not in valid_df.columns:
        logger.warning(
            "model_card: '%s' column not found, skipping segment eval", _SEGMENT_COL
        )
        return {}

    for segment in _KNOWN_SEGMENTS:
        seg_df = valid_df[valid_df[_SEGMENT_COL] == segment]
        if seg_df.empty:
            continue

        for cid, user_df in seg_df.groupby("customer_id"):
            positives = user_df[user_df["label"] > 0]["product_id"].tolist()
            if not positives:
                continue

            with torch.no_grad():
                user_ids = torch.tensor(
                    user_df["customer_id_idx"].values, dtype=torch.long
                ).to(device)
                item_ids = torch.tensor(
                    user_df["product_id_idx"].values, dtype=torch.long
                ).to(device)
                cat_ids = torch.tensor(
                    user_df["category_id_idx"].values, dtype=torch.long
                ).to(device)

                scores = model(user_ids, item_ids, cat_ids).squeeze(-1).cpu().numpy()

            ranked_items = user_df["product_id"].values[np.argsort(-scores)].tolist()
            ndcg = _ndcg_at_k(positives, ranked_items, k=k)
            results[segment].append(ndcg)

    return {
        seg: float(np.mean(scores)) if scores else 0.0
        for seg, scores in results.items()
    }


# ---------------------------------------------------------------------------
# Model card markdown generation
# ---------------------------------------------------------------------------


def _render_model_card(
    segment_ndcg: dict[str, float],
    overall_ndcg: float,
    bias_flag: bool,
    run_id: str,
    model_name: str = "recsys-two-tower",
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    bias_str = (
        "⚠️ YES — segments differ by >20%"
        if bias_flag
        else "✅ NO — segments within 20%"
    )

    rows = "\n".join(
        f"| {seg} | {score:.4f} |"
        for seg, score in sorted(segment_ndcg.items(), key=lambda x: -x[1])
    )

    return textwrap.dedent(
        f"""
    # Model Card: {model_name}

    **Generated:** {now}
    **MLflow Run ID:** `{run_id}`

    ## Performance

    | Metric | Value |
    |--------|-------|
    | Overall NDCG@10 | {overall_ndcg:.4f} |
    | Bias Flag | {bias_str} |

    ## Per-Segment NDCG@10 (RFM Segments)

    | Segment | NDCG@10 |
    |---------|---------|
    {rows}

    ## Bias Analysis

    Bias is flagged when `min_segment_ndcg < max_segment_ndcg × {_BIAS_THRESHOLD}`.
    Segments with fewer interactions (New customers) typically score lower.
    Investigate if At Risk segment drops below 0.05.

    ## Usage

    - **Retrieval stage:** Two-Tower embedding similarity (top-200 candidates)
    - **Reranking stage:** Rule-based + optional CrossEncoder
    - **Serving:** `recsys_api` via FAISS/Qdrant ANN search

    ## Limitations

    - Trained on Vietnamese e-commerce data (Tiki-style)
    - Cold-start users fall back to popularity-based recommendations
    - Evaluation on holdout split, not true A/B test
    """
    ).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_model_card(
    model=None,
    valid_df: pd.DataFrame | None = None,
    run_id: str | None = None,
    overall_ndcg: float = 0.0,
) -> str:
    """
    Generate and log model card for the current or given MLflow run.

    Can be called inline from train_model.py or standalone via CLI.
    """
    mlflow.set_tracking_uri(_MLFLOW_URI)

    if model is not None and valid_df is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segment_ndcg = evaluate_per_segment(model, valid_df, device)
    else:
        segment_ndcg = {seg: 0.0 for seg in _KNOWN_SEGMENTS}

    # Bias flag: worst segment < best segment * threshold
    scores = [v for v in segment_ndcg.values() if v > 0]
    bias_flag = bool(scores and min(scores) < max(scores) * _BIAS_THRESHOLD)

    active_run_id = run_id or (
        mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
    )
    card_md = _render_model_card(segment_ndcg, overall_ndcg, bias_flag, active_run_id)

    # Write to temp file and log as MLflow artifact
    card_path = Path("/tmp/model_card.md")
    card_path.write_text(card_md, encoding="utf-8")

    try:
        mlflow.log_artifact(str(card_path), "model_card")
        for seg, score in segment_ndcg.items():
            mlflow.log_metric(f"ndcg_segment_{seg.lower().replace(' ', '_')}", score)
        mlflow.log_metric("bias_flag", int(bias_flag))
        logger.info(
            "model_card logged to MLflow run %s (bias_flag=%s)",
            active_run_id,
            bias_flag,
        )
    except Exception as e:
        logger.warning("model_card: mlflow log failed: %s", e)

    return card_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id", default=None, help="MLflow run ID to attach card to"
    )
    parser.add_argument("--overall-ndcg", type=float, default=0.0)
    args = parser.parse_args()

    card = generate_model_card(run_id=args.run_id, overall_ndcg=args.overall_ndcg)
    print(card)
