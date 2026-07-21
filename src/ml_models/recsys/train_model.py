# =========================================================
# FILE: train_model.py
# =========================================================

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.ml_models.recsys.data.build_dataset import build_dataset
from src.ml_models.recsys.training.mlflow_logger import (
    log_metrics,
    log_model,
    setup_mlflow,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
# =========================================================
# UTILS
# =========================================================

# =========================================================
# ARTIFACTS
# =========================================================
from src.ml_models.recsys.artifacts.save_artifacts import save_artifacts

# =========================================================
# DATASET
# =========================================================
from src.ml_models.recsys.datasets.recsys_dataset import EcommerceRecSysDataset

# =========================================================
# MODEL
# =========================================================
from src.ml_models.recsys.models.two_tower import TwoTowerModel

# =========================================================
# TRAINING
# =========================================================
from src.ml_models.recsys.training.trainer import (
    train_one_epoch_infonce,
    validate,
)
from src.ml_models.recsys.utils.encoding import fit_encoders, safe_transform
from src.ml_models.recsys.utils.scaling import scale_features

# =========================================================
# CONFIG
# =========================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 1024
EPOCHS = 10
EMBEDDING_DIM = 32

# Path where Airflow DAG exports aggregated feedback from silver.user_feedback_events
_FEEDBACK_PATH = Path(
    os.getenv(
        "FEEDBACK_SIGNAL_PATH",
        "artifacts/recsys_models/data_menu/user_feedback_events.parquet",
    )
)
_FEEDBACK_WEIGHTS = {"purchase": 3, "click": 1}


def _load_feedback_signal() -> pd.DataFrame | None:
    """
    Load user feedback aggregated by feedback_aggregator Airflow DAG.

    Returns DataFrame with columns [customer_id, product_id, action, weight]
    or None if file doesn't exist yet (first run before any feedback collected).
    """
    if not _FEEDBACK_PATH.exists():
        logger.info(
            "No feedback signal found at %s — training without RLHF data",
            _FEEDBACK_PATH,
        )
        return None
    try:
        df = pd.read_parquet(_FEEDBACK_PATH)
        df = df[df["action"].isin(_FEEDBACK_WEIGHTS)].copy()
        df["label"] = df["action"].map(_FEEDBACK_WEIGHTS).astype(float)
        logger.info("Loaded %d feedback events (purchases+clicks)", len(df))
        return df[["customer_id", "product_id", "label"]]
    except Exception as exc:
        logger.warning("Failed to load feedback signal: %s — skipping", exc)
        return None


def _merge_feedback(
    train_pos_df: pd.DataFrame, feedback_df: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Merge RLHF feedback into training data.

    Purchases (weight=3) are repeated 3x, clicks (weight=1) appended once.
    This upsamples high-signal interactions without changing the dataset schema.
    """
    if feedback_df is None or len(feedback_df) == 0:
        return train_pos_df

    # Only keep feedback for known users/items (avoid OOV during encoding)
    known_users = set(train_pos_df["customer_id"].astype(str))
    known_items = set(train_pos_df["product_id"].astype(str))
    feedback_df = feedback_df[
        feedback_df["customer_id"].astype(str).isin(known_users)
        & feedback_df["product_id"].astype(str).isin(known_items)
    ].copy()

    if len(feedback_df) == 0:
        logger.info(
            "No overlapping feedback with training users/items — skipping merge"
        )
        return train_pos_df

    # Repeat high-signal rows by weight
    rows = []
    for _, row in feedback_df.iterrows():
        repeat = int(row["label"])
        for _ in range(repeat):
            rows.append(row)
    extra = pd.DataFrame(rows)

    # Add missing columns (fill with neutral defaults so Dataset doesn't break)
    for col in train_pos_df.columns:
        if col not in extra.columns:
            extra[col] = train_pos_df[col].iloc[0] if col != "label" else 1.0

    merged = pd.concat([train_pos_df, extra[train_pos_df.columns]], ignore_index=True)
    logger.info(
        "Feedback merged: original=%d + feedback_rows=%d = total=%d",
        len(train_pos_df),
        len(extra),
        len(merged),
    )
    return merged


# =========================================================
# MAIN
# =========================================================


def main(prebuilt: tuple | None = None):
    """
    prebuilt: (train_df, valid_df, test_df, item_lookup, raw_df) đã build sẵn
    — dùng khi negative sampling (bên trong build_dataset(), vòng loop O(n×m)
    CPU thuần) đã chạy LOCAL từ trước (xem
    src/data_pipeline/jobs/build_negative_samples_local.py), để không chạy
    lại lần nữa trên Kaggle chỉ tổ tốn quota GPU cho việc không cần GPU.
    None (mặc định) — hành vi cũ, tự gọi build_dataset() như trước.
    """

    # =====================================================
    # DEVICE
    # =====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Using device: %s", device)

    feedback_df = _load_feedback_signal()

    if prebuilt is not None:
        train_df, valid_df, test_df, item_lookup, raw_df = prebuilt
        logger.info(
            "Dùng train/valid/test đã build sẵn (bỏ qua build_dataset() nội bộ)"
        )
    else:
        train_df, valid_df, test_df, item_lookup, raw_df = build_dataset()

    logger.info("Train=%d Valid=%d Test=%d", len(train_df), len(valid_df), len(test_df))

    # =====================================================
    # ENCODERS
    # =====================================================

    logger.info("Encoding IDs...")

    (train_df, user_encoder, item_encoder, cat_encoder) = fit_encoders(train_df)

    # =====================================================
    # VALID ENCODE
    # =====================================================

    valid_df["customer_id_idx"] = safe_transform(user_encoder, valid_df["customer_id"])

    valid_df["product_id_idx"] = safe_transform(item_encoder, valid_df["product_id"])

    valid_df["category_id_idx"] = safe_transform(
        cat_encoder, valid_df["category_id"].astype(str)
    )

    # =====================================================
    # TEST ENCODE
    # =====================================================

    test_df["customer_id_idx"] = safe_transform(user_encoder, test_df["customer_id"])

    test_df["product_id_idx"] = safe_transform(item_encoder, test_df["product_id"])

    test_df["category_id_idx"] = safe_transform(
        cat_encoder, test_df["category_id"].astype(str)
    )

    # =====================================================
    # SCALE FEATURES
    # =====================================================

    logger.info("Scaling features...")

    (train_df, valid_df, test_df, scaler) = scale_features(train_df, valid_df, test_df)

    # =====================================================
    # SAVE TRAIN/VALID/TEST
    # =====================================================

    train_df.to_parquet("train_df.parquet")
    valid_df.to_parquet("valid_df.parquet")
    test_df.to_parquet("test_df.parquet")

    # =====================================================
    # SAVE HISTORY
    # =====================================================

    history_cols = [
        "customer_id",
        "product_id",
        "purchased_at",
        "total_reviews_so_far",
        "avg_price_preference",
        "positive_review_ratio",
        "has_history",
    ]

    history_df = raw_df[history_cols].copy()

    history_df.to_parquet("artifacts/recsys_models/data_menu/user_history.parquet")

    # =====================================================
    # SAVE ITEM LOOKUP
    # =====================================================

    item_lookup.reset_index().to_parquet(
        "artifacts/recsys_models/data_menu/item_lookup.parquet"
    )

    # =====================================================
    # DATASET
    # =====================================================

    # InfoNCE trains on positive interactions only (label > 0 = Positive or Neutral review).
    # In each batch, other items are implicit in-batch negatives — no pre-sampling needed.
    train_pos_df = train_df[train_df["label"] > 0.0].copy()

    # Merge RLHF feedback signal — upsamples purchases/clicks from real user actions
    train_pos_df = _merge_feedback(train_pos_df, feedback_df)

    train_pos_dataset = EcommerceRecSysDataset(train_pos_df)
    valid_dataset = EcommerceRecSysDataset(valid_df)

    # =====================================================
    # DATALOADER
    # =====================================================

    train_loader = DataLoader(
        train_pos_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # =====================================================
    # MODEL
    # =====================================================

    num_users = len(user_encoder.classes_) + 1
    num_items = len(item_encoder.classes_) + 1
    num_categories = len(cat_encoder.classes_) + 1

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # =====================================================
    # TRAIN LOOP
    # =====================================================

    best_auc = 0.0

    with mlflow.start_run():

        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("lr", 3e-4)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_metric(
            "n_feedback_events", len(feedback_df) if feedback_df is not None else 0
        )
        mlflow.log_metric("n_train_samples", len(train_pos_df))

        for epoch in range(EPOCHS):

            train_loss, item_collapse_metric = train_one_epoch_infonce(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            valid_loss, valid_auc, valid_recall, valid_ndcg = validate(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
            )

            log_metrics(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                valid_auc=valid_auc,
                recall_at_k=valid_recall,
                ndcg_at_k=valid_ndcg,
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f valid_loss=%.4f auc=%.4f recall@10=%.4f ndcg@10=%.4f item_collapse_cos=%.4f",
                epoch + 1,
                EPOCHS,
                train_loss,
                valid_loss,
                valid_auc,
                valid_recall,
                valid_ndcg,
                item_collapse_metric,
            )
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.4f} "
                f"valid_loss={valid_loss:.4f} auc={valid_auc:.4f} "
                f"recall@10={valid_recall:.4f} ndcg@10={valid_ndcg:.4f} "
                f"item_collapse_cos={item_collapse_metric:.4f}",
                flush=True,
            )
            mlflow.log_metric("item_collapse_cos", item_collapse_metric, step=epoch)
            # > 0.5 = item embeddings crowding onto a narrow cone (near-collapse);
            # a healthy spread on the 64-d hypersphere sits closer to 0.
            if item_collapse_metric > 0.5:
                logger.warning(
                    "Item embedding collapse detected (mean pairwise cosine=%.4f > 0.5) — "
                    "recommendations may look near-identical across users",
                    item_collapse_metric,
                )

            if valid_auc > best_auc:

                best_auc = valid_auc

                save_artifacts(
                    model=model,
                    scaler=scaler,
                    user_encoder=user_encoder,
                    item_encoder=item_encoder,
                    cat_encoder=cat_encoder,
                )

                log_model(model)

                # Register model to MLflow registry (promotion gate in retrain.py
                # will transition the best version to Production stage)
                try:
                    run_id = mlflow.active_run().info.run_id
                    mlflow.register_model(f"runs:/{run_id}/model", "recsys-two-tower")
                    logger.info(
                        "Model registered in MLflow registry (run_id=%s)", run_id
                    )
                except Exception as e:
                    logger.warning("Failed to register model: %s", e)

                logger.info("Best AUC updated: %.4f", best_auc)

    logger.info("Training completed! Best AUC: %.4f", best_auc)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    setup_mlflow()
    main()
