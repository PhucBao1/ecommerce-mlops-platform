import mlflow
import numpy as np
import pandas as pd

from src.ml_models.recsys.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.serving.recsys_api.inference import recommend

TOP_K = 10

USER_HISTORY_PATH = "artifacts/recsys_models/data_menu/" "user_history.parquet"

df = pd.read_parquet(USER_HISTORY_PATH)

df["customer_id"] = df["customer_id"].astype(str)

df["product_id"] = df["product_id"].astype(str)

# =====================================================
# SORT
# =====================================================

sort_col = None

for col in ["purchased_at", "created_at", "review_timestamp", "event_timestamp"]:
    if col in df.columns:
        sort_col = col
        break

if sort_col is None:

    raise ValueError("Cannot find timestamp column")

df = df.sort_values(["customer_id", sort_col])

# =====================================================
# EVALUATION
# =====================================================

recalls = []
precisions = []
ndcgs = []

users = df["customer_id"].unique()

print(f"Evaluating {len(users)} users...")

for customer_id in users:

    user_df = df[df["customer_id"] == customer_id]

    # cần ít nhất 2 interaction
    if len(user_df) < 2:
        continue

    # ==========================================
    # HOLDOUT LAST ITEM
    # ==========================================

    train_history = user_df.iloc[:-1]

    test_item = str(user_df.iloc[-1]["product_id"])

    actual = [test_item]

    try:

        result = recommend(
            customer_id=customer_id, top_k=TOP_K, history_override=train_history
        )

        predicted = [str(x["product_id"]) for x in result["recommendations"]]

        recalls.append(recall_at_k(actual, predicted, TOP_K))

        precisions.append(precision_at_k(actual, predicted, TOP_K))

        ndcgs.append(ndcg_at_k(actual, predicted, TOP_K))

    except Exception as e:

        print(f"User {customer_id} failed: {e}")

# =====================================================
# RESULT
# =====================================================

print("\n" + "=" * 60)

print(f"Users evaluated: " f"{len(recalls)}")

print(f"Recall@{TOP_K}: " f"{np.mean(recalls):.4f}")

print(f"Precision@{TOP_K}: " f"{np.mean(precisions):.4f}")

print(f"NDCG@{TOP_K}: " f"{np.mean(ndcgs):.4f}")

print("=" * 60)

mlflow.set_experiment("recsys_offline_eval")

with mlflow.start_run():

    mlflow.log_metric("recall_at_10", float(np.mean(recalls)))

    mlflow.log_metric("precision_at_10", float(np.mean(precisions)))

    mlflow.log_metric("ndcg_at_10", float(np.mean(ndcgs)))
