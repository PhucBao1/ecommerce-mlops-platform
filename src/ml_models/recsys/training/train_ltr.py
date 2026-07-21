"""
LTR (Learning to Rank) training with XGBoost LambdaMART.

Stage 2 reranker: takes Two-Tower top-N candidates and reranks with
rich features to produce top-K final recommendations. Trained model replaces
the hand-tuned weighted-sum formula in recsys_api/reranker.py (0.05*price +
0.03*sentiment + 0.02*quality) with weights actually learned from real
interaction data.

Bug thật 17/7/2026 (BENCHMARK_RESULTS.md mục 13/14): bản trước KHÔNG chạy
được — `history["action"]` (purchase/click/ignore) không tồn tại trong
user_history.parquet thật (schema thật: customer_id, product_id,
purchased_at, total_reviews_so_far, avg_price_preference,
positive_review_ratio, has_history — không có "action"). Nguồn dữ liệu
click/ignore đúng cách (`/feedback` → Kafka `user_actions` →
feedback_processor.py) tồn tại trong code nhưng KHÔNG được deploy (không có
trong bất kỳ docker-compose nào) — và kể cả có cũng ghi vào Feast (online
feature) chứ không phải log lịch sử dùng để train offline. `semantic_score`/
`tfidf_score`/`rrf_score` cũ bị fill cứng 0.0 (3/7 feature vô dụng).

Fix trong bản này:
  - Positive = mỗi dòng trong user_history.parquet (review/mua thật).
  - Negative = random sampling item KHÔNG nằm trong tập positive của khách đó
    (cùng pattern đã dùng + verify trong train_sasrec.py).
  - semantic_score = cosine similarity THẬT giữa user_vector/item_vector từ
    chính Two-Tower model đã train (best_two_tower.pt), không fill giả.
  - category_complement (MỚI) = 1 nếu category của item là phụ kiện đi kèm
    category khách đã tương tác gần đây (vd điện thoại → phụ kiện điện
    thoại) — gộp yêu cầu cross-sell vào làm 1 feature cho model tự học
    trọng số, thay vì hard-code boost tay trong inference_faiss.py.

Data reality check: user_history.parquet — 78,076 customer, chỉ 13,219
(~17%) có >=2 review dùng được để suy ra "category đã tương tác trước đó"
đáng tin; phần còn lại category_match/category_complement sẽ =0 cho khách
mới (đúng, không giả định gì thêm). Đây là giới hạn quy mô data (catalog
1744 item/3 category), không phải giới hạn của cách làm — cùng khuyến nghị
đã note trong BENCHMARK_RESULTS.md: xử lý bronze 35k-item sẽ cải thiện trực
tiếp NDCG khi có nhiều sequence dài hơn.

Usage:
    python -m src.ml_models.recsys.training.train_ltr
    python -m src.ml_models.recsys.training.train_ltr --n-estimators 500 --max-depth 6
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_DATA_DIR = os.getenv("DATA_DIR", "artifacts/recsys_models/data_menu")
_ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts/recsys_models")
_MODEL_OUT = os.getenv("LTR_MODEL_PATH", "artifacts/recsys_models/ltr_model.json")
_SEED = 42
_N_NEGATIVES_PER_POSITIVE = 4  # cùng bậc với negative sampling của SASRec

_FEATURE_COLS = [
    "semantic_score",
    "bayes_sentiment",
    "price_closeness",
    "category_match",
    "category_complement",
    "popularity_norm",
    "sasrec_score",
]
_LABEL_COL = "relevance"
_GROUP_COL = "customer_id"

# Cross-sell category map + sasrec_score — định nghĩa DUY NHẤT tại
# reranker.py (serving), import lại ở đây để training/serving luôn khớp
# nhau, tránh 2 nơi định nghĩa lệch.
from src.serving.recsys_api.reranker import (
    CATEGORY_COMPLEMENTS as _CATEGORY_COMPLEMENTS,
)

# noqa: E402
from src.serving.recsys_api.reranker import (
    compute_sasrec_scores_batch as _compute_sasrec_scores_batch,
)

_BAYES_PRIOR_VOTES = 5


# ---------------------------------------------------------------------------
# Two-Tower model loading (độc lập với recsys_api/loaders.py — tránh side
# effect kết nối Qdrant/MLflow registry lúc import, script này chỉ cần đọc
# trọng số cục bộ để tính embedding, không cần phục vụ request).
# ---------------------------------------------------------------------------


def _load_two_tower():
    from src.serving.recsys_api.model import TwoTowerModel

    user_encoder = joblib.load(f"{_ARTIFACT_DIR}/encoders/user_encoder.pkl")
    item_encoder = joblib.load(f"{_ARTIFACT_DIR}/encoders/item_encoder.pkl")
    cat_encoder = joblib.load(f"{_ARTIFACT_DIR}/encoders/cat_encoder.pkl")
    scaler = joblib.load(f"{_ARTIFACT_DIR}/scalers/scaler.pkl")

    model = TwoTowerModel(
        num_users=len(user_encoder.classes_) + 1,
        num_items=len(item_encoder.classes_) + 1,
        num_categories=len(cat_encoder.classes_) + 1,
        embedding_dim=32,
    )
    model.load_state_dict(
        torch.load(f"{_ARTIFACT_DIR}/model/best_two_tower.pt", map_location="cpu")
    )
    model.eval()

    user_mapping = {
        str(c).replace(".0", ""): i for i, c in enumerate(user_encoder.classes_)
    }
    item_mapping = {c: i for i, c in enumerate(item_encoder.classes_)}
    cat_mapping = {c: i for i, c in enumerate(cat_encoder.classes_)}
    return model, scaler, user_mapping, item_mapping, cat_mapping


def _safe_map(mapping: dict, values) -> np.ndarray:
    unknown = len(mapping)
    return pd.Series(values).map(mapping).fillna(unknown).astype(int).values


def _compute_item_vectors(
    model, scaler, item_lookup: pd.DataFrame, item_mapping, cat_mapping
):
    item_idx = _safe_map(item_mapping, item_lookup["product_id"])
    cat_idx = _safe_map(cat_mapping, item_lookup["category_id"].astype(str))

    scaled = scaler.transform(
        pd.DataFrame(
            {
                "total_reviews_so_far": 0,
                "avg_price_preference": 0,
                "positive_review_ratio": 0,
                "price": item_lookup["price"],
                "avg_item_sentiment": item_lookup["avg_item_sentiment"],
            }
        )
    )
    item_num = scaled[:, 3:5]

    with torch.no_grad():
        vecs = model.item_tower(
            item_id=torch.tensor(item_idx, dtype=torch.long),
            item_category=torch.tensor(cat_idx, dtype=torch.long),
            item_num=torch.tensor(item_num, dtype=torch.float32),
        )
    return vecs.numpy(), {pid: i for i, pid in enumerate(item_lookup["product_id"])}


def _compute_user_vector(
    model, scaler, user_mapping, customer_id: str, user_feat_row: dict
):
    user_idx = user_mapping.get(str(customer_id), len(user_mapping))
    scale_input = np.array(
        [
            [
                user_feat_row["total_reviews_so_far"],
                user_feat_row["avg_price_preference"],
                user_feat_row["positive_review_ratio"],
                0.0,
                0.0,
            ]
        ],
        dtype=np.float64,
    )
    scaled = scaler.transform(scale_input)
    user_num = np.concatenate(
        [scaled[:, [0, 1, 2]], np.array([[user_feat_row["has_history"]]])], axis=1
    )
    with torch.no_grad():
        vec = model.user_tower(
            user_id=torch.tensor([user_idx], dtype=torch.long),
            user_num=torch.tensor(user_num, dtype=torch.float32),
        )
    return vec.numpy()[0]


def _bayesian_sentiment(df: pd.DataFrame, prior: float) -> pd.Series:
    if "item_review_count" not in df.columns:
        return df["avg_item_sentiment"]
    v = df["item_review_count"].fillna(0)
    R = df["avg_item_sentiment"].fillna(0.0)
    m = _BAYES_PRIOR_VOTES
    return (v / (v + m)) * R + (m / (v + m)) * prior


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
_CHECKPOINT_ROWS = Path(_ARTIFACT_DIR) / ".ltr_checkpoint_rows.parquet"
_CHECKPOINT_SESSIONS = Path(_ARTIFACT_DIR) / ".ltr_checkpoint_sessions.json"


def _build_ltr_dataset() -> pd.DataFrame:
    if _CHECKPOINT_ROWS.exists() and _CHECKPOINT_SESSIONS.exists():
        import json

        logger.info(
            "Tìm thấy checkpoint (%s) — bỏ qua bước build dataset, load lại thay vì chạy lại từ đầu",
            _CHECKPOINT_ROWS,
        )
        df_rows = pd.read_parquet(_CHECKPOINT_ROWS)
        ckpt = json.loads(_CHECKPOINT_SESSIONS.read_text())
        customer_sessions = ckpt["customer_sessions"]
        customer_candidates = ckpt["customer_candidates"]
        sentiment_prior = ckpt["sentiment_prior"]
        return _finish_ltr_dataset(
            df_rows, customer_sessions, customer_candidates, sentiment_prior
        )

    history = pd.read_parquet(Path(_DATA_DIR) / "user_history.parquet")
    items = pd.read_parquet(Path(_DATA_DIR) / "item_lookup.parquet")
    items["product_id"] = items["product_id"].astype(str)
    items["category_id"] = items["category_id"].astype(str)
    history["product_id"] = history["product_id"].astype(str)
    history["customer_id"] = (
        history["customer_id"].astype(str).str.replace(".0", "", regex=False)
    )

    # customer_id="0" — placeholder fillna artifact từ preprocessing, không
    # phải khách thật (cùng loại trừ đã áp dụng trong train_sasrec.py).
    history = history[history["customer_id"] != "0"].copy()

    rated = (
        items[items.get("item_review_count", 0) > 0]
        if "item_review_count" in items.columns
        else items
    )
    sentiment_prior = float(rated["avg_item_sentiment"].mean()) if len(rated) else 0.185

    popularity = history["product_id"].value_counts()
    pop_max = float(popularity.max()) if len(popularity) else 1.0

    model, scaler, user_mapping, item_mapping, cat_mapping = _load_two_tower()
    item_vectors, item_vec_idx = _compute_item_vectors(
        model, scaler, items, item_mapping, cat_mapping
    )
    item_by_id = items.set_index("product_id")

    rng = np.random.default_rng(_SEED)
    all_product_ids = items["product_id"].tolist()

    rows = []
    customer_sessions: dict = {}
    customer_candidates: dict = {}
    for customer_id, grp in history.groupby("customer_id"):
        positive_ids = set(grp["product_id"])
        bought_cats = set(
            item_by_id.loc[item_by_id.index.intersection(positive_ids), "category_id"]
        )
        complement_cats: set[str] = set()
        for c in bought_cats:
            complement_cats |= _CATEGORY_COMPLEMENTS.get(c, set())

        user_feat_row = grp.iloc[-1][
            [
                "total_reviews_so_far",
                "avg_price_preference",
                "positive_review_ratio",
                "has_history",
            ]
        ].to_dict()
        target_price = float(user_feat_row["avg_price_preference"]) or 0.0
        try:
            u_vec = _compute_user_vector(
                model, scaler, user_mapping, customer_id, user_feat_row
            )
        except Exception:
            continue

        # Negatives: random item KHÔNG nằm trong tập positive của khách này.
        neg_candidates = []
        attempts = 0
        target_n = min(
            _N_NEGATIVES_PER_POSITIVE * len(positive_ids), len(all_product_ids)
        )
        while len(neg_candidates) < target_n and attempts < target_n * 5:
            attempts += 1
            pid = all_product_ids[rng.integers(0, len(all_product_ids))]
            if pid not in positive_ids:
                neg_candidates.append(pid)

        # sasrec_score: dùng TOÀN BỘ lịch sử mua/review thật của khách (sắp
        # xếp cũ→mới theo purchased_at) làm "session" cho SASRec — cùng mức
        # đơn giản hoá với bought_cats/category_match ở trên (không tách
        # leave-one-out point-in-time cho từng dòng, dùng chung 1 session
        # cho cả positive/negative của khách này). CHỈ thu thập ở đây, chấm
        # điểm batch 1 lần cho TẤT CẢ khách sau khi loop này xong (xem dưới)
        # — gọi model từng khách một (493k lần forward pass batch=1) mất
        # ~66 phút đo thử, batch lại nhanh hơn nhiều lần.
        session_items = grp.sort_values("purchased_at")["product_id"].tolist()
        all_candidates_for_customer = list(positive_ids) + neg_candidates
        customer_sessions[customer_id] = session_items
        customer_candidates[customer_id] = all_candidates_for_customer

        for pid, label in [(pid, 1) for pid in positive_ids] + [
            (pid, 0) for pid in neg_candidates
        ]:
            if pid not in item_by_id.index or pid not in item_vec_idx:
                continue
            item_row = item_by_id.loc[pid]
            if isinstance(item_row, pd.DataFrame):  # duplicate product_id guard
                item_row = item_row.iloc[0]

            i_vec = item_vectors[item_vec_idx[pid]]
            semantic_score = float(np.dot(u_vec, i_vec))

            price = float(item_row.get("price", 0) or 0)
            price_closeness = 1.0 / (
                1.0 + abs(price - target_price) / max(target_price, 1.0)
            )

            cat = str(item_row.get("category_id", ""))
            rows.append(
                {
                    "customer_id": customer_id,
                    "product_id": pid,
                    "semantic_score": semantic_score,
                    "avg_item_sentiment": float(
                        item_row.get("avg_item_sentiment", 0) or 0
                    ),
                    "item_review_count": float(
                        item_row.get("item_review_count", 0) or 0
                    ),
                    "price_closeness": price_closeness,
                    "category_match": int(cat in bought_cats),
                    "category_complement": int(cat in complement_cats),
                    "popularity_norm": float(popularity.get(pid, 0)) / pop_max,
                    "relevance": label,
                }
            )

    # Checkpoint NGAY TRƯỚC bước sasrec_score batch (bước nặng memory nhất,
    # dễ bị kill nhất) — nếu crash sau điểm này, lần chạy sau load lại đây
    # thay vì lặp lại ~30-40 phút build dataset ở trên.
    df_rows = pd.DataFrame(rows)
    df_rows.to_parquet(_CHECKPOINT_ROWS)
    import json

    _CHECKPOINT_SESSIONS.write_text(
        json.dumps(
            {
                "customer_sessions": customer_sessions,
                "customer_candidates": customer_candidates,
                "sentiment_prior": sentiment_prior,
            }
        )
    )
    logger.info("Đã lưu checkpoint tại %s", _CHECKPOINT_ROWS)

    return _finish_ltr_dataset(
        df_rows, customer_sessions, customer_candidates, sentiment_prior
    )


def _finish_ltr_dataset(
    df_rows: pd.DataFrame,
    customer_sessions: dict,
    customer_candidates: dict,
    sentiment_prior: float,
) -> pd.DataFrame:
    logger.info(
        "Chấm sasrec_score batch cho %d khách (%d candidate/khách trung bình)...",
        len(customer_sessions),
        sum(len(v) for v in customer_candidates.values())
        // max(len(customer_candidates), 1),
    )
    batch_scores = _compute_sasrec_scores_batch(customer_sessions, customer_candidates)

    df = df_rows.copy()
    df["sasrec_score"] = [
        batch_scores.get(cid, {}).get(pid, 0.0)
        for cid, pid in zip(df["customer_id"], df["product_id"])
    ]
    df["bayes_sentiment"] = _bayesian_sentiment(df, sentiment_prior)
    result = df[_FEATURE_COLS + [_LABEL_COL, _GROUP_COL]].dropna(subset=_FEATURE_COLS)

    # Thành công hết → xoá checkpoint (không cần giữ nữa, tránh lần sau vô
    # tình dùng data cũ nếu user_history/item_lookup đã đổi).
    _CHECKPOINT_ROWS.unlink(missing_ok=True)
    _CHECKPOINT_SESSIONS.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_ltr(n_estimators: int = 300, max_depth: int = 6, lr: float = 0.1) -> float:
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("ltr_training")

    df = _build_ltr_dataset()
    logger.info(
        "LTR dataset: %d rows (%d positive, %d negative), %d users",
        len(df),
        int((df[_LABEL_COL] == 1).sum()),
        int((df[_LABEL_COL] == 0).sum()),
        df[_GROUP_COL].nunique(),
    )

    groups = df[_GROUP_COL].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=_SEED)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].sort_values(_GROUP_COL)
    val_df = df.iloc[val_idx].sort_values(_GROUP_COL)

    X_train = train_df[_FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[_LABEL_COL].values.astype(int)
    g_train = train_df.groupby(_GROUP_COL, sort=False).size().values

    X_val = val_df[_FEATURE_COLS].values.astype(np.float32)
    y_val = val_df[_LABEL_COL].values.astype(int)
    g_val = val_df.groupby(_GROUP_COL, sort=False).size().values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=_FEATURE_COLS)
    dtrain.set_group(g_train)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=_FEATURE_COLS)
    dval.set_group(g_val)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": lr,
        "max_depth": max_depth,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": _SEED,
    }

    with mlflow.start_run(run_name="ltr_lambdamart"):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": lr,
                "n_features": len(_FEATURE_COLS),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_users_train": train_df[_GROUP_COL].nunique(),
                "n_users_val": val_df[_GROUP_COL].nunique(),
            }
        )

        evals_result: dict = {}
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=30,
            verbose_eval=50,
        )

        best_ndcg = max(evals_result["val"]["ndcg@10"])
        mlflow.log_metric("best_val_ndcg10", best_ndcg)
        mlflow.log_metric("best_iteration", model.best_iteration)

        importance = model.get_score(importance_type="gain")
        for feat, score in importance.items():
            mlflow.log_metric(f"fi_{feat}", round(score, 4))
        logger.info("Feature importance (gain): %s", importance)

        Path(_MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(_MODEL_OUT)
        # File local đã lưu xong ở dòng trên — đây là artifact serving THẬT
        # SỰ dùng. Upload MLflow chỉ để tracking/hiển thị, lỗi artifact-store
        # (MinIO 500, transient) không được phép làm mất kết quả training đã
        # chạy xong (bug thật gặp 17/7/2026: log_artifact lỗi khiến cả hàm
        # crash trước khi return, dù model đã train + save xong).
        try:
            mlflow.log_artifact(_MODEL_OUT, "ltr_model")
        except Exception as exc:
            logger.warning(
                "mlflow.log_artifact thất bại (không chặn training): %s", exc
            )
        logger.info("LTR model saved to %s (NDCG@10=%.4f)", _MODEL_OUT, best_ndcg)

    return best_ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    ndcg = train_ltr(
        n_estimators=args.n_estimators, max_depth=args.max_depth, lr=args.lr
    )
    print(f"\nBest NDCG@10: {ndcg:.4f}")
