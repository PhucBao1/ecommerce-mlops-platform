# =========================================================
# FILE: loaders.py
# =========================================================

import logging
import os
import socket
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

from src.serving.recsys_api.model import TwoTowerModel
from src.serving.recsys_api.reranker import set_item_popularity, set_sentiment_prior
from src.serving.recsys_api.utils import safe_transform
from src.serving.recsys_api.vector_store import get_vector_store

logger = logging.getLogger(__name__)

BASE_DIR = Path("/app")

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# READINESS
#
# Mọi thứ bên dưới vẫn chạy lúc import module như cũ (retrieval.py/
# inference.py/inference_faiss.py đều `from loaders import <tên>` ngay lúc
# CHÍNH CÁC FILE ĐÓ được import — chuyển hẳn sang lifespan async sẽ phải
# sửa cả 4 file, không chỉ file này). Cái thay đổi: nếu load lỗi giữa
# chừng, exception được bắt lại ở cuối file thay vì rơi thẳng ra ngoài làm
# crash cả quá trình import — READY/LOAD_ERROR cho main.py biết trạng thái
# thật để trả lời /ready đúng, thay vì để cả process chết không rõ lý do.
# =========================================================

READY = False
LOAD_ERROR: str | None = None

# Giá trị mặc định an toàn — sẽ bị ghi đè bởi phần load bên dưới nếu thành
# công. Cần khai báo trước vì `inference.py` dùng `from loaders import *`,
# nếu load lỗi giữa chừng mà biến chưa từng được gán thì file đó sẽ crash
# với ImportError ở một nơi khác, khó truy vết hơn nhiều so với đọc thẳng
# LOAD_ERROR ở đây.
user_encoder = None
item_encoder = None
cat_encoder = None
scaler = None
item_lookup_df = pd.DataFrame()
df_history = pd.DataFrame()
global_avg_price = 0.0
USER_MAPPING: dict = {}
ITEM_MAPPING: dict = {}
CAT_MAPPING: dict = {}
model = None
ALL_ITEM_VECTORS = None
ALL_ITEM_IDS = np.array([])
VECTOR_STORE = None
TRENDING_ITEM_IDS: list = []
USER_HISTORY_DICT: dict = {}
LIGHTGCN_USER_EMBEDDINGS: np.ndarray | None = None
LIGHTGCN_ITEM_EMBEDDINGS: np.ndarray | None = None
LIGHTGCN_USER_IDS: list | None = None
LIGHTGCN_ITEM_IDS: list | None = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _download_mlflow_weights(run_id: str) -> str:
    return mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/data/model.pth"
    )


def _mlflow_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_model_weights(m: TwoTowerModel, dev) -> TwoTowerModel:
    """Load from MLflow Production registry; fall back to local .pt file."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    host = mlflow_uri.split("//")[-1].split(":")[0]
    port = (
        int(mlflow_uri.split(":")[-1].rstrip("/"))
        if ":" in mlflow_uri.split("//")[-1]
        else 80
    )

    if _mlflow_reachable(host, port):
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            client = mlflow.MlflowClient()
            prod = client.get_latest_versions("recsys-two-tower", stages=["Production"])
            if prod:
                local_path = _download_mlflow_weights(prod[0].run_id)
                m.load_state_dict(torch.load(local_path, map_location=dev))
                logger.info(
                    "Loaded Production model from MLflow (run_id=%s)", prod[0].run_id
                )
                return m
        except Exception as e:
            logger.warning("MLflow registry unavailable: %s — falling back to local", e)
    else:
        logger.warning("MLflow at %s not reachable — skipping registry", mlflow_uri)

    local_model_path = BASE_DIR / "artifacts/recsys_models/model/best_two_tower.pt"
    m.load_state_dict(torch.load(local_model_path, map_location=dev))
    logger.info("Loaded model from local path (fallback)")
    return m


def _load_artifacts() -> None:
    """Load every recsys artifact. Raises on failure — caller (module bottom)
    catches it, sets READY/LOAD_ERROR instead of crashing the import."""
    global user_encoder, item_encoder, cat_encoder, scaler
    global item_lookup_df, df_history, global_avg_price
    global USER_MAPPING, ITEM_MAPPING, CAT_MAPPING
    global model, ALL_ITEM_VECTORS, ALL_ITEM_IDS, VECTOR_STORE
    global TRENDING_ITEM_IDS, USER_HISTORY_DICT
    global LIGHTGCN_USER_EMBEDDINGS, LIGHTGCN_ITEM_EMBEDDINGS
    global LIGHTGCN_USER_IDS, LIGHTGCN_ITEM_IDS

    logger.info("Loading artifacts...")

    # =========================================================
    # LOAD ENCODERS
    # =========================================================
    user_encoder = joblib.load(
        BASE_DIR / "artifacts/recsys_models/encoders/user_encoder.pkl"
    )

    item_encoder = joblib.load(
        BASE_DIR / "artifacts/recsys_models/encoders/item_encoder.pkl"
    )

    cat_encoder = joblib.load(
        BASE_DIR / "artifacts/recsys_models/encoders/cat_encoder.pkl"
    )

    scaler = joblib.load(BASE_DIR / "artifacts/recsys_models/scalers/scaler.pkl")

    # =========================================================
    # LOAD DATA
    # =========================================================
    item_lookup_df = pd.read_parquet(
        BASE_DIR / "artifacts/recsys_models/data_menu/item_lookup.parquet"
    )

    df_history = pd.read_parquet(
        BASE_DIR / "artifacts/recsys_models/data_menu/user_history.parquet"
    )

    # =========================================================
    # FIX TYPES
    # =========================================================

    df_history["customer_id"] = (
        df_history["customer_id"].astype(str).str.replace(".0", "", regex=False)
    )

    item_lookup_df["product_id"] = item_lookup_df["product_id"].astype(str)

    item_lookup_df["category_id"] = item_lookup_df["category_id"].astype(str)

    # =========================================================
    # GLOBALS
    # =========================================================

    global_avg_price = df_history["avg_price_preference"].mean()

    # Prior cho Bayesian sentiment trong reranker — phải là trung bình của
    # CATALOG THẬT, không phải tính từ tập candidate của từng request (làm vậy
    # thì prior bị chính outlier trong tập đó kéo lệch và mất tác dụng).
    if "item_review_count" in item_lookup_df.columns:
        rated = item_lookup_df[item_lookup_df["item_review_count"] > 0]
        if len(rated) > 0:
            set_sentiment_prior(rated["avg_item_sentiment"].mean())

    # popularity_norm feature cho LTR model (train_ltr.py) — đếm lượt
    # mua/review thật theo product_id từ user_history.parquet.
    set_item_popularity(df_history["product_id"].astype(str).value_counts().to_dict())

    # =========================================================
    # MAPPINGS
    # =========================================================

    USER_MAPPING = {
        str(cls).replace(".0", ""): idx for idx, cls in enumerate(user_encoder.classes_)
    }

    ITEM_MAPPING = {cls: idx for idx, cls in enumerate(item_encoder.classes_)}

    CAT_MAPPING = {cls: idx for idx, cls in enumerate(cat_encoder.classes_)}

    # =========================================================
    # MODEL — load from MLflow Production registry, fallback to local
    # =========================================================

    num_users = len(user_encoder.classes_) + 1
    num_items = len(item_encoder.classes_) + 1
    num_categories = len(cat_encoder.classes_) + 1

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_categories=num_categories,
        embedding_dim=32,
    )

    model = _load_model_weights(model, device)
    model.to(device)
    model.eval()

    logger.info("Model loaded!")

    # =========================================================
    # PRECOMPUTE ITEM EMBEDDINGS
    # =========================================================
    logger.info("Precomputing item vectors...")

    item_df = item_lookup_df.copy()

    item_df["product_id_idx"] = safe_transform(ITEM_MAPPING, item_df["product_id"])

    item_df["category_id_idx"] = safe_transform(
        CAT_MAPPING, item_df["category_id"].astype(str)
    )

    scaled_item_features = scaler.transform(
        pd.DataFrame(
            {
                "total_reviews_so_far": 0,
                "avg_price_preference": 0,
                "positive_review_ratio": 0,
                "price": item_df["price"],
                "avg_item_sentiment": item_df["avg_item_sentiment"],
            }
        )
    )

    item_num_features = scaled_item_features[:, 3:5]

    with torch.no_grad():

        item_batch = {
            "item_id": torch.tensor(
                item_df["product_id_idx"].values, dtype=torch.long
            ).to(device),
            "item_category": torch.tensor(
                item_df["category_id_idx"].values, dtype=torch.long
            ).to(device),
            "item_num": torch.tensor(item_num_features, dtype=torch.float32).to(device),
        }

        ALL_ITEM_VECTORS = model.item_tower(
            item_id=item_batch["item_id"],
            item_category=item_batch["item_category"],
            item_num=item_batch["item_num"],
        )

        ALL_ITEM_VECTORS = ALL_ITEM_VECTORS.cpu()

    ALL_ITEM_IDS = item_df["product_id"].values

    logger.info("Item embeddings ready!")

    # =========================================================
    # VECTOR STORE — index ALL_ITEM_VECTORS for ANN search
    # Backed by FAISS (dev) or Qdrant (prod) via VECTOR_STORE_BACKEND env var
    # =========================================================

    VECTOR_STORE = get_vector_store()

    _payloads = [
        {
            "product_id": row["product_id"],
            "price": float(row.get("price", 0)),
            "category_id": str(row.get("category_id", "")),
        }
        for _, row in item_df.iterrows()
    ]

    VECTOR_STORE.upsert(
        ids=list(ALL_ITEM_IDS),
        vectors=ALL_ITEM_VECTORS.numpy(),
        payloads=_payloads,
    )
    logger.info(
        "Vector store indexed %d items (backend=%s)",
        len(ALL_ITEM_IDS),
        os.getenv("VECTOR_STORE_BACKEND", "faiss"),
    )

    # =========================================================
    # TRENDING ITEMS — cold start fallback
    # Top 200 items by purchase frequency in last 30 days
    # =========================================================

    _cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    _recent = (
        df_history[df_history["purchased_at"] >= _cutoff]
        if "purchased_at" in df_history.columns
        else df_history
    )
    _trending = (
        _recent["product_id"].value_counts().head(200).index.astype(str).tolist()
    )
    TRENDING_ITEM_IDS = (
        _trending
        if _trending
        else item_lookup_df.nlargest(200, "avg_item_sentiment")["product_id"]
        .astype(str)
        .tolist()
    )

    logger.info("Trending items computed: %d items", len(TRENDING_ITEM_IDS))

    # =========================================================
    # USER HISTORY DICT
    # =========================================================

    logger.info("Building user history dictionary...")

    USER_HISTORY_DICT = {
        k: v.sort_values("purchased_at") for k, v in df_history.groupby("customer_id")
    }

    logger.info("User history dictionary ready!")

    # =========================================================
    # LIGHTGCN EMBEDDINGS — loaded lazily when RETRIEVAL_BACKEND=lightgcn
    # =========================================================

    _RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "twotower")

    if _RETRIEVAL_BACKEND == "lightgcn":
        _lgcn_path = BASE_DIR / "artifacts/recsys_models/lightgcn_embeddings.npz"
        if _lgcn_path.exists():
            _lgcn_data = np.load(str(_lgcn_path), allow_pickle=True)
            LIGHTGCN_USER_EMBEDDINGS = _lgcn_data["user_embeddings"]
            LIGHTGCN_ITEM_EMBEDDINGS = _lgcn_data["item_embeddings"]
            LIGHTGCN_USER_IDS = list(_lgcn_data["user_ids"])
            LIGHTGCN_ITEM_IDS = list(_lgcn_data["item_ids"])
            logger.info(
                "LightGCN embeddings loaded: %d users, %d items",
                len(LIGHTGCN_USER_IDS),
                len(LIGHTGCN_ITEM_IDS),
            )
        else:
            logger.warning(
                "RETRIEVAL_BACKEND=lightgcn but %s not found — falling back to "
                "Two-Tower",
                _lgcn_path,
            )


try:
    _load_artifacts()
    READY = True
    logger.info("recsys_artifacts_ready")
except Exception:
    LOAD_ERROR = "artifact loading failed — see traceback in logs above"
    logger.error("recsys_artifacts_load_failed", exc_info=True)
    # Cố tình KHÔNG raise lại: để module import xong, FastAPI vẫn tạo được
    # app object — /health trả lời "alive" bình thường, /ready báo đúng
    # 503 (chưa sẵn sàng) thay vì cả process crash-loop không rõ lý do.
