import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_explanation(row: pd.Series, user_features: dict) -> dict:
    """Compute human-readable explanation for why this item was recommended."""
    target_price = user_features.get("avg_price_preference", 0)
    sentiment = float(row.get("bayes_sentiment", row.get("avg_item_sentiment", 0)))

    # Price match: 100% = exact match, 0% = very far
    if target_price > 0:
        price_match_pct = max(
            0.0, 100.0 - abs(float(row["price"]) - target_price) / target_price * 100
        )
    else:
        price_match_pct = 50.0

    factors = []
    if price_match_pct >= 70:
        factors.append(f"Phù hợp tầm giá ({price_match_pct:.0f}%)")
    if sentiment >= 0.7:
        factors.append(f"Đánh giá tốt ({sentiment * 5:.1f}/5⭐)")
    elif sentiment >= 0.5:
        factors.append(f"Đánh giá khá ({sentiment * 5:.1f}/5⭐)")
    if row.get("category_complement"):
        factors.append("Phụ kiện phù hợp với sản phẩm bạn từng mua")

    top_reason = factors[0] if factors else "Phù hợp sở thích cá nhân"

    return {
        "top_reason": top_reason,
        "model_confidence": round(float(row.get("predict_score", 0)), 3),
        "price_match_pct": round(price_match_pct, 1),
        "sentiment_score": round(sentiment, 3),
        "factors": factors[:2],
    }


_BAYES_PRIOR_VOTES = 5  # m: số review "ảo" kéo sentiment về mức trung bình chung

# C trong công thức Bayesian: trung bình sentiment của TOÀN CATALOG.
#
# PHẢI là hằng số ổn định, KHÔNG được tính từ tập candidate đang xét. Bản đầu
# tiên tính C = mean của chính candidates_df và nó vô dụng: với 2 item
# [1 review, sentiment 1.0] và [500 review, sentiment 0.9] thì C = 0.95 (bị
# chính cái outlier kéo lên), co 1.0 về 0.95 ra 0.958 — vẫn thắng item 500
# review (0.900), tức là đúng cái bug đang muốn sửa. Test bắt được.
#
# Giá trị thật của catalog hiện tại ~0.185 (1130/1744 item có review). Được
# set_sentiment_prior() ghi đè lúc load artifact để khỏi hardcode lệch dữ liệu.
_sentiment_prior = 0.185

# Popularity (đếm lượt mua/review theo product_id) — set_item_popularity() ghi
# đè lúc load artifact (loaders.py), cùng pattern với set_sentiment_prior().
_item_popularity: dict = {}
_popularity_max = 1.0


def set_sentiment_prior(mean: float) -> None:
    """Gọi lúc load item_lookup — prior phải khớp catalog thật, không phải số cứng."""
    global _sentiment_prior
    _sentiment_prior = float(mean)


def set_item_popularity(popularity: dict) -> None:
    """Gọi lúc load user_history — dùng cho feature popularity_norm của LTR model."""
    global _item_popularity, _popularity_max
    _item_popularity = popularity
    _popularity_max = float(max(popularity.values())) if popularity else 1.0


def _bayesian_sentiment(df):
    if "item_review_count" not in df.columns:
        return df["avg_item_sentiment"]

    v = df["item_review_count"].fillna(0)
    R = df["avg_item_sentiment"].fillna(0.0)
    m = _BAYES_PRIOR_VOTES
    return (v / (v + m)) * R + (m / (v + m)) * _sentiment_prior


# Cross-sell: category điện thoại/máy tính bảng -> category phụ kiện đi kèm.
# category_id lấy từ item_lookup.parquet thật (xem BENCHMARK_RESULTS.md mục
# 13) — chỉ dùng category phụ kiện ĐIỆN THOẠI cụ thể, KHÔNG dùng "Thiết Bị
# Số - Phụ Kiện Số" (1815, 631 item, quá rộng, lẫn phụ kiện laptop không
# liên quan — cùng bài học rút ra từ bug #18, sparse fusion cho search).
# Dùng chung định nghĩa này cho cả training (train_ltr.py import từ đây) và
# serving, tránh 2 nơi định nghĩa lệch nhau.
PHONE_CATEGORIES = {"1789", "1795", "1794"}
PHONE_ACCESSORY_CATEGORIES = {
    "8214",  # Phụ Kiện Điện Thoại và Máy Tính Bảng
    "28484",  # Miếng Dán Màn Hình Điện Thoại
    "28574",  # Bao Da - Ốp Lưng Điện Thoại iPhone
    "28576",  # Bao Da - Ốp Lưng Điện Thoại Samsung
    "5006",  # Dây Cáp Sạc iPhone, iPad
    "1820",  # Thẻ Nhớ Điện Thoại
    "28460",  # Bộ Adapter Sạc Kèm Cáp Sạc
    "1804",  # Tai Nghe Có Dây — nghe nhạc/gọi điện qua điện thoại
    "1811",  # Tai Nghe Bluetooth
    "8400",  # Tai Nghe True Wireless
    "4428",  # Tai Nghe Có Dây Nhét Tai
    "5531",  # Tai Nghe Bluetooth Nhét Tai
    "2324",  # Loa Bluetooth — dùng kèm điện thoại
}

# Laptop/PC (1846, category lớn thứ 2 catalog, 561 item) -> phụ kiện laptop cụ
# thể. Cùng nguyên tắc như phone: KHÔNG dùng "Thiết Bị Số - Phụ Kiện Số"
# (1815) vì quá rộng, chỉ chọn category linh kiện/phụ kiện PC/laptop rõ ràng.
LAPTOP_CATEGORIES = {"1846"}
LAPTOP_ACCESSORY_CATEGORIES = {
    "1838",  # Chuột Văn Phòng Không Dây
    "1831",  # Bàn Di Chuột - Miếng Lót Chuột
    "28682",  # Giá Đỡ Laptop
    "28696",  # Bộ Phím Chuột Chơi Game
    "1828",  # USB
    "1827",  # Ổ Cứng Di Động
    "11958",  # Card Màn Hình - VGA
    "11956",  # Mainboard - Board Mạch Chủ
    "11960",  # Vỏ Case - Thùng Máy
    "28900",  # Nguồn Máy Tính
    "12672",  # Màn Hình Gaming
    "28932",  # Màn Hình Phổ Thông
    "28930",  # Màn Hình Đồ Họa
}

CATEGORY_COMPLEMENTS: dict = {
    cat: PHONE_ACCESSORY_CATEGORIES for cat in PHONE_CATEGORIES
}
CATEGORY_COMPLEMENTS.update(
    {cat: LAPTOP_ACCESSORY_CATEGORIES for cat in LAPTOP_CATEGORIES}
)

# Tính THÊM cặp category hay mua CÙNG NHAU từ chính
# `user_history.parquet` thật (market-basket analysis — lift = P(A,B) /
# (P(A)·P(B)), lọc lift≥1.5 + tối thiểu 20 khách mua cả 2 + mỗi category
# nguồn tối thiểu 30 khách để tránh nhiễu mẫu nhỏ) — xem script
# `build_category_complements.py`. Kết quả: 189 category có complement thật
# (vs 4 hard-code), ví dụ Sách tiếng Việt → Sách hướng nghiệp/Kỹ năng sống,
# Đồ Chơi Mẹ&Bé → Sữa bột/Sữa công thức. Giữ nguyên mapping tay cho
# phone/laptop (đã curate kỹ, tránh category quá rộng — bug #18 cũ), chỉ bổ
# sung cho các category KHÔNG có trong mapping tay.
_COMPLEMENTS_PATH = os.getenv(
    "CATEGORY_COMPLEMENTS_PATH",
    "/app/artifacts/recsys_models/category_complements.json",
)
try:
    import json as _json

    with open(_COMPLEMENTS_PATH, encoding="utf-8") as _f:
        _data_driven_complements = _json.load(_f)
    for _cat, _comps in _data_driven_complements.items():
        if _cat not in CATEGORY_COMPLEMENTS:
            CATEGORY_COMPLEMENTS[_cat] = set(_comps)
    logger.info(
        "category_complements_loaded data_driven=%d hardcoded=%d total=%d",
        len(_data_driven_complements),
        len(PHONE_CATEGORIES) + len(LAPTOP_CATEGORIES),
        len(CATEGORY_COMPLEMENTS),
    )
except FileNotFoundError:
    logger.warning(
        "category_complements.json không tìm thấy tại %s — chỉ dùng mapping "
        "tay phone/laptop (%d category). Chạy build_category_complements.py "
        "để sinh file này từ user_history.parquet thật.",
        _COMPLEMENTS_PATH,
        len(CATEGORY_COMPLEMENTS),
    )


# =====================================================
# LTR MODEL (XGBoost LambdaMART) — train_ltr.py
# =====================================================

_LTR_MODEL_PATH = os.getenv(
    "LTR_MODEL_PATH", "/app/artifacts/recsys_models/ltr_model.json"
)
_LTR_FEATURE_COLS = [
    "semantic_score",
    "bayes_sentiment",
    "price_closeness",
    "category_match",
    "category_complement",
    "popularity_norm",
    "sasrec_score",
]
_ltr_model = None
_ltr_load_attempted = False

# =====================================================
# SASREC — feature phụ cho LTR (sasrec_score), KHÔNG phải endpoint riêng.
# Two-Tower (semantic_score) nắm bắt sở thích DÀI HẠN của khách (feature
# tổng hợp total_reviews/avg_price/positive_ratio), còn SASRec nắm bắt Ý
# ĐỊNH THEO PHIÊN/THỨ TỰ gần đây — cộng thêm feature này để XGBoost tự học
# trọng số kết hợp 2 tín hiệu, thay vì tự đoán tay 1 công thức blend cố định.
# Cùng pattern lazy-load với _get_ltr_model() — trả None nếu chưa train
# SASRec (vd môi trường dev/test), rerank_candidates() fallback sasrec_score=0.
# =====================================================

_SASREC_MODEL_PATH = os.getenv(
    "SASREC_MODEL_PATH", "/app/artifacts/recsys_models/sasrec.pt"
)
_SASREC_ENC_PATH = os.getenv(
    "SASREC_ENC_PATH", "/app/artifacts/recsys_models/sasrec_item_enc.json"
)
_SASREC_MAX_SEQ = 50
_sasrec_model = None
_sasrec_item_enc: dict | None = None
_sasrec_load_attempted = False


def _get_sasrec():
    """Lazy-load SASRec model + item encoder. Trả (None, None) nếu chưa có
    file (chưa train, hoặc môi trường dev/test) — không chặn rerank."""
    global _sasrec_model, _sasrec_item_enc, _sasrec_load_attempted
    if _sasrec_load_attempted:
        return _sasrec_model, _sasrec_item_enc
    _sasrec_load_attempted = True
    if not os.path.exists(_SASREC_MODEL_PATH) or not os.path.exists(_SASREC_ENC_PATH):
        logger.warning(
            "SASRec model không tìm thấy tại %s — sasrec_score sẽ = 0.0 cho mọi item",
            _SASREC_MODEL_PATH,
        )
        return None, None
    try:
        import json

        import torch

        from src.ml_models.recsys.models.sasrec import SASRec

        item_enc = json.loads(open(_SASREC_ENC_PATH).read())
        model = SASRec(n_items=len(item_enc), max_seq_len=_SASREC_MAX_SEQ)
        model.load_state_dict(torch.load(_SASREC_MODEL_PATH, map_location="cpu"))
        model.eval()
        logger.info(
            "sasrec_model_loaded path=%s n_items=%d", _SASREC_MODEL_PATH, len(item_enc)
        )
        _sasrec_model, _sasrec_item_enc = model, item_enc
        return model, item_enc
    except Exception as exc:
        logger.warning("sasrec_model_load_failed, sasrec_score sẽ = 0.0: %s", exc)
        return None, None


def compute_sasrec_scores(
    candidate_product_ids: list[str], session_items: list[str] | None
) -> dict:
    """Chấm điểm SASRec cho từng candidate dựa trên session/lịch sử gần đây
    (thứ tự cũ → mới). Trả dict rỗng nếu SASRec chưa sẵn sàng hoặc không có
    session — gọi nơi dùng tự fillna(0.0), KHÔNG raise để không chặn rerank."""
    if not session_items:
        return {}
    model, item_enc = _get_sasrec()
    if model is None:
        return {}
    import torch

    item_indices = [item_enc.get(str(iid), 0) for iid in session_items]
    item_indices = item_indices[-_SASREC_MAX_SEQ:]
    padded = [0] * (_SASREC_MAX_SEQ - len(item_indices)) + item_indices
    seq_t = torch.tensor([padded], dtype=torch.long)

    cand_indices = [item_enc.get(str(pid), 0) for pid in candidate_product_ids]
    cand_t = torch.tensor(cand_indices, dtype=torch.long)

    with torch.no_grad():
        scores = model.predict_next(seq_t, cand_t)[0].numpy()

    return {pid: float(s) for pid, s in zip(candidate_product_ids, scores)}


def compute_sasrec_scores_batch(
    customer_sessions: dict, customer_candidates: dict
) -> dict:
    """Bản batch của compute_sasrec_scores() — CHỈ dùng cho train_ltr.py xây
    dataset (~493k khách). Gọi compute_sasrec_scores() từng khách một tốn
    ~66 phút (mỗi forward pass Transformer chỉ batch=1) — bản này encode
    TOÀN BỘ session cùng lúc theo chunk (nhanh hơn nhiều lần), rồi chấm điểm
    candidate bằng dot product trực tiếp với ma trận embedding item (không
    cần gọi lại model nữa, đúng công thức seq_out @ cand_emb.T của
    predict_next() nhưng tách phần forward pass ra làm 1 lần duy nhất).
    Serving path (rerank_candidates(), 1 request/lần) không có gì để batch
    nên vẫn dùng compute_sasrec_scores() ở trên, không đổi.

    customer_sessions: {customer_id: [item_id, ...] (cũ→mới)}
    customer_candidates: {customer_id: [item_id, ...] (ứng viên cần chấm)}
    Trả {customer_id: {product_id: score}}.
    """
    model, item_enc = _get_sasrec()
    if model is None:
        return {}
    import torch

    customer_ids = list(customer_sessions.keys())
    all_padded = []
    for cid in customer_ids:
        items = customer_sessions[cid]
        idxs = [item_enc.get(str(i), 0) for i in items][-_SASREC_MAX_SEQ:]
        padded = [0] * (_SASREC_MAX_SEQ - len(idxs)) + idxs
        all_padded.append(padded)
    seq_tensor = torch.tensor(all_padded, dtype=torch.long)

    chunk = 512
    session_vecs = []
    with torch.no_grad():
        for i in range(0, len(seq_tensor), chunk):
            batch = seq_tensor[i : i + chunk]
            out = model.encode_session(batch)[:, -1, :]  # [B, D] — vị trí cuối
            session_vecs.append(out)
    session_vecs = torch.cat(session_vecs, dim=0)  # [N, D]

    item_emb_matrix = model.item_emb.weight.detach()  # [n_items+1, D]

    result = {}
    with torch.no_grad():
        for i, cid in enumerate(customer_ids):
            candidates = customer_candidates.get(cid, [])
            if not candidates:
                continue
            cand_idx = torch.tensor(
                [item_enc.get(str(pid), 0) for pid in candidates], dtype=torch.long
            )
            cand_vecs = item_emb_matrix[cand_idx]  # [C, D]
            scores = (cand_vecs @ session_vecs[i]).numpy()
            result[cid] = {pid: float(s) for pid, s in zip(candidates, scores)}

    return result


def _get_ltr_model():
    """Lazy-load XGBoost Booster — chỉ load 1 lần, trả None nếu file không có
    (môi trường chưa chạy train_ltr.py, vd test/dev) để rerank_candidates()
    fallback về công thức rule-based thay vì crash."""
    global _ltr_model, _ltr_load_attempted
    if _ltr_load_attempted:
        return _ltr_model
    _ltr_load_attempted = True
    if not os.path.exists(_LTR_MODEL_PATH):
        logger.warning(
            "LTR model không tìm thấy tại %s — dùng rule-based rerank (chạy "
            "`python -m src.ml_models.recsys.training.train_ltr` để train)",
            _LTR_MODEL_PATH,
        )
        return None
    try:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(_LTR_MODEL_PATH)
        logger.info("ltr_model_loaded path=%s", _LTR_MODEL_PATH)
        return booster
    except Exception as exc:
        logger.warning("ltr_model_load_failed, dùng rule-based rerank: %s", exc)
        return None


def rerank_candidates(
    candidates_df,
    user_features,
    diversity_limit: int = 3,
    bought_cats: set | None = None,
    session_items: list[str] | None = None,
):
    df = candidates_df.copy()

    target_price = user_features["avg_price_preference"]
    pos_ratio = user_features.get("positive_review_ratio", 0.5)
    bought_cats = bought_cats or set()
    complement_cats: set = set()
    for c in bought_cats:
        complement_cats |= CATEGORY_COMPLEMENTS.get(str(c), set())

    # =====================================
    # FEATURES — dùng chung cho cả LTR model lẫn fallback rule-based
    # =====================================

    bayes_sentiment = _bayesian_sentiment(df)
    df["bayes_sentiment"] = bayes_sentiment
    df["price_closeness"] = 1 / (
        1 + abs(df["price"] - target_price) / max(target_price, 1.0)
    )
    sasrec_scores = compute_sasrec_scores(
        df["product_id"].astype(str).tolist(), session_items
    )
    df["sasrec_score"] = df["product_id"].astype(str).map(sasrec_scores).fillna(0.0)
    df["category_match"] = df["category_id"].astype(str).isin(bought_cats).astype(int)
    df["category_complement"] = (
        df["category_id"].astype(str).isin(complement_cats).astype(int)
    )
    df["popularity_norm"] = (
        df["product_id"].astype(str).map(_item_popularity).fillna(0) / _popularity_max
    )

    df["semantic_score"] = df["predict_score"].astype(float)
    # Đổi sang ENSEMBLE BLEND tường minh — đúng pattern hệ thống enterprise
    # thật (Amazon/Shopee không dùng 1 model độc quyền quyết định thứ hạng,
    # mà blend nhiều signal độc lập với trọng số rõ ràng, dễ giải thích/tune
    # theo business, không phụ thuộc hoàn toàn vào gain học được của 1 model):
    #   - RETRIEVAL (Two-Tower, sở thích DÀI HẠN)
    #   - SESSION (SASRec, ý định GẦN ĐÂY/trong phiên)
    #   - TRENDING (popularity real-time)
    #   - SENTIMENT (chất lượng sản phẩm, Bayesian-smoothed)
    #   - CATEGORY (khớp lịch sử mua — vẫn giữ nhưng KHÔNG còn áp đảo)
    #   - COMPLEMENT (cross-sell)
    #   - PRICE (phù hợp tầm giá)
    # Trọng số cộng = 1.0, mỗi input normalize về [0,1] TRONG CHÍNH batch
    # candidate đang xét (semantic_score/sasrec_score không tự nhiên nằm
    # trong [0,1] như category_match/popularity_norm) để so sánh công bằng.
    W_RETRIEVAL = 0.30
    W_SESSION = 0.20
    W_TRENDING = 0.15
    W_SENTIMENT = 0.15
    W_CATEGORY = 0.10
    W_COMPLEMENT = 0.05
    W_PRICE = 0.05

    def _normalize(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    df["final_score"] = (
        W_RETRIEVAL * _normalize(df["semantic_score"])
        + W_SESSION * _normalize(df["sasrec_score"])
        + W_TRENDING * df["popularity_norm"].clip(0, 1)
        + W_SENTIMENT * df["bayes_sentiment"].clip(0, 1)
        + W_CATEGORY * df["category_match"]
        + W_COMPLEMENT * df["category_complement"]
        + W_PRICE * df["price_closeness"].clip(0, 1)
    )

    _get_ltr_model()

    df = df.sort_values("final_score", ascending=False)

    # Compute explanation per item (Task 70)
    df["_explanation"] = df.apply(
        lambda row: _build_explanation(row, user_features), axis=1
    )

    # =====================================
    # DIVERSITY — max 3 items per category
    # =====================================

    result: list = []
    cat_count: dict = {}

    for _, row in df.iterrows():
        cat = row["category_id"]
        if cat_count.get(cat, 0) < diversity_limit:
            result.append(row)
            cat_count[cat] = cat_count.get(cat, 0) + 1

    return pd.DataFrame(result)
