"""
Reranker của recsys-api — thứ quyết định user THẬT SỰ nhìn thấy gì.

Lưu ý bản chất: rerank ở đây KHÔNG phải model ML — nó là công thức cộng có trọng
số (hệ số 0.05/0.03/0.02 chọn tay) trên `predict_score` do Two-Tower sinh ra, cộng
thêm bước diversity đếm theo category. Vì không có nhãn để tune, những ràng buộc
dưới đây là thứ duy nhất gác nó.
"""

import pandas as pd

from src.serving.recsys_api.reranker import rerank_candidates

_USER = {"avg_price_preference": 100.0, "positive_review_ratio": 0.6}


def _candidates(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bayesian sentiment: review count phải làm giảm độ tin của sentiment
# ---------------------------------------------------------------------------


def test_item_with_one_review_does_not_beat_item_with_many():
    """
    Bug cũ: avg_item_sentiment dùng thô, nên item có ĐÚNG 1 review 5 sao
    thắng áp đảo item có 500 review trung bình 4.5 sao. Bayesian average kéo
    item ít review về mức trung bình chung.
    """
    df = _candidates(
        [
            # sentiment tuyệt đối nhưng chỉ 1 review
            {
                "product_id": "ít_review",
                "predict_score": 0.80,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 1.0,
                "item_review_count": 1,
            },
            # sentiment thấp hơn chút nhưng có 500 review
            {
                "product_id": "nhiều_review",
                "predict_score": 0.80,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.9,
                "item_review_count": 500,
            },
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=5)
    scores = dict(zip(out["product_id"], out["bayes_sentiment"]))

    # item 1-review bị kéo XUỐNG mạnh (chỉ 1 phiếu thật, 5 phiếu "ảo" từ prior)
    assert scores["ít_review"] < 1.0
    # item nhiều review gần như giữ nguyên 0.9 (500 phiếu thật lấn át prior)
    assert scores["nhiều_review"] > 0.85
    # và nhờ vậy nó KHÔNG còn bị item 1-review vượt mặt
    assert scores["nhiều_review"] > scores["ít_review"]


def test_prior_is_not_derived_from_the_candidate_set():
    """
    Prior C phải là hằng số của TOÀN CATALOG, không được tính từ tập candidate.

    Bản đầu tiên tính C = mean(candidates_df) và nó tự vô hiệu hoá: outlier
    1-review-5-sao tự kéo C lên sát chính nó, nên bước co về trung bình chẳng
    làm được gì. Test này khoá lại: cùng 1 item, kết quả không được đổi chỉ vì
    có thêm item khác trong tập candidate.
    """
    item = {
        "product_id": "p1",
        "predict_score": 0.5,
        "price": 100,
        "category_id": "c1",
        "avg_item_sentiment": 1.0,
        "item_review_count": 1,
    }
    alone = rerank_candidates(_candidates([item]), _USER, diversity_limit=5)

    noisy = dict(item)
    noisy["product_id"] = "p2"
    noisy["category_id"] = "c2"
    noisy["avg_item_sentiment"] = 0.0  # item khác, sentiment cực thấp
    with_others = rerank_candidates(
        _candidates([item, noisy]), _USER, diversity_limit=5
    )

    score_alone = alone.set_index("product_id").loc["p1", "bayes_sentiment"]
    score_with = with_others.set_index("product_id").loc["p1", "bayes_sentiment"]
    assert score_alone == score_with


def test_missing_review_count_falls_back_to_raw_sentiment():
    """item_lookup cũ (chưa có cột item_review_count) không được làm sập rerank."""
    df = _candidates(
        [
            {
                "product_id": "p1",
                "predict_score": 0.5,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.8,
            }
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=5)
    assert len(out) == 1
    assert out.iloc[0]["bayes_sentiment"] == 0.8


# ---------------------------------------------------------------------------
# Diversity: tối đa N item / category
# ---------------------------------------------------------------------------


def test_diversity_limit_caps_items_per_category():
    """Không có bước này thì top-K dễ bị 1 category chiếm sạch."""
    df = _candidates(
        [
            {
                "product_id": f"p{i}",
                "predict_score": 0.9 - i * 0.01,
                "price": 100,
                "category_id": "cùng_category",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            }
            for i in range(6)
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=2)
    assert len(out) == 2, "chỉ được giữ 2 item cho mỗi category"


def test_diversity_keeps_highest_scoring_item_of_each_category():
    df = _candidates(
        [
            {
                "product_id": "c1_thấp",
                "predict_score": 0.10,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            },
            {
                "product_id": "c1_cao",
                "predict_score": 0.90,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            },
            {
                "product_id": "c2",
                "predict_score": 0.50,
                "price": 100,
                "category_id": "c2",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            },
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=1)
    kept = set(out["product_id"])
    assert kept == {"c1_cao", "c2"}, "phải giữ item điểm cao nhất của mỗi category"


# ---------------------------------------------------------------------------
# Price boost + explanation
# ---------------------------------------------------------------------------


def test_price_closer_to_preference_scores_higher():
    df = _candidates(
        [
            {
                "product_id": "đúng_tầm_giá",
                "predict_score": 0.5,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            },
            {
                "product_id": "quá_đắt",
                "predict_score": 0.5,
                "price": 10_000,
                "category_id": "c2",
                "avg_item_sentiment": 0.5,
                "item_review_count": 10,
            },
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=5)
    assert out.iloc[0]["product_id"] == "đúng_tầm_giá"


def test_explanation_attached_to_every_item():
    df = _candidates(
        [
            {
                "product_id": "p1",
                "predict_score": 0.5,
                "price": 100,
                "category_id": "c1",
                "avg_item_sentiment": 0.9,
                "item_review_count": 100,
            }
        ]
    )
    out = rerank_candidates(df, _USER, diversity_limit=5)
    expl = out.iloc[0]["_explanation"]
    assert expl["top_reason"]
    assert "price_match_pct" in expl
    assert "sentiment_score" in expl
