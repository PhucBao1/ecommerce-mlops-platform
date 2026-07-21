"""
Ranking metrics — Recall@K / NDCG@K per user.

Đây là con số `promotion_gate` (retrain.py) dùng để quyết định có đẩy model mới
lên Production hay không. Bản cũ lấy top-K trên TOÀN BỘ validation set rồi mới
average, tức là hàng nghìn user chia nhau đúng 10 "slot" — gần như vô nghĩa, và
model random-init vẫn ra recall ~0.98. Metric sai = ship model tệ lên Production,
nên phải có test gác.
"""

import numpy as np

from src.ml_models.recsys.training.evalute import (
    compute_ndcg_at_k,
    compute_recall_at_k,
)

# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def test_perfect_ranking_gives_recall_one():
    """Item positive được xếp đầu cho từng user => recall = 1.0."""
    users = ["u1", "u1", "u2", "u2"]
    logits = [9.0, 0.1, 8.0, 0.2]  # phần tử positive có điểm cao nhất
    labels = [1.0, 0.0, 1.0, 0.0]
    assert compute_recall_at_k(users, logits, labels, k=1) == 1.0


def test_positive_ranked_last_gives_recall_zero():
    users = ["u1", "u1"]
    logits = [0.1, 9.0]  # positive (index 0) bị xếp SAU
    labels = [1.0, 0.0]
    assert compute_recall_at_k(users, logits, labels, k=1) == 0.0


def test_recall_is_averaged_per_user_not_pooled():
    """
    Đây chính là bug cũ. u1 xếp đúng (recall 1), u2 xếp sai (recall 0)
    => trung bình PER-USER = 0.5.

    Cách cũ (gộp chung, top-K toàn cục) sẽ lấy 1 top-1 duy nhất trên cả 4 dòng
    và cho ra con số khác hẳn — không phản ánh chất lượng gợi ý cho từng user.
    """
    users = ["u1", "u1", "u2", "u2"]
    logits = [9.0, 0.1, 0.1, 9.0]
    labels = [1.0, 0.0, 1.0, 0.0]  # positive của u2 nằm ở logit thấp
    assert compute_recall_at_k(users, logits, labels, k=1) == 0.5


def test_user_without_any_positive_is_skipped():
    """Recall không định nghĩa được khi user không có item positive nào."""
    users = ["u1", "u1", "u2", "u2"]
    logits = [9.0, 0.1, 5.0, 1.0]
    labels = [1.0, 0.0, 0.0, 0.0]  # u2 toàn negative
    # chỉ u1 được tính => 1.0, không bị u2 kéo xuống 0.5
    assert compute_recall_at_k(users, logits, labels, k=1) == 1.0


def test_neutral_label_counts_as_positive():
    """Label mềm: Positive=1.0, Neutral=0.5, Negative=0.0. Ngưỡng >= 0.5."""
    users = ["u1", "u1"]
    logits = [9.0, 0.1]
    labels = [0.5, 0.0]
    assert compute_recall_at_k(users, logits, labels, k=1) == 1.0


def test_empty_input_returns_zero_not_crash():
    assert compute_recall_at_k([], [], [], k=10) == 0.0


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------


def test_ndcg_perfect_order_is_one():
    users = ["u1", "u1", "u1"]
    logits = [9.0, 5.0, 1.0]
    labels = [1.0, 0.5, 0.0]  # thứ tự điểm khớp đúng thứ tự relevance
    assert compute_ndcg_at_k(users, logits, labels, k=3) == 1.0


def test_ndcg_reversed_order_is_worse_than_perfect():
    users = ["u1", "u1", "u1"]
    labels = [1.0, 0.5, 0.0]
    good = compute_ndcg_at_k(users, [9.0, 5.0, 1.0], labels, k=3)
    bad = compute_ndcg_at_k(users, [1.0, 5.0, 9.0], labels, k=3)
    assert bad < good


def test_ndcg_uses_graded_relevance():
    """
    Xếp item relevance 1.0 lên đầu phải TỐT HƠN xếp item 0.5 lên đầu.
    Nếu code binarize nhãn (coi 1.0 và 0.5 như nhau) thì 2 giá trị này sẽ bằng
    nhau và test đỏ.
    """
    users = ["u1", "u1"]
    labels = [1.0, 0.5]
    strong_first = compute_ndcg_at_k(users, [9.0, 1.0], labels, k=2)
    weak_first = compute_ndcg_at_k(users, [1.0, 9.0], labels, k=2)
    assert strong_first > weak_first


def test_ndcg_averaged_per_user():
    users = ["u1", "u1", "u2", "u2"]
    logits = [9.0, 1.0, 1.0, 9.0]
    labels = [1.0, 0.0, 1.0, 0.0]  # u1 xếp đúng, u2 xếp ngược
    result = compute_ndcg_at_k(users, logits, labels, k=2)
    assert 0.0 < result < 1.0


def test_user_with_all_zero_labels_is_skipped():
    """IDCG = 0 => NDCG không định nghĩa được, phải bỏ qua chứ không chia cho 0."""
    users = ["u1", "u1", "u2", "u2"]
    logits = [9.0, 1.0, 5.0, 2.0]
    labels = [1.0, 0.0, 0.0, 0.0]
    result = compute_ndcg_at_k(users, logits, labels, k=2)
    assert result == 1.0  # chỉ u1 được tính
    assert not np.isnan(result)
