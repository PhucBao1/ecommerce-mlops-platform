import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(logits, labels):
    probs = 1 / (1 + np.exp(-np.array(logits)))
    return roc_auc_score(labels, probs)


def _group_by_user(user_ids, logits, labels):
    user_ids = np.asarray(user_ids)
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    return {
        uid: (logits[user_ids == uid], labels[user_ids == uid])
        for uid in np.unique(user_ids)
    }


def compute_recall_at_k(user_ids, logits, labels, k=10):
    """
    Recall@K computed PER USER then averaged -- not a single top-K over the
    whole validation set (that was the bug: with many users sharing one
    validation batch, a global top-10 is nearly meaningless as a
    recommendation-quality signal, and it's what promotion_gate in
    retrain.py used to decide whether to promote a model to Production).
    """
    binary_labels = (np.asarray(labels) >= 0.5).astype(int)
    recalls = []
    for u_logits, u_labels in _group_by_user(user_ids, logits, binary_labels).values():
        total_pos = u_labels.sum()
        if total_pos == 0:
            continue  # recall undefined without a positive to find
        top_k_idx = np.argsort(-u_logits)[:k]
        recalls.append(u_labels[top_k_idx].sum() / total_pos)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_ndcg_at_k(user_ids, logits, labels, k=10):
    """Per-user NDCG@K with graded relevance (soft labels), averaged across users."""
    ndcgs = []
    for u_logits, u_labels in _group_by_user(user_ids, logits, labels).values():
        order = np.argsort(-u_logits)[:k]
        gains = np.asarray(u_labels)[order]
        discounts = 1 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))

        ideal_order = np.argsort(-np.asarray(u_labels))[:k]
        ideal_gains = np.asarray(u_labels)[ideal_order]
        ideal_discounts = 1 / np.log2(np.arange(2, len(ideal_gains) + 2))
        idcg = float(np.sum(ideal_gains * ideal_discounts))

        if idcg == 0:
            continue
        ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
