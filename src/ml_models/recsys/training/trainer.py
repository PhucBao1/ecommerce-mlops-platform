import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.ml_models.recsys.training.evalute import compute_ndcg_at_k, compute_recall_at_k


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    losses = []

    for batch in loader:

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def _uniformity_loss(x, t=2.0):
    """Wang & Isola (2020) uniformity term — penalizes embeddings clustering
    on the hypersphere instead of spreading out. x is already L2-normalized,
    so pairwise squared distance reduces to 2 - 2*cos_sim."""
    sq_dist = torch.pdist(x, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()


def _mean_pairwise_cosine(x, n_sample=256):
    """Diagnostic only (no grad): mean off-diagonal cosine similarity of a
    random subsample — near 1.0 means the embedding space has collapsed
    (every item/user points the same direction, see BENCHMARK_RESULTS.md §17)."""
    with torch.no_grad():
        if x.size(0) > n_sample:
            idx = torch.randperm(x.size(0), device=x.device)[:n_sample]
            x = x[idx]
        sim = x @ x.T
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        return sim[mask].mean().item()


def train_one_epoch_infonce(
    model, loader, optimizer, device, temperature=0.07, uniformity_weight=0.5
):
    """InfoNCE loss with in-batch negatives for Two-Tower training.

    uniformity_weight adds the Wang & Isola anti-collapse regularizer on top
    of the plain contrastive loss — without it, this training setup was
    observed to converge to a degenerate solution where nearly all item
    embeddings point in the same direction (mean pairwise cosine ~0.915
    across 500 random catalog items), making recommendations effectively
    identical regardless of user (see BENCHMARK_RESULTS.md §17 investigation).
    """
    model.train()
    losses = []
    collapse_metrics = []

    for batch in loader:

        batch = {k: v.to(device) for k, v in batch.items()}

        user_vec = model.user_tower(
            user_id=batch["user_id"], user_num=batch["user_num"]
        )
        item_vec = model.item_tower(
            item_id=batch["item_id"],
            item_category=batch["item_category"],
            item_num=batch["item_num"],
        )

        # [B, B] similarity — off-diagonal items are in-batch negatives
        logits = torch.matmul(user_vec, item_vec.T) / temperature
        labels = torch.arange(len(user_vec), device=device)

        contrastive_loss = F.cross_entropy(logits, labels)
        uniformity = _uniformity_loss(item_vec) + _uniformity_loss(user_vec)
        loss = contrastive_loss + uniformity_weight * uniformity

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        collapse_metrics.append(_mean_pairwise_cosine(item_vec.detach()))

    return np.mean(losses), np.mean(collapse_metrics)


def validate(model, loader, criterion, device):

    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    all_user_ids = []

    with torch.no_grad():

        for batch in loader:

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["label"]
            logits = model(batch)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_user_ids.extend(batch["user_id"].cpu().numpy())

    soft_labels = np.array(all_labels)
    # AUC + Recall cần binary: Positive+Neutral (>=0.5) = 1, Negative/no-interaction = 0
    # roc_auc_score không nhận continuous labels nên phải binarize
    binary_labels = (soft_labels >= 0.5).astype(int)
    probs = 1 / (1 + np.exp(-np.array(all_logits)))

    auc = roc_auc_score(binary_labels, probs)
    # Recall@10/NDCG@10 PER USER rồi mới average
    recall = compute_recall_at_k(all_user_ids, all_logits, soft_labels, k=10)
    # NDCG dùng soft labels (graded relevance: 1.0 > 0.5 > 0.0)
    ndcg = compute_ndcg_at_k(all_user_ids, all_logits, soft_labels, k=10)

    return np.mean(losses), auc, recall, ndcg
