"""
LightGCN training script.

Builds bipartite user-item graph from purchase history,
trains with BPR loss, exports embeddings for serving.

Usage:
    python -m src.ml_models.recsys.training.train_lightgcn
    python -m src.ml_models.recsys.training.train_lightgcn --epochs 30 --layers 3 --dim 64
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.ml_models.recsys.evaluation.metrics import ndcg_at_k
from src.ml_models.recsys.models.lightgcn import LightGCN
from src.ml_models.recsys.training.mlflow_logger import setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_DATA_DIR = os.getenv("DATA_DIR", "/app/artifacts/recsys_models/data_menu")
_MODEL_DIR = os.getenv("MODEL_DIR", "/app/artifacts/recsys_models")
_SEED = 42

torch.manual_seed(_SEED)
np.random.seed(_SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BPRDataset(Dataset):
    """Samples (user, pos_item, neg_item) triples for BPR training."""

    def __init__(self, interactions: pd.DataFrame, n_items: int):
        pos = interactions[interactions["label"] > 0]
        self.users = pos["user_idx"].values
        self.pos_items = pos["item_idx"].values
        self.n_items = n_items
        self.all_pos = pos.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        uid = self.users[idx]
        pos = self.pos_items[idx]
        # Sample negative not in user's positives
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in self.all_pos.get(uid, set()):
                break
        return uid, pos, neg


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_edge_index(interactions: pd.DataFrame, n_users: int) -> torch.Tensor:
    """
    Build bipartite graph as COO edge_index (both directions).
    Item nodes are offset by n_users so all nodes share a single index space.
    """
    users = torch.tensor(interactions["user_idx"].values, dtype=torch.long)
    items = torch.tensor(interactions["item_idx"].values + n_users, dtype=torch.long)

    # user→item and item→user edges
    edge_index = torch.stack(
        [
            torch.cat([users, items]),
            torch.cat([items, users]),
        ],
        dim=0,
    )
    return edge_index


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model: LightGCN, val_df: pd.DataFrame, edge_index: torch.Tensor, device, k: int = 10
) -> float:
    """Compute mean NDCG@K on validation set."""
    model.eval()
    user_embs, item_embs = model.get_embeddings(edge_index.to(device))
    user_embs = user_embs.cpu().numpy()
    item_embs = item_embs.cpu().numpy()

    ndcg_scores = []
    for uid, group in val_df.groupby("user_idx"):
        positives = group[group["label"] > 0]["item_idx"].tolist()
        if not positives:
            continue
        u_vec = user_embs[uid]
        scores = item_embs @ u_vec
        ranked = np.argsort(-scores).tolist()
        ndcg_scores.append(ndcg_at_k(positives, ranked, k=k))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    n_epochs: int = 20,
    embedding_dim: int = 64,
    n_layers: int = 3,
    lr: float = 1e-3,
    batch_size: int = 1024,
) -> float:

    setup_mlflow()
    mlflow.set_experiment("lightgcn_training")

    # Load data
    history = pd.read_parquet(Path(_DATA_DIR) / "user_history.parquet")
    history["label"] = (
        history["action"].map({"purchase": 3, "click": 1, "ignore": 0}).fillna(1)
    )

    # Encode user/item IDs
    user_enc = {uid: i for i, uid in enumerate(history["customer_id"].unique())}
    item_enc = {iid: i for i, iid in enumerate(history["product_id"].unique())}
    history["user_idx"] = history["customer_id"].map(user_enc)
    history["item_idx"] = history["product_id"].map(item_enc)

    n_users = len(user_enc)
    n_items = len(item_enc)

    # Train/val split (last 20% of interactions by time for each user)
    history = history.sort_values("purchased_at")
    cutoff = int(len(history) * 0.8)
    train_df = history.iloc[:cutoff]
    val_df = history.iloc[cutoff:]

    edge_index = _build_edge_index(train_df[train_df["label"] > 0], n_users)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)

    model = LightGCN(
        n_users, n_items, embedding_dim=embedding_dim, n_layers=n_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = BPRDataset(train_df, n_items)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    with mlflow.start_run(run_name="lightgcn"):
        mlflow.log_params(
            {
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": embedding_dim,
                "n_layers": n_layers,
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
            }
        )

        best_ndcg = 0.0
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            for user_ids, pos_ids, neg_ids in loader:
                user_ids = user_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)

                loss = model(user_ids, pos_ids, neg_ids, edge_index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            ndcg = _evaluate(model, val_df, edge_index, device)
            if ndcg > best_ndcg:
                best_ndcg = ndcg

            mlflow.log_metrics({"loss": avg_loss, "ndcg_at_10": ndcg}, step=epoch)
            logger.info("epoch=%d loss=%.4f ndcg@10=%.4f", epoch, avg_loss, ndcg)

        # Export final embeddings (user + item) as npz for serving
        user_embs, item_embs = model.get_embeddings(edge_index)
        out_path = Path(_MODEL_DIR) / "lightgcn_embeddings.npz"
        np.savez(
            str(out_path),
            user_embeddings=user_embs.cpu().numpy(),
            item_embeddings=item_embs.cpu().numpy(),
            user_ids=list(user_enc.keys()),
            item_ids=list(item_enc.keys()),
        )
        mlflow.log_artifact(str(out_path), "lightgcn_embeddings")
        mlflow.log_metric("best_ndcg_at_10", best_ndcg)
        logger.info("embeddings saved to %s (best NDCG@10=%.4f)", out_path, best_ndcg)

    return best_ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    best = train(
        n_epochs=args.epochs,
        embedding_dim=args.dim,
        n_layers=args.layers,
        lr=args.lr,
        batch_size=args.batch,
    )
    print(f"\nBest NDCG@10: {best:.4f}")
