"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
He et al., SIGIR 2020.

Architecture:
  - User/item embeddings as learnable parameters
  - L layers of LGConv (linear graph convolution, no non-linearity)
  - Final embedding = mean of all layer embeddings (layer combination)
  - Score = dot product of user and item final embeddings
  - Loss = BPR (Bayesian Personalized Ranking): positive > negative

Compared to Two-Tower:
  - Two-Tower: learns from (user features, item features) pairs — 1st-order
  - LightGCN: propagates signals over the purchase graph — captures multi-hop
    "users who bought A also bought B and C" patterns
  - Switch via RETRIEVAL_BACKEND=lightgcn env var in serving

Requires: torch-geometric>=2.4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LightGCNConv(nn.Module):
    """Single LightGCN layer: symmetric normalized aggregation, no weights."""

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Args:
            x:          Node features [num_nodes, emb_dim]
            edge_index: COO adjacency [2, num_edges] (user→item + item→user)
            num_nodes:  Total nodes (n_users + n_items)

        Returns:
            Aggregated embeddings [num_nodes, emb_dim]
        """
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device).scatter_add_(
            0, row, torch.ones(row.size(0), device=x.device)
        )
        deg_inv_sqrt = deg.pow(-0.5).nan_to_num(nan=0.0, posinf=0.0)

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.zeros_like(x)
        out.scatter_add_(
            0, col.unsqueeze(1).expand(-1, x.size(1)), norm.unsqueeze(1) * x[row]
        )
        return out


class LightGCN(nn.Module):
    """
    Full LightGCN model.

    Args:
        n_users:       Number of users in the dataset.
        n_items:       Number of items in the dataset.
        embedding_dim: Size of user/item embedding vectors.
        n_layers:      Number of graph convolution layers (typical: 3).
    """

    def __init__(
        self, n_users: int, n_items: int, embedding_dim: int = 64, n_layers: int = 3
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.conv = LightGCNConv()

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def _propagate(self, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Run L-layer graph propagation, return final (user, item) embeddings."""
        num_nodes = self.n_users + self.n_items
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        layer_embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self.conv(all_emb, edge_index, num_nodes)
            layer_embs.append(all_emb)

        # Layer combination: mean of all layers (Eq. 11 in paper)
        final = torch.stack(layer_embs, dim=1).mean(dim=1)
        return final[: self.n_users], final[self.n_users :]

    def forward(
        self,
        user_ids: Tensor,
        pos_item_ids: Tensor,
        neg_item_ids: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute BPR loss for a batch of (user, pos_item, neg_item) triples.

        Returns:
            Scalar BPR loss.
        """
        user_embs, item_embs = self._propagate(edge_index)

        u = user_embs[user_ids]
        pos = item_embs[pos_item_ids]
        neg = item_embs[neg_item_ids]

        pos_score = (u * pos).sum(dim=-1)
        neg_score = (u * neg).sum(dim=-1)

        bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
        reg_loss = (
            self.user_emb.weight[user_ids].norm(2).pow(2)
            + self.item_emb.weight[pos_item_ids].norm(2).pow(2)
            + self.item_emb.weight[neg_item_ids].norm(2).pow(2)
        ) / len(user_ids)

        return bpr_loss + 1e-5 * reg_loss

    def get_embeddings(self, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Get final user and item embeddings after propagation (for inference/export)."""
        with torch.no_grad():
            return self._propagate(edge_index)

    def score(self, user_ids: Tensor, item_ids: Tensor, edge_index: Tensor) -> Tensor:
        """Score user-item pairs (for ranking/evaluation)."""
        user_embs, item_embs = self._propagate(edge_index)
        u = user_embs[user_ids]
        i = item_embs[item_ids]
        return (u * i).sum(dim=-1)
