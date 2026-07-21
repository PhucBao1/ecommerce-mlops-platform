"""
DIN: Deep Interest Network.
Zhou et al., KDD 2018 (Alibaba).

Computes attention-weighted user history representation with respect to
each target item. Instead of a fixed user vector (Two-Tower), DIN creates
a different user representation for each candidate item.

Use case in this system:
  - Two-Tower retrieves top-200 candidates
  - DIN reranks the 200 candidates using attention over user click history
  - More effective than static reranking because user interest is item-adaptive

Architecture:
  - Item embedding
  - Activation Unit: MLP that scores relevance of each history item to target
  - Weighted sum of history embeddings → attended user representation
  - Final MLP: concat(attended_user, target_item) → relevance score

Benchmark on internal Alibaba data: DIN +3-5% CTR over DeepFM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ActivationUnit(nn.Module):
    """
    Attention scoring: how relevant is a history item to the target item?

    Input: [target, history_item, target - history_item, target * history_item]
    Output: scalar attention weight
    """

    def __init__(self, embedding_dim: int, hidden_units: list[int] = None):
        super().__init__()
        hidden_units = hidden_units or [64, 32]
        input_dim = embedding_dim * 4  # concat of 4 interaction terms

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.PReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, target: Tensor, history: Tensor) -> Tensor:
        """
        Args:
            target:  [batch, 1, emb_dim] — target item embedding (broadcast over history)
            history: [batch, seq_len, emb_dim] — user history embeddings

        Returns:
            Attention weights [batch, seq_len, 1]
        """
        target = target.expand_as(history)
        interaction = torch.cat(
            [target, history, target - history, target * history], dim=-1
        )
        return self.mlp(interaction)  # [B, L, 1]


class DIN(nn.Module):
    """
    Deep Interest Network.

    Args:
        n_items:       Vocabulary size.
        embedding_dim: Item embedding size.
        mlp_units:     Hidden units for final MLP.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        mlp_units: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        mlp_units = mlp_units or [128, 64]
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.activation_unit = ActivationUnit(embedding_dim)

        # Final MLP: attended_user (emb_dim) + target (emb_dim) → score
        layers: list[nn.Module] = []
        prev = embedding_dim * 2
        for h in mlp_units:
            layers += [nn.Linear(prev, h), nn.PReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, target_item_ids: Tensor, history_item_ids: Tensor) -> Tensor:
        """
        Score target items given user click history.

        Args:
            target_item_ids:  [batch] candidate item IDs to score
            history_item_ids: [batch, seq_len] user history item IDs (0 = padding)

        Returns:
            Relevance scores [batch]
        """
        target_emb = self.item_emb(target_item_ids).unsqueeze(1)  # [B, 1, D]
        history_emb = self.item_emb(history_item_ids)  # [B, L, D]

        # Attention weights from Activation Unit
        attn_weights = self.activation_unit(target_emb, history_emb)  # [B, L, 1]

        # Mask padding (history_item_ids == 0)
        pad_mask = (history_item_ids != 0).unsqueeze(-1).float()  # [B, L, 1]
        attn_weights = attn_weights * pad_mask

        # Softmax over non-padded positions
        attn_weights = attn_weights - (1 - pad_mask) * 1e9
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B, L, 1]

        # Attended user representation
        user_repr = (attn_weights * history_emb).sum(dim=1)  # [B, D]

        # Final scoring
        combined = torch.cat([user_repr, target_emb.squeeze(1)], dim=-1)  # [B, 2D]
        return self.mlp(combined).squeeze(-1)  # [B]
