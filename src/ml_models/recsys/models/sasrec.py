"""
SASRec: Self-Attentive Sequential Recommendation.
Kang & McAuley, ICDM 2018.

Models sequential patterns in user sessions using causal self-attention.
Predicts the next item a user will interact with given their session history.

Architecture:
  - Item embedding layer
  - L Transformer blocks (causal: each position attends only to past positions)
  - Output: probability over all items for next position

Compared to Two-Tower (non-sequential):
  - Two-Tower: "this user bought X, Y, Z in the past" → static user vector
  - SASRec: "this session: X → Y → Z → ?" → next-item prediction
  - Endpoint: POST /recommend/session {session_items: ["X", "Y", "Z"]}

Session definition: purchases/clicks within a 30-minute gap window.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SASRecBlock(nn.Module):
    """Single Transformer encoder block with causal mask."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = PointWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        # Causal self-attention (position i can only attend to positions ≤ i)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SASRec(nn.Module):
    """
    SASRec model.

    Args:
        n_items:        Vocabulary size (number of unique items).
        max_seq_len:    Maximum session length (padding/truncation target).
        d_model:        Embedding dimension.
        n_heads:        Number of attention heads.
        n_layers:       Number of Transformer blocks.
        d_ff:           Feed-forward hidden size (typically 4 * d_model).
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # 0 = padding token
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [SASRecBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len: int, device) -> Tensor:
        """Upper-triangular mask for causal attention (True = mask out)."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def encode_session(self, item_seq: Tensor) -> Tensor:
        """
        Encode a session sequence.

        Args:
            item_seq: [batch, seq_len] item ID tensor (0 = padding)

        Returns:
            Sequence output [batch, seq_len, d_model]
        """
        bs, seq_len = item_seq.shape
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        x = self.emb_drop(self.item_emb(item_seq) + self.pos_emb(positions))

        mask = self._causal_mask(seq_len, item_seq.device)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, item_seq: Tensor, pos_items: Tensor, neg_items: Tensor) -> Tensor:
        """
        Compute BCE loss for next-item prediction.

        Args:
            item_seq:  [batch, seq_len] session history (shifted right)
            pos_items: [batch, seq_len] ground truth next items
            neg_items: [batch, seq_len] negative samples

        Returns:
            Scalar loss.
        """
        seq_out = self.encode_session(item_seq)  # [B, L, D]
        pos_emb = self.item_emb(pos_items)  # [B, L, D]
        neg_emb = self.item_emb(neg_items)  # [B, L, D]

        pos_logits = (seq_out * pos_emb).sum(-1)  # [B, L]
        neg_logits = (seq_out * neg_emb).sum(-1)  # [B, L]

        # Mask padding positions
        pad_mask = (pos_items != 0).float()
        loss = -torch.log(torch.sigmoid(pos_logits) + 1e-8) * pad_mask
        loss -= torch.log(1 - torch.sigmoid(neg_logits) + 1e-8) * pad_mask
        return loss.sum() / (pad_mask.sum() + 1e-8)

    def predict_next(self, item_seq: Tensor, candidate_items: Tensor) -> Tensor:
        """
        Score candidate items for next-item recommendation.

        Args:
            item_seq:        [batch, seq_len] session history
            candidate_items: [n_candidates] item IDs to score

        Returns:
            Scores [batch, n_candidates]
        """
        seq_out = self.encode_session(item_seq)[:, -1, :]  # [B, D] — last position
        cand_emb = self.item_emb(candidate_items)  # [C, D]
        return seq_out @ cand_emb.T  # [B, C]
