import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# ITEM TOWER
# =====================================================
class ItemTower(nn.Module):

    def __init__(self, num_items, num_categories, embedding_dim=32):

        super().__init__()

        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        # item_emb + cat_emb + 2 numerical
        input_dim = embedding_dim * 2 + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64)
        )

    def forward(self, item_id, item_category, item_num):

        i_emb = self.item_embedding(item_id)

        c_emb = self.category_embedding(item_category)

        x = torch.cat([i_emb, c_emb, item_num], dim=1)

        item_vector = self.mlp(x)

        # normalize embeddings
        item_vector = F.normalize(item_vector, p=2, dim=1)

        return item_vector
