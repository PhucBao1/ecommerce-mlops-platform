# =========================================================
# FILE: model.py
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# USER TOWER
# =====================================================
class UserTower(nn.Module):

    def __init__(self, num_users, embedding_dim=32):

        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # user_id_emb + 3 numerical features
        input_dim = embedding_dim + 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64)
        )

    def forward(self, user_id, user_num):

        u_emb = self.user_embedding(user_id)

        x = torch.cat([u_emb, user_num], dim=1)

        user_vector = self.mlp(x)

        # normalize embeddings
        user_vector = F.normalize(user_vector, p=2, dim=1)

        return user_vector


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


# =====================================================
# TWO TOWER MODEL
# =====================================================
class TwoTowerModel(nn.Module):

    def __init__(self, num_users, num_items, num_categories, embedding_dim=32):

        super().__init__()

        self.user_tower = UserTower(num_users=num_users, embedding_dim=embedding_dim)

        self.item_tower = ItemTower(
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=embedding_dim,
        )

    def forward(self, batch):

        # =========================
        # USER VECTOR
        # =========================
        user_vector = self.user_tower(
            user_id=batch["user_id"], user_num=batch["user_num"]
        )

        # =========================
        # ITEM VECTOR
        # =========================
        item_vector = self.item_tower(
            item_id=batch["item_id"],
            item_category=batch["item_category"],
            item_num=batch["item_num"],
        )

        # =========================
        # DOT PRODUCT
        # =========================
        scores = torch.sum(user_vector * item_vector, dim=1) * 10

        return scores
