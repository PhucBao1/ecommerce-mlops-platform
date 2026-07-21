import torch
import torch.nn as nn

from src.ml_models.recsys.models.item_tower import ItemTower
from src.ml_models.recsys.models.user_tower import UserTower


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
