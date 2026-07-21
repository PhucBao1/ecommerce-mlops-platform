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

        # user_id_emb + 4 numerical features
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
