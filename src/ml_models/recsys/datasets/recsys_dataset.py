import pandas as pd
import torch
from torch.utils.data import Dataset


class EcommerceRecSysDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Nạp toàn bộ dataframe vào RAM và chuyển thành Pytorch Tensors.
        """
        # 1. User Categorical Features (Dùng kiểu Long cho Embedding layer)
        self.user_ids = torch.tensor(df["customer_id_idx"].values, dtype=torch.long)

        # 2. User Numerical Features (Gộp chung thành 1 vector, dùng kiểu Float)
        self.user_num_features = torch.tensor(
            df[
                [
                    "total_reviews_so_far",
                    "avg_price_preference",
                    "positive_review_ratio",
                    "has_history",
                ]
            ].values,
            dtype=torch.float32,
        )

        # 3. Item Categorical Features
        self.item_ids = torch.tensor(df["product_id_idx"].values, dtype=torch.long)
        self.item_category = torch.tensor(
            df["category_id_idx"].values, dtype=torch.long
        )

        # 4. Item Numerical Features
        self.item_num_features = torch.tensor(
            df[["price", "avg_item_sentiment"]].values, dtype=torch.float32
        )

        # 5. Label (Mục tiêu dự đoán)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        """Trả về tổng số lượng mẫu (samples) trong dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Lấy ra 1 mẫu dữ liệu tại vị trí idx."""
        return {
            # =====================
            # USER
            # =====================
            "user_id": self.user_ids[idx],
            "user_num": self.user_num_features[idx],
            # =====================
            # ITEM
            # =====================
            "item_id": self.item_ids[idx],
            "item_category": self.item_category[idx],
            "item_num": self.item_num_features[idx],
            "label": self.labels[idx],
        }
