# RecSys Models

Two-Tower, SASRec, LightGCN — training, evaluation, export artifacts cho serving.

## File interactions

```
train_model.py
  ├── data/build_dataset.py         — merge user_history + item_lookup, split train/val
  ├── data/feature_engineering.py   — tạo user/item features leak-proof
  ├── data/preprocessing.py         — normalize, sort by purchased_at
  ├── datasets/recsys_dataset.py    — EcommerceRecSysDataset (PyTorch Dataset)
  ├── utils/encoding.py             — fit LabelEncoder cho user/item/category
  ├── utils/scaling.py              — fit StandardScaler (5 features)
  ├── models/two_tower.py
  │     ├── models/user_tower.py    — UserTower: embedding + MLP
  │     └── models/item_tower.py    — ItemTower: embedding + category + MLP
  ├── training/trainer.py           — train_one_epoch_infonce(), validate()
  ├── training/evalute.py           — Recall@K, NDCG@K, MAP
  └── training/mlflow_logger.py     — log metrics, artifacts, model registry

retrieval/
  ├── export_embeddings.py          — chạy item_tower → save item_embeddings.npy
  └── build_faiss_index.py          — load embeddings → build IndexFlatIP → save .faiss

training/train_sasrec.py            — độc lập, dùng models/sasrec.py
training/train_lightgcn.py          — độc lập, dùng models/lightgcn.py
training/train_ltr.py               — độc lập, dùng XGBRanker cho reranker
training/tune_optuna.py             — wrap train_model.py với Optuna HPO
```

## ASCII flow

```
user_history.parquet + item_lookup.parquet
           │
           ▼
┌──────────────────────────────────────┐
│  build_dataset.py                    │
│  merge on product_id                 │
│  sort by purchased_at per user       │
│  train/val split (time-based)        │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  feature_engineering.py              │
│  user: expanding().mean().shift(1)   │  ← leak-proof
│    avg_price_preference              │
│    positive_review_ratio             │
│    total_reviews_so_far (log1p)      │
│  item: avg_item_sentiment            │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  encoding.py + scaling.py            │
│  LabelEncoder: user/item/category    │
│  StandardScaler: 5 numerical features│
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  TwoTowerModel training              │
│  InfoNCE loss + in-batch negatives   │
│  embedding_dim=32                    │
│  best checkpoint → best_two_tower.pt │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│  export_embeddings.py                │
│  item_tower(all items) → .npy        │
│  build_faiss_index.py                │
│  IndexFlatIP(dim=32) → .faiss        │
└──────────────────────────────────────┘
```

## Cấu trúc

```
recsys/
├── train_model.py
├── export_onnx.py
├── config.yaml
├── models/
│   ├── two_tower.py / user_tower.py / item_tower.py
│   ├── sasrec.py / lightgcn.py / din.py
├── data/
│   ├── build_dataset.py / feature_engineering.py / preprocessing.py
├── datasets/recsys_dataset.py
├── training/
│   ├── trainer.py / evalute.py / mlflow_logger.py
│   ├── train_sasrec.py / train_lightgcn.py / train_ltr.py
│   └── tune_optuna.py
├── retrieval/
│   ├── build_faiss_index.py / export_embeddings.py / retrieve.py
└── utils/encoding.py / scaling.py
```

## Features

### User features (leak-proof via `expanding().shift(1)`)

| Feature | Cách tính | Ví dụ | Mô tả |
|---|---|---|---|
| `avg_price_preference` | expanding mean của `price`, shifted 1 | `8500000.0` | Tầm giá trung bình user đã mua **trước** đơn hiện tại |
| `positive_review_ratio` | expanding mean của `is_positive`, shifted 1 | `0.75` | Tỷ lệ review tích cực trước đơn hiện tại |
| `total_reviews_so_far` | `log1p(cumcount)` | `1.098` (= log1p(2)) | Log số lần mua để giảm skewness |
| `has_history` | bool | `True` / `False` | False = cold start |

### Item features

| Feature | Nguồn | Ví dụ | Mô tả |
|---|---|---|---|
| `price` | `item_lookup.parquet` | `20990000.0` | Giá sản phẩm (VND) |
| `avg_item_sentiment` | `item_lookup.parquet` | `0.82` | Sentiment trung bình tất cả reviews của item |
| `category_id` | `item_lookup.parquet` | `"dien-thoai"` → `idx 3` | Sau LabelEncode |

### Model inputs (sau StandardScaler)

**User tower tensor:**

| Tensor | Shape | Nội dung |
|---|---|---|
| `user_id` | `(B,)` int64 | User embedding index sau LabelEncode |
| `user_num` | `(B, 3)` float32 | `[total_reviews_so_far, avg_price_preference, positive_review_ratio]` scaled |

**Item tower tensor:**

| Tensor | Shape | Nội dung |
|---|---|---|
| `item_id` | `(B,)` int64 | Item embedding index sau LabelEncode |
| `item_category` | `(B,)` int64 | Category embedding index |
| `item_num` | `(B, 2)` float32 | `[price, avg_item_sentiment]` scaled |

**Example feature values trước scaling:**

| customer_id | avg_price_preference | positive_review_ratio | total_reviews_so_far | product_id | price | avg_item_sentiment |
|---|---|---|---|---|---|---|
| `"7290341"` | `18000000.0` | `1.0` | `0.0` | `"277512620"` | `20990000.0` | `0.82` |
| `"7290341"` | `19495000.0` | `1.0` | `0.693` | `"279257510"` | `15990000.0` | `0.71` |

> Dòng 2: user mua lần 2 — `avg_price_preference` đã tính cả đơn 1 (expanding), `total_reviews_so_far = log1p(1) = 0.693`.

### Model output

- **TwoTowerModel.forward():** dot product scores (B,) — dùng khi training
- **user_tower():** user_vector (B, embedding_dim=32) — dùng lúc serving
- **item_tower():** item_vector (B, embedding_dim=32) — precomputed tất cả items lúc startup

## Train

```bash
# Two-Tower (main model)
python src/ml_models/recsys/train_model.py

# SASRec (session-based)
python src/ml_models/recsys/training/train_sasrec.py

# LightGCN (graph-based, optional)
python src/ml_models/recsys/training/train_lightgcn.py

# LTR reranker
python src/ml_models/recsys/training/train_ltr.py

# HPO với Optuna
python src/ml_models/recsys/training/tune_optuna.py --n-trials 50
```

## Artifacts

```
artifacts/recsys_models/
├── encoders/
│   ├── user_encoder.pkl
│   ├── item_encoder.pkl
│   └── cat_encoder.pkl
├── scalers/scaler.pkl
├── model/best_two_tower.pt
├── data_menu/
│   ├── item_lookup.parquet
│   └── user_history.parquet
├── faiss_index/item_index.faiss
└── lightgcn_embeddings.npz     (optional)
```

### `item_lookup.parquet` — example rows

| product_id | product_name | category_id | brand_name | price | avg_item_sentiment | thumbnail_url |
|---|---|---|---|---|---|---|
| `"277512620"` | `"Samsung Galaxy S24 Ultra"` | `"dien-thoai"` | `"Samsung"` | `20990000.0` | `0.82` | `"https://cdn.tiki.vn/..."` |
| `"279257510"` | `"iPhone 15 Pro Max 256GB"` | `"dien-thoai"` | `"Apple"` | `28990000.0` | `0.91` | `"https://cdn.tiki.vn/..."` |

### `user_history.parquet` — example rows

| customer_id | product_id | purchased_at | avg_price_preference | positive_review_ratio | total_reviews_so_far |
|---|---|---|---|---|---|
| `"7290341"` | `"277512620"` | `2025-10-15 14:22:00` | `18000000.0` | `1.0` | `0.0` |
| `"7290341"` | `"279257510"` | `2025-10-28 09:05:00` | `19495000.0` | `1.0` | `0.693` |
| `"8801234"` | `"277574339"` | `2025-10-20 17:30:00` | `5000000.0` | `0.0` | `0.0` |

## MLflow

Experiment: `recsys_two_tower`

| Metric | Ví dụ | Mô tả |
|---|---|---|
| `train/loss` | `2.34` | InfoNCE loss per epoch |
| `val/recall_at_10` | `0.18` | Recall@10 trên validation set |
| `val/ndcg_at_10` | `0.12` | NDCG@10 |
| `val/map_at_10` | `0.09` | MAP@10 |

Model registry: `recsys-two-tower` stage `Production` → serving API auto-load.

## Thiết kế kỹ thuật

**Tại sao `expanding().mean().shift(1)` thay vì `.mean()` thẳng?**
`.mean()` thẳng sẽ tính trung bình bao gồm cả đơn hàng hiện tại — tức là model biết thông tin về tương lai lúc training (data leakage). Ví dụ: đơn thứ 3 của user dùng `avg_price = mean([order1, order2, order3])` — nhưng lúc serving, đơn 3 chưa xảy ra nên không thể biết giá order3. `shift(1)` đảm bảo feature tại thời điểm t chỉ dùng thông tin đến t-1.

**Tại sao InfoNCE thay vì BPR (Bayesian Personalized Ranking)?**
BPR sample 1 negative per positive — với sparse data (1 user mua 1 item), tỷ lệ negative/positive thấp, loss noisy. InfoNCE dùng in-batch negatives: với batch size B, mỗi sample có B-1 negatives — hiệu quả hơn rất nhiều mà không cần sampling thêm. Cũng có theoretical backing: InfoNCE maximize mutual information giữa user và item representations.

**Tại sao `embedding_dim=32` thay vì 64/128?**
Dataset nhỏ (~15k interactions). High embedding dim với sparse data → overfitting mạnh, user embedding không học được meaningful representation. Dim 32 đủ để encode category + price preference. Khi có nhiều data hơn (>100k interactions) có thể tăng lên 64.

**Tại sao `log1p(cumcount)` cho `total_reviews_so_far`?**
Cumcount phân phối rất skewed: 60% users có 1 interaction, một số users có 20+ interactions. Log transform → scale về cùng range, tránh users nhiều interaction dominate gradient. `log1p` thay vì `log` để xử lý cumcount=0 (first interaction).

**Tại sao FAISS IndexFlatIP (inner product) thay vì L2?**
Two-Tower output vectors đã được normalize (unit norm). Inner product giữa unit vectors = cosine similarity. IndexFlatIP với normalized vectors là ANN search chính xác theo cosine similarity — đúng với optimization objective của InfoNCE (maximize dot product giữa user/item vectors match).

## Troubleshooting

**`FileNotFoundError: user_encoder.pkl`**
Training chưa chạy, artifacts chưa được generate. Chạy:
```bash
python src/ml_models/recsys/train_model.py
```
Hoặc dùng fake data nếu chưa có real data:
```bash
python scripts/generate_fake_data.py  # tạo item_lookup + user_history
python src/ml_models/recsys/train_model.py
```

**MLflow `ConnectionRefused` khi log metrics**
MLflow server chưa chạy. Train vẫn hoạt động bình thường (model save local), chỉ không log được. Kiểm tra:
```bash
docker compose -f docker-compose.infra.yml ps mlflow
```

**`FAISS index not found` khi khởi động recsys_api**
Export embeddings và build index sau training:
```bash
python src/ml_models/recsys/retrieval/export_embeddings.py
python src/ml_models/recsys/retrieval/build_faiss_index.py
```

**val/recall_at_10 = 0.0 sau epoch 1**
Bình thường với sparse data. Cần ít nhất 5–10 epochs để embedding space ổn định. Nếu sau 20 epochs vẫn 0.0 → kiểm tra data leakage (features có dùng `.shift(1)` chưa).

**`ValueError: could not convert string to float`**
`customer_id` hoặc `product_id` chứa ký tự `.0` (float string). `loaders.py` có xử lý nhưng kiểm tra encoding step:
```python
df["customer_id"] = df["customer_id"].astype(str).str.replace(".0", "", regex=False)
```
