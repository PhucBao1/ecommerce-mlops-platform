# NLP — PhoBERT Sentiment Analysis

Fine-tune PhoBERT cho phân tích cảm xúc review tiếng Việt (3 class: Negative/Neutral/Positive). Kết quả feed vào scoring pipeline của RecSys.

## File interactions

```
train_phobert.py
  └── artifacts/nlp_models/phobert/version_001/  — lưu model weights + tokenizer

spark_batch_infer.py
  ├── spark/session.py              — SparkSession factory
  ├── artifacts/nlp_models/phobert/ — load model
  └── lakehouse.silver.cleaned_comment  — đọc input
        └── lakehouse.gold.comment_predictions  — ghi output (MERGE INTO)

src/data_pipeline/streaming/sentiment_model.py
  └── artifacts/nlp_models/phobert/ — load model (serving real-time)

src/serving/nlp_api/phobert_api.py
  └── artifacts/nlp_models/phobert/ — load model (HTTP API)
```

## ASCII flow

```
silver.cleaned_comment
  clean_comment (text)
  rating (1–5 → label)
        │
        ▼
┌────────────────────────────────────┐
│  train_phobert.py                  │
│                                    │
│  rating → label mapping:           │
│    ≤2 → Negative(0)                │
│    =3 → Neutral(2)                 │
│    ≥4 → Positive(1)                │
│                                    │
│  tokenize (max_length=256)         │
│  PhoBERT encoder                   │
│  linear classifier (3 classes)     │
│  cross-entropy + class weights     │
│  → save model + tokenizer          │
└───────────────┬────────────────────┘
                │
                ▼
  artifacts/nlp_models/phobert/version_001/

                │  batch inference
                ▼
┌────────────────────────────────────┐
│  spark_batch_infer.py              │
│  Spark UDF wraps PhoBERT           │
│  → sentiment_label, scores (4)     │
│  MERGE INTO gold.comment_predictions│
└────────────────────────────────────┘
```

## Files

| File | Mô tả |
|---|---|
| `train_phobert.py` | Fine-tuning script — Hugging Face Trainer |
| `spark_batch_infer.py` | Batch inference trên Silver table bằng Spark UDF |

## Model

| Thuộc tính | Giá trị |
|---|---|
| Base model | `wonrax/phobert-base-vietnamese-sentiment` |
| Task | 3-class sequence classification |
| Labels | `0=Negative`, `1=Positive`, `2=Neutral` |
| Max length | 256 tokens |
| Loss | Cross-entropy + class weights (chống imbalanced) |

## Training data schema (`silver.cleaned_comment`)

Input vào training:

| Field | Type | Ví dụ |
|---|---|---|
| `review_id` | str | `"48392011"` |
| `clean_comment` | str | `"Pin trâu, camera chụp siêu nét"` |
| `rating` | float32 | `5.0` |

Label mapping từ rating:

| rating | label_id | label_name |
|---|---|---|
| 1–2 | 0 | `Negative` |
| 3 | 2 | `Neutral` |
| 4–5 | 1 | `Positive` |

**Example training batch:**

| clean_comment | rating | label_id | label_name |
|---|---|---|---|
| `"Pin trâu, camera chụp siêu nét"` | `5.0` | `1` | `Positive` |
| `"Máy nóng, pin tụt nhanh quá"` | `2.0` | `0` | `Negative` |
| `"Tạm được, không quá tệ"` | `3.0` | `2` | `Neutral` |
| `null` (empty text) | `4.0` | auto `Neutral` | skip model |

## Model output schema

### Online (API response — `POST /predict`)

| Field | Type | Ví dụ |
|---|---|---|
| `text` | str | `"Pin trâu, camera chụp siêu nét"` |
| `sentiment` | str | `"Positive"` |
| `confidence` | str | `"95.21%"` |
| `neg_score` | float | `0.02` |
| `pos_score` | float | `0.95` |
| `neu_score` | float | `0.03` |
| `label_id` | int | `1` |

### Batch inference (`gold.comment_predictions`)

| Field | Type | Ví dụ |
|---|---|---|
| *(tất cả fields Silver)* | | Giữ nguyên từ cleaned_comment  |
| `sentiment_label` | str | `"Positive"` |
| `sentiment_score` | float32 | `0.95` | Confidence của class được chọn (0–1)
| `neg_score` | float32 | `0.02` | P(Negative)
| `neu_score` | float32 | `0.03` | P(Neutral)
| `pos_score` | float32 | `0.95` | P(Positive)
## Train

```bash
# Fine-tune (cần GPU, hoặc Colab/Kaggle — xem TRAIN.md)
python src/ml_models/nlp/train_phobert.py \
  --model-path wonrax/phobert-base-vietnamese-sentiment \
  --output-dir artifacts/nlp_models/phobert/version_001 \
  --epochs 3 \
  --batch-size 16

# Batch inference: Silver → Gold
spark-submit src/ml_models/nlp/spark_batch_infer.py \
  --input-table lakehouse.silver.cleaned_comment \
  --output-table lakehouse.gold.comment_predictions \
  --model-path artifacts/nlp_models/phobert/version_001
```

## Artifacts

```
artifacts/nlp_models/phobert/
└── version_001/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── special_tokens_map.json
    └── tokenizer_config.json
```

## MLflow tracking

Experiment: `nlp_sentiment`

| Metric logged | Mô tả |
|---|---|
| `eval/f1_macro` | F1 macro trên validation set |
| `eval/accuracy` | Accuracy |
| `eval/loss` | Validation loss |
| `train/loss` | Training loss per step |

## Thiết kế kỹ thuật

**Tại sao dùng `wonrax/phobert-base-vietnamese-sentiment`?**
Đây là PhoBERT đã được pre-train trên corpus tiếng Việt lớn, có sẵn Vietnamese tokenizer (BPE với vocabulary 64k tokens tiếng Việt). Fine-tune từ checkpoint này → hội tụ nhanh hơn train từ BERT tiếng Anh, và hiểu được đặc trưng ngôn ngữ tiếng Việt (dấu thanh, từ ghép, teen code thông dụng trên Tiki).

**Tại sao `max_length=256` thay vì 512?**
Phần lớn review Tiki ngắn: 90% review dưới 200 tokens. Tăng 512 → tốn bộ nhớ gấp đôi (attention O(n²)), giảm batch size xuống còn 8 thay vì 16, training chậm hơn mà accuracy không cải thiện đáng kể. Truncate tại 256 chỉ ảnh hưởng ~10% review dài bất thường.

**Tại sao dùng class weights cho loss?**
Data Tiki tự nhiên highly imbalanced: ~70% Positive, ~15% Neutral, ~15% Negative. Train với cross-entropy thuần → model bias toward Positive, F1 Negative thấp. Class weights = {Negative: 3.0, Neutral: 2.0, Positive: 1.0} → buộc model học Negative features tốt hơn — quan trọng vì Negative là signal mạnh nhất cho RecSys suppress.

**Tại sao text rỗng → auto `Neutral` thay vì chạy model?**
Tokenizer với text rỗng ra `[CLS][SEP]` — model không có signal, output gần uniform distribution, confidence thấp (~33%). Gán thẳng Neutral với confidence 100% là safer: không tạo noise trong feature store, không tốn GPU compute cho input vô nghĩa.

## Troubleshooting

**CUDA OOM khi train với `batch_size=16`**
Giảm batch size và dùng gradient accumulation để giữ effective batch:
```python
# Trong train_phobert.py
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # effective batch = 16
)
```

**`OSError: model path not found`**
Model chưa được train. Chạy training trước hoặc tải checkpoint từ HuggingFace:
```bash
# Dùng base model chưa fine-tune để test API
MODEL_PATH=wonrax/phobert-base-vietnamese-sentiment \
  uvicorn src.serving.nlp_api.phobert_api:app --port 8000
```

**Spark UDF OOM khi batch inference trên dataset lớn**
PhoBERT load vào executor memory. Mỗi executor cần ~1.5GB. Với 4 executor × 1.5GB = 6GB. Điều chỉnh:
```bash
SPARK_EXECUTOR_MEMORY=2g SPARK_EXECUTOR_CORES=1 \
spark-submit src/ml_models/nlp/spark_batch_infer.py
```

**F1 Negative thấp (<0.6) dù F1 macro ổn**
Data imbalanced chưa được xử lý đủ. Thêm class weight mạnh hơn hoặc oversample Negative:
```python
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
# Thường ra: Negative≈3.2, Neutral≈2.1, Positive≈0.8
```

**Batch inference rất chậm trên CPU (>1h cho 100k reviews)**
Bình thường — PhoBERT trên CPU ~50 reviews/giây. Dùng GPU hoặc Kaggle/Colab cho batch inference. Xem [TRAIN.md](../../../TRAIN.md) để hướng dẫn chạy trên Colab.
