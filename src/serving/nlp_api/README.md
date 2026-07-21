# PhoBERT Sentiment API

FastAPI service — phân tích cảm xúc review tiếng Việt (batch inference). Port **8000**.

## File interactions

```
phobert_api.py
  └── artifacts/nlp_models/phobert/version_001/  — load tokenizer + model khi startup

(không import module nào khác trong project — standalone service)
```

## ASCII flow

```
POST /predict  {"texts": ["...", "..."]}
        │
        ▼
┌──────────────────────────────────────┐
│  1. Tách valid texts / empty texts   │
│     empty → Neutral, confidence 100% │
│     (không gọi model)                │
│                                      │
│  2. Tokenize valid texts             │
│     AutoTokenizer (max_length=256)   │
│     truncation=True, padding=True    │
│                                      │
│  3. model.forward() → logits         │
│     softmax → [neg, pos, neu]        │
│                                      │
│  4. Merge kết quả valid + empty      │
│     giữ đúng thứ tự index            │
└──────────────────────────────────────┘
        │
        ▼
{"results": [{sentiment, confidence, scores...}]}
```

## Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "model_version": "v1.0.0", "device": "cpu"}
```

### `POST /predict`

**Request schema:**

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `texts` | List[str] | ✅ | Danh sách texts cần phân tích |

**Request example:**
```json
{
  "texts": [
    "Pin trâu, camera chụp siêu nét, màn hình đẹp!",
    "Hàng kém chất lượng, giao hàng chậm.",
    "Tạm được, không quá tệ.",
    ""
  ]
}
```

**Response schema (per item):**

| Field | Type | Mô tả | Ví dụ |
|---|---|---|---|
| `text` | str | Text đầu vào gốc | `"Pin trâu, camera..."` |
| `sentiment` | str | Class được dự đoán | `"Positive"` |
| `confidence` | str | Confidence của class (%) | `"95.21%"` |
| `neg_score` | float | P(Negative) ∈ [0, 1] | `0.02` |
| `pos_score` | float | P(Positive) ∈ [0, 1] | `0.95` |
| `neu_score` | float | P(Neutral) ∈ [0, 1] | `0.03` |
| `label_id` | int | 0=Negative, 1=Positive, 2=Neutral | `1` |

**Response example:**
```json
{
  "results": [
    {
      "text": "Pin trâu, camera chụp siêu nét, màn hình đẹp!",
      "sentiment": "Positive",
      "confidence": "95.21%",
      "neg_score": 0.02, "pos_score": 0.95, "neu_score": 0.03,
      "label_id": 1
    },
    {
      "text": "Hàng kém chất lượng, giao hàng chậm.",
      "sentiment": "Negative",
      "confidence": "88.43%",
      "neg_score": 0.88, "pos_score": 0.05, "neu_score": 0.07,
      "label_id": 0
    },
    {
      "text": "Tạm được, không quá tệ.",
      "sentiment": "Neutral",
      "confidence": "61.20%",
      "neg_score": 0.22, "pos_score": 0.17, "neu_score": 0.61,
      "label_id": 2
    },
    {
      "text": "",
      "sentiment": "Neutral",
      "confidence": "100.00%",
      "neg_score": 0.0, "pos_score": 0.0, "neu_score": 1.0,
      "label_id": 2
    }
  ]
}
```

## Model

- **Base:** `wonrax/phobert-base-vietnamese-sentiment`
- **Path:** `artifacts/nlp_models/phobert/version_001` (env `MODEL_PATH`)
- Load một lần duy nhất lúc startup, giữ trong memory suốt lifetime process

## Xử lý edge cases

| Input | Xử lý |
|---|---|
| Text rỗng / chỉ spaces | → `Neutral`, confidence 100%, không gọi model |
| `texts = []` | → `{"results": []}` ngay lập tức |
| Text quá dài | → Tokenizer tự truncate tại `max_length=256` |
| Mixed valid + empty | → Process valid texts theo batch, merge kết quả giữ đúng thứ tự |

## Env vars

| Var | Default | Mô tả |
|---|---|---|
| `MODEL_PATH` | `./artifacts/nlp_models/phobert/version_001` | Đường dẫn tới model directory |
| `MODEL_VERSION` | `v1.0.0` | Version string cho `/health` |

## Chạy local

```bash
uvicorn src.serving.nlp_api.phobert_api:app --port 8000 --reload
# hoặc Docker
docker compose -f docker-compose.app.yml up nlp-api
```

Swagger UI: `http://localhost:8000/docs`

## Thiết kế kỹ thuật

**Tại sao tách valid/empty texts trước khi inference?**
Tránh tốn GPU compute cho empty strings. Quan trọng hơn: batch với mixed empty/valid texts gây padding không đều, ảnh hưởng attention mask. Tách rõ → model chỉ xử lý batch thuần valid texts → kết quả consistent hơn.

**Tại sao giữ đúng thứ tự index khi merge kết quả?**
Caller của API (streaming consumer, batch inference) ghép kết quả với input theo index. Nếu thứ tự sai → sentiment sai người, ảnh hưởng Feast feature push và Gold table.

**Tại sao model load synchronous lúc startup, không lazy?**
Nếu lazy load (load lần đầu khi có request), request đầu tiên sẽ chậm 10–30 giây. Với API serving production, p99 latency đột biến là không acceptable. Load upfront → mọi request có latency consistent từ đầu.

## Troubleshooting

**Startup rất chậm (30–60s)**
Bình thường — PhoBERT ~430MB, load từ disk vào RAM/VRAM mất thời gian. Chỉ xảy ra một lần. Subsequent requests < 100ms.

**`OSError: MODEL_PATH not found`**
Model artifacts chưa được tạo. Chạy training trước hoặc dùng base model:
```bash
MODEL_PATH=wonrax/phobert-base-vietnamese-sentiment \
  docker compose -f docker-compose.app.yml up nlp-api
```

**`RuntimeError: CUDA out of memory` khi inference batch lớn**
Batch quá lớn (>64 texts). Split phía client:
```python
def predict_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = requests.post("/predict", json={"texts": chunk})
        results.extend(resp.json()["results"])
    return results
```

**Kết quả trả về sai thứ tự (result[0] không correspond với texts[0])**
Bug trong merge logic. Kiểm tra `valid_indices` list có được sort đúng không. Đây là known gotcha khi mix empty và valid texts.

**API `/predict` block quá lâu với batch 100+ texts**
API hiện tại synchronous. Không dùng `asyncio.to_thread` → block event loop. Workaround: giảm batch size hoặc gọi API qua nhiều request song song từ client.
