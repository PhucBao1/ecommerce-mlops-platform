# Agent API

LLM agent service — hỏi đáp sản phẩm, so sánh, tư vấn dựa trên RAG + LangGraph. Port **8003**.

## File interactions

```
main.py
  ├── rag.py (RAGPipeline)
  │     ├── sentence_transformers      — embed product_name (bkai vietnamese-bi-encoder)
  │     ├── FAISS IndexFlatIP          — dense ANN search
  │     ├── TfidfVectorizer            — sparse search (char ngram 2-4)
  │     └── vector_store.py            — Qdrant (nếu VECTOR_STORE_BACKEND=qdrant)
  ├── reranker.py (CrossEncoderReranker)
  │     └── cross-encoder/ms-marco-MiniLM-L-6-v2 — neural reranker (nếu RERANKER_BACKEND=neural)
  ├── indexer.py (KBIndexer)
  │     └── chunker.py                 — KB policy/FAQ docs → chunks → FAISS KB index
  ├── ingestion/
  │     ├── pdf_loader.py              — load PDF files vào KB
  │     ├── excel_loader.py            — load Excel/CSV vào KB
  │     └── web_loader.py              — scrape web pages vào KB
  ├── memory.py (MemoryStore)
  │     └── Redis                      — lưu conversation history per session (TTL 24h)
  ├── guardrails.py (Guardrails)       — check prompt injection, PII, unsafe content
  ├── cache.py (SemanticCache)
  │     ├── rag.py._model              — reuse cùng embedding model (không load 2 lần)
  │     └── Redis                      — cache answers by cosine similarity
  ├── tracing.py                       — OTel TracerProvider + Prometheus metrics
  ├── security.py                      — auth /admin/* (fail-CLOSED) + rate limit theo IP
  ├── cost_guard.py                    — budget/user/ngày + circuit breaker toàn cục
  ├── graph.py (build_graph)           — LangGraph StateGraph, 7 node:
  │     │                                summarize → router →
  │     │                                (kb_search|product_search|recommend|agent) ↔ tools
  │     │                                checkpointer: RedisSaver (thread_id=customer_id)
  │     ├── prompt_registry.py         — get_active_prompt() đọc lúc RUNTIME (hot-swap)
  │     └── tools.py                   — search_products, get_recommendations,
  │                                      filter_by_price, get_product_detail, search_kb
  ├── policy_engine.py (PolicyEngine)  — rate limit, user tier, tool permission
  └── prompt_registry.py              — versioned prompts, log version vào MLflow

chunker.py                             — standalone util, dùng khi index KB docs dài

eval/
  ├── ragas_eval.py                    — RAGAS: faithfulness, answer_relevancy, context_precision
  └── deepeval_eval.py                 — DeepEval: AnswerRelevancy, Faithfulness, ContextualRecall
```

## Files

| File | Mô tả |
|---|---|
| `main.py` | FastAPI app + lifespan (khởi tạo all singletons) |
| `graph.py` | `build_graph()` — LangGraph StateGraph 5 node + `run_graph_stream()` (SSE) |
| `security.py` | Auth `/admin/*` (fail-CLOSED), rate limit theo IP |
| `cost_guard.py` | Budget/user/ngày + circuit breaker chi phí toàn cục |
| `rag.py` | `RAGPipeline` — embed + hybrid search + scoring |
| `memory.py` | `MemoryStore` — Redis-backed conversation history |
| `guardrails.py` | `Guardrails` — chặn prompt injection, PII, unsafe content |
| `policy_engine.py` | `PolicyEngine` — rate limit, user tier, tool permission |
| `cache.py` | `SemanticCache` — cache answer theo cosine similarity |
| `tools.py` | LangGraph tools: search_products, get_reviews, compare |
| `chunker.py` | Chunker cho knowledge base docs dài (policy, manual) |
| `indexer.py` | `KBIndexer` — embed + index KB chunks vào FAISS riêng |
| `reranker.py` | `CrossEncoderReranker` — neural reranker (RERANKER_BACKEND=neural) |
| `tracing.py` | OTel TracerProvider + Prometheus metrics cho agent_api |
| `prompt_registry.py` | Versioned prompt templates + MLflow tracking |
| `ingestion/pdf_loader.py` | Load PDF vào KB |
| `ingestion/excel_loader.py` | Load Excel/CSV vào KB |
| `ingestion/web_loader.py` | Scrape web pages vào KB |
| `eval/ragas_eval.py` | RAGAS eval: faithfulness, answer_relevancy, context_precision |
| `eval/deepeval_eval.py` | DeepEval: AnswerRelevancy, Faithfulness, ContextualRecall |

## Endpoints

### `POST /chat`

Chat thường (non-streaming).

**Request schema:**

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `message` | str | ✅ | Câu hỏi của user |
| `session_id` | str | ❌ | ID session để giữ conversation history |
| `customer_id` | str | ❌ | ID user (optional, cho personalization) |

```bash
curl -X POST http://localhost:8003/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "So sánh iPhone 15 vs Samsung S24", "session_id": "sess_abc"}'
```

**Response schema:**

| Field | Type | Mô tả |
|---|---|---|
| `answer` | str | Câu trả lời của agent |
| `sources` | List[str] | Product IDs / review IDs làm nguồn |
| `tool_calls` | List[str] | Tools đã được gọi trong request |
| `cached` | bool | True nếu lấy từ SemanticCache |
| `trace_id` | str | OTel trace ID |

### `POST /chat/stream`

Streaming response (Server-Sent Events).

**Response:** `text/event-stream`, mỗi dòng là một token.

### `GET /health`

```json
{"status": "ok", "llm_backend": "ollama", "rag_ready": true}
```

### `GET /metrics`

Prometheus scrape endpoint.

## Kiến trúc

```
POST /chat  {"message": "Điện thoại chụp ảnh đẹp dưới 10 triệu", "session_id": "s1"}
        │
        ├─► Guardrails.check()
        │   ├── prompt injection keyword → BLOCK
        │   ├── PII (email/phone/CMND) → mask trước khi xử lý
        │   └── unsafe topic → BLOCK
        │
        ├─► SemanticCache.lookup(message, threshold=0.95)
        │   └── cosine ≥ 0.95 → return cached answer (~10ms)
        │
        ├─► PolicyEngine.check()
        │   └── rate limit / user tier check
        │
        ├─► MemoryStore.load_history(session_id)
        │   └── Redis → last N turns của session
        │
        ├─► LangGraph StateGraph.run()
        │   │
        │   ├── Router node: classify intent
        │   │   ├── "search"  → RAG node
        │   │   ├── "compare" → compare_items tool
        │   │   └── "reviews" → get_reviews tool
        │   │
        │   ├── RAG node: RAGPipeline.search(query)
        │   │   ├── extract_price_filter() ← regex tiếng Việt
        │   │   │   "dưới 10 triệu" → max_price=10_000_000
        │   │   ├── dense_search  (FAISS)   → top-50
        │   │   ├── sparse_search (TF-IDF)  → top-50
        │   │   ├── RRF fusion: Σ 1/(60+rank)
        │   │   ├── price filter
        │   │   ├── score = 0.6×rrf + 0.2×sentiment + 0.2×(1-price/max)
        │   │   └── [optional] CrossEncoder reranker
        │   │
        │   └── LLM.generate(context=top_k_products + history)
        │       Ollama Qwen2.5 7B  hoặc  Claude API
        │
        ├─► MemoryStore.save_turn(session_id, user_msg, assistant_msg)
        └─► SemanticCache.set(message, answer)
```

## RAG Pipeline (`rag.py`)

Dữ liệu nguồn là **product records** từ `item_lookup.parquet` (ngắn — không cần chunk).
Tài liệu dài (policy, manual) được xử lý qua `chunker.py` trước khi index riêng.

### Index (build lúc startup)

| Index | Kỹ thuật | Mô tả |
|---|---|---|
| Dense | FAISS IndexFlatIP | Encode `product_name` bằng `bkai-foundation-models/vietnamese-bi-encoder` |
| Sparse | TF-IDF (char ngram 2-4, 20k features) | Lexical matching tên sản phẩm |

### Search flow

```
query
  ├── extract_price_filter() — regex Vietnamese: "dưới 5 triệu", "từ 2tr đến 10tr"
  ├── dense_search (FAISS) → top-50 indices
  ├── sparse_search (TF-IDF cosine) → top-50 indices
  ├── RRF fusion: score = Σ 1/(60 + rank)
  ├── price filter (pandas post-filter hoặc Qdrant server-side)
  ├── Rule-based scoring:
  │     0.6×(rrf/max) + 0.2×(sentiment/max) + 0.2×(1 - price/max_price)
  └── Optional neural reranker (CrossEncoderReranker, default=off)
```

### Document chunker (`chunker.py`)

Dành cho knowledge base docs dài (FAQ, policy, hướng dẫn sử dụng):
- Sentence-aware sliding window
- `chunk_size = 1500 chars` (~300 tokens), `overlap = 200 chars`
- Output: `List[Chunk]` với `source`, `chunk_index`, `metadata`
- Product records **không đi qua chunker** (đã ngắn sẵn)

### RAG item schema (trả về từ `search()`)

| Field | Type | Mô tả |
|---|---|---|
| `product_id` | str | ID sản phẩm |
| `product_name` | str | Tên sản phẩm |
| `price` | float | Giá (VND) |
| `category_name` | str | Tên danh mục |
| `brand_name` | str | Thương hiệu |
| `avg_sentiment` | float | Sentiment trung bình (0–1) |
| `thumbnail_url` | str | URL ảnh |
| `score` | float | Điểm tổng hợp (RRF + sentiment + price) |

**Example output** (query: `"điện thoại chụp ảnh đẹp dưới 10 triệu"`):

| product_id | product_name | price | brand_name | avg_sentiment | score |
|---|---|---|---|---|---|
| `"276348291"` | `"Samsung Galaxy A55 5G 128GB"` | `9990000.0` | `"Samsung"` | `0.78` | `0.842` |
| `"279100234"` | `"iPhone SE 2023 64GB"` | `9490000.0` | `"Apple"` | `0.85` | `0.801` |
| `"278201045"` | `"Xiaomi Redmi Note 13 Pro 256GB"` | `7990000.0` | `"Xiaomi"` | `0.71` | `0.763` |

## LangGraph tools

| Tool | Mô tả | Input | Output |
|---|---|---|---|
| `search_products` | Tìm sản phẩm theo query | query: str, category?: str | List[Product] |
| `get_reviews` | Lấy reviews của product | product_id: str, limit: int | List[Review] |
| `compare_items` | So sánh 2+ sản phẩm | product_ids: List[str] | ComparisonTable |

## Memory schema (Redis)

Key: `memory:{session_id}`, TTL: 24h

Mỗi turn lưu:

| Field | Mô tả |
|---|---|
| `role` | `"user"` / `"assistant"` |
| `content` | Nội dung message |
| `timestamp` | ISO 8601 |
| `tool_calls` | Tools được gọi trong turn đó |

**Example conversation history:**
```json
[
  {
    "role": "user",
    "content": "Tui cần mua điện thoại tầm 10 triệu, chụp ảnh đẹp",
    "timestamp": "2025-11-01T10:00:00Z",
    "tool_calls": []
  },
  {
    "role": "assistant",
    "content": "Với tầm giá 10 triệu, tôi gợi ý Samsung Galaxy A55 5G...",
    "timestamp": "2025-11-01T10:00:02Z",
    "tool_calls": ["search_products"]
  },
  {
    "role": "user",
    "content": "Còn option nào rẻ hơn không?",
    "timestamp": "2025-11-01T10:00:30Z",
    "tool_calls": []
  }
]
```

## Guardrails

| Check | Hành động khi vi phạm | Ví dụ trigger |
|---|---|---|
| Prompt injection keywords | Block, return error message | `"ignore previous instructions"` |
| PII (email, phone, CMND) | Mask trước khi gửi LLM | `"email tui là abc@gmail.com"` |
| Unsafe topics | Block | `"hướng dẫn làm bom"` |
| Response length > 2000 chars | Truncate | Câu trả lời dài bất thường |

## Prometheus metrics

| Metric | Type | Mô tả |
|---|---|---|
| `guardrail_block_total` | Counter | Số requests bị guardrails chặn |
| `search_cache_hit_total` | Counter | SemanticCache hits |
| `agent_latency_seconds` | Histogram | End-to-end agent latency |
| `llm_token_count_total` | Counter | Tổng tokens đã dùng (input + output) |

## Env vars

| Var | Default | Mô tả |
|---|---|---|
| `AGENT_LLM_BACKEND` | `ollama` | `"ollama"` (local) hoặc `"claude"` (Anthropic API) |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama service |
| `ANTHROPIC_API_KEY` | — | Dùng khi `AGENT_LLM_BACKEND=claude` |
| `VECTOR_STORE_BACKEND` | `faiss` | `"faiss"` hoặc `"qdrant"` — cho RAG index |
| `RAG_QDRANT_COLLECTION` | `rag_items` | Qdrant collection cho RAG (tách khỏi recsys collection) |
| `RERANKER_BACKEND` | `rule` | `"rule"` (default) hoặc `"neural"` (CrossEncoder) |
| `EMBEDDING_MODEL` | `bkai-foundation-models/vietnamese-bi-encoder` | Sentence-transformer model |
| `REDIS_HOST` | `redis` | Memory + SemanticCache store |
| `AGENT_PROMPT_VERSION` | `v1` | Prompt version (tracked trong MLflow) |

## Chạy

```bash
# Ollama backend (local, miễn phí)
docker compose -f docker-compose.app.yml --profile agent up agent-api

# Claude backend (cần API key)
AGENT_LLM_BACKEND=claude ANTHROPIC_API_KEY=sk-... \
  docker compose -f docker-compose.app.yml --profile agent up agent-api
```

Swagger UI: `http://localhost:8003/docs`

## KB Indexer (`indexer.py` + `ingestion/`)

Tách biệt với product FAISS index trong `rag.py`. Dùng cho policy/FAQ/warranty KB docs.

```bash
# Thêm KB từ PDF
from src.serving.agent_api.ingestion.pdf_loader import PDFLoader
from src.serving.agent_api.chunker import chunk_documents
from src.serving.agent_api.indexer import KBIndexer

docs = PDFLoader().load("docs/tiki_policy.pdf")
chunks = chunk_documents(docs)          # chunk_size=1500, overlap=200
indexer = KBIndexer()
indexer.add_chunks(chunks)
indexer.save("/app/artifacts/kb_index/")
```

---

## Evaluation (`eval/`)

Đánh giá chất lượng RAG pipeline theo 2 framework: **RAGAS** và **DeepEval**.

### RAGAS (`eval/ragas_eval.py`)

Đánh giá `RAGPipeline.search()` với 20 Vietnamese product queries. LLM judge: Claude Haiku.

| Metric | Ý nghĩa | Threshold tốt |
|---|---|---|
| `faithfulness` | Câu trả lời có dựa trên context retrieved không? | ≥ 0.7 |
| `answer_relevancy` | Context có trả lời đúng câu hỏi không? | ≥ 0.7 |
| `context_precision` | Các context retrieved có liên quan không? | ≥ 0.7 |

```bash
# Dùng Claude Haiku làm LLM judge
ANTHROPIC_API_KEY=sk-ant-... \
  python -m src.serving.agent_api.eval.ragas_eval --prompt-version 1

# Neural reranker
RERANKER_BACKEND=neural ANTHROPIC_API_KEY=sk-ant-... \
  python -m src.serving.agent_api.eval.ragas_eval

# Không log MLflow
python -m src.serving.agent_api.eval.ragas_eval --no-mlflow
```

Kết quả log vào MLflow experiment `ragas_rag_eval`:
```
ragas_faithfulness:       0.82
ragas_answer_relevancy:   0.79
ragas_context_precision:  0.74
```

### DeepEval (`eval/deepeval_eval.py`)

Đánh giá toàn bộ agent flow (RAG + LLM response) với 10 test cases gồm cả **policy** và **product** queries. LLM judge: Ollama hoặc Claude (theo `AGENT_LLM_BACKEND`).

| Metric | Ý nghĩa |
|---|---|
| `AnswerRelevancyMetric` | Response có trả lời đúng câu hỏi không? |
| `FaithfulnessMetric` | Response có bịa đặt ngoài context không? |
| `ContextualRecallMetric` | Context có chứa expected answer không? |

```bash
# Dùng Ollama local (cần ollama đang chạy)
python -m src.serving.agent_api.eval.deepeval_eval --prompt-version v1

# Dùng Claude judge
AGENT_LLM_BACKEND=claude ANTHROPIC_API_KEY=sk-ant-... \
  python -m src.serving.agent_api.eval.deepeval_eval --prompt-version v2
```

Kết quả log vào MLflow experiment `agent_eval_deepeval`.

### RAGAS vs DeepEval — khi nào dùng cái nào?

| | RAGAS | DeepEval |
|---|---|---|
| Phạm vi | RAG retrieval only | Full agent (RAG + LLM response) |
| LLM judge | Claude Haiku (tốt cho tiếng Việt) | Ollama hoặc Claude |
| Dùng khi | Tune RAG / reranker | Tune prompt, so sánh version |
| CI gate | Kiểm tra retrieval recall | Kiểm tra answer quality |

---

## Thiết kế kỹ thuật

**Tại sao hybrid search (dense + sparse) thay vì chỉ dense?**
Dense search tốt cho semantic similarity nhưng yếu với exact name matching: query `"iPhone 15 Pro Max"` → embedding model có thể rank `"Samsung Galaxy S25"` cao hơn vì học theo behavior pattern. Sparse TF-IDF char ngram (2,4) bắt subword match chính xác: `"iphone 15"` → tìm đúng model iPhone 15. RRF fusion lấy best of both mà không cần tune thêm weight.

**Tại sao RRF dùng rank thay vì weighted sum scores?**
Score của dense (cosine, range 0–1) và sparse (TF-IDF cosine, range khác) có distribution rất khác — normalize không đơn giản. RRF dùng rank → independent của magnitude, robust hơn, mặc định k=60 đã hoạt động tốt cho hầu hết cases mà không cần tune.

**Tại sao SemanticCache threshold = 0.95?**
Threshold thấp hơn (0.85) → cache hit nhiều hơn nhưng risk trả lời sai context (câu hỏi về điện thoại 5 triệu nhưng cache trả lời của câu về laptop 5 triệu). 0.95 đảm bảo chỉ cache hit khi query gần như identical về semantic, answer vẫn relevant, trong khi vẫn cache các repeated queries phổ biến.

**Tại sao SemanticCache reuse embedding model từ RAGPipeline?**
Embedding model (~600MB) tốn RAM. Load 2 instance riêng → waste 600MB. SemanticCache nhận `embedding_model=rag_pipeline.model` → dùng chung cùng instance đã load sẵn trong memory.

**Tại sao products không qua chunker.py?**
Product records ~50–80 tokens, vừa trong single BERT context window. Chunk một product record → phân mảnh thông tin không cần thiết, mất ngữ cảnh. `chunker.py` dành riêng cho long-form docs (FAQ 2000+ words, policy PDF) — cần split thành chunks ~300 tokens để embed meaningful.

---

## Troubleshooting

**Startup rất chậm (2–5 phút)**
Embedding model `bkai-foundation-models/vietnamese-bi-encoder` (~600MB) download lần đầu từ HuggingFace. Sau khi cache tại `SENTENCE_TRANSFORMERS_HOME=/app/model_cache/sentence_transformers`, startup < 30s.

**RAG trả về kết quả không liên quan với query tiếng Việt**
Kiểm tra bằng cách search thử trực tiếp:
```python
rag = RAGPipeline()
results = rag.search("điện thoại chụp ảnh đẹp dưới 10 triệu", top_k=3)
print([r["product_name"] for r in results])
```
Nếu kết quả lạ: bật `RERANKER_BACKEND=neural` để dùng CrossEncoder reranker cho chính xác hơn (chậm hơn ~3×).

**LLM không trả lời, request timeout sau 60s**
Ollama model chưa được pull về. Lần đầu cần pull ~4GB:
```bash
docker exec -it ollama ollama pull qwen2.5:7b
# Kiểm tra model đã load
docker exec -it ollama ollama list
```

**`Guardrails: PII detected` với câu hỏi bình thường**
False positive — regex PII bắt nhầm số giá tiền. Ví dụ `"0.35 triệu"` trùng pattern phone number. Điều chỉnh regex trong `guardrails.py`.

**Session history không giữ giữa các requests**
`session_id` không được truyền trong request. Mỗi request không có `session_id` → tạo session UUID mới → không có history. Đảm bảo client gửi đúng `session_id` trong mọi request của cùng conversation.

**`SemanticCache` hit với câu hỏi khác nhau rõ ràng**
Threshold đang thấp hơn 0.95. Kiểm tra:
```bash
echo $SEMANTIC_CACHE_THRESHOLD   # nếu set thì phải >= 0.90
```
