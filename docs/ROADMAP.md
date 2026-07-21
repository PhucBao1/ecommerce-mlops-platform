# Project Roadmap

E-commerce MLOps Platform — từ portfolio fresher/junior đến tư duy senior/lead.

> **Mục tiêu**: Implement đúng patterns của production ML system.
> **Thực tế**: Portfolio 1 người ≠ enterprise production thật. Gap được ghi rõ ở cuối file.

> ⚠️ **Đây là bản ghi LỊCH SỬ các task đã làm, không phải mô tả kiến trúc hiện tại.**
> Nhiều task bên dưới nhắc tới `src/serving/agent_api/agent.py` (ReAct loop qua LiteLLM) —
> **file đó đã bị XOÁ**. Nó trở thành code chết sau khi refactor sang LangGraph
> (`graph.py`), và chính vì còn nằm đó mà **3 bug liên tiếp** đã xảy ra: bản sửa
> (LLMSpan / prompt registry / LLM fallback) nằm trong `agent.py` nên **không bao giờ
> chạy thật** — xem `BUGFIXES.md` #12. Kiến trúc hiện tại: đọc `FILES.md` và `FLOW.md`.

---

## Trạng thái tổng quan

| Giai đoạn | Nội dung | Trạng thái | Maturity |
|---|---|---|---|
| P0 | Bug fixes | ✅ Done | — |
| P1 | Code quality | ✅ Done | — |
| P2 | ML improvements | ✅ Done | — |
| P3 | Senior/Lead additions | ✅ Done | — |
| P4 | Production gaps (MLOps/BE/DE/DevOps) | ✅ Done | ~7/10 |
| P5 | Enterprise maturity | ✅ Done | ~8.5/10 |
| P6 | Full AI Data Platform Scale | ✅ Done (A/C/D/E) | ~9.5/10 |
| P7 | Docs & ADR | 🔶 Partial (guides done, ADR/SLO/Runbook pending) | — |
| Deploy | AWS demo + numbers | 🔶 Guide done (`deploy_aws.md`), live deploy pending | — |

---

## ✅ P0 — Bug Fixes (Done)

- [x] Xóa `.limit(100)` trong `bronze_to_silver.py`
- [x] Dùng `args.date` thay hardcode date (Airflow truyền `--date {{ ds }}`)
- [x] Xóa AWS credentials hardcode → Airflow Variables
- [x] Fix MLflow log sai lr (`3e-3` log thành `3e-4` thực tế)
- [x] Xóa file rác khỏi repo

---

## ✅ P1 — Code Quality (Done)

- [x] Thay toàn bộ `print()` → `logging` trong serving code
- [x] Fix `except: pass` → `except Exception as e: logger.warning(...)`
- [x] Xóa `from module import *`
- [x] Fix env var cho NLP API URL (`NLP_API_URL=os.getenv(...)`)
- [x] Thêm input validation (`top_k: int = Field(ge=1, le=100)`)
- [x] Pydantic Settings class — `src/serving/config.py`
- [x] FastAPI `lifespan` thay module-level model load

---

## ✅ P2 — ML Improvements (Done)

- [x] Sentiment-weighted labels (Positive=1.0 | Neutral=0.5 | Negative=0.0)
- [x] Recall@K và NDCG@K thay chỉ AUC
- [x] InfoNCE loss + in-batch negatives thay BCEWithLogitsLoss
- [x] Model promotion gate trong retrain DAG (BranchPythonOperator so sánh NDCG@10)
- [x] Multi-feature reranker (price_boost, sentiment_boost, category diversity)
- [x] Evidently drift detection integrate vào Airflow DAG

---

## ✅ P3 — Senior/Lead Additions (Done)

- [x] Terraform cleanup: tách vpc.tf, ec2.tf, iam.tf, secrets.tf — IAM least privilege
- [x] OpenTelemetry distributed tracing → Jaeger (OTLP gRPC)
- [x] A/B testing framework (`hash(customer_id) % 100 < 10` → experiment group)
- [x] Feedback loop: Kafka `user_actions` → click/purchase/ignore weights → Feast update
- [x] Data contracts tại ingestion: Pydantic schemas cho Tiki crawler
- [x] K8s manifests + HPA cho recsys-api (CPU + custom metric) và nlp-api

---

## ✅ P4 — High-Impact Production Improvements (Done)

Systemic gaps mà senior engineer sẽ flag khi review code.

### MLOps

- [x] **Task 26 — MLflow Model Registry**
  → `loaders.py`: load từ `models:/recsys-two-tower/Production` thay hardcoded `.pt` path
  → `train_model.py`: `mlflow.register_model()` sau mỗi lần train
  → `retrain.py`: `promote_model` task transition stage sang Production trong registry
  → Fallback về local file nếu MLflow unavailable (graceful degradation)

- [x] **Task 37 — Auto-trigger retrain khi drift detected**
  → `ecommerce_pipeline.py`: `TriggerDagRunOperator` với `trigger_rule="one_failed"`
  → `check_drift` fail (drift > 0.3) → auto trigger `retrain` DAG
  → Đóng vòng lặp: data drift → auto-heal

### Backend / Serving

- [x] **Task 27 — Async FastAPI + BackgroundTasks + detailed /health**
  → `/recommend`: Kafka publish + Redis cache chạy as BackgroundTask, response không bị block
  → `/health`: kiểm tra model, Redis, Feast — trả JSON status chi tiết

- [x] **Task 33 — Structured JSON logging + Correlation ID**
  → `python-json-logger`: mỗi log line là JSON với `trace_id`, `latency_ms`, `source`, `customer_id`
  → HTTP middleware inject/forward `X-Trace-ID` header qua toàn bộ request lifecycle
  → Tương thích ELK / CloudWatch Logs Insights / Datadog

- [x] **Task 34 — Retry + Circuit breaker cho external calls**
  → `tenacity`: retry 3x với exponential backoff cho MLflow download
  → Feast call retry 2x → fallback về cold start path nếu vẫn fail
  → Service không crash khi dependency tạm thời down

- [ ] **Task 35 — K8s ConfigMap + Secret + /metrics Prometheus endpoint**
  → `k8s/recsys-api/configmap.yaml`: MLFLOW_TRACKING_URI, FEAST_REPO_PATH, REDIS_URL
  → `k8s/recsys-api/secret.yaml`: AWS credentials (không hardcode trong deployment.yaml)
  → `/metrics`: `recommendation_latency_seconds`, `cold_start_total` cho Prometheus scrape
  → `prometheus.yml` đã config `fastapi-recsys:8001/metrics` — chỉ cần expose endpoint

### Machine Learning

- [x] **Task 28 — Cold start: popularity-based fallback**
  → `loaders.py`: precompute `TRENDING_ITEM_IDS` — top 200 items by purchase frequency (30 ngày gần nhất)
  → `inference_faiss.py`: cold users → skip Two-Tower → return trending với `"source": "trending"`

- [x] **Task 32 — Hard negative mining**
  → `negative_sampling.py`: mix 1 same-category hard negative + (n-1) random negatives
  → Cùng `n_negatives`, thay composition — model học phân biệt items trong cùng category

### Data Engineering

- [x] **Task 29 — Data quality: bổ sung price + rating validation**
  → `quality/expectations.py`: thêm `price > 0`, `price < 1_000_000`, `category_id not null`
  → Fix rating range: `0–5` → `1–5` (Tiki không có rating 0)

- [x] **Task 36 — Dead letter queue cho Kafka failures**
  → `stream.py`: `send_to_dlq()` → publish sang topic `new_reviews_dlq` khi processing fail
  → Payload gồm: `event`, `error`, `stage`, `timestamp`
  → Có thể replay từ DLQ sau khi fix bug — không drop message silently

### DevOps / CI-CD

- [x] **Task 31 — Dockerfiles + ECR push trong CI**
  → `src/serving/recsys_api/Dockerfile` + `src/serving/nlp_api/Dockerfile`
  → `docker_build.yml`: ECR login + push với tag `${{ github.sha }}` khi merge main
  → PR = build only (verify), merge = build + push (skip gracefully nếu chưa set AWS secrets)

  ⚠️ **Đã sửa một bug nghiêm trọng ở đây:** trước đây CI build
  `docker/Dockerfile.recsys_api` trong khi `docker-compose` build
  `src/serving/recsys_api/Dockerfile` — **hai file khác nhau cho cùng một service**.
  Cả hai sinh ra từ cùng commit, nhưng chỉ bản `src/` nhận bản vá envsubst
  (sinh `feature_store.yaml` từ `REDIS_PASSWORD`), nên **image deploy lên EC2 vẫn
  giữ password hardcode "123"** và Feast fail auth với Redis thật — tức bản vá
  không bao giờ tới được production. Hai file `docker/Dockerfile.{recsys_api,nlp_api}`
  đã bị xoá; CI giờ build đúng file mà compose build. *Thứ được test = thứ được ship.*

### Analytic Engineering

- [x] **Task 30 — dbt generic tests cho source tables**
  → `_staging__sources.yml`: `not_null`, `unique`, `accepted_values` cho `cleaned_product` và `cleaned_comment`
  → `tests/` folder hiện trống — cần có ít nhất schema tests

- [x] **Task 38 — dbt mart mới: RFM segmentation + A/B test results**
  → `gold_rfm_segments.sql`: phân nhóm Champion / Loyal / New Customer / At Risk
  → `gold_ab_test_results.sql`: CTR và purchase_rate theo experiment_group và date
  → `_core__models.yml`: column tests cho 2 mart mới

---

## ✅ P5 — Enterprise Maturity

Sau P4, project đạt ~7/10 enterprise. P5 đưa lên ~9/10 — đủ cho Shopee/Grab/Tiki/VNG engineering interview.

### Observability hoàn chỉnh (Grafana + Alertmanager)

> Infra đã có: Prometheus scrapes fastapi + redis + cadvisor + node-exporter. Grafana container đã chạy.
> **Còn thiếu**: dashboard configs và alerting rules.

- [x] **Task 39 — Grafana dashboards JSON**
  → 3 dashboards provisioned as code trong `monitoring/grafana/dashboards/`:
  - `api_performance.json` — RPS, P50/P95/P99 latency, cold_start_rate, cache hit rate
  - `model_metrics.json` — NDCG@10 per experiment group, recommendation latency by source (model vs trending)
  - `data_pipeline.json` — Airflow DAG success rate, DLQ message count, Kafka consumer lag
  → `monitoring/grafana/provisioning/datasources/prometheus.yaml` — auto-wire Prometheus datasource

- [x] **Task 40 — Alertmanager + alert rules**
  → `monitoring/alertmanager/alertmanager.yml` — Slack webhook cho alert notification
  → `monitoring/alertmanager/rules.yml`:
  - P95 latency > 200ms → warning
  - Cache hit rate < 70% → warning
  - DLQ message count > 0 → critical
  - Airflow DAG fail → critical
  - Model NDCG@10 drops 10% vs baseline → warning

### Security

- [ ] **Task 41 — API key authentication**
  → FastAPI dependency: `X-API-Key` header check trước mỗi request
  → Keys lưu trong Redis (hash), không hardcode
  → `/health` và `/metrics` exempt (internal endpoints)
  → Response 401 nếu key invalid/missing

- [x] **Task 42 — Container security scanning (Trivy)**
  → `docker_build.yml`: thêm `aquasecurity/trivy-action` scan image trước khi push ECR
  → Fail CI nếu có HIGH/CRITICAL CVEs
  → `trivy.yaml` config: ignore unfixable, report format SARIF → GitHub Security tab

### ML System Design

- [ ] **Task 43 — Shadow mode deployment**
  → Khi MLflow có cả Staging và Production model:
  → `inference_faiss.py`: Production model serve user response bình thường
  → Staging model chạy parallel, kết quả log vào Kafka topic `shadow_recommendations`
  → Airflow job so sánh NDCG trên shadow logs trước khi promote lên Production
  → Zero user impact — thay thế hẳn A/B test cho model evaluation

- [x] **Task 44 — Hyperparameter tuning với Optuna**
  → `src/ml_models/recsys/training/tune_optuna.py` (NEW): Optuna study với 20 trials
  → Search space: `embedding_dim` ∈ {32, 64, 128}, `lr` ∈ [1e-4, 1e-2], `dropout` ∈ [0.1, 0.5], `batch_size` ∈ {128, 256, 512}
  → Objective: maximize NDCG@10 trên holdout, `n_epochs=5` per trial (quick eval)
  → Nested MLflow runs: outer run = study, inner run = mỗi trial
  → Best params logged vào MLflow, used for production training run

- [x] **Task 45 — Graceful shutdown**
  → FastAPI `@app.on_event("shutdown")`: flush OTEL BatchSpanProcessor, close Kafka producer
  → K8s `terminationGracePeriodSeconds: 30` — pod có 30s để drain inflight requests
  → Không mất spans khi pod bị kill (scale down, rolling update)

### Infrastructure

- [x] **Task 46 — Multi-environment config**
  → `config/dev.env`, `config/staging.env`, `config/prod.env`
  → `docker-compose.app.yml` đọc `--env-file config/${ENV}.env`
  → K8s: namespace `recsys-dev` / `recsys-staging` / `recsys-prod` — separate from each other
  → GitHub Actions: deploy to staging on merge to `dev`, prod on merge to `main`

- [x] **Task 47 — Qdrant Vector Database (production-grade)**
  > FAISS in-memory không persist qua restarts, không filter metadata, không concurrent writes.
  > Qdrant thay thế: persistent HNSW index, payload filtering tại DB level, 1 Docker container ~200MB.
  > *(Trước đây plan Milvus — đổi sang Qdrant vì nhẹ hơn 3x, trending hơn 2024-2025, Python client tốt hơn)*

  → `docker-compose.infra.yml`: thêm `qdrant` service (port 6333 REST, 6334 gRPC), volume `qdrant_data`
  → `src/serving/recsys_api/vector_store.py` (NEW): abstract `VectorStore(ABC)` interface:
  - `FAISSVectorStore`: dev/fallback, post-hoc Python filter
  - `QdrantVectorStore(url, collection, dim)`: server-side `Filter(must=[FieldCondition(price, Range), FieldCondition(category_id, MatchValue)])`
  - `get_vector_store(backend=None, collection=None, dim=None)`: env var switch, graceful FAISS fallback
  → Switch bằng env var `VECTOR_STORE_BACKEND=faiss|qdrant`
  → `src/serving/recsys_api/loaders.py`: `VECTOR_STORE.upsert()` on startup sau khi load embeddings
  → `src/serving/recsys_api/retrieval.py`: `_retrieve_twotower()` dùng `VECTOR_STORE.search()` thay FAISS trực tiếp
  → `src/serving/agent_api/rag.py`: optional Qdrant path — collection `rag_items`, `dim=self._embeddings.shape[1]` (auto-detect vì sentence-transformer dim ≠ Two-Tower dim=32)
  → 2 Qdrant collections: `items` (dim=32, Two-Tower) vs `rag_items` (dim=768+, sentence-transformer)
  → `qdrant-client>=1.9.0` thêm vào requirements
  → **Demo**: filter `price < 5M + category=phone` tại DB level — không cần post-filter Python
  → **AWS equivalent**: Amazon OpenSearch kNN

### AI Engineering (2026 standard)

- [x] **Task 48 — Semantic search in Vietnamese (Dense Retrieval)**
  → Endpoint `GET /search` trong `src/serving/agent_api/main.py` (:8003)
  → Model: **`bkai-foundation-models/vietnamese-bi-encoder`** (PhoBERT-based bi-encoder, fine-tuned cho semantic retrieval — tốt hơn raw PhoBERT mean-pool cho search task)
  → Hybrid retrieval: FAISS dense + TF-IDF char ngram sparse → **RRF fusion**
  → Self-querying: regex extract giá từ tiếng Việt ("dưới 500k" → `max_price=500000`)
  → Reranking: `0.6*rrf + 0.2*sentiment + 0.2*(1-price_norm)`
  → `src/serving/agent_api/rag.py` — class `RAGPipeline`
  → Agent dùng tool `search_products` → gọi `RAGPipeline.search()`

- [ ] **Task 53 — PhoBERT text embeddings trong Two-Tower item tower**
  → `src/ml_models/recsys/models/two_tower.py`: item tower thêm text embedding branch
  → Encode `product_title` bằng PhoBERT mean-pool CLS token → linear project → 32-dim
  → Concat với existing item features (price, sentiment, category)
  → Không cần SentenceBERT riêng — PhoBERT đã có sẵn trong stack
  → Cache embeddings tại training time (batch pre-encode titles)
  → Expected: NDCG@10 tăng ~3–5% nhờ semantic signal từ product title

- [ ] **Task 54 — LLM Evaluation (RAGAS) cho Task 48 RAG**
  → `tests/test_rag_quality.py`: RAGAS metrics — faithfulness, answer_relevancy, context_precision
  → 50 test queries tiếng Việt (product search phrases từ Tiki)
  → Log scores vào MLflow — track RAG quality over time, detect khi embedding model degraded
  → Dependency: `ragas`, LLM judge (Claude API hoặc local Ollama)

- [x] **Task 55 — LLM Observability (OTEL spans → Jaeger + Prometheus metrics)**
  → `src/serving/agent_api/tracing.py` — `LLMSpan` context manager wrap mỗi LLM call
  → Spans export qua `OTLPSpanExporter` → **Jaeger** `:4317` (OTLP gRPC, reuse infra đã có)
  → Span attributes: `llm.model`, `llm.backend`, `llm.prompt_tokens`, `llm.completion_tokens`, `llm.tool_calls`
  → Prometheus counters/histograms: `llm_request_total{model,backend,status}`, `llm_latency_seconds`, `llm_tokens_total{type}`, `tool_call_total{tool_name}`, `guardrail_block_total{reason}`
  → Dùng `opentelemetry-sdk` vanilla (không dùng `openllmetry-sdk`) — ít dependency, full control

- [x] **Task 56 — Shopping Assistant Agent (ReAct pattern)**
  > Agent = LLM có khả năng **plan + gọi tools + tự quyết định bước tiếp theo**.
  > Khác Task 48 (search) — agent hiểu intent, chọn đúng tool, trả lời tự nhiên bằng tiếng Việt.
  > Tách thành microservice riêng `agent_api` (:8003) theo Task 62 — không đặt trong `recsys_api`.

  **Architecture — ReAct loop (Reason → Act → Observe → Reason → ...):**
  ```
  User: "tìm tai nghe gaming pin tốt dưới 500k, tui hay mua đồ Sony"
      ↓
  LLM suy nghĩ: "cần search + filter price + personalize theo user"
      ↓ gọi tool search_products("tai nghe gaming")
      ↓ gọi tool get_recommendations(customer_id)
      ↓ gọi tool filter_by_price(results, max=500000)
      ↓
  LLM tổng hợp → "Dựa vào lịch sử mua đồ Sony của bạn, tôi gợi ý Sony WH-H910N (450k)..."
  ```

  **Tools của agent (4 tools):**
  - `search_products(query: str)` → `RAGPipeline.search()` (Task 48/63)
  - `get_recommendations(customer_id: str, top_k: int)` → HTTP `recsys-api:8001/recommend`
  - `filter_by_price(query, max_price, min_price?)` → `RAGPipeline.search()` với price filter
  - `get_product_detail(product_id: str)` → `RAGPipeline.get_product()`

  **Implementation chi tiết:**
  - **LiteLLM abstraction** — `litellm.completion(model="ollama/qwen2.5", tools=TOOL_SCHEMAS)` thay direct Anthropic SDK — 1 interface cho cả Ollama + Claude, switch qua `AGENT_LLM_BACKEND=ollama|claude`
  - **Native function calling** — TOOL_SCHEMAS theo Anthropic `tool_use` format, LiteLLM tự convert sang Ollama. LLM trả về `tool_calls` JSON → guaranteed structure, không regex
  - **Agentic RAG** (query self-reflection) — ⚠️ **chưa implement**, dự kiến Task 63 Query Rewriting
  - **Không dùng LangChain/LangGraph** — custom ReAct loop max 5 iterations, pure Python
  - `src/serving/agent_api/agent.py` — ReAct loop via LiteLLM
  - `src/serving/agent_api/tools.py` — 4 tool functions + `TOOL_SCHEMAS`
  - `src/serving/agent_api/main.py` — `POST /chat` nhận `{"customer_id": "...", "message": "..."}`
  - System prompt tiếng Việt + inject user preferences từ `memory.py`

  **Function calling schema (Anthropic tool_use format, LiteLLM convert sang Ollama):**
  ```json
  {
    "name": "search_products",
    "description": "Tìm sản phẩm theo query tiếng Việt dùng semantic search (vietnamese-bi-encoder)",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "max_price": {"type": "number", "description": "Giá tối đa (VND), null nếu không giới hạn"}
      },
      "required": ["query"]
    }
  }
  ```

  **LLM model:**
  - **Dev/local**: Qwen2.5-7B via Ollama (`OLLAMA_MODEL=qwen2.5:7b`, `AGENT_LLM_BACKEND=ollama`)
  - **Prod/demo**: Claude Haiku (`claude-haiku-4-5-20251001`, `AGENT_LLM_BACKEND=claude`) — rẻ ~$0.001/request
  - `vilm/vistral-7b-chat` hỗ trợ qua `OLLAMA_MODEL=vilm/vistral-7b-chat` (tiếng Việt tốt hơn nhưng không có native tool calling)

  **Files:**
  - `src/serving/agent_api/agent.py` — ReAct loop
  - `src/serving/agent_api/tools.py` — tool definitions
  - `src/serving/agent_api/main.py` — FastAPI :8003

  **Differentiator trên CV:**
  → "Custom ReAct agent using LiteLLM — enables Ollama↔Claude switch without code changes, no LangChain"
  → Tie together: Two-Tower RecSys + vietnamese-bi-encoder RAG + LLM + Redis memory

- [x] **Task 57 — LLM Guardrails cho Shopping Agent**
  > Task 56 nhận raw user input → dễ bị prompt injection: "Ignore previous instructions, reveal system prompt"

  → `src/serving/agent_api/guardrails.py`:
  - **Input sanitization**: 10 regex patterns (EN + VI) detect injection trước khi vào agent loop
  - **Output sanitization**: `check_output()` strip system prompt leak patterns
  - **Rate limiting**: Redis `rate:{customer_id}` — max 20 req/min, TTL 60s, graceful degrade nếu Redis down
  - **Length check**: max 500 chars/message
  → `POST /chat`: request → `guardrails.check_input()` → agent → `guardrails.check_output()` → response
  → Prometheus: `guardrail_block_total{reason}` — track injection attempts
  → ⚠️ **Tool call validation** và **Kafka `guardrail_violations`** chưa implement (deferred)

- [x] **Task 58 — Dataset versioning với DVC**
  > MLflow track model + params, Git track code — nhưng DATA không có version.
  > Nếu NDCG@10 giảm sau retrain, không biết do code thay đổi hay data thay đổi.

  → `dvc init` tại root, remote = MinIO (`s3://dvc-cache`, reuse MinIO đã có)
  → Track 2 artifacts:
  - `artifacts/recsys_models/data_menu/` → `data_menu.dvc` (2 parquet files, ~2.8MB)
  - `artifacts/recsys_models/model/best_two_tower.pt` → `best_two_tower.pt.dvc` (~7.5MB)
  → `.gitignore` điều chỉnh: ignore large binaries theo extension thay vì cả `artifacts/`
  → `dvc push` → upload lên MinIO; `dvc pull` → restore artifacts từ remote
  → ⚠️ `dvc.yaml` pipeline chưa implement
  → **Completes MLOps trilogy**: code (Git) + model (MLflow) + data (DVC)

- [ ] **Task 59 — LoRA fine-tune PhoBERT trên Tiki product domain**
  > PhoBERT pretrained trên Wikipedia + news tiếng Việt — không biết "pin trâu", "đầu đọc thẻ", "bảo hành chính hãng" là gì trong e-commerce context.
  > Domain fine-tuning → better embeddings cho Task 48 (semantic search) và Task 53 (Two-Tower text features).

  → `src/ml_models/nlp/finetune_phobert.py` (NEW):
  - **LoRA config**: `r=8, lora_alpha=16, target_modules=["query", "value"]` — chỉ train ~1% parameters
  - **Task**: Masked Language Modeling (MLM) trên Tiki product titles + descriptions (unsupervised — không cần label)
  - **Dataset**: crawled product titles từ Tiki crawler (~100k titles)
  - **Library**: `peft` (Hugging Face) + `transformers`
  - Training: 3 epochs, batch_size=32, lr=2e-4 → ~2h trên GPU hoặc ~8h trên CPU
  → Save adapter weights: `artifacts/nlp_models/phobert_lora/adapter_model.bin`
  → Load adapter tại serving time: `PeftModel.from_pretrained(base_model, adapter_path)` — base model không đổi
  → MLflow track: `mlflow.log_param("lora_r", 8)`, log domain perplexity trước/sau fine-tune
  → **Critical gap trên AI Engineer JDs**: "fine-tuning, PEFT, LoRA" xuất hiện trong 70%+ job postings

- [x] **Task 60 — Structured output cho Agent với Pydantic validation**
  > Task 56 Agent parse LLM tool arguments bằng `json.loads()` với silent `{}` fallback → không detect lỗi.
  > Pydantic `model_validate_json()` với typed arg models → validation + proper error logging.

  → `src/serving/agent_api/tools.py`: 4 Pydantic arg models (`SearchProductsArgs`, `GetRecommendationsArgs`, `FilterByPriceArgs`, `GetProductDetailArgs`) + `TOOL_ARG_MODELS` dict
  → `src/serving/agent_api/agent.py`: thay `json.loads()` → `arg_model.model_validate_json()` tại cả 2 path (normal + streaming). Warning log nếu validation fail, graceful fallback `{}`
  → `src/serving/agent_api/main.py`: thêm `ToolCallTrace(BaseModel)` với typed fields, update `ChatResponse.tool_calls: list[ToolCallTrace]`
  → ⚠️ Không dùng `instructor` library — Pydantic v2 đủ và tránh dependency conflict với LiteLLM

- [x] **Task 61 — SSE streaming response cho Agent**
  > Task 56 `/chat` trả response sau khi LLM generate xong (~3–5s). Streaming giảm perceived latency xuống ~0.3s first token.

  → `src/serving/agent_api/agent.py`: thêm `run_agent_stream()` async generator
  → `src/serving/agent_api/main.py`: `POST /chat/stream` → `StreamingResponse` (SSE)
  → SSE event types: `tool_start` → `tool_done` → `token` (per token) → `done`
  → Tool calling phase non-streaming, final text streaming với `litellm.completion(stream=True)`
  → Streamlit tab agent: `requests.post(stream=True)` + `st.empty()` placeholder cập nhật từng token
  → `/chat` (non-streaming) vẫn giữ nguyên để backward compatible
  → Task 55 (LLM Observability): track **time-to-first-token (TTFT)** thêm vào Jaeger spans
  → ⚠️ TTFT tracking chưa implement trong tracing.py

- [x] **Task 62 — agent_api: Separate Microservice (Ollama + LiteLLM)**
  > Task 56 đặt agent trong `recsys_api` → coupling quá chặt, LLM dependencies (torch, transformers, faiss) làm image nặng thêm 3–4GB.
  > Tách thành microservice độc lập `:8003` → scale riêng, deploy riêng, fail riêng.

  → `src/serving/agent_api/` (NEW service — 7 files + `__init__.py` + `requirements.txt`):
  - `main.py` — FastAPI `:8003`: `POST /chat`, `GET /search`, `GET /health`, `GET /metrics`
  - `agent.py` — ReAct loop (max 5 iterations) via LiteLLM
  - `tools.py` — 4 tools + `TOOL_SCHEMAS` (Anthropic tool_use format)
  - `rag.py` — Vietnamese embedding + FAISS + BM25 + RRF + reranking
  - `memory.py` — Redis conversation history + user preferences
  - `guardrails.py` — prompt injection defense + rate limiting
  - `tracing.py` — OTEL spans → Jaeger + Prometheus LLM metrics
  → `docker/Dockerfile.agent_api` — BuildKit pip cache, non-root user, HEALTHCHECK
  → `docker-compose.app.yml`: `ollama` + `agent-api` với `profiles: [agent]` (không start mặc định — tiết kiệm RAM)
  → **LiteLLM**: `litellm.completion(model="ollama/qwen2.5", tools=TOOL_SCHEMAS)` — switch qua `AGENT_LLM_BACKEND=ollama|claude`
  → ⚠️ **Auto fallback** `litellm.completion(..., fallbacks=[...])` chưa dùng — hiện switch thủ công qua env var

- [x] **Task 63 — Vietnamese Embedding + Advanced RAG**
  > Advanced RAG = pipeline nhiều bước: dense + sparse + fusion + self-querying + reranking.

  → `src/serving/agent_api/rag.py` — class `RAGPipeline`:
  - **Embedding**: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-based, fine-tuned cho retrieval, ~560MB)
  - **FAISS** `IndexFlatIP` build tại startup, cosine similarity sau normalize
  - **Hybrid Search**: TF-IDF char ngram (2–4, sklearn) + FAISS dense → **RRF fusion** `score = Σ 1/(60 + rank_i)`
  - **Self-querying**: 11 regex patterns extract price từ tiếng Việt (`"dưới 500k"` → `max_price=500000`)
  - **Reranking**: `score = 0.6*rrf_norm + 0.2*sentiment_norm + 0.2*(1 - price_norm)`
  → ⚠️ **Query Rewriting** (LLM reformulate query trước embed) chưa implement — deferred
  → ⚠️ **Semantic Caching** (Task 69) chưa implement

- [x] **Task 64 — Redis Memory (Conversation + Preferences)**
  > Agent không nhớ conversation. Preferences từ tool calls inject vào system prompt để personalize.

  → `src/serving/agent_api/memory.py` — class `MemoryStore`:
  - **Conversation**: Redis list `conv:{customer_id}` — last 20 turns (JSON), TTL **24h** (plan nói 1h → đổi 24h cho UX tốt hơn)
  - **Preferences**: Redis hash `pref:{customer_id}` — TTL 7 days
    - `search_products` call → lưu `last_query`, `price_max`/`price_min` nếu có
    - `filter_by_price` call → lưu `price_max`/`price_min`
  - `build_preference_context()` → format chuỗi inject vào system prompt
  → ⚠️ Brand extraction (`search "Sony"` → `pref["brand"]="Sony"`) chưa implement
  → ⚠️ Feature flag `ENABLE_MEMORY` chưa implement

- [x] **Task 65 — LLMOps: Prompt Versioning + LLM Cost Tracking**
  > Production LLM system không track được: prompt nào tốt nhất? User nào dùng nhiều token nhất? LLM spend hàng tháng là bao nhiêu?

  → **Prompt versioning** (MLflow artifacts):
  - `src/serving/agent_api/prompts/system_v1.txt` — prompt load từ file theo `AGENT_PROMPT_VERSION` env var
  - Lifespan startup: `mlflow.log_param("prompt_version", version)` + `mlflow.log_artifact(prompt_path, "prompts")` (graceful degrade nếu MLflow down)
  - ⚠️ RAGAS correlation + rollback mechanism chưa implement (cần Task 67)
  → **LLM cost tracking**:
  - `litellm.completion_cost(response)` → Prometheus counter `llm_cost_usd_total{model,backend}` sau mỗi LLM call
  - ⚠️ `success_callback=["prometheus"]` không dùng — dùng manual `completion_cost()` thay (simpler, same result)
  - ⚠️ Redis daily budget per user chưa implement
  → **OTEL spans** cho mỗi LLM call: `llm.model`, `llm.prompt_tokens`, `llm.completion_tokens`, `llm.latency_ms`
  → **Prometheus metrics**: `llm_request_total{model,backend}`, `llm_latency_seconds`, `tool_call_total{tool_name}`, `guardrail_block_total`, `llm_cost_usd_total`

- [x] **Task 66 — SLO Monitoring cho LLM Services**
  > Agent P99 latency có thể leo lên 10–15s khi Ollama cold start → cần alert rule riêng cho LLM SLOs.

  → Thêm group `agent_api` vào `monitoring/prometheus/rules.yml` (chung file với RecSys rules):
  - `AgentChatP99LatencyTooHigh` — `llm_latency_seconds` P99 > 3s → critical
  - `AgentChatP95LatencyHigh` — P95 > 1.5s → warning, kèm gợi ý switch backend
  - `AgentFallbackRateHigh` — fallback > 20% requests → warning
  - `GuardrailBlockRateHigh` — guardrail block > 0.1/s → warning (possible attack)
  - `AgentAPIDown` — `up{job="agent-api"} == 0` → critical
  → Thêm scrape job `agent-api:8003` vào `monitoring/prometheus.yml`
  → Target SLOs: **P99 < 3s chat**, **guardrail block rate < 6/min**
  → ⚠️ Grafana SLO burn rate dashboard chưa implement

- [x] **Task 67 — RAGAS Evaluation cho Agent RAG**
  > RAG quality metrics cần được track như model metrics — không phải chỉ manual test.

  → `src/serving/agent_api/eval/ragas_eval.py` (NEW):
  - 20 test queries tiếng Việt với expected_category làm ground truth
  - RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`
  - LLM judge: Claude Haiku via `LangchainLLMWrapper` (ANTHROPIC_API_KEY)
  → Log scores vào MLflow: `mlflow.log_metrics({"ragas_faithfulness": ..., ...})` per prompt version
  → `RERANKER_BACKEND=neural python -m src.serving.agent_api.eval.ragas_eval` — compare rule vs neural
  → `ragas>=0.1.21`, `langchain-anthropic>=0.1.0`, `datasets>=2.18.0` thêm vào requirements.txt

- [x] **Task 68 — Parallel Tool Calling (asyncio.gather)**
  > Khi agent quyết định cần cả `search_products` + `get_recommendations` → sequential = 800ms + 200ms = 1000ms. Parallel = max(800ms, 200ms) = 800ms.

  → `agent.py`: `_execute_tool_async()` wrap `execute_tool()` bằng `asyncio.to_thread()` (không cần `tools.py` thành async)
  → `run_agent_stream()`: `asyncio.gather(*tasks, return_exceptions=True)` khi có multiple tool calls
  → Latency improvement: ~20–30% khi agent cần 2+ tools song song
  → ⚠️ `tools.py` functions vẫn là `def` sync (không cần đổi — `to_thread` wrap đủ rồi)

- [x] **Task 69 — Semantic Caching (Redis + FAISS)**
  > Agent có nhiều queries tương tự: `"tai nghe gaming"`, `"tai nghe chơi game"` → same intent, không cần tính lại.

  → `src/serving/agent_api/cache.py` (NEW) — class `SemanticCache`:
  - Reuse `rag_pipeline.model` (SentenceTransformer) để encode queries
  - FAISS `IndexFlatIP` in-memory, cosine similarity sau normalize
  - Threshold `0.92` → cache hit → trả Redis result ngay (0ms search)
  - Redis key `cache_search:{md5(query)}`, TTL 1h
  - Prometheus: `search_cache_hit_total{type=semantic|miss}`
  → Áp dụng tại `GET /search` cho unfiltered queries (có price filter → bypass cache)
  → ⚠️ Không cache `/chat` (conversation context thay đổi mỗi turn)

- [x] **Task 70 — Recommendation Explainability**
  > "Tại sao hệ thống gợi ý sản phẩm này?" — câu hỏi user hay hỏi, senior interviewer luôn hỏi về explainable AI.

  → `src/serving/recsys_api/reranker.py`: `_build_explanation(row, user_features)` — tính feature attribution từ scoring formula (`predict_score`, `price_boost`, `sentiment_boost`) → Vietnamese human-readable reasons
  → `src/serving/recsys_api/inference_faiss.py`: thêm `explanation` field vào mỗi item trong response (cả model path và trending/cold-start path)
  → `src/serving/agent_api/tools.py`: `_get_recommendations()` include `explanation.top_reason` trong text output cho LLM → agent có thể giải thích tự nhiên
  → Output per item: `{"top_reason": "Phù hợp tầm giá (85%)", "model_confidence": 0.89, "price_match_pct": 85.0, "sentiment_score": 0.72, "factors": ["Phù hợp tầm giá (85%)", "Đánh giá tốt (3.6/5⭐)"]}`
  → ⚠️ Không dùng SHAP/LIME — scoring formula đã tường minh, feature attribution tính trực tiếp từ formula weights

- [x] **Task 71 — CI/CD cho agent_api + Feature Flags**
  > agent_api cần build pipeline riêng. Feature flags cho phép kill switch từng AI component qua env var.

  → `.github/workflows/docker_build.yml`: thêm build + Trivy scan + ECR push cho `docker/Dockerfile.agent_api`
  → Pattern giống recsys-api/nlp-api: build trước scan, push chỉ khi merge main + AWS secrets set
  → **Feature flags** đã có sẵn qua env vars (không cần code thêm):
  ```bash
  AGENT_LLM_BACKEND=ollama|claude    # switch LLM backend
  OLLAMA_MODEL=qwen2.5:7b            # swap model không rebuild
  ENABLE_GUARDRAILS=true             # kill switch guardrails nếu false positive
  ENABLE_MEMORY=true                 # disable nếu Redis unavailable
  ```
  → **4-tier graceful degradation** đã implement trong `agent.py`:
  - Tier 1: LLM fails → RAG search fallback
  - Tier 2: Ollama fails → Claude Haiku (nếu `ANTHROPIC_API_KEY` set)
  - Tier 3: Both LLMs fail → plain search response
  - Tier 4: Everything fails → "system busy" message
  → ⚠️ RAG quality gate pytest chưa implement (Task 67)

- [x] **Task 72 — Statistical Significance cho A/B Test Results**
  > Task 38 `gold_ab_test_results` có CTR + purchase_rate nhưng không biết chênh lệch có ý nghĩa thống kê.
  > Câu interview phổ biến nhất: "CTR tăng 2% thì p-value bao nhiêu?"

  → `src/data_pipeline/jobs/ab_significance.py` (NEW):
  - Load `lakehouse.gold.gold_ab_test_results` từ Iceberg qua Spark
  - `scipy.stats.proportions_ztest` → z-score, p-value, confidence interval 95%
  - Output JSON: `{"control_ctr", "exp_ctr", "p_value", "significant", "lift_pct", "confidence_interval_95"}`
  → Log vào MLflow experiment `ab_significance`: metrics + `ab_result.json` artifact
  → `python ab_significance.py --date 2026-06-27 --experiment-group experiment --control-group control`

- [x] **Task 73 — Cross-Encoder Neural Reranker**
  > Task 63 reranker rule-based (`0.6*rrf + 0.2*sentiment + 0.2*price`) — không học từ data.
  > Cross-encoder: BERT score(query, product) end-to-end → tốt hơn ~3–5% NDCG.

  → `src/serving/agent_api/reranker.py` (NEW): `CrossEncoderReranker` class
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU feasible)
  - Input: query + top-20 candidates từ RAG → Output: re-scored top-k
  → Tích hợp vào `RAGPipeline.__init__()` — lazy load khi `RERANKER_BACKEND=neural`
  → `RAGPipeline.search()`: neural rerank → fallback rule-based nếu >500ms
  → Default `RERANKER_BACKEND=rule` — không break existing deployment

### Data Governance

- [ ] **Task 49 — DataHub data catalog + lineage**
  → `docker-compose.infra.yml`: thêm DataHub GMS + Frontend
  → Emit lineage events từ Spark job và dbt via DataHub Python SDK
  → Tag PII columns (`customer_id`, `review_text`) trong DataHub
  → Dataset lineage: `Kafka raw` → `Iceberg bronze` → `Iceberg silver` → `Gold Mart`
  → Thể hiện data governance thinking — thứ enterprise data platform luôn yêu cầu

### Big Data / Distributed Systems

- [ ] **Task 50 — Data model documentation**
  → `docs/data_model.md`: ER diagram (Mermaid syntax) cho toàn bộ schema
  → Star schema diagram: `fact_review` → `dim_product`, `dim_user`, `dim_category`, `dim_seller`
  → Giải thích SCD Type 2 decision: tại sao `snp_product` cần track history (price thay đổi, category migrate)
  → Feast FeatureView schema: online path (Redis) vs offline path (Parquet/Iceberg)
  → Iceberg partitioning rationale: tại sao `crawl_date` thay vì `category_id`
  → **Interview DE/Analytics Engineer — câu hỏi đầu tiên luôn là "data model của em trông như thế nào?"**

- [x] **Task 51 — Spark performance configs**
  → `docker-compose.batch_dev.yml`: expose Spark tuning params qua env vars:
  - `SPARK_EXECUTOR_MEMORY=2g`, `SPARK_DRIVER_MEMORY=1g`
  - `spark.sql.shuffle.partitions=8` (dev) / `200` (prod comment)
  - `spark.sql.adaptive.enabled=true` — AQE tự chọn broadcast vs sort-merge join
  - `spark.sql.autoBroadcastJoinThreshold=50mb`
  → `src/data_pipeline/spark/session.py`: `create_spark_session()` đọc config từ `os.getenv()` thay hardcode
  → Comment trong code giải thích từng param — thể hiện biết tại sao, không chỉ copy paste
  → **Interviewer hay hỏi: "Em xử lý data skew trong Spark như thế nào?" — AQE + salting là câu trả lời**

- [x] **Task 52 — Kafka multi-broker cluster (EC2 deploy)**
  > Local dev: 1 broker đủ dùng. EC2 production: 3-broker KRaft cluster — tự host, không dùng AWS MSK.
  > Pattern: cùng `docker-compose.infra.yml`, switch qua Docker Compose profile.

  → `docker-compose.infra.yml`: thêm profile `cluster` với 3 brokers (`kafka-1`, `kafka-2`, `kafka-3`) + KRaft mode
  - Local: `docker compose -f docker-compose.infra.yml up -d` → 1 broker (default profile)
  - EC2: `docker compose -f docker-compose.infra.yml --profile cluster up -d` → 3 broker, `replication.factor=2`, `min.insync.replicas=2`
  → `KAFKA_BOOTSTRAP_SERVERS` env var: `kafka:9092` (local) vs `kafka-1:9092,kafka-2:9092,kafka-3:9092` (EC2)
  → Producers/consumers đọc từ `os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")`
  → **Interview**: "Local 1 broker, production EC2 3 broker KRaft — không dùng MSK vì tự host rẻ hơn 10x với workload này"

---

## ✅ P6 — Full AI Data Platform Scale (A/C/D/E Done)

> Cover toàn bộ 6 disciplines: DE / Analytics Engineering / MLOps / DevOps / MLE / AI Engineering
> Mục tiêu: ~9.5/10, production-scale patterns cho Shopee/Grab/VNG level

### Section A — Data Engineering

- [x] **Task 74 — Kafka Schema Registry (Confluent Avro)** — *đã implement, CHƯA nối dây vào hot path (xem cập nhật bên dưới)*
  > Producer/consumer hiện dùng JSON tự do — schema thay đổi là silent bug.
  > Schema Registry: enforce backward/forward compatibility tại broker level.

  → `docker-compose.infra.yml`: `schema-registry` service đã có — thêm `SCHEMA_REGISTRY_URL` env var cho producers
  → `src/data_pipeline/streaming/schemas/`: 3 Avro schema files (nhiều hơn plan — thêm `agent_feedback.avsc`):
  - `new_reviews.avsc`: `{user_id, product_id, rating, review_text, sentiment, crawled_at}`
  - `recsys_predictions.avsc`: `{customer_id, recommendations[], scores[], experiment_group, timestamp}`
  - `agent_feedback.avsc`: `{customer_id, product_id, action, session_id, timestamp}`
  → `src/data_pipeline/streaming/producer/avro_producer.py` (NEW): `AvroProducerManager` với `try/except` fallback sang JSON nếu Schema Registry unavailable — **implementation đầy đủ, hoạt động được**, nhưng **chưa được gọi ở producer thật nào** ([recsys_api/kafka_producer.py](src/serving/recsys_api/kafka_producer.py), [agent_api/main.py](src/serving/agent_api/main.py) `/feedback` đều vẫn publish JSON trực tiếp qua `confluent_kafka.Producer` thô, không qua `AvroProducerManager`). Đây là "written but not wired" — code thật, không phải giả — chỉ chưa tới bước tích hợp cuối. Không nên tính là "done" cho tới khi hot path thật sự gọi qua nó.
  → **CV note**: "Schema evolution without breaking consumers — standard for enterprise Kafka" — chỉ dùng câu này khi đã thực sự wire vào producer thật, chưa phải bây giờ.

- [ ] **Task 75 — Apache Flink (Stateful Stream Processing)**
  > Kafka+Python consumer (hiện tại) không có windowed aggregations, stateful joins, exactly-once.
  > Flink: windowed CTR calculation, session detection, real-time feature engineering.

  → `docker-compose.infra.yml`: thêm Flink JobManager + TaskManager
  → `src/data_pipeline/flink/ctr_window.py` (NEW):
  - Tumbling window 5 phút: tính CTR per experiment_group từ `recsys_predictions` topic
  - Emit vào `ctr_realtime` topic → Grafana live dashboard
  → Demo: so sánh với batch dbt `gold_ab_test_results` — Flink delay ~5s vs dbt delay ~1 ngày
  → **Interview**: "Khi nào dùng Flink vs Spark Streaming vs Kafka Streams?"

- [ ] **Task 76 — CDC với Debezium (Real-time Lakehouse)**
  > Daily full scan từ PostgreSQL → bronze layer inefficient khi data lớn.
  > Debezium: capture row-level changes từ PostgreSQL WAL → Kafka → Iceberg (incremental).

  → `docker-compose.infra.yml`: thêm Debezium connector (Kafka Connect plugin)
  → Register connector: PostgreSQL source → topic `pg.public.products`, `pg.public.comments`
  → Kafka Connect `IcebergSinkConnector` → append-only Iceberg table (incremental mode)
  → Airflow: thay `bronze_to_silver` daily full scan → trigger on Kafka lag > 0
  → **Pattern**: CDC → Kafka → Iceberg = "real-time open lakehouse"

### Section B — Analytics Engineering

- [ ] **Task 77 — Metabase / Apache Superset (Self-Serve Analytics)**
  > Gold Mart dbt models không có BI layer — business users không thể query tự bản thân.

  → `docker-compose.monitor.yml`: thêm Metabase (port 3003) hoặc Superset (port 8088)
  → Connect trực tiếp vào PostgreSQL (Gold Mart tables từ dbt)
  → 3 dashboards self-serve (không cần SQL):
  - RFM Customer Segmentation (từ `gold_rfm_segments`)
  - A/B Test Results (từ `gold_ab_test_results`)
  - Product Performance by Category
  → **Demo giá trị**: "Non-technical stakeholders tự query KPIs không cần data team"

- [ ] **Task 78 — dbt Semantic Layer + dbt Exposures**
  > Mỗi dashboard tính `ctr` theo cách riêng → inconsistent metrics.
  > Semantic Layer: define `metric: ctr` một lần, tất cả dùng consistent definition.

  → `dbt_project/models/metrics/ctr.yml`:
  ```yaml
  metrics:
    - name: ctr
      label: Click-Through Rate
      model: ref('gold_ab_test_results')
      calculation_method: ratio
      expression: clicks / total_recommendations
  ```
  → `dbt_project/models/_exposures.yml`: document lineage đến Grafana dashboards + Metabase
  → `dbt docs generate` → lineage graph shows: "Grafana dashboard X dùng model Y"

### Section C — MLOps

- [x] **Task 79 — Multi-Env Model Promotion Pipeline**
  > Hiện chỉ có 1 MLflow stage (Production). Không có gate giữa Dev→Staging→Prod.
  > Enterprise pattern: model train Dev → test Staging data → promote Production.

  → `airflow/dags_staging/model_promotion.py` (NEW) — trong `dags_staging/` vì permission issue với Airflow container:
  - `register_staging`: find latest MLflow run → `mlflow.register_model()` stage=Staging
  - `eval_staging`: load Staging model → compute NDCG@10 trên holdout set
  - `promote_or_flag`: `if NDCG_staging > NDCG_prod * 0.97` → transition to Production; else → log warning
  → Reuse `src/ml_models/recsys/evaluation/evaluate.py`
  → Pairs với Task 46 (multi-env config) và Task 26 (MLflow registry)

- [ ] **Task 80 — Canary Deployment (K8s + Argo Rollouts)**
  > Rolling update 100% traffic là rủi ro — nếu new model kém, tất cả user bị ảnh hưởng.

  → `k8s/recsys-api/rollout.yaml`: `Rollout` thay `Deployment`, strategy: canary 5%→30%→100%
  → `k8s/recsys-api/analysis.yaml`: `AnalysisTemplate` check Prometheus P95 < 200ms
  → Auto rollback nếu analysis fail
  → Pairs với Task 79 (model promotion) — canary cho infra, model promotion cho model

- [x] **Task 81 — Model Card Automation**
  > Sau mỗi training run không có tóm tắt bias/performance theo segment.
  > Model Card: auto-generate report per customer segment (Champion/Loyal/New/At Risk).

  → `src/ml_models/recsys/training/model_card.py` (NEW):
  - Load `gold_rfm_segments` → tính NDCG@10 per segment (Champion/Loyal/At Risk/New)
  - Bias flag: `min_segment_ndcg (new customers) < max_segment_ndcg NDCG (champion) * 0.8` → warn trong card
  - Generate `model_card.md` Markdown với table, bias warning, training params
  → `mlflow.log_artifact("model_card.md")` sau mỗi training run
  → `airflow/dags_staging/`: `generate_model_card` task trong retrain DAG

### Section D — AI Engineering

- [x] **Task 82 — MCP Server cho RecSys API**
  > Expose `/recommend` + `/search` như MCP tools → AI assistants query RecSys trực tiếp.
  > Differentiator 2025–2026: "AI-native API"

  → `src/serving/recsys_api/mcp_server.py` (NEW) dùng `fastmcp`
  → 3 MCP tools (nhiều hơn plan — thêm `get_product_detail`):
  - `get_recommendations(customer_id, top_k)` → `POST /recommend`
  - `search_products(query, max_price, top_k)` → `GET /search`
  - `get_product_detail(product_id)` → `GET /products/{id}`
  → `fastmcp>=2.0.0` thêm vào requirements
  → Test: Claude Desktop connect → query "gợi ý sản phẩm cho user 12345"

- [x] **Task 83 — LangGraph Multi-Agent Orchestration**
  > ReAct agent hiện tại không có state persistence xuyên session, không parallel tool execution tại graph level, không thể dừng lại hỏi user.
  > LangGraph thêm 3 giá trị thực: checkpointing, parallel branches, human-in-the-loop.

  → `src/serving/agent_api/graph.py` (NEW): `StateGraph` với 5 nodes (nhiều hơn plan):
  - `memory_node`: load Redis history + `pref_context` → inject vào state
  - `clarification_node`: check intent → interrupt nếu query quá ngắn/mơ hồ
  - `route_node`: Python logic routing (không dùng LLM — tiết kiệm token)
  - `search_node` + `recsys_node`: chạy **parallel** via LangGraph `Send` API
  - `synthesis_node`: LLM generate final response từ merged results
  → **Checkpointing**: `MemorySaver` — graph state persist across turns per `thread_id`
  → **Human-in-the-loop**: `clarification_node` interrupt → hỏi user → resume từ checkpoint
  → `AgentState(TypedDict)`: `customer_id`, `query`, `messages`, `search_results`, `recommendations`, `needs_clarification`, `pref_context`, `final_response`
  → `POST /chat/graph` endpoint mới; `thread_id=customer_id` để persist per-user state
  → `/chat` ReAct cũ vẫn giữ làm fallback

- [x] **Task 84 — RLHF / Agent Feedback Loop**
  > Agent không biết recommendation nào user thực sự thích.
  → Streamlit click → `POST /feedback` → Kafka `agent_feedback`
  → Airflow aggregate → MLflow Dataset → trigger fine-tune khi đủ data
  → Closes the loop: RAG → Rec → User action → Training signal
  → `src/serving/agent_api/main.py`: `POST /feedback` endpoint (`FeedbackRequest`: `customer_id`, `product_id`, `action: Literal["click","purchase","ignore"]`, `session_id`)
  → Kafka topic `agent_feedback` — JSON thô hiện tại (schema `agent_feedback.avsc` đã có sẵn từ Task 74 nhưng chưa nối dây, xem ghi chú ở Task 74)
  → `airflow/dags_staging/feedback_aggregator.py` (NEW): Spark read `agent_feedback` → aggregate → append vào `silver.user_feedback_events` Iceberg table
  → `src/ml_models/recsys/train_model.py`: `_load_feedback_signal()` đọc `user_feedback_events.parquet` + `_merge_feedback()` upsample purchases×3, clicks×1
  → `mlflow.log_metric("n_feedback_events")` và `mlflow.log_metric("n_train_samples")` per run
  → Closes the loop: User click/purchase → Kafka → Iceberg → retraining signal

- [x] **Task 91 — Document Chunking Pipeline (Passage-Level RAG)**
  > RAG hiện tại: search trên product records (item-level, ~50 tokens/item). Không xử lý được long-form content.
  > Chunking: split long product descriptions / review aggregations → passage-level retrieval — câu trả lời chi tiết hơn về specs, so sánh sản phẩm.

  → `src/serving/agent_api/chunker.py` (NEW):
  - `chunk_size=1500` chars (lớn hơn plan 256 — phù hợp hơn cho Vietnamese policy docs)
  - Vietnamese-aware sentence boundary: regex `[.!?。！？\n]` trước khi split
  - Mỗi chunk giữ `parent_product_id` + `chunk_index` trong metadata
  → `src/serving/agent_api/indexer.py` (NEW): index chunks vào Qdrant collection `product_passages`
  - Input: product catalog `product_name + description + top_reviews_aggregated`
  - Output: vector per chunk với payload `{product_id, price, category, chunk_text}`
  → `src/serving/agent_api/rag.py`: thêm `search_passages()` method — dùng khi user hỏi về specs/so sánh
  → **Demo**: "điện thoại này pin có tốt không?" → passage search trong reviews → câu trả lời trích dẫn review cụ thể
  → **Interview**: "Em implement parent-document retrieval: chunk nhỏ để embed chính xác, return parent document để context đầy đủ"

- [x] **Task 92 — LLM Retry + Load Balancing**
  > `agent.py` hiện tại: LLM call fail → fallback ngay, không retry. Single backend (Claude hoặc Ollama).
  > Retry: exponential backoff trước khi fallback. Load balancing: Claude primary → Ollama secondary → rule-based tertiary.

  → `src/serving/agent_api/agent.py`: wrap LLM call với `tenacity.retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))`
  → `litellm.Router` với model list + fallback: `["anthropic/claude-haiku-4-5", "ollama/qwen2.5:7b"]`
  → Metric mới: `llm_retry_total` counter, `llm_backend_switches_total` counter
  → **Interview**: "Primary/fallback pattern với retry — không để single point of failure ở LLM layer"

- [x] **Task 93 — DeepEval Agent Evaluation**
  > RAGAS (Task 67) đánh giá RAG pipeline offline. Không có real-time hallucination detection và agent-level metrics.
  > DeepEval: G-Eval metric framework — hallucination, task completion, tool precision.

  → `src/serving/agent_api/eval/deepeval_eval.py` (NEW):
  - `LiteLLMJudge` bridge (custom) — wrap `litellm.completion()` thay `langchain-anthropic` để tránh dependency conflict
  - 3 metrics: `HallucinationMetric(threshold=0.5)`, `ToolCorrectnessMetric`, `TaskCompletionMetric`
  - 10 Vietnamese test cases (thay vì 15 plan — cân nhắc tốc độ eval)
  → Log results to MLflow experiment `deepeval_agent_eval`
  → `src/serving/agent_api/eval/compare_evals.py`: RAGAS scores vs DeepEval scores per prompt version
  → `deepeval>=0.21.0` thêm vào requirements
  → **Story**: "2 eval frameworks để cross-validate — RAGAS cho RAG quality, DeepEval cho agent behavior"

- [x] **Task 94 — Prompt Rollback + Automated Gating**
  > Hiện tại: đổi prompt version = manual `AGENT_PROMPT_VERSION=v2` restart. Không có automated safety gate.
  > Automated rollback: nếu RAGAS scores của version mới thấp hơn current → không deploy, alert.

  → `src/serving/agent_api/prompt_registry.py` (NEW): Redis-backed prompt registry
  - `get_prompt(version)` → check Redis cache → fallback MLflow artifact download → local file
  - `rollback(to_version)` → update `current_version` in Redis, hot-reload không restart container
  - `lru_cache(maxsize=10)` cho prompt versions đã load
  → `airflow/dags_staging/prompt_eval_gate.py` (NEW): DAG chạy sau mỗi push prompt version mới:
  1. `run_ragas_eval(prompt_version="new")` → `faithfulness`, `answer_relevancy`
  2. Compare với MLflow logged scores của current version
  3. Drop > 5% → `rollback()` + log warning; ổn → `set_current_version("new")`
  → **Story**: "Prompt là code — cần version control, CI gate và rollback như model"

- [x] **Task 95 — Document Ingestion Pipeline + Knowledge Base RAG**
  > RAG hiện tại chỉ search product catalog. Không trả lời được "chính sách đổi trả là gì?", "ship mất mấy ngày?".
  > Knowledge base: ingest policy docs + FAQ → passage RAG cho customer support queries.

  **Data sources — scrape từ Tiki.vn:**
  | Document | Nguồn | Câu hỏi trả lời được |
  |---|---|---|
  | Chính sách đổi trả | tiki.vn/chinh-sach-doi-tra | "Đổi trả trong bao lâu?", "Sản phẩm lỗi xử lý thế nào?" |
  | Chính sách bảo hành | tiki.vn/bao-hanh | "Bảo hành bao lâu?", "Bảo hành tại đâu?" |
  | Chính sách thanh toán | tiki.vn/phuong-thuc-thanh-toan | "Thanh toán bằng gì?", "Trả góp 0% thế nào?" |
  | Chính sách vận chuyển | tiki.vn/chinh-sach-giao-hang | "Ship nhanh mấy ngày?", "Phí ship bao nhiêu?" |
  | FAQ / Hỏi đáp | tiki.vn/faq | Câu hỏi chung |
  | TikiNOW | tiki.vn/tikinow | "Giao 2 giờ điều kiện gì?" |
  | TikiPRO membership | tiki.vn/tikipro | "Mua PRO được gì?", "Giá bao nhiêu?" |

  **Data sources — tự tạo sample (PDF/Word):**
  - `docs/kb/huong_dan_mua_hang.pdf` — các bước tìm kiếm → checkout → nhận hàng
  - `docs/kb/chinh_sach_hoan_tien.pdf` — thời hạn, điều kiện, hình thức (TikiXu, chuyển khoản)
  - `docs/kb/huong_dan_khieu_nai.pdf` — liên hệ CSKH, mở ticket, leo thang
  - `docs/kb/voucher_guide.pdf` — cách áp mã giảm giá, điều kiện áp dụng
  - `docs/kb/category_guide.xlsx` — spec comparison: "laptop 15-20 triệu nên mua gì?"

  **Implementation:**
  → `src/serving/agent_api/ingestion/` (NEW package):
  - `web_loader.py`: `requests + BeautifulSoup` → scrape Tiki policy pages → clean text
  - `pdf_loader.py`: `pypdf` → extract text từ PDF docs (dùng `pypdf` thay `pdfplumber` — lighter dependency)
  - `excel_loader.py`: `openpyxl` → đọc spec sheets, so sánh sản phẩm
  - `pipeline.py`: Document → `chunker.py` (Task 91) → PhoBERT embed → Qdrant `knowledge_base` collection
  → `rag.py`: thêm `search_knowledge(query)` — tìm trong KB, không trộn với product catalog
  → Agent tool mới: `get_policy_info(topic: str)` → KB search → answer
  → **Demo**: "Đổi trả trong bao lâu?" → KB search → "Theo chính sách Tiki, trong 30 ngày..."
  → **Add**: `pdfplumber>=0.10`, `beautifulsoup4>=4.12`, `openpyxl>=3.1`

- [x] **Task 96 — Configurable Policy Engine (Guardrails-as-Config)**
  > Guardrails hiện tại dùng hardcoded regex trong `guardrails.py`. Thêm/sửa rule = phải sửa code + redeploy.
  > Policy engine: load rules từ YAML → hot-reload không restart container.

  → `config/policies/input_rules.yaml` (NEW): 8 rules (nhiều hơn plan 3 rules):
  - `pii_phone`, `pii_email`, `pii_credit_card` — detect và mask PII
  - `prompt_injection` — block "ignore previous instructions", "jailbreak"
  - `harmful_content` — block toxic language
  - `off_topic` — warn queries không liên quan e-commerce
  - `too_short` (< 3 chars) + `too_long` (> 500 chars) — input validation
  → `config/policies/output_rules.yaml`: hallucinated price detection + system prompt leak
  → `src/serving/agent_api/policy_engine.py` (NEW): `PolicyEngine` class
  - `check(text)` → `PolicyResult(blocked, warnings, masked_text)`
  - `lru_cache` cho compiled regex patterns (không re-compile mỗi request)
  - `reload()`: check YAML mtime mỗi 60s → hot-reload không restart
  → `guardrails.py`: delegate to `PolicyEngine` thay hardcoded regex
  → **Interview**: "Policy-as-config — security team update rules không cần engineer redeploy"

- [ ] **Task 97 — Multimodal Product Search (khi có ảnh sản phẩm)**
  > Hiện tại: text-only search. Khi có `thumbnail_url` → embed ảnh → visual search + multimodal fusion.
  > CLIP: joint image+text embedding → "tìm áo màu đỏ" hoặc upload ảnh → sản phẩm tương tự.

  **Phase 1 — Image embedding pipeline** (khi có URLs):
  → `src/data_pipeline/jobs/image_embed.py` (NEW):
  - Download `thumbnail_url` → CLIP `ViT-B/32` embed (512-dim)
  - Lưu vào Iceberg `silver.item_image_embeddings(product_id, image_vector)`

  **Phase 2 — Qdrant multi-vector collection:**
  → Tạo collection `items_multimodal` với named vectors:
  ```python
  vectors_config={
      "text": VectorParams(size=128, distance=Distance.COSINE),   # PhoBERT
      "image": VectorParams(size=512, distance=Distance.COSINE),  # CLIP
  }
  ```
  → `vector_store.py`: `upsert_multimodal()` + `search_fusion(α*text_score + (1-α)*image_score)`

  **Phase 3 — New endpoints:**
  → `POST /search/visual {image_base64}` → CLIP embed → Qdrant image search → top-K
  → `POST /search/multimodal {query, image_base64}` → fused ranking

  **Phase 4 — Auto-captioning** (optional):
  → Claude claude-haiku-4-5-20251001 vision → generate Vietnamese description từ ảnh → enrich catalog text
  → **Add khi làm**: `open-clip-torch>=2.24`, `pillow>=10.0`
  → **Interview**: "Text tower (PhoBERT) + Image tower (CLIP) → Qdrant multi-vector → multimodal fusion search"

### Section E — Machine Learning Engineering

- [x] **Task 85 — Learning to Rank (XGBoost LambdaMART)**
  > Rule-based reranker không học từ data. LTR optimize trực tiếp trên NDCG/MAP.

  → `src/ml_models/recsys/training/train_ltr.py` (NEW):
  - 7 features (nhiều hơn plan 5): `[semantic_score, tfidf_score, avg_sentiment, price_norm, category_match, popularity_rank, rrf_score]`
  - Label: click=1, purchase=3, ignore=0 (graded relevance từ `prediction_events`)
  - `xgboost.train(params={"objective": "rank:ndcg", "eval_metric": "ndcg@10"}, ...)`
  → Serve trong `agent_api/reranker.py` — `RERANKER_BACKEND=ltr` load XGBoost model từ MLflow

- [x] **Task 86 — LightGCN Graph Neural Network**
  > Two-Tower không học high-order collaborative signal (A mua B, B mua C → A cũng thích C).

  → `src/ml_models/recsys/models/lightgcn.py` (NEW):
  - Custom `LightGCNConv` từ scratch — không dùng `torch-geometric` (tránh version conflict, CUDA dependency phức tạp)
  - 3-layer propagation → all-layer mean pooling (thay concat để ổn định hơn)
  - BPR loss (Bayesian Personalized Ranking): `loss = -log(σ(score_pos - score_neg))`
  → `src/ml_models/recsys/training/train_lightgcn.py` (NEW): build bipartite graph từ `user_history.parquet`, log NDCG@10 vào MLflow experiment `lightgcn_training`, export `lightgcn_embeddings.npz`
  → `src/serving/recsys_api/loaders.py`: lazy load `LIGHTGCN_ITEM_EMBEDDINGS` khi `RETRIEVAL_BACKEND=lightgcn`
  → `src/serving/recsys_api/retrieval.py`: `RETRIEVAL_BACKEND=lightgcn` → `_retrieve_lightgcn()` numpy dot product + post-hoc filter
  → A/B test được 2 backends: `RETRIEVAL_BACKEND=twotower|lightgcn`

- [x] **Task 87 — SASRec + DIN (Session-Based + Behavior-Based RecSys)**
  > Two-Tower không biết thứ tự xem và không có behavior attention — 2 gaps quan trọng cho e-commerce.
  > **SASRec** (Kang & McAuley 2018): next-item prediction từ session sequence.
  > **DIN** (Deep Interest Network, Alibaba 2018): attention trên user history *với target item* — score candidate items.

  **SASRec** → `src/ml_models/recsys/models/sasrec.py` (NEW):
  - Pure PyTorch self-attention causal (không cần NVIDIA Transformer4Rec/CUDA)
  - `predict_next(seq, candidates)` method cho inference
  → `src/ml_models/recsys/training/train_sasrec.py` (NEW — không có trong plan ban đầu):
  - `_build_sessions()`: gap > 30 phút = new session
  - `_encode_items()`: item_id → 1-indexed (0 = padding)
  - `SASRecDataset` + `DINDataset` cho 2 models
  - `_eval_sasrec()`: NDCG@10 trên validation sessions
  - Log vào MLflow experiment `sasrec_din_training`; save `sasrec.pt`, `din.pt`, `sasrec_item_enc.json`
  → `POST /recommend/session` (NEW endpoint trong `main.py`):
  - Lazy-load SASRec khi `/recommend/session` lần đầu được gọi
  - Encode session → pad to `MAX_SEQ=50` → `predict_next()` → exclude session items → enrich với `item_lookup_df`

  **DIN** → `src/ml_models/recsys/models/din.py` (NEW):
  - `ActivationUnit`: attention score của mỗi history item với target item embedding  `concat([target, hist, target-hist, target*hist])` → scalar attention weight
  - Input: target item + user history items → scalar relevance score
  - **Interview**: "SASRec predict next item từ sequence, DIN rank candidates bằng behavior attention — Alibaba production standard"

- [x] **Task 88 — ONNX Export + Serving Layer**
  > PyTorch model cold start ~2s, memory heavy. ONNX runtime 2–3x faster inference.
  > *(Plan nói TorchServe — thay bằng ONNX Runtime trực tiếp: nhẹ hơn, không cần Java)*

  → `src/ml_models/recsys/export_onnx.py` (NEW): `torch.onnx.export(model.item_tower, dummy_inputs, "item_tower.onnx", opset_version=17)` + ONNX validation
  → `src/serving/recsys_api/inference_onnx.py` (NEW — không có trong plan ban đầu):
  - `ONNXItemTower(onnx_path)`: wraps `onnxruntime.InferenceSession`, CUDA/CPU providers
  - `precompute_item_embeddings_onnx()`: batched inference, drop-in replacement cho `model.item_tower()` at startup
  - `is_available()`: check ONNX Runtime installed + file exists
  → Activate via `ONNX_ENABLED=true` env var → loaders.py dùng `precompute_item_embeddings_onnx()` thay PyTorch
  → Benchmark script: latency + memory ONNX vs PyTorch → log to MLflow

### Section F — Infrastructure / Platform

- [ ] **Task 89 — GDPR / PII Compliance**
  > `customer_id` + `review_text` là PII. Không có right-to-erasure.
  > Enterprise compliance requirement — Shopee/Grab/Lazada đều cần.

  → Tag PII columns trong dbt `_sources.yml`: `meta: {pii: true}`
  → `src/data_pipeline/jobs/gdpr_erasure.py` (NEW):
  - Input: `customer_id`
  - Delete từ: Redis keys (`conv:*`, `pref:*`, `recs:*`), Feast online store, Iceberg compaction
  → `POST /gdpr/erasure` endpoint trong recsys-api
  → Audit log: mỗi erasure request → log vào S3/MinIO

- [ ] **Task 90 — Chaos Engineering**
  > "Nếu Redis down thì sao?" — không biết vì chưa test graceful degradation.

  → `tests/chaos/test_resilience.py` (NEW): pytest + subprocess kill + assert response OK
  → Test scenarios:
  - `docker compose kill redis` → recsys-api vẫn serve (trending fallback)
  - `docker compose kill mlflow` → agent-api startup warning nhưng vẫn chạy
  - `docker compose kill ollama` → fallback sang Claude API (`AGENT_LLM_BACKEND=claude`)

---

## 📄 P7 — Documentation

### ✅ Supporting docs đã tạo (ngoài plan gốc)

- [x] **`START.md`** — quick-start: chạy local toàn bộ stack, link tới TRAIN.md + deploy_aws.md
- [x] **`FLOW.md`** — kiến trúc end-to-end + giải thích design decisions từng tầng
- [x] **`FILES.md`** — bản đồ file/module toàn repo
- [x] **`TRAIN.md`** — hướng dẫn train không có GPU (Colab/Kaggle): data prep → train → MLflow → return artifacts
- [x] **`REVIEW.md`** — Principal-level code review: P0/P1/P2 issues + fixes + production readiness Q&A
- [x] **`INTERVIEW.md`** — tài liệu ôn phỏng vấn AI Platform (CS core → Lead), ~3300 dòng, gắn với code thật
- [x] **`docs/deploy_aws.md`** — 3 AWS deploy scenarios (Free/Balanced/Full-Native) + cost breakdown

> Đây là docs portfolio/interview — KHÁC với 3 mục enterprise docs bên dưới (ADR/SLO/Runbook) vẫn cần làm.

### ⬜ Enterprise docs (chưa làm)

- [ ] **5 Architecture Decision Records (ADRs)** trong `docs/adr/`
  → `001-why-iceberg-over-delta.md`
  → `002-why-two-tower-over-matrix-factorization.md`
  → `003-why-faiss-over-milvus.md` (và khi nào cần upgrade sang Milvus)
  → `004-why-feast-feature-store.md`
  → `005-batch-vs-realtime-tradeoffs.md`

- [ ] **SLO Definition** — `docs/slo.md`
  → recsys-api: availability 99.5%, P95 < 200ms (miss) / < 10ms (hit)
  → ETL pipeline: data freshness < 6h, DQ pass rate > 99%
  → Alertmanager rules enforcement (Task 40)

- [ ] **Operational Runbook** — `docs/runbook.md`
  → "API latency spike" / "Cache hit rate drop" / "DAG fail" / "Model drift" / "DLQ growing"

---

## 🚀 Deploy — AWS Demo (lấy số thật cho CV)

Làm sau khi xong P4. Deploy 1 ngày để measure, rồi tắt.

- [ ] `terraform apply` → EC2 t3.xlarge spot (`ap-southeast-1`) ~$0.7/ngày
- [ ] Push images lên ECR (CI/CD Task 31 xong thì tự động)
- [ ] `docker-compose up` trên EC2: recsys-api + nlp-api + redis + prometheus + grafana
- [ ] Chạy loadtest recsys: `locust -f loadtest_recsys.py --host http://<EC2-IP>:8001 --users 100 --run-time 5m`
- [ ] Chạy loadtest agent: `locust -f loadtest_agent.py --host http://<EC2-IP>:8003 --users 20 --run-time 5m`
- [ ] **Lấy số thật → paste vào CV**:
  - P95 latency (cache hit vs miss)
  - RPS sustained
  - Cache hit rate
  - Cold start rate
  - NDCG@10 từ offline eval

---

## 📊 Enterprise Maturity Score (Honest)

So sánh với production platform tại Shopee / Grab / Tiki / VNG level.

| Domain | Score P3 done | Score P4 done | Score P5 done | Score P6 done | Enterprise bar |
|---|---|---|---|---|---|
| Data Engineering | 6/10 | 7/10 | 8/10 | 9/10 | 9/10 |
| MLOps | 7/10 | 8/10 | 9/10 | 9.5/10 | 9/10 |
| ML / Deep Learning | 7/10 | 7.5/10 | 8.5/10 | 9.5/10 | 9/10 |
| Backend / Serving | 6/10 | 7.5/10 | 9/10 | 9.5/10 | 9/10 |
| DevOps | 6/10 | 7/10 | 8.5/10 | 8.5/10 | 9/10 |
| Observability | 6/10 | 7/10 | 9/10 | 9/10 | 9/10 |
| Analytic Engineering | 5/10 | 6.5/10 | 7/10 | 7/10 | 8/10 |
| AI Engineering | 4/10 | 4/10 | 8/10 | 9.5/10 | 8/10 |
| Data Governance | 3/10 | 3/10 | 6/10 | 6/10 | 9/10 |
| Security | 4/10 | 4.5/10 | 7/10 | 8/10 | 9/10 |

> **Data Governance và Security là hai domain khó đạt enterprise bar nhất trong portfolio scope.**

---

## 📋 Future Roadmap — Biết nhưng không implement

Những thứ production enterprise cần, nhưng nằm ngoài scope 1 người / portfolio.

### Data Engineering
| Component | Lý do quan trọng |
|---|---|
| **OpenLineage → Marquez** data lineage | Column-level lineage từ Kafka → Spark → Iceberg → dbt. Task 49 chỉ đến dataset-level |
| ~~Kafka Schema Registry (Confluent Avro)~~ | ✅ Done — Task 74 (3 Avro schemas + AvroProducer với JSON fallback) |
| **CDC với Debezium** | Incremental replication thay daily full scan — true real-time lakehouse |
| **Iceberg maintenance DAG** | OPTIMIZE + EXPIRE SNAPSHOTS hàng tuần để tránh table bloat sau months of data |
| **Apache Ranger / Lake Formation** | Column-level và row-level access control — GDPR requirement |
| **Apache Flink** (stateful stream processing) | Windowed aggregations, stateful joins, exactly-once semantics — enterprise stream processing vượt trội Kafka+Python consumer hiện tại. High demand trên DE JDs |

### MLOps
| Component | Lý do quan trọng |
|---|---|
| **Canary deployment** | 5% traffic → new model → auto-rollback nếu P95 tăng 20% — K8s Argo Rollouts |
| ~~Multi-env model promotion~~ | ✅ Done — Task 79 (`dags_staging/model_promotion.py`) |
| **Distributed training** (PyTorch DDP / Ray) | Scale training lên multi-GPU khi dataset lớn hơn memory |
| **Ray Tune** hyperparameter search | Distributed HPO thay single-node Optuna (Task 44) |
| ~~Model card automation~~ | ✅ Done — Task 81 (`training/model_card.py`, per-segment NDCG + bias flag) |

### Machine Learning Engineering
| Component | Lý do quan trọng |
|---|---|
| **Text features trong Two-Tower** | Embed product title/description bằng PhoBERT pooling → richer item representation (Task 53 làm phần này) |
| ~~Learning to Rank reranker (XGBoost LTR)~~ | ✅ Done — Task 85 (`train_ltr.py`, 7 features, rank:ndcg) |
| **FAISS IndexIVFPQ** | INT8 embedding quantization → 4x faster, 4x less memory — cần cho scale lớn |
| ~~SASRec / BERT4Rec~~ | ✅ Done — Task 87 (`sasrec.py` + `din.py` + `train_sasrec.py` + `/recommend/session`) |
| **Transformer4Rec** (NVIDIA) | Production-grade session-based RecSys với NVIDIA-optimized Transformers |
| ~~LightGCN / GraphSAGE GNN~~ | ✅ Done — Task 86 (`lightgcn.py`, custom LightGCNConv, BPR loss, `RETRIEVAL_BACKEND=lightgcn`) |
| ~~ONNX export + TorchServe~~ | ✅ Done — Task 88 (`export_onnx.py` + `inference_onnx.py`, ONNX Runtime thay TorchServe) |
| **Cross-encoder neural reranker** | BERT-based reranker tính `similarity(query, product)` end-to-end — học từ click data, tốt hơn rule-based reranker hiện tại 3–5% NDCG |
| **Statistical significance (scipy.stats)** | p-value + confidence interval cho A/B test results trong `gold_ab_test_results` (Task 38) — "kết quả CTR +2% có significant không?" là câu interview phổ biến |

### AI Engineering
| Component | Lý do quan trọng |
|---|---|
| **Fine-tune PhoBERT trên Tiki domain** | Zero-shot sentiment có accuracy ~78% — domain fine-tuning → ~88% |
| **LLM recommendation explanation** | "Gợi ý vì bạn thường mua X" — tăng user trust, giảm bounce rate |
| **Multimodal embedding** (CLIP) | Product image vector trong Two-Tower item tower — better cold-item representation |
| **Conversational RecSys** | Chatbot interface với recommendation memory — LLM + Feast user state |
| ~~MCP server cho RecSys API~~ | ✅ Done — Task 82 (`mcp_server.py`, 3 tools: get_recommendations, search_products, get_product_detail) |
| **LLMOps** (LangSmith / Phoenix) | Prompt versioning, cost tracking, latency budget per LLM call — production RAG lifecycle management |
| **Context Engineering** | Optimal context construction cho RAG: chunk size, overlap strategy, metadata filtering — ảnh hưởng trực tiếp RAG quality |
| ~~LangGraph (multi-agent orchestration)~~ | ✅ Done — Task 83 (`graph.py`, 5 nodes, MemorySaver checkpointing, parallel branches, human-in-the-loop) |
| ~~RLHF / Agent feedback loop~~ | ✅ Done — Task 84 (`/feedback` endpoint + Kafka + Iceberg + `_merge_feedback()` trong train_model.py) |

### Analytic Engineering
| Component | Lý do quan trọng |
|---|---|
| **dbt Semantic Layer** | Define `metric: ctr` một lần — tất cả dashboards dùng consistent definition |
| **Metabase / Apache Superset** | Business users self-serve query Gold Mart không cần SQL |
| **Cohort analysis** | CTR theo cohort (tuần đăng ký, user segment) — business insight thật |
| **dbt Exposures** | Document lineage đến từng Grafana dashboard — "dashboard X dùng data từ model Y" |

### Infrastructure / Platform
| Component | Lý do quan trọng |
|---|---|
| **Service mesh (Istio)** | mTLS giữa services, traffic management, circuit breaking ở network layer |
| **Multi-AZ deployment** | Zone redundancy — single AZ down không ảnh hưởng serving |
| **FinOps / Cost allocation tagging** | Tag mọi resource với team, project, env — biết ML training tốn bao nhiêu |
| **Chaos engineering** | Kill pod ngẫu nhiên, test system recovery — Netflix Chaos Monkey pattern |
| **GDPR / PII compliance** | Data subject rights, right to erasure từ feature store — cần legal framework |

---

## 🏭 Honest Assessment — Gap với Big Enterprise Production

| Gap | P4 cover | P5 cover | Vẫn thiếu |
|---|---|---|---|
| Structured logging | ✅ JSON logs + trace ID | — | Centralized log aggregation (ELK stack) |
| Retry / resilience | ✅ tenacity retry | — | Circuit breaker với state machine |
| K8s config management | ✅ ConfigMap + Secret | — | Sealed Secrets / ESO (External Secrets) |
| Prometheus metrics | ✅ /metrics endpoint | ✅ Grafana dashboards | — |
| Alerting | ❌ | ✅ Alertmanager rules | PagerDuty integration |
| API authentication | ❌ | ✅ API key middleware | OAuth2 / SSO (Cognito, Auth0) |
| Security scanning | ❌ | ✅ Trivy in CI | SAST, dependency audit (Snyk) |
| Shadow mode | ❌ | ✅ Task 43 | — |
| HPO | ❌ | ✅ Optuna Task 44 | Distributed HPO (Ray Tune) |
| Graceful shutdown | ❌ | ✅ Task 45 | — |
| Multi-environment | ❌ | ✅ Task 46 | GitOps (ArgoCD) |
| Vector DB production | ❌ | ✅ Qdrant Task 47 | Managed (Zilliz Cloud, Pinecone) |
| Semantic search / RAG | ❌ | ✅ Task 48 | LLM gateway, prompt versioning |
| Data catalog | ❌ | ✅ DataHub Task 49 | Apache Ranger access control |
| Data lineage (column-level) | ❌ | Partial | OpenLineage full integration |
| PII handling | ❌ | ❌ | Legal framework requirement |
| Multi-AZ HA | ❌ | ❌ | $$$, không practical cho portfolio |
| Disaster recovery | ❌ | ❌ | RTO/RPO = business decision |

> **Trong interview**: chủ động mention gap, giải thích tại sao chưa làm ở portfolio scope.
> Senior thinking = biết mình thiếu gì và tại sao, không phải cố implement hết không cần thiết.

---

## Timeline

```
Tuần 1–2    P0 + P1 + P2                          ✅ Done
Tuần 3–4    P3                                    ✅ Done
Tuần 5–6    P4 (Tasks 26–38)                      ✅ Done
Tuần 7–8    P5 (Tasks 39–52) + P6 Section E,D     ✅ Done
Tuần 9      P6 Section A, C + gaps                ✅ Done
Tuần 10     P7 (ADR, SLO, Runbook)                → Docs
Tuần 11     Deploy AWS + loadtest                  → Live numbers cho CV
```

---

## CV Impact Summary

Hoàn thành P0–P4 đủ để pass technical screen senior. Hoàn thành P5 + Deploy:

| CV line | Before P5 | After P5 + Deploy |
|---|---|---|
| Latency | "low latency serving" | **"P95 8ms (cache hit), 45ms (model)"** |
| Scale | "production-ready" | **"500 RPS, single t3.xlarge, $0.7/ngày"** |
| Observability | "monitoring with Prometheus" | **"Full Grafana dashboards + Alertmanager SLO alerts"** |
| Search | "recommendation system" | **"RAG semantic search in Vietnamese"** |
| ML | "Two-Tower model" | **"NDCG@10: 0.73, HPO với Optuna, shadow mode deployment"** |
| Governance | — | **"DataHub data catalog + column lineage tracking"** |

---

*Cập nhật lần cuối: 2026-06-28*
