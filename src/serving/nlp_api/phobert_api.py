# File: src/ml_models/inference/api.py
import logging
import os
import time
from typing import Any, Dict, List, Optional

import redis
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentAPI")
start_time = time.time()
app = FastAPI(title="PhoBERT Sentiment API", version="1.0")

# =========================================================
# AUTH + RATE LIMIT
#
# Cùng pattern với recsys_api/main.py và agent_api/guardrails.py (Redis
# INCR theo key, window 60s) — không bịa cơ chế khác cho nhất quán giữa
# các service. NLP_API_KEY không set = auth tắt (dev local).
# =========================================================

NLP_API_KEY = os.getenv("NLP_API_KEY")
_RATE_LIMIT_PER_MIN = int(os.getenv("NLP_RATE_LIMIT_PER_MIN", "60"))
_PUBLIC_PATHS = {"/health"}

# db=3 — cùng concern rate-limit với recsys_api/agent_api (P2-7), tách khỏi
# cache (db=0) / Feast (db=1) / agent memory (db=2).
_redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD") or None,
    db=3,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
)


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if request.url.path in _PUBLIC_PATHS:
        return await call_next(request)

    if NLP_API_KEY:
        provided_key = request.headers.get("X-API-Key")
        if provided_key != NLP_API_KEY:
            return JSONResponse(
                status_code=401, content={"detail": "Missing or invalid X-API-Key"}
            )
        rate_key = f"ratelimit:apikey:{provided_key}"
    else:
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"ratelimit:ip:{client_ip}"

    try:
        count = _redis_client.incr(rate_key)
        if count == 1:
            _redis_client.expire(rate_key, 60)
        if count > _RATE_LIMIT_PER_MIN:
            return JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"}
            )
    except Exception:
        pass  # Redis down — fail open, cùng trade-off với recsys_api/agent_api

    return await call_next(request)


# 1. ĐƯỜNG DẪN TỚI THƯ MỤC MODEL CỦA BẠN
MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/nlp_models/phobert/version_001")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
logger.info("Đang tải model PhoBERT từ thư mục local...")

# 2. Load Model & Tokenizer một lần duy nhất khi khởi động server
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Đang tải model PhoBERT từ {MODEL_PATH} lên {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()  # Bật chế độ dự đoán

logger.info("✅ Load model thành công!")

_ONNX_PATH = os.path.join(MODEL_PATH, "model.onnx")
onnx_session = None
try:
    import onnxruntime as ort

    if os.path.exists(_ONNX_PATH):
        onnx_session = ort.InferenceSession(
            _ONNX_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        logger.info(
            "✅ ONNX Runtime enabled (%s, provider=%s)",
            _ONNX_PATH,
            onnx_session.get_providers()[0],
        )
    else:
        logger.info("ONNX model không tồn tại tại %s — dùng PyTorch", _ONNX_PATH)
except ImportError:
    logger.info("onnxruntime chưa cài — dùng PyTorch")

# Class mapping
id2label = {0: "Negative", 1: "Positive", 2: "Neutral"}


# -----------------------
# Input schema
# -----------------------


class TextBatch(BaseModel):
    texts: List[str]


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_version": MODEL_VERSION, "device": device}


@app.post("/predict")
async def predict_sentiment(batch: TextBatch):
    # Khởi tạo logger thời gian ngay tại lúc bắt đầu request
    request_start = time.time()
    try:
        raw_texts = batch.texts
        if not raw_texts:
            return {"results": []}

        # final_results = [None] * len(raw_texts)
        final_results: List[Optional[Dict[str, Any]]] = [None] * len(raw_texts)
        # Bước 1: Phân loại dữ liệu Ngon và Rác
        # results = [None] * len(batch.texts)
        valid_indices = []
        valid_texts = []

        # Bước 1: Tách dữ liệu (O(n))
        for i, text in enumerate(raw_texts):
            clean_text = str(text).strip() if text else ""
            if clean_text:
                valid_indices.append(i)
                valid_texts.append(clean_text)
            else:
                # Gán trực tiếp vào vị trí i
                final_results[i] = {
                    "text": text,
                    "sentiment": "Neutral",
                    "confidence": "100.00%",
                    "note": "Empty input bypass",
                }

        # Tokenize batch
        if valid_texts:
            inputs = tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            if onnx_session is not None:
                ort_out = onnx_session.run(
                    None,
                    {
                        "input_ids": inputs["input_ids"].cpu().numpy(),
                        "attention_mask": inputs["attention_mask"].cpu().numpy(),
                    },
                )[0]
                logits = torch.from_numpy(ort_out)
                probs = torch.softmax(logits, dim=-1)
            else:
                with torch.inference_mode():  # Dùng inference_mode thay no_grad (nhanh hơn xíu)():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)

            # Bước 3: Map kết quả từ Model vào đúng vị trí ban đầu
            # j là index chạy trong batch kết quả của model
            # original_idx là index thật trong list khách gửi lên
            for j, original_idx in enumerate(valid_indices):
                prob = probs[j]
                class_id = prob.argmax().item()
                confidence = prob.max().item() * 100

                final_results[original_idx] = {
                    "text": raw_texts[original_idx],  # Trả về text gốc
                    "sentiment": id2label[class_id],
                    "confidence": f"{confidence:.2f}%",
                }

        latency = time.time() - request_start
        logger.info(f"Inference completed in: {latency:.4f}s")

        # Luôn trả về cấu trúc nhất quán {"results": [...]}
        # Trong production, việc thay đổi kiểu trả về (Object vs Array) khiến Client rất khó xử lý.
        return {"results": final_results, "latency": f"{latency:.4f}s"}

    except Exception as e:
        logger.error(f"Lỗi khi dự đoán: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Model Error")
