FROM python:3.10-slim

# Cài đặt các công cụ cơ bản. gettext-base (envsubst) — dùng để sinh
# feature_store.yaml thật từ REDIS_PASSWORD env var lúc container start (xem CMD),
# tránh hardcode password (bug thật: hardcode "123" gây Feast auth fail trên AWS
# vì Redis thật dùng password khác).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

# WORKDIR
WORKDIR /app


# Copy file requirements và cài đặt
COPY src/data_pipeline/streaming/consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================
# COPY PROJECT
# =========================================================
COPY artifacts/nlp_models /app/artifacts/nlp_models

# Copy source code
COPY src /app/src

# 🔥 QUAN TRỌNG
ENV PYTHONPATH=/app

#EXPOSE 8010

# Sinh feature_store.yaml thật từ template (envsubst REDIS_PASSWORD) trước khi chạy
CMD ["sh", "-c", "envsubst < /app/src/feature_store/feature_repo/feature_store.yaml > /tmp/fs.yaml && mv /tmp/fs.yaml /app/src/feature_store/feature_repo/feature_store.yaml && exec python src/data_pipeline/streaming/consumer/stream.py"]
