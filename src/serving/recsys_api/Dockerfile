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
COPY src/serving/recsys_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================
# COPY PROJECT
# =========================================================
COPY artifacts/recsys_models /app/artifacts/recsys_models

# Copy source code API vào container
#COPY src/serving/recsys_api /app/src/serving/recsys_api
COPY src /app/src


RUN useradd --no-create-home --system --uid 1001 app \
    && chown -R app:app /app
USER app

EXPOSE 8001

# Sinh feature_store.yaml thật từ template (envsubst REDIS_PASSWORD) trước khi chạy
CMD ["sh", "-c", "envsubst < /app/src/feature_store/feature_repo/feature_store.yaml > /tmp/fs.yaml && mv /tmp/fs.yaml /app/src/feature_store/feature_repo/feature_store.yaml && exec uvicorn src.serving.recsys_api.main:app --host 0.0.0.0 --port 8001"]
