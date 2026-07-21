# File: docker/Dockerfile.api
FROM python:3.10-slim

# Cài đặt các công cụ cơ bản
#RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy file requirements và cài đặt
COPY src/serving/nlp_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#4. Copy thư mục model đã train vào Docker
# Đường dẫn bên trái là ở máy bạn, bên phải là ở trong Docker
COPY ./artifacts/nlp_models/phobert/version_001 /app/artifacts/nlp_models/phobert/version_001

# Copy source code API vào container
COPY src/serving/nlp_api/phobert_api.py /app/phobert_api.py

# 6. Set biến môi trường để code api.py biết đường dẫn model nằm ở đâu trong Docker
ENV MODEL_PATH="/app/artifacts/nlp_models/phobert/version_001"
ENV MODEL_VERSION="v1.0.0"

ENV HF_HOME="/app/model_cache"

RUN mkdir -p /app/model_cache \
    && useradd --no-create-home --system --uid 1001 app \
    && chown -R app:app /app
USER app

# Mở port 8000 để Spark có thể gọi vào
EXPOSE 8000

# Lệnh khởi động server khi container chạy
CMD ["uvicorn", "phobert_api:app", "--host", "0.0.0.0", "--port", "8000"]
