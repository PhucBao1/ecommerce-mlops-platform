#!/bin/bash
set -euxo pipefail
exec > /var/log/user-data.log 2>&1

# ─── 1. Docker (DLAMI có thể đã có sẵn — cài lại vô hại, apt bỏ qua nếu đã cài) ───
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

# ─── 2. NVIDIA Container Toolkit — để `docker run --gpus all` thấy được GPU ───
# --batch --yes: user_data/cloud-init chạy không có TTY, gpg mặc định cố mở
# /dev/tty (kể cả cho --dearmor không cần tương tác) và fail dưới set -e —
# script dừng ngay đây, mọi bước sau (cài NVIDIA toolkit, tải model, chạy vLLM)
# KHÔNG BAO GIỜ chạy. Bug thật bắt được: cả 2 GPU replica đứng im ở đúng dòng này.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit awscli
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# ─── 3. Kéo model từ S3 (KHÔNG tải từ HuggingFace lúc chạy — tránh phụ thuộc
# mạng ra ngoài VPC + rate-limit khi 2 instance cùng tải song song). Model PHẢI
# được upload sẵn 1 lần trước đó (xem terraform/phase2/README.md) ───
mkdir -p /opt/models/model
aws s3 sync "s3://${s3_bucket}/${model_s3_prefix}/" /opt/models/model/ --region "${aws_region}"

# ─── 4. DCGM Exporter — GPU utilization/VRAM/temperature → Prometheus ───
docker run -d \
  --name dcgm-exporter \
  --restart unless-stopped \
  --gpus all \
  -p 9400:9400 \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04

# ─── 5. vLLM — image chính thức (đóng gói đúng CUDA runtime khớp vLLM, tránh lệch
# version với driver DLAMI như khi tự pip install trên host). Docker healthcheck
# thay cho readiness/liveness probe (không có K8s ở quy mô demo này) ───
#
# ⚠️ --enable-auto-tool-choice + --tool-call-parser hermes là BẮT BUỘC, đừng gỡ:
# thiếu 2 flag này, server OpenAI-compatible của vLLM KHÔNG parse tool call —
# model trả tool call về dưới dạng text thô trong `content`, field `tool_calls`
# rỗng, nên LangChain (ChatOpenAI trong graph.py) không thấy tool nào được gọi
# và toàn bộ 5 tool của agent chết IM LẶNG (không lỗi, chỉ là agent trả lời chay).
# Đây đúng là triệu chứng <tool_call> lộ ra text thô đang phải vá bằng regex ở
# Ollama — trên vLLM thì fix được tận gốc ở tầng server. Parser `hermes` là đúng
# format tool-call của họ Qwen2.5 (xem docs vLLM: tool_calling supported models).
# health-start-period=900s: 180s là quá ngắn thật — đo được trên T4 +
# Qwen2.5-7B-AWQ: load weight từ EBS gp3 mất ~12 phút (không phải GPU chậm,
# mà disk throughput), cộng torch.compile warmup ~1 phút nữa. Container không
# chết trong lúc đó (chỉ health status báo "unhealthy" sai lệch), nhưng nếu
# có logic tự động dựa vào health status (vd ALB target group) thì sẽ bị đá
# ra sớm oan.
#
# ⚠️ KHÔNG chèn comment vào giữa các dòng nối \ bên dưới — bash nuốt comment
# làm literal argument, làm gãy lệnh im lặng (bug thật đã gặp: "docker run
# requires at least 1 argument" trên replica tạo lại lần 2).
#
# --enable-prefix-caching + --enable-chunked-prefill: đo GPU util lúc load
# test (mục 10, BENCHMARK_RESULTS.md) cho thấy cả 2 GPU đã 100% suốt bài test
# — trần RPS hiện tại là compute, KHÔNG phải memory (đã loại trừ hướng tăng
# gpu-memory-utilization/giảm max-model-len vì lý do đó). 2 flag này giảm
# compute PHẢI làm lại chứ không tăng dung lượng: prefix-caching bỏ qua việc
# tính lại system prompt (system_v1.txt) vốn lặp lại giống hệt mọi request;
# chunked-prefill cho phép request dài (prefill) xen kẽ với request đang
# decode trong cùng batch step, tránh 1 request dài chặn cả batch. KHÔNG
# đổi --gpu-memory-utilization/--max-model-len — memory không phải nút thắt,
# và giảm max-model-len có rủi ro cắt cụt context thật (system prompt + tool
# schema + RAG) của agent nhiều lượt hội thoại.
docker run -d \
  --name vllm \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  -v /opt/models/model:/model \
  --health-cmd="curl -f http://localhost:8000/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=900s \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name "${vllm_model}" \
%{ if vllm_quantization != "" ~}
  --quantization ${vllm_quantization} \
%{ endif ~}
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --enable-chunked-prefill

touch /home/ubuntu/bootstrap_done
