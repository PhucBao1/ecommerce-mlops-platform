# Terraform Phase 2 — vLLM GPU replicas (Ollama CPU → vLLM GPU benchmark)

Tạo 2× EC2 Spot `g4dn.xlarge` (Deep Learning AMI — driver NVIDIA sẵn), mỗi cái
chạy vLLM (`vllm/vllm-openai` — Docker, không phải bare-metal pip install) +
DCGM exporter (GPU metrics), cùng VPC/subnet với Phase 1 — **KHÔNG chung
Terraform state** với Phase 1 (đọc output Phase 1 qua `terraform_remote_state`,
chỉ read-only).

**Chỉ chạy sau khi:**
1. Phase 1 (`terraform/phase1/`) đã `terraform apply` xong (cần state file tồn tại).
2. AWS account đã **upgrade Paid plan** (GPU EC2 thường bị Free plan chặn/hạn chế).
3. **Model đã upload sẵn lên S3** (bước tay bắt buộc, xem dưới) — KHÔNG tải trực
   tiếp từ HuggingFace lúc container start (tránh phụ thuộc mạng ra ngoài VPC +
   rate-limit khi 2 instance cùng tải song song).

## Bước 0 — Upload model lên S3 (làm 1 lần, TRƯỚC khi apply)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir ./qwen2.5-7b-instruct-awq

# Lấy tên bucket từ Phase 1
BUCKET=$(cd ../phase1 && terraform output -raw s3_bucket)

aws s3 sync ./qwen2.5-7b-instruct-awq "s3://$BUCKET/models/qwen2.5-7b-instruct-awq/"
```

**Vì sao 7B AWQ (không phải 3B FP16 như bản thảo đầu):** `graph.py` đã note sẵn
Qwen2.5:3b không đủ tin cậy cho tool-calling (Router phải ép chọn tool tường
minh thay vì để LLM tự quyết). 7B AWQ (~4-5GB weights) thực ra **nhẹ VRAM hơn**
3B FP16 (~6GB) nhờ quantize INT4, trong khi chất lượng cao hơn hẳn — không còn
đánh đổi giữa "nhẹ" và "tốt" ở trường hợp này.

(Nếu đổi sang model khác, đổi `model_s3_prefix` + `vllm_model` trong
`terraform.tfvars` cho khớp. Nếu model không phải AWQ/GPTQ, set
`vllm_quantization = ""`.)

## Deploy

```bash
cd terraform/phase2
cp terraform.tfvars.example terraform.tfvars
# Điền: my_ip, key_pair_name

terraform init
terraform plan
terraform apply
```

Đợi ~5-10 phút (`user_data` cài Docker + nvidia-container-toolkit + `aws s3
sync` model + pull image `vllm/vllm-openai`). Verify:

```bash
terraform output ssh_commands   # copy lệnh SSH cho Replica B/C

ssh -i your-key.pem ubuntu@<replica-B-ip>
docker ps                        # thấy 2 container: vllm, dcgm-exporter
docker inspect --format='{{.State.Health.Status}}' vllm   # kỳ vọng "healthy"
docker logs vllm --tail 50
```

## Nối vào nginx + agent-api (Instance A)

Lấy config nginx sẵn từ output, dán thẳng — không cần gõ tay private IP:

```bash
terraform output -raw nginx_upstream_snippet
```

Các bước còn lại (cài nginx trên Instance A, sửa `.env.aws` cho `agent-api` trỏ
`AGENT_LLM_BACKEND=vllm`, loadtest so sánh) — xem chi tiết trong
**[`AWS_VLLM_DEPLOY.md`](../../AWS_VLLM_DEPLOY.md)** (không lặp lại ở đây để tránh
2 nơi cùng nói 1 việc, dễ lệch nhau khi sửa sau này).

## Monitoring — GPU + vLLM metrics vào Grafana có sẵn

Sau khi apply xong, nối Prometheus (Phase 1) vào scrape 2 replica:

```bash
terraform output vllm_replica_private_ips
```

SSH vào Instance A, sửa `monitoring/prometheus.yml`, bỏ comment 2 job
`vllm-gpu-metrics`/`dcgm-exporter` ở cuối file, điền đúng private IP vừa lấy,
rồi reload (không cần restart container):

```bash
curl -X POST http://localhost:9090/-/reload
```

Dashboard **`Phase 2 — GPU / vLLM`** (`monitoring/grafana/dashboards/gpu_vllm.json`)
đã có sẵn, tự nhận data ngay sau khi scrape target lên — GPU utilization/VRAM/
temperature (DCGM) + request running/waiting + KV-cache usage + time-to-first-
token p50/p95 (vLLM `/metrics` — đây là con số so sánh trực tiếp với baseline
Ollama CPU cho CV).

## Dọn dẹp

```bash
terraform destroy
```

**QUAN TRỌNG** — terminate ngay sau khi lấy đủ số liệu benchmark, Spot vẫn tính
phí theo giờ dù không dùng.

## Gotcha đã biết

- **Tên AMI DLAMI đổi theo thời gian** — nếu `terraform apply` báo không tìm thấy
  AMI khớp filter trong `vllm.tf`, vào AWS Console → EC2 → Launch Instance → AMI
  → search "Deep Learning AMI GPU PyTorch" → copy đúng pattern tên hiện hành.
- **DCGM exporter image (`nvcr.io/nvidia/k8s/dcgm-exporter`) trên NGC registry**
  — thường pull public không cần login, nhưng nếu gặp lỗi auth: tạo tài khoản
  NGC miễn phí rồi `docker login nvcr.io` trước khi user_data chạy lại (hoặc
  SSH vào chạy tay `docker login` rồi re-run lệnh `docker run` trong
  `user_data.sh.tpl`).
- **Model 3B upload lên S3 lần đầu ~6GB** — sync + tải lần đầu trên EC2 mất vài
  phút, `docker inspect` container `vllm` có thể báo "starting" (chưa qua
  `--health-start-period=180s`) một lúc trước khi "healthy".
- **`vllm:*` metric names có thể đổi theo version image `vllm-openai:latest`**
  — nếu dashboard "No data" dù đã scrape đúng target, `curl
  http://<replica-ip>:8000/metrics` xem tên metric thật đang expose, sửa lại
  query trong `gpu_vllm.json` cho khớp.

## Đã lưu lại tham khảo — bản đầu tiên (bare-metal, không Docker)

`vllm.tf` còn giữ nguyên văn local
`vllm_user_data_v1_bare_metal_reference_only` (pip install vllm + systemd unit
tay, không dùng Docker) — KHÔNG còn được dùng, chỉ giữ lại để đối chiếu/học lại
sau này (thấy rõ khác biệt so với bản Docker hiện tại).
