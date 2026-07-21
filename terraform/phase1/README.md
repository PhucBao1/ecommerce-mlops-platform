# Terraform Phase 1 — Full platform (trừ GPU/vLLM) trên AWS

IaC cho AWS deployment. `terraform apply` một lần → EC2, S3, RDS, ECR, IAM, VPC lên
đủ để chạy toàn bộ platform (serving + Kafka + Spark/Iceberg + Airflow + dbt +
Feast + monitoring) qua docker-compose có sẵn. GPU/vLLM/EKS/MSK/MWAA nằm ở
**Phase 2** — thư mục Terraform HOÀN TOÀN RIÊNG (state riêng), chỉ thực hiện sau
khi upgrade AWS account sang Paid plan (Free plan chặn các service đó).

---

## Kiến trúc AWS

```
Internet
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│  VPC  10.0.0.0/16                                                 │
│                                                                   │
│  ┌────────────────────────────────────────────────┐               │
│  │  Public Subnet  10.0.1.0/24                    │               │
│  │                                                  │               │
│  │  EC2 m7i-flex.large (2 vCPU/8GB — free-tier-      │               │
│  │  eligible lớn nhất, thay m5.2xlarge dự kiến ban đầu)│               │
│  │  docker-compose:                                │               │
│  │  ├── recsys-api        :8001                    │               │
│  │  ├── sentiment-api      :8000  (PhoBERT ONNX)   │               │
│  │  ├── agent-api          :8003  (Ollama tự host) │               │
│  │  ├── recsys-consumer/producer  (Kafka)          │               │
│  │  ├── Redis              :6379                    │               │
│  │  ├── Kafka (broker)     :9092                    │               │
│  │  ├── Qdrant             :6333                    │               │
│  │  ├── Spark+Iceberg      :8888/:4040               │               │
│  │  ├── Airflow webserver  :8080                    │               │
│  │  ├── dbt (target prod_airflow → spark-dbt-thrift)│               │
│  │  ├── Feast (local provider, baked vào recsys-api,│               │
│  │  │           online store = Redis db=1)          │               │
│  │  ├── MLflow             :5000                    │               │
│  │  ├── Prometheus/Grafana :9090/:3000               │               │
│  │  └── Jaeger             :16686                    │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                   │
│  ┌──────────────────────────────────────────────────┐               │
│  │  Private Subnet  10.0.2.0/24                     │               │
│  │  RDS PostgreSQL (db.t3.micro) — CHỈ Iceberg catalog│               │
│  │  port 5432 — EC2 only, không public               │               │
│  │  (MLflow + Airflow vẫn dùng Postgres tự host riêng│               │
│  │   trong docker-compose, không dùng RDS này)        │               │
│  └──────────────────────────────────────────────────┘               │
└───────────────────────────────────────────────────────────────────┘

S3 bucket: ecommerce-mlops-lakehouse
  └── warehouse/          ← Iceberg tables (thay MinIO)

ECR: 8 repositories (bản gốc chỉ có 3, thiếu recsys-consumer/producer + spark/airflow/dbt)
  ├── recsys-api
  ├── sentiment-api   (đổi tên từ "phobert-api" — khớp đúng tên service compose)
  ├── agent-api
  ├── recsys-consumer
  ├── recsys-producer
  ├── spark            (dùng chung cho spark-iceberg + spark-thrift)
  ├── airflow          (dùng chung cho mọi service airflow-*)
  └── dbt

Build TẤT CẢ 8 image ở máy local, EC2 chỉ pull (không build — EC2 2 vCPU/30GB
disk, đã gặp lỗi hết ổ đĩa khi thử build trực tiếp trên đó).

Secrets Manager: ecommerce-mlops/app-secrets
  └── passwords, GITHUB_PAT (clone repo private), hostname nội bộ (redis/broker/mlflow)
```

---

## Chi phí ước tính

| Service | Loại | Giá/tháng |
|---|---|---|
| EC2 `m7i-flex.large` | Compute | ~$50/tháng (~$0.0864/h On-Demand — chỉ bật khi cần demo, không để chạy 24/7) |
| RDS db.t3.micro | PostgreSQL | ~$13 |
| S3 | Storage (<10GB) | ~$0.23 |
| ECR | Image storage (8 repo) | Vài $ (vượt 500MB free vì image nặng, nhưng lifecycle policy giữ tối đa 5 tag/repo) |
| Secrets Manager | 1 secret | ~$0.40 |
| Data transfer | Nhỏ | ~$1 |
| **Thực tế (bật vài giờ/lần demo)** | | **~$0.15-0.25/giờ** |

**⚠️ Thực tế deploy: `m5.2xlarge` dự kiến ban đầu KHÔNG dùng được** — AWS Free
plan chặn hẳn instance type không "free-tier-eligible" (lỗi
`InvalidParameterCombination: not eligible for Free Tier`). Danh sách free-tier-
eligible thật (kiểm tra qua `aws ec2 describe-instance-types --filters
"Name=free-tier-eligible,Values=true"`) chỉ có `t2/t3/t4g.micro/small`,
`c7i-flex.large`, `m7i-flex.large` — chọn **`m7i-flex.large` (2 vCPU/8GB)**, lớn
nhất trong danh sách. Với RAM này, KHÔNG chạy được Airflow+Spark+Kafka+dbt+
monitoring+serving đồng thời thoải mái như tính toán ban đầu (~20-25GB cần) —
chấp nhận đánh đổi (build ở local để giảm tải CPU/disk cho EC2, xem bước 8), nếu
cần chạy mượt hơn thì upgrade Paid plan rồi đổi `instance_type` sang máy to hơn.

**Tại sao không dùng ElastiCache và MSK:**
- ElastiCache cache.t3.micro: +$12/tháng — Redis chạy Docker trên EC2 là đủ
- MSK (1 broker): +$150/tháng — Kafka chạy Docker trên EC2 là đủ
- Tổng tiết kiệm: ~$162/tháng so với fully managed

**⚠️ Nhớ terminate/stop EC2 sau khi chụp hình xong** — đây là On-Demand (không
phải Spot), vẫn tính tiền dù không hoạt động.

---

## Files

| File | Tạo resource |
|---|---|
| `provider.tf` | AWS provider, region |
| `vpc.tf` | VPC, public subnet, private subnet, internet gateway, route table |
| `ec2.tf` | EC2 instance, security group (SSH giới hạn IP), user_data (bootstrap only) |
| `rds.tf` | RDS PostgreSQL (chỉ Iceberg catalog), DB subnet group, security group |
| `s3.tf` | S3 bucket + versioning |
| `ecr.tf` | 8 ECR repos (5 app + spark/airflow/dbt) + lifecycle policy (giữ 5 image mới nhất/repo) |
| `iam.tf` | EC2 IAM role (S3 + ECR pull + SSM + Secrets Manager) |
| `github_oidc.tf` | OIDC provider + IAM role cho GitHub Actions, giới hạn đúng repo+branch |
| `secrets.tf` | Secrets Manager — passwords, GITHUB_PAT, hostname nội bộ |
| `variables.tf` | Input variables (bao gồm `my_ip`, `key_pair_name`, `github_pat` mới) |
| `outputs.tf` | Outputs: IP, endpoints, ECR URLs, GitHub Actions role ARN, VPC/subnet/SG ID (Phase 2 tái dùng) |

---

## Deploy lần đầu

### 1. Prerequisites

```bash
# Cài Terraform >= 1.5
terraform -version

# Cài AWS CLI và configure (Paid hoặc Free account plan đều được — Phase 1
# không đụng service nào bị Free plan chặn)
aws configure

# Tạo EC2 Key Pair trước trong AWS Console (EC2 → Key Pairs → Create) — Terraform
# không tự tạo được vì file .pem cần tải về đúng lúc tạo, không an toàn để tự động hóa

# Tạo GitHub PAT (Settings → Developer settings → Personal access tokens →
# Fine-grained, scope "Contents: Read-only" trên đúng repo này) — dùng để EC2
# clone code (repo đang PRIVATE, HTTPS clone trơn không auth sẽ lỗi)
```

### 2. Init và apply

```bash
cd terraform/phase1
cp terraform.tfvars.example terraform.tfvars
# Điền: my_ip (curl -s ifconfig.me), key_pair_name

terraform init

export TF_VAR_db_password="your_secure_password"   # gitleaks:allow
export TF_VAR_github_pat="ghp_..."                  # KHÔNG hardcode vào .tfvars

terraform plan    # xem trước sẽ tạo gì
terraform apply   # ~5 phút
```

### 3. Lấy outputs và setup GitHub

```bash
terraform output deploy_instructions

# GitHub → Settings → Secrets and variables → Actions → Variables (KHÔNG phải
# Secrets — các giá trị này không nhạy cảm):
#   AWS_ROLE_ARN      = <github_actions_role_arn>
#   ECR_REGISTRY      = <ecr_registry>
#   EC2_INSTANCE_ID   = <ec2_instance_id>
#   AWS_REGION        = ap-southeast-1
```

### 4. Cập nhật API key thật (nếu muốn thử AGENT_LLM_BACKEND=claude sau này)

```bash
aws secretsmanager update-secret \
  --secret-id ecommerce-mlops/app-secrets \
  --secret-string "$(aws secretsmanager get-secret-value \
    --secret-id ecommerce-mlops/app-secrets \
    --query SecretString --output text \
  | jq '.ANTHROPIC_API_KEY = "sk-ant-..."')"
```

### 5. Đợi bootstrap xong, rsync artifacts + kb-docs (BẮT BUỘC — hay bị quên)

`artifacts/` (~1.15GB: model weights, ONNX, item_lookup.parquet...) và `kb-docs/`
(588K) **không nằm trong git** (đã kiểm tra: `git ls-files` = 0 cho cả 2 thư mục;
DVC cũng chỉ cover 2 file nhỏ trong `artifacts/recsys_models/`) — `user_data` chỉ
clone code, KHÔNG có 2 thư mục này. Thiếu bước dưới thì `recsys-api`/
`sentiment-api`/`agent-api` không load được model/KB docs.

```bash
EC2_IP=$(terraform output -raw ec2_public_ip)

# Đợi bootstrap xong (~2-3 phút), kiểm tra:
ssh -i your-key.pem ubuntu@$EC2_IP "test -f ~/bootstrap_done && echo OK"

# Rsync từ máy LOCAL lên EC2 (không thể làm ngược lại — EC2 không có access vào
# máy local, đây luôn là bước push từ local)
rsync -avz --progress -e "ssh -i your-key.pem" \
  ../../artifacts/ ubuntu@$EC2_IP:~/repo/artifacts/
rsync -avz --progress -e "ssh -i your-key.pem" \
  ../../kb-docs/ ubuntu@$EC2_IP:~/repo/kb-docs/
```

### 6. Migrate raw bronze data lên S3 (BẮT BUỘC — DAG không tự crawl lại)

`airflow/dags/ecommerce_pipeline.py` (dag_id thật bên trong là
**`daily_bronze_to_silver_job_2`** — KHÁC tên file, dùng đúng ID này khi
trigger) có task crawl **bị comment out** (dòng 44-56) — DAG này giả định
bronze ĐÃ CÓ SẴN, không tự crawl fresh từ Tiki. Nếu bỏ qua bước này, trigger DAG
sẽ chạy `bronze_to_silver` trên catalog RDS rỗng → lỗi.

`bronze_to_silver.py` đọc đúng 2 path (đọc kỹ code, KHÔNG phải `raw_products`/
`raw_comments` như tên gọi trong `validate_data_quality()` — đó chỉ là nhãn log):
```python
f"s3a://warehouse/bronze/products/{execution_date}/products_*.parquet"
f"s3a://warehouse/bronze/comments/{execution_date}/comments_*.parquet"
```

**Gotcha thật:** data local có ở 2 ngày KHÁC NHAU — `bronze/products/2026-05-29/`
và `bronze/comments/2026-05-30/`. DAG gọi cả 2 hàm với CÙNG 1 `execution_date`
→ không có ngày nào chạy đúng cho cả 2 cùng lúc nếu giữ nguyên tên thư mục gốc.
**Cách xử lý (không sửa code Python):** tự chọn 1 ngày dùng chung khi upload:

```bash
# Từ máy LOCAL (đã cấu hình AWS credentials của bạn, không phải trên EC2)
BUCKET=$(cd .. && terraform output -raw s3_bucket)
DATE="2026-05-29"   # chọn 1 ngày dùng chung cho cả products lẫn comments

aws s3 sync ../../data/warehouse/warehouse/bronze/products/2026-05-29/ \
  "s3://$BUCKET/warehouse/bronze/products/$DATE/"
aws s3 sync ../../data/warehouse/warehouse/bronze/comments/2026-05-30/ \
  "s3://$BUCKET/warehouse/bronze/comments/$DATE/"
```

Khi trigger DAG (bước 9), phải set **Logical date = `2026-05-29`** (khớp
`$DATE` ở trên) — trigger theo ngày hiện tại (mặc định) sẽ tìm sai thư mục.

(KHÔNG cần migrate `silver/`/`gold/`/bảng Iceberg đã tính sẵn — DAG sẽ tự tạo lại
khi chạy, đúng nghĩa "chạy pipeline thật" thay vì hiển thị dữ liệu cũ.)

### 7. Feast apply (registry.db không nằm trong git, phải sinh lại trên EC2)

```bash
ssh -i your-key.pem ubuntu@$EC2_IP
cd ~/repo/src/feature_store/feature_repo
feast apply   # cần: pip install feast (hoặc chạy trong venv/container tạm)
```

### 8. Build image — build ở máy LOCAL rồi push ECR (KHÔNG build trên EC2)

**Đã thử build trực tiếp trên EC2 lúc deploy thật — gặp lỗi `no space left on
device`** (EBS chỉ 30GB, nhiều image tự cài `torch` full-CUDA nặng ~700MB/cái dù
không có GPU + Spark/Airflow vốn đã nặng). Máy local mạnh hơn nhiều + có sẵn
Docker cache từ lúc dev — build local rồi push lên ECR, EC2 chỉ **pull**, không
**build**, tránh hẳn vấn đề CPU yếu/disk nhỏ.

8 image cần build (5 image "app" + 3 image hạ tầng):

```bash
# Từ máy LOCAL, tại thư mục gốc repo
cd ~/BaoBao/Ecommerce
ECR_REGISTRY=$(cd terraform/phase1 && terraform output -raw ecr_registry)
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

docker build --platform linux/amd64 -f src/serving/recsys_api/Dockerfile -t $ECR_REGISTRY/recsys-api:latest .
docker build --platform linux/amd64 -f src/serving/nlp_api/Dockerfile -t $ECR_REGISTRY/sentiment-api:latest .
docker build --platform linux/amd64 -f src/data_pipeline/streaming/consumer/Dockerfile -t $ECR_REGISTRY/recsys-consumer:latest .
docker build --platform linux/amd64 -f src/data_pipeline/streaming/producer/Dockerfile -t $ECR_REGISTRY/recsys-producer:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.agent_api -t $ECR_REGISTRY/agent-api:latest .
docker build --platform linux/amd64 -f spark/Dockerfile.spark -t $ECR_REGISTRY/spark:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.airflow --build-arg DOCKER_GID=984 -t $ECR_REGISTRY/airflow:latest .
docker build --platform linux/amd64 -f docker/Dockerfile.dbt -t $ECR_REGISTRY/dbt:latest .

for img in recsys-api sentiment-api recsys-consumer recsys-producer agent-api spark airflow dbt; do
  docker push $ECR_REGISTRY/$img:latest
done
```

**Gotcha đã gặp thật:** `sentiment-api`/`recsys-consumer` ban đầu build ra
**10.7GB/10.8GB** vì `requirements.txt` thiếu `--extra-index-url
https://download.pytorch.org/whl/cpu` (tải nhầm bản `torch` full-CUDA dù máy
không GPU) — đã sửa trong code (`src/serving/nlp_api/requirements.txt`,
`src/data_pipeline/streaming/consumer/requirements.txt`), sau khi sửa còn
~4-5GB. Nếu tự thêm service mới có `torch`, nhớ luôn dùng bản `+cpu` cho service
chạy trên EC2 không GPU.

### 9. Trên EC2 — chỉ pull, không build

```bash
ssh -i your-key.pem ubuntu@$EC2_IP
cd ~/repo
export ECR_REGISTRY="<region-account>.dkr.ecr.ap-southeast-1.amazonaws.com"   # lấy từ `terraform output ecr_registry`
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

# CHỈ pull đúng service cần cho AWS — bỏ minio/mc/postgres-catalog/pgadmin/
# metabase (đã thay bằng S3/RDS thật, còn khai báo trong file nhưng không cần chạy)
docker compose --env-file .env.aws -f docker-compose.infra.yml pull broker schema-registry redis qdrant spark-thrift
docker compose --env-file .env.aws -f docker-compose.infra.yml up -d broker schema-registry redis qdrant spark-thrift

docker compose --env-file .env.aws -f docker-compose.app.yml pull
docker compose --env-file .env.aws -f docker-compose.app.yml up -d

docker compose --env-file .env.aws -f docker-compose.batch_dev.yml pull
docker compose --env-file .env.aws -f docker-compose.batch_dev.yml up -d

docker compose --env-file .env.aws -f docker-compose.monitor.yml up -d
```

**Lưu ý:** `ollama` + `agent-api` có `profiles: [agent]` trong
`docker-compose.app.yml` → lệnh `up` ở trên **KHÔNG khởi động 2 service này**
(chủ đích — Ollama CPU không chạy ở Phase 1, đợi Phase 2 trỏ thẳng vLLM). Nếu
sau này muốn bật `agent-api` sớm hơn (khi chưa có Phase 2), thêm
`--profile agent` vào lệnh `up` của `docker-compose.app.yml` (image `agent-api`
đã build+push sẵn ở bước 8 dù chưa dùng ngay).

### 10. Verify deploy

```bash
curl http://$EC2_IP:8001/health   # RecSys API
curl http://$EC2_IP:8000/health   # Sentiment API (PhoBERT)
# agent-api KHÔNG chạy ở Phase 1 (profile "agent", đợi Phase 2 trỏ vLLM) — bỏ qua
# Airflow: http://$EC2_IP:8080 — trigger dag_id "daily_bronze_to_silver_job_2"
#   (file airflow/dags/ecommerce_pipeline.py — DAG "thật" duy nhất được mount,
#   airflow/dags_staging/ và root dags/ KHÔNG mount, ngoài phạm vi Phase 1) —
#   NHỚ set Logical date = 2026-05-29 (khớp bước 6), không dùng ngày mặc định.
#   DAG dùng image SPARK_IMAGE/DBT_IMAGE tự dựng từ biến ECR_REGISTRY (đã set
#   trong docker-compose.batch_dev.yml's environment) — không cần sửa gì thêm.
# Grafana: http://$EC2_IP:3000
# MLflow:  http://$EC2_IP:5000
```

---

## Checklist chụp hình (portfolio)

**Hạ tầng/IaC:**
- [ ] `terraform apply` output
- [ ] AWS Console → EC2 `running`, đúng type `m7i-flex.large`
- [ ] AWS Console → S3 bucket có object thật trong `warehouse/`

**CI/CD:**
- [ ] GitHub Actions run xanh (build 5 image → Trivy scan → push ECR → SSM deploy)
- [ ] ECR repository thấy image tag mới nhất

**Serving APIs:**
- [ ] `curl`/Swagger UI `recsys-api` `/recommend` trả kết quả thật
- [ ] `curl` `agent-api` `/chat/stream` trả lời + tool_calls thật (Ollama CPU)
- [ ] `curl` `sentiment-api` `/predict` (ONNX) trả kết quả thật

**Data pipeline:**
- [ ] Airflow UI — Graph view DAG `daily_bronze_to_silver_job_2` + 1 lần chạy thành công (Logical date = 2026-05-29)
- [ ] Qdrant dashboard (`:6333/dashboard`) có collection thật
- [ ] MLflow UI có experiment/run thật
- [ ] `feast feature-views list` (SSH vào EC2) trả về feature thật cho 1 customer_id

**Monitoring:**
- [ ] Grafana 4 dashboard có traffic thật (không "No data")
- [ ] Prometheus `/targets` toàn bộ `UP`
- [ ] Jaeger UI có trace thật của `agent-api`

**Load test:**
- [ ] Locust nhắm vào `$EC2_IP` thật (không phải localhost)

---

## CI/CD — GitHub Actions

Sau khi setup GitHub Variables xong, pipeline tự động:

```
git push origin main
    │
    ▼
GitHub Actions (.github/workflows/docker_build.yml)
    │
    ├─ Build 5 Docker images (recsys-api, sentiment-api, agent-api,
    │                          recsys-consumer, recsys-producer)
    ├─ Trivy security scan → GitHub Security tab
    ├─ OIDC → assume IAM role (không cần AWS key)
    ├─ Push to ECR (cả 5 image, tag SHA + latest)
    └─ SSM Run Command → EC2:
         git pull --ff-only
         export ECR_REGISTRY=...
         docker compose --env-file .env.aws -f docker-compose.app.yml pull
         docker compose --env-file .env.aws -f docker-compose.app.yml up -d --remove-orphans
         (wait + check status)
```

CI/CD chỉ tự động deploy lại 5 service "app" (đổi code thường xuyên). Spark/
Airflow/dbt không đổi theo mỗi lần push code — build+push thủ công theo bước 8
ở trên (1 lần, hoặc khi nào Dockerfile/dependency của riêng chúng đổi), không
nằm trong CI/CD hot-path.

**Tại sao OIDC thay vì AWS_ACCESS_KEY_ID trong Secrets:**
- Long-lived keys phải rotate thủ công, có thể bị leak
- OIDC: GitHub tự generate JWT per-run, key hết hạn sau ~1 giờ
- IAM role chỉ trust đúng repo + branch (`repo:PhucBao1/Ecommerce:ref:refs/heads/main`
  — bản gốc dùng `:*` cho phép mọi ref, đã siết lại)

**Tại sao SSM thay vì SSH cho CI/CD:**
- Không cần lưu SSH private key trong GitHub Secrets
- EC2 dùng IAM role, SSH (port 22) chỉ mở cho đúng IP cá nhân — CI/CD không cần
  port 22 mở public
- Audit trail: mỗi SSM command log vào CloudTrail

**Vì sao repo cần GITHUB_PAT dù đã dùng OIDC:** OIDC chỉ cấp quyền AWS (ECR/SSM),
không giúp `git clone`/`git pull` một repo GitHub private — đó là 2 hệ thống auth
khác nhau. PAT lưu trong Secrets Manager, `user_data` lấy ra dùng 1 lần lúc clone
rồi bỏ khỏi `.env.aws` (không phải runtime env của app).

---

## Destroy (dọn dẹp)

```bash
terraform destroy
# Confirm với "yes" — cần TF_VAR_db_password/TF_VAR_github_pat vẫn đang set trong shell
```

> **Lưu ý:** S3 bucket chứa data sẽ bị xóa nếu không có `prevent_destroy`. Backup
> data trước khi destroy nếu cần.

---

## Phase 2 (GPU/vLLM, EKS, MSK, MWAA) — KHÔNG nằm trong thư mục này

Thư mục Terraform HOÀN TOÀN RIÊNG (state riêng), chỉ đọc `vpc_id`/`subnet_id`/
`security_group_id` (output ở trên) làm input để đặt cùng VPC — tránh 1 lần
`terraform apply` ở Phase 1 vô tình động vào resource Phase 2 (bị Free-plan chặn
nếu account chưa upgrade Paid). Chi tiết: `AWS_VLLM_DEPLOY.md`.

## Mở rộng sang production

| Nhu cầu | Thay đổi |
|---|---|
| Redis HA | Thêm `elasticache.tf` (cache.t3.micro ~$12/tháng) |
| Kafka managed | Thêm `msk.tf` (MSK ~$150/tháng) |
| Multi-AZ DB | `rds.tf`: đổi `multi_az = true` |
| Auto scaling | Thêm ASG + ALB |
| GPU inference | Xem Phase 2 (`AWS_VLLM_DEPLOY.md`) |
