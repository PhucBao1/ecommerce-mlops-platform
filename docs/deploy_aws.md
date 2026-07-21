# AWS Deployment Guide

> Deploy AI data platform lên AWS cho demo/interview. 3 scenarios từ free đến full stack.

---

## Mapping: Local Stack → AWS Services

| Component | Local (hiện tại) | AWS Equivalent | Ghi chú |
|---|---|---|---|
| Object storage | MinIO | **Amazon S3** | Free 5GB, sau đó $0.023/GB |
| Data lake query | Spark local | **Amazon Athena** | Serverless SQL trên S3/Iceberg, $5/TB scanned |
| ETL Spark | Spark local | **AWS Glue** (Spark managed) | $0.44/DPU-hr, chỉ tính khi job chạy |
| Data catalog | Hive metastore | **AWS Glue Data Catalog** | Free 1M objects |
| Kafka | Local Kafka (1 broker) | **Self-hosted Kafka trên EC2** (3-broker KRaft) | Cùng docker-compose, EC2 t3.medium ~$0.04/hr. Rẻ hơn MSK 10x cho workload này |
| Kafka managed alt | — | **Amazon MSK Serverless** | $0.75/hr — chỉ dùng khi cần fully managed + multi-AZ auto |
| Streaming alt | — | **Kinesis Data Streams** | $0.015/shard-hr — AWS-native nếu không muốn ops Kafka |
| Airflow | Airflow local | **Amazon MWAA** (Managed Airflow) | ~$0.49/hr (đắt), dùng Step Functions rẻ hơn |
| MLflow | MLflow local | **Amazon SageMaker** (full ML platform) | Thay cả tracking + registry + serving |
| Vector search | FAISS in-memory | **Amazon OpenSearch** (kNN plugin) | t3.small.search $0.036/hr |
| Feature Store | Feast | **SageMaker Feature Store** | Online + offline store |
| PostgreSQL | PostgreSQL local | **Amazon RDS** hoặc **Aurora Serverless v2** | RDS t3.micro free 12 tháng |
| Redis | Redis local | **Amazon ElastiCache for Redis** | t3.micro $0.017/hr |
| Data warehouse | — | **Amazon Redshift Serverless** | $0.36/RPU-hr, scale to zero khi idle |
| API serving | FastAPI EC2 | **ECS Fargate** hoặc **Lambda + API Gateway** | Lambda free 1M req/mo |
| LLM (Ollama) | Ollama local | **Amazon Bedrock** | Claude Haiku $0.0008/1K tokens, Llama free tier |
| Monitoring | Prometheus + Grafana | **CloudWatch** + Dashboards | 10 dashboards free |
| Distributed tracing | Jaeger | **AWS X-Ray** | 100K traces/mo free |
| Docker registry | Local | **Amazon ECR** | 500MB/mo free |

---

## Scenario 1 — Free Tier + Tí Tiền (~$0.30 cho 4h demo)

Giữ toàn bộ docker-compose, chỉ dùng cloud cho compute + S3. Đơn giản nhất.

```
EC2 t3.medium ($0.04/hr)    ← chạy tất cả via docker-compose (all-in-one)
S3 (free 5GB)               ← DVC artifacts + model weights thay MinIO
ECR (500MB free)            ← Docker images
Amazon Bedrock / Claude API ← thay Ollama, pay-per-token (không cần GPU)
```

**Tất cả chạy trên EC2:** Redis, PostgreSQL, MLflow, FAISS — không managed.

**Chi phí 4h:** EC2 $0.04×4 = $0.16 + Claude Haiku ~$0.10 = **~$0.30**

### Setup

```bash
# 1. Launch EC2 t3.medium (Ubuntu 24.04)
#    Security Group: inbound 8000-8003, 3000, 9090, 22
ssh -i key.pem ubuntu@<EC2-IP>
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu && newgrp docker

# 2. Clone + config
git clone <repo> && cd repo
cp .env.example .env
# Điền: ANTHROPIC_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# 3. DVC pull từ S3 (thay MinIO)
pip install "dvc[s3]"
dvc remote modify minio url s3://your-bucket/dvc-cache
dvc remote modify minio --local access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify minio --local secret_access_key $AWS_SECRET_ACCESS_KEY
dvc pull

# 4. Start services
docker network create my_shared_network
docker compose -f docker-compose.infra.yml up -d
docker compose -f docker-compose.app.yml up -d recsys-api sentiment-api
AGENT_LLM_BACKEND=claude docker compose -f docker-compose.app.yml --profile agent up -d agent-api
```

**Pros:** cực rẻ, setup 30 phút, debug dễ
**Cons:** single EC2 = single point of failure, data mất nếu EC2 crash, không auto-scale

---

## Scenario 2 — Full Stack AWS-Native (~$10–15/ngày)

Thay toàn bộ local services bằng AWS-managed. Architecture giống production thật tại Shopee/Grab.

```
Compute:
  ECS Fargate             ← recsys-api (0.5vCPU/1GB) + sentiment-api (1vCPU/2GB)
                            agent-api (0.5vCPU/1GB) + mlflow-tracking (0.25vCPU/0.5GB)
  ALB                     ← load balancer, health checks

Storage & Query:
  S3                      ← data lake (free tier) + DVC artifacts
  Glue Data Catalog       ← Iceberg metadata, thay Hive metastore (free 1M objects)
  Athena                  ← serverless SQL query Iceberg ($5/TB scanned, ~free cho demo)
  Redshift Serverless     ← analytics warehouse, Gold Mart dbt models ($0.36/RPU-hr)

Streaming:
  Self-hosted Kafka 3-broker trên EC2 t3.medium ($0.04/hr × 1 shared)
  (hoặc MSK Serverless $0.75/hr nếu muốn zero-ops, hoặc Kinesis $0.015/shard-hr)

ML Platform:
  SageMaker Experiments   ← thay MLflow tracking
  SageMaker Model Registry← model versioning + stage promotion
  SageMaker Endpoints     ← model serving (ml.t3.medium $0.05/hr)
  Amazon Bedrock          ← thay Ollama, Claude Haiku pay-per-token

Feature Store:
  SageMaker Feature Store ← thay Feast (online: $0.0025/100 reads)

Search:
  OpenSearch t3.small     ← thay FAISS in-memory, kNN plugin ($0.036/hr)

Database & Cache:
  Aurora Serverless v2    ← PostgreSQL auto-scale ($0.12/ACU-hr minimum)
  ElastiCache t3.micro    ← Redis ($0.017/hr)

Monitoring:
  CloudWatch + X-Ray      ← thay Prometheus/Grafana/Jaeger (mostly free)
  ECR                     ← Docker images ($0.10/GB/mo)
```

### Chi phí 1 ngày

| Service | Chi phí |
|---|---|
| ECS Fargate (3 tasks) | ~$2.30 |
| Aurora Serverless v2 | ~$2.88 |
| MSK Serverless | ~$1.80 |
| OpenSearch t3.small | ~$0.86 |
| ElastiCache t3.micro | ~$0.41 |
| SageMaker endpoint | ~$1.20 |
| ALB | ~$0.19 |
| **Tổng** | **~$10–15/ngày** |

### Setup

```bash
# Thêm modules Glue/Athena/SageMaker/MSK vào terraform/phase1/
cd infrastructure && terraform apply

# Build + push images lên ECR
for svc in recsys_api sentiment_api agent_api; do
  docker build -f docker/Dockerfile.$svc -t <ECR_URI>/$svc:latest .
  docker push <ECR_URI>/$svc:latest
done

# Deploy
aws ecs update-service --cluster recsys-cluster --service recsys-api --force-new-deployment
aws ecs update-service --cluster recsys-cluster --service sentiment-api --force-new-deployment
aws ecs update-service --cluster recsys-cluster --service agent-api --force-new-deployment
```

**Khi nào dùng:** interview Shopee/Grab/VNG level. Bật 1 ngày trước, demo, `terraform destroy` → tổng ~$10-15.

### Key talking points

- "Glue Data Catalog + Athena thay Hive metastore — serverless, pay-per-query, không cần Spark cluster cho ad-hoc"
- "SageMaker thay MLflow + custom training + serving — single ML platform, built-in model registry với approval gates"
- "Bedrock thay Ollama — không cần GPU instance, pay-per-token, switch model không rebuild"
- "MSK thay local Kafka — fully managed, multi-AZ replication, zero ops"
- "Redshift Serverless scale to zero khi idle — không tốn tiền ngoài giờ làm việc"

---

## Scenario 3 — Balanced (~$2.20/ngày) ⭐ Recommended cho fresher

EC2 + RDS + S3/Athena. Redis/Kafka vẫn tự host trên EC2 — rẻ hơn ElastiCache/MSK đáng kể.

```
EC2 t3.large ($0.083/hr)         ← APIs + Redis + Kafka via docker-compose
RDS t3.micro (free tier 12mo)    ← PostgreSQL managed, không mất data khi EC2 restart
S3 + Glue Data Catalog           ← data lake + metadata (free tier)
Athena                           ← ad-hoc query Iceberg ($5/TB, ~free cho demo)
ECR                              ← Docker images
GitHub Actions OIDC + SSM        ← CI/CD tự động, không cần SSH key
Amazon Bedrock / Claude API      ← LLM, pay-per-token, không cần GPU
```

Redis và Kafka chạy Docker trên cùng EC2 — tiết kiệm ~$170/tháng so với ElastiCache + MSK.

### Chi phí 1 ngày

| Service | Chi phí |
|---|---|
| EC2 t3.large | $2.00 |
| RDS db.t3.micro | **free** (12 tháng đầu) |
| S3 + ECR | ~$0.01 |
| Secrets Manager | ~$0.01 |
| **Tổng** | **~$2.02/ngày** |

4h demo: ~$0.33

### Setup

```bash
# 1. Provision với Terraform (từ terraform/phase1/)
cd terraform/phase1
export TF_VAR_db_password="<CHANGE_ME>"
terraform apply

# 2. Copy outputs → GitHub Variables (cho CI/CD)
terraform output deploy_instructions
# → set AWS_ROLE_ARN, ECR_REGISTRY, EC2_INSTANCE_ID, AWS_REGION trên GitHub

# 3. Push code → GitHub Actions tự build + deploy qua SSM
git push origin main
# → build images → push ECR → SSM RunCommand → docker compose up

# Verify
EC2_IP=$(terraform output -raw ec2_public_ip)
curl http://$EC2_IP:8001/health
curl http://$EC2_IP:8003/health
```

> EC2 tự boot cấu hình qua `user_data`: clone repo → fetch secrets từ Secrets Manager → ECR login → start compose. Không cần SSH vào tay.

### Tại sao Scenario 3 là best cho fresher portfolio

1. **EC2 + docker-compose** = dễ debug, không cần biết ECS/Fargate, setup dưới 1h
2. **RDS** = managed PostgreSQL → data không mất khi EC2 restart, nói được "biết dùng managed services"
3. **OIDC + SSM** = zero long-lived credentials, audit trail qua CloudTrail — điểm enterprise rõ ràng
4. **Athena** = câu hỏi hay gặp "tại sao Athena?" → "serverless, $5/TB, không cần Spark cluster chỉ để query"
5. **Bedrock** = "pay-per-token, không cần GPU, switch từ Ollama chỉ cần đổi env var"
6. **Cost** = ~$2/ngày → affordable để bật lên demo cho recruiter, không lo bill

---

## So sánh nhanh

| Component | Scenario 1 (free+) | Scenario 2 (full AWS) | Scenario 3 (balanced) |
|---|---|---|---|
| **Compute** | EC2 t3.medium | ECS Fargate | EC2 t3.large |
| **PostgreSQL** | trên EC2 | Aurora Serverless v2 | RDS t3.micro free |
| **Redis** | trên EC2 | ElastiCache r6g | trên EC2 (Docker) |
| **Data lake** | S3 only | S3 + Athena + Glue | S3 + Athena |
| **ETL** | — | AWS Glue | — |
| **Streaming** | — | MSK Serverless | — |
| **ML platform** | MLflow local | SageMaker full | MLflow local |
| **Vector search** | FAISS local | OpenSearch kNN | FAISS local |
| **Analytics DW** | — | Redshift Serverless | Athena |
| **LLM** | Claude API | Amazon Bedrock | Claude API |
| **Monitoring** | CloudWatch basic | CloudWatch + X-Ray | CloudWatch basic |
| **Load balancer** | — | ALB | ALB |
| **Chi phí 4h** | **~$0.30** | **~$2.50** | **~$0.33** |
| **Chi phí 1 ngày** | **~$1.00** | **~$10–15** | **~$2.02** |
| **Setup effort** | ⭐ 30 phút | ⭐⭐⭐⭐⭐ vài giờ | ⭐⭐ 1h |
| **Interview level** | entry | Shopee/Grab senior | mid-level |

---

## Cleanup — Tắt sau demo

```bash
# Scenario 1: terminate EC2
aws ec2 terminate-instances --instance-ids <id>

# Scenario 2 & 3: destroy everything
cd terraform/phase1 && terraform destroy
# Kiểm tra console sau: S3 buckets + ECR repos không auto-delete, cần xóa thủ công nếu có data
```

> **Lưu ý:** Terraform destroy không xóa S3 buckets có data (safety guard). Xóa thủ công trên console nếu muốn tránh storage cost.
