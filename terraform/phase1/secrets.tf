resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "${var.project_name}/app-secrets"
  description = "All runtime env vars for the MLOps platform (passwords + API keys)"
  # 0 = xóa ngay lập tức khi destroy (không giữ recovery window 7 ngày) — hợp lý
  # cho demo/dev vì secret luôn được Terraform tạo lại từ đầu, không có dữ liệu
  # cần khôi phục. Giá trị > 0 từng gây lỗi "already scheduled for deletion" khi
  # apply lại ngay sau destroy (tên secret bị giữ chỗ suốt recovery window).
  recovery_window_in_days = 0

  tags = { Project = var.project_name }
}

# EC2 user_data fetches this secret và ghi thành 2 chỗ:
#   - GITHUB_PAT được lấy riêng ra để clone repo (private), KHÔNG ghi vào .env.aws
#   - phần còn lại ghi vào /home/ubuntu/repo/.env.aws
#
# Sensitive values (passwords, API keys) go here.
# Non-sensitive config (hostnames, ports) is baked into user_data via Terraform interpolation.
resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id

  secret_string = jsonencode({
    # GitHub PAT (repo scope, read-only đủ) — CHỈ dùng để `git clone` repo private lúc
    # boot, bị xóa khỏi .env.aws (xem ec2.tf user_data bước 4) — không phải runtime env.
    GITHUB_PAT = var.github_pat

    # Passwords — rotate here, EC2 picks up on next restart
    REDIS_PASSWORD    = var.db_password
    POSTGRES_PASSWORD = var.db_password

    # BẮT BUỘC — agent-api fail-CLOSED: thiếu key này thì /admin/* từ chối hết,
    ADMIN_API_KEY = var.admin_api_key

    # Trần chi phí LLM/ngày (USD) — rate limit chặn số REQUEST, cái này chặn TIỀN
    USER_DAILY_BUDGET_USD   = "0.50"
    GLOBAL_DAILY_BUDGET_USD = "5.00"

    # API keys — optional, chỉ cần nếu muốn thử AGENT_LLM_BACKEND=claude sau này
    ANTHROPIC_API_KEY = "REPLACE_ME_BEFORE_APPLY"

    # RDS PostgreSQL — CHỈ thay thế `postgres-catalog` (Iceberg catalog). MLflow và
    # Airflow vẫn dùng Postgres tự host riêng của chúng (docker-compose.monitor.yml,
    # docker-compose.batch_dev.yml) — RDS multi-database cho cả 3 mục đích cần thêm
    # provider/bootstrap phức tạp hơn, không đáng cho quy mô demo này.
    POSTGRES_ICEBERG_HOST     = aws_db_instance.postgres.address
    POSTGRES_ICEBERG_PORT     = "5432"
    POSTGRES_ICEBERG_USER     = "iceberg_admin"
    POSTGRES_ICEBERG_PASSWORD = var.db_password
    POSTGRES_ICEBERG_DB       = "iceberg_metadata"

    POSTGRES_MLFLOW_HOST     = aws_db_instance.postgres.address
    POSTGRES_MLFLOW_USER     = "iceberg_admin"
    POSTGRES_MLFLOW_PASSWORD = var.db_password
    POSTGRES_MLFLOW_DB       = "mlflow"

    # Artifact store MLflow (model weights thật, KHÁC backend store ở trên chỉ
    # lưu params/metrics) — mặc định docker-compose.monitor.yml là bucket tên
    # "mlflow" (dùng được với MinIO local vì không cần unique toàn cầu), nhưng
    # trên AWS S3 thật bucket tên "mlflow" đã bị chiếm bởi account khác (bug
    # thật đã gặp, xem BUGFIXES.md) → override sang bucket + prefix riêng.
    MLFLOW_ARTIFACT_BUCKET = "${aws_s3_bucket.lakehouse.bucket}/mlflow"

    POSTGRES_AIRFLOW_HOST     = aws_db_instance.postgres.address
    POSTGRES_AIRFLOW_USER     = "iceberg_admin"
    POSTGRES_AIRFLOW_PASSWORD = var.db_password
    POSTGRES_AIRFLOW_DB       = "airflow"

    # S3 — thay MinIO. Bucket thật + IAM instance role (không cần static
    # AWS_ACCESS_KEY_ID/SECRET — container Docker mặc định với tới EC2 instance
    # metadata service qua bridge network để lấy credentials tạm thời).
    CATALOG_WAREHOUSE = "s3a://${aws_s3_bucket.lakehouse.bucket}/warehouse/"
    KB_BUCKET         = aws_s3_bucket.lakehouse.bucket
    AWS_REGION        = var.aws_region

    # Rỗng = tắt hẳn nhánh MinIO trong session.py (dùng IAM role thay vì static
    # key/endpoint MinIO). Đọc bởi Spark job qua Airflow Variable AIRFLOW_VAR_S3_ENDPOINT_URL
    # (xem docker-compose.batch_dev.yml) — không set biến này thì Spark job mặc định
    # nối MinIO cục bộ (http://minio:9000), sẽ lỗi DNS trên AWS vì không có container đó.
    S3_ENDPOINT_URL         = ""
    S3A_ENDPOINT            = "s3.${var.aws_region}.amazonaws.com"
    ICEBERG_S3_ENDPOINT     = "https://s3.${var.aws_region}.amazonaws.com"
    S3_CREDENTIALS_PROVIDER = "com.amazonaws.auth.InstanceProfileCredentialsProvider"
    KB_LOCAL_PATH           = ""

    # Self-hosted trên cùng EC2 qua Docker — dùng TÊN CONTAINER (không phải
    # "localhost", gotcha đã gặp ở local dev: REDIS_HOST=localhost làm service
    # trong container không kết nối được vì "localhost" trỏ vào chính container đó)
    REDIS_HOST              = "redis"
    REDIS_PORT              = "6379"
    KAFKA_BOOTSTRAP_SERVERS = "broker:29092" # internal listener, xem docker-compose.infra.yml
    MLFLOW_TRACKING_URI     = "http://mlflow:5000"

    # Agent config — GIỮ Ollama tự host (không chuyển Claude API) vì Phase 2 cần
    # baseline Ollama CPU để so sánh với vLLM GPU
    AGENT_LLM_BACKEND    = "ollama"
    OLLAMA_URL           = "http://ollama:11434"
    OLLAMA_MODEL         = "qwen2.5:3b"
    EMBEDDING_MODEL      = "dangvantuan/vietnamese-embedding"
    AGENT_PROMPT_VERSION = "v1"
  })
}
