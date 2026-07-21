# NOTE: repo tên "sentiment-api" (không phải "phobert-api" như bản gốc) — khớp
# đúng tên service trong docker-compose.app.yml, tránh lệch tên khi CI/CD
# pull/tag image (bản gốc dùng "phobert-api", không khớp với service key thật).
resource "aws_ecr_repository" "recsys_api" {
  name                 = "recsys-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "sentiment_api" {
  name                 = "sentiment-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "agent_api" {
  name                 = "agent-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Bổ sung — bản gốc thiếu 2 repo này dù docker-compose.app.yml có 2 service tự
# build này (recsys-consumer, recsys-producer — Kafka consumer/producer)
resource "aws_ecr_repository" "recsys_consumer" {
  name                 = "recsys-consumer"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "recsys_producer" {
  name                 = "recsys-producer"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Bổ sung — build Spark/Airflow/dbt trên EC2 (2 vCPU) rất chậm + dễ hết disk
# (đã gặp lỗi "no space left on device" thật). Build local rồi push lên đây,
# EC2 chỉ pull — giống hệt pattern 5 repo app ở trên.
# "spark" dùng chung cho CẢ spark-iceberg VÀ spark-thrift (cùng 1 Dockerfile,
# spark/Dockerfile.spark) — không cần 2 repo riêng.
resource "aws_ecr_repository" "spark" {
  name                 = "spark"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# "airflow" dùng chung cho mọi service airflow-* (webserver/scheduler/init/worker...
# — tất cả cùng build 1 image qua x-airflow-common trong docker-compose.batch_dev.yml)
resource "aws_ecr_repository" "airflow" {
  name                 = "airflow"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "dbt" {
  name                 = "dbt"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Auto-delete untagged images older than 14 days (keep costs down)
resource "aws_ecr_lifecycle_policy" "cleanup" {
  for_each   = toset(["recsys-api", "sentiment-api", "agent-api", "recsys-consumer", "recsys-producer", "spark", "airflow", "dbt"])
  repository = each.key

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Remove untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })

  depends_on = [
    aws_ecr_repository.recsys_api,
    aws_ecr_repository.sentiment_api,
    aws_ecr_repository.agent_api,
    aws_ecr_repository.recsys_consumer,
    aws_ecr_repository.recsys_producer,
    aws_ecr_repository.spark,
    aws_ecr_repository.airflow,
    aws_ecr_repository.dbt,
  ]
}
