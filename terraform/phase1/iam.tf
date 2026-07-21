resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "s3_access" {
  name = "s3-least-privilege"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.lakehouse.arn,
        "${aws_s3_bucket.lakehouse.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "ecr_access" {
  name = "ecr-pull"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = [
          aws_ecr_repository.recsys_api.arn,
          aws_ecr_repository.sentiment_api.arn,
          aws_ecr_repository.agent_api.arn,
          aws_ecr_repository.recsys_consumer.arn,
          aws_ecr_repository.recsys_producer.arn,
          aws_ecr_repository.spark.arn,
          aws_ecr_repository.airflow.arn,
          aws_ecr_repository.dbt.arn,
        ]
      }
    ]
  })
}

# SSM: allows GitHub Actions to send deploy commands via SSM (no SSH key needed)
resource "aws_iam_role_policy_attachment" "ssm_managed" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "secrets_access" {
  name = "secrets-manager-read"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.app_secrets.arn
    }]
  })
}

# Prometheus EC2 Service Discovery (Phase 2) — cho phép Prometheus tự tìm
# GPU instance (vLLM replica) qua tag thay vì phải sửa tay IP vào
# monitoring/prometheus.yml mỗi lần tạo/xóa instance (GPU instance ephemeral,
# bật/tắt liên tục để tiết kiệm chi phí). Chỉ quyền đọc (Describe*), không
# thay đổi được gì.
resource "aws_iam_role_policy" "ec2_describe_for_prometheus_sd" {
  name = "ec2-describe-prometheus-sd"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ec2:DescribeInstances", "ec2:DescribeTags"]
      Resource = "*" # Describe* không hỗ trợ resource-level permission
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-profile"
  role = aws_iam_role.ec2_role.name
}
