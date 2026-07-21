# IAM role cho GPU instance — CHỈ quyền đọc đúng prefix models/ trong bucket Phase 1
# (least-privilege, không cấp quyền ghi/xóa vì GPU instance chỉ cần pull model xuống)

resource "aws_iam_role" "vllm" {
  name = "${var.project_name}-vllm-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "vllm_s3_read_model" {
  name = "s3-read-model-only"
  role = aws_iam_role.vllm.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject"]
      Resource = [
        "arn:aws:s3:::${local.s3_bucket}/${var.model_s3_prefix}/*",
      ]
      }, {
      Effect   = "Allow"
      Action   = ["s3:ListBucket"]
      Resource = "arn:aws:s3:::${local.s3_bucket}"
      Condition = {
        StringLike = { "s3:prefix" = ["${var.model_s3_prefix}/*"] }
      }
    }]
  })
}

# ECR — mirror vllm/vllm-openai (Docker Hub) sang ECR để tránh rate-limit pull
# ẩn danh (~100 pull/6h/IP). Bug thật gặp phải: pull trực tiếp Docker Hub kẹt
# liên tục trên 1 replica sau nhiều lần retry cùng đêm — nghi hết quota ẩn danh.
# GetAuthorizationToken bắt buộc Resource="*" (AWS không cho scope theo repo cho
# action này), các action push/pull thật thì scope đúng 1 repo.
resource "aws_iam_role_policy" "vllm_ecr_mirror" {
  name = "ecr-vllm-openai-mirror"
  role = aws_iam_role.vllm.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ecr:GetAuthorizationToken"]
      Resource = "*"
      }, {
      Effect = "Allow"
      Action = [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
      ]
      Resource = "arn:aws:ecr:${var.aws_region}:*:repository/vllm-openai"
    }]
  })
}

# ECR — pull recsys-api:gpu-bench (biến thể torch CUDA, benchmark GPU vs CPU cho
# Two-Tower/SASRec, 16/7/2026) — repo riêng, đã có sẵn từ CI/CD pipeline
# (.github/workflows/docker_build.yml), chỉ thêm quyền pull cho GPU instance.
resource "aws_iam_role_policy" "vllm_ecr_recsys_pull" {
  name = "ecr-recsys-api-pull"
  role = aws_iam_role.vllm.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
      ]
      Resource = "arn:aws:ecr:${var.aws_region}:*:repository/recsys-api"
    }]
  })
}

resource "aws_iam_instance_profile" "vllm" {
  name = "${var.project_name}-vllm-profile"
  role = aws_iam_role.vllm.name
}

# SSM Session Manager — thiếu ở bản gốc, khiến chỉ SSH được (phụ thuộc IP tĩnh
# + key-pair, dễ vướng NAT xoay IP như đã gặp ở Phase 1). Thêm để debug/exec
# vào instance GPU không cần mở port 22 hay lo IP đổi.
resource "aws_iam_role_policy_attachment" "vllm_ssm" {
  role       = aws_iam_role.vllm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
