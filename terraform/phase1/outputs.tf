output "ec2_public_ip" {
  description = "Elastic IP của MLOps server — cố định, không đổi dù instance bị stop/start (khác aws_instance.mlops_server.public_ip vốn là IP động)"
  value       = aws_eip.mlops_server.public_ip
}

output "ec2_instance_id" {
  description = "EC2 instance ID — set as GitHub variable EC2_INSTANCE_ID for SSM deploy"
  value       = aws_instance.mlops_server.id
}

output "s3_bucket" {
  description = "S3 Data Lakehouse bucket name"
  value       = aws_s3_bucket.lakehouse.bucket
}

output "ecr_recsys_url" {
  value = aws_ecr_repository.recsys_api.repository_url
}

output "ecr_sentiment_url" {
  value = aws_ecr_repository.sentiment_api.repository_url
}

output "ecr_agent_url" {
  value = aws_ecr_repository.agent_api.repository_url
}

output "ecr_recsys_consumer_url" {
  value = aws_ecr_repository.recsys_consumer.repository_url
}

output "ecr_recsys_producer_url" {
  value = aws_ecr_repository.recsys_producer.repository_url
}

output "ecr_registry" {
  description = "ECR registry hostname — set as GitHub variable ECR_REGISTRY"
  value       = split("/", aws_ecr_repository.recsys_api.repository_url)[0]
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint — already stored in Secrets Manager automatically"
  value       = aws_db_instance.postgres.address
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC — set as GitHub variable AWS_ROLE_ARN"
  value       = aws_iam_role.github_actions.arn
}

output "secrets_arn" {
  description = "Secrets Manager ARN — app fetches passwords at startup"
  value       = aws_secretsmanager_secret.app_secrets.arn
}

# Phase 2 (GPU/vLLM, EKS, MSK, MWAA — thư mục Terraform state HOÀN TOÀN RIÊNG,
# KHÔNG chung state với Phase 1) đọc 3 output dưới đây làm input, để đặt GPU
# instance cùng VPC/subnet/SG này thay vì dựng mạng mới.
output "vpc_id" {
  description = "VPC ID — Phase 2 tái dùng, không tạo VPC mới"
  value       = aws_vpc.main.id
}

output "subnet_id" {
  description = "Public subnet ID — Phase 2 tái dùng"
  value       = aws_subnet.public.id
}

output "security_group_id" {
  description = "Security Group ID — Phase 2 có thể tham chiếu hoặc tạo SG riêng cho phép traffic từ SG này"
  value       = aws_security_group.main.id
}

output "deploy_instructions" {
  description = "Copy these values to GitHub → Settings → Variables (not Secrets)"
  value       = <<-EOT
    GitHub Variables (Settings → Secrets and variables → Actions → Variables):
      AWS_ROLE_ARN       = ${aws_iam_role.github_actions.arn}
      ECR_REGISTRY       = ${split("/", aws_ecr_repository.recsys_api.repository_url)[0]}
      EC2_INSTANCE_ID    = ${aws_instance.mlops_server.id}
      AWS_REGION         = ${var.aws_region}
  EOT
}
