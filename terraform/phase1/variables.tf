variable "aws_region" {
  default = "ap-southeast-1"
}

variable "project_name" {
  default = "ecommerce-mlops"
}

variable "instance_type" {
  description = "m5.2xlarge (8 vCPU/32GB) — full platform chạy đồng thời: Spark+Kafka+Airflow+MLflow+monitoring+serving (~20-25GB RAM thực tế). t3.large (2vCPU/8GB) chỉ đủ cho 3 service serving, không đủ khi thêm Airflow/Spark/dbt."
  default     = "m5.2xlarge"
}

variable "github_repo" {
  description = "GitHub repo in owner/name format — used to scope OIDC trust. PHẢI khớp đúng `git remote -v` thật, không phải tên cũ/tên dự kiến."
  default     = "PhucBao1/Ecommerce"
}

variable "github_branch" {
  description = "Branch được phép trigger OIDC — giới hạn đúng branch, không để mọi ref đều assume role được"
  type        = string
  default     = "main"
}

variable "db_instance_class" {
  description = "RDS instance class"
  default     = "db.t3.micro"
}

variable "db_password" {
  description = "Master password for RDS PostgreSQL — set via TF_VAR_db_password env var, never hardcode"
  sensitive   = true
}

variable "admin_api_key" {
  description = <<-EOT
    Key bảo vệ /admin/kb/* của agent-api. BẮT BUỘC — agent-api fail-CLOSED và
    docker-compose từ chối khởi động nếu thiếu. Sinh bằng: openssl rand -hex 32
    Set qua TF_VAR_admin_api_key, không hardcode.
  EOT
  sensitive   = true
}

variable "my_ip" {
  description = "IP public của bạn (CIDR, vd 1.2.3.4/32) — giới hạn SSH, KHÔNG mở 0.0.0.0/0 cho port 22. Lấy bằng: curl -s ifconfig.me"
  type        = string
}

variable "key_pair_name" {
  description = "Tên EC2 key-pair đã tạo sẵn trong AWS Console (EC2 → Key Pairs) — dùng để SSH vào instance khi cần chạy tay (rsync artifacts, feast apply, trigger Airflow DAG lần đầu)"
  type        = string
}

variable "github_pat" {
  description = "GitHub Personal Access Token (repo scope, read-only đủ dùng) — để EC2 user_data clone được repo PRIVATE. Set qua TF_VAR_github_pat, KHÔNG hardcode. Lưu vào Secrets Manager, không lưu trong user_data script gốc."
  type        = string
  sensitive   = true
}
