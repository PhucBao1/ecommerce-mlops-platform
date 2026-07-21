terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Đọc output của Phase 1 qua chính state file local của nó (KHÔNG chung state —
# đây chỉ là data source READ-ONLY, `terraform apply` ở Phase 2 không bao giờ
# động vào/khóa state Phase 1). Bắt buộc Phase 1 đã `terraform apply` xong trước
# (state file phải tồn tại ở ../phase1/terraform.tfstate) — đây là chủ đích, vì
# Phase 2 (GPU) phụ thuộc VPC/SG đã có sẵn từ Phase 1.
data "terraform_remote_state" "phase1" {
  backend = "local"
  config = {
    path = "${path.module}/../phase1/terraform.tfstate"
  }
}

locals {
  vpc_id                   = data.terraform_remote_state.phase1.outputs.vpc_id
  subnet_id                = var.gpu_subnet_id_override != "" ? var.gpu_subnet_id_override : data.terraform_remote_state.phase1.outputs.subnet_id
  phase1_security_group_id = data.terraform_remote_state.phase1.outputs.security_group_id
  s3_bucket                = data.terraform_remote_state.phase1.outputs.s3_bucket
}
