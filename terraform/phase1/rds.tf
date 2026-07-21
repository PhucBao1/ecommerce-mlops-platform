# RDS PostgreSQL — replaces local postgres-catalog container
# Used CHỈ bởi Spark Iceberg catalog. MLflow và Airflow vẫn dùng Postgres tự host
# riêng của chúng (docker-compose.monitor.yml / docker-compose.batch_dev.yml) —
# dùng chung 1 RDS instance cho cả 3 mục đích cần multi-database bootstrap phức
# tạp hơn (Terraform postgresql provider hoặc script riêng), không đáng cho demo.
#
# Local dev:  POSTGRES_HOST=postgres-catalog (docker-compose)
# AWS prod:   POSTGRES_ICEBERG_HOST=<rds_endpoint from output>

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = [aws_subnet.public.id, aws_subnet.private.id]

  tags = { Project = var.project_name }
}

resource "aws_security_group" "rds" {
  name   = "${var.project_name}-rds-sg"
  vpc_id = aws_vpc.main.id

  # Only accept connections from EC2 (not public internet)
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.main.id]
  }

  tags = { Name = "${var.project_name}-rds-sg" }
}

resource "aws_db_instance" "postgres" {
  identifier        = "${var.project_name}-postgres"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  storage_type      = "gp3"

  db_name  = "iceberg_metadata"
  username = "iceberg_admin" # "admin" là từ khóa dành riêng của Postgres engine, RDS từ chối
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Single-AZ for portfolio (set multi_az=true for production)
  multi_az            = false
  publicly_accessible = false
  skip_final_snapshot = true
  deletion_protection = false

  # AWS Free Plan báo lỗi FreeTierRestrictionError nếu backup_retention_period
  # vượt giới hạn cho phép — hạ xuống 1 (thấp nhất còn bật backup) để an toàn
  backup_retention_period = 1
  backup_window           = "03:00-04:00"

  tags = { Project = var.project_name }
}
