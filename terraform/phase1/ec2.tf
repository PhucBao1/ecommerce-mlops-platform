data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "main" {
  name   = "${var.project_name}-sg"
  vpc_id = aws_vpc.main.id # must be same VPC as EC2

  # SSH — giới hạn đúng IP của Bao, KHÔNG mở 0.0.0.0/0 (gap thật đã sửa — bản gốc mở public)
  ingress {
    description = "SSH - restricted to my IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  ingress {
    description = "sentiment-api (PhoBERT)"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "recsys-api" # port thật của recsys-api — bản gốc chỉ mở 8000 và gán nhầm nhãn "RecSys API" (8000 thực ra là sentiment-api)
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "recsys-producer"
    from_port   = 8002
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "agent-api"
    from_port   = 8003
    to_port     = 8003
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Redis (Feast online store, db=1) — chỉ mở nội bộ VPC (10.0.0.0/16), KHÔNG
  # public. Cần cho recsys-api-gpu chạy trên Phase 2 (GPU, cùng VPC khác subnet
  # AZ) gọi vào Feast — benchmark GPU vs CPU cho Two-Tower/SASRec (16/7/2026).
  ingress {
    description = "Redis (Feast online store) - VPC-internal only, for recsys-api-gpu on Phase 2"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # Jaeger OTLP gRPC receiver + Kafka external listener — chỉ mở nội bộ VPC.
  # Phát hiện thật: 2 port này CHƯA từng có rule (dù docker-compose publish ra
  # host) — recsys-api-gpu set đúng OTEL_EXPORTER_OTLP_ENDPOINT/KAFKA_BOOTSTRAP_SERVERS
  # trỏ về IP Phase 1 nhưng vẫn bị SG chặn ở tầng network, gây retry/backoff
  # ngầm (benchmark GPU vs CPU cho Two-Tower/SASRec, 16/7/2026).
  ingress {
    description = "Jaeger OTLP gRPC receiver - VPC-internal only"
    from_port   = 4317
    to_port     = 4317
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    description = "Kafka external listener (PLAINTEXT_HOST) - VPC-internal only"
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # nginx-recsys — load balancer tạm cho 2 replica recsys-api-gpu (Option 3,
  # benchmark horizontal scaling 16/7/2026) — chỉ mở cho my_ip để loadtest
  # trực tiếp từ máy local, không public. Xoá sau khi benchmark xong.
  ingress {
    description = "nginx-recsys LB benchmark - restricted to my IP only"
    from_port   = 8090
    to_port     = 8090
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  ingress {
    description = "Airflow Webserver"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Grafana"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MLflow"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Prometheus"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Jaeger UI"
    from_port   = 16686
    to_port     = 16686
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Qdrant REST + dashboard"
    from_port   = 6333
    to_port     = 6333
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Spark notebook / master UI"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Spark job UI (thrift)"
    from_port   = 4040
    to_port     = 4040
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-sg" }
}

resource "aws_instance" "mlops_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  subnet_id            = aws_subnet.public.id # public subnet → gets public IP
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name
  key_name             = var.key_pair_name # bản gốc thiếu field này — không có cách nào SSH vào được để chạy rsync/feast apply/trigger DAG lần đầu

  vpc_security_group_ids = [aws_security_group.main.id]

  root_block_device {
    # 30GB (mức free-tier "an toàn" giả định ban đầu) KHÔNG đủ thật — đã gặp
    # "no space left on device" liên tục khi pull đủ 7 image (spark+5 app+airflow
    # +dbt cộng lại >25-30GB). Free plan có vẻ KHÔNG chặn cứng volume >30GB như
    # đã chặn instance type — chỉ tính phí phần vượt free tier (rất rẻ, gp3
    # ~$0.08/GB-tháng). Tăng lên 80GB để đủ chỗ cho toàn bộ image + margin.
    volume_size = 80
    volume_type = "gp3" # cheaper + faster than gp2
  }

  # Runs once on first boot — CHỈ bootstrap (không tự "docker compose up" toàn bộ
  # ở đây), vì artifacts/ (~1.15GB, không nằm trong git) + kb-docs/ + Feast registry
  # PHẢI rsync từ máy local lên SAU khi instance đã tồn tại — không thể có sẵn tại
  # thời điểm user_data chạy. Trình tự đầy đủ (rsync → feast apply → compose up)
  # nằm trong AWS_DEPLOY_PHASE1.md, chạy tay qua SSH sau khi terraform apply xong.
  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    exec > /var/log/user-data.log 2>&1

    # 1. Install base tools (KHÔNG cài docker.io/docker-compose-plugin ở đây —
    # "docker-compose-plugin" không tồn tại trên apt repo gốc Ubuntu, chỉ có sau
    # khi thêm repo chính thức Docker ở bước dưới; cài chung 1 lệnh làm cả script
    # exit sớm vì set -euxo pipefail, không bao giờ chạy tới bootstrap_done)
    apt-get update -y
    apt-get install -y awscli jq git curl

    # Cài Docker qua repo chính thức (không dùng gói docker.io cũ trên Ubuntu repo)
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    systemctl enable docker && systemctl start docker
    usermod -aG docker ubuntu

    # 2. Fetch secrets from Secrets Manager TRƯỚC (cần GITHUB_PAT để clone repo private)
    aws secretsmanager get-secret-value \
      --region ${var.aws_region} \
      --secret-id ${aws_secretsmanager_secret.app_secrets.name} \
      --query SecretString --output text > /tmp/secrets.json

    GITHUB_PAT=$(jq -r '.GITHUB_PAT' /tmp/secrets.json)

    # 3. Clone repo — dùng PAT vì repo đang PRIVATE (git clone HTTPS trơn sẽ lỗi 404/auth)
    cd /home/ubuntu
    git clone "https://$${GITHUB_PAT}@github.com/${var.github_repo}.git" repo
    chown -R ubuntu:ubuntu repo

    # 4. Ghi .env từ secret (bỏ GITHUB_PAT ra khỏi .env — chỉ cần lúc clone, không phải runtime env)
    jq -r 'del(.GITHUB_PAT) | to_entries | .[] | "\(.key)=\(.value)"' /tmp/secrets.json \
      > /home/ubuntu/repo/.env.aws
    chown ubuntu:ubuntu /home/ubuntu/repo/.env.aws
    chmod 600 /home/ubuntu/repo/.env.aws
    rm -f /tmp/secrets.json

    # 5. Login ECR (để `docker compose pull` sau này qua CI/CD hoạt động được)
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.${var.aws_region}.amazonaws.com"
    aws ecr get-login-password --region ${var.aws_region} | \
      docker login --username AWS --password-stdin "$ECR_REGISTRY"

    # 6. Network dùng chung cho toàn bộ docker-compose (tất cả file compose của
    # project đều tham chiếu network external này)
    docker network create my_shared_network || true

    touch /home/ubuntu/bootstrap_done
  EOF

  # hop_limit mặc định = 1 chỉ đủ cho process chạy trực tiếp trên host — container
  # Docker (qua network bridge) tính thêm 1 hop, khiến request IMDSv2 (PUT lấy token)
  # bị drop/timeout. Bug thật đã gặp: Spark job treo ~10 phút mỗi lần vì AWS SDK
  # (bên trong container) phải đợi PUT token timeout rồi mới fallback IMDSv1.
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  tags = { Name = "${var.project_name}-server" }

  lifecycle {
    # data.aws_ami.ubuntu dùng most_recent=true — Canonical phát hành AMI mới là
    # data source trôi giá trị, khiến MỌI `terraform apply` sau này (kể cả chỉ để
    # sửa 1 dòng secret) đòi DESTROY + TẠO LẠI instance đang chạy thật (mất toàn bộ
    # state trên đĩa local: repo đã clone, image đã build, artifacts đã rsync).
    # Bug thật bắt được trước khi apply: AMI đổi từ ami-0d0d... sang ami-019c...
    # chỉ vì Ubuntu ra bản patch mới, không phải do ai chủ đích đổi. Nâng cấp AMI
    # phải là hành động CHỦ ĐÍCH (xoá dòng này hoặc taint resource), không phải
    # tác dụng phụ ngoài ý muốn của 1 lần apply không liên quan.
    ignore_changes = [ami]
  }
}

# Elastic IP — không có thì mỗi lần instance bị stop/start, AWS cấp public IP
resource "aws_eip" "mlops_server" {
  instance = aws_instance.mlops_server.id
  domain   = "vpc"

  tags = { Name = "${var.project_name}-eip" }
}
