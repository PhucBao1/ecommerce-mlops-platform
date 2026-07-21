# AWS Deep Learning AMI (Ubuntu, driver NVIDIA đã cài sẵn) — né việc tự cài driver
# NVIDIA/CUDA thủ công trên Ubuntu trơn (dễ lỗi version + cần reboot). Dùng bản
# "Base OSS Nvidia Driver GPU AMI" (chỉ driver, không cần full PyTorch preinstall)
# vì vLLM chạy qua Docker container (vllm/vllm-openai image tự mang theo CUDA
# runtime riêng) — host chỉ cần driver + nvidia-container-toolkit (user_data tự
# cài toolkit), không phụ thuộc PyTorch cài sẵn trên host.
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_security_group" "vllm" {
  name   = "${var.project_name}-vllm-sg"
  vpc_id = local.vpc_id

  ingress {
    description = "SSH - restricted to my IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # vLLM API — CHỈ cho phép từ Security Group của Instance A (Phase 1), không mở
  # public — Instance A là nơi duy nhất cần gọi vào (qua nginx LB)
  ingress {
    description     = "vLLM OpenAI-compatible API - only from Instance A (Phase 1)"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [local.phase1_security_group_id]
  }

  # DCGM exporter — Prometheus trên Instance A (Phase 1) scrape GPU metrics qua đây
  ingress {
    description     = "DCGM exporter (GPU metrics) - only from Instance A (Phase 1)"
    from_port       = 9400
    to_port         = 9400
    protocol        = "tcp"
    security_groups = [local.phase1_security_group_id]
  }

  # recsys-api-gpu (tạm, benchmark GPU vs CPU cho Two-Tower/SASRec, 16/7/2026) —
  # chỉ mở cho my_ip để loadtest_recsys.py chạy trực tiếp từ máy local gọi vào,
  # không mở public. Có thể xoá sau khi benchmark xong.
  ingress {
    description = "recsys-api-gpu benchmark - restricted to my IP only"
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  # Cho Prometheus (Instance A/Phase 1) scrape metrics recsys-api-gpu — cùng
  # job "fastapi-recsys" với bản CPU, gắn label backend=gpu để phân biệt.
  ingress {
    description     = "recsys-api-gpu metrics - only from Instance A (Phase 1) for Prometheus"
    from_port       = 8001
    to_port         = 8001
    protocol        = "tcp"
    security_groups = [local.phase1_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-vllm-sg" }
}

# ⚠️ THAM KHẢO/HỌC LẠI SAU — KHÔNG DÙNG NỮA. Đây là bản đầu tiên (bare-metal
# pip install + systemd, không Docker) — giữ lại nguyên văn để so sánh/học cách
# viết systemd unit qua user_data, KHÔNG được aws_instance nào tham chiếu tới
# local này nữa (đã đổi sang user_data.sh.tpl dùng Docker bên dưới).
locals {
  vllm_user_data_v1_bare_metal_reference_only = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    exec > /var/log/user-data.log 2>&1

    # DLAMI có sẵn conda + CUDA driver — chỉ cần cài vllm. Một số bản DLAMI có
    # conda env riêng tên "pytorch", số khác dùng "base" — thử activate "pytorch"
    # trước, fallback "base" nếu không có, cuối cùng fallback python3 hệ thống.
    PIP_BIN="pip3"
    if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
      source /opt/conda/etc/profile.d/conda.sh
      conda activate pytorch 2>/dev/null || conda activate base 2>/dev/null || true
      PIP_BIN="pip"
    fi

    $PIP_BIN install --no-cache-dir vllm

    # systemd service — tự restart nếu crash, tự start lại sau reboot
    cat > /etc/systemd/system/vllm.service <<'UNIT'
    [Unit]
    Description=vLLM OpenAI-compatible API server
    After=network.target

    [Service]
    Type=simple
    ExecStart=/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh 2>/dev/null; conda activate pytorch 2>/dev/null || conda activate base 2>/dev/null || true; exec python3 -m vllm.entrypoints.openai.api_server --model ${var.vllm_model} --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.85 --max-model-len 4096'
    Restart=on-failure
    RestartSec=10
    User=root

    [Install]
    WantedBy=multi-user.target
    UNIT

    systemctl daemon-reload
    systemctl enable --now vllm.service

    touch /home/ubuntu/bootstrap_done
  EOF
}

resource "aws_instance" "vllm_replica" {
  count = 2 # B (index 0) và C (index 1)

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.gpu_instance_type
  subnet_id              = local.subnet_id
  vpc_security_group_ids = [aws_security_group.vllm.id]
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.vllm.name

  # Spot — chấp nhận được vì đây là benchmark/test ngắn hạn (bật, đo, tắt trong
  # 1 buổi), KHÔNG dùng cho traffic thật/production (xem lý do trong AWS_VLLM_DEPLOY.md)
  # use_spot=false (tạm thời): On-Demand khi Spot capacity căng (500 liên tục
  # trên RunInstances / bị reclaim ngay sau khi cấp — gặp thật sáng nay).
  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        max_price                      = var.spot_max_price
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 100 # model weights + CUDA libs cần nhiều hơn root mặc định
  }

  # Docker + nvidia-container-toolkit + DCGM exporter + vLLM container — xem
  # user_data.sh.tpl. Dùng `docker run --restart unless-stopped` (không cần
  # systemd unit riêng) — Docker daemon tự khởi động lại container sau crash/reboot.
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    vllm_model        = var.vllm_model
    vllm_quantization = var.vllm_quantization
    s3_bucket         = local.s3_bucket
    model_s3_prefix   = var.model_s3_prefix
    aws_region        = var.aws_region
  })

  # Role=vllm-gpu — Prometheus ở Instance A (Phase 1) dùng EC2 Service Discovery
  tags = {
    Name = "${var.project_name}-vllm-replica-${count.index == 0 ? "B" : "C"}"
    Role = "vllm-gpu"
  }

  lifecycle {
    # Cùng bug đã gặp ở Phase 1 ec2.tf: most_recent=true trên data.aws_ami khiến
    # apply lại (vd thêm replica, sửa tag) có thể đòi destroy+recreate instance
    # Spot đang chạy chỉ vì AWS ra AMI mới. Vá phòng ngừa trước khi apply lần đầu.
    ignore_changes = [ami]
  }
}
