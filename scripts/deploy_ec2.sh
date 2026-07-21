#!/bin/bash
# Deploy recsys-api + sentiment-api lên EC2 (chạy sau terraform apply)
# Usage: bash scripts/deploy_ec2.sh <path-to-key.pem>

set -e

KEY=$1
if [ -z "$KEY" ]; then
  echo "Usage: bash scripts/deploy_ec2.sh <path-to-key.pem>"
  exit 1
fi

# Lấy EC2 IP từ terraform output
EC2_IP=$(cd terraform/phase1 && terraform output -raw ec2_public_ip)
echo "→ EC2 IP: $EC2_IP"

# ─── 1. Cài Docker trên EC2 (lần đầu) ───────────────────────────────
ssh -i "$KEY" -o StrictHostKeyChecking=no ec2-user@"$EC2_IP" << 'REMOTE'
  if ! command -v docker &>/dev/null; then
    sudo yum update -y
    sudo yum install -y docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
      -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker installed."
  else
    echo "Docker already installed."
  fi
REMOTE

# ─── 2. Copy file cần thiết lên EC2 ─────────────────────────────────
echo "→ Copying files..."
scp -i "$KEY" -o StrictHostKeyChecking=no .env ec2-user@"$EC2_IP":~/ecommerce/.env
scp -i "$KEY" docker-compose.infra.yml   ec2-user@"$EC2_IP":~/ecommerce/
scp -i "$KEY" docker-compose.app.yml     ec2-user@"$EC2_IP":~/ecommerce/
scp -i "$KEY" docker-compose.monitor.yml ec2-user@"$EC2_IP":~/ecommerce/

# Copy source code và artifacts
ssh -i "$KEY" ec2-user@"$EC2_IP" "mkdir -p ~/ecommerce/src ~/ecommerce/artifacts"
rsync -avz --progress -e "ssh -i $KEY" \
  src/ ec2-user@"$EC2_IP":~/ecommerce/src/
rsync -avz --progress -e "ssh -i $KEY" \
  artifacts/ ec2-user@"$EC2_IP":~/ecommerce/artifacts/

# ─── 3. Tạo network + khởi động services ────────────────────────────
echo "→ Starting services on EC2..."
ssh -i "$KEY" ec2-user@"$EC2_IP" << 'REMOTE'
  cd ~/ecommerce
  docker network create my_shared_network 2>/dev/null || true

  # Chỉ cần Redis + 2 APIs để demo
  docker-compose -f docker-compose.infra.yml up -d redis
  sleep 5
  docker-compose -f docker-compose.app.yml up --build -d recsys-api sentiment-api
  echo "→ Services started!"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
REMOTE

echo ""
echo "✅ Deploy xong!"
echo ""
echo "APIs:"
echo "  recsys-api:    http://$EC2_IP:8001/health"
echo "  sentiment-api: http://$EC2_IP:8000/health"
echo ""
echo "Cập nhật Streamlit:"
echo "  export RECSYS_URL=http://$EC2_IP:8001"
echo "  export SENTIMENT_URL=http://$EC2_IP:8000"
echo "  streamlit run src/serving/streamlit_app/app.py"
echo ""
echo "Nhớ mở Security Group cho port 8000 và 8001 trên AWS Console."
