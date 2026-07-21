#!/usr/bin/env bash
# Chạy 1 lệnh shell trên EC2 Phase 1 qua AWS SSM (không cần SSH — security group
# chỉ mở port 22 cho đúng 1 IP, SSM là cách duy nhất can thiệp từ máy khác).
#
# Dùng:
#   ./scripts/ssm.sh "docker ps"
#   ./scripts/ssm.sh "docker logs agent-api --tail 50"
#
# Không cần nhớ cú pháp aws ssm send-command dài dòng — script tự gửi lệnh,
# tự chờ xong, tự in kết quả (cả stdout lẫn stderr).

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Dùng: $0 \"<lệnh shell muốn chạy trên EC2>\"" >&2
  exit 1
fi

INSTANCE_ID="${SSM_INSTANCE_ID:-i-097e562737069678e}"
REGION="${AWS_REGION:-ap-southeast-1}"
COMMAND="$1"

CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[\"$COMMAND\"]" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID (đang chạy...)" >&2

# Chờ tới khi command xong (Success/Failed/Cancelled/TimedOut)
while true; do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Pending")
  case "$STATUS" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
  sleep 3
done

aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query '{Status:Status,Output:StandardOutputContent,Error:StandardErrorContent}' \
  --output json
