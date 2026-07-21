#!/bin/bash
set -euo pipefail

# Chown các path bind-mount từ host (Docker tự tạo thư mục host chưa tồn tại với
# quyền root, "RUN chown" lúc build image không chạm tới được các path này vì
# chúng chỉ tồn tại lúc container chạy) — chạy 1 lần mỗi lần start container,
# rẻ vì các thư mục này nhỏ, không lặp lại trên toàn bộ /app.
for path in /app/model_cache/sentence_transformers /app/model_cache/huggingface /app/artifacts/kb_index; do
    if [ -d "$path" ]; then
        chown -R app:app "$path" || true
    fi
done

exec gosu app "$@"
