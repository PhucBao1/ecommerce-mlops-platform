#!/bin/bash
set -e

echo "Starting Spark Iceberg..."

# có thể thêm wait-for nếu cần
# ví dụ: chờ MinIO
# sleep 5


exec "$@"
