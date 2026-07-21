"""
S3/MinIO abstraction for KB document storage.

Local dev  → set S3_ENDPOINT_URL=http://minio:9000 (MinIO)
AWS prod   → leave S3_ENDPOINT_URL unset (boto3 uses default AWS endpoint)

No other code change needed to switch environments.
"""

import logging
import os
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_BUCKET = os.getenv("KB_BUCKET", "warehouse")
_PREFIX = os.getenv("KB_PREFIX", "kb-docs/")


class S3KBLoader:
    def __init__(self):
        endpoint = os.getenv("S3_ENDPOINT_URL") or None  # None → AWS, str → MinIO
        self._bucket = _BUCKET
        self._prefix = _PREFIX
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

    def upload_bytes(
        self, data: bytes, filename: str, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload raw bytes as kb-docs/<filename>. Returns the full S3 key."""
        key = f"{self._prefix}{filename}"
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info(
            "s3_kb_upload bucket=%s key=%s bytes=%d", self._bucket, key, len(data)
        )
        return key

    def list_files(self) -> list[dict]:
        """List all files under the kb-docs/ prefix."""
        paginator = self._s3.get_paginator("list_objects_v2")
        files: list[dict] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                if obj["Key"] == self._prefix:  # skip the folder marker itself
                    continue
                files.append(
                    {
                        "key": obj["Key"],
                        "filename": obj["Key"].removeprefix(self._prefix),
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "etag": obj["ETag"].strip('"'),
                    }
                )
        return files

    def download_bytes(self, key: str) -> bytes:
        resp = self._s3.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    def download_to_tmp(self, key: str) -> str:
        """Download file to a temp path. Caller is responsible for os.unlink."""
        suffix = Path(key).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(self.download_bytes(key))
            return f.name

    def ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist (useful for fresh MinIO installs)."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except ClientError:
            self._s3.create_bucket(Bucket=self._bucket)
            logger.info("s3_kb_bucket_created bucket=%s", self._bucket)
