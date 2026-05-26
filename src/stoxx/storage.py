"""Storage abstraction for STOXX data with Cloudflare R2 implementation."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


class Storage(ABC):
    """Abstract base class for remote object storage."""

    @abstractmethod
    def list_review_dates(self, prefix: str) -> list[date]:
        """List review dates from partition directories under the given prefix.

        Args:
            prefix: Object key prefix (e.g. "STOXX600/membership").

        Returns:
            Sorted list of dates parsed from review_date=YYYY-MM-DD partitions.
        """

    @abstractmethod
    def upload_directory(self, local_dir: Path, prefix: str) -> int:
        """Upload all files in a local directory tree to storage.

        Args:
            local_dir: Local directory root to upload.
            prefix: Remote key prefix for uploaded objects.

        Returns:
            Number of files uploaded.
        """

    @abstractmethod
    def download_file(self, key: str, local_path: Path) -> None:
        """Download a single file from storage.

        Args:
            key: Remote object key.
            local_path: Local destination path.
        """


class R2Storage(Storage):
    """Cloudflare R2 storage backend using the S3-compatible API."""

    def __init__(self, account_id: str, access_key_id: str, secret_access_key: str, bucket_name: str) -> None:
        """Initialize R2 storage client.

        Args:
            account_id: Cloudflare account ID.
            access_key_id: R2 access key ID.
            secret_access_key: R2 secret access key.
            bucket_name: R2 bucket name.
        """
        import boto3

        self._bucket_name = bucket_name
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    def list_review_dates(self, prefix: str) -> list[date]:
        """List review dates from partition directories under the given prefix."""
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        response = self._client.list_objects_v2(Bucket=self._bucket_name, Prefix=prefix, Delimiter="/")

        dates: list[date] = []
        for cp in response.get("CommonPrefixes", []):
            dirname = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            match = re.match(r"review_date=(\d{4}-\d{2}-\d{2})", dirname)
            if match:
                dates.append(date.fromisoformat(match.group(1)))

        return sorted(dates)

    def upload_directory(self, local_dir: Path, prefix: str) -> int:
        """Upload all files in a local directory tree to R2."""
        count = 0
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{prefix}/{relative}" if prefix else str(relative)
            with file_path.open("rb") as f:
                self._client.put_object(Bucket=self._bucket_name, Key=key, Body=f.read())
            count += 1
            logger.info("Uploaded %s -> %s", file_path, key)
        return count

    def download_file(self, key: str, local_path: Path) -> None:
        """Download a single file from R2."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        response = self._client.get_object(Bucket=self._bucket_name, Key=key)
        local_path.write_bytes(response["Body"].read())
        logger.info("Downloaded %s -> %s", key, local_path)


def from_env() -> R2Storage:
    """Create an R2Storage instance from environment variables.

    Reads R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME.

    Returns:
        Configured R2Storage instance.
    """
    return R2Storage(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket_name=os.environ["R2_BUCKET_NAME"],
    )
