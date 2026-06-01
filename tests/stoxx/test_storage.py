"""Tests for the storage abstraction layer."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from botocore.stub import Stubber

from stoxx.storage import R2Storage, from_env


@pytest.fixture
def r2_storage():
    """Create an R2Storage instance with test credentials."""
    return R2Storage(
        account_id="test-account",
        access_key_id="test-key",
        secret_access_key="test-secret",  # noqa: S106
        bucket_name="test-bucket",
    )


@pytest.fixture
def stubber(r2_storage):
    """Create a botocore Stubber for the R2 client."""
    s = Stubber(r2_storage._client)
    s.activate()
    yield s
    s.deactivate()


class TestListReviewDates:
    """Tests for R2Storage.list_review_dates."""

    def test_parses_partitions(self, r2_storage, stubber):
        """Partition directories are parsed into date objects."""
        stubber.add_response(
            "list_objects_v2",
            {
                "CommonPrefixes": [
                    {"Prefix": "STOXX600/membership/review_date=2024-03-01/"},
                    {"Prefix": "STOXX600/membership/review_date=2024-06-01/"},
                ],
                "Contents": [],
            },
            expected_params={
                "Bucket": "test-bucket",
                "Prefix": "STOXX600/membership/",
                "Delimiter": "/",
            },
        )

        result = r2_storage.list_review_dates("STOXX600/membership")

        assert result == [date(2024, 3, 1), date(2024, 6, 1)]

    def test_empty(self, r2_storage, stubber):
        """Empty bucket returns empty list."""
        stubber.add_response(
            "list_objects_v2",
            {"Contents": []},
            expected_params={
                "Bucket": "test-bucket",
                "Prefix": "STOXX600/membership/",
                "Delimiter": "/",
            },
        )

        result = r2_storage.list_review_dates("STOXX600/membership")

        assert result == []


class TestUploadDirectory:
    """Tests for R2Storage.upload_directory."""

    def test_uploads_files(self, r2_storage, stubber, tmp_path):
        """All files in directory tree are uploaded."""
        # Create test file structure
        sub = tmp_path / "membership" / "review_date=2024-03-01"
        sub.mkdir(parents=True)
        data_file = sub / "data.parquet"
        data_file.write_bytes(b"fake-parquet-data")

        stubber.add_response(
            "put_object",
            {},
            expected_params={
                "Bucket": "test-bucket",
                "Key": "STOXX600/membership/review_date=2024-03-01/data.parquet",
                "Body": b"fake-parquet-data",
            },
        )

        count = r2_storage.upload_directory(tmp_path, "STOXX600")

        assert count == 1

    def test_uploads_multiple_files(self, r2_storage, stubber, tmp_path):
        """Multiple files across subdirectories are all uploaded."""
        for name in ["a.parquet", "b.parquet"]:
            (tmp_path / name).write_bytes(b"data")
            stubber.add_response(
                "put_object",
                {},
                expected_params={
                    "Bucket": "test-bucket",
                    "Key": f"STOXX600/{name}",
                    "Body": b"data",
                },
            )

        count = r2_storage.upload_directory(tmp_path, "STOXX600")

        assert count == 2


class TestDownloadFile:
    """Tests for R2Storage.download_file."""

    def test_downloads_to_local_path(self, r2_storage, stubber, tmp_path):
        """File is downloaded and written to the specified local path."""
        from io import BytesIO

        from botocore.response import StreamingBody

        body = StreamingBody(BytesIO(b"parquet-content"), len(b"parquet-content"))

        stubber.add_response(
            "get_object",
            {"Body": body},
            expected_params={
                "Bucket": "test-bucket",
                "Key": "STOXX600/membership/review_date=2024-03-01/data.parquet",
            },
        )

        dest = tmp_path / "sub" / "data.parquet"
        r2_storage.download_file("STOXX600/membership/review_date=2024-03-01/data.parquet", dest)

        assert dest.read_bytes() == b"parquet-content"


class TestFromEnv:
    """Tests for the from_env factory function."""

    @pytest.mark.asyncio
    async def test_reads_secret_blocks(self):
        """Factory reads all required Prefect Secret blocks."""
        secrets = {
            "r2-account-id": "test-account",
            "r2-access-key-id": "test-key",
            "r2-secret-access-key": "test-secret",
            "r2-bucket-name": "test-bucket",
        }

        async def fake_load(name):
            mock = MagicMock()
            mock.get.return_value = secrets[name]
            return mock

        with patch("prefect.blocks.system.Secret.load", side_effect=fake_load):
            storage = await from_env()

        assert isinstance(storage, R2Storage)
        assert storage._bucket_name == "test-bucket"

    @pytest.mark.asyncio
    async def test_missing_block_raises(self):
        """Missing Secret block raises AttributeError."""

        async def fake_load(name):
            return None

        with patch("prefect.blocks.system.Secret.load", side_effect=fake_load), pytest.raises(AttributeError):
            await from_env()
