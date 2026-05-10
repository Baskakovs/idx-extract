"""Tests for Parquet dataset writing."""

from __future__ import annotations

import logging

import polars as pl

from stoxx.extract import compute_membership
from stoxx.load import write_parquet_dataset


def _make_membership(entries):
    """Generate bootstrap membership from parsed entries."""
    return compute_membership(entries, prior_membership=None)


class TestWriteParquetDataset:
    """Tests for write_parquet_dataset."""

    def test_assets_parquet_written(self, tmp_path, parsed_csv, membership):
        """Write dataset and verify assets.parquet exists and is readable."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        assets_path = tmp_path / "assets.parquet"
        assert assets_path.exists()
        df = pl.read_parquet(assets_path)
        assert len(df) == len(assets)

    def test_entries_partitioned_by_review_date(self, tmp_path, parsed_csv, membership):
        """Entries are partitioned into review_date=YYYY-MM-DD directories."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        partition_dir = tmp_path / "entries" / "review_date=2024-03-01"
        assert partition_dir.exists()
        assert (partition_dir / "data.parquet").exists()

    def test_membership_partitioned_by_review_date(self, tmp_path, parsed_csv, membership):
        """Membership is partitioned into review_date=YYYY-MM-DD directories."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        partition_dir = tmp_path / "membership" / "review_date=2024-03-01"
        assert partition_dir.exists()
        assert (partition_dir / "data.parquet").exists()

    def test_assets_overwrite_is_idempotent(self, tmp_path, parsed_csv, membership):
        """Writing assets twice produces no error and same content."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)
        df1 = pl.read_parquet(tmp_path / "assets.parquet")

        write_parquet_dataset(assets, entries, membership, tmp_path)
        df2 = pl.read_parquet(tmp_path / "assets.parquet")

        assert df1.equals(df2)

    def test_partition_overwrite_logs_warning(self, tmp_path, parsed_csv, membership, caplog):
        """Writing the same review_date twice logs a warning."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        with caplog.at_level(logging.WARNING):
            write_parquet_dataset(assets, entries, membership, tmp_path)

        assert any("already exists" in msg for msg in caplog.messages)

    def test_entry_reason_stored_as_string(self, tmp_path, parsed_csv, membership):
        """EntryReason enum values are stored as strings in membership parquet."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        df = pl.read_parquet(tmp_path / "membership" / "review_date=2024-03-01" / "data.parquet")
        assert df["entry_reason"].dtype == pl.Utf8
        assert "bootstrap" in df["entry_reason"].to_list()

    def test_atomic_write_no_tmp_files_remain(self, tmp_path, parsed_csv, membership):
        """No .tmp files remain after writing."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert tmp_files == []

    def test_roundtrip_preserves_data(self, tmp_path, parsed_csv, membership):
        """Write assets and read back, verifying all fields match."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path)

        df = pl.read_parquet(tmp_path / "assets.parquet")
        assert set(df.columns) == {"isin", "internal_key", "ric", "name", "country", "currency", "sedol"}
        assert len(df) == len(assets)

        # Verify a specific asset roundtrips correctly
        first_asset = assets[0]
        row = df.filter(pl.col("isin") == first_asset.isin).to_dicts()[0]
        assert row["isin"] == first_asset.isin
        assert row["internal_key"] == first_asset.internal_key
        assert row["ric"] == first_asset.ric
        assert row["name"] == first_asset.name
        assert row["country"] == first_asset.country
        assert row["currency"] == first_asset.currency
