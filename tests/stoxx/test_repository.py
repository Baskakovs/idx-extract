"""Tests for STOXX Parquet repository classes."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from repository import ParquetRepository
from stoxx.load import write_parquet_dataset
from stoxx.repository import StoxxIndex


class TestStoxxIndex:
    """Tests for the StoxxIndex class."""

    def test_available_review_dates(self, tmp_path, parsed_csv, membership):
        """Finds dates from partition dirs."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        dates = idx.available_review_dates
        assert len(dates) >= 1
        assert all(isinstance(d, date) for d in dates)

    def test_available_review_dates_empty(self, tmp_path):
        """Returns [] when no data exists."""
        idx = StoxxIndex(name="empty", path=tmp_path / "nonexistent")
        assert idx.available_review_dates == []

    def test_assets_returns_dataframe(self, tmp_path, parsed_csv, membership):
        """Assets property returns correct columns and row count."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        df = idx.assets
        assert len(df) == len(assets)
        assert "isin" in df.columns
        assert "name" in df.columns

    def test_entries_for_review_date(self, tmp_path, parsed_csv, membership):
        """Entries includes review_date column."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        rd = idx.available_review_dates[0]
        df = idx.entries(rd)
        assert "review_date" in df.columns
        assert len(df) > 0

    def test_membership_for_review_date(self, tmp_path, parsed_csv, membership):
        """Membership has entry_reason as string."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        rd = idx.available_review_dates[0]
        df = idx.membership(rd)
        assert "entry_reason" in df.columns
        assert df["entry_reason"].dtype == pl.String

    def test_constituents_returns_joined_members(self, tmp_path, parsed_csv, membership):
        """Joined df has both asset and membership columns."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        rd = idx.available_review_dates[0]
        df = idx.constituents(rd)
        assert "isin" in df.columns
        assert "name" in df.columns
        assert "entry_reason" in df.columns
        assert len(df) > 0

    def test_constituents_default_latest_date(self, tmp_path, parsed_csv, membership):
        """None resolves to latest review date."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        idx = StoxxIndex(name="stoxx600", path=tmp_path / "stoxx600")
        df_default = idx.constituents()
        df_explicit = idx.constituents(idx.available_review_dates[-1])
        assert df_default.shape == df_explicit.shape

    def test_resolve_review_date_raises_on_empty(self, tmp_path):
        """ValueError when no dates available."""
        idx = StoxxIndex(name="empty", path=tmp_path / "nonexistent")
        with pytest.raises(ValueError, match="No review dates available"):
            idx.constituents()


class TestParquetRepository:
    """Tests for the ParquetRepository class."""

    def test_discovers_index_directories(self, tmp_path, parsed_csv, membership):
        """Finds subdirs with assets.parquet."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        repo = ParquetRepository(root=tmp_path, index_factory=StoxxIndex)
        assert len(repo.indexes) == 1
        assert repo.indexes[0].name == "stoxx600"

    def test_empty_root_returns_empty_list(self, tmp_path):
        """Empty/missing root returns empty list."""
        repo = ParquetRepository(root=tmp_path / "nonexistent", index_factory=StoxxIndex)
        assert repo.indexes == []

    def test_get_index_by_name(self, tmp_path, parsed_csv, membership):
        """Returns correct StoxxIndex."""
        assets, entries = parsed_csv
        write_parquet_dataset(assets, entries, membership, tmp_path / "stoxx600")
        repo = ParquetRepository(root=tmp_path, index_factory=StoxxIndex)
        idx = repo.get_index("stoxx600")
        assert isinstance(idx, StoxxIndex)
        assert idx.name == "stoxx600"

    def test_get_index_raises_on_missing(self, tmp_path):
        """KeyError for unknown name."""
        repo = ParquetRepository(root=tmp_path, index_factory=StoxxIndex)
        with pytest.raises(KeyError, match="not found"):
            repo.get_index("nonexistent")
