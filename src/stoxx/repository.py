"""Concrete STOXX index backed by a Parquet dataset directory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from repository import Index


@dataclass
class StoxxIndex(Index):
    """A STOXX index backed by a Parquet dataset directory."""

    name: str
    path: Path

    @property
    def available_review_dates(self) -> list[date]:
        """Return sorted list of available review dates from membership partitions."""
        membership_dir = self.path / "membership"
        if not membership_dir.exists():
            return []
        dates = []
        for partition in sorted(membership_dir.iterdir()):
            match = re.match(r"review_date=(\d{4}-\d{2}-\d{2})", partition.name)
            if match:
                dates.append(date.fromisoformat(match.group(1)))
        return sorted(dates)

    @property
    def assets(self) -> pl.DataFrame:
        """Read the assets.parquet file."""
        return pl.read_parquet(self.path / "assets.parquet")

    def entries(self, review_date: date | None = None) -> pl.DataFrame:
        """Read entries for a review date, re-attaching the review_date column."""
        rd = self._resolve_review_date(review_date)
        path = self.path / "entries" / f"review_date={rd}" / "data.parquet"
        df = pl.read_parquet(path)
        return df.with_columns(pl.lit(rd).alias("review_date"))

    def membership(self, review_date: date | None = None) -> pl.DataFrame:
        """Read membership for a review date, re-attaching the review_date column."""
        rd = self._resolve_review_date(review_date)
        path = self.path / "membership" / f"review_date={rd}" / "data.parquet"
        df = pl.read_parquet(path)
        return df.with_columns(pl.lit(rd).alias("review_date"))

    def constituents(self, review_date: date | None = None) -> pl.DataFrame:
        """Return members (is_member=True) joined with assets on isin."""
        rd = self._resolve_review_date(review_date)
        members = self.membership(rd).filter(pl.col("is_member"))
        return members.join(self.assets, on="isin")

    def _resolve_review_date(self, review_date: date | None) -> date:
        """Resolve None to the latest available review date."""
        if review_date is not None:
            return review_date
        dates = self.available_review_dates
        if not dates:
            msg = f"No review dates available for index '{self.name}'"
            raise ValueError(msg)
        return dates[-1]
