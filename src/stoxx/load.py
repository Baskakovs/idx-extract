"""Write extraction results to Parquet datasets."""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
from pathlib import Path
from typing import Any

import polars as pl
from prefect import task

from stoxx.extract import Asset, IndexMembership, SelectionListEntry

logger = logging.getLogger(__name__)


def _dataclass_to_dict(item: Any) -> dict:
    """Convert a frozen dataclass to a dict, converting enums to their values."""
    result = dataclasses.asdict(item)
    for key, value in result.items():
        if isinstance(value, enum.Enum):
            result[key] = value.value
    return result


def _dataclasses_to_df(items: list) -> pl.DataFrame:
    """Convert a list of frozen dataclasses to a Polars DataFrame."""
    return pl.DataFrame([_dataclass_to_dict(x) for x in items], infer_schema_length=None)


def _write_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet atomically using a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp_path, compression="zstd")
    os.replace(tmp_path, path)


def _write_partitioned(df: pl.DataFrame, base_dir: Path, partition_col: str) -> None:
    """Write a DataFrame partitioned by a column into subdirectories."""
    for value in df[partition_col].unique().sort().to_list():
        partition_value = str(value)
        partition_dir = base_dir / f"{partition_col}={partition_value}"
        if partition_dir.exists():
            logger.debug("Partition directory already exists, overwriting: %s", partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        partition_df = df.filter(pl.col(partition_col) == value).drop(partition_col)
        _write_atomic(partition_df, partition_dir / "data.parquet")


@task
def write_parquet_dataset(
    assets: list[Asset],
    entries: list[SelectionListEntry],
    membership: list[IndexMembership],
    output_dir: Path,
) -> None:
    """Write assets, entries, and membership to a partitioned Parquet dataset.

    Args:
        assets: List of Asset dataclasses (written as a flat file).
        entries: List of SelectionListEntry dataclasses (partitioned by review_date).
        membership: List of IndexMembership dataclasses (partitioned by review_date).
        output_dir: Root directory for the output dataset.
    """
    # Assets: flat file, full overwrite
    assets_df = _dataclasses_to_df(assets)
    _write_atomic(assets_df, output_dir / "assets.parquet")

    # Entries: partitioned by review_date
    entries_df = _dataclasses_to_df(entries)
    entries_df = entries_df.with_columns(pl.col("review_date").cast(pl.Date))
    _write_partitioned(entries_df, output_dir / "entries", "review_date")

    # Membership: partitioned by review_date
    # IndexMembership doesn't have review_date, so derive it from entries
    review_dates = {e.review_date for e in entries}
    if len(review_dates) != 1:
        msg = f"Expected exactly one review_date in entries, got {len(review_dates)}"
        raise ValueError(msg)
    review_date = review_dates.pop()

    membership_df = _dataclasses_to_df(membership)
    membership_df = membership_df.with_columns(pl.lit(review_date).alias("review_date"))
    _write_partitioned(membership_df, output_dir / "membership", "review_date")
