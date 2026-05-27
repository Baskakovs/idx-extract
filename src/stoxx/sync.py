"""Incremental sync orchestrator for STOXX 600 data."""

from __future__ import annotations

import logging
import tempfile
from datetime import date
from pathlib import Path

import polars as pl
from prefect import flow, get_run_logger, task

from .download import START_DATE, download_selection_lists, get_periods
from .extract import Asset, compute_membership, parse_selection_list
from .load import _dataclasses_to_df, _write_atomic, write_parquet_dataset
from .storage import Storage, from_env

logger = logging.getLogger(__name__)


def _read_local_membership(output_dir: Path, review_date: date) -> set[str] | None:
    """Read member ISINs from a local membership partition.

    Args:
        output_dir: Root output directory containing the membership subdirectory.
        review_date: The review date partition to read.

    Returns:
        Set of ISINs with is_member=True, or None if partition does not exist.
    """
    path = output_dir / "membership" / f"review_date={review_date}" / "data.parquet"
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return set(df.filter(pl.col("is_member"))["isin"].to_list())


def _download_prior_membership(storage: Storage, review_date: date, tmp_dir: Path) -> set[str] | None:
    """Download and read prior membership from remote storage.

    Args:
        storage: Storage backend to download from.
        review_date: The review date partition to download.
        tmp_dir: Temporary directory for downloaded files.

    Returns:
        Set of ISINs with is_member=True, or None if not found remotely.
    """
    key = f"STOXX600/membership/review_date={review_date}/data.parquet"
    local_path = tmp_dir / f"prior_{review_date}.parquet"
    try:
        storage.download_file(key, local_path)
    except Exception:
        logger.info("No prior membership found in remote for %s", review_date)
        return None
    df = pl.read_parquet(local_path)
    return set(df.filter(pl.col("is_member"))["isin"].to_list())


def _compute_intervals(member_dates: list[date], all_dates: list[date]) -> list[tuple[date, date]]:
    """Compute contiguous membership intervals from review dates.

    Args:
        member_dates: Sorted review dates when an ISIN was a member.
        all_dates: Sorted global list of all review dates.

    Returns:
        List of (first_included, last_included) tuples for each contiguous span.
    """
    if not member_dates:
        return []

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    indices = [date_to_idx[d] for d in member_dates]

    intervals: list[tuple[date, date]] = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx != prev + 1:
            intervals.append((all_dates[start], all_dates[prev]))
            start = idx
        prev = idx

    intervals.append((all_dates[start], all_dates[prev]))
    return intervals


def _build_isin_membership(output_dir: Path) -> tuple[dict[str, list[date]], list[date]]:
    """Build ISIN→member dates mapping from all membership partitions on disk.

    Args:
        output_dir: Root output directory containing the membership subdirectory.

    Returns:
        Tuple of (isin_to_dates dict, sorted global date list).
    """
    membership_dir = output_dir / "membership"
    isin_dates: dict[str, list[date]] = {}
    all_dates: set[date] = set()

    if not membership_dir.exists():
        return {}, []

    for partition in sorted(membership_dir.iterdir()):
        if not partition.name.startswith("review_date="):
            continue
        review_date = date.fromisoformat(partition.name.split("=", 1)[1])
        all_dates.add(review_date)

        parquet_path = partition / "data.parquet"
        if not parquet_path.exists():
            continue

        df = pl.read_parquet(parquet_path)
        member_isins = df.filter(pl.col("is_member"))["isin"].to_list()
        for isin in member_isins:
            isin_dates.setdefault(isin, []).append(review_date)

    # Sort all collected dates
    sorted_dates = sorted(all_dates)
    for isin in isin_dates:
        isin_dates[isin].sort()

    return isin_dates, sorted_dates


@task
def _write_merged_assets(output_dir: Path, new_assets: list[Asset], member_isins: set[str]) -> None:
    """Build assets.parquet with membership interval columns from all partitions.

    Reads all membership partitions to compute per-ISIN intervals, then joins
    with asset metadata. Each (ISIN, interval) pair becomes one row.

    Args:
        output_dir: Root output directory.
        new_assets: Newly parsed assets to merge in.
        member_isins: Set of ISINs that were members in newly processed dates.
    """
    assets_path = output_dir / "assets.parquet"

    # Build asset metadata lookup from new + existing
    new_df = _dataclasses_to_df(new_assets)
    new_df = new_df.unique(subset=["isin"], keep="first")

    if assets_path.exists():
        existing_df = pl.read_parquet(assets_path)
        # Drop interval columns from existing if present (will be recomputed)
        drop_cols = [c for c in ("first_included", "last_included") if c in existing_df.columns]
        if drop_cols:
            existing_df = existing_df.drop(drop_cols)
        existing_df = existing_df.unique(subset=["isin"], keep="first")
        metadata_df = pl.concat([existing_df, new_df], how="diagonal_relaxed").unique(subset=["isin"], keep="first")
    else:
        metadata_df = new_df

    # Build intervals from all membership partitions on disk
    isin_dates, all_dates = _build_isin_membership(output_dir)

    log = get_run_logger()

    if not all_dates:
        # No membership data yet — write metadata without intervals
        _write_atomic(metadata_df, assets_path)
        log.info("Wrote %d assets (no membership data for intervals)", len(metadata_df))
        return

    # Build interval rows
    rows: list[dict[str, date | str]] = []
    for isin, member_dates in isin_dates.items():
        for first, last in _compute_intervals(member_dates, all_dates):
            rows.append({"isin": isin, "first_included": first, "last_included": last})

    if not rows:
        _write_atomic(metadata_df, assets_path)
        log.info("Wrote %d assets (no member intervals found)", len(metadata_df))
        return

    intervals_df = pl.DataFrame(rows).with_columns(
        pl.col("first_included").cast(pl.Date),
        pl.col("last_included").cast(pl.Date),
    )

    # Join metadata with intervals — one row per (ISIN, interval)
    combined = intervals_df.join(metadata_df, on="isin", how="left")

    _write_atomic(combined, assets_path)
    log.info("Wrote %d asset rows (%d unique ISINs)", len(combined), combined["isin"].n_unique())


@flow(log_prints=True)
async def sync(
    output_dir: Path | str = "output/STOXX600",
    cache_dir: Path | str = "cache/stoxx",
) -> list[date]:
    """Run incremental sync: download missing periods, compute membership, upload.

    Args:
        output_dir: Local directory for Parquet output.
        cache_dir: Local directory for downloaded selection list files.

    Returns:
        List of newly processed review dates.
    """
    log = get_run_logger()
    storage = from_env()
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    today = date.today()

    remote_dates = storage.list_review_dates("STOXX600/membership")
    all_periods = get_periods(START_DATE, today)

    remote_months = {(d.year, d.month) for d in remote_dates}
    missing_periods = [(y, m) for y, m in all_periods if (y, m) not in remote_months]

    if not missing_periods:
        log.info("All periods already synced, nothing to do.")
        return []

    log.info("Found %d missing periods to process", len(missing_periods))

    result = await download_selection_lists(output_dir=cache_dir, periods=missing_periods)

    if not result.downloaded:
        log.warning("No files downloaded for missing periods")
        return []

    log.info("Downloaded %d files", len(result.downloaded))

    # Parse all downloaded files and group by review_date
    review_date_groups: dict[date, tuple[list, list]] = {}
    for filepath in result.downloaded:
        assets, entries = parse_selection_list(filepath)
        if entries:
            rd = entries[0].review_date
            if rd in review_date_groups:
                existing_assets, existing_entries = review_date_groups[rd]
                existing_assets.extend(assets)
                existing_entries.extend(entries)
            else:
                review_date_groups[rd] = (assets, entries)

    log.info("Parsed %d review dates", len(review_date_groups))

    sorted_dates = sorted(review_date_groups.keys())
    new_dates: list[date] = []
    all_member_isins: set[str] = set()
    all_assets: list[Asset] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Find the most recent remote date for initial prior membership
        prior_date = remote_dates[-1] if remote_dates else None

        for rd in sorted_dates:
            assets, entries = review_date_groups[rd]

            # Get prior membership: local first, then remote, then None (bootstrap)
            prior_membership: set[str] | None = None
            if new_dates:
                prior_membership = _read_local_membership(output_dir, new_dates[-1])
            elif prior_date is not None:
                prior_membership = _download_prior_membership(storage, prior_date, tmp_path)

            membership = compute_membership(entries, prior_membership)
            write_parquet_dataset(assets, entries, membership, output_dir)

            # Accumulate assets and member ISINs for final merged assets file
            all_assets.extend(assets)
            all_member_isins.update(m.isin for m in membership if m.is_member)

            new_dates.append(rd)
            log.info("Processed review date %s", rd)

    # Merge new assets with existing assets.parquet, keeping only members
    _write_merged_assets(output_dir, all_assets, all_member_isins)

    # Upload once at the end
    if new_dates:
        storage.upload_directory(output_dir, "STOXX600")

    log.info("Sync complete: %d new dates processed", len(new_dates))
    return new_dates
