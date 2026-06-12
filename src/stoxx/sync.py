"""Incremental sync orchestrator for STOXX 600 data."""

import logging
import tempfile
from datetime import date
from pathlib import Path

import polars as pl
from prefect import flow, get_run_logger, task

from stoxx.download import START_DATE, download_selection_lists, get_periods
from stoxx.extract import Asset, compute_membership, parse_selection_list
from stoxx.load import _dataclasses_to_df, _write_atomic, write_parquet_dataset
from stoxx.storage import Storage, from_env
from yukka import report_unresolved_assets, resolve_yukka_ids

logger = logging.getLogger(__name__)

_download_task = task(download_selection_lists, retries=3, retry_delay_seconds=5)  # type: ignore[call-overload]


def _build_isin_lookup(storage: Storage, output_dir: Path, tmp_path: Path) -> dict[str, str]:
    """Build an internal_key→ISIN lookup from existing assets data.

    Args:
        storage: Storage backend for downloading remote files.
        output_dir: Local output directory that may contain assets.parquet.
        tmp_path: Temporary directory for downloaded files.

    Returns:
        Dict mapping internal_key to ISIN. Empty dict if no assets data exists.
    """
    assets_path = output_dir / "assets.parquet"
    if not assets_path.exists():
        remote_key = "STOXX600/assets.parquet"
        local_path = tmp_path / "assets_lookup.parquet"
        try:
            storage.download_file(remote_key, local_path)
            assets_path = local_path
        except Exception:
            logger.info("No existing assets file found for ISIN lookup")
            return {}

    df = pl.read_parquet(assets_path)
    if "internal_key" not in df.columns or "isin" not in df.columns:
        return {}

    lookup: dict[str, str] = {}
    for row in df.select(["internal_key", "isin"]).iter_rows():
        key, isin = row
        if key and isin and str(isin).strip() and not str(isin).startswith("KEY_"):
            lookup[str(key).strip()] = str(isin).strip()
    return lookup


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
def _build_merged_assets(output_dir: Path, new_assets: list[Asset], member_isins: set[str]) -> pl.DataFrame:
    """Build merged assets DataFrame with membership interval columns from all partitions.

    Reads all membership partitions to compute per-ISIN intervals, then joins
    with asset metadata. Each (ISIN, interval) pair becomes one row.

    Args:
        output_dir: Root output directory.
        new_assets: Newly parsed assets to merge in.
        member_isins: Set of ISINs that were members in newly processed dates.

    Returns:
        Merged DataFrame with asset metadata and membership intervals.
    """
    assets_path = output_dir / "assets.parquet"

    # Build asset metadata lookup from new + existing
    new_df = _dataclasses_to_df(new_assets)
    new_df = new_df.unique(subset=["isin"], keep="first")

    if assets_path.exists():
        existing_df = pl.read_parquet(assets_path)
        # Drop interval/yukka columns from existing if present (will be recomputed)
        drop_cols = [c for c in ("first_included", "last_included", "yukka_id") if c in existing_df.columns]
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
        log.info("Built %d assets (no membership data for intervals)", len(metadata_df))
        return metadata_df

    # Build interval rows
    rows: list[dict[str, date | str]] = []
    for isin, member_dates in isin_dates.items():
        for first, last in _compute_intervals(member_dates, all_dates):
            rows.append({"isin": isin, "first_included": first, "last_included": last})

    if not rows:
        log.info("Built %d assets (no member intervals found)", len(metadata_df))
        return metadata_df

    intervals_df = pl.DataFrame(rows).with_columns(
        pl.col("first_included").cast(pl.Date),
        pl.col("last_included").cast(pl.Date),
    )

    # Join metadata with intervals — one row per (ISIN, interval)
    combined = intervals_df.join(metadata_df, on="isin", how="left")

    log.info("Built %d asset rows (%d unique ISINs)", len(combined), combined["isin"].n_unique())
    return combined


@task
def _build_ranking_table(output_dir: Path) -> pl.DataFrame:
    """Build a wide-format ranking table with RICs as columns and daily dates as rows.

    Reads entries and membership partitions to produce a forward-filled daily
    ranking view. Members get their rank; non-members get null. Uses 0 as a
    sentinel during forward-fill to correctly propagate exits.

    Args:
        output_dir: Root output directory containing entries and membership subdirectories.

    Returns:
        DataFrame with a ``date`` column and one column per RIC containing forward-filled ranks.
    """
    entries_dir = output_dir / "entries"
    membership_dir = output_dir / "membership"

    if not entries_dir.exists() or not membership_dir.exists():
        return pl.DataFrame({"date": []}).cast({"date": pl.Date})

    # Build ISIN->RIC lookup from assets for entries that lack a ric column
    isin_to_ric: dict[str, str] = {}
    assets_path = output_dir / "assets.parquet"
    if assets_path.exists():
        assets_df = pl.read_parquet(assets_path)
        if "isin" in assets_df.columns and "ric" in assets_df.columns:
            for row in assets_df.select(["isin", "ric"]).iter_rows():
                isin_to_ric[row[0]] = row[1]

    all_known_rics: set[str] = set()
    long_rows: list[dict] = []

    review_dates: list[date] = []
    for partition in sorted(entries_dir.iterdir()):
        if not partition.name.startswith("review_date="):
            continue
        rd = date.fromisoformat(partition.name.split("=", 1)[1])
        review_dates.append(rd)

    for rd in review_dates:
        entries_path = entries_dir / f"review_date={rd}" / "data.parquet"
        membership_path = membership_dir / f"review_date={rd}" / "data.parquet"

        if not entries_path.exists() or not membership_path.exists():
            continue

        entries_df = pl.read_parquet(entries_path)
        membership_df = pl.read_parquet(membership_path)

        # Join RIC from assets if entries lack a ric column
        if "ric" not in entries_df.columns:
            if not isin_to_ric:
                continue
            ric_series = entries_df["isin"].map_elements(lambda x: isin_to_ric.get(x), return_dtype=pl.Utf8)
            entries_df = entries_df.with_columns(ric_series.alias("ric"))

        entries_df = entries_df.filter(pl.col("ric").is_not_null())

        # Get member ISINs
        member_isins = set(membership_df.filter(pl.col("is_member"))["isin"].to_list())

        # Build ric->rank for members, deduplicate
        ric_rank: dict[str, int] = {}
        for row in entries_df.iter_rows(named=True):
            ric = row["ric"]
            if ric in ric_rank:
                continue
            all_known_rics.add(ric)
            if row["isin"] in member_isins:
                ric_rank[ric] = row["rank"]

        # Members get their rank; all other known RICs get 0 (sentinel for exit)
        for ric in all_known_rics:
            rank = ric_rank.get(ric, 0)
            long_rows.append({"date": rd, "ric": ric, "rank": rank})

    if not long_rows:
        return pl.DataFrame({"date": []}).cast({"date": pl.Date})

    long_df = pl.DataFrame(long_rows).with_columns(pl.col("date").cast(pl.Date))

    # Pivot to wide format: rows=date, columns=RIC, values=rank
    wide_df = long_df.pivot(on="ric", index="date", values="rank")

    # Expand to daily date range
    min_date: date = wide_df["date"].min()  # type: ignore[assignment]
    max_date = date.today()
    daily_dates = pl.DataFrame({"date": pl.date_range(min_date, max_date, eager=True)})

    # Join with daily range, sort, forward-fill
    result = daily_dates.join(wide_df, on="date", how="left").sort("date")
    ric_cols = [c for c in result.columns if c != "date"]
    result = result.with_columns(pl.col(c).forward_fill() for c in ric_cols)

    # Replace sentinel 0 with null
    result = result.with_columns(pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c) for c in ric_cols)

    return result


@task
def _validate_ranking_table(ranking_df: pl.DataFrame, review_dates: list[date]) -> None:
    """Check that each review date row in the ranking table has ranks covering 1-100.

    Args:
        ranking_df: Wide-format ranking DataFrame (date column + RIC columns).
        review_dates: Review dates that should be validated.
    """
    log = get_run_logger()
    ric_cols = [c for c in ranking_df.columns if c != "date"]

    if not ric_cols or ranking_df.is_empty():
        log.warning("Ranking table is empty, skipping validation")
        return

    for rd in review_dates:
        row = ranking_df.filter(pl.col("date") == rd)
        if row.is_empty():
            log.warning("Ranking validation: no row for review date %s", rd)
            continue

        ranks = set()
        for col in ric_cols:
            val = row[col][0]
            if val is not None:
                ranks.add(int(val))

        expected = set(range(1, 101))
        missing = expected - ranks
        if missing:
            log.warning(
                "Ranking validation FAILED for %s: missing %d ranks in 1-100 (e.g. %s). Only %d distinct ranks found.",
                rd,
                len(missing),
                sorted(missing)[:10],
                len(ranks),
            )
        else:
            log.info("Ranking validation passed for %s: ranks 1-100 all present (%d total ranks)", rd, len(ranks))


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

    result = await _download_task(output_dir=cache_dir, periods=missing_periods)

    if not result.downloaded:
        log.warning("No files downloaded for missing periods")
        return []

    log.info("Downloaded %d files", len(result.downloaded))

    new_dates: list[date] = []
    all_member_isins: set[str] = set()
    all_assets: list[Asset] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build ISIN lookup from existing assets for resolving missing ISINs
        isin_lookup = _build_isin_lookup(storage, output_dir, tmp_path)
        if isin_lookup:
            log.info("Built ISIN lookup with %d entries from historical assets", len(isin_lookup))

        # Parse all downloaded files and group by review_date
        review_date_groups: dict[date, tuple[list, list]] = {}
        for filepath in result.downloaded:
            assets, entries = parse_selection_list(filepath, isin_lookup=isin_lookup)
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

    # Merge new assets, enrich with Yukka IDs, and write
    assets_path = output_dir / "assets.parquet"
    merged_df = _build_merged_assets(output_dir, all_assets, all_member_isins)
    enriched_df = resolve_yukka_ids(merged_df)
    _write_atomic(enriched_df, assets_path)
    report_unresolved_assets(assets_path)

    # Build, validate, and write ranking table
    ranking_df = _build_ranking_table(output_dir)
    _validate_ranking_table(ranking_df, new_dates)
    _write_atomic(ranking_df, output_dir / "ranking.parquet")

    # Upload once at the end
    if new_dates:
        storage.upload_directory(output_dir, "STOXX600")

    log.info("Sync complete: %d new dates processed", len(new_dates))
    return new_dates
