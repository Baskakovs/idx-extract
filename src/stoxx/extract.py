"""Extract STOXX selection list data and compute index membership."""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Asset:
    """Time-invariant security identifiers."""

    isin: str
    internal_key: str
    ric: str
    name: str
    country: str
    currency: str
    sedol: str | None = None


@dataclass(frozen=True)
class SelectionListEntry:
    """One row per asset per review date."""

    isin: str
    review_date: date
    ff_mcap: float | None
    rank: int | None
    comment: str | None = None


class EntryReason(enum.Enum):
    """Why a stock was selected for membership."""

    TOP_550 = "top_550"
    BUFFER_RETAINED = "buffer_retained"
    FILL_TO_600 = "fill_to_600"
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True)
class IndexMembership:
    """Membership result for a single ISIN."""

    isin: str
    is_member: bool
    entry_reason: EntryReason


def _normalize_column_name(name: str) -> str:
    """Normalize column name to lowercase with underscores."""
    normalized = name.lower().strip()
    normalized = re.sub(r"[()\[\]]+", "", normalized)
    normalized = re.sub(r"[\s\-\.]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def parse_selection_list_csv(filepath: Path) -> tuple[list[Asset], list[SelectionListEntry]]:
    """Parse a STOXX selection list CSV into Assets and SelectionListEntries.

    Args:
        filepath: Path to a semicolon-delimited STOXX selection list CSV.

    Returns:
        A tuple of (assets, entries) where assets has one per unique ISIN.
    """
    df = pl.read_csv(filepath, separator=";", infer_schema_length=10000)
    df = df.rename({col: _normalize_column_name(col) for col in df.columns})

    # Extract review_date from creation_date column (YYYYMMDD -> date)
    creation_date_str = str(df["creation_date"][0])
    review_date = date(int(creation_date_str[:4]), int(creation_date_str[4:6]), int(creation_date_str[6:8]))

    # Build assets: one per unique ISIN
    asset_df = df.unique(subset=["isin"], keep="first")
    assets = []
    for row in asset_df.to_dicts():
        sedol_val = row.get("sedol")
        sedol = str(sedol_val).strip() if sedol_val is not None and str(sedol_val).strip() else None
        assets.append(
            Asset(
                isin=str(row["isin"]).strip(),
                internal_key=str(row["internal_key"]).strip(),
                ric=str(row["ric"]).strip(),
                name=str(row["instrument_name"]).strip(),
                country=str(row["country"]).strip(),
                currency=str(row["currency"]).strip(),
                sedol=sedol,
            )
        )

    # Build entries: one per row
    entries = []
    for row in df.to_dicts():
        rank_val = row.get("rank_final")
        rank = int(rank_val) if rank_val is not None and str(rank_val).strip() != "" else None

        comment_val = row.get("comment")
        comment = str(comment_val).strip() if comment_val is not None and str(comment_val).strip() else None

        ff_mcap_val = row.get("ff_mcap_meur")
        ff_mcap = float(ff_mcap_val) if ff_mcap_val is not None and str(ff_mcap_val).strip() != "" else None
        entries.append(
            SelectionListEntry(
                isin=str(row["isin"]).strip(),
                review_date=review_date,
                ff_mcap=ff_mcap,
                rank=rank,
                comment=comment,
            )
        )

    return assets, entries


def compute_membership(
    entries: list[SelectionListEntry],
    prior_membership: set[str] | None,
) -> list[IndexMembership]:
    """Compute STOXX Europe 600 index membership using the buffer rule.

    Args:
        entries: Selection list entries (may include unranked entries).
        prior_membership: Set of ISINs that were members in the prior review,
            or None for bootstrap mode (first review).

    Returns:
        List of exactly 600 IndexMembership results.
    """
    # Filter to ranked entries, sort by FF Mcap DESC then ISIN ASC for deterministic tiebreaker
    ranked = [e for e in entries if e.rank is not None]
    ranked.sort(key=lambda e: (-(e.ff_mcap or 0), e.isin))

    if prior_membership is None:
        logger.warning("Bootstrap mode: no prior membership provided, taking top 600 by FF Mcap")
        return [IndexMembership(isin=e.isin, is_member=True, entry_reason=EntryReason.BOOTSTRAP) for e in ranked[:600]]

    members: list[IndexMembership] = []

    # Positions 1-550: automatic members
    for entry in ranked[:550]:
        members.append(IndexMembership(isin=entry.isin, is_member=True, entry_reason=EntryReason.TOP_550))

    # Positions 551-750: retain prior members (buffer zone)
    buffer_zone = ranked[550:750]
    for entry in buffer_zone:
        if len(members) >= 600:
            break
        if entry.isin in prior_membership:
            members.append(IndexMembership(isin=entry.isin, is_member=True, entry_reason=EntryReason.BUFFER_RETAINED))

    # Fill remaining slots to 600 from largest remaining by FF Mcap
    member_isins = {m.isin for m in members}
    remaining = [e for e in ranked if e.isin not in member_isins]
    for entry in remaining:
        if len(members) >= 600:
            break
        members.append(IndexMembership(isin=entry.isin, is_member=True, entry_reason=EntryReason.FILL_TO_600))

    return members
