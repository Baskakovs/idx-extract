"""Tests for STOXX selection list extraction and membership computation."""

from __future__ import annotations

import logging
from datetime import date

from stoxx.extract import (
    Asset,
    EntryReason,
    SelectionListEntry,
    compute_membership,
)


def _make_entries(n: int, ff_mcap_start: float = 100000.0) -> list[SelectionListEntry]:
    """Create n synthetic entries with descending FF Mcap and sequential ISINs."""
    return [
        SelectionListEntry(
            isin=f"ISIN{i:08d}",
            review_date=date(2024, 3, 1),
            ff_mcap=ff_mcap_start - i,
            rank=i + 1,
        )
        for i in range(n)
    ]


class TestParseSelectionListCSV:
    """Tests for parse_selection_list_csv with real CSV fixture."""

    def test_parse_returns_expected_row_count(self, parsed_csv):
        """CSV has 1862 data rows."""
        _, entries = parsed_csv
        assert len(entries) == 1862

    def test_parse_handles_semicolon_delimiter(self, parsed_csv):
        """Rank-1 asset has RIC NOVOb.CO."""
        _, entries = parsed_csv
        rank_1 = [e for e in entries if e.rank == 1]
        assert len(rank_1) == 1
        # Verify via assets that rank-1 ISIN maps to NOVOb.CO
        assets, _ = parsed_csv
        rank_1_isin = rank_1[0].isin
        asset = next(a for a in assets if a.isin == rank_1_isin)
        assert asset.ric == "NOVOb.CO"

    def test_parse_handles_encoding_and_floats(self, parsed_csv):
        """L'OREAL name round-trips and ff_mcap is a positive float."""
        assets, entries = parsed_csv
        loreal = [a for a in assets if "OREAL" in a.name.upper()]
        assert len(loreal) >= 1
        loreal_entry = next(e for e in entries if e.isin == loreal[0].isin)
        assert isinstance(loreal_entry.ff_mcap, float)
        assert loreal_entry.ff_mcap > 0

    def test_parse_yields_unique_isins(self, parsed_csv):
        """No duplicate ISINs in assets list."""
        assets, _ = parsed_csv
        isins = [a.isin for a in assets]
        assert len(isins) == len(set(isins))

    def test_asset_and_entry_separation(self, parsed_csv):
        """Asset has no ff_mcap/rank attrs; all entry ISINs exist in asset set."""
        assets, entries = parsed_csv
        assert not hasattr(Asset, "ff_mcap")
        assert not hasattr(Asset, "rank")
        asset_isins = {a.isin for a in assets}
        entry_isins = {e.isin for e in entries}
        assert entry_isins.issubset(asset_isins)


class TestComputeMembership:
    """Tests for compute_membership with synthetic fixtures."""

    def test_top_550_all_automatic_members(self):
        """First 550 entries get TOP_550 reason."""
        entries = _make_entries(700)
        result = compute_membership(entries, prior_membership=set())
        top_550 = [m for m in result if m.entry_reason == EntryReason.TOP_550]
        assert len(top_550) == 550

    def test_buffer_retains_prior_members(self):
        """Prior members in buffer zone (551-750) are retained."""
        entries = _make_entries(800)
        # Put 11 prior members in buffer zone (positions 551-561)
        prior = {f"ISIN{i:08d}" for i in range(550, 561)}
        result = compute_membership(entries, prior_membership=prior)
        retained = [m for m in result if m.entry_reason == EntryReason.BUFFER_RETAINED]
        assert len(retained) == 11

    def test_fill_to_600_when_buffer_insufficient(self):
        """When buffer retains fewer than 50, remaining slots are filled."""
        entries = _make_entries(800)
        prior = {f"ISIN{i:08d}" for i in range(550, 560)}  # 10 in buffer
        result = compute_membership(entries, prior_membership=prior)
        filled = [m for m in result if m.entry_reason == EntryReason.FILL_TO_600]
        assert len(filled) == 40  # 600 - 550 - 10 = 40

    def test_output_is_exactly_600(self):
        """Result always contains exactly 600 members."""
        entries = _make_entries(1000)
        result = compute_membership(entries, prior_membership=set())
        assert len(result) == 600

    def test_bootstrap_mode_takes_top_600(self, caplog):
        """Bootstrap mode (prior=None) takes top 600 with BOOTSTRAP reason and logs warning."""
        entries = _make_entries(800)
        with caplog.at_level(logging.WARNING):
            result = compute_membership(entries, prior_membership=None)
        assert len(result) == 600
        assert all(m.entry_reason == EntryReason.BOOTSTRAP for m in result)
        assert "Bootstrap mode" in caplog.text

    def test_membership_is_deterministic(self):
        """Same inputs produce identical outputs."""
        entries = _make_entries(800)
        prior = {f"ISIN{i:08d}" for i in range(550, 560)}
        result1 = compute_membership(entries, prior_membership=prior)
        result2 = compute_membership(entries, prior_membership=prior)
        assert result1 == result2

    def test_ties_are_handled_consistently(self):
        """Two entries with same FF Mcap are ordered by ISIN alphabetically."""
        entries = [
            SelectionListEntry(isin="ISIN_B", review_date=date(2024, 3, 1), ff_mcap=50000.0, rank=1),
            SelectionListEntry(isin="ISIN_A", review_date=date(2024, 3, 1), ff_mcap=50000.0, rank=2),
        ]
        # Add 598 more entries with lower mcap
        for i in range(598):
            entries.append(
                SelectionListEntry(
                    isin=f"ISIN_Z{i:06d}",
                    review_date=date(2024, 3, 1),
                    ff_mcap=40000.0 - i,
                    rank=i + 3,
                )
            )
        result = compute_membership(entries, prior_membership=None)
        # ISIN_A should come before ISIN_B due to alphabetical tiebreak
        isins = [m.isin for m in result]
        assert isins[0] == "ISIN_A"
        assert isins[1] == "ISIN_B"


class TestIntegration:
    """Integration tests using real CSV fixture."""

    def test_full_pipeline_bootstrap(self, parsed_csv):
        """Parse real CSV -> bootstrap membership -> exactly 600."""
        _, entries = parsed_csv
        result = compute_membership(entries, prior_membership=None)
        assert len(result) == 600
        assert all(m.entry_reason == EntryReason.BOOTSTRAP for m in result)

    def test_full_pipeline_with_prior(self, parsed_csv):
        """Bootstrap -> use as prior -> recompute -> 600 with mixed reasons."""
        _, entries = parsed_csv
        bootstrap = compute_membership(entries, prior_membership=None)
        prior = {m.isin for m in bootstrap}
        result = compute_membership(entries, prior_membership=prior)
        assert len(result) == 600
        reasons = {m.entry_reason for m in result}
        assert EntryReason.TOP_550 in reasons
