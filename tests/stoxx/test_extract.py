"""Tests for STOXX selection list extraction and membership computation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from stoxx.extract import (
    Asset,
    EntryReason,
    SelectionListEntry,
    _parse_pdf_date,
    compute_membership,
    parse_selection_list,
    parse_selection_list_pdf,
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


PDF_FIXTURE = Path(__file__).parent.parent.parent / "cache" / "stoxx" / "sl_sxxp_201510.pdf"


class TestParsePdfDate:
    """Tests for _parse_pdf_date helper."""

    def test_yyyymmdd_format(self):
        """Parses YYYYMMDD date from header."""
        lines = ["STOXX EUROPE 600", "Last Updated: 20230901"]
        assert _parse_pdf_date(lines) == date(2023, 9, 1)

    def test_dd_mm_yyyy_format(self):
        """Parses DD.MM.YYYY date from header."""
        lines = ["STOXX Europe 600", "Last Updated: 13.10.2015"]
        assert _parse_pdf_date(lines) == date(2015, 10, 13)

    def test_raises_on_missing_date(self):
        """Raises ValueError when no date found."""
        with pytest.raises(ValueError, match="Could not find review date"):
            _parse_pdf_date(["STOXX Europe 600", "No date here"])


@pytest.fixture(scope="module")
def parsed_pdf():
    """Parse the PDF fixture once for all PDF tests."""
    if not PDF_FIXTURE.exists():
        pytest.skip("PDF fixture not available")
    return parse_selection_list_pdf(PDF_FIXTURE)


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason="PDF fixture not available")
class TestParseSelectionListPdf:
    """Tests for parse_selection_list_pdf with real PDF fixture."""

    def test_parse_returns_assets_and_entries(self, parsed_pdf):
        """PDF parsing returns non-empty assets and entries."""
        assets, entries = parsed_pdf
        assert len(assets) > 600
        assert len(entries) > 600

    def test_review_date_extracted(self, parsed_pdf):
        """Review date is correctly parsed from PDF header."""
        _, entries = parsed_pdf
        assert entries[0].review_date == date(2015, 10, 13)

    def test_ff_mcap_converted_to_meur(self, parsed_pdf):
        """PDF BEUR values are converted to MEUR (multiplied by 1000)."""
        _, entries = parsed_pdf
        rank_1 = next(e for e in entries if e.rank == 1)
        assert rank_1.ff_mcap is not None
        assert rank_1.ff_mcap > 100000  # Should be in MEUR range

    def test_unique_isins_in_assets(self, parsed_pdf):
        """No duplicate ISINs in assets list."""
        assets, _ = parsed_pdf
        isins = [a.isin for a in assets]
        assert len(isins) == len(set(isins))

    def test_asset_fields_populated(self, parsed_pdf):
        """Assets have non-empty key fields."""
        assets, _ = parsed_pdf
        for asset in assets[:10]:
            assert asset.isin
            assert asset.name
            assert asset.country
            assert asset.currency


@pytest.mark.skipif(not PDF_FIXTURE.exists(), reason="PDF fixture not available")
class TestParseSelectionList:
    """Tests for the unified parse_selection_list dispatcher."""

    def test_dispatches_to_csv(self, csv_fixture_path):
        """CSV files are dispatched to the CSV parser."""
        _assets, entries = parse_selection_list(csv_fixture_path)
        assert len(entries) == 1862

    def test_dispatches_to_pdf(self, parsed_pdf):
        """PDF files are dispatched to the PDF parser."""
        _assets, entries = parsed_pdf
        assert len(entries) > 600
