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
    _resolve_isin,
    compute_membership,
    parse_selection_list,
    parse_selection_list_csv,
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
        """Two entries with same rank are ordered by ISIN alphabetically."""
        entries = [
            SelectionListEntry(isin="ISIN_B", review_date=date(2024, 3, 1), ff_mcap=50000.0, rank=1),
            SelectionListEntry(isin="ISIN_A", review_date=date(2024, 3, 1), ff_mcap=50000.0, rank=1),
        ]
        # Add 598 more entries with lower mcap
        for i in range(598):
            entries.append(
                SelectionListEntry(
                    isin=f"ISIN_Z{i:06d}",
                    review_date=date(2024, 3, 1),
                    ff_mcap=40000.0 - i,
                    rank=i + 2,
                )
            )
        result = compute_membership(entries, prior_membership=None)
        # ISIN_A should come before ISIN_B due to alphabetical tiebreak
        isins = [m.isin for m in result]
        assert isins[0] == "ISIN_A"
        assert isins[1] == "ISIN_B"


class TestIntegration:
    """Integration tests using real CSV fixture."""

    def test_every_cycle_has_at_least_600_with_ranks_1_to_600(self, parsed_csv):
        """Every review cycle must have at least 600 entries covering ranks 1-600."""
        _, entries = parsed_csv
        by_date: dict[date, list[SelectionListEntry]] = {}
        for e in entries:
            by_date.setdefault(e.review_date, []).append(e)
        for review_date, cycle_entries in by_date.items():
            assert len(cycle_entries) >= 600, f"{review_date}: only {len(cycle_entries)} entries"
            ranks = {e.rank for e in cycle_entries if e.rank is not None}
            expected = set(range(1, 601))
            missing = expected - ranks
            assert not missing, f"{review_date}: missing ranks {sorted(missing)}"

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


MOCK_PDF_HEADER = ["STOXX Europe 600", "Last Updated: 13.10.2015", "ISIN ..."]
MOCK_PDF_TABLE_PAGE0 = [
    [
        "ISIN",
        "Sedol",
        "RIC",
        "Int.Key",
        "Company Name",
        "Country",
        "Currency",
        "Component",
        "FF Mcap (BEUR)",
        "Rank\n(FINAL)",
        "Rank\n(PREVIOUS\n)",
    ],
    ["CH0038863350", "7123870", "NESN.VX", "461669", "NESTLE", "CH", "CHF", "Large", "214.1", "1", ""],
    ["NL0010273215", "B7DRGX5", "ASML.AS", "443333", "ASML HLDG", "NL", "EUR", "Large", "150.5", "2", "1"],
    ["DE0007164600", "4846288", "SAP.DE", "479831", "SAP", "DE", "EUR", "Large", "120.3", "3", "2"],
]
MOCK_PDF_TABLE_PAGE1 = [
    ["GB0002374006", "0237400", "DGE.L", "039600", "DIAGEO", "GB", "GBP", "Large", "80.2", "4", "3"],
]


def _make_mock_pdf():
    """Create a mock pdfplumber PDF object."""
    from unittest.mock import MagicMock

    page0 = MagicMock()
    page0.extract_text.return_value = "\n".join(MOCK_PDF_HEADER)
    page0.extract_table.return_value = MOCK_PDF_TABLE_PAGE0

    page1 = MagicMock()
    page1.extract_text.return_value = ""
    page1.extract_table.return_value = MOCK_PDF_TABLE_PAGE1

    pdf = MagicMock()
    pdf.pages = [page0, page1]
    return pdf


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


class TestParseSelectionListPdf:
    """Tests for parse_selection_list_pdf with mocked PDF data."""

    def test_parse_returns_assets_and_entries(self, monkeypatch):
        """PDF parsing returns correct number of assets and entries."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        assets, entries = parse_selection_list_pdf(Path("fake.pdf"))
        assert len(assets) == 4
        assert len(entries) == 4

    def test_review_date_extracted(self, monkeypatch):
        """Review date is correctly parsed from PDF header."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        _, entries = parse_selection_list_pdf(Path("fake.pdf"))
        assert entries[0].review_date == date(2015, 10, 13)

    def test_ff_mcap_converted_to_meur(self, monkeypatch):
        """PDF BEUR values are converted to MEUR (multiplied by 1000)."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        _, entries = parse_selection_list_pdf(Path("fake.pdf"))
        rank_1 = next(e for e in entries if e.rank == 1)
        assert rank_1.ff_mcap == 214100.0

    def test_unique_isins_in_assets(self, monkeypatch):
        """No duplicate ISINs in assets list."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        assets, _ = parse_selection_list_pdf(Path("fake.pdf"))
        isins = [a.isin for a in assets]
        assert len(isins) == len(set(isins))

    def test_asset_fields_populated(self, monkeypatch):
        """Assets have non-empty key fields."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        assets, _ = parse_selection_list_pdf(Path("fake.pdf"))
        for asset in assets:
            assert asset.isin
            assert asset.name
            assert asset.country
            assert asset.currency

    def test_multi_page_extraction(self, monkeypatch):
        """Rows from multiple pages are combined."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        assets, entries = parse_selection_list_pdf(Path("fake.pdf"))
        assert any(a.isin == "GB0002374006" for a in assets)
        assert any(e.rank == 4 for e in entries)


class TestParseSelectionList:
    """Tests for the unified parse_selection_list dispatcher."""

    def test_dispatches_to_csv(self, csv_fixture_path):
        """CSV files are dispatched to the CSV parser."""
        _assets, entries = parse_selection_list(csv_fixture_path)
        assert len(entries) == 1862

    def test_dispatches_to_pdf(self, monkeypatch):
        """PDF files are dispatched to the PDF parser."""
        monkeypatch.setitem(
            __import__("sys").modules, "pdfplumber", type("M", (), {"open": staticmethod(lambda _: _make_mock_pdf())})()
        )
        _assets, entries = parse_selection_list(Path("fake.pdf"))
        assert len(entries) == 4

    def test_passes_isin_lookup_to_csv(self, csv_fixture_path):
        """isin_lookup parameter is forwarded to CSV parser."""
        _assets, entries = parse_selection_list(csv_fixture_path, isin_lookup={"FAKE": "RESOLVED"})
        assert len(entries) == 1862


class TestResolveIsin:
    """Tests for ISIN resolution from lookup and fallback."""

    def test_valid_isin_returned_as_is(self):
        """A valid ISIN is returned unchanged regardless of lookup."""
        assert _resolve_isin("CH0038863350", "461669", {"461669": "OTHER"}) == "CH0038863350"

    def test_empty_isin_resolved_from_lookup(self):
        """Empty ISIN is resolved via lookup dict."""
        assert _resolve_isin("", "461669", {"461669": "CH0038863350"}) == "CH0038863350"

    def test_null_isin_resolved_from_lookup(self):
        """Null-string ISIN is resolved via lookup dict."""
        assert _resolve_isin("null", "461669", {"461669": "CH0038863350"}) == "CH0038863350"

    def test_none_string_isin_resolved_from_lookup(self):
        """None-string ISIN is resolved via lookup dict."""
        assert _resolve_isin("None", "461669", {"461669": "CH0038863350"}) == "CH0038863350"

    def test_fallback_to_internal_key(self):
        """When lookup has no entry, falls back to KEY_ prefix."""
        assert _resolve_isin("", "461669", {"OTHER": "XX"}) == "KEY_461669"

    def test_fallback_without_lookup(self):
        """When no lookup provided, falls back to KEY_ prefix."""
        assert _resolve_isin("", "461669", None) == "KEY_461669"

    def test_empty_lookup_falls_back(self):
        """Empty lookup dict falls back to KEY_ prefix."""
        assert _resolve_isin("", "461669", {}) == "KEY_461669"


class TestParseSelectionListCsvWithLookup:
    """Tests for parse_selection_list_csv with isin_lookup parameter."""

    def test_resolves_missing_isins_from_lookup(self, tmp_path):
        """CSV rows with empty ISINs are resolved via isin_lookup."""
        csv_content = (
            "Internal_Key;ISIN;RIC;Instrument_Name;Country;Currency;Sedol;"
            "FF_Mcap_MEUR;Rank_Final;Comment;Creation_Date\n"
            "100001;;RIC1.DE;Company A;DE;EUR;SEDOL1;50000.0;1;;20260601\n"
            "100002;;RIC2.FR;Company B;FR;EUR;SEDOL2;40000.0;2;;20260601\n"
            "100003;;RIC3.GB;Company C;GB;GBP;SEDOL3;30000.0;3;;20260601\n"
        )
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        lookup = {"100001": "DE000AAAA111", "100002": "FR000BBBB222"}
        assets, entries = parse_selection_list_csv(csv_path, isin_lookup=lookup)

        assert len(assets) == 3
        assert len(entries) == 3

        asset_map = {a.internal_key: a for a in assets}
        assert asset_map["100001"].isin == "DE000AAAA111"
        assert asset_map["100002"].isin == "FR000BBBB222"
        assert asset_map["100003"].isin == "KEY_100003"

        entry_isins = {e.isin for e in entries}
        assert "DE000AAAA111" in entry_isins
        assert "FR000BBBB222" in entry_isins
        assert "KEY_100003" in entry_isins

    def test_existing_isins_not_overwritten(self, tmp_path):
        """CSV rows with valid ISINs are not overwritten by lookup."""
        csv_content = (
            "Internal_Key;ISIN;RIC;Instrument_Name;Country;Currency;Sedol;"
            "FF_Mcap_MEUR;Rank_Final;Comment;Creation_Date\n"
            "100001;ORIGINAL_ISIN;RIC1.DE;Company A;DE;EUR;;50000.0;1;;20260601\n"
        )
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(csv_content)

        lookup = {"100001": "LOOKUP_ISIN"}
        assets, entries = parse_selection_list_csv(csv_path, isin_lookup=lookup)

        assert assets[0].isin == "ORIGINAL_ISIN"
        assert entries[0].isin == "ORIGINAL_ISIN"
