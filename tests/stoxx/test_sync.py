"""Tests for the incremental sync orchestrator."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from stoxx.download import DownloadResult
from stoxx.extract import (
    Asset,
    EntryReason,
    IndexMembership,
    SelectionListEntry,
)
from stoxx.storage import Storage
from stoxx.sync import (
    _build_merged_assets,
    _build_ranking_table,
    _compute_intervals,
    _download_prior_membership,
    _read_local_membership,
    sync,
)


class MemoryStorage(Storage):
    """In-memory storage backend for testing."""

    def __init__(self) -> None:
        """Initialize empty storage."""
        self._objects: dict[str, bytes] = {}

    def list_review_dates(self, prefix: str) -> list[date]:
        """List review dates from stored object keys."""
        import re

        if not prefix.endswith("/"):
            prefix = prefix + "/"
        dates: list[date] = []
        seen: set[str] = set()
        for key in self._objects:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            dirname = rest.split("/")[0]
            if dirname in seen:
                continue
            seen.add(dirname)
            match = re.match(r"review_date=(\d{4}-\d{2}-\d{2})", dirname)
            if match:
                dates.append(date.fromisoformat(match.group(1)))
        return sorted(dates)

    def upload_directory(self, local_dir: Path, prefix: str) -> int:
        """Upload files from local directory to memory."""
        count = 0
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{prefix}/{relative}" if prefix else str(relative)
            self._objects[key] = file_path.read_bytes()
            count += 1
        return count

    def download_file(self, key: str, local_path: Path) -> None:
        """Download a file from memory storage."""
        if key not in self._objects:
            msg = f"Key not found: {key}"
            raise FileNotFoundError(msg)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self._objects[key])


def _make_entries(review_date: date, count: int = 700) -> list[SelectionListEntry]:
    """Create synthetic selection list entries for testing."""
    return [
        SelectionListEntry(
            isin=f"ISIN{i:06d}",
            review_date=review_date,
            ff_mcap=float(count - i) * 1000,
            rank=i + 1,
        )
        for i in range(count)
    ]


def _make_assets(count: int = 700) -> list[Asset]:
    """Create synthetic assets for testing."""
    return [
        Asset(
            isin=f"ISIN{i:06d}",
            internal_key=f"KEY{i:06d}",
            ric=f"RIC{i:06d}",
            name=f"Company {i}",
            country="DE",
            currency="EUR",
        )
        for i in range(count)
    ]


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _mock_download(review_date: date, assets: list[Asset], entries: list[SelectionListEntry]):
    """Create a mock for download_selection_lists that returns a fixture CSV path."""

    async def _download(**kwargs):
        # Return the real fixture path so parse_selection_list can parse it
        result = DownloadResult()
        csv_path = FIXTURE_DIR / "slpublic_sxxp_20240301.csv"
        if csv_path.exists():
            result.downloaded.append(csv_path)
        return result

    return _download


class TestReadLocalMembership:
    """Tests for _read_local_membership helper."""

    def test_reads_existing_partition(self, tmp_path):
        """Reads member ISINs from an existing local partition."""
        partition = tmp_path / "membership" / "review_date=2024-03-01"
        partition.mkdir(parents=True)
        df = pl.DataFrame(
            {
                "isin": ["ISIN1", "ISIN2", "ISIN3"],
                "is_member": [True, True, False],
                "entry_reason": ["top_550", "top_550", "top_550"],
            }
        )
        df.write_parquet(partition / "data.parquet")

        result = _read_local_membership(tmp_path, date(2024, 3, 1))

        assert result == {"ISIN1", "ISIN2"}

    def test_returns_none_for_missing(self, tmp_path):
        """Returns None when partition directory does not exist."""
        result = _read_local_membership(tmp_path, date(2024, 3, 1))

        assert result is None


class TestDownloadPriorMembership:
    """Tests for _download_prior_membership helper."""

    def test_downloads_and_reads(self, tmp_path):
        """Downloads membership parquet from storage and reads members."""
        storage = MemoryStorage()

        # Create a membership parquet in memory storage
        df = pl.DataFrame(
            {
                "isin": ["ISIN1", "ISIN2"],
                "is_member": [True, False],
                "entry_reason": ["top_550", "top_550"],
            }
        )
        parquet_path = tmp_path / "temp.parquet"
        df.write_parquet(parquet_path)
        storage._objects["STOXX600/membership/review_date=2024-03-01/data.parquet"] = parquet_path.read_bytes()

        result = _download_prior_membership(storage, date(2024, 3, 1), tmp_path)

        assert result == {"ISIN1"}

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when remote key does not exist."""
        storage = MemoryStorage()

        result = _download_prior_membership(storage, date(2024, 3, 1), tmp_path)

        assert result is None


class TestSync:
    """Tests for the main sync function."""

    @pytest.mark.asyncio
    async def test_no_missing_periods_exits_early(self, tmp_path):
        """Sync returns empty list when all periods are already remote."""
        storage = MemoryStorage()

        # Pre-populate storage with all periods up to a known date
        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.date") as mock_date,
        ):
            mock_date.today.return_value = date(2024, 3, 15)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            # Add all periods as existing in remote
            from stoxx.download import START_DATE, get_periods

            all_periods = get_periods(START_DATE, date(2024, 3, 15))
            for year, month in all_periods:
                key = f"STOXX600/membership/review_date={date(year, month, 1)}/data.parquet"
                storage._objects[key] = b"fake"

            result = await sync(output_dir=tmp_path / "output", cache_dir=tmp_path / "cache")

        assert result == []

    @pytest.mark.asyncio
    async def test_sync_processes_new_dates(self, tmp_path):
        """Sync downloads, parses, computes membership, and uploads for missing periods."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        rd = date(2024, 3, 1)
        assets = _make_assets()
        entries = _make_entries(rd)

        async def mock_download(**kwargs):
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd]
        assert any("membership" in k for k in storage._objects)

    @pytest.mark.asyncio
    async def test_sync_bootstrap_no_prior(self, tmp_path):
        """First sync with no remote data uses bootstrap mode (no prior membership)."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        rd = date(2024, 3, 1)
        assets = _make_assets()
        entries = _make_entries(rd)

        async def mock_download(**kwargs):
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch("stoxx.sync.compute_membership") as mock_compute,
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            mock_compute.return_value = [
                IndexMembership(isin=f"ISIN{i:06d}", is_member=True, entry_reason=EntryReason.BOOTSTRAP)
                for i in range(600)
            ]
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        mock_compute.assert_called_once()
        _, kwargs = mock_compute.call_args
        if not kwargs:
            args = mock_compute.call_args[0]
            assert args[1] is None
        assert result == [rd]

    @pytest.mark.asyncio
    async def test_sync_chains_prior_membership(self, tmp_path):
        """When processing multiple dates, prior membership chains from previous result."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        rd1 = date(2024, 3, 1)
        rd2 = date(2024, 6, 1)
        assets = _make_assets()
        entries1 = _make_entries(rd1)
        entries2 = _make_entries(rd2)

        async def mock_download(**kwargs):
            result = DownloadResult()
            for name in ["a.csv", "b.csv"]:
                csv_path = cache_dir / name
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_path.write_text("placeholder")
                result.downloaded.append(csv_path)
            return result

        parse_results = iter([(assets, entries1), (assets, entries2)])

        def mock_parse(filepath):
            return next(parse_results)

        compute_calls = []

        def mock_compute(entries, prior_membership):
            compute_calls.append(prior_membership)
            return [
                IndexMembership(isin=e.isin, is_member=True, entry_reason=EntryReason.BOOTSTRAP) for e in entries[:600]
            ]

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3), (2024, 6)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", side_effect=mock_parse),
            patch("stoxx.sync.compute_membership", side_effect=mock_compute),
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd1, rd2]
        assert compute_calls[0] is None
        assert compute_calls[1] is not None
        assert isinstance(compute_calls[1], set)

    @pytest.mark.asyncio
    async def test_sync_returns_new_dates(self, tmp_path):
        """Sync returns the list of newly processed dates."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        rd = date(2024, 3, 1)
        assets = _make_assets()
        entries = _make_entries(rd)

        async def mock_download(**kwargs):
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        assert len(result) == 1
        assert result[0] == rd

    @pytest.mark.asyncio
    async def test_sync_skips_existing_dates(self, tmp_path):
        """Periods already in remote storage are not re-downloaded."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        # Mark 2024-03 as already existing with valid parquet data
        prior_df = pl.DataFrame(
            {
                "isin": [f"ISIN{i:06d}" for i in range(600)],
                "is_member": [True] * 600,
                "entry_reason": ["bootstrap"] * 600,
            }
        )
        import io

        buf = io.BytesIO()
        prior_df.write_parquet(buf)
        storage._objects["STOXX600/membership/review_date=2024-03-01/data.parquet"] = buf.getvalue()

        rd = date(2024, 6, 1)
        assets = _make_assets()
        entries = _make_entries(rd)

        async def mock_download(**kwargs):
            periods = kwargs.get("periods", [])
            assert (2024, 3) not in periods
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3), (2024, 6)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd]


class TestComputeIntervals:
    """Tests for _compute_intervals helper."""

    def test_single_continuous_span(self):
        """Consecutive dates produce a single interval."""
        all_dates = [date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)]
        member_dates = [date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)]

        result = _compute_intervals(member_dates, all_dates)

        assert result == [(date(2024, 3, 1), date(2024, 9, 1))]

    def test_gap_produces_two_intervals(self):
        """A gap in the middle produces two intervals."""
        all_dates = [
            date(2024, 3, 1),
            date(2024, 6, 1),
            date(2024, 9, 1),
            date(2024, 12, 1),
            date(2025, 3, 1),
        ]
        member_dates = [date(2024, 3, 1), date(2024, 6, 1), date(2024, 12, 1), date(2025, 3, 1)]

        result = _compute_intervals(member_dates, all_dates)

        assert result == [
            (date(2024, 3, 1), date(2024, 6, 1)),
            (date(2024, 12, 1), date(2025, 3, 1)),
        ]

    def test_multiple_reentries(self):
        """Multiple gaps produce multiple intervals."""
        all_dates = [
            date(2016, 2, 1),
            date(2016, 5, 1),
            date(2016, 8, 1),
            date(2016, 11, 1),
            date(2017, 2, 1),
            date(2017, 5, 1),
            date(2017, 8, 1),
        ]
        member_dates = [
            date(2016, 2, 1),
            date(2016, 5, 1),
            date(2016, 8, 1),
            date(2017, 5, 1),
            date(2017, 8, 1),
        ]

        result = _compute_intervals(member_dates, all_dates)

        assert result == [
            (date(2016, 2, 1), date(2016, 8, 1)),
            (date(2017, 5, 1), date(2017, 8, 1)),
        ]

    def test_single_date(self):
        """A single member date produces an interval with first == last."""
        all_dates = [date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)]
        member_dates = [date(2024, 6, 1)]

        result = _compute_intervals(member_dates, all_dates)

        assert result == [(date(2024, 6, 1), date(2024, 6, 1))]

    def test_empty_member_dates(self):
        """Empty member dates return no intervals."""
        all_dates = [date(2024, 3, 1), date(2024, 6, 1)]

        result = _compute_intervals([], all_dates)

        assert result == []


class TestBuildMergedAssetsIntervals:
    """Tests for _build_merged_assets with interval columns."""

    def test_intervals_appear_in_output(self, tmp_path):
        """Assets parquet contains first_included and last_included columns."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        for rd, isins in [
            (date(2024, 3, 1), ["ISIN001", "ISIN002"]),
            (date(2024, 6, 1), ["ISIN001", "ISIN003"]),
        ]:
            partition = output_dir / "membership" / f"review_date={rd}"
            partition.mkdir(parents=True)
            df = pl.DataFrame(
                {
                    "isin": isins,
                    "is_member": [True] * len(isins),
                    "entry_reason": ["top_550"] * len(isins),
                }
            )
            df.write_parquet(partition / "data.parquet")

        assets = [
            Asset(isin="ISIN001", internal_key="K1", ric="R1", name="Co1", country="DE", currency="EUR"),
            Asset(isin="ISIN002", internal_key="K2", ric="R2", name="Co2", country="FR", currency="EUR"),
            Asset(isin="ISIN003", internal_key="K3", ric="R3", name="Co3", country="GB", currency="GBP"),
        ]

        result = _build_merged_assets(output_dir, assets, {"ISIN001", "ISIN002", "ISIN003"})
        assert "first_included" in result.columns
        assert "last_included" in result.columns

        # ISIN001 is in both dates (contiguous) -> 1 row
        isin001 = result.filter(pl.col("isin") == "ISIN001")
        assert len(isin001) == 1
        assert isin001["first_included"][0] == date(2024, 3, 1)
        assert isin001["last_included"][0] == date(2024, 6, 1)

        # ISIN002 only in first date -> 1 row
        isin002 = result.filter(pl.col("isin") == "ISIN002")
        assert len(isin002) == 1
        assert isin002["first_included"][0] == date(2024, 3, 1)
        assert isin002["last_included"][0] == date(2024, 3, 1)

        # ISIN003 only in second date -> 1 row
        isin003 = result.filter(pl.col("isin") == "ISIN003")
        assert len(isin003) == 1

        # Total: 3 rows
        assert len(result) == 3

    def test_gap_creates_multiple_rows(self, tmp_path):
        """An ISIN with a gap in membership gets multiple rows."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        dates_list = [date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)]
        membership_data = {
            dates_list[0]: ["ISIN001"],
            dates_list[1]: [],
            dates_list[2]: ["ISIN001"],
        }

        for rd, isins in membership_data.items():
            partition = output_dir / "membership" / f"review_date={rd}"
            partition.mkdir(parents=True)
            all_isins = isins if isins else ["ISIN_OTHER"]
            is_member = [True] * len(isins) + [False] * (len(all_isins) - len(isins))
            df = pl.DataFrame(
                {
                    "isin": all_isins,
                    "is_member": is_member,
                    "entry_reason": ["top_550"] * len(all_isins),
                }
            )
            df.write_parquet(partition / "data.parquet")

        assets = [
            Asset(isin="ISIN001", internal_key="K1", ric="R1", name="Co1", country="DE", currency="EUR"),
        ]

        result = _build_merged_assets(output_dir, assets, {"ISIN001"})
        isin001 = result.filter(pl.col("isin") == "ISIN001")
        assert len(isin001) == 2
        assert isin001["first_included"][0] == date(2024, 3, 1)
        assert isin001["last_included"][0] == date(2024, 3, 1)
        assert isin001["first_included"][1] == date(2024, 9, 1)
        assert isin001["last_included"][1] == date(2024, 9, 1)


def _write_entries_partition(output_dir: Path, review_date: date, entries: list[dict]) -> None:
    """Write an entries partition with ric, isin, rank columns."""
    partition = output_dir / "entries" / f"review_date={review_date}"
    partition.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(entries)
    df.write_parquet(partition / "data.parquet")


def _write_membership_partition(output_dir: Path, review_date: date, membership: list[dict]) -> None:
    """Write a membership partition with isin, is_member, entry_reason columns."""
    partition = output_dir / "membership" / f"review_date={review_date}"
    partition.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(membership)
    df.write_parquet(partition / "data.parquet")


class TestBuildRankingTable:
    """Tests for _build_ranking_table."""

    def test_basic_ranking_table(self, tmp_path):
        """Forward-fills ranks correctly across two review dates with entries/exits."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        rd1 = date(2024, 3, 1)
        rd2 = date(2024, 6, 1)

        # rd1: A (rank 1, member), B (rank 2, member), C (rank 3, not member)
        _write_entries_partition(
            output_dir,
            rd1,
            [
                {"isin": "ISIN_A", "ric": "RIC_A", "rank": 1, "ff_mcap": 1000.0},
                {"isin": "ISIN_B", "ric": "RIC_B", "rank": 2, "ff_mcap": 900.0},
                {"isin": "ISIN_C", "ric": "RIC_C", "rank": 3, "ff_mcap": 800.0},
            ],
        )
        _write_membership_partition(
            output_dir,
            rd1,
            [
                {"isin": "ISIN_A", "is_member": True, "entry_reason": "bootstrap"},
                {"isin": "ISIN_B", "is_member": True, "entry_reason": "bootstrap"},
                {"isin": "ISIN_C", "is_member": False, "entry_reason": "bootstrap"},
            ],
        )

        # rd2: A (rank 1, member), B (rank 5, NOT member), C (rank 2, member)
        _write_entries_partition(
            output_dir,
            rd2,
            [
                {"isin": "ISIN_A", "ric": "RIC_A", "rank": 1, "ff_mcap": 1100.0},
                {"isin": "ISIN_B", "ric": "RIC_B", "rank": 5, "ff_mcap": 500.0},
                {"isin": "ISIN_C", "ric": "RIC_C", "rank": 2, "ff_mcap": 950.0},
            ],
        )
        _write_membership_partition(
            output_dir,
            rd2,
            [
                {"isin": "ISIN_A", "is_member": True, "entry_reason": "top_550"},
                {"isin": "ISIN_B", "is_member": False, "entry_reason": "top_550"},
                {"isin": "ISIN_C", "is_member": True, "entry_reason": "top_550"},
            ],
        )

        with patch("stoxx.sync.date") as mock_date:
            mock_date.today.return_value = date(2024, 6, 3)
            mock_date.fromisoformat = date.fromisoformat
            result = _build_ranking_table(output_dir)

        assert "date" in result.columns
        assert "RIC_A" in result.columns
        assert "RIC_B" in result.columns
        assert "RIC_C" in result.columns

        # On rd1 day: A=1, B=2, C=null (not member)
        row_rd1 = result.filter(pl.col("date") == rd1)
        assert row_rd1["RIC_A"][0] == 1
        assert row_rd1["RIC_B"][0] == 2
        assert row_rd1["RIC_C"][0] is None

        # On rd2 day: A=1, B=null (exited), C=2 (entered)
        row_rd2 = result.filter(pl.col("date") == rd2)
        assert row_rd2["RIC_A"][0] == 1
        assert row_rd2["RIC_B"][0] is None
        assert row_rd2["RIC_C"][0] == 2

        # Day after rd2: forward-filled from rd2
        row_after = result.filter(pl.col("date") == date(2024, 6, 2))
        assert row_after["RIC_A"][0] == 1
        assert row_after["RIC_B"][0] is None
        assert row_after["RIC_C"][0] == 2

    def test_reentry(self, tmp_path):
        """A company that exits then re-enters has null in the gap period."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        rd1 = date(2024, 3, 1)
        rd2 = date(2024, 6, 1)
        rd3 = date(2024, 9, 1)

        for rd, rank, is_member in [(rd1, 1, True), (rd2, 5, False), (rd3, 2, True)]:
            _write_entries_partition(
                output_dir,
                rd,
                [{"isin": "ISIN_X", "ric": "RIC_X", "rank": rank, "ff_mcap": 1000.0}],
            )
            _write_membership_partition(
                output_dir,
                rd,
                [{"isin": "ISIN_X", "is_member": is_member, "entry_reason": "top_550"}],
            )

        with patch("stoxx.sync.date") as mock_date:
            mock_date.today.return_value = date(2024, 9, 2)
            mock_date.fromisoformat = date.fromisoformat
            result = _build_ranking_table(output_dir)

        # rd1: member with rank 1
        assert result.filter(pl.col("date") == rd1)["RIC_X"][0] == 1
        # Between rd1 and rd2: forward-filled rank 1
        assert result.filter(pl.col("date") == date(2024, 4, 1))["RIC_X"][0] == 1
        # rd2: not member -> null
        assert result.filter(pl.col("date") == rd2)["RIC_X"][0] is None
        # Between rd2 and rd3: forward-filled null
        assert result.filter(pl.col("date") == date(2024, 7, 1))["RIC_X"][0] is None
        # rd3: re-entered with rank 2
        assert result.filter(pl.col("date") == rd3)["RIC_X"][0] == 2

    def test_exactly_600_members_per_day(self, tmp_path):
        """Each day has exactly 600 non-null ranks (the STOXX 600 constituent count)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        rd = date(2024, 3, 1)
        n_total = 700
        n_members = 600

        entries = [
            {"isin": f"ISIN_{i}", "ric": f"RIC_{i}", "rank": i + 1, "ff_mcap": float(n_total - i)}
            for i in range(n_total)
        ]
        membership = [
            {"isin": f"ISIN_{i}", "is_member": i < n_members, "entry_reason": "bootstrap"} for i in range(n_total)
        ]

        _write_entries_partition(output_dir, rd, entries)
        _write_membership_partition(output_dir, rd, membership)

        with patch("stoxx.sync.date") as mock_date:
            mock_date.today.return_value = date(2024, 3, 3)
            mock_date.fromisoformat = date.fromisoformat
            result = _build_ranking_table(output_dir)

        ric_cols = [c for c in result.columns if c != "date"]
        for row in result.iter_rows(named=True):
            non_null = sum(1 for c in ric_cols if row[c] is not None)
            assert non_null == n_members, f"Expected {n_members} members on {row['date']}, got {non_null}"

    def test_empty_partitions(self, tmp_path):
        """No data produces a DataFrame with only a date column."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = _build_ranking_table(output_dir)

        assert "date" in result.columns
        assert len(result) == 0

    def test_uses_ric_columns(self, tmp_path):
        """Columns are RIC codes, not ISINs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        rd = date(2024, 3, 1)
        _write_entries_partition(
            output_dir,
            rd,
            [{"isin": "ISIN_A", "ric": "VOW3.DE", "rank": 1, "ff_mcap": 1000.0}],
        )
        _write_membership_partition(
            output_dir,
            rd,
            [{"isin": "ISIN_A", "is_member": True, "entry_reason": "bootstrap"}],
        )

        with patch("stoxx.sync.date") as mock_date:
            mock_date.today.return_value = date(2024, 3, 2)
            mock_date.fromisoformat = date.fromisoformat
            result = _build_ranking_table(output_dir)

        assert "VOW3.DE" in result.columns
        assert "ISIN_A" not in result.columns

    @pytest.mark.asyncio
    async def test_ranking_written_during_sync(self, tmp_path):
        """Sync produces a ranking.parquet file in the output directory."""
        storage = MemoryStorage()
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"

        rd = date(2024, 3, 1)
        assets = _make_assets()
        entries = _make_entries(rd)

        async def mock_download(**kwargs):
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.from_env", return_value=storage),
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch(
                "stoxx.sync.resolve_yukka_ids",
                side_effect=lambda df: df.with_columns(pl.lit(None).cast(pl.Utf8).alias("yukka_id")),
            ),
            patch("stoxx.sync.report_unresolved_assets"),
        ):
            result = await sync(output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd]
        assert (output_dir / "ranking.parquet").exists()
        ranking_df = pl.read_parquet(output_dir / "ranking.parquet")
        assert "date" in ranking_df.columns
        assert len(ranking_df) > 0
