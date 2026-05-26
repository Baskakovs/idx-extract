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
    _compute_intervals,
    _download_prior_membership,
    _read_local_membership,
    _write_merged_assets,
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
        with patch("stoxx.sync.date") as mock_date:
            mock_date.today.return_value = date(2024, 3, 15)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            # Add all periods as existing in remote
            from stoxx.download import START_DATE, get_periods

            all_periods = get_periods(START_DATE, date(2024, 3, 15))
            for year, month in all_periods:
                key = f"STOXX600/membership/review_date={date(year, month, 1)}/data.parquet"
                storage._objects[key] = b"fake"

            result = await sync(storage, output_dir=tmp_path / "output", cache_dir=tmp_path / "cache")

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
            # Write a file that parse_selection_list can't use, so we mock parsing too
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
        ):
            result = await sync(storage, output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd]
        # Verify files were uploaded to storage
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
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
            patch("stoxx.sync.compute_membership") as mock_compute,
        ):
            mock_compute.return_value = [
                IndexMembership(isin=f"ISIN{i:06d}", is_member=True, entry_reason=EntryReason.BOOTSTRAP)
                for i in range(600)
            ]
            result = await sync(storage, output_dir=output_dir, cache_dir=cache_dir)

        # compute_membership should be called with prior_membership=None
        mock_compute.assert_called_once()
        _, kwargs = mock_compute.call_args
        if not kwargs:
            args = mock_compute.call_args[0]
            assert args[1] is None  # prior_membership
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
            patch("stoxx.sync.get_periods", return_value=[(2024, 3), (2024, 6)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", side_effect=mock_parse),
            patch("stoxx.sync.compute_membership", side_effect=mock_compute),
        ):
            result = await sync(storage, output_dir=output_dir, cache_dir=cache_dir)

        assert result == [rd1, rd2]
        # First call should have no prior (bootstrap)
        assert compute_calls[0] is None
        # Second call should have prior membership from first
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
            patch("stoxx.sync.get_periods", return_value=[(2024, 3)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
        ):
            result = await sync(storage, output_dir=output_dir, cache_dir=cache_dir)

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
            # Verify only the missing period is requested
            periods = kwargs.get("periods", [])
            assert (2024, 3) not in periods
            result = DownloadResult()
            csv_path = cache_dir / "test.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_text("placeholder")
            result.downloaded.append(csv_path)
            return result

        with (
            patch("stoxx.sync.get_periods", return_value=[(2024, 3), (2024, 6)]),
            patch("stoxx.sync.download_selection_lists", side_effect=mock_download),
            patch("stoxx.sync.parse_selection_list", return_value=(assets, entries)),
        ):
            result = await sync(storage, output_dir=output_dir, cache_dir=cache_dir)

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


class TestWriteMergedAssetsIntervals:
    """Tests for _write_merged_assets with interval columns."""

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

        _write_merged_assets(output_dir, assets, {"ISIN001", "ISIN002", "ISIN003"})

        result = pl.read_parquet(output_dir / "assets.parquet")
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

        _write_merged_assets(output_dir, assets, {"ISIN001"})

        result = pl.read_parquet(output_dir / "assets.parquet")
        isin001 = result.filter(pl.col("isin") == "ISIN001")
        assert len(isin001) == 2
        assert isin001["first_included"][0] == date(2024, 3, 1)
        assert isin001["last_included"][0] == date(2024, 3, 1)
        assert isin001["first_included"][1] == date(2024, 9, 1)
        assert isin001["last_included"][1] == date(2024, 9, 1)
