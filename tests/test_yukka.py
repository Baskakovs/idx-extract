"""Tests for the Yukka entity ID resolution module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl


class TestResolveYukkaIds:
    """Tests for resolve_yukka_ids task."""

    def _make_assets_df(self, n: int = 5) -> pl.DataFrame:
        """Create a test assets DataFrame with n rows."""
        return pl.DataFrame(
            {
                "isin": [f"ISIN{i:06d}" for i in range(n)],
                "ric": [f"RIC{i}.DE" for i in range(n)],
                "name": [f"Company {i}" for i in range(n)],
                "country": ["DE"] * n,
                "currency": ["EUR"] * n,
            }
        )

    def _mock_isin_response(self, isins: list[str], resolved: set[str]) -> dict:
        """Build a mock ISIN lookup response."""
        return {isin: {"alpha_id": f"yukka_{isin}"} if isin in resolved else None for isin in isins}

    def _mock_ric_response(self, rics: list[str], resolved: set[str]) -> dict:
        """Build a mock RIC lookup response."""
        return {ric: {"alpha_id": f"yukka_{ric}"} if ric in resolved else None for ric in rics}

    def test_row_count_preserved_all_resolved(self):
        """Row count must be identical before and after enrichment."""
        from yukka import resolve_yukka_ids

        df = self._make_assets_df(10)
        all_isins = set(df["isin"].to_list())

        mock_client = MagicMock()
        mock_client.post.return_value.raise_for_status = MagicMock()
        mock_client.post.return_value.json.return_value = self._mock_isin_response(list(all_isins), all_isins)

        with patch("yukka._build_client", return_value=mock_client):
            result = resolve_yukka_ids(df)

        assert len(result) == len(df)
        assert result["yukka_id"].null_count() == 0

    def test_row_count_preserved_none_resolved(self):
        """All rows kept even when no ISIN or RIC resolves."""
        from yukka import resolve_yukka_ids

        df = self._make_assets_df(10)

        mock_client = MagicMock()
        mock_client.post.return_value.raise_for_status = MagicMock()
        mock_client.post.return_value.json.return_value = dict.fromkeys(df["isin"].to_list())

        with patch("yukka._build_client", return_value=mock_client):
            result = resolve_yukka_ids(df)

        assert len(result) == len(df)
        assert result["yukka_id"].null_count() == len(df)

    def test_row_count_preserved_partial_resolution(self):
        """Partially resolved ISINs still preserve all rows."""
        from yukka import resolve_yukka_ids

        df = self._make_assets_df(10)
        isins = df["isin"].to_list()
        resolved_isins = set(isins[:6])

        isin_resp = self._mock_isin_response(isins, resolved_isins)
        unresolved_rics = [f"RIC{i}.DE" for i in range(6, 10)]
        ric_resp = self._mock_ric_response(unresolved_rics, set())

        call_count = 0

        def mock_post(endpoint, json):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "isin" in endpoint:
                resp.json.return_value = {k: isin_resp[k] for k in json if k in isin_resp}
            else:
                resp.json.return_value = {k: ric_resp.get(k) for k in json}
            return resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post

        with patch("yukka._build_client", return_value=mock_client):
            result = resolve_yukka_ids(df)

        assert len(result) == len(df)
        assert result["yukka_id"].null_count() == 4
        assert result["yukka_id"].drop_nulls().len() == 6

    def test_row_count_preserved_with_duplicate_isins(self):
        """Multiple rows per ISIN (e.g. interval rows) are all preserved."""
        from yukka import resolve_yukka_ids

        df = pl.DataFrame(
            {
                "isin": ["ISIN001", "ISIN001", "ISIN002", "ISIN003"],
                "ric": ["R1.DE", "R1.DE", "R2.DE", "R3.DE"],
                "first_included": ["2024-03-01", "2024-09-01", "2024-03-01", "2024-06-01"],
                "last_included": ["2024-06-01", "2024-12-01", "2024-12-01", "2024-12-01"],
            }
        )

        mock_client = MagicMock()
        mock_client.post.return_value.raise_for_status = MagicMock()
        mock_client.post.return_value.json.return_value = {
            "ISIN001": {"alpha_id": "yukka_1"},
            "ISIN002": None,
            "ISIN003": {"alpha_id": "yukka_3"},
        }

        with patch("yukka._build_client", return_value=mock_client):
            result = resolve_yukka_ids(df)

        assert len(result) == 4
        isin001_rows = result.filter(pl.col("isin") == "ISIN001")
        assert len(isin001_rows) == 2
        assert isin001_rows["yukka_id"].null_count() == 0
        assert result.filter(pl.col("isin") == "ISIN002")["yukka_id"].null_count() == 1

    def test_ric_fallback_resolves_unmatched_isins(self):
        """ISINs not found via ISIN lookup fall back to RIC lookup."""
        from yukka import resolve_yukka_ids

        df = self._make_assets_df(3)
        isins = df["isin"].to_list()

        def mock_post(endpoint, json):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "isin" in endpoint:
                resp.json.return_value = {
                    isins[0]: {"alpha_id": "yukka_via_isin"},
                    isins[1]: None,
                    isins[2]: None,
                }
            else:
                resp.json.return_value = {
                    "RIC1.DE": {"alpha_id": "yukka_via_ric"},
                    "RIC2.DE": None,
                }
            return resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_post

        with patch("yukka._build_client", return_value=mock_client):
            result = resolve_yukka_ids(df)

        assert len(result) == 3
        assert result.filter(pl.col("isin") == isins[0])["yukka_id"][0] == "yukka_via_isin"
        assert result.filter(pl.col("isin") == isins[1])["yukka_id"][0] == "yukka_via_ric"
        assert result.filter(pl.col("isin") == isins[2])["yukka_id"][0] is None


class TestReportUnresolvedAssets:
    """Tests for report_unresolved_assets task."""

    def test_no_artifact_when_all_resolved(self, tmp_path):
        """No artifact created when every row has a yukka_id."""
        from yukka import report_unresolved_assets

        df = pl.DataFrame(
            {
                "isin": ["ISIN001", "ISIN002"],
                "ric": ["R1", "R2"],
                "yukka_id": ["y1", "y2"],
            }
        )
        path = tmp_path / "assets.parquet"
        df.write_parquet(path)

        with patch("yukka.create_table_artifact") as mock_artifact:
            report_unresolved_assets(path)

        mock_artifact.assert_not_called()

    def test_artifact_created_for_unresolved(self, tmp_path):
        """Artifact is created listing all unresolved assets from the full file."""
        from yukka import report_unresolved_assets

        df = pl.DataFrame(
            {
                "isin": ["ISIN001", "ISIN002", "ISIN003"],
                "ric": ["R1", "R2", "R3"],
                "name": ["Co1", "Co2", "Co3"],
                "country": ["DE", "FR", "GB"],
                "currency": ["EUR", "EUR", "GBP"],
                "yukka_id": ["y1", None, None],
            }
        )
        path = tmp_path / "assets.parquet"
        df.write_parquet(path)

        with patch("yukka.create_table_artifact") as mock_artifact:
            report_unresolved_assets(path)

        mock_artifact.assert_called_once()
        call_kwargs = mock_artifact.call_args[1]
        assert call_kwargs["key"] == "unresolved-yukka-assets"
        assert len(call_kwargs["table"]) == 2
        unresolved_isins = {row["isin"] for row in call_kwargs["table"]}
        assert unresolved_isins == {"ISIN002", "ISIN003"}
