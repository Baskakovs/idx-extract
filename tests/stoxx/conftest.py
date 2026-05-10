"""Shared fixtures for STOXX extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stoxx.extract import parse_selection_list_csv

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def csv_fixture_path() -> Path:
    """Return the path to the March 2024 selection list CSV fixture."""
    return FIXTURE_DIR / "slpublic_sxxp_20240301.csv"


@pytest.fixture
def parsed_csv(csv_fixture_path: Path):
    """Parse the CSV fixture and return (assets, entries)."""
    return parse_selection_list_csv(csv_fixture_path)
