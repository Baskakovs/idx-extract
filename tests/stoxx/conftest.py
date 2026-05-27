"""Shared fixtures for STOXX extraction tests.

Security exceptions (S101, S603, S607) are inherited from the project-wide
ruff per-file-ignores for tests/**/*.py — assert statements and subprocess
calls are expected in test code.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW"] = "ignore"

import pytest

from stoxx.extract import compute_membership, parse_selection_list_csv

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def csv_fixture_path() -> Path:
    """Return the path to the March 2024 selection list CSV fixture."""
    return FIXTURE_DIR / "slpublic_sxxp_20240301.csv"


@pytest.fixture
def parsed_csv(csv_fixture_path: Path):
    """Parse the CSV fixture and return (assets, entries)."""
    return parse_selection_list_csv(csv_fixture_path)


@pytest.fixture
def membership(parsed_csv):
    """Compute bootstrap membership from parsed CSV entries."""
    _, entries = parsed_csv
    return compute_membership(entries, prior_membership=None)
