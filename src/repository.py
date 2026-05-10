"""Abstract base classes for data repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl


@dataclass
class Index(ABC):
    """Abstract index with constituent lookup."""

    name: str

    @property
    @abstractmethod
    def available_review_dates(self) -> list[date]:
        """Return sorted list of available review dates."""

    @abstractmethod
    def constituents(self, review_date: date | None = None) -> pl.DataFrame:
        """Return the constituents for a given review date."""


@dataclass
class Repository(ABC):
    """Abstract repository providing access to index data."""

    root: Path

    @property
    @abstractmethod
    def indexes(self) -> list[Index]:
        """Return a list of index instances for stoxx600, sp500, etc."""


@dataclass
class ParquetRepository(Repository):
    """Repository that discovers indexes from Parquet dataset directories."""

    index_factory: Callable[[str, Path], Index]
    _index_cache: list[Index] | None = field(default=None, init=False, repr=False)

    @property
    def indexes(self) -> list[Index]:
        """Discover subdirectories containing assets.parquet."""
        if self._index_cache is not None:
            return self._index_cache
        if not self.root.exists():
            self._index_cache = []
            return self._index_cache
        found: list[Index] = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "assets.parquet").exists():
                found.append(self.index_factory(child.name, child))
        self._index_cache = found
        return self._index_cache

    def get_index(self, name: str) -> Index:
        """Look up an index by name, raising KeyError if not found."""
        for idx in self.indexes:
            if idx.name == name:
                return idx
        msg = f"Index '{name}' not found. Available: {[i.name for i in self.indexes]}"
        raise KeyError(msg)
