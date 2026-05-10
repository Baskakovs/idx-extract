"""Abstract base classes for data repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Index:
    """Abstract index with constituent lookup."""

    @abstractmethod
    def constituents(self, **kwargs) -> list:
        """Return the list of index constituents."""


@dataclass
class Repository(ABC):
    """Abstract repository providing access to index data."""

    @property
    @abstractmethod
    def indexes(self) -> list[Index]:
        """Return a list of index instances for stoxx600, sp500, etc."""
