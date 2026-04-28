"""Tests for astroai.core.io.__init__ lazy __getattr__ imports (lines 21-30)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import astroai.core.io as io_mod


class TestIoInitLazyImports:
    def test_read_raw_accessible_via_getattr(self) -> None:
        """Cover lines 22-23: read_raw lazy import through __getattr__."""
        with patch.dict("sys.modules", {"rawpy": MagicMock()}):
            fn = io_mod.__getattr__("read_raw")
        assert callable(fn)

    def test_read_raw_metadata_accessible_via_getattr(self) -> None:
        """Cover lines 25-26: read_raw_metadata lazy import through __getattr__."""
        with patch.dict("sys.modules", {"rawpy": MagicMock()}):
            fn = io_mod.__getattr__("read_raw_metadata")
        assert callable(fn)

    def test_raw_extensions_accessible_via_getattr(self) -> None:
        """Cover lines 27-29: RAW_EXTENSIONS lazy import through __getattr__."""
        with patch.dict("sys.modules", {"rawpy": MagicMock()}):
            raw_ext = io_mod.__getattr__("RAW_EXTENSIONS")
        assert isinstance(raw_ext, frozenset)

    def test_unknown_attr_raises_attribute_error(self) -> None:
        """Cover line 30: AttributeError raised for unknown names."""
        with pytest.raises(AttributeError, match="no attribute"):
            io_mod.__getattr__("totally_unknown_function")
