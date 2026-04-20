import pytest
from unittest.mock import patch, MagicMock

from astroai.inference.coordinate_extractor import (
    CoordinateExtractor,
    Coordinates,
    ExtractionMethod,
    _parse_sexagesimal_ra,
    _parse_sexagesimal_dec,
)


@pytest.fixture()
def extractor():
    return CoordinateExtractor(timeout=5.0)


class TestSexagesimalParsing:
    def test_parse_ra_space_separated(self):
        result = _parse_sexagesimal_ra("05 35 17.3")
        assert result is not None
        assert abs(result - 83.8221) < 0.01

    def test_parse_ra_colon_separated(self):
        result = _parse_sexagesimal_ra("12:30:00")
        assert result is not None
        assert abs(result - 187.5) < 0.01

    def test_parse_dec_positive(self):
        result = _parse_sexagesimal_dec("+22 00 52")
        assert result is not None
        assert abs(result - 22.0144) < 0.01

    def test_parse_dec_negative(self):
        result = _parse_sexagesimal_dec("-05 23 28")
        assert result is not None
        assert abs(result - (-5.3911)) < 0.01

    def test_parse_invalid_returns_none(self):
        assert _parse_sexagesimal_ra("invalid") is None
        assert _parse_sexagesimal_dec("invalid") is None


class TestObjctraExtraction:
    def test_extracts_from_objctra_objctdec(self, extractor):
        header = {"OBJCTRA": "05 35 17.3", "OBJCTDEC": "-05 23 28"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.OBJCTRA_DEC
        assert abs(result.ra_deg - 83.8221) < 0.01
        assert abs(result.dec_deg - (-5.3911)) < 0.01
        assert result.confidence == 0.95

    def test_missing_objctra_falls_through(self, extractor):
        header = {"OBJCTRA": "05 35 17.3"}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.OBJCTRA_DEC


class TestRaDecExtraction:
    def test_extracts_numeric_ra_dec(self, extractor):
        header = {"RA": 83.8221, "DEC": -5.3911}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.RA_DEC
        assert abs(result.ra_deg - 83.8221) < 0.001
        assert result.confidence == 0.90

    def test_extracts_string_ra_dec(self, extractor):
        header = {"RA": "12:30:00", "DEC": "+45 00 00"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.RA_DEC
        assert abs(result.ra_deg - 187.5) < 0.01
        assert abs(result.dec_deg - 45.0) < 0.01


class TestCrvalExtraction:
    def test_extracts_crval_with_ctype(self, extractor):
        header = {
            "CRVAL1": 210.8,
            "CRVAL2": 54.35,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.CRVAL
        assert abs(result.ra_deg - 210.8) < 0.01
        assert result.confidence == 0.85

    def test_crval_without_ctype_assumes_ra_dec(self, extractor):
        header = {"CRVAL1": 100.0, "CRVAL2": 30.0}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.CRVAL
        assert result.ra_deg == 100.0
        assert result.dec_deg == 30.0


class TestObjectResolve:
    def test_skips_calibration_frame_names(self, extractor):
        header = {"OBJECT": "Dark"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.NONE

    @patch("astroai.inference.coordinate_extractor.httpx.Client")
    def test_resolves_via_sesame(self, mock_client_cls, extractor):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "%J 83.6331 22.0145 = 2000.0\n%T M1\n"
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        header = {"OBJECT": "M1"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.OBJECT_RESOLVE
        assert abs(result.ra_deg - 83.6331) < 0.001
        assert result.object_name == "M1"

    def test_uses_cache_on_second_call(self, extractor):
        cached = Coordinates(
            ra_deg=10.68, dec_deg=41.27,
            method=ExtractionMethod.OBJECT_RESOLVE,
            confidence=0.75, object_name="M31",
        )
        extractor._cache["M31"] = cached
        header = {"OBJECT": "M31"}
        result = extractor.extract(header)
        assert result is cached


class TestPriorityCascade:
    def test_objctra_takes_priority_over_ra_dec(self, extractor):
        header = {
            "OBJCTRA": "05 35 17.3",
            "OBJCTDEC": "-05 23 28",
            "RA": 999.0,
            "DEC": 999.0,
        }
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.OBJCTRA_DEC

    def test_empty_header_returns_none_method(self, extractor):
        result = extractor.extract({})
        assert result.method == ExtractionMethod.NONE
        assert result.confidence == 0.0
