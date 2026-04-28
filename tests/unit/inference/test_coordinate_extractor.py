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


# ---------------------------------------------------------------------------
# Coverage gap tests
# ---------------------------------------------------------------------------

class TestSexagesimalParsingEdgeCases:
    """Cover lines 48-49 and 64-65: ValueError/IndexError except branches."""

    def test_parse_ra_raises_value_error_returns_none(self):
        # Only one part => len(parts) < 2 returns None at line 41-42
        # Provide two parts where parts[0] is non-numeric to trigger ValueError at line 48-49
        result = _parse_sexagesimal_ra("XX YY")
        assert result is None

    def test_parse_ra_two_parts_first_invalid(self):
        # Two parts: parts[0] invalid float -> ValueError caught at line 48
        result = _parse_sexagesimal_ra("abc 30")
        assert result is None

    def test_parse_dec_value_error_returns_none(self):
        # Regex matches but groups contain non-float -> ValueError at line 64-65
        # Patch re.match to return a match with invalid group values
        import re
        from unittest.mock import patch as _patch, MagicMock
        mock_match = MagicMock()
        mock_match.group.side_effect = lambda n: ("notafloat" if n == 1 else "0")
        with _patch("astroai.inference.coordinate_extractor.re.match", return_value=mock_match):
            result = _parse_sexagesimal_dec("+10 20 30")
        assert result is None


class TestTryObjctraDecInvalidParsed:
    """Cover line 119: ra/dec parse returns None -> _try_objctra_dec returns None."""

    def test_invalid_objctra_string_returns_none_method(self, extractor):
        # "INVALID" cannot be parsed by _parse_sexagesimal_ra -> ra is None
        header = {"OBJCTRA": "INVALID", "OBJCTDEC": "+22 00 00"}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.OBJCTRA_DEC

    def test_invalid_objctdec_string_returns_none_method(self, extractor):
        # Valid RA but invalid DEC string -> dec is None -> line 118-119
        header = {"OBJCTRA": "05 35 17.3", "OBJCTDEC": "INVALID"}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.OBJCTRA_DEC


class TestTryRaDecInvalidParsed:
    """Cover line 141: ra/dec parse returns None -> _try_ra_dec returns None."""

    def test_invalid_ra_string_falls_through(self, extractor):
        # String RA that cannot be parsed -> ra is None at line 141
        header = {"RA": "not_a_number", "DEC": 45.0}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.RA_DEC

    def test_invalid_dec_string_falls_through(self, extractor):
        # Valid string RA but invalid string DEC -> dec is None at line 141
        header = {"RA": "12:30:00", "DEC": "not_a_number"}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.RA_DEC

    def test_non_parseable_numeric_dec_falls_through(self, extractor):
        # None DEC value -> _try_float(None) returns None -> line 141
        header = {"RA": 100.0, "DEC": None}
        result = extractor.extract(header)
        assert result.method != ExtractionMethod.RA_DEC


class TestTryCrvalDecInCtype1:
    """Cover lines 163-170: 'DEC' in ctype1 branch -> RA/Dec swapped."""

    def test_crval_swaps_ra_dec_when_dec_in_ctype1_only(self, extractor):
        # CTYPE1="DEC--TAN", CTYPE2="SOMETHING" (no "RA")
        # Line 156: "RA" in ctype1 -> False, "RA" in ctype2 -> False
        # Line 163: "DEC" in ctype1 -> True -> ra_deg=crval2, dec_deg=crval1
        header = {
            "CRVAL1": 45.0,
            "CRVAL2": 210.0,
            "CTYPE1": "DEC--TAN",
            "CTYPE2": "SOMETHING",
        }
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.CRVAL
        assert result.ra_deg == 210.0
        assert result.dec_deg == 45.0
        assert result.confidence == 0.85

    def test_crval_low_confidence_when_no_ra_dec_in_ctype(self, extractor):
        # CTYPE1/CTYPE2 present but contain neither RA nor DEC -> line 170-175
        header = {
            "CRVAL1": 55.0,
            "CRVAL2": 25.0,
            "CTYPE1": "GLON-TAN",
            "CTYPE2": "GLAT-TAN",
        }
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.CRVAL
        assert result.ra_deg == 55.0
        assert result.dec_deg == 25.0
        assert result.confidence == 0.70


class TestResolveSesameErrors:
    """Cover lines 200-201 and 203-205: HTTP error and httpx.HTTPError paths."""

    @patch("astroai.inference.coordinate_extractor.httpx.Client")
    def test_sesame_non_200_response_returns_none(self, mock_client_cls, extractor):
        # Lines 199-201: HTTP status != 200 -> log warning, return None
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        header = {"OBJECT": "UnknownObject"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.NONE

    @patch("astroai.inference.coordinate_extractor.httpx.Client")
    def test_sesame_httpx_error_returns_none(self, mock_client_cls, extractor):
        # Lines 203-205: httpx.HTTPError raised -> log warning, return None
        import httpx
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_cls.return_value = mock_client

        header = {"OBJECT": "SomeNebula"}
        result = extractor.extract(header)
        assert result.method == ExtractionMethod.NONE


class TestParseSesameResponseNoMatch:
    """Cover line 224: _parse_sesame_response returns None when no %J line found."""

    def test_sesame_response_without_j_line_returns_none(self, extractor):
        from astroai.inference.coordinate_extractor import CoordinateExtractor
        result = CoordinateExtractor._parse_sesame_response(
            "%T M1\n%C 83.63 +22.01\n# comment\n", "M1"
        )
        assert result is None

    def test_sesame_response_j_line_insufficient_parts(self, extractor):
        from astroai.inference.coordinate_extractor import CoordinateExtractor
        # %J line present but only one value -> len(parts) < 2
        result = CoordinateExtractor._parse_sesame_response("%J 83.6331\n", "M1")
        assert result is None
