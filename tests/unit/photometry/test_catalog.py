from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from astroai.engine.photometry.catalog import AAVSOCatalogClient, GAIACatalogClient


class TestGAIACatalogClient:
    def test_query_returns_parsed_stars(self) -> None:
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "metadata": [
                {"name": "ra"},
                {"name": "dec"},
                {"name": "phot_g_mean_mag"},
            ],
            "data": [
                [180.0, 45.0, 12.5],
                [180.1, 45.1, 13.0],
            ],
        }
        fake_response.raise_for_status = MagicMock()

        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = fake_response

            client = GAIACatalogClient()
            result = client.query(180.0, 45.0, 1.0)

        assert len(result) == 2
        assert result[0]["ra"] == 180.0
        assert result[0]["phot_g_mean_mag"] == 12.5

    def test_fail_silently_returns_empty_on_error(self) -> None:
        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPError("connection failed")

            client = GAIACatalogClient(fail_silently=True)
            result = client.query(180.0, 45.0, 1.0)

        assert result == []

    def test_fail_silently_false_raises(self) -> None:
        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPError("connection failed")

            client = GAIACatalogClient(fail_silently=False)
            with pytest.raises(httpx.HTTPError):
                client.query(180.0, 45.0, 1.0)


class TestAAVSOCatalogClient:
    def test_query_returns_variable_stars(self) -> None:
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "VSXObjects": {
                "VSXObject": [
                    {"Name": "V* RR Lyr", "RA2000": "19 25 27.91"},
                    {"Name": "V* SS Cyg", "RA2000": "21 42 42.80"},
                ],
            },
        }
        fake_response.raise_for_status = MagicMock()

        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.return_value = fake_response

            client = AAVSOCatalogClient()
            result = client.query(290.0, 42.0, 2.0)

        assert len(result) == 2
        assert result[0]["Name"] == "V* RR Lyr"

    def test_fail_silently_returns_empty_on_error(self) -> None:
        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPError("timeout")

            client = AAVSOCatalogClient(fail_silently=True)
            result = client.query(180.0, 45.0, 1.0)

        assert result == []

    def test_fail_silently_false_raises(self) -> None:
        with patch("astroai.engine.photometry.catalog.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=mock_client.return_value)
            mock_client.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.return_value.get.side_effect = httpx.HTTPError("timeout")

            client = AAVSOCatalogClient(fail_silently=False)
            with pytest.raises(httpx.HTTPError):
                client.query(180.0, 45.0, 1.0)
