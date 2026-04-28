"""Unit tests for LicenseClient — all HTTP calls mocked via patch."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from astroai.licensing.client import LicenseClient
from astroai.licensing.exceptions import (
    ActivationError,
    LicenseError,
    RefreshError,
    TierInsufficientError,
)


def _mock_response(
    status_code: int,
    json_body: dict | None = None,
    content_type: str = "application/json",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.headers = {"content-type": content_type}
    return resp


class TestLicenseClientActivate:
    def test_success_returns_jwt_and_attestation(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {"token": "jwt123", "attestation": "att456"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            token, att = client.activate("KEY-1234", "1.0.0")
        assert token == "jwt123"
        assert att == "att456"

    def test_success_attestation_may_be_none(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {"token": "jwt_only"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            token, att = client.activate("KEY", "1.0.0")
        assert token == "jwt_only"
        assert att is None

    def test_error_raises_activation_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(422, {"error": "max_seats_reached", "detail": "Limit hit"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(ActivationError, match="max_seats_reached"):
                client.activate("KEY", "1.0.0")

    def test_network_error_raises_license_error(self) -> None:
        import httpx
        client = LicenseClient(base_url="http://fake")
        with patch("astroai.licensing.client.httpx.post", side_effect=httpx.RequestError("down")):
            with pytest.raises(LicenseError, match="Network error"):
                client.activate("KEY", "1.0.0")

    def test_non_json_error_uses_status_code(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(500, content_type="text/html")
        resp.json.return_value = {}
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(ActivationError) as exc_info:
                client.activate("KEY", "1.0.0")
        assert "500" in exc_info.value.code


class TestLicenseClientRefresh:
    def test_success_returns_new_token(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {"token": "new_jwt", "attestation": "new_att"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            token, att = client.refresh("old_jwt")
        assert token == "new_jwt"
        assert att == "new_att"

    def test_error_raises_refresh_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(401, {"error": "subscription_expired"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(RefreshError, match="subscription_expired"):
                client.refresh("old_jwt")

    def test_network_error_raises_license_error(self) -> None:
        import httpx
        client = LicenseClient(base_url="http://fake")
        with patch("astroai.licensing.client.httpx.post", side_effect=httpx.RequestError("timeout")):
            with pytest.raises(LicenseError):
                client.refresh("jwt")

    def test_non_json_error_uses_status_code(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(503, content_type="text/plain")
        resp.json.return_value = {}
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(RefreshError) as exc_info:
                client.refresh("jwt")
        assert "503" in exc_info.value.code


class TestLicenseClientDeactivate:
    def test_success_returns_seats(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {"seats_released": 2})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            seats = client.deactivate("jwt")
        assert seats == 2

    def test_success_default_seat_count(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            seats = client.deactivate("jwt")
        assert seats == 1

    def test_error_raises_license_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(500, {})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(LicenseError, match="500"):
                client.deactivate("jwt")

    def test_network_error_raises_license_error(self) -> None:
        import httpx
        client = LicenseClient(base_url="http://fake")
        with patch("astroai.licensing.client.httpx.post", side_effect=httpx.RequestError("conn")):
            with pytest.raises(LicenseError, match="deactivation"):
                client.deactivate("jwt")


class TestLicenseClientGetModelManifest:
    def test_success(self) -> None:
        client = LicenseClient(base_url="http://fake")
        models = [{"name": "denoise_pro", "size": 100}]
        resp = _mock_response(200, {"models": models})
        with patch("astroai.licensing.client.httpx.get", return_value=resp):
            result = client.get_model_manifest("jwt")
        assert result == models

    def test_error_raises_license_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(403, {})
        with patch("astroai.licensing.client.httpx.get", return_value=resp):
            with pytest.raises(LicenseError, match="403"):
                client.get_model_manifest("jwt")

    def test_network_error_raises_license_error(self) -> None:
        import httpx
        client = LicenseClient(base_url="http://fake")
        with patch("astroai.licensing.client.httpx.get", side_effect=httpx.RequestError("timeout")):
            with pytest.raises(LicenseError, match="manifest"):
                client.get_model_manifest("jwt")


class TestLicenseClientGetDownloadUrl:
    def test_success(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(200, {"url": "https://r2.example.com/model.onnx"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            url = client.get_download_url("jwt", "denoise_pro")
        assert url == "https://r2.example.com/model.onnx"

    def test_403_raises_tier_insufficient(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(403, {"required_tier": "pro_annual"})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(TierInsufficientError) as exc_info:
                client.get_download_url("jwt", "starnet_pro")
        assert exc_info.value.model_name == "starnet_pro"
        assert exc_info.value.required_tier == "pro_annual"

    def test_404_raises_license_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(404, {})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(LicenseError, match="not found"):
                client.get_download_url("jwt", "unknown_model")

    def test_other_error_raises_license_error(self) -> None:
        client = LicenseClient(base_url="http://fake")
        resp = _mock_response(500, {})
        with patch("astroai.licensing.client.httpx.post", return_value=resp):
            with pytest.raises(LicenseError, match="500"):
                client.get_download_url("jwt", "model")

    def test_network_error_raises_license_error(self) -> None:
        import httpx
        client = LicenseClient(base_url="http://fake")
        with patch("astroai.licensing.client.httpx.post", side_effect=httpx.RequestError("conn")):
            with pytest.raises(LicenseError, match="download URL"):
                client.get_download_url("jwt", "model")
