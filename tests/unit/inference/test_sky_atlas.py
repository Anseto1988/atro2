import pytest
from unittest.mock import patch, MagicMock

from astroai.inference.sky_atlas import (
    SkyAtlas,
    SkyObject,
    SkyAtlasResult,
    _angular_separation,
)


@pytest.fixture()
def atlas():
    return SkyAtlas(timeout=5.0, use_online=False)


@pytest.fixture()
def atlas_online():
    return SkyAtlas(timeout=5.0, use_online=True)


class TestAngularSeparation:
    def test_same_point_is_zero(self):
        sep = _angular_separation(100.0, 45.0, 100.0, 45.0)
        assert sep < 0.001

    def test_known_separation(self):
        sep = _angular_separation(0.0, 0.0, 1.0, 0.0)
        assert abs(sep - 60.0) < 0.1

    def test_symmetric(self):
        s1 = _angular_separation(10.0, 20.0, 30.0, 40.0)
        s2 = _angular_separation(30.0, 40.0, 10.0, 20.0)
        assert abs(s1 - s2) < 0.001


class TestLocalSearch:
    def test_finds_m42_near_orion(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, radius_arcmin=30.0)
        names = [o.name for o in result.objects]
        assert "M42" in names

    def test_returns_empty_for_void_region(self, atlas):
        result = atlas.query(ra_deg=180.0, dec_deg=89.0, radius_arcmin=10.0)
        assert len(result.objects) == 0

    def test_objects_sorted_by_distance(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, radius_arcmin=300.0)
        distances = [o.angular_distance_arcmin for o in result.objects]
        assert distances == sorted(distances)

    def test_finds_m81_m82_pair(self, atlas):
        result = atlas.query(ra_deg=148.9, dec_deg=69.3, radius_arcmin=30.0)
        names = [o.name for o in result.objects]
        assert "M81" in names
        assert "M82" in names


class TestSolveQuality:
    def test_excellent_rms(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, solve_rms_arcsec=0.3)
        assert result.solve_quality == 1.0

    def test_poor_rms(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, solve_rms_arcsec=10.0)
        assert result.solve_quality == 0.1

    def test_no_rms_defaults_to_half(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39)
        assert result.solve_quality == 0.5


class TestConfidence:
    def test_confidence_higher_with_objects(self, atlas):
        result_orion = atlas.query(ra_deg=83.82, dec_deg=-5.39, radius_arcmin=300.0)
        result_void = atlas.query(ra_deg=180.0, dec_deg=89.0, radius_arcmin=10.0)
        assert result_orion.confidence > result_void.confidence

    def test_confidence_bounded_0_1(self, atlas):
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, radius_arcmin=600.0, solve_rms_arcsec=0.1)
        assert 0.0 <= result.confidence <= 1.0


class TestOnlineQuery:
    @patch("astroai.inference.sky_atlas.httpx.Client")
    def test_simbad_fallback_when_few_local_results(self, mock_client_cls, atlas_online):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "main_id\tra\tdec\totype_txt\tV\n"
            "NGC 1234\t50.0\t-10.0\tGalaxy\t12.5\n"
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = atlas_online.query(ra_deg=50.0, dec_deg=-10.0, radius_arcmin=30.0)
        names = [o.name for o in result.objects]
        assert "NGC 1234" in names

    @patch("astroai.inference.sky_atlas.httpx.Client")
    def test_handles_simbad_failure_gracefully(self, mock_client_cls, atlas_online):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Network error")
        mock_client_cls.return_value = mock_client

        result = atlas_online.query(ra_deg=50.0, dec_deg=-10.0, radius_arcmin=30.0)
        assert isinstance(result, SkyAtlasResult)


class TestResultStructure:
    def test_result_contains_fov_info(self, atlas):
        result = atlas.query(ra_deg=100.0, dec_deg=30.0, radius_arcmin=45.0)
        assert result.fov_center_ra == 100.0
        assert result.fov_center_dec == 30.0
        assert result.search_radius_arcmin == 45.0

    def test_max_results_limit(self):
        atlas = SkyAtlas(use_online=False, max_results=3)
        result = atlas.query(ra_deg=83.82, dec_deg=-5.39, radius_arcmin=600.0)
        assert len(result.objects) <= 3
