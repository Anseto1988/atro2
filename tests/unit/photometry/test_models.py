from __future__ import annotations

from astroai.engine.photometry.models import PhotometryResult, StarMeasurement


class TestStarMeasurement:
    def test_required_fields(self) -> None:
        s = StarMeasurement(
            star_id=1, ra=180.0, dec=45.0,
            x_pixel=100.0, y_pixel=200.0, instr_mag=-12.5,
        )
        assert s.star_id == 1
        assert s.ra == 180.0
        assert s.dec == 45.0
        assert s.x_pixel == 100.0
        assert s.y_pixel == 200.0
        assert s.instr_mag == -12.5

    def test_default_optional_fields(self) -> None:
        s = StarMeasurement(
            star_id=0, ra=0.0, dec=0.0,
            x_pixel=0.0, y_pixel=0.0, instr_mag=0.0,
        )
        assert s.catalog_mag == 0.0
        assert s.cal_mag == 0.0
        assert s.residual == 0.0

    def test_optional_fields_set(self) -> None:
        s = StarMeasurement(
            star_id=2, ra=10.0, dec=20.0,
            x_pixel=50.0, y_pixel=60.0, instr_mag=-11.0,
            catalog_mag=10.5, cal_mag=10.6, residual=0.1,
        )
        assert s.catalog_mag == 10.5
        assert s.cal_mag == 10.6
        assert s.residual == 0.1


class TestPhotometryResult:
    def test_empty_result(self) -> None:
        r = PhotometryResult()
        assert r.stars == []
        assert r.r_squared == 0.0
        assert r.n_matched == 0

    def test_result_with_stars(self) -> None:
        stars = [
            StarMeasurement(
                star_id=i, ra=float(i), dec=float(i),
                x_pixel=float(i), y_pixel=float(i), instr_mag=-10.0,
            )
            for i in range(5)
        ]
        r = PhotometryResult(stars=stars, r_squared=0.98, n_matched=5)
        assert len(r.stars) == 5
        assert r.r_squared == 0.98
        assert r.n_matched == 5

    def test_stars_list_is_independent(self) -> None:
        r1 = PhotometryResult()
        r2 = PhotometryResult()
        r1.stars.append(
            StarMeasurement(
                star_id=0, ra=0.0, dec=0.0,
                x_pixel=0.0, y_pixel=0.0, instr_mag=0.0,
            )
        )
        assert len(r2.stars) == 0
