import numpy as np
import pytest

from mapping import MAPPINGS, equatorial_cartesian, radial_mapping, radial_mapping_derivatives


def test_unknown_mapping_raises():
    with pytest.raises(ValueError):
        radial_mapping(10.0, M=1.0, mapping="nope")


def test_boyer_lindquist_is_identity():
    r = np.array([3.0, 6.0, 10.0, 100.0])
    assert np.array_equal(radial_mapping(r, M=1.0, mapping="boyer_lindquist"), r)


def test_mappings_converge_at_large_r():
    M = 1.0
    r = 1e6 * M
    R_bl = radial_mapping(r, M, "boyer_lindquist")
    R_harm = radial_mapping(r, M, "harmonic")
    R_iso = radial_mapping(r, M, "isotropic")
    assert R_harm == pytest.approx(R_bl, rel=1e-4)
    assert R_iso == pytest.approx(R_bl, rel=1e-4)


def test_mappings_disagree_in_strong_field():
    # At the ISCO (r = 6M) the three mappings should differ by O(M/r) ~ 17%,
    # not agree -- this spread IS the systematic-error estimate.
    M = 1.0
    r = 6.0 * M
    R_bl = radial_mapping(r, M, "boyer_lindquist")
    R_harm = radial_mapping(r, M, "harmonic")
    R_iso = radial_mapping(r, M, "isotropic")
    assert abs(R_bl - R_harm) / R_bl > 0.05
    assert abs(R_bl - R_iso) / R_bl > 0.01
    assert R_harm != R_iso


def test_isotropic_sqrt_argument_nonnegative_outside_horizon():
    M = 1.0
    rs = 2 * M
    r = np.linspace(rs * 1.001, 50 * M, 500)
    assert np.all(r**2 - 2 * M * r >= 0)
    R_iso = radial_mapping(r, M, "isotropic")
    assert np.all(np.isfinite(R_iso))


def test_equatorial_cartesian_matches_polar_radius():
    M = 1.0
    r = np.array([6.0, 10.0, 30.0])
    phi = np.array([0.0, np.pi / 3, np.pi])
    for mapping in MAPPINGS:
        x, y = equatorial_cartesian(r, phi, M, mapping=mapping)
        R = radial_mapping(r, M, mapping)
        assert np.allclose(np.hypot(x, y), R)


def test_equatorial_cartesian_vectorized_shapes():
    M = 1.0
    r = np.linspace(6.0, 20.0, 50)
    phi = np.linspace(0.0, 4 * np.pi, 50)
    x, y = equatorial_cartesian(r, phi, M)
    assert x.shape == r.shape
    assert y.shape == r.shape


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_radial_mapping_derivatives_match_finite_difference(mapping):
    M = 1.0
    r = np.linspace(6.5, 40.0, 30)
    dr = 1e-3
    dR_dr, d2R_dr2 = radial_mapping_derivatives(r, M, mapping)

    R_plus = radial_mapping(r + dr, M, mapping)
    R_minus = radial_mapping(r - dr, M, mapping)
    dR_dr_fd = (R_plus - R_minus) / (2 * dr)
    d2R_dr2_fd = (R_plus - 2 * radial_mapping(r, M, mapping) + R_minus) / dr**2

    assert np.allclose(dR_dr, dR_dr_fd, atol=1e-6, rtol=1e-6)
    assert np.allclose(d2R_dr2, d2R_dr2_fd, atol=1e-3, rtol=1e-3)


def test_boyer_lindquist_and_harmonic_derivatives_are_trivial():
    r = np.array([6.0, 10.0, 100.0])
    for mapping in ("boyer_lindquist", "harmonic"):
        dR_dr, d2R_dr2 = radial_mapping_derivatives(r, M=1.0, mapping=mapping)
        assert np.all(dR_dr == 1.0)
        assert np.all(d2R_dr2 == 0.0)