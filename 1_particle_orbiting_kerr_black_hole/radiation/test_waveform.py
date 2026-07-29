import os
import sys

import numpy as np
import pytest

from quadrupole import mass_quadrupole
from waveform import polarization_basis, strain_plus_cross

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))

from schwarzschild import SchwarzschildGeodesicCoordTime


def _exact_circular_orbit_params(M, a):
    h2 = M * a**2 / (a - 3 * M)
    E2 = (a - 2 * M) ** 2 / (a * (a - 3 * M))
    return np.sqrt(E2), np.sqrt(h2)


def _circular_orbit_Qddot(M=1.0, a=20.0, mu=1e-3, t_max=50.0, n_steps=500):
    E, h = _exact_circular_orbit_params(M, a)
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=t_max, n_steps=n_steps)
    )
    _, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    return traj, Qddot, M, a, mu


def test_polarization_basis_orthonormal_away_from_poles():
    theta, phi = 0.7, 1.3
    n_hat, e_theta, e_phi = polarization_basis(theta, phi)
    for v in (n_hat, e_theta, e_phi):
        assert np.isclose(np.dot(v, v), 1.0)
    assert np.isclose(np.dot(n_hat, e_theta), 0.0)
    assert np.isclose(np.dot(n_hat, e_phi), 0.0)
    assert np.isclose(np.dot(e_theta, e_phi), 0.0)


@pytest.mark.parametrize("theta_obs", [0.0, np.pi])
def test_poles_are_numerically_safe(theta_obs):
    traj, Qddot, M, a, mu = _circular_orbit_Qddot()
    for phi_obs in (0.0, 1.0, np.pi, 5.0):
        h_plus, h_cross = strain_plus_cross(Qddot, theta_obs, phi_obs, D=1000.0)
        assert np.all(np.isfinite(h_plus))
        assert np.all(np.isfinite(h_cross))


def test_face_on_circular_polarization():
    # theta=0 (face-on): |h_+| = |h_x|, amplitude 4*mu*M/(D*a), 90 deg out of phase.
    traj, Qddot, M, a, mu = _circular_orbit_Qddot()
    D = 1000.0
    h_plus, h_cross = strain_plus_cross(Qddot, theta_obs=0.0, phi_obs=0.0, D=D)

    amp = 4 * mu * M / (D * a)
    assert np.allclose(h_plus**2 + h_cross**2, amp**2, rtol=1e-4)

    Phi = traj["phi"]
    expected_plus = -amp * np.cos(2 * Phi)
    expected_cross = -amp * np.sin(2 * Phi)
    assert np.allclose(h_plus, expected_plus, atol=1e-4 * amp)
    assert np.allclose(h_cross, expected_cross, atol=1e-4 * amp)


def test_edge_on_cross_polarization_vanishes_exactly():
    # theta=pi/2 (edge-on): h_x = 0 exactly -- the strongest test of the TT
    # projection, per the report.
    traj, Qddot, M, a, mu = _circular_orbit_Qddot()
    D = 1000.0
    h_plus, h_cross = strain_plus_cross(Qddot, theta_obs=np.pi / 2, phi_obs=0.0, D=D)

    assert np.allclose(h_cross, 0.0, atol=1e-12)

    amp = 2 * mu * M / (D * a)
    Phi = traj["phi"]
    expected_plus = -amp * np.cos(2 * Phi)
    assert np.allclose(h_plus, expected_plus, rtol=1e-4)


@pytest.mark.parametrize("iota", [0.0, np.pi / 6, np.pi / 3, np.pi / 2])
def test_general_inclination_matches_closed_form(iota):
    # h_+ = -(2 mu M)/(D a) (1+cos^2(iota)) cos(2 Phi)
    # h_x = -(4 mu M)/(D a)      cos(iota)  sin(2 Phi)
    traj, Qddot, M, a, mu = _circular_orbit_Qddot()
    D = 1000.0
    h_plus, h_cross = strain_plus_cross(Qddot, theta_obs=iota, phi_obs=0.0, D=D)

    Phi = traj["phi"]
    expected_plus = -(2 * mu * M) / (D * a) * (1 + np.cos(iota) ** 2) * np.cos(2 * Phi)
    expected_cross = -(4 * mu * M) / (D * a) * np.cos(iota) * np.sin(2 * Phi)

    assert np.allclose(h_plus, expected_plus, rtol=1e-3, atol=1e-15)
    assert np.allclose(h_cross, expected_cross, rtol=1e-3, atol=1e-15)
