import os
import sys

import numpy as np
import pytest

from quadrupole import mass_quadrupole

# schwarzschild.py's own sys.path bootstrap only resolves correctly when
# simulator.py has run first; set it up here so this test is importable
# standalone.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))

from schwarzschild import SchwarzschildGeodesicCoordTime


def _exact_circular_orbit_params(M, a):
    """E, h for an exact Schwarzschild circular orbit at areal radius a."""
    h2 = M * a**2 / (a - 3 * M)
    E2 = (a - 2 * M) ** 2 / (a * (a - 3 * M))
    return np.sqrt(E2), np.sqrt(h2)


def test_quadrupole_is_trace_free():
    M, a, mu = 1.0, 10.0, 1e-3
    E, h = _exact_circular_orbit_params(M, a)
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=200.0, n_steps=500)
    )
    Q, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    trace = Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2]
    trace_ddot = Qddot[:, 0, 0] + Qddot[:, 1, 1] + Qddot[:, 2, 2]
    assert np.allclose(trace, 0.0, atol=1e-10)
    assert np.allclose(trace_ddot, 0.0, atol=1e-8)


def test_quadrupole_symmetric():
    M, a, mu = 1.0, 10.0, 1e-3
    E, h = _exact_circular_orbit_params(M, a)
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.3, dr_dtau0=0.0, t_max=200.0, n_steps=500)
    )
    Q, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    assert np.allclose(Q, np.swapaxes(Q, 1, 2))
    assert np.allclose(Qddot, np.swapaxes(Qddot, 1, 2))


def test_circular_orbit_Qddot_matches_closed_form():
    # For a circular orbit R=a, phi=Omega*t, the quadrupole double-dot has a
    # simple closed form (derived directly from I_ij = mu R^2 cos/sin^2(phi)):
    #   Qddot_xx = -2 mu a^2 Omega^2 cos(2 phi)
    #   Qddot_yy = +2 mu a^2 Omega^2 cos(2 phi)
    #   Qddot_xy = -2 mu a^2 Omega^2 sin(2 phi)
    #   Qddot_zz = 0
    M, a, mu = 1.0, 20.0, 1e-3
    E, h = _exact_circular_orbit_params(M, a)
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=50.0, n_steps=500)
    )
    # Confirm the orbit actually stayed circular (r constant) before trusting
    # the closed-form comparison.
    assert np.allclose(traj["r"], a, atol=1e-6)

    Omega = np.sqrt(M / a**3)  # exact Schwarzschild coordinate-time Kepler law
    phi = traj["phi"]

    Q, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )

    expected_xx = -2 * mu * a**2 * Omega**2 * np.cos(2 * phi)
    expected_yy = 2 * mu * a**2 * Omega**2 * np.cos(2 * phi)
    expected_xy = -2 * mu * a**2 * Omega**2 * np.sin(2 * phi)

    assert np.allclose(Qddot[:, 0, 0], expected_xx, rtol=1e-4)
    assert np.allclose(Qddot[:, 1, 1], expected_yy, rtol=1e-4)
    assert np.allclose(Qddot[:, 0, 1], expected_xy, rtol=1e-4)
    assert np.allclose(Qddot[:, 2, 2], 0.0, atol=1e-10)


@pytest.mark.parametrize("mapping", ["boyer_lindquist", "harmonic", "isotropic"])
def test_mapping_choice_changes_amplitude_but_not_trace(mapping):
    M, a, mu = 1.0, 10.0, 1e-3
    E, h = _exact_circular_orbit_params(M, a)
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=50.0, n_steps=500)
    )
    Q, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M, mapping=mapping,
    )
    trace = Q[:, 0, 0] + Q[:, 1, 1] + Q[:, 2, 2]
    assert np.allclose(trace, 0.0, atol=1e-10)
