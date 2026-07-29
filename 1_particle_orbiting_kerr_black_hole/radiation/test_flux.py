import os
import sys

import numpy as np
import pytest

from flux import angular_momentum_flux, energy_flux, orbit_period_average, quadrupole_jerk
from quadrupole import mass_quadrupole

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))

from schwarzschild import SchwarzschildGeodesicCoordTime


def _exact_circular_orbit_params(M, a):
    h2 = M * a**2 / (a - 3 * M)
    E2 = (a - 2 * M) ** 2 / (a * (a - 3 * M))
    return np.sqrt(E2), np.sqrt(h2)


def _circular_orbit_Qddot(M=1.0, a=20.0, mu=1e-3, n_periods=6, n_steps_per_period=200):
    E, h = _exact_circular_orbit_params(M, a)
    Omega = np.sqrt(M / a**3)
    period = 2 * np.pi / Omega
    t_max = n_periods * period
    n_steps = n_periods * n_steps_per_period
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=t_max, n_steps=n_steps)
    )
    _, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    return traj, Qddot, M, a, mu, Omega, period


def test_quadrupole_jerk_matches_finite_difference():
    traj, Qddot, M, a, mu, Omega, period = _circular_orbit_Qddot()
    t = traj["t"]
    Qdddot = quadrupole_jerk(Qddot, t)

    dt = t[1] - t[0]
    Qdddot_fd = np.zeros_like(Qddot)
    Qdddot_fd[1:-1] = (Qddot[2:] - Qddot[:-2]) / (2 * dt)
    assert np.allclose(Qdddot[2:-2], Qdddot_fd[2:-2], rtol=1e-3, atol=1e-10)


def test_energy_flux_matches_closed_form_circular_orbit():
    # dE/dt = (32/5) mu^2 M^3 / a^5
    traj, Qddot, M, a, mu, Omega, period = _circular_orbit_Qddot()
    t = traj["t"]
    Qdddot = quadrupole_jerk(Qddot, t)
    dEdt = energy_flux(Qdddot)

    expected = 32.0 / 5.0 * mu**2 * M**3 / a**5
    averaged = orbit_period_average(dEdt, t, period)
    assert averaged == pytest.approx(expected, rel=1e-3)


def test_energy_flux_is_nonnegative():
    traj, Qddot, M, a, mu, Omega, period = _circular_orbit_Qddot()
    t = traj["t"]
    Qdddot = quadrupole_jerk(Qddot, t)
    dEdt = energy_flux(Qdddot)
    assert np.all(dEdt >= -1e-12)


def test_angular_momentum_flux_is_along_z_for_equatorial_orbit():
    traj, Qddot, M, a, mu, Omega, period = _circular_orbit_Qddot()
    t = traj["t"]
    Qdddot = quadrupole_jerk(Qddot, t)
    dLdt = angular_momentum_flux(Qddot, Qdddot)

    dLx = orbit_period_average(dLdt[:, 0], t, period)
    dLy = orbit_period_average(dLdt[:, 1], t, period)
    dLz = orbit_period_average(dLdt[:, 2], t, period)

    assert dLx == pytest.approx(0.0, abs=1e-12)
    assert dLy == pytest.approx(0.0, abs=1e-12)
    assert dLz != pytest.approx(0.0, abs=1e-12)


def test_orbit_period_average_rejects_short_span():
    traj, Qddot, M, a, mu, Omega, period = _circular_orbit_Qddot(n_periods=1, n_steps_per_period=3)
    t = traj["t"]
    with pytest.raises(ValueError):
        orbit_period_average(Qddot, t, period=10 * period)
