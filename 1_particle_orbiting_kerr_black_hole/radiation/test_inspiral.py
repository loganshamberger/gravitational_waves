import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))

from inspiral import SchwarzschildAdiabaticInspiral, _da_dt, _omega
from schwarzschild import circular_orbit_E_h, circular_orbit_dE_da


def test_circular_orbit_dE_da_matches_finite_difference():
    M = 1.0
    a = np.linspace(6.5, 100.0, 30) * M
    da = 1e-4
    dE_da = circular_orbit_dE_da(M, a)

    E_plus, _ = circular_orbit_E_h(M, a + da)
    E_minus, _ = circular_orbit_E_h(M, a - da)
    dE_da_fd = (E_plus - E_minus) / (2 * da)

    assert np.allclose(dE_da, dE_da_fd, rtol=1e-5)


def test_da_dt_matches_newtonian_weak_field():
    # da/dt -> -(64/5) M^2 mu / a^3 (Peters 1964) as a -> infinity.
    M, mu = 1.0, 1e-4
    a = 1000.0 * M
    da_dt = _da_dt(a, M, mu)
    expected = -(64.0 / 5.0) * M**2 * mu / a**3
    assert da_dt == pytest.approx(expected, rel=1e-2)


def test_da_dt_is_negative_above_isco():
    M, mu = 1.0, 1e-4
    for a in (6.5, 10.0, 50.0, 500.0):
        assert _da_dt(a * M, M, mu) < 0


def _run_inspiral(M=1.0, mu=1e-4, a0=20.0, t_max=None, n_steps=2000):
    if t_max is None:
        # Long enough to noticeably shrink a for this mu, short of the ISCO.
        t_max = 5e6
    result = SchwarzschildAdiabaticInspiral().simulate(
        dict(M=M, mu=mu, a0=a0, phi0=0.0, t_max=t_max, n_steps=n_steps)
    )
    return result


def test_radius_decreases_monotonically():
    result = _run_inspiral()
    r = result["r"]
    assert np.all(np.diff(r) <= 0)
    assert r[-1] < r[0]


def test_orbital_frequency_increases_chirp():
    result = _run_inspiral()
    Omega = _omega(result["r"], result["params"]["M"])
    assert np.all(np.diff(Omega) >= 0)
    assert Omega[-1] > Omega[0]


def test_stops_before_isco():
    # a0 close enough to the ISCO, and t_max generous enough, that the
    # inspiral should reach the ISCO stopping event well within the run
    # (da/dt ~ -5e-5 near a=8M for these params, so ~1e5 is ample) --
    # terminate on the event rather than integrating a into or through it.
    M = 1.0
    result = SchwarzschildAdiabaticInspiral().simulate(
        dict(M=M, mu=1e-3, a0=8.0, phi0=0.0, t_max=2e5, n_steps=5000)
    )
    assert result["reached_isco"]
    assert result["r"][-1] > 6.0 * M
    assert np.all(result["r"] > 6.0 * M)


def test_rddot_and_phiddot_are_finite_and_small():
    # Rddot, phiddot are numerically differentiated (from analytic Rdot,
    # phidot); sanity-check they're well-behaved and much smaller than the
    # leading O(1) circular-orbit terms (Omega^2 a, Omega), consistent with
    # mu being a small parameter.
    result = _run_inspiral()
    M = result["params"]["M"]
    assert np.all(np.isfinite(result["Rddot"]))
    assert np.all(np.isfinite(result["phiddot"]))
    Omega = _omega(result["r"], M)
    assert np.all(np.abs(result["phiddot"]) < Omega**2)
