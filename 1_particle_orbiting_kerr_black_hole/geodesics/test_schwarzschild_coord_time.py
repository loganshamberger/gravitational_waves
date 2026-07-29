import os
import sys

import numpy as np
import pytest

# schwarzschild.py's own sys.path bootstrap only resolves correctly when
# simulator.py has run first; set it up here so this test is importable
# standalone.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))

from schwarzschild import SchwarzschildGeodesic, SchwarzschildGeodesicCoordTime

# Bound, non-plunging orbit (from config.yaml's baseline case) so both
# integrators run to completion without hitting the horizon.
PARAMS = dict(M=1.0, E=0.95, h=4.0, r0=10.0, phi0=0.0, dr_dtau0=0.0)


def test_coord_time_matches_tau_integrator_via_interpolation():
    tau_result = SchwarzschildGeodesic().simulate({**PARAMS, "tau_max": 500.0, "n_steps": 4000})
    t_result = SchwarzschildGeodesicCoordTime().simulate({**PARAMS, "t_max": 500.0, "n_steps": 4000})

    # Restrict to the t-range covered by both solutions.
    t_common_max = min(tau_result["t"][-1], t_result["t"][-1])
    mask = t_result["t"] <= t_common_max

    r_interp = np.interp(t_result["t"][mask], tau_result["t"], tau_result["r"])
    phi_interp = np.interp(t_result["t"][mask], tau_result["t"], tau_result["phi"])

    assert np.allclose(t_result["r"][mask], r_interp, atol=1e-5, rtol=1e-6)
    assert np.allclose(t_result["phi"][mask], phi_interp, atol=1e-5, rtol=1e-6)


def test_analytic_Rdot_matches_finite_difference():
    result = SchwarzschildGeodesicCoordTime().simulate({**PARAMS, "t_max": 500.0, "n_steps": 4000})
    Rdot_fd = np.gradient(result["r"], result["t"])
    # Loose tolerance: finite differencing is the thing we're avoiding, this
    # is a sanity cross-check, not the primary correctness test.
    assert np.allclose(result["Rdot"], Rdot_fd, atol=1e-2, rtol=1e-2)


def test_analytic_Rddot_matches_finite_difference_of_Rdot():
    result = SchwarzschildGeodesicCoordTime().simulate({**PARAMS, "t_max": 500.0, "n_steps": 4000})
    Rddot_fd = np.gradient(result["Rdot"], result["t"])
    assert np.allclose(result["Rddot"], Rddot_fd, atol=5e-2, rtol=5e-2)


def test_plunging_orbit_trims_trailing_samples():
    # Coordinate time diverges near the horizon (infinite redshift), so t_max
    # must be well past the ~65 (t units) it actually takes to approach rs.
    plunge_params = dict(M=1.0, E=1.0, h=0.0, r0=20.0, phi0=0.0, dr_dtau0=-0.3162278, t_max=200.0)
    result = SchwarzschildGeodesicCoordTime().simulate(plunge_params)
    assert result["hit_horizon"]
    assert len(result["t"]) > 0
    rs = 2.0
    # No trailing sample should sit right on the near-horizon event threshold.
    assert np.all((1 - rs / result["r"]) > 1e-3)
