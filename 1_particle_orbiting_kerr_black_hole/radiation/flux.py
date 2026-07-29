"""Energy and angular momentum flux from the quadrupole formula, per
notes/gravitational_wave_quadrupole_report.md section 5.

Spacetime-agnostic: depends only on Qddot(t) (and the time grid), so it is
reusable for Kerr unchanged.

Units: G = c = 1, consistent with the rest of the project.
"""

import numpy as np

_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0


def quadrupole_jerk(Qddot, t):
    """Q_dddot(t) via a single numerical derivative of the analytic Qddot(t)
    (report section 4c, option 2: differentiating the already-analytic Qddot
    once is well-conditioned, unlike finite-differencing the trajectory
    itself twice).

    t must be a uniform grid (as produced by the coordinate-time
    integrators). Uses `np.gradient`, a second-order central difference with
    one-sided differences at the endpoints.
    """
    dt = t[1] - t[0]
    return np.gradient(Qddot, dt, axis=0)


def energy_flux(Qdddot):
    """Instantaneous dE/dt(t) = (1/5) Qdddot_ij Qdddot_ij (sum over i, j)."""
    return 0.2 * np.einsum("tij,tij->t", Qdddot, Qdddot)


def angular_momentum_flux(Qddot, Qdddot):
    """Instantaneous dL_i/dt(t) = (2/5) eps_ijk Qddot_jl Qdddot_kl (sum over
    j, k, l), returned as shape (N, 3).
    """
    return 0.4 * np.einsum("ijk,tjl,tkl->ti", _EPS, Qddot, Qdddot)


def orbit_period_average(x, t, period):
    """Average x(t) (shape (N, ...)) over the trailing one orbital period,
    per the report's angle-bracket time-averaging. Returns a single value
    (or array, if x has trailing dimensions) averaged over the last
    `period` of the trajectory, matching the report's use of period-averaged
    fluxes rather than instantaneous ones.
    """
    if t[-1] - t[0] < period:
        raise ValueError("time span shorter than one orbital period")
    mask = t >= t[-1] - period
    return np.trapezoid(x[mask], t[mask], axis=0) / (t[mask][-1] - t[mask][0])
