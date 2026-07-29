"""Adiabatic radiation-reaction inspiral for a Schwarzschild circular orbit,
per notes/gravitational_wave_quadrupole_report.md section 7.

Physical basis: valid in the extreme-mass-ratio limit (mu << M), where the
radiation-reaction timescale a/|da/dt| is much longer than the orbital
period, so the orbit is well-approximated at every instant by the exact
circular geodesic for the current a (a "sequence of circular orbits"). This
breaks down near the ISCO (a -> 6M), where dE/da -> 0 and da/dt formally
diverges -- the same transition-to-plunge regime where the adiabatic
approximation is known to fail in the EMRI literature. Integration is
stopped just above the ISCO for that reason, not as an arbitrary cutoff.

Registered as its own System (SchwarzschildAdiabaticInspiral) rather than a
flag on the existing geodesic Systems, per the "radiation reaction must be
composable in/out" decision -- see whatidid space particle-orbiting-kerr.
core.py is unchanged.

The da/dt right-hand side needs an instantaneous flux at a single radius,
not a time series, so it reuses the closed-form circular-orbit flux
dE/dt = (32/5) mu^2 M^3 / a^5 (validated in flux.py's tests) rather than
calling quadrupole.py/flux.py at each step -- those operate on a finished
trajectory's Q-dddot(t), which doesn't exist yet during integration.

Once produced, the resulting (r=a(t), phi(t)) trajectory has the same shape
(t, r, phi, Rdot, phidot, Rddot, phiddot) as
geodesics.schwarzschild.SchwarzschildGeodesicCoordTime's output, so it flows
through mass_quadrupole/quadrupole_jerk/energy_flux/strain_plus_cross
unmodified to produce the actual chirping waveform.

Units: G = c = 1, consistent with the rest of the project.
"""

import os
import sys
from typing import Any, Dict

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from core import MissingParameterError, System
from schwarzschild import circular_orbit_dE_da
from visualization import plot_orbit_panels

# Fraction of the ISCO radius (6M) at which the adiabatic sequence-of-
# circular-orbits approximation is stopped, since dE/da -> 0 there and da/dt
# formally diverges (see module docstring).
_ISCO_FRACTION = 1.001


def _da_dt(a, M, mu):
    dE_da = circular_orbit_dE_da(M, a)
    return -(32.0 / 5.0) * mu * M**3 / (a**5 * dE_da)


def _omega(a, M):
    return np.sqrt(M / a**3)


class SchwarzschildAdiabaticInspiral(System):
    """Adiabatic inspiral of a circular Schwarzschild orbit under quadrupole
    radiation reaction.

    State integrated in coordinate time t: [a, phi], where a is the
    (instantaneously circular) orbital radius.
    """

    REQUIRED_PARAMETERS = (
        "M",       # black hole mass
        "mu",      # orbiting particle's mass (test-particle limit: mu << M)
        "a0",      # initial orbital radius (a0 > 6M for a stable start)
        "phi0",    # initial azimuthal angle
        "t_max",   # coordinate time to integrate to
    )

    def validate(self, params: Dict[str, Any]) -> None:
        missing = [p for p in self.REQUIRED_PARAMETERS if p not in params]
        if missing:
            raise MissingParameterError(
                f"{type(self).__name__} is missing required parameters: {missing}"
            )

    def simulate(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        M = params["M"]
        mu = params["mu"]
        isco = 6.0 * M
        n_steps = params.get("n_steps", 2000)

        y0 = [params["a0"], params["phi0"]]
        t_eval = np.linspace(0, params["t_max"], n_steps)

        def rhs(t, y):
            a, _ = y
            return [_da_dt(a, M, mu), _omega(a, M)]

        def hit_isco(t, y):
            return y[0] - isco * _ISCO_FRACTION
        hit_isco.terminal = True
        hit_isco.direction = -1

        sol = solve_ivp(
            rhs,
            [0, params["t_max"]],
            y0,
            t_eval=t_eval,
            events=hit_isco,
            rtol=1e-10,
            atol=1e-12,
        )

        t, a, phi = sol.t, sol.y[0], sol.y[1]
        reached_isco = sol.status == 1

        Rdot = _da_dt(a, M, mu)
        phidot = _omega(a, M)
        # Rddot, phiddot are O(mu) and O(mu) respectively but algebraically
        # messy to differentiate analytically a second time; since Rdot and
        # phidot are themselves smooth analytic functions of a(t), a single
        # numerical derivative is well-conditioned (same reasoning as
        # flux.py's quadrupole_jerk).
        dt = t[1] - t[0]
        Rddot = np.gradient(Rdot, dt)
        phiddot = np.gradient(phidot, dt)

        return {
            "t": t,
            "r": a,
            "phi": phi,
            "Rdot": Rdot,
            "phidot": phidot,
            "Rddot": Rddot,
            "phiddot": phiddot,
            "reached_isco": reached_isco,
            "params": {"M": M, "mu": mu, "a0": params["a0"], "phi0": params["phi0"]},
        }

    def visualize(self, result: Dict[str, np.ndarray]) -> None:
        p = result["params"]
        rs = 2 * p["M"]
        filename = (
            f"schwarzschild_inspiral_M{p['M']:g}_mu{p['mu']:g}"
            f"_a0{p['a0']:g}_phi0{p['phi0']:g}.png"
        )
        plot_orbit_panels(
            time_values=result["t"],
            r=result["r"],
            phi=result["phi"],
            rs=rs,
            time_label="t (coordinate time)",
            title="Schwarzschild adiabatic inspiral (radiation reaction)",
            filename=filename,
        )
