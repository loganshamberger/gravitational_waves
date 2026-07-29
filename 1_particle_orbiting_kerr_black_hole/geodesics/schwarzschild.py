"""Schwarzschild-spacetime Systems: massive-particle and photon geodesics.

Units: G = c = 1 throughout, so mass and distance share units (e.g.
multiples of M).
"""

import logging
import os
import sys
from typing import Any, Dict

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from core import MissingParameterError, System
from visualization import plot_orbit_panels

logger = logging.getLogger(__name__)

# Horizon-proximity fraction (of rs) at which a geodesic is considered to have
# hit the horizon; also used to trim badly-conditioned trailing samples where
# dt/dtau diverges.
_HORIZON_FRACTION = 1.001


def _dt_dtau(r, E, rs):
    """Gravitational redshift factor: coordinate-time rate per unit proper time."""
    return E / (1 - rs / r)


def _dphi_dtau(r, h):
    """Angular rate per unit proper time (specific angular momentum conservation)."""
    return h / r**2


def _dpr_dtau(r, M, h):
    """Radial geodesic acceleration, d(dr/dtau)/dtau, from the effective potential."""
    return -M / r**2 + h**2 / r**3 - 3 * M * h**2 / r**4


def _coordinate_time_derivatives(r, pr, M, E, h, rs):
    """Analytic Rdot, phidot, Rddot, phiddot (coordinate-time derivatives).

    Derived by the chain rule from the tau-derivatives above, using only
    already-integrated state (r, pr = dr/dtau) -- zero numerical
    differentiation, so no amplification of solver noise. Feeds the
    quadrupole radiation module (see notes/gravitational_wave_quadrupole_report.md
    section 4b).
    """
    dt_dtau = _dt_dtau(r, E, rs)
    dphi_dtau = _dphi_dtau(r, h)
    dpr_dtau = _dpr_dtau(r, M, h)

    Rdot = pr / dt_dtau
    phidot = dphi_dtau / dt_dtau

    ddt_dtau_dr = E * rs / (r**2 * (1 - rs / r) ** 2)
    ddt_dtau_dt = ddt_dtau_dr * Rdot
    dpr_dt = dpr_dtau / dt_dtau
    Rddot = (dpr_dt * dt_dtau - pr * ddt_dtau_dt) / dt_dtau**2

    ddphi_dtau_dr = -2 * h / r**3
    ddphi_dtau_dt = ddphi_dtau_dr * Rdot
    phiddot = (ddphi_dtau_dt * dt_dtau - dphi_dtau * ddt_dtau_dt) / dt_dtau**2

    return Rdot, phidot, Rddot, phiddot


def circular_orbit_E_h(M, a):
    """Exact specific energy and angular momentum of a Schwarzschild circular
    orbit at areal radius a (only defined for a > 3M; physically stable only
    for a >= 6M, the ISCO).
    """
    h2 = M * a**2 / (a - 3 * M)
    E2 = (a - 2 * M) ** 2 / (a * (a - 3 * M))
    return np.sqrt(E2), np.sqrt(h2)


def circular_orbit_dE_da(M, a):
    """d(specific energy)/da for a Schwarzschild circular orbit, from
    differentiating circular_orbit_E_h's E(a)^2 closed form analytically.
    Used by the adiabatic-inspiral radiation-reaction module to convert an
    energy-loss rate into a rate of change of orbital radius.
    """
    E, _ = circular_orbit_E_h(M, a)
    f = (a - 2 * M) ** 2
    g = a * (a - 3 * M)
    df_da = 2 * (a - 2 * M)
    dg_da = 2 * a - 3 * M
    dE2_da = (df_da * g - f * dg_da) / g**2
    return dE2_da / (2 * E)


class SchwarzschildGeodesic(System):
    """Timelike geodesic of a massive test particle in Schwarzschild spacetime.

    State integrated in proper time tau: [t, r, phi, dr/dtau].
    """

    REQUIRED_PARAMETERS = (
        "M",         # black hole mass
        "E",         # specific energy, E / (m c^2)
        "h",         # specific angular momentum, L / mu
        "r0",        # initial radial coordinate
        "phi0",      # initial azimuthal angle
        "dr_dtau0",  # initial dr/dtau
        "tau_max",   # proper time to integrate to
    )

    def validate(self, params: Dict[str, Any]) -> None:
        missing = [p for p in self.REQUIRED_PARAMETERS if p not in params]
        if missing:
            raise MissingParameterError(
                f"{type(self).__name__} is missing required parameters: {missing}"
            )

    def simulate(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        M = params["M"]
        E = params["E"]
        h = params["h"]
        rs = 2 * M
        logger.info("%s starting: rs=%.6g, params=%s", type(self).__name__, rs, params)
        t0 = params.get("t0", 0.0)
        n_steps = params.get("n_steps", 2000)

        y0 = [t0, params["r0"], params["phi0"], params["dr_dtau0"]]
        tau_eval = np.linspace(0, params["tau_max"], n_steps)

        def rhs(tau, y):
            _, r, _, pr = y
            return [_dt_dtau(r, E, rs), pr, _dphi_dtau(r, h), _dpr_dtau(r, M, h)]

        def hit_horizon(tau, y):
            return y[1] - rs * 1.001
        hit_horizon.terminal = True
        hit_horizon.direction = -1

        sol = solve_ivp(
            rhs,
            [0, params["tau_max"]],
            y0,
            t_eval=tau_eval,
            events=hit_horizon,
            rtol=1e-9,
            atol=1e-9,
        )

        return {
            "tau": sol.t,
            "t": sol.y[0],
            "r": sol.y[1],
            "phi": sol.y[2],
            "dr_dtau": sol.y[3],
            "hit_horizon": sol.status == 1,
            "params": {"M": M, "E": E, "h": h, "r0": params["r0"], "phi0": params["phi0"]},
        }

    def visualize(self, result: Dict[str, np.ndarray]) -> None:
        p = result["params"]
        rs = 2 * p["M"]
        filename = (
            f"schwarzschild_M{p['M']:g}_E{p['E']:g}_h{p['h']:g}"
            f"_r0{p['r0']:g}_phi0{p['phi0']:g}.png"
        )
        plot_orbit_panels(
            time_values=result["t"],
            r=result["r"],
            phi=result["phi"],
            rs=rs,
            time_label="t (coordinate time)",
            title="Schwarzschild geodesic",
            filename=filename,
        )


class SchwarzschildGeodesicCoordTime(System):
    """Timelike geodesic of a massive test particle in Schwarzschild spacetime,
    integrated on a uniform coordinate-time grid instead of proper time.

    Shares its physics (_dt_dtau, _dphi_dtau, _dpr_dtau) with
    SchwarzschildGeodesic rather than re-deriving it -- the two are the same
    orbit, just parameterized differently. Proper time is the natural
    parameter for the orbit; coordinate time is the natural parameter for a
    gravitational-wave waveform (the quadrupole formula's t), which is why
    this integrator exists alongside, not instead of, the tau one.

    State integrated in coordinate time t: [r, phi, dr/dtau]. Keeping
    dr/dtau (rather than dr/dt) as the third state component means each
    tau-derivative just gets divided by dt/dtau -- no quotient rule needed
    in the RHS itself.
    """

    REQUIRED_PARAMETERS = (
        "M",         # black hole mass
        "E",         # specific energy, E / (m c^2)
        "h",         # specific angular momentum, L / mu
        "r0",        # initial radial coordinate
        "phi0",      # initial azimuthal angle
        "dr_dtau0",  # initial dr/dtau
        "t_max",     # coordinate time to integrate to
    )

    def validate(self, params: Dict[str, Any]) -> None:
        missing = [p for p in self.REQUIRED_PARAMETERS if p not in params]
        if missing:
            raise MissingParameterError(
                f"{type(self).__name__} is missing required parameters: {missing}"
            )

    def simulate(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        M = params["M"]
        E = params["E"]
        h = params["h"]
        rs = 2 * M
        logger.info("%s starting: rs=%.6g, params=%s", type(self).__name__, rs, params)
        n_steps = params.get("n_steps", 2000)

        y0 = [params["r0"], params["phi0"], params["dr_dtau0"]]
        t_eval = np.linspace(0, params["t_max"], n_steps)

        def rhs(t, y):
            r, _, pr = y
            dt_dtau = _dt_dtau(r, E, rs)
            return [pr / dt_dtau, _dphi_dtau(r, h) / dt_dtau, _dpr_dtau(r, M, h) / dt_dtau]

        def hit_horizon(t, y):
            return y[0] - rs * _HORIZON_FRACTION
        hit_horizon.terminal = True
        hit_horizon.direction = -1

        sol = solve_ivp(
            rhs,
            [0, params["t_max"]],
            y0,
            t_eval=t_eval,
            events=hit_horizon,
            rtol=1e-9,
            atol=1e-9,
        )

        t, r, phi, pr = sol.t, sol.y[0], sol.y[1], sol.y[2]
        hit_horizon_flag = sol.status == 1
        if hit_horizon_flag:
            # dt/dtau grows large (not yet infinite) near the horizon
            # threshold; drop trailing samples once the redshift factor gets
            # badly conditioned, rather than a fixed sample count (which can
            # over-trim a coarse grid down to nothing useful).
            well_conditioned = np.where(1 - rs / r > 1e-3)[0]
            if len(well_conditioned) > 0:
                cutoff = well_conditioned[-1] + 1
                t, r, phi, pr = t[:cutoff], r[:cutoff], phi[:cutoff], pr[:cutoff]

        Rdot, phidot, Rddot, phiddot = _coordinate_time_derivatives(r, pr, M, E, h, rs)

        return {
            "t": t,
            "r": r,
            "phi": phi,
            "dr_dtau": pr,
            "Rdot": Rdot,
            "phidot": phidot,
            "Rddot": Rddot,
            "phiddot": phiddot,
            "hit_horizon": hit_horizon_flag,
            "params": {"M": M, "E": E, "h": h, "r0": params["r0"], "phi0": params["phi0"]},
        }

    def visualize(self, result: Dict[str, np.ndarray]) -> None:
        p = result["params"]
        rs = 2 * p["M"]
        filename = (
            f"schwarzschild_coordtime_M{p['M']:g}_E{p['E']:g}_h{p['h']:g}"
            f"_r0{p['r0']:g}_phi0{p['phi0']:g}.png"
        )
        plot_orbit_panels(
            time_values=result["t"],
            r=result["r"],
            phi=result["phi"],
            rs=rs,
            time_label="t (coordinate time)",
            title="Schwarzschild geodesic (coordinate-time integrator)",
            filename=filename,
        )


class SchwarzschildPhotonGeodesic(System):
    """Null geodesic of a photon in Schwarzschild spacetime.

    Energy is normalized to 1 (affine-parameter rescaling freedom), so the
    orbit is characterized by the impact parameter b = L/E instead of
    separate energy/angular-momentum constants.
    State integrated in the affine parameter lambda: [t, r, phi, dr/dlambda].
    """

    REQUIRED_PARAMETERS = (
        "M",             # black hole mass
        "b",             # impact parameter, L / E
        "r0",            # initial radial coordinate
        "phi0",          # initial azimuthal angle
        "dr_dlambda0",   # initial dr/dlambda
        "lambda_max",    # affine parameter to integrate to
    )

    def validate(self, params: Dict[str, Any]) -> None:
        missing = [p for p in self.REQUIRED_PARAMETERS if p not in params]
        if missing:
            raise MissingParameterError(
                f"{type(self).__name__} is missing required parameters: {missing}"
            )

    def simulate(self, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        M = params["M"]
        b = params["b"]
        rs = 2 * M
        logger.info("%s starting: rs=%.6g, params=%s", type(self).__name__, rs, params)
        t0 = params.get("t0", 0.0)
        n_steps = params.get("n_steps", 2000)

        y0 = [t0, params["r0"], params["phi0"], params["dr_dlambda0"]]
        lambda_eval = np.linspace(0, params["lambda_max"], n_steps)

        def rhs(lam, y):
            _, r, _, pr = y
            dt_dlambda = 1 / (1 - rs / r)
            dphi_dlambda = b / r**2
            dpr_dlambda = b**2 / r**3 - 1.5 * rs * b**2 / r**4
            return [dt_dlambda, pr, dphi_dlambda, dpr_dlambda]

        def hit_horizon(lam, y):
            return y[1] - rs * 1.001
        hit_horizon.terminal = True
        hit_horizon.direction = -1

        sol = solve_ivp(
            rhs,
            [0, params["lambda_max"]],
            y0,
            t_eval=lambda_eval,
            events=hit_horizon,
            rtol=1e-9,
            atol=1e-9,
        )

        return {
            "lam": sol.t,
            "t": sol.y[0],
            "r": sol.y[1],
            "phi": sol.y[2],
            "dr_dlambda": sol.y[3],
            "hit_horizon": sol.status == 1,
            "params": {"M": M, "b": b, "r0": params["r0"], "phi0": params["phi0"]},
        }

    def visualize(self, result: Dict[str, np.ndarray]) -> None:
        p = result["params"]
        rs = 2 * p["M"]
        filename = f"schwarzschild_photon_M{p['M']:g}_b{p['b']:g}_r0{p['r0']:g}_phi0{p['phi0']:g}.png"
        plot_orbit_panels(
            time_values=result["lam"],
            r=result["r"],
            phi=result["phi"],
            rs=rs,
            time_label="lambda (affine parameter)",
            title="Schwarzschild photon geodesic",
            filename=filename,
        )
