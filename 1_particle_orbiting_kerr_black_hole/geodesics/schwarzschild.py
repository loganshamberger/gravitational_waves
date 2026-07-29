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
            dt_dtau = E / (1 - rs / r)
            dphi_dtau = h / r**2
            dpr_dtau = -M / r**2 + h**2 / r**3 - 3 * M * h**2 / r**4
            return [dt_dtau, pr, dphi_dtau, dpr_dtau]

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
