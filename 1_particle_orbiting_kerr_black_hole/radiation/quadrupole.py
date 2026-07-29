"""The (reduced, trace-free) mass quadrupole moment for a point particle on
an equatorial orbit, and its second time derivative -- computed fully
analytically from the geodesic's coordinate-time state, per
notes/gravitational_wave_quadrupole_report.md sections 2 and 4b.

Spacetime-agnostic: takes r, phi and their coordinate-time derivatives plus
a particle mass and radial mapping; nothing here is Schwarzschild-specific,
so it is reusable for Kerr's equatorial plane unchanged.

Units: G = c = 1, consistent with the rest of the project.
"""

import numpy as np

from mapping import radial_mapping, radial_mapping_derivatives


def _equatorial_kinematics(r, phi, rdot, phidot, rddot, phiddot, M, mapping):
    """Flat-space pseudo-Cartesian position, velocity, acceleration (z=0)."""
    R = radial_mapping(r, M, mapping)
    dR_dr, d2R_dr2 = radial_mapping_derivatives(r, M, mapping)
    Rdot = dR_dr * rdot
    Rddot = d2R_dr2 * rdot**2 + dR_dr * rddot

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    x = R * cos_phi
    y = R * sin_phi

    vx = Rdot * cos_phi - R * phidot * sin_phi
    vy = Rdot * sin_phi + R * phidot * cos_phi

    radial_term = Rddot - R * phidot**2
    tangential_term = 2 * Rdot * phidot + R * phiddot
    ax = radial_term * cos_phi - tangential_term * sin_phi
    ay = radial_term * sin_phi + tangential_term * cos_phi

    return x, y, vx, vy, ax, ay


def mass_quadrupole(r, phi, rdot, phidot, rddot, phiddot, mu, M, mapping="boyer_lindquist"):
    """Trace-free quadrupole moment Q_ij(t) and its second time derivative.

    Inputs are 1-D arrays over the trajectory (r, phi and their coordinate-
    time first/second derivatives, as returned by
    geodesics.schwarzschild.SchwarzschildGeodesicCoordTime), plus the
    particle's mass mu and the radial mapping used to define flat space.

    Returns (Q, Qddot), each shape (N, 3, 3), symmetric. The orbit is
    confined to z=0, but the trace subtraction still gives Q_zz a nonzero
    value -- required for observers away from the orbital plane (see the TT
    projection module).
    """
    x, y, vx, vy, ax, ay = _equatorial_kinematics(r, phi, rdot, phidot, rddot, phiddot, M, mapping)

    N = np.shape(x)[0] if np.ndim(x) else 1
    I = np.zeros((N, 3, 3))
    Iddot = np.zeros((N, 3, 3))

    I[:, 0, 0] = mu * x * x
    I[:, 1, 1] = mu * y * y
    I[:, 0, 1] = I[:, 1, 0] = mu * x * y

    Iddot[:, 0, 0] = 2 * mu * vx * vx + 2 * mu * ax * x
    Iddot[:, 1, 1] = 2 * mu * vy * vy + 2 * mu * ay * y
    Iddot[:, 0, 1] = Iddot[:, 1, 0] = mu * (vx * vy + vy * vx) + mu * (ax * y + x * ay)

    trace = I[:, 0, 0] + I[:, 1, 1]  # I_zz = 0 exactly, z is always 0
    trace_ddot = Iddot[:, 0, 0] + Iddot[:, 1, 1]

    Q = I.copy()
    Q[:, 0, 0] -= trace / 3
    Q[:, 1, 1] -= trace / 3
    Q[:, 2, 2] = -trace / 3

    Qddot = Iddot.copy()
    Qddot[:, 0, 0] -= trace_ddot / 3
    Qddot[:, 1, 1] -= trace_ddot / 3
    Qddot[:, 2, 2] = -trace_ddot / 3

    return Q, Qddot
