"""TT-gauge strain scalars h_+, h_x for an arbitrary observer direction, from
the quadrupole module's Qddot. See
notes/gravitational_wave_quadrupole_report.md section 3.

Only the h_+/h_x scalars are built (not the full h_ij^TT tensor / explicit
Lambda projector) -- e^+ and e^x are already transverse-traceless, so
contracting them with Qddot directly gives the same result as going through
Lambda, at a fraction of the algebra (section 3b).

Spacetime-agnostic: depends only on Qddot, an observer direction, and a
distance, so it is reusable for Kerr unchanged.

Units: G = c = 1, consistent with the rest of the project.
"""

import numpy as np


def polarization_basis(theta_obs, phi_obs):
    """Right-handed orthonormal (n_hat, e_theta, e_phi) for observer direction
    (theta_obs, phi_obs), via closed forms -- not cross products with z_hat,
    since those are singular at the poles while these are not (section 3c).
    """
    sin_t, cos_t = np.sin(theta_obs), np.cos(theta_obs)
    sin_p, cos_p = np.sin(phi_obs), np.cos(phi_obs)

    n_hat = np.array([sin_t * cos_p, sin_t * sin_p, cos_t])
    e_theta = np.array([cos_t * cos_p, cos_t * sin_p, -sin_t])
    e_phi = np.array([-sin_p, cos_p, 0.0])
    return n_hat, e_theta, e_phi


def strain_plus_cross(Qddot, theta_obs, phi_obs, D):
    """h_+(t), h_x(t) from Qddot(t) (shape (N,3,3)) for an observer at
    distance D in direction (theta_obs, phi_obs).

    OPEN QUESTION (unresolved, see whatidid space particle-orbiting-kerr):
    theta_obs/phi_obs intentionally have no default -- there's no observer
    direction that's more "correct" than another, so forcing every call site
    to state it seemed safer than silently picking one. That reasoning has
    been pushed back on and is not settled; revisit before treating this
    signature as final.

    h_+ = (1/D)(Qddot_ThetaTheta - Qddot_PhiPhi)
    h_x = (2/D) Qddot_ThetaPhi
    """
    _, e_theta, e_phi = polarization_basis(theta_obs, phi_obs)

    Q_tt = np.einsum("i,tij,j->t", e_theta, Qddot, e_theta)
    Q_pp = np.einsum("i,tij,j->t", e_phi, Qddot, e_phi)
    Q_tp = np.einsum("i,tij,j->t", e_theta, Qddot, e_phi)

    h_plus = (Q_tt - Q_pp) / D
    h_cross = 2 * Q_tp / D
    return h_plus, h_cross
