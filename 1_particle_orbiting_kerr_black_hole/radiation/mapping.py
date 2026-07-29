"""Pseudo-Cartesian radial mappings from Schwarzschild-like (r, phi) to flat
space, for feeding the quadrupole formula (which wants Cartesian coordinates
on flat space, but the geodesic integrators produce areal-radius r).

Spacetime-agnostic: only depends on r and M, so it is reusable for Kerr's
equatorial plane unchanged.

Units: G = c = 1, consistent with the rest of the project.
"""

import numpy as np

MAPPINGS = ("boyer_lindquist", "harmonic", "isotropic")


def radial_mapping(r, M, mapping="boyer_lindquist"):
    """Map areal radius r to a flat-space pseudo-Cartesian radius R.

    All three options agree as r -> infinity; they differ at O(M/r), which is
    a free, cheap estimate of the systematic error of the whole quadrupole
    scheme (see notes/gravitational_wave_quadrupole_report.md section 2b).
    """
    r = np.asarray(r, dtype=float)
    if mapping == "boyer_lindquist":
        return r
    if mapping == "harmonic":
        return r - M
    if mapping == "isotropic":
        return 0.5 * (r - M + np.sqrt(r**2 - 2 * M * r))
    raise ValueError(f"Unknown mapping {mapping!r}; expected one of {MAPPINGS}")


def equatorial_cartesian(r, phi, M, mapping="boyer_lindquist"):
    """Flat-space pseudo-Cartesian (x, y) for an equatorial (theta=pi/2) orbit."""
    R = radial_mapping(r, M, mapping)
    return R * np.cos(phi), R * np.sin(phi)
