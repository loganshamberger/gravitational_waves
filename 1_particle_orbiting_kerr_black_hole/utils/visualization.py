"""Shared plotting helpers for orbit/geodesic Systems.

Every System's visualize() ends up wanting the same 3-panel layout (a
radial-coordinate trace, an angular trace, and the orbit itself in the
equatorial plane with the horizon drawn in). Centralizing it here means a
Kerr system can reuse the exact same panel layout and output location.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","visualizations")


def output_path(filename: str) -> str:
    """Return the path under VISUALIZATION_DIR for `filename`, creating the dir if needed."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    return os.path.join(VISUALIZATION_DIR, filename)


def plot_orbit_panels(
    time_values: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    rs: float,
    time_label: str,
    title: str,
    filename: str,
) -> None:
    """Render the standard 3-panel (r vs time, phi vs time, orbit) figure and save it."""
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    ax_r = fig.add_subplot(gs[0, 0])
    ax_phi = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_orbit = fig.add_subplot(gs[:, 1])

    ax_r.plot(time_values, r)
    ax_r.set_ylabel("r")
    ax_phi.plot(time_values, phi)
    ax_phi.set_ylabel("phi")
    ax_phi.set_xlabel(time_label)

    x, y = r * np.cos(phi), r * np.sin(phi)
    ax_orbit.plot(x, y, linewidth=1)
    ax_orbit.add_patch(plt.Circle((0, 0), rs, color="black"))
    ax_orbit.set_aspect("equal")
    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.set_title("orbit in the equatorial plane")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path(filename))
    plt.close(fig)
