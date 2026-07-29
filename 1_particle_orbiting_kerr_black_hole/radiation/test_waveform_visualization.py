import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geodesics"))

from inspiral import SchwarzschildAdiabaticInspiral
from quadrupole import mass_quadrupole
from schwarzschild import SchwarzschildGeodesicCoordTime, circular_orbit_E_h
from waveform import strain_plus_cross
from waveform_visualization import (
    dominant_frequency,
    instantaneous_frequency,
    plot_chirp_spectrogram,
    plot_power_spectrum,
    plot_strain_timeseries,
)


def _exact_circular_orbit_params(M, a):
    E, h = circular_orbit_E_h(M, a)
    return E, h


def _circular_orbit_h_plus(M=1.0, a=20.0, mu=1e-3, n_periods=8, n_steps_per_period=200):
    E, h = _exact_circular_orbit_params(M, a)
    Omega = np.sqrt(M / a**3)
    period = 2 * np.pi / Omega
    t_max = n_periods * period
    n_steps = n_periods * n_steps_per_period
    traj = SchwarzschildGeodesicCoordTime().simulate(
        dict(M=M, E=E, h=h, r0=a, phi0=0.0, dr_dtau0=0.0, t_max=t_max, n_steps=n_steps)
    )
    _, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    h_plus, h_cross = strain_plus_cross(Qddot, theta_obs=np.pi / 3, phi_obs=0.0, D=1000.0)
    return traj["t"], h_plus, h_cross, Omega


def test_dominant_frequency_matches_frequency_doubling():
    t, h_plus, h_cross, Omega = _circular_orbit_h_plus()
    expected_freq = 2 * Omega / (2 * np.pi)
    freq = dominant_frequency(t, h_plus)
    assert freq == pytest.approx(expected_freq, rel=0.05)


def test_instantaneous_frequency_matches_orbital_frequency_for_circular_orbit():
    t, h_plus, h_cross, Omega = _circular_orbit_h_plus()
    _, freq = instantaneous_frequency(t, h_plus)
    expected_freq = 2 * Omega / (2 * np.pi)
    # Ignore edge effects from the Hilbert-transform boundary and the
    # np.gradient endpoints.
    interior = freq[10:-10]
    assert np.allclose(interior, expected_freq, rtol=0.05)


def test_instantaneous_frequency_increases_for_chirp():
    # a0 close to the ISCO so the frequency shift is large and fast (~50%
    # rise in orbital frequency by merger, per the diagnostic run used to
    # pick these parameters); n_steps chosen for ~140 samples per initial
    # orbital period, comfortably above Nyquist.
    M, mu, a0 = 1.0, 1e-3, 8.0
    traj = SchwarzschildAdiabaticInspiral().simulate(
        dict(M=M, mu=mu, a0=a0, phi0=0.0, t_max=2e5, n_steps=200000)
    )
    _, Qddot = mass_quadrupole(
        traj["r"], traj["phi"], traj["Rdot"], traj["phidot"], traj["Rddot"], traj["phiddot"],
        mu, M,
    )
    h_plus, _ = strain_plus_cross(Qddot, theta_obs=np.pi / 3, phi_obs=0.0, D=1000.0)
    t, freq = instantaneous_frequency(traj["t"], h_plus)

    # Smooth out short-timescale wiggle (the phase derivative is noisy on
    # scales below one GW cycle) by comparing block-averaged frequency in
    # the first vs. last quarter of the run -- the chirp should be
    # unambiguous over that span even if not monotonic sample-to-sample.
    interior = freq[20:-20]
    n = len(interior)
    first_quarter = np.mean(interior[: n // 4])
    last_quarter = np.mean(interior[-n // 4 :])
    assert last_quarter > first_quarter


def test_plotting_functions_produce_files(tmp_path, monkeypatch):
    import waveform_visualization as wv

    monkeypatch.setattr(wv, "output_path", lambda filename: str(tmp_path / filename))

    t, h_plus, h_cross, Omega = _circular_orbit_h_plus(n_periods=3, n_steps_per_period=100)
    plot_strain_timeseries(t, h_plus, h_cross, "strain.png")
    plot_power_spectrum(t, h_plus, "spectrum.png")
    plot_chirp_spectrogram(t, h_plus, "spectrogram.png")

    assert (tmp_path / "strain.png").exists()
    assert (tmp_path / "spectrum.png").exists()
    assert (tmp_path / "spectrogram.png").exists()
