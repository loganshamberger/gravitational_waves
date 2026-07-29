"""Waveform plots and frequency-domain checks for h_+/h_x, per
notes/gravitational_wave_quadrupole_report.md section 6.

These operate on already-computed (t, h_plus, h_cross) arrays (from
waveform.strain_plus_cross), not on a System's trajectory dict directly --
consistent with the rest of radiation/ being pure post-processing rather
than System subclasses. theta_obs/phi_obs stay a required argument
everywhere upstream (see waveform.py's open question); nothing here picks a
default on the caller's behalf.

Units: G = c = 1, consistent with the rest of the project.
"""

import os
import sys

import numpy as np
from scipy.signal import hilbert, spectrogram

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from visualization import output_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def dominant_frequency(t, h):
    """Peak frequency (cycles per unit t) of h(t) via an FFT on a uniform
    grid. For a circular orbit this should match 2 * Omega / (2 pi) -- the
    report's frequency-doubling check.
    """
    dt = t[1] - t[0]
    spectrum = np.abs(np.fft.rfft(h - np.mean(h)))
    freqs = np.fft.rfftfreq(len(h), d=dt)
    return freqs[np.argmax(spectrum)]


def instantaneous_frequency(t, h):
    """Instantaneous frequency of h(t) via the analytic signal (Hilbert
    transform): freq(t) = (1 / 2 pi) d(phase)/dt. Used to verify the chirp
    signature (increasing frequency) for an inspiraling source, where a
    single FFT peak is no longer meaningful since the signal isn't
    stationary.

    Returns (t_mid, freq), one shorter than t/h since it's built from a
    finite difference of the unwrapped phase.
    """
    phase = np.unwrap(np.angle(hilbert(h)))
    dt = t[1] - t[0]
    freq = np.gradient(phase, dt) / (2 * np.pi)
    return t, freq


def plot_strain_timeseries(t, h_plus, h_cross, filename, title="Gravitational-wave strain"):
    """h_+(t), h_x(t) on one time axis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, h_plus, label="h_+")
    ax.plot(t, h_cross, label="h_x")
    ax.set_xlabel("t (coordinate time)")
    ax.set_ylabel("strain")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path(filename))
    plt.close(fig)


def plot_power_spectrum(t, h, filename, title="Strain power spectrum"):
    """|FFT(h)| vs. frequency -- for a circular orbit this should show a
    single sharp peak at 2 * orbital frequency (the frequency-doubling
    check).
    """
    dt = t[1] - t[0]
    spectrum = np.abs(np.fft.rfft(h - np.mean(h)))
    freqs = np.fft.rfftfreq(len(h), d=dt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, spectrum)
    ax.set_xlabel("frequency (1 / t)")
    ax.set_ylabel("|FFT(h)|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path(filename))
    plt.close(fig)


def plot_chirp_spectrogram(t, h, filename, title="Chirp spectrogram"):
    """Time-frequency spectrogram of h(t) -- shows the rising instantaneous
    frequency of an inspiral directly, unlike a single stationary FFT.
    """
    dt = t[1] - t[0]
    fs = 1.0 / dt
    f, times, Sxx = spectrogram(h, fs=fs, nperseg=min(256, len(h) // 4))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(times, f, Sxx, shading="gouraud")
    ax.set_xlabel("t (coordinate time)")
    ax.set_ylabel("frequency (1 / t)")
    ax.set_title(title)
    # Zoom to where the signal's power actually is -- the chirp is a thin
    # rising line near the bottom of the full Nyquist range otherwise.
    power_per_freq = Sxx.sum(axis=1)
    if power_per_freq.any():
        peak_freq = f[np.argmax(power_per_freq)]
        ax.set_ylim(0, 4 * peak_freq)
    fig.tight_layout()
    fig.savefig(output_path(filename))
    plt.close(fig)
