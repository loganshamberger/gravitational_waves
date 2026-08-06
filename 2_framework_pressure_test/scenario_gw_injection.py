"""Scenario 4 -- the motivating problem: inject a chirp into a detector, recover it.

    ChirpSource -> GeometricToSI -> ProjectOntoDetector -> Resample
                                                             |
                                            AddColoredNoise -+-> MatchedFilter -> Report

This is the pipeline the whole framework was designed for (§5). It is built
here at a DELIBERATELY LOW FIDELITY -- see the warning below -- because its job
is to answer "does the architecture carry an end-to-end multi-stage,
multi-unit, cross-domain problem?", not "is this publication-grade".

===========================================================================
FIDELITY WARNING -- this is a TOY. Known departures from real GW analysis:

  * Leading-order Newtonian/quadrupole chirp only. No post-Newtonian terms,
    no spin, no merger, no ringdown. Integration stops at the Newtonian ISCO,
    which is conservative and truncates the loudest part of a real signal.
  * A stellar-mass binary (30+30 Msun) is used, NOT an EMRI. Real EMRI
    waveforms last months and sit in the LISA band, not LIGO's -- a
    LIGO injection demo needs a LIGO-band source. Swapping in the project's
    actual SchwarzschildAdiabaticInspiral is the follow-up.
  * The noise PSD is a crude analytic fit, not a measured aLIGO curve.
  * One detector, no network, no sky localisation, no calibration.
  * The matched filter uses the exact injected template, so recovered SNR is
    an optimistic upper bound. No template bank, no chi-squared veto, no
    search over sky position or coalescence phase.

None of the physics below should be quoted. The ARCHITECTURE is the claim.
===========================================================================
"""

from dataclasses import dataclass

import numpy as np

from core import Context, DataProduct, Graph, Process
from kinds import Signal
from scenario_oscillator import SIGNAL

# Physical constants (SI)
G = 6.67430e-11
C = 2.99792458e8
MSUN = 1.98892e30
MPC = 3.0856775814913673e22


# ---------------------------------------------------------------------------
# L2 products
# ---------------------------------------------------------------------------


@dataclass
class Waveform(DataProduct):
    """Two polarisations on a shared time base, plus the unit system they are in.

    `units` rides ON THE WIRE rather than in the Context -- the core never
    reconciles units, a node does [§2].
    """

    t: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray
    units: str = "geometric"

    def __post_init__(self) -> None:
        if self.units not in ("geometric", "SI"):
            raise ValueError(f"unknown unit system {self.units!r}")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class ChirpSource(Process):
    """Leading-order inspiral chirp in GEOMETRIC units (G = c = M_total = 1).

    Uses the standard Newtonian time-to-coalescence relation
        omega(tau) = (5 / (256 eta tau))**(3/8)
    and integrates the orbital phase. Amplitude is in units of (M / D).
    """

    inputs = {}
    outputs = {"waveform": Waveform}

    def __init__(self, eta=0.25, f_start_geom=0.0186, inclination=0.0, n=200_000):
        self.eta = eta
        self.f_start_geom = f_start_geom      # orbital angular frequency, geometric
        self.inclination = inclination
        self.n = n

    def validate(self, ctx: Context, **inputs) -> None:
        if not 0 < self.eta <= 0.25:
            raise ValueError(f"symmetric mass ratio must be in (0, 0.25], got {self.eta}")

    def _tau_of_omega(self, w):
        return 5.0 / (256.0 * self.eta * w ** (8.0 / 3.0))

    def run(self, ctx: Context, **inputs):
        w_isco = 6.0 ** -1.5                       # Newtonian ISCO, M = 1
        tau0 = self._tau_of_omega(self.f_start_geom)
        tau1 = self._tau_of_omega(w_isco)

        t = np.linspace(0.0, tau0 - tau1, self.n)
        tau = tau0 - t
        w = (5.0 / (256.0 * self.eta * tau)) ** (3.0 / 8.0)

        # Orbital phase by cumulative integration; GW phase is twice this.
        phase = 2.0 * np.cumsum(w) * (t[1] - t[0])

        amp = 4.0 * self.eta * w ** (2.0 / 3.0)
        ci = np.cos(self.inclination)
        return {
            "waveform": Waveform(
                t=t,
                h_plus=amp * 0.5 * (1 + ci**2) * np.cos(phase),
                h_cross=amp * ci * np.sin(phase),
                units="geometric",
            )
        }


# ---------------------------------------------------------------------------
# Unit reconciliation -- A NODE, not core machinery
# ---------------------------------------------------------------------------


class GeometricToSI(Process):
    """Waveform(geometric) -> Waveform(SI). Scales time by GM/c^3, strain by GM/(c^2 D).

    This node is the entire answer to "how do two domains with different unit
    systems talk to each other" (§2). The core knows nothing about it, and a
    downstream node that requires SI will refuse geometric input itself.
    """

    inputs = {"waveform": Waveform}
    outputs = {"waveform": Waveform}

    def __init__(self, total_mass_msun: float, distance_mpc: float):
        self.total_mass_msun = total_mass_msun
        self.distance_mpc = distance_mpc

    def validate(self, ctx: Context, waveform: Waveform = None, **inputs) -> None:
        if waveform is not None and waveform.units != "geometric":
            raise ValueError(
                f"GeometricToSI needs geometric input, got {waveform.units!r}"
            )
        if self.distance_mpc <= 0:
            raise ValueError("distance must be positive")

    def run(self, ctx: Context, waveform: Waveform):
        m = self.total_mass_msun * MSUN
        t_scale = G * m / C**3                       # seconds per geometric time unit
        h_scale = G * m / (C**2 * self.distance_mpc * MPC)
        return {
            "waveform": Waveform(
                t=waveform.t * t_scale,
                h_plus=waveform.h_plus * h_scale,
                h_cross=waveform.h_cross * h_scale,
                units="SI",
            )
        }


# ---------------------------------------------------------------------------
# Detector chain
# ---------------------------------------------------------------------------


class ProjectOntoDetector(Process):
    """Waveform -> Signal. h(t) = F_plus h_plus + F_cross h_cross.

    Standard L-shaped interferometer antenna pattern for source direction
    (theta, phi) and polarisation angle psi.
    """

    inputs = {"waveform": Waveform}
    outputs = {"strain": SIGNAL}

    def __init__(self, theta=0.0, phi=0.0, psi=0.0):
        self.theta, self.phi, self.psi = theta, phi, psi

    def validate(self, ctx: Context, waveform: Waveform = None, **inputs) -> None:
        if waveform is not None and waveform.units != "SI":
            raise ValueError(
                f"detector needs SI strain, got {waveform.units!r} -- "
                "insert a GeometricToSI node"
            )

    def antenna_pattern(self):
        ct, s2p, c2p = np.cos(self.theta), np.sin(2 * self.phi), np.cos(2 * self.phi)
        s2s, c2s = np.sin(2 * self.psi), np.cos(2 * self.psi)
        f_plus = 0.5 * (1 + ct**2) * c2p * c2s - ct * s2p * s2s
        f_cross = 0.5 * (1 + ct**2) * c2p * s2s + ct * s2p * c2s
        return float(f_plus), float(f_cross)

    def run(self, ctx: Context, waveform: Waveform):
        fp, fc = self.antenna_pattern()
        return {
            "strain": Signal(
                t=waveform.t,
                values=fp * waveform.h_plus + fc * waveform.h_cross,
                name="projected strain",
            )
        }


def aligo_psd(f: np.ndarray) -> np.ndarray:
    """Crude analytic stand-in for the aLIGO design sensitivity curve.

    Right order of magnitude and roughly the right shape (seismic wall at low
    f, bucket near 150 Hz, shot noise rising above). Not a real curve.
    """
    f = np.maximum(np.abs(f), 1.0)
    x = f / 150.0
    return 1e-46 * ((20.0 / f) ** 8 + 1.0 / x + 1.0 + x**2)


@dataclass
class InjectionTruth(DataProduct):
    """Ground truth about where the signal was placed. Carried on a wire so a
    downstream sink can score the recovery without reaching into a node."""

    t_inject: float
    segment_duration: float


class InjectIntoSegment(Process):
    """Place a short signal at a RANDOM time inside a longer, empty segment.

    Without this the injection sits at t = 0 and the matched filter merely
    confirms a known answer at zero lag. With it, the filter has to search --
    and the recovered lag is a real measurement that can be scored against
    `truth`.

    Uses ctx.rng, so the injection time is reproducible from the seed.
    """

    inputs = {"signal": SIGNAL}
    outputs = {"data": SIGNAL, "truth": InjectionTruth}

    def __init__(self, segment_s: float = 8.0, margin_s: float = 0.5):
        self.segment_s = segment_s
        self.margin_s = margin_s

    def validate(self, ctx: Context, signal: Signal = None, **inputs) -> None:
        if signal is None:
            return
        need = signal.t[-1] - signal.t[0] + 2 * self.margin_s
        if self.segment_s <= need:
            raise ValueError(
                f"segment of {self.segment_s}s cannot hold a "
                f"{signal.t[-1] - signal.t[0]:.2f}s signal plus "
                f"{self.margin_s}s margins (need > {need:.2f}s)"
            )

    def run(self, ctx: Context, signal: Signal):
        dt = signal.dt
        n_seg = int(round(self.segment_s / dt))
        n_sig = len(signal.values)
        lo = int(round(self.margin_s / dt))
        hi = n_seg - n_sig - lo
        i0 = int(ctx.rng.integers(lo, hi))

        buf = np.zeros(n_seg)
        buf[i0 : i0 + n_sig] = signal.values
        return {
            "data": Signal(t=np.arange(n_seg) * dt, values=buf, name="segment"),
            "truth": InjectionTruth(
                t_inject=i0 * dt, segment_duration=self.segment_s
            ),
        }


class AddColoredNoise(Process):
    """Signal -> Signal, with stationary Gaussian noise coloured by a PSD.

    Uses ctx.rng, so a seeded Context makes the whole injection reproducible.
    """

    inputs = {"strain": SIGNAL}
    outputs = {"data": SIGNAL, "psd": SIGNAL}

    def __init__(self, psd=aligo_psd, scale: float = 1.0):
        self.psd = psd
        self.scale = scale

    def run(self, ctx: Context, strain: Signal):
        n = len(strain.values)
        dt = strain.dt
        freqs = np.fft.rfftfreq(n, dt)
        psd = self.psd(freqs)

        # Draw white Gaussian noise, colour it in the frequency domain.
        white = ctx.rng.normal(size=n)
        coloured = np.fft.irfft(
            np.fft.rfft(white) * np.sqrt(psd / (4 * dt)), n=n
        )
        return {
            "data": Signal(
                t=strain.t,
                values=strain.values + self.scale * coloured,
                name="detector data",
            ),
            # Report the PSD of the noise ACTUALLY added, i.e. including the
            # scale factor -- otherwise a downstream filter whitens against the
            # wrong noise level and its SNR becomes independent of how noisy
            # the data really is.
            "psd": Signal(t=freqs, values=self.scale**2 * psd, name="psd"),
        }


class MatchedFilter(Process):
    """Data + template + PSD -> SNR time series.

    Textbook frequency-domain matched filter. The template here is the exact
    injected signal, so this is the optimistic no-mismatch case.
    """

    inputs = {"data": SIGNAL, "template": SIGNAL, "psd": SIGNAL}
    outputs = {"snr": SIGNAL}

    def run(self, ctx: Context, data: Signal, template: Signal, psd: Signal):
        n = len(data.values)
        dt = data.dt
        df = 1.0 / (n * dt)

        d = np.fft.rfft(data.values) * dt
        h = np.fft.rfft(template.values, n=n) * dt
        s = np.interp(np.fft.rfftfreq(n, dt), psd.t, psd.values)

        # Optimal filter and its normalisation.
        # z(t) = 4 Re int d(f) h*(f)/S(f) e^{2 pi i f t} df; irfft carries a 1/n.
        sigma_sq = 4 * df * np.sum(np.abs(h) ** 2 / s)
        z = 4 * df * n * np.fft.irfft(d * np.conj(h) / s, n=n)

        return {
            "snr": Signal(
                t=data.t - data.t[0],
                values=np.abs(z) / np.sqrt(sigma_sq) if sigma_sq > 0 else z * 0,
                name="snr",
            )
        }


class InjectionReport(Process):
    """Sink: scores the recovery against ground truth.

    Two inputs from two different upstream nodes -- the measurement and the
    truth it should reproduce. Neither node knows about the other.
    """

    inputs = {"snr": SIGNAL, "truth": InjectionTruth}
    outputs = {}

    def __init__(self, filename="injection.txt"):
        self.filename = filename

    def run(self, ctx: Context, snr: Signal, truth: InjectionTruth):
        i = int(np.argmax(snr.values))
        recovered, peak = float(snr.t[i]), float(snr.values[i])
        (ctx.workdir / self.filename).write_text(
            f"injected at t = {truth.t_inject:.4f} s\n"
            f"recovered at t = {recovered:.4f} s "
            f"(error {1e3 * (recovered - truth.t_inject):+.2f} ms)\n"
            f"peak SNR {peak:.2f} over a {truth.segment_duration:g}s segment\n"
        )
        return {}


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def build_injection_graph(
    total_mass_msun=30.0,
    distance_mpc=400.0,
    eta=0.25,
    inclination=0.0,
    f_start_hz=20.0,
    theta=0.4, phi=0.9, psi=0.2,
    sample_rate=4096.0,
    noise_scale=1.0,
    segment_s=8.0,
):
    """The §5 pipeline, wired.

    Note the template branch reuses the SAME nodes as the injection branch --
    which is the composability claim doing real work: a template is just the
    clean strain, tapped off before noise is added.

    **Why `f_start_hz` is converted here and not inside ChirpSource.** The
    source works in geometric units and does NOT know the mass -- that lives
    downstream in `GeometricToSI`, because reconciliation is a node. So the
    source cannot be told "start at 20 Hz"; only this builder, which knows both
    the geometric side and the mass, can do that conversion.

    That has a consequence worth stating: the SWEEP RATIO is set entirely by
    the geometric start frequency against the geometric ISCO, and both are
    mass-independent. Changing the mass slides the whole band up or down in Hz
    without widening it. To get a longer chirp you must start lower, not
    lighter.
    """
    from l1_nodes import Resample, Window

    # GW angular frequency 2*pi*f = 2*omega_orbital, so omega = pi*f; convert
    # to geometric using the time unit GM/c^3.
    t_scale = G * total_mass_msun * MSUN / C**3
    f_start_geom = np.pi * f_start_hz * t_scale

    g = Graph()
    g.add("chirp", ChirpSource(eta=eta, inclination=inclination,
                               f_start_geom=f_start_geom))
    g.add("to_si", GeometricToSI(total_mass_msun, distance_mpc))
    g.add("detector", ProjectOntoDetector(theta=theta, phi=phi, psi=psi))
    g.add("resample", Resample(sample_rate))
    g.add("taper", Window(alpha=0.2))
    g.add("inject", InjectIntoSegment(segment_s=segment_s))
    g.add("noise", AddColoredNoise(scale=noise_scale))
    g.add("filter", MatchedFilter())
    g.add("report", InjectionReport())

    g.connect("chirp.waveform", "to_si.waveform")
    g.connect("to_si.waveform", "detector.waveform")
    g.connect("detector.strain", "resample.signal")
    g.connect("resample.signal", "taper.signal")
    g.connect("taper.signal", "inject.signal")     # place it at a random time
    g.connect("inject.data", "noise.strain")
    g.connect("noise.data", "filter.data")
    g.connect("taper.signal", "filter.template")   # template starts at zero lag
    g.connect("noise.psd", "filter.psd")
    g.connect("filter.snr", "report.snr")
    g.connect("inject.truth", "report.truth")      # score against ground truth
    return g
