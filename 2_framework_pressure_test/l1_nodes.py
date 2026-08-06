"""Layer 1 -- generic nodes, written once against shared kinds.

Per §4: these are domain-neutral transforms over `Signal`. `Resample` exists
here rather than in a domain package because the design's L1 rule says define
it when it has two consumers -- the GW injection chain needs it for both the
waveform and the detector stream.

Nothing here knows any physics.
"""

import numpy as np

from core import Context, Process
from kinds import Signal
from scenario_oscillator import SIGNAL


class Resample(Process):
    """Signal -> Signal on a uniform grid at a fixed sample rate.

    Linear interpolation. Crude on purpose -- the point is that resampling is
    an ordinary NODE, not machinery hidden in the core [§2].
    """

    inputs = {"signal": SIGNAL}
    outputs = {"signal": SIGNAL}

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def validate(self, ctx: Context, **inputs) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    def run(self, ctx: Context, signal: Signal):
        t0, t1 = float(signal.t[0]), float(signal.t[-1])
        n = int((t1 - t0) * self.sample_rate)
        t = t0 + np.arange(n) / self.sample_rate
        return {
            "signal": Signal(
                t=t,
                values=np.interp(t, signal.t, signal.values),
                name=f"{signal.name} @{self.sample_rate:g}Hz",
            )
        }


class Window(Process):
    """Taper both ends of a Signal (Tukey). Suppresses FFT edge artefacts."""

    inputs = {"signal": SIGNAL}
    outputs = {"signal": SIGNAL}

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def run(self, ctx: Context, signal: Signal):
        n = len(signal.values)
        w = np.ones(n)
        edge = int(self.alpha * n / 2)
        if edge > 0:
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
            w[:edge] = ramp
            w[-edge:] = ramp[::-1]
        return {
            "signal": Signal(t=signal.t, values=signal.values * w,
                             name=f"{signal.name} (windowed)")
        }
