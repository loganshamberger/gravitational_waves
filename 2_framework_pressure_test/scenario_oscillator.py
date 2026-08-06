"""Scenario 1 -- simple and driven harmonic oscillator.

Purpose: pressure-test the framework on a continuous-time physics problem
where an external forcing term is WIRED IN as a graph edge rather than baked
into the integrator.

    Free:   Oscillator -> EnergyTrace -> Report
    Driven: DrivingForce -> DrivenOscillator -> EnergyTrace -> Report

The interesting part is not the physics (it is a spring). It is what happens
at the DrivingForce -> DrivenOscillator edge, which is where this scenario
breaks the framework. See FINDINGS.md, finding 1.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from core import Context, Free, Parametric, Process
from kinds import Real, Signal

# A port carrying a sampled signal whose representation the scheduler grounds.
SIGNAL = Parametric(Signal, Free("F"))


@dataclass
class Trajectory:
    """Oscillator state history. Local to this scenario, not a shared kind."""

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray


class TrajectoryProduct(Trajectory):
    """Marker subclass so Trajectory can be used as a port type."""


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class DrivingForce(Process):
    """Source: a sinusoidal driving force sampled on a fixed grid.

    NOTE the awkwardness this creates downstream: the consumer is an adaptive
    ODE solver that needs F at solver-chosen times, but a data edge can only
    hand it F on THIS grid. See FINDINGS.md finding 1 -- this is the oracle
    question arriving unprompted.
    """

    inputs = {}
    outputs = {"force": SIGNAL}

    def __init__(self, amplitude: float, omega_d: float, t_max: float, n: int = 2001):
        self.amplitude = amplitude
        self.omega_d = omega_d
        self.t_max = t_max
        self.n = n

    def validate(self, ctx: Context, **inputs) -> None:
        if self.n < 2:
            raise ValueError("need at least two samples to define a grid")

    def run(self, ctx: Context, **inputs):
        t = np.linspace(0.0, self.t_max, self.n)
        f = self.amplitude * np.cos(self.omega_d * t)
        return {"force": Signal(t=t, values=f, name="driving force")}


class Oscillator(Process):
    """Source: undriven damped harmonic oscillator. m x'' + c x' + k x = 0."""

    inputs = {}
    outputs = {"trajectory": TrajectoryProduct}

    def __init__(
        self,
        m: float,
        k: float,
        c: float = 0.0,
        x0: float = 1.0,
        v0: float = 0.0,
        t_max: float = 50.0,
        n: int = 2001,
    ):
        self.m, self.k, self.c = m, k, c
        self.x0, self.v0 = x0, v0
        self.t_max, self.n = t_max, n

    def validate(self, ctx: Context, **inputs) -> None:
        if self.m <= 0 or self.k <= 0:
            raise ValueError(f"need m>0 and k>0, got m={self.m}, k={self.k}")
        if self.c < 0:
            raise ValueError(f"damping must be non-negative, got c={self.c}")

    def _rhs(self, t, y):
        x, v = y
        return [v, (-self.k * x - self.c * v) / self.m]

    def run(self, ctx: Context, **inputs):
        t_eval = np.linspace(0.0, self.t_max, self.n)
        sol = solve_ivp(
            self._rhs, (0.0, self.t_max), [self.x0, self.v0],
            t_eval=t_eval, rtol=1e-10, atol=1e-12,
        )
        return {
            "trajectory": TrajectoryProduct(t=sol.t, x=sol.y[0], v=sol.y[1])
        }


class DrivenOscillator(Process):
    """Transform: same oscillator, with the forcing term arriving on a port.

    m x'' + c x' + k x = F(t), where F comes from upstream.
    """

    inputs = {"force": SIGNAL}
    outputs = {"trajectory": TrajectoryProduct}

    def __init__(
        self,
        m: float,
        k: float,
        c: float = 0.0,
        x0: float = 0.0,
        v0: float = 0.0,
        n: int = 2001,
    ):
        self.m, self.k, self.c = m, k, c
        self.x0, self.v0 = x0, v0
        self.n = n

    def validate(self, ctx: Context, **inputs) -> None:
        if self.m <= 0 or self.k <= 0:
            raise ValueError(f"need m>0 and k>0, got m={self.m}, k={self.k}")
        force = inputs.get("force")
        if force is not None and len(force.t) < 2:
            raise ValueError("driving force must have at least two samples")

    def run(self, ctx: Context, force: Signal):
        # THE COMPROMISE (FINDINGS.md finding 1): the solver picks its own
        # internal times, but the edge only delivered F on a fixed grid, so we
        # interpolate. Accuracy is now capped by the PRODUCER's grid, a choice
        # the consumer cannot see or influence.
        def f_of_t(t):
            return np.interp(t, force.t, force.values)

        def rhs(t, y):
            x, v = y
            return [v, (f_of_t(t) - self.k * x - self.c * v) / self.m]

        t_max = float(force.t[-1])
        t_eval = np.linspace(0.0, t_max, self.n)
        sol = solve_ivp(
            rhs, (0.0, t_max), [self.x0, self.v0],
            t_eval=t_eval, rtol=1e-10, atol=1e-12,
        )
        return {
            "trajectory": TrajectoryProduct(t=sol.t, x=sol.y[0], v=sol.y[1])
        }


# ---------------------------------------------------------------------------
# Transforms and sinks
# ---------------------------------------------------------------------------


class EnergyTrace(Process):
    """Transform: total mechanical energy over time. Trajectory -> Signal."""

    inputs = {"trajectory": TrajectoryProduct}
    outputs = {"energy": SIGNAL}

    def __init__(self, m: float, k: float):
        self.m, self.k = m, k

    def run(self, ctx: Context, trajectory: TrajectoryProduct):
        e = 0.5 * self.m * trajectory.v**2 + 0.5 * self.k * trajectory.x**2
        return {"energy": Signal(t=trajectory.t, values=e, name="energy")}


class Report(Process):
    """Sink: writes a summary line. Demonstrates a sink over ANY Signal.

    The same sink serves the driving force, the energy trace, and (in the SDLC
    scenario) a throughput series -- which is the point of a shared kind.
    """

    inputs = {"signal": SIGNAL}
    outputs = {}

    def __init__(self, filename: str):
        self.filename = filename

    def run(self, ctx: Context, signal: Signal):
        path = ctx.workdir / self.filename
        peak = float(np.max(np.abs(signal.values)))
        mean = float(np.mean(signal.values))
        path.write_text(
            f"{signal.name}: n={len(signal.t)} "
            f"t=[{signal.t[0]:.3f}, {signal.t[-1]:.3f}] "
            f"peak={peak:.6g} mean={mean:.6g}\n"
        )
        ctx.logger.info("wrote %s", path)
        return {}


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_free_graph(m=1.0, k=4.0, c=0.0, t_max=50.0):
    """Undriven oscillator: Oscillator -> EnergyTrace -> Report."""
    from core import Graph

    g = Graph()
    g.add("osc", Oscillator(m=m, k=k, c=c, x0=1.0, v0=0.0, t_max=t_max))
    g.add("energy", EnergyTrace(m=m, k=k))
    g.add("report", Report("free_energy.txt"))
    g.connect("osc.trajectory", "energy.trajectory")
    g.connect("energy.energy", "report.signal")
    return g


def build_driven_graph(m=1.0, k=4.0, c=0.2, amplitude=1.0, omega_d=2.0, t_max=50.0):
    """Driven oscillator, with the force wired in as an edge.

    omega_d = sqrt(k/m) puts it on resonance.
    """
    from core import Graph

    g = Graph()
    g.add("force", DrivingForce(amplitude=amplitude, omega_d=omega_d, t_max=t_max))
    g.add("osc", DrivenOscillator(m=m, k=k, c=c))
    g.add("energy", EnergyTrace(m=m, k=k))
    g.add("report", Report("driven_energy.txt"))
    g.connect("force.force", "osc.force")
    g.connect("osc.trajectory", "energy.trajectory")
    g.connect("energy.energy", "report.signal")
    return g


# The batch driver grounds Signal's free parameter to real samples.
BATCH_BINDINGS = {"F": Real}
