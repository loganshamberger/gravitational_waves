"""Scenario 3 -- coupling the free oscillator to the driven one THROUGH the graph.

Two masses joined by a coupling spring:

    m1 x1'' = -(k1 + kc) x1 + kc x2
    m2 x2'' = -(k2 + kc) x2 + kc x1

Each mass feels the other's displacement. Note the symmetry: this is genuinely
bidirectional, so the honest dataflow graph has a CYCLE.

Three graphs are built here, and the difference between them is the whole point:

  1. `build_cascade_graph`   -- one-way coupling. osc1 drives osc2, no back
                               reaction. A DAG. Runs today.
  2. `build_coupled_graph`   -- the honest bidirectional topology. Wires fine,
                               and BatchScheduler refuses it (cycle).
  3. `build_monolith_graph`  -- both masses collapsed into one node. Runs, and
                               is exact, and buries the seam the framework
                               exists to expose.

So the physics question "how good is the one-way approximation?" and the
framework question "what does the batch scheduler cost me?" turn out to be the
same question, measured by the same number. See FINDINGS.md F5.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from core import Context, Graph, Process
from kinds import Signal
from scenario_oscillator import (
    SIGNAL,
    DrivenOscillator,
    Oscillator,
    Report,
    TrajectoryProduct,
)


# ---------------------------------------------------------------------------
# The coupling element -- a transform, i.e. an ordinary node
# ---------------------------------------------------------------------------


class CouplingForce(Process):
    """Transform: Trajectory -> Signal. The force one mass exerts on the other.

    F(t) = kc * x(t). This is "reconciliation is a node" applied to a physical
    interaction: the coupling is not machinery inside the scheduler, it is a
    box on the diagram with a typed input and a typed output.
    """

    inputs = {"trajectory": TrajectoryProduct}
    outputs = {"force": SIGNAL}

    def __init__(self, kc: float, name: str = "coupling force"):
        self.kc = kc
        self.name = name

    def validate(self, ctx: Context, **inputs) -> None:
        if self.kc < 0:
            raise ValueError(f"coupling constant must be non-negative, got {self.kc}")

    def run(self, ctx: Context, trajectory: TrajectoryProduct):
        return {
            "force": Signal(
                t=trajectory.t, values=self.kc * trajectory.x, name=self.name
            )
        }


# ---------------------------------------------------------------------------
# The monolith -- both masses inside one node
# ---------------------------------------------------------------------------


class TwoMassSystem(Process):
    """Source: solves the FULL bidirectional 2-DOF system in one node.

    This is exact, and it is the monolith escape hatch [6.3]. It is included as
    the reference solution AND as the thing the design says not to do -- the
    coupling spring, the single most interesting element in the system, is now
    invisible to the graph. You cannot swap it, plot the force it carries, or
    replace one mass with a different model.
    """

    inputs = {}
    outputs = {"m1": TrajectoryProduct, "m2": TrajectoryProduct}

    def __init__(self, m1, k1, m2, k2, kc, x1_0=1.0, x2_0=0.0, t_max=60.0, n=6001):
        self.m1, self.k1 = m1, k1
        self.m2, self.k2 = m2, k2
        self.kc = kc
        self.x1_0, self.x2_0 = x1_0, x2_0
        self.t_max, self.n = t_max, n

    def _rhs(self, t, y):
        x1, v1, x2, v2 = y
        a1 = (-(self.k1 + self.kc) * x1 + self.kc * x2) / self.m1
        a2 = (-(self.k2 + self.kc) * x2 + self.kc * x1) / self.m2
        return [v1, a1, v2, a2]

    def run(self, ctx: Context, **inputs):
        t_eval = np.linspace(0.0, self.t_max, self.n)
        sol = solve_ivp(
            self._rhs, (0.0, self.t_max), [self.x1_0, 0.0, self.x2_0, 0.0],
            t_eval=t_eval, rtol=1e-12, atol=1e-14,
        )
        return {
            "m1": TrajectoryProduct(t=sol.t, x=sol.y[0], v=sol.y[1]),
            "m2": TrajectoryProduct(t=sol.t, x=sol.y[2], v=sol.y[3]),
        }


# ---------------------------------------------------------------------------
# Analytic reference (equal masses and stiffnesses, x1(0)=1, x2(0)=0, at rest)
# ---------------------------------------------------------------------------


def normal_mode_frequencies(m: float, k: float, kc: float) -> tuple[float, float]:
    """In-phase and out-of-phase mode frequencies."""
    return np.sqrt(k / m), np.sqrt((k + 2 * kc) / m)


def analytic_two_mass(t: np.ndarray, m: float, k: float, kc: float):
    """Closed-form x1(t), x2(t) for the symmetric case -- the beat solution."""
    wa, wb = normal_mode_frequencies(m, k, kc)
    return 0.5 * (np.cos(wa * t) + np.cos(wb * t)), 0.5 * (
        np.cos(wa * t) - np.cos(wb * t)
    )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_cascade_graph(m1=1.0, k1=4.0, m2=1.0, k2=4.0, kc=0.2, t_max=60.0, n=6001):
    """ONE-WAY: Oscillator -> CouplingForce -> DrivenOscillator -> Report.

    osc1 runs free (its stiffness absorbs the coupling spring, but it never
    sees x2), its motion becomes a force, and that force drives osc2. A DAG,
    so the shipped scheduler runs it -- at the cost of dropping the kc*x2 term
    from mass 1's equation.
    """
    g = Graph()
    g.add("osc1", Oscillator(m=m1, k=k1 + kc, c=0.0, x0=1.0, v0=0.0,
                             t_max=t_max, n=n))
    g.add("coupling", CouplingForce(kc=kc, name="k_c x1"))
    g.add("osc2", DrivenOscillator(m=m2, k=k2 + kc, c=0.0, x0=0.0, v0=0.0, n=n))
    g.add("report", Report("cascade_force.txt"))
    g.connect("osc1.trajectory", "coupling.trajectory")
    g.connect("coupling.force", "osc2.force")
    g.connect("coupling.force", "report.signal")
    return g


def build_coupled_graph(m1=1.0, k1=4.0, m2=1.0, k2=4.0, kc=0.2, t_max=60.0, n=6001):
    """THE HONEST TOPOLOGY: each mass drives the other. Contains a cycle.

    Wiring succeeds. `BatchScheduler` raises CycleError. Nothing here is a
    mistake -- this is the correct diagram of the physics, and the framework is
    accurately reporting that the only shipped driver cannot execute it.
    """
    g = Graph()
    g.add("osc1", DrivenOscillator(m=m1, k=k1 + kc, c=0.0, x0=1.0, v0=0.0, n=n))
    g.add("osc2", DrivenOscillator(m=m2, k=k2 + kc, c=0.0, x0=0.0, v0=0.0, n=n))
    g.add("force_on_2", CouplingForce(kc=kc, name="k_c x1"))
    g.add("force_on_1", CouplingForce(kc=kc, name="k_c x2"))
    g.connect("osc1.trajectory", "force_on_2.trajectory")
    g.connect("force_on_2.force", "osc2.force")
    g.connect("osc2.trajectory", "force_on_1.trajectory")
    g.connect("force_on_1.force", "osc1.force")   # <-- closes the loop
    return g


def build_monolith_graph(m1=1.0, k1=4.0, m2=1.0, k2=4.0, kc=0.2, t_max=60.0, n=6001):
    """Exact, single node. Runs today; the coupling is invisible to the graph."""
    g = Graph()
    g.add("pair", TwoMassSystem(m1=m1, k1=k1, m2=m2, k2=k2, kc=kc,
                                t_max=t_max, n=n))
    return g
