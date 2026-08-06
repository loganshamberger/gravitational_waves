"""Scenario 1 tests: the physics must be right, and the framework must carry it."""

import numpy as np
import pytest

from core import BatchScheduler, Context, GroundTypeError, Graph, Parametric
from kinds import Complex, Real, Signal
from scenario_oscillator import (
    BATCH_BINDINGS,
    DrivenOscillator,
    DrivingForce,
    EnergyTrace,
    Oscillator,
    Report,
    build_driven_graph,
    build_free_graph,
)


class Batch(BatchScheduler):
    ground_bindings = BATCH_BINDINGS


def ctx(tmp_path):
    return Context(rng=np.random.default_rng(0), workdir=tmp_path)


# ---------------------------------------------------------------------------
# The physics is actually correct
# ---------------------------------------------------------------------------


def test_undamped_oscillator_has_the_analytic_period(tmp_path):
    m, k = 1.0, 4.0
    out = Batch().run(build_free_graph(m=m, k=k, c=0.0, t_max=50.0), ctx(tmp_path))
    traj = out["osc"]["trajectory"]

    # x(t) = cos(w t) with w = sqrt(k/m) = 2 -> T = pi
    expected = np.cos(np.sqrt(k / m) * traj.t)
    assert np.max(np.abs(traj.x - expected)) < 1e-6


def test_undamped_oscillator_conserves_energy(tmp_path):
    m, k = 1.0, 4.0
    out = Batch().run(build_free_graph(m=m, k=k, c=0.0, t_max=50.0), ctx(tmp_path))
    e = out["energy"]["energy"].values
    assert np.ptp(e) / e.mean() < 1e-6


def test_damping_monotonically_removes_energy(tmp_path):
    out = Batch().run(build_free_graph(m=1.0, k=4.0, c=0.3, t_max=50.0), ctx(tmp_path))
    e = out["energy"]["energy"].values
    assert e[-1] < e[0]
    # Envelope decays as exp(-c t / m); check the ratio over one span.
    assert e[-1] / e[0] < 1e-4


def test_driven_oscillator_reaches_the_analytic_steady_state_amplitude(tmp_path):
    """Validates the wired-in force, not just that the graph ran."""
    m, k, c, F0 = 1.0, 4.0, 0.4, 1.0
    w = 1.3                       # deliberately off-resonance
    g = build_driven_graph(m=m, k=k, c=c, amplitude=F0, omega_d=w, t_max=400.0)
    out = Batch().run(g, ctx(tmp_path))
    traj = out["osc"]["trajectory"]

    # A = F0 / sqrt((k - m w^2)^2 + (c w)^2)
    expected = F0 / np.sqrt((k - m * w**2) ** 2 + (c * w) ** 2)
    tail = traj.x[traj.t > 300.0]
    assert np.max(np.abs(tail)) == pytest.approx(expected, rel=2e-2)


def test_resonance_amplifies_relative_to_off_resonance(tmp_path):
    def amplitude(w):
        g = build_driven_graph(m=1.0, k=4.0, c=0.2, amplitude=1.0,
                               omega_d=w, t_max=400.0)
        traj = Batch().run(g, ctx(tmp_path))["osc"]["trajectory"]
        return np.max(np.abs(traj.x[traj.t > 300.0]))

    assert amplitude(2.0) > 3 * amplitude(1.0)      # w0 = sqrt(k/m) = 2


def test_zero_amplitude_driving_reproduces_the_free_solution(tmp_path):
    """Composition sanity: the driven graph degenerates to the free one."""
    g = build_driven_graph(m=1.0, k=4.0, c=0.0, amplitude=0.0, t_max=20.0)
    g.nodes["osc"].x0, g.nodes["osc"].v0 = 1.0, 0.0
    driven = Batch().run(g, ctx(tmp_path))["osc"]["trajectory"]
    free = Batch().run(
        build_free_graph(m=1.0, k=4.0, c=0.0, t_max=20.0), ctx(tmp_path)
    )["osc"]["trajectory"]
    assert np.max(np.abs(driven.x - free.x)) < 1e-6


# ---------------------------------------------------------------------------
# The framework carried it
# ---------------------------------------------------------------------------


def test_config_is_separated_from_data_on_the_wire(tmp_path):
    """m and k are constructor args; the force arrives on a port. [3.1]"""
    node = DrivenOscillator(m=1.0, k=4.0)
    assert node.inputs == {"force": Parametric(Signal, node.inputs["force"].param)}
    assert "m" not in node.inputs and "k" not in node.inputs


def test_the_same_sink_serves_two_different_signals(tmp_path):
    """A sink over a shared kind, not a method welded to one System. [3.1]"""
    Batch().run(build_free_graph(t_max=20.0), ctx(tmp_path))
    Batch().run(build_driven_graph(t_max=20.0), ctx(tmp_path))
    assert (tmp_path / "free_energy.txt").exists()
    assert (tmp_path / "driven_energy.txt").exists()


def test_validate_rejects_bad_config_before_anything_runs(tmp_path):
    g = Graph().add("osc", Oscillator(m=-1.0, k=4.0))
    g.add("energy", EnergyTrace(m=1.0, k=4.0))
    g.connect("osc.trajectory", "energy.trajectory")
    with pytest.raises(ValueError, match="need m>0"):
        Batch().run(g, ctx(tmp_path))


def test_solver_accuracy_is_capped_by_the_producers_grid(tmp_path):
    """FINDING 1, as evidence rather than assertion.

    The consumer is an adaptive solver running at rtol=1e-10, but the force
    arrived on a fixed grid, so it must interpolate. Accuracy is therefore set
    by a choice made by the PRODUCER, which cannot know what the consumer
    needs. Refining only the producer's grid -- changing nothing about the
    consumer -- improves the answer by orders of magnitude. That is the
    signature of a missing oracle edge [§9].
    """
    m, k, c, F0, w = 1.0, 4.0, 0.4, 1.0, 1.3
    exact = F0 / np.sqrt((k - m * w**2) ** 2 + (c * w) ** 2)

    def amplitude_with_grid(n):
        g = Graph()
        g.add("force", DrivingForce(amplitude=F0, omega_d=w, t_max=400.0, n=n))
        g.add("osc", DrivenOscillator(m=m, k=k, c=c))
        g.connect("force.force", "osc.force")
        traj = Batch().run(g, ctx(tmp_path))["osc"]["trajectory"]
        return np.max(np.abs(traj.x[traj.t > 300.0]))

    coarse = abs(amplitude_with_grid(101) - exact) / exact
    fine = abs(amplitude_with_grid(40001) - exact) / exact

    assert coarse > 0.05, "expected the coarse producer grid to visibly degrade"
    assert fine < 0.005
    assert coarse > 20 * fine


def test_a_frequency_domain_driver_is_refused_at_bind_time(tmp_path):
    """These nodes only produce real samples; a complex-grounding driver says so.

    This is §6.4's claim in miniature: the STRUCTURE is fine, the driver is
    what disagrees. Note the graph is byte-identical to the one that runs.
    """

    class ACDriver(BatchScheduler):
        ground_bindings = {"F": Complex}

    class RealOnlyForce(DrivingForce):
        outputs = {"force": Parametric(Signal, Real)}

    g = Graph()
    g.add("force", RealOnlyForce(amplitude=1.0, omega_d=2.0, t_max=10.0))
    g.add("osc", DrivenOscillator(m=1.0, k=4.0))
    g.connect("force.force", "osc.force")          # phase 1 passes

    with pytest.raises(GroundTypeError, match="do not match"):
        ACDriver().run(g, ctx(tmp_path))           # phase 2 refuses
