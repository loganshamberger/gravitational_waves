"""Scenario 3 tests: coupling two oscillators through the framework."""

import numpy as np
import pytest

from core import BatchScheduler, Context, CycleError
from scenario_oscillator import BATCH_BINDINGS
from scenario_coupled_oscillators import (
    CouplingForce,
    analytic_two_mass,
    build_cascade_graph,
    build_coupled_graph,
    build_monolith_graph,
    normal_mode_frequencies,
)


class Batch(BatchScheduler):
    ground_bindings = BATCH_BINDINGS


def ctx(tmp_path):
    return Context(rng=np.random.default_rng(0), workdir=tmp_path)


# ---------------------------------------------------------------------------
# The reference solution is genuinely correct
# ---------------------------------------------------------------------------


def test_monolith_matches_the_closed_form_beat_solution(tmp_path):
    m, k, kc = 1.0, 4.0, 0.2
    out = Batch().run(build_monolith_graph(kc=kc), ctx(tmp_path))
    t = out["pair"]["m1"].t
    x1_exact, x2_exact = analytic_two_mass(t, m, k, kc)

    assert np.max(np.abs(out["pair"]["m1"].x - x1_exact)) < 1e-8
    assert np.max(np.abs(out["pair"]["m2"].x - x2_exact)) < 1e-8


def test_energy_transfers_completely_between_the_masses(tmp_path):
    """The physical signature of coupling: mass 2 starts at rest and ends up
    carrying essentially all the motion, then hands it back."""
    out = Batch().run(build_monolith_graph(kc=0.2, t_max=120.0), ctx(tmp_path))
    x1, x2 = out["pair"]["m1"].x, out["pair"]["m2"].x
    assert x2[0] == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(x2)) > 0.99          # full transfer at the beat node
    assert np.min(np.abs(x1)) < 0.01          # mass 1 momentarily at rest


def test_beat_period_matches_the_normal_mode_splitting(tmp_path):
    m, k, kc = 1.0, 4.0, 0.2
    wa, wb = normal_mode_frequencies(m, k, kc)
    out = Batch().run(build_monolith_graph(kc=kc, t_max=200.0, n=20001), ctx(tmp_path))
    t, x2 = out["pair"]["m2"].t, out["pair"]["m2"].x

    # x2 = sin((wb+wa)t/2) sin((wb-wa)t/2): the envelope's FIRST maximum is at
    # t = pi/(wb - wa), and it recurs at odd multiples, so search only the
    # first beat -- a global argmax picks arbitrarily among near-equal peaks.
    first_peak = np.pi / (wb - wa)
    window = t <= 1.5 * first_peak
    peak_t = t[window][np.argmax(np.abs(x2[window]))]

    # The product's maximum snaps to the nearest CARRIER peak, so it can sit up
    # to half a carrier period away from the envelope maximum. That is physics,
    # not error -- so that is the tolerance.
    carrier_period = 4 * np.pi / (wa + wb)
    assert abs(peak_t - first_peak) < carrier_period / 2

    # And the energy really has moved to mass 2 there.
    assert np.abs(x2[window]).max() > 0.99


# ---------------------------------------------------------------------------
# The cascade -- the graph the framework CAN run
# ---------------------------------------------------------------------------


def test_cascade_runs_and_couples_the_two_systems(tmp_path):
    """The headline: the free oscillator's motion drives the second one,
    entirely through typed graph edges."""
    out = Batch().run(build_cascade_graph(), ctx(tmp_path))
    x2 = out["osc2"]["trajectory"].x
    assert np.max(np.abs(x2)) > 0.0, "mass 2 never moved -- coupling did nothing"
    assert (tmp_path / "cascade_force.txt").exists()


def test_the_coupling_force_is_a_visible_product_on_the_wire(tmp_path):
    """Contrast with the monolith: here you can inspect the interaction."""
    out = Batch().run(build_cascade_graph(kc=0.2), ctx(tmp_path))
    force = out["coupling"]["force"]
    x1 = out["osc1"]["trajectory"].x
    assert np.allclose(force.values, 0.2 * x1)
    assert force.name == "k_c x1"


def test_zero_coupling_leaves_the_second_mass_at_rest(tmp_path):
    out = Batch().run(build_cascade_graph(kc=0.0), ctx(tmp_path))
    assert np.max(np.abs(out["osc2"]["trajectory"].x)) < 1e-9


def test_cascade_is_accurate_for_weak_coupling(tmp_path):
    """One-way coupling is a real physical limit, not a fudge: as kc -> 0 the
    neglected back-reaction vanishes and the cascade converges on the truth."""
    m, k = 1.0, 4.0
    kc = 0.002
    out = Batch().run(build_cascade_graph(kc=kc, t_max=60.0), ctx(tmp_path))
    t = out["osc2"]["trajectory"].t
    _, x2_exact = analytic_two_mass(t, m, k, kc)
    err = np.max(np.abs(out["osc2"]["trajectory"].x - x2_exact))
    assert err < 0.02


def test_cascade_diverges_secularly_for_identical_masses(tmp_path):
    """FINDING F5 -- the sharp version. Dropping the cycle is not a loss of
    accuracy, it is a change of QUALITATIVE BEHAVIOUR.

    For two identical undamped oscillators the drive frequency equals mass 2's
    natural frequency exactly, so one-way coupling drives it on undamped
    resonance and its amplitude grows without bound. The exact solution is
    bounded (energy sloshes back and forth). The back-reaction -- precisely the
    term the cycle would have carried -- is what prevents the runaway.
    """
    amps = []
    for t_max in (60.0, 120.0, 240.0, 480.0):
        out = Batch().run(
            build_cascade_graph(kc=0.2, t_max=t_max, n=20001), ctx(tmp_path)
        )
        amps.append(np.max(np.abs(out["osc2"]["trajectory"].x)))

    # Linear growth: each doubling of the window roughly doubles the amplitude.
    for a, b in zip(amps, amps[1:]):
        assert b / a == pytest.approx(2.0, rel=0.05)
    assert amps[-1] > 20.0

    # Meanwhile the true solution never exceeds 1.
    out = Batch().run(build_monolith_graph(kc=0.2, t_max=480.0, n=20001), ctx(tmp_path))
    assert np.max(np.abs(out["pair"]["m2"].x)) < 1.001


def test_cascade_error_grows_with_coupling_strength(tmp_path):
    """FINDING F5, as evidence: the framework-imposed approximation has a cost,
    and the cost is a function of exactly the term the cycle would have carried.
    """
    m, k = 1.0, 4.0

    def cascade_error(kc):
        out = Batch().run(build_cascade_graph(kc=kc, t_max=60.0), ctx(tmp_path))
        t = out["osc2"]["trajectory"].t
        _, x2_exact = analytic_two_mass(t, m, k, kc)
        return np.max(np.abs(out["osc2"]["trajectory"].x - x2_exact))

    weak, strong = cascade_error(0.002), cascade_error(1.0)
    assert weak < 0.02
    assert strong > 0.5
    assert strong > 20 * weak


# ---------------------------------------------------------------------------
# The honest topology -- the graph the framework CANNOT run
# ---------------------------------------------------------------------------


def test_the_bidirectional_topology_wires_successfully():
    """Physics that is genuinely a cycle is legal STRUCTURE."""
    g = build_coupled_graph()
    assert g.has_cycle()
    assert {"osc1", "osc2", "force_on_1", "force_on_2"} == g.cycle_nodes()


def test_the_batch_driver_refuses_the_bidirectional_topology(tmp_path):
    g = build_coupled_graph()
    with pytest.raises(CycleError, match="cannot execute a cyclic"):
        Batch().run(g, ctx(tmp_path))


def test_all_three_graphs_describe_the_same_physics(tmp_path):
    """The framework's choice is not between right and wrong models -- it is
    between three encodings of one system, of which it can run two."""
    kc = 0.2
    cascade = build_cascade_graph(kc=kc)
    coupled = build_coupled_graph(kc=kc)
    monolith = build_monolith_graph(kc=kc)

    Batch().run(cascade, ctx(tmp_path))                  # runs, approximate
    Batch().run(monolith, ctx(tmp_path))                 # runs, exact, opaque
    with pytest.raises(CycleError):
        Batch().run(coupled, ctx(tmp_path))              # exact, transparent, unrunnable

    # And the one that is both exact AND transparent is the one refused.
    assert len(coupled.nodes) == 4                       # coupling spring visible
    assert len(monolith.nodes) == 1                      # coupling spring buried


def test_the_monolith_hides_the_coupling_force(tmp_path):
    """The concrete cost of the escape hatch, mirroring the SDLC finding."""
    mono = Batch().run(build_monolith_graph(kc=0.2), ctx(tmp_path))
    casc = Batch().run(build_cascade_graph(kc=0.2), ctx(tmp_path))

    assert set(mono["pair"]) == {"m1", "m2"}             # no force anywhere
    assert "force" in casc["coupling"]                   # visible in the cascade


def test_the_coupling_node_is_swappable(tmp_path):
    """Why exposing the seam matters: a different interaction law is a
    different node, with no change to either oscillator."""

    class NonlinearCoupling(CouplingForce):
        def run(self, ctx, trajectory):
            from kinds import Signal

            return {
                "force": Signal(
                    t=trajectory.t,
                    values=self.kc * np.tanh(3.0 * trajectory.x),
                    name="nonlinear",
                )
            }

    g = build_cascade_graph(kc=0.2)
    g.nodes["coupling"] = NonlinearCoupling(kc=0.2)
    out = Batch().run(g, ctx(tmp_path))
    assert out["coupling"]["force"].name == "nonlinear"
    assert np.max(np.abs(out["osc2"]["trajectory"].x)) > 0.0


def test_validate_rejects_a_negative_coupling_constant(tmp_path):
    g = build_cascade_graph()
    g.nodes["coupling"] = CouplingForce(kc=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        Batch().run(g, ctx(tmp_path))
