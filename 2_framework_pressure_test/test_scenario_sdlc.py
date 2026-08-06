"""Scenario 2 tests: cycles, two drivers over one structure, and the monolith cost."""

import numpy as np
import pytest

from core import (
    BatchScheduler,
    CapabilityError,
    Context,
    CycleError,
    Graph,
    GroundTypeError,
    Process,
)
from scenario_sdlc import (
    BATCH_BINDINGS,
    AnalyticQueueScheduler,
    Backlog,
    CodeReview,
    Deploy,
    RateFlow,
    TokenFlow,
    Triage,
    build_linear_graph,
    build_rework_graph,
)


class Batch(BatchScheduler):
    ground_bindings = BATCH_BINDINGS


def ctx(tmp_path, seed=0):
    return Context(rng=np.random.default_rng(seed), workdir=tmp_path)


# ---------------------------------------------------------------------------
# It runs, and the numbers are sane
# ---------------------------------------------------------------------------


def test_pipeline_runs_end_to_end(tmp_path):
    out = Batch().run(build_linear_graph(n_items=300), ctx(tmp_path))
    assert (tmp_path / "deploy.txt").exists()
    assert len(out["ci"]["passing"]) > 0


def test_items_are_conserved_across_every_split(tmp_path):
    """Nothing is invented or lost at a two-output stage."""
    out = Batch().run(build_linear_graph(n_items=300), ctx(tmp_path))
    reviewed = len(out["dev"]["for_review"])
    assert len(out["review"]["approved"]) + len(out["review"]["rejected"]) == reviewed
    approved = len(out["review"]["approved"])
    assert len(out["ci"]["passing"]) + len(out["ci"]["failing"]) == approved


def test_triage_drops_only_undersized_items(tmp_path):
    out = Batch().run(build_linear_graph(n_items=500), ctx(tmp_path))
    kept = out["triage"]["ready"].items
    assert all(i.size >= 0.5 for i in kept)
    assert len(kept) < len(out["backlog"]["items"])


def test_deployed_count_tracks_the_reject_and_failure_rates(tmp_path):
    """Statistical check with a fixed seed, not just 'it produced something'."""
    g = build_linear_graph(n_items=4000)
    g.nodes["review"].reject_rate = 0.25
    g.nodes["ci"].failure_rate = 0.10
    out = Batch().run(g, ctx(tmp_path, seed=11))

    reviewed = len(out["dev"]["for_review"])
    deployed = len(out["ci"]["passing"])
    assert deployed / reviewed == pytest.approx(0.75 * 0.90, rel=0.05)


def test_run_is_reproducible_from_the_seed(tmp_path):
    def once():
        out = Batch().run(build_linear_graph(n_items=300), ctx(tmp_path, seed=5))
        return len(out["ci"]["passing"])

    assert once() == once()


# ---------------------------------------------------------------------------
# The cycle: legal structure, unexecutable under this driver [6]
# ---------------------------------------------------------------------------


def test_the_rework_topology_wires_successfully():
    """Wiring a feedback edge is NOT an error. The graph holds cycles."""
    g = build_rework_graph()
    assert g.has_cycle()
    assert {"dev", "review"} <= g.cycle_nodes()


def test_the_batch_driver_refuses_the_rework_topology(tmp_path):
    g = build_rework_graph()
    with pytest.raises(CycleError, match="cannot execute a cyclic"):
        Batch().run(g, ctx(tmp_path))


def test_the_refusal_names_the_offending_nodes_and_says_it_is_the_drivers_fault(tmp_path):
    g = build_rework_graph()
    with pytest.raises(CycleError) as e:
        Batch().run(g, ctx(tmp_path))
    msg = str(e.value)
    assert "dev" in msg and "review" in msg
    assert "graph itself is legal" in msg


def test_burying_the_loop_is_what_makes_it_runnable(tmp_path):
    """The monolith escape hatch works -- and that is the problem [6.3].

    Identical pipeline semantics; the only difference is whether the rework
    loop is a graph edge or hidden inside a node. Exactly one of them runs.
    """
    with pytest.raises(CycleError):
        Batch().run(build_rework_graph(n_items=100), ctx(tmp_path))

    Batch().run(build_linear_graph(n_items=100), ctx(tmp_path))  # no raise


def test_the_buried_loop_is_invisible_to_the_graph(tmp_path):
    """The cost, made concrete: no port exposes the rework iterations.

    In the honest topology the rework volume is a product on an edge you could
    hang a sink on. Buried, the graph cannot see it at all -- you must reach
    inside the node's private result to learn anything.
    """
    out = Batch().run(build_linear_graph(n_items=200), ctx(tmp_path))
    assert "rework" not in out["dev"]
    assert set(out["dev"]) == {"for_review"}

    retried = [i for i in out["dev"]["for_review"].items if i.attempts > 1]
    assert retried, "rework did happen -- it is simply not on any wire"


# ---------------------------------------------------------------------------
# One structure, two drivers, two representations [6.4]
# ---------------------------------------------------------------------------


def test_the_same_graph_runs_under_both_drivers(tmp_path):
    """The SPICE .tran/.ac claim, in miniature."""
    tokens = Batch().run(build_linear_graph(n_items=300), ctx(tmp_path))
    rates = AnalyticQueueScheduler().run(build_linear_graph(n_items=300), ctx(tmp_path))

    assert isinstance(tokens["ci"]["passing"], TokenFlow)
    assert isinstance(rates["ci"]["passing"], RateFlow)


def test_the_two_drivers_agree_on_the_deployed_fraction(tmp_path):
    """Different representations, same underlying pipeline -- so they must agree.

    This is the closest thing available to §6.3's observational-equivalence
    invariant across representations, and note how weak it has to be: the two
    products are not comparable, so only a DERIVED scalar can be checked. See
    FINDINGS.md finding 4.
    """
    g_tok = build_linear_graph(n_items=6000)
    tok = Batch().run(g_tok, ctx(tmp_path, seed=3))
    simulated = len(tok["ci"]["passing"]) / len(tok["dev"]["for_review"])

    rates = AnalyticQueueScheduler().run(build_linear_graph(), ctx(tmp_path))
    analytic = (
        rates["ci"]["passing"].arrival_rate / rates["dev"]["for_review"].arrival_rate
    )
    assert simulated == pytest.approx(analytic, rel=0.05)


def test_the_analytic_driver_needed_no_core_change():
    """It is defined in this scenario file, against the public core API only."""
    import core

    import scenario_sdlc

    assert not hasattr(core, "Analytic")
    assert not hasattr(core, "AnalyticQueueScheduler")
    assert AnalyticQueueScheduler.__module__ == scenario_sdlc.__name__


def test_a_batch_only_node_is_refused_by_the_analytic_driver(tmp_path):
    """Capability check, on a node that is perfectly valid under the other driver."""

    class SimulationOnlyDeploy(Deploy):
        analyze = None      # explicitly does not support closed form

    g = build_linear_graph(n_items=50)
    g.nodes["deploy"] = SimulationOnlyDeploy()
    with pytest.raises(CapabilityError, match="does not provide Analytic"):
        AnalyticQueueScheduler().run(g, ctx(tmp_path))


def test_a_driver_that_binds_no_representation_is_refused(tmp_path):
    class Unbound(BatchScheduler):
        ground_bindings = {}

    with pytest.raises(GroundTypeError, match="unbound"):
        Unbound().run(build_linear_graph(n_items=10), ctx(tmp_path))


# ---------------------------------------------------------------------------
# Non-physics canary properties [7]
# ---------------------------------------------------------------------------


def test_this_entire_scenario_imports_no_physics():
    import scenario_sdlc

    src = open(scenario_sdlc.__file__).read().lower()
    for noun in ("numpy", "scipy", "schwarzschild", "orbit", "waveform"):
        assert noun not in src.split('"""', 2)[-1], f"{noun} leaked into the SDLC scenario"
