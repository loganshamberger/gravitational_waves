"""L0 tests: the two-phase type check, the structure/scheduler split, capabilities."""

import numpy as np
import pytest

from core import (
    BatchScheduler,
    Batchable,
    CapabilityError,
    Context,
    CycleError,
    EventDriven,
    Free,
    Graph,
    GroundTypeError,
    Parametric,
    PortError,
    Process,
    SchemaMismatchError,
    Steppable,
    ground_types_compatible,
    schemas_compatible,
)


class Base:
    pass


class Derived(Base):
    pass


class Other:
    pass


class Alpha:
    pass


class Beta:
    pass


# ---------------------------------------------------------------------------
# Phase 1 vs phase 2 [3.3]
# ---------------------------------------------------------------------------


def test_plain_types_check_by_subclassing_at_wiring():
    assert schemas_compatible(Derived, Base)
    assert not schemas_compatible(Base, Derived)
    assert not schemas_compatible(Other, Base)


def test_parametric_and_plain_never_mix():
    assert not schemas_compatible(Parametric(Base, Alpha), Base)
    assert not schemas_compatible(Base, Parametric(Base, Alpha))


def test_free_parameter_defers_to_bind_time():
    """The heart of the two-phase claim: a free variable PASSES phase 1."""
    free = Parametric(Base, Free("F"))
    assert schemas_compatible(free, Parametric(Base, Alpha))
    assert schemas_compatible(Parametric(Base, Alpha), free)


def test_ground_mismatch_is_caught_at_phase_1_when_both_are_ground():
    assert not schemas_compatible(Parametric(Base, Alpha), Parametric(Base, Beta))


def test_phase_2_rejects_what_phase_1_deferred():
    produced = Parametric(Base, Free("F")).resolve({"F": Alpha})
    expected = Parametric(Base, Beta)
    assert not ground_types_compatible(produced, expected)


def test_phase_2_refuses_to_run_on_unresolved_variables():
    with pytest.raises(GroundTypeError, match="free variable"):
        ground_types_compatible(Parametric(Base, Free("F")), Parametric(Base, Alpha))


def test_unbound_variable_names_the_scheduler_bindings():
    with pytest.raises(GroundTypeError, match="unbound"):
        Parametric(Base, Free("Z")).resolve({"F": Alpha})


# ---------------------------------------------------------------------------
# Minimal nodes
# ---------------------------------------------------------------------------


class Src(Process):
    inputs = {}
    outputs = {"out": Parametric(Base, Free("F"))}

    def run(self, ctx, **kw):
        return {"out": Base()}


class Snk(Process):
    inputs = {"in": Parametric(Base, Free("F"))}
    outputs = {}

    def run(self, ctx, **kw):
        return {}


class PlainSrc(Process):
    inputs = {}
    outputs = {"out": Derived}

    def run(self, ctx, **kw):
        return {"out": Derived()}


class PlainSnk(Process):
    inputs = {"in": Other}
    outputs = {}

    def run(self, ctx, **kw):
        return {}


def test_wiring_rejects_incompatible_schemas_immediately():
    g = Graph().add("a", PlainSrc()).add("b", PlainSnk())
    with pytest.raises(SchemaMismatchError, match="not compatible"):
        g.connect("a.out", "b.in")


def test_unknown_ports_are_named_helpfully():
    g = Graph().add("a", Src()).add("b", Snk())
    with pytest.raises(PortError, match="no output port"):
        g.connect("a.nope", "b.in")
    with pytest.raises(PortError, match="no input port"):
        g.connect("a.out", "b.nope")


def test_an_input_cannot_be_wired_twice():
    g = Graph().add("a", Src()).add("a2", Src()).add("b", Snk())
    g.connect("a.out", "b.in")
    with pytest.raises(PortError, match="already wired"):
        g.connect("a2.out", "b.in")


def test_scheduler_rejects_unbound_free_variable_at_bind_time():
    """Wiring succeeded; the driver still refuses. That is phase 2 earning its keep."""

    class NoBindings(BatchScheduler):
        ground_bindings = {}

    g = Graph().add("a", Src()).add("b", Snk())
    g.connect("a.out", "b.in")
    with pytest.raises(GroundTypeError, match="unbound"):
        NoBindings().run(g, Context())


def test_same_graph_accepted_by_one_driver_and_refused_by_another():
    """The claim that ground types are scheduler-dependent, as a test [6.4]."""

    class SrcAlpha(Process):
        inputs = {}
        outputs = {"out": Parametric(Base, Alpha)}

        def run(self, ctx, **kw):
            return {"out": Base()}

    class SnkFree(Process):
        inputs = {"in": Parametric(Base, Free("F"))}
        outputs = {}

        def run(self, ctx, **kw):
            return {}

    def make():
        g = Graph().add("a", SrcAlpha()).add("b", SnkFree())
        g.connect("a.out", "b.in")
        return g

    class BindsAlpha(BatchScheduler):
        ground_bindings = {"F": Alpha}

    class BindsBeta(BatchScheduler):
        ground_bindings = {"F": Beta}

    BindsAlpha().run(make(), Context())          # fine
    with pytest.raises(GroundTypeError):
        BindsBeta().run(make(), Context())       # same structure, different driver


# ---------------------------------------------------------------------------
# Structure vs scheduler [6]
# ---------------------------------------------------------------------------


class Loop(Process):
    inputs = {"a": Base, "b": Base}
    outputs = {"out": Base}

    def run(self, ctx, **kw):
        return {"out": Base()}


class Pass(Process):
    inputs = {"x": Base}
    outputs = {"y": Base}

    def run(self, ctx, **kw):
        return {"y": Base()}


class BaseSrc(Process):
    inputs = {}
    outputs = {"out": Base}

    def run(self, ctx, **kw):
        return {"out": Base()}


def test_graph_accepts_a_cycle_and_the_scheduler_refuses_it():
    g = Graph().add("s", BaseSrc()).add("loop", Loop()).add("p", Pass())
    g.connect("s.out", "loop.a")
    g.connect("loop.out", "p.x")
    g.connect("p.y", "loop.b")        # cycle: loop -> p -> loop

    assert g.has_cycle()                             # structure is fine
    assert g.cycle_nodes() == {"loop", "p"}
    with pytest.raises(CycleError, match="cannot execute a cyclic"):
        BatchScheduler().run(g, Context())           # the DRIVER objects


def test_unwired_inputs_are_caught_at_bind_time():
    g = Graph().add("s", BaseSrc()).add("loop", Loop())
    g.connect("s.out", "loop.a")
    with pytest.raises(PortError, match="unwired required inputs"):
        BatchScheduler().run(g, Context())


# ---------------------------------------------------------------------------
# Capabilities [6.2]
# ---------------------------------------------------------------------------


def test_a_node_without_the_capability_is_refused_by_name():
    class NotBatchable(Process):
        inputs = {}
        outputs = {"out": Base}
        # no run()

    g = Graph().add("x", NotBatchable())
    with pytest.raises(CapabilityError, match="does not provide Batchable"):
        BatchScheduler().run(g, Context())


def test_capability_protocols_are_structurally_checkable():
    class Stepper(Process):
        inputs = {}
        outputs = {}

        def init(self, ctx):
            pass

        def step(self, ctx, dt, **kw):
            return {}

        def run(self, ctx, **kw):
            return {}

    s = Stepper()
    assert isinstance(s, Batchable)
    assert isinstance(s, Steppable)
    assert not isinstance(s, EventDriven)


def test_declared_output_ports_are_enforced():
    class Liar(Process):
        inputs = {}
        outputs = {"out": Base}

        def run(self, ctx, **kw):
            return {"surprise": Base()}

    g = Graph().add("x", Liar())
    with pytest.raises(PortError, match="undeclared output ports"):
        BatchScheduler().run(g, Context())

    class Slacker(Process):
        inputs = {}
        outputs = {"out": Base, "other": Base}

        def run(self, ctx, **kw):
            return {"out": Base()}

    g2 = Graph().add("x", Slacker())
    with pytest.raises(PortError, match="did not return declared"):
        BatchScheduler().run(g2, Context())


def test_context_rng_is_seeded_and_reproducible():
    def draw():
        ctx = Context(rng=np.random.default_rng(1234))
        return ctx.rng.random(5)

    assert np.array_equal(draw(), draw())
