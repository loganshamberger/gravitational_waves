"""Scenario 2 -- SDLC digital twin: backlog to deployment.

Purpose: pressure-test the framework on a NON-PHYSICS, feedback-shaped,
discrete problem. It doubles as the design's mandated reuse canary: if this
graph cannot be written cleanly on L0, the core is contaminated with physics.

    Backlog -> Triage -> Development -> CodeReview -> CI -> Deploy

Two things here are the actual test, not the simulation:

1. REWORK IS A CYCLE. CodeReview.rejected feeds back into Development, and CI
   failures feed back too. `build_rework_graph` wires that cycle; the Graph
   accepts it and BatchScheduler refuses to execute it. That is the
   structure/scheduler split doing visible work.

2. THE SAME GRAPH UNDER TWO DRIVERS. `AnalyticQueueScheduler` (defined here, in
   L3, without touching the core) grounds the wires to RATES where
   BatchScheduler grounds them to TOKENS. This is the SPICE .tran/.ac
   counterexample reproduced in miniature -- one structure, two
   representations. See FINDINGS.md, finding 2.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from core import (
    Context,
    DataProduct,
    Free,
    Graph,
    Parametric,
    Process,
    Scheduler,
)


# ---------------------------------------------------------------------------
# Products -- two representations of the same flow [6.4 family 2]
# ---------------------------------------------------------------------------


class Flow(DataProduct):
    """Base kind for "work moving between stages"."""


@dataclass
class TokenFlow(Flow):
    """Individual work items. What a simulation driver puts on a wire."""

    items: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class RateFlow(Flow):
    """Arrival rate and service statistics. What an ANALYTIC driver puts on a
    wire for the very same edge -- not a chunking of TokenFlow, a different
    mathematical object."""

    arrival_rate: float
    service_rate: float

    @property
    def utilisation(self) -> float:
        return self.arrival_rate / self.service_rate


# Ground-type markers the schedulers bind the free variable "R" to.
class Tokens:
    """Ground type: wires carry individual work items."""


class Rates:
    """Ground type: wires carry rate/queueing statistics."""


FLOW = Parametric(Flow, Free("R"))


@dataclass
class WorkItem:
    ident: int
    size: float
    attempts: int = 0


# ---------------------------------------------------------------------------
# Capabilities -- Batchable ships in core; this one is scenario-local
# ---------------------------------------------------------------------------


@runtime_checkable
class Analytic(Protocol):
    """Steady-state closed-form evaluation. A driver verb of its own [6.2].

    Deliberately NOT added to core: it is not universal, and adding it there
    would be exactly the layering violation the reuse guarantee forbids.
    """

    def analyze(self, ctx: Context, **inputs: Any) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class Backlog(Process):
    """Source: the incoming backlog."""

    inputs = {}
    outputs = {"items": FLOW}

    def __init__(self, n_items: int, mean_size: float = 3.0, arrival_rate: float = 5.0):
        self.n_items = n_items
        self.mean_size = mean_size
        self.arrival_rate = arrival_rate

    def validate(self, ctx: Context, **inputs) -> None:
        if self.n_items < 0:
            raise ValueError("n_items must be non-negative")

    def run(self, ctx: Context, **inputs):
        sizes = ctx.rng.exponential(self.mean_size, self.n_items)
        items = [WorkItem(ident=i, size=float(s)) for i, s in enumerate(sizes)]
        return {"items": TokenFlow(items)}

    def analyze(self, ctx: Context, **inputs):
        return {
            "items": RateFlow(
                arrival_rate=self.arrival_rate, service_rate=1.0 / self.mean_size
            )
        }


class Triage(Process):
    """Transform: drops items below a size threshold (won't-fix)."""

    inputs = {"incoming": FLOW}
    outputs = {"ready": FLOW}

    def __init__(self, min_size: float = 0.5):
        self.min_size = min_size

    def run(self, ctx: Context, incoming: TokenFlow):
        kept = [i for i in incoming.items if i.size >= self.min_size]
        ctx.logger.debug("triage kept %d/%d", len(kept), len(incoming.items))
        return {"ready": TokenFlow(kept)}

    def analyze(self, ctx: Context, incoming: RateFlow):
        # Triage thins the arrival stream; service is instantaneous.
        return {
            "ready": RateFlow(
                arrival_rate=incoming.arrival_rate * 0.85,
                service_rate=incoming.service_rate,
            )
        }


class Development(Process):
    """Transform WITH a rework input -- this is what creates the cycle.

    `rework` is fed by CodeReview.rejected. Structurally legal; not executable
    under BatchScheduler.
    """

    inputs = {"ready": FLOW, "rework": FLOW}
    outputs = {"for_review": FLOW}

    def __init__(self, capacity: int = 100):
        self.capacity = capacity

    def run(self, ctx: Context, ready: TokenFlow, rework: TokenFlow):
        items = (ready.items + rework.items)[: self.capacity]
        return {"for_review": TokenFlow(items)}

    def analyze(self, ctx: Context, ready: RateFlow, rework: RateFlow):
        return {
            "for_review": RateFlow(
                arrival_rate=ready.arrival_rate + rework.arrival_rate,
                service_rate=float(self.capacity),
            )
        }


class DevelopmentWithInternalRework(Process):
    """The SAME stage with the feedback loop buried inside the node.

    This is the monolith escape hatch the design warns about [6.3]. It is the
    only way to run this pipeline under the shipped scheduler, and it is
    included precisely so the cost is visible: the rework loop -- the single
    most interesting dynamic in an SDLC -- becomes invisible to the graph. You
    cannot attach a sink to it, count its iterations, or swap its policy.
    """

    inputs = {"ready": FLOW}
    outputs = {"for_review": FLOW}

    def __init__(self, capacity: int = 100, reject_rate: float = 0.3, max_attempts: int = 3):
        self.capacity = capacity
        self.reject_rate = reject_rate
        self.max_attempts = max_attempts

    def run(self, ctx: Context, ready: TokenFlow):
        done, queue = [], list(ready.items)[: self.capacity]
        while queue:
            item = queue.pop(0)
            item.attempts += 1
            rejected = ctx.rng.random() < self.reject_rate
            if rejected and item.attempts < self.max_attempts:
                queue.append(item)   # <-- the buried loop
            else:
                done.append(item)
        return {"for_review": TokenFlow(done)}

    def analyze(self, ctx: Context, ready: RateFlow):
        inflation = 1.0 / (1.0 - self.reject_rate)
        return {
            "for_review": RateFlow(
                arrival_rate=ready.arrival_rate * inflation,
                service_rate=float(self.capacity),
            )
        }


class CodeReview(Process):
    """Transform with TWO outputs: approvals and rejections."""

    inputs = {"for_review": FLOW}
    outputs = {"approved": FLOW, "rejected": FLOW}

    def __init__(self, reject_rate: float = 0.3):
        self.reject_rate = reject_rate

    def run(self, ctx: Context, for_review: TokenFlow):
        approved, rejected = [], []
        for item in for_review.items:
            (rejected if ctx.rng.random() < self.reject_rate else approved).append(item)
        return {"approved": TokenFlow(approved), "rejected": TokenFlow(rejected)}

    def analyze(self, ctx: Context, for_review: RateFlow):
        r = for_review.arrival_rate
        return {
            "approved": RateFlow(r * (1 - self.reject_rate), for_review.service_rate),
            "rejected": RateFlow(r * self.reject_rate, for_review.service_rate),
        }


class CI(Process):
    """Transform: continuous integration; flaky by construction."""

    inputs = {"candidate": FLOW}
    outputs = {"passing": FLOW, "failing": FLOW}

    def __init__(self, failure_rate: float = 0.15):
        self.failure_rate = failure_rate

    def run(self, ctx: Context, candidate: TokenFlow):
        passing, failing = [], []
        for item in candidate.items:
            (failing if ctx.rng.random() < self.failure_rate else passing).append(item)
        return {"passing": TokenFlow(passing), "failing": TokenFlow(failing)}

    def analyze(self, ctx: Context, candidate: RateFlow):
        r = candidate.arrival_rate
        return {
            "passing": RateFlow(r * (1 - self.failure_rate), candidate.service_rate),
            "failing": RateFlow(r * self.failure_rate, candidate.service_rate),
        }


class Deploy(Process):
    """Sink: writes a deployment summary."""

    inputs = {"release": FLOW}
    outputs = {}

    def __init__(self, filename: str = "deploy.txt"):
        self.filename = filename

    def run(self, ctx: Context, release: TokenFlow):
        path = ctx.workdir / self.filename
        total = sum(i.size for i in release.items)
        path.write_text(
            f"deployed {len(release.items)} items, total size {total:.3f}\n"
        )
        return {}

    def analyze(self, ctx: Context, release: RateFlow):
        path = ctx.workdir / self.filename
        path.write_text(
            f"steady state: throughput {release.arrival_rate:.3f}/day, "
            f"utilisation {release.utilisation:.3f}\n"
        )
        return {}


# ---------------------------------------------------------------------------
# A second scheduler, defined OUTSIDE the core [6, 6.4]
# ---------------------------------------------------------------------------


class AnalyticQueueScheduler(Scheduler):
    """Steady-state queueing driver. Grounds every wire to RATES, not tokens.

    Written entirely in L3 against the public core API -- no core change was
    needed to add a driver with a different verb AND a different
    representation. That is the claim under test.
    """

    capability = Analytic
    ground_bindings = {"R": Rates}

    def _bind_structure(self, graph: Graph) -> None:
        # A steady-state analysis of a cyclic network is well-defined (it is
        # just a linear system), but solving one is out of scope here, so this
        # driver declines cycles too -- for a DIFFERENT reason than the batch
        # driver, which is the point.
        if graph.has_cycle():
            raise NotImplementedError(
                "closed-form steady state for cyclic networks not implemented"
            )

    def run(self, graph: Graph, ctx: Context) -> Dict[str, Dict[str, Any]]:
        self.bind(graph, ctx)
        products: Dict[str, Dict[str, Any]] = {}
        for name in graph.topological_order():
            node = graph.nodes[name]
            kwargs = {
                e.dst_port: products[e.src][e.src_port]
                for e in graph.edges
                if e.dst == name
            }
            node.validate(ctx, **kwargs)
            products[name] = node.analyze(ctx, **kwargs) or {}
        return products


BATCH_BINDINGS = {"R": Tokens}


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_linear_graph(n_items: int = 200) -> Graph:
    """Feed-forward pipeline. Runs under BOTH drivers.

    Rework is buried inside DevelopmentWithInternalRework -- see that class for
    what this costs.
    """
    g = Graph()
    g.add("backlog", Backlog(n_items=n_items))
    g.add("triage", Triage())
    g.add("dev", DevelopmentWithInternalRework())
    g.add("review", CodeReview())
    g.add("ci", CI())
    g.add("deploy", Deploy())
    g.connect("backlog.items", "triage.incoming")
    g.connect("triage.ready", "dev.ready")
    g.connect("dev.for_review", "review.for_review")
    g.connect("review.approved", "ci.candidate")
    g.connect("ci.passing", "deploy.release")
    return g


def build_rework_graph(n_items: int = 200) -> Graph:
    """The HONEST topology: rejections feed back into development.

    Wiring succeeds -- the structure is legal. Executing it under
    BatchScheduler raises CycleError. Nothing here is a workaround; this is the
    framework correctly reporting that the shipped driver cannot run this
    shape.
    """
    g = Graph()
    g.add("backlog", Backlog(n_items=n_items))
    g.add("triage", Triage())
    g.add("dev", Development())
    g.add("review", CodeReview())
    g.add("ci", CI())
    g.add("deploy", Deploy())
    g.connect("backlog.items", "triage.incoming")
    g.connect("triage.ready", "dev.ready")
    g.connect("dev.for_review", "review.for_review")
    g.connect("review.rejected", "dev.rework")   # <-- the cycle
    g.connect("review.approved", "ci.candidate")
    g.connect("ci.passing", "deploy.release")
    return g
