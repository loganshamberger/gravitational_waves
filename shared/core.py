"""Reusable simulation framework — Layer 0 (the core spine).

See `shared/README.md` for a how-to guide with runnable examples, and
`docs/composable_simulation_framework.md` for the design rationale. This
docstring is the map; the README is the manual.

WHAT THIS MODULE IS
-------------------
A typed dataflow framework. You write `Process` nodes that declare named,
typed input and output ports; you wire them into a `Graph`; you hand the graph
to a `Scheduler`, which executes it. The framework's job is to make the
connections between steps explicit, checkable, and inspectable, instead of
leaving them buried inside one long function.

Two APIs live here, deliberately:

1. The ORIGINAL System/Runner/run_from_config catalog. Unchanged, still used by
   the geodesic project's simulator. Nothing about it breaks.

2. The COMPOSABLE GRAPH framework: Process / Graph / Scheduler / Context.
   `SystemProcess` adapts (1) into (2) as a degenerate Process, which is the
   migration path in the design doc's §8.

THIS MODULE CONTAINS NO PHYSICS. It imports nothing from any domain package and
names no domain-specific concept. That is asserted by a test, not a convention
— see 2_framework_pressure_test/test_reuse_guarantee.py, which also builds a
non-scientific graph (CSV → features → classifier → report) on this core to
prove the point.

THE FIVE CONCEPTS
-----------------
  Process     A node. Declares `inputs` / `outputs` (name → port type) and
              carries its configuration on the instance. Source / transform /
              sink are *shapes* (empty inputs / empty outputs / both), not
              subclasses.
  DataProduct Marker base for values that travel on edges. The core never
              inspects their fields.
  Context     The shared world every run needs: `rng`, `workdir`, `logger`.
              Nothing domain-specific — a domain that needs more subclasses it.
  Graph       Nodes plus typed edges. Pure STRUCTURE. May contain cycles.
  Scheduler   SEMANTICS: how a structure executes. Ships one implementation,
              `BatchScheduler`.

THE TWO IDEAS THAT ARE EASY TO MISS
-----------------------------------
1. STRUCTURE AND SEMANTICS ARE SEPARATE. `Graph` will happily hold a cycle;
   whether a cycle can *execute* is the scheduler's opinion. `BatchScheduler`
   refuses one, and says so in a message that states the graph itself is legal.
   That is what makes adding a cycle-tolerant scheduler later a non-breaking
   change rather than a redesign.

2. TYPE-CHECKING HAPPENS IN TWO PHASES.
      Phase 1, at wiring time  (`Graph.connect`)   — the port type *schema*.
      Phase 2, at bind time    (`Scheduler.bind`)  — the *ground* type, plus
                                                     node capabilities.
   A port may declare `Parametric(Series, Free("F"))`, leaving the element type
   open; the scheduler instantiates `F`. This exists because a driver can
   change the *representation* on a wire, not just how it is chunked in time —
   the same node might carry real samples under one driver and complex
   coefficients under another. Phase 1 cannot know which, so it defers.

Design commitments encoded below (design-doc sections in brackets):

  [3.1] A Process splits into a UNIVERSAL half (port names, arity, config,
        validate) and an EXECUTION half (the capability a scheduler binds to).
        Port *type schemas* are universal; port *ground types* are not.
  [3.3] Type-checking is two-phase, as above.
  [6]   Graph is structure; Scheduler is semantics.
  [6.2] Execution verbs are CAPABILITIES a node opts into: `Batchable` (ships),
        `Steppable` and `EventDriven` (declared, undriven). No verb is
        privileged.
  [6.4] Ports are parameterised, because drivers change representations.
  [9]   Callable / oracle ports are NOT built. Shape reserved only.

KNOWN LIMITATIONS (measured, not guessed — see the pressure-test FINDINGS.md)
----------------------------------------------------------------------------
  * No optional ports. Every declared input must be wired, so a node usable in
    two topologies has to be written twice.
  * Unconsumed outputs are dropped silently.
  * Only `Batchable` is drivable. Feedback loops must be buried inside a node,
    which for some problems changes the answer rather than merely approximating
    it.
  * `validate` sees only its own node's inputs, so cross-node validity
    conditions cannot be expressed.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Union, runtime_checkable

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
#
# Each error names the PHASE it belongs to, because which phase raised tells
# you whose fault it is: a schema error is a wiring mistake, a ground-type or
# capability error means this particular driver cannot run an otherwise-valid
# graph.
# ---------------------------------------------------------------------------


class MissingParameterError(ValueError):
    """Raised when a System is run without all of its required parameters.

    Belongs to the legacy catalog API, not the graph framework.
    """


class UnknownSimulationKindError(ValueError):
    """Raised when a config entry's `kind` has no matching System.

    Belongs to the legacy catalog API, not the graph framework.
    """


class PortError(ValueError):
    """A port does not exist, is wired twice, is left unwired, or was not returned.

    Structural mistakes about port *names and arity*, as opposed to their
    types. Raised at wiring time by `Graph.connect`, and at bind/run time by a
    scheduler when a required input is unwired or a node returns the wrong set
    of output ports.
    """


class SchemaMismatchError(TypeError):
    """PHASE 1 failure: these two ports can never be connected, under any driver.

    Raised at WIRING time by `Graph.connect`, and independent of any scheduler.
    If you see this, the graph is wrong — no choice of driver will help.
    """


class GroundTypeError(TypeError):
    """PHASE 2 failure: the schemas fit, but *this* driver's ground types do not.

    Raised at BIND time by a `Scheduler`. A graph that raises this under one
    scheduler may be perfectly legal under another — that is the entire point
    of the two-phase split. Also raised when a scheduler leaves a `Free`
    variable unbound.
    """


class CapabilityError(TypeError):
    """PHASE 2 failure: a node cannot be driven by this scheduler.

    The node does not implement the capability protocol the scheduler requires
    (e.g. it has no `run` and the scheduler needs `Batchable`). Like
    `GroundTypeError`, this is a statement about a graph-driver *pairing*, not
    about the graph alone.
    """


class CycleError(ValueError):
    """A scheduler cannot execute a cyclic structure.

    This is always a SCHEDULER error, never a `Graph` error. Cycles are legal
    structure; a cycle-tolerant scheduler added later would not raise. Also
    raised by `Graph.topological_order`, which is a convenience for schedulers
    that want a linear order and therefore inherits the same constraint.
    """


# ---------------------------------------------------------------------------
# Port type schemas [3.1, 3.3, 6.4]
#
# A port type is either a plain `type` or a `Parametric`. Plain types check by
# ordinary subclassing. Parametric types have one type argument, which may be
# left open as a `Free` variable for the scheduler to fill in.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Free:
    """A free type variable in a port schema, instantiated by the scheduler.

    Writing `Parametric(Series, Free("F"))` says "this port carries a Series of
    *something*, and which something depends on who is driving." One driver may
    bind `F` to a real-valued element type, another to a complex one, without
    either node changing.

    A scheduler declares its bindings in `Scheduler.ground_bindings`. A free
    variable no scheduler binds is an error — at bind time, not wiring time.

    Attributes:
        name: The variable's name, matched against a scheduler's
            `ground_bindings` keys. Conventionally a single capital letter.
    """

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Parametric:
    """A parameterised port type: a base kind plus exactly one type argument.

    Only one argument is supported; that has been sufficient so far, and
    keeping it to one keeps the compatibility rules readable.

    Attributes:
        base: The container/kind type, checked by subclassing.
        param: The element type. Either a concrete `type` (already ground) or
            a `Free` variable, resolved by the scheduler at bind time.

    Compatibility: `Parametric` never matches a plain `type` and vice versa.
    Mixing the two forms on a single edge is always a phase-1 error, because
    "a Series of X" and "an X" are different shapes, not different flavours.
    """

    base: type
    param: Union[type, Free]

    @property
    def is_ground(self) -> bool:
        """True when `param` is a concrete type rather than a `Free` variable."""
        return not isinstance(self.param, Free)

    def resolve(self, bindings: Mapping[str, type]) -> "Parametric":
        """Substitute this schema's free variable using a scheduler's bindings.

        Args:
            bindings: Map of variable name → concrete type, normally a
                scheduler's `ground_bindings`.

        Returns:
            A ground `Parametric`. Returns `self` unchanged when already ground.

        Raises:
            GroundTypeError: The variable is not present in `bindings`. The
                message lists what the scheduler *does* bind, since the usual
                cause is a typo or a scheduler that forgot a slot.
        """
        if self.is_ground:
            return self
        name = self.param.name
        if name not in bindings:
            raise GroundTypeError(
                f"free type variable {name!r} in {self} is unbound by this "
                f"scheduler (it binds {sorted(bindings)})"
            )
        return Parametric(self.base, bindings[name])

    def __str__(self) -> str:
        param = self.param.__name__ if isinstance(self.param, type) else str(self.param)
        return f"{self.base.__name__}[{param}]"


#: A port's declared type: either a plain class or a `Parametric` schema.
PortType = Union[type, Parametric]


def _name(t: PortType) -> str:
    """Human-readable name for either form of port type, for error messages."""
    return t.__name__ if isinstance(t, type) else str(t)


def schemas_compatible(produced: PortType, expected: PortType) -> bool:
    """PHASE 1 — the wiring-time check. Scheduler-independent.

    Answers: "could an output of type `produced` ever feed an input of type
    `expected`, under *some* driver?" It deliberately passes any disagreement
    that is confined to an unresolved free variable, because at wiring time
    nobody knows what that variable will become. Phase 2 settles it.

    The rules:
      * Plain vs plain — ordinary `issubclass`.
      * Plain vs parametric (either order) — never compatible.
      * Parametric vs parametric — bases must be subclass-compatible; if either
        parameter is still free, defer (return True); otherwise the parameters
        must be subclass-compatible too.

    Args:
        produced: The upstream output port's declared type.
        expected: The downstream input port's declared type.

    Returns:
        True if the edge is at least *potentially* valid.
    """
    if isinstance(produced, Parametric) != isinstance(expected, Parametric):
        return False
    if isinstance(produced, type):
        return issubclass(produced, expected)
    if not issubclass(produced.base, expected.base):
        return False
    if not produced.is_ground or not expected.is_ground:
        return True  # deferred to bind time
    return issubclass(produced.param, expected.param)


def ground_types_compatible(produced: PortType, expected: PortType) -> bool:
    """PHASE 2 — the bind-time check, after free variables have been resolved.

    Stricter than `schemas_compatible`: nothing is deferred, because by now the
    driver is known.

    Args:
        produced: The upstream output type, already passed through `resolve`.
        expected: The downstream input type, already passed through `resolve`.

    Returns:
        True if this driver can actually move a value along this edge.

    Raises:
        GroundTypeError: Either argument still contains a free variable, which
            means the caller skipped resolution — a framework bug, not a user
            error.
    """
    if isinstance(produced, type) and isinstance(expected, type):
        return issubclass(produced, expected)
    if isinstance(produced, Parametric) and isinstance(expected, Parametric):
        if not produced.is_ground or not expected.is_ground:
            raise GroundTypeError(
                f"{produced} / {expected} still contain free variables at bind time"
            )
        return issubclass(produced.base, expected.base) and issubclass(
            produced.param, expected.param
        )
    return False


def _resolve(t: PortType, bindings: Mapping[str, type]) -> PortType:
    """Resolve a port type if parametric; pass plain types through untouched."""
    return t.resolve(bindings) if isinstance(t, Parametric) else t


# ---------------------------------------------------------------------------
# Data products and context [3.2, 3.4]
# ---------------------------------------------------------------------------


class DataProduct:
    """Marker base for "a value that flows along an edge".

    Subclassing this is optional — any type works as a port type — but it
    documents intent and gives a domain a common ancestor to hang shared
    behaviour on. The core never inspects a product's fields; it only compares
    the *declared* port types.

    A frozen dataclass makes a good product: values on edges are read by
    several consumers and should not be mutated in place.
    """


@dataclass
class Context:
    """The shared world — only what *every* run needs.

    Deliberately tiny. There is nothing here about units, time bases, or
    coordinate frames, because those are domain concerns; a domain that needs a
    shared world of its own subclasses this and its nodes require the subclass.
    Core code only ever touches the three base fields, so the core stays
    domain-free.

    Attributes:
        rng: The single source of randomness. Nodes must draw from this and
            never from module-level `np.random`, or runs stop being
            reproducible from a seed.
        workdir: Where sinks write artefacts. Nodes should not write anywhere
            else, so a run can be redirected wholesale (tests point it at a
            temporary directory).
        logger: For progress and diagnostics. Nodes should log rather than
            print, so output can be captured or silenced.
    """

    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0)
    )
    workdir: Path = field(default_factory=lambda: Path("."))
    logger: logging.Logger = field(default_factory=lambda: logger)


# ---------------------------------------------------------------------------
# Process and capabilities [3.1, 6.2]
# ---------------------------------------------------------------------------


class Process(ABC):
    """A node in the graph. Declares typed ports; configuration lives on `self`.

    The single most important distinction this class draws is between
    CONFIGURATION and DATA:

      * Configuration — knobs fixed when the node is built — goes in
        `__init__` and lives on the instance. Not visible to the graph.
      * Data — values produced by other nodes — arrives through *ports*, as
        keyword arguments to `run`.

    Getting that split right is what makes nodes reusable: the same class
    configured differently is a different node, but the wiring is unchanged.

    Class attributes:
        inputs: Mapping of input port name → `PortType`. Empty for a source.
        outputs: Mapping of output port name → `PortType`. Empty for a sink.

    Source, transform, and sink are *shapes*, not subclasses:

        source     inputs = {}          outputs = {...}
        transform  inputs = {...}       outputs = {...}
        sink       inputs = {...}       outputs = {}

    A Process also splits along a second axis:

      * The UNIVERSAL half — port names, arity, configuration, and `validate`
        — is the same no matter who executes the node.
      * The EXECUTION half is a *capability* (`Batchable`, `Steppable`,
        `EventDriven`) that a scheduler binds to. Only `Batchable` is drivable
        today.

    To be runnable under the shipped scheduler, a subclass needs a `run`
    method matching the `Batchable` protocol. `Process` does not declare `run`
    abstract, precisely because the execution half is optional and
    capability-dependent.
    """

    inputs: Mapping[str, PortType] = {}
    outputs: Mapping[str, PortType] = {}

    @property
    def label(self) -> str:
        """Short display name, used in diagnostics and diagrams.

        Defaults to the class name. Override when one class is used for
        several roles and the class name alone would be ambiguous.
        """
        return type(self).__name__

    def validate(self, ctx: Context, **inputs: Any) -> None:
        """Check configuration and received inputs; raise if unfit.

        Called by the scheduler immediately before the node executes, with the
        same keyword arguments `run` is about to receive. The default accepts
        everything.

        This is where SEMANTIC checks belong — the ones the type system cannot
        express. Port types catch "you connected the wrong kind of thing"; this
        catches "the thing is the right kind but the values are wrong":
        negative masses, empty arrays, an input in the wrong unit system, a
        window shorter than the data it must hold.

        Two caveats worth knowing:
          * Inputs may be absent. `validate` is also useful to call directly in
            tests, so guard with `inputs.get(...)` or a default of `None`
            rather than assuming every port is present.
          * It sees only THIS node's inputs. Conditions spanning several nodes
            cannot be expressed here, and the framework has no other place for
            them (see the module docstring's limitations).

        Args:
            ctx: The run context.
            **inputs: The products about to be passed to `run`, keyed by input
                port name.

        Raises:
            Exception: Any exception aborts the run. Prefer `ValueError` with a
                message naming the offending value and what was expected.
        """


@runtime_checkable
class Batchable(Protocol):
    """Capability: runs to completion in one call. THE ONLY CAPABILITY THAT SHIPS.

    A node satisfies this by defining a `run` method with the signature below.
    No inheritance is needed — the protocol is structural, and
    `isinstance(node, Batchable)` checks for the method's presence.

    Contract for `run`:
      * Consume products via keyword arguments named after `inputs` ports.
      * Return a dict keyed by `outputs` port names. The scheduler enforces
        that the returned keys match the declared ports EXACTLY — extra or
        missing keys are a `PortError`. A sink returns `{}` (or `None`).
      * Draw randomness only from `ctx.rng`, and write only under
        `ctx.workdir`, or reproducibility and redirectability are lost.
    """

    def run(self, ctx: Context, **inputs: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class Steppable(Protocol):
    """Capability: advances in fixed time steps. RESERVED — no scheduler drives it.

    Declared so the seam is visible and `isinstance(node, Steppable)` is
    answerable today, and so that adding a fixed-step co-simulation driver
    later requires no change to `Process`, ports, or the type system.

    Building the actual stepper waits for a feedback-coupled problem that needs
    it — though note the pressure-test findings argue that point has arrived.
    """

    def init(self, ctx: Context) -> None:
        ...

    def step(self, ctx: Context, dt: float, **inputs: Any) -> Dict[str, Any]:
        ...


@runtime_checkable
class EventDriven(Protocol):
    """Capability: irregular event times, THE NODE owns the clock. RESERVED.

    Orthogonal to the batch/step ladder rather than a rung on it: a stepper
    driven to completion subsumes a one-shot run, but neither subsumes a node
    that decides for itself when its next event occurs.
    """

    def next_event_time(self) -> float:
        ...

    def handle_event(self, ctx: Context, t: float, **inputs: Any) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Graph -- structure only [3.3, 6]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One wire: a named output port feeding a named input port.

    Attributes:
        src: Name of the producing node, as registered with `Graph.add`.
        src_port: Output port name on `src`.
        dst: Name of the consuming node.
        dst_port: Input port name on `dst`.
    """

    src: str
    src_port: str
    dst: str
    dst_port: str

    def __str__(self) -> str:
        return f"{self.src}.{self.src_port} -> {self.dst}.{self.dst_port}"


class Graph:
    """Nodes plus typed edges. Pure structure. MAY CONTAIN CYCLES.

    A `Graph` describes what is connected to what. It does not execute
    anything, and it holds no opinion about whether it *can* be executed —
    that belongs to a `Scheduler`. In particular a cycle is perfectly legal
    here; the shipped batch driver is what refuses it.

    The only checking `Graph` performs is phase 1: port existence, duplicate
    wiring, and schema compatibility. It performs no ground-type or capability
    checking, because both depend on a driver it has not been told about.

    `add` and `connect` return `self`, so wiring can be chained.

    Attributes:
        nodes: Registered nodes, keyed by the name given to `add`.
        edges: Wires, in the order they were connected.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Process] = {}
        self.edges: List[Edge] = []

    # -- construction -------------------------------------------------------

    def add(self, name: str, node: Process) -> "Graph":
        """Register a node under a unique name.

        The name is how the node is referenced in `connect`, in results, and in
        diagnostics — short and role-descriptive beats long and type-descriptive
        (`"smooth"` rather than `"the_savitzky_golay_filter"`).

        Args:
            name: Unique identifier within this graph.
            node: The `Process` instance, already configured.

        Returns:
            self, for chaining.

        Raises:
            PortError: A node is already registered under `name`.
        """
        if name in self.nodes:
            raise PortError(f"node {name!r} already in graph")
        self.nodes[name] = node
        return self

    def _split(self, ref: str, side: str) -> tuple[str, str]:
        """Parse a `'node.port'` reference and check the node exists."""
        if "." not in ref:
            raise PortError(f"{side} {ref!r} must be 'node.port'")
        name, port = ref.split(".", 1)
        if name not in self.nodes:
            raise PortError(f"no node {name!r} in graph")
        return name, port

    def connect(self, src_ref: str, dst_ref: str) -> "Graph":
        """Wire one output port to one input port. PHASE 1 CHECKING HAPPENS HERE.

        Fan-OUT is allowed: one output may feed any number of inputs. Fan-IN is
        not: each input accepts exactly one edge, so that the value arriving on
        a port is never ambiguous. A node that must combine several upstreams
        declares several input ports.

        Args:
            src_ref: Producing side, as `'node.port'`.
            dst_ref: Consuming side, as `'node.port'`.

        Returns:
            self, for chaining.

        Raises:
            PortError: Malformed reference, unknown node, unknown port, or the
                destination input is already wired. The message lists the
                available port names, which usually identifies a typo directly.
            SchemaMismatchError: The two port types are not compatible under
                any driver.
        """
        src, src_port = self._split(src_ref, "source")
        dst, dst_port = self._split(dst_ref, "destination")

        produced = self.nodes[src].outputs.get(src_port)
        if produced is None:
            raise PortError(
                f"{src!r} has no output port {src_port!r} "
                f"(has: {sorted(self.nodes[src].outputs)})"
            )
        expected = self.nodes[dst].inputs.get(dst_port)
        if expected is None:
            raise PortError(
                f"{dst!r} has no input port {dst_port!r} "
                f"(has: {sorted(self.nodes[dst].inputs)})"
            )

        for e in self.edges:
            if e.dst == dst and e.dst_port == dst_port:
                raise PortError(f"input {dst}.{dst_port} is already wired from {e}")

        if not schemas_compatible(produced, expected):
            raise SchemaMismatchError(
                f"cannot wire {src_ref} -> {dst_ref}: "
                f"{_name(produced)} is not compatible with {_name(expected)}"
            )

        self.edges.append(Edge(src, src_port, dst, dst_port))
        return self

    # -- queries ------------------------------------------------------------

    def unwired_inputs(self) -> List[str]:
        """Input ports with no incoming edge, as `'node.port'` strings.

        Every declared input must be wired: there are no optional ports and no
        defaults. Schedulers call this during `bind`, so an unwired input is a
        bind-time failure rather than a confusing `TypeError` inside `run`.
        """
        wired = {(e.dst, e.dst_port) for e in self.edges}
        return [
            f"{n}.{p}"
            for n, node in self.nodes.items()
            for p in node.inputs
            if (n, p) not in wired
        ]

    def predecessors(self, name: str) -> set[str]:
        """Names of nodes with an edge into `name`."""
        return {e.src for e in self.edges if e.dst == name}

    def _kahn_order(self) -> List[str]:
        """Kahn's algorithm, returning as much order as exists.

        The single implementation behind `_toposort`, `cycle_nodes`, and
        `depths`, which previously each had their own copy with subtly
        different tie-breaking.

        Returns:
            Nodes in a deterministic topological order. For an acyclic graph
            this is every node. For a cyclic one it is the acyclic PREFIX —
            the nodes never made ready are exactly those in or downstream of a
            cycle, which is what makes this useful to all three callers.
        """
        indeg = {n: len(self.predecessors(n)) for n in self.nodes}
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: List[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for e in self.edges:
                if e.src != n:
                    continue
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    ready.append(e.dst)
            ready.sort()
        return order

    def _toposort(self) -> Optional[List[str]]:
        """Full topological order, or None when the structure is cyclic."""
        order = self._kahn_order()
        return order if len(order) == len(self.nodes) else None

    def has_cycle(self) -> bool:
        """True if any cycle exists. Says nothing about whether that is a problem."""
        return self._toposort() is None

    def cycle_nodes(self) -> set[str]:
        """Nodes participating in — or downstream of — a cycle.

        Exactly the nodes Kahn's algorithm can never make ready. Note the
        "or downstream of": a node fed by a cycle is also unschedulable by a
        topological driver even though it sits on no loop itself, and reporting
        it is more useful than hiding it.

        Returns:
            The offending node names; empty for an acyclic graph.
        """
        return set(self.nodes) - set(self._kahn_order())

    def topological_order(self) -> List[str]:
        """A safe execution order for acyclic structures.

        Offered as a convenience for schedulers that want one. It is NOT a
        claim that every scheduler executes in this order — an event-driven
        driver would ignore it entirely.

        Returns:
            Node names in dependency order.

        Raises:
            CycleError: The structure is cyclic. The message names the nodes
                involved.
        """
        order = self._toposort()
        if order is None:
            raise CycleError(f"structure is cyclic: {sorted(self.cycle_nodes())}")
        return order

    def depths(self) -> Dict[str, int]:
        """Longest-path depth per node — the column index for a layered layout.

        Sources are at depth 0 and each node sits one past its deepest
        predecessor, so a renderer can place depth on one axis and get a
        left-to-right flow.

        Cyclic graphs have no well-defined depth for the nodes in the loop, so
        those are laid out by BREADTH-FIRST ORDER OF ARRIVAL from the acyclic
        frontier — i.e. the order work actually traverses them. (Sorting them
        by name instead produces a diagram that reads backwards.) When every
        node is in the cycle there is no frontier at all, so the least-depended
        -on node seeds the walk and the result reads as a chain plus one
        back-edge.

        Returns:
            Node name → depth, defined for every node.
        """
        order = self._kahn_order()
        settled = set(order)

        depth = {n: 0 for n in self.nodes}
        for n in order:
            for e in self.edges:
                if e.src == n and e.dst in settled:
                    depth[e.dst] = max(depth[e.dst], depth[n] + 1)

        stuck = set(self.nodes) - settled
        if not stuck:
            return depth

        assigned = set(settled)
        if not assigned:
            indeg = {n: len(self.predecessors(n)) for n in stuck}
            seed = min(sorted(stuck), key=lambda n: indeg[n])
            depth[seed] = 0
            assigned = {seed}

        frontier = sorted(assigned, key=lambda n: depth[n])
        while frontier:
            nxt = []
            for n in frontier:
                for e in self.edges:
                    if e.src != n or e.dst in assigned:
                        continue
                    depth[e.dst] = depth[n] + 1
                    assigned.add(e.dst)
                    nxt.append(e.dst)
            frontier = nxt

        # A cycle with no entry edge at all is unreachable; trail it off the end.
        base = max(depth.values(), default=-1)
        for i, n in enumerate(sorted(stuck - assigned)):
            depth[n] = base + 1 + i
        return depth

    # -- inspection ---------------------------------------------------------

    def describe(self) -> str:
        """Plain-text dump of nodes, ports, edges, and cycle status.

        The quickest way to see what you actually wired, as opposed to what you
        believe you wired. Depends on nothing, so it is safe to call anywhere,
        including from a failing test.

        Returns:
            A multi-line string.
        """
        lines = [f"Graph: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        cyc = self.cycle_nodes()
        lines.append(
            f"  cyclic: {'YES -- ' + ', '.join(sorted(cyc)) if cyc else 'no'}"
        )
        for name in sorted(self.nodes):
            node = self.nodes[name]
            lines.append(f"  [{name}] {node.label}")
            for port, t in node.inputs.items():
                lines.append(f"      in  {port}: {_name(t)}")
            for port, t in node.outputs.items():
                lines.append(f"      out {port}: {_name(t)}")
        for e in self.edges:
            lines.append(f"  {e}")
        return "\n".join(lines)

    def to_dot(self, name: str = "graph") -> str:
        """Graphviz DOT source for this structure.

        String building only — graphviz is not imported and need not be
        installed. Pipe the result to `dot -Tpng` if you have it. Nodes in a
        cycle are shaded.

        Args:
            name: The digraph's name in the DOT output.

        Returns:
            DOT source as a string.
        """
        cyc = self.cycle_nodes()
        out = [f"digraph {name} {{", "  rankdir=LR;", "  node [shape=box];"]
        for n, node in self.nodes.items():
            style = ', style=filled, fillcolor="#ffe6e6"' if n in cyc else ""
            out.append(f'  "{n}" [label="{n}\\n{node.label}"{style}];')
        for e in self.edges:
            out.append(
                f'  "{e.src}" -> "{e.dst}" [label="{e.src_port}->{e.dst_port}"];'
            )
        out.append("}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Schedulers -- semantics [6]
# ---------------------------------------------------------------------------


class Scheduler(ABC):
    """Decides HOW a structure executes. Owns phase-2 checking.

    A scheduler is the *driver* of a graph. It declares two things and
    implements one method:

      * `capability` — the protocol every node must satisfy to be driven.
      * `ground_bindings` — concrete types for the `Free` variables appearing
        in port schemas.
      * `run` — execute the graph and return its products.

    Each scheduler has its own verb; none is privileged. `BatchScheduler` calls
    `run` on each node, but a fixed-step driver would call `step` and an
    event-driven one `handle_event`, and all three would share this base class,
    this checking, and the same `Graph`.

    Subclasses override `_bind_structure` to add their own structural rules
    (the batch driver rejects cycles there) and implement `run`. They should
    call `bind` first thing in `run`.

    Class attributes:
        capability: Protocol required of every node. Defaults to `Batchable`.
        ground_bindings: Free-variable name → concrete type. Defaults to empty,
            which is correct only for graphs whose ports are all ground.
    """

    capability: type = Batchable
    ground_bindings: Mapping[str, type] = {}

    def bind(self, graph: Graph, ctx: Context) -> None:
        """PHASE 2. Check capabilities, ground types, wiring completeness, structure.

        Raising here rather than at wiring time is the design's central claim:
        none of this information exists until a driver has been chosen. The
        same graph may bind cleanly under one scheduler and fail under another,
        and that is correct behaviour rather than an inconsistency.

        Checks run in this order, chosen so the most fundamental failure is
        reported first:
          1. Every node provides `capability`.
          2. Every edge's ground types agree once `ground_bindings` is applied.
          3. Every declared input is wired.
          4. `_bind_structure` — whatever else this driver requires.

        Args:
            graph: The structure to check.
            ctx: The run context. Unused by the base implementation; available
                to subclasses that need context-dependent admissibility.

        Raises:
            CapabilityError: A node cannot be driven by this scheduler.
            GroundTypeError: An edge's types disagree under this driver's
                bindings, or a free variable is unbound.
            PortError: A declared input has no incoming edge.
            CycleError: Raised by `BatchScheduler._bind_structure` for cyclic
                structures; other drivers may raise other errors here.
        """
        for name, node in graph.nodes.items():
            if not isinstance(node, self.capability):
                raise CapabilityError(
                    f"node {name!r} ({node.label}) does not provide "
                    f"{self.capability.__name__}, required by "
                    f"{type(self).__name__}"
                )

        for e in graph.edges:
            produced = _resolve(
                graph.nodes[e.src].outputs[e.src_port], self.ground_bindings
            )
            expected = _resolve(
                graph.nodes[e.dst].inputs[e.dst_port], self.ground_bindings
            )
            if not ground_types_compatible(produced, expected):
                raise GroundTypeError(
                    f"under {type(self).__name__}, edge {e} grounds to "
                    f"{_name(produced)} -> {_name(expected)}, which do not match"
                )

        missing = graph.unwired_inputs()
        if missing:
            raise PortError(f"unwired required inputs: {sorted(missing)}")

        self._bind_structure(graph)

    def _bind_structure(self, graph: Graph) -> None:
        """Hook for driver-specific structural rules. Default: accept anything.

        Override to reject shapes this driver cannot handle. Prefer an error
        message that states what the DRIVER cannot do, not what the graph got
        wrong — the graph is usually fine.
        """

    @abstractmethod
    def run(self, graph: Graph, ctx: Context) -> Dict[str, Dict[str, Any]]:
        """Execute the graph.

        Implementations should call `self.bind(graph, ctx)` before doing
        anything else.

        Args:
            graph: The structure to execute.
            ctx: The run context, passed to every node.

        Returns:
            Nested mapping `{node_name: {output_port: product}}`, covering
            every node that produced anything. Sinks appear with an empty dict.
        """


class BatchScheduler(Scheduler):
    """Topological, run-to-completion driver. Requires `Batchable`. Rejects cycles.

    The only scheduler that ships. It sorts the graph topologically, then calls
    each node's `validate` and `run` exactly once, threading products along
    edges.

    Its refusal of cycles is a property of THIS DRIVER, not of `Graph` — which
    is what makes adding a cycle-tolerant scheduler later a purely additive
    change. The error message says so explicitly, because the natural
    assumption on seeing it is that the graph is malformed.

    To bind free port variables, subclass and set `ground_bindings`:

        class MyDriver(BatchScheduler):
            ground_bindings = {"F": float}

    See `shared/README.md` for worked examples.
    """

    capability = Batchable

    def _bind_structure(self, graph: Graph) -> None:
        """Reject cyclic structures, naming the nodes and blaming the driver."""
        if graph.has_cycle():
            raise CycleError(
                "BatchScheduler cannot execute a cyclic structure; nodes in or "
                f"downstream of the cycle: {sorted(graph.cycle_nodes())}. "
                "The graph itself is legal -- a cycle-tolerant scheduler would "
                "accept it."
            )

    def run(self, graph: Graph, ctx: Context) -> Dict[str, Dict[str, Any]]:
        """Bind, then execute every node once in topological order.

        For each node: gather incoming products into keyword arguments named
        after its input ports, call `validate`, call `run`, and check that the
        returned keys match the declared output ports exactly. That last check
        is deliberately strict — a typo in a returned port name would otherwise
        surface much later as a mysteriously missing input downstream.

        Args:
            graph: The structure to execute.
            ctx: The run context, passed to every node.

        Returns:
            `{node_name: {output_port: product}}` for every node.

        Raises:
            PortError: A node returned undeclared or incomplete output ports.
            CycleError, CapabilityError, GroundTypeError: From `bind`.
            Exception: Anything a node's `validate` or `run` raises propagates
                unchanged; the run stops at the first failure.
        """
        self.bind(graph, ctx)
        order = graph.topological_order()

        products: Dict[str, Dict[str, Any]] = {}
        for name in order:
            node = graph.nodes[name]
            kwargs = {
                e.dst_port: products[e.src][e.src_port]
                for e in graph.edges
                if e.dst == name
            }
            node.validate(ctx, **kwargs)
            ctx.logger.debug("running %s (%s)", name, node.label)
            out = node.run(ctx, **kwargs) or {}

            unexpected = set(out) - set(node.outputs)
            if unexpected:
                raise PortError(
                    f"{name!r} returned undeclared output ports {sorted(unexpected)}"
                )
            missing = set(node.outputs) - set(out)
            if missing:
                raise PortError(
                    f"{name!r} did not return declared output ports {sorted(missing)}"
                )
            products[name] = out
        return products


# ---------------------------------------------------------------------------
# Migration: the original catalog API, unchanged [8]
#
# This half predates the graph framework and is kept working verbatim. New work
# should use Process/Graph/Scheduler; `SystemProcess` bridges the two so that
# adopting the new API never requires rewriting the old nodes first.
# ---------------------------------------------------------------------------


class System(ABC):
    """LEGACY. Base class for a self-contained, parameter-driven simulation.

    Predates the graph framework. A `System` takes a flat parameter dict, runs
    to completion, and renders itself. It cannot be composed — there is no way
    to declare that one System's output feeds another's input — which is the
    limitation `Process` exists to remove.

    Kept working unchanged for existing code. Wrap one in `SystemProcess` to
    use it as a graph node; write new nodes as `Process` subclasses directly.
    """

    @abstractmethod
    def validate(self, params: Dict[str, Any]) -> None:
        """Raise MissingParameterError (or similar) if params is unfit to simulate."""

    @abstractmethod
    def simulate(self, params: Dict[str, Any]) -> Any:
        """Run the simulation and return the result."""

    @abstractmethod
    def visualize(self, result: Any) -> None:
        """Render the result of simulate()."""


class Runner:
    """LEGACY. Validates params against a `System`, then runs it."""

    def __init__(self, system: System):
        self.system = system

    def run(self, params: Dict[str, Any]) -> Any:
        """Validate then simulate.

        Args:
            params: Flat parameter dict.

        Returns:
            Whatever `System.simulate` returns.
        """
        self.system.validate(params)
        return self.system.simulate(params)


class SystemProcess(Process):
    """Adapts a legacy `System` into a degenerate `Process`, so nothing breaks.

    The adapted node is a SOURCE: `inputs = {}`, with a single `result` output
    port typed `object`, carrying whatever `simulate` returned. Parameters are
    configuration, supplied at construction, exactly as they were before.

    The `object` port type means the framework can check almost nothing about
    what comes out — legacy Systems return ad-hoc dicts with no declared type.
    That is the cost of adaptation, and the reason to migrate a System properly
    (splitting it into real nodes with real product types) rather than leaving
    it wrapped forever.

    Args:
        system: The legacy `System` instance.
        params: The parameter dict it expects.
    """

    inputs: Mapping[str, PortType] = {}
    outputs: Mapping[str, PortType] = {"result": object}

    def __init__(self, system: System, params: Dict[str, Any]):
        self.system = system
        self.params = params

    @property
    def label(self) -> str:
        return f"SystemProcess({type(self.system).__name__})"

    def validate(self, ctx: Context, **inputs: Any) -> None:
        """Delegate to the wrapped System's own parameter validation."""
        self.system.validate(self.params)

    def run(self, ctx: Context, **inputs: Any) -> Dict[str, Any]:
        """Run the wrapped System and publish its result on the `result` port."""
        return {"result": self.system.simulate(self.params)}


def run_from_config(config_path: str, registry: Dict[str, type]) -> None:
    """LEGACY. Read a YAML file of simulations and run + visualize each one.

    The pre-graph entry point: a flat list of independent Systems, each run to
    completion and rendered. There is no way to express a connection between
    two entries — that is what `Graph` adds.

    `registry` maps a `kind` string to a System subclass. Expected YAML shape:
        simulations:
          - kind: <registry key>
            params: {<name>: <value>, ...}
          - kind: <another registry key>
            params: {...}

    Domain examples are deliberately omitted: this module is L0 and the no-leak
    test forbids naming any domain concept, even in a docstring. See each
    project's own config.yaml for a worked example.

    Args:
        config_path: Path to the YAML file.
        registry: Map of `kind` string → `System` subclass.

    Raises:
        UnknownSimulationKindError: A `kind` has no entry in `registry`. The
            message lists the known kinds.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for entry in config["simulations"]:
        kind = entry["kind"]
        system_cls = registry.get(kind)
        if system_cls is None:
            raise UnknownSimulationKindError(
                f"Unknown simulation kind {kind!r}; known kinds: {list(registry)}"
            )

        system = system_cls()
        runner = Runner(system)
        result = runner.run(entry["params"])
        system.visualize(result)
