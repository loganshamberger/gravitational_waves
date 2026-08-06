# `core.py` — a typed dataflow framework for simulations

Build a simulation out of **nodes with typed ports**, wire them into a
**graph**, and hand the graph to a **scheduler** that runs it.

The point is to make the connections between steps explicit and checkable
instead of leaving them buried inside one long function. If step A's output
feeds step B's input, that is an edge you can see, type-check, draw, and tap a
plot onto — not a variable passed down a call stack.

- **How to use it** — this file.
- **Why it is shaped this way** — [`docs/composable_simulation_framework.md`](../docs/composable_simulation_framework.md).
- **What breaks when you push it** — [`2_framework_pressure_test/FINDINGS.md`](../2_framework_pressure_test/FINDINGS.md), and four worked scenarios beside it.

Every code block below has been executed as written.

---

## Contents

1. [The mental model](#1-the-mental-model)
2. [Quick start](#2-quick-start)
3. [Writing a Process](#3-writing-a-process)
4. [Port types](#4-port-types)
5. [Context](#5-context)
6. [Wiring a Graph](#6-wiring-a-graph)
7. [Schedulers](#7-schedulers)
8. [Cycles](#8-cycles)
9. [Inspecting a graph](#9-inspecting-a-graph)
10. [Adopting it in existing code](#10-adopting-it-in-existing-code)
11. [Testing your nodes](#11-testing-your-nodes)
12. [Error reference](#12-error-reference)
13. [What is deliberately not built](#13-what-is-deliberately-not-built)

---

## 1. The mental model

Five concepts, and two ideas that are easy to miss.

| Concept | What it is |
|---|---|
| `Process` | A node. Declares named, typed `inputs` and `outputs`; holds its configuration on the instance. |
| `DataProduct` | Marker base for values that travel on edges. |
| `Context` | The shared world: `rng`, `workdir`, `logger`. Nothing else. |
| `Graph` | Nodes plus typed edges. Pure **structure**. May contain cycles. |
| `Scheduler` | **Semantics** — how a structure executes. One ships: `BatchScheduler`. |

**Idea 1 — structure and semantics are separate.** A `Graph` will happily hold
a cycle. Whether that cycle can *execute* is the scheduler's opinion, and
`BatchScheduler` says no. This is why adding a cycle-tolerant driver later is
additive rather than a redesign.

**Idea 2 — type-checking happens in two phases.**

```
Graph.connect(...)     →  phase 1: port type SCHEMA        (scheduler-independent)
Scheduler.run(...)     →  phase 2: GROUND types + capabilities  (driver-specific)
```

A port can declare "a series of *something*" and let the scheduler decide what
the something is. Phase 1 cannot know, so it defers. See
[§4](#4-port-types).

---

## 2. Quick start

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from core import BatchScheduler, Context, DataProduct, Graph, Process

@dataclass
class Table(DataProduct):
    rows: np.ndarray

class MakeData(Process):                       # source: no inputs
    inputs, outputs = {}, {"table": Table}
    def __init__(self, n):
        self.n = n                             # configuration
    def run(self, ctx, **kw):
        return {"table": Table(ctx.rng.normal(size=(self.n, 2)))}

class Standardise(Process):                    # transform
    inputs, outputs = {"table": Table}, {"table": Table}
    def run(self, ctx, table):                 # data arrives by port name
        x = table.rows
        return {"table": Table((x - x.mean(0)) / x.std(0))}

class Summary(Process):                        # sink: no outputs
    inputs, outputs = {"table": Table}, {}
    def __init__(self, filename):
        self.filename = filename
    def run(self, ctx, table):
        (ctx.workdir / self.filename).write_text(f"n={len(table.rows)}\n")
        return {}

g = (Graph()
     .add("src",  MakeData(500))
     .add("norm", Standardise())
     .add("out",  Summary("summary.txt")))
g.connect("src.table",  "norm.table")
g.connect("norm.table", "out.table")

ctx = Context(rng=np.random.default_rng(0), workdir=Path("/tmp"))
products = BatchScheduler().run(g, ctx)

products["norm"]["table"].rows.shape       # (500, 2)
```

`run` returns `{node_name: {port_name: product}}` for every node, so you can
reach any intermediate value, not just the final one.

---

## 3. Writing a Process

### Configuration vs. data — the split that matters

This is the distinction to get right; almost everything else follows from it.

- **Configuration** — knobs fixed when you build the node — goes in
  `__init__` and lives on `self`. The graph never sees it.
- **Data** — values produced by other nodes — arrives through **ports**, as
  keyword arguments to `run`.

```python
class Resample(Process):
    inputs, outputs = {"signal": Series}, {"signal": Series}

    def __init__(self, sample_rate):   # configuration: fixed at build time
        self.sample_rate = sample_rate

    def run(self, ctx, signal):        # data: arrives on the "signal" port
        ...
```

Same class, two different rates, two different nodes — and the wiring is
identical. Had `sample_rate` been a port, every graph would need a node to
supply a constant.

### Source, transform, sink are shapes, not base classes

```python
class Source(Process):    inputs = {};        outputs = {"out": T}
class Transform(Process): inputs = {"in": T}; outputs = {"out": T}
class Sink(Process):      inputs = {"in": T}; outputs = {}
```

A sink returns `{}` (or `None`). Nothing else distinguishes them.

### The `run` contract

- Keyword arguments are named after your `inputs` port names.
- Return a dict keyed by your `outputs` port names — **exactly**. Extra or
  missing keys raise `PortError`. This is strict on purpose: a typo in a
  returned port name would otherwise surface much later as a mysteriously
  missing input three nodes downstream.
- Draw randomness **only** from `ctx.rng` and write **only** under
  `ctx.workdir`.

### `validate` — semantic checks the type system cannot express

Port types catch *"you connected the wrong kind of thing."* `validate` catches
*"it is the right kind but the values are wrong."*

```python
class Window(Process):
    inputs, outputs = {"signal": Series}, {"signal": Series}

    def __init__(self, width):
        self.width = width

    def validate(self, ctx, signal=None, **kw):
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if signal is not None and len(signal.values) < self.width:
            raise ValueError(
                f"window of {self.width} needs at least that many samples, "
                f"got {len(signal.values)}"
            )
```

Two things to know:

- **Default inputs to `None`.** `validate` is also handy to call directly in
  tests, so guard with `signal is not None` rather than assuming every port is
  present.
- **It sees only this node's inputs.** Conditions spanning several nodes cannot
  be expressed, and there is nowhere else to put them
  ([§13](#13-what-is-deliberately-not-built)).

---

## 4. Port types

A port type is either a **plain class** or a **`Parametric` schema**.

### Plain types

Checked by ordinary subclassing — an output of type `Derived` may feed an input
of type `Base`.

```python
inputs = {"table": Table}
```

### Parametric types — when the driver picks the representation

Sometimes a node carries "a series of *something*", where the something depends
on how the graph is being run: real samples under a time-domain driver, complex
coefficients under a frequency-domain one. Declare that with a `Free` variable:

```python
from core import Free, Parametric

SERIES = Parametric(Series, Free("R"))     # "a Series of R, for some R"

class Source(Process):
    inputs, outputs = {}, {"out": SERIES}
```

The scheduler fills `R` in:

```python
class CountDriver(BatchScheduler):
    ground_bindings = {"R": Counts}

class RateDriver(BatchScheduler):
    ground_bindings = {"R": Rates}
```

Both drivers run the *same graph object*; the edges mean different things under
each. A driver that binds nothing fails at bind time, not wiring time:

```
GroundTypeError: free type variable 'R' in Series[R] is unbound by this
scheduler (it binds [])
```

If all your ports are plain types you never need any of this — leave
`ground_bindings` empty and ignore the feature.

### The two phases, precisely

| | Phase 1 — `Graph.connect` | Phase 2 — `Scheduler.bind` |
|---|---|---|
| Knows the driver? | No | Yes |
| Checks | Port exists, not already wired, **schemas** compatible | **Ground** types, node capabilities, all inputs wired, driver-specific structure |
| Raises | `PortError`, `SchemaMismatchError` | `GroundTypeError`, `CapabilityError`, `PortError`, `CycleError` |
| Meaning of failure | The graph is wrong. No driver will help. | This graph and this driver are incompatible. Another driver may be fine. |

Compatibility rules for phase 1:

- plain vs plain → `issubclass`
- plain vs parametric, either order → **never** compatible ("a Series of X" and
  "an X" are different shapes)
- parametric vs parametric → bases must be subclass-compatible; if either
  parameter is still free, **defer**; otherwise parameters must be
  subclass-compatible too

---

## 5. Context

```python
@dataclass
class Context:
    rng: np.random.Generator     # the ONLY source of randomness
    workdir: Path                # the ONLY place to write
    logger: logging.Logger
```

Deliberately tiny. There is nothing about units, time bases, or coordinate
frames, because those are domain concerns — the core stays domain-free by
having no opinion about them.

If your domain needs a shared world of its own, **subclass** it and have your
nodes require the subclass:

```python
@dataclass
class MyContext(Context):
    target_units: str = "SI"
```

Core code only ever touches the three base fields, so this is safe.

**Reproducibility is a discipline, not a guarantee.** The framework cannot stop
a node calling `np.random.normal()` directly. If it does, seeded runs stop
reproducing. Same for writing outside `workdir` — do that and tests can no
longer redirect a run into a temporary directory.

---

## 6. Wiring a Graph

```python
g = Graph()
g.add("name", node)               # returns self, so calls chain
g.connect("src.port", "dst.port") # returns self
```

- **Fan-out is allowed.** One output may feed any number of inputs.
- **Fan-in is not.** Each input accepts exactly one edge, so the value arriving
  on a port is never ambiguous. A node that combines several upstreams declares
  several input ports.
- **Every declared input must be wired.** There are no optional ports and no
  defaults; an unwired input fails at bind time.

Error messages list the available names, which usually pinpoints a typo:

```
PortError: 's' has no output port 'nope' (has: ['y'])
```

---

## 7. Schedulers

### Running one

```python
products = BatchScheduler().run(graph, ctx)
```

`BatchScheduler` sorts topologically, then for each node gathers its incoming
products, calls `validate`, calls `run`, and checks the returned port names.

### Capabilities

A scheduler declares which execution verb it requires. Nodes opt in
*structurally* — no inheritance, just define the method:

| Capability | Method(s) | Status |
|---|---|---|
| `Batchable` | `run(ctx, **inputs)` | **Ships.** The only drivable one. |
| `Steppable` | `init(ctx)`, `step(ctx, dt, **inputs)` | Declared, no driver. |
| `EventDriven` | `next_event_time()`, `handle_event(ctx, t, **inputs)` | Declared, no driver. |

A node missing the required method is refused by name at bind time:

```
CapabilityError: node 'n' (NoRun) does not provide Batchable, required by
BatchScheduler
```

### Writing your own driver

You do not need to touch `core.py` to add one. Declare a capability, declare
your bindings, implement `run`, and call `bind` first:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Estimable(Protocol):
    def estimate(self, ctx, **inputs): ...

class RateDriver(Scheduler):
    capability = Estimable            # a verb the core has never heard of
    ground_bindings = {"R": Rates}

    def run(self, graph, ctx):
        self.bind(graph, ctx)         # phase 2: always do this first
        out = {}
        for n in graph.topological_order():
            kw = {e.dst_port: out[e.src][e.src_port]
                  for e in graph.edges if e.dst == n}
            out[n] = graph.nodes[n].estimate(ctx, **kw) or {}
        return out
```

Override `_bind_structure(graph)` to reject shapes your driver cannot handle.
When you do, **blame the driver, not the graph** — the graph is usually fine:

```python
def _bind_structure(self, graph):
    if graph.has_cycle():
        raise CycleError(
            f"{type(self).__name__} cannot execute a cyclic structure: "
            f"{sorted(graph.cycle_nodes())}. The graph itself is legal."
        )
```

---

## 8. Cycles

A cycle is legal **structure** and an illegal **batch execution**:

```python
g.has_cycle()        # True
g.cycle_nodes()      # {'loop', 'relay'}
BatchScheduler().run(g, ctx)
```

```
CycleError: BatchScheduler cannot execute a cyclic structure; nodes in or
downstream of the cycle: ['loop', 'relay']. The graph itself is legal -- a
cycle-tolerant scheduler would accept it.
```

`cycle_nodes()` reports nodes in a cycle **or downstream of one** — a node fed
by a cycle is equally unschedulable, and naming it is more useful than hiding
it.

**This matters more than it looks.** The tempting workaround — collapse the
loop into one node — hides the feedback from the graph, and for some problems
that is not an approximation but a different answer. See FINDINGS F5, where two
coupled oscillators diverge without their feedback edge while the true solution
stays bounded.

---

## 9. Inspecting a graph

```python
print(g.describe())      # nodes, ports, edges, cycle status — no dependencies
print(g.to_dot())        # Graphviz DOT source; graphviz need not be installed
g.depths()               # {node: column index} for layered layout
```

`describe()` is the fastest way to see what you actually wired rather than what
you believe you wired, and it is safe to call from a failing test.

For rendered diagrams, `2_framework_pressure_test/viz.py` has `plot_graph`,
which draws the layout with cycle members shaded and back-edges dashed. It
lives there rather than here so that `core.py` imports no plotting library.

---

## 10. Adopting it in existing code

Nothing has to be rewritten up front. The legacy `System` / `Runner` /
`run_from_config` API still works unchanged, and `SystemProcess` turns any
existing `System` into a graph node:

```python
from core import SystemProcess, Graph, BatchScheduler, Context

g = Graph().add("legacy", SystemProcess(MySystem(), {"x": 21}))
BatchScheduler().run(g, Context())["legacy"]["result"]
```

The adapted node is a source with a single `result` port typed `object`.

That `object` is the catch: the framework can check almost nothing about what
comes out, because legacy Systems return ad-hoc dicts with no declared type. So
treat `SystemProcess` as a bridge, not a destination — the value appears when
you split a System into real nodes with real product types and the seams
between them become visible.

Suggested order: wrap it → run it in a graph → split the largest hidden
pipeline inside it into two nodes → give the value passing between them a
product type → repeat.

---

## 11. Testing your nodes

A `Process` is an ordinary object, so unit-test it without a graph:

```python
node = Standardise()
out = node.run(Context(), table=Table(np.array([[1.0, 2.0], [3.0, 4.0]])))
assert out["table"].rows.mean() == pytest.approx(0.0)
```

Test `validate` separately, since it is where your error messages live:

```python
with pytest.raises(ValueError, match="width must be positive"):
    Window(width=-1).validate(Context())
```

Then test the wiring, which is a different failure mode:

```python
with pytest.raises(SchemaMismatchError):
    g.connect("producer.out", "consumer.wrong_kind")
```

Point `Context.workdir` at pytest's `tmp_path` and pass an explicit seed, and
your runs become reproducible and self-cleaning:

```python
ctx = Context(rng=np.random.default_rng(0), workdir=tmp_path)
```

**The reuse guarantee is a test, not a promise.**
`2_framework_pressure_test/test_reuse_guarantee.py` asserts that `core.py`
imports nothing domain-specific and names no domain concept, and builds a
non-scientific graph (CSV → features → classifier → report) on it. If you add
anything domain-flavoured to the core, that test fails — as it did once
already, catching a domain example left in a docstring.

---

## 12. Error reference

| Error | Phase | Means |
|---|---|---|
| `PortError` | 1 and 2 | Port does not exist, input wired twice, input unwired, or `run` returned the wrong set of port names. |
| `SchemaMismatchError` | 1 | These ports can never connect. **The graph is wrong.** |
| `GroundTypeError` | 2 | Schemas fit, but this driver's ground types do not — or a `Free` variable is unbound. **Another driver might work.** |
| `CapabilityError` | 2 | This node cannot be driven by this scheduler. |
| `CycleError` | 2 | This *driver* cannot execute a cyclic structure. The graph is still legal. |
| `MissingParameterError`, `UnknownSimulationKindError` | — | Legacy catalog API only. |

The useful habit: **which phase raised tells you whose fault it is.** Phase 1
means fix the wiring. Phase 2 means the wiring is fine and the graph/driver
pairing is not.

---

## 13. What is deliberately not built

Measured limitations, not speculation — each was found by a scenario in
`2_framework_pressure_test/` and is written up in its `FINDINGS.md`.

| Limitation | Consequence |
|---|---|
| **No optional ports** | Every declared input must be wired, so a node usable in two topologies must be written twice. (F2) |
| **Unconsumed outputs vanish silently** | An output wired to nothing is dropped with no warning. (F4) |
| **Only `Batchable` is drivable** | Feedback must be buried inside a node — which for some problems changes the answer rather than approximating it. (F5) |
| **No callable / oracle ports** | A consumer that needs values at points it discovers at runtime must accept a pre-sampled series instead, capping its accuracy at the producer's grid. (F1) |
| **No cross-node validity conditions** | `validate` sees only its own inputs, so every node can be correct while the composition is meaningless. (F6) |
| **Observational equivalence is representation-local** | "Two drivers agree" is only statable within one representation. (F3) |

Priority for the next round of work is in FINDINGS.md; the short version is
that a cycle-tolerant scheduler comes first, because it is the only one of
these that produces wrong results rather than imprecise ones.
