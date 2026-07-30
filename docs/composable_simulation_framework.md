# Composable Simulation Framework — Design Note

Design notes for evolving `shared/core.py` from a **catalog of independent
Systems** into a **typed dataflow graph of composable Processes** — with a hard
guarantee that the framework core stays domain-agnostic and reusable for
simulations that have nothing to do with gravitational waves. No code yet —
this is the abstraction set, the layering discipline, the type contracts, and
an honest accounting of what changes and what stays.

Motivating example: inject an EMRI inspiral waveform into a simulated LIGO
detector stream and recover it — two systems with nothing internally in common,
whose *interaction* is the science. But that is only the first application. The
same core must be able to drive a circuit simulation, a climate box model, or a
data pipeline with **zero physics vocabulary in it**.

---

## 1. What the current interface cannot express

`shared/core.py`'s `System` is `validate / simulate / visualize`: each System
takes a flat `params` dict, runs to completion, writes a PNG. A catalog of
things you run in isolation. Three structural limits:

1. **"My input is another system's output."** `simulate(params)` conflates
   *config knobs* (`M`, `mu`) with *data*. A result is an ad-hoc
   `Dict[str, np.ndarray]`, not a typed thing another system declares it
   consumes. No wire connects one system's output to another's input.
2. **"These two systems live in different worlds."** Inspiral is geometric
   units, coordinate time, dimensionless radius; LIGO is SI seconds, physical
   strain, ~16 kHz sampling, an antenna pattern. Composition is only physically
   meaningful once units/time-base/frame are reconciled — but see §2: that
   reconciliation must NOT live in the framework core.
3. **"Show me the thing in the middle."** `visualize` is a terminal method on
   each System — no way to view an intermediate product; every composite
   reimplements plotting.

**The tell:** the inspiral already contains a hidden pipeline — its docstring
says the trajectory *"flows through mass_quadrupole → quadrupole_jerk →
strain_plus_cross unmodified."* That composition exists today as straight-line
calls buried in one class. This design mostly *extracts and reifies* it.

---

## 2. The reframe — and the reuse guarantee

> From a catalog of Systems you run one at a time
> to a typed dataflow graph of Processes connected by typed data products,
> **on a core that contains no physics.**

The central design commitment, and the answer to "don't let LIGO decisions
lock me out of reusing this elsewhere":

> **The core never reconciles anything.** No units logic, no resampling, no
> frames. Every reconciliation is just another *node* in the graph
> (`GeometricToSI`, `Resample`, `UnitConvert` are ordinary transforms in a
> domain library, not machinery in the spine). This leaves the core with zero
> physics surface to contaminate.

Enforced as a checkable rule (§7): **`shared/core.py` imports nothing from any
domain package and contains no physics noun**, verified by a non-physics canary
graph in its own test suite.

### The four layers

| Layer | Lives in | Knows about | Example contents |
|------|----------|-------------|------------------|
| **L0 Core** | `shared/core.py` | nothing domain-specific | `Process`, `Graph`, `Context(rng, workdir, logger)`, `DataProduct` marker |
| **L1 Shared kinds** *(optional)* | `shared/kinds.py` | generic data *shapes* | `TimeSeries`/`Dimensioned` protocols; generic `Resample`, `Window` |
| **L2 Domain library** | e.g. `gw/` | one physics domain | `Waveform`, `Trajectory`, `StrainSeries`, `Units`, `SourceFrame`, GW nodes |
| **L3 Application** | a config/script | one problem | the specific graph you wire |

Dependencies point **downward only**: L2 imports L1 and L0; L0 imports neither.

---

## 3. Layer 0 — the core spine (domain-agnostic)

Everything here is free of physics. This is the reusable framework.

### 3.1 Process — generalizes `System`

Declares *typed input/output ports*, splitting upstream data from static config.
This is the single most important change: it separates "knobs I'm configured
with" (constructor args) from "data I receive" (ports).

```python
class Process(ABC):
    inputs:  dict[str, type]   # port name -> expected product type ({} for a source)
    outputs: dict[str, type]   # port name -> produced product type

    def validate(self, ctx: "Context", **inputs) -> None:
        """Raise if config or received inputs are unfit. Generalizes
        System.validate: from 'params present' to 'inputs present AND typed'."""

    @abstractmethod
    def run(self, ctx: "Context", **inputs) -> dict[str, Any]:
        """Consume typed inputs, return {output_port: product}. Runs to
        completion. This is the `Batchable` capability (§6): the only execution
        verb the core ships now. Stepping / event-driven verbs are additional
        capabilities a node opts into later — they do NOT change ports or types."""
```

Source / transform / sink are not new base classes — just shapes of
`inputs`/`outputs`: a **source** has `inputs = {}`, a **sink** has
`outputs = {}`, a **transform** has both. Visualization is a sink over any
product; it is no longer a method welded to one System.

**The two halves of a Process (see §6).** A Process splits along a line that
scenario-testing (§6.1) made sharp:

- **Universal half — `inputs`/`outputs`/config/`validate`.** Identical no matter
  how the node is driven. This is the part every scheduler agrees on.
- **Execution half — the *capability*** (`run` today; `step`, event-handlers
  later). This is the *only* part a scheduler binds to.

Ports, product types, and `validate` never depend on the scheduler; the verb
does.

### 3.2 Context — only what *every* simulation needs

```python
@dataclass
class Context:
    rng: np.random.Generator     # seeded -> reproducible (any stochastic sim)
    workdir: Path                # where sinks write artifacts
    logger: logging.Logger
    # NOTHING about units, time, or frames. Those are domain concerns (L2).
```

A domain that needs a shared "world" (e.g. a target unit system) **subclasses**
this — `GWContext(Context)` adds `target_units` — and its nodes require the
subclass. Core code only ever touches the base fields, so core stays clean.

### 3.3 Graph — replaces `run_from_config`'s flat loop

```python
g = Graph()
g.add("a", SomeSource(...))
g.add("b", SomeTransform(...))
g.connect("a.out", "b.in")   # structural type check at WIRING time
g.run(ctx)                   # topological sort, pass products along edges
```

The only compatibility check the core performs is **structural**:
`issubclass(produced_type, expected_type)` (or Protocol satisfaction). It does
NOT know that "geometric" and "SI" are incompatible — that mismatch is caught by
a domain node refusing bad input, not by the core. This is what keeps the wiring
engine universal.

`Graph` is **structure only** — nodes plus typed edges. It is allowed to contain
cycles; whether a cycle is *legal to execute* is a property of the scheduler
(§6), not of the structure. This yields a clean symmetry:

> **`Graph` checks port *types* at wiring time. The scheduler checks node
> *capabilities* at run time.** Same refusal mechanism, two orthogonal axes.

### 3.4 DataProduct

Just a marker/base for "a value that flows on an edge" — likely nothing more
than "any frozen dataclass." The core does not care what fields it has.

---

## 4. Layer 1 — shared kinds (optional, cross-domain)

Domain-neutral *shapes* that many physics (and non-physics) domains share, so
that generic transforms are written once and reused everywhere. Protocols, not
inheritance:

```python
class TimeSeries(Protocol):      # a GW waveform, a temperature series, a stock price...
    t: np.ndarray
    fs: float

class Dimensioned(Protocol):     # anything carrying a unit system
    units: "Units"
```

A generic `Resample` node written against `TimeSeries` then works on **any**
time series — a GW `Waveform`, a climate series, market data — none of which
import each other. That is where cross-domain reuse actually cashes out. A
domain opts in by having its data products satisfy the protocol.

---

## 5. Layer 2 — the GW domain library (the first plug-in, not the framework)

This is where every physics noun lives. It is one example domain; a completely
different domain would replace this wholesale without touching L0/L1.

```python
@dataclass(frozen=True)
class Units:
    system: Literal["geometric", "SI"]
    total_mass_msun: float | None = None   # sets GM/c^3 geometric->SI time scale

@dataclass(frozen=True)
class SourceFrame:
    theta_obs: float; phi_obs: float       # observer direction (open default question)
    distance: float; psi: float = 0.0      # luminosity distance, polarization angle

@dataclass
class Waveform:                            # implements TimeSeries + Dimensioned
    t: np.ndarray; fs: float
    h_plus: np.ndarray; h_cross: np.ndarray
    frame: SourceFrame; units: Units

# ...Trajectory, StrainSeries likewise. Plus the nodes:
#   SchwarzschildInspiral (source), GeometricToSI, Resample (reused from L1),
#   ProjectOntoDetector (antenna pattern F+/Fx), AddColoredNoise (PSD), MatchedFilter.
```

Design rule shared with L1: **time base and units travel WITH the product**,
never as a side channel (today `fs = 1/(t[1]-t[0])` is recomputed ad hoc).

### The LIGO injection, as an L2 graph

```
SchwarzschildInspiral  source     -> Waveform (geometric, coord time)
GeometricToSI          transform  pick total mass; geometric time -> seconds
Resample               transform  (L1, generic) -> Waveform at detector fs
ProjectOntoDetector    transform  antenna pattern F+, Fx(sky, psi, t) -> StrainSeries
AddColoredNoise        transform  PSD-colored realization from ctx.rng
MatchedFilter          sink       recover SNR time series / report
```

Every arrow is a typed edge. Every unit/time reconciliation is a *node*, not
core magic. New physics required (all greenfield, none in repo today):
`GeometricToSI`, `ProjectOntoDetector`, `AddColoredNoise`, `MatchedFilter`.

---

## 6. Orchestration: structure vs. scheduler

Split the two things a naive `Graph.run()` conflates:

- **`Graph` = structure.** Nodes + typed edges. May contain cycles. Knows only
  structural type-checking (§3.3).
- **`Scheduler`/`Driver` = semantics.** *How* the structure executes. The core
  ships exactly one; others slot in later without touching `Process`, ports, or
  types.

### 6.1 Why one scheduler is not enough — scenario stress-test

The batch-DAG model looked sufficient only because the motivating GW injection
is unusually **feed-forward** (one waveform, one detector, no feedback, fixed
sampling). That is selection bias. Pressure-testing against two unrelated
scenarios exposed it:

- **MD + DFT + photocatalysis.** DFT is an inner **SCF fixed-point**; BOMD calls
  a force provider **every timestep** (positions↔forces is a *cycle*);
  photon absorption and surface hopping are **stochastic events that branch
  trajectories at runtime**.
- **Digital twin of a capture→SDLC→prod pipeline.** Discrete-event, stateful
  queueing network: work-items arrive on **irregular event times**, contend for
  **finite resources** (engineers, CI runners), and loop backward (review/CI
  fail → dev, prod incident → hotfix). A live twin source **never completes**.

Both, from completely different domains, break the *same* piece — orchestration
— in the same three ways: **cycles are the norm**, **there are ≥3 scheduling
semantics**, and **coupling/topology is richer than static data-on-an-edge**.
Everything that *held* in both scenarios was the universal half of a Process
(ports/types/config/`validate`/reconciliation-as-a-node); everything that
*broke* was the execution/orchestration half. That is the empirical basis for
the split above.

### 6.2 Three scheduling semantics; capability-typed nodes

The three schedulers do not share one verb — they ask nodes for *different
methods*. Model these as **execution capabilities** a node opts into; each
scheduler requires the capability it drives.

| Capability | Verb | Who owns the clock | State | Driven by |
|---|---|---|---|---|
| `Batchable` *(ships now)* | `run(ctx, **in)` | scheduler (trivially) | optional | batch-topological |
| `Steppable` | `step(ctx, dt, **in)` + `init/reset` | scheduler | required | fixed-step co-sim (SCF/BOMD) |
| `EventDriven` | `next_event_time()` + `handle_event(...)` | **the node** | required | discrete-event (twin) |

The old framing — "`run` is the batch special-case of a future `step(dt)`" — was
half right and is now corrected:

- **True:** `Steppable ⊃ Batchable` (a stepper driven to completion subsumes a
  one-shot run). There is a batch ⊂ step ladder.
- **False:** that the ladder covers everything. `EventDriven` is **orthogonal** —
  irregular time, node owns the clock. It sits *beside* the ladder, not on it.
  Continuous-time physics never forced this, which is why the first draft missed
  discrete-event entirely.

### 6.3 Decision

- **Ship now:** `Batchable` + the batch-topological scheduler only. (Same scope
  as before — no new machinery built.)
- **Two cheap moves to avoid painting into a corner:** (1) do not name/shape the
  base verb as if `run` were the privileged special case of `step`; (2) reserve
  the capability seam so `Steppable`/`EventDriven` slot in without touching
  ports, product types, or `validate`.
- **Do NOT** build the stepper, the event scheduler, or a coupling algorithm
  until a feedback-coupled or event-driven problem actually needs one.
- **Invariant to enforce (a test, not a hope):** a node that advertises multiple
  capabilities must be **observationally identical** across them — `run` must
  equal step-to-completion. This blocks the "hidden behavior depending on who
  drives me" failure mode, the same disease the redesign exists to cure (§8).
- **Granularity is a domain choice, not core magic.** Whether `positions` is an
  instant (`X@t`) or a whole trajectory (`Series[X]`) depends on the domain;
  bridge the two with an L1 `Aggregate` sink and a `Stream[X]` vs `X`
  distinction — consistent with "reconciliation is a node," no core change.

**Sharp tension, recorded:** the tempting escape hatch — collapse every feedback
loop into one monolithic node — re-buries exactly the seams the redesign exists
to expose (§8). SCF-inside-DFT is a legitimate internal detail; BOMD-as-one-node
is not. This is the argument *for* generalizing the scheduler and *against* the
monolith.

---

## 7. The reuse guarantee, made mechanical

Layering is only real if something enforces it. The discipline:

1. **No-leak rule:** `shared/core.py` (L0) imports nothing from any domain
   package and mentions no physics noun. A one-line import-linter test asserts
   this.
2. **Canary graph:** the core test suite builds a *non-physics* graph on L0 —
   e.g. `LoadCSV (source) -> FeatureEngineer (transform) -> LogisticRegression
   (transform) -> Report (sink)`, which maps onto the campaign-finance /
   win-prediction projects in this workspace. It exercises source/transform/
   sink, config-vs-port separation, `ctx.rng` (train/test seed), and
   `ctx.workdir` (model artifact) — with zero physics imports. If it can't be
   written cleanly, the core is contaminated.

Together these turn "I want to reuse this for a different system" from a hope
into a passing test.

---

## 8. Migration path — nothing breaks

- `System` becomes a **degenerate Process** (`inputs = {}`, one output, old
  `visualize()` attached as a sink). The four registered geodesic/inspiral
  Systems keep running unchanged.
- `run_from_config` becomes "build a graph." A config with no `connect:` section
  = today's independent single-node runs.
- First extraction target: split the inspiral's hidden pipeline into explicit
  `InspiralTrajectory` (→ `Trajectory`), `QuadrupoleStrain` (`Trajectory` →
  `Waveform`), and strain/spectrum plotter sinks. Low-risk: it already *is* one
  pipeline; the change is making the seams visible.
- Retire the `sys.path.insert` import hacks for a real package while doing this.

---

## 9. Open questions / deferred

- **Observer direction default** (`theta_obs`/`phi_obs`) — still unresolved
  (separate whatidid page); now lives on L2's `SourceFrame`, question unchanged.
- **Where L1 stops:** is a `TimeSeries` protocol worth it now, or premature
  until a second time-series domain exists? Leaning: define it when the LIGO
  demo forces `Resample`, so it has two consumers (waveform + detector) from
  day one.
- **Time-grid reconciliation** — explicit `Resample` node (leaning) vs. an
  implicit context policy. Explicit keeps L0 dumb, which serves the reuse goal.
- **Serialization** — PNG-only today; typed products invite an HDF5/npz
  `Save`/`Load` sink (L1, generic). Deferred.
- **Coupling algorithm** — Jacobi vs. Gauss-Seidel exchange, error control across
  coupled steppers. Deferred until a `Steppable` scheduler is built (§6).
- **Service ports / oracles vs. data edges** *(open, deferred)* — MD needs a
  callable force *oracle* queried on demand, and SDLC stages need shared resource
  pools; neither is a producer→consumer data edge. Is that a port whose type is a
  callable/Process reference, or is tight coupling always one node internally?
  Bridges to §6: a force oracle looks like a `Batchable` node invoked on-demand by
  a `Steppable` one.
- **Port-type stability across schedulers** *(open)* — is there a node whose port
  *types* genuinely cannot stay fixed between batch and stepped execution (beyond
  the `X` vs `Stream[X]` granularity bridge)? If so, the "universal half is
  scheduler-independent" claim (§6.1) needs qualification.
