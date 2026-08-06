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
        capabilities a node opts into later."""
```

Source / transform / sink are not new base classes — just shapes of
`inputs`/`outputs`: a **source** has `inputs = {}`, a **sink** has
`outputs = {}`, a **transform** has both. Visualization is a sink over any
product; it is no longer a method welded to one System.

**The two halves of a Process (see §6).** A Process splits along a line that
scenario-testing (§6.1) made sharp:

- **Universal half — port *names*/arity/config/`validate`.** Identical no matter
  how the node is driven. This is the part every scheduler agrees on.
- **Execution half — the *capability*** (`run` today; `step`, event-handlers
  later). This is the part a scheduler binds to.

**Port types sit on the line, not cleanly on the universal side.** An earlier
draft claimed ports, product types, and `validate` never depend on the
scheduler. That is over-claimed, and §6.4 gives the counterexamples: a driver can
change the *representation* of what flows on a wire, not merely how it is chunked
in time. So the accurate statement is:

> A port declares a **type schema**, which is scheduler-independent. The
> **ground type** is instantiated by the scheduler at bind time.

Concretely, a port is `Signal[F]`, not `Signal[Real]`; a transient driver
instantiates `F = Series[Real]`, a frequency-domain driver `F = Complex`. The
schema — arity, shape, which output feeds which input — is fixed and checkable
the moment you wire the graph. The ground type is not knowable until a scheduler
is chosen, because in some cases (§6.4, family 3) it depends on the scheduler's
*tuning*, not just its identity.

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
(§6), not of the structure.

**Type-checking is two-phase**, mirroring the capability check:

> **Wiring time (`Graph`):** the *type schema* — arity, shape, and
> schema-compatibility of each edge. Catches "you connected a waveform port to a
> scalar port."
>
> **Bind time (scheduler):** the *ground type*, instantiated for this driver,
> plus node *capabilities*. Catches "this driver needs complex phasors and that
> node can only emit real samples."

Both checks are the same refusal mechanism applied at the two moments where the
required information actually exists. Wiring time still catches the large
majority of real mistakes — you do not need to know the scheduler to know a
mis-wire is a mis-wire — but it is genuinely weaker than "all types verified
before you run." §6.4 explains why that concession is forced.

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
  the capability seam so `Steppable`/`EventDriven` slot in without touching port
  *names*, arity, config, or `validate`. **Note the narrowing:** an earlier draft
  said "without touching ports, product types, or `validate`." Product types are
  *not* covered by that promise — see §6.4.
- **Do NOT** build the stepper, the event scheduler, or a coupling algorithm
  until a feedback-coupled or event-driven problem actually needs one.
- **Invariant to enforce (a test, not a hope):** a node that advertises multiple
  capabilities must be **observationally identical** across them — `run` must
  equal step-to-completion. This blocks the "hidden behavior depending on who
  drives me" failure mode, the same disease the redesign exists to cure (§8).
  **Narrowed by implementation:** this holds only *within* one representation.
  Across representations the two outputs are not comparable values at all
  (`TokenFlow` vs `RateFlow`), so the strongest available statement is
  agreement on a *derived scalar*, chosen by the domain and unenforceable by
  the core. See `FINDINGS.md` F3.
- **Granularity is a domain choice, not core magic.** Whether `positions` is an
  instant (`X@t`) or a whole trajectory (`Series[X]`) depends on the domain;
  bridge the two with an L1 `Aggregate` sink and a `Stream[X]` vs `X`
  distinction — consistent with "reconciliation is a node," no core change.

**Sharp tension, recorded:** the tempting escape hatch — collapse every feedback
loop into one monolithic node — re-buries exactly the seams the redesign exists
to expose (§8). SCF-inside-DFT is a legitimate internal detail; BOMD-as-one-node
is not. This is the argument *for* generalizing the scheduler and *against* the
monolith.

### 6.4 Port types are NOT scheduler-independent — the counterexample

§6.2 assumed the only way a driver can change what a port carries is
*granularity* — the whole series (`Series[X]`) versus one instant (`X`) — bridged
by an L1 `Aggregate` sink. That assumption is false. Hunting deliberately for a
node that (i) legitimately supports more than one driver and (ii) needs different
port types under each turned one up immediately.

**The counterexample: SPICE `.tran` vs `.ac`.** A netlist is the canonical
multi-driver artifact — one structure, run under `.op` (DC operating point),
`.tran` (transient), `.ac` (small-signal frequency response), `.noise`. That is
exactly this design's Graph/Scheduler split, and it is the most standard
multi-driver workflow in engineering simulation.

Take a MOSFET model. The universal half looks intact: same config (W, L, model
card), same four ports (drain/gate/source/bulk), same `validate`. It is not a
`Batchable`-only node ducking the question — it genuinely supports every
analysis. But the wire types change with the driver:

| Driver | What a wire carries |
|---|---|
| `.op` | `Real` — a scalar node voltage |
| `.tran` | `Series[Real]` — time-domain samples |
| `.ac` | `Complex` — a phasor at ω, linearized about the operating point |

`.ac` breaks the claim. A phasor is not a chunk of a waveform and not an instant
of a stream; it is a different mathematical object in a different space, obtained
by linearizing and moving to the frequency domain. No `Aggregate` recovers it.
SPICE concedes the point internally: devices implement `load` *and a separate*
`acLoad` assembling into a **complex** matrix. Worse, `.ac` requires an input
`.tran` does not — the DC operating point it linearizes about — so the driver
changes the port *set*, not only the types.

**Three families of the same disease.** Drivers that change the *representation*
of the quantity on the wire, not just its chunking in time:

1. **Time-domain ↔ frequency-domain linearization.** SPICE `.ac`. Also inside the
   MD/DFT scenario of §6.1: ground-state SCF carries a density, while
   density-functional perturbation theory / TDDFT carries response functions
   χ(ω). Same electronic-structure node, different object on the wire.
2. **Token ↔ distribution.** Inside the digital-twin scenario of §6.1: driven
   discrete-event, a wire carries a `WorkItem`; driven by a steady-state analytic
   queueing model, the same wire carries `Distribution[ServiceTime]` (λ, μ,
   utilization). Both are legitimate ways to run the same model. Same split as
   PIC `ParticleList` vs. Vlasov `f(x,v)`, and Lagrangian vs. Eulerian fluids.
3. **Value ↔ value-plus-combining-algebra.** In FMI co-simulation with unequal
   macro-steps a consumer cannot use the producer's value *at t* — it must
   extrapolate across the step, so the producer ships value **plus k
   derivatives** (FMI has `fmi2GetRealOutputDerivatives` for precisely this). In
   batch you never need it: you hold the whole `Series` and interpolate post hoc.
   Recurs as covariance in sequential filtering (EnKF vs. 4D-Var) and as
   sample-count-plus-variance in progressive estimation.

Family 3 stings hardest: **k is set by the scheduler's error-control policy**, so
the port type is parameterized not just by *which* scheduler but by *how it is
tuned*. No wiring-time check can know it.

The usual escape hatches all fail here. "It's a different node" — no, it is one
MOSFET model, one DFT code, one pipeline model. "It's `Batchable`-only" — no,
these support every driver. "It's just chunking" — no, in all three families.

**Resolution: two-phase type-checking** (§3.3). The port declares a polymorphic
*schema* (`Signal[F]`); the scheduler instantiates the ground type at bind time
and re-checks there, alongside the capability check. The §3.3 symmetry survives —
arguably tightened, since types and capabilities now both check in two phases at
the same two moments — but §3.1's "types never depend on the scheduler" had to be
weakened to "type *schemas* never depend on the scheduler."

**Rejected alternative:** declare that a representation change means building a
*different graph*, restricting scheduler-swapping to within one representation.
Cheaper and honest, but it gives up the SPICE workflow — one netlist, four
analyses — which is too central a use case for a framework whose entire pitch is
reuse across domains.

**Consequence for (b).** The service-port / oracle question in §9 is partly the
same disease seen from the coupling side: the wire needs to carry something
richer than a value. Track them together.

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

> **Status: the framework is now IMPLEMENTED.** `shared/core.py` is L0;
> `2_framework_pressure_test/` is L1–L3 plus 50 tests, covering a harmonic
> oscillator scenario and an SDLC pipeline scenario. Read
> `2_framework_pressure_test/FINDINGS.md` alongside this section — it confirms
> four claims made here and falsifies three more, including two entries below.
> New problems it surfaced that this section never anticipated: **ports cannot
> be optional** (F2, forced one pipeline stage to be duplicated into two
> classes), and **unconsumed outputs are silently dropped** (F4).

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
- **Service ports / oracles vs. data edges** — *analysed, shape reserved, NOT
  built. Revisit when something in the repo actually needs one.* MD needs a
  callable force *oracle* queried on demand, and SDLC stages need shared resource
  pools; neither is a producer→consumer data edge. The candidate encoding is a
  port whose type is a callable — `Oracle[Positions, Forces]` — wired like any
  other edge, with the scheduler handing the consumer something it can call.
  Judge it against the two alternatives, not in isolation: (i) collapse consumer
  and provider into one node — the monolith §6.3 rejects; (ii) a data edge plus a
  `Steppable` scheduler running fixed-point iteration — the "proper" answer,
  deferred. The oracle is the cheap middle.

  **For it.** It converts a cycle into a DAG *without* collapsing the nodes: DFT
  becomes a dependency of MD rather than a mutual peer, so the batch-topological
  scheduler can run it. It buries only the timestep loop (MD's actual job — no
  seam lost) while keeping the force-provider seam swappable (DFT → classical
  force field → ML potential, consumer untouched). That buys feedback coupling
  without building `Steppable`. For genuinely *data-dependent* access — adaptive
  integrators, root-finders, optimisers, MCTS — a data edge cannot express the
  problem at all, since the query points are discovered at runtime over an
  effectively infinite domain. And it is plainly domain-neutral, so it does not
  threaten §7: the canary graph can exercise it via a hyperparameter search
  querying a train-and-score oracle, exactly as `scipy.minimize` calls an
  objective. It also completes the §6.4 ladder — `X` → `Jet[X,k]` →
  `Series[X]` → `Oracle[D, X]`, each rung answering "where can I get the value?"

  **Against it.** The edge stays visible but *the traffic does not*: no record of
  how many calls, at what arguments, in what order — and no way to hang a sink on
  an oracle edge and plot it. That is a direct hit on intermediate visualization,
  one of the three founding complaints in §1. The scheduler also loses control it
  cannot recover: the consumer decides when the provider runs, so ordering,
  parallelism, checkpointing between calls, and cross-coupling error control all
  become unavailable — the whole coupling-algorithm question above is meaningless
  on an oracle edge. It punches a hole in the capability system: a `Batchable`
  node invoked by *another node* means "the scheduler binds capabilities" is no
  longer true, and there is a second undeclared driver. Statefulness leaks (DFT
  warm-starting SCF from the previous geometry is order-dependent), and with
  `ctx.rng` in play, call order determines draw order — reproducibility becomes
  coupling-dependent rather than seed-dependent, which is exactly the
  "behaviour depends on who drives me" disease. Typing it properly needs
  contravariant-argument / covariant-result checking, machinery §3.3 keeps out of
  L0 on purpose, and under §3.3's two-phase scheme it would mean instantiating
  ground types *inside* a higher-order type. Finally, callables do not serialize,
  so the deferred Save/Load sink cannot cross an oracle edge.

  **Leaning, when this is revisited.** The discriminator is whether the query
  pattern is **data-dependent**. Known-up-front or finite-and-enumerable → use a
  data edge; nothing real is given up and provenance, visualization,
  checkpointing and scheduler control are all retained. Query points discovered
  at runtime → the oracle is honest rather than lazy. Two mitigations worth
  keeping with the idea: (1) hand over a **mediated** callable — the scheduler
  wraps the provider so calls are logged, counted and cached, which restores
  provenance and makes the call log itself a plottable product, killing the
  largest drawback; (2) require an oracle port to **declare purity**, so a
  stateful oracle is legal but cannot hide.

  **Why not now:** the LIGO demo is entirely feed-forward with no oracle
  anywhere, and only `Batchable` ships. Same posture as `Steppable` and the
  bind-time type check — reserve the shape, build nothing.

  > ⚠ **CONTRADICTED BY THE IMPLEMENTATION.** The "why not now" above did not
  > survive first contact. `2_framework_pressure_test/` scenario 1 — a *driven
  > harmonic oscillator*, the simplest forced system there is — needs an oracle
  > edge. Its adaptive ODE solver evaluates the RHS at times it discovers at
  > runtime, but a data edge can only deliver the force on the producer's grid,
  > so accuracy is capped by an upstream choice the consumer cannot see:
  > measured relative error moves from 3.7e-1 to 2.8e-5 purely by refining the
  > *producer's* grid, with the solver pinned at `rtol=1e-10`. That is exactly
  > the data-dependent discriminator this entry wrote down. See
  > `FINDINGS.md` F1, which recommends building the oracle *ahead of*
  > `Steppable`. Note also the recurring failure mode: this is the third time
  > the feed-forward LIGO example has hidden a requirement (cf. §6.1, §6.4).
- **Port-type stability across schedulers** — ~~open~~ **RESOLVED, and the claim
  was falsified.** A counterexample exists (SPICE `.tran` vs `.ac`, plus two more
  families); see §6.4. The universal half now covers port *names*/arity/config/
  `validate` but **not** ground port types, and type-checking became two-phase
  (§3.3). Follow-on questions this opens:
  - How much of the schema/ground-type split does L0 need *now*, given only the
    batch driver ships? Leaning: state the two-phase rule in the design, but
    implement only the wiring-time check until a second driver exists —
    a single-driver framework cannot exercise the bind-time half.
  - Is a node allowed to support a driver whose ground type it cannot produce
    (the MOSFET that has no `acLoad`)? Probably yes, and it is refused at bind
    time like a missing capability — but that makes "advertises a capability"
    and "can satisfy this driver's types" two separate predicates.
  - Does the §6.3 observational-equivalence invariant (`run` == step-to-
    completion) even typecheck when the two drivers use different ground types?
    It may only be statable *within* a representation.
