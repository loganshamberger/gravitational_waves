# Pressure-test findings — composable framework, first implementation

Built `shared/core.py` (L0) against `docs/composable_simulation_framework.md`,
then stress-tested it with two deliberately dissimilar scenarios:

- **Scenario 1 — harmonic oscillator, free and driven.** Continuous-time
  physics, feed-forward, with the driving force wired in as a graph edge.
- **Scenario 2 — SDLC backlog→deployment.** Non-physics, discrete, feedback-
  shaped. Doubles as a second reuse canary.
- **Scenario 3 — two oscillators coupled through the graph.** Scenario 1's free
  oscillator wired into its driven oscillator. Added last, and it produced the
  most serious finding.

65 tests pass. The design mostly held. Four findings below are things the
design got right and can now stop hedging about; five are places it is wrong or
incomplete — **F5 is severe** (the shipped scheduler yields divergent physics on
two coupled pendulums), and F1 is urgent.

---

## What held up

### H1. Two-phase type-checking works, and the second phase earns its keep

The §6.4 claim — one structure, two drivers, two representations on the wires —
is now executable rather than argued. `AnalyticQueueScheduler` (in
`scenario_sdlc.py`) drives the *same graph object* as `BatchScheduler`, grounds
every wire to `Rates` instead of `Tokens`, and produces a steady-state answer
where the batch driver produces simulated tokens.

Both drivers agree on the deployed fraction to within 5%
(`test_the_two_drivers_agree_on_the_deployed_fraction`), so this is not two
unrelated computations wearing the same wiring.

The check is load-bearing in both directions:
`test_same_graph_accepted_by_one_driver_and_refused_by_another` builds one
graph that passes phase 1, then passes under a driver binding `F→Alpha` and
raises `GroundTypeError` under one binding `F→Beta`.

### H2. A new driver needed no core change

`AnalyticQueueScheduler` and its `Analytic` capability are defined entirely in
`scenario_sdlc.py`, against the public core API, with a verb (`analyze`) the
core has never heard of. `test_the_analytic_driver_needed_no_core_change`
asserts the core does not know either name. This was the central bet of the
Graph/Scheduler split, and it paid.

### H3. Cycles-in-structure / rejection-by-scheduler is real, and the error is honest

`build_rework_graph` wires `CodeReview.rejected → Development.rework`. Wiring
**succeeds** — the graph holds the cycle. `BatchScheduler` then refuses with a
message that names the offending nodes and explicitly says *"The graph itself
is legal — a cycle-tolerant scheduler would accept it."* A future scheduler
that handles feedback is therefore a pure addition.

### H4. The no-leak test caught a real leak on its first run

Not hypothetically — `run_from_config`'s docstring contained a Schwarzschild
YAML example, i.e. physics sitting in L0. The test failed, the docstring was
generalised, the test passes. The mechanical guarantee did the exact job §7
claimed for it, immediately.

*(Small pitfall worth recording: the first version of the check matched
substrings and flagged `metric` inside `Parametric`. Word boundaries required.)*

---

## What broke

### F1 — URGENT. The simplest possible physics problem needs an oracle edge

**This is the significant finding.** §9 concluded "oracles: reserve the shape,
build nothing — the LIGO demo is feed-forward and nothing needs one." Scenario 1
is a *spring*, and it needs one.

`DrivingForce → DrivenOscillator` is a data edge, so the force arrives sampled
on a grid the producer chose. But the consumer is an adaptive ODE solver: it
evaluates the RHS at times *it* picks, discovered at runtime. So
`DrivenOscillator.run` interpolates (`np.interp`), and its accuracy is capped by
a decision made upstream by a node that cannot know what it needs.

Measured, with the solver held at `rtol=1e-10` and only the *producer's* grid
varying:

| producer grid | Δt | rel. error in steady-state amplitude |
|---|---|---|
| n=101 | 4.0 | 3.7 × 10⁻¹ |
| n=401 | 1.0 | 1.3 × 10⁻¹ |
| n=2001 | 0.2 | 5.6 × 10⁻³ |
| n=40001 | 0.01 | 2.8 × 10⁻⁵ |

Four orders of magnitude, entirely from the producer's grid. The consumer's own
tolerance is irrelevant. (`test_solver_accuracy_is_capped_by_the_producers_grid`)

This lands *exactly* on the discriminator §9 wrote down — "query points
discovered at runtime → the oracle is honest" — which is reassuring for the
analysis and damning for the scheduling. **Note the pattern: this is the third
time the motivating example's feed-forward shape has hidden a problem.** GW
injection hid the need for non-batch schedulers (§6.1); GW injection hid the
`.ac`-style representation change (§6.4); now it hid this. Deferring "until
something needs it" keeps mis-firing because the LIGO demo is a systematically
unrepresentative probe.

**Recommendation:** promote the oracle from "reserved shape" to the next thing
built, ahead of `Steppable`. The `Oracle[D, X]` port with the §9 mitigation
(scheduler hands over a *mediated* callable that logs/counts/caches) would let
`DrivenOscillator` ask for `F(t)` at its own quadrature points, and the call log
would keep the edge visible.

### F2. Ports cannot be optional, so one stage became two classes

`Development` declares `inputs = {ready, rework}`. Unwired inputs are a bind-time
error, so that node **cannot be used in the linear topology at all** — there is
no rework producer. Supporting both topologies forced a duplicate class,
`DevelopmentWithInternalRework`, differing only in whether the loop is a port or
a `while` loop.

That is a real modelling failure: the two classes are the same stage, and a user
choosing between topologies must now maintain two implementations that can drift.
The design has no notion of an optional port, a default value for an unwired
input, or a port that only exists under some topologies. Neither §3.1 nor §3.3
mentions this.

### F3. The observational-equivalence invariant does not survive across representations

§6.3 requires that a node advertising multiple capabilities be observationally
identical across them, and §9 flagged the worry that this may not typecheck
across representations. Confirmed: it does not.

`TokenFlow` and `RateFlow` are not comparable values. The strongest statement
available was a *derived scalar* — the deployed fraction — checked
statistically at 5% tolerance
(`test_the_two_drivers_agree_on_the_deployed_fraction`). There is no way to
write "`run` == `analyze`" as an equality.

**The invariant must be restated:** observational equivalence holds *within* a
representation. Across representations, the most you get is an agreed derived
quantity, and choosing that quantity is a domain judgement the core cannot make
or enforce. §6.3's "a test, not a hope" is therefore weaker than advertised —
it is a test only inside a representation.

### F5 — SEVERE. Refusing the cycle does not cost accuracy; it changes the answer qualitatively

Scenario 3 couples the free oscillator to the driven one through the graph. The
physics is two masses joined by a spring:

    m1 x1'' = -(k1 + kc) x1 + kc x2
    m2 x2'' = -(k2 + kc) x2 + kc x1

Each mass feels the other, so **the honest dataflow graph is a cycle.** Three
encodings exist, and the framework can run exactly the wrong two:

| Graph | Exact? | Coupling visible? | Runs? |
|---|---|---|---|
| `build_cascade_graph` (one-way) | no | **yes** | yes |
| `build_coupled_graph` (bidirectional) | **yes** | **yes** | **no — CycleError** |
| `build_monolith_graph` (one node) | **yes** | no | yes |

The one encoding that is both exact *and* transparent is the one refused.

F1 was about accuracy. This is worse. For two **identical** undamped
oscillators the drive frequency equals mass 2's natural frequency exactly
(`√((k+kc)/m)` on both sides), so one-way coupling drives mass 2 on **undamped
resonance** and its amplitude grows without bound:

| integration window | cascade max‖x₂‖ | exact max‖x₂‖ |
|---|---|---|
| t ≤ 60 | 2.880 | 0.999 |
| t ≤ 120 | 5.797 | 0.999 |
| t ≤ 240 | 11.705 | 1.000 |
| t ≤ 480 | 23.369 | 1.000 |

Amplitude doubles as the window doubles — linear secular growth, diverging.
The true solution is bounded: energy sloshes between the masses and comes back.
**The back-reaction — exactly the term the cycle would have carried — is what
prevents the runaway.** (`test_cascade_diverges_secularly_for_identical_masses`)

Error vs. coupling strength, at t ≤ 60:

| kc | kc/k | rel. error in x₂ |
|---|---|---|
| 0.002 | 0.0005 | 0.0% |
| 0.02 | 0.005 | 1.4% |
| 0.2 | 0.05 | 262% |
| 1.0 | 0.25 | 1278% |

One-way coupling is a legitimate physical limit — as kc → 0 the cascade
converges on the truth (3.3e-6 absolute at kc=0.002), and the monolith matches
the closed-form beat solution to 1.3e-11. So none of this is a coding error.
It is the framework's scheduler forcing a modelling choice that is *silently
catastrophic* in the symmetric case, with nothing in the API hinting that the
approximation has a validity range.

**Why this outranks F1.** F1 said a deferred feature would cost precision. F5
says the shipped scheduler, on the most standard problem in undergraduate
physics — two coupled pendulums — will hand you a divergent answer if you take
the only runnable transparent route. And "collapse it into one node" is not a
neutral fallback: `test_the_monolith_hides_the_coupling_force` shows the
coupling force has no port at all in that encoding, so the term whose omission
caused the divergence is also the term you can no longer inspect.

**Recommendation.** This makes a cycle-tolerant scheduler (fixed-point /
`Steppable`) considerably more urgent than §6.3's "wait until a feedback-coupled
problem needs one" implies — the second scenario anyone writes is such a
problem. Two coupled oscillators is a ready-made acceptance test with a
closed-form answer. Note this *reverses* the priority ordering suggested below
before scenario 3 existed: F1's oracle and F5's scheduler are now the top two
items, and F5 arguably comes first because it produces wrong physics rather
than imprecise physics.

### F4. Unconsumed outputs are silently dropped

`CI.failing` is produced by every run and wired to nothing. Nothing complains.
For a pipeline whose whole subject is where work gets stuck, silently discarding
the failure stream is exactly the wrong default. Inputs must be wired; outputs
need not be, and the asymmetry is undocumented and unargued. At minimum this
deserves an opt-in strictness flag on the scheduler.

---

---

## Scenario 4 — the motivating problem, finally run

The GW injection chain from §5 (`ChirpSource → GeometricToSI →
ProjectOntoDetector → Resample → Window → AddColoredNoise → MatchedFilter →
Report`) is built in `scenario_gw_injection.py` at deliberately low fidelity —
see the fidelity warning at the top of that file. It is a toy: leading-order
Newtonian chirp, a stellar-mass binary rather than an EMRI, an analytic
stand-in PSD, one detector, exact-template filtering. **None of its physics
should be quoted.**

What it does establish:

- **It runs on the shipped batch scheduler, unmodified.** Prediction confirmed:
  the injection chain is feed-forward, single-representation and acyclic, so
  none of F1/F5 bite. Notably this is *why* it was such a poor probe — it
  exercises the one shape the framework already handled.
- **Units-on-the-wire works, and the enforcement lands in the right place.**
  `Waveform` carries `units`; `ProjectOntoDetector` refuses geometric input
  while the core's type check *passes* (both ports are `Waveform`). That is
  §3.3's "a domain node refuses bad input, not the core" doing exactly what was
  designed. Reconciliation is a node (`GeometricToSI`) and the core stayed dumb.
- **L1 finally earned its keep.** `Resample` now has two genuine consumers
  (oscillator and injection), which is precisely the condition §4 set for
  defining it. Waiting was the right call.
- **Fan-out composes cleanly.** The template is the clean strain tapped off
  before noise is added — one output feeding two inputs, no special support
  needed.
- Sanity scalings all hold: h ≈ 1.2e-21 for 30+30 M☉ at 400 Mpc, SNR ∝ 1/D and
  ∝ 1/noise to within 2%, edge-on quieter than face-on, monotonically rising
  chirp frequency, antenna nulls where they should be.

### Later additions to scenario 4

- **Random injection time.** `InjectIntoSegment` places the chirp at a
  seed-reproducible random time inside an 8 s segment and emits an
  `InjectionTruth` product alongside the data. The matched filter now has to
  *search* rather than confirm a known zero lag, and `InjectionReport` scores
  the recovery against truth — two inputs from two upstream nodes that know
  nothing about each other. Recovery is sub-millisecond (≤ 2 samples at 4096 Hz)
  across seeds at SNR ≈ 23–25, and degrades to a noise peak at 8000 Mpc, which
  is the control.
- **A layering consequence worth recording.** `ChirpSource` cannot be told
  "start at 20 Hz": it works in geometric units and does not know the mass,
  which lives downstream in `GeometricToSI` because reconciliation is a node.
  Only the L3 graph builder knows both, so the Hz→geometric conversion happens
  there. A corollary caught the first attempt at widening the chirp: the sweep
  *ratio* is fixed by the geometric start frequency against the geometric
  ISCO, and **both are mass-independent** — changing the mass slides the band
  in Hz without widening it. The fix was to start lower, not lighter.

### F6 (minor). Cross-node validity conditions are inexpressible

Every stage can be individually correct and the composition still meaningless.
In an antenna null the graph's template and its data are *both* ≈ 0, so the
matched filter computes 0/0 and returns a finite, entirely spurious SNR of ~3.2.
No node is wrong. The framework has no way to state "this output is only
meaningful if that other quantity is non-negligible," because `validate` sees
only one node's own inputs.

Recorded as a documented trap
(`test_matched_filter_snr_is_meaningless_for_a_null_template`) rather than
patched, since the real fix is a template bank, not a framework feature. But it
is worth knowing that graph-level type safety buys nothing against this class of
error.

*(Also caught during the build: `AddColoredNoise` initially reported the
unscaled PSD while scaling the noise it added, which made SNR independent of
the actual noise level. Having the PSD be an explicit output **port** rather
than an implicit shared assumption is what made the inconsistency visible.)*

## Layering scorecard

| Claim | Verdict |
|---|---|
| L0 contains no physics | Holds — enforced by test, caught one real leak |
| L0 needs no change to add a driver | Holds — `AnalyticQueueScheduler` is proof |
| Graph = structure, scheduler = semantics | Holds — cycle case demonstrates it |
| Port *schemas* are scheduler-independent | Holds |
| Port *ground types* are not | Holds — this is why phase 2 exists |
| `System` still works untouched | Holds — `SystemProcess` adapter, legacy API intact |
| Reconciliation is a node, never core | Holds — nothing in L0 interpolates or converts |
| Oracles can wait | **False (F1)** |
| Ports are fully universal | **False (F2)** — no optional ports |
| Observational equivalence is testable | **Partly false (F3)** — within one representation only |
| Batch-only is a safe place to start | **False (F5)** — divergent physics on coupled oscillators |
| The monolith is an acceptable fallback | **False (F5)** — it hides the term that caused the divergence |

---

## Suggested next steps, in priority order

**Revised after scenario 3.** The pre-scenario-3 list ended with "only then
consider `Steppable`; nothing needed it." Scenario 3 needed it, and produced
divergent physics without it. Corrected ordering:

1. **Build a cycle-tolerant scheduler** (F5). Two coupled oscillators is the
   acceptance test and it has a closed-form answer. This is first because it is
   the only finding that yields *wrong* results rather than imprecise ones.
2. **Build the `Oracle` port** with the mediated-callable mitigation (F1). The
   driven oscillator is a ready-made acceptance test with an analytic answer.
3. **Decide optional ports** (F2) — a default, an `Optional[...]` schema, or an
   explicit "this node has topology variants" concept. Note scenario 3 hit this
   too: `build_coupled_graph` needs both oscillators to accept a force port,
   while `build_cascade_graph` needs one of them not to.
4. **Restate the §6.3 invariant** as within-representation (F3).
5. **Decide the unconsumed-output policy** (F4).

**Meta-observation across all three scenarios.** Every deferral this design made
on the grounds of "no current problem needs it" was falsified by the next
scenario written — schedulers (§6.1), representation change (§6.4), oracles
(F1), and now cycle tolerance (F5). The common cause is that the motivating GW
injection is feed-forward, single-representation, and acyclic, which makes it a
systematically unrepresentative probe. Future "defer until needed" calls should
be tested against a deliberately dissimilar scenario *before* being recorded,
not after.
