# Framework pressure test

Implements and stress-tests the composable simulation framework designed in
[`docs/composable_simulation_framework.md`](../docs/composable_simulation_framework.md).

The framework itself lives in [`shared/core.py`](../shared/core.py) (Layer 0).
This folder is Layers 1–3: shared kinds, four domain scenarios, and the tests
that try to break them.

## Read this first

**[FINDINGS.md](FINDINGS.md)** — what held up and what broke. Four design claims
confirmed, five problems found. **F5 is severe:** two coupled oscillators is a
cycle, the batch scheduler refuses it, and the only runnable transparent
encoding diverges — amplitude grows without bound where the true solution stays
bounded. F1 is urgent: even a driven oscillator needs the callable/oracle port
the design had deferred indefinitely.

## Layout

| File | Layer | What it is |
|---|---|---|
| `kinds.py` | L1 | `Signal`, `Table` — domain-neutral products with two consumers |
| `l1_nodes.py` | L1 | Generic `Resample`, `Window` — written once, used by two scenarios |
| `scenario_oscillator.py` | L2/L3 | Free + driven harmonic oscillator; force wired in as an edge |
| `scenario_sdlc.py` | L2/L3 | Backlog→deployment pipeline, plus a second scheduler |
| `scenario_coupled_oscillators.py` | L2/L3 | Two oscillators coupled through the graph — cascade, cyclic, and monolith encodings |
| `scenario_gw_injection.py` | L2/L3 | **The motivating problem:** chirp → detector → noise → matched filter. Deliberately low fidelity |
| `test_core.py` | — | L0: two-phase typing, cycles, capabilities |
| `test_reuse_guarantee.py` | — | §7: no-leak rule + non-physics canary graph |
| `test_scenario_oscillator.py` | — | Physics validated against closed-form answers |
| `test_scenario_sdlc.py` | — | Cycle refusal, two drivers over one structure |
| `test_scenario_coupled_oscillators.py` | — | Normal modes, beats, and the secular-divergence finding |
| `test_scenario_gw_injection.py` | — | Pipeline scalings, antenna pattern, unit guards, time recovery |
| `viz.py` | L1/L3 | `PlotSignal` sink + `plot_graph` structure renderer |
| `make_figures.py` | — | Renders every scenario to `figures/` |

## Running

```bash
python -m pytest 2_framework_pressure_test/ -q          # 90 tests
python 2_framework_pressure_test/make_figures.py        # 11 figures
python -m pytest 2_framework_pressure_test/ -q -k core  # just L0
```

`conftest.py` handles the `sys.path` setup, matching the existing repo
convention. Retiring those hacks for a real package is noted in the design doc
(§8) and deliberately not done here — it would touch the existing simulator.

## What each scenario is for

**Scenario 1 (oscillator)** is continuous-time, feed-forward physics with an
analytic answer at every step, so the tests check real numbers: the undamped
period is `2π/√(k/m)` to 1e-6, energy is conserved to 1e-6, and the driven
steady-state amplitude matches `F₀/√((k−mω²)² + (cω)²)` to 2%. It exists to
answer "does the framework survive contact with actual physics, and what
happens when you make a forcing term a graph edge?"

**Scenario 2 (SDLC)** is discrete, non-physics, and feedback-shaped. It exists
to answer "is L0 really domain-agnostic, what does the framework do with a
genuine cycle, and can one structure be driven two different ways?" The rework
loop (`CodeReview.rejected → Development`) is the interesting part: the graph
accepts it, the batch scheduler refuses it, and the version that *does* run has
the loop buried inside a node — with a test measuring exactly what that costs.

## Design commitments this code encodes

Referenced by section number in `core.py`'s module docstring:

- **§3.1** Process splits into a universal half and an execution half.
- **§3.3** Type-checking is two-phase: schema at wiring, ground type at bind.
- **§6** Graph is structure (may hold cycles); Scheduler is semantics.
- **§6.4** Ports are parameterised, because a driver can change the
  representation on a wire.
- **§7** The reuse guarantee is a test, not a convention.
- **§8** `System`/`Runner`/`run_from_config` are untouched; `SystemProcess`
  adapts a legacy `System` into a graph node.
- **§9** Oracle ports are *not* built — see FINDINGS.md F1, which argues that
  was the wrong call.

## Scenario 3 in one line

The free oscillator's motion becomes the driving force for the second
oscillator, through typed graph edges — and building it revealed that the
*correct* diagram of two coupled masses is a cycle the shipped scheduler cannot
run, while the two encodings it *can* run are respectively wrong (one-way
cascade, secularly divergent) and opaque (both masses in one node).

## Figures

`make_figures.py` writes two families to `figures/`:

- **`graph_*.png`** — the wiring of each scenario. Cycle members are shaded red
  and back-edges drawn as dashed red arcs, so `graph_03` and `graph_05` show at
  a glance exactly what the batch scheduler refuses. This is something the old
  `System` catalog could not produce at all: there was no structure to draw.
- **`result_*.png`** — what each simulation produced. `result_02` is the F5
  plot: the one-way cascade's amplitude growing without bound beside the exact
  solution pinned at 1.

## Scenario 4 fidelity warning

`scenario_gw_injection.py` is a **toy**. Leading-order Newtonian chirp only (no
PN terms, spin, merger or ringdown); a 30+30 M☉ stellar-mass binary rather than
an EMRI, because real EMRIs are LISA-band and months long; an analytic
stand-in for the aLIGO PSD; one detector; and matched filtering against the
exact injected template, so recovered SNR is an optimistic upper bound with no
template bank or vetoes.

The architecture is the claim, not the astrophysics. None of its numbers should
be quoted.
