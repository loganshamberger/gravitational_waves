# Particle Orbiting a (Schwarzschild) Black Hole

Step one toward a Kerr geodesic simulator: understand and implement Schwarzschild
orbits first (the non-rotating special case), then generalize. Now extended
with gravitational-wave emission (quadrupole formula) and radiation-reaction
inspiral.

## Contents

- **`notes/gravitational_wave_quadrupole_report.md`** — research notes on
  extending the geodesic simulator to emit gravitational waves via the
  quadrupole formula: pseudo-Cartesian radial mappings, the reduced mass
  quadrupole, the TT-gauge polarization basis, energy/angular-momentum flux,
  and validation targets (frequency doubling, face-on/edge-on polarization,
  closed-form circular-orbit flux). `radiation/` implements this report task
  by task.

- **`../shared/core.py`** — the reusable framework: `System` (ABC), `Runner`,
  `run_from_config`, and the error types. No physics here — it lives outside
  this exercise's directory so future exercises (e.g. a Kerr simulator) can
  import it unchanged. `simulator.py` and `geodesics/schwarzschild.py` add
  `../shared` to `sys.path` before importing it.

- **`utils/visualization.py`** — shared plotting: `plot_orbit_panels` renders
  the standard 3-panel (r vs. time, φ vs. time, orbit) figure, and every
  figure is written under `visualizations/` (created automatically).

- **`geodesics/schwarzschild.py`** — the Schwarzschild `System`
  implementations (physics only; imports `core` and `visualization`), plus
  shared circular-orbit closed forms (`circular_orbit_E_h`,
  `circular_orbit_dE_da`) used by both the geodesic Systems and the radiation
  module.

- **`radiation/`** — gravitational-wave emission as pure post-processing
  functions over a trajectory, not `System` subclasses (see "Gravitational
  waves" below), plus one new `System` for radiation-reaction inspiral.

- **`simulator.py`** — thin entry point: imports `core` + `schwarzschild` +
  `radiation.inspiral`, declares `SYSTEM_REGISTRY`, and runs the CLI.

- **`config.yaml`** — a batch of simulations covering the regimes identified in
  the geodesics report (weak-field/Mercury-like, high eccentricity, ISCO,
  unstable circular orbit, marginally bound, radial plunge, and the photon
  analogues: flyby, near-critical deflection, capture, radial infall), plus
  two gravitational-wave-oriented runs (a coordinate-time circular orbit, and
  an adiabatic inspiral through the ISCO).

- **`visualizations/`** — output folder for every generated figure.

## Framework design

```
System (ABC)                                     [../shared/core.py]
  validate(params)   — subclass-implemented; raises if params are unfit to run
  simulate(params)   — subclass-implemented; integrates the geodesic, returns a trajectory
  visualize(result)  — subclass-implemented; plots and saves a figure

Runner(system)                                   [../shared/core.py]
  run(params) = system.validate(params); return system.simulate(params)

plot_orbit_panels(...)                           [utils/visualization.py]
  shared r/phi/orbit figure, saved under visualizations/
```

`Runner` is generic — it doesn't know or care what kind of system it's driving.
Each concrete `System` declares its own required parameters and validates
itself; `Runner` just enforces that validation happens before `simulate()`
ever runs. `core.py` has no Schwarzschild-specific code and lives in
`../shared/`, so a Kerr (or any other) exercise can import it directly
instead of re-implementing the framework. `utils/visualization.py` stays
local to this exercise for now (it saves to this exercise's own
`visualizations/` folder) but is likewise Schwarzschild-agnostic.

Four systems currently implement this interface:

- **`SchwarzschildGeodesic`** (`geodesics/schwarzschild.py`) — timelike
  geodesic of a massive test particle. Integrated in proper time `τ`.
  Params: `M, E, h, r0, phi0, dr_dtau0, tau_max`.
- **`SchwarzschildPhotonGeodesic`** (`geodesics/schwarzschild.py`) — null
  geodesic of a photon. Integrated in an affine parameter `λ` (proper time is
  identically zero for light). Characterized by impact parameter `b = L/E`
  instead of separate `E`/`h`. Params: `M, b, r0, phi0, dr_dlambda0, lambda_max`.
- **`SchwarzschildGeodesicCoordTime`** (`geodesics/schwarzschild.py`) — the
  same massive-particle orbit as `SchwarzschildGeodesic`, reparameterized in
  coordinate time `t` instead of `τ` (sharing the same underlying physics
  functions), because `t` is the natural time variable for a gravitational
  waveform. Params: `M, E, h, r0, phi0, dr_dtau0, t_max`.
- **`SchwarzschildAdiabaticInspiral`** (`radiation/inspiral.py`) — a circular
  orbit whose radius decays under quadrupole radiation reaction, until just
  above the ISCO. See "Gravitational waves" below. Params:
  `M, mu, a0, phi0, t_max`.

Both integrate the radial acceleration derived from the effective potential
(`d²r/dτ² = -M/r² + h²/r³ - 3Mh²/r⁴` for massive particles, missing the
Newtonian `-M/r²` term for massless photons) using `scipy.integrate.solve_ivp`
(adaptive-step RK45), with an event that stops integration if the trajectory
crosses the event horizon. Every run logs `rs = 2M` and its full parameter
dict at the start.

`visualize()` calls `plot_orbit_panels()` (`utils/visualization.py`), producing a
3-panel figure per run: `r` vs. time, `φ` vs. time, and the actual orbit
`(r cos φ, r sin φ)` in the equatorial plane with the horizon drawn as a
filled disk. Files are named after their physical parameters, e.g.
`schwarzschild_M1_E0.95_h4_r010_phi00.png`, and saved under
`visualizations/`.

## Batch runs from YAML

`simulator.py` declares `SYSTEM_REGISTRY`, mapping a `kind` string to a
`System` class, and passes it to `run_from_config` (from `../shared/core.py`),
which reads a YAML file shaped like:

```yaml
simulations:
  - kind: schwarzschild_massive
    params: {M: 1.0, E: 0.95, h: 4.0, r0: 10.0, phi0: 0.0, dr_dtau0: 0.0, tau_max: 500.0}
  - kind: schwarzschild_photon
    params: {M: 1.0, b: 6.0, r0: 30.0, phi0: 0.0, dr_dlambda0: -0.99, lambda_max: 80.0}
```

and runs + visualizes every entry. Unknown `kind` raises
`UnknownSimulationKindError`; missing params raise `MissingParameterError`
(from the relevant system's `validate()`).

```
python3 simulator.py            # runs config.yaml
python3 simulator.py other.yaml # or a specific file
```

## Why `scipy.integrate.solve_ivp`

Non-stiff 4-equation ODE system with adaptive step size — RK45 handles the
slow-then-fast dynamics near the photon sphere/ISCO without hand-tuning a
fixed step, and its `events` mechanism gives clean horizon-crossing detection
via root-finding instead of polling. Trade-off worth knowing: it isn't
symplectic, so `E`/`h`-like conserved quantities drift slowly over very long
integrations (observed: the ISCO circular orbit crept from `r=6.000` to
`r≈5.999` over ~3000 units of proper time). If long-run conservation becomes
the binding constraint — especially once there's no closed-form Kerr solution
to validate against — a symplectic/geometric integrator would be the natural
upgrade.

## Verified physical checks

- ISCO circular orbit (`r=6M`) stays circular to within numerical drift.
- Unstable circular orbit (`r=4.5M`, between photon sphere and ISCO) sits on
  a knife's edge as expected.
- Photon with `b=6 > b_crit=3√3` deflects and escapes (gravitational
  lensing); `b=5.25` (just above critical) grazes near the photon sphere with
  a large deflection angle before escaping; `b=0` free-falls radially.

## Gravitational waves (`radiation/`)

Added a full pipeline that turns a Schwarzschild trajectory into a
gravitational waveform via the (Newtonian) quadrupole formula, plus an
adiabatic inspiral driven by the resulting radiated energy. Every module
here is a set of pure functions operating on trajectory arrays, not a
`System` subclass — radiation post-processing has to be composable in or out
of an existing orbit without modifying `core.py` or the geodesic `System`s,
so it's layered on top instead of baked in. `SchwarzschildAdiabaticInspiral`
is the one exception: since radiation reaction feeds back into the orbit
itself, it's a new `System` you choose to run, not a flag on an existing one.

- **`mapping.py`** — three pseudo-Cartesian radial mappings for turning
  Schwarzschild `r` into flat-space coordinates for the quadrupole formula:
  Boyer-Lindquist (`R=r`, the default), harmonic (`R=r-M`), and isotropic.
  They agree at large `r` and differ by `O(M/r)` in the strong field — that
  spread is itself the mapping-choice systematic-error estimate.
- **`quadrupole.py`** — `mass_quadrupole()` computes the trace-free reduced
  quadrupole `Q_ij` and its second time derivative `Q̈_ij` analytically from
  a trajectory's `r, phi` and their coordinate-time derivatives — zero
  numerical differentiation, so no `1/Δt²` noise amplification.
- **`waveform.py`** — `strain_plus_cross()` projects `Q̈_ij` onto the
  TT-gauge polarization basis for an observer at `(theta_obs, phi_obs, D)`,
  giving `h_+(t)`, `h_×(t)`. **Open question, not yet resolved:**
  `theta_obs`/`phi_obs` are required arguments with no default — there's no
  observer direction that's more "correct" than another, but this reasoning
  has been pushed back on and isn't settled (tracked in the `whatidid` space
  `particle-orbiting-kerr`).
- **`flux.py`** — `energy_flux()`/`angular_momentum_flux()` from `Q⃛`
  (one numerical derivative of the already-analytic `Q̈`, well-conditioned
  since `Q̈` itself has no solver noise). Validated against the closed-form
  circular-orbit `dE/dt = (32/5)μ²M³/a⁵`.
- **`inspiral.py`** — `SchwarzschildAdiabaticInspiral`, an adiabatic
  ("sequence of circular orbits") inspiral: integrates `a(t)`/`phi(t)`
  directly, using the closed-form circular-orbit flux and
  `circular_orbit_dE_da` to get `da/dt`, stopping just above the ISCO where
  the adiabatic approximation itself is known to break down (transition to
  plunge). Valid in the extreme-mass-ratio limit (`μ ≪ M`); see the module
  docstring for the full physical justification. Its output has the same
  shape as `SchwarzschildGeodesicCoordTime`'s, so it flows through
  `mass_quadrupole` → `flux`/`waveform` unmodified to produce a chirping
  waveform.
- **`waveform_visualization.py`** — `h_+`/`h_×` time series, a power
  spectrum (verifies the frequency-doubling check: GW frequency = 2× orbital
  frequency), and a chirp spectrogram for the inspiral. Also two
  frequency-domain checks usable outside of plotting:
  `dominant_frequency()` (single FFT peak) and `instantaneous_frequency()`
  (Hilbert-transform phase derivative, for verifying a rising chirp where a
  single stationary FFT no longer applies).

Two new `config.yaml` entries exercise this pipeline directly:
`schwarzschild_massive_coord_time` (a stable circular orbit, ready to feed
into the quadrupole/waveform functions) and `schwarzschild_adiabatic_inspiral`
(a full decay from `a=8M` to just above the ISCO — see its plot for the
inward spiral).

What this deliberately gets wrong, and why it's still useful: the quadrupole
formula is a weak-field approximation that's known to be off by ~5% even at
`r₀~50M` in the literature, worse near the ISCO — this is the same
"numerical kludge" approach used for real EMRI waveform modeling, valid for
getting the qualitative shape/frequency evolution right, not for
precision-matching a true relativistic waveform.

## Next steps

- Implement the Kerr generalization (`a ≠ 0`): third conserved quantity
  (Carter constant), spin-dependent photon sphere/ISCO, frame dragging,
  ergosphere.
- Consider a symplectic integrator if long-duration conservation checks
  become important for validating Kerr orbits.
- Resolve the open `theta_obs`/`phi_obs` default question in `waveform.py`.
- Extend the adiabatic inspiral (or the quadrupole formula generally) to
  Kerr once the Kerr geodesic integrator exists — `mass_quadrupole()` and
  everything downstream of it is already spacetime-agnostic (equatorial
  `r, phi` in, `Q_ij`/`Q̈_ij` out), so only the geodesic and circular-orbit
  closed forms need to change.
