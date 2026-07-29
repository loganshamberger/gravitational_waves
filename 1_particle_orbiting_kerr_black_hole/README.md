# Particle Orbiting a (Schwarzschild) Black Hole

Step one toward a Kerr geodesic simulator: understand and implement Schwarzschild
orbits first (the non-rotating special case), then generalize.

## Contents

- **`schwarzschild_geodesics_report.md`** — research notes distilled from the
  [Wikipedia article on Schwarzschild geodesics](https://en.wikipedia.org/wiki/Schwarzschild_geodesics):
  the metric, the distinction between coordinate time and proper time, the two
  conserved quantities (specific energy `E` and specific angular momentum `h`),
  the effective potential, orbit classification (circular, ISCO, photon sphere,
  marginally bound, plunge), perihelion precession, and the physical regimes
  worth stress-testing a simulator against. Also notes what changes once we
  move to Kerr (Carter constant, ergosphere, spin-dependent ISCO/photon sphere).

- **`../shared/core.py`** — the reusable framework: `System` (ABC), `Runner`,
  `run_from_config`, and the error types. No physics here — it lives outside
  this exercise's directory so future exercises (e.g. a Kerr simulator) can
  import it unchanged. `simulator.py` and `schwarzschild.py` add `../shared`
  to `sys.path` before importing it.

- **`visualization.py`** — shared plotting: `plot_orbit_panels` renders the
  standard 3-panel (r vs. time, φ vs. time, orbit) figure, and every figure
  is written under `visualizations/` (created automatically).

- **`schwarzschild.py`** — the two Schwarzschild `System` implementations
  (physics only; imports `core` and `visualization`).

- **`simulator.py`** — thin entry point: imports `core` + `schwarzschild`,
  declares `SYSTEM_REGISTRY`, and runs the CLI.

- **`config.yaml`** — a batch of simulations covering the regimes identified in
  the report (weak-field/Mercury-like, high eccentricity, ISCO, unstable
  circular orbit, marginally bound, radial plunge, and the photon analogues:
  flyby, near-critical deflection, capture, radial infall).

- **`visualizations/`** — output folder for every generated figure.

## Framework design

```
System (ABC)                                     [../shared/core.py]
  validate(params)   — subclass-implemented; raises if params are unfit to run
  simulate(params)   — subclass-implemented; integrates the geodesic, returns a trajectory
  visualize(result)  — subclass-implemented; plots and saves a figure

Runner(system)                                   [../shared/core.py]
  run(params) = system.validate(params); return system.simulate(params)

plot_orbit_panels(...)                           [visualization.py]
  shared r/phi/orbit figure, saved under visualizations/
```

`Runner` is generic — it doesn't know or care what kind of system it's driving.
Each concrete `System` declares its own required parameters and validates
itself; `Runner` just enforces that validation happens before `simulate()`
ever runs. `core.py` has no Schwarzschild-specific code and lives in
`../shared/`, so a Kerr (or any other) exercise can import it directly
instead of re-implementing the framework. `visualization.py` stays local to
this exercise for now (it saves to this exercise's own `visualizations/`
folder) but is likewise Schwarzschild-agnostic.

Two systems currently implement this interface (in `schwarzschild.py`):

- **`SchwarzschildGeodesic`** — timelike geodesic of a massive test particle.
  Integrated in proper time `τ`. Params: `M, E, h, r0, phi0, dr_dtau0, tau_max`.
- **`SchwarzschildPhotonGeodesic`** — null geodesic of a photon. Integrated in
  an affine parameter `λ` (proper time is identically zero for light).
  Characterized by impact parameter `b = L/E` instead of separate `E`/`h`.
  Params: `M, b, r0, phi0, dr_dlambda0, lambda_max`.

Both integrate the radial acceleration derived from the effective potential
(`d²r/dτ² = -M/r² + h²/r³ - 3Mh²/r⁴` for massive particles, missing the
Newtonian `-M/r²` term for massless photons) using `scipy.integrate.solve_ivp`
(adaptive-step RK45), with an event that stops integration if the trajectory
crosses the event horizon. Every run logs `rs = 2M` and its full parameter
dict at the start.

`visualize()` calls `plot_orbit_panels()` (`visualization.py`), producing a
3-panel figure per run: `r` vs. time, `φ` vs. time, and the actual orbit
`(r cos φ, r sin φ)` in the equatorial plane with the horizon drawn as a
filled disk. Files are named after their physical parameters, e.g.
`schwarzschild_M1_E0.95_h4_r010_phi00.png`, and saved under
`visualizations/`.

## Batch runs from YAML

`simulator.py` declares `SYSTEM_REGISTRY`, mapping a `kind` string to a
`System` class, and passes it to `core.run_from_config`, which reads a YAML
file shaped like:

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

## Next steps

- Implement the Kerr generalization (`a ≠ 0`): third conserved quantity
  (Carter constant), spin-dependent photon sphere/ISCO, frame dragging,
  ergosphere.
- Consider a symplectic integrator if long-duration conservation checks
  become important for validating Kerr orbits.
