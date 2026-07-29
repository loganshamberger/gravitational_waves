# Schwarzschild Geodesics — Report

Source: [Wikipedia — Schwarzschild geodesics](https://en.wikipedia.org/wiki/Schwarzschild_geodesics)

This report summarizes the non-rotating (Schwarzschild) case as a foundation before
generalizing to Kerr (rotating) black holes.

## 1. Metric and coordinates

The Schwarzschild metric, in units where the line element has dimensions of `c²dτ²`:

```
ds² = c²dτ² = (1 - rs/r) c²dt² - dr²/(1 - rs/r) - r²(dθ² + sin²θ dφ²)
```

- `t` — coordinate time: time measured by a stationary observer at infinity. Only valid for `r > rs`.
- `τ` — proper time: time measured by a clock moving along with the particle.
- `r` — radial (areal) coordinate, defined so that a sphere at radius `r` has circumference `2πr`. Valid for `r > rs`.
- `θ, φ` — colatitude and longitude.
- `rs = 2GM/c²` — Schwarzschild radius (event horizon).

**Key point on time**: coordinate time `t` and proper time `τ` are distinct and related through the
conserved energy (see below). This is the source of gravitational time dilation — as measured by a
distant observer, a clock falling toward `rs` appears to run increasingly slowly and coordinate time
diverges as the particle approaches the horizon, even though proper time along the geodesic stays finite.

For **massive particles**, proper time `τ` is the natural affine-like parameter along the geodesic.
For **photons** (null geodesics), proper time is identically zero along the path, so an arbitrary
affine parameter `λ` is used instead.

## 2. Constants of motion

Two conserved quantities arise from the metric's symmetries (time-translation and rotational Killing
vectors), directly analogous to energy and angular momentum conservation in Newtonian orbital mechanics:

**Specific energy** (energy per unit rest mass, conserved along the geodesic):
```
(1 - rs/r)(dt/dτ) = E/(mc²)   =>   E = constant
```
`E` includes rest energy, so a particle at rest at infinity has `E = mc²`.

**Specific angular momentum**:
```
h = L/μ = r²(dφ/dτ) = constant
```
where `μ` is the reduced mass (`μ ≈ m` when `M ≫ m`, i.e., a test particle).

Because the metric is spherically symmetric, motion is confined to a plane, so a single angular
momentum component fully describes it (no analog of Lense-Thirring frame dragging — that only appears
in Kerr).

For photons, `E` and `h` individually are not meaningful (a photon has no rest mass to normalize
against), but the **ratio** `mh/E`, equivalently the **impact parameter** `b = h/E·(mc²/c)` (units vary
by convention), is conserved and characterizes the trajectory.

## 3. Equation of motion / effective potential

From the metric normalization condition `ds² = c²dτ²` (or `= 0` for photons), one derives a radial
equation of the same form as a 1D energy-conservation problem:

```
(dr/dτ)² = E²/(m²c²) - (1 - rs/r)(c² + h²/r²)
```

which can be recast with an **effective potential** `V(r)`:

```
V(r) = -GMm/r + L²/(2μr²) - G(M+m)L²/(c²μr³)
```

Three terms:
1. `-GMm/r` — ordinary Newtonian gravity (attractive).
2. `L²/(2μr²)` — centrifugal barrier (repulsive), same as Newtonian.
3. `-G(M+m)L²/(c²μr³)` — **purely relativistic correction**, attractive, falls off as `1/r³`. This term
   is what makes Schwarzschild orbits qualitatively different from Keplerian ellipses: it can dominate
   at small `r`, removing the centrifugal barrier entirely below a critical angular momentum and
   allowing particles to plunge into the horizon regardless of angular momentum.

For **photons**, using `u = 1/r`:
```
(du/dφ)² = rs u³ - u² + 1/b²
```
Note there is no rest-mass/Newtonian term — the leading behavior is governed entirely by the `rs·u³`
relativistic term and the impact parameter `b`.

## 4. Orbit classification

**Circular orbits** occur where `dV/dr = 0`. Solving with `a = h/c` gives two roots:
```
r_outer = (a²/rs)[1 + √(1 - 3rs²/a²)]     (stable)
r_inner = (a²/rs)[1 - √(1 - 3rs²/a²)]     (unstable)
```

- **ISCO (innermost stable circular orbit)**: the discriminant `1 - 3rs²/a²` vanishes at `a = rs√3`,
  giving `r_ISCO = 6rs = 6GM/c²`. Below this radius no stable circular orbit exists for a massive
  particle.
- **Photon sphere**: unstable circular orbit for light, at `r_photon = 3rs` (half the ISCO radius,
  independent of energy since photons have no rest mass to set a scale). Critical impact parameter
  `b_crit = (3√3/2) rs`; photons with `b < b_crit` are captured, `b > b_crit` are deflected and escape.
- **Marginally bound orbit**: `E = mc²` (zero net energy at infinity — parabolic-like), located at
  `r_mb = 4rs`.
- **Bound / precessing elliptical orbits**: the cubic in `u` has three real roots `u1 < u2 < u3`; the
  particle oscillates between `r_min = 1/u2` and `r_max = 1/u1`, precessing each radial period rather
  than closing into a fixed ellipse (unlike Newtonian orbits).
- **Unbound / hyperbolic orbits**: `u1 ≤ 0`; particle arrives from and escapes to infinity.
- **Marginal/separatrix orbits**: `u2 = u3` (double root) — particle asymptotically spirals onto the
  unstable circular orbit rather than reaching it in finite proper time/angle.
- **Plunge orbits**: angular momentum too low to maintain any circular orbit; particle falls through
  the horizon after a finite change in `φ`.

| Feature | Radius |
|---|---|
| Event horizon | `rs = 2GM/c²` |
| Photon sphere (unstable, light only) | `3rs` |
| Marginally bound orbit | `4rs` |
| ISCO (innermost stable circular orbit, massive particles) | `6rs` |

## 5. Perihelion precession

The `1/r³` relativistic term in `V(r)` causes bound elliptical orbits to precess by a small angle per
orbit rather than closing:

```
δφ ≈ 6πGM / (c² A (1 - e²))
```

where `A` is the semi-major axis and `e` the eccentricity. This is the classic GR test explaining
Mercury's anomalous perihelion precession, and is a good sanity check for a numerical integrator: for
weak fields (`rs/A` small) the simulated precession per orbit should converge to this formula.

## 6. Interesting/extreme regimes to test a simulator against

- **Weak-field limit**: far from the black hole (`r ≫ rs`), orbits should reduce to Newtonian ellipses;
  precession should match the perturbative formula above (Mercury-like check).
- **Near the photon sphere (`r ≈ 3rs`)**: extreme light bending; small changes in impact parameter `b`
  near `b_crit` produce large deflection-angle changes — good for testing numerical sensitivity/step size.
- **Near ISCO (`r ≈ 6rs`)**: last stable circular orbit; orbits just inside should slowly plunge, just
  outside should remain (quasi-)stable — useful to validate stability boundaries against the analytic
  `r_ISCO = 6rs` prediction.
- **Marginally bound (`E = mc²`, `r ≈ 4rs`)**: transition between bound and unbound behavior.
- **High-eccentricity bound orbits**: large precession accumulation per orbit; tests long-term numerical
  accuracy/energy-angular-momentum conservation over many periods.
- **Photon capture vs. escape**: sweep impact parameter `b` across `b_crit = (3√3/2) rs` and confirm the
  transition from capture to escape.
- **Radial plunge (`h = 0`)**: purely radial infall has a closed-form solution and is a good unit test
  for coordinate-time divergence at the horizon vs. finite proper time.
- **Conservation checks**: `E` and `h` (equivalently `L`) should remain constant along any integrated
  trajectory to numerical tolerance — this is the most direct correctness check for an integrator,
  regardless of orbit type.

## 7. Relevance to the Kerr generalization

Schwarzschild is the non-rotating (`a = 0`) special case of Kerr. Moving to Kerr will require:
- A third "hidden" constant of motion (the Carter constant), since Kerr's reduced symmetry no longer
  confines geodesics to a single orbital plane.
- Separate inner/outer horizons and an ergosphere, replacing the single horizon at `rs`.
- Frame dragging, which couples `φ` motion to the black hole's spin even for zero angular momentum
  particles.
- Distinct ISCO/photon-sphere radii for prograde vs. retrograde orbits (no longer single numbers like
  `3rs`/`6rs`, but functions of spin `a` and orbit orientation).

The energy/angular-momentum conservation checks and near-horizon/near-photon-sphere/near-ISCO test
regimes identified here carry over directly as validation targets once spin is introduced.
