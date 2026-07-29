# Gravitational Waves from an Orbiting Particle: the Quadrupole Formula in TT Gauge

Research notes for extending the Schwarzschild geodesic simulator to emit a
waveform. Companion to `schwarzschild_geodesics_report.md`. No code yet — this
is the formalism, the conventions we'll adopt, the numerical strategy, and an
honest accounting of what the approximation does and does not capture.

Units: `G = c = 1`, consistent with the rest of the project.

---

## 1. Where the formula comes from

Linearize the Einstein equations about flat space, `g_μν = η_μν + h_μν` with
`|h| ≪ 1`. In Lorenz gauge (`∂^μ h̄_μν = 0`, with `h̄_μν = h_μν - ½η_μν h`)
the field equation becomes a flat-space wave equation:

```
□ h̄_μν = -16π T_μν
```

whose retarded solution is

```
h̄_μν(t, x) = 4 ∫ T_μν(t - |x - x'|, x') / |x - x'| d³x'
```

Expand for a far, slow-moving, compact source (`D ≫ source size ≫`
wavelength-suppressed corrections). Stress-energy conservation `∂_μ T^μν = 0`
converts the `T_ij` integral into second time derivatives of the second mass
moment. Keeping only the leading term:

```
h̄_ij(t, x) = (2/D) Ïij(t - D)
```

with the **second mass moment**

```
I_ij(t) = ∫ ρ(t, x) x_i x_j d³x
```

The physical, gauge-invariant radiation content is the transverse–traceless
part of this (§3). This is Einstein's 1918 quadrupole formula.

**Three approximations are baked in**, and all three are strained by a
strong-field orbit:

1. **Weak field** — the source's own gravity is treated as a perturbation on
   flat space. False at `r ~ 6M`.
2. **Slow motion** (`v ≪ 1`) — the multipole expansion is an expansion in
   `v`. At the ISCO, `v ≈ 0.4`, so `v²` corrections are ~15%.
3. **Flat-space wave propagation** — the wave is assumed to travel on
   Minkowski space, ignoring backscatter off the background curvature
   ("tails").

§7 quantifies this and names the rigorous alternative.

---

## 2. The quadrupole moment for a point particle

For a single test particle of mass `μ` at pseudo-Cartesian position `x(t)`,
`ρ(t, x) = μ δ³(x - x(t))`, so the integral collapses:

```
I_ij(t) = μ x_i(t) x_j(t)
```

Two definitions circulate in the literature and it matters which you use where:

| symbol | definition | where it's used |
|---|---|---|
| `I_ij` | `μ x_i x_j` (has a trace) | fine for `h_+`/`h_×` — the TT projection removes the trace anyway (see §3) |
| `Q_ij` | `I_ij - ⅓ δ_ij x·x` (trace-free, "reduced") | **required** for the energy/angular-momentum flux formulas (§5) |

Using `I_ij` in the flux formula gives a wrong answer. Using `Q_ij` for the
strain gives the same answer as `I_ij`. Safest convention: **build `Q_ij`,
use it everywhere.**

### 2a. Whose mass, and about which origin?

Strictly, `I_ij` is the moment of the *whole system* in its center-of-mass
frame. For a two-body system of masses `m₁, m₂` that yields `I_ij = μ x_i x_j`
with `μ = m₁m₂/(m₁+m₂)` the reduced mass and `x` the *relative* separation.

For an extreme mass ratio (`μ ≪ M`) the central black hole barely recoils, so
"particle position relative to the hole" and "relative separation" coincide to
`O(μ/M)`, and `μ` is just the particle mass. This is consistent with the
geodesic approximation we're already making — the particle doesn't back-react
on the metric. Good; but it does mean the code needs `μ` as a new parameter,
since the geodesic itself is mass-independent.

### 2b. The pseudo-Cartesian mapping — a real choice, not a formality

The quadrupole formula wants Cartesian coordinates on *flat* space. We have
Schwarzschild `(r, θ, φ)`, which are **not** flat-space spherical polars —
`r` is an areal radius, and proper radial distance is `dr/√(1-2M/r)`.

There is no unique right answer; the standard options differ at `O(M/r)`:

| mapping | `x = R sinθ cosφ`, etc., with `R =` | note |
|---|---|---|
| **Boyer–Lindquist / "numerical kludge"** | `r` (Schwarzschild `r` directly) | what the EMRI kludge waveform literature uses; empirically matches Teukolsky well |
| harmonic | `r - M` | natural if comparing to PN results in harmonic gauge |
| isotropic | `½(r - M + √(r² - 2Mr))` | the metric is conformally flat in these; arguably the most defensible "flat space" identification |

All three agree as `r → ∞`. **Recommendation: Boyer–Lindquist (`R = r`)** as
the default, because it's the choice validated against exact results in the
kludge literature — but make it a switchable option, since the spread between
mappings is a free, cheap estimate of the systematic error of the whole
scheme.

Since our orbits are equatorial (`θ = π/2`), the mapping reduces to:

```
x = R cos φ,   y = R sin φ,   z = 0
```

The orbital plane is the `xy` plane. Note the particle motion is planar but
the *radiation* is not confined to that plane — the observer direction is
fully general (§3).

---

## 3. TT projection for an arbitrary observer direction

Let `n̂` be the unit vector from the source to the observer. Define the
spatial projector onto the plane transverse to `n̂`:

```
P_ij = δ_ij - n_i n_j
```

and the **TT (Lambda) projection tensor**:

```
Λ_ij,kl(n̂) = P_ik P_jl - ½ P_ij P_kl
```

`Λ` is symmetric in `(ij)`, in `(kl)`, under exchange of pairs, is transverse
(`n^i Λ_ij,kl = 0`), traceless (`δ^ij Λ_ij,kl = 0`), and idempotent. The
waveform is then

```
h_ij^TT(t) = (2/D) Λ_ij,kl(n̂) Q̈_kl(t - D)
```

### 3a. Polarization basis

Pick two orthonormal vectors spanning the transverse plane. Parameterize the
observer direction by `(Θ, Φ_obs)` in the same pseudo-Cartesian frame:

```
n̂  = ( sinΘ cosΦ_obs,  sinΘ sinΦ_obs,  cosΘ )
ê_Θ = ( cosΘ cosΦ_obs,  cosΘ sinΦ_obs, -sinΘ )
ê_Φ = (     -sinΦ_obs,       cosΦ_obs,      0 )
```

`(n̂, ê_Θ, ê_Φ)` is right-handed and orthonormal. Because the orbit is
equatorial, `Θ` is the **inclination** `ι` (`Θ = 0` → face-on, viewed down the
orbital angular momentum axis; `Θ = π/2` → edge-on), and `Φ_obs` just sets the
zero of the waveform's phase.

The polarization tensors:

```
e^+_ij  = (ê_Θ)_i (ê_Θ)_j - (ê_Φ)_i (ê_Φ)_j
e^×_ij  = (ê_Θ)_i (ê_Φ)_j + (ê_Φ)_i (ê_Θ)_j
```

Both are already transverse and traceless w.r.t. `n̂`, and satisfy
`e^A_ij e^B_ij = 2 δ^AB`. So

```
h_ij^TT = h_+ e^+_ij + h_× e^×_ij,     h_A = ½ h_ij^TT e^A_ij
```

### 3b. The shortcut worth using

Because `e^+` and `e^×` are *already* TT, contracting them with `Λ` returns
them unchanged. So `Λ` never needs to be built explicitly:

```
h_+ = (1/D) Q̈_kl e^+_kl = (1/D) ( Q̈_ΘΘ - Q̈_ΦΦ )
h_× = (1/D) Q̈_kl e^×_kl = (2/D)   Q̈_ΘΦ
```

where `Q̈_ΘΘ ≡ (ê_Θ)_k (ê_Θ)_l Q̈_kl`, etc. This is a handful of dot products
instead of a rank-4 tensor contraction, and — since the trace is removed by
the contraction too — it works identically with `I_ij` in place of `Q_ij`.

Build `Λ` explicitly only if we want the full `h_ij^TT` tensor for a
visualization (e.g. animating the ring-of-particles deformation).

### 3c. Two gotchas

- **Polarization angle ambiguity.** Rotating `(ê_Θ, ê_Φ)` by `ψ` about `n̂`
  mixes the polarizations: `h_+ + i h_× → e^{-2iψ}(h_+ + i h_×)`. Our basis
  choice fixes `ψ = 0`. It's a convention, not physics — but it must be
  documented, because `h_+` alone is meaningless without it.
- **Poles.** At `Θ = 0` or `π`, `ê_Θ` and `ê_Φ` are individually
  discontinuous in `Φ_obs` (though still orthonormal — the formulas above
  don't divide by `sinΘ`, so they're numerically safe). Don't construct the
  basis via cross products with `ẑ`; use the closed forms.

---

## 4. Getting `Q̈` — the numerical strategy

This is where the implementation lives or dies, and there's a clean way to
avoid numerical differentiation almost entirely.

### 4a. Coordinate time, not proper time

The quadrupole formula's `t` is the **asymptotic coordinate time** (≈ observer
proper time at large `D`), not the particle's proper time `τ`. The current
integrator advances in `τ` and returns `t(τ)`, which is a non-uniform,
monotonically increasing grid.

Two options:

- **(A) Re-parameterize the integration to `t`.** Divide the existing RHS by
  `dt/dτ = E/(1 - 2M/r)`. State becomes `[r, φ, dr/dt]` on a uniform `t`
  grid. Downstream everything (finite differences, FFT, plotting against
  detector time) is trivial.
- **(B) Keep `τ`, chain-rule.** `d/dt = (dτ/dt) d/dτ`, then interpolate onto a
  uniform `t` grid for the FFT.

**Recommendation: (A).** It costs one line in the RHS and removes an entire
class of interpolation error. Worth keeping the `τ` integrator too — proper
time is the natural parameter for the *orbit*; coordinate time is the natural
parameter for the *waveform*. Two entry points, one physics.

Caveat for (A): `dt/dτ` diverges at the horizon, so a plunging orbit stalls in
`t`. That's physically correct (infinite redshift) and the existing horizon
event already terminates it, but the last few samples will be badly
conditioned and should be trimmed.

### 4b. Differentiate analytically, not numerically

Don't finite-difference `Q_ij` twice. Expand by the product rule:

```
I_ij   = μ x_i x_j
İ_ij   = μ ( v_i x_j + x_i v_j )
Ï_ij   = 2μ v_i v_j + μ ( a_i x_j + x_i a_j )
```

and `v = dx/dt`, `a = d²x/dt²` are available **analytically** from the
geodesic equations we already integrate. For the equatorial case, with
`R(t)` and `φ(t)` and `Ṙ, φ̇` from the state vector, and `R̈, φ̈` from the RHS:

```
x  = ( R cosφ, R sinφ, 0 )
v  = ( Ṙcosφ - Rφ̇ sinφ,  Ṙ sinφ + Rφ̇ cosφ, 0 )
a  = ( (R̈ - Rφ̇²)cosφ - (2Ṙφ̇ + Rφ̈) sinφ,
       (R̈ - Rφ̇²)sinφ + (2Ṙφ̇ + Rφ̈) cosφ,  0 )
```

So the strain requires **zero numerical derivatives** — a large accuracy win,
since naive second-order finite differencing of an ODE solution amplifies
solver noise by `~1/Δt²`.

Then subtract the trace once at the end: `Q̈_ij = Ï_ij - ⅓ δ_ij Ï_kk`.

### 4c. The third derivative (only needed for flux)

The energy flux (§5) needs `Q⃛`. Options, in order of preference:

1. Differentiate the `a` expression analytically once more (the jerk), using
   `dR̈/dt` from differentiating the radial RHS. Fully analytic, most accurate,
   most algebra.
2. Numerically differentiate the analytic `Q̈` once. One derivative of a clean
   analytic function on a uniform grid is well-conditioned — spectral or
   high-order central differences are fine. **Recommended starting point.**
3. Compute the flux from the strain instead: `dE/dt = (D²/16π) ∮ (ḣ_+² + ḣ_×²) dΩ`.
   Equivalent, but requires a sky integral.

### 4d. Retarded time

`h(t_obs) = h(t_src - D)`. For a single source at fixed `D` this is a constant
offset — physically important for labeling, numerically a no-op. Just record
it rather than shifting arrays.

---

## 5. Energy and angular momentum flux

Using the **trace-free** `Q_ij`, angle-brackets denoting an average over
several wave periods:

```
dE/dt  = (1/5) ⟨ Q⃛_ij Q⃛_ij ⟩
dL_i/dt = (2/5) ε_ijk ⟨ Q̈_jl Q⃛_kl ⟩
```

Angular distribution of the flux:

```
dE / (dt dΩ) = (D²/16π) ⟨ ḣ_+² + ḣ_×² ⟩
```

Integrating the last over the sphere must reproduce the first — a good
internal consistency check that the TT projection and polarization basis are
implemented correctly.

---

## 6. Validation targets

Closed-form results the implementation must reproduce. A test particle `μ` on
a **circular orbit** of pseudo-Cartesian radius `a`, orbital phase
`Φ(t) = Ω t`, with Kepler `Ω² = M/a³`, observed at inclination `ι`:

```
h_+ = -(2 μ M) / (D a) · (1 + cos²ι) · cos 2Φ
h_× = -(4 μ M) / (D a) ·      cos ι  · sin 2Φ
dE/dt = (32/5) μ² M³ / a⁵
```

Note the amplitude `∝ μ M / (D a)`, and the sign convention that follows from
`Φ` measured from the `x` axis with the observer in the `xz` plane.

Checks this gives us, cheaply:

- **Frequency doubling.** GW frequency is `2 ×` the orbital frequency. Falls
  straight out of the FFT.
- **Face-on (`ι = 0`).** `|h_+| = |h_×|`, `90°` out of phase → circular
  polarization. Amplitude `4μM/(Da)`.
- **Edge-on (`ι = π/2`).** `h_× = 0` exactly, `h_+` amplitude `2μM/(Da)` —
  linear polarization. A strong test of the TT projection.
- **Peanut-shaped beaming.** The flux is largest along the orbital axis,
  smallest in the plane, by a factor of 8 — check via the sky integral in §5.
- **Eccentric orbits** emit at harmonics of the orbital frequency, not just
  the second — a rich FFT test.
- **Precessing orbits** (the relativistic perihelion advance already in the
  simulator) split each harmonic into a doublet separated by the precession
  frequency. This is a genuinely relativistic feature that survives even
  though the wave-generation formula is Newtonian.

---

## 7. What this will get wrong, and by how much

Being blunt about this now saves confusion later:

- **Amplitude error grows into the strong field.** The literature finding for
  static spherically symmetric holes is that the quadrupole formula deviates
  from the exact (Teukolsky/black-hole-perturbation) result by **~5% even at
  `r₀ ~ 50M`**, and considerably worse near the ISCO. The waveform *shape* and
  *phasing* hold up far better than the amplitude — which is exactly why the
  "numerical kludge" family of EMRI waveforms is useful despite being built on
  this approximation.
- **No radiation reaction.** The geodesic is fixed; energy leaves the system
  but the orbit does not shrink. This is internally inconsistent, but fine for
  a snapshot of a few orbits. Adding adiabatic inspiral — evolving `E` and `h`
  using the computed `dE/dt`, `dL/dt` — is a natural, well-defined next step
  and would produce a chirp.
- **Gauge ambiguity of the mapping.** The `O(M/r)` spread between the three
  radial mappings in §2b is a *lower bound* on the systematic error.
- **Missing physics:** current quadrupole (spin/mass-current radiation),
  higher mass multipoles, tails/backscatter, and any horizon absorption.
- **The rigorous alternative**, for when this is no longer good enough:
  Regge–Wheeler–Zerilli black hole perturbation theory (Schwarzschild) or the
  Teukolsky equation (Kerr) — solve for the perturbation on the *curved*
  background, decomposed in spin-weighted spheroidal harmonics. Much heavier,
  and worth doing only after the quadrupole version is working as a baseline
  to compare against.

---

## 8. Implementation sketch (for the next session)

Proposed shape, consistent with the existing `System` framework:

1. **A coordinate-time geodesic integrator** — the `τ` RHS divided by
   `dt/dτ`; returns `r, φ, ṙ, φ̇, r̈, φ̈` on a uniform `t` grid.
2. **A quadrupole module** — pure function of the trajectory:
   `(t, R, φ, Ṙ, φ̇, R̈, φ̈, μ, mapping) → Q_ij(t), Q̈_ij(t)`. Analytic, no
   differentiation. Spacetime-agnostic, so Kerr reuses it unchanged.
3. **A TT/observer module** — `(Q̈_ij, n̂(Θ,Φ_obs), D) → h_+(t), h_×(t)`, via
   §3b. Also spacetime-agnostic. Optional full `h_ij^TT` via explicit `Λ`.
4. **A flux module** — `Q⃛` (one numerical derivative to start) → `dE/dt`,
   `dL/dt`, sky distribution.
5. **Visualization** — `h_+`/`h_×` vs. time; power spectrum; a sky map of
   strain amplitude; optionally the ring-of-test-particles animation.

Modules 2–4 have no Schwarzschild-specific content, so they belong in a `radiation/` package rather than inside `geodesics/`.

**New parameters:** `mu` (particle mass), `D` (observer distance),
`theta_obs`, `phi_obs`, and a `mapping` selector.

---

## 9. Open questions to settle before writing code

1. Which radial mapping is the default — and do we want all three switchable
   to expose the systematic error? ( BL default, all three available.) 
2. Do we replace the proper-time integrator with a coordinate-time one, or
   keep both? (keep both, share the physics.)
3. Is adiabatic radiation reaction in scope now, or is a fixed-geodesic
   snapshot the goal for this stage? (This is in scope now)
4. Do we want the full `h_ij^TT` tensor (for a deformation animation), or are
   `h_+`/`h_×` scalars enough? (Only h_+ and h_x)

---

## References

- [Quadrupole formula — Wikipedia](https://en.wikipedia.org/wiki/Quadrupole_formula)
- [T. Moore, Les Houches lectures, Session 5: Gravitational Waves](https://pages.pomona.edu/~tmoore/LesHouches/les-houches-5.pdf)
- [S. Bernuzzi & M. Breschi, *Notes on Gravitational Waves*](http://sbernuzzi.gitpages.tpi.uni-jena.de/gw/notes/2020/main_v0.0.pdf)
- [Rational Orbits and Gravitational Waves in Static Spherical Spacetimes: An Open-Source Numerical Framework (arXiv:2606.21029)](https://arxiv.org/html/2606.21029)
- [Gravitational waves from bodies orbiting the Galactic center black hole and their detectability by LISA (A&A 2019)](https://www.aanda.org/articles/aa/full_html/2019/07/aa35406-19/aa35406-19.html)
- [Gravitational waves from merging compact binaries (arXiv:0903.4877)](https://arxiv.org/pdf/0903.4877)
- [Low-Frequency Sources of Gravitational Waves: A Tutorial (arXiv:gr-qc/9710079)](https://arxiv.org/pdf/gr-qc/9710079)
