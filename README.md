# Compressible-velocity active fluid 2D

**Live demo: https://csviragh.github.io/compressible-ns-sim/**

Interactive 2D active fluid with a *compressible* velocity field (no pressure projection) and a
Lamé viscous operator (shear viscosity μ, second coefficient λ), plus the self-propulsion term of its
sister project [navier-stokes-sim](https://github.com/csviragh/navier-stokes-sim) (incompressible).
Vite + TypeScript, plain canvas, no runtime dependencies.

![screenshot](screenshot.png)

## Equations

A Stable-Fluids-style solver like navier-stokes-sim, but **without the pressure projection and
without any equation of state**: the velocity is free to compress (∇·u ≠ 0). On the periodic unit
square (default 256 × 256, side 1, unit density):

```
∂u/∂t + (u·∇)u = μ∇²u + (λ+μ)∇(∇·u) − (1/τ)(|u| − v₀) u/|u| + f
```

where the Lamé operator is **delayed** (below). There is no Cucker–Smale or other nonlocal alignment
kernel in this model: the only coupling between neighbouring points is the (delayed) Lamé operator.

### Delayed Lamé operator (the model)

The discrete Lamé operator (periodic, collocated: μ × 5-point Laplacian, (λ+μ)[∂xx u + ∂xy v, ∂xy u + ∂yy v]
with 3-point ∂xx and 4-corner ∂xy) is split as L = D + N:

* **self part** D: the cell's own stencil weight, `D = −(4μ + 2(λ+μ))/h²` (same for both components);
* **neighbour part** N = L − D: all off-centre stencil weights.

The model uses

```
L_delayed[u](x, t) = D·u(x, t − t_self) + N[u(·, t − t_nb)](x)
```

* t_self, t_nb ∈ [0, 2] s, default 0, with a **lock t_self = t_nb** checkbox (on by default).
* Past fields are read **Eulerian**, at the fixed grid point, from a ring buffer of past velocity fields
  (`FieldHistory`, one frame per Δt, linear interpolation in time; the current sub-step field is the newest point).
* With t_self = t_nb = 0 it is exactly the undelayed operator L (tested).
* Advection, self-propulsion and density transport are not delayed.

**Delay-driven checkerboard instability (physics of the delayed model, not a bug).** For equal delays t
the grid-scale checkerboard mode (k = (π/h, π/h)) obeys `â' = −Λ_cb â(t − t)` with Λ_cb = 8μ/h²
(transverse) or (8μ + 4(λ+μ))/h² (compressive). This delay equation is unstable for **Λ_cb·t > π/2**,
the same criterion as the delayed local viscosity in navier-stokes-sim (8νt/h² = π/2). At the default
μ = 1e-4 on 256², Λ_cb ≈ 52 /s, so already t ≳ 0.03 s destabilises the grid scale; smaller μ or a
coarser grid moves the threshold up (t_c ∝ h²/μ). Unequal delays give a two-delay characteristic equation
μ = D e^{−μ t_self} + N̂(k) e^{−μ t_nb}. No hidden regularisation is added; the panel shows Λ_cb·t and warns above π/2.

### Order parameter / correlations

Same panel as navier-stokes-sim: histogram of P(cos θ_ij) over pairs of cell directions (30 000 sampled pairs), exact ⟨cos θ_ij⟩ = (|Σv̂|² − N)/(N(N−1)), radial correlation C(r) (full and connected) with correlation length ξ (first zero of C_conn), and a 20 s time trace of ⟨cos θ⟩, φ and ξ.

### Passive density

A density field ρ (initially 1, reset by the init buttons) is carried by the continuity equation
`∂ρ/∂t + ∇·(ρu) = 0` with **no feedback on u** (no pressure, no EOS). Finite volume in flux form:
MUSCL reconstruction (MC limiter) with upwind face fluxes, face velocity = mean of the two cells,
SSP-RK2 sub-steps at per-direction CFL ≤ 0.25. Total mass is conserved to round-off; a ρ ≥ 0
floor is a last-resort safeguard, and the HUD reports it if it ever fires.

### Lamé convention

The viscous term is the Navier–Lamé operator ∇·σ with

```
σ = μ(∇u + ∇uᵀ) + λ(∇·u) I     ⇒   ∇·σ = μ∇²u + (λ+μ)∇(∇·u)
```

μ is the shear viscosity and λ the second (Lamé) coefficient. Per Fourier mode the operator is
−μk²û − (λ+μ)k(k·û), so

* transverse (shear, k ⊥ û) modes decay at **μk²**,
* longitudinal (compressive, k ∥ û) modes decay at **(λ + 2μ)k²**.

Well-posedness (ellipticity) needs μ ≥ 0 and **λ + 2μ ≥ 0**; the solver clamps λ to −2μ and the
UI warns. For reference the panel also shows the 2D bulk viscosity ζ = λ + μ (σ = 2μ·dev D + ζ(∇·u)I).

### Self-propulsion

Identical to navier-stokes-sim: −(1/τ)(|u| − v₀) u/|u|, integrated exactly per step,
|u| → v₀ + (|u| − v₀)e^{−Δt/τ} along u/|u| (defaults τ = 0.5 s, v₀ = 0.5).

## Numerics

Operator splitting per step (Δt = 1/60 s):

1. forces (mouse drag);
2. **delayed Lamé step, explicit forward Euler**, sub-stepped: ns = ceil(Δt·Λ/1.8) with the Gershgorin
   bound Λ = (8μ + 5|λ+μ|)/h² of the spectrum of −L, so Δt_sub·Λ ≤ 1.8 < 2 (forward-Euler limit for the
   undelayed operator; e.g. μ = 1e-4, λ = 0, 256²: Λ ≈ 85 /s, Δt_sub ≤ 0.024 s, 1 sub-step at Δt = 1/60).
   Each sub-step reads its delayed fields at its own time. Conserves the mean momentum. (The exact
   Fourier step of earlier versions cannot carry the delay and was removed.)
3. **semi-Lagrangian advection** of u and the dye (midpoint back-trace, bilinear, periodic);
4. exact self-propulsion relaxation.

There is no pressure solve, so compression waves are not sound waves: a compressive mode only
decays at (λ+2μ)k² and is carried/steepened by advection.

## Controls

* **Drag**: push the fluid and inject dye.
* **🎲 Random initial conditions**, **➡ Ordered + perturbation** (uniform flock along +x at v₀ with a
  small disordered swirl patch, σ = 0.065, ⟨cos⟩ ≈ 0.95), with seed and amplitude fields — same
  generators as navier-stokes-sim.
* Views: density ρ (diverging map centred at 1, auto or fixed ± range), velocity direction (hue, brightness = speed), speed, vorticity, divergence, dye.
* **Velocity-arrow overlay** (toggle, spacing and scale sliders).
* Delay sliders t_self, t_nb (0–2 s) with lock t_self = t_nb; info line with Λ_cb·t and the sub-step count; correlation panel.
* Sliders: μ (log), λ, self-propulsion on/off, τ, v₀, grid (64/128/256), steps per frame.
* HUD: grid, t, Δt, kinetic energy, ⟨|u|⟩, polar order |⟨u/|u|⟩|, rms ∇·u, total mass, min/max ρ.
* URL parameter `?view=density` (or speed, vorticity, divergence, direction, dye) picks the start view.
* Keys: `Space` pause, `S` step, `R` reset, `V` next view, `A` arrows, `H` hide panel.

## Tests (`npm test`)

* zero delay: shear wave decays at μk² (±1 %), longitudinal wave at (λ+2μ)k² (±1 %) for λ = 0, 3μ, −1.5μ
* self-propulsion drives ⟨|u|⟩ → v₀; exact relaxation of a uniform flow
* periodicity: a uniformly advected pattern wraps around the torus unchanged; viscosity conserves mean momentum
* ∇·u ≠ 0 is retained; λ clamping
* delay buffer reproduces a known shifted history; split operator D·u + N[u] equals the undelayed L to 1e-13; zero delays = forward Euler with L, no history kept; equal-delay checkerboard decays at Λt = 0.82 and grows at Λt = 3.3
* correlations: aligned field gives ⟨cos⟩ = 1 and C(r) = 1; ordered + perturbation start gives ⟨cos⟩ ≈ 0.95
* density: mass conserved to 1e-12 in a compressive flow with ρ ≥ 0; uniform flow translates ρ around the torus (max error < 1 %)

## Development

```
npm install
npm run dev     # local server
npm test
npm run build
```

CI runs tests and the build on every push; pushes to `main` deploy to GitHub Pages.
