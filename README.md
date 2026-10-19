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
2. **viscous step, exact in Fourier space**: transverse part × e^{−μκ²Δt}, longitudinal part ×
   e^{−(λ+2μ)κ²Δt}, κ = 2 sin(kh/2)/h (the 5-point-Laplacian symbol). Unconditionally stable,
   conserves the mean momentum;
3. **semi-Lagrangian advection** of u and the dye (midpoint back-trace, bilinear, periodic);
4. exact self-propulsion relaxation.

There is no pressure solve, so compression waves are not sound waves: a compressive mode only
decays at (λ+2μ)k² and is carried/steepened by advection.

## Controls

* **Drag**: push the fluid and inject dye.
* **🎲 Random initial conditions**, **➡ Ordered + perturbation** (uniform flock along +x at v₀ with a
  small disordered swirl patch, σ = 0.065, ⟨cos⟩ ≈ 0.95), with seed and amplitude fields — same
  generators as navier-stokes-sim.
* Views: velocity direction (hue, brightness = speed), speed, vorticity, divergence, dye.
* **Velocity-arrow overlay** (toggle, spacing and scale sliders).
* Sliders: μ (log), λ, self-propulsion on/off, τ, v₀, grid (64/128/256), steps per frame.
* HUD: grid, t, Δt, kinetic energy, ⟨|u|⟩, polar order |⟨u/|u|⟩|, rms ∇·u.
* Keys: `Space` pause, `S` step, `R` reset, `V` next view, `A` arrows, `H` hide panel.

## Tests (`npm test`)

* shear wave decays at μk² (±1 %), longitudinal wave at (λ+2μ)k² (±1 %) for λ = 0, 3μ, −1.5μ
* self-propulsion drives ⟨|u|⟩ → v₀; exact relaxation of a uniform flow
* periodicity: a uniformly advected pattern wraps around the torus unchanged; viscosity conserves mean momentum
* ∇·u ≠ 0 is retained; λ clamping

## Development

```
npm install
npm run dev     # local server
npm test
npm run build
```

CI runs tests and the build on every push; pushes to `main` deploy to GitHub Pages.
