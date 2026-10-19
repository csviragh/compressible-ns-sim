# Compressible Navier–Stokes 2D

**Live demo: https://csviragh.github.io/compressible-ns-sim/**

Interactive 2D *compressible* (barotropic) Navier–Stokes simulator in the browser, with a
Lamé-form viscous stress (shear viscosity μ and second coefficient λ). Sister project of
[navier-stokes-sim](https://github.com/csviragh/navier-stokes-sim) (incompressible).
Vite + TypeScript, plain canvas, no runtime dependencies.

![screenshot](screenshot.png)

## Equations

On the periodic unit square (default 256 × 256, side 1):

```
∂ρ/∂t + ∇·(ρu) = 0
∂(ρu)/∂t + ∇·(ρu⊗u + p I − σ) = f
σ = μ(∇u + ∇uᵀ) + λ(∇·u) I
```

Equation of state (barotropic, no energy equation):

* **isothermal** (default) `p = c²ρ`
* **polytropic** `p = Kρ^γ`, with `K = c²/γ` so that `c` is the sound speed at ρ = 1.

A passive dye `s` is carried as the conserved quantity ρs (for visualisation only).

### Viscosity conventions (2D)

Write `D = ½(∇u + ∇uᵀ)`. In 2D the deviator is `D − ½(tr D) I`, so

```
σ = 2μ (D − ½(∇·u) I) + ζ (∇·u) I,     ζ = λ + μ   (2D bulk viscosity)
```

(In 3D the same σ gives ζ = λ + 2μ/3.) Thermodynamics requires μ ≥ 0 and ζ ≥ 0, i.e.
**λ ≥ −μ**; the app and solver clamp λ to −μ and show a warning. The **Stokes hypothesis**
(ζ = 0) is λ = −μ in 2D (checkbox in the UI).

Linear damping rates about ρ₀ = 1 (wavenumber k):

* shear (transverse) wave: amplitude ∝ e^{−μk²t/ρ₀}
* sound (longitudinal) wave: amplitude ∝ e^{−Γt}, **Γ = (2μ + λ)k²/(2ρ₀) = (μ + ζ)k²/(2ρ₀)**,
  oscillating at ω ≈ ck.

## Numerics

* Cell-centred finite volume, conservative variables (ρ, ρu, ρv, ρs).
* MUSCL reconstruction of primitives (ρ, u, v, s) with the monotonized-central (MC) limiter, **Rusanov**
  (local Lax–Friedrichs) flux with wave speed |uₙ| + c(ρ).
* Viscous fluxes centred on faces: compact normal derivatives across the face, tangential
  derivatives averaged from neighbouring cell-centred differences (no odd–even decoupling).
* **SSP-RK3** time stepping, adaptive Δt = min(CFL·h / max(|u|+c), 0.2 h²ρ_min/(2μ+|λ|)), CFL 0.4.
* Positivity: fall back to first order at a face if a reconstructed density is ≤ 0; a density
  floor (1e-6) is a last-resort safeguard. The HUD reports if it ever fires (it is the only thing
  that can break mass conservation).
* Periodic boundaries; mass and momentum are conserved to round-off (flux form).

## Controls

* **Drag**: push the fluid (Gaussian body force along the drag). **Click**: add a density/pressure pulse.
* Presets: acoustic pulse, shear layers (Kelvin–Helmholtz), colliding jets, random turbulence,
  rotating stirrer (a moving forcing disc; a periodic box has no lid), fluid at rest.
* Views: density, vorticity, divergence (shows sound waves), speed, Mach number, dye.
* Sliders: μ (log), λ, Stokes checkbox, EOS, sound speed c, γ, grid (64/128/256), steps per frame.
* HUD: grid, t, Δt, total mass, kinetic energy, max Mach number.
* Keys: `Space` pause, `S` step, `R` reset, `V` next view, `H` hide panel.

## Tests (`npm test`)

* mass conserved to 1e-12 relative (turbulence with viscosity), momentum conserved
* acoustic pulse front moves at c (±3 %) for c = 0.5 and 1
* shear wave decays at μk²/ρ (±2 %)
* sound wave damps at (2μ + λ)k²/(2ρ) (±5 %) for λ = 0, 2μ and −μ (Stokes)
* λ clamping (ζ ≥ 0), polytropic colliding jets stay positive and finite

## Development

```
npm install
npm run dev     # local server
npm test
npm run build
```

CI runs tests and the build on every push; pushes to `main` deploy to GitHub Pages.
