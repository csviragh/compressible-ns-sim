/**
 * 2D *compressible-velocity* active fluid on the periodic unit square (no pressure, no
 * projection, no equation of state):
 *
 *   ∂u/∂t + (u·∇)u = μ∇²u + (λ+μ)∇(∇·u) − (1/τ)(|u| − v₀) u/|u| + f
 *
 * The viscous operator is the Navier–Lamé operator ∇·σ with σ = μ(∇u + ∇uᵀ) + λ(∇·u)I
 * (unit density). Per Fourier mode it damps the transverse (shear, k⊥u) part at μk² and the
 * longitudinal (compressive, k∥u) part at (λ+2μ)k². Stability/ellipticity needs μ ≥ 0 and
 * λ + 2μ ≥ 0 (λ is clamped).
 *
 * Delays: the discrete Lamé operator L = D + N is split into its self part D (the cell's own,
 * diagonal stencil coefficient, D = −(4μ + 2(λ+μ))/h² for both components) and its neighbour
 * part N = L − D. The model is  L_delayed[u](x,t) = D·u(x, t − t_self) + N[u(·, t − t_nb)](x),
 * Eulerian (fixed grid point) read from a ring buffer. t_self = t_nb = 0 gives L exactly.
 *
 * Step (operator splitting): forces → explicit (forward-Euler, sub-stepped) delayed Lamé step →
 * semi-Lagrangian advection (bilinear, periodic) → exact self-propulsion relaxation.
 */
import { FieldHistory } from './delay';

export interface Params {
  mu: number;       // shear viscosity μ
  lambda: number;   // Lamé second coefficient λ (clamped to ≥ −2μ)
  selfProp: boolean;
  tau: number;      // self-propulsion relaxation time τ (s)
  v0: number;       // target speed v₀
  dt: number;       // time step (s)
  dyeFade: number;  // 1/s
  /** Delays (s) of the Lamé operator: self (cell-own, diagonal) part reads u(t − t_self), neighbour part u(t − t_nb). */
  tSelf: number; tNb: number;
}

export const defaultParams = (): Params => ({
  mu: 1e-4, lambda: 0, selfProp: true, tau: 0.5, v0: 0.5, dt: 1 / 60, dyeFade: 0,
  tSelf: 0, tNb: 0,
});

/** Longitudinal (compressive) viscosity λ + 2μ. */
export const longitudinalViscosity = (p: Params) => p.lambda + 2 * p.mu;
/** 2D bulk viscosity ζ = λ + μ (informational). */
export const bulkViscosity = (p: Params) => p.lambda + p.mu;

/** Clamp μ ≥ 0 and λ ≥ −2μ. Returns true if λ was clamped. */
export function clampParams(p: Params): boolean {
  if (p.mu < 0) p.mu = 0;
  if (p.lambda < -2 * p.mu) { p.lambda = -2 * p.mu; return true; }
  return false;
}

export function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export class Solver {
  readonly n: number; readonly h: number;
  params: Params;
  t = 0;
  u: Float64Array; v: Float64Array; dye: Float64Array;
  /** Passive density ρ (continuity equation, no feedback on u). */
  rho: Float64Array; floorHits = 0;
  private r1: Float64Array; private dr: Float64Array;
  fx: Float64Array; fy: Float64Array;
  private tmp: Float64Array; private tmp2: Float64Array;
  readonly history = new FieldHistory();
  private du: Float64Array; private dv: Float64Array; private su: Float64Array; private sv: Float64Array;
  private lx: Float64Array; private ly: Float64Array;
  /** Number of explicit Lamé sub-steps used in the last step. */
  lastSubsteps = 1;

  constructor(n = 256, params: Params = defaultParams()) {
    this.n = n; this.h = 1 / n; this.params = params;
    const N = n * n, A = () => new Float64Array(N);
    this.u = A(); this.v = A(); this.dye = A(); this.rho = A().fill(1); this.r1 = A(); this.dr = A(); this.fx = A(); this.fy = A();
    this.tmp = A(); this.tmp2 = A();
    this.du = A(); this.dv = A(); this.su = A(); this.sv = A(); this.lx = A(); this.ly = A();
  }

  reset() { this.u.fill(0); this.v.fill(0); this.dye.fill(0); this.fx.fill(0); this.fy.fill(0); this.rho.fill(1); this.floorHits = 0; this.t = 0; this.history.clear(); }

  /** Self (diagonal) coefficient of the discrete Lamé stencil, the same for both components. */
  selfCoeff(): number { const { mu, lambda } = this.params; return -(4 * mu + 2 * (lambda + mu)) / (this.h * this.h); }

  /**
   * Upper bound of the spectral radius of −L (Gershgorin): Λ = (8μ + 5|λ+μ|)/h².
   * Forward Euler is stable for Δt_sub·Λ ≤ 2 (undelayed).
   */
  spectralBound(): number { const { mu, lambda } = this.params; return (8 * mu + 5 * Math.abs(lambda + mu)) / (this.h * this.h); }

  /**
   * Discrete Lamé operator, split: out = D·(su, sv) + N[(nu, nv)], N = L − D.
   * Stencil (periodic, collocated): μ·5-point Laplacian + (λ+μ)[∂xx u + ∂xy v, ∂xy u + ∂yy v],
   * ∂xx 3-point, ∂xy 4-corner centred. With s = n the result is exactly L[u].
   */
  lameSplit(su: Float64Array, sv: Float64Array, nu: Float64Array, nv: Float64Array, ox: Float64Array, oy: Float64Array) {
    const { n, h } = this, { mu, lambda } = this.params, b = lambda + mu, ih2 = 1 / (h * h), D = this.selfCoeff();
    for (let j = 0; j < n; j++) {
      const jm = ((j - 1 + n) % n) * n, jp = ((j + 1) % n) * n, jj = j * n;
      for (let i = 0; i < n; i++) {
        const im = (i - 1 + n) % n, ip = (i + 1) % n, k = jj + i;
        const uE = nu[jj + ip], uW = nu[jj + im], uN = nu[jp + i], uS = nu[jm + i];
        const vE = nv[jj + ip], vW = nv[jj + im], vN = nv[jp + i], vS = nv[jm + i];
        const cxyV = (nv[jp + ip] - nv[jp + im] - nv[jm + ip] + nv[jm + im]) * 0.25;
        const cxyU = (nu[jp + ip] - nu[jp + im] - nu[jm + ip] + nu[jm + im]) * 0.25;
        // neighbour part (all off-centre stencil entries)
        const nx = (mu * (uE + uW + uN + uS) + b * (uE + uW) + b * cxyV) * ih2;
        const ny = (mu * (vE + vW + vN + vS) + b * (vN + vS) + b * cxyU) * ih2;
        ox[k] = nx + D * su[k]; oy[k] = ny + D * sv[k];
      }
    }
  }

  /** Undelayed discrete Lamé operator L[u] (independent single-stencil form, used by tests). */
  lame(u: Float64Array, v: Float64Array, ox: Float64Array, oy: Float64Array) {
    const { n, h } = this, { mu, lambda } = this.params, b = lambda + mu, ih2 = 1 / (h * h);
    for (let j = 0; j < n; j++) {
      const jm = ((j - 1 + n) % n) * n, jp = ((j + 1) % n) * n, jj = j * n;
      for (let i = 0; i < n; i++) {
        const im = (i - 1 + n) % n, ip = (i + 1) % n, k = jj + i;
        const lapU = u[jj + ip] + u[jj + im] + u[jp + i] + u[jm + i] - 4 * u[k];
        const lapV = v[jj + ip] + v[jj + im] + v[jp + i] + v[jm + i] - 4 * v[k];
        const uxx = u[jj + ip] - 2 * u[k] + u[jj + im], vyy = v[jp + i] - 2 * v[k] + v[jm + i];
        const vxy = (v[jp + ip] - v[jp + im] - v[jm + ip] + v[jm + im]) * 0.25;
        const uxy = (u[jp + ip] - u[jp + im] - u[jm + ip] + u[jm + im]) * 0.25;
        ox[k] = (mu * lapU + b * (uxx + vxy)) * ih2; oy[k] = (mu * lapV + b * (vyy + uxy)) * ih2;
      }
    }
  }

  /**
   * Delayed Lamé step over dt, forward Euler with ns = ceil(dt·Λ/1.8) sub-steps (Λ = spectralBound()).
   * Sub-step m at time t_m reads the self field at t_m − t_self and the neighbour field at
   * t_m − t_nb (history frames + the current sub-step field as the newest point).
   */
  viscous(dt: number) {
    const { mu, lambda, tSelf, tNb } = this.params;
    if (mu === 0 && lambda === 0) { this.lastSubsteps = 0; return; }
    const ns = Math.max(1, Math.ceil(dt * this.spectralBound() / 1.8)), d = dt / ns;
    this.lastSubsteps = ns;
    const { u, v, du, dv, su, sv, lx, ly } = this, N = u.length;
    for (let m = 0; m < ns; m++) {
      const tm = this.t + m * d;
      let S0u: Float64Array = u, S0v: Float64Array = v, N0u: Float64Array = u, N0v: Float64Array = v;
      if (tSelf > 0) { this.history.read(tm - tSelf, tm, u, v, su, sv); S0u = su; S0v = sv; }
      if (tNb > 0) { this.history.read(tm - tNb, tm, u, v, du, dv); N0u = du; N0v = dv; }
      this.lameSplit(S0u, S0v, N0u, N0v, lx, ly);
      for (let k = 0; k < N; k++) { u[k] += d * lx[k]; v[k] += d * ly[k]; }
    }
  }

  /** dρ/dt = −∇·(ρu): MUSCL (MC limiter) upwind fluxes, face velocity = mean of the two cells. */
  private densityRhs(R: Float64Array, out: Float64Array) {
    const { n, u, v } = this, ih = 1 / this.h;
    out.fill(0);
    const mc = (a: number, b: number) => { if (a * b <= 0) return 0; const m = Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b)); return a > 0 ? m : -m; };
    for (let dir = 0; dir < 2; dir++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let ka: number, kb: number, kc: number, ke: number;
      if (dir === 0) { const r = j * n; ka = r + (i - 1 + n) % n; kb = r + i; kc = r + (i + 1) % n; ke = r + (i + 2) % n; }
      else { ka = ((j - 1 + n) % n) * n + i; kb = j * n + i; kc = ((j + 1) % n) * n + i; ke = ((j + 2) % n) * n + i; }
      const w = dir === 0 ? u : v, uf = 0.5 * (w[kb] + w[kc]);
      const rf = uf >= 0 ? R[kb] + 0.5 * mc(R[kb] - R[ka], R[kc] - R[kb]) : R[kc] - 0.5 * mc(R[kc] - R[kb], R[ke] - R[kc]);
      const F = uf * rf * ih;
      out[kb] -= F; out[kc] += F;
    }
  }

  /** Continuity step with SSP-RK2 sub-steps (per-direction CFL ≤ 0.25); mass is conserved to round-off. */
  advectDensity(dt: number) {
    let umax = 1e-12;
    for (let k = 0; k < this.u.length; k++) umax = Math.max(umax, Math.abs(this.u[k]), Math.abs(this.v[k]));
    const ns = Math.max(1, Math.ceil(umax * dt / this.h / 0.25)), d = dt / ns, { rho, r1, dr } = this, N = rho.length;
    for (let s = 0; s < ns; s++) {
      this.densityRhs(rho, dr);
      for (let k = 0; k < N; k++) r1[k] = rho[k] + d * dr[k];
      this.densityRhs(r1, dr);
      for (let k = 0; k < N; k++) rho[k] = 0.5 * (rho[k] + r1[k] + d * dr[k]);
      // positivity safeguard (should not fire at this CFL; breaks exact conservation if it does)
      for (let k = 0; k < N; k++) if (!(rho[k] >= 0)) { rho[k] = 0; this.floorHits++; }
    }
  }
  totalMass() { let m = 0; for (const r of this.rho) m += r; return m * this.h * this.h; }

  /** Bilinear periodic sample of q at grid coordinates (x, y) in cells. */
  private sample(q: Float64Array, x: number, y: number) {
    const n = this.n;
    const fx = Math.floor(x), fy = Math.floor(y), sx = x - fx, sy = y - fy;
    const i0 = ((fx % n) + n) % n, j0 = ((fy % n) + n) % n, i1 = (i0 + 1) % n, j1 = (j0 + 1) % n;
    return (1 - sy) * ((1 - sx) * q[j0 * n + i0] + sx * q[j0 * n + i1]) + sy * ((1 - sx) * q[j1 * n + i0] + sx * q[j1 * n + i1]);
  }

  /** Semi-Lagrangian advection of u, v and dye (midpoint back-trace). */
  advect(dt: number) {
    const { n, u, v, tmp, tmp2 } = this, s = dt / this.h;
    const nu = new Float64Array(n * n), nd = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      const xm = i - 0.5 * s * u[k], ym = j - 0.5 * s * v[k];
      const x = i - s * this.sample(u, xm, ym), y = j - s * this.sample(v, xm, ym);
      nu[k] = this.sample(u, x, y); tmp2[k] = this.sample(v, x, y); nd[k] = this.sample(this.dye, x, y);
    }
    u.set(nu); v.set(tmp2); this.dye.set(nd); void tmp;
  }

  /** Exact relaxation of the local speed toward v₀ along u/|u| (same as navier-stokes-sim). */
  selfPropulsion(dt: number) {
    const { tau, v0 } = this.params;
    if (!(tau > 0)) return;
    const decay = Math.exp(-dt / tau), gain = Math.max(0, v0) * (1 - decay);
    for (let k = 0; k < this.u.length; k++) {
      const sp = Math.hypot(this.u[k], this.v[k]);
      if (sp < 1e-9) continue;
      const f = decay + gain / sp; this.u[k] *= f; this.v[k] *= f;
    }
  }

  /** Longest delay the history must cover (0 = nothing delayed). */
  delayMax(): number { return Math.max(this.params.tSelf, this.params.tNb, 0); }

  step(dt = this.params.dt) {
    clampParams(this.params);
    const N = this.u.length, tauMax = this.delayMax();
    if (tauMax > 0) {
      const h = this.history;
      h.configure(N, N, dt, tauMax + dt, 256 * 2 ** 20);
      if (h.wants(this.t)) h.push(this.t, this.u, this.v);
      h.prune(this.t + dt);
    } else if (this.history.frames.length) this.history.clear();
    for (let k = 0; k < N; k++) { this.u[k] += dt * this.fx[k]; this.v[k] += dt * this.fy[k]; }
    this.viscous(dt);
    this.advectDensity(dt);
    this.advect(dt);
    if (this.params.selfProp) this.selfPropulsion(dt);
    if (this.params.dyeFade > 0) { const f = Math.exp(-this.params.dyeFade * dt); for (let k = 0; k < N; k++) this.dye[k] *= f; }
    this.t += dt;
  }

  // ---------- initial conditions (same generators as navier-stokes-sim) ----------
  /** Weak random swirls (amplitude ≈ peak speed), deterministic per seed. Divergence-free start. */
  seedRandom(seed: number, amplitude = 0.05, vortices = 40) {
    this.reset();
    const { n, h } = this, r = rng(seed), rad = Math.max(2, n / 12), psi = new Float64Array(n * n);
    for (let m = 0; m < vortices; m++) {
      const cx = r() * n, cy = r() * n, s = (r() < 0.5 ? -1 : 1) * amplitude * rad * h * 1.65 * (0.5 + r());
      this.addSwirlPsi(psi, cx, cy, rad, s, 1);
    }
    this.psiToVelocity(psi, 0, 0);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) this.dye[j * n + i] = ((Math.floor(i / (n / 8)) + Math.floor(j / (n / 8))) & 1);
  }

  /** Ordered flock along heading at v₀ plus a Gaussian-tapered disordered swirl patch (⟨cos⟩ ≈ 0.95). */
  seedOrdered(seed: number, opts: { heading?: number; sigma?: number; amplitude?: number; core?: number } = {}) {
    this.reset();
    const { n, h } = this, r = rng(seed), v0 = this.params.v0;
    const sigma = (opts.sigma ?? 0.065) / h, amp = (opts.amplitude ?? 2) * v0, core = (opts.core ?? 2.5 / 72) / h;
    const cx = r() * n, cy = r() * n;
    const nv = Math.max(1, Math.round(((2 * sigma) ** 2) / ((2 * core) ** 2) * 4));
    const vort: [number, number, number][] = [];
    for (let m = 0; m < nv; m++) {
      const rr = 2 * sigma * Math.sqrt(r()), th = 2 * Math.PI * r();
      vort.push([cx + rr * Math.cos(th), cy + rr * Math.sin(th), (r() < 0.5 ? -1 : 1) * amp * core * h * 1.65 * (0.5 + r())]);
    }
    const wrap = (d: number) => d - n * Math.round(d / n), psi = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const ex = wrap(i - cx), ey = wrap(j - cy), env = Math.exp(-(ex * ex + ey * ey) / (2 * sigma * sigma));
      if (env < 1e-6) continue;
      let sum = 0;
      for (const [vx, vy, st] of vort) { const dx = wrap(i - vx), dy = wrap(j - vy); sum += st * Math.exp(-(dx * dx + dy * dy) / (2 * core * core)); }
      psi[j * n + i] = env * sum;
    }
    const hd = opts.heading ?? 0;
    this.psiToVelocity(psi, v0 * Math.cos(hd), v0 * Math.sin(hd));
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) this.dye[j * n + i] = (Math.floor(j / (n / 16)) & 1);
  }

  private addSwirlPsi(psi: Float64Array, cx: number, cy: number, rad: number, s: number, _w: number) {
    const n = this.n, reach = 3 * rad;
    for (let j = 0; j < n; j++) {
      let dy = j - cy; dy -= n * Math.round(dy / n); if (Math.abs(dy) > reach) continue;
      for (let i = 0; i < n; i++) {
        let dx = i - cx; dx -= n * Math.round(dx / n); if (Math.abs(dx) > reach) continue;
        psi[j * n + i] += s * Math.exp(-(dx * dx + dy * dy) / (2 * rad * rad));
      }
    }
  }
  private psiToVelocity(psi: Float64Array, U0: number, V0: number) {
    const { n, h } = this;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, ip = j * n + (i + 1) % n, im = j * n + (i - 1 + n) % n, jp = ((j + 1) % n) * n + i, jm = ((j - 1 + n) % n) * n + i;
      this.u[k] = U0 + (psi[jp] - psi[jm]) / (2 * h);
      this.v[k] = V0 - (psi[ip] - psi[im]) / (2 * h);
    }
  }

  // ---------- diagnostics ----------
  kineticEnergy() { let e = 0; for (let k = 0; k < this.u.length; k++) e += this.u[k] ** 2 + this.v[k] ** 2; return 0.5 * e * this.h * this.h; }
  meanSpeed() { let s = 0; for (let k = 0; k < this.u.length; k++) s += Math.hypot(this.u[k], this.v[k]); return s / this.u.length; }
  /** Polar order |⟨u/|u|⟩|. */
  polarOrder() {
    let sx = 0, sy = 0, c = 0;
    for (let k = 0; k < this.u.length; k++) { const s = Math.hypot(this.u[k], this.v[k]); if (s > 1e-12) { sx += this.u[k] / s; sy += this.v[k] / s; c++; } }
    return c ? Math.hypot(sx, sy) / c : 0;
  }
  rmsDivergence() { const f = new Float64Array(this.u.length); this.field('divergence', f); let s = 0; for (const x of f) s += x * x; return Math.sqrt(s / f.length); }

  field(kind: 'density' | 'speed' | 'vorticity' | 'divergence' | 'direction' | 'dye', out: Float64Array) {
    const { n, h, u, v } = this;
    for (let j = 0; j < n; j++) {
      const jm = ((j - 1 + n) % n) * n, jp = ((j + 1) % n) * n, jj = j * n;
      for (let i = 0; i < n; i++) {
        const k = jj + i, im = jj + (i - 1 + n) % n, ip = jj + (i + 1) % n;
        switch (kind) {
          case 'density': out[k] = this.rho[k]; break;
          case 'speed': out[k] = Math.hypot(u[k], v[k]); break;
          case 'vorticity': out[k] = ((v[ip] - v[im]) - (u[jp + i] - u[jm + i])) / (2 * h); break;
          case 'divergence': out[k] = ((u[ip] - u[im]) + (v[jp + i] - v[jm + i])) / (2 * h); break;
          case 'direction': out[k] = Math.atan2(v[k], u[k]); break;
          case 'dye': out[k] = this.dye[k]; break;
        }
      }
    }
  }
}
