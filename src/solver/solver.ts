/**
 * 2D compressible (barotropic) Navier–Stokes on the periodic unit square.
 *
 *   ∂ρ/∂t + ∇·(ρu) = 0
 *   ∂(ρu)/∂t + ∇·(ρu⊗u + pI − σ) = f
 *   σ = μ(∇u + ∇uᵀ) + λ(∇·u)I        (Lamé form; 2D bulk viscosity ζ = λ + μ)
 *   p = c²ρ (isothermal)  or  p = Kρ^γ with K = c²/γ (so c is the sound speed at ρ = 1)
 *
 * Finite volume: MUSCL (MC limiter, primitive variables) + Rusanov flux, centred
 * face-based viscous fluxes, SSP-RK3, CFL-adaptive dt. A passive dye s is carried
 * as the conserved quantity ρs.
 */
export type EOS = 'isothermal' | 'polytropic';

export interface Params {
  mu: number;      // shear viscosity
  lambda: number;  // second (Lamé) coefficient; clamped to ≥ −μ (ζ ≥ 0)
  c: number;       // sound speed at ρ = 1
  eos: EOS;
  gamma: number;
  cfl: number;
  rhoFloor: number;
}

export const defaultParams = (): Params => ({
  mu: 2e-4, lambda: 0, c: 1, eos: 'isothermal', gamma: 1.4, cfl: 0.4, rhoFloor: 1e-6,
});

/** Bulk viscosity in 2D: trace part of σ is (λ + μ)(∇·u)·I relative to the deviator. */
export const bulkViscosity = (p: Params) => p.lambda + p.mu;

/** Clamp λ so that ζ = λ + μ ≥ 0. Returns true if clamping happened. */
export function clampLambda(p: Params): boolean {
  if (p.lambda < -p.mu) { p.lambda = -p.mu; return true; }
  return false;
}

/** Monotonized-central (MC) limiter: full limited slope from one-sided differences a, b. */
function mc(a: number, b: number): number {
  if (a * b <= 0) return 0;
  const m = Math.min(2 * Math.abs(a), 2 * Math.abs(b), 0.5 * Math.abs(a + b));
  return a > 0 ? m : -m;
}

export class Solver {
  readonly n: number;
  readonly h: number;
  params: Params;
  t = 0;
  lastDt = 0;
  floorHits = 0;
  // conserved state
  rho: Float64Array; mx: Float64Array; my: Float64Array; rs: Float64Array;
  // external force density (cleared by the caller as needed)
  fx: Float64Array; fy: Float64Array;
  private u: Float64Array; private v: Float64Array; private s: Float64Array;
  private ux: Float64Array; private uy: Float64Array; private vx: Float64Array; private vy: Float64Array;
  private d: Float64Array[]; private s0: Float64Array[]; private s1: Float64Array[];

  constructor(n = 256, params: Params = defaultParams()) {
    this.n = n; this.h = 1 / n; this.params = params;
    const N = n * n, A = () => new Float64Array(N);
    this.rho = A().fill(1); this.mx = A(); this.my = A(); this.rs = A();
    this.fx = A(); this.fy = A();
    this.u = A(); this.v = A(); this.s = A();
    this.ux = A(); this.uy = A(); this.vx = A(); this.vy = A();
    this.d = [A(), A(), A(), A()]; this.s0 = [A(), A(), A(), A()]; this.s1 = [A(), A(), A(), A()];
  }

  get state(): Float64Array[] { return [this.rho, this.mx, this.my, this.rs]; }

  pressure(r: number): number {
    const P = this.params;
    return P.eos === 'isothermal' ? P.c * P.c * r : (P.c * P.c / P.gamma) * Math.pow(r, P.gamma);
  }
  soundSpeed(r: number): number {
    const P = this.params;
    return P.eos === 'isothermal' ? P.c : P.c * Math.pow(r, 0.5 * (P.gamma - 1));
  }

  /** Stable time step from CFL and the explicit viscous limit. */
  computeDt(): number {
    const { n, h } = this;
    let smax = 1e-12, rmin = Infinity;
    for (let k = 0; k < n * n; k++) {
      const r = this.rho[k];
      const a = Math.max(Math.abs(this.mx[k]), Math.abs(this.my[k])) / r + this.soundSpeed(r);
      if (a > smax) smax = a;
      if (r < rmin) rmin = r;
    }
    let dt = this.params.cfl * h / smax;
    const D = (2 * this.params.mu + Math.abs(this.params.lambda)) / Math.max(rmin, 1e-3);
    if (D > 0) dt = Math.min(dt, 0.2 * h * h / D);
    return dt;
  }

  /** dU/dt for state S into out. */
  private rhs(S: Float64Array[], out: Float64Array[]) {
    const { n, h, u, v, s } = this;
    const [R, MX, MY, RS] = S;
    const [dR, dMX, dMY, dRS] = out;
    const { mu, lambda } = this.params;
    const N = n * n, ih = 1 / h;
    for (let k = 0; k < N; k++) { const r = R[k]; u[k] = MX[k] / r; v[k] = MY[k] / r; s[k] = RS[k] / r; }
    const visc = mu !== 0 || lambda !== 0;
    if (visc) {
      for (let j = 0; j < n; j++) {
        const jm = ((j - 1 + n) % n) * n, jp = ((j + 1) % n) * n, jj = j * n;
        for (let i = 0; i < n; i++) {
          const im = (i - 1 + n) % n, ip = (i + 1) % n, k = jj + i;
          this.ux[k] = 0.5 * ih * (u[jj + ip] - u[jj + im]);
          this.vx[k] = 0.5 * ih * (v[jj + ip] - v[jj + im]);
          this.uy[k] = 0.5 * ih * (u[jp + i] - u[jm + i]);
          this.vy[k] = 0.5 * ih * (v[jp + i] - v[jm + i]);
        }
      }
    }
    for (let k = 0; k < N; k++) { dR[k] = 0; dMX[k] = this.fx[k]; dMY[k] = this.fy[k]; dRS[k] = 0; }

    // sweep over faces in both directions; dir 0: x-faces (i+1/2), dir 1: y-faces
    for (let dir = 0; dir < 2; dir++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          // cells along the sweep: a (i−1), b (i), c (i+1), e (i+2) — face between b and c
          let ka: number, kb: number, kc: number, ke: number;
          if (dir === 0) {
            const r = j * n;
            ka = r + (i - 1 + n) % n; kb = r + i; kc = r + (i + 1) % n; ke = r + (i + 2) % n;
          } else {
            ka = ((j - 1 + n) % n) * n + i; kb = j * n + i; kc = ((j + 1) % n) * n + i; ke = ((j + 2) % n) * n + i;
          }
          // normal (un) and tangential (ut) velocity
          const un = dir === 0 ? u : v, ut = dir === 0 ? v : u;
          // reconstruct ρ, un, ut, s
          const sb = (q: Float64Array) => 0.5 * mc(q[kb] - q[ka], q[kc] - q[kb]);
          const sc = (q: Float64Array) => 0.5 * mc(q[kc] - q[kb], q[ke] - q[kc]);
          let rL = R[kb] + sb(R), rR = R[kc] - sc(R);
          let nL = un[kb] + sb(un), nR = un[kc] - sc(un);
          let tL = ut[kb] + sb(ut), tR = ut[kc] - sc(ut);
          let cL = s[kb] + sb(s), cR = s[kc] - sc(s);
          if (!(rL > 0 && rR > 0)) { // positivity fallback: first order
            rL = R[kb]; rR = R[kc]; nL = un[kb]; nR = un[kc]; tL = ut[kb]; tR = ut[kc]; cL = s[kb]; cR = s[kc];
          }
          const pL = this.pressure(rL), pR = this.pressure(rR);
          const a = Math.max(Math.abs(nL) + this.soundSpeed(rL), Math.abs(nR) + this.soundSpeed(rR));
          const mL = rL * nL, mR = rR * nR;
          const F0 = 0.5 * (mL + mR) - 0.5 * a * (rR - rL);
          let Fn = 0.5 * (mL * nL + pL + mR * nR + pR) - 0.5 * a * (mR - mL);
          let Ft = 0.5 * (mL * tL + mR * tR) - 0.5 * a * (rR * tR - rL * tL);
          const Fs = 0.5 * (mL * cL + mR * cR) - 0.5 * a * (rR * cR - rL * cL);
          if (visc) {
            // compact normal derivatives across the face, averaged tangential ones
            let dnn: number, dtn: number, dnt: number, dtt: number; // d(un)/dn, d(ut)/dn, d(un)/dt, d(ut)/dt
            if (dir === 0) {
              dnn = (u[kc] - u[kb]) * ih; dtn = (v[kc] - v[kb]) * ih;
              dnt = 0.5 * (this.uy[kb] + this.uy[kc]); dtt = 0.5 * (this.vy[kb] + this.vy[kc]);
            } else {
              dnn = (v[kc] - v[kb]) * ih; dtn = (u[kc] - u[kb]) * ih;
              dnt = 0.5 * (this.vx[kb] + this.vx[kc]); dtt = 0.5 * (this.ux[kb] + this.ux[kc]);
            }
            Fn -= 2 * mu * dnn + lambda * (dnn + dtt);
            Ft -= mu * (dnt + dtn);
          }
          const FX = dir === 0 ? Fn : Ft, FY = dir === 0 ? Ft : Fn;
          dR[kb] -= F0 * ih; dR[kc] += F0 * ih;
          dMX[kb] -= FX * ih; dMX[kc] += FX * ih;
          dMY[kb] -= FY * ih; dMY[kc] += FY * ih;
          dRS[kb] -= Fs * ih; dRS[kc] += Fs * ih;
        }
      }
    }
  }

  /** One SSP-RK3 step; returns the dt used. */
  step(dtMax = Infinity): number {
    const dt = Math.min(this.computeDt(), dtMax);
    const U = this.state, N = this.n * this.n, { d, s0, s1 } = this;
    for (let q = 0; q < 4; q++) s0[q].set(U[q]);
    this.rhs(U, d);
    for (let q = 0; q < 4; q++) for (let k = 0; k < N; k++) s1[q][k] = s0[q][k] + dt * d[q][k];
    this.floor(s1);
    this.rhs(s1, d);
    for (let q = 0; q < 4; q++) for (let k = 0; k < N; k++) s1[q][k] = 0.75 * s0[q][k] + 0.25 * (s1[q][k] + dt * d[q][k]);
    this.floor(s1);
    this.rhs(s1, d);
    for (let q = 0; q < 4; q++) for (let k = 0; k < N; k++) U[q][k] = s0[q][k] / 3 + (2 / 3) * (s1[q][k] + dt * d[q][k]);
    this.floor(U);
    this.t += dt; this.lastDt = dt;
    return dt;
  }

  /** Positivity safeguard (only active in extreme cases; breaks exact mass conservation when it fires). */
  private floor(S: Float64Array[]) {
    const R = S[0], f = this.params.rhoFloor;
    for (let k = 0; k < R.length; k++) if (!(R[k] >= f)) { R[k] = f; S[1][k] = 0; S[2][k] = 0; this.floorHits++; }
  }

  totalMass(): number { let m = 0; for (const r of this.rho) m += r; return m * this.h * this.h; }
  kineticEnergy(): number {
    let e = 0;
    for (let k = 0; k < this.rho.length; k++) e += (this.mx[k] ** 2 + this.my[k] ** 2) / this.rho[k];
    return 0.5 * e * this.h * this.h;
  }
  maxMach(): number {
    let m = 0;
    for (let k = 0; k < this.rho.length; k++) {
      const r = this.rho[k], sp = Math.hypot(this.mx[k], this.my[k]) / r;
      m = Math.max(m, sp / this.soundSpeed(r));
    }
    return m;
  }

  /** Fill output with a diagnostic field. */
  field(kind: 'density' | 'vorticity' | 'divergence' | 'speed' | 'mach' | 'dye', out: Float64Array) {
    const { n, h } = this, N = n * n;
    if (kind === 'density') { out.set(this.rho); return; }
    for (let k = 0; k < N; k++) { const r = this.rho[k]; this.u[k] = this.mx[k] / r; this.v[k] = this.my[k] / r; }
    for (let j = 0; j < n; j++) {
      const jm = ((j - 1 + n) % n) * n, jp = ((j + 1) % n) * n, jj = j * n;
      for (let i = 0; i < n; i++) {
        const k = jj + i, im = jj + (i - 1 + n) % n, ip = jj + (i + 1) % n;
        const r = this.rho[k];
        switch (kind) {
          case 'vorticity': out[k] = ((this.v[ip] - this.v[im]) - (this.u[jp + i] - this.u[jm + i])) / (2 * h); break;
          case 'divergence': out[k] = ((this.u[ip] - this.u[im]) + (this.v[jp + i] - this.v[jm + i])) / (2 * h); break;
          case 'speed': out[k] = Math.hypot(this.u[k], this.v[k]); break;
          case 'mach': out[k] = Math.hypot(this.u[k], this.v[k]) / this.soundSpeed(r); break;
          case 'dye': out[k] = this.rs[k] / r; break;
        }
      }
    }
  }
}
