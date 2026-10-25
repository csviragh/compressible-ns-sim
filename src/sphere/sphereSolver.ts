/**
 * Compressible active fluid on a rotating unit sphere (CPU reference solver).
 *
 * State per cell: tangent velocity u stored as 3D Cartesian components (re-projected onto the tangent
 * plane after every sub-step) and density ρ. Time units: seconds; lengths: sphere radius R = 1.
 *
 *   ∂u/∂t + ∇_u u = μ(Δ_B u + K u) + (λ+μ)∇(∇·u)            Lamé (switchable), K = 1/R² = 1
 *                   − (1/τ)(|u| − v₀) u/|u|                   self-propulsion
 *                   − 2Ω (ẑ·n) n × u                           Coriolis (f = 2Ω cos colatitude)
 *                   + A Σ_y [ρ(y, t − d/v_int) − ρ(x, t)] K(d) ê_{y→x} dA_y   retarded repulsion
 *   ∂ρ/∂t + ∇·(ρu) = 0
 *
 * Viscous term: for a tangent field on the unit sphere, P Δ(u_cart) = Δ_B u − u (Δ applied to each
 * Cartesian component, P = tangent projector, Δ_B = Bochner/rough Laplacian). The Lamé stress divergence
 * ∇·[μ(∇u+∇uᵀ) + λ(∇·u)g] = μ(Δ_B u + Ric·u) + (λ+μ)∇(∇·u) with Ric = 1, so we use
 * μ(P Δ u_cart + 2u) + (λ+μ)∇(∇·u). Solid-body rotation (a Killing field) is then exactly stress-free.
 * Centrifugal force is assumed absorbed into the (absent) geopotential, as in shallow-water models.
 */
import { CubedSphere, gcDist, type V3 } from './grid';

export interface SphereParams {
  lameOn: boolean; mu: number; lambda: number;
  selfProp: boolean; tau: number; v0: number;
  omega: number;                    // rotation rate Ω (rad/s), sign = sense of rotation about +z
  repOn: boolean; repA: number; R0: number; alpha: number; vInt: number;  // vInt = Infinity: instantaneous
  dt: number;
}
export const defaultSphereParams = (): SphereParams => ({
  lameOn: true, mu: 1e-3, lambda: 0, selfProp: true, tau: 0.5, v0: 0.5, omega: 1,
  repOn: true, repA: 100, R0: 0.15, alpha: 1, vInt: Infinity, dt: 1 / 60,
});
export function repulsionKernel(d: number, R0: number, alpha: number) {
  return d <= 0 || d >= R0 ? 0 : ((R0 / d) ** alpha - 1) * (1 - d / R0) ** 2;
}
export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
/** v ↦ minimal rotation taking unit a to unit b, applied to v (parallel transport along the great circle) */
export function transport(v: V3, a: V3, b: V3): V3 {
  const c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const k: V3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  if (c < -0.999999) return [-v[0], -v[1], -v[2]];
  const kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2], f = kv / (1 + c);
  return [v[0] * c + (k[1] * v[2] - k[2] * v[1]) + k[0] * f,
    v[1] * c + (k[2] * v[0] - k[0] * v[2]) + k[1] * f,
    v[2] * c + (k[0] * v[1] - k[1] * v[0]) + k[2] * f];
}

export class SphereSolver {
  g: CubedSphere; params: SphereParams; N: number;
  u: Float64Array; rho: Float64Array; time = 0;
  private tmp: Float64Array; private tmp2: Float64Array; private sc = new Int32Array(4); private sw = new Float64Array(4);
  // repulsion pair list
  private pairKey = ''; pStart = new Int32Array(0); pJ = new Int32Array(0); pW = new Float64Array(0); pE = new Float64Array(0); pD = new Float64Array(0);
  // ρ history ring buffer (one slot per step)
  private hist: Float64Array[] = []; private histHead = 0; private histCount = 0;
  lastForce: Float64Array;
  constructor(n: number, params: SphereParams) {
    this.g = new CubedSphere(n); this.params = params; this.N = this.g.N;
    this.u = new Float64Array(3 * this.N); this.rho = new Float64Array(this.N).fill(1);
    this.tmp = new Float64Array(3 * this.N); this.tmp2 = new Float64Array(3 * this.N); this.lastForce = new Float64Array(3 * this.N);
  }
  project() {
    const { u, g } = this;
    for (let k = 0; k < this.N; k++) {
      const o = 3 * k, x = g.pos[o], y = g.pos[o + 1], z = g.pos[o + 2], d = u[o] * x + u[o + 1] * y + u[o + 2] * z;
      u[o] -= d * x; u[o + 1] -= d * y; u[o + 2] -= d * z;
    }
  }
  totalMass() { let m = 0; for (let k = 0; k < this.N; k++) m += this.rho[k] * this.g.area[k]; return m; }

  // ---------- operators ----------
  private gu = new Float64Array(0);
  /**
   * Normal velocity u·n at every edge midpoint from a linear (least-squares) reconstruction on both
   * sides, averaged: exact for linear Cartesian components, so the FV divergence stays consistent at the
   * panel-edge kinks of the cubed sphere (a plain (u_i + u_j)/2 is not, and piles mass up along panel edges).
   */
  faceNormalVelocity(u: Float64Array, out: Float64Array) {
    const g = this.g, N = this.N; if (this.gu.length !== 9 * N) this.gu = new Float64Array(9 * N);
    const G = this.gu;
    for (let k = 0; k < N; k++) for (let q = 0; q < 4; q++) {
      const m = g.nbr[4 * k + q], o = 12 * k + 3 * q;
      for (let c = 0; c < 3; c++) { const df = u[3 * m + c] - u[3 * k + c]; for (let d = 0; d < 3; d++) { if (q === 0) G[9 * k + 3 * c + d] = 0; G[9 * k + 3 * c + d] += g.lsq[o + d] * df; } }
    }
    for (let e = 0; e < g.ei.length; e++) {
      const i = g.ei[e], j = g.ej[e], o = 3 * e, oi = 6 * e, oj = 6 * e + 3; let un = 0;
      for (let c = 0; c < 3; c++) {
        let a = u[3 * i + c] + u[3 * j + c];
        for (let d = 0; d < 3; d++) a += G[9 * i + 3 * c + d] * g.eoff[oi + d] + G[9 * j + 3 * c + d] * g.eoff[oj + d];
        un += 0.5 * a * g.enrm[o + c];
      }
      out[e] = un;
    }
  }
  /** divergence of the tangent field u (finite volume) */
  divergence(u: Float64Array, out: Float64Array) {
    const g = this.g; out.fill(0); const un = new Float64Array(g.ei.length); this.faceNormalVelocity(u, un);
    for (let e = 0; e < g.ei.length; e++) {
      const i = g.ei[e], j = g.ej[e], F = g.elen[e] * un[e]; out[i] += F; out[j] -= F;
    }
    for (let k = 0; k < this.N; k++) out[k] /= g.area[k];
  }
  /** surface divergence Σ_c (∇_s u_c)_c from least-squares gradients (consistent at panel edges; used in the Lamé term) */
  divergenceLsq(u: Float64Array, out: Float64Array) {
    const g = this.g;
    for (let k = 0; k < this.N; k++) {
      let d = 0;
      for (let q = 0; q < 4; q++) { const m = g.nbr[4 * k + q], o = 12 * k + 3 * q; for (let c = 0; c < 3; c++) d += g.lsq[o + c] * (u[3 * m + c] - u[3 * k + c]); }
      out[k] = d;
    }
  }
  /** least-squares tangent gradient of scalar component c (stride st) of f into g (3N) */
  gradient(f: Float64Array, g3: Float64Array, st = 1, c = 0) {
    const g = this.g;
    for (let k = 0; k < this.N; k++) {
      let x = 0, y = 0, z = 0; const fk = f[st * k + c];
      for (let q = 0; q < 4; q++) {
        const df = f[st * g.nbr[4 * k + q] + c] - fk, o = 12 * k + 3 * q;
        x += g.lsq[o] * df; y += g.lsq[o + 1] * df; z += g.lsq[o + 2] * df;
      }
      g3[3 * k] = x; g3[3 * k + 1] = y; g3[3 * k + 2] = z;
    }
  }
  /** out += s ∇φ */
  addGradient(phi: Float64Array, s: number, out: Float64Array) {
    const gr = this.tmp2; this.gradient(phi, gr);
    for (let k = 0; k < 3 * this.N; k++) out[k] += s * gr[k];
  }
  private gbuf = new Float64Array(0);
  /**
   * out += s Δ(f_c) for each Cartesian component (finite volume, face-normal gradient with
   * non-orthogonal correction: ∇f·n = ḡ·n + ((f_j − f_i)/d − ḡ·t̂)/(t̂·n), exact for linear f)
   */
  addCartLaplacian(u: Float64Array, s: number, out: Float64Array) {
    const g = this.g; if (this.gbuf.length !== 3 * this.N) this.gbuf = new Float64Array(3 * this.N);
    const gr = this.gbuf;
    for (let c = 0; c < 3; c++) {
      this.gradient(u, gr, 3, c);
      for (let e = 0; e < g.ei.length; e++) {
        const i = g.ei[e], j = g.ej[e], o = 3 * e;
        const nx = g.enrm[o], ny = g.enrm[o + 1], nz = g.enrm[o + 2], tx = g.etan[o], ty = g.etan[o + 1], tz = g.etan[o + 2];
        const gx = 0.5 * (gr[3 * i] + gr[3 * j]), gy = 0.5 * (gr[3 * i + 1] + gr[3 * j + 1]), gz = 0.5 * (gr[3 * i + 2] + gr[3 * j + 2]);
        const tn = tx * nx + ty * ny + tz * nz;
        const fn = gx * nx + gy * ny + gz * nz + ((u[3 * j + c] - u[3 * i + c]) / g.edist[e] - (gx * tx + gy * ty + gz * tz)) / tn;
        const F = s * g.elen[e] * fn; out[3 * i + c] += F / g.area[i]; out[3 * j + c] -= F / g.area[j];
      }
    }
  }
  /** Lamé acceleration μ(PΔu + 2u) + (λ+μ)∇(∇·u), written into acc (unprojected) */
  lameAccel(u: Float64Array, acc: Float64Array) {
    const { mu, lambda } = this.params; acc.fill(0);
    this.addCartLaplacian(u, mu, acc);
    for (let k = 0; k < 3 * this.N; k++) acc[k] += 2 * mu * u[k];
    const div = new Float64Array(this.N); this.divergenceLsq(u, div);
    this.addGradient(div, lambda + mu, acc);
  }
  private maxDiag = 0;
  viscousStep(dt: number) {
    const { mu, lambda } = this.params, nu = Math.max(mu, lambda + 2 * mu, 0);
    if (!(nu > 0)) return;
    if (!this.maxDiag) { const g = this.g, s = new Float64Array(this.N); for (let e = 0; e < g.ei.length; e++) { const w = g.elen[e] / g.edist[e]; s[g.ei[e]] += w / g.area[g.ei[e]]; s[g.ej[e]] += w / g.area[g.ej[e]]; } this.maxDiag = Math.max(...s); }
    const nsub = Math.max(1, Math.ceil(dt * nu * this.maxDiag / 0.4)), h = dt / nsub;
    for (let s = 0; s < nsub; s++) {
      this.lameAccel(this.u, this.tmp);
      for (let k = 0; k < 3 * this.N; k++) this.u[k] += h * this.tmp[k];
      this.project();
    }
  }
  /** exact rotation of u about the local normal by −2Ω(ẑ·n)dt: |u| is preserved exactly */
  coriolisStep(dt: number) {
    const { u, g } = this, W = this.params.omega; if (!W) return;
    for (let k = 0; k < this.N; k++) {
      const o = 3 * k, nx = g.pos[o], ny = g.pos[o + 1], nz = g.pos[o + 2];
      const th = -2 * W * nz * dt, c = Math.cos(th), s = Math.sin(th);
      const ux = u[o], uy = u[o + 1], uz = u[o + 2];
      // u tangent: R u = u cos θ + (n × u) sin θ
      u[o] = ux * c + (ny * uz - nz * uy) * s; u[o + 1] = uy * c + (nz * ux - nx * uz) * s; u[o + 2] = uz * c + (nx * uy - ny * ux) * s;
    }
  }
  selfPropulsion(dt: number) {
    const { tau, v0 } = this.params; if (!(tau > 0)) return;
    const decay = Math.exp(-dt / tau), gain = Math.max(0, v0) * (1 - decay), u = this.u;
    for (let k = 0; k < this.N; k++) {
      const sp = Math.hypot(u[3 * k], u[3 * k + 1], u[3 * k + 2]); if (sp < 1e-9) continue;
      const f = decay + gain / sp; u[3 * k] *= f; u[3 * k + 1] *= f; u[3 * k + 2] *= f;
    }
  }
  /** semi-Lagrangian advection of u along great circles, bilinear in equiangular coords + parallel transport */
  advectVelocity(dt: number) {
    const { u, g, sc, sw } = this, out = this.tmp;
    for (let k = 0; k < this.N; k++) {
      const o = 3 * k, p = g.p(k), ux = u[o], uy = u[o + 1], uz = u[o + 2], sp = Math.hypot(ux, uy, uz);
      if (sp * dt < 1e-12) { out[o] = ux; out[o + 1] = uy; out[o + 2] = uz; continue; }
      const s = sp * dt, cs = Math.cos(s), sn = Math.sin(s) / sp;
      const pd: V3 = [p[0] * cs - ux * sn, p[1] * cs - uy * sn, p[2] * cs - uz * sn];
      g.stencil(pd, sc, sw);
      let ax = 0, ay = 0, az = 0;
      for (let q = 0; q < 4; q++) {
        const m = sc[q], v = transport([u[3 * m], u[3 * m + 1], u[3 * m + 2]], g.p(m), pd);
        ax += sw[q] * v[0]; ay += sw[q] * v[1]; az += sw[q] * v[2];
      }
      const r = transport([ax, ay, az], pd, p); out[o] = r[0]; out[o + 1] = r[1]; out[o + 2] = r[2];
    }
    u.set(out); this.project();
  }
  /** conservative FV continuity, upwind fluxes, CFL sub-stepping */
  advectDensity(dt: number) {
    const { g, u, rho } = this, E = g.ei.length, flux = new Float64Array(E);
    let cmax = 0; this.faceNormalVelocity(u, flux);
    for (let e = 0; e < E; e++) {
      const i = g.ei[e], j = g.ej[e]; flux[e] *= g.elen[e];
      cmax = Math.max(cmax, Math.abs(flux[e]) / Math.min(g.area[i], g.area[j]));
    }
    const nsub = Math.max(1, Math.ceil(dt * cmax * 4 / 0.8)), h = dt / nsub, m = new Float64Array(this.N);
    for (let k = 0; k < this.N; k++) m[k] = rho[k] * g.area[k];
    for (let s = 0; s < nsub; s++) {
      for (let e = 0; e < E; e++) {
        const i = g.ei[e], j = g.ej[e], F = flux[e];
        const dm = h * F * (F > 0 ? rho[i] : rho[j]); m[i] -= dm; m[j] += dm;
      }
      for (let k = 0; k < this.N; k++) rho[k] = m[k] / g.area[k];
    }
  }
  // ---------- retarded repulsion ----------
  private buildPairs() {
    const { R0, alpha } = this.params, key = `${R0}|${alpha}`; if (key === this.pairKey) return;
    this.pairKey = key; const g = this.g, N = this.N, chord = 2 * Math.sin(Math.min(R0, Math.PI) / 2);
    // spatial hash on a 3D lattice with spacing = chord length of R0
    const cs = Math.max(chord, 1e-3), M = Math.ceil(2 / cs) + 1, cellOf = (x: number) => Math.min(M - 1, Math.floor((x + 1) / cs));
    const buckets = new Map<number, number[]>();
    for (let k = 0; k < N; k++) { const key = (cellOf(g.pos[3 * k]) * M + cellOf(g.pos[3 * k + 1])) * M + cellOf(g.pos[3 * k + 2]); (buckets.get(key) ?? buckets.set(key, []).get(key)!).push(k); }
    const st = [0], J: number[] = [], W: number[] = [], Ev: number[] = [], D: number[] = [];
    for (let i = 0; i < N; i++) {
      const pi = g.p(i), cx = cellOf(pi[0]), cy = cellOf(pi[1]), cz = cellOf(pi[2]);
      for (let a = cx - 1; a <= cx + 1; a++) for (let b = cy - 1; b <= cy + 1; b++) for (let c = cz - 1; c <= cz + 1; c++) {
        if (a < 0 || b < 0 || c < 0 || a >= M || b >= M || c >= M) continue;
        for (const j of buckets.get((a * M + b) * M + c) ?? []) {
          if (j === i) continue; const pj = g.p(j), d = gcDist(pi, pj); if (d >= R0) continue;
          // unit tangent at x_i pointing away from y_j (−log map direction)
          const cd = pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2];
          let ex = pi[0] * cd - pj[0], ey = pi[1] * cd - pj[1], ez = pi[2] * cd - pj[2]; const l = Math.hypot(ex, ey, ez) || 1;
          ex /= l; ey /= l; ez /= l;
          J.push(j); W.push(repulsionKernel(d, R0, alpha) * g.area[j]); Ev.push(ex, ey, ez); D.push(d);
        }
      }
      st.push(J.length);
    }
    this.pStart = Int32Array.from(st); this.pJ = Int32Array.from(J); this.pW = Float64Array.from(W); this.pE = Float64Array.from(Ev); this.pD = Float64Array.from(D);
  }
  get pairCount() { return this.pJ.length; }
  private pushHistory(cap: number) {
    if (this.hist.length !== cap) { this.hist = Array.from({ length: cap }, () => new Float64Array(this.N)); this.histHead = 0; this.histCount = 0; }
    this.histHead = (this.histHead + 1) % cap; this.hist[this.histHead].set(this.rho); this.histCount = Math.min(cap, this.histCount + 1);
  }
  clearHistory() { this.histCount = 0; }
  /** repulsion acceleration (tangent), written into this.lastForce */
  repulsion(dt: number) {
    this.buildPairs(); const { repA, vInt, R0 } = this.params, f = this.lastForce, rho = this.rho; f.fill(0);
    const retarded = Number.isFinite(vInt) && vInt > 0;
    let cap = 1;
    if (retarded) { cap = Math.min(Math.ceil(R0 / vInt / dt) + 2, Math.max(2, Math.floor(256e6 / (8 * this.N)))); this.pushHistory(cap); }
    for (let i = 0; i < this.N; i++) {
      let fx = 0, fy = 0, fz = 0;
      for (let q = this.pStart[i]; q < this.pStart[i + 1]; q++) {
        const j = this.pJ[q]; let rj: number;
        if (retarded) {
          const lag = Math.min(this.pD[q] / vInt / dt, this.histCount - 1), l0 = Math.floor(lag), s = lag - l0;
          const a = this.hist[(this.histHead - l0 + cap) % cap][j], b = this.hist[(this.histHead - Math.min(l0 + 1, this.histCount - 1) + cap) % cap][j];
          rj = a + s * (b - a);
        } else rj = rho[j];
        const w = repA * (rj - rho[i]) * this.pW[q];
        fx += w * this.pE[3 * q]; fy += w * this.pE[3 * q + 1]; fz += w * this.pE[3 * q + 2];
      }
      f[3 * i] = fx; f[3 * i + 1] = fy; f[3 * i + 2] = fz;
    }
  }
  step(dt = this.params.dt) {
    const p = this.params;
    if (p.repOn && p.repA !== 0) { this.repulsion(dt); for (let k = 0; k < 3 * this.N; k++) this.u[k] += dt * this.lastForce[k]; this.project(); }
    this.coriolisStep(dt);
    if (p.lameOn) this.viscousStep(dt);
    this.advectVelocity(dt);
    if (p.selfProp) this.selfPropulsion(dt);
    this.advectDensity(dt);
    this.time += dt;
  }

  // ---------- initial conditions ----------
  /** random: smooth field from random vortices + sources (rms speed ~ v0), plus white-noise directions and density noise of relative size amp */
  seedRandom(seed: number, amp = 0.5) {
    const r = rng(seed), g = this.g, v0 = this.params.v0 || 0.5, M = 24;
    const c: V3[] = [], s: number[] = [], kind: number[] = [];
    for (let m = 0; m < M; m++) {
      const z = 2 * r() - 1, ph = 2 * Math.PI * r(), rr = Math.sqrt(1 - z * z);
      c.push([rr * Math.cos(ph), rr * Math.sin(ph), z]); s.push((r() < 0.5 ? -1 : 1) * (0.5 + r())); kind.push(r() < 0.7 ? 0 : 1);
    }
    const sig = 0.35; this.u.fill(0); this.time = 0; this.clearHistory();
    for (let k = 0; k < this.N; k++) {
      const p = g.p(k);
      for (let m = 0; m < M; m++) {
        const cd = p[0] * c[m][0] + p[1] * c[m][1] + p[2] * c[m][2], d = Math.acos(Math.max(-1, Math.min(1, cd)));
        const w = s[m] * Math.exp(-(d * d) / (2 * sig * sig));
        // tangent direction away from c (source) or around c (vortex)
        let ex = p[0] * cd - c[m][0], ey = p[1] * cd - c[m][1], ez = p[2] * cd - c[m][2]; const l = Math.hypot(ex, ey, ez); if (l < 1e-9) continue;
        ex /= l; ey /= l; ez /= l;
        if (kind[m] === 0) { const tx = p[1] * ez - p[2] * ey, ty = p[2] * ex - p[0] * ez, tz = p[0] * ey - p[1] * ex; ex = tx; ey = ty; ez = tz; }
        const fac = w * Math.sin(Math.min(d, Math.PI - d) * 2) ;
        this.u[3 * k] += fac * ex; this.u[3 * k + 1] += fac * ey; this.u[3 * k + 2] += fac * ez;
      }
    }
    this.project();
    let rms = 0; for (let k = 0; k < this.N; k++) rms += this.u[3 * k] ** 2 + this.u[3 * k + 1] ** 2 + this.u[3 * k + 2] ** 2;
    rms = Math.sqrt(rms / this.N) || 1;
    for (let k = 0; k < this.N; k++) {
      const z = 2 * r() - 1, ph = 2 * Math.PI * r(), rr = Math.sqrt(1 - z * z);
      for (let q = 0; q < 3; q++) this.u[3 * k + q] = (this.u[3 * k + q] / rms) * v0 * (1 - Math.min(1, amp)) + amp * v0 * [rr * Math.cos(ph), rr * Math.sin(ph), z][q] * 1.5;
    }
    this.project();
    // density: 1 + amp × smoothed noise (two passes of neighbour averaging)
    const n0 = new Float64Array(this.N).map(() => 2 * r() - 1);
    for (let pass = 0; pass < 2; pass++) { const t = n0.slice(); for (let k = 0; k < this.N; k++) { let a = t[k]; for (let q = 0; q < 4; q++) a += t[g.nbr[4 * k + q]]; n0[k] = a / 5; } }
    let mx = 1e-9; for (const x of n0) mx = Math.max(mx, Math.abs(x));
    for (let k = 0; k < this.N; k++) this.rho[k] = Math.max(0.05, 1 + Math.min(amp, 0.9) * n0[k] / mx);
    const m0 = this.totalMass() / (4 * Math.PI); for (let k = 0; k < this.N; k++) this.rho[k] /= m0;
  }
  /** eastward solid-body rotation u = v0 ẑ × x (speed v0 sin θ), uniform density */
  seedZonal(v0 = this.params.v0) {
    const g = this.g; this.time = 0; this.clearHistory(); this.rho.fill(1);
    for (let k = 0; k < this.N; k++) { const x = g.pos[3 * k], y = g.pos[3 * k + 1]; this.u[3 * k] = -v0 * y; this.u[3 * k + 1] = v0 * x; this.u[3 * k + 2] = 0; }
  }

  // ---------- order parameters ----------
  /**
   * Polar order ⟨u/|u|⟩ is not defined on a sphere (vectors in different tangent planes; and a smooth
   * unit tangent field cannot exist — hairy-ball theorem). We report instead:
   *  - localAlign  ⟨cos θ_ij⟩ over edge-neighbour pairs, u_j parallel-transported to x_i (unweighted)
   *  - localAlignRho  same, each pair weighted by its mass (m_i + m_j)/2
   *  - Lz: zonal order Σ m (x × u)·ẑ / Σ m |u| sin θ ∈ [−1, 1]  (+1 = everything moving east, solid-body-like)
   *  - PhiL: rotational polar order |Σ m x × û| / Σ m ∈ [0, 1]  (x × û is a unit vector: the axis of the great circle a parcel is on)
   *  - meanSpeed, meanSpeedRho, rhoMax
   */
  orderStats() {
    const { g, u, rho } = this, N = this.N;
    let sp = 0, spR = 0, M = 0, mean = 0;
    const speed = new Float64Array(N);
    for (let k = 0; k < N; k++) { speed[k] = Math.hypot(u[3 * k], u[3 * k + 1], u[3 * k + 2]); mean += speed[k]; }
    mean /= N; const thr = 0.02 * mean;
    let ca = 0, cn = 0, cr = 0, cw = 0;
    for (let e = 0; e < g.ei.length; e++) {
      const i = g.ei[e], j = g.ej[e]; if (speed[i] <= thr || speed[j] <= thr) continue;
      const t = transport([u[3 * j], u[3 * j + 1], u[3 * j + 2]], g.p(j), g.p(i));
      const c = (t[0] * u[3 * i] + t[1] * u[3 * i + 1] + t[2] * u[3 * i + 2]) / (speed[i] * speed[j]);
      const w = 0.5 * (rho[i] * g.area[i] + rho[j] * g.area[j]);
      ca += c; cn++; cr += w * c; cw += w;
    }
    let Lz = 0, Ld = 0, Lx = 0, Ly = 0, Lzz = 0, Mv = 0, rhoMax = 0;
    for (let k = 0; k < N; k++) {
      const m = rho[k] * g.area[k], x = g.pos[3 * k], y = g.pos[3 * k + 1], z = g.pos[3 * k + 2];
      sp += speed[k]; spR += m * speed[k]; M += m; rhoMax = Math.max(rhoMax, rho[k]);
      Lz += m * (x * u[3 * k + 1] - y * u[3 * k]); Ld += m * speed[k] * Math.hypot(x, y);
      if (speed[k] > thr) {
        const a = u[3 * k] / speed[k], b = u[3 * k + 1] / speed[k], c = u[3 * k + 2] / speed[k];
        Lx += m * (y * c - z * b); Ly += m * (z * a - x * c); Lzz += m * (x * b - y * a); Mv += m;
      }
    }
    const PhiL = Mv > 0 ? Math.hypot(Lx, Ly, Lzz) / Mv : 0;
    return {
      localAlign: cn ? ca / cn : 0, localAlignRho: cw ? cr / cw : 0,
      Lz: Ld > 0 ? Lz / Ld : 0, PhiL, PhiLaxis: [Lx, Ly, Lzz].map((v) => v / (Math.hypot(Lx, Ly, Lzz) || 1)) as V3,
      meanSpeed: sp / N, meanSpeedRho: M ? spR / M : 0, rhoMax, mass: M,
    };
  }
}
