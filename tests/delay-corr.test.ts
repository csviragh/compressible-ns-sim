import { describe, it, expect } from 'vitest';
import { Solver, defaultParams, repulsionKernel, ALPHA_MIN, ALPHA_MAX } from '../src/solver/solver';
import { FieldHistory } from '../src/solver/delay';
import { OrderAnalyzer } from '../src/solver/correlations';

describe('delay buffer', () => {
  it('reproduces a known shifted history (linear interpolation in time)', () => {
    const h = new FieldHistory(), N = 8, dt = 0.1;
    h.configure(N, N, dt, 1, 1 << 20);
    const f = (t: number) => Float64Array.from({ length: N }, (_, k) => t * 3 + k);
    for (let s = 0; s <= 20; s++) { const t = s * dt; if (h.wants(t)) h.push(t, f(t), f(-t)); h.prune(t + dt); }
    const now = 2, ou = new Float64Array(N), ov = new Float64Array(N);
    for (const d of [0.3, 0.55, 0.85]) {
      h.read(now - d, now, f(now), f(-now), ou, ov);
      for (let k = 0; k < N; k++) { expect(ou[k]).toBeCloseTo(f(now - d)[k], 5); expect(ov[k]).toBeCloseTo(f(-(now - d))[k], 5); }
    }
    h.read(now, now, f(now), f(-now), ou, ov); // zero delay → current field
    expect(Array.from(ou)).toEqual(Array.from(f(now)));
  });

  it('lookup returns the current field for T ≥ now and clamps before the oldest frame', () => {
    const h = new FieldHistory(); h.configure(4, 0, 0.1, 1, 1 << 20);
    for (let s = 0; s <= 10; s++) h.push(s * 0.1, new Float64Array(4).fill(s), new Float64Array(0));
    expect(h.lookup(1.5, 1.5).a).toBe(-1);
    expect(h.lookup(-5, 1.5)).toEqual({ a: 0, b: 0, theta: 0 });
    const l = h.lookup(0.35, 1.5); expect(l.a).toBe(3); expect(l.b).toBe(4); expect(l.theta).toBeCloseTo(0.5, 9);
  });
});

describe('repulsion', () => {
  it('kernel: diverges as d → 0 and decreases monotonically', () => {
    for (const a of [0.1, 1, 1.95]) {
      let prev = Infinity;
      for (let q = 1; q < 200; q++) { const K = repulsionKernel(0.04 * q / 200, 0.04, a); expect(K).toBeLessThan(prev); prev = K; }
      expect(repulsionKernel(1e-9, 0.04, a)).toBeGreaterThan(repulsionKernel(1e-6, 0.04, a));
    }
    expect(repulsionKernel(1e-8, 0.04, 1)).toBeGreaterThan(1e5);
  });
  it('kernel: K(R0) = 0, K\'(R0) = 0, zero for d ≥ R0 and the own cell; α clamped to [0.1, 1.95]', () => {
    const R0 = 0.04, e = 1e-6;
    expect(repulsionKernel(R0, R0, 1)).toBe(0);
    expect((repulsionKernel(R0 - e, R0, 1) - repulsionKernel(R0 - 2 * e, R0, 1)) / e).toBeCloseTo(0, 6);
    for (const d of [R0, 0.05, 1]) expect(repulsionKernel(d, R0, 1.5)).toBe(0);
    expect(repulsionKernel(0, R0, 1)).toBe(0);
    expect(repulsionKernel(0.01, R0, 0)).toBe(repulsionKernel(0.01, R0, ALPHA_MIN));
    expect(repulsionKernel(0.01, R0, 3)).toBe(repulsionKernel(0.01, R0, ALPHA_MAX));
    expect(repulsionKernel(0.01, R0, 1)).toBeCloseTo((4 - 1) * 0.75 ** 2, 12);
  });
  it('uniform density feels no net force; a density bump pushes fluid outward (compressible, ∇·u > 0)', () => {
    const n = 32, s = new Solver(n, { ...defaultParams(), repOn: true, R0: 0.15 });
    const fx = new Float64Array(n * n), fy = new Float64Array(n * n);
    s.repulsion(fx, fy); expect(Math.max(...fx.map(Math.abs), ...fy.map(Math.abs))).toBeLessThan(1e-12);
    s.rho[16 * n + 16] = 5; s.repulsion(fx, fy);
    expect(fx[16 * n + 18]).toBeGreaterThan(0); expect(fx[16 * n + 14]).toBeLessThan(0); expect(fy[18 * n + 16]).toBeGreaterThan(0);
  });
  it('retarded: a density change reaches distance d only after d/v_int', () => {
    const n = 32, vInt = 0.5, p = { ...defaultParams(), repOn: true, R0: 0.2, vInt, mu: 0, lambda: 0, selfProp: false };
    const s = new Solver(n, p), fx = new Float64Array(n * n), fy = new Float64Array(n * n), dt = 1 / 60;
    const probe = 16 * n + 16 + 4, d = 4 / n; // target 4 cells east of the source, delay 0.25 s
    let tArrive = -1;
    while (s.t < 0.6) {
      if (s.t >= 0.1 - 1e-9) s.rho[16 * n + 16] = 3; // source switched on at t = 0.1
      // record ρ(t) exactly as step() does, but keep ρ frozen (no transport) to isolate the retardation
      s.history.configure(n * n, 0, dt, s.delayMax() + dt, 1 << 26); s.history.push(s.t, s.rho, new Float64Array(0)); s.t += dt;
      s.repulsion(fx, fy);
      if (tArrive < 0 && fx[probe] > 1e-9) tArrive = s.t;
    }
    expect(tArrive - 0.1).toBeGreaterThan(d / vInt - 2 * dt);
    expect(tArrive - 0.1).toBeLessThan(d / vInt + 2 * dt);
    p.vInt = Infinity; s.repulsion(fx, fy); expect(fx[probe]).toBeGreaterThan(0);
  });
});

describe('Lamé switch', () => {
  it('lameOn = false leaves u untouched by viscosity; on = forward Euler with L', () => {
    const p = { ...defaultParams(), mu: 1e-3, lambda: 2e-3, selfProp: false, lameOn: false };
    const s = new Solver(32, p); s.seedRandom(4, 0.05);
    const u0 = s.u.slice(); s.viscous(p.dt); expect(Array.from(s.u)).toEqual(Array.from(u0));
    p.lameOn = true; const N = u0.length, lx = new Float64Array(N), ly = new Float64Array(N);
    s.lame(s.u, s.v, lx, ly); const ns = Math.ceil(p.dt * s.spectralBound() / 1.8); s.viscous(p.dt);
    expect(ns).toBe(s.lastSubsteps); expect(Math.abs(s.u[5] - u0[5])).toBeGreaterThan(0);
  });
});

describe('correlations', () => {
  it('aligned field gives ⟨cos⟩ = 1, C(r) = 1', () => {
    const s = new Solver(32, defaultParams()); s.u.fill(0.3); s.v.fill(0.4);
    const st = new OrderAnalyzer().analyze(32, 32, (i, j) => [s.u[j * 32 + i], s.v[j * 32 + i]], { periodicX: true, periodicY: true });
    expect(st.meanCos).toBeCloseTo(1, 12);
    expect(st.C[3]).toBeCloseTo(1, 9);
  });
  it('ordered + perturbation start has ⟨cos⟩ ≈ 0.95 on 256²', () => {
    const s = new Solver(256, defaultParams()); s.seedOrdered(5000);
    const st = new OrderAnalyzer().analyze(256, 256, (i, j) => [s.u[j * 256 + i], s.v[j * 256 + i]], { periodicX: true, periodicY: true, correlation: false });
    expect(st.meanCos).toBeGreaterThan(0.92); expect(st.meanCos).toBeLessThan(0.98);
  });
});
