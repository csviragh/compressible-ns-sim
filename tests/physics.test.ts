import { describe, it, expect } from 'vitest';
import { Solver, defaultParams, clampParams } from '../src/solver/solver';

const mk = (n: number, over: Partial<ReturnType<typeof defaultParams>> = {}) =>
  new Solver(n, { ...defaultParams(), selfProp: false, kernel: 'none', ...over });
const proj = (s: Solver, q: Float64Array, fn: (x: number) => number) => {
  let a = 0, w = 0;
  for (let j = 0; j < s.n; j++) for (let i = 0; i < s.n; i++) { const f = fn((i + 0.5) * s.h); a += q[j * s.n + i] * f; w += f * f; }
  return a / w;
};
const k = 2 * Math.PI, kap2 = (n: number) => (2 * n * Math.sin(Math.PI / n)) ** 2; // discrete k² for mode 1

describe('compressible-velocity Lamé fluid', () => {
  const fn = (x: number) => Math.sin(k * x);

  it('shear wave v = sin(kx) decays at μk²', () => {
    const mu = 2e-3, s = mk(64, { mu, lambda: 5e-3 });
    for (let j = 0; j < 64; j++) for (let i = 0; i < 64; i++) s.v[j * 64 + i] = 1e-3 * fn((i + 0.5) / 64);
    const a0 = proj(s, s.v, fn), T = 2; while (s.t < T - 1e-9) s.step(0.01);
    const rate = -Math.log(proj(s, s.v, fn) / a0) / T;
    expect(Math.abs(rate - mu * k * k) / (mu * k * k)).toBeLessThan(0.01);
    expect(Math.abs(rate - mu * kap2(64)) / (mu * kap2(64))).toBeLessThan(2e-3);
  });

  it('longitudinal wave u = sin(kx) decays at (λ+2μ)k²', () => {
    for (const [mu, lambda] of [[2e-3, 0], [2e-3, 6e-3], [2e-3, -3e-3]]) {
      const s = mk(64, { mu, lambda });
      for (let j = 0; j < 64; j++) for (let i = 0; i < 64; i++) s.u[j * 64 + i] = 1e-3 * fn((i + 0.5) / 64);
      const a0 = proj(s, s.u, fn), T = 2; while (s.t < T - 1e-9) s.step(0.01);
      const rate = -Math.log(proj(s, s.u, fn) / a0) / T, want = (lambda + 2 * mu) * k * k;
      expect(Math.abs(rate - want) / want).toBeLessThan(0.01);
    }
  });

  it('self-propulsion drives |u| → v0', () => {
    const s = new Solver(32, { ...defaultParams(), mu: 1e-4, selfProp: true, tau: 0.5, v0: 0.5 });
    s.seedOrdered(3);
    for (let m = 0; m < 600; m++) s.step();
    expect(Math.abs(s.meanSpeed() - 0.5)).toBeLessThan(0.02);
    // a uniform flow is exactly relaxed: |u| = v0 + (|u0| − v0) e^{−t/τ}
    const q = mk(16, { selfProp: true, tau: 0.5, v0: 0.5 });
    q.u.fill(0.1); for (let m = 0; m < 30; m++) q.step(1 / 60);
    expect(q.u[0]).toBeCloseTo(0.5 - 0.4 * Math.exp(-0.5 / 0.5), 10);
  });

  it('is periodic: a uniform flow translates a pattern around the torus unchanged in shape', () => {
    const n = 64, s = mk(n, { mu: 0, lambda: 0 });
    s.u.fill(0.5);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s.dye[j * n + i] = Math.sin(k * (i + 0.5) / n) * Math.cos(k * (j + 0.5) / n);
    const d0 = s.dye.slice();
    // u Δt = h exactly → pure shifts; after n steps the pattern has wrapped once
    for (let m = 0; m < n; m++) s.step(1 / (0.5 * n));
    let err = 0; for (let q = 0; q < d0.length; q++) err = Math.max(err, Math.abs(s.dye[q] - d0[q]));
    expect(err).toBeLessThan(1e-9);
    // mean momentum is conserved by viscosity on the torus
    const t = mk(32, { mu: 5e-3, lambda: 1e-2 }); t.seedRandom(9, 0.1); t.u.forEach((_, q) => (t.u[q] += 0.2));
    const m0 = t.u.reduce((a, b) => a + b, 0); t.viscous(0.5);
    expect(Math.abs(t.u.reduce((a, b) => a + b, 0) - m0) / 1024).toBeLessThan(1e-12);
  });

  it('allows compression (∇·u ≠ 0) and clamps λ ≥ −2μ', () => {
    const s = mk(32); for (let i = 0; i < 32 * 32; i++) s.u[i] = Math.sin(k * ((i % 32) + 0.5) / 32);
    expect(s.rmsDivergence()).toBeGreaterThan(1);
    const p = { ...defaultParams(), mu: 1e-3, lambda: -1 };
    expect(clampParams(p)).toBe(true); expect(p.lambda + 2 * p.mu).toBe(0);
  });

  it('density: mass conserved to round-off in a compressive active flow, ρ stays ≥ 0', () => {
    const s = new Solver(64, { ...defaultParams(), mu: 1e-4, lambda: 0 });
    s.seedRandom(5, 0.3);
    for (let i = 0; i < 64 * 64; i++) s.u[i] += 0.3 * Math.sin(k * ((i % 64) + 0.5) / 64); // add compression
    const m0 = s.totalMass();
    for (let m = 0; m < 120; m++) s.step();
    expect(s.floorHits).toBe(0);
    expect(Math.min(...s.rho)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...s.rho)).toBeGreaterThan(1.05); // compression actually happened
    expect(Math.abs(s.totalMass() - m0) / m0).toBeLessThan(1e-12);
  });

  it('density: a uniform flow translates ρ around the torus', () => {
    const n = 64, s = mk(n, { mu: 0, lambda: 0 });
    s.u.fill(0.5);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) s.rho[j * n + i] = 1 + 0.5 * Math.sin(k * (i + 0.5) / n);
    const r0 = s.rho.slice();
    while (s.t < 2 - 1e-9) s.step(1 / 64); // one full wrap at u = 0.5
    let err = 0; for (let q = 0; q < r0.length; q++) err = Math.max(err, Math.abs(s.rho[q] - r0[q]));
    expect(err).toBeLessThan(0.01);
    expect(Math.abs(s.totalMass() - 1)).toBeLessThan(1e-12);
  });
});
