import { describe, it, expect } from 'vitest';
import { Solver, defaultParams } from '../src/solver/solver';
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

  it('split operator with self = neighbour field equals the undelayed Lamé operator', () => {
    const s = new Solver(32, { ...defaultParams(), mu: 2e-3, lambda: 5e-3 }); s.seedRandom(7, 0.2);
    for (let q = 0; q < s.u.length; q++) s.u[q] += 0.1 * Math.sin(q * 0.37);
    const N = s.u.length, ax = new Float64Array(N), ay = new Float64Array(N), bx = new Float64Array(N), by = new Float64Array(N);
    s.lame(s.u, s.v, ax, ay); s.lameSplit(s.u, s.v, s.u, s.v, bx, by);
    let e = 0, m = 0; for (let q = 0; q < N; q++) { e = Math.max(e, Math.abs(ax[q] - bx[q]), Math.abs(ay[q] - by[q])); m = Math.max(m, Math.abs(ax[q])); }
    expect(e / m).toBeLessThan(1e-13);
  });

  it('zero delays: stepping equals forward Euler with the undelayed operator, no history kept', () => {
    const p = { ...defaultParams(), mu: 1e-3, lambda: 2e-3, selfProp: false };
    const s = new Solver(32, p); s.seedRandom(4, 0.05);
    const u = s.u.slice(), v = s.v.slice(), N = u.length, lx = new Float64Array(N), ly = new Float64Array(N);
    const ns = Math.max(1, Math.ceil(p.dt * s.spectralBound() / 1.8)), d = p.dt / ns;
    for (let m = 0; m < ns; m++) { s.lame(u, v, lx, ly); for (let q = 0; q < N; q++) { u[q] += d * lx[q]; v[q] += d * ly[q]; } }
    s.viscous(p.dt);
    for (let q = 0; q < N; q++) { expect(s.u[q]).toBeCloseTo(u[q], 13); expect(s.v[q]).toBeCloseTo(v[q], 13); }
    for (let m = 0; m < 10; m++) s.step();
    expect(s.history.frames.length).toBe(0);
  });

  it('equal delays: grid-scale checkerboard stable for Λ·t < π/2, unstable for Λ·t > π/2', () => {
    // n = 32, μ = 1e-3, λ = −μ (pure shear-type stencil): checkerboard eigenvalue Λ = 8μ/h² = 8.19 /s
    const grow = (t: number) => {
      const s = new Solver(32, { ...defaultParams(), mu: 1e-3, lambda: -1e-3, selfProp: false, tSelf: t, tNb: t });
      for (let j = 0; j < 32; j++) for (let i = 0; i < 32; i++) s.u[j * 32 + i] = 1e-3 * ((i + j) & 1 ? 1 : -1);
      const a0 = Math.max(...s.u.map(Math.abs));
      while (s.t < 8) s.step();
      return Math.max(...s.u.map(Math.abs)) / a0;
    };
    expect(grow(0.1)).toBeLessThan(0.1);   // Λt = 0.82
    expect(grow(0.4)).toBeGreaterThan(10); // Λt = 3.3
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
