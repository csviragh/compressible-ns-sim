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
    for (const d of [0.3, 0.55, 0.95]) {
      h.read(now - d, now, f(now), f(-now), ou, ov);
      for (let k = 0; k < N; k++) { expect(ou[k]).toBeCloseTo(f(now - d)[k], 5); expect(ov[k]).toBeCloseTo(f(-(now - d))[k], 5); }
    }
    h.read(now, now, f(now), f(-now), ou, ov); // zero delay → current field
    expect(Array.from(ou)).toEqual(Array.from(f(now)));
  });

  it('zero delays give exactly the undelayed dynamics', () => {
    const a = new Solver(32, defaultParams()), b = new Solver(32, { ...defaultParams(), delayLame: true });
    a.seedOrdered(4); b.seedOrdered(4);
    for (let m = 0; m < 30; m++) { a.step(); b.step(); }
    expect(Array.from(b.u)).toEqual(Array.from(a.u));
    expect(b.history.frames.length).toBe(0);
  });

  it('a tiny delay is close to the undelayed run; a long delay with S·t > π/2 destabilises the ordered state', () => {
    const base = { ...defaultParams(), mu: 1e-4, selfProp: false };
    const a = new Solver(32, base), b = new Solver(32, { ...base, tSelf: 1e-3, tNb: 1e-3 });
    a.seedRandom(2, 0.1); b.seedRandom(2, 0.1);
    for (let m = 0; m < 60; m++) { a.step(); b.step(); }
    let e = 0; for (let k = 0; k < a.u.length; k++) e = Math.max(e, Math.abs(a.u[k] - b.u[k]));
    expect(e).toBeLessThan(2e-3);
    // linear check of û' = −S û(t − t): perturbation grows when S·t > π/2 (here 2·1 = 2)
    const c = new Solver(32, { ...base, tSelf: 1, tNb: 1, mu: 0 });
    c.seedRandom(3, 1e-3); const amp0 = Math.max(...c.u.map(Math.abs));
    for (let m = 0; m < 60 * 12; m++) c.step();
    expect(Math.max(...c.u.map(Math.abs))).toBeGreaterThan(amp0);
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
