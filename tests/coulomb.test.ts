import { describe, it, expect } from 'vitest';
import { Solver, defaultParams } from '../src/solver/solver';
import { coulombWeights, minImageDist } from '../src/solver/coulomb';

const mk = (n: number, over: Partial<ReturnType<typeof defaultParams>> = {}) =>
  new Solver(n, { ...defaultParams(), selfProp: false, lameOn: false, repOn: true, repKind: 'coulomb', repA: 1, ...over });
const bump = (s: Solver, ci: number, cj: number, amp = 1, w = 1.5) => {
  const n = s.n;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    let dx = i - ci, dy = j - cj; dx -= n * Math.round(dx / n); dy -= n * Math.round(dy / n);
    s.rho[j * n + i] = 1 + amp * Math.exp(-(dx * dx + dy * dy) / (2 * w * w));
  }
};

describe('periodic 2D Coulomb repulsion', () => {
  it('uniform density gives zero force (instantaneous FFT and retarded direct sum)', () => {
    for (const vInt of [Infinity, 0.5]) {
      const s = mk(16, { vInt }), fx = new Float64Array(256), fy = new Float64Array(256);
      s.repulsion(fx, fy);
      for (let k = 0; k < 256; k++) { expect(Math.abs(fx[k])).toBeLessThan(1e-10); expect(Math.abs(fy[k])).toBeLessThan(1e-10); }
    }
  });

  it('kernel W(o) ≈ h² ô/|o| at moderate distance and W(0) = 0', () => {
    const n = 64, { wx, wy } = coulombWeights(n), h = 1 / n;
    expect(wx[0]).toBe(0); expect(wy[0]).toBe(0);
    for (const d of [4, 8]) { // along +x: free-space h²/(d h) = h/d; images add < few %
      const w = wx[d], want = h * h / (d * h);
      expect(Math.abs(w - want) / want).toBeLessThan(0.05);
      expect(Math.abs(wy[d])).toBeLessThan(1e-12);
    }
  });

  it('force around a small density bump is radially outward and ~1/d', () => {
    const n = 64, s = mk(n), fx = new Float64Array(n * n), fy = new Float64Array(n * n);
    bump(s, 32, 32, 1, 1);
    s.repulsion(fx, fy);
    let M = 0; for (let k = 0; k < n * n; k++) M += (s.rho[k] - 1) / (n * n); // bump mass
    for (const [di, dj] of [[6, 0], [0, -6], [5, 5], [-10, 0], [0, 10]]) {
      const k = (32 + dj) * n + 32 + di, d = Math.hypot(di, dj) / n;
      const radial = (fx[k] * di + fy[k] * dj) / Math.hypot(di, dj), tang = (fx[k] * dj - fy[k] * di) / Math.hypot(di, dj);
      expect(radial).toBeGreaterThan(0);
      expect(Math.abs(tang)).toBeLessThan(1e-6 * Math.abs(radial) + 1e-12);
      // free-space M/d minus background term π ρ̄_excess... periodic correction ~ 2π M d / 2 (small at d ≲ 0.15)
      expect(Math.abs(radial - M / d) / (M / d)).toBeLessThan(0.1);
    }
  });

  it('FFT instantaneous force equals the direct periodic-kernel sum', () => {
    const n = 24, s = mk(n), a = [new Float64Array(n * n), new Float64Array(n * n)], b = [new Float64Array(n * n), new Float64Array(n * n)];
    for (let k = 0; k < n * n; k++) s.rho[k] = 1 + 0.5 * Math.sin(0.37 * k) + 0.2 * Math.cos(k * k * 0.01);
    s.coulomb(a[0], a[1]); s.coulomb(b[0], b[1], true);
    let e = 0, M = 0; for (let k = 0; k < n * n; k++) { e = Math.max(e, Math.abs(a[0][k] - b[0][k]), Math.abs(a[1][k] - b[1][k])); M = Math.max(M, Math.abs(a[0][k])); }
    expect(e / M).toBeLessThan(1e-10);
  });

  it('retarded: a source switched on at t = 0 is felt at distance d only after d/v_int', () => {
    const n = 32, vInt = 0.5, s = mk(n, { vInt, dt: 0.02 }); s.params.repA = 1;
    // freeze dynamics: we only probe the force, stepping time through the history
    s.rho.fill(1); const fx = new Float64Array(n * n), fy = new Float64Array(n * n);
    // record a uniform history first
    for (let m = 0; m < 5; m++) { s.history.configure(n * n, 0, 0.02, s.delayMax(), 2 ** 28); s.history.push(s.t, s.rho, []); s.t += 0.02; }
    const t0 = s.t; bump(s, 16, 16, 1, 1);
    const di = 8, d = minImageDist(n, di, 0), probe = 16 * n + 16 + di;
    const arrival: number[] = [];
    for (let m = 0; m < 60; m++) {
      s.history.configure(n * n, 0, 0.02, s.delayMax(), 2 ** 28); s.history.push(s.t, s.rho, []);
      s.repulsion(fx, fy);
      if (fx[probe] > 0.3 / d * 0.01 && arrival.length === 0) arrival.push(s.t - t0);
      s.t += 0.02;
    }
    expect(arrival.length).toBe(1);
    expect(Math.abs(arrival[0] - d / vInt)).toBeLessThanOrEqual(0.02 * 1.5 + 1e-9);
  });

  it('full step with Coulomb (retarded) conserves mass and stays finite; R0/α ignored', () => {
    const s = mk(16, { vInt: 1, R0: 0.01, alpha: 1.9 });
    s.seedRandom(1, 0.1); bump(s, 8, 8, 0.5, 2); const m0 = s.totalMass();
    for (let m = 0; m < 20; m++) s.step();
    expect(Math.abs(s.totalMass() - m0)).toBeLessThan(1e-12);
    expect(s.u.every(Number.isFinite)).toBe(true);
    expect(s.repulsionStencil().length).toBe(0);
  });
});
