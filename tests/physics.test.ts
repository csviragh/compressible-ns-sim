import { describe, it, expect } from 'vitest';
import { Solver, defaultParams, clampLambda, bulkViscosity } from '../src/solver/solver';
import { applyPreset, setState } from '../src/solver/presets';

const mk = (n: number, over: Partial<ReturnType<typeof defaultParams>> = {}) =>
  new Solver(n, { ...defaultParams(), ...over });

/** Projection of field q(x) onto sin(kx) (k = 2π), averaged over y. */
function sinAmp(s: Solver, q: (k: number) => number, fn: (x: number) => number) {
  let a = 0, w = 0;
  for (let j = 0; j < s.n; j++) for (let i = 0; i < s.n; i++) {
    const f = fn((i + 0.5) * s.h); a += q(j * s.n + i) * f; w += f * f;
  }
  return a / w;
}
const run = (s: Solver, T: number) => { while (s.t < T - 1e-12) s.step(T - s.t); };

describe('compressible NS', () => {
  it('conserves mass to round-off (turbulence + viscosity)', () => {
    const s = mk(48, { mu: 1e-3, lambda: 2e-3 });
    applyPreset(s, 'turbulence', 7);
    const m0 = s.totalMass();
    for (let k = 0; k < 150; k++) s.step();
    expect(s.floorHits).toBe(0);
    expect(Math.abs(s.totalMass() - m0) / m0).toBeLessThan(1e-12);
  });

  it('conserves momentum on the torus', () => {
    const s = mk(32, { mu: 1e-3 });
    applyPreset(s, 'jets');
    const P = () => s.mx.reduce((a, b) => a + b, 0);
    const p0 = P();
    for (let k = 0; k < 100; k++) s.step();
    expect(Math.abs(P() - p0) * s.h * s.h).toBeLessThan(1e-12);
  });

  it('acoustic pulse propagates at speed ≈ c', () => {
    for (const c of [0.5, 1]) {
      const s = mk(256, { mu: 0, c });
      // 1D Gaussian pulse in x, small amplitude; it splits into two fronts at ±c t
      setState(s, (x) => [1 + 1e-3 * Math.exp(-((x - 0.5) ** 2) / (2 * 0.02 ** 2)), 0, 0, 0]);
      const T = 0.3 / c;
      run(s, T);
      let best = 0, bi = 0;
      for (let i = s.n / 2; i < s.n; i++) if (s.rho[i] > best) { best = s.rho[i]; bi = i; }
      const xpk = (bi + 0.5) * s.h;
      expect(Math.abs((xpk - 0.5) / T - c) / c).toBeLessThan(0.03);
    }
  });

  // The scheme's own (Rusanov/limiter) dissipation is measured with μ = λ = 0 and subtracted;
  // it is reported so it stays small compared with the physical rates.
  const shearRate = (mu: number, lambda: number) => {
    const s = mk(64, { mu, lambda });
    const fn = (x: number) => Math.sin(2 * Math.PI * x);
    setState(s, (x) => [1, 0, 1e-3 * fn(x), 0]);
    const a0 = sinAmp(s, (k) => s.my[k] / s.rho[k], fn);
    const T = 1; run(s, T);
    return -Math.log(sinAmp(s, (k) => s.my[k] / s.rho[k], fn) / a0) / T;
  };
  it('shear wave decays at rate μk²/ρ', () => {
    const mu = 0.01, k = 2 * Math.PI, base = shearRate(0, 0), rate = shearRate(mu, 0.02);
    console.log(`shear: numerical ${base.toExponential(2)}/s, total ${rate.toFixed(4)}, theory ${(mu * k * k).toFixed(4)}`);
    expect(base).toBeLessThan(0.02 * mu * k * k);
    expect(Math.abs(rate - base - mu * k * k) / (mu * k * k)).toBeLessThan(0.02);
  });

  it('longitudinal (sound) wave damps at (2μ+λ)k²/(2ρ)', () => {
    const soundRate = (mu: number, lambda: number) => {
      const s = mk(64, { mu, lambda, c: 1 });
      const fn = (x: number) => Math.sin(2 * Math.PI * x);
      setState(s, (x) => [1, 1e-4 * fn(x), 0, 0]);
      // quadratic acoustic energy: ½ρ₀u² + c²ρ'²/(2ρ₀); sampled at whole periods
      const E = () => { let e = 0; for (let k = 0; k < s.rho.length; k++) { const u = s.mx[k] / s.rho[k]; e += u * u + (s.rho[k] - 1) ** 2; } return e; };
      const e0 = E(), T = 4; run(s, T);
      return -Math.log(E() / e0) / (2 * T);
    };
    const base = soundRate(0, 0), k = 2 * Math.PI;
    for (const [mu, lambda] of [[5e-3, 0], [5e-3, 1e-2], [5e-3, -5e-3]]) {
      const want = (2 * mu + lambda) * k * k / 2, rate = soundRate(mu, lambda);
      console.log(`sound μ=${mu} λ=${lambda}: numerical ${base.toExponential(2)}/s, total ${rate.toFixed(4)}, theory ${want.toFixed(4)}`);
      expect(Math.abs(rate - base - want) / want).toBeLessThan(0.03);
    }
  });

  it('clamps λ so the bulk viscosity ζ = λ + μ ≥ 0', () => {
    const p = { ...defaultParams(), mu: 1e-3, lambda: -5e-3 };
    expect(clampLambda(p)).toBe(true);
    expect(bulkViscosity(p)).toBe(0);
  });

  it('polytropic EOS runs stably and keeps ρ > 0 in colliding jets', () => {
    const s = mk(48, { eos: 'polytropic', gamma: 1.4, mu: 5e-4 });
    applyPreset(s, 'jets');
    for (let k = 0; k < 200; k++) s.step();
    expect(Math.min(...s.rho)).toBeGreaterThan(0);
    expect(Number.isFinite(s.kineticEnergy())).toBe(true);
  });
});
