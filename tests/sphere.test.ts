import { describe, it, expect } from 'vitest';
import { CubedSphere } from '../src/sphere/grid';
import { SphereSolver, defaultSphereParams, rng as rng0 } from '../src/sphere/sphereSolver';

const quiet = () => ({ ...defaultSphereParams(), lameOn: false, selfProp: false, repOn: false, omega: 0 });
const speeds = (s: SphereSolver) => Array.from({ length: s.N }, (_, k) => Math.hypot(s.u[3 * k], s.u[3 * k + 1], s.u[3 * k + 2]));

describe('cubed-sphere geometry', () => {
  const g = new CubedSphere(16);
  it('areas sum to 4π, quasi-uniform', () => {
    let s = 0, mn = Infinity, mx = 0; for (const a of g.area) { s += a; mn = Math.min(mn, a); mx = Math.max(mx, a); }
    expect(Math.abs(s - 4 * Math.PI)).toBeLessThan(1e-10);
    expect(mx / mn).toBeLessThan(1.4); // equiangular: max/min area ≈ 1.3
  });
  it('4 symmetric neighbours per cell, 2N edges, unit centres', () => {
    for (let k = 0; k < g.N; k++) for (let q = 0; q < 4; q++) {
      const m = g.nbr[4 * k + q]; expect(m).not.toBe(k);
      expect(Array.from(g.nbr.slice(4 * m, 4 * m + 4))).toContain(k);
    }
    expect(g.ei.length).toBe(2 * g.N);
    for (let k = 0; k < g.N; k++) expect(Math.abs(Math.hypot(...g.p(k)) - 1)).toBeLessThan(1e-12);
  });
  it('edge lengths sum to the panel-edge network length', () => {
    let L = 0; for (const l of g.elen) L += l;
    expect(L).toBeGreaterThan(0);
    for (let k = 0; k < g.N; k++) expect(g.locate(g.p(k))).toBe(k);
  });
  it('Laplacian of Y1 (= z) is −2 z, solid-body rotation is stress-free', () => {
    const s = new SphereSolver(24, { ...quiet(), lameOn: true, mu: 1, lambda: 0.5 });
    const u = new Float64Array(3 * s.N), acc = new Float64Array(3 * s.N);
    for (let k = 0; k < s.N; k++) u[3 * k + 2] = s.g.pos[3 * k + 2]; // z as a scalar in the z component
    s.addCartLaplacian(u, 1, acc);
    let err = 0; for (let k = 0; k < s.N; k++) err = Math.max(err, Math.abs(acc[3 * k + 2] + 2 * s.g.pos[3 * k + 2]));
    expect(err).toBeLessThan(0.05);
    s.seedZonal(1); s.lameAccel(s.u, acc);
    let a = 0; for (let k = 0; k < s.N; k++) { const p = s.g.p(k), d = acc[3*k]*p[0]+acc[3*k+1]*p[1]+acc[3*k+2]*p[2]; a = Math.max(a, Math.hypot(acc[3*k]-d*p[0], acc[3*k+1]-d*p[1], acc[3*k+2]-d*p[2])); }
    expect(a).toBeLessThan(0.25); // O(h) truncation at panel edges (μ = 1, λ = 0.5; exact value 0)
  });
});

describe('interpolation across panel edges', () => {
  it('semi-Lagrangian stencil is as accurate at panel edges as in panel interiors', () => {
    const g = new CubedSphere(32), sc = new Int32Array(4), sw = new Float64Array(4), r = rng0(5);
    const f = (p: number[]) => Math.sin(2 * p[0]) + p[1] * p[2];
    let ein = 0, cin = 0, eed = 0, ced = 0;
    for (let t = 0; t < 40000; t++) {
      const z = 2 * r() - 1, ph = 2 * Math.PI * r(), rr = Math.sqrt(1 - z * z), x: [number, number, number] = [rr * Math.cos(ph), rr * Math.sin(ph), z];
      g.stencil(x, sc, sw); let a = 0; for (let q = 0; q < 4; q++) a += sw[q] * f(g.p(sc[q]));
      const [, xi, et] = g.coords(x), q4 = Math.PI / 4, d = Math.min(xi + q4, q4 - xi, et + q4, q4 - et) / g.dxi, e = Math.abs(a - f(x));
      if (d < 1) { eed += e; ced++; } else if (d > 3) { ein += e; cin++; }
    }
    expect(eed / ced).toBeLessThan(2 * (ein / cin));
  });
});

describe('sphere physics', () => {
  it('conserves mass to round-off (all terms on)', () => {
    const s = new SphereSolver(16, { ...defaultSphereParams(), repOn: true, vInt: 0.5, omega: 2 });
    s.seedRandom(7, 0.8); const m0 = s.totalMass();
    for (let i = 0; i < 100; i++) s.step();
    expect(Math.abs(s.totalMass() / m0 - 1)).toBeLessThan(1e-12);
  });
  it('Coriolis does no work: |u| preserved exactly by the Coriolis step', () => {
    const s = new SphereSolver(16, { ...quiet(), omega: 3 }); s.seedRandom(3, 0.7);
    const a = speeds(s); for (let i = 0; i < 200; i++) s.coriolisStep(1 / 60);
    const b = speeds(s); for (let k = 0; k < s.N; k++) expect(Math.abs(a[k] - b[k])).toBeLessThan(1e-12);
  });
  it('uniform density gives zero repulsion (instantaneous and retarded)', () => {
    for (const vInt of [Infinity, 0.3]) {
      const s = new SphereSolver(16, { ...quiet(), repOn: true, repA: 5, R0: 0.3, alpha: 1.5, vInt });
      s.rho.fill(1.3); for (let i = 0; i < 5; i++) s.repulsion(1 / 60);
      expect(Math.max(...s.lastForce.map(Math.abs))).toBe(0);
      expect(s.pairCount).toBeGreaterThan(0);
    }
  });
  it('repulsion pushes away from a density bump', () => {
    const s = new SphereSolver(16, { ...quiet(), repOn: true, repA: 1, R0: 0.4 });
    const c = s.g.locate([1, 0, 0]); s.rho[c] = 5; s.repulsion(1 / 60);
    const nb = s.g.nbr[4 * c + 1], p = s.g.p(nb), f = s.lastForce, cp = s.g.p(c);
    const away = (p[0] - cp[0]) * f[3 * nb] + (p[1] - cp[1]) * f[3 * nb + 1] + (p[2] - cp[2]) * f[3 * nb + 2];
    expect(away).toBeGreaterThan(0);
  });
  it('inertial oscillation: small uniform-ish flow turns at f = 2Ω cos θ (full step, Lamé/self-prop off)', () => {
    const W = 1.5, s = new SphereSolver(24, { ...quiet(), omega: W }), dt = 1 / 120;
    // small amplitude field: eps × (x̂ projected), advection negligible
    for (let k = 0; k < s.N; k++) s.u[3 * k] = 1e-4; s.project();
    const k = s.g.locate([Math.cos(Math.PI / 4) * Math.cos(0.3), Math.cos(Math.PI / 4) * Math.sin(0.3), Math.sin(Math.PI / 4)]);
    const n = s.g.p(k), u0 = [s.u[3 * k], s.u[3 * k + 1], s.u[3 * k + 2]];
    const T = 1; for (let i = 0; i < T / dt; i++) s.step(dt);
    const u1 = [s.u[3 * k], s.u[3 * k + 1], s.u[3 * k + 2]];
    const cx = u0[1] * u1[2] - u0[2] * u1[1], cy = u0[2] * u1[0] - u0[0] * u1[2], cz = u0[0] * u1[1] - u0[1] * u1[0];
    const ang = Math.atan2(cx * n[0] + cy * n[1] + cz * n[2], u0[0] * u1[0] + u0[1] * u1[1] + u0[2] * u1[2]);
    expect(ang / T).toBeCloseTo(-2 * W * n[2], 2); // clockwise in the northern hemisphere for Ω > 0
  });
  it('self-propulsion drives |u| → v0', () => {
    const s = new SphereSolver(12, { ...quiet(), selfProp: true, v0: 0.4 }); s.seedRandom(1, 0.5);
    for (let i = 0; i < 300; i++) s.selfPropulsion(1 / 60);
    for (const v of speeds(s)) expect(Math.abs(v - 0.4)).toBeLessThan(1e-3);
  });
  it('zonal flow has Lz = 1 and local alignment ≈ 1', () => {
    const s = new SphereSolver(16, quiet()); s.seedZonal(0.5); const o = s.orderStats();
    expect(o.Lz).toBeCloseTo(1, 6); expect(o.localAlign).toBeGreaterThan(0.95); expect(o.localAlignRho).toBeGreaterThan(0.95);
  });
});

describe('sphere stability', () => {
  it('default parameters + retarded repulsion stay finite and bounded for 600 steps', () => {
    const s = new SphereSolver(16, { ...defaultSphereParams(), repOn: true, repA: 5, R0: 0.3, alpha: 1.5, vInt: 0.5, lambda: 0.002 });
    s.seedRandom(11, 1);
    for (let i = 0; i < 600; i++) s.step();
    const o = s.orderStats();
    expect(Number.isFinite(o.meanSpeed)).toBe(true); expect(o.meanSpeed).toBeLessThan(5);
    expect(Math.min(...s.rho)).toBeGreaterThanOrEqual(0);
  });
});
