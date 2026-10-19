import type { Solver } from './solver';

export type Preset = 'acoustic' | 'kh' | 'jets' | 'turbulence' | 'stir' | 'rest';

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** Periodic distance component on the unit torus. */
export const pd = (a: number) => a - Math.round(a);

export function setState(s: Solver, f: (x: number, y: number) => [number, number, number, number]) {
  const { n, h } = s;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i, [r, u, v, c] = f((i + 0.5) * h, (j + 0.5) * h);
    s.rho[k] = r; s.mx[k] = r * u; s.my[k] = r * v; s.rs[k] = r * c;
  }
  s.t = 0; s.floorHits = 0; s.fx.fill(0); s.fy.fill(0);
}

/** Add a Gaussian density (pressure) pulse at (x0, y0). */
export function addPulse(s: Solver, x0: number, y0: number, amp = 0.2, sigma = 0.025) {
  const { n, h } = s;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i, dx = pd((i + 0.5) * h - x0), dy = pd((j + 0.5) * h - y0);
    const g = amp * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    const r0 = s.rho[k], r1 = r0 * (1 + g);
    s.mx[k] *= r1 / r0; s.my[k] *= r1 / r0; s.rs[k] *= r1 / r0; s.rho[k] = r1;
  }
}

export function applyPreset(s: Solver, p: Preset, seed = 1) {
  const c = s.params.c;
  switch (p) {
    case 'rest': setState(s, () => [1, 0, 0, 0]); break;
    case 'acoustic':
      setState(s, () => [1, 0, 0, 0]);
      addPulse(s, 0.5, 0.5, 0.2, 0.03);
      break;
    case 'kh': {
      const U = 0.25 * c, d = 0.01;
      setState(s, (x, y) => {
        const band = Math.tanh((y - 0.25) / d) - Math.tanh((y - 0.75) / d) - 1; // +1 inside, −1 outside
        const pert = 0.01 * c * Math.sin(4 * Math.PI * x) * (Math.exp(-((y - 0.25) ** 2) / 0.002) + Math.exp(-((y - 0.75) ** 2) / 0.002));
        return [1, U * band, pert, 0.5 * (band + 1)];
      });
      break;
    }
    case 'jets': {
      const U = 0.6 * c;
      setState(s, (x, y) => {
        const w = Math.exp(-((y - 0.5) ** 2) / (2 * 0.04 ** 2)) * (1 + 0.02 * Math.sin(10 * Math.PI * y));
        const u = x < 0.5 ? U * w * Math.sin(Math.PI * x / 0.5) : -U * w * Math.sin(Math.PI * (x - 0.5) / 0.5);
        return [1, u, 0, x < 0.5 ? w : 0];
      });
      break;
    }
    case 'turbulence': {
      const r = rng(seed), modes: [number, number, number, number][] = [];
      for (let m = 0; m < 40; m++) {
        const kx = Math.floor(r() * 13) - 6, ky = Math.floor(r() * 13) - 6;
        if (kx === 0 && ky === 0) continue;
        modes.push([kx, ky, r() * 2 * Math.PI, 1 / Math.hypot(kx, ky)]);
      }
      const amp = 0.08 * c;
      setState(s, (x, y) => {
        let u = 0, v = 0;
        for (const [kx, ky, ph, a] of modes) { // streamfunction modes: solenoidal start
          const sn = Math.cos(2 * Math.PI * (kx * x + ky * y) + ph) * a;
          u += ky * sn; v -= kx * sn;
        }
        return [1, amp * u / 3, amp * v / 3, x < 0.5 ? 1 : 0];
      });
      break;
    }
    case 'stir':
      setState(s, (x, y) => [1, 0, 0, Math.hypot(pd(x - 0.5), pd(y - 0.5)) < 0.15 ? 1 : 0]);
      break;
  }
}

/** Rotating stirrer: a force disk circling the centre (used by the 'stir' preset). */
export function stirForce(s: Solver, t: number, strength = 3) {
  const { n, h } = s, R = 0.25, w = 2 * Math.PI * 0.25;
  const x0 = 0.5 + R * Math.cos(w * t), y0 = 0.5 + R * Math.sin(w * t);
  const dirx = -Math.sin(w * t), diry = Math.cos(w * t);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i, dx = pd((i + 0.5) * h - x0), dy = pd((j + 0.5) * h - y0);
    const g = strength * s.params.c * Math.exp(-(dx * dx + dy * dy) / (2 * 0.03 ** 2));
    s.fx[k] = g * dirx * s.rho[k]; s.fy[k] = g * diry * s.rho[k];
  }
}
