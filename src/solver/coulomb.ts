/**
 * 2D Coulomb repulsion (log potential, force ∝ 1/d, no cutoff) on the periodic unit square.
 *
 * Potential with neutralising uniform background:  −∇²φ = 2π (ρ − ρ̄),  force  f = −A ∇φ.
 * In free space this is f(x) = A ∫ ρ(y) (x − y)/|x − y|² dy; on the torus the periodic Green's
 * function automatically sums all periodic images (and the background makes the sum converge).
 *
 * Grid form: f(x) = A Σ_{o ≠ 0} W(o) ρ(x − o) with W(o) = h² F_per(o h), F_per = exact periodic
 * point-source force (all images + neutralising background, Ewald-summed, see periodicCoulombForce).
 * Far from the images W(o) ≈ h² o/|o|². Σ_o W(o) = 0, so uniform ρ gives exactly zero force; the self
 * cell (o = 0) is excluded. Instantaneous: FFT circular convolution with W (identical to the direct sum).
 * Retarded version: the source at offset o is seen with delay τ(o) = d_min(o)/v_int, d_min =
 * minimum-image distance, quantised to bins of the time step: τ = q·Δt, q = round(d_min/(v_int Δt)).
 */
import { FFT } from './fft1';
export interface CoulombKernel {
  n: number;
  /** W_x, W_y per offset index o = oj·n + oi (offset (oi, oj) mod n), unit strength. */
  wx: Float64Array; wy: Float64Array;
  /** Delay bin q per offset (0 = instantaneous). */
  bin: Uint32Array; nBins: number;
  /** Offsets sorted by bin, and start index of each bin in `order` (length nBins + 1). */
  order: Uint32Array; binStart: Uint32Array;
  binDt: number; vInt: number;
}

/** In-place 2D DFT (forward or inverse) of re/im (n×n, row-major). */
export function fft2(fft: FFT, re: Float64Array, im: Float64Array, inverse: boolean) {
  const n = fft.n, r = new Float64Array(n), c = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const o = j * n;
    for (let i = 0; i < n; i++) { r[i] = re[o + i]; c[i] = im[o + i]; }
    inverse ? fft.inverse(r, c) : fft.forward(r, c);
    for (let i = 0; i < n; i++) { re[o + i] = r[i]; im[o + i] = c[i]; }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) { r[j] = re[j * n + i]; c[j] = im[j * n + i]; }
    inverse ? fft.inverse(r, c) : fft.forward(r, c);
    for (let j = 0; j < n; j++) { re[j * n + i] = r[j]; im[j * n + i] = c[j]; }
  }
}

const fftCache = new Map<number, FFT>();
const getFFT = (n: number) => { let f = fftCache.get(n); if (!f) { f = new FFT(n); fftCache.set(n, f); } return f; };

/**
 * Exact periodic point-source force (unit mass, neutralising background) at displacement (x, y):
 *   F(r) = Σ_L (r+L)/|r+L|²  − background,  evaluated by Ewald splitting with width σ:
 *   real part   Σ_L ê_L e^{−|r+L|²/(2σ²)} / |r+L|          (3×3 images)
 *   recip. part Σ_{k≠0} 2π k sin(k·r) e^{−k²σ²/2} / k²      (|m| ≤ 12, k = 2πm)
 * Accurate to ~1e-7 relative; the k = 0 term is absent (= uniform neutralising background).
 */
const SIG = 0.08, MMAX = 12;
export function periodicCoulombForce(x: number, y: number): [number, number] {
  let fx = 0, fy = 0;
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
    const X = x + a, Y = y + b, r2 = X * X + Y * Y;
    if (r2 === 0) continue;
    const g = Math.exp(-r2 / (2 * SIG * SIG)) / r2; fx += g * X; fy += g * Y;
  }
  const tp = 2 * Math.PI;
  for (let mi = -MMAX; mi <= MMAX; mi++) for (let mj = -MMAX; mj <= MMAX; mj++) {
    if (!mi && !mj) continue;
    const kx = tp * mi, ky = tp * mj, k2 = kx * kx + ky * ky, w = tp * Math.sin(kx * x + ky * y) * Math.exp(-k2 * SIG * SIG / 2) / k2;
    fx += w * kx; fy += w * ky;
  }
  return [fx, fy];
}

const wCache = new Map<number, { wx: Float64Array; wy: Float64Array; Wr: Float64Array; Wi: Float64Array }>();
/**
 * Direct-sum weights W(o) = h²·F_periodic(o·h) (unit strength), o = 0 excluded, plus their DFT
 * (Wr + i·Wi = DFT[W_x + i W_y]) for the FFT convolution. Σ_o W(o) = 0 (odd kernel).
 */
export function coulombWeights(n: number) {
  let c = wCache.get(n);
  if (c) return c;
  const N = n * n, h = 1 / n, wx = new Float64Array(N), wy = new Float64Array(N);
  for (let oj = 0; oj < n; oj++) for (let oi = 0; oi < n; oi++) {
    if (!oi && !oj) continue;
    const k = oj * n + oi, a = oi > n / 2 ? oi - n : oi, b = oj > n / 2 ? oj - n : oj;
    // exact antisymmetry on even grids at the n/2 lines: F is odd and periodic → 0 component there
    let [fx, fy] = periodicCoulombForce(a * h, b * h);
    if (2 * oi === n) fx = 0; if (2 * oj === n) fy = 0;
    wx[k] = fx * h * h; wy[k] = fy * h * h;
  }
  const Wr = Float64Array.from(wx), Wi = Float64Array.from(wy);
  fft2(getFFT(n), Wr, Wi, false);
  c = { wx, wy, Wr, Wi }; wCache.set(n, c); return c;
}

/**
 * Instantaneous periodic Coulomb force (unit strength) via FFT circular convolution with W:
 *   f = Σ_o W(o) ρ(x − o)  (identical to the direct sum), O(N log N).
 */
export function coulombFFT(n: number, rho: ArrayLike<number>, ox: Float64Array, oy: Float64Array) {
  const N = n * n, re = new Float64Array(N), im = new Float64Array(N), { Wr, Wi } = coulombWeights(n), f = getFFT(n);
  for (let k = 0; k < N; k++) re[k] = rho[k];
  fft2(f, re, im, false);
  // ρ real → DFT(W_x ⋆ ρ) + i DFT(W_y ⋆ ρ) = (Wr + i Wi)·ρ̂
  for (let k = 0; k < N; k++) { const r = re[k], i = im[k]; re[k] = Wr[k] * r - Wi[k] * i; im[k] = Wr[k] * i + Wi[k] * r; }
  fft2(f, re, im, true);
  for (let k = 0; k < N; k++) { ox[k] = re[k]; oy[k] = im[k]; }
}

/** Minimum-image distance of grid offset (oi, oj) in units of the box (h = 1/n). */
export function minImageDist(n: number, oi: number, oj: number) {
  const a = oi > n / 2 ? oi - n : oi, b = oj > n / 2 ? oj - n : oj;
  return Math.hypot(a, b) / n;
}
export const COULOMB_DMAX = Math.SQRT1_2; // largest minimum-image distance on the unit torus

let kerCache: CoulombKernel | null = null;
/** Kernel + delay bins for grid n, propagation speed vInt (Infinity = 1 bin), bin width binDt (= Δt). */
export function coulombKernel(n: number, vInt: number, binDt: number): CoulombKernel {
  const inst = !(vInt > 0) || !Number.isFinite(vInt);
  const c = kerCache;
  if (c && c.n === n && c.binDt === binDt && (c.vInt === vInt || (inst && c.nBins === 1))) return c;
  const N = n * n, { wx, wy } = coulombWeights(n), bin = new Uint32Array(N);
  let nBins = 1;
  if (!inst) for (let oj = 0; oj < n; oj++) for (let oi = 0; oi < n; oi++) {
    const q = Math.round(minImageDist(n, oi, oj) / (vInt * binDt));
    bin[oj * n + oi] = q; if (q + 1 > nBins) nBins = q + 1;
  }
  const binStart = new Uint32Array(nBins + 1);
  for (let k = 1; k < N; k++) binStart[bin[k] + 1]++;
  for (let q = 0; q < nBins; q++) binStart[q + 1] += binStart[q];
  const fill = binStart.slice(0, nBins), order = new Uint32Array(N - 1);
  for (let k = 1; k < N; k++) order[fill[bin[k]]++] = k;
  kerCache = { n, wx, wy, bin, nBins, order, binStart, binDt, vInt };
  return kerCache;
}
