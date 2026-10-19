/**
 * Flocking-style order parameters of a velocity field (Vicsek et al. 1995;
 * Cavagna et al., PNAS 107 (2010) 11865): pairwise alignment distribution
 * P(cos θ_ij), its exact mean, and the radial velocity correlation C(r).
 *
 * Works on cell-centred velocities (FluidSolver.uc / vc after
 * updateDiagnostics()). Only DIRECTIONS v̂ = u/|u| enter, as in the starling
 * analysis; cells whose speed is below `minSpeedFrac` × mean speed have no
 * meaningful direction and are excluded (mask m = 0).
 *
 * - ⟨cos θ_ij⟩ over all pairs i ≠ j is exact: (|Σv̂|² − N) / (N(N − 1)), which
 *   equals φ_dir² − (1 − φ_dir²)/(N − 1) with φ_dir = |Σv̂|/N.
 * - The histogram of cos θ_ij is estimated from `pairs` random pairs (i ≠ j).
 * - C(r) = Σ_{x} m(x)m(x+r) v̂(x)·v̂(x+r) / Σ m(x)m(x+r), radially binned in
 *   cells (pair-weighted), via FFT autocorrelation: circular on periodic axes
 *   (minimum-image separations), zero-padded (non-wrapping) on walled axes.
 *   The connected version uses δv̂ = v̂ − ⟨v̂⟩ (mean over included cells) and
 *   is normalised by ⟨|δv̂|²⟩, so C_conn(0) = 1. Its first zero crossing is the
 *   correlation length ξ (Cavagna et al.); the 1/e crossing is reported too.
 */
import { FFT } from './fft1';

export interface OrderOptions {
  bins?: number;            // histogram bins on [−1, 1] (default 40)
  pairs?: number;           // sampled pairs for the histogram (default 30000)
  minSpeedFrac?: number;    // exclusion threshold, fraction of the mean speed (default 0.02)
  periodicX?: boolean;
  periodicY?: boolean;
  correlation?: boolean;    // compute C(r) (default true)
  seed?: number;            // RNG seed for the pair sampling
}

export interface OrderStats {
  n: number;                // included cells
  excluded: number;
  meanCos: number;          // exact ⟨cos θ_ij⟩, i ≠ j
  phiDir: number;           // |Σv̂| / N
  hist: Float64Array;       // probability density on [−1, 1] (∫ = 1)
  sampledMeanCos: number;   // mean over the sampled pairs (check of meanCos)
  r: Float64Array;          // bin centres in cells (bin k = separations in [k − ½, k + ½))
  C: Float64Array;          // full C(r)
  Cconn: Float64Array;      // connected, normalised C(r)
  xi0: number;              // first zero of Cconn (cells), NaN if none within range
  xiE: number;              // first 1/e crossing of Cconn (cells), NaN if none
}

/** 2D complex FFT workspace of size A × B (row-major, index a + A·b). */
class FFT2 {
  readonly re: Float64Array; readonly im: Float64Array;
  private readonly f1: FFT; private readonly f2: FFT;
  private readonly t1r: Float64Array; private readonly t1i: Float64Array;
  private readonly t2r: Float64Array; private readonly t2i: Float64Array;
  constructor(readonly A: number, readonly B: number) {
    this.re = new Float64Array(A * B); this.im = new Float64Array(A * B);
    this.f1 = new FFT(A); this.f2 = new FFT(B);
    this.t1r = new Float64Array(A); this.t1i = new Float64Array(A);
    this.t2r = new Float64Array(B); this.t2i = new Float64Array(B);
  }
  run(forward: boolean): void {
    const { A, B, re, im, t1r, t1i, t2r, t2i } = this;
    for (let b = 0; b < B; b++) {
      const o = A * b;
      for (let a = 0; a < A; a++) { t1r[a] = re[o + a]; t1i[a] = im[o + a]; }
      if (forward) this.f1.forward(t1r, t1i); else this.f1.inverse(t1r, t1i);
      for (let a = 0; a < A; a++) { re[o + a] = t1r[a]; im[o + a] = t1i[a]; }
    }
    for (let a = 0; a < A; a++) {
      for (let b = 0; b < B; b++) { t2r[b] = re[a + A * b]; t2i[b] = im[a + A * b]; }
      if (forward) this.f2.forward(t2r, t2i); else this.f2.inverse(t2r, t2i);
      for (let b = 0; b < B; b++) { re[a + A * b] = t2r[b]; im[a + A * b] = t2i[b]; }
    }
  }
  /** In place: (re, im) ← Re[ Σ_x conj(z(x)) z(x + r) ] (autocorrelation, real part). */
  autocorr(): void {
    this.run(true);
    const { re, im } = this;
    for (let k = 0; k < re.length; k++) { re[k] = re[k] * re[k] + im[k] * im[k]; im[k] = 0; }
    this.run(false);
  }
}

const nextPow2 = (n: number) => { let m = 1; while (m < n) m <<= 1; return m; };

/** xorshift32 */
function rng(seed: number): () => number {
  let s = (seed >>> 0) || 0x9e3779b9;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

export class OrderAnalyzer {
  private ws: FFT2 | null = null;
  private key = '';
  private ux = new Float64Array(0); private uy = new Float64Array(0); private m = new Uint8Array(0);
  private ids = new Int32Array(0);
  private calls = 0;
  private mapKey = '';
  private binMap = new Int32Array(0);
  private zr = new Float64Array(0); private zi = new Float64Array(0);
  private mr = new Float64Array(0); private mi = new Float64Array(0);

  /**
   * Analyse an nx × ny field given by a getter (cell (i, j), 0-based) or by
   * cell-centred arrays with stride (i + 1 + stride·(j + 1), the solver layout).
   */
  analyze(nx: number, ny: number, get: (i: number, j: number) => [number, number], opt: OrderOptions = {}): OrderStats {
    return this.run(nx, ny, (ux, uy) => {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const [a, b] = get(i, j); ux[i + nx * j] = a; uy[i + nx * j] = b; }
    }, opt);
  }

  /** Convenience wrapper for the solver layout (cell-centred arrays with ghost ring). */
  analyzeCells(nx: number, ny: number, stride: number, uc: ArrayLike<number>, vc: ArrayLike<number>, opt: OrderOptions = {}): OrderStats {
    return this.run(nx, ny, (ux, uy) => {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const k = i + 1 + stride * (j + 1); ux[i + nx * j] = uc[k]; uy[i + nx * j] = vc[k]; }
    }, opt);
  }

  private run(nx: number, ny: number, fill: (ux: Float64Array, uy: Float64Array) => void, opt: OrderOptions): OrderStats {
    const bins = opt.bins ?? 40, pairs = opt.pairs ?? 30000, frac = opt.minSpeedFrac ?? 0.02;
    const px = !!opt.periodicX, py = !!opt.periodicY;
    const N0 = nx * ny;
    if (this.ux.length !== N0) {
      this.ux = new Float64Array(N0); this.uy = new Float64Array(N0); this.m = new Uint8Array(N0); this.ids = new Int32Array(N0);
    }
    const { ux, uy, m, ids } = this;
    fill(ux, uy);
    let sp = 0;
    for (let k = 0; k < N0; k++) sp += Math.hypot(ux[k], uy[k]);
    const thr = Math.max(1e-12, frac * sp / N0);
    let n = 0, sx = 0, sy = 0;
    for (let k = 0; k < N0; k++) {
      const s = Math.hypot(ux[k], uy[k]);
      if (s > thr) { ux[k] /= s; uy[k] /= s; m[k] = 1; ids[n++] = k; sx += ux[k]; sy += uy[k]; }
      else { ux[k] = 0; uy[k] = 0; m[k] = 0; }
    }
    const meanCos = n > 1 ? (sx * sx + sy * sy - n) / (n * (n - 1)) : NaN;
    const phiDir = n > 0 ? Math.hypot(sx, sy) / n : NaN;

    // histogram from random pairs
    const hist = new Float64Array(bins);
    let sampled = 0, sumCos = 0;
    if (n > 1) {
      const r = rng(opt.seed ?? (0x2545f491 + 7919 * this.calls++));
      for (let p = 0; p < pairs; p++) {
        const a = ids[Math.floor(r() * n)];
        let b = ids[Math.floor(r() * (n - 1))];
        if (b === a) b = ids[n - 1];
        const c = Math.max(-1, Math.min(1, ux[a] * ux[b] + uy[a] * uy[b]));
        hist[Math.min(bins - 1, Math.floor(((c + 1) / 2) * bins))]++;
        sumCos += c; sampled++;
      }
      const w = bins / (2 * sampled);
      for (let k = 0; k < bins; k++) hist[k] *= w;
    }

    const rMax = Math.floor(Math.min(nx, ny) / 2);
    const nb = rMax + 1;
    const rr = new Float64Array(nb); for (let k = 0; k < nb; k++) rr[k] = k;
    const C = new Float64Array(nb).fill(NaN), Cc = new Float64Array(nb).fill(NaN);
    let xi0 = NaN, xiE = NaN;
    if (opt.correlation !== false && n > 1) {
      const A = px ? nx : nextPow2(2 * nx), B = py ? ny : nextPow2(2 * ny);
      const key = `${A}x${B}`;
      if (this.key !== key || !this.ws) { this.ws = new FFT2(A, B); this.key = key; }
      const ws = this.ws;
      const mapKey = `${A}x${B}:${nx}x${ny}:${px}${py}`;
      if (this.mapKey !== mapKey) {
        // offset (a, b) of the correlation array → radial bin, −1 if out of range / impossible
        this.binMap = new Int32Array(A * B);
        for (let b = 0; b < B; b++) for (let a = 0; a < A; a++) {
          const dx = a <= A / 2 ? a : a - A, dy = b <= B / 2 ? b : b - B;
          let d = Math.round(Math.hypot(dx, dy));
          if ((!px && Math.abs(dx) >= nx) || (!py && Math.abs(dy) >= ny) || d > rMax) d = -1;
          this.binMap[a + A * b] = d;
        }
        this.mapKey = mapKey;
      }
      const binMap = this.binMap;
      const L = A * B;
      if (this.zr.length !== L) {
        this.zr = new Float64Array(L); this.zi = new Float64Array(L); this.mr = new Float64Array(L); this.mi = new Float64Array(L);
      }
      const { zr, zi, mr, mi } = this;
      const cnt = new Float64Array(nb), num = new Float64Array(nb), numc = new Float64Array(nb);
      const accumulate = (src: Float64Array, into: Float64Array) => {
        for (let q = 0; q < L; q++) { const d = binMap[q]; if (d >= 0) into[d] += src[q]; }
      };
      // forward transforms of z = v̂x + i v̂y and of the mask m
      ws.re.fill(0); ws.im.fill(0);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const k = i + nx * j; ws.re[i + A * j] = ux[k]; ws.im[i + A * j] = uy[k]; }
      ws.run(true); zr.set(ws.re); zi.set(ws.im);
      const trivialMask = n === N0 && px && py; // every offset pairs all N cells
      if (trivialMask) { mr.fill(0); mi.fill(0); mr[0] = N0; }
      else {
        ws.re.fill(0); ws.im.fill(0);
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) ws.re[i + A * j] = m[i + nx * j];
        ws.run(true); mr.set(ws.re); mi.set(ws.im);
      }
      // Re autocorr(f) = IFFT of the even part of |F|²; pack ⟨v̂·v̂'⟩ (real) and ⟨δ·δ'⟩ (imag)
      // into one inverse transform. δ = v̂ − μ m with μ = mean direction, so Δ̂ = Ẑ − μ M̂.
      const mx = sx / n, my = sy / n;
      const P = (k: number, pr: Float64Array, pi: Float64Array, cr: number, ci: number) => {
        const r = zr[k] - (cr * pr[k] - ci * pi[k]), i = zi[k] - (cr * pi[k] + ci * pr[k]);
        return r * r + i * i;
      };
      for (let bq = 0; bq < B; bq++) {
        const bm = bq === 0 ? 0 : B - bq;
        for (let aq = 0; aq < A; aq++) {
          const k = aq + A * bq, km = (aq === 0 ? 0 : A - aq) + A * bm;
          ws.re[k] = 0.5 * (zr[k] * zr[k] + zi[k] * zi[k] + zr[km] * zr[km] + zi[km] * zi[km]);
          ws.im[k] = 0.5 * (P(k, mr, mi, mx, my) + P(km, mr, mi, mx, my));
        }
      }
      ws.run(false); accumulate(ws.re, num); accumulate(ws.im, numc);
      if (trivialMask) { for (let q = 0; q < L; q++) { const d = binMap[q]; if (d >= 0) cnt[d] += N0; } }
      else {
        for (let k = 0; k < L; k++) { ws.re[k] = mr[k] * mr[k] + mi[k] * mi[k]; ws.im[k] = 0; }
        ws.run(false); accumulate(ws.re, cnt);
      }
      const c0 = numc[0] / cnt[0];
      for (let d = 0; d < nb; d++) {
        if (cnt[d] > 0.5) { C[d] = num[d] / cnt[d]; Cc[d] = c0 > 1e-12 ? numc[d] / cnt[d] / c0 : NaN; }
      }
      const cross = (level: number): number => {
        for (let d = 1; d < nb; d++) {
          const a = Cc[d - 1], b = Cc[d];
          if (Number.isFinite(a) && Number.isFinite(b) && a > level && b <= level) return d - 1 + (a - level) / (a - b);
        }
        return NaN;
      };
      xi0 = cross(0); xiE = cross(Math.exp(-1));
    }
    return { n, excluded: N0 - n, meanCos, phiDir, hist, sampledMeanCos: sampled ? sumCos / sampled : NaN,
      r: rr, C, Cconn: Cc, xi0, xiE };
  }
}
