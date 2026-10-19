/** In-place radix-2 complex FFT and a 2D wrapper for n × n power-of-two grids. */
export class FFT2 {
  readonly n: number;
  private cos: Float64Array; private sin: Float64Array; private rev: Uint32Array;
  private tr = new Float64Array(0); private ti = new Float64Array(0);
  constructor(n: number) {
    if (n & (n - 1)) throw new Error('n must be a power of two');
    this.n = n;
    this.cos = new Float64Array(n / 2); this.sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) { this.cos[k] = Math.cos(2 * Math.PI * k / n); this.sin[k] = -Math.sin(2 * Math.PI * k / n); }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); this.rev[i] = r; }
    this.tr = new Float64Array(n); this.ti = new Float64Array(n);
  }
  /** 1D transform of re/im (length n), sign = −1 forward, +1 inverse (unnormalised). */
  fft1(re: Float64Array, im: Float64Array, inverse: boolean) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    const sg = inverse ? -1 : 1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sg * sin[k * step], a = s + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
      }
    }
  }
  /** 2D transform in place (row-major n × n). Inverse is normalised by 1/n². */
  fft2(re: Float64Array, im: Float64Array, inverse = false) {
    const { n, tr, ti } = this;
    for (let j = 0; j < n; j++) {
      const o = j * n; for (let i = 0; i < n; i++) { tr[i] = re[o + i]; ti[i] = im[o + i]; }
      this.fft1(tr, ti, inverse);
      for (let i = 0; i < n; i++) { re[o + i] = tr[i]; im[o + i] = ti[i]; }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) { tr[j] = re[j * n + i]; ti[j] = im[j * n + i]; }
      this.fft1(tr, ti, inverse);
      for (let j = 0; j < n; j++) { re[j * n + i] = tr[j]; im[j * n + i] = ti[j]; }
    }
    if (inverse) { const s = 1 / (n * n); for (let k = 0; k < n * n; k++) { re[k] *= s; im[k] *= s; } }
  }
}
