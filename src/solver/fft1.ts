/**
 * Small complex FFT for arbitrary lengths (in place on separate re/im
 * Float64Arrays): iterative radix-2 for powers of two, Bluestein's chirp-z
 * algorithm (via a power-of-two convolution) for every other length, so it is
 * O(n log n) for any grid size. Used by the direct periodic Poisson solver.
 */
export class FFT {
  readonly n: number;
  private readonly pow2: boolean;
  // radix-2 tables (for n itself, or for the Bluestein convolution length m)
  private readonly m: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  // Bluestein
  private readonly wr?: Float64Array;
  private readonly wi?: Float64Array;
  private readonly br?: Float64Array;
  private readonly bi?: Float64Array;
  private readonly ar?: Float64Array;
  private readonly ai?: Float64Array;

  constructor(n: number) {
    if (!Number.isInteger(n) || n < 1) throw new Error(`FFT length must be a positive integer (got ${n})`);
    this.n = n;
    this.pow2 = (n & (n - 1)) === 0;
    let m = n;
    if (!this.pow2) {
      m = 1;
      while (m < 2 * n - 1) m <<= 1;
    }
    this.m = m;
    let bits = 0;
    while (1 << bits < m) bits++;
    this.rev = new Uint32Array(m);
    for (let i = 0; i < m; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(m >> 1 || 1);
    this.sin = new Float64Array(m >> 1 || 1);
    for (let k = 0; k < m >> 1; k++) {
      this.cos[k] = Math.cos((-2 * Math.PI * k) / m);
      this.sin[k] = Math.sin((-2 * Math.PI * k) / m);
    }
    if (!this.pow2) {
      // chirp w_k = exp(−iπk²/n); k² is reduced mod 2n to keep the angle accurate
      this.wr = new Float64Array(n);
      this.wi = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        const a = (-Math.PI * ((k * k) % (2 * n))) / n;
        this.wr[k] = Math.cos(a);
        this.wi[k] = Math.sin(a);
      }
      this.br = new Float64Array(m);
      this.bi = new Float64Array(m);
      for (let k = 0; k < n; k++) {
        this.br[k] = this.wr[k];
        this.bi[k] = -this.wi[k];
        if (k > 0) {
          this.br[m - k] = this.wr[k];
          this.bi[m - k] = -this.wi[k];
        }
      }
      this.radix2(this.br, this.bi);
      this.ar = new Float64Array(m);
      this.ai = new Float64Array(m);
    }
  }

  /** Forward DFT X_k = Σ x_j e^(−2πijk/n), in place. */
  forward(re: Float64Array, im: Float64Array): void {
    if (this.pow2) this.radix2(re, im);
    else this.bluestein(re, im);
  }

  /** Inverse DFT (including the 1/n factor), in place. */
  inverse(re: Float64Array, im: Float64Array): void {
    const n = this.n;
    for (let k = 0; k < n; k++) im[k] = -im[k];
    this.forward(re, im);
    const inv = 1 / n;
    for (let k = 0; k < n; k++) {
      re[k] *= inv;
      im[k] = -im[k] * inv;
    }
  }

  private radix2(re: Float64Array, im: Float64Array): void {
    const m = this.m;
    const rev = this.rev;
    for (let i = 0; i < m; i++) {
      const r = rev[i];
      if (r > i) {
        let t = re[i]; re[i] = re[r]; re[r] = t;
        t = im[i]; im[i] = im[r]; im[r] = t;
      }
    }
    for (let size = 2; size <= m; size <<= 1) {
      const half = size >> 1;
      const step = m / size;
      for (let start = 0; start < m; start += size) {
        for (let k = 0; k < half; k++) {
          const c = this.cos[k * step];
          const s = this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * c - im[b] * s;
          const ti = re[b] * s + im[b] * c;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }

  private bluestein(re: Float64Array, im: Float64Array): void {
    const { n, m } = this;
    const wr = this.wr!, wi = this.wi!, br = this.br!, bi = this.bi!, ar = this.ar!, ai = this.ai!;
    for (let k = 0; k < n; k++) {
      ar[k] = re[k] * wr[k] - im[k] * wi[k];
      ai[k] = re[k] * wi[k] + im[k] * wr[k];
    }
    ar.fill(0, n);
    ai.fill(0, n);
    this.radix2(ar, ai);
    for (let k = 0; k < m; k++) {
      const r = ar[k] * br[k] - ai[k] * bi[k];
      ai[k] = ar[k] * bi[k] + ai[k] * br[k];
      ar[k] = r;
    }
    // inverse radix-2 via conjugation
    for (let k = 0; k < m; k++) ai[k] = -ai[k];
    this.radix2(ar, ai);
    const inv = 1 / m;
    for (let k = 0; k < n; k++) {
      const cr = ar[k] * inv;
      const ci = -ai[k] * inv;
      re[k] = cr * wr[k] - ci * wi[k];
      im[k] = cr * wi[k] + ci * wr[k];
    }
  }
}
