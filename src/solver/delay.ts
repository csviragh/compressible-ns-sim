/**
 * Ring buffer of past velocity fields for the delayed viscosity terms.
 *
 * Every frame holds the full staggered u and v arrays (ghost layers included,
 * Float32, exactly as the solver stores them), time-stamped with simulation
 * time. Frames are recorded at most every `interval` seconds (= Δt unless the
 * memory budget forces coarser sampling) and kept only as long as the longest
 * delay τ_max needs them. Reads interpolate linearly in time; a read before
 * the oldest frame returns the oldest frame (start-up: the history is
 * "pre-filled" with the earliest state), a read at or after the present
 * returns the current field supplied by the caller.
 */

interface Frame { t: number; u: Float32Array; v: Float32Array }

export class FieldHistory {
  frames: Frame[] = [];
  private pool: Frame[] = [];
  private uLen = 0;
  private vLen = 0;
  interval = 0;
  capacity = 0;
  tauMax = 0;
  static readonly MAX_FRAMES = 512;

  configure(uLen: number, vLen: number, dt: number, tauMax: number, budgetBytes: number): void {
    if (uLen !== this.uLen || vLen !== this.vLen) { this.clear(); this.pool = []; }
    this.uLen = uLen;
    this.vLen = vLen;
    this.tauMax = tauMax;
    this.capacity = Math.max(3, Math.min(FieldHistory.MAX_FRAMES, Math.floor(budgetBytes / this.frameBytes())));
    // capacity − 2 intervals must cover τ_max (one extra frame brackets it)
    this.interval = Math.max(dt, tauMax / (this.capacity - 2));
    while (this.frames.length > this.capacity) this.pool.push(this.frames.shift()!);
  }

  frameBytes(): number { return 4 * (this.uLen + this.vLen); }
  bytes(): number { return this.frames.length * this.frameBytes(); }
  clear(): void { this.pool.push(...this.frames); this.frames = []; }

  /** Should the field at time t be recorded? */
  wants(t: number): boolean {
    const f = this.frames;
    return f.length === 0 || t - f[f.length - 1].t >= this.interval * (1 - 1e-9);
  }

  /** Record (a copy of) the fields at time t. */
  push(t: number, u: ArrayLike<number>, v: ArrayLike<number>): void {
    let f = this.pool.pop();
    if (!f || f.u.length !== this.uLen || f.v.length !== this.vLen) {
      f = { t, u: new Float32Array(this.uLen), v: new Float32Array(this.vLen) };
    }
    f.t = t;
    f.u.set(u);
    f.v.set(v);
    this.frames.push(f);
    while (this.frames.length > this.capacity) this.pool.push(this.frames.shift()!);
  }

  /** Drop frames no read at time ≥ tNext can need. */
  prune(tNext: number): void {
    const f = this.frames;
    while (f.length >= 2 && f[1].t <= tNext - this.tauMax) this.pool.push(f.shift()!);
  }

  /**
   * Interpolation weights for the field at time T, given the present time
   * `now` (index −1 = the caller's current field): value = (1 − θ)·F[a] + θ·F[b].
   */
  lookup(T: number, now: number): { a: number; b: number; theta: number } {
    const f = this.frames;
    if (T >= now || f.length === 0) return { a: -1, b: -1, theta: 0 };
    if (T <= f[0].t) return { a: 0, b: 0, theta: 0 };
    let lo = 0, hi = f.length; // last frame with t ≤ T
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid].t <= T) lo = mid; else hi = mid; }
    const ta = f[lo].t;
    const b = lo + 1 < f.length ? lo + 1 : -1;
    const tb = b < 0 ? now : f[b].t;
    const theta = tb > ta ? (T - ta) / (tb - ta) : 0;
    if (theta <= 0) return { a: lo, b: lo, theta: 0 };
    if (theta >= 1) return { a: b, b, theta: 0 };
    return { a: lo, b, theta };
  }

  /**
   * Write u(·, T) and v(·, T) into outU / outV. `curU`/`curV` stand for the
   * field at `now` (index −1) and for any T ≥ now.
   */
  read(
    T: number, now: number, curU: ArrayLike<number>, curV: ArrayLike<number>,
    outU: Float64Array, outV: Float64Array,
  ): { a: number; b: number; theta: number } {
    const l = this.lookup(T, now);
    const fill = (out: Float64Array, cur: ArrayLike<number>, pick: (fr: Frame) => Float32Array) => {
      const fa = l.a < 0 ? cur : pick(this.frames[l.a]);
      if (l.theta > 0) {
        const fb = l.b < 0 ? cur : pick(this.frames[l.b]);
        const wa = 1 - l.theta, wb = l.theta;
        for (let k = 0; k < out.length; k++) out[k] = wa * fa[k] + wb * fb[k];
      } else for (let k = 0; k < out.length; k++) out[k] = fa[k];
    };
    fill(outU, curU, (fr) => fr.u);
    fill(outV, curV, (fr) => fr.v);
    return l;
  }
}
