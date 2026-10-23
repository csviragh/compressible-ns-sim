/**
 * WebGPU compute backend for Solver.step(). The CPU Solver object stays the source of truth for
 * parameters, time, initial conditions and diagnostics: `upload()` copies its fields to the GPU,
 * `advance()` runs steps entirely on the GPU and reads u, v, ρ, dye back into the CPU arrays
 * (for rendering, HUD, arrows and correlations). Physics is the same splitting as the CPU code,
 * in f32 instead of f64.
 *
 * Retarded repulsion: the bookkeeping of the ρ ring buffer (which frames exist, their times,
 * interval, pruning, interpolation weights) is done by the same `FieldHistory` class as on the CPU
 * (with 1-element dummy frames); every frame maps to a slot of a GPU ring buffer that receives a
 * copy of ρ when the frame is pushed. Per step, the stencil (offset, weights, slot a/b, θ) is
 * built on the CPU and uploaded.
 */
import { Solver } from '../solver/solver';
import { FieldHistory } from '../solver/delay';
import code from './shaders.wgsl?raw';
import coulombCode from './coulomb.wgsl?raw';
import { coulombKernel } from '../solver/coulomb';

const CUR = 0xffffffff;
const WG = 8;
const ENTRY_BYTES = 32;
const MAX_STEPS = 16;   // steps per advance() call
const MAX_LAME = 64;    // Lamé sub-steps per step (uniform slots)
const NS_MAX_CAP = 64;  // density sub-steps per step
const C_CHUNK = 4096;   // Coulomb offsets per dispatch (keeps single dispatches short, avoids GPU watchdog resets)
const C_MAX_CHUNKS = 64; // → grids up to 512²

type Frame = FieldHistory['frames'][number];

export class GpuSolver {
  private device!: GPUDevice;
  private pipes: Record<string, GPUComputePipeline> = {};
  private layout!: GPUBindGroupLayout;
  private bufs!: Record<'P' | 'S' | 'U' | 'U2' | 'R' | 'D' | 'F' | 'ring' | 'st' | 'cnt' | 'read', GPUBuffer>;
  private bind!: GPUBindGroup;
  private n = 0;
  private stBlock = 256;
  private ringSlots = 1;
  private hist = new FieldHistory();
  private slotOf = new Map<Frame, number>();
  private free: number[] = [];
  /** Max density sub-steps requested by the GPU beyond the cap (0 = never clamped). */
  clampedNs = 0;
  floorHits = 0;
  lastGpuMs = 0;
  // --- Coulomb direct sum ---
  private cPipe!: GPUComputePipeline;
  private cLayout!: GPUBindGroupLayout;
  private cBufs: { CP: GPUBuffer; K: GPUBuffer; bins: GPUBuffer } | null = null;
  private cBind: GPUBindGroup | null = null;
  private cKey = '';
  private cBinBlock = 256;
  private cBindKey: unknown[] = [];
  adapterInfo = '';

  static async create(): Promise<GpuSolver | null> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    const g = new GpuSolver();
    g.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;
    g.adapterInfo = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : '';
    g.initPipelines();
    return g;
  }

  private initPipelines() {
    const d = this.device, C = GPUShaderStage.COMPUTE;
    const st = (i: number, ro = false, dyn = false): GPUBindGroupLayoutEntry => ({ binding: i, visibility: C, buffer: { type: ro ? 'read-only-storage' : 'storage', hasDynamicOffset: dyn } });
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true } },
        st(2), st(3), st(4), st(5), st(6), st(7, true), st(8, true, true), st(9),
      ],
    });
    const module = d.createShaderModule({ code });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.cLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true } },
        st(1), st(2, true), st(3, true), st(4, true), st(5, true, true),
      ],
    });
    this.cPipe = d.createComputePipeline({ layout: d.createPipelineLayout({ bindGroupLayouts: [this.cLayout] }), compute: { module: d.createShaderModule({ code: coulombCode }), entryPoint: 'coulomb' } });
    for (const e of ['repulse', 'lame', 'umax', 'dens1', 'dens2', 'dens3', 'advect'])
      this.pipes[e] = d.createComputePipeline({ layout: pl, compute: { module, entryPoint: e } });
  }

  private alloc(n: number, M: number, slots: number) {
    const d = this.device, N = n * n, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const old = this.bufs;
    const stBlock = Math.max(256, Math.ceil(Math.max(1, M) * ENTRY_BYTES / 256) * 256);
    const sizeChanged = !old || n !== this.n;
    const mk = (size: number, usage = S) => d.createBuffer({ size, usage });
    if (sizeChanged) {
      if (old) for (const b of Object.values(old)) b.destroy();
      this.bufs = {
        P: mk(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
        S: mk(256 * MAX_LAME, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
        U: mk(8 * N), U2: mk(12 * N), R: mk(12 * N), D: mk(8 * N), F: mk(8 * N),
        ring: mk(4 * N * slots), st: mk(stBlock * MAX_STEPS), cnt: mk(16),
        read: mk(4 * (5 * N + 4), GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
      };
      const sub = new Uint32Array(64 * MAX_LAME); for (let s = 0; s < MAX_LAME; s++) sub[s * 64] = s;
      d.queue.writeBuffer(this.bufs.S, 0, sub);
      this.n = n; this.stBlock = stBlock; this.ringSlots = slots; this.resetHistory();
    } else {
      if (stBlock > this.stBlock) { this.bufs.st.destroy(); this.bufs.st = mk(stBlock * MAX_STEPS); this.stBlock = stBlock; }
      if (slots !== this.ringSlots) { this.bufs.ring.destroy(); this.bufs.ring = mk(4 * N * slots); this.ringSlots = slots; this.resetHistory(); }
      else if (stBlock <= this.stBlock && this.bind) return;
    }
    const b = this.bufs, e = (i: number, buf: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding: i, resource: { buffer: buf, size } });
    this.bind = d.createBindGroup({
      layout: this.layout,
      entries: [e(0, b.P), e(1, b.S, 16), e(2, b.U), e(3, b.U2), e(4, b.R), e(5, b.D), e(6, b.F), e(7, b.ring), e(8, b.st, this.stBlock), e(9, b.cnt)],
    });
  }

  /** Upload the Coulomb kernel W(o)/bins for (n, v_int, Δt) and size the per-step bin table. Returns the kernel. */
  private prepCoulomb(n: number, vInt: number, dt: number, dtA: number) {
    const d = this.device, K = coulombKernel(n, vInt, dt), N = n * n, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const binBlock = Math.max(256, Math.ceil(K.nBins * 16 / 256) * 256), key = `${n}|${K.nBins}|${vInt}|${dt}`;
    if (!this.cBufs || this.cBinBlock < binBlock || this.cBufs.K.size !== 16 * N) {
      this.cBufs?.CP.destroy(); this.cBufs?.K.destroy(); this.cBufs?.bins.destroy();
      this.cBinBlock = Math.max(binBlock, this.cBinBlock);
      this.cBufs = {
        CP: d.createBuffer({ size: 256 * C_MAX_CHUNKS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        K: d.createBuffer({ size: 16 * N, usage: S }),
        bins: d.createBuffer({ size: this.cBinBlock * MAX_STEPS, usage: S }),
      };
      this.cKey = ''; this.cBind = null;
    }
    if (this.cKey !== key) {
      const kb = new ArrayBuffer(16 * N), kf = new Float32Array(kb), ku = new Uint32Array(kb);
      for (let o = 0; o < N; o++) { kf[4 * o] = K.wx[o]; kf[4 * o + 1] = K.wy[o]; ku[4 * o + 2] = K.bin[o]; }
      d.queue.writeBuffer(this.cBufs.K, 0, kb); this.cKey = key;
    }
    const chunks = Math.ceil(N / C_CHUNK);
    if (chunks > C_MAX_CHUNKS) throw new Error(`Coulomb: grid too large (${n}²)`);
    const cp = new ArrayBuffer(256 * chunks), cu = new Uint32Array(cp), cf = new Float32Array(cp);
    for (let c = 0; c < chunks; c++) { const o = 64 * c; cu[o] = n; cu[o + 1] = N; cu[o + 2] = c * C_CHUNK; cu[o + 3] = Math.min(N, (c + 1) * C_CHUNK); cf[o + 4] = dtA; }
    d.queue.writeBuffer(this.cBufs.CP, 0, cp);
    const b = this.bufs, bk = [b.U, b.R, b.ring, this.cBufs.K, this.cBufs.bins];
    if (!this.cBind || bk.some((x, i) => x !== this.cBindKey[i])) {
      const e = (i: number, buf: GPUBuffer, size?: number): GPUBindGroupEntry => ({ binding: i, resource: { buffer: buf, size } });
      this.cBind = d.createBindGroup({ layout: this.cLayout, entries: [e(0, this.cBufs.CP, 32), e(1, b.U), e(2, b.R, 4 * N), e(3, b.ring), e(4, this.cBufs.K), e(5, this.cBufs.bins, this.cBinBlock)] });
      this.cBindKey = bk;
    }
    return { K, chunks };
  }

  private resetHistory() { this.hist.clear(); this.slotOf.clear(); this.free = Array.from({ length: this.ringSlots }, (_, i) => i); }

  /** Copy the CPU solver state (u, v, ρ, dye) to the GPU and restart the ρ history. */
  upload(s: Solver) {
    this.alloc(s.n, s.repulsionStencil().length, this.ringSlots);
    const q = this.device.queue, N = s.n * s.n, f = (a: Float64Array) => Float32Array.from(a);
    q.writeBuffer(this.bufs.U, 0, f(s.u)); q.writeBuffer(this.bufs.U, 4 * N, f(s.v));
    q.writeBuffer(this.bufs.R, 0, f(s.rho)); q.writeBuffer(this.bufs.D, 0, f(s.dye));
    this.floorHits = s.floorHits;
    this.resetHistory();
  }

  /**
   * Run `steps` solver steps on the GPU (forces s.fx/fy act in the first step only, like main.ts),
   * then read the state back into s. If `dyeDirty`, s.dye is uploaded first (mouse dye injection).
   */
  advance(s: Solver, steps: number, dyeDirty = false): Promise<void> {
    // serialise: one submission + readback at a time
    const run = this.chain.then(() => this.advanceNow(s, steps, dyeDirty));
    this.chain = run.catch(() => undefined);
    return run;
  }
  private chain: Promise<unknown> = Promise.resolve();

  private async advanceNow(s: Solver, steps: number, dyeDirty: boolean): Promise<void> {
    steps = Math.min(steps, MAX_STEPS);
    const p = s.params, dt = p.dt, N = s.n * s.n, n = s.n;
    const stencil = p.repOn && p.repA !== 0 ? s.repulsionStencil() : [];
    const tauMax = s.delayMax();
    // ring size: what FieldHistory can keep alive (capacity), limited by the device
    const maxSlots = Math.max(1, Math.floor(Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize) / (4 * N)));
    let want = 1;
    if (tauMax > 0) {
      this.hist.configure(1, 0, dt, tauMax + dt, 4 * Math.min(maxSlots, FieldHistory.MAX_FRAMES));
      want = this.hist.capacity;
    }
    this.alloc(n, stencil.length, want);
    const coul = p.repOn && p.repA !== 0 && p.repKind === 'coulomb' ? this.prepCoulomb(n, p.vInt, dt, dt * p.repA) : null;
    const binData = coul ? new ArrayBuffer(this.cBinBlock * steps) : null;
    const q = this.device.queue, b = this.bufs;
    if (dyeDirty) q.writeBuffer(b.D, 0, Float32Array.from(s.dye));
    q.writeBuffer(b.F, 0, Float32Array.from(s.fx)); q.writeBuffer(b.F, 4 * N, Float32Array.from(s.fy));

    // Params (constant for this call)
    const lameNs = !p.lameOn || (p.mu === 0 && p.lambda === 0) ? 0 : Math.max(1, Math.ceil(dt * s.spectralBound() / 1.8));
    if (lameNs > MAX_LAME) throw new Error(`Lamé needs ${lameNs} sub-steps > ${MAX_LAME}`);
    let um = 1e-12; for (let k = 0; k < N; k++) um = Math.max(um, Math.abs(s.u[k]), Math.abs(s.v[k]));
    const nsEst = Math.max(1, Math.ceil(um * dt / s.h / 0.25)), nsMax = Math.min(NS_MAX_CAP, 2 * nsEst + 2);
    const decay = Math.exp(-dt / p.tau), sp = p.selfProp && p.tau > 0;
    const P = new ArrayBuffer(64), pu = new Uint32Array(P), pf = new Float32Array(P);
    pu[0] = n; pu[1] = N; pu[2] = stencil.length; pu[3] = nsMax;
    pf[4] = s.h; pf[5] = dt; pf[6] = lameNs ? dt / lameNs : 0; pf[7] = p.mu;
    pf[8] = p.lambda + p.mu; pf[9] = decay; pf[10] = Math.max(0, p.v0) * (1 - decay); pu[11] = sp ? 1 : 0;
    pf[12] = p.dyeFade > 0 ? Math.exp(-p.dyeFade * dt) : 1;
    q.writeBuffer(b.P, 0, P);

    const enc = this.device.createCommandEncoder(), wg = Math.ceil(n / WG);
    const stData = new ArrayBuffer(this.stBlock * steps), si = new Int32Array(stData), sf = new Float32Array(stData), su = new Uint32Array(stData);
    const pass = (name: string, sub: number, step: number) => {
      const c = enc.beginComputePass(); c.setPipeline(this.pipes[name]);
      c.setBindGroup(0, this.bind, [256 * sub, this.stBlock * step]); c.dispatchWorkgroups(wg, wg); c.end();
    };
    enc.clearBuffer(b.cnt);
    let t = s.t;
    for (let m = 0; m < steps; m++) {
      // --- ρ history (same bookkeeping as Solver.step) ---
      if (tauMax > 0) {
        const h = this.hist;
        if (h.wants(t)) h.push(t, [0], []);
        h.prune(t + dt);
        const live = new Set(h.frames);
        for (const [f, sl] of this.slotOf) if (!live.has(f)) { this.slotOf.delete(f); this.free.push(sl); }
        for (const f of h.frames) if (!this.slotOf.has(f)) {
          const sl = this.free.pop()!; this.slotOf.set(f, sl);
          enc.copyBufferToBuffer(b.R, 0, b.ring, 4 * N * sl, 4 * N); // the new frame is ρ(t)
        }
      } else if (this.hist.frames.length) this.resetHistory();
      // --- stencil block for this step ---
      const base = (this.stBlock * m) / 4;
      stencil.forEach((e, idx) => {
        const o = base + idx * 8;
        let a = CUR, bb = CUR, theta = 0;
        if (e.delay > 0) {
          const l = this.hist.lookup(t - e.delay, t);
          a = l.a < 0 ? CUR : this.slotOf.get(this.hist.frames[l.a])!;
          bb = l.b < 0 ? CUR : this.slotOf.get(this.hist.frames[l.b])!;
          theta = l.theta;
        }
        si[o] = e.di; si[o + 1] = e.dj; sf[o + 2] = e.wx; sf[o + 3] = e.wy; su[o + 4] = a; su[o + 5] = bb; sf[o + 6] = theta;
      });
      // --- kernels ---
      pass('repulse', 0, m);
      if (coul && binData) {
        const bu = new Uint32Array(binData), bf = new Float32Array(binData), base2 = (this.cBinBlock * m) / 4;
        for (let qb = 0; qb < coul.K.nBins; qb++) {
          let a = CUR, bb = CUR, theta = 0;
          if (qb > 0) {
            const l = this.hist.lookup(t - qb * dt, t);
            a = l.a < 0 ? CUR : this.slotOf.get(this.hist.frames[l.a])!;
            bb = l.b < 0 ? CUR : this.slotOf.get(this.hist.frames[l.b])!;
            theta = l.theta;
          }
          bu[base2 + 4 * qb] = a; bu[base2 + 4 * qb + 1] = bb; bf[base2 + 4 * qb + 2] = theta;
        }
        for (let c = 0; c < coul.chunks; c++) {
          const cpass = enc.beginComputePass(); cpass.setPipeline(this.cPipe);
          cpass.setBindGroup(0, this.cBind!, [256 * c, this.cBinBlock * m]); cpass.dispatchWorkgroups(wg, wg); cpass.end();
        }
      }
      if (m === 0) enc.clearBuffer(b.F);
      for (let l = 0; l < lameNs; l++) { pass('lame', 0, m); enc.copyBufferToBuffer(b.U2, 0, b.U, 0, 8 * N); }
      enc.clearBuffer(b.cnt, 0, 4); enc.clearBuffer(b.cnt, 8, 4);
      pass('umax', 0, m);
      for (let d = 0; d < nsMax; d++) { pass('dens1', d, m); pass('dens2', d, m); pass('dens3', d, m); }
      pass('advect', 0, m);
      enc.copyBufferToBuffer(b.U2, 0, b.U, 0, 8 * N);
      enc.copyBufferToBuffer(b.U2, 8 * N, b.D, 0, 4 * N);
      t += dt;
    }
    q.writeBuffer(b.st, 0, stData);
    if (binData && this.cBufs) q.writeBuffer(this.cBufs.bins, 0, binData);
    enc.copyBufferToBuffer(b.U, 0, b.read, 0, 8 * N);
    enc.copyBufferToBuffer(b.R, 0, b.read, 8 * N, 4 * N);
    enc.copyBufferToBuffer(b.D, 0, b.read, 12 * N, 4 * N);
    enc.copyBufferToBuffer(b.cnt, 0, b.read, 16 * N, 16);
    const t0 = performance.now();
    q.submit([enc.finish()]);
    await b.read.mapAsync(GPUMapMode.READ);
    this.lastGpuMs = performance.now() - t0;
    const r = new Float32Array(b.read.getMappedRange()), rc = new Uint32Array(r.buffer, 16 * N, 4);
    for (let k = 0; k < N; k++) { s.u[k] = r[k]; s.v[k] = r[N + k]; s.rho[k] = r[2 * N + k]; s.dye[k] = r[3 * N + k]; }
    this.floorHits += rc[1]; s.floorHits = this.floorHits;
    if (rc[2]) this.clampedNs = Math.max(this.clampedNs, rc[2]);
    b.read.unmap();
    s.t = t;
    s.lastSubsteps = lameNs;
    s.fx.fill(0); s.fy.fill(0);
  }
}

/**
 * Dev check: run `steps` steps from a copy of the current state on the CPU (f64) and on the GPU (f32),
 * return max |Δ| of u, v, ρ, dye relative to the field's max magnitude. Leaves `s` advanced by the GPU.
 */
export async function compareBackends(gpu: GpuSolver, s: Solver, steps = 10) {
  const c = new Solver(s.n, { ...s.params });
  c.u.set(s.u); c.v.set(s.v); c.rho.set(s.rho); c.dye.set(s.dye); c.t = s.t;
  s.fx.fill(0); s.fy.fill(0);
  gpu.upload(s);
  const t0 = performance.now(); for (let m = 0; m < steps; m++) c.step(); const cpuMs = (performance.now() - t0) / steps;
  const t1 = performance.now();
  for (let m = 0; m < steps; m += 8) await gpu.advance(s, Math.min(8, steps - m));
  const gpuMs = (performance.now() - t1) / steps;
  const rel = (a: Float64Array, b: Float64Array) => {
    let e = 0, M = 1e-30; for (let k = 0; k < a.length; k++) { e = Math.max(e, Math.abs(a[k] - b[k])); M = Math.max(M, Math.abs(b[k])); }
    return e / M;
  };
  return { steps, u: rel(s.u, c.u), v: rel(s.v, c.v), rho: rel(s.rho, c.rho), dye: rel(s.dye, c.dye), cpuMsPerStep: cpuMs, gpuMsPerStep: gpuMs, tCpu: c.t, tGpu: s.t };
}
