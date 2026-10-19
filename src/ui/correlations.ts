import { OrderAnalyzer, type OrderStats } from '../solver/correlations';
import type { Solver as FluidSolver } from '../solver/solver';

/** Run the analysis every UPDATE_EVERY frames (only while enabled and the panel is open). */
const UPDATE_EVERY = 6;
const SPAN = 20; // s of simulated time in the time series
const PAIRS = 30000;
/** C(r) is refreshed on every CORR_EVERY-th analysis (it changes slowly; FFTs dominate the cost). */
const CORR_EVERY = 2;
const BINS = 40;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function setupCanvas(cv: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; w: number; h: number } | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth || cv.width, h = cv.clientHeight || cv.height;
  const W = Math.round(w * dpr), H = Math.round(h * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px ui-monospace, monospace';
  return { ctx, w, h };
}

/**
 * "Order parameter / correlations" panel: pairwise-alignment histogram
 * P(cos θ_ij), radial velocity correlation C(r) (full and connected) and a
 * time series of ⟨cos θ_ij⟩, φ and ξ. See src/solver/correlations.ts.
 */
export class CorrelationPanel {
  private readonly an = new OrderAnalyzer();
  private readonly details = $<HTMLDetailsElement>('corr-panel');
  private readonly enabled = $<HTMLInputElement>('corr-on');
  private readonly info = $<HTMLParagraphElement>('corr-info');
  private frame = 0;
  private solver: FluidSolver | null = null;
  last: OrderStats | null = null;
  /** Cost of the last full analysis (with C(r)) and of the last histogram-only one, ms. */
  lastMs = 0;
  lastHistMs = 0;
  private count = 0;
  /** Time series (exposed for automated checks): t, ⟨cos⟩, φ, ξ (domain units). */
  readonly series = { t: [] as number[], cos: [] as number[], phi: [] as number[], xi: [] as number[] };

  constructor() {
    this.enabled.checked = true;
    this.enabled.addEventListener('change', () => { if (!this.enabled.checked) this.info.textContent = 'off'; });
  }

  get active(): boolean { return this.enabled.checked && this.details.open; }

  reset(): void {
    const s = this.series; s.t = []; s.cos = []; s.phi = []; s.xi = [];
    this.frame = UPDATE_EVERY - 1; // analyse on the next frame
    this.count = 0;
  }

  /** Call once per frame after solver.updateDiagnostics(); phi = speed-weighted polar order. */
  update(s: FluidSolver, phiOf: (s: FluidSolver) => number): void {
    if (s !== this.solver) { this.solver = s; this.reset(); }
    const ser = this.series;
    if (ser.t.length && s.t < ser.t[ser.t.length - 1]) this.reset();
    if (!this.active || ++this.frame < UPDATE_EVERY) return;
    if (ser.t.length && s.t === ser.t[ser.t.length - 1] && this.last) return; // paused
    this.frame = 0;
    const phi = phiOf(s);
    const g = { nx: s.n, ny: s.n, h: s.h };
    const t0 = performance.now();
    // C(r) (three 2D FFTs) on every CORR_EVERY-th analysis; histogram + exact mean every time
    const doCorr = this.count++ % CORR_EVERY === 0 || !this.last || this.last.r.length !== Math.floor(Math.min(g.nx, g.ny) / 2) + 1;
    const n = s.n, u = s.u, v = s.v, cell: [number, number] = [0, 0];
    const st = this.an.analyze(n, n, (i, j) => { cell[0] = u[j * n + i]; cell[1] = v[j * n + i]; return cell; }, {
      bins: BINS, pairs: PAIRS, periodicX: true, periodicY: true, correlation: doCorr,
    });
    if (!doCorr && this.last) { st.C = this.last.C; st.Cconn = this.last.Cconn; st.xi0 = this.last.xi0; st.xiE = this.last.xiE; }
    const ms = performance.now() - t0;
    if (doCorr) this.lastMs = ms; else this.lastHistMs = ms;
    this.last = st;
    ser.t.push(s.t); ser.cos.push(st.meanCos); ser.phi.push(phi); ser.xi.push(st.xi0 * g.h);
    while (ser.t.length > 2 && ser.t[0] < s.t - SPAN) { ser.t.shift(); ser.cos.shift(); ser.phi.shift(); ser.xi.shift(); }
    this.draw(g.h, Math.floor(Math.min(g.nx, g.ny) / 2) * g.h);
  }

  private draw(h: number, rMaxDomain: number): void {
    const st = this.last;
    if (!st) return;
    const f = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
    this.info.textContent =
      `⟨cos θ_ij⟩ = ${f(st.meanCos, 3)} (exact; φ_dir² = ${f(st.phiDir ** 2, 3)}, φ = ${f(this.series.phi.at(-1) ?? NaN, 3)}) · ` +
      `ξ = ${Number.isFinite(st.xi0) ? f(st.xi0 * h, 3) : `> ${f(rMaxDomain, 2)}`} (C_conn = 0), ` +
      `ξ₁/ₑ = ${f(st.xiE * h, 3)} · ${st.n} cells (${st.excluded} slow excluded) · ${this.lastMs.toFixed(1)} ms (C(r)) / ${this.lastHistMs.toFixed(1)} ms`;
    this.drawHist(st);
    this.drawCr(st, h);
    this.drawSeries(rMaxDomain);
  }

  private drawHist(st: OrderStats): void {
    const c = setupCanvas($<HTMLCanvasElement>('corr-hist'));
    if (!c) return;
    const { ctx, w, h } = c, top = 12, bot = h - 12, B = st.hist.length;
    const pMax = Math.max(1, ...st.hist);
    ctx.fillStyle = '#3fa9ff';
    for (let k = 0; k < B; k++) {
      const x0 = 2 + (k / B) * (w - 4), x1 = 2 + ((k + 1) / B) * (w - 4);
      const y = bot - (st.hist[k] / pMax) * (bot - top);
      ctx.fillRect(x0, y, Math.max(1, x1 - x0 - 0.5), bot - y);
    }
    const X = (c: number) => 2 + ((c + 1) / 2) * (w - 4);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.beginPath(); ctx.moveTo(X(0), top); ctx.lineTo(X(0), bot); ctx.stroke();
    if (Number.isFinite(st.meanCos)) {
      ctx.strokeStyle = '#ffa657'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(X(st.meanCos), top); ctx.lineTo(X(st.meanCos), bot); ctx.stroke(); ctx.lineWidth = 1;
    }
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('−1', 2, h - 2); ctx.fillText('0', X(0) - 3, h - 2); ctx.fillText('cos θ_ij  1', w - 62, h - 2);
    ctx.fillStyle = '#3fa9ff'; ctx.fillText(`P(cos θ_ij), max ${pMax.toFixed(1)}`, 4, 10);
    ctx.fillStyle = '#ffa657'; ctx.fillText(`mean ${Number.isFinite(st.meanCos) ? st.meanCos.toFixed(2) : '—'}`, w - 62, 10);
  }

  private drawCr(st: OrderStats, hCell: number): void {
    const c = setupCanvas($<HTMLCanvasElement>('corr-r'));
    if (!c) return;
    const { ctx, w, h } = c, top = 12, bot = h - 12, n = st.r.length;
    const X = (r: number) => 2 + (r / Math.max(1, n - 1)) * (w - 4);
    const Y = (y: number) => top + ((1 - y) / 2) * (bot - top); // y ∈ [−1, 1]
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.beginPath(); ctx.moveTo(2, Y(0)); ctx.lineTo(w - 2, Y(0)); ctx.stroke();
    const line = (ys: Float64Array, col: string) => {
      ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.beginPath(); let started = false;
      ys.forEach((y, k) => { if (!Number.isFinite(y)) return; const px = X(k), py = Y(Math.max(-1, Math.min(1, y))); if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; } });
      ctx.stroke(); ctx.lineWidth = 1;
    };
    line(st.C, '#58a6ff');
    line(st.Cconn, '#7ee787');
    if (Number.isFinite(st.xi0)) {
      ctx.strokeStyle = '#ffa657'; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(st.xi0), top); ctx.lineTo(X(st.xi0), bot); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.fillStyle = '#58a6ff'; ctx.fillText('C(r)', 4, 10);
    ctx.fillStyle = '#7ee787'; ctx.fillText('connected', 36, 10);
    ctx.fillStyle = '#ffa657'; ctx.fillText(`ξ ${Number.isFinite(st.xi0) ? (st.xi0 * hCell).toFixed(3) : '—'}`, 104, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('0', 2, h - 2); ctx.fillText(`r = ${((n - 1) * hCell).toFixed(2)}`, w - 60, h - 2);
  }

  private drawSeries(rMaxDomain: number): void {
    const c = setupCanvas($<HTMLCanvasElement>('corr-t'));
    const s = this.series;
    if (!c || !s.t.length) return;
    const { ctx, w, h } = c, top = 12, bot = h - 4;
    const t1 = s.t[s.t.length - 1], t0 = t1 - SPAN;
    const X = (t: number) => 2 + ((t - t0) / SPAN) * (w - 4);
    const Y = (y: number) => bot - Math.max(-0.05, Math.min(1.05, y)) * (bot - top);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath(); ctx.moveTo(2, Y(0)); ctx.lineTo(w - 2, Y(0)); ctx.moveTo(2, Y(1)); ctx.lineTo(w - 2, Y(1)); ctx.stroke();
    const line = (ys: number[], f: (y: number) => number, col: string) => {
      ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath(); let started = false;
      ys.forEach((y, k) => { const v = f(y); if (!Number.isFinite(v)) { started = false; return; } const px = X(s.t[k]), py = Y(v); if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; } });
      ctx.stroke(); ctx.lineWidth = 1;
    };
    line(s.phi, (y) => y, 'rgba(88,166,255,0.7)');
    line(s.xi, (y) => y / rMaxDomain, '#7ee787');
    line(s.cos, (y) => y, '#ffa657');
    ctx.fillStyle = '#ffa657'; ctx.fillText('⟨cos θ⟩', 4, 10);
    ctx.fillStyle = '#58a6ff'; ctx.fillText('φ', 54, 10);
    ctx.fillStyle = '#7ee787'; ctx.fillText(`ξ / ${rMaxDomain.toFixed(2)}`, 68, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.45)'; ctx.fillText(`${SPAN} s`, w - 28, h - 6);
  }
}
