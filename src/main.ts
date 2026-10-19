import './style.css';
import { Solver, defaultParams, clampParams, bulkViscosity, longitudinalViscosity } from './solver/solver';
import { CorrelationPanel } from './ui/correlations';
import type { KernelType } from './solver/solver';
import { viridis, diverging, hsv, type LUT } from './render/colormaps';

type View = 'density' | 'direction' | 'speed' | 'vorticity' | 'divergence' | 'dye';
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const params = defaultParams();
let solver: Solver, buf: Float64Array, img: ImageData;
let view: View = 'direction', running = true, spf = 1, arrows = true, aSpace = 12, aScale = 1;
const canvas = $<HTMLCanvasElement>('sim'), ctx = canvas.getContext('2d')!;
const over = $<HTMLCanvasElement>('overlay'), octx = over.getContext('2d')!;

const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0] >>> 0;
function seedFromInput(): number {
  const t = $<HTMLInputElement>('seed-input').value.trim();
  const s = /^\d+$/.test(t) && Number(t) <= 0xffffffff ? Number(t) >>> 0 : freshSeed();
  $('seed-used').textContent = String(s); return s;
}
const corrPanel = new CorrelationPanel();
let lastInit: 'ordered' | 'random' = 'ordered', lastSeed = 1;
function init(kind = lastInit, seed = lastSeed) {
  lastInit = kind; lastSeed = seed; corrPanel.reset();
  if (kind === 'ordered') solver.seedOrdered(seed);
  else solver.seedRandom(seed, parseFloat($<HTMLInputElement>('seed-amp').value) || 0.05);
}
function build(n: number) {
  solver = new Solver(n, params); buf = new Float64Array(n * n);
  canvas.width = n; canvas.height = n; img = ctx.createImageData(n, n); init();
}

const fmt = (x: number) => (x === 0 ? '0' : Math.abs(x) < 1e-2 || Math.abs(x) >= 1e3 ? x.toExponential(2) : x.toPrecision(3));
function bindSlider(id: string, get: () => number, set: (v: number) => void, to = (v: number) => v, from = (v: number) => v) {
  const el = $<HTMLInputElement>(id), out = $<HTMLOutputElement>(id + '-out');
  el.value = String(to(get())); out.textContent = fmt(get());
  el.addEventListener('input', () => { set(from(parseFloat(el.value))); out.textContent = fmt(get()); refresh(); });
  return el;
}
bindSlider('mu', () => params.mu, (v) => (params.mu = v), Math.log10, (v) => 10 ** v);
const lamEl = bindSlider('lambda', () => params.lambda, (v) => (params.lambda = v));
bindSlider('tau', () => params.tau, (v) => (params.tau = v));
bindSlider('v0', () => params.v0, (v) => (params.v0 = v));
bindSlider('spf', () => spf, (v) => (spf = v));
bindSlider('aspace', () => aSpace, (v) => (aSpace = v));
bindSlider('ascale', () => aScale, (v) => (aScale = v));
function refresh() {
  const clamped = clampParams(params);
  lamEl.value = String(params.lambda); $('lambda-out').textContent = fmt(params.lambda);
  $('zeta').innerHTML = `λ + 2μ = ${fmt(longitudinalViscosity(params))} (compressive) · ζ = λ + μ = ${fmt(bulkViscosity(params))}` +
    (clamped ? '<br><span class="warn">λ clamped to −2μ: λ + 2μ < 0 is ill-posed (anti-diffusion)</span>' : '');
  const di = document.getElementById('d-info');
  if (di) {
    const on = [params.delayKernel && params.kernel !== 'none' ? 'kernel' : '', params.delayLame ? 'Lamé' : ''].filter(Boolean).join(' + ') || 'nothing';
    const St = params.kernelStrength * params.tSelf;
    di.innerHTML = `delays act on: ${on}` + (params.kernel !== 'none' && params.kernel !== 'fractional' ? ` · S·t_self = ${St.toFixed(2)}${St > Math.PI / 2 ? ' <span class="warn">(> π/2)</span>' : ''}` : '') +
      (solver ? ` · history ${solver.history.frames.length} frames` : '');
  }
}
bindSlider('kS', () => params.kernelStrength, (v) => (params.kernelStrength = v), Math.log10, (v) => 10 ** v);
bindSlider('kR', () => params.kernelRange, (v) => (params.kernelRange = v));
bindSlider('kB', () => params.kernelBeta, (v) => (params.kernelBeta = v));
bindSlider('kA', () => params.kernelAlpha, (v) => (params.kernelAlpha = v));
bindSlider('kF', () => params.kernelS, (v) => (params.kernelS = v));
const lock = $<HTMLInputElement>('d-lock');
const dS = bindSlider('dS', () => params.tSelf, (v) => { params.tSelf = v; if (lock.checked) { params.tNb = v; dN.value = String(v); $('dN-out').textContent = fmt(v); } });
const dN = bindSlider('dN', () => params.tNb, (v) => { params.tNb = v; if (lock.checked) { params.tSelf = v; dS.value = String(v); $('dS-out').textContent = fmt(v); } });
lock.addEventListener('change', () => { if (lock.checked) { params.tNb = params.tSelf; dN.value = String(params.tSelf); $('dN-out').textContent = fmt(params.tSelf); } refresh(); });
$<HTMLSelectElement>('kernel').addEventListener('change', (e) => { params.kernel = (e.target as HTMLSelectElement).value as KernelType; refresh(); });
$<HTMLInputElement>('d-kernel').addEventListener('change', (e) => { params.delayKernel = (e.target as HTMLInputElement).checked; refresh(); });
$<HTMLInputElement>('d-lame').addEventListener('change', (e) => { params.delayLame = (e.target as HTMLInputElement).checked; refresh(); });
refresh();
$<HTMLInputElement>('selfprop').addEventListener('change', (e) => (params.selfProp = (e.target as HTMLInputElement).checked));
const arrowsEl = $<HTMLInputElement>('arrows');
arrowsEl.addEventListener('change', () => (arrows = arrowsEl.checked));
const viewSel = $<HTMLSelectElement>('view');
viewSel.addEventListener('change', () => (view = viewSel.value as View));
{ // ?view=density etc. selects the initial view
  const q = new URLSearchParams(location.search).get('view');
  if (q && [...viewSel.options].some((o) => o.value === q)) { viewSel.value = q; view = q as View; }
}
$<HTMLSelectElement>('grid').addEventListener('change', (e) => build(parseInt((e.target as HTMLSelectElement).value)));
$('random-init').onclick = () => init('random', seedFromInput());
$('ordered-init').onclick = () => init('ordered', seedFromInput());
const playBtn = $<HTMLButtonElement>('play');
const toggle = () => { running = !running; playBtn.textContent = running ? '⏸ Pause' : '▶ Play'; };
playBtn.onclick = toggle;
$('step').onclick = () => advance(1);
$('reset').onclick = () => init();
const panel = $('panel');
$('panel-toggle').onclick = () => panel.classList.toggle('hidden');
window.addEventListener('keydown', (e) => {
  const tg = (e.target as HTMLElement).tagName; if (tg === 'INPUT' || tg === 'SELECT') return;
  if (e.key === ' ') { toggle(); e.preventDefault(); }
  else if (e.key === 's') advance(1);
  else if (e.key === 'r') init();
  else if (e.key === 'h') panel.classList.toggle('hidden');
  else if (e.key === 'a') { arrows = !arrows; arrowsEl.checked = arrows; }
  else if (e.key === 'v') { viewSel.selectedIndex = (viewSel.selectedIndex + 1) % viewSel.options.length; view = viewSel.value as View; }
});

// drag pushes the fluid and injects dye
let drag: { x: number; y: number; dx: number; dy: number } | null = null;
const toDomain = (e: PointerEvent) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
canvas.addEventListener('pointerdown', (e) => { const [x, y] = toDomain(e); drag = { x, y, dx: 0, dy: 0 }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => { if (!drag) return; const [x, y] = toDomain(e); drag.dx += x - drag.x; drag.dy += y - drag.y; drag.x = x; drag.y = y; });
canvas.addEventListener('pointerup', () => (drag = null));
function applyForces() {
  solver.fx.fill(0); solver.fy.fill(0);
  if (!drag || (!drag.dx && !drag.dy)) return;
  const { n, h } = solver, s = 0.03, dt = params.dt * spf, fx = drag.dx / dt * 0.5 / params.dt, fy = drag.dy / dt * 0.5 / params.dt;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    let dx = (i + 0.5) * h - drag.x, dy = (j + 0.5) * h - drag.y; dx -= Math.round(dx); dy -= Math.round(dy);
    const g = Math.exp(-(dx * dx + dy * dy) / (2 * s * s)); if (g < 1e-3) continue;
    const q = j * n + i; solver.fx[q] += fx * g; solver.fy[q] += fy * g; solver.dye[q] = Math.max(solver.dye[q], g);
  }
  drag.dx = 0; drag.dy = 0;
}
function advance(steps: number) { applyForces(); solver.step(); solver.fx.fill(0); solver.fy.fill(0); for (let i = 1; i < steps; i++) solver.step(); }

function render() {
  const { n } = solver; solver.field(view, buf);
  const d = img.data;
  if (view === 'direction') {
    const sp = new Float64Array(n * n); solver.field('speed', sp);
    let mx = 1e-9; for (const s of sp) mx = Math.max(mx, s);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i, [r, g, b] = hsv(buf[k] / (2 * Math.PI) + 0.5, 0.85, 0.25 + 0.75 * Math.min(1, sp[k] / mx));
      const p = ((n - 1 - j) * n + i) * 4; d[p] = r * 255; d[p + 1] = g * 255; d[p + 2] = b * 255; d[p + 3] = 255;
    }
  } else {
    let lut: LUT = viridis, lo = 0, hi = 1;
    if (view === 'density') {
      const fixed = parseFloat($<HTMLInputElement>('rho-range').value);
      let a = 1e-6; for (const r of buf) a = Math.max(a, Math.abs(r - 1));
      if (fixed > 0) a = fixed;
      lut = diverging; lo = 1 - a; hi = 1 + a;
    } else if (view === 'vorticity' || view === 'divergence') {
      const s = Array.from(buf, Math.abs).sort((a, b) => a - b); const a = Math.max(s[Math.floor(s.length * 0.995)], 1e-6);
      lut = diverging; lo = -a; hi = a;
    } else if (view === 'speed') { let a = 1e-6; for (const v of buf) a = Math.max(a, v); hi = a; }
    const sc = 255 / (hi - lo);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const c = Math.max(0, Math.min(255, Math.round((buf[j * n + i] - lo) * sc))) * 3, p = ((n - 1 - j) * n + i) * 4;
      d[p] = lut[c]; d[p + 1] = lut[c + 1]; d[p + 2] = lut[c + 2]; d[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  drawArrows();
  let rmin = Infinity, rmax = -Infinity; for (const r of solver.rho) { if (r < rmin) rmin = r; if (r > rmax) rmax = r; }
  $('stats').textContent = `${n}² · t=${solver.t.toFixed(2)} s · dt=${params.dt.toFixed(4)} · Ekin=${solver.kineticEnergy().toExponential(3)} · ⟨|u|⟩=${solver.meanSpeed().toFixed(3)} · polar order=${solver.polarOrder().toFixed(3)} · rms ∇·u=${solver.rmsDivergence().toExponential(2)} · mass=${solver.totalMass().toFixed(12)} · ρ∈[${rmin.toFixed(3)}, ${rmax.toFixed(3)}]` + (solver.floorHits ? ` · ρ-floor hits ${solver.floorHits}` : '');
}
function drawArrows() {
  const r = over.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (over.width !== Math.round(r.width * dpr)) { over.width = Math.round(r.width * dpr); over.height = Math.round(r.height * dpr); }
  octx.clearRect(0, 0, over.width, over.height);
  if (!arrows) return;
  const { n, u, v } = solver, W = over.width, cell = W / n, step = Math.max(2, Math.round(aSpace * n / 256));
  let mx = 1e-9; for (let k = 0; k < u.length; k++) mx = Math.max(mx, Math.hypot(u[k], v[k]));
  const L = aScale * step * cell * 0.9 / mx;
  octx.strokeStyle = view === 'density' ? 'rgba(20,20,30,0.8)' : 'rgba(255,255,255,0.85)'; octx.fillStyle = octx.strokeStyle; octx.lineWidth = Math.max(1, dpr);
  for (let j = step >> 1; j < n; j += step) for (let i = step >> 1; i < n; i += step) {
    const k = j * n + i, x = (i + 0.5) * cell, y = W - (j + 0.5) * cell, ax = u[k] * L, ay = -v[k] * L, len = Math.hypot(ax, ay);
    if (len < 0.5) continue;
    octx.beginPath(); octx.moveTo(x - ax / 2, y - ay / 2); octx.lineTo(x + ax / 2, y + ay / 2); octx.stroke();
    const hx = ax / len, hy = ay / len, hs = Math.min(len * 0.35, 5 * dpr);
    octx.beginPath(); octx.moveTo(x + ax / 2, y + ay / 2);
    octx.lineTo(x + ax / 2 - hs * (hx - 0.5 * hy), y + ay / 2 - hs * (hy + 0.5 * hx));
    octx.lineTo(x + ax / 2 - hs * (hx + 0.5 * hy), y + ay / 2 - hs * (hy - 0.5 * hx)); octx.fill();
  }
}

let last = performance.now(), fpsAvg = 60;
function frame(now: number) {
  const dt = (now - last) / 1000; last = now; fpsAvg = 0.9 * fpsAvg + 0.1 / Math.max(dt, 1e-3);
  if (running) advance(spf);
  render(); corrPanel.update(solver, (s) => s.polarOrder()); if (running && (solver.t * 60 | 0) % 30 === 0) refresh();
  $('fps').textContent = `${fpsAvg.toFixed(0)} fps`;
  requestAnimationFrame(frame);
}
build(256);
requestAnimationFrame(frame);
(window as unknown as { app: unknown }).app = { get solver() { return solver; }, params, corr: corrPanel };
