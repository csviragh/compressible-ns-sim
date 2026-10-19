import './style.css';
import { Solver, defaultParams, clampLambda, bulkViscosity, type EOS } from './solver/solver';
import { applyPreset, addPulse, stirForce, type Preset } from './solver/presets';
import { viridis, diverging, type LUT } from './render/colormaps';

type View = 'density' | 'vorticity' | 'divergence' | 'speed' | 'mach' | 'dye';
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const params = defaultParams();
let solver: Solver;
let preset: Preset = 'acoustic';
let view: View = 'density';
let running = true;
let seed = 1;
let buf: Float64Array, img: ImageData;
const canvas = $<HTMLCanvasElement>('sim');
const ctx = canvas.getContext('2d')!;

function build(n: number) {
  solver = new Solver(n, params);
  buf = new Float64Array(n * n);
  canvas.width = n; canvas.height = n;
  img = ctx.createImageData(n, n);
  reset();
}
function reset() { applyPreset(solver, preset, seed++); }

// ---------- controls ----------
const fmt = (x: number) => (x === 0 ? '0' : Math.abs(x) < 1e-2 || Math.abs(x) >= 1e3 ? x.toExponential(2) : x.toPrecision(3));
function bindSlider(id: string, get: () => number, set: (v: number) => void, toSlider = (v: number) => v, fromSlider = (v: number) => v) {
  const el = $<HTMLInputElement>(id), out = $<HTMLOutputElement>(id + '-out');
  el.value = String(toSlider(get()));
  const upd = () => { set(fromSlider(parseFloat(el.value))); out.textContent = fmt(get()); refreshVisc(); };
  el.addEventListener('input', upd); out.textContent = fmt(get());
  return el;
}
const muEl = bindSlider('mu', () => params.mu, (v) => { params.mu = v; if (stokes.checked) params.lambda = -v; }, Math.log10, (v) => 10 ** v);
const lamEl = bindSlider('lambda', () => params.lambda, (v) => { params.lambda = v; stokes.checked = false; });
const stokes = $<HTMLInputElement>('stokes');
stokes.addEventListener('change', () => { if (stokes.checked) params.lambda = -params.mu; refreshVisc(); });
bindSlider('c', () => params.c, (v) => { params.c = v; });
bindSlider('gamma', () => params.gamma, (v) => { params.gamma = v; });
let spf = 2;
bindSlider('spf', () => spf, (v) => { spf = v; });
void muEl;

function refreshVisc() {
  const clamped = clampLambda(params);
  lamEl.value = String(params.lambda);
  $('lambda-out').textContent = fmt(params.lambda);
  const z = bulkViscosity(params);
  const el = $('zeta');
  el.innerHTML = `ζ = λ + μ = ${fmt(z)}  ·  2μ + λ = ${fmt(2 * params.mu + params.lambda)}` +
    (clamped ? '<br><span class="warn">λ clamped to −μ: ζ < 0 is unphysical</span>' : '');
  $('gamma-row').hidden = params.eos !== 'polytropic';
}
$<HTMLSelectElement>('eos').addEventListener('change', (e) => { params.eos = (e.target as HTMLSelectElement).value as EOS; refreshVisc(); });
$<HTMLSelectElement>('preset').addEventListener('change', (e) => { preset = (e.target as HTMLSelectElement).value as Preset; reset(); });
const viewSel = $<HTMLSelectElement>('view');
viewSel.addEventListener('change', () => { view = viewSel.value as View; });
$<HTMLSelectElement>('grid').addEventListener('change', (e) => build(parseInt((e.target as HTMLSelectElement).value)));
const playBtn = $<HTMLButtonElement>('play');
const toggle = () => { running = !running; playBtn.textContent = running ? '⏸ Pause' : '▶ Play'; };
playBtn.onclick = toggle;
$('step').onclick = () => advance(1);
$('reset').onclick = reset;
const panel = $('panel');
$('panel-toggle').onclick = () => panel.classList.toggle('hidden');
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'SELECT') return;
  if (e.key === ' ') { toggle(); e.preventDefault(); }
  else if (e.key === 's') advance(1);
  else if (e.key === 'r') reset();
  else if (e.key === 'h') panel.classList.toggle('hidden');
  else if (e.key === 'v') { viewSel.selectedIndex = (viewSel.selectedIndex + 1) % viewSel.options.length; view = viewSel.value as View; }
});
refreshVisc();

// ---------- pointer: drag pushes, click pulses ----------
let drag: { x: number; y: number; x0: number; y0: number; moved: boolean; dx: number; dy: number } | null = null;
const toDomain = (e: PointerEvent) => { const r = canvas.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height]; };
canvas.addEventListener('pointerdown', (e) => { const [x, y] = toDomain(e); drag = { x, y, x0: x, y0: y, moved: false, dx: 0, dy: 0 }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return; const [x, y] = toDomain(e);
  drag.dx += x - drag.x; drag.dy += y - drag.y; drag.x = x; drag.y = y;
  if (Math.hypot(x - drag.x0, y - drag.y0) > 0.01) drag.moved = true;
});
canvas.addEventListener('pointerup', () => { if (drag && !drag.moved) addPulse(solver, drag.x, drag.y, 0.3, 0.02); drag = null; });

function applyForces(frameDt: number) {
  solver.fx.fill(0); solver.fy.fill(0);
  if (preset === 'stir') stirForce(solver, solver.t);
  if (drag && drag.moved && (drag.dx || drag.dy)) {
    const { n, h } = solver, s = 0.04, k = 1 / Math.max(frameDt, 1e-3);
    const fx = drag.dx * k * 30, fy = drag.dy * k * 30;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      let dx = (i + 0.5) * h - drag.x, dy = (j + 0.5) * h - drag.y; dx -= Math.round(dx); dy -= Math.round(dy);
      const g = Math.exp(-(dx * dx + dy * dy) / (2 * s * s)); if (g < 1e-3) continue;
      const q = j * n + i; solver.fx[q] += fx * g * solver.rho[q]; solver.fy[q] += fy * g * solver.rho[q];
    }
    drag.dx = 0; drag.dy = 0;
  }
}
function advance(steps: number, frameDt = 1 / 60) { applyForces(frameDt); for (let i = 0; i < steps; i++) solver.step(); }

// ---------- render ----------
function render() {
  const { n } = solver; solver.field(view, buf);
  let lut: LUT = viridis, lo = 0, hi = 1;
  if (view === 'density') {
    let m = 0; for (const r of buf) m += r; m /= buf.length;
    let a = 1e-6; for (const r of buf) a = Math.max(a, Math.abs(r - m));
    lut = diverging; lo = m - a; hi = m + a;
  } else if (view === 'vorticity' || view === 'divergence') {
    const s = [...buf].map(Math.abs).sort((a, b) => a - b); const a = Math.max(s[Math.floor(s.length * 0.995)], 1e-6);
    lut = diverging; lo = -a; hi = a;
  } else if (view === 'dye') { lo = 0; hi = 1; }
  else { let a = 1e-6; for (const v of buf) a = Math.max(a, v); hi = a; }
  const d = img.data, sc = 255 / (hi - lo);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const v = buf[j * n + i], c = Math.max(0, Math.min(255, Math.round((v - lo) * sc))) * 3;
    const p = ((n - 1 - j) * n + i) * 4; d[p] = lut[c]; d[p + 1] = lut[c + 1]; d[p + 2] = lut[c + 2]; d[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  $('stats').textContent = `${n}² · t=${solver.t.toFixed(3)} · dt=${solver.lastDt.toExponential(2)} · M=${solver.totalMass().toFixed(12)} · Ekin=${solver.kineticEnergy().toExponential(3)} · Ma_max=${solver.maxMach().toFixed(2)}` +
    (solver.floorHits ? ` · ρ-floor hits ${solver.floorHits}` : '');
}

let last = performance.now(), fpsAvg = 60;
function frame(now: number) {
  const dt = (now - last) / 1000; last = now; fpsAvg = 0.9 * fpsAvg + 0.1 / Math.max(dt, 1e-3);
  if (running) advance(spf, dt);
  render();
  $('fps').textContent = `${fpsAvg.toFixed(0)} fps`;
  requestAnimationFrame(frame);
}
build(256);
requestAnimationFrame(frame);
