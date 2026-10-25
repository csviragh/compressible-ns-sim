import '../style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SphereSolver, defaultSphereParams } from './sphereSolver';
import { viridis, hsv, type LUT } from '../render/colormaps';

type View = 'density' | 'speed' | 'direction';
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = defaultSphereParams();
let solver: SphereSolver, view: View = 'density', running = true, spf = 1, arrows = true, aScale = 0.08, spin = false, msPerStep = 0;
let lastInit: 'random' | 'zonal' = 'random', lastSeed = 1;

// ---------- three.js scene ----------
const host = $('globe');
THREE.ColorManagement.enabled = false; // colormap values are display (sRGB) values already
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(2, devicePixelRatio)); host.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x05060a);
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100); camera.position.set(2.6, -2.2, 1.6); camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true; controls.minDistance = 1.3; controls.maxDistance = 10;
const frame = new THREE.Group(); scene.add(frame); // rotates with Ω when "inertial view" is on
let mesh: THREE.Mesh, colors: THREE.BufferAttribute, arrowGeo: THREE.BufferGeometry, arrowLines: THREE.LineSegments, arrowCells: number[] = [];
{ // rotation axis + arrow head
  const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, -1.6), new THREE.Vector3(0, 0, 1.6)]), new THREE.LineBasicMaterial({ color: 0xffd24a }));
  scene.add(axis);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.12, 16), new THREE.MeshBasicMaterial({ color: 0xffd24a }));
  cone.rotation.x = Math.PI / 2; cone.position.z = 1.66; cone.name = 'axisCone'; scene.add(cone);
  // graticule: equator + meridian, attached to the rotating frame
  const ring = (f: (t: number) => THREE.Vector3) => new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 128 }, (_, i) => f((2 * Math.PI * i) / 128))), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
  frame.add(ring((t) => new THREE.Vector3(Math.cos(t), Math.sin(t), 0).multiplyScalar(1.004)));
  frame.add(ring((t) => new THREE.Vector3(Math.sin(t), 0, Math.cos(t)).multiplyScalar(1.004)));
}
function buildMesh() {
  if (mesh) { frame.remove(mesh); mesh.geometry.dispose(); frame.remove(arrowLines); arrowGeo.dispose(); }
  const g = solver.g, N = g.N, posA = new Float32Array(N * 18);
  for (let k = 0; k < N; k++) {
    const c = [0, 1, 2, 0, 2, 3];
    for (let t = 0; t < 6; t++) for (let q = 0; q < 3; q++) posA[18 * k + 3 * t + q] = g.corners[12 * k + 3 * c[t] + q];
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(posA, 3));
  colors = new THREE.BufferAttribute(new Float32Array(N * 18), 3); geo.setAttribute('color', colors);
  mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.FrontSide })); frame.add(mesh);
  // arrows on a sub-lattice of cells
  const n = g.n, stride = Math.max(1, Math.round(n / 10)); arrowCells = [];
  for (let p = 0; p < 6; p++) for (let j = Math.floor(stride / 2); j < n; j += stride) for (let i = Math.floor(stride / 2); i < n; i += stride) arrowCells.push(g.index(p, i, j));
  arrowGeo = new THREE.BufferGeometry(); arrowGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(arrowCells.length * 18), 3));
  arrowLines = new THREE.LineSegments(arrowGeo, new THREE.LineBasicMaterial({ color: 0xffffff })); frame.add(arrowLines);
}
function lut(map: LUT, t: number, out: Float32Array, o: number) {
  const n = Math.max(0, Math.min(255, Math.round(t * 255)));
  const r = map[3 * n] / 255, g = map[3 * n + 1] / 255, b = map[3 * n + 2] / 255;
  for (let v = 0; v < 6; v++) { out[o + 3 * v] = r; out[o + 3 * v + 1] = g; out[o + 3 * v + 2] = b; }
}
function paint() {
  const g = solver.g, N = g.N, c = colors.array as Float32Array, u = solver.u, rho = solver.rho;
  if (view === 'density') {
    // log colour scale centred on ρ = 1; half-width in decades (auto: 98th percentile of |log10 ρ|)
    const L = new Float64Array(N); for (let k = 0; k < N; k++) L[k] = Math.log10(Math.max(rho[k], 1e-6));
    let w = parseFloat($<HTMLInputElement>('rho-range').value) || 0;
    if (!(w > 0)) { const a = Float64Array.from(L, Math.abs).sort(); w = Math.max(0.01, a[Math.floor(0.98 * (N - 1))]); }
    for (let k = 0; k < N; k++) lut(viridis, 0.5 + L[k] / (2 * w), c, 18 * k);
  } else if (view === 'speed') {
    let mx = 1e-9; const sp = new Float64Array(N);
    for (let k = 0; k < N; k++) { sp[k] = Math.hypot(u[3 * k], u[3 * k + 1], u[3 * k + 2]); mx = Math.max(mx, sp[k]); }
    for (let k = 0; k < N; k++) lut(viridis, sp[k] / mx, c, 18 * k);
  } else {
    for (let k = 0; k < N; k++) {
      const x = g.pos[3 * k], y = g.pos[3 * k + 1], z = g.pos[3 * k + 2], r = Math.hypot(x, y) || 1e-9;
      const ue = (-y * u[3 * k] + x * u[3 * k + 1]) / r, un = u[3 * k + 2] * r - z * (x * u[3 * k] + y * u[3 * k + 1]) / r;
      const [R, G, B] = hsv(((Math.atan2(un, ue) / (2 * Math.PI)) + 1) % 1, 0.85, 0.95);
      for (let v = 0; v < 6; v++) { c[18 * k + 3 * v] = R; c[18 * k + 3 * v + 1] = G; c[18 * k + 3 * v + 2] = B; }
    }
  }
  colors.needsUpdate = true;
  arrowLines.visible = arrows;
  if (arrows) {
    const a = arrowGeo.getAttribute('position') as THREE.BufferAttribute, A = a.array as Float32Array;
    let mx = 1e-9; for (const k of arrowCells) mx = Math.max(mx, Math.hypot(u[3 * k], u[3 * k + 1], u[3 * k + 2]));
    const L = aScale / Math.max(mx, params.v0 || mx);
    arrowCells.forEach((k, q) => {
      const p = [g.pos[3 * k], g.pos[3 * k + 1], g.pos[3 * k + 2]], v = [u[3 * k] * L, u[3 * k + 1] * L, u[3 * k + 2] * L];
      const s = 1.006, tip = p.map((x, i) => (x + v[i]) * s), base = p.map((x) => x * s);
      // shaft + two barbs
      const vl = Math.hypot(...v) || 1e-9, side = [p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]];
      const b1 = tip.map((x, i) => x - 0.3 * v[i] + 0.15 * side[i] / vl * vl), b2 = tip.map((x, i) => x - 0.3 * v[i] - 0.15 * side[i] / vl * vl);
      A.set([...base, ...tip, ...tip, ...b1, ...tip, ...b2], 18 * q);
    });
    a.needsUpdate = true;
  }
}
function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

// ---------- controls ----------
const freshSeed = () => crypto.getRandomValues(new Uint32Array(1))[0] >>> 0;
function seedFromInput() { const t = $<HTMLInputElement>('seed-input').value.trim(); const s = /^\d+$/.test(t) && Number(t) <= 0xffffffff ? Number(t) >>> 0 : freshSeed(); $('seed-used').textContent = String(s); return s; }
function init(kind = lastInit, seed = lastSeed) {
  lastInit = kind; lastSeed = seed; series.length = 0;
  if (kind === 'random') solver.seedRandom(seed, parseFloat($<HTMLInputElement>('seed-amp').value) || 0);
  else solver.seedZonal();
  frame.rotation.z = 0;
}
function build(n: number) { solver = new SphereSolver(n, params); buildMesh(); init(); }
const fmt = (x: number) => (!Number.isFinite(x) ? '∞' : x === 0 ? '0' : Math.abs(x) < 1e-2 || Math.abs(x) >= 1e3 ? x.toExponential(2) : x.toPrecision(3));
function bindSlider(id: string, get: () => number, set: (v: number) => void, to = (v: number) => v, from = (v: number) => v) {
  const el = $<HTMLInputElement>(id), out = $<HTMLOutputElement>(id + '-out');
  el.value = String(to(get())); out.textContent = fmt(get());
  el.addEventListener('input', () => { set(from(parseFloat(el.value))); out.textContent = fmt(get()); repInfo(); });
}
const check = (id: string, get: () => boolean, set: (v: boolean) => void) => { const el = $<HTMLInputElement>(id); el.checked = get(); el.addEventListener('change', () => { set(el.checked); repInfo(); }); };
bindSlider('omega', () => params.omega, (v) => (params.omega = v));
bindSlider('mu', () => params.mu, (v) => (params.mu = v), Math.log10, (v) => 10 ** v);
bindSlider('lambda', () => params.lambda, (v) => (params.lambda = Math.max(v, -2 * params.mu)));
bindSlider('tau', () => params.tau, (v) => (params.tau = v));
bindSlider('v0', () => params.v0, (v) => (params.v0 = v));
bindSlider('repA', () => params.repA, (v) => (params.repA = v));
bindSlider('R0', () => params.R0, (v) => (params.R0 = v));
bindSlider('alpha', () => params.alpha, (v) => (params.alpha = v));
bindSlider('vInt', () => params.vInt, (v) => (params.vInt = v >= 2 ? Infinity : 10 ** v), (v) => (Number.isFinite(v) ? Math.log10(v) : 2));
bindSlider('spf', () => spf, (v) => (spf = v));
bindSlider('ascale', () => aScale, (v) => (aScale = v));
check('lame-on', () => params.lameOn, (v) => (params.lameOn = v));
check('selfprop', () => params.selfProp, (v) => (params.selfProp = v));
check('rep-on', () => params.repOn, (v) => { params.repOn = v; solver.clearHistory(); });
check('arrows', () => arrows, (v) => (arrows = v));
check('spin', () => spin, (v) => { spin = v; if (!v) frame.rotation.z = 0; });
function repInfo() {
  const d = Number.isFinite(params.vInt) ? params.R0 / params.vInt : 0;
  $('rep-info').textContent = `max delay R₀/v_int = ${fmt(d)} s (${Math.ceil(d / params.dt)} steps)` + (solver?.pairCount ? ` · ${solver.pairCount} pairs` : '');
}
$<HTMLSelectElement>('view').addEventListener('change', (e) => (view = (e.target as HTMLSelectElement).value as View));
$<HTMLSelectElement>('grid').addEventListener('change', (e) => build(Number((e.target as HTMLSelectElement).value)));
const playBtn = $('play');
const setRunning = (r: boolean) => { running = r; playBtn.textContent = r ? '⏸ Pause' : '▶ Play'; };
playBtn.addEventListener('click', () => setRunning(!running));
$('step').addEventListener('click', () => advance(1));
$('reset').addEventListener('click', () => init());
$('random-init').addEventListener('click', () => init('random', seedFromInput()));
$('zonal-init').addEventListener('click', () => init('zonal'));
$('panel-toggle').addEventListener('click', () => $('panel').classList.toggle('hidden'));
addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).tagName === 'INPUT') return;
  if (e.key === ' ') { setRunning(!running); e.preventDefault(); } else if (e.key === 's') advance(1); else if (e.key === 'r') init();
  else if (e.key === 'h') $('panel').classList.toggle('hidden'); else if (e.key === 'a') { arrows = !arrows; $<HTMLInputElement>('arrows').checked = arrows; }
  else if (e.key === 'v') { const s = $<HTMLSelectElement>('view'); s.selectedIndex = (s.selectedIndex + 1) % s.options.length; view = s.value as View; }
});

// ---------- loop ----------
const series: number[] = [];
function advance(k: number) {
  const t0 = performance.now(); for (let i = 0; i < k; i++) solver.step();
  const ms = (performance.now() - t0) / k; msPerStep = msPerStep ? 0.9 * msPerStep + 0.1 * ms : ms;
  if (spin) frame.rotation.z = params.omega * solver.time;
}
let frameNo = 0, lastT = performance.now(), fps = 0;
function loop() {
  requestAnimationFrame(loop);
  if (running) advance(spf);
  paint(); controls.update();
  (scene.getObjectByName('axisCone') as THREE.Mesh).rotation.x = params.omega >= 0 ? Math.PI / 2 : -Math.PI / 2;
  (scene.getObjectByName('axisCone') as THREE.Mesh).position.z = params.omega >= 0 ? 1.66 : -1.66;
  renderer.render(scene, camera);
  const now = performance.now(); fps = 0.9 * fps + 0.1 * (1000 / Math.max(1, now - lastT)); lastT = now;
  if (frameNo++ % 10 === 0) {
    const o = solver.orderStats();
    $('fps').textContent = `${fps.toFixed(0)} fps`;
    $('stats').textContent = `t = ${solver.time.toFixed(1)} s · ${msPerStep.toFixed(1)} ms/step · ${solver.N} cells`;
    (window as unknown as { msPerStep: number }).msPerStep = msPerStep;
    $('order-info').textContent = `⟨cos⟩ ${o.localAlign.toFixed(3)} · ⟨cos⟩_ρ ${o.localAlignRho.toFixed(3)}\nL_z ${o.Lz.toFixed(3)} · Φ_L ${o.PhiL.toFixed(3)} (axis ${o.PhiLaxis.map((v) => v.toFixed(2)).join(', ')})\n⟨|u|⟩ ${o.meanSpeed.toFixed(3)} · ⟨|u|⟩_ρ ${o.meanSpeedRho.toFixed(3)} · ρ_max ${o.rhoMax.toFixed(2)} · mass/4π ${(o.mass / (4 * Math.PI)).toFixed(6)}`;
    repInfo();
  }
}
build(32); init('random', 1); $('seed-used').textContent = '1'; loop();
(window as unknown as { app: unknown }).app = { get solver() { return solver; }, params, renderer };
