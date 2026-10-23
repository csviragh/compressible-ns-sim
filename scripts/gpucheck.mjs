// Dev check: GPU (WebGPU, f32) vs CPU (f64) on the same initial state, in headless Chrome.
// Usage: npm run build && npx vite preview --port 4173 &  then  CHROME=/usr/bin/google-chrome npm run gpucheck
// Without a real GPU it uses SwiftShader (software Vulkan), so the timings are not GPU performance.
import puppeteer from 'puppeteer-core';
const b = await puppeteer.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome', headless: 'new',
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-webgpu-adapter=swiftshader', '--use-vulkan=swiftshader'] });
const p = await b.newPage(); p.on('pageerror', e => console.log('pageerror', e.message)); p.on('console', m => { const t = m.text(); if (!t.startsWith('gpucheck')) console.log('console:', t.slice(0, 300)); });
await p.goto(process.argv[2] || 'http://localhost:4173/');
await p.waitForFunction(() => window.app.gpu, { timeout: 60000 });
await p.evaluate(() => document.getElementById('play').click());
await new Promise((r) => setTimeout(r, 2000)); // let an in-flight GPU frame finish
const cases = [
  ['defaults (Lamé on, repulsion off)', {}],
  ['Lamé off', { lameOn: false }],
  ['repulsion instantaneous', { repOn: true, vInt: Infinity, lambda: 2e-4 }],
  ['repulsion retarded v_int=0.3 (after 40 steps of history)', { repOn: true, vInt: 0.3, lambda: 2e-4 }],
];
const coulombCases = [
  ['Coulomb instantaneous (CPU FFT vs GPU direct sum), 64²', { repOn: true, repKind: 'coulomb', vInt: Infinity, repA: 0.5, lambda: 2e-4 }],
  ['Coulomb retarded v_int=0.3, 64²', { repOn: true, repKind: 'coulomb', vInt: 0.3, repA: 0.5, lambda: 2e-4 }],
  ['Coulomb retarded v_int=2, 64²', { repOn: true, repKind: 'coulomb', vInt: 2, repA: 0.5, lambda: 2e-4 }],
];
const grid = async (n) => { await p.evaluate((n) => { const g = document.getElementById('grid'); if (![...g.options].some(o => o.value === String(n))) { const o = document.createElement('option'); o.value = String(n); o.textContent = n; g.appendChild(o); } g.value = String(n); g.dispatchEvent(new Event('change')); }, n); await new Promise((r) => setTimeout(r, 500)); };
const only = process.env.CASES; // 'coulomb' → only the Coulomb cases
const all = only === 'coulomb' ? [] : cases.map(c => [...c, 256]);
for (const c of coulombCases) all.push([...c, 64]);
for (const [name, pp, n] of all) {
  await grid(n);
  const r = await p.evaluate(async (pp) => {
    const a = window.app; Object.assign(a.params, { lameOn: true, repOn: false, vInt: Infinity, lambda: 0, repKind: 'short', repA: 0.5 }, pp);
    a.solver.seedRandom(3, 0.3); a.solver.u.forEach((_, k) => a.solver.rho[k] = 1 + 0.3 * Math.sin(k * 0.05));
    return await a.compare(40);
  }, pp);
  console.log(name, JSON.stringify(r, (k, v) => typeof v === 'number' ? +v.toPrecision(3) : v));
}
await b.close();
