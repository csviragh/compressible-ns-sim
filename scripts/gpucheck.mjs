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
for (const [name, pp] of cases) {
  const r = await p.evaluate(async (pp) => {
    const a = window.app; Object.assign(a.params, { lameOn: true, repOn: false, vInt: Infinity, lambda: 0 }, pp);
    a.solver.seedRandom(3, 0.3); a.solver.u.forEach((_, k) => a.solver.rho[k] = 1 + 0.3 * Math.sin(k * 0.05));
    return await a.compare(40);
  }, pp);
  console.log(name, JSON.stringify(r, (k, v) => typeof v === 'number' ? +v.toPrecision(3) : v));
}
await b.close();
