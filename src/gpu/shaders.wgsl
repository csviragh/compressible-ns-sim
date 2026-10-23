// WebGPU port of Solver.step() (src/solver/solver.ts). All fields are f32, row-major k = j·n + i, periodic.
// Buffers: U = [u | v], U2 = scratch [u | v | dye], R = [ρ | r1 | ρ_new], D = [dye | unused], F = [fx | fy].
struct Params {
  n: u32, N: u32, M: u32, nsMax: u32,
  h: f32, dt: f32, dl: f32, mu: f32,
  b: f32, decay: f32, gain: f32, sp: u32,
  fade: f32, p0: f32, p1: f32, p2: f32,
};
struct Sub { s: u32 };
struct Entry { di: i32, dj: i32, wx: f32, wy: f32, a: u32, bb: u32, theta: f32, pad: f32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> S: Sub;
@group(0) @binding(2) var<storage, read_write> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> U2: array<f32>;
@group(0) @binding(4) var<storage, read_write> R: array<f32>;
@group(0) @binding(5) var<storage, read_write> D: array<f32>;
@group(0) @binding(6) var<storage, read_write> F: array<f32>;
@group(0) @binding(7) var<storage, read> ring: array<f32>;
@group(0) @binding(8) var<storage, read> st: array<Entry>;
@group(0) @binding(9) var<storage, read_write> cnt: array<atomic<u32>>;

const CUR: u32 = 0xffffffffu;
fn wrap(x: i32) -> u32 { let n = i32(P.n); return u32(((x % n) + n) % n); }
fn id(i: i32, j: i32) -> u32 { return wrap(j) * P.n + wrap(i); }

fn src(s: u32, q: u32) -> f32 { if (s == CUR) { return R[q]; } return ring[s * P.N + q]; }

// forces + retarded repulsion: u += Δt·f, then u += Δt·f_rep
@compute @workgroup_size(8, 8) fn repulse(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = g.y * P.n + g.x;
  var ax = 0.0; var ay = 0.0;
  for (var m = 0u; m < P.M; m++) {
    let e = st[m]; let q = id(i + e.di, j + e.dj);
    var r = src(e.a, q);
    if (e.theta > 0.0) { r = (1.0 - e.theta) * r + e.theta * src(e.bb, q); }
    ax += e.wx * r; ay += e.wy * r;
  }
  U[k] = (U[k] + P.dt * F[k]) + P.dt * ax;
  U[P.N + k] = (U[P.N + k] + P.dt * F[P.N + k]) + P.dt * ay;
}

// one forward-Euler Lamé sub-step: U2 = U + d·L[U]
@compute @workgroup_size(8, 8) fn lame(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let i = i32(g.x); let j = i32(g.y); let k = g.y * P.n + g.x; let N = P.N;
  let e = id(i + 1, j); let w = id(i - 1, j); let nn = id(i, j + 1); let s = id(i, j - 1);
  let ne = id(i + 1, j + 1); let nw = id(i - 1, j + 1); let se = id(i + 1, j - 1); let sw = id(i - 1, j - 1);
  let u = U[k]; let v = U[N + k];
  let lapU = U[e] + U[w] + U[nn] + U[s] - 4.0 * u;
  let lapV = U[N + e] + U[N + w] + U[N + nn] + U[N + s] - 4.0 * v;
  let uxx = U[e] - 2.0 * u + U[w]; let vyy = U[N + nn] - 2.0 * v + U[N + s];
  let vxy = (U[N + ne] - U[N + nw] - U[N + se] + U[N + sw]) * 0.25;
  let uxy = (U[ne] - U[nw] - U[se] + U[sw]) * 0.25;
  let ih2 = 1.0 / (P.h * P.h);
  U2[k] = u + P.dl * ((P.mu * lapU + P.b * (uxx + vxy)) * ih2);
  U2[N + k] = v + P.dl * ((P.mu * lapV + P.b * (vyy + uxy)) * ih2);
}

@compute @workgroup_size(8, 8) fn umax(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let k = g.y * P.n + g.x;
  atomicMax(&cnt[0], bitcast<u32>(max(abs(U[k]), abs(U[P.N + k]))));
}

fn nsEff() -> u32 {
  let um = max(bitcast<f32>(atomicLoad(&cnt[0])), 1e-12);
  let ns = max(1u, u32(ceil(um * P.dt / P.h / 0.25)));
  if (ns > P.nsMax) { atomicMax(&cnt[2], ns); }
  return min(ns, P.nsMax);
}
fn mc(a: f32, b: f32) -> f32 {
  if (a * b <= 0.0) { return 0.0; }
  let m = min(min(2.0 * abs(a), 2.0 * abs(b)), 0.5 * abs(a + b));
  return select(-m, m, a > 0.0);
}
// MUSCL upwind flux through the face kb|kc (ka = kb − 1, ke = kc + 1), velocity component at offset wo
fn flux(o: u32, wo: u32, ka: u32, kb: u32, kc: u32, ke: u32) -> f32 {
  let uf = 0.5 * (U[wo + kb] + U[wo + kc]);
  let Ra = R[o + ka]; let Rb = R[o + kb]; let Rc = R[o + kc]; let Re = R[o + ke];
  var rf: f32;
  if (uf >= 0.0) { rf = Rb + 0.5 * mc(Rb - Ra, Rc - Rb); } else { rf = Rc - 0.5 * mc(Rc - Rb, Re - Rc); }
  return uf * rf / P.h;
}
fn rhs(o: u32, i: i32, j: i32) -> f32 {
  let k = id(i, j);
  let fxl = flux(o, 0u, id(i - 2, j), id(i - 1, j), k, id(i + 1, j));
  let fxr = flux(o, 0u, id(i - 1, j), k, id(i + 1, j), id(i + 2, j));
  let fyl = flux(o, P.N, id(i, j - 2), id(i, j - 1), k, id(i, j + 1));
  let fyr = flux(o, P.N, id(i, j - 1), k, id(i, j + 1), id(i, j + 2));
  return (fxl - fxr) + (fyl - fyr);
}
@compute @workgroup_size(8, 8) fn dens1(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let ns = nsEff(); if (S.s >= ns) { return; }
  let k = g.y * P.n + g.x; let d = P.dt / f32(ns);
  R[P.N + k] = R[k] + d * rhs(0u, i32(g.x), i32(g.y));
}
@compute @workgroup_size(8, 8) fn dens2(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let ns = nsEff(); if (S.s >= ns) { return; }
  let k = g.y * P.n + g.x; let d = P.dt / f32(ns);
  R[2u * P.N + k] = 0.5 * (R[k] + R[P.N + k] + d * rhs(P.N, i32(g.x), i32(g.y)));
}
@compute @workgroup_size(8, 8) fn dens3(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let ns = nsEff(); if (S.s >= ns) { return; }
  let k = g.y * P.n + g.x;
  var r = R[2u * P.N + k];
  if (!(r >= 0.0)) { r = 0.0; atomicAdd(&cnt[1], 1u); }
  R[k] = r;
}

fn sample(o: u32, x: f32, y: f32) -> f32 {
  let fx = floor(x); let fy = floor(y); let sx = x - fx; let sy = y - fy;
  let i0 = i32(fx); let j0 = i32(fy);
  let a = id(i0, j0); let b = id(i0 + 1, j0); let c = id(i0, j0 + 1); let d = id(i0 + 1, j0 + 1);
  return (1.0 - sy) * ((1.0 - sx) * Uo(o, a) + sx * Uo(o, b)) + sy * ((1.0 - sx) * Uo(o, c) + sx * Uo(o, d));
}
// o < 2N selects U (u: 0, v: N); o = 2N selects the dye D
fn Uo(o: u32, q: u32) -> f32 { if (o >= 2u * P.N) { return D[q]; } return U[o + q]; }

// semi-Lagrangian advection (midpoint back-trace) of u, v, dye + exact self-propulsion + dye fade
@compute @workgroup_size(8, 8) fn advect(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= P.n || g.y >= P.n) { return; }
  let k = g.y * P.n + g.x; let N = P.N; let s = P.dt / P.h;
  let fi = f32(g.x); let fj = f32(g.y);
  let xm = fi - 0.5 * s * U[k]; let ym = fj - 0.5 * s * U[N + k];
  let x = fi - s * sample(0u, xm, ym); let y = fj - s * sample(N, xm, ym);
  var nu = sample(0u, x, y); var nv = sample(N, x, y);
  let nd = sample(2u * N, x, y);
  if (P.sp != 0u) {
    let sp = sqrt(nu * nu + nv * nv);
    if (sp >= 1e-9) { let f = P.decay + P.gain / sp; nu *= f; nv *= f; }
  }
  U2[k] = nu; U2[N + k] = nv; U2[2u * N + k] = nd * P.fade;
}
