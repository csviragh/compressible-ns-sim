// Retarded / instantaneous periodic 2D Coulomb repulsion, direct sum over all grid offsets (O(N²)).
// u += Δt·A Σ_{o in chunk} W(o) ρ(x − o, t − q(o)Δt);  W, q from src/solver/coulomb.ts (K buffer),
// per-bin history slots/θ in the per-step bin table.
struct CP { n: u32, N: u32, o0: u32, o1: u32, dtA: f32, p0: f32, p1: f32, p2: f32 };
struct Bin { a: u32, b: u32, theta: f32, pad: f32 };
@group(0) @binding(0) var<uniform> C: CP;
@group(0) @binding(1) var<storage, read_write> U: array<f32>;
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read> ring: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<vec4f>;
@group(0) @binding(5) var<storage, read> bins: array<Bin>;
const CUR: u32 = 0xffffffffu;
fn src(s: u32, q: u32) -> f32 { if (s == CUR) { return R[q]; } return ring[s * C.N + q]; }
@compute @workgroup_size(8, 8) fn coulomb(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= C.n || g.y >= C.n) { return; }
  let n = C.n; var ax = 0.0; var ay = 0.0;
  for (var o = C.o0; o < C.o1; o++) {
    let e = K[o]; let oi = o % n; let oj = o / n;
    let q = ((g.y + n - oj) % n) * n + (g.x + n - oi) % n;
    let B = bins[bitcast<u32>(e.z)];
    var r = src(B.a, q);
    if (B.theta > 0.0) { r = (1.0 - B.theta) * r + B.theta * src(B.b, q); }
    ax += e.x * r; ay += e.y * r;
  }
  let k = g.y * n + g.x;
  U[k] += C.dtA * ax; U[C.N + k] += C.dtA * ay;
}
