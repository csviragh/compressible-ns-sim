/**
 * Equiangular cubed-sphere grid on the UNIT sphere (radius R = 1; all lengths are in units of R,
 * so a distance d is also the great-circle angle in radians).
 *
 * 6 panels × n × n cells. Panel p maps (ξ, η) ∈ [−π/4, π/4]² to x = normalize(c + tan ξ a + tan η b).
 * Cell areas are exact spherical quadrilateral areas (sum = 4π); every cell has exactly 4 edge
 * neighbours (across panel edges too, found geometrically), and each edge stores its exact
 * great-circle length and the tangent face normal at its midpoint, so finite-volume fluxes are
 * exactly antisymmetric (mass is conserved to round-off).
 */
export type V3 = [number, number, number];
const PANELS: [V3, V3, V3][] = [
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
  [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
  [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
  [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
];
const norm = (v: V3): V3 => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const gcDist = (a: V3, b: V3) => Math.atan2(Math.hypot(...cross(a, b)), dot(a, b));

function triArea(a: V3, b: V3, c: V3) {
  return 2 * Math.atan2(Math.abs(dot(a, cross(b, c))), 1 + dot(a, b) + dot(b, c) + dot(c, a));
}

export class CubedSphere {
  readonly n: number; readonly N: number; readonly dxi: number;
  readonly pos: Float64Array;      // 3N cell centres (unit vectors)
  readonly area: Float64Array;     // N
  readonly nbr: Int32Array;        // 4N: neighbours in −ξ, +ξ, −η, +η directions
  readonly corners: Float64Array;  // 12N: 4 corner unit vectors per cell (for rendering)
  // edges (i < j)
  readonly ei: Int32Array; readonly ej: Int32Array;
  readonly elen: Float64Array;     // great-circle edge length l_ij
  readonly edist: Float64Array;    // centre distance d_ij
  readonly enrm: Float64Array;     // 3E tangent face normal at edge midpoint, pointing i → j
  readonly etan: Float64Array;     // 3E unit tangent at the edge midpoint along the great circle i → j
  readonly eoff: Float64Array;     // 6E log-map offsets of the edge midpoint from centre i and from centre j (tangent vectors)
  readonly lsq: Float64Array;      // 12N least-squares gradient weights: ∇f_i ≈ Σ_q lsq[i,q] (f_nbr(q) − f_i)
  constructor(n: number) {
    this.n = n; const N = (this.N = 6 * n * n); this.dxi = Math.PI / 2 / n;
    this.pos = new Float64Array(3 * N); this.area = new Float64Array(N);
    this.nbr = new Int32Array(4 * N); this.corners = new Float64Array(12 * N);
    const vert = (p: number, xi: number, et: number) => {
      const [c, a, b] = PANELS[p], tx = Math.tan(xi), ty = Math.tan(et);
      return norm([c[0] + tx * a[0] + ty * b[0], c[1] + tx * a[1] + ty * b[1], c[2] + tx * a[2] + ty * b[2]]);
    };
    const q = Math.PI / 4, d = this.dxi;
    for (let p = 0; p < 6; p++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = this.index(p, i, j);
      const ctr = vert(p, -q + (i + 0.5) * d, -q + (j + 0.5) * d);
      this.pos.set(ctr, 3 * k);
      const c0 = vert(p, -q + i * d, -q + j * d), c1 = vert(p, -q + (i + 1) * d, -q + j * d);
      const c2 = vert(p, -q + (i + 1) * d, -q + (j + 1) * d), c3 = vert(p, -q + i * d, -q + (j + 1) * d);
      this.corners.set([...c0, ...c1, ...c2, ...c3], 12 * k);
      this.area[k] = triArea(c0, c1, c2) + triArea(c0, c2, c3);
      const step = (di: number, dj: number) =>
        (i + di >= 0 && i + di < n && j + dj >= 0 && j + dj < n) ? this.index(p, i + di, j + dj)
          : this.locate(vert(p, -q + (i + 0.5 + di) * d, -q + (j + 0.5 + dj) * d));
      this.nbr[4 * k] = step(-1, 0); this.nbr[4 * k + 1] = step(1, 0);
      this.nbr[4 * k + 2] = step(0, -1); this.nbr[4 * k + 3] = step(0, 1);
    }
    const ei: number[] = [], ej: number[] = [], el: number[] = [], ed: number[] = [], en: number[] = [], et: number[] = [], eo: number[] = [];
    for (let k = 0; k < N; k++) for (let s = 0; s < 4; s++) {
      const m = this.nbr[4 * k + s]; if (m <= k) continue;
      const shared: V3[] = [];
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) {
        const A = this.corner(k, a), B = this.corner(m, b);
        if (Math.hypot(...sub(A, B)) < 1e-9) shared.push(A);
      }
      if (shared.length !== 2) throw new Error(`edge ${k}-${m}: ${shared.length} shared corners`);
      const [A, B] = shared, mid = norm([A[0] + B[0], A[1] + B[1], A[2] + B[2]]);
      let nv = norm(cross(mid, sub(B, A)));
      if (dot(nv, sub(this.p(m), this.p(k))) < 0) nv = [-nv[0], -nv[1], -nv[2]];
      ei.push(k); ej.push(m); el.push(gcDist(A, B)); ed.push(gcDist(this.p(k), this.p(m))); en.push(...nv);
      const logm = (c: V3): V3 => { const cd = dot(c, mid), t = sub(mid, [c[0] * cd, c[1] * cd, c[2] * cd]), l = Math.hypot(...t) || 1, d = gcDist(c, mid); return [t[0] / l * d, t[1] / l * d, t[2] / l * d]; };
      eo.push(...logm(this.p(k)), ...logm(this.p(m)));
      const pk = this.p(k), pm = this.p(m), dd = sub(pm, pk), mm = norm([pk[0] + pm[0], pk[1] + pm[1], pk[2] + pm[2]]), dm = dot(dd, mm);
      et.push(...norm([dd[0] - dm * mm[0], dd[1] - dm * mm[1], dd[2] - dm * mm[2]]));
    }
    this.etan = Float64Array.from(et); this.eoff = Float64Array.from(eo);
    // least-squares gradient on the 4-neighbour stencil, offsets = log map of the neighbour centre
    this.lsq = new Float64Array(12 * N);
    for (let k = 0; k < N; k++) {
      const pk = this.p(k), T: V3[] = [];
      for (let q = 0; q < 4; q++) {
        const pm = this.p(this.nbr[4 * k + q]), c = dot(pk, pm), t = sub(pm, [pk[0] * c, pk[1] * c, pk[2] * c]);
        const l = Math.hypot(...t), dd = gcDist(pk, pm); T.push([t[0] / l * dd, t[1] / l * dd, t[2] / l * dd]);
      }
      const e1 = norm(Math.abs(pk[0]) < 0.9 ? cross(pk, [1, 0, 0]) : cross(pk, [0, 1, 0])), e2 = cross(pk, e1);
      let a = 0, b = 0, cc = 0; const tt = T.map((t) => [dot(t, e1), dot(t, e2)]);
      for (const [x, y] of tt) { a += x * x; b += x * y; cc += y * y; }
      const det = a * cc - b * b;
      for (let q = 0; q < 4; q++) {
        const [x, y] = tt[q], g1 = (cc * x - b * y) / det, g2 = (-b * x + a * y) / det;
        for (let c = 0; c < 3; c++) this.lsq[12 * k + 3 * q + c] = g1 * e1[c] + g2 * e2[c];
      }
    }
    this.ei = Int32Array.from(ei); this.ej = Int32Array.from(ej);
    this.elen = Float64Array.from(el); this.edist = Float64Array.from(ed); this.enrm = Float64Array.from(en);
  }
  index(p: number, i: number, j: number) { return (p * this.n + j) * this.n + i; }
  p(k: number): V3 { return [this.pos[3 * k], this.pos[3 * k + 1], this.pos[3 * k + 2]]; }
  corner(k: number, a: number): V3 { const o = 12 * k + 3 * a; return [this.corners[o], this.corners[o + 1], this.corners[o + 2]]; }
  /** panel and continuous equiangular coordinates of a point */
  coords(x: V3): [number, number, number] {
    let best = 0, bv = -Infinity;
    for (let p = 0; p < 6; p++) { const v = dot(x, PANELS[p][0]); if (v > bv) { bv = v; best = p; } }
    const [, a, b] = PANELS[best];
    return [best, Math.atan(dot(x, a) / bv), Math.atan(dot(x, b) / bv)];
  }
  locate(x: V3): number {
    const [p, xi, et] = this.coords(x), n = this.n, q = Math.PI / 4;
    const i = Math.min(n - 1, Math.max(0, Math.floor((xi + q) / this.dxi)));
    const j = Math.min(n - 1, Math.max(0, Math.floor((et + q) / this.dxi)));
    return this.index(p, i, j);
  }
  /** cell (p, i, j) with i, j allowed one step outside the panel (resolved via neighbour table) */
  cellAt(p: number, i: number, j: number): number {
    const n = this.n, iin = i >= 0 && i < n, jin = j >= 0 && j < n;
    if (iin && jin) return this.index(p, i, j);
    const ci = Math.min(n - 1, Math.max(0, i)), cj = Math.min(n - 1, Math.max(0, j)), k = this.index(p, ci, cj);
    if (!iin && !jin) return k; // cube corner: fall back to the corner cell
    if (!iin) return this.nbr[4 * k + (i < 0 ? 0 : 1)];
    return this.nbr[4 * k + (j < 0 ? 2 : 3)];
  }
  /** bilinear stencil (4 cells, weights) in equiangular coordinates around point x */
  stencil(x: V3, cells: Int32Array, w: Float64Array) {
    const [p, xi, et] = this.coords(x), q = Math.PI / 4;
    const fx = (xi + q) / this.dxi - 0.5, fy = (et + q) / this.dxi - 0.5;
    const i0 = Math.floor(fx), j0 = Math.floor(fy), sx = fx - i0, sy = fy - j0;
    cells[0] = this.cellAt(p, i0, j0); cells[1] = this.cellAt(p, i0 + 1, j0);
    cells[2] = this.cellAt(p, i0, j0 + 1); cells[3] = this.cellAt(p, i0 + 1, j0 + 1);
    w[0] = (1 - sx) * (1 - sy); w[1] = sx * (1 - sy); w[2] = (1 - sx) * sy; w[3] = sx * sy;
    if (i0 >= 0 && j0 >= 0 && i0 + 1 < this.n && j0 + 1 < this.n) return;
    // stencil crosses a panel edge: the ghost cells are not where the panel's coordinates put them
    // (grid lines kink), so use the exact positions instead — weights of the least-squares linear fit
    // f ≈ a + g·t through the 4 cells (t = tangent offset at x), evaluated at x (linear-exact).
    const e1 = norm(Math.abs(x[0]) < 0.9 ? cross(x, [1, 0, 0]) : cross(x, [0, 1, 0])), e2 = cross(x, e1);
    // inverse-distance weights make the fit (nearly) interpolating when x sits on a cell centre
    const T: number[][] = [], W: number[] = [], eps = (0.05 * this.dxi) ** 2;
    for (let q = 0; q < 4; q++) { const c = this.p(cells[q]), s2 = 1 / Math.max(1e-12, dot(c, x)), a = dot(c, e1) * s2, b = dot(c, e2) * s2; T.push([1, a, b]); W.push(1 / (a * a + b * b + eps)); }
    // normal equations M = Σ τ τᵀ (3×3), weights w_q = (M⁻¹ τ_q)_0
    const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    T.forEach((t, q) => { for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[3 * a + b] += W[q] * t[a] * t[b]; });
    const [a, b, c, d, e, f, g2, h, i] = M, A = e * i - f * h, B = -(d * i - f * g2), C = d * h - e * g2, det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-30) return;
    const r0 = [A / det, -(b * i - c * h) / det, (b * f - c * e) / det]; // first row of M⁻¹ (M symmetric)
    for (let q = 0; q < 4; q++) w[q] = W[q] * (r0[0] * T[q][0] + r0[1] * T[q][1] + r0[2] * T[q][2]);
  }
}
