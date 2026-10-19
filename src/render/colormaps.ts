/**
 * Colormaps as 256-entry RGB lookup tables (Uint8Array of length 768).
 */
export type LUT = Uint8Array;

function buildLUT(stops: readonly (readonly [number, number, number, number])[]): LUT {
  const lut = new Uint8Array(256 * 3);
  for (let n = 0; n < 256; n++) {
    const t = n / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, r0, g0, b0] = stops[k];
    const [t1, r1, g1, b1] = stops[k + 1];
    const s = t1 > t0 ? Math.min(Math.max((t - t0) / (t1 - t0), 0), 1) : 0;
    lut[3 * n] = Math.round(r0 + s * (r1 - r0));
    lut[3 * n + 1] = Math.round(g0 + s * (g1 - g0));
    lut[3 * n + 2] = Math.round(b0 + s * (b1 - b0));
  }
  return lut;
}

/** Perceptually uniform sequential map (sampled from matplotlib's viridis). */
export const viridis: LUT = buildLUT([
  [0.0, 68, 1, 84],
  [0.125, 71, 44, 122],
  [0.25, 59, 81, 139],
  [0.375, 44, 113, 142],
  [0.5, 33, 144, 141],
  [0.625, 39, 173, 129],
  [0.75, 92, 200, 99],
  [0.875, 170, 220, 50],
  [1.0, 253, 231, 37],
]);

/** Diverging blue–white–red map (similar to "coolwarm"/"RdBu"), 0.5 = zero. */
export const diverging: LUT = buildLUT([
  [0.0, 5, 48, 97],
  [0.2, 33, 102, 172],
  [0.4, 146, 197, 222],
  [0.5, 247, 247, 247],
  [0.6, 244, 165, 130],
  [0.8, 178, 24, 43],
  [1.0, 103, 0, 31],
]);

/** Convert HSV (h in [0, 1)) to RGB in [0, 1]. */
export function hsv(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (((i % 6) + 6) % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

/** Parse a #rrggbb string to RGB in [0, 1]. */
export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return [1, 1, 1];
  return [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
}
