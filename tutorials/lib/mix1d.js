// Exact fields of a 1D Gaussian mixture on the linear path x_t = t x_1 + (1 - t) x_0, x_0 ~ N(0, 1),
// shared by the §4, §5 and §6 tutorials. A mixture is { mu, sd, w } with weights summing to one.
// Component k of the noised mixture is N(t mu_k, t² sd_k² + (1 - t)²). No DOM access.

export const MIX1 = { mu: [-1.6, 0.3, 1.9], sd: [0.45, 0.35, 0.5], w: [0.45, 0.3, 0.25] };
export const XMIN = -4.5, XMAX = 4.5, NG = 361;
export const XG = Float64Array.from({ length: NG }, (_, i) => XMIN + (XMAX - XMIN) * i / (NG - 1));
export const DX = (XMAX - XMIN) / (NG - 1);
const R2PI = Math.sqrt(2 * Math.PI);

export function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}
export const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// Component means and variances at time t.
export function comps(m, t) {
  return { mt: m.mu.map((u) => t * u), vt: m.sd.map((s) => t * t * s * s + (1 - t) ** 2) };
}
// Posterior over components and all fields at (x, t). u is E[x_1 - x_0 | x_t = x], stable up to t = 1.
export function fields(m, x, t) {
  const { mt, vt } = comps(m, t), K = m.mu.length;
  let lmax = -Infinity; const lw = new Array(K);
  for (let k = 0; k < K; k++) { lw[k] = Math.log(m.w[k]) - 0.5 * Math.log(vt[k]) - (x - mt[k]) ** 2 / (2 * vt[k]); lmax = Math.max(lmax, lw[k]); }
  let Z = 0; for (let k = 0; k < K; k++) { lw[k] = Math.exp(lw[k] - lmax); Z += lw[k]; }
  let s = 0, u = 0, x1 = 0;
  for (let k = 0; k < K; k++) {
    const r = lw[k] / Z, d = x - mt[k], s2 = m.sd[k] ** 2;
    s -= r * d / vt[k];
    u += r * (m.mu[k] + (t * s2 - (1 - t)) * d / vt[k]);
    x1 += r * (m.mu[k] + t * s2 * d / vt[k]);
  }
  return { logp: lmax + Math.log(Z) - Math.log(R2PI), score: s, u, xhat: x1 };
}
export const dens = (m, x, t = 1) => Math.exp(fields(m, x, t).logp);
export const score = (m, x, t = 1) => fields(m, x, t).score;
export const vel = (m, x, t) => fields(m, x, t).u;
export function cdf(m, x, t = 1) {
  const { mt, vt } = comps(m, t);
  return m.w.reduce((a, w, k) => a + w * Phi((x - mt[k]) / Math.sqrt(vt[k])), 0);
}
export function quantile(m, q, t = 1) {
  let lo = -12, hi = 12;
  for (let i = 0; i < 60; i++) { const mid = 0.5 * (lo + hi); if (cdf(m, mid, t) < q) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}
// Exact flow map of the probability-flow ODE in 1D: the monotone map that carries p_s to p_t.
export const flowMap = (m, x, s, t) => (s === t ? x : quantile(m, cdf(m, x, s), t));
// Stratified quantiles of a mixture (for exact-in-distribution sample sets).
export const strat = (m, n, t = 1) => Float64Array.from({ length: n }, (_, i) => quantile(m, (i + 0.5) / n, t));
export const densGrid = (m, t = 1) => Float64Array.from(XG, (x) => dens(m, x, t));

// Tilting by a quadratic reward r(x) = -(x - a)²/2 at temperature beta keeps a Gaussian mixture.
export function tiltQuad(m, a, beta) {
  const sd = [], mu = [], lw = [];
  m.mu.forEach((u, k) => {
    const v = m.sd[k] ** 2, vp = 1 / (1 / v + 1 / beta);
    sd.push(Math.sqrt(vp)); mu.push(vp * (u / v + a / beta));
    lw.push(Math.log(m.w[k]) - 0.5 * Math.log(v + beta) - (a - u) ** 2 / (2 * (v + beta)));
  });
  const mx = Math.max(...lw), e = lw.map((l) => Math.exp(l - mx)), Z = e.reduce((p, q) => p + q, 0);
  return { mu, sd, w: e.map((q) => q / Z) };
}

// ---------- Grid quantities ----------
export function normalize(p) { const Z = p.reduce((a, b) => a + b, 0) * DX; return p.map((q) => q / Z); }
export function klGrid(p, q) { let s = 0; for (let i = 0; i < p.length; i++) if (p[i] > 1e-300) s += p[i] * Math.log(p[i] / Math.max(q[i], 1e-300)); return s * DX; }
export const meanGrid = (p, f) => p.reduce((a, q, i) => a + q * f(XG[i]), 0) * DX;
// W1 between two densities on the grid: integral of |F - G|.
export function w1Grid(p, q) { let F = 0, G = 0, s = 0; for (let i = 0; i < p.length; i++) { F += p[i] * DX; G += q[i] * DX; s += Math.abs(F - G); } return s * DX; }
// W1 between a sample set and a mixture: integral of |F_n - F| on a fine grid.
export function w1Samples(xs, m, t = 1) {
  const a = Float64Array.from(xs).sort(), n = a.length, lo = Math.min(a[0], -8), hi = Math.max(a[n - 1], 8), M = 4000, h = (hi - lo) / M;
  let j = 0, s = 0;
  for (let i = 0; i < M; i++) { const x = lo + (i + 0.5) * h; while (j < n && a[j] <= x) j++; s += Math.abs(j / n - cdf(m, x, t)); }
  return s * h;
}
// W1 between two sample sets of equal size.
export function w1Pair(xs, ys) {
  const a = Float64Array.from(xs).sort(), b = Float64Array.from(ys).sort();
  let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length;
}
// Gaussian kernel density estimate on the grid.
export function kde(xs, bw = 0.12) {
  const out = new Float64Array(NG), c = 1 / (xs.length * bw * R2PI);
  for (const x of xs) {
    const i0 = Math.max(0, Math.floor((x - 4 * bw - XMIN) / DX)), i1 = Math.min(NG - 1, Math.ceil((x + 4 * bw - XMIN) / DX));
    for (let i = i0; i <= i1; i++) out[i] += c * Math.exp(-((XG[i] - x) ** 2) / (2 * bw * bw));
  }
  return out;
}
// A Gaussian mixture with free weights (softmax of logits), means and log-widths, as a flat vector.
export const toVec = (m) => [...m.w.map(Math.log), ...m.mu, ...m.sd.map(Math.log)];
export function fromVec(v) {
  const K = v.length / 3, l = v.slice(0, K), mx = Math.max(...l), e = l.map((q) => Math.exp(q - mx)), Z = e.reduce((a, b) => a + b, 0);
  return { w: e.map((q) => q / Z), mu: v.slice(K, 2 * K), sd: v.slice(2 * K).map(Math.exp) };
}
