// Numerics of the §5 tutorial. The teacher is the exact probability-flow ODE of the 1D mixture MIX1 on
// the linear path; in 1D its flow map is F_t^{-1}(F_s(x)). Students are piecewise-linear maps fitted by
// least squares. Generated sets are 2000 stratified quantiles of the noise, so W1 to p_data has no
// Monte Carlo noise except where a step draws fresh noise. No DOM access.
import { MIX1, XG, DX, NG, fields, vel, cdf, quantile, flowMap, strat, densGrid, w1Samples, normalize, klGrid } from '../lib/mix1d.js';
import { makeRng } from '../../animations/lib/rng.js';
import { MIX2, gmm2 } from '../score-matching/sim.js';

export { MIX1, XG, vel, flowMap, densGrid, MIX2 };
export const pData = densGrid(MIX1);
export const NS = 2000;
export const Z0 = strat(MIX1, NS, 0);               // p_0 of the linear path is N(0, 1)
export const w1 = (xs) => w1Samples(xs, MIX1);
export const W1_EXACT = w1(Z0.map((z) => flowMap(MIX1, z, 0, 1)));

// ---------- Piecewise-linear maps ----------
// Knots at fixed positions, values fitted by least squares with the hat basis; linear beyond the ends.
export function plEval(P, x) {
  const { k, v } = P, n = k.length;
  let i = 0;
  if (x <= k[0]) i = 0; else if (x >= k[n - 1]) i = n - 2;
  else { let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (k[m] <= x) lo = m; else hi = m; } i = lo; }
  const a = (x - k[i]) / (k[i + 1] - k[i]);
  return v[i] + a * (v[i + 1] - v[i]);
}
export function plFit(k, xs, ys, ridge = 1e-6) {
  const n = k.length, A = Array.from({ length: n }, () => new Float64Array(n)), b = new Float64Array(n);
  for (let s = 0; s < xs.length; s++) {
    const x = xs[s]; let i;
    if (x <= k[0]) i = 0; else if (x >= k[n - 1]) i = n - 2;
    else { let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (k[m] <= x) lo = m; else hi = m; } i = lo; }
    const a = (x - k[i]) / (k[i + 1] - k[i]), w0 = 1 - a, w1_ = a;
    A[i][i] += w0 * w0; A[i][i + 1] += w0 * w1_; A[i + 1][i] += w0 * w1_; A[i + 1][i + 1] += w1_ * w1_;
    b[i] += w0 * ys[s]; b[i + 1] += w1_ * ys[s];
  }
  for (let i = 0; i < n; i++) A[i][i] += ridge * xs.length;
  // Tridiagonal solve.
  const c = new Float64Array(n), d = new Float64Array(n), v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = i ? A[i][i - 1] : 0, den = A[i][i] - lo * (i ? c[i - 1] : 0);
    c[i] = i < n - 1 ? A[i][i + 1] / den : 0; d[i] = (b[i] - lo * (i ? d[i - 1] : 0)) / den;
  }
  for (let i = n - 1; i >= 0; i--) v[i] = d[i] - (i < n - 1 ? c[i] * v[i + 1] : 0);
  return { k: Float64Array.from(k), v };
}
// M knots spread over the quantiles 0.002 to 0.998 of p_t.
export const knotsAt = (M, t) => Float64Array.from({ length: M }, (_, i) => quantile(MIX1, 0.002 + 0.996 * i / (M - 1), t));

// ---------- Step 2: Euler on the exact field ----------
export function eulerPath(x, K) {
  const xs = [x];
  for (let k = 0; k < K; k++) { const t = k / K; x += vel(MIX1, x, t) / K; xs.push(x); }
  return xs;
}
export const euler = (x, K) => { for (let k = 0; k < K; k++) x += vel(MIX1, x, k / K) / K; return x; };
export const KS2 = [1, 2, 4, 8, 16, 32, 64, 128];
let e2 = null;
export const eulerW1 = () => (e2 ??= KS2.map((K) => w1(Z0.map((z) => euler(z, K)))));

// ---------- Step 3: progressive distillation ----------
// A model is a list of steps on the grid t_j = j/n; each step maps t_j to t_{j+1}.
export const TEACH3 = 64;
const marg = new Map();
const marginal = (t) => { if (!marg.has(t)) marg.set(t, strat(MIX1, NS, t)); return marg.get(t); };
export function teacher3() {
  return { n: TEACH3, steps: Array.from({ length: TEACH3 }, (_, j) => (x) => x + vel(MIX1, x, j / TEACH3) / TEACH3) };
}
export function distilRound(prev, M) {
  const n = prev.n / 2, steps = [], maps = [];
  for (let j = 0; j < n; j++) {
    const t = j / n, xs = marginal(t), ys = xs.map((x) => prev.steps[2 * j + 1](prev.steps[2 * j](x)));
    const P = plFit(knotsAt(M, t), xs, ys); maps.push(P); steps.push((x) => plEval(P, x));
  }
  return { n, steps, maps };
}
export const runModel = (Mo, x) => Mo.steps.reduce((y, f) => f(y), x);
export const modelW1 = (Mo) => w1(Z0.map((z) => runModel(Mo, z)));

// ---------- Step 4: consistency models ----------
// f_j approximates X_{t_j -> 1} on the grid t_j = j/N, with f_N the identity. Distillation fits f_j to
// f_{j+1} after one Euler step of the teacher; training fits it to f_{j+1} at the same (x_0, x_1) pair.
export const MC = 16, NPAIR4 = 4000;
const PAIRS4 = (() => {
  const r = makeRng(17), x0 = new Float64Array(NPAIR4), x1 = new Float64Array(NPAIR4);
  for (let i = 0; i < NPAIR4; i++) {
    x0[i] = r.gauss(); const u = r.uniform();
    const k = u < MIX1.w[0] ? 0 : u < MIX1.w[0] + MIX1.w[1] ? 1 : 2;
    x1[i] = MIX1.mu[k] + MIX1.sd[k] * r.gauss();
  }
  return { x0, x1 };
})();
export function consistency(N, kind) {
  const f = new Array(N + 1); f[N] = (x) => x;
  const maps = new Array(N);
  for (let j = N - 1; j >= 0; j--) {
    const t = j / N, t2 = (j + 1) / N; let xs, ys;
    if (kind === 'cd') { xs = marginal(t); ys = xs.map((x) => f[j + 1](x + vel(MIX1, x, t) / N)); }
    else {
      const { x0, x1 } = PAIRS4;
      xs = Array.from(x0, (z, i) => t * x1[i] + (1 - t) * z); ys = Array.from(x0, (z, i) => f[j + 1](t2 * x1[i] + (1 - t2) * z));
    }
    const P = plFit(knotsAt(MC, t), xs, ys); maps[j] = P; f[j] = (x) => plEval(P, x);
  }
  return { N, f, maps };
}
// Two steps: jump to data, re-noise to t' = 1/2 with fresh seeded noise, jump again.
const ZR = (() => { const r = makeRng(23), z = strat(MIX1, NS, 0).slice(); for (let i = NS - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [z[i], z[j]] = [z[j], z[i]]; } return z; })();
export function cmSamples(C) {
  const one = Z0.map((z) => C.f[0](z)), jh = C.N / 2;
  const two = one.map((x, i) => C.f[jh](0.5 * x + 0.5 * ZR[i]));
  return { one, two };
}
// The same two-step scheme with the exact flow map, as a reference for the re-noising error.
export const EXACT_TWO = w1(Z0.map((z, i) => flowMap(MIX1, 0.5 * flowMap(MIX1, z, 0, 1) + 0.5 * ZR[i], 0.5, 1)));

// ---------- Step 6: average velocity ----------
export const ubar = (x, s, t) => (t - s < 1e-6 ? vel(MIX1, x, s) : (flowMap(MIX1, x, s, t) - x) / (t - s));
// Both sides of the MeanFlow identity at (x, s, t); the total derivative is a difference along the path.
export function meanflowSides(x, s, t, d = 1e-4) {
  const xs = flowMap(MIX1, x, s, s + d);
  const dds = (ubar(xs, s + d, t) - ubar(x, s, t)) / d;
  return { lhs: ubar(x, s, t), rhs: vel(MIX1, x, s) + (t - s) * dds, u: vel(MIX1, x, s), dds };
}
export const jumpsW1 = (K) => w1(Z0.map((z) => { let x = z; for (let k = 0; k < K; k++) x = flowMap(MIX1, x, k / K, (k + 1) / K); return x; }));

// ---------- Step 5: reflow on the 2D mixture ----------
export const NR5 = 4000, NSHOW5 = 400, KRK = 48, BW5 = 0.22;
const u2 = (x, y, t) => gmm2.fields(x, y, t).u;
function rk4Path(x, y, K) {
  const P = [[x, y]], h = 1 / K;
  for (let k = 0; k < K; k++) {
    const t = k * h, a = u2(x, y, t), b = u2(x + h / 2 * a[0], y + h / 2 * a[1], t + h / 2), c = u2(x + h / 2 * b[0], y + h / 2 * b[1], t + h / 2), d = u2(x + h * c[0], y + h * c[1], t + h);
    x += h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); y += h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1]); P.push([x, y]);
  }
  return P;
}
let R5 = null;
export function reflowData() {
  if (R5) return R5;
  const r = makeRng(51), x0 = new Float64Array(2 * NR5), x1 = new Float64Array(2 * NR5), paths = [];
  for (let i = 0; i < NR5; i++) {
    x0[2 * i] = r.gauss(); x0[2 * i + 1] = r.gauss();
    const P = rk4Path(x0[2 * i], x0[2 * i + 1], KRK); x1[2 * i] = P[KRK][0]; x1[2 * i + 1] = P[KRK][1];
    if (i < NSHOW5) paths.push(P);
  }
  return (R5 = { x0, x1, paths });
}
// The reflowed velocity: Nadaraya–Watson regression of x_1 - x_0 on x_t over the coupled pairs.
export function reflowVel(x, y, t, D = reflowData()) {
  let sw = 0, sx = 0, sy = 0; const h2 = 2 * BW5 * BW5;
  for (let i = 0; i < NR5; i++) {
    const ax = D.x0[2 * i], ay = D.x0[2 * i + 1], bx = D.x1[2 * i], by = D.x1[2 * i + 1];
    const dx = x - (t * bx + (1 - t) * ax), dy = y - (t * by + (1 - t) * ay), w = Math.exp(-(dx * dx + dy * dy) / h2);
    sw += w; sx += w * (bx - ax); sy += w * (by - ay);
  }
  return sw > 0 ? [sx / sw, sy / sw] : [0, 0];
}
// K Euler steps of a 2D field from the shown starts; returns end points.
export function euler2(field, K, D = reflowData()) {
  return Array.from({ length: NSHOW5 }, (_, i) => {
    let x = D.x0[2 * i], y = D.x0[2 * i + 1];
    for (let k = 0; k < K; k++) { const v = field(x, y, k / K); x += v[0] / K; y += v[1] / K; }
    return [x, y];
  });
}
// Straightness: mean over paths and time of |u(x_t, t) - (x_1 - x_0)|², for paths stored at KRK + 1 times.
export function straightness(paths) {
  let s = 0, n = 0;
  for (const P of paths) {
    const K = P.length - 1, dx = P[K][0] - P[0][0], dy = P[K][1] - P[0][1];
    for (let k = 0; k < K; k++) { const vx = (P[k + 1][0] - P[k][0]) * K, vy = (P[k + 1][1] - P[k][1]) * K; s += (vx - dx) ** 2 + (vy - dy) ** 2; n++; }
  }
  return s / n;
}
export function reflowPaths(K = 24, D = reflowData()) {
  return Array.from({ length: NSHOW5 }, (_, i) => {
    let x = D.x0[2 * i], y = D.x0[2 * i + 1]; const P = [[x, y]];
    for (let k = 0; k < K; k++) { const v = reflowVel(x, y, k / K, D); x += v[0] / K; y += v[1] / K; P.push([x, y]); }
    return P;
  });
}
// Mean distance of one-step samples to the teacher's end points for the same noise.
export const endErr = (ends, D = reflowData()) => ends.reduce((s, [x, y], i) => s + Math.hypot(x - D.x1[2 * i], y - D.x1[2 * i + 1]), 0) / ends.length;
export const teacherVel2 = u2;

// ---------- Step 7: distribution matching ----------
// The one-step generator is represented by NP7 particles g_i = G(z_i) at stratified noise z_i, each a free
// parameter. The fake score is then the exact score of the noised particle cloud, a Gaussian mixture.
export const NP7 = 400, Z7 = strat(MIX1, NP7, 0), TARGET7 = Z7.map((z) => flowMap(MIX1, z, 0, 1));
export const BOUND = [-0.65, 1.1];   // midpoints between the modes
export const TR7 = { low: [0.85, 0.98], wide: [0.2, 0.98] };
function fakeScore(g, x, t) {
  const s2 = (1 - t) ** 2; let a = 0, b = 0, best = Infinity;
  for (let i = 0; i < g.length; i++) { const d = x - t * g[i]; if (d * d < best) best = d * d; }
  for (let i = 0; i < g.length; i++) { const d = x - t * g[i], w = Math.exp(-(d * d - best) / (2 * s2)); a += w; b -= w * d / s2; }
  return b / a;
}
// Density of the particle cloud on XG, smoothed with a Gaussian of width 0.08 for display and KL.
export function cloudDensity(g, bw = 0.08) {
  const out = new Float64Array(NG), c = 1 / (g.length * bw * Math.sqrt(2 * Math.PI));
  for (const y of g) { const j0 = Math.max(0, Math.floor((y - 5 * bw - XG[0]) / DX)), j1 = Math.min(NG - 1, Math.ceil((y + 5 * bw - XG[0]) / DX)); for (let j = j0; j <= j1; j++) out[j] += c * Math.exp(-0.5 * ((XG[j] - y) / bw) ** 2); }
  return out;
}
export const modeMass = (g) => { const m = [0, 0, 0]; for (const y of g) m[y < BOUND[0] ? 0 : y < BOUND[1] ? 1 : 2]++; return m.map((k) => k / g.length); };
export const DATA_MASS = MIX1.w.slice();
export function makeDMD({ range = 'low', lam = 0, seed = 7, lr = 0.5, draws = 4 } = {}) {
  const rng = makeRng(seed), [ta, tb] = TR7[range];
  const g = Float64Array.from(Z7, (z) => 1.9 + 0.15 * z);   // narrow start on the right mode
  const S = { g, it: 0, hist: [] };
  const measure = () => {
    const p = normalize(cloudDensity(g));
    S.cur = { p, kl: klGrid(p.map((q) => Math.max(q, 1e-12)), pData), mass: modeMass(g), w1: w1Samples(Array.from(g), MIX1) };
    S.hist.push({ it: S.it, w1: S.cur.w1, mass: S.cur.mass });
  };
  measure();
  S.step = () => {
    const gs = Float64Array.from(g), dg = new Float64Array(NP7);
    for (let i = 0; i < NP7; i++) {
      for (let n = 0; n < draws; n++) {
        const t = ta + (tb - ta) * rng.uniform(), x = t * gs[i] + (1 - t) * rng.gauss();
        // DMD: (s_fake - s_real)(x_t) times dx_t/dg = t, weighted by (1 - t)^2.
        dg[i] += (fakeScore(gs, x, t) - fields(MIX1, x, t).score) * t * (1 - t) ** 2 / draws;
      }
      if (lam > 0) dg[i] += lam * 2 * (gs[i] - TARGET7[i]);
    }
    for (let i = 0; i < NP7; i++) g[i] -= lr * dg[i];
    S.it++; measure();
  };
  return S;
}
