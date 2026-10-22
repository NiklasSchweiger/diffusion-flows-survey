// Fields, a trainable velocity model and samplers of the §2.4 tutorial. No DOM access, so the numbers
// can be checked in Node. Convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data, linear
// path alpha_t = t, sigma_t = 1 - t unless a schedule is passed. All fields are exact (Gaussian mixtures).
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, MIX2, gmm2, W2, p2, LIN, SCHEDULES, T_MIN, dens1, pt1, pfVel1, sample1, quantile1, NR, TR, reverse1, quality2, LL_EXACT } from '../sde/sim.js';
import { pt2, PAIRS2, posteriorSamples } from '../score-matching/sim.js';

export { MIX1, MIX2, gmm2, W2, p2, LIN, SCHEDULES, T_MIN, dens1, pt1, pfVel1, sample1, NR, TR, reverse1, quality2, LL_EXACT, pt2, PAIRS2, posteriorSamples };

// ---------- 1D marginal velocity, eq. (marginal-velocity) ----------
// u_t(x) = alpha'_t E[x1 | x_t = x] + sigma'_t E[x0 | x_t = x], with both posterior means in closed form.
const phi = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
export function post1(x, t, S = LIN) {
  const a = S.alpha(t), s = S.sigma(t);
  let z = 0, e1 = 0, e0 = 0;
  MIX1.mu.forEach((m, k) => {
    const s2 = MIX1.sd[k] ** 2, v = a * a * s2 + s * s, r = MIX1.w[k] * phi(x, a * m, v);
    z += r; e1 += r * (m + (a * s2 / v) * (x - a * m)); e0 += r * (s / v) * (x - a * m);
  });
  return { e1: e1 / z, e0: e0 / z };
}
export function uExact1(x, t, S = LIN) { const { e1, e0 } = post1(x, t, S); return S.dalpha(t) * e1 + S.dsigma(t) * e0; }
// The two parts of the velocity-score identity, eq. (velocity-score).
export const idParts = (x, t, S = LIN) => ({ lin: (S.dalpha(t) / S.alpha(t)) * x, sc: 0.5 * S.g2(t) * dens1(x, t, S).score, ca: S.dalpha(t) / S.alpha(t), cs: 0.5 * S.g2(t) });

// ---------- Step 1: the continuity equation and divergence-free additions (2D) ----------
// dx/dt = u_t(x) + c sigma_t^2 J grad log p_t(x), J the rotation by 90 degrees. The extra flux
// p_t c sigma_t^2 J grad log p_t = c sigma_t^2 J grad p_t has zero divergence, so every c generates p_t.
// RK2 on NC uniform steps; the linear-path fields are finite on [0, 1].
export const NC = 200, NP = 1000;
// Start points: a stratified radius and golden-angle directions, so 1000 points already match N(0, I) closely.
const P0 = (() => {
  const a = new Float64Array(2 * NP), g = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < NP; i++) { const r = Math.sqrt(-2 * Math.log(1 - (i + 0.5) / NP)); a[2 * i] = r * Math.cos(i * g); a[2 * i + 1] = r * Math.sin(i * g); }
  return a;
})();
export function swirlVel(x, y, t, c) {
  const f = gmm2.fields(x, y, t), k = c * (1 - t) ** 2;
  return [f.u[0] - k * f.score[1], f.u[1] + k * f.score[0]];
}
export function swirlPaths(c) {
  const path = [Float64Array.from(P0)], h = 1 / NC;
  for (let j = 0; j < NC; j++) {
    const t = j * h, cur = path[j], nxt = new Float64Array(2 * NP);
    for (let i = 0; i < NP; i++) {
      const x = cur[2 * i], y = cur[2 * i + 1], a = swirlVel(x, y, t, c), b = swirlVel(x + h * a[0], y + h * a[1], t + h, c);
      nxt[2 * i] = x + 0.5 * h * (a[0] + b[0]); nxt[2 * i + 1] = y + 0.5 * h * (a[1] + b[1]);
    }
    path.push(nxt);
  }
  return path;
}
// Mean log p_t of a set of 2D points, and its value under exact samples of p_t (Monte Carlo, fixed draws).
export function meanLogPt(xy, t) { let s = 0; const n = xy.length / 2; for (let i = 0; i < n; i++) s += Math.log(Math.max(pt2(xy[2 * i], xy[2 * i + 1], t), 1e-300)) / n; return s; }
const EX2 = (() => { const r = makeRng(92), n = 3000, x1 = [], x0 = []; for (let i = 0; i < n; i++) { const s = gmm2.sample(r); x1.push(s.x, s.y); x0.push(r.gauss(), r.gauss()); } return { n, x1, x0 }; })();
export function exactLogPt(t) {
  const { n, x1, x0 } = EX2, xy = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) xy[i] = t * x1[i] + (1 - t) * x0[i];
  return meanLogPt(xy, t);
}

// ---------- Step 4: flow matching with a model that is linear in its parameters (1D) ----------
// u_theta(x, t) = sum_k hat_k(t) (sum_i theta_ki rbf_i(x) + a_k x + b_k): NB Gaussian bumps in x and
// x, 1 for the tails, each multiplied by NK hat functions in t. Only two hats are nonzero at any t.
export const NB = 25, NK = 11, XC = Float64Array.from({ length: NB }, (_, i) => -4.5 + 9 * i / (NB - 1)), RW = 0.42;
const NF = NB + 2, NPAR = NK * NF;
function feats(x, t, out) {
  const u = Math.min(Math.max(t, 0), 1) * (NK - 1), k = Math.min(NK - 2, Math.floor(u)), w = u - k;
  for (let i = 0; i < NB; i++) out[i] = Math.exp(-((x - XC[i]) ** 2) / (2 * RW * RW));
  out[NB] = x; out[NB + 1] = 1;
  return [k, w];
}
const FB = new Float64Array(NF);
export function uModel(th, x, t) {
  const [k, w] = feats(x, t, FB);
  let a = 0, b = 0; const o0 = k * NF, o1 = o0 + NF;
  for (let i = 0; i < NF; i++) { a += th[o0 + i] * FB[i]; b += th[o1 + i] * FB[i]; }
  return (1 - w) * a + w * b;
}
// Training triples (t, x0, x1) come from a seeded stream, so a run is reproducible from its step count.
export const BATCH = 128;
export function makeTrainer(seed = 101) {
  const th = new Float64Array(NPAR), m = new Float64Array(NPAR), v = new Float64Array(NPAR), g = new Float64Array(NPAR);
  const st = { th, step: 0, rng: makeRng(seed), hist: [] };
  const b1 = 0.9, b2 = 0.999;
  // The labels are noisy (the floor below is about 2.3), so the step size decays to average the noise out.
  const lr = (k) => 0.03 / Math.sqrt(1 + k / 100);
  st.train = (n) => {
    for (let s = 0; s < n; s++) {
      g.fill(0); let loss = 0;
      for (let q = 0; q < BATCH; q++) {
        const t = st.rng.uniform(), x0 = st.rng.gauss(), x1 = sample1(st.rng), x = t * x1 + (1 - t) * x0, y = x1 - x0;
        const [k, w] = feats(x, t, FB), o0 = k * NF, o1 = o0 + NF;
        let a = 0, b = 0; for (let i = 0; i < NF; i++) { a += th[o0 + i] * FB[i]; b += th[o1 + i] * FB[i]; }
        const r = (1 - w) * a + w * b - y; loss += r * r / BATCH;
        for (let i = 0; i < NF; i++) { g[o0 + i] += 2 * r * (1 - w) * FB[i] / BATCH; g[o1 + i] += 2 * r * w * FB[i] / BATCH; }
      }
      st.step++;
      const c1 = 1 - b1 ** st.step, c2 = 1 - b2 ** st.step;
      for (let j = 0; j < NPAR; j++) { m[j] = b1 * m[j] + (1 - b1) * g[j]; v[j] = b2 * v[j] + (1 - b2) * g[j] * g[j]; th[j] -= lr(st.step) * (m[j] / c1) / (Math.sqrt(v[j] / c2) + 1e-8); }
    }
  };
  return st;
}
// Fixed evaluation set: the flow-matching loss of a model, and the part of it no model can remove,
// E || label - u_t(x_t) ||^2. The difference is E || u_theta - u ||^2.
const EVAL = (() => {
  const r = makeRng(102), n = 4000, t = new Float64Array(n), x = new Float64Array(n), y = new Float64Array(n), u = new Float64Array(n);
  for (let i = 0; i < n; i++) { t[i] = r.uniform(); const x0 = r.gauss(), x1 = sample1(r); x[i] = t[i] * x1 + (1 - t[i]) * x0; y[i] = x1 - x0; u[i] = uExact1(x[i], t[i]); }
  let floor = 0; for (let i = 0; i < n; i++) floor += (y[i] - u[i]) ** 2 / n;
  return { n, t, x, y, u, floor };
})();
export const LOSS_FLOOR = EVAL.floor;
export function evalLoss(th) {
  let loss = 0, gap = 0; const n = EVAL.n;
  for (let i = 0; i < n; i++) { const p = uModel(th, EVAL.x[i], EVAL.t[i]); loss += (p - EVAL.y[i]) ** 2 / n; gap += (p - EVAL.u[i]) ** 2 / n; }
  return { loss, gap };
}
// A few labels near time t for the scatter plot: fixed (x0, x1) draws placed at x_t.
const LAB = (() => { const r = makeRng(103), n = 400; return { n, x0: Float64Array.from({ length: n }, () => r.gauss()), x1: Float64Array.from({ length: n }, () => sample1(r)) }; })();
export function labelsAt(t) { const out = []; for (let i = 0; i < LAB.n; i++) out.push([t * LAB.x1[i] + (1 - t) * LAB.x0[i], LAB.x1[i] - LAB.x0[i]]); return out; }
// Samples of a velocity field: Heun steps from stratified quantiles of N(0, 1) at t = 0 to t = 1.
const Z0 = Float64Array.from({ length: 1000 }, (_, i) => quantile1((i + 0.5) / 1000, 0));
export function sampleField(vel, K = 40) {
  const out = new Float64Array(Z0.length), h = 1 / K;
  for (let i = 0; i < Z0.length; i++) {
    let x = Z0[i];
    for (let j = 0; j < K; j++) { const t = j * h, a = vel(x, t), b = vel(x + h * a, t + h); x += 0.5 * h * (a + b); }
    out[i] = x;
  }
  return out;
}
export const meanLogP1 = (xs) => xs.reduce((a, x) => a + Math.log(Math.max(pt1(x, 1), 1e-300)), 0) / xs.length;
export const LL1_EXACT = (() => { const r = makeRng(104); let s = 0; const n = 20000; for (let i = 0; i < n; i++) s += Math.log(pt1(sample1(r), 1)) / n; return s; })();

// ---------- Step 6: the family of samplers, eq. (sde-family) ----------
// reverse1(eta) from §2.3 integrates dx = (u + 1/2 eps^2 score) dt + eps dW with eps_t^2 = eta^2 sigma_t^2 lambda'_t.
// Its particles start at stratified quantiles, so particle i starts at rank i. Spearman correlation of start and end.
export function rankCorr(x) {
  const n = x.n, end = Array.from({ length: n }, (_, i) => [x.x[i][NR], i]).sort((a, b) => a[0] - b[0]), rk = new Float64Array(n);
  end.forEach(([, i], r) => { rk[i] = r; });
  const m = (n - 1) / 2; let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - m) * (rk[i] - m); den += (i - m) ** 2; }
  return num / den;
}

// ---------- Step 7: straight lines, curved paths and few-step sampling (2D) ----------
// Plain Euler with K uniform steps on [0, 1] and the exact marginal velocity; the start points are those of step 1.
export function euler2(K) {
  const path = [Float64Array.from(P0)], h = 1 / K;
  for (let j = 0; j < K; j++) {
    const t = j * h, cur = path[j], nxt = new Float64Array(2 * NP);
    for (let i = 0; i < NP; i++) { const u = gmm2.fields(cur[2 * i], cur[2 * i + 1], t).u; nxt[2 * i] = cur[2 * i] + h * u[0]; nxt[2 * i + 1] = cur[2 * i + 1] + h * u[1]; }
    path.push(nxt);
  }
  return path;
}
// Straightness of a path set: chord length |x_1 - x_0| divided by arc length, averaged over paths. It is 1 for straight paths.
export function straightness(path) {
  const K = path.length - 1, n = path[0].length / 2; let s = 0;
  for (let i = 0; i < n; i++) {
    let arc = 0; for (let j = 0; j < K; j++) arc += Math.hypot(path[j + 1][2 * i] - path[j][2 * i], path[j + 1][2 * i + 1] - path[j][2 * i + 1]);
    s += Math.hypot(path[K][2 * i] - path[0][2 * i], path[K][2 * i + 1] - path[0][2 * i + 1]) / Math.max(arc, 1e-12) / n;
  }
  return s;
}
export const DATA_MEAN2 = MIX2.means.reduce((a, [mx, my], k) => [a[0] + W2[k] * mx, a[1] + W2[k] * my], [0, 0]);
