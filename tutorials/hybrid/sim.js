// Numbers of the §2.10 tutorial. No DOM access, so they can be checked in Node. Convention: t = 0 is noise,
// t = 1 is data, x_t = alpha_t x1 + sigma_t x0 with alpha_t = t and sigma_t = 1 - t. Nothing here is trained.
// The data are an isotropic Gaussian mixture over 12 x 12 images whose means are the eight digits of §2.7,
// so every conditional, every block-average map and every noised marginal has a closed form.
import { makeRng } from '../../animations/lib/rng.js';
import { GMM } from '../../animations/lib/gmm.js';
import { DIGITS, N_IMG } from '../generalization/sim.js';

export { DIGITS, N_IMG };
export const D = N_IMG * N_IMG, K = DIGITS.length, S0 = 0.3, V0 = S0 * S0;
const LOGW = -Math.log(K), L2PI = Math.log(2 * Math.PI);
const logN = (x, m, v) => -0.5 * ((x - m) ** 2 / v + L2PI + Math.log(v));
function pick(w, u) {
  let c = 0;
  for (let i = 0; i < w.length; i++) { c += w[i]; if (u < c) return i; }
  return w.length - 1;
}
// Normalized posterior from log weights.
function norm(lw) {
  const m = Math.max(...lw), e = lw.map((l) => Math.exp(l - m)), z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}
function logsumexp(a) { const m = Math.max(...a); return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); }

// Log likelihood of x under each component, and the exact log density of the mixture.
export function compLog(x) {
  return DIGITS.map((mu) => { let l = 0; for (let i = 0; i < D; i++) l += logN(x[i], mu[i], V0); return l; });
}
export const logp = (x) => logsumexp(compLog(x).map((l) => l + LOGW));
export const posterior = (x) => norm(compLog(x));
export function sampleData(r) {
  const k = Math.floor(r.uniform() * K), x = new Float64Array(D);
  for (let i = 0; i < D; i++) x[i] = DIGITS[k][i] + S0 * r.gauss();
  return { x, k };
}

// ---------- Typicality test ----------
// A sample is typical when its exact log density lies between the 1% and 99% quantiles of real draws.
export const QLO_HI = (() => {
  const r = makeRng(1701), M = 20000, l = new Float64Array(M);
  for (let i = 0; i < M; i++) l[i] = logp(sampleData(r).x);
  l.sort();
  return [l[Math.floor(0.01 * M)], l[Math.floor(0.99 * M)], l[Math.floor(0.5 * M)]];
})();
export const typ = (x) => { const l = logp(x); return l < QLO_HI[0] ? -1 : l > QLO_HI[1] ? 1 : 0; };
export function typStats(xs) {
  let lo = 0, hi = 0;
  for (const x of xs) { const c = typ(x); if (c < 0) lo++; else if (c > 0) hi++; }
  const n = xs.length;
  return { typical: (n - lo - hi) / n, low: lo / n, high: hi / n };
}

// ---------- Step 1: the chain rule in any order ----------
// One image generated pixel by pixel. Before each pixel the posterior over the digits is that of the mixture
// given the visible pixels; the pixel is drawn from sum_k post_k N(mu_k[i], s^2).
export function arRun(order, seed) {
  const r = makeRng(seed), ord = Array.from({ length: D }, (_, i) => i);
  if (order === 'random') for (let i = D - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
  const x = new Float64Array(D), lw = Array(K).fill(0), post = [], logc = [];
  for (const i of ord) {
    const pi = norm(lw);
    post.push(pi);
    const k = pick(pi, r.uniform()), v = DIGITS[k][i] + S0 * r.gauss();
    x[i] = v;
    logc.push(Math.log(pi.reduce((s, p, j) => s + p * Math.exp(logN(v, DIGITS[j][i], V0)), 0)));
    for (let j = 0; j < K; j++) lw[j] += logN(v, DIGITS[j][i], V0);
  }
  post.push(norm(lw));
  return { ord, x, post, logc, logp: logp(x) };
}
export const RUN1 = { raster: arRun('raster', 1101), random: arRun('random', 1102) };
// The sum of log conditionals of an image in a given order (to show it does not depend on the order).
export function scoreOrder(x, ord) {
  const lw = Array(K).fill(0);
  let s = 0;
  for (const i of ord) {
    const pi = norm(lw);
    s += Math.log(pi.reduce((a, p, j) => a + p * Math.exp(logN(x[i], DIGITS[j][i], V0)), 0));
    for (let j = 0; j < K; j++) lw[j] += logN(x[i], DIGITS[j][i], V0);
  }
  return s;
}

// ---------- Step 2: next-scale prediction ----------
// The map at scale r holds the averages of (12/r)^2 pixels. Under component k the block averages are
// independent N(mean of mu_k over the block, s^2/n). Given the parent average, the J children of a parent are
// iid Gaussians conditioned on their mean: child = mu_child + (parent - mean of mu_children) + (e - mean e).
export function blockMap(im, r) {
  const b = N_IMG / r, out = new Float64Array(r * r);
  for (let j = 0; j < N_IMG; j++) for (let i = 0; i < N_IMG; i++) out[Math.floor(j / b) * r + Math.floor(i / b)] += im[j * N_IMG + i] / (b * b);
  return out;
}
const MAPS = new Map();
const muMap = (k, r) => { const key = k * 100 + r; if (!MAPS.has(key)) MAPS.set(key, blockMap(DIGITS[k], r)); return MAPS.get(key); };
// One run. scales: increasing divisors of 12 ending in 12. joint: one component per scale (exact);
// otherwise every token draws its own component from the posterior (a parallel per-token head).
export function nextScale(scales, joint, r) {
  const maps = [];
  let cur = null, rc = 0;
  for (const rs of scales) {
    const n = (N_IMG / rs) ** 2, v = V0 / n, out = new Float64Array(rs * rs);
    let pi;
    if (!cur) pi = Array(K).fill(1 / K);
    else {
      const nc = (N_IMG / rc) ** 2;
      pi = norm(DIGITS.map((_, k) => { const m = muMap(k, rc); let l = 0; for (let b = 0; b < cur.length; b++) l += logN(cur[b], m[b], V0 / nc); return l; }));
    }
    const kj = pick(pi, r.uniform()), f = cur ? rs / rc : 0, J = f * f;
    // Sum of mu over the children of each parent, per component, for the residual.
    const pm = cur ? DIGITS.map((_, k) => { const m = muMap(k, rs), a = new Float64Array(rc * rc); for (let q = 0; q < rs * rs; q++) a[Math.floor(Math.floor(q / rs) / f) * rc + Math.floor((q % rs) / f)] += m[q] / J; return a; }) : null;
    if (joint) {
      const e = Array.from({ length: rs * rs }, () => Math.sqrt(v) * r.gauss()), m = muMap(kj, rs);
      const eb = cur ? new Float64Array(rc * rc) : null;
      if (cur) for (let q = 0; q < rs * rs; q++) eb[Math.floor(Math.floor(q / rs) / f) * rc + Math.floor((q % rs) / f)] += e[q] / J;
      for (let q = 0; q < rs * rs; q++) {
        if (!cur) { out[q] = m[q] + e[q]; continue; }
        const p = Math.floor(Math.floor(q / rs) / f) * rc + Math.floor((q % rs) / f);
        out[q] = m[q] + cur[p] - pm[kj][p] + e[q] - eb[p];
      }
    } else {
      const sd = Math.sqrt(cur ? v * (1 - 1 / J) : v);
      for (let q = 0; q < rs * rs; q++) {
        const k = pick(pi, r.uniform()), m = muMap(k, rs);
        if (!cur) { out[q] = m[q] + sd * r.gauss(); continue; }
        const p = Math.floor(Math.floor(q / rs) / f) * rc + Math.floor((q % rs) / f);
        out[q] = m[q] + cur[p] - pm[k][p] + sd * r.gauss();
      }
    }
    maps.push({ r: rs, map: out, pi });
    cur = out; rc = rs;
  }
  return { maps, x: cur };
}
export const SCALES2 = [1, 3, 6, 12], NS2 = 400;
const memo = (f) => { const c = new Map(); return (...a) => { const k = JSON.stringify(a); if (!c.has(k)) c.set(k, f(...a)); return c.get(k); }; };
export const runs2 = memo((joint) => { const r = makeRng(joint ? 1201 : 1202); return Array.from({ length: NS2 }, () => nextScale(SCALES2, joint, r)); });
export const stats2 = memo((joint) => typStats(runs2(joint).map((q) => q.x)));

// ---------- Step 3: one axis, steps against dependence ----------
// Autoregressive sampling with k pixels per step: each pixel of a step draws its own component from the
// posterior given the visible pixels, so the pixels of one step are independent given the visible ones.
export function arParallel(order, k, r) {
  const ord = Array.from({ length: D }, (_, i) => i);
  if (order === 'random') for (let i = D - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
  const x = new Float64Array(D), lw = Array(K).fill(0);
  for (let a = 0; a < D; a += k) {
    const pi = norm(lw), grp = ord.slice(a, a + k);
    for (const i of grp) { const c = pick(pi, r.uniform()); x[i] = DIGITS[c][i] + S0 * r.gauss(); }
    for (const i of grp) for (let j = 0; j < K; j++) lw[j] += logN(x[i], DIGITS[j][i], V0);
  }
  return x;
}
// Euler on the exact probability-flow ODE with N uniform steps from t = 0. Under component k,
// x_t ~ N(t mu_k, v I) with v = t^2 s^2 + (1 - t)^2, and E[x1 | x_t, k] = mu_k + (t s^2 / v)(x_t - t mu_k).
export function denoise(x, t, out) {
  const v = t * t * V0 + (1 - t) ** 2;
  const lw = DIGITS.map((mu) => { let l = 0; for (let i = 0; i < D; i++) l -= (x[i] - t * mu[i]) ** 2; return l / (2 * v); });
  const g = norm(lw);
  out.fill(0);
  for (let k = 0; k < K; k++) if (g[k] > 1e-12) for (let i = 0; i < D; i++) out[i] += g[k] * (DIGITS[k][i] + (t * V0 / v) * (x[i] - t * DIGITS[k][i]));
  return out;
}
export function flowEuler(N, r) {
  const x = Float64Array.from({ length: D }, () => r.gauss()), m = new Float64Array(D);
  for (let n = 0; n < N; n++) {
    const t = n / N, h = 1 / N;
    denoise(x, t, m);
    for (let i = 0; i < D; i++) x[i] += h * (m[i] - x[i]) / (1 - t);
  }
  return x;
}
export const K3_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 144];
export const SCALES3 = [[12], [1, 12], [1, 3, 12], [1, 3, 6, 12]];
export const N3_LIST = [1, 2, 4, 8, 16, 32, 64, 128];
export const SAMPLERS3 = ['raster', 'random', 'scale', 'flow'];
export const NS3 = 300;
// Settings of a sampler as [{steps, label, gen(r)}].
export function settings3(s) {
  if (s === 'raster' || s === 'random') return K3_LIST.map((k) => ({ steps: Math.ceil(D / k), label: `k = ${k}`, gen: (r) => arParallel(s, k, r) }));
  if (s === 'scale') return SCALES3.map((sc) => ({ steps: sc.length, label: `scales ${sc.join(', ')}`, gen: (r) => nextScale(sc, false, r).x }));
  return N3_LIST.map((N) => ({ steps: N, label: `N = ${N}`, gen: (r) => flowEuler(N, r) }));
}
export const runs3 = memo((s, j) => { const r = makeRng(1300 + 40 * SAMPLERS3.indexOf(s) + j), g = settings3(s)[j].gen; return Array.from({ length: NS3 }, () => g(r)); });
export const stats3 = memo((s, j) => ({ steps: settings3(s)[j].steps, ...typStats(runs3(s, j)) }));

// ---------- Step 4: discrete tokens lose information ----------
// 2 x 2 patches (d = 4) quantized by a k-means codebook trained on patches of 400 seeded images.
export const P4 = 2, DP = P4 * P4, NT4 = (N_IMG / P4) ** 2;
export function patches(x) {
  const out = [];
  for (let pj = 0; pj < N_IMG; pj += P4) for (let pi = 0; pi < N_IMG; pi += P4) {
    const v = new Float64Array(DP);
    for (let b = 0; b < P4; b++) for (let a = 0; a < P4; a++) v[b * P4 + a] = x[(pj + b) * N_IMG + pi + a];
    out.push(v);
  }
  return out;
}
export function unpatch(ps) {
  const x = new Float64Array(D), n = N_IMG / P4;
  ps.forEach((v, t) => { const pj = Math.floor(t / n) * P4, pi = (t % n) * P4; for (let b = 0; b < P4; b++) for (let a = 0; a < P4; a++) x[(pj + b) * N_IMG + pi + a] = v[b * P4 + a]; });
  return x;
}
const TRAIN4 = (() => { const r = makeRng(1401), P = []; for (let i = 0; i < 400; i++) P.push(...patches(sampleData(r).x)); return P; })();
export const TEST4 = (() => { const r = makeRng(1402); return Array.from({ length: 200 }, () => sampleData(r).x); })();
const d2 = (a, b) => { let s = 0; for (let i = 0; i < DP; i++) s += (a[i] - b[i]) ** 2; return s; };
function nearestCode(C, v) { let bi = 0, bd = Infinity; for (let c = 0; c < C.length; c++) { const d = d2(C[c], v); if (d < bd) { bd = d; bi = c; } } return bi; }
export const K4_LIST = [2, 4, 8, 16, 32, 64, 128, 256];
export const codebook = memo((Kc) => {
  const r = makeRng(1410 + Kc), P = TRAIN4, n = P.length, C = [P[Math.floor(r.uniform() * n)]];
  const dist = new Float64Array(n).fill(Infinity);
  while (C.length < Kc) {
    let tot = 0;
    for (let i = 0; i < n; i++) { dist[i] = Math.min(dist[i], d2(P[i], C[C.length - 1])); tot += dist[i]; }
    let u = r.uniform() * tot, i = 0;
    while (i < n - 1 && (u -= dist[i]) > 0) i++;
    C.push(P[i]);
  }
  let cb = C.map((c) => Float64Array.from(c));
  for (let it = 0; it < 15; it++) {
    const S = cb.map(() => new Float64Array(DP)), cnt = new Float64Array(Kc);
    for (const v of P) { const c = nearestCode(cb, v); cnt[c]++; for (let a = 0; a < DP; a++) S[c][a] += v[a]; }
    cb = cb.map((c, j) => (cnt[j] ? S[j].map((s) => s / cnt[j]) : c));
  }
  return cb;
});
export const quantize = (x, Kc) => { const C = codebook(Kc); return unpatch(patches(x).map((v) => C[nearestCode(C, v)])); };
export const stats4 = memo((Kc) => {
  let e = 0;
  for (const x of TEST4) { const q = quantize(x, Kc); for (let i = 0; i < D; i++) e += (q[i] - x[i]) ** 2; }
  return { mse: e / (TEST4.length * D), bits: NT4 * Math.log2(Kc) };
});

// ---------- Step 5: a diffusion head for one continuous token ----------
// Tokens are horizontal pairs of pixels, generated in raster order. Given the tokens before it, token j is
// distributed as the 2D mixture sum_k post_k N((mu_k[a], mu_k[b]), s^2 I).
export const NT5 = D / 2;
export const tokPix = (j) => [2 * j, 2 * j + 1];
export function tokenGMM(pi, j) {
  const [a, b] = tokPix(j);
  return new GMM(DIGITS.map((mu) => [mu[a], mu[b]]), S0, { weights: pi.map((p) => Math.max(p, 1e-300)) });
}
// Probability-flow path of the 2D head from x0 ~ N(0, I), RK4 on a uniform grid of t.
export function headPath(g, x0, y0, n = 40) {
  const P = [[x0, y0]];
  let x = x0, y = y0;
  const u = (x, y, t) => { const f = g.fields(x, y, Math.min(t, 1 - 1e-6)).u; return f; };
  for (let i = 0; i < n; i++) {
    const t = i / n, h = 1 / n;
    const k1 = u(x, y, t), k2 = u(x + h / 2 * k1[0], y + h / 2 * k1[1], t + h / 2), k3 = u(x + h / 2 * k2[0], y + h / 2 * k2[1], t + h / 2), k4 = u(x + h * k3[0], y + h * k3[1], t + h);
    x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    P.push([x, y]);
  }
  return P;
}
// One image generated token by token in a seeded random order (as in MAR), with the diffusion head
// (head = 'diff') or with a regression head, which predicts the conditional mean (head = 'mse').
export function marRun(head, seed) {
  const r = makeRng(seed), x = new Float64Array(D), lw = Array(K).fill(0), ord = Array.from({ length: NT5 }, (_, j) => j);
  for (let i = NT5 - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
  for (const j of ord) {
    const pi = norm(lw), [a, b] = tokPix(j);
    if (head === 'mse') { x[a] = 0; x[b] = 0; for (let k = 0; k < K; k++) { x[a] += pi[k] * DIGITS[k][a]; x[b] += pi[k] * DIGITS[k][b]; } }
    else { const P = headPath(tokenGMM(pi, j), r.gauss(), r.gauss(), 24); [x[a], x[b]] = P[P.length - 1]; }
    for (const i of [a, b]) for (let k = 0; k < K; k++) lw[k] += logN(x[i], DIGITS[k][i], V0);
  }
  return { x, ord };
}
export const RUN5 = marRun('diff', 1501);
export const IMGS5 = memo((head) => Array.from({ length: 4 }, (_, i) => marRun(head, 1510 + i).x));
// Token j of RUN5 given its first m tokens: the posterior, the 2D mixture, the regression prediction and its
// expected squared error, and probability-flow paths of the diffusion head from 12 seeded noise points.
export const field5 = memo((j, m) => {
  const lw = Array(K).fill(0);
  for (const q of RUN5.ord.slice(0, m)) for (const i of tokPix(q)) for (let k = 0; k < K; k++) lw[k] += logN(RUN5.x[i], DIGITS[k][i], V0);
  const pi = norm(lw), g = tokenGMM(pi, j), r = makeRng(1520 + j), [a, b] = tokPix(j);
  const mean = [0, 0];
  for (let k = 0; k < K; k++) { mean[0] += pi[k] * DIGITS[k][a]; mean[1] += pi[k] * DIGITS[k][b]; }
  let mse = 0;
  for (let k = 0; k < K; k++) mse += pi[k] * ((DIGITS[k][a] - mean[0]) ** 2 + (DIGITS[k][b] - mean[1]) ** 2 + 2 * V0);
  const paths = Array.from({ length: 12 }, () => headPath(g, r.gauss(), r.gauss()));
  return { g, pi, mean, mse, paths, visible: RUN5.ord.slice(0, m).includes(j), value: [RUN5.x[a], RUN5.x[b]] };
});
