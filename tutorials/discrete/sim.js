// Numbers of the §2.9 tutorial. No DOM access, so they can be checked in Node. Convention: the corruption
// runs from data at t = 1 to noise at t = 0, and the generative chain runs forward in t. Nothing here is
// trained: every distribution is a closed form or an exact enumeration, and random draws use fixed seeds.
import { makeRng } from '../../animations/lib/rng.js';

const sum = (a) => a.reduce((s, v) => s + v, 0);
function pick(w, u) {
  let c = 0;
  for (let i = 0; i < w.length; i++) { c += w[i]; if (u < c) return i; }
  return w.length - 1;
}

// ---------- Step 1: a Markov chain on a finite set ----------
// Five words with probabilities P1 and a mask symbol. Going down from t = 1, a token is kept until a time
// U ~ U[0, 1] and then replaced, by a uniform draw over the five words or by [M]. So the corruption rate is
// 1/t, P(kept at t) = t, and the marginals are p_t = t p1 + (1 - t)/K (uniform) or t p1 and 1 - t for [M].
export const WORDS1 = ['cat', 'dog', 'owl', 'cow', 'fox'], K1 = 5, M1 = 5;
export const P1 = [0.35, 0.25, 0.2, 0.12, 0.08];
// Marginal at time t over the five words and [M] (index 5).
export function marg1(kind, t) {
  if (kind === 'uniform') return [...P1.map((p) => t * p + (1 - t) / K1), 0];
  return [...P1.map((p) => t * p), 1 - t];
}
// Corruption rate Q_t(y, x) of jumping from y to x as t decreases.
export function Q1(kind, t, y, x) {
  if (x === y) return 0;
  if (kind === 'uniform') return x < K1 && y < K1 ? 1 / (t * K1) : 0;
  return x === M1 && y < K1 ? 1 / t : 0;
}
// Generative rate R_t(x, y) = Q_t(y, x) p_t(y)/p_t(x).
export function R1(kind, t, x, y) {
  const p = marg1(kind, t);
  return p[x] > 0 ? Q1(kind, t, y, x) * p[y] / p[x] : 0;
}
// Inflow to y from every other state and outflow from y, as in the Kolmogorov forward equation.
export function flows1(kind, t, y) {
  const p = marg1(kind, t), inflow = [], n = K1 + 1;
  let out = 0;
  for (let x = 0; x < n; x++) {
    inflow.push(x === y ? 0 : p[x] * R1(kind, t, x, y));
    if (x !== y) out += p[y] * R1(kind, t, y, x);
  }
  const e = 1e-4, dpdt = (marg1(kind, t + e)[y] - marg1(kind, t - e)[y]) / (2 * e);
  return { inflow, inTot: sum(inflow), out, dpdt };
}
// 400 seeded chains: a clean word, a replacement time U and a uniform replacement word.
export const N1 = 400;
const CH1 = (() => {
  const r = makeRng(9101), x1 = [], u = [], z = [];
  for (let i = 0; i < N1; i++) { x1.push(pick(P1, r.uniform())); u.push(r.uniform()); z.push(Math.floor(r.uniform() * K1)); }
  return { x1, u, z };
})();
export const chain1 = (kind, t, i) => (t >= CH1.u[i] ? CH1.x1[i] : kind === 'uniform' ? CH1.z[i] : M1);
export function hist1(kind, t) {
  const h = Array(K1 + 1).fill(0);
  for (let i = 0; i < N1; i++) h[chain1(kind, t, i)] += 1 / N1;
  return h;
}

// ---------- Step 2: time reversal on a lattice ----------
// 32 states with a two-bump p_data (a discretized smooth density g) and the uniform kernel.
export const K2 = 32;
const BUMPS = [[0.6, 9, 2.6], [0.4, 22, 3.2]];
const gs = (u) => sum(BUMPS.map(([w, m, s]) => w * Math.exp(-0.5 * ((u - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))));
const gd = (u) => sum(BUMPS.map(([w, m, s]) => -w * (u - m) / (s * s) * Math.exp(-0.5 * ((u - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))));
const Z2 = sum(Array.from({ length: K2 }, (_, x) => gs(x)));
export const P2 = Array.from({ length: K2 }, (_, x) => gs(x) / Z2);
export const marg2 = (t) => P2.map((p) => t * p + (1 - t) / K2);
// The smooth density behind the lattice, t g/Z + (1 - t)/K, and its log-derivative.
export const smooth2 = (t, u) => t * gs(u) / Z2 + (1 - t) / K2;
export const dlog2 = (t, u) => t * gd(u) / Z2 / smooth2(t, u);
// Concrete score towards the right neighbour, p_t(x + 1)/p_t(x).
export const ratio2 = (t, x) => { const p = marg2(t); return p[x + 1] / p[x]; };
// 400 chains run with the generative rates R_t(x, y) = p_t(y)/(t K p_t(x)) from p at t = 0.001, recorded at
// t = j/100. Each substep jumps with probability 1 - exp(-rate dt) to y != x drawn proportionally to p_t(y).
export const N2 = 400, T2 = 100;
export const CH2 = (() => {
  const r = makeRng(9201), t0 = 1e-3, grid = [];
  for (let i = 0; i <= 200; i++) grid.push(t0 * Math.pow(10, i / 100));
  for (let i = 1; i <= 2000; i++) grid.push(0.01 + 0.99 * i / 2000);
  const p0 = marg2(t0), x = Array.from({ length: N2 }, () => pick(p0, r.uniform()));
  const rec = [x.slice()];
  let next = 1;
  for (let k = 1; k < grid.length; k++) {
    const ta = grid[k - 1], tb = grid[k], tm = 0.5 * (ta + tb), dt = tb - ta, p = marg2(tm);
    for (let i = 0; i < N2; i++) {
      const rate = (1 - p[x[i]]) / (tm * K2 * p[x[i]]);
      const u = r.uniform(), v = r.uniform();
      if (u < 1 - Math.exp(-rate * dt)) {
        const w = p.map((q, y) => (y === x[i] ? 0 : q)), s = sum(w);
        x[i] = pick(w.map((q) => q / s), v);
      }
    }
    while (next <= T2 && tb >= next / T2 - 1e-9) { rec.push(x.slice()); next++; }
  }
  return rec;
})();
export function hist2(j) {
  const h = Array(K2).fill(0);
  for (const x of CH2[j]) h[x] += 1 / N2;
  return h;
}

// ---------- Step 3: score entropy ----------
// For a state x and its neighbour y = x + 1, the conditional ratio r(x1) = p_t(y | x1)/p_t(x | x1) under the
// uniform kernel, p_t(z | x1) = t 1{z = x1} + (1 - t)/K, and the posterior p(x1 | x) = p1(x1) p_t(x | x1)/p_t(x).
export const cond2 = (t, z, x1) => t * (z === x1 ? 1 : 0) + (1 - t) / K2;
export function entropy3(t, x) {
  const y = x + 1, pt = marg2(t);
  const post = P2.map((p, a) => p * cond2(t, x, a) / pt[x]);
  const r = P2.map((_, a) => cond2(t, y, a) / cond2(t, x, a));
  const target = sum(post.map((q, a) => q * r[a]));
  // Expected loss E[s - r log s] over the posterior, and its minimum.
  const loss = (s) => s - target * Math.log(s);
  return { post, r, target, marginal: pt[y] / pt[x], loss, minLoss: loss(target) };
}

// ---------- The toy language of Animation A4 (copied, not imported, so this file stands alone) ----------
// 64 eight-token sentences with three coupled pairs of positions and known probabilities.
export const WTS = [0.4, 0.3, 0.2, 0.1];
export const ANIMAL = [['cat', 'meows'], ['dog', 'barks'], ['owl', 'hoots'], ['cow', 'moos']];
export const CITY = [['new', 'york'], ['san', 'francisco'], ['los', 'angeles'], ['hong', 'kong']];
export const TIME = [['at', 'night'], ['in', 'spring'], ['on', 'sunday'], ['by', 'noon']];
export const L = 8, PAIRS = [[1, 5], [3, 4], [6, 7]], GROUP = [-1, 0, -1, 1, 1, 0, 2, 2];
export const SENT = [], PS = [];
ANIMAL.forEach((a, i) => CITY.forEach((c, j) => TIME.forEach((m, k) => {
  SENT.push(['the', a[0], 'from', c[0], c[1], a[1], m[0], m[1]]);
  PS.push(WTS[i] * WTS[(j + 1) % 4] * WTS[(k + 2) % 4]);
})));
export const VOCAB = [...new Set(SENT.flat())];
export const V = VOCAB.length;
export const H_LANG = -sum(PS.map((p) => p * Math.log2(p)));
export const isValid = (x) => SENT.some((s) => s.every((w, i) => w === x[i]));
export const pairOk = (x, a, b) => SENT.some((s) => s[a] === x[a] && s[b] === x[b]);
export const probOf = (x) => { const j = SENT.findIndex((s) => s.every((w, i) => w === x[i])); return j < 0 ? 0 : PS[j]; };
// Exact conditional marginals of every position given the visible tokens (null = [M]), as maps word -> p.
// A sequence that no sentence matches can never become valid; the page then conditions on the sentences that
// agree with the most visible tokens.
export function marginals(x) {
  let ok = SENT.map((s) => s.every((w, i) => x[i] === null || x[i] === w));
  if (!ok.some(Boolean)) {
    const m = SENT.map((s) => s.reduce((a, w, i) => a + (x[i] === w ? 1 : 0), 0)), mx = Math.max(...m);
    ok = m.map((v) => v === mx);
  }
  const M = Array.from({ length: L }, () => new Map());
  let z = 0;
  SENT.forEach((s, j) => {
    if (!ok[j]) return;
    z += PS[j];
    s.forEach((w, i) => M[i].set(w, (M[i].get(w) || 0) + PS[j]));
  });
  for (const m of M) for (const [k, v] of m) m.set(k, v / z);
  return M;
}
function drawMap(m, u) {
  let c = 0, last;
  for (const [w, p] of m) { c += p; last = [w, p]; if (u < c) return last; }
  return last;
}

// ---------- Step 4: masked diffusion and its ELBO ----------
// The three sentences of the buttons: the most probable, a middle one and a rare one.
export const SENT4 = [
  ['the', 'cat', 'from', 'hong', 'kong', 'meows', 'on', 'sunday'],
  ['the', 'dog', 'from', 'new', 'york', 'barks', 'at', 'night'],
  ['the', 'cow', 'from', 'los', 'angeles', 'moos', 'in', 'spring'],
];
// Predictor at temperature tau: the exact conditional raised to 1/tau and renormalized.
function predLogp(M, i, w, tau) {
  let z = 0;
  for (const [, p] of M[i]) z += Math.pow(p, 1 / tau);
  return Math.log(Math.pow(M[i].get(w) || 0, 1 / tau) / z);
}
// C(S) = sum over masked i of -log p_theta(x^i | visible tokens) for each of the 2^8 mask sets S, grouped by |S|.
export function maskCosts(x, tau) {
  const S = Array(L + 1).fill(0), cost = new Float64Array(1 << L);
  for (let b = 1; b < 1 << L; b++) {
    const v = x.map((w, i) => (b >> i & 1 ? null : w)), M = marginals(v);
    let c = 0, k = 0;
    for (let i = 0; i < L; i++) if (b >> i & 1) { c -= predLogp(M, i, x[i], tau); k++; }
    cost[b] = c; S[k] += c;
  }
  return { S, cost };
}
const fact = (n) => (n <= 1 ? 1 : n * fact(n - 1));
// Beta weights B(k, L + 1 - k) = (k - 1)! (L - k)!/L!: the integral of m^(k-1) (1 - m)^(L-k) dm over [0, 1].
export const BETA = Array.from({ length: L + 1 }, (_, k) => (k === 0 ? 0 : fact(k - 1) * fact(L - k) / fact(L)));
export const bound4 = (S) => sum(S.map((s, k) => s * BETA[k]));
// -log p_theta(x) of the generative chain at temperature tau. In continuous time the masked tokens are revealed
// one at a time in a uniformly random order, so p_theta(x) averages the chain rule over the 8! orders. A sum
// over the visible sets computes the average exactly.
export function nllModel(x, tau) {
  const g = new Float64Array(1 << L);
  g[0] = 1;
  for (let b = 0; b < (1 << L) - 1; b++) {
    if (!g[b]) continue;
    const v = x.map((w, i) => (b >> i & 1 ? w : null)), M = marginals(v);
    let n = 0;
    for (let i = 0; i < L; i++) if (!(b >> i & 1)) n++;
    for (let i = 0; i < L; i++) if (!(b >> i & 1)) g[b | 1 << i] += g[b] / n * Math.exp(predLogp(M, i, x[i], tau));
  }
  return -Math.log(g[(1 << L) - 1]);
}
// The bound and the model's negative log-likelihood averaged over the language, weighted by p_data.
const AVG4 = new Map();
export function avg4(tau) {
  if (!AVG4.has(tau)) {
    let b = 0, n = 0;
    SENT.forEach((s, j) => { b += PS[j] * bound4(maskCosts(s, tau).S); n += PS[j] * nllModel(s, tau); });
    AVG4.set(tau, { bound: b, nll: n });
  }
  return AVG4.get(tau);
}
export const H_LANG_NATS = -sum(PS.map((p) => p * Math.log(p)));
export const TAU4_LIST = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2];
// Schedules alpha_t and their derivatives.
export const SCHED4 = {
  linear: { name: 'linear', a: (t) => t, da: () => 1 },
  cosine: { name: 'cosine', a: (t) => 1 - Math.cos(Math.PI * t / 2), da: (t) => Math.PI / 2 * Math.sin(Math.PI * t / 2) },
  steep: { name: 'steep, α = t⁴', a: (t) => t ** 4, da: (t) => 4 * t ** 3 },
};
// Integrand of the ELBO at time t: alpha'/(1 - alpha) E[sum over masked -log p] with mask probability m = 1 - alpha,
// which is alpha' sum_k m^(k-1) (1 - m)^(L-k) S_k and stays finite as m -> 0.
export function integrand4(sched, S, t) {
  const m = 1 - sched.a(t);
  let e = 0;
  for (let k = 1; k <= L; k++) e += Math.pow(m, k - 1) * Math.pow(1 - m, L - k) * S[k];
  return sched.da(t) * e;
}
export function area4(sched, S, n = 2000) {
  let a = 0;
  for (let i = 0; i < n; i++) a += integrand4(sched, S, (i + 0.5) / n) / n;
  return a;
}

// ---------- Step 5: discrete flow matching ----------
// Per-token path p_t(x^i | x1) = kappa_t delta(x1^i) + (1 - kappa_t) p0, with kappa_t = t and p0 either [M] or
// uniform over the V words. The marginal rate to y is kappa'/(1 - kappa) P(x1^i = y | x_t). An Euler step of
// size h redraws each token from its posterior marginal with probability h/(1 - t).
function postSent(x, t, source) {
  if (source === 'mask') return null;
  const w = SENT.map((s, j) => PS[j] * s.reduce((a, v, i) => a * (t * (x[i] === v ? 1 : 0) + (1 - t) / V), 1));
  const z = sum(w);
  return w.map((v) => v / z);
}
function postMarg(x, t, source) {
  if (source === 'mask') return marginals(x);
  const w = postSent(x, t, source), M = Array.from({ length: L }, () => new Map());
  SENT.forEach((s, j) => s.forEach((v, i) => M[i].set(v, (M[i].get(v) || 0) + w[j])));
  return M;
}
export function runDFM(source, K, r) {
  let x = source === 'mask' ? Array(L).fill(null) : Array.from({ length: L }, () => VOCAB[Math.floor(r.uniform() * V)]);
  const path = [x.slice()], jumps = Array(L).fill(0);
  let changes = 0, revisions = 0;
  for (let k = 0; k < K; k++) {
    const t = k / K, h = 1 / K, M = postMarg(x, t, source), nx = x.slice();
    for (let i = 0; i < L; i++) {
      const u = r.uniform(), v = r.uniform();
      // A visible token of the masked source has the posterior marginal delta(x^i), so it never jumps.
      if (source === 'mask' && x[i] !== null) continue;
      if (u < h / (1 - t)) nx[i] = drawMap(M[i], v)[0];
    }
    for (let i = 0; i < L; i++) if (nx[i] !== x[i]) { changes++; if (++jumps[i] > 1) revisions++; }
    x = nx; path.push(x.slice());
  }
  return { x, path, changes, revisions, ok: isValid(x) };
}
export const K5_LIST = [1, 2, 4, 8, 16, 32, 64];
const STATS5 = new Map();
// Share of valid sentences, mean number of changes per token, and mean number of revisions per sentence
// (changes of a token after its first one) over n seeded runs.
export function stats5(source, K, n = 1000) {
  const key = `${source}:${K}`;
  if (!STATS5.has(key)) {
    const r = makeRng(9500 + K + (source === 'mask' ? 0 : 1000));
    let ok = 0, ch = 0, rv = 0;
    for (let i = 0; i < n; i++) { const o = runDFM(source, K, r); ok += o.ok; ch += o.changes; rv += o.revisions; }
    STATS5.set(key, { valid: ok / n, changes: ch / (n * L), revisions: rv / n });
  }
  return STATS5.get(key);
}
// The eight runs on screen.
export const runs5 = (source, K) => { const r = makeRng(9600 + K); return Array.from({ length: 8 }, () => runDFM(source, K, r)); };

// ---------- Step 6: parallel decoding ----------
// One decoding run. rule: {k} (k random masked positions per step) or {tau} (confidence threshold, at least one).
export function decode(rule, r) {
  const x = Array(L).fill(null), steps = [];
  while (x.includes(null)) {
    const M = marginals(x), masked = x.map((v, i) => (v === null ? i : -1)).filter((i) => i >= 0);
    const preds = masked.map((i) => { const [w, p] = drawMap(M[i], r.uniform()); return { i, w, p, j: r.uniform() }; });
    let keep;
    if (rule.k) keep = [...preds].sort((a, b) => b.j - a.j).slice(0, rule.k).map((p) => p.i);
    else {
      keep = preds.filter((p) => p.p > rule.tau).map((p) => p.i);
      if (!keep.length) keep = [[...preds].sort((a, b) => b.p + 1e-6 * b.j - a.p - 1e-6 * a.j)[0].i];
    }
    for (const p of preds) if (keep.includes(p.i)) x[p.i] = p.w;
    steps.push({ preds, keep });
  }
  const bad = PAIRS.filter(([a, b]) => !pairOk(x, a, b));
  return { x, steps, ok: bad.length === 0, bad };
}
export const K6_LIST = [1, 2, 3, 4, 5, 6, 7, 8];
export const TAU6_LIST = [0.9, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
const STATS6 = new Map();
// Valid share, mean steps and the entropy (bits) of the produced sentences over n seeded runs.
export function stats6(rule, n = 1000) {
  const key = rule.k ? `k${rule.k}` : `t${rule.tau}`;
  if (!STATS6.has(key)) {
    const r = makeRng(rule.k ? 9700 + rule.k : 9800 + Math.round(rule.tau * 100));
    let ok = 0, st = 0;
    const cnt = new Map();
    for (let i = 0; i < n; i++) {
      const o = decode(rule, r), s = o.x.join(' ');
      ok += o.ok; st += o.steps.length; cnt.set(s, (cnt.get(s) || 0) + 1);
    }
    let H = 0;
    for (const c of cnt.values()) H -= c / n * Math.log2(c / n);
    STATS6.set(key, { valid: ok / n, steps: st / n, H, distinct: cnt.size });
  }
  return STATS6.get(key);
}
export const runs6 = (rule) => { const r = makeRng(rule.k ? 9900 + rule.k : 9950 + Math.round(rule.tau * 100)); return Array.from({ length: 6 }, () => decode(rule, r)); };
