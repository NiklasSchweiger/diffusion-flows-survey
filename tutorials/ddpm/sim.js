// Exact quantities of the DDPM tutorial. No DOM access, so the numbers can be checked in Node.
// DDPM notation: steps k = 0..K, k = 0 is the data, x_k = sqrt(abar_k) x_0 + sqrt(1 - abar_k) eps.
// In the survey's notation x_k is x_t with alpha_t = sqrt(abar_k), sigma_t = sqrt(1 - abar_k).
// The data are the 1D mixture MIX1, so marginals, posteriors and reverse kernels are mixtures of Gaussians.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1 } from '../lib/mix1d.js';

export { MIX1 };
const R2PI = Math.sqrt(2 * Math.PI), NC = MIX1.mu.length;

// ---------- Schedule ----------
// abar(tau) of the continuous variance-preserving schedule with beta(tau) linear from 0.1 to 20.
// K steps take abar_k = abar(k/K); for K = 1000 the betas run from about 1e-4 to 0.02 (Ho et al.).
export const BMIN = 0.1, BMAX = 20;
export const abarC = (tau) => Math.exp(-BMIN * tau - 0.5 * (BMAX - BMIN) * tau * tau);
const schedCache = new Map();
export function schedule(K) {
  if (schedCache.has(K)) return schedCache.get(K);
  const abar = Float64Array.from({ length: K + 1 }, (_, k) => abarC(k / K));
  const alpha = new Float64Array(K + 1), beta = new Float64Array(K + 1), btil = new Float64Array(K + 1);
  for (let k = 1; k <= K; k++) {
    alpha[k] = abar[k] / abar[k - 1]; beta[k] = 1 - alpha[k];
    btil[k] = (1 - abar[k - 1]) / (1 - abar[k]) * beta[k];
  }
  const s = { K, abar, alpha, beta, btil };
  schedCache.set(K, s);
  return s;
}
// Coefficients of the posterior mean mu~ = c0 x_0 + ck x_k.
export function postCoef(S, k) {
  const { abar, alpha, beta } = S;
  return { c0: Math.sqrt(abar[k - 1]) * beta[k] / (1 - abar[k]), ck: Math.sqrt(alpha[k]) * (1 - abar[k - 1]) / (1 - abar[k]) };
}

// ---------- Fields of the noised mixture ----------
// Level with signal a = sqrt(abar) and noise s = sqrt(1 - abar). Returns log density, score,
// E[x_0 | x], Var[x_0 | x] and the optimal noise prediction eps* = E[eps | x] = -s * score.
export function fields(x, abar) {
  const a = Math.sqrt(abar), s2 = 1 - abar;
  let lmax = -Infinity; const lw = new Array(NC), v = new Array(NC);
  for (let j = 0; j < NC; j++) {
    v[j] = abar * MIX1.sd[j] ** 2 + s2;
    lw[j] = Math.log(MIX1.w[j]) - 0.5 * Math.log(v[j]) - (x - a * MIX1.mu[j]) ** 2 / (2 * v[j]);
    if (lw[j] > lmax) lmax = lw[j];
  }
  let Z = 0; for (let j = 0; j < NC; j++) { lw[j] = Math.exp(lw[j] - lmax); Z += lw[j]; }
  let score = 0, m1 = 0, m2 = 0;
  for (let j = 0; j < NC; j++) {
    const r = lw[j] / Z, d = x - a * MIX1.mu[j], sd2 = MIX1.sd[j] ** 2;
    const mj = MIX1.mu[j] + a * sd2 * d / v[j], cj = sd2 * s2 / v[j];
    score -= r * d / v[j]; m1 += r * mj; m2 += r * (mj * mj + cj);
  }
  return { logp: lmax + Math.log(Z) - Math.log(R2PI), score, x0: m1, var0: Math.max(0, m2 - m1 * m1), eps: -Math.sqrt(s2) * score };
}
export const dens = (x, abar) => Math.exp(fields(x, abar).logp);
export const epsStar = (x, abar) => fields(x, abar).eps;

// ---------- Step 2: the exact reverse kernel q(x_{k-1} | x_k) ----------
// q(x_{k-1} | x_k) is proportional to N(x_k; sqrt(alpha_k) y, beta_k) p_{k-1}(y), a mixture of Gaussians in y.
export function reverseKernel(S, k, xk) {
  const ab = S.abar[k - 1], al = S.alpha[k], be = S.beta[k], a = Math.sqrt(ab), ra = Math.sqrt(al);
  const comps = [];
  for (let j = 0; j < NC; j++) {
    const m = a * MIX1.mu[j], v = ab * MIX1.sd[j] ** 2 + (1 - ab);
    const prec = 1 / v + al / be, vv = 1 / prec, mm = vv * (m / v + ra * xk / be);
    const lw = Math.log(MIX1.w[j]) - 0.5 * Math.log(al * v + be) - (xk - ra * m) ** 2 / (2 * (al * v + be));
    comps.push({ m: mm, v: vv, lw });
  }
  const mx = Math.max(...comps.map((c) => c.lw)); let Z = 0;
  comps.forEach((c) => { c.w = Math.exp(c.lw - mx); Z += c.w; });
  comps.forEach((c) => { c.w /= Z; });
  const mean = comps.reduce((s, c) => s + c.w * c.m, 0), vr = comps.reduce((s, c) => s + c.w * (c.v + c.m * c.m), 0) - mean * mean;
  return { comps, mean, var: vr, pdf: (y) => comps.reduce((s, c) => s + c.w * Math.exp(-((y - c.m) ** 2) / (2 * c.v)) / Math.sqrt(2 * Math.PI * c.v), 0) };
}
// KL(exact kernel || N(m, v)) by quadrature on +-8 standard deviations of the kernel.
export function klToGauss(ker, m, v) {
  const sd = Math.sqrt(ker.var), lo = ker.mean - 8 * sd, hi = ker.mean + 8 * sd, n = 1600, h = (hi - lo) / n;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const y = lo + (i + 0.5) * h, p = ker.pdf(y);
    if (p > 1e-300) s += p * (Math.log(p) + 0.5 * Math.log(2 * Math.PI * v) + (y - m) ** 2 / (2 * v));
  }
  return Math.max(0, s * h);
}
// Stratified posterior draws of x_0 given x_k (quantiles of the posterior mixture), for the thin Gaussians.
export function posteriorX0(xk, abar, n) {
  const a = Math.sqrt(abar), s2 = 1 - abar, comps = [];
  for (let j = 0; j < NC; j++) {
    const sd2 = MIX1.sd[j] ** 2, v = abar * sd2 + s2;
    comps.push({ lw: Math.log(MIX1.w[j]) - 0.5 * Math.log(v) - (xk - a * MIX1.mu[j]) ** 2 / (2 * v), m: MIX1.mu[j] + a * sd2 * (xk - a * MIX1.mu[j]) / v, sd: Math.sqrt(sd2 * s2 / v) });
  }
  const mx = Math.max(...comps.map((c) => c.lw)); let Z = 0;
  comps.forEach((c) => { c.w = Math.exp(c.lw - mx); Z += c.w; });
  const cdf = (x) => comps.reduce((s, c) => s + c.w / Z * 0.5 * (1 + erf((x - c.m) / (c.sd * Math.SQRT2))), 0);
  return Array.from({ length: n }, (_, i) => {
    const q = (i + 0.5) / n; let lo = -8, hi = 8;
    for (let it = 0; it < 50; it++) { const mid = 0.5 * (lo + hi); if (cdf(mid) < q) lo = mid; else hi = mid; }
    return 0.5 * (lo + hi);
  });
}
function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}

// ---------- Step 3: the ELBO of the exact-mean model ----------
// V_k = E_{x_k}[Var(x_0 | x_k)], the error of the best data prediction at level k, by quadrature.
const GX = Float64Array.from({ length: 561 }, (_, i) => -7 + 14 * i / 560), GH = 14 / 560;
const mmseCache = new Map();
export function mmse(abar) {
  const key = abar.toPrecision(12);
  if (mmseCache.has(key)) return mmseCache.get(key);
  let s = 0;
  for (const x of GX) { const f = fields(x, abar); s += Math.exp(f.logp) * f.var0 * GH; }
  if (mmseCache.size > 20000) mmseCache.clear();
  mmseCache.set(key, s);
  return s;
}
export const DATA_M2 = MIX1.mu.reduce((s, m, j) => s + MIX1.w[j] * (m * m + MIX1.sd[j] ** 2), 0);
// Entropy of the data by quadrature (1.58 nats, see the §2.0 primer).
export const ENTROPY = (() => { let s = 0; for (const x of GX) { const p = dens(x, 1); if (p > 1e-300) s -= p * Math.log(p) * GH; } return s; })();
// Terms of the bound for a model with the exact mean and variance sigma_k^2 = btil_k ('tilde') or beta_k ('beta').
// terms[k] is L_{k-1} for k = 2..K; L_0 uses a Gaussian decoder with variance beta_1 in both cases.
const elboCache = new Map();
export function elbo(K, vchoice) {
  const key = `${K}|${vchoice}`;
  if (elboCache.has(key)) return elboCache.get(key);
  const S = schedule(K), terms = new Float64Array(K + 1);
  for (let k = 2; k <= K; k++) {
    const v1 = S.btil[k], v2 = vchoice === 'beta' ? S.beta[k] : S.btil[k], { c0 } = postCoef(S, k);
    terms[k] = 0.5 * (Math.log(v2 / v1) + v1 / v2 - 1) + c0 * c0 * mmse(S.abar[k]) / (2 * v2);
  }
  const aK = S.abar[K];
  const LK = 0.5 * (-Math.log(1 - aK) + (1 - aK) + aK * DATA_M2 - 1);
  const L0 = 0.5 * Math.log(2 * Math.PI * S.beta[1]) + mmse(S.abar[1]) / (2 * S.beta[1]);
  let sum = 0; for (let k = 2; k <= K; k++) sum += terms[k];
  const out = { terms, LK, L0, sum, total: LK + sum + L0 };
  elboCache.set(key, out);
  return out;
}

// ---------- Step 4: weights of the noise loss ----------
// L_{k-1} = lambda_k E|eps - eps_theta|^2 + const, lambda_k = beta_k^2 / (2 sigma_k^2 alpha_k (1 - abar_k)).
export function lambdas(K, vchoice) {
  const S = schedule(K), out = new Float64Array(K + 1);
  for (let k = 2; k <= K; k++) {
    const s2 = vchoice === 'beta' ? S.beta[k] : S.btil[k];
    out[k] = S.beta[k] ** 2 / (2 * s2 * S.alpha[k] * (1 - S.abar[k]));
  }
  return out;
}

// ---------- Step 5: training pairs (x_k, eps) ----------
function sampleData(r) {
  const u = r.uniform(); let j = 0, c = MIX1.w[0];
  while (u > c && j < NC - 1) c += MIX1.w[++j];
  return MIX1.mu[j] + MIX1.sd[j] * r.gauss();
}
export const PAIRS = (() => {
  const r = makeRng(61), n = 2500, x0 = new Float64Array(n), eps = new Float64Array(n);
  for (let i = 0; i < n; i++) { x0[i] = sampleData(r); eps[i] = r.gauss(); }
  return { n, x0, eps };
})();

// ---------- Step 6: samplers ----------
// One generalized DDIM step from level k to k - 1 with sigma_k = eta sqrt(btil_k):
// x_{k-1} = sqrt(abar_{k-1}) x0hat + sqrt(1 - abar_{k-1} - sigma_k^2) eps* + sigma_k z.
// eta = 1 is DDPM ancestral sampling with variance btil_k, eta = 0 is deterministic DDIM.
export const NP = 2000;
const NOISE = (() => {
  // Start points: stratified quantiles of N(0, 1), so DDIM measures only the discretization error.
  const r = makeRng(71), x = new Float64Array(NP);
  for (let i = 0; i < NP; i++) { const q = (i + 0.5) / NP; let lo = -8, hi = 8; for (let it = 0; it < 50; it++) { const m = 0.5 * (lo + hi); if (0.5 * (1 + erf(m / Math.SQRT2)) < q) lo = m; else hi = m; } x[i] = 0.5 * (lo + hi); }
  return { x, r };
})();
const runCache = new Map();
// Returns final samples and the paths of every 'stride'-th particle at every level (for drawing).
export function runSampler(K, eta, keepEvery = 40) {
  const key = `${K}|${eta}`;
  if (runCache.has(key)) return runCache.get(key);
  const S = schedule(K), x = Float64Array.from(NOISE.x), r = makeRng(1000 + K);
  const nKeep = Math.ceil(NP / keepEvery), paths = Array.from({ length: nKeep }, () => new Float64Array(K + 1));
  for (let i = 0; i < nKeep; i++) paths[i][K] = x[i * keepEvery];
  for (let k = K; k >= 1; k--) {
    const ab = S.abar[k], abp = S.abar[k - 1], sg2 = eta * eta * S.btil[k];
    const cdir = Math.sqrt(Math.max(0, 1 - abp - sg2)), sg = Math.sqrt(sg2);
    for (let i = 0; i < NP; i++) {
      const f = fields(x[i], ab), x0h = (x[i] - Math.sqrt(1 - ab) * f.eps) / Math.sqrt(ab);
      const z = r.gauss();
      x[i] = Math.sqrt(abp) * x0h + cdir * f.eps + (eta > 0 ? sg * z : 0);
    }
    for (let i = 0; i < nKeep; i++) paths[i][k - 1] = x[i * keepEvery];
  }
  const out = { x, paths, w1: w1Data(x) };
  if (runCache.size > 40) runCache.clear();
  runCache.set(key, out);
  return out;
}
// W1 between samples and the data mixture: integral of |F_n - F|.
const cdfData = (x) => MIX1.w.reduce((s, w, j) => s + w * 0.5 * (1 + erf((x - MIX1.mu[j]) / (MIX1.sd[j] * Math.SQRT2))), 0);
export function w1Data(xs) {
  const a = Float64Array.from(xs).sort(), n = a.length, lo = Math.min(a[0], -7), hi = Math.max(a[n - 1], 7), M = 4000, h = (hi - lo) / M;
  let j = 0, s = 0;
  for (let i = 0; i < M; i++) { const x = lo + (i + 0.5) * h; while (j < n && a[j] <= x) j++; s += Math.abs(j / n - cdfData(x)); }
  return s * h;
}
// W1 of 2000 exact draws from the data, the floor that sampling noise sets for eta = 1.
export const W1_FLOOR = (() => { const r = makeRng(81); return w1Data(Float64Array.from({ length: NP }, () => sampleData(r))); })();
export const KS = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];

// ---------- Step 7: the same path in flow-matching time ----------
// x_k / (sqrt(abar) + sqrt(1 - abar)) = t x_data + (1 - t) eps with t = sqrt(abar) / (sqrt(abar) + sqrt(1 - abar)).
export function fmTime(abar) { const a = Math.sqrt(abar), s = Math.sqrt(1 - abar); return { t: a / (a + s), c: 1 / (a + s) }; }
