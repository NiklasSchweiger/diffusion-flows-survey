// Conditioning, classifier-free guidance, its exact target and Feynman–Kac reweighting of the §2.11 tutorial.
// No DOM access, so the numbers can be checked in Node. Convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise,
// t = 1 data, linear path alpha_t = t, sigma_t = 1 - t. The data are a 1D mixture of five Gaussians: class A
// owns the two left modes, class B the two right ones, and the middle mode belongs to both with probability 1/2.
import { makeRng } from '../../animations/lib/rng.js';

// ---------- Normal distribution ----------
function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

export const MU = [-3, -1.5, 0, 1.5, 3], SD = 0.3, PA = [1, 1, 0.5, 0, 0], NM = 5;
export const XMIN = -4.5, XMAX = 4.5;
const LOG2PI = Math.log(2 * Math.PI);

// A mixture of Gaussians; sd is one number or one per component, w is normalized.
export function mix(mu, sd, w) {
  const z = w.reduce((a, q) => a + q, 0), keep = mu.map((_, k) => w[k] > 0);
  const f = (a) => a.filter((_, k) => keep[k]);
  return { mu: f(mu), sd: f(typeof sd === 'number' ? mu.map(() => sd) : sd), w: f(w.map((q) => q / z)) };
}
export const DATA = mix(MU, SD, [1, 1, 1, 1, 1]);
export const COND = mix(MU, SD, PA);                     // p(x | A): weights 0.4, 0.4, 0.2
export const P_SHARED = 0.2;                             // weight of the shared mode in p(x | A)

// Posterior of the component and of x1 given x_t = x. Fills R (component weights), M (posterior means of x1),
// V (posterior variances), DM (dM/dx) and SC (score of each noised component); returns log p_t(x).
const R = new Float64Array(8), M = new Float64Array(8), V = new Float64Array(8), DM = new Float64Array(8), SC = new Float64Array(8);
function post(Q, x, t) {
  const s = 1 - t, n = Q.mu.length; let mx = -Infinity;
  for (let k = 0; k < n; k++) {
    const s2 = Q.sd[k] * Q.sd[k], v = t * t * s2 + s * s, d = x - t * Q.mu[k];
    R[k] = Math.log(Q.w[k]) - d * d / (2 * v) - 0.5 * (LOG2PI + Math.log(v));
    M[k] = Q.mu[k] + (t * s2 / v) * d; V[k] = s2 * s * s / v; DM[k] = t * s2 / v; SC[k] = -d / v;
    if (R[k] > mx) mx = R[k];
  }
  let z = 0;
  for (let k = 0; k < n; k++) { R[k] = Math.exp(R[k] - mx); z += R[k]; }
  for (let k = 0; k < n; k++) R[k] /= z;
  return mx + Math.log(z);
}
export const logp = (Q, x, t) => post(Q, x, t);
export function score(Q, x, t) { post(Q, x, t); let s = 0; for (let k = 0; k < Q.mu.length; k++) s += R[k] * SC[k]; return s; }
export function xhat(Q, x, t) { post(Q, x, t); let m = 0; for (let k = 0; k < Q.mu.length; k++) m += R[k] * M[k]; return m; }
// The noisy classifier p_t(A | x) = E[p(A | x1) | x_t = x] = sum_k p_t(k | x) p(A | k), and its log.
export function classA(x, t) { post(DATA, x, t); let p = 0; for (let k = 0; k < NM; k++) p += R[k] * PA[k]; return p; }
export function logClassA(x, t) {
  const s = 1 - t; let a = -Infinity, b = -Infinity;
  for (let k = 0; k < NM; k++) {
    const v = t * t * SD * SD + s * s, l = -((x - t * MU[k]) ** 2) / (2 * v);
    b = lse2(b, l); if (PA[k] > 0) a = lse2(a, l + Math.log(PA[k]));
  }
  return a - b;
}
function lse2(a, b) { if (a === -Infinity) return b; if (b === -Infinity) return a; const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); }
// The score of p_t(A | x) by a central difference of its log.
export const dlogClassA = (x, t, d = 1e-4) => (logClassA(x + d, t) - logClassA(x - d, t)) / (2 * d);

// ---------- Exact tilt h_t(x) = E[p(A | x1)^w | x_t = x] ----------
// Table of w log p(A | y) on a fine grid, and a quadrature over the Gaussian posterior of x1 in each component.
const YLO = -8, YHI = 8, NY = 3201, DY = (YHI - YLO) / (NY - 1);
export const YG = Float64Array.from({ length: NY }, (_, i) => YLO + DY * i);
const LPA = Float64Array.from(YG, (y) => logClassA(y, 1));
const NZ = 61, ZN = Float64Array.from({ length: NZ }, (_, j) => -6 + 12 * j / (NZ - 1));
const ZW = (() => { const a = Float64Array.from(ZN, (z) => Math.exp(-z * z / 2)), s = a.reduce((p, q) => p + q, 0); return a.map((q) => q / s); })();
const lpa = (y) => { const u = (y - YLO) / DY, i = Math.max(0, Math.min(NY - 2, Math.floor(u))), f = u - i; return LPA[i] + f * (LPA[i + 1] - LPA[i]); };
const LF = new Float64Array(8 * NZ);
// Returns [log h_t(x), d/dx log h_t(x)].
export function logh(x, t, w) {
  post(DATA, x, t);
  let mx = -Infinity, sbar = 0;
  for (let k = 0; k < NM; k++) {
    sbar += R[k] * SC[k];
    const sd = Math.sqrt(V[k]);
    for (let j = 0; j < NZ; j++) { const l = w * lpa(M[k] + sd * ZN[j]); LF[k * NZ + j] = l; if (R[k] > 0 && l > mx) mx = l; }
  }
  let h = 0, dh = 0;
  for (let k = 0; k < NM; k++) {
    if (R[k] === 0) continue;
    const sd = Math.sqrt(V[k]); let I = 0, J = 0;
    for (let j = 0; j < NZ; j++) { const f = Math.exp(LF[k * NZ + j] - mx); I += ZW[j] * f; J += ZW[j] * ZN[j] * f; }
    h += R[k] * I;
    dh += R[k] * ((SC[k] - sbar) * I + (sd > 1e-12 ? DM[k] * J / sd : 0));
  }
  return [Math.log(h) + mx, dh / h];
}

// ---------- Densities on a grid ----------
export const NG = 361, XG = Float64Array.from({ length: NG }, (_, i) => XMIN + (XMAX - XMIN) * i / (NG - 1));
const DXG = (XMAX - XMIN) / (NG - 1);
// Normalizes log densities on XG into a density.
export function normLog(ls) {
  const m = Math.max(...ls), p = Float64Array.from(ls, (l) => Math.exp(l - m)), z = p.reduce((a, q) => a + q, 0) * DXG;
  return p.map((q) => q / z);
}
export const densAt = (Q, t) => Float64Array.from(XG, (x) => Math.exp(logp(Q, x, t)));
// pi^w_t ∝ p_t(x) p_t(A | x)^w, the density whose score CFG follows at time t.
export const piW = (w, t) => normLog(Float64Array.from(XG, (x) => logp(DATA, x, t) + w * logClassA(x, t)));
// The tilted marginal p_t(x) h_t(x) / Z of the exact guidance target.
export const tiltW = (w, t) => normLog(Float64Array.from(XG, (x) => logp(DATA, x, t) + logh(x, t, w)[0]));
// Mass of a density on XG within 2 SD of the shared mode.
export const sharedMass = (p) => { let s = 0; for (let i = 0; i < NG; i++) if (Math.abs(XG[i]) < 2 * SD) s += p[i] * DXG; return s; };

// ---------- Samplers ----------
// The probability-flow ODE from t = T0 to T1 on a grid uniform in log-SNR, RK4, started from NQ stratified
// quantiles of p_T0. In 1D the flow is monotone, so the end points are the quantiles of the pushforward.
export const T0 = 0.002, T1 = 0.999, NK = 120, NQ = 400;
const lam = (t) => 2 * Math.log(t / (1 - t));
const grid = (a, b, n) => Float64Array.from({ length: n + 1 }, (_, k) => { const e = Math.exp((lam(a) + (lam(b) - lam(a)) * k / n) / 2); return e / (1 + e); });
export const TG = grid(T0, T1, NK);
function quantiles(Q, t, n) {
  const cdf = (x) => { const s = 1 - t; let c = 0; for (let k = 0; k < Q.mu.length; k++) { const v = Math.sqrt(t * t * Q.sd[k] ** 2 + s * s); c += Q.w[k] * Phi((x - t * Q.mu[k]) / v); } return c; };
  return Float64Array.from({ length: n }, (_, i) => { const p = (i + 0.5) / n; let lo = -10, hi = 10; for (let k = 0; k < 60; k++) { const m = (lo + hi) / 2; if (cdf(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; });
}
export const X0 = quantiles(DATA, T0, NQ);
export function flow(vel, x0 = X0) {
  return Float64Array.from(x0, (x) => {
    for (let k = 0; k < NK; k++) {
      const t = TG[k], h = TG[k + 1] - t, tm = t + h / 2;
      const k1 = vel(x, t), k2 = vel(x + h * k1 / 2, tm), k3 = vel(x + h * k2 / 2, tm), k4 = vel(x + h * k3, t + h);
      x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
    return x;
  });
}
// The same flow for NTR of the starting points, with the position at every grid time, for drawing paths.
export const NTR = 40;
export function trails(vel) {
  return Array.from({ length: NTR }, (_, j) => {
    let x = X0[Math.floor((j + 0.5) * NQ / NTR)]; const tr = new Float64Array(NK + 1); tr[0] = x;
    for (let k = 0; k < NK; k++) {
      const t = TG[k], h = TG[k + 1] - t, tm = t + h / 2;
      const k1 = vel(x, t), k2 = vel(x + h * k1 / 2, tm), k3 = vel(x + h * k2 / 2, tm), k4 = vel(x + h * k3, t + h);
      x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6; tr[k + 1] = x;
    }
    return tr;
  });
}
// Guided velocity of two models: the data prediction (1 - w) xhat_neg + w xhat_pos, with the velocity
// u = (xhat - x) / (1 - t) of the linear path. With neg = the unconditional model this is CFG.
// The guidance weight is w only for t in [a, b] and 1 outside.
export const guidedVel = (pos, neg, w, a = 0, b = 1) => (x, t) => {
  const ww = t >= a && t <= b ? w : 1;
  return ((1 - ww) * xhat(neg, x, t) + ww * xhat(pos, x, t) - x) / (1 - t);
};
// The exact tilted flow: the unconditional velocity plus (1 - t) / t times the score of h_t.
export const tiltVel = (w) => (x, t) => (xhat(DATA, x, t) - x) / (1 - t) + ((1 - t) / t) * logh(x, t, w)[1];

// ---------- Targets and metrics ----------
// Quantiles of a density given as log values on the fine grid YG.
function quantFromLog(ls, n) {
  const m = Math.max(...ls), p = Float64Array.from(ls, (l) => Math.exp(l - m)), c = new Float64Array(NY);
  for (let i = 1; i < NY; i++) c[i] = c[i - 1] + (p[i] + p[i - 1]) * DY / 2;
  const z = c[NY - 1];
  return Float64Array.from({ length: n }, (_, k) => {
    const q = (k + 0.5) / n * z; let a = 0, b = NY - 1;
    while (b - a > 1) { const mid = (a + b) >> 1; if (c[mid] < q) a = mid; else b = mid; }
    return YG[a] + DY * (q - c[a]) / Math.max(c[b] - c[a], 1e-300);
  });
}
// Quantiles of pi^w_1 ∝ p(x) p(A | x)^w.
const QCACHE = new Map();
export function targetQ(w, n = NQ) {
  const key = `${w}|${n}`;
  if (!QCACHE.has(key)) QCACHE.set(key, quantFromLog(Float64Array.from(YG, (y) => logp(DATA, y, 1) + w * LPA[Math.round((y - YLO) / DY)]), n));
  return QCACHE.get(key);
}
// W1 of NRUN independent exact draws from pi^w_1 to its quantiles: the floor that any sampler with NRUN outputs reaches.
export function iidW1(w, n) { const F = targetQ(w, 4000), r = makeRng(77 + w); return w1(Float64Array.from({ length: n }, () => F[Math.floor(r.uniform() * 4000)]), targetQ(w, n)); }
export function w1(xs, Q) { const s = Float64Array.from(xs).sort(), n = s.length; let d = 0; for (let i = 0; i < n; i++) d += Math.abs(s[i] - Q[i]) / n; return d; }
// A Gaussian kernel density of equally weighted samples on XG.
export function kde(xs, bw = 0.07) {
  const out = new Float64Array(NG), c = 1 / (xs.length * bw * Math.sqrt(2 * Math.PI));
  for (const x of xs) { const i0 = Math.max(0, Math.floor((x - 4 * bw - XMIN) / DXG)), i1 = Math.min(NG - 1, Math.ceil((x + 4 * bw - XMIN) / DXG)); for (let i = i0; i <= i1; i++) out[i] += c * Math.exp(-0.5 * ((XG[i] - x) / bw) ** 2); }
  return out;
}
// Quality against p(x | A): precision is the share of samples within 2 SD of a mode of class A; shares are the
// fractions of samples whose nearest mode is each of the five; coverage is sum_k min(share_k, p(k | A)), which is
// 1 when every mode of A gets its share and nothing goes to B.
export const PK_A = [0.4, 0.4, 0.2, 0, 0];
export function quality(xs) {
  const share = new Float64Array(NM); let prec = 0;
  for (const x of xs) {
    let k = 0; for (let j = 1; j < NM; j++) if (Math.abs(x - MU[j]) < Math.abs(x - MU[k])) k = j;
    share[k] += 1 / xs.length;
    if (PA[k] > 0 && Math.abs(x - MU[k]) < 2 * SD) prec += 1 / xs.length;
  }
  return { prec, share, cov: share.reduce((a, q, k) => a + Math.min(q, PK_A[k]), 0) };
}

// ---------- Imperfect models for steps 4 and 5 ----------
// A trained conditional model is too wide and partly ignores the condition; the unconditional model is the
// same network with the condition dropped. The weak model of autoguidance is a wider version of the conditional one.
export const SD_MODEL = 0.45, SD_WEAK = 0.65, LEAK = 0.15;
const condW = (sd) => mix(MU, sd, PA.map((a) => (1 - LEAK) * a / 2.5 + LEAK / 5));
export const MODEL_C = condW(SD_MODEL), MODEL_U = mix(MU, SD_MODEL, [1, 1, 1, 1, 1]), MODEL_W = condW(SD_WEAK);

// ---------- Feynman–Kac SMC (step 6) ----------
// Proposal: Euler–Maruyama on the exact conditional SDE, whose marginals are p_t(x | A); drift
// u + (1 - t)/t s with u and s of p_t(x | A), diffusion g^2 = 2(1 - t)/t. Potentials G_t = p_t(A | x)^{w-1}, so the
// weights telescope to p(A | x1)^{w-1} / p_T(A | x_T)^{w-1} and the target is p(x | A) p(A | x)^{w-1} ∝ pi^w_1.
// Resampling (multinomial) when the effective sample size drops below N/2; one particle per run is returned.
export const NRUN = 400, NK6 = 100, N6 = [1, 2, 4, 8, 16, 32];
const TG6 = grid(0.02, 0.995, NK6);
const sdeDrift = (x, t) => { post(COND, x, t); let m = 0, s = 0; for (let k = 0; k < COND.mu.length; k++) { m += R[k] * M[k]; s += R[k] * SC[k]; } return (m - x) / (1 - t) + ((1 - t) / t) * s; };
export function smc(N, w) {
  const r = makeRng(611 + 7 * N + 101 * w), out = new Float64Array(NRUN), x = new Float64Array(N), lw = new Float64Array(N), lg = new Float64Array(N), nx = new Float64Array(N);
  const t0 = TG6[0], v0 = Math.sqrt(t0 * t0 * SD * SD + (1 - t0) ** 2);
  let resamples = 0;
  for (let run = 0; run < NRUN; run++) {
    for (let i = 0; i < N; i++) {
      let u = r.uniform(), k = 0; while (k < COND.w.length - 1 && (u -= COND.w[k]) > 0) k++;
      x[i] = t0 * COND.mu[k] + v0 * r.gauss(); lg[i] = (w - 1) * logClassA(x[i], t0); lw[i] = lg[i];
    }
    for (let k = 0; k < NK6; k++) {
      const t = TG6[k], dt = TG6[k + 1] - t, g2 = 2 * (1 - t) / t, last = k === NK6 - 1, tn = last ? 1 : TG6[k + 1];
      for (let i = 0; i < N; i++) {
        x[i] += sdeDrift(x[i], t) * dt + Math.sqrt(g2 * dt) * r.gauss();
        const l = (w - 1) * logClassA(x[i], tn); lw[i] += l - lg[i]; lg[i] = l;
      }
      if (last || N === 1) continue;
      const m = Math.max(...lw); let s = 0, s2 = 0;
      for (let i = 0; i < N; i++) { const q = Math.exp(lw[i] - m); s += q; s2 += q * q; }
      if (s * s / s2 < N / 2) {
        resamples++;
        for (let i = 0; i < N; i++) { let u = r.uniform() * s, j = 0; while (j < N - 1 && (u -= Math.exp(lw[j] - m)) > 0) j++; nx[i] = x[j]; }
        for (let i = 0; i < N; i++) { x[i] = nx[i]; lw[i] = 0; lg[i] = (w - 1) * logClassA(x[i], tn); }
      }
    }
    const m = Math.max(...lw); let s = 0; for (let i = 0; i < N; i++) s += Math.exp(lw[i] - m);
    let u = r.uniform() * s, j = 0; while (j < N - 1 && (u -= Math.exp(lw[j] - m)) > 0) j++;
    out[run] = x[j];
  }
  return { out, resamples: resamples / NRUN };
}
