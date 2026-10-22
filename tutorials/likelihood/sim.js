// Bounds, weightings and exact likelihoods of the §2.5 tutorial. No DOM access, so the numbers can be
// checked in Node. Convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data. The data are the
// 1D Gaussian mixture MIX1 of §2.2 to §2.4, so every posterior mean and error below is exact. Where a
// level is given by its log-SNR lambda, the path is variance preserving: alpha^2 = sigmoid(lambda).
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, LIN, dens1 } from '../sde/sim.js';
import { uExact1 } from '../flow-matching/sim.js';

export { MIX1 };
export const LAM_MIN = -9, LAM_MAX = 9;
export const sigm = (z) => 1 / (1 + Math.exp(-z));
const phi = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
export const pdata = (x) => MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * phi(x, m, MIX1.sd[k] ** 2), 0);
export const EX2 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * (m * m + MIX1.sd[k] ** 2), 0);   // E[x1^2]

// ---------- Posterior of x1 given a noisy observation ----------
// At log-SNR lam, x_t / alpha_t = y = x1 + r x0 with r = e^{-lam/2}. Given y, x1 is again a mixture:
// component k has weight pi_k(y), mean m_k(y) and variance v_k(y).
export function postY(y, lam) {
  const r2 = Math.exp(-lam), out = [];
  let z = 0;
  MIX1.mu.forEach((m, k) => {
    const s2 = MIX1.sd[k] ** 2, v = s2 + r2, w = MIX1.w[k] * phi(y, m, v);
    z += w; out.push([w, m + (s2 / v) * (y - m), s2 * r2 / v]);
  });
  let e1 = 0, ex2 = 0;
  for (const c of out) { c[0] /= z; e1 += c[0] * c[1]; ex2 += c[0] * (c[2] + c[1] * c[1]); }
  return { comps: out, e1, var1: Math.max(ex2 - e1 * e1, 0), py: z };
}
// Data prediction and its mean squared error E||x1 - hx1||^2 at log-SNR lam, by quadrature over y.
export function mmse1(lam) {
  const r = Math.exp(-lam / 2), lo = -4.2 - 7 * r, hi = 4.6 + 7 * r, n = 900, h = (hi - lo) / n;
  let s = 0;
  for (let i = 0; i <= n; i++) { const p = postY(lo + i * h, lam); s += (i === 0 || i === n ? 0.5 : 1) * p.py * p.var1 * h; }
  return s;
}
// The VDM integrand l(lam) = 1/2 E||x0 - hx0||^2 = 1/2 e^lam E||x1 - hx1||^2, tabulated on a fine grid.
export const NL = 720;
export const LAMS = Float64Array.from({ length: NL + 1 }, (_, i) => LAM_MIN + (LAM_MAX - LAM_MIN) * i / NL);
export const M1 = LAMS.map(mmse1);
export const ELL = LAMS.map((l, i) => 0.5 * Math.exp(l) * M1[i]);
const lerp = (tab, lam) => {
  const u = (lam - LAM_MIN) / (LAM_MAX - LAM_MIN) * NL, i = Math.min(NL - 1, Math.max(0, Math.floor(u))), w = u - i;
  return tab[i] * (1 - w) + tab[i + 1] * w;
};
export const ell = (lam) => lerp(ELL, lam);
export const m1 = (lam) => Math.exp(-lam) * 2 * ell(lam);
// Cumulative integral of l from LAM_MIN, by the trapezoid rule on the table.
export const CUM = (() => { const c = new Float64Array(NL + 1), h = (LAM_MAX - LAM_MIN) / NL; for (let i = 1; i <= NL; i++) c[i] = c[i - 1] + 0.5 * h * (ELL[i] + ELL[i - 1]); return c; })();
export const L_INF = CUM[NL];

// ---------- Step 1 to 3: the discrete bound ----------
// Levels lam_0 < ... < lam_{N-1}, equally spaced from LAM_MIN to LAM_MAX. The decoder of step i uses the
// exact data prediction at the noisier level, so each denoising term is 1/2 (SNR_i - SNR_{i-1}) mmse1(lam_{i-1}).
// The reconstruction decoder is N(hx1, e^{-LAM_MAX}), the posterior scale at the last level.
export const levels = (N) => Float64Array.from({ length: N }, (_, i) => LAM_MIN + (LAM_MAX - LAM_MIN) * i / (N - 1));
const PRIOR = (() => { const a2 = sigm(LAM_MIN), s2 = 1 - a2; return 0.5 * (a2 * EX2 + s2 - 1 - Math.log(s2)); })();
const RECON = (() => { const v = Math.exp(-LAM_MAX); return 0.5 * Math.log(2 * Math.PI * v) + mmse1(LAM_MAX) / (2 * v); })();
export function bound(N) {
  const L = levels(N), terms = [];
  for (let i = 1; i < N; i++) terms.push(0.5 * (Math.exp(L[i]) - Math.exp(L[i - 1])) * mmse1(L[i - 1]));
  const den = terms.reduce((a, b) => a + b, 0);
  return { prior: PRIOR, terms, den, recon: RECON, total: PRIOR + den + RECON, lams: L };
}
// The entropy of the data, -E log p_data(x1): the smallest value any bound on -log p can take on average.
export const H_DATA = (() => { const n = 4000, lo = -5, hi = 6, h = (hi - lo) / n; let s = 0; for (let i = 0; i <= n; i++) { const p = pdata(lo + i * h); if (p > 0) s -= p * Math.log(p) * h; } return s; })();
export const BOUND_INF = PRIOR + L_INF + RECON;

// ---------- Step 2: one KL term at a single noisy point ----------
// VP levels with alpha^2 = sigmoid(lam). s is the noisier level, t the cleaner one (lam_s < lam_t).
export const vp = (lam) => ({ a: Math.sqrt(sigm(lam)), s: Math.sqrt(sigm(-lam)) });
// The encoder posterior q(x_t | x_s, x1) = N(mu, vs) of paper app:elbo, as coefficients of x_s and x1.
export function stepPost(ls, lt) {
  const S = vp(ls), T = vp(lt), ast = S.a / T.a, s2st = S.s ** 2 - ast ** 2 * T.s ** 2;
  return { cs: ast * T.s ** 2 / S.s ** 2, c1: T.a * s2st / S.s ** 2, v: s2st * T.s ** 2 / S.s ** 2, dsnr: Math.exp(lt) - Math.exp(ls) };
}
// Density of x_s at level lam: the mixture with each component scaled by alpha and widened by sigma.
export const pLam = (x, lam) => { const { a, s } = vp(lam); return MIX1.mu.reduce((acc, m, k) => acc + MIX1.w[k] * phi(x, a * m, a * a * MIX1.sd[k] ** 2 + s * s), 0); };
// Fixed uniforms and normals for the posterior draws of x1 given x_s, so the draws move smoothly with x_s.
export const NDRAW = 16;
const DRAWU = (() => { const r = makeRng(251); return Array.from({ length: NDRAW }, (_, i) => [(i + r.uniform()) / NDRAW, r.gauss()]); })();
export function postDraws(xs, ls) {
  const { a } = vp(ls), P = postY(xs / a, ls);
  return { ...P, draws: DRAWU.map(([u, g]) => {
    let k = 0, c = P.comps[0][0];
    while (u > c && k < P.comps.length - 1) c += P.comps[++k][0];
    const [, m, v] = P.comps[k];
    return { k, x1: m + Math.sqrt(v) * g };
  }) };
}

// ---------- Step 4: the schedule changes only the variance ----------
// Each schedule maps t in [0, 1] onto [LAM_MIN, LAM_MAX] and returns lam and dlam/dt. The integrand in t is
// f(t) = lam'(t) l(lam(t)), whose integral is L_INF for every schedule.
const invCum = (c) => {
  let lo = 0, hi = NL;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (CUM[m] < c) lo = m; else hi = m; }
  const w = (c - CUM[lo]) / Math.max(CUM[hi] - CUM[lo], 1e-300);
  return LAMS[lo] + w * (LAMS[hi] - LAMS[lo]);
};
const TA_LIN = sigm(LAM_MIN / 2), TB_LIN = sigm(LAM_MAX / 2);
const TA_COS = (2 / Math.PI) * Math.atan(Math.exp(LAM_MIN / 2)), TB_COS = (2 / Math.PI) * Math.atan(Math.exp(LAM_MAX / 2));
export const SCHED = {
  loglin: { name: 'log-SNR linear', at: (t) => ({ lam: LAM_MIN + (LAM_MAX - LAM_MIN) * t, dlam: LAM_MAX - LAM_MIN }) },
  linear: { name: 'linear path', at: (t) => { const u = TA_LIN + (TB_LIN - TA_LIN) * t; return { lam: 2 * Math.log(u / (1 - u)), dlam: 2 * (TB_LIN - TA_LIN) / (u * (1 - u)) }; } },
  cosine: { name: 'cosine path', at: (t) => { const u = TA_COS + (TB_COS - TA_COS) * t; return { lam: 2 * Math.log(Math.tan(Math.PI * u / 2)), dlam: 2 * Math.PI * (TB_COS - TA_COS) / Math.sin(Math.PI * u) }; } },
  flat: { name: 'flat integrand', at: (t) => { const lam = invCum(t * L_INF); return { lam, dlam: L_INF / Math.max(ell(lam), 1e-12) }; } },
};
export const fSched = (key, t) => { const { lam, dlam } = SCHED[key].at(t); return dlam * ell(lam); };
// Running Monte Carlo means of f(t) with the same NMC uniform draws for every schedule.
export const NMC = 2000;
const TDRAW = (() => { const r = makeRng(317); return Float64Array.from({ length: NMC }, () => r.uniform()); })();
export const RUN = Object.fromEntries(Object.keys(SCHED).map((k) => {
  const run = new Float64Array(NMC); let s = 0, s2 = 0;
  for (let i = 0; i < NMC; i++) { const f = fSched(k, TDRAW[i]); s += f; s2 += f * f; run[i] = s / (i + 1); }
  // Standard deviation of f under U[0, 1], on a fine grid rather than from the draws.
  let m = 0, q = 0; const n = 4000;
  for (let i = 0; i < n; i++) { const f = fSched(k, (i + 0.5) / n); m += f / n; q += f * f / n; }
  return [k, { run, mean: m, sd: Math.sqrt(Math.max(q - m * m, 0)) }];
}));

// ---------- Steps 5 and 6: weightings ----------
// Raw weights w(lam) in log-SNR; each is scaled so that its weighted loss int w l dlam equals L_INF.
const nrm = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
const SIGMA_D = 0.5;
const RAW = {
  elbo: () => 1,
  fm: (l) => Math.exp(-l / 2),
  sd3: (l) => { const t = sigm(l / 2); return nrm(l / 2, 0, 1) / (t * (1 - t)) * Math.exp(-l / 2); },
  edm: (l) => nrm(-l / 2, -1.2, 1.2) * (Math.exp(-l) + SIGMA_D ** 2) / SIGMA_D ** 2,
};
export const WNAMES = { elbo: 'ELBO', fm: 'Flow matching, uniform t', sd3: 'SD3, logit-normal t', edm: 'EDM' };
export const W = Object.fromEntries(Object.entries(RAW).map(([k, f]) => {
  const raw = LAMS.map(f), h = (LAM_MAX - LAM_MIN) / NL;
  let I = 0; for (let i = 0; i < NL; i++) I += 0.5 * h * (raw[i] * ELL[i] + raw[i + 1] * ELL[i + 1]);
  const c = L_INF / I, w = raw.map((v) => c * v);
  // -w' by central differences, and the two sides of the Fubini identity:
  // int w l = int (-w'(l')) CUM(l') dl' + w(LAM_MAX) L_INF.
  const dw = w.map((v, i) => -(w[Math.min(NL, i + 1)] - w[Math.max(0, i - 1)]) / ((Math.min(NL, i + 1) - Math.max(0, i - 1)) * h));
  let rhs = 0, neg = 0, pos = 0;
  for (let i = 0; i < NL; i++) {
    const a = dw[i] * CUM[i], b = dw[i + 1] * CUM[i + 1]; rhs += 0.5 * h * (a + b);
    const d = 0.5 * h * (dw[i] + dw[i + 1]); if (d < 0) neg -= d; else pos += d;
  }
  rhs += w[NL] * L_INF;
  return [k, { w, dw, lhs: L_INF, rhs, neg, pos, end: w[NL], monotone: neg < 1e-6 * (pos + 1e-12) }];
}));
export const wAt = (k, lam) => lerp(W[k].w, lam);

// ---------- Step 7: exact likelihood through the probability-flow ODE ----------
// Linear path. From x1 at t = 1, integrate the ODE backwards to t = 0 with K RK4 steps on the state
// (x, D), D = int div u dt; then log p1(x1) = log N(x0; 0, 1) - D. The divergence is a finite difference.
const T_END = 1 - 1e-6;
const du = (x, t) => (uExact1(x + 1e-4, t) - uExact1(x - 1e-4, t)) / 2e-4;
export function odeLogp(x1, K) {
  const h = -T_END / K, f = (x, t) => [uExact1(x, t), du(x, t)], path = [x1];
  let x = x1, D = 0, t = T_END;
  for (let k = 0; k < K; k++) {
    const a = f(x, t), b = f(x + 0.5 * h * a[0], t + 0.5 * h), c = f(x + 0.5 * h * b[0], t + 0.5 * h), d = f(x + h * c[0], t + h);
    x += h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); D += -h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1]); t += h;
    path.push(x);
  }
  const lp0 = -0.5 * x * x - 0.5 * Math.log(2 * Math.PI);
  return { x0: x, lp0, div: D, logp: lp0 - D, path };
}
// Stratified quantiles of the data, for the average -log p over the data.
export const NQ = 120;
export const XQ = (() => {
  const out = [], n = 6000, lo = -5, hi = 6, hh = (hi - lo) / n; let c = 0, j = 0;
  for (let i = 0; i < n && j < NQ; i++) { c += pdata(lo + (i + 0.5) * hh) * hh; while (j < NQ && c >= (j + 0.5) / NQ) out[j++] = lo + (i + 1) * hh; }
  return Float64Array.from(out);
})();
