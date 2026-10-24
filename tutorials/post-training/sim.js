// Models, samplers and training loops of the §4 tutorial. No DOM access, so the numbers can be
// checked in Node. The base model is the exact mixture MIX1; every fine-tuned model is a Gaussian
// mixture with three components and free weights, means and widths (nine parameters), so its density
// and its KL to any target are exact. Samplers are Euler–Maruyama on the exact fields of that mixture.
import { MIX1, XG, NG, DX, fields, comps, tiltQuad, densGrid, normalize, klGrid, meanGrid, w1Grid, w1Samples, toVec, fromVec, kde } from '../lib/mix1d.js';
import { makeRng } from '../../animations/lib/rng.js';

export { MIX1, XG, NG, DX, densGrid, normalize, klGrid, meanGrid, w1Grid, tiltQuad, fromVec, toVec, kde, w1Samples };
export const A0 = 1.2, B0 = 0.5;                    // reward centre and KL weight of steps 4 to 7
export const reward = (x, a = A0) => -((x - a) ** 2) / 2;
export const pData = normalize(densGrid(MIX1));
export const pStar = (a = A0, beta = B0) => normalize(densGrid(tiltQuad(MIX1, a, beta)));

// ---------- Step 1 and 2: the tilted target on a grid ----------
// p*_beta ∝ p_data e^{r/beta} for any reward given on the grid.
export function tiltGrid(rg, beta) {
  const l = Float64Array.from(XG, (_, i) => Math.log(Math.max(pData[i], 1e-300)) + rg[i] / beta), mx = Math.max(...l);
  return normalize(l.map((q) => Math.exp(q - mx)));
}
// Step 2: gold reward, a bump between the middle and the right mode, and a straight-line proxy fitted
// by least squares to n noisy labels at seeded draws from p_data.
export const gold = (x) => Math.exp(-((x - 1.1) ** 2) / (2 * 0.55 ** 2));
export function sampleData(rng, n, m = MIX1) {
  return Float64Array.from({ length: n }, () => {
    let u = rng.uniform(), k = 0; while (k < m.w.length - 1 && u > m.w[k]) { u -= m.w[k]; k++; }
    return m.mu[k] + m.sd[k] * rng.gauss();
  });
}
export function fitProxy(seed, n = 20, noise = 0.1) {
  const rng = makeRng(seed), xs = sampleData(rng, n), ys = Float64Array.from(xs, (x) => gold(x) + noise * rng.gauss());
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; });
  const b = sxy / sxx, c = my - b * mx;
  return { xs, ys, b, c, f: (x) => c + b * x };
}
export const BETAS2 = Float64Array.from({ length: 41 }, (_, i) => 10 ** (1.5 - 3 * i / 40));   // 31.6 down to 0.0316
export function overopt(proxy) {
  const rg = Float64Array.from(XG, proxy.f);
  return Array.from(BETAS2, (beta) => {
    const p = tiltGrid(rg, beta);
    return { beta, p, kl: klGrid(p, pData), gold: meanGrid(p, gold), proxy: meanGrid(p, proxy.f) };
  });
}

// ---------- The sampler (steps 3 to 6) ----------
// Time grid from T0 to 1; the start is N(0, 1) at T0, within 0.02 of the exact marginal there.
export const T0 = 0.02, KS = 40;
export const TG = Float64Array.from({ length: KS + 1 }, (_, i) => T0 + (1 - T0) * i / KS);
// Noise level as a multiple lam of the memoryless level eps_t² = 2(1 - t)/t.
export const eps2 = (t, lam) => lam * lam * 2 * (1 - t) / t;

// Drift b = u + (eps²/2) s of a mixture model, with its gradient in the nine parameters and in x.
// Parameters per component: logit of the weight, mean, log of the width.
export function drift(m, x, t, e2, grad = false) {
  const K = m.mu.length, { mt, vt } = comps(m, t), l = new Array(K);
  let lmax = -Infinity;
  for (let k = 0; k < K; k++) { l[k] = Math.log(m.w[k]) - 0.5 * Math.log(vt[k]) - (x - mt[k]) ** 2 / (2 * vt[k]); lmax = Math.max(lmax, l[k]); }
  let Z = 0; for (let k = 0; k < K; k++) { l[k] = Math.exp(l[k] - lmax); Z += l[k]; }
  const r = l.map((q) => q / Z), d = mt.map((q) => x - q), s2 = m.sd.map((q) => q * q);
  const c = s2.map((q) => t * q - (1 - t));
  const bk = r.map((_, k) => m.mu[k] + c[k] * d[k] / vt[k] - 0.5 * e2 * d[k] / vt[k]);
  const b = bk.reduce((a, q, k) => a + r[k] * q, 0);
  if (!grad) return { b };
  const g = new Float64Array(3 * K);
  let bx = 0;
  for (let k = 0; k < K; k++) {
    const v = vt[k], dk = d[k], df = bk[k] - b;
    g[k] = r[k] * df;                                                      // logit
    g[K + k] = r[k] * (1 - c[k] * t / v + 0.5 * e2 * t / v) + r[k] * df * dk * t / v;   // mean
    const dv = 2 * t * t * s2[k];                                          // log width
    const dl = -0.5 * dv / v + dk * dk * dv / (2 * v * v);
    g[2 * K + k] = r[k] * (2 * t * s2[k] * dk / v - c[k] * dk * dv / (v * v) + 0.5 * e2 * dk * dv / (v * v)) + r[k] * df * dl;
    bx += r[k] * (c[k] / v - 0.5 * e2 / v) + r[k] * df * (-dk / v);
  }
  return { b, g, bx };
}

// One path of the sampler. Returns the path, the noise used and the log-probability of each step.
export function path(m, x, lam, rng, zs) {
  const xs = new Float64Array(KS + 1), z = zs ?? new Float64Array(KS), lp = new Float64Array(KS);
  xs[0] = x;
  for (let k = 0; k < KS; k++) {
    const t = TG[k], h = TG[k + 1] - t, e2 = eps2(t, lam), { b } = drift(m, x, t, e2);
    if (!zs) z[k] = rng.gauss();
    x += b * h + Math.sqrt(e2 * h) * z[k];
    xs[k + 1] = x;
    lp[k] = e2 > 0 ? -0.5 * z[k] * z[k] - 0.5 * Math.log(2 * Math.PI * e2 * h) : NaN;
  }
  return { xs, z, lp };
}
export function ends(m, lam, n, seed) {
  const rng = makeRng(seed), out = new Float64Array(n);
  for (let i = 0; i < n; i++) { const p = path(m, rng.gauss(), lam, rng); out[i] = p.xs[KS]; }
  return out;
}
// Log-probability of a whole path under a model with the same noise, and its gradient in the parameters.
export function pathLogp(m, xs, lam, grad = false) {
  let lp = 0; const K = m.mu.length, g = new Float64Array(3 * K);
  for (let k = 0; k < KS; k++) {
    const t = TG[k], h = TG[k + 1] - t, e2 = eps2(t, lam), D = drift(m, xs[k], t, e2, grad), res = xs[k + 1] - xs[k] - D.b * h;
    lp += -res * res / (2 * e2 * h);
    if (grad) for (let j = 0; j < 3 * K; j++) g[j] += res / e2 * D.g[j];
  }
  return { lp, g };
}

// ---------- Step 4: policy gradient (DDPO, GRPO) ----------
// Maximizes E[r] - beta KL(path || base path) with the score-function estimator. The KL enters through
// the per-path log-ratio, which is an unbiased sample of the path KL.
export function adam(lr) {
  let m = null, v = null, k = 0;
  return (theta, g) => {
    if (!m) { m = new Float64Array(g.length); v = new Float64Array(g.length); }
    k++;
    for (let j = 0; j < g.length; j++) {
      m[j] = 0.9 * m[j] + 0.1 * g[j]; v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
      theta[j] += lr * (m[j] / (1 - 0.9 ** k)) / (Math.sqrt(v[j] / (1 - 0.999 ** k)) + 1e-8);
    }
  };
}
export function makePG({ algo = 'grpo', G = 16, groups = 4, beta = B0, a = A0, seed = 11, lr = 0.04, clip = 0.2, epochs = 2 }) {
  const rng = makeRng(seed), theta = toVec(MIX1), step = adam(lr), lam = 1;
  const S = { theta, it: 0, hist: [], group: null, baseline: null };
  const evalModel = () => {
    const m = fromVec(theta), p = normalize(densGrid(m));
    return { m, p, er: meanGrid(p, (x) => reward(x, a)), klD: klGrid(p, pData), klS: klGrid(p, pStar(a, beta)) };
  };
  S.cur = evalModel();
  S.hist.push({ it: 0, er: S.cur.er, klD: S.cur.klD, klS: S.cur.klS, pathKL: 0 });
  S.step = () => {
    const m = fromVec(theta), batch = [];
    let pk = 0;
    for (let gi = 0; gi < groups; gi++) {
      const x0 = rng.gauss(), grp = [];
      for (let i = 0; i < G; i++) {
        const p = path(m, x0, lam, rng), lt = pathLogp(m, p.xs, lam).lp, lb = pathLogp(MIX1, p.xs, lam).lp;
        const R = reward(p.xs[KS], a) - beta * (lt - lb);
        pk += lt - lb; grp.push({ xs: p.xs, R, r: reward(p.xs[KS], a), old: lt });
      }
      if (algo === 'grpo') {
        const mu = grp.reduce((s, q) => s + q.R, 0) / G, sd = Math.sqrt(grp.reduce((s, q) => s + (q.R - mu) ** 2, 0) / G);
        grp.forEach((q) => { q.A = sd > 1e-12 ? (q.R - mu) / sd : 0; });
      }
      batch.push(...grp);
      if (gi === 0) S.group = grp;
    }
    if (algo === 'ddpo') {
      const mean = batch.reduce((s, q) => s + q.R, 0) / batch.length;
      S.baseline = S.baseline === null ? mean : 0.9 * S.baseline + 0.1 * mean;
      const sd = Math.sqrt(batch.reduce((s, q) => s + (q.R - S.baseline) ** 2, 0) / batch.length) || 1;
      batch.forEach((q) => { q.A = (q.R - S.baseline) / sd; });
    }
    // GRPO reuses the batch for a few epochs with a clipped ratio; DDPO takes one plain step.
    const E = algo === 'grpo' ? epochs : 1;
    for (let e = 0; e < E; e++) {
      const mm = fromVec(theta), g = new Float64Array(theta.length);
      for (const q of batch) {
        const { lp, g: gl } = pathLogp(mm, q.xs, lam, true), ratio = Math.exp(lp - q.old);
        const active = algo !== 'grpo' || !((q.A > 0 && ratio > 1 + clip) || (q.A < 0 && ratio < 1 - clip));
        if (active) for (let j = 0; j < g.length; j++) g[j] += q.A * ratio * gl[j] / batch.length;
      }
      step(theta, g);
    }
    S.it++;
    S.cur = evalModel();
    S.hist.push({ it: S.it, er: S.cur.er, klD: S.cur.klD, klS: S.cur.klS, pathKL: pk / batch.length });
  };
  return S;
}

// ---------- Step 5: pathwise gradient (DRaFT, ReFL) ----------
// Gradient of E[r(x_1)] in the parameters at the base model, from a batch of n paths. 'sf' is the
// score-function estimator with the batch mean as baseline; 'pw' backpropagates through the last K
// steps of the sampler (K = KS is the full gradient).
export function gradEstimate(kind, n, rng, K = KS, a = A0, m = MIX1, lam = 1) {
  const P = 3 * m.mu.length, g = new Float64Array(P), paths = [];
  for (let i = 0; i < n; i++) paths.push(path(m, rng.gauss(), lam, rng));
  if (kind === 'sf') {
    const rs = paths.map((p) => reward(p.xs[KS], a)), mean = rs.reduce((s, q) => s + q, 0) / n;
    paths.forEach((p, i) => { const { g: gl } = pathLogp(m, p.xs, lam, true); for (let j = 0; j < P; j++) g[j] += (n > 1 ? (rs[i] - mean) * n / (n - 1) : rs[i]) * gl[j] / n; });
    return g;
  }
  for (const p of paths) {
    // Forward sensitivity dx/dtheta over the last K steps; earlier steps are treated as constants.
    const J = new Float64Array(P);
    for (let k = KS - K; k < KS; k++) {
      const t = TG[k], h = TG[k + 1] - t, e2 = eps2(t, lam), D = drift(m, p.xs[k], t, e2, true);
      for (let j = 0; j < P; j++) J[j] = J[j] * (1 + D.bx * h) + D.g[j] * h;
    }
    const dr = -(p.xs[KS] - a);
    for (let j = 0; j < P; j++) g[j] += dr * J[j] / n;
  }
  return g;
}
export const NAMES = ['logit w₁', 'logit w₂', 'logit w₃', 'μ₁', 'μ₂', 'μ₃', 'log s₁', 'log s₂', 'log s₃'];

// ---------- Step 6: Diffusion-DPO ----------
// Pairs from p_data labelled by a Bradley–Terry draw on r: P(a beats b) = σ(r(a) - r(b)).
export function makePairs(n, seed, a = A0) {
  const rng = makeRng(seed), xs = sampleData(rng, 2 * n), pairs = [];
  for (let i = 0; i < n; i++) {
    const u = xs[2 * i], v = xs[2 * i + 1], pu = 1 / (1 + Math.exp(-(reward(u, a) - reward(v, a))));
    pairs.push(rng.uniform() < pu ? [u, v] : [v, u]);
  }
  return pairs;
}
// Log density of a mixture and its gradient in the nine parameters.
export function logpGrad(m, x) {
  const K = m.mu.length, l = m.mu.map((u, k) => Math.log(m.w[k]) - Math.log(m.sd[k]) - (x - u) ** 2 / (2 * m.sd[k] ** 2));
  const mx = Math.max(...l), e = l.map((q) => Math.exp(q - mx)), Z = e.reduce((a, b) => a + b, 0), r = e.map((q) => q / Z);
  const g = new Float64Array(3 * K);
  for (let k = 0; k < K; k++) {
    const z = (x - m.mu[k]) / m.sd[k];
    g[k] = r[k] - m.w[k]; g[K + k] = r[k] * z / m.sd[k]; g[2 * K + k] = r[k] * (z * z - 1);
  }
  return { lp: mx + Math.log(Z) - 0.5 * Math.log(2 * Math.PI), g };
}
// The ELBO surrogate of log p_theta(x) - log p_ref(x): minus the weighted flow-matching loss difference,
// -E_t[t/(1 - t) (|u_theta - (x - x0)|² - |u_ref - (x - x0)|²)], with t uniform on [0.05, 0.95] and
// fixed seeded draws (t, x0) per point. The weight t/(1 - t) makes it the path KL of the memoryless SDE.
export const NMC = 6;
export function elboDiff(m, x, draws, grad = false) {
  let s = 0; const g = new Float64Array(3 * m.mu.length);
  for (const [t, z] of draws) {
    const xt = t * x + (1 - t) * z, tgt = x - z, wgt = t / (1 - t) * 0.9;
    const Dm = drift(m, xt, t, 0, grad), Dr = fields(MIX1, xt, t).u;
    s -= wgt * ((Dm.b - tgt) ** 2 - (Dr - tgt) ** 2) / draws.length;
    if (grad) for (let j = 0; j < g.length; j++) g[j] -= wgt * 2 * (Dm.b - tgt) * Dm.g[j] / draws.length;
  }
  return { d: s, g };
}
export function makeDPO({ pairs, beta = B0, kind = 'exact', seed = 5, lr = 0.03 }) {
  const theta = toVec(MIX1), step = adam(lr), rng = makeRng(seed);
  const draws = pairs.map(() => [0, 1].map(() => Array.from({ length: NMC }, () => [0.05 + 0.9 * rng.uniform(), rng.gauss()])));
  const lref = pairs.map(([w, l]) => [logpGrad(MIX1, w).lp, logpGrad(MIX1, l).lp]);
  const S = { theta, it: 0, hist: [] };
  const evalModel = () => { const p = normalize(densGrid(fromVec(theta))); return { p, klS: klGrid(p, pStar(A0, beta)), klD: klGrid(p, pData) }; };
  S.cur = evalModel(); S.hist.push({ it: 0, klS: S.cur.klS });
  S.step = () => {
    const m = fromVec(theta), g = new Float64Array(theta.length);
    let loss = 0;
    pairs.forEach(([w, l], i) => {
      let dw, dl;
      if (kind === 'exact') {
        const A = logpGrad(m, w), B = logpGrad(m, l);
        dw = { d: A.lp - lref[i][0], g: A.g }; dl = { d: B.lp - lref[i][1], g: B.g };
      } else { dw = elboDiff(m, w, draws[i][0], true); dl = elboDiff(m, l, draws[i][1], true); }
      const z = beta * (dw.d - dl.d), sg = 1 / (1 + Math.exp(z));     // d/dz of log σ(z) is σ(-z)
      loss += Math.log(1 + Math.exp(-z)) / pairs.length;
      for (let j = 0; j < g.length; j++) g[j] += sg * beta * (dw.g[j] - dl.g[j]) / pairs.length;
    });
    step(theta, g); S.it++; S.cur = evalModel(); S.hist.push({ it: S.it, klS: S.cur.klS, loss });
  };
  return S;
}

// ---------- Step 7: the optimal control for a given noise level ----------
// h_t(x) = E[e^{r(x_1)/beta} | x_t = x] solves the backward equation of the base SDE; the controlled
// SDE started from the fixed start density ends at e^{r/beta} g_1, where g solves the base Fokker–Planck
// equation from g_0 = p_0 / h_0. Both are implicit finite-difference schemes on a grid of 301 points on
// [-7, 7] and 500 steps uniform in log(t / (1 - t)) from 0.01 to 0.9995.
const PX = 301, PL = -7, PR = 7, PD = (PR - PL) / (PX - 1);
export const PXG = Float64Array.from({ length: PX }, (_, i) => PL + PD * i);
const NT = 500, Ta = 0.01, Tb = 0.9995;
const PT = Float64Array.from({ length: NT + 1 }, (_, i) => { const l = Math.log(Ta / (1 - Ta)) + (Math.log(Tb / (1 - Tb)) - Math.log(Ta / (1 - Ta))) * i / NT; return 1 / (1 + Math.exp(-l)); });
function tridiag(a, b, c, d) {
  const n = d.length, cp = new Float64Array(n), dp = new Float64Array(n), x = new Float64Array(n);
  cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0];
  for (let i = 1; i < n; i++) { const den = b[i] - a[i] * cp[i - 1]; cp[i] = c[i] / den; dp[i] = (d[i] - a[i] * dp[i - 1]) / den; }
  x[n - 1] = dp[n - 1]; for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
  return x;
}
export function control(lam, a = A0, beta = B0) {
  const bAt = (t) => { const e2 = eps2(t, lam); return { e2, b: Float64Array.from(PXG, (x) => drift(MIX1, x, t, e2).b) }; };
  // Backward equation: dh/dt + b h' + (eps²/2) h'' = 0, solved in log-scaled form for stability.
  const rmax = Math.max(...Array.from(PXG, (x) => reward(x, a) / beta));
  let h = Float64Array.from(PXG, (x) => Math.exp(reward(x, a) / beta - rmax));
  for (let k = NT - 1; k >= 0; k--) {
    const t = PT[k], dt = PT[k + 1] - t, { e2, b } = bAt(t), A = new Float64Array(PX), B = new Float64Array(PX), C = new Float64Array(PX);
    for (let i = 0; i < PX; i++) {
      const D = 0.5 * e2 / (PD * PD), up = b[i] > 0 ? b[i] / PD : 0, dn = b[i] < 0 ? -b[i] / PD : 0;
      A[i] = -dt * (D + dn); C[i] = -dt * (D + up); B[i] = 1 + dt * (2 * D + up + dn);
    }
    B[0] += A[0]; A[0] = 0; B[PX - 1] += C[PX - 1]; C[PX - 1] = 0;           // zero slope at the ends
    h = tridiag(A, B, C, h);
  }
  // Forward equation of the base SDE in flux form from g_0 = p_start / h_0.
  const p0 = Float64Array.from(PXG, (x) => Math.exp(fields(MIX1, x, Ta).logp));
  let g = Float64Array.from(PXG, (_, i) => p0[i] / h[i]);
  for (let k = 0; k < NT; k++) {
    const t = PT[k + 1], dt = PT[k + 1] - PT[k], { e2, b } = bAt(t), A = new Float64Array(PX), B = new Float64Array(PX).fill(1), C = new Float64Array(PX);
    const D = 0.5 * e2 / (PD * PD);
    for (let i = 0; i < PX - 1; i++) {
      // Flux between i and i + 1: upwind advection plus diffusion.
      const bf = 0.5 * (b[i] + b[i + 1]), fp = Math.max(bf, 0) / PD, fm = Math.min(bf, 0) / PD;
      B[i] += dt * (fp + D); C[i] += dt * (fm - D);
      A[i + 1] += dt * (-fp - D); B[i + 1] += dt * (-fm + D);
    }
    g = tridiag(A, B, C, g);
  }
  const q = Float64Array.from(PXG, (x, i) => Math.exp(reward(x, a) / beta - rmax) * g[i]);
  const Z = q.reduce((s, v) => s + v, 0) * PD; q.forEach((_, i) => { q[i] /= Z; });
  const ps = Float64Array.from(PXG, (x) => Math.exp(fields(tiltQuad(MIX1, a, beta), x, 1).logp));
  let F = 0, G = 0, w1 = 0; for (let i = 0; i < PX; i++) { F += q[i] * PD; G += ps[i] * PD; w1 += Math.abs(F - G) * PD; }
  // Spread of log h_0 over the start density: its standard deviation.
  const lh = Array.from(h, Math.log), mh = lh.reduce((s, v, i) => s + v * p0[i] * PD, 0), sdh = Math.sqrt(lh.reduce((s, v, i) => s + (v - mh) ** 2 * p0[i] * PD, 0));
  return { q, ps, h0: h, p0, w1, sdh };
}

// ---------- Step 3 demo ----------
export const LAMS = [0, 0.25, 0.5, 1, 1.5], X3 = 0.15;
export function sdeDemo(lam) {
  const rng = makeRng(7), paths = Array.from({ length: 40 }, () => path(MIX1, X3, lam, rng));
  const ode = path(MIX1, X3, 0, rng), xs = ends(MIX1, lam, 2000, 13);
  const lp = paths[0].lp, lpSum = lam > 0 ? lp.reduce((s, v) => s + v, 0) : NaN;
  return { paths, ode, xs, w1: w1Samples(xs, MIX1), lp, lpSum };
}
// ---------- Step 5: estimator study ----------
export const KTR = [1, 2, 5, 10, 20, 40], BATCH = [1, 2, 4, 8, 16, 32];
// Pathwise gradients truncated to the last K steps, for every K in KTR, from one set of paths.
export function gradAllK(n, rng, a = A0, m = MIX1, lam = 1) {
  const P = 3 * m.mu.length, out = KTR.map(() => new Float64Array(P));
  for (let i = 0; i < n; i++) {
    const p = path(m, rng.gauss(), lam, rng), D = [];
    for (let k = 0; k < KS; k++) { const t = TG[k]; D.push(drift(m, p.xs[k], t, eps2(t, lam), true)); }
    const dr = -(p.xs[KS] - a);
    KTR.forEach((K, ki) => {
      const J = new Float64Array(P);
      for (let k = KS - K; k < KS; k++) { const h = TG[k + 1] - TG[k]; for (let j = 0; j < P; j++) J[j] = J[j] * (1 + D[k].bx * h) + D[k].g[j] * h; }
      for (let j = 0; j < P; j++) out[ki][j] += dr * J[j] / n;
    });
  }
  return out;
}
