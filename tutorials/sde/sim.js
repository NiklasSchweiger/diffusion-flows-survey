// Densities, fields and samplers of the §2.3 tutorial. No DOM access, so the numbers can be checked in Node.
// Our convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data. The forward (corruption)
// process runs in tau = 1 - t, y_tau = x_{1 - tau}. All fields are exact because the data are Gaussian mixtures.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, MIX2, gmm2, W2, p2 } from '../score-matching/sim.js';

export { MIX1, MIX2, gmm2, W2, p2 };

// ---------- Schedules ----------
// Each gives alpha, sigma and their derivatives in t. lam is the log-SNR log(alpha^2 / sigma^2).
const sched = (alpha, sigma, dalpha, dsigma) => {
  const s = { alpha, sigma, dalpha, dsigma };
  s.lam = (t) => 2 * Math.log(alpha(t) / sigma(t));
  s.dlam = (t) => 2 * (dalpha(t) / alpha(t) - dsigma(t) / sigma(t));
  s.f = (t) => -dalpha(t) / alpha(t);                       // drift coefficient of the forward SDE, eq. (fg)
  // Squared diffusion coefficient sigma^2 lambda', eq. (fg), written so that it stays finite at sigma = 0.
  s.g2 = (t) => 2 * (sigma(t) ** 2 * dalpha(t) / alpha(t) - sigma(t) * dsigma(t));
  return s;
};
const H = Math.PI / 2;
// Ratio r = sigma_t / alpha_t of the variance-exploding schedule runs geometrically from R_MAX to R_MIN.
export const R_MAX = 4, R_MIN = 0.03;
export const SCHEDULES = {
  linear: sched((t) => t, (t) => 1 - t, () => 1, () => -1),
  cosine: sched((t) => Math.sin(H * t), (t) => Math.cos(H * t), (t) => H * Math.cos(H * t), (t) => -H * Math.sin(H * t)),
  ve: sched(() => 1, (t) => R_MAX ** (1 - t) * R_MIN ** t, () => 0, (t) => Math.log(R_MIN / R_MAX) * R_MAX ** (1 - t) * R_MIN ** t),
};
export const LIN = SCHEDULES.linear;
// The forward process stops at t = T_MIN (tau = 1 - T_MIN), where p_t is close to N(0, 1).
export const T_MIN = 0.03;

// ---------- 1D mixture under a schedule ----------
const phi = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
const comps = (t, S) => { const a = S.alpha(t), s = S.sigma(t); return MIX1.mu.map((m, k) => [MIX1.w[k], a * m, a * a * MIX1.sd[k] ** 2 + s * s]); };
// Density, first and second x-derivative and score of p_t.
export function dens1(x, t, S = LIN) {
  let p = 0, d1 = 0, d2 = 0;
  for (const [w, m, v] of comps(t, S)) { const f = w * phi(x, m, v), u = (x - m) / v; p += f; d1 -= f * u; d2 += f * (u * u - 1 / v); }
  return { p, d1, d2, score: d1 / Math.max(p, 1e-300) };
}
export const pt1 = (x, t, S = LIN) => dens1(x, t, S).p;
export const scoret1 = (x, t, S = LIN) => dens1(x, t, S).score;
// Time derivative of rho_tau = p_{1 - tau}, by a central difference of the exact density.
export const dtau1 = (x, tau, S = LIN, h = 1e-4) => -(pt1(x, 1 - tau + h, S) - pt1(x, 1 - tau - h, S)) / (2 * h);
// The two terms of Fokker-Planck at time tau for the linear forward SDE with b = f y.
export function fpTerms(x, tau, S = LIN) {
  const t = 1 - tau, f = S.f(t), g2 = S.g2(t), { p, d1, d2, score } = dens1(x, t, S);
  return {
    p, score, f, g2,
    transport: -(f * p + f * x * d1),        // -d/dy (b rho)
    diffusion: 0.5 * g2 * d2,                // 1/2 g^2 rho''
    b: f * x,                                // drift
    vdiff: -0.5 * g2 * score,                // diffusion written as a velocity, -1/2 g^2 d/dy log rho
  };
}
// Velocity of the probability-flow ODE in t (our convention): (alpha'/alpha) x + 1/2 sigma^2 lambda' score.
export const pfVel1 = (x, t, S = LIN) => (S.dalpha(t) / S.alpha(t)) * x + 0.5 * S.g2(t) * scoret1(x, t, S);
export function sample1(rng) {
  const u = rng.uniform(); let k = 0, c = MIX1.w[0];
  while (u > c && k < 2) c += MIX1.w[++k];
  return MIX1.mu[k] + MIX1.sd[k] * rng.gauss();
}
// Quantiles of p_t by bisection on the exact CDF.
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
function erf(x) { const s = Math.sign(x), a = Math.abs(x), k = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * Math.exp(-a * a)); }
export function quantile1(q, t, S = LIN) {
  const cs = comps(t, S), F = (x) => cs.reduce((a, [w, m, v]) => a + w * Phi((x - m) / Math.sqrt(v)), 0);
  let lo = -20, hi = 20;
  for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (F(mid) < q) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}

// ---------- Steps 1 and 4: forward SDE paths and probability-flow paths, in tau ----------
// Grid tau_j = j / NT * (1 - T_MIN). The SDE uses its exact Gaussian transition between grid points,
// x_{t'} | x_t ~ N((alpha'/alpha) x_t, sigma'^2 - (alpha'/alpha)^2 sigma^2), so the paths carry no discretization error.
export const NT = 160, TAU_MAX = 1 - T_MIN;
export const TAUS = Float64Array.from({ length: NT + 1 }, (_, j) => TAU_MAX * j / NT);
export const FWD = (() => {
  const r = makeRng(55), n = 1000, y = Array.from({ length: n }, () => new Float64Array(NT + 1));
  for (let i = 0; i < n; i++) {
    y[i][0] = quantile1((i + 0.5) / n, 1);   // stratified start, so the histograms are not noisy at tau = 0
    for (let j = 1; j <= NT; j++) {
      const t0 = 1 - TAUS[j - 1], t1 = 1 - TAUS[j], q = LIN.alpha(t1) / LIN.alpha(t0);
      y[i][j] = q * y[i][j - 1] + Math.sqrt(Math.max(0, LIN.sigma(t1) ** 2 - q * q * LIN.sigma(t0) ** 2)) * r.gauss();
    }
  }
  return { n, y };
})();
// Probability-flow paths from the 2%, 4%, ..., 98% quantiles of the data, by RK4 with 8 substeps per grid interval.
export const PF = (() => {
  const qs = Array.from({ length: 49 }, (_, i) => 0.02 * (i + 1));
  const y = qs.map((q) => { const a = new Float64Array(NT + 1); a[0] = quantile1(q, 1); return a; });
  const M = 8;
  for (const a of y) {
    let x = a[0];
    for (let j = 1; j <= NT; j++) {
      let t = 1 - TAUS[j - 1]; const h = -(TAUS[j] - TAUS[j - 1]) / M;   // t decreases
      for (let m = 0; m < M; m++) {
        const k1 = pfVel1(x, t), k2 = pfVel1(x + h / 2 * k1, t + h / 2), k3 = pfVel1(x + h / 2 * k2, t + h / 2), k4 = pfVel1(x + h * k3, t + h);
        x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += h;
      }
      a[j] = x;
    }
  }
  return { qs, y };
})();

// ---------- Step 5: the reverse family from noise to data, in t ----------
// dx = (u_t(x) + 1/2 eta^2 g_t^2 score) dt + eta g_t dW. eta = 0 is the probability-flow ODE, eta = 1 the
// reverse SDE. Starts from stratified quantiles of the exact p_{T_MIN}; Euler-Maruyama on NR steps, fine enough that the step error is invisible.
export const NR = 400;
export const TR = Float64Array.from({ length: NR + 1 }, (_, j) => T_MIN + (1 - T_MIN) * j / NR);
const REV = (() => { const r = makeRng(75), n = 1000; return { n, x0: Float64Array.from({ length: n }, (_, i) => quantile1((i + 0.5) / n, T_MIN)), xi: Float64Array.from({ length: n * NR }, () => r.gauss()) }; })();
export function reverse1(eta) {
  const { n, x0, xi } = REV, x = Array.from({ length: n }, () => new Float64Array(NR + 1));
  for (let i = 0; i < n; i++) {
    let v = x0[i]; x[i][0] = v;
    for (let j = 0; j < NR; j++) {
      const t = TR[j], dt = TR[j + 1] - t, g2 = LIN.g2(t);
      v += (pfVel1(v, t) + 0.5 * eta * eta * g2 * scoret1(v, t)) * dt + eta * Math.sqrt(g2 * dt) * xi[i * NR + j];
      x[i][j + 1] = v;
    }
  }
  return { n, x };
}

// ---------- Step 6: samplers in our convention, 2D ----------
// K Euler (ODE) or Euler-Maruyama (SDE) steps in t on a grid that is uniform in the log-SNR lambda,
// from x ~ N(0, I) at t = T0 to t = T1. Fields are exact (gmm.js uses the linear schedule).
export const T0 = 0.02, T1 = 0.995;
export function gridLam(K) {
  const l0 = LIN.lam(T0), l1 = LIN.lam(T1);
  return Float64Array.from({ length: K + 1 }, (_, k) => { const l = l0 + (l1 - l0) * k / K, e = Math.exp(l / 2); return e / (1 + e); });
}
const S2 = (() => { const r = makeRng(81), n = 500, K = 400; return { n, K, x0: Float64Array.from({ length: 2 * n }, () => r.gauss()), xi: Float64Array.from({ length: 2 * n * K }, () => r.gauss()) }; })();
export const N2 = S2.n;
// Returns positions at every grid point, path[k] = Float64Array(2n).
export function sample2(K, sde) {
  const ts = gridLam(K), n = S2.n, path = [Float64Array.from(S2.x0)];
  for (let k = 0; k < K; k++) {
    const t = ts[k], dt = ts[k + 1] - t, cur = path[k], nxt = new Float64Array(2 * n);
    const ra = LIN.dalpha(t) / LIN.alpha(t), c = LIN.sigma(t) ** 2 * LIN.dlam(t), nz = LIN.sigma(t) * Math.sqrt(LIN.dlam(t) * dt);
    for (let i = 0; i < n; i++) {
      const x = cur[2 * i], y = cur[2 * i + 1], sc = gmm2.fields(x, y, t).score;
      if (sde) {
        // Eq. (reverse-sde-ours): drift (alpha'/alpha) x + sigma^2 lambda' score, noise sigma sqrt(lambda') dW.
        // The noise draws are indexed by the time of the step, so K = 400 reuses the first draws for coarser grids.
        const j = Math.min(S2.K - 1, Math.floor(k * S2.K / K));
        nxt[2 * i] = x + (ra * x + c * sc[0]) * dt + nz * S2.xi[2 * (i * S2.K + j)];
        nxt[2 * i + 1] = y + (ra * y + c * sc[1]) * dt + nz * S2.xi[2 * (i * S2.K + j) + 1];
      } else {
        nxt[2 * i] = x + (ra * x + 0.5 * c * sc[0]) * dt;
        nxt[2 * i + 1] = y + (ra * y + 0.5 * c * sc[1]) * dt;
      }
    }
    path.push(nxt);
  }
  return { ts, path };
}
// Quality of a set of 2D samples: mass per component (by responsibility at t = 1) and mean log p.
export function quality2(xy) {
  const n = xy.length / 2, mass = [0, 0, 0]; let ll = 0;
  for (let i = 0; i < n; i++) {
    const x = xy[2 * i], y = xy[2 * i + 1];
    mass[gmm2.argmaxComponent(x, y, 1)] += 1 / n;
    ll += Math.log(Math.max(p2(x, y), 1e-300)) / n;
  }
  return { mass, ll };
}
// Mean log p of exact data samples, for comparison.
export const LL_EXACT = (() => { const r = makeRng(82); let ll = 0; const n = 4000; for (let i = 0; i < n; i++) { const s = gmm2.sample(r); ll += Math.log(p2(s.x, s.y)) / n; } return ll; })();

// ---------- Step 7: schedules as reparametrizations (1D) ----------
// In y = x_t / alpha_t and r = sigma_t / alpha_t the probability-flow ODE is dy/dr = -r d/dy log p(y; r),
// with p(y; r) = p_data * N(0, r^2). Paths from quantiles of p(y; R_MAX), integrated in log r.
const dens1r = (y, r) => { let p = 0, d1 = 0; MIX1.mu.forEach((m, k) => { const v = MIX1.sd[k] ** 2 + r * r, f = MIX1.w[k] * phi(y, m, v); p += f; d1 -= f * (y - m) / v; }); return d1 / Math.max(p, 1e-300); };
export const NE = 200;
export const LOGR = Float64Array.from({ length: NE + 1 }, (_, j) => Math.log(R_MAX) + (Math.log(R_MIN) - Math.log(R_MAX)) * j / NE);
export const EDM = (() => {
  const qs = Array.from({ length: 31 }, (_, i) => 0.03 + 0.94 * i / 30);
  // Quantiles of p(y; R_MAX): the 1D mixture with variances sd_k^2 + R_MAX^2.
  const F = (y) => MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * Phi((y - m) / Math.sqrt(MIX1.sd[k] ** 2 + R_MAX ** 2)), 0);
  const q0 = (q) => { let lo = -40, hi = 40; for (let i = 0; i < 60; i++) { const mid = (lo + hi) / 2; if (F(mid) < q) lo = mid; else hi = mid; } return (lo + hi) / 2; };
  // dy/d(log r) = r dy/dr = -r^2 score(y; r).
  const vel = (y, l) => { const r = Math.exp(l); return -r * r * dens1r(y, r); };
  const y = qs.map((q) => {
    const a = new Float64Array(NE + 1); let v = a[0] = q0(q);
    for (let j = 1; j <= NE; j++) {
      const M = 6, h = (LOGR[j] - LOGR[j - 1]) / M; let l = LOGR[j - 1];
      for (let m = 0; m < M; m++) { const k1 = vel(v, l), k2 = vel(v + h / 2 * k1, l + h / 2), k3 = vel(v + h / 2 * k2, l + h / 2), k4 = vel(v + h * k3, l + h); v += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); l += h; }
      a[j] = v;
    }
    return a;
  });
  return { qs, y };
})();
// t at which a schedule reaches the ratio r = sigma_t / alpha_t (the ratio decreases in t), by bisection.
export function tOfRatio(S, r) {
  let lo = 1e-6, hi = 1 - 1e-9;
  for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (S.sigma(mid) / S.alpha(mid) > r) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
