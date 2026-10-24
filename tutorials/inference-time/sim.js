// Numerics of the §6 tutorial. The base model is the exact field of the 1D mixture MIX1 (steps 1, 2, 6
// and 7) or of the 2D mixture MIX2 (steps 3 to 5) on the linear path x_t = t x_1 + (1 - t) x_0. Compute
// is counted in evaluations of the field (NFE). Steps 1, 2 and 4 and the systematic verifier of step 7
// are exact; search, SMC and the noisy verifier are Monte Carlo over seeded runs. No DOM access.
import { MIX1, XG, DX, NG, fields, cdf, strat, densGrid, w1Samples, normalize, klGrid, meanGrid } from '../lib/mix1d.js';
import { makeRng } from '../../animations/lib/rng.js';
import { MIX2 } from '../score-matching/sim.js';

export { MIX1, XG, DX, MIX2 };
export const pData = densGrid(MIX1);
export const NS = 2000;
export const Z0 = strat(MIX1, NS, 0);
export const w1 = (xs) => w1Samples(xs, MIX1);

// W1 between a sample set and a density on the grid (integral of |F_n - F|).
export function w1ToGrid(xs, p) {
  const a = Float64Array.from(xs).sort(), n = a.length; let j = 0, F = 0, s = 0;
  for (let i = 0; i < NG; i++) { F += p[i] * DX; const x = XG[i] + DX / 2; while (j < n && a[j] <= x) j++; s += Math.abs(j / n - F); }
  return s * DX;
}
// W1 between two densities on the grid.
export function w1Dens(p, q) { let F = 0, G = 0, s = 0; for (let i = 0; i < NG; i++) { F += p[i] * DX; G += q[i] * DX; s += Math.abs(F - G); } return s * DX; }

// ---------- Step 1: solvers ----------
// K + 1 grid times from 0 to 1. The log-SNR grid is uniform in lambda = log(t / (1 - t)) on [-L, L],
// with its two end points moved to 0 and 1.
export const LSNR = 4;
export function grid(K, kind) {
  if (kind === 'uniform') return Float64Array.from({ length: K + 1 }, (_, i) => i / K);
  return Float64Array.from({ length: K + 1 }, (_, i) => (i === 0 ? 0 : i === K ? 1 : 1 / (1 + Math.exp(LSNR - 2 * LSNR * i / K))));
}
const u1 = (x, t) => fields(MIX1, x, t).u;
// Euler: one NFE per step. On the linear path it coincides with DDIM.
export function solveEuler(x, ts) { for (let i = 0; i + 1 < ts.length; i++) x += (ts[i + 1] - ts[i]) * u1(x, ts[i]); return x; }
// Heun: two NFE per step.
export function solveHeun(x, ts) {
  for (let i = 0; i + 1 < ts.length; i++) {
    const h = ts[i + 1] - ts[i], d1 = u1(x, ts[i]), d2 = u1(x + h * d1, ts[i + 1]);
    x += 0.5 * h * (d1 + d2);
  }
  return x;
}
// DPM-Solver++(2M) in data prediction with alpha = t, sigma = 1 - t: one NFE per step, second order
// from the previous prediction, first order in the first step and in the last step to sigma = 0.
export function solveDPM(x, ts) {
  const n = ts.length - 1, lam = (t) => Math.log(t / (1 - t));
  let prev = 0;
  for (let i = 0; i < n; i++) {
    const t = ts[i], t2 = ts[i + 1], xh = fields(MIX1, x, t).xhat;
    let D = xh;
    if (i > 0 && i < n - 1) {
      const r = (lam(t) - lam(ts[i - 1])) / (lam(t2) - lam(t));
      if (Number.isFinite(r)) D = (1 + 1 / (2 * r)) * xh - (1 / (2 * r)) * prev;
    }
    const ratio = (1 - t2) / (1 - t), eh = t === 0 || t2 === 1 ? 0 : ((1 - t2) * t) / (t2 * (1 - t));
    x = ratio * x - t2 * (eh - 1) * D;
    prev = xh;
  }
  return x;
}
export const SOLVERS = { euler: [solveEuler, 1], heun: [solveHeun, 2], dpm: [solveDPM, 1] };
export const NFE1 = [2, 4, 8, 16, 32, 64];
const cache1 = new Map();
export function solverW1(name, kind) {
  const key = name + kind;
  if (!cache1.has(key)) {
    const [f, per] = SOLVERS[name];
    cache1.set(key, NFE1.map((nfe) => { const ts = grid(nfe / per, kind); return w1(Z0.map((z) => f(z, ts))); }));
  }
  return cache1.get(key);
}
export const W1_EXACT = w1(strat(MIX1, NS, 1));

// ---------- Rewards ----------
// A narrow bump between the left and the middle mode, where p_data has little mass.
export const RA = -0.65, RW = 0.22;
export const reward = (x) => Math.exp(-((x - RA) ** 2) / (2 * RW * RW));
export const dreward = (x) => -((x - RA) / (RW * RW)) * reward(x);
export const rGrid = Float64Array.from(XG, reward);
export const tilt = (beta) => normalize(Float64Array.from(pData, (p, i) => p * Math.exp(rGrid[i] / beta)));

// ---------- Step 2: best-of-N ----------
// F_r(r(x)) = P(r(X) <= r(x)) = 1 - P(|X - a| < |x - a|) for the bump reward.
const Fr = Float64Array.from(XG, (x) => { const d = Math.abs(x - RA); return 1 - (cdf(MIX1, RA + d) - cdf(MIX1, RA - d)); });
export const bestOfN = (N) => normalize(Float64Array.from(pData, (p, i) => N * p * Fr[i] ** (N - 1)));
export const klBoN = (N) => Math.log(N) - (N - 1) / N;
export const NS2 = [1, 2, 4, 8, 16, 32, 64, 128];
// The beta at which KL(p*_beta || p_data) equals a given value, by bisection in log beta.
export function betaForKL(kl) {
  if (kl <= 0) return Infinity;
  let lo = Math.log(1e-4), hi = Math.log(1e4);
  for (let i = 0; i < 60; i++) { const m = 0.5 * (lo + hi); if (klGrid(tilt(Math.exp(m)), pData) > kl) lo = m; else hi = m; }
  return Math.exp(0.5 * (lo + hi));
}
export const meanR = (p) => meanGrid(p, reward);
export { klGrid };

// ---------- 2D fields of MIX2 (isotropic components) ----------
const S2 = MIX2.std ** 2, MU2 = MIX2.means, WT2 = (() => { const z = MIX2.w.reduce((a, b) => a + b, 0); return MIX2.w.map((w) => w / z); })();
// Velocity and E[x_1 | x_t] at (x, y, t).
export function f2(x, y, t) {
  const v = t * t * S2 + (1 - t) ** 2, lw = MU2.map(([mx, my], k) => Math.log(WT2[k]) - ((x - t * mx) ** 2 + (y - t * my) ** 2) / (2 * v));
  const m = Math.max(...lw), r = lw.map((l) => Math.exp(l - m)), Z = r.reduce((a, b) => a + b, 0);
  let ux = 0, uy = 0, hx = 0, hy = 0; const c = (t * S2 - (1 - t)) / v, g = (t * S2) / v;
  MU2.forEach(([mx, my], k) => {
    const q = r[k] / Z, dx = x - t * mx, dy = y - t * my;
    ux += q * (mx + c * dx); uy += q * (my + c * dy); hx += q * (mx + g * dx); hy += q * (my + g * dy);
  });
  return { ux, uy, hx, hy };
}
export function p2data(x, y) {
  return MU2.reduce((a, [mx, my], k) => a + WT2[k] * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * S2)) / (2 * Math.PI * S2), 0);
}
// RK4 flow map from t = 0 to 1 with K steps (4K NFE).
export const K3 = 10;
export function flow2(x, y, K = K3) {
  const h = 1 / K;
  for (let i = 0; i < K; i++) {
    const t = i * h, a = f2(x, y, t), b = f2(x + 0.5 * h * a.ux, y + 0.5 * h * a.uy, t + 0.5 * h);
    const c = f2(x + 0.5 * h * b.ux, y + 0.5 * h * b.uy, t + 0.5 * h), d = f2(x + h * c.ux, y + h * c.uy, t + h);
    x += (h / 6) * (a.ux + 2 * b.ux + 2 * c.ux + d.ux); y += (h / 6) * (a.uy + 2 * b.uy + 2 * c.uy + d.uy);
  }
  return [x, y];
}

// ---------- Step 3: searching the noise ----------
// Two rewards. 'far' pulls towards a point a beyond the data, r(x) = -|x - a|² / 2. 'rare' is a narrow
// bump of width 0.15 at b, on the edge of the right mode, where about one data sample in 590 has r > 0.5.
export const A3 = [2.4, 1.5], B3 = [2.3, -0.3], W3 = 0.15;
export const REW3 = {
  far: { f: ([x, y]) => -0.5 * ((x - A3[0]) ** 2 + (y - A3[1]) ** 2), grad: ([x, y]) => [A3[0] - x, A3[1] - y], at: A3 },
  rare: {
    f: ([x, y]) => Math.exp(-((x - B3[0]) ** 2 + (y - B3[1]) ** 2) / (2 * W3 * W3)),
    grad: (p) => { const r = REW3.rare.f(p); return [(r * (B3[0] - p[0])) / (W3 * W3), (r * (B3[1] - p[1])) / (W3 * W3)]; },
    at: B3,
  },
};
export const reward2 = REW3.far.f;
export const NFE_EVAL3 = 4 * K3;
export const LIM3 = 4.5, NG3 = 64;
const pull3 = {};
// The reward pulled back to the noise plane, on a grid over [-LIM3, LIM3]².
export function pullback3(rk = 'far') {
  if (pull3[rk]) return pull3[rk];
  const r = new Float64Array(NG3 * NG3), f = REW3[rk].f;
  for (let j = 0; j < NG3; j++) for (let i = 0; i < NG3; i++) {
    const x = -LIM3 + (2 * LIM3) * (i + 0.5) / NG3, y = -LIM3 + (2 * LIM3) * (j + 0.5) / NG3;
    r[j * NG3 + i] = f(flow2(x, y));
  }
  return (pull3[rk] = { r });
}
// Random search: n seeded draws of the noise. Local search: from the first draw, Gaussian proposals of
// size rho, kept when they raise the reward. Both return the visited points in order.
export function search3(kind, n, rho = 0.4, seed = 5, rk = 'far') {
  if (kind === 'trs') return trs3(n, seed, rk);
  const rng = makeRng(seed), pts = [], f = REW3[rk].f;
  let best = null;
  const visit = (z) => { const x = flow2(z[0], z[1]), r = f(x), p = { z, x, r }; pts.push(p); if (!best || r > best.r) best = p; return p; };
  let cur = visit([rng.gauss(), rng.gauss()]);
  for (let i = 1; i < n; i++) {
    if (kind === 'random') visit([rng.gauss(), rng.gauss()]);
    else { const p = visit([cur.z[0] + rho * rng.gauss(), cur.z[1] + rho * rng.gauss()]); if (p.r > cur.r) cur = p; }
  }
  return { pts, best };
}
// Trust-region search (as in animation A13): a random warm-up on 20% of the budget, then k boxes centred
// on the k best noises so far. Each box proposes masked uniform steps inside it, grows by alpha after cs
// rounds with an improvement, shrinks after cf rounds without, and restarts at l0 when it fails at lmin.
export const TRS3 = { k: 3, per: 2, warm: 0.2, l0: 0.8, lmin: 0.05, lmax: 2.4, alpha: 1.5, cs: 3, cf: 3, pmin: 0.1, pmax: 0.9 };
function trs3(n, seed, rk) {
  const P = TRS3, rng = makeRng(seed), pts = [], f = REW3[rk].f;
  const visit = (z) => { const x = flow2(z[0], z[1]), p = { z, x, r: f(x) }; pts.push(p); return p; };
  const nw = Math.min(n, Math.max(P.k, Math.round(P.warm * n)));
  for (let i = 0; i < nw; i++) visit([rng.gauss(), rng.gauss()]);
  const reg = Array.from({ length: P.k }, () => ({ l: P.l0, s: 0, f: 0, c: null }));
  while (pts.length < n) {
    const top = [...pts].sort((a, b) => b.r - a.r).slice(0, P.k);
    for (let j = 0; j < P.k; j++) {
      const R = reg[j], ctr = top[j % top.length]; let imp = false; R.c = ctr.z;
      for (let b = 0; b < P.per && pts.length < n; b++) {
        let m; do { const q = P.pmin + (P.pmax - P.pmin) * rng.uniform(); m = [rng.uniform() < q, rng.uniform() < q]; } while (!m[0] && !m[1]);
        const p = visit(ctr.z.map((v, d) => v + (m[d] ? (rng.uniform() - 0.5) * R.l : 0))); if (p.r > ctr.r) imp = true;
      }
      if (imp) { R.s++; R.f = 0; } else { R.f++; R.s = 0; }
      if (R.s >= P.cs) { R.l = Math.min(R.l * P.alpha, P.lmax); R.s = 0; }
      if (R.f >= P.cf) { R.l = R.l <= P.lmin + 1e-9 ? P.l0 : Math.max(R.l / P.alpha, P.lmin); R.f = 0; }
    }
  }
  const best = pts.reduce((a, b) => (b.r > a.r ? b : a));
  return { pts, best, boxes: reg.filter((R) => R.c).map((R) => ({ c: R.c, l: R.l })), warm: nw };
}
// Probability that a standard 2D Gaussian has a larger norm than z.
export const tail2 = (z) => Math.exp(-(z[0] ** 2 + z[1] ** 2) / 2);

// ---------- Step 4: optimizing through the flow ----------
// The velocity of MIX2 and its Jacobian, grad u = c I + t (1 - c t) / v Cov_q(m), with q the posterior
// weights of the components and m their means.
function uJ(x, y, t) {
  const v = t * t * S2 + (1 - t) ** 2, c = (t * S2 - (1 - t)) / v;
  const lw = MU2.map(([mx, my], k) => Math.log(WT2[k]) - ((x - t * mx) ** 2 + (y - t * my) ** 2) / (2 * v));
  const m = Math.max(...lw), e = lw.map((l) => Math.exp(l - m)), Z = e.reduce((a, b) => a + b, 0), q = e.map((a) => a / Z);
  let mbx = 0, mby = 0, ux = 0, uy = 0, cxx = 0, cxy = 0, cyy = 0;
  MU2.forEach(([mx, my], k) => { mbx += q[k] * mx; mby += q[k] * my; ux += q[k] * (mx + c * (x - t * mx)); uy += q[k] * (my + c * (y - t * my)); });
  MU2.forEach(([mx, my], k) => { const a = mx - mbx, b = my - mby; cxx += q[k] * a * a; cxy += q[k] * a * b; cyy += q[k] * b * b; });
  const s = (t * (1 - c * t)) / v;
  return { ux, uy, J: [c + s * cxx, s * cxy, s * cxy, c + s * cyy] };
}
const mm = (A, B) => [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]];
const madd = (A, B, s) => A.map((a, i) => a + s * B[i]);
// RK4 flow map and its 2 × 2 Jacobian dX/dx_0, carried along the solve (the tangent equation dP = J P dt).
function flowJ([x, y], K = K3) {
  const h = 1 / K; let P = [1, 0, 0, 1];
  for (let i = 0; i < K; i++) {
    const t = i * h, a = uJ(x, y, t), Pa = mm(a.J, P);
    const b = uJ(x + (h / 2) * a.ux, y + (h / 2) * a.uy, t + h / 2), Pb = mm(b.J, madd(P, Pa, h / 2));
    const c = uJ(x + (h / 2) * b.ux, y + (h / 2) * b.uy, t + h / 2), Pc = mm(c.J, madd(P, Pb, h / 2));
    const d = uJ(x + h * c.ux, y + h * c.uy, t + h), Pd = mm(d.J, madd(P, Pc, h));
    x += (h / 6) * (a.ux + 2 * b.ux + 2 * c.ux + d.ux); y += (h / 6) * (a.uy + 2 * b.uy + 2 * c.uy + d.uy);
    P = P.map((p, k) => p + (h / 6) * (Pa[k] + 2 * Pb[k] + 2 * Pc[k] + Pd[k]));
  }
  return { x: [x, y], P };
}
// The RK4 path from a noise, K + 1 points.
export function path2([x, y], K = K3) {
  const h = 1 / K, out = [[x, y]];
  for (let i = 0; i < K; i++) {
    const t = i * h, a = f2(x, y, t), b = f2(x + 0.5 * h * a.ux, y + 0.5 * h * a.uy, t + 0.5 * h);
    const c = f2(x + 0.5 * h * b.ux, y + 0.5 * h * b.uy, t + 0.5 * h), d = f2(x + h * c.ux, y + h * c.uy, t + h);
    x += (h / 6) * (a.ux + 2 * b.ux + 2 * c.ux + d.ux); y += (h / 6) * (a.uy + 2 * b.uy + 2 * c.uy + d.uy); out.push([x, y]);
  }
  return out;
}
// Both methods start from the first noise of the searches of step 3. One iteration costs a forward solve
// and a backward pass, counted as 2 × 40 NFE (a vector-Jacobian product costs about one evaluation).
export const ZG = (() => { const r = makeRng(5); return [r.gauss(), r.gauss()]; })();
export const NFE_ITG = 2 * NFE_EVAL3, KOC = 40;
// D-Flow: gradient ascent on the noise for r(X(z)) - beta |z|² / 2, with a fixed step eta.
export function dflowG(iters, beta, rk = 'far', eta = 0.1) {
  const R = REW3[rk]; let z = ZG.slice(); const zs = [z];
  for (let i = 0; i < iters; i++) {
    const { x, P } = flowJ(z), g = R.grad(x);
    z = [z[0] + eta * (P[0] * g[0] + P[2] * g[1] - beta * z[0]), z[1] + eta * (P[1] * g[0] + P[3] * g[1] - beta * z[1])]; zs.push(z);
  }
  const xs = path2(z), x = xs[xs.length - 1];
  return { z, zs, xs, x, r: R.f(x), p: p2data(x[0], x[1]) };
}
// OC-Flow: Euler with KOC steps and a control theta_k added to the velocity at each step, maximizing
// r(x_1) - beta/2 sum_k h |theta_k|². The costate mu runs backward from grad r(x_1) through
// mu_k = (I + h grad u_k)^T mu_{k+1}, and each control moves along mu_{k+1} - beta theta_k.
export function ocflowG(iters, beta, rk = 'far', eta = 0.5) {
  const R = REW3[rk], K = KOC, h = 1 / K, th = Array.from({ length: K }, () => [0, 0]);
  let res, free;
  for (let it = 0; it <= iters; it++) {
    const xs = [ZG.slice()], Js = [];
    for (let k = 0; k < K; k++) { const [x, y] = xs[k], a = uJ(x, y, k * h); Js.push(a.J); xs.push([x + h * (a.ux + th[k][0]), y + h * (a.uy + th[k][1])]); }
    const x = xs[K];
    res = { xs, x, r: R.f(x), p: p2data(x[0], x[1]), E: th.reduce((s, q) => s + 0.5 * h * (q[0] ** 2 + q[1] ** 2), 0) };
    if (!it) free = xs;
    if (it === iters) break;
    let mu = R.grad(x);
    for (let k = K - 1; k >= 0; k--) {
      const J = Js[k], prev = [mu[0] + h * (J[0] * mu[0] + J[2] * mu[1]), mu[1] + h * (J[1] * mu[0] + J[3] * mu[1])];
      th[k] = [th[k][0] + eta * (mu[0] - beta * th[k][0]), th[k][1] + eta * (mu[1] - beta * th[k][1])]; mu = prev;
    }
  }
  return { ...res, free };
}

// ---------- Step 5: inverse problems ----------
// y = x^(1) + N(0, sy²). The exact posterior is a mixture with diagonal components.
export function posterior4(y, sy) {
  const v = S2 + sy * sy, lw = MU2.map(([mx], k) => Math.log(WT2[k]) - ((y - mx) ** 2) / (2 * v));
  const m = Math.max(...lw), e = lw.map((l) => Math.exp(l - m)), Z = e.reduce((a, b) => a + b, 0);
  return {
    w: e.map((q) => q / Z),
    mu: MU2.map(([mx, my]) => [(mx * sy * sy + y * S2) / v, my]),
    var: MU2.map(() => [(S2 * sy * sy) / v, S2]),
  };
}
// Velocity of a diagonal mixture noised along the linear path.
function uDiag(P, x, y, t) {
  const lw = P.mu.map(([mx, my], k) => {
    const vx = t * t * P.var[k][0] + (1 - t) ** 2, vy = t * t * P.var[k][1] + (1 - t) ** 2;
    return Math.log(P.w[k]) - 0.5 * Math.log(vx * vy) - (x - t * mx) ** 2 / (2 * vx) - (y - t * my) ** 2 / (2 * vy);
  });
  const m = Math.max(...lw), r = lw.map((l) => Math.exp(l - m)), Z = r.reduce((a, b) => a + b, 0);
  let ux = 0, uy = 0;
  P.mu.forEach(([mx, my], k) => {
    const vx = t * t * P.var[k][0] + (1 - t) ** 2, vy = t * t * P.var[k][1] + (1 - t) ** 2, q = r[k] / Z;
    ux += q * (mx + (t * P.var[k][0] - (1 - t)) * (x - t * mx) / vx); uy += q * (my + (t * P.var[k][1] - (1 - t)) * (y - t * my) / vy);
  });
  return [ux, uy];
}
// Posterior mean and covariance of x_1 given x_t under the prior MIX2.
function post2(x, y, t) {
  const v = t * t * S2 + (1 - t) ** 2, lw = MU2.map(([mx, my], k) => Math.log(WT2[k]) - ((x - t * mx) ** 2 + (y - t * my) ** 2) / (2 * v));
  const m = Math.max(...lw), r = lw.map((l) => Math.exp(l - m)), Z = r.reduce((a, b) => a + b, 0), g = (t * S2) / v, c = (S2 * (1 - t) ** 2) / v;
  let hx = 0, hy = 0, sxx = 0, sxy = 0, syy = 0;
  MU2.forEach(([mx, my], k) => {
    const q = r[k] / Z, ax = mx + g * (x - t * mx), ay = my + g * (y - t * my);
    hx += q * ax; hy += q * ay; sxx += q * (c + ax * ax); sxy += q * ax * ay; syy += q * (c + ay * ay);
  });
  return { hx, hy, cxx: sxx - hx * hx, cxy: sxy - hx * hy, cyy: syy - hy * hy };
}
export const NP4 = 400, K4 = 1000;
const Z4 = (() => { const r = makeRng(41); return Array.from({ length: NP4 }, () => [r.gauss(), r.gauss()]); })();
// kind: 'dps' (p(y | E[x_1 | x_t]) with scale zeta), 'pigdm' (adds Var[x_1^(1) | x_t] to the noise),
// 'exact' (the exact posterior field). Euler with K4 steps from t = 0 to 1.
export function sample4(kind, y, sy, zeta = 1) {
  const P = posterior4(y, sy), h = 1 / K4;
  return Z4.map(([x0, y0]) => {
    let x = x0, z = y0;
    for (let i = 0; i < K4; i++) {
      const t = i * h;
      if (kind === 'exact') { const [ux, uy] = uDiag(P, x, z, t); x += h * ux; z += h * uy; continue; }
      const f = f2(x, z, t), q = post2(x, z, t);
      // (1 - t)/t times the gradient of log N(y; xhat^(1), s²) through J = t Cov / (1 - t)².
      const s2 = kind === 'dps' ? sy * sy : sy * sy + q.cxx, k = (kind === 'dps' ? zeta : 1) * (y - q.hx) / (s2 * (1 - t));
      x += h * (f.ux + k * q.cxx); z += h * (f.uy + k * q.cxy);
    }
    return [x, z];
  });
}
// W1 of each coordinate to the exact posterior marginal, and the shares by nearest mode.
const mixCdf = (w, mu, sd) => (x) => w.reduce((a, q, k) => a + q * 0.5 * (1 + erfL((x - mu[k]) / (sd[k] * Math.SQRT2))), 0);
function erfL(x) { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); }
function w1Cdf(xs, F) {
  const a = Float64Array.from(xs).sort(), n = a.length, lo = Math.min(a[0], -6), hi = Math.max(a[n - 1], 6), M = 3000, hh = (hi - lo) / M;
  let j = 0, s = 0; for (let i = 0; i < M; i++) { const x = lo + (i + 0.5) * hh; while (j < n && a[j] <= x) j++; s += Math.abs(j / n - F(x)); }
  return s * hh;
}
export const nearest2 = ([x, y]) => { let b = 0, d = Infinity; MU2.forEach(([mx, my], k) => { const e = (x - mx) ** 2 + (y - my) ** 2; if (e < d) { d = e; b = k; } }); return b; };
export function score4(S, y, sy) {
  const P = posterior4(y, sy), sh = [0, 0, 0]; S.forEach((p) => sh[nearest2(p)]++);
  const Fx = mixCdf(P.w, P.mu.map((m) => m[0]), P.var.map((v) => Math.sqrt(v[0]))), Fy = mixCdf(P.w, P.mu.map((m) => m[1]), P.var.map((v) => Math.sqrt(v[1])));
  return { w1x: w1Cdf(S.map((p) => p[0]), Fx), w1y: w1Cdf(S.map((p) => p[1]), Fy), shares: sh.map((c) => c / S.length), post: P };
}

// ---------- Step 6: SMC against best-of-N at equal compute ----------
// The SDE dx = [u + (eps²/2) s] dt + eps dW with eps_t² = 2 (1 - t) keeps the marginals p_t.
export const K5 = 50, RUNS5 = 400;
const eps2 = (t) => 2 * (1 - t);
// One SDE step from t to t + h; returns the new point and E[x_1 | x_t] at the old point (one NFE).
function sdeStep(x, t, h, g, guide = 0) {
  const F = fields(MIX1, x, t);
  let drift = F.u + 0.5 * eps2(t) * F.score;
  if (guide) {
    // Reward-gradient guidance: grad log h ~ r'(xhat) dxhat/dx / beta, with dxhat/dx = t Var / (1 - t)².
    const V = varX1(x, t), gr = (dreward(F.xhat) * t * V) / ((1 - t) ** 2) / guide;
    drift += ((1 - t) / Math.max(t, 1e-9) + 0.5 * eps2(t)) * gr;
  }
  return x + h * drift + Math.sqrt(eps2(t) * h) * g;
}
function varX1(x, t) {
  const m = MIX1, K = m.mu.length, lw = [], vt = [];
  for (let k = 0; k < K; k++) { vt[k] = t * t * m.sd[k] ** 2 + (1 - t) ** 2; lw[k] = Math.log(m.w[k]) - 0.5 * Math.log(vt[k]) - (x - t * m.mu[k]) ** 2 / (2 * vt[k]); }
  const mx = Math.max(...lw); let Z = 0; const r = lw.map((l) => { const e = Math.exp(l - mx); Z += e; return e; });
  let a = 0, b = 0;
  for (let k = 0; k < K; k++) { const s2 = m.sd[k] ** 2, mk = m.mu[k] + (t * s2 * (x - t * m.mu[k])) / vt[k], ck = (s2 * (1 - t) ** 2) / vt[k]; a += (r[k] / Z) * mk; b += (r[k] / Z) * (ck + mk * mk); }
  return Math.max(b - a * a, 0);
}
const xhat1 = (x, t) => (t >= 1 ? x : fields(MIX1, x, t).xhat);
// Best-of-N: N independent SDE runs, keep the end point with the largest reward.
export function runBoN(N, run) {
  const rng = makeRng(1000 + run), h = 1 / K5; let best = -Infinity, out = 0;
  for (let n = 0; n < N; n++) {
    let x = rng.gauss();
    for (let k = 0; k < K5; k++) x = sdeStep(x, k * h, h, rng.gauss());
    if (reward(x) > best) { best = reward(x); out = x; }
  }
  return out;
}
// SMC with N particles and potentials exp((r(xhat_{k+1}) - r(xhat_k)) / beta); multinomial resampling
// when the effective sample size falls below N/2; one output drawn by the final weights.
export function runSMC(N, beta, run, trace = null) {
  const rng = makeRng(5000 + run), h = 1 / K5;
  let xs = Array.from({ length: N }, () => rng.gauss()), lw = new Float64Array(N);
  let rh = xs.map((x) => reward(xhat1(x, 0)));
  for (let k = 0; k < K5; k++) {
    xs = xs.map((x) => sdeStep(x, k * h, h, rng.gauss()));
    const t2 = (k + 1) * h, rn = xs.map((x) => reward(xhat1(x, t2)));
    for (let i = 0; i < N; i++) lw[i] += (rn[i] - rh[i]) / beta;
    rh = rn;
    const m = Math.max(...lw), w = Array.from(lw, (l) => Math.exp(l - m)), Z = w.reduce((a, b) => a + b, 0);
    const ess = (Z * Z) / w.reduce((a, b) => a + b * b, 0);
    if (trace) trace.push(ess / N);
    if (ess < N / 2 && k < K5 - 1) {
      const nx = [], nr = [];
      for (let i = 0; i < N; i++) { let u = rng.uniform() * Z, j = 0; while (j < N - 1 && (u -= w[j]) > 0) j++; nx.push(xs[j]); nr.push(rh[j]); }
      xs = nx; rh = nr; lw = new Float64Array(N);
    }
  }
  const m = Math.max(...lw), w = Array.from(lw, (l) => Math.exp(l - m)), Z = w.reduce((a, b) => a + b, 0);
  let u = rng.uniform() * Z, j = 0; while (j < N - 1 && (u -= w[j]) > 0) j++;
  return xs[j];
}
export function runGuided(beta, run) {
  const rng = makeRng(9000 + run), h = 1 / K5; let x = rng.gauss();
  for (let k = 0; k < K5; k++) x = sdeStep(x, k * h, h, rng.gauss(), beta);
  return x;
}
export const NS5 = [1, 2, 4, 8, 16, 32, 64];
export const BETAS5 = [0.1, 0.2, 0.4];

// ---------- Step 7: an imperfect verifier ----------
// The verifier sees r + noise (independent per sample) or r + b times a bump at a spurious place.
export const SPUR = 3.1, SPW = 0.25;
export const spur = (x) => Math.exp(-((x - SPUR) ** 2) / (2 * SPW * SPW));
// Exact density of the best of N under a deterministic score q on a fine grid.
const FINE = 6000, FX = Float64Array.from({ length: FINE }, (_, i) => -6 + 12 * (i + 0.5) / FINE), FDX = 12 / FINE;
const FP = (() => { const p = FX.map((x) => Math.exp(fields(MIX1, x, 1).logp)); const Z = p.reduce((a, b) => a + b, 0) * FDX; return p.map((q) => q / Z); })();
export function goldBoNSystematic(b, NS) {
  const q = FX.map((x) => reward(x) + b * spur(x)), idx = Array.from(q.keys()).sort((i, j) => q[i] - q[j]);
  const F = new Float64Array(FINE); let c = 0;
  for (const i of idx) { c += FP[i] * FDX; F[i] = c; }
  return NS.map((N) => { let s = 0, z = 0; for (let i = 0; i < FINE; i++) { const d = N * FP[i] * F[i] ** (N - 1); s += d * reward(FX[i]); z += d; } return s / z; });
}
// Monte Carlo for independent noise: RUNS6 seeded runs of N draws from p_data.
export const RUNS6 = 2000;
let U6c = null;
const U6f = () => (U6c ??= (() => { const r = makeRng(61), n = RUNS6 * 1024, x = new Float64Array(n), g = new Float64Array(n); const Q = strat(MIX1, 4096, 1); for (let i = 0; i < n; i++) { x[i] = Q[Math.floor(r.uniform() * 4096)]; g[i] = r.gauss(); } return { x, g }; })());
export function goldBoNNoise(sigma, NS) {
  const U6 = U6f();
  return NS.map((N) => {
    let s = 0;
    for (let run = 0; run < RUNS6; run++) {
      let best = -Infinity, gold = 0;
      for (let n = 0; n < N; n++) { const i = run * 1024 + n, x = U6.x[i], rr = reward(x), v = rr + sigma * U6.g[i]; if (v > best) { best = v; gold = rr; } }
      s += gold;
    }
    return s / RUNS6;
  });
}
export const RMAX = 1;
