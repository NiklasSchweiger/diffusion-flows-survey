// Couplings, transport maps, Sinkhorn and controlled samplers of the §2.6 tutorial. No DOM access, so the
// numbers can be checked in Node. Convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data, linear
// path alpha_t = t, sigma_t = 1 - t. The data are the 1D mixture of §2.1 and the 2D mixture of §2.2.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, MIX2, gmm2, W2, LIN, pt1, dens1, sample1, quantile1 } from '../sde/sim.js';
import { uExact1 } from '../flow-matching/sim.js';
import { pt2 } from '../score-matching/sim.js';

export { MIX1, MIX2, gmm2, W2, LIN, pt1, uExact1, pt2 };

const phi = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
function erf(x) { const s = Math.sign(x), a = Math.abs(x), k = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * Math.exp(-a * a)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
export const pdata1 = (x) => pt1(x, 1);
export const DATA_M2 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * (MIX1.sd[k] ** 2 + m * m), 0);   // E[x1^2]

// ---------- Optimal assignment (Hungarian algorithm, O(n^3)) ----------
// cost is a function (i, j) -> number; returns col[i], the column assigned to row i.
export function assign(n, cost) {
  const INF = 1e18, u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  const minv = new Float64Array(n + 1), used = new Uint8Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; minv.fill(INF); used.fill(0);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const col = new Int32Array(n);
  for (let j = 1; j <= n; j++) col[p[j] - 1] = j - 1;
  return col;
}

// ---------- Step 1: minibatch OT pairs in 2D ----------
// NS noise and data points from one seeded stream. Minibatch OT with batch size B splits both lists into
// consecutive blocks of B and solves the assignment with cost |x1 - x0|^2 inside each block. B = 1 keeps
// the independent pairing.
export const NS = 128, NV = 2048, BATCHES = [1, 2, 4, 8, 16, 32, 64, 128];
function draw2(seed, n) {
  const r = makeRng(seed), x0 = new Float64Array(2 * n), x1 = new Float64Array(2 * n), k = new Uint8Array(n);
  for (let i = 0; i < n; i++) { x0[2 * i] = r.gauss(); x0[2 * i + 1] = r.gauss(); const s = gmm2.sample(r); x1[2 * i] = s.x; x1[2 * i + 1] = s.y; k[i] = s.k; }
  return { n, x0, x1, k };
}
const SHOW = draw2(201, NS), EST = draw2(202, NV);
// Returns pairs as indices into the data list: data index m[i] goes with noise point i.
function batchOT(D, B) {
  const m = new Int32Array(D.n);
  for (let s = 0; s < D.n; s += B) {
    const col = assign(B, (i, j) => (D.x1[2 * (s + j)] - D.x0[2 * (s + i)]) ** 2 + (D.x1[2 * (s + j) + 1] - D.x0[2 * (s + i) + 1]) ** 2);
    for (let i = 0; i < B; i++) m[s + i] = s + col[i];
  }
  return m;
}
const seg = (D, m, i) => [D.x0[2 * i], D.x0[2 * i + 1], D.x1[2 * m[i]], D.x1[2 * m[i] + 1]];
function crossings(D, m) {
  const side = (ax, ay, bx, by, cx, cy) => Math.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
  let n = 0;
  for (let i = 0; i < D.n; i++) for (let j = i + 1; j < D.n; j++) {
    const [a, b, c, d] = seg(D, m, i), [e, f, g, h] = seg(D, m, j);
    if (side(a, b, c, d, e, f) !== side(a, b, c, d, g, h) && side(e, f, g, h, a, b) !== side(e, f, g, h, c, d)) n++;
  }
  return n;
}
// Spread of the regression target x1 - x0 among pairs whose x_t (t = 0.5) falls in the same square of side
// 0.35, as a share of its total variance. It estimates E Var[x1 - x0 | x_t] / Var[x1 - x0]; the binning makes
// it a slight overestimate.
function condVar(D, m) {
  const bins = new Map(), all = [0, 0, 0];
  for (let i = 0; i < D.n; i++) {
    const [a, b, c, d] = seg(D, m, i), vx = c - a, vy = d - b, k = `${Math.floor((a + c) / 2 / 0.35)},${Math.floor((b + d) / 2 / 0.35)}`;
    if (!bins.has(k)) bins.set(k, [0, 0, 0, 0]);
    const s = bins.get(k); s[0]++; s[1] += vx; s[2] += vy; s[3] += vx * vx + vy * vy;
    all[0] += vx; all[1] += vy; all[2] += vx * vx + vy * vy;
  }
  let within = 0;
  for (const [n, sx, sy, ss] of bins.values()) within += ss - (sx * sx + sy * sy) / n;
  return (within / D.n) / (all[2] / D.n - (all[0] / D.n) ** 2 - (all[1] / D.n) ** 2);
}
const cache1 = new Map();
export function couple(B) {
  if (!cache1.has(B)) {
    const m = batchOT(SHOW, B), me = batchOT(EST, B);
    let len = 0; for (let i = 0; i < NV; i++) { const [a, b, c, d] = seg(EST, me, i); len += ((c - a) ** 2 + (d - b) ** 2) / NV; }
    cache1.set(B, { D: SHOW, m, cross: crossings(SHOW, m), len, cvar: condVar(EST, me) });
  }
  return cache1.get(B);
}

// ---------- Steps 2 and 3: the 1D optimal map and the two couplings ----------
// The optimal coupling of N(0, 1) and the 1D mixture is the monotone map T = F_data^{-1}(Phi), tabulated
// on a fine grid of x0. Under it x_t = (1 - t) x0 + t T(x0) is a straight line at constant speed.
export const X0LO = -6, X0HI = 6, NT0 = 2401, DX0 = (X0HI - X0LO) / (NT0 - 1);
export const X0G = Float64Array.from({ length: NT0 }, (_, i) => X0LO + DX0 * i);
export const TMAP = Float64Array.from(X0G, (x) => quantile1(Math.min(Math.max(Phi(x), 1e-12), 1 - 1e-12), 1));
const TD = Float64Array.from(X0G, (x, i) => phi(x, 0, 1) / Math.max(pdata1(TMAP[i]), 1e-300));   // T'(x0)
export function T(x0) {
  const u = (x0 - X0LO) / DX0, i = Math.min(NT0 - 2, Math.max(0, Math.floor(u))), w = u - i;
  return (1 - w) * TMAP[i] + w * TMAP[i + 1];
}
// The x0 whose line passes through x at time t (the lines are ordered, so bisection on the grid works).
function x0At(x, t) {
  const pos = (i) => (1 - t) * X0G[i] + t * TMAP[i];
  if (x <= pos(0)) return X0G[0];
  if (x >= pos(NT0 - 1)) return X0G[NT0 - 1];
  let lo = 0, hi = NT0 - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pos(mid) < x) lo = mid; else hi = mid; }
  const a = pos(lo), b = pos(hi);
  return X0G[lo] + DX0 * (x - a) / Math.max(b - a, 1e-12);
}
export const uOT = (x, t) => { const z = x0At(x, t); return T(z) - z; };
// Density of the OT path at time t as a curve: points (x, p) with x increasing.
export function ptOTcurve(t) {
  const xs = [], ps = [];
  for (let i = 0; i < NT0; i += 2) { xs.push((1 - t) * X0G[i] + t * TMAP[i]); ps.push(phi(X0G[i], 0, 1) / ((1 - t) + t * TD[i])); }
  return { xs, ps };
}
export function ptOT(x, t) {
  const z = x0At(x, t), u = (z - X0LO) / DX0, i = Math.min(NT0 - 2, Math.max(0, Math.floor(u))), w = u - i;
  return phi(z, 0, 1) / ((1 - t) + t * ((1 - w) * TD[i] + w * TD[i + 1]));
}
// W_2^2 between N(0, 1) and the data, from the map.
export const W22 = (() => { let s = 0; for (let i = 0; i < NT0; i++) s += (TMAP[i] - X0G[i]) ** 2 * phi(X0G[i], 0, 1) * DX0; return s; })();
// E|x1 - x0|^2 under the independent coupling.
export const COST_IND1 = DATA_M2 + 1;

// Starts at stratified quantiles of N(0, 1); paths of the independent field by RK4, K-step Euler for both fields.
export const NQ = 1000;
export const Z0 = Float64Array.from({ length: NQ }, (_, i) => quantile1((i + 0.5) / NQ, 0));
const DATA_Q = Float64Array.from({ length: NQ }, (_, i) => quantile1((i + 0.5) / NQ, 1));
export function euler1(vel, K, x0s = Z0) {
  const out = new Float64Array(x0s.length), h = 1 / K;
  for (let i = 0; i < x0s.length; i++) { let x = x0s[i]; for (let j = 0; j < K; j++) x += h * vel(x, j * h); out[i] = x; }
  return out;
}
// Euler polygons for a few starts: positions at t = j / K.
export function eulerPaths(vel, K, x0s) {
  return Array.from(x0s, (x0) => { const p = [x0]; let x = x0; for (let j = 0; j < K; j++) { x += vel(x, j / K) / K; p.push(x); } return p; });
}
// Accurate paths of the independent field: RK4 on NA steps.
export const NA = 100;
export function rk4Paths(x0s) {
  return Array.from(x0s, (x0) => {
    const p = [x0]; let x = x0; const h = 1 / NA;
    for (let j = 0; j < NA; j++) { const t = j * h, k1 = uExact1(x, t), k2 = uExact1(x + h / 2 * k1, t + h / 2), k3 = uExact1(x + h / 2 * k2, t + h / 2), k4 = uExact1(x + h * k3, t + h); x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); p.push(x); }
    return p;
  });
}
// W_1 distance of a sample set to the data, computed from the sorted samples and the data quantiles.
export function w1Data(xs) { const s = Float64Array.from(xs).sort(); let d = 0; for (let i = 0; i < NQ; i++) d += Math.abs(s[i] - DATA_Q[i]) / NQ; return d; }

// Kinetic energy rate of the independent field, int u_t^2 p_t dx, by quadrature on 1401 points.
const QX = Float64Array.from({ length: 1401 }, (_, i) => -7 + 14 * i / 1400), QDX = 14 / 1400;
export const keRate = (t) => { let s = 0; for (const x of QX) s += uExact1(x, t) ** 2 * pt1(x, t) * QDX; return s; };
export const KE_T = Float64Array.from({ length: 101 }, (_, j) => keRate(j / 100));
// Simpson on 101 points.
export const KE_IND = (() => { let s = 0; for (let j = 0; j <= 100; j++) s += KE_T[j] * (j === 0 || j === 100 ? 1 : j % 2 ? 4 : 2); return s / 300; })();

// ---------- Step 4: rectification in 2D ----------
// NR4 noise points are integrated by RK4 along the exact field of the independent coupling. The new pairs are
// (X0, X1); the middle term of the chain is int E|u_t(X_t)|^2 dt along the same paths (trapezoid rule).
export const NR4 = 1500, NSTEP4 = 80;
export const RECT = (() => {
  const D = draw2(203, NR4), path = [Float64Array.from(D.x0)], h = 1 / NSTEP4, ke = new Float64Array(NSTEP4 + 1);
  const sq = (P, t) => { let s = 0; for (let i = 0; i < NR4; i++) { const u = gmm2.fields(P[2 * i], P[2 * i + 1], t).u; s += (u[0] ** 2 + u[1] ** 2) / NR4; } return s; };
  for (let j = 0; j < NSTEP4; j++) {
    const t = j * h, P = path[j], Q = new Float64Array(2 * NR4);
    for (let i = 0; i < NR4; i++) {
      const x = P[2 * i], y = P[2 * i + 1], f = (a, b, s) => gmm2.fields(a, b, s).u;
      const k1 = f(x, y, t), k2 = f(x + h / 2 * k1[0], y + h / 2 * k1[1], t + h / 2), k3 = f(x + h / 2 * k2[0], y + h / 2 * k2[1], t + h / 2), k4 = f(x + h * k3[0], y + h * k3[1], t + h);
      Q[2 * i] = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); Q[2 * i + 1] = y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    }
    path.push(Q);
  }
  for (let j = 0; j <= NSTEP4; j++) ke[j] = sq(path[j], j * h);
  let mid = 0; for (let j = 0; j < NSTEP4; j++) mid += h * (ke[j] + ke[j + 1]) / 2;
  let rect = 0, old = 0; const X1 = path[NSTEP4];
  for (let i = 0; i < NR4; i++) {
    rect += ((X1[2 * i] - D.x0[2 * i]) ** 2 + (X1[2 * i + 1] - D.x0[2 * i + 1]) ** 2) / NR4;
    old += ((D.x1[2 * i] - D.x0[2 * i]) ** 2 + (D.x1[2 * i + 1] - D.x0[2 * i + 1]) ** 2) / NR4;
  }
  // Exact E|x1 - x0|^2 = E|x1|^2 + 2 for the independent coupling.
  const exact = MIX2.means.reduce((a, [mx, my], k) => a + W2[k] * (mx * mx + my * my + 2 * MIX2.std ** 2), 0) + 2;
  return { D, path, rect, mid, old, exact };
})();

// ---------- Step 5: the Schrödinger bridge as entropic OT, by Sinkhorn on a grid ----------
// Grid of NG points on [-4, 4] for both x0 and x1, weights a (noise) and b (data). The coupling minimizes
// E_pi |x1 - x0|^2 - 2 eps^2 H(pi); its Gibbs kernel exp(-|x1 - x0|^2 / (2 eps^2)) is the Brownian transition.
// Log-domain iterations, warm-started from the potentials of the previous eps.
export const NG = 121, GLO = -4, GHI = 4, GD = (GHI - GLO) / (NG - 1);
export const GX = Float64Array.from({ length: NG }, (_, i) => GLO + GD * i);
const norm = (a) => { const z = a.reduce((p, q) => p + q, 0); return a.map((q) => q / z); };
export const GA = norm(Float64Array.from(GX, (x) => phi(x, 0, 1))), GB = norm(Float64Array.from(GX, (x) => pdata1(x)));
const LA = GA.map(Math.log), LB = GB.map(Math.log);
let pot = { f: new Float64Array(NG), g: new Float64Array(NG) };   // f = lambda F, warm start
const cache5 = new Map();
export function sinkhorn(eps) {
  const key = eps.toFixed(3);
  if (cache5.has(key)) return cache5.get(key);
  const lam = 2 * eps * eps, C = (i, j) => (GX[j] - GX[i]) ** 2 / lam;
  const F = Float64Array.from(pot.f, (q) => q / lam), G = Float64Array.from(pot.g, (q) => q / lam), tmp = new Float64Array(NG);
  const lse = (n, f) => { let m = -Infinity; for (let k = 0; k < n; k++) { tmp[k] = f(k); if (tmp[k] > m) m = tmp[k]; } let s = 0; for (let k = 0; k < n; k++) s += Math.exp(tmp[k] - m); return m + Math.log(s); };
  let it = 0, err = 1;
  for (; it < 5000 && err > 1e-7; it++) {
    for (let i = 0; i < NG; i++) F[i] = -lse(NG, (j) => G[j] + LB[j] - C(i, j));
    for (let j = 0; j < NG; j++) G[j] = -lse(NG, (i) => F[i] + LA[i] - C(i, j));
    if (it % 10 === 9) { err = 0; for (let i = 0; i < NG; i++) err += Math.abs(Math.exp(lse(NG, (j) => F[i] + LA[i] + G[j] + LB[j] - C(i, j))) - GA[i]); }
  }
  pot = { f: F.map((q) => q * lam), g: G.map((q) => q * lam) };
  const P = new Float64Array(NG * NG); let cost = 0, mi = 0;
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) {
    const l = F[i] + LA[i] + G[j] + LB[j] - C(i, j), p = Math.exp(l); P[i * NG + j] = p;
    if (p > 0) { cost += p * (GX[j] - GX[i]) ** 2; mi += p * (l - LA[i] - LB[j]); }
  }
  const out = { eps, P, cost, mi, it, err };
  if (cache5.size > 40) cache5.clear();
  cache5.set(key, out);
  return out;
}
// Bridge paths: endpoints drawn from pi with fixed uniforms, plus eps times a fixed Brownian bridge.
export const NBR = 40, NBS = 100;
const BR = (() => {
  const r = makeRng(205), u = Float64Array.from({ length: NBR }, () => r.uniform());
  const b = Array.from({ length: NBR }, () => { const w = [0]; for (let k = 0; k < NBS; k++) w.push(w[k] + r.gauss() / Math.sqrt(NBS)); return w.map((q, k) => q - (k / NBS) * w[NBS]); });
  return { u, b };
})();
export function bridges(S) {
  const cum = new Float64Array(NG * NG); let c = 0;
  for (let k = 0; k < NG * NG; k++) cum[k] = (c += S.P[k]);
  return Array.from(BR.u, (u, n) => {
    let lo = 0, hi = NG * NG - 1; const q = u * c;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cum[mid] < q) lo = mid + 1; else hi = mid; }
    const x0 = GX[Math.floor(lo / NG)], x1 = GX[lo % NG];
    return { x0, x1, path: BR.b[n].map((w, k) => (1 - k / NBS) * x0 + (k / NBS) * x1 + S.eps * w) };
  });
}

// ---------- Steps 6 and 7: reward tilting and the h-transform (1D) ----------
// Base process: the member of the family of §2.4 with memoryless noise, dx = (alpha'/alpha x + eps_t^2 score) dt
// + eps_t dW with eps_t^2 = sigma_t^2 lambda'_t, on a grid uniform in log-SNR from t = T_S to t = T_E.
// Reward r(x) = -(x - c)^2 / 2, so e^{r/beta} is an unnormalized Gaussian of variance beta.
export const T_S = 0.03, T_E = 0.995, NK6 = 200;
export const TG = (() => { const l0 = LIN.lam(T_S), l1 = LIN.lam(T_E); return Float64Array.from({ length: NK6 + 1 }, (_, k) => { const e = Math.exp((l0 + (l1 - l0) * k / NK6) / 2); return e / (1 + e); }); })();
export const reward = (x, c) => -((x - c) ** 2) / 2;
// Posterior of x1 given x_t = x: log-weights, means and variances of its three components.
function post(x, t) {
  const a = LIN.alpha(t), s = LIN.sigma(t);
  return MIX1.mu.map((m, k) => {
    const s2 = MIX1.sd[k] ** 2, v = a * a * s2 + s * s;
    return { lw: Math.log(MIX1.w[k]) - ((x - a * m) ** 2) / (2 * v) - 0.5 * Math.log(v), m: m + (a * s2 / v) * (x - a * m), v: s2 * s * s / v };
  });
}
const lseArr = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((p, q) => p + Math.exp(q - m), 0)); };
// log h_t(x) = log E[exp(r(x1)/beta) | x_t = x], exact for the mixture posterior.
export function logh(x, t, beta, c) {
  const P = post(x, t), z = lseArr(P.map((q) => q.lw));
  return lseArr(P.map((q) => q.lw - z + 0.5 * Math.log(beta / (beta + q.v)) - ((q.m - c) ** 2) / (2 * (beta + q.v))));
}
export const dlogh = (x, t, beta, c, d = 1e-4) => (logh(x + d, t, beta, c) - logh(x - d, t, beta, c)) / (2 * d);
export function xhat(x, t) { const P = post(x, t), z = lseArr(P.map((q) => q.lw)); return P.reduce((a, q) => a + Math.exp(q.lw - z) * q.m, 0); }
// Guidance: log h_t is replaced by r(xhat_1(x, t)) / beta, differentiated through the denoiser.
export const dloghGuid = (x, t, beta, c, d = 1e-4) => (reward(xhat(x + d, t), c) - reward(xhat(x - d, t), c)) / (beta * 2 * d);
// The tilted target p* ∝ p_data e^{r/beta}: again a mixture.
export function tilted(beta, c) {
  const comp = MIX1.mu.map((m, k) => { const s2 = MIX1.sd[k] ** 2; return { lw: Math.log(MIX1.w[k]) - ((m - c) ** 2) / (2 * (s2 + beta)) - 0.5 * Math.log(s2 + beta), m: (m * beta + c * s2) / (s2 + beta), v: s2 * beta / (s2 + beta) }; });
  const z = lseArr(comp.map((q) => q.lw));
  comp.forEach((q) => { q.w = Math.exp(q.lw - z); });
  const pdf = (x) => comp.reduce((a, q) => a + q.w * phi(x, q.m, q.v), 0);
  const cdf = (x) => comp.reduce((a, q) => a + q.w * Phi((x - q.m) / Math.sqrt(q.v)), 0);
  const Q = Float64Array.from({ length: NQ }, (_, i) => { const p = (i + 0.5) / NQ; let lo = -10, hi = 10; for (let k = 0; k < 50; k++) { const mid = (lo + hi) / 2; if (cdf(mid) < p) lo = mid; else hi = mid; } return (lo + hi) / 2; });
  let kl = 0; for (const x of QX) { const p = pdf(x); if (p > 1e-300) kl += p * Math.log(p / pdata1(x)) * QDX; }
  return { comp, pdf, Q, kl };
}
export function w1To(xs, Q) { const s = Float64Array.from(xs).sort(), n = s.length; let d = 0; for (let i = 0; i < n; i++) d += Math.abs(s[i] - Q[Math.min(NQ - 1, Math.floor((i + 0.5) * NQ / n))]) / n; return d; }

// Stratified starts from p_{T_S} and fixed Gaussian increments, shared by base and controlled runs.
const NP6 = 1000;
const S6 = (() => { const r = makeRng(206); return { x0: Float64Array.from({ length: NP6 }, (_, i) => quantile1((i + 0.5) / NP6, T_S)), xi: Float64Array.from({ length: NP6 * NK6 }, () => r.gauss()) }; })();
const drift0 = (x, t) => (LIN.dalpha(t) / LIN.alpha(t)) * x + LIN.g2(t) * dens1(x, t).score;
// Euler-Maruyama; ctrl(x, t) is the extra term of log h in the drift (0 for the base process). Records the
// positions of NTR particles for drawing and the Girsanov cost E int 1/2 |c_t|^2 dt with c_t = eps_t dlogh.
export const NTR = 30;
const TRI = Array.from({ length: NTR }, (_, j) => Math.floor((j + 0.5) * NP6 / NTR));   // particles whose paths are drawn
export function simulate(ctrl, x0s = S6.x0) {
  const end = new Float64Array(NP6), trails = Array.from({ length: NTR }, () => new Float64Array(NK6 + 1));
  let cost = 0;
  for (let i = 0; i < NP6; i++) {
    let x = x0s[i]; const tr = trails[TRI.indexOf(i)] ?? null;
    if (tr) tr[0] = x;
    for (let k = 0; k < NK6; k++) {
      const t = TG[k], dt = TG[k + 1] - t, g2 = LIN.g2(t), d = ctrl ? ctrl(x, t) : 0;
      cost += 0.5 * g2 * d * d * dt / NP6;
      x += (drift0(x, t) + g2 * d) * dt + Math.sqrt(g2 * dt) * S6.xi[i * NK6 + k];
      if (tr) tr[k + 1] = x;
    }
    end[i] = x;
  }
  return { end, trails, cost };
}
// The h-transform also tilts the start: p_{T_S} h_{T_S} / Z. Its stratified quantiles, from the CDF on a grid,
// and the KL between the tilted and the plain start (the part of the cost paid before the first step).
export function startTilted(beta, c) {
  const n = 2401, lo = -7, d = 14 / (n - 1), x = Float64Array.from({ length: n }, (_, i) => lo + d * i);
  const p = Float64Array.from(x, (q) => pt1(q, T_S)), lh = Float64Array.from(x, (q) => logh(q, T_S, beta, c));
  const m = Math.max(...lh), w = Float64Array.from(p, (q, i) => q * Math.exp(lh[i] - m)), cum = new Float64Array(n);
  for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + (w[i] + w[i - 1]) * d / 2;
  const z = cum[n - 1]; let kl = 0;
  const zp = p.reduce((a, q) => a + q * d, 0), zw = z;
  for (let i = 0; i < n; i++) if (w[i] > 0) kl += (w[i] / zw) * Math.log((w[i] / zw) / (p[i] / zp)) * d;
  const x0 = Float64Array.from({ length: NP6 }, (_, k) => {
    const q = (k + 0.5) / NP6 * z; let a = 0, b = n - 1;
    while (b - a > 1) { const mid = (a + b) >> 1; if (cum[mid] < q) a = mid; else b = mid; }
    return x[a] + d * (q - cum[a]) / Math.max(cum[b] - cum[a], 1e-300);
  });
  return { x0, kl };
}
// Spread of h at the start time: max / min of h_{T_S} over the central 99.8% of p_{T_S}, minus 1.
export function hSpread(beta, c) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < 101; i++) { const x = quantile1(0.001 + 0.998 * i / 100, T_S), l = logh(x, T_S, beta, c); lo = Math.min(lo, l); hi = Math.max(hi, l); }
  return Math.exp(hi - lo) - 1;
}
// Feynman–Kac SMC: NRUN independent runs of N particles under the base process. The weights follow the
// guidance estimate h~_t = exp(r(xhat_1)/beta) step by step, with h~ = exp(r/beta) exactly at the end, and the
// particles are resampled (multinomial, fixed uniforms) when the effective sample size drops below N/2.
// One particle per run is returned, drawn by its final weight, so the output has NRUN samples.
export const NRUN = 600, NK7 = 100;
const TG7 = Float64Array.from({ length: NK7 + 1 }, (_, k) => TG[2 * k]);
export function smc(N, beta, c) {
  const r = makeRng(207 + N), out = new Float64Array(NRUN), x = new Float64Array(N), lw = new Float64Array(N), lh = new Float64Array(N), nx = new Float64Array(N);
  const lht = (y, t, last) => (last ? reward(y, c) / beta : reward(xhat(y, t), c) / beta);
  for (let run = 0; run < NRUN; run++) {
    for (let i = 0; i < N; i++) { x[i] = quantile1(Math.min(0.9999, Math.max(0.0001, r.uniform())), T_S); lh[i] = lht(x[i], T_S, false); lw[i] = 0; }
    for (let k = 0; k < NK7; k++) {
      const t = TG7[k], dt = TG7[k + 1] - t, g2 = LIN.g2(t), last = k === NK7 - 1;
      for (let i = 0; i < N; i++) {
        x[i] += drift0(x[i], t) * dt + Math.sqrt(g2 * dt) * r.gauss();
        const l = lht(x[i], TG7[k + 1], last); lw[i] += l - lh[i]; lh[i] = l;
      }
      const m = Math.max(...lw); let s = 0, s2 = 0;
      for (let i = 0; i < N; i++) { const w = Math.exp(lw[i] - m); s += w; s2 += w * w; }
      if (!last && s * s / s2 < N / 2) {
        for (let i = 0; i < N; i++) { let u = r.uniform() * s, j = 0; while (j < N - 1 && (u -= Math.exp(lw[j] - m)) > 0) j++; nx[i] = x[j]; }
        for (let i = 0; i < N; i++) { x[i] = nx[i]; lw[i] = 0; lh[i] = lht(x[i], TG7[k + 1], false); }
      }
    }
    const m = Math.max(...lw); let s = 0; for (let i = 0; i < N; i++) s += Math.exp(lw[i] - m);
    let u = r.uniform() * s, j = 0; while (j < N - 1 && (u -= Math.exp(lw[j] - m)) > 0) j++;
    out[run] = x[j];
  }
  return out;
}
