// Numerics of steps 1, 3, 4 and 7 of the §7 tutorial: the robot policy, a linear world model, the CRPS
// and score distillation. The steps on real data are in real.js. All four toys are exact (closed form)
// or Monte Carlo over seeded draws, and no function here touches the DOM. The policy's flow time is τ.
import { makeRng } from '../../animations/lib/rng.js';
import { GMM } from '../../animations/lib/gmm.js';
import { Phi } from '../lib/mix1d.js';

const sum = (a) => a.reduce((s, v) => s + v, 0);
const mean = (a) => sum(a) / a.length;
// E|Z| for Z ~ N(mu, v).
export const absMoment = (mu, v) => (v <= 0 ? Math.abs(mu) : Math.sqrt(v) * Math.sqrt(2 / Math.PI) * Math.exp(-mu * mu / (2 * v)) + mu * (1 - 2 * Phi(-mu / Math.sqrt(v))));

// ---------- Step 1: robot policy ----------
// Demonstrations pass an obstacle on the left or on the right. Along the forward coordinate s the
// lateral position is y = m b(s) + d with a mode m = ±1 and an AR(1) deviation d of stationary
// standard deviation PS and correlation PR per step. The observation is the current (s, y). The flow
// policy samples the exact conditional of the next H waypoints: first m from p(m | y), then the
// deviation forward. The regression policy outputs the conditional mean of the same chunk.
export const PK = 40, PS = 0.05, PR = 0.9, PB = 0.3, PW = 0.1, NP = 400, NPS = 30;
export const OBS = { s: 0.5, hs: 0.06, hy: 0.12 };
export const pb = (s) => PB * Math.exp(-((s - 0.5) ** 2) / (2 * PW * PW));
export const pPlus = (s, y) => { const b = pb(s); return 1 / (1 + Math.exp(-2 * y * b / (PS * PS))); };
export const PH = [1, 2, 4, 8, 16, 40];
function policyRoll(kind, H, rng) {
  const ys = [PS * rng.gauss()], modes = [];
  while (ys.length - 1 < PK) {
    const k = ys.length - 1, s = k / PK, y = ys[k], bk = pb(s), p = pPlus(s, y);
    if (kind === 'reg') {
      const e = 2 * p - 1;
      for (let j = 1; j <= H && k + j <= PK; j++) ys.push(e * pb((k + j) / PK) + PR ** j * (y - e * bk));
    } else {
      const m = rng.uniform() < p ? 1 : -1; let d = y - m * bk;
      modes.push([k, m]);
      for (let j = 1; j <= H && k + j <= PK; j++) { d = PR * d + Math.sqrt(1 - PR * PR) * PS * rng.gauss(); ys.push(m * pb((k + j) / PK) + d); }
    }
  }
  const hit = ys.some((v, i) => Math.abs(i / PK - OBS.s) <= OBS.hs && Math.abs(v) < OBS.hy);
  // A switch counts once the two modes are apart, b(s) > 0.05.
  let sw = 0; for (let i = 1; i < modes.length; i++) if (modes[i][1] !== modes[i - 1][1] && pb(modes[i][0] / PK) > 0.05) sw++;
  return { ys, hit, sw };
}
const polCache = new Map();
export function policy(kind, H) {
  const key = kind + H; if (polCache.has(key)) return polCache.get(key);
  const rng = makeRng(7), runs = Array.from({ length: NP }, () => policyRoll(kind, H, rng));
  const r = { show: runs.slice(0, NPS).map((q) => q.ys), success: mean(runs.map((q) => (q.hit ? 0 : 1))), switches: mean(runs.map((q) => q.sw)) };
  polCache.set(key, r);
  return r;
}
// Demonstrations themselves: the mode is fixed for the whole path.
export const DEMOS = (() => {
  const rng = makeRng(5);
  return Array.from({ length: NPS }, () => { const m = rng.uniform() < 0.5 ? 1 : -1; let d = PS * rng.gauss(); return Array.from({ length: PK + 1 }, (_, k) => { if (k) d = PR * d + Math.sqrt(1 - PR * PR) * PS * rng.gauss(); return m * pb(k / PK) + d; }); });
})();

// ---------- Step 4: weather and the CRPS ----------
// The true predictive distribution G is a two-component mixture with mean 0. The forecast F_k scales
// it about its mean by the spread factor k, so k = 0 is the deterministic forecast at the mean and
// k = 1 is the truth. All expectations are exact sums of E|N(mu, v)| over pairs of components.
export const G = { mu: [-0.8, 1.2], sd: [0.6, 0.8], w: [0.6, 0.4] };
const pairSum = (f) => { let s = 0; for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) s += G.w[j] * G.w[k] * f(j, k); return s; };
export const EXY = (k) => pairSum((j, i) => absMoment(k * G.mu[j] - G.mu[i], k * k * G.sd[j] ** 2 + G.sd[i] ** 2));
export const EYY = pairSum((j, i) => absMoment(G.mu[j] - G.mu[i], G.sd[j] ** 2 + G.sd[i] ** 2));
// Expected CRPS of F_k against observations from G; the plain M-member estimator; the fair one.
export const crps = (k) => EXY(k) - 0.5 * k * EYY;
export const crpsPlain = (k, M) => EXY(k) - 0.5 * (1 - 1 / M) * k * EYY;
export const VARG = G.w.reduce((a, w, j) => a + w * (G.sd[j] ** 2 + G.mu[j] ** 2), 0);
export const KAP = Array.from({ length: 201 }, (_, i) => i / 100);
export const argminK = (f) => KAP.reduce((b, k) => (f(k) < f(b) ? k : b), KAP[0]);
export const MS6 = [2, 3, 5, 10, 20, 50];
const rngW = makeRng(41);
const drawG = () => { const j = rngW.uniform() < G.w[0] ? 0 : 1; return G.mu[j] + G.sd[j] * rngW.gauss(); };
export const YMEM = Array.from({ length: 50 }, drawG);
export const YOBS = drawG();
// The empirical scores of one ensemble of M members against one observation.
export function ensembleScore(xs, y, fair) {
  const M = xs.length; let a = 0, b = 0;
  xs.forEach((x) => { a += Math.abs(x - y); xs.forEach((z) => { b += Math.abs(x - z); }); });
  return a / M - b / (2 * (fair ? M * (M - 1) : M * M));
}
export const gDens = (x, k = 1) => G.w.reduce((a, w, j) => { const s = Math.max(k, 1e-3) * G.sd[j]; return a + w * Math.exp(-((x - k * G.mu[j]) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI)); }, 0);

// ---------- Step 7: score distillation ----------
// The image model is an exact 2D mixture with four components; the prompt c selects the upper two.
// The renderer is the identity, so a particle is its own rendering. One iteration draws for every
// particle a time t ~ U[0.05, 0.95] and a noise x_0 and moves the particle by SETA times the
// difference of two denoisers at x_t = t x + (1 - t) x_0: the guided denoiser of the model, and for SDS
// the particle itself (the denoiser of a point mass) or for VSD the exact denoiser of the noised
// particle cloud. This is a step of size SETA along the paper's eq:sds with the weight w(t) = (1 - t) / t.
export const SMEANS = [[-1.6, 1.0], [1.5, 1.2], [0.2, -1.5], [1.9, -1.0]], SW = [0.4, 0.3, 0.2, 0.1], SSD = 0.3;
export const SN = 200, SIT = 600, SETA = 0.05, SGAM = [1, 3, 7.5, 20];
const GU = new GMM(SMEANS, SSD, { weights: SW }), GC = new GMM(SMEANS.slice(0, 2), SSD, { weights: SW.slice(0, 2) });
export const gmmDens = (x, y, cond) => {
  const K = cond ? 2 : 4, z = cond ? SW[0] + SW[1] : 1;
  let p = 0; for (let k = 0; k < K; k++) p += SW[k] / z * Math.exp(-((x - SMEANS[k][0]) ** 2 + (y - SMEANS[k][1]) ** 2) / (2 * SSD * SSD));
  return p / (2 * Math.PI * SSD * SSD);
};
// Seeded samples of the prompt's distribution p(x | c), for comparison.
export const SAMPLES = (() => { const rng = makeRng(31); return Array.from({ length: SN }, () => { const s = GC.sample(rng); return [s.x, s.y]; }); })();
// The particle cloud after every iteration, computed on demand. kind is 'sds' or 'vsd'.
export function distill(kind, gam) {
  const init = makeRng(5), rng = makeRng(9);
  let P = Array.from({ length: SN }, () => [init.gauss(), init.gauss()]);
  const hist = [P];
  const step = () => {
    const Q = P.map((p) => p.slice());
    for (let i = 0; i < SN; i++) {
      const t = 0.05 + 0.9 * rng.uniform(), s = 1 - t, x = t * P[i][0] + s * rng.gauss(), y = t * P[i][1] + s * rng.gauss();
      const du = GU.fields(x, y, t).d1, dc = GC.fields(x, y, t).d1, D = [du[0] + gam * (dc[0] - du[0]), du[1] + gam * (dc[1] - du[1])];
      let E = P[i];
      if (kind === 'vsd') {
        let mx = -Infinity; const l = P.map((q) => { const a = -((x - t * q[0]) ** 2 + (y - t * q[1]) ** 2) / (2 * s * s); if (a > mx) mx = a; return a; });
        let z = 0, ex = 0, ey = 0; l.forEach((a, j) => { const w = Math.exp(a - mx); z += w; ex += w * P[j][0]; ey += w * P[j][1]; });
        E = [ex / z, ey / z];
      }
      Q[i][0] += SETA * (D[0] - E[0]); Q[i][1] += SETA * (D[1] - E[1]);
    }
    P = Q; hist.push(P);
  };
  return { at: (k) => { while (hist.length <= Math.min(k, SIT)) step(); return hist[Math.min(k, SIT)]; } };
}
// For a cloud of points: the share nearest to each prompt mode, the share within 2.5 standard
// deviations of its mode, and the spread, the RMS distance per coordinate to the centroid of the
// points that share its nearest mode (SSD for exact samples).
export function cloudStats(P) {
  const n = [0, 0], c = [[0, 0], [0, 0]], ks = []; let inside = 0;
  P.forEach(([x, y]) => {
    const d = SMEANS.slice(0, 2).map(([a, b]) => (x - a) ** 2 + (y - b) ** 2), k = d[0] <= d[1] ? 0 : 1;
    n[k]++; c[k][0] += x; c[k][1] += y; ks.push(k); if (d[k] < (2.5 * SSD) ** 2) inside++;
  });
  let sq = 0; P.forEach(([x, y], i) => { const k = ks[i]; sq += (x - c[k][0] / n[k]) ** 2 + (y - c[k][1] / n[k]) ** 2; });
  return { share: n.map((v) => v / P.length), inside: inside / P.length, spread: Math.sqrt(sq / P.length / 2) };
}

// ---------- Step 3: a linear world model of a 1D game ----------
// The game is a strip of WD pixels with a bright blob at the player's position s; the action a in
// {-1, 0, 1} moves it by WSTEP pixels. The model predicts the next frame linearly, x' = [x, 1] W_a,
// and is fitted by least squares on WN real frame pairs. Training on contexts with added noise of
// standard deviation σ equals, in expectation, ridge regression with λ = σ² (the bias is not
// penalized), so every fit here is the closed-form ridge solution. In a rollout, each generated
// frame gets pixel noise σ_g, the residual of a few-step sampler, and becomes the next context.
export const WD = 48, WSTEP = 1.5, WSIG = 2.5, WN = 600, WK = 60, WR = 20, WFLOOR = 1e-7;
const WLO = 3, WHI = WD - 4, WA = [-1, 0, 1], WP = WD + 1;
export const wRender = (s) => Float64Array.from({ length: WD }, (_, i) => Math.exp(-((i - s) ** 2) / (2 * WSIG * WSIG)));
const wMove = (s, a) => Math.min(WHI, Math.max(WLO, s + a * WSTEP));
// Gram matrix [X, 1]^T [X, 1] and right-hand sides [X, 1]^T Y_a of the training pairs.
const WGRAM = (() => {
  const rng = makeRng(1), G = new Float64Array(WP * WP), B = WA.map(() => new Float64Array(WP * WD));
  for (let n = 0; n < WN; n++) {
    const s = WLO + (WHI - WLO) * rng.uniform(), x = [...wRender(s), 1];
    for (let i = 0; i < WP; i++) for (let j = 0; j < WP; j++) G[i * WP + j] += x[i] * x[j];
    WA.forEach((a, ai) => { const y = wRender(wMove(s, a)); for (let i = 0; i < WP; i++) for (let j = 0; j < WD; j++) B[ai][i * WD + j] += x[i] * y[j]; });
  }
  return { G, B };
})();
// Spectral norm of the frame part of W (WD × WD, rows act on the context) by power iteration.
function specNorm(W) {
  let v = new Float64Array(WD).fill(1 / Math.sqrt(WD)), nrm = 0;
  for (let it = 0; it < 200; it++) {
    const u = new Float64Array(WD), w = new Float64Array(WD);
    for (let i = 0; i < WD; i++) for (let j = 0; j < WD; j++) u[j] += v[i] * W[i * WD + j];
    for (let i = 0; i < WD; i++) for (let j = 0; j < WD; j++) w[i] += W[i * WD + j] * u[j];
    nrm = Math.sqrt(Math.hypot(...w)); const z = Math.hypot(...w); v = w.map((q) => q / z);
  }
  return nrm;
}
// Ridge fit for λ: one W per action, and the largest spectral norm over the actions.
export function wFit(lam) {
  const { G, B } = WGRAM, M = Float64Array.from(G), L = new Float64Array(WP * WP);
  for (let i = 0; i < WD; i++) M[i * WP + i] += lam * WN;
  for (let j = 0; j < WP; j++) {   // Cholesky, M = L L^T
    let d = M[j * WP + j]; for (let k = 0; k < j; k++) d -= L[j * WP + k] ** 2;
    L[j * WP + j] = Math.sqrt(d);
    for (let i = j + 1; i < WP; i++) { let s = M[i * WP + j]; for (let k = 0; k < j; k++) s -= L[i * WP + k] * L[j * WP + k]; L[i * WP + j] = s / L[j * WP + j]; }
  }
  const Ws = B.map((Bm) => {
    const W = new Float64Array(WP * WD);
    for (let c = 0; c < WD; c++) {
      const z = new Float64Array(WP);
      for (let i = 0; i < WP; i++) { let s = Bm[i * WD + c]; for (let k = 0; k < i; k++) s -= L[i * WP + k] * z[k]; z[i] = s / L[i * WP + i]; }
      for (let i = WP - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < WP; k++) s -= L[k * WP + i] * W[k * WD + c]; W[i * WD + c] = s / L[i * WP + i]; }
    }
    return W;
  });
  return { Ws, norm: Math.max(...Ws.map(specNorm)) };
}
// WR rollouts of WK frames from a real first frame, with the same actions and the same noise draws
// for every model. Returns the median error per frame, and the frames and positions of the first rollout.
export function wRoll(fit, sg) {
  const ra = makeRng(9), rz = makeRng(11), errs = Array.from({ length: WK + 1 }, () => []), film = [], truth = [], path = [];
  for (let r = 0; r < WR; r++) {
    let s = 10 + (WD - 20) * ra.uniform(), x = wRender(s);
    errs[0].push(0); if (!r) { film.push(x); truth.push(x); path.push(s); }
    for (let k = 1; k <= WK; k++) {
      const ai = Math.min(2, Math.floor(3 * ra.uniform())), W = fit.Ws[ai], y = new Float64Array(WD);
      s = wMove(s, WA[ai]);
      for (let j = 0; j < WD; j++) { let v = W[WD * WD + j]; for (let i = 0; i < WD; i++) v += x[i] * W[i * WD + j]; y[j] = v + sg * rz.gauss(); }
      x = y; const t = wRender(s);
      errs[k].push(Math.hypot(...x.map((v, j) => v - t[j])));
      if (!r) { film.push(x); truth.push(t); path.push(s); }
    }
  }
  const med = errs.map((e) => { const q = [...e].sort((a, b) => a - b); return (q[(WR >> 1) - 1] + q[WR >> 1]) / 2; });
  return { med, film, truth, path };
}
// One-step error on real contexts, the quantity that teacher forcing minimizes, averaged over a grid.
export function wOneStep(fit) {
  let tot = 0, n = 0;
  for (let q = 0; q < 50; q++) {
    const s = WLO + 1 + (WHI - WLO - 2) * q / 49, x = wRender(s);
    WA.forEach((a, ai) => {
      const W = fit.Ws[ai], t = wRender(wMove(s, a)); let e = 0;
      for (let j = 0; j < WD; j++) { let v = W[WD * WD + j]; for (let i = 0; i < WD; i++) v += x[i] * W[i * WD + j]; e += (v - t[j]) ** 2; }
      tot += Math.sqrt(e); n++;
    });
  }
  return tot / n;
}
