// Numerics of steps 3 and 6 of the §7 tutorial, the robot policy and the CRPS; the steps on real data
// are in real.js. Both toys are exact (closed form) or Monte Carlo over seeded draws, and no function
// here touches the DOM. The policy's flow time is τ.
import { makeRng } from '../../animations/lib/rng.js';
import { Phi } from '../lib/mix1d.js';

const sum = (a) => a.reduce((s, v) => s + v, 0);
const mean = (a) => sum(a) / a.length;
// E|Z| for Z ~ N(mu, v).
export const absMoment = (mu, v) => (v <= 0 ? Math.abs(mu) : Math.sqrt(v) * Math.sqrt(2 / Math.PI) * Math.exp(-mu * mu / (2 * v)) + mu * (1 - 2 * Phi(-mu / Math.sqrt(v))));

// ---------- Step 3: robot policy ----------
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

// ---------- Step 6: weather and the CRPS ----------
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
