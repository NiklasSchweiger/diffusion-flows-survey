// Targets and samplers for the §2.1 tutorial. No DOM access, so the numbers can be checked in Node.
// All fields are exact: the targets are Gaussian mixtures, and smoothing a mixture with N(0, σ² I)
// gives another mixture whose component variance is s² + σ².
import { GMM } from '../../animations/lib/gmm.js';
import { makeRng } from '../../animations/lib/rng.js';

// ---------- 1D target (steps 1 and 4) ----------
export const MIX1 = { mu: [-1.6, 0.3, 1.9], sd: [0.45, 0.35, 0.5], w: [0.45, 0.3, 0.25] };
const phi = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (Math.sqrt(2 * Math.PI) * s);
export const p1 = (x) => MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * phi(x, m, MIX1.sd[k]), 0);
export function score1(x) {
  let num = 0, den = 0;
  MIX1.mu.forEach((m, k) => { const f = MIX1.w[k] * phi(x, m, MIX1.sd[k]); num -= f * (x - m) / MIX1.sd[k] ** 2; den += f; });
  return num / Math.max(den, 1e-300);
}

// ---------- 2D target (steps 2, 3 and 5) ----------
export const MIX2 = { means: [[-1.4, -0.55], [1.35, -0.85], [0.15, 1.3]], std: 0.5, w: [0.45, 0.3, 0.25] };
export const gmm2 = new GMM(MIX2.means, MIX2.std, { weights: MIX2.w });
// At t = 1 the interpolant marginal in gmm.js is the data mixture itself.
export const score2 = (x, y) => gmm2.fields(x, y, 1).score;
export function p2(x, y, means = MIX2.means, std = MIX2.std, w = MIX2.w) {
  const v = std * std, W = w.reduce((a, b) => a + b, 0);
  return means.reduce((a, [mx, my], k) => a + (w[k] / W) * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * v)) / (2 * Math.PI * v), 0);
}
// Total variance tr Cov of the 2D target, the reference for the ULA spread readout.
export const spread2 = (() => {
  const W = MIX2.w.reduce((a, b) => a + b, 0), w = MIX2.w.map((x) => x / W);
  const m = [0, 1].map((d) => MIX2.means.reduce((a, mu, k) => a + w[k] * mu[d], 0));
  return 2 * MIX2.std ** 2 + MIX2.means.reduce((a, mu, k) => a + w[k] * ((mu[0] - m[0]) ** 2 + (mu[1] - m[1]) ** 2), 0);
})();

// A cloud of 2D particles moved by x <- x + eta * s(x) + sqrt(2 eta) * xi. Particle 0 is tracked:
// prev is its position before the last step and drift its drift displacement eta * s(prev).
export class Cloud {
  constructor(n, seed, box = [3, 2.2]) { this.n = n; this.seed = seed; this.box = box; this.P = new Float64Array(2 * n); this.reset(); }
  reset() {
    this.rng = makeRng(this.seed); this.k = 0;
    for (let i = 0; i < this.n; i++) {
      this.P[2 * i] = (this.rng.uniform() * 2 - 1) * this.box[0];
      this.P[2 * i + 1] = (this.rng.uniform() * 2 - 1) * this.box[1];
    }
    this.prev = [this.P[0], this.P[1]]; this.drift = [0, 0]; this.trail = [[this.P[0], this.P[1]]];
  }
  step(eta, score, { drift = true, noise = true } = {}) {
    const P = this.P, sq = Math.sqrt(2 * eta);
    for (let i = 0; i < this.n; i++) {
      const x = P[2 * i], y = P[2 * i + 1], sc = drift ? score(x, y) : [0, 0];
      const gx = this.rng.gauss(), gy = this.rng.gauss();   // drawn even when unused, so toggles keep the same stream
      if (i === 0) { this.prev = [x, y]; this.drift = [eta * sc[0], eta * sc[1]]; }
      P[2 * i] = x + eta * sc[0] + (noise ? sq * gx : 0);
      P[2 * i + 1] = y + eta * sc[1] + (noise ? sq * gy : 0);
    }
    this.k++;
    this.trail.push([P[0], P[1]]);
    if (this.trail.length > 80) this.trail.shift();
  }
  spread() {
    const { P, n } = this; let mx = 0, my = 0, v = 0;
    for (let i = 0; i < n; i++) { mx += P[2 * i] / n; my += P[2 * i + 1] / n; }
    for (let i = 0; i < n; i++) v += ((P[2 * i] - mx) ** 2 + (P[2 * i + 1] - my) ** 2) / n;
    return v;
  }
}

// ---------- Step 4: Langevin in 1D, histogram and KL over tau ----------
export const FP = { N: 3000, dtau: 0.01, T: 8, bins: 64, lo: -4, hi: 4, start: [3.2, 0.15] };
export function fokkerPlanckRun() {
  const { N, dtau, T, bins, lo, hi, start } = FP, rng = makeRng(11), steps = Math.round(T / dtau);
  const X = new Float64Array(N), dx = (hi - lo) / bins, sq = Math.sqrt(2 * dtau);
  for (let i = 0; i < N; i++) X[i] = start[0] + start[1] * rng.gauss();
  const pBin = Array.from({ length: bins }, (_, b) => {
    let s = 0; for (let j = 0; j < 8; j++) s += p1(lo + (b + (j + 0.5) / 8) * dx) / 8; return s;
  });
  const frames = [], kl = [], every = 5;
  const record = () => {
    const h = new Float32Array(bins);
    for (let i = 0; i < N; i++) { const b = Math.floor((X[i] - lo) / dx); if (b >= 0 && b < bins) h[b] += 1 / (N * dx); }
    let d = 0;
    for (let b = 0; b < bins; b++) if (h[b] > 0) d += h[b] * dx * Math.log(h[b] / Math.max(pBin[b], 1e-12));
    frames.push(h); kl.push(d);
  };
  record();
  for (let k = 1; k <= steps; k++) {
    for (let i = 0; i < N; i++) X[i] += dtau * score1(X[i]) + sq * rng.gauss();
    if (k % every === 0) record();
  }
  // Histogram KL of an exact sample of p: the floor that finite N puts under the estimate.
  return { frames, kl, dtauFrame: dtau * every, pBin, dx };
}

// ---------- Step 5: ULA on one Gaussian N(0, v I) ----------
// Exact stationary variance of x <- (1 - eta/v) x + sqrt(2 eta) xi, stable for eta < 2v.
export const ulaVar = (eta, v) => v / (1 - eta / (2 * v));

// ---------- Step 6: annealed Langevin ----------
export const MIX3 = { means: [[-2.0, -1.15], [2.1, -1.0], [0.05, 1.65]], std: 0.25, w: [0.55, 0.3, 0.15] };
export const ANNEAL = { N: 450, sigHi: 3, sigLo: 0.25, K: 30, gamma: 0.15, Lmax: 16 };
export const sigmas = (L) => L === 1 ? [ANNEAL.sigLo] : Array.from({ length: L }, (_, i) => ANNEAL.sigHi * (ANNEAL.sigLo / ANNEAL.sigHi) ** (i / (L - 1)));
export const levelGMM = (s) => new GMM(MIX3.means, Math.sqrt(MIX3.std ** 2 + s * s), { weights: MIX3.w });
const gmm3 = new GMM(MIX3.means, MIX3.std, { weights: MIX3.w });
export const nearest3 = (x, y) => gmm3.argmaxComponent(x, y, 1);

// Runs annealed ULA with L levels and K steps per level, and single-level ULA at sigLo with the
// same total number of steps. Both start from the same uniform layout and use the same noise.
// Returns frames (one per step) as Float32Array of xy pairs, and the mass per mode at the end.
export function annealRuns(L) {
  const { N, K, gamma } = ANNEAL, init = new Float64Array(2 * N), r0 = makeRng(3);
  for (let i = 0; i < N; i++) { init[2 * i] = (r0.uniform() * 2 - 1) * 3; init[2 * i + 1] = (r0.uniform() * 2 - 1) * 2.3; }
  const run = (levels) => {
    const P = Float64Array.from(init), rng = makeRng(9), frames = [Float32Array.from(P)], lev = [0];
    levels.forEach((s, li) => {
      const g = levelGMM(s), eta = gamma * s * s, sq = Math.sqrt(2 * eta);
      for (let k = 0; k < K; k++) {
        for (let i = 0; i < N; i++) {
          const sc = g.fields(P[2 * i], P[2 * i + 1], 1).score;
          P[2 * i] += eta * sc[0] + sq * rng.gauss();
          P[2 * i + 1] += eta * sc[1] + sq * rng.gauss();
        }
        frames.push(Float32Array.from(P)); lev.push(li);
      }
    });
    return { frames, lev, levels };
  };
  const single = run(Array(L).fill(ANNEAL.sigLo)), anneal = run(sigmas(L));
  return { single, anneal };
}
export function massPerMode(P) {
  const m = [0, 0, 0], n = P.length / 2;
  for (let i = 0; i < n; i++) m[nearest3(P[2 * i], P[2 * i + 1])] += 1 / n;
  return m;
}
