// Targets and estimators for the §2.2 tutorial. No DOM access, so the numbers can be checked in Node.
// Interpolant x_t = alpha_t x1 + sigma_t x0 with the linear schedule alpha_t = t, sigma_t = 1 - t:
// t = 0 is noise, t = 1 is data. All fields are exact because the data are Gaussian mixtures.
import { GMM } from '../../animations/lib/gmm.js';
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, p1, score1 } from '../ebm-langevin/sim.js';

export { MIX1, p1, score1 };
export const alpha = (t) => t, sigma = (t) => 1 - t;

// ---------- 1D data (steps 1, 2 and 5) ----------
const phi = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
// Marginal p_t of the 1D mixture: components N(alpha mu_k, alpha^2 sd_k^2 + sigma^2).
export function pt1(x, t) {
  const a = alpha(t), s = sigma(t);
  return MIX1.mu.reduce((acc, m, k) => acc + MIX1.w[k] * phi(x, a * m, a * a * MIX1.sd[k] ** 2 + s * s), 0);
}
export function scoret1(x, t) {
  const a = alpha(t), s = sigma(t);
  let num = 0, den = 0;
  MIX1.mu.forEach((m, k) => { const v = a * a * MIX1.sd[k] ** 2 + s * s, f = MIX1.w[k] * phi(x, a * m, v); num -= f * (x - a * m) / v; den += f; });
  return num / Math.max(den, 1e-300);
}
export function sample1(rng) {
  const u = rng.uniform(); let k = 0, c = MIX1.w[0];
  while (u > c && k < 2) c += MIX1.w[++k];
  return MIX1.mu[k] + MIX1.sd[k] * rng.gauss();
}
// Fixed samples of p (step 2) and fixed (x1, x0) pairs (step 5), drawn once so every frame agrees.
export const SAMPLES1 = (() => { const r = makeRng(21); return Float64Array.from({ length: 200 }, () => sample1(r)); })();
export const PAIRS1 = (() => {
  const r = makeRng(22), n = 2500, x1 = new Float64Array(n), x0 = new Float64Array(n);
  for (let i = 0; i < n; i++) { x1[i] = sample1(r); x0[i] = r.gauss(); }
  return { n, x1, x0 };
})();

// Score model of steps 1 and 2: the score of N(m, v), s_theta(x) = -(x - m) / v, with theta = (m, v).
export const model = (x, m, v) => -(x - m) / v;
// Explicit and implicit objectives by quadrature on [-6, 6]. C = J_ESM - J_ISM = 1/2 E_p ||grad log p||^2.
const GX = Array.from({ length: 1201 }, (_, i) => -6 + 12 * i / 1200), GDX = 12 / 1200;
export function objectives(m, v) {
  let esm = 0, ism = 0;
  for (const x of GX) {
    const p = p1(x), s = model(x, m, v);
    esm += 0.5 * p * (s - score1(x)) ** 2 * GDX;
    ism += p * (0.5 * s * s - 1 / v) * GDX;
  }
  return { esm, ism };
}
// Sample estimate of J_ISM from SAMPLES1 and its gradient in (m, log v), for the fit button.
export function ismSample(m, v) {
  let j = 0, gm = 0, gl = 0; const n = SAMPLES1.length;
  for (const x of SAMPLES1) {
    const d = x - m;
    j += (0.5 * d * d / (v * v) - 1 / v) / n;
    gm += -d / (v * v) / n;
    gl += (-d * d / (v * v) + 1 / v) / n;     // d/d(log v) = v d/dv
  }
  return { j, gm, gl };
}

// ---------- 2D data (steps 3, 5, 6 and 7) ----------
export const MIX2 = { means: [[-1.4, -0.55], [1.35, -0.85], [0.15, 1.3]], std: 0.5, w: [0.45, 0.3, 0.25] };
export const gmm2 = new GMM(MIX2.means, MIX2.std, { weights: MIX2.w });
export const W2 = (() => { const z = MIX2.w.reduce((a, b) => a + b, 0); return MIX2.w.map((x) => x / z); })();
// Data density p (t = 1) for the heat maps.
export function p2(x, y) {
  const v = MIX2.std ** 2;
  return MIX2.means.reduce((a, [mx, my], k) => a + W2[k] * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * v)) / (2 * Math.PI * v), 0);
}
export function pt2(x, y, t) {
  const a = alpha(t), s = sigma(t);
  return MIX2.means.reduce((acc, [mx, my], k) => {
    const v = a * a * MIX2.std ** 2 + s * s;
    return acc + W2[k] * Math.exp(-((x - a * mx) ** 2 + (y - a * my) ** 2) / (2 * v)) / (2 * Math.PI * v);
  }, 0);
}
// Fixed data points and noise draws in 2D.
export const PAIRS2 = (() => {
  const r = makeRng(31), n = 300, x1 = new Float64Array(2 * n), x0 = new Float64Array(2 * n), k = new Uint8Array(n);
  for (let i = 0; i < n; i++) { const s = gmm2.sample(r); x1[2 * i] = s.x; x1[2 * i + 1] = s.y; k[i] = s.k; x0[2 * i] = r.gauss(); x0[2 * i + 1] = r.gauss(); }
  return { n, x1, x0, k };
})();
// Posterior of x1 given x_t = x: component k with responsibility r_k, and inside it
// N(mu_k + (a s^2 / v)(x - a mu_k), (s^2 sigma^2 / v) I) with v = a^2 s^2 + sigma^2.
// Draws M posterior samples with fixed uniforms and normals, so the picture depends only on (x, t).
const POST = (() => { const r = makeRng(41), M = 60; return { M, u: Float64Array.from({ length: M }, () => r.uniform()), g: Float64Array.from({ length: 2 * M }, () => r.gauss()) }; })();
export function posteriorSamples(x, y, t) {
  const a = alpha(t), s = sigma(t), s2 = MIX2.std ** 2, v = a * a * s2 + s * s;
  const lr = MIX2.means.map(([mx, my], k) => Math.log(W2[k]) - ((x - a * mx) ** 2 + (y - a * my) ** 2) / (2 * v));
  const mx = Math.max(...lr), r = lr.map((l) => Math.exp(l - mx)), z = r.reduce((p, q) => p + q, 0);
  const cum = []; let c = 0; r.forEach((q) => cum.push((c += q / z)));
  const sd = Math.sqrt(s2 * s * s / v), out = [];
  for (let i = 0; i < POST.M; i++) {
    let k = cum.findIndex((q) => POST.u[i] <= q); if (k < 0) k = 2;
    const [ux, uy] = MIX2.means[k];
    out.push({ k, x: ux + (a * s2 / v) * (x - a * ux) + sd * POST.g[2 * i], y: uy + (a * s2 / v) * (y - a * uy) + sd * POST.g[2 * i + 1] });
  }
  return { samples: out, resp: r.map((q) => q / z) };
}
// Exact fields at time t: score, E[x1 | x_t] and E[x0 | x_t] (gmm.js uses the same linear schedule).
export const fields2 = (x, y, t) => gmm2.fields(x, y, t);
// Loss weights of table 1 for the linear path, as functions of t.
export const WEIGHTS = {
  data: () => 1,
  noise: (t) => alpha(t) ** 2 / sigma(t) ** 2,
  score: (t) => alpha(t) ** 2 / sigma(t) ** 4,
  velocity: (t) => (1 + alpha(t) / sigma(t)) ** 2,   // (alpha' - sigma' alpha / sigma)^2 with alpha' = 1, sigma' = -1
};
