// Numbers of the §2.8 tutorial. No DOM access, so they can be checked in Node. Convention:
// x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data, linear path alpha_t = t, sigma_t = 1 - t.
// Nothing here is trained: every quantity is exact arithmetic, an exact Gaussian-mixture field, or the
// best linear predictor of a given width, which depends only on second moments.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX2, gmm2, logp2, LOGP_REF, DIGITS, N_IMG } from '../generalization/sim.js';

export { MIX2, gmm2, logp2, LOGP_REF, DIGITS, N_IMG };
export const alpha = (t) => t, sigma = (t) => 1 - t;
const gaussArray = (seed, m) => { const r = makeRng(seed); return Float64Array.from({ length: m }, () => r.gauss()); };

// ---------- Step 1: from pixels to tokens ----------
// Coordinates of the input, of the latent and the token count N = H W (T/f_T)/(f^2 p^2).
export function tokens({ H, W, T = 1, fT = 1, f, p, C }) {
  const lat = (H / f) * (W / f) * Math.ceil(T / fT), N = lat / (p * p);
  return { pixels: H * W * T * 3, latent: lat * C, N, pairs: N * N, reduction: (H * W * T * 3) / (lat * C) };
}
export const PRESETS = {
  sd: { name: 'SD-VAE image', H: 1024, W: 1024, T: 1, fT: 1, f: 8, p: 2, C: 4 },
  dcae: { name: 'DC-AE image', H: 1024, W: 1024, T: 1, fT: 1, f: 32, p: 1, C: 32 },
  video: { name: 'Video, 121 frames', H: 704, W: 1216, T: 121, fT: 8, f: 32, p: 1, C: 128 },
};

// ---------- Step 2: the same noise level hides less in more pixels ----------
// A 12 x 12 digit upsampled by b has (12b)^2 pixels. The average of x_t over a b x b block is
// alpha * (base pixel) + sigma * (mean of b^2 independent N(0, 1)), whose noise has standard deviation sigma/b.
export const B_LIST = [1, 2, 4, 8];
export const snrEff = (t, b) => (b * alpha(t) / sigma(t)) ** 2;
// The time at resolution factor b whose coarse content has the SNR of time t at the base resolution.
export const shiftT = (t, b) => t / (t + b * (1 - t));
const NOISE2 = Object.fromEntries(B_LIST.map((b) => [b, gaussArray(500 + b, (N_IMG * b) ** 2)]));
export function upsample(im, b) {
  const n = N_IMG * b, out = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) out[j * n + i] = im[Math.floor(j / b) * N_IMG + Math.floor(i / b)];
  return out;
}
export function blockAvg(im, b) {
  const n = N_IMG * b, out = new Float64Array(N_IMG * N_IMG);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) out[Math.floor(j / b) * N_IMG + Math.floor(i / b)] += im[j * n + i] / (b * b);
  return out;
}
// x_t of the digit upsampled by b, with the fixed noise of that resolution.
export function noisy(im, b, t) {
  const up = upsample(im, b), z = NOISE2[b], a = alpha(t), s = sigma(t);
  return up.map((v, i) => a * v + s * z[i]);
}
// Empirical standard deviation of the block-averaged noise, to check sigma/b on the page.
export const blockNoiseSD = (b) => { const m = blockAvg(NOISE2[b], b); return Math.sqrt(m.reduce((a, v) => a + v * v, 0) / m.length); };

// ---------- Step 3: an encoder that cuts the plane into strips ----------
// On [-L, L) the x-axis is cut into k strips of width w = 2L/k, and strip j is moved to position pi(j).
// The map is an isometry on each strip and exactly invertible; points outside [-L, L) stay where they are.
export const L3 = 3.6, K3_LIST = [1, 2, 3, 4, 6, 8, 12], H3_LIST = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];
const PERM = new Map();
function perm(k) {
  if (!PERM.has(k)) {
    const r = makeRng(700 + k);
    let p;
    // A fixed permutation without fixed points, so every strip moves.
    do { p = Array.from({ length: k }, (_, i) => i); for (let i = k - 1; i > 0; i--) { const j = Math.floor(r.uniform() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } } while (k > 1 && p.some((v, i) => v === i));
    const inv = []; p.forEach((v, i) => { inv[v] = i; });
    PERM.set(k, { p, inv });
  }
  return PERM.get(k);
}
export const permutation = (k) => perm(k).p;
export function encode(x, k) {
  if (k === 1 || x < -L3 || x >= L3) return x;
  const w = 2 * L3 / k, j = Math.min(k - 1, Math.floor((x + L3) / w));
  return x + (perm(k).p[j] - j) * w;
}
export function decode(z, k) {
  if (k === 1 || z < -L3 || z >= L3) return z;
  const w = 2 * L3 / k, jj = Math.min(k - 1, Math.floor((z + L3) / w));
  return z + (perm(k).inv[jj] - jj) * w;
}
export const N_TRAIN3 = 400, NS3 = 600;
export const TRAIN3 = (() => { const r = makeRng(710), y = new Float64Array(2 * N_TRAIN3); for (let i = 0; i < N_TRAIN3; i++) { const q = gmm2.sample(r); y[2 * i] = q.x; y[2 * i + 1] = q.y; } return y; })();
// Samples of the smoothed model (1/n) sum_i N(z_i, h^2 I): a training latent plus h times a fixed noise draw.
const IDX3 = (() => { const r = makeRng(720); return Uint16Array.from({ length: NS3 }, () => Math.floor(r.uniform() * N_TRAIN3)); })();
const Z3 = gaussArray(721, 2 * NS3);
// log p_data below which 1% of fresh data falls.
export const Q01 = (() => { const r = makeRng(730), M = 20000, l = new Float64Array(M); for (let i = 0; i < M; i++) { const q = gmm2.sample(r); l[i] = logp2(q.x, q.y); } l.sort(); return l[Math.floor(0.01 * M)]; })();
export function run3(k, h) {
  const lat = new Float64Array(2 * N_TRAIN3), Z = new Float64Array(2 * NS3), X = new Float64Array(2 * NS3);
  let rec = 0;
  for (let i = 0; i < N_TRAIN3; i++) { lat[2 * i] = encode(TRAIN3[2 * i], k); lat[2 * i + 1] = TRAIN3[2 * i + 1]; rec = Math.max(rec, Math.abs(decode(lat[2 * i], k) - TRAIN3[2 * i])); }
  let lp = 0, far = 0;
  for (let i = 0; i < NS3; i++) {
    const q = IDX3[i];
    Z[2 * i] = lat[2 * q] + h * Z3[2 * i]; Z[2 * i + 1] = lat[2 * q + 1] + h * Z3[2 * i + 1];
    X[2 * i] = decode(Z[2 * i], k); X[2 * i + 1] = Z[2 * i + 1];
    const l = logp2(X[2 * i], X[2 * i + 1]); lp += l / NS3; if (l < Q01) far++;
  }
  return { lat, Z, X, rec, logp: lp, far: far / NS3 };
}

// ---------- Step 4: representation alignment through Tweedie ----------
// Frozen encoder f(x1) = P(k | x1), the Bayes classifier of the three components. The REPA target
// E[f(x1) | x_t] equals P(k | x_t) by the tower property, because k -> x1 -> x_t is a Markov chain.
const resp = (x, y, t, out) => { gmm2._resp(x, y, t); out[0] = gmm2._r[0]; out[1] = gmm2._r[1]; out[2] = gmm2._r[2]; return out; };
export const encoderF = (x, y, out = new Float64Array(3)) => resp(x, y, 1, out);
export const repaTarget = (x, y, t, out = new Float64Array(3)) => resp(x, y, t, out);
export const denoiser = (x, y, t) => gmm2.fields(x, y, t).d1;
export const T4 = Float64Array.from({ length: 41 }, (_, i) => Math.min(0.995, i / 40));
export const M4 = 4000;
// R^2 of two predictors of f(x1) along t: the REPA target P(k | x_t), and f(x_hat_1), the feature of the denoised point.
export const R2 = (() => {
  const r = makeRng(740), x1 = new Float64Array(2 * M4), x0 = new Float64Array(2 * M4), F = new Float64Array(3 * M4), f = new Float64Array(3), g = new Float64Array(3);
  for (let i = 0; i < M4; i++) { const q = gmm2.sample(r); x1[2 * i] = q.x; x1[2 * i + 1] = q.y; x0[2 * i] = r.gauss(); x0[2 * i + 1] = r.gauss(); encoderF(q.x, q.y, f); F.set(f, 3 * i); }
  const mean = (A, d) => Array.from({ length: d }, (_, c) => { let s = 0; for (let i = 0; i < M4; i++) s += A[d * i + c]; return s / M4; });
  const tv = (A, d) => { const m = mean(A, d); let s = 0; for (let i = 0; i < M4; i++) for (let c = 0; c < d; c++) s += (A[d * i + c] - m[c]) ** 2; return s; };
  const vf = tv(F, 3), sem = [], fx = [];
  for (const t of T4) {
    const a = alpha(t), s = sigma(t);
    let es = 0, ef = 0;
    for (let i = 0; i < M4; i++) {
      const x = a * x1[2 * i] + s * x0[2 * i], y = a * x1[2 * i + 1] + s * x0[2 * i + 1], [hx, hy] = denoiser(x, y, t);
      repaTarget(x, y, t, g); encoderF(hx, hy, f);
      for (let c = 0; c < 3; c++) { es += (g[c] - F[3 * i + c]) ** 2; ef += (f[c] - F[3 * i + c]) ** 2; }
    }
    sem.push(1 - es / vf); fx.push(1 - ef / vf);
  }
  return { sem, fx };
})();

// ---------- Step 5: which target a network of width r can learn ----------
// The eight digits, as equally likely images of D = 144 pixels, define a mean m and a covariance of rank 7.
// In the basis of its eigenvectors (completed by orthonormalized cosine images) every direction is an
// independent scalar regression, and the best predictor of width r keeps the r directions whose fitted
// values have the largest variance.
export const D5 = N_IMG * N_IMG;
export const MEAN5 = (() => { const m = new Float64Array(D5); DIGITS.forEach((im) => im.forEach((v, i) => { m[i] += v / DIGITS.length; })); return m; })();
// Symmetric eigendecomposition by cyclic Jacobi rotations; returns eigenvalues and eigenvectors (columns of V).
function jacobi(A0, n) {
  const A = A0.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), tn = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(tn * tn + 1), s = tn * c;
      for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  return { vals: A.map((r, i) => r[i]), V };
}
export const BASIS5 = (() => {
  const n = DIGITS.length, Y = DIGITS.map((im) => im.map((v, i) => v - MEAN5[i]));
  const G = Y.map((a) => Y.map((b) => a.reduce((s, v, i) => s + v * b[i], 0) / n));
  const { vals, V } = jacobi(G, n), order = vals.map((v, i) => i).sort((i, j) => vals[j] - vals[i]);
  const E = [], lam = [];
  for (const q of order) {
    if (vals[q] < 1e-9) continue;
    const e = new Float64Array(D5); Y.forEach((y, i) => y.forEach((v, p) => { e[p] += V[i][q] * v; }));
    const nr = Math.hypot(...e); E.push(e.map((v) => v / nr)); lam.push(vals[q]);
  }
  const rank = E.length;
  // Complement: 2D cosine images in order of frequency, orthonormalized against everything before.
  const freqs = []; for (let u = 0; u < N_IMG; u++) for (let v = 0; v < N_IMG; v++) freqs.push([u, v]);
  freqs.sort((a, b) => a[0] + a[1] - b[0] - b[1] || a[0] - b[0]);
  for (const [u, v] of freqs) {
    if (E.length === D5) break;
    const e = new Float64Array(D5);
    for (let j = 0; j < N_IMG; j++) for (let i = 0; i < N_IMG; i++) e[j * N_IMG + i] = Math.cos(Math.PI * (i + 0.5) * u / N_IMG) * Math.cos(Math.PI * (j + 0.5) * v / N_IMG);
    for (let pass = 0; pass < 2; pass++) for (const b of E) { const d = b.reduce((s, x, i) => s + x * e[i], 0); for (let i = 0; i < D5; i++) e[i] -= d * b[i]; }
    const nr = Math.hypot(...e); if (nr < 1e-6) continue;
    E.push(e.map((x) => x / nr)); lam.push(0);
  }
  return { E, lam: Float64Array.from(lam), rank };
})();
export const TARGETS = ['x1', 'x0', 'v'];
const AB = { x1: [1, 0], x0: [0, 1], v: [1, -1] };
// Best linear predictor of width r of one target: returns the coefficient k_j by which the implied
// x_hat_1 multiplies the j-th coordinate of x_t - alpha m, the kept directions and the expected squared error of x_hat_1.
export function linPred(target, t, r) {
  const { lam } = BASIS5, a = alpha(t), s = sigma(t), [ca, cb] = AB[target];
  const cov = Float64Array.from(lam, (l) => ca * a * l + cb * s), vr = Float64Array.from(lam, (l) => a * a * l + s * s);
  const order = Array.from(lam, (_, j) => j).sort((i, j) => cov[j] ** 2 / vr[j] - cov[i] ** 2 / vr[i] || i - j);
  const keep = new Uint8Array(D5); for (let q = 0; q < r; q++) keep[order[q]] = 1;
  const k = new Float64Array(D5); let err = 0;
  for (let j = 0; j < D5; j++) {
    const c = keep[j] ? cov[j] / vr[j] : 0;
    k[j] = target === 'x1' ? c : target === 'x0' ? (1 - s * c) / a : 1 + s * c;
    err += lam[j] - 2 * k[j] * a * lam[j] + k[j] ** 2 * vr[j];
  }
  return { k, keep, err };
}
const NOISE5 = DIGITS.map((_, i) => gaussArray(750 + i, D5));
export const xt5 = (d, t) => DIGITS[d].map((v, i) => alpha(t) * v + sigma(t) * NOISE5[d][i]);
// x_hat_1 = m + sum_j k_j <x_t - alpha m, e_j> e_j.
export function predImage(x, t, P) {
  const { E } = BASIS5, a = alpha(t), out = Float64Array.from(MEAN5);
  for (let j = 0; j < D5; j++) {
    if (!P.k[j]) continue;
    let c = 0; for (let i = 0; i < D5; i++) c += (x[i] - a * MEAN5[i]) * E[j][i];
    c *= P.k[j]; for (let i = 0; i < D5; i++) out[i] += c * E[j][i];
  }
  return out;
}

// ---------- Step 6: attention is the empirical denoiser ----------
export const N6 = 24;
export const TRAIN6 = (() => { const r = makeRng(760), y = new Float64Array(2 * N6); for (let i = 0; i < N6; i++) { const q = gmm2.sample(r); y[2 * i] = q.x; y[2 * i + 1] = q.y; } return y; })();
const softmax = (l) => { const m = Math.max(...l), e = l.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
// Weights as dot-product attention (query x, keys alpha y_i, temperature sigma^2, bias -alpha^2 |y_i|^2 / 2 sigma^2)
// and as the distance softmax of §2.7, with the output of the attention and the largest difference.
export function attention(x, t) {
  const a = alpha(t), v2 = sigma(t) ** 2, la = [], ld = [];
  for (let i = 0; i < N6; i++) {
    const yx = TRAIN6[2 * i], yy = TRAIN6[2 * i + 1];
    la.push((x[0] * a * yx + x[1] * a * yy) / v2 - a * a * (yx * yx + yy * yy) / (2 * v2));
    ld.push(-((x[0] - a * yx) ** 2 + (x[1] - a * yy) ** 2) / (2 * v2));
  }
  const wa = softmax(la), wd = softmax(ld);
  let ox = 0, oy = 0, diff = 0;
  for (let i = 0; i < N6; i++) { ox += wa[i] * TRAIN6[2 * i]; oy += wa[i] * TRAIN6[2 * i + 1]; diff = Math.max(diff, Math.abs(wa[i] - wd[i])); }
  return { wa, wd, out: [ox, oy], diff };
}

// ---------- Step 7: joint attention ----------
// Per layer, with model width d: one weight set (Q, K, V, O and an MLP of width 4d) has 12 d^2 parameters.
// Cross-attention adds Q, K, V, O for the text (4 d^2). MMDiT keeps two full weight sets.
export const ARCHS = {
  cross: { name: 'Cross-attention (U-Net, DiT)', params: (d) => 16 * d * d, blocks: [1, 1, 0, 0], pairs: (ni, nt) => ni * ni + ni * nt },
  mmdit: { name: 'MMDiT', params: (d) => 24 * d * d, blocks: [1, 1, 1, 1], pairs: (ni, nt) => (ni + nt) ** 2 },
  single: { name: 'Single stream', params: (d) => 12 * d * d, blocks: [1, 1, 1, 1], pairs: (ni, nt) => (ni + nt) ** 2 },
};
// Image tokens of a square image of side R with f = 8 and p = 2.
export const imgTokens = (R) => (R / 16) ** 2;
