// Empirical and smoothed denoisers of the §2.7 tutorial. No DOM access, so the numbers can be checked in
// Node. Convention: x_t = alpha_t x1 + sigma_t x0, t = 0 noise, t = 1 data, linear path alpha_t = t,
// sigma_t = 1 - t. The data are the 2D mixture of §2.2, padded with independent N(0, s^2) coordinates
// when the dimension d is larger than 2.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX2, gmm2 } from '../score-matching/sim.js';

export { MIX2, gmm2 };
export const alpha = (t) => t, sigma = (t) => 1 - t;
export const PAD_SD = MIX2.std;

// ---------- Training sets ----------
// A stream of points of p_data in dimension d; training sets are its first n points, so they are nested.
const streams = new Map();
export function stream(seed, d, nmax = 2048) {
  const key = `${seed}|${d}`;
  if (!streams.has(key)) {
    const r = makeRng(seed), y = new Float64Array(nmax * d), k = new Uint8Array(nmax);
    for (let i = 0; i < nmax; i++) {
      const s = gmm2.sample(r); y[i * d] = s.x; y[i * d + 1] = s.y; k[i] = s.k;
      for (let j = 2; j < d; j++) y[i * d + j] = PAD_SD * r.gauss();
    }
    streams.set(key, { y, k, d });
  }
  return streams.get(key);
}
export const SEED_A = 301, SEED_B = 302;
// Training set of size n, as a view on the stream.
export const train = (seed, d, n) => { const s = stream(seed, d); return { y: s.y.subarray(0, n * d), k: s.k.subarray(0, n), n, d }; };

// ---------- The denoiser of a training set, smoothed with bandwidth h ----------
// Noising the smoothed empirical law (1/n) sum_i N(y_i, h^2 I) gives a mixture with components
// N(alpha y_i, v I), v = alpha^2 h^2 + sigma^2. Its denoiser is
//   xhat_1(x) = sum_i w_i (y_i + (alpha h^2 / v)(x - alpha y_i)),  w_i = softmax_i(-|x - alpha y_i|^2 / (2v)).
// h = 0 is the empirical optimum of the survey. Writes xhat into out and the weights into w.
// T may carry log weights logw (the mixture p_data itself is such a T with h = its component width).
export function denoise(T, x, t, h, out, w) {
  const { y, n, d, logw } = T, a = alpha(t), v = a * a * h * h + sigma(t) ** 2;
  let mx = -Infinity;
  for (let i = 0; i < n; i++) {
    let q = 0;
    for (let j = 0; j < d; j++) { const e = x[j] - a * y[i * d + j]; q += e * e; }
    w[i] = -q / (2 * v) + (logw ? logw[i] : 0); if (w[i] > mx) mx = w[i];
  }
  let z = 0;
  for (let i = 0; i < n; i++) { w[i] = Math.exp(w[i] - mx); z += w[i]; }
  const c = a * h * h / v;
  out.fill(0);
  for (let i = 0; i < n; i++) {
    w[i] /= z; if (w[i] < 1e-12) continue;
    for (let j = 0; j < d; j++) out[j] += w[i] * (y[i * d + j] + c * (x[j] - a * y[i * d + j]));
  }
  return out;
}

// ---------- Probability-flow ODE ----------
// On the linear path u_t = (xhat_1 - x)/(1 - t). In s = log sigma_t this is dx/ds = x - xhat_1, which RK4
// integrates on a uniform grid of s from sigma = 1 (t = 0) to sigma = SIG_END.
export const SIG_END = 1e-3, NSTEP = 60;
export const S_GRID = Float64Array.from({ length: NSTEP + 1 }, (_, k) => Math.log(SIG_END) * k / NSTEP);
export const T_GRID = S_GRID.map((s) => 1 - Math.exp(s));
// Noise draws shared by every model, so paths with the same index start at the same point.
export function noise(seed, d, m) { const r = makeRng(seed), z = new Float64Array(m * d); for (let i = 0; i < z.length; i++) z[i] = r.gauss(); return z; }

// Integrates m paths; returns end points, and if keep is set the first two coordinates at every grid time
// plus the trajectory statistics of step 3 (largest weight and largest class weight).
export function flow(T, h, x0, m, { keep = false, every = 1 } = {}) {
  const { d, n } = T, X = Float64Array.from(x0.subarray(0, m * d)), out = new Float64Array(d), w = new Float64Array(n);
  const k1 = new Float64Array(d), k2 = new Float64Array(d), k3 = new Float64Array(d), k4 = new Float64Array(d), tmp = new Float64Array(d);
  const nk = Math.floor(NSTEP / every) + 1;
  const path = keep ? new Float32Array(m * nk * 2) : null, wmax = keep ? new Float32Array(m * nk) : null, cmax = keep ? new Float32Array(m * nk) : null;
  const f = (x, s, dst) => { denoise(T, x, 1 - Math.exp(s), h, out, w); for (let j = 0; j < d; j++) dst[j] = x[j] - out[j]; };
  const stats = (x, s, i, kk) => {
    denoise(T, x, 1 - Math.exp(s), h, out, w);
    let wm = 0; const cw = [0, 0, 0];
    for (let q = 0; q < n; q++) { if (w[q] > wm) wm = w[q]; cw[T.k[q]] += w[q]; }
    path[(i * nk + kk) * 2] = x[0]; path[(i * nk + kk) * 2 + 1] = x[1]; wmax[i * nk + kk] = wm; cmax[i * nk + kk] = Math.max(...cw);
  };
  for (let i = 0; i < m; i++) {
    const x = X.subarray(i * d, (i + 1) * d);
    if (keep) stats(x, S_GRID[0], i, 0);
    for (let k = 0; k < NSTEP; k++) {
      const s = S_GRID[k], ds = S_GRID[k + 1] - s;
      f(x, s, k1);
      for (let j = 0; j < d; j++) tmp[j] = x[j] + 0.5 * ds * k1[j]; f(tmp, s + 0.5 * ds, k2);
      for (let j = 0; j < d; j++) tmp[j] = x[j] + 0.5 * ds * k2[j]; f(tmp, s + 0.5 * ds, k3);
      for (let j = 0; j < d; j++) tmp[j] = x[j] + ds * k3[j]; f(tmp, s + ds, k4);
      for (let j = 0; j < d; j++) x[j] += ds * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]) / 6;
      if (keep && (k + 1) % every === 0) stats(x, S_GRID[k + 1], i, (k + 1) / every);
    }
  }
  return { X, path, wmax, cmax, nk, m, d };
}

// ---------- Copy test ----------
// A sample counts as a copy when its nearest training point is much closer than the second nearest:
// d1 / d2 < 1/3, the criterion used to detect extracted training images. Returns the ratios and the index
// of the nearest training point per sample.
export function nearest(T, X, m) {
  const { y, n, d } = T, ratio = new Float64Array(m), idx = new Int32Array(m), d1s = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let a = Infinity, b = Infinity, ia = -1;
    for (let q = 0; q < n; q++) {
      let s = 0; for (let j = 0; j < d; j++) { const e = X[i * d + j] - y[q * d + j]; s += e * e; }
      if (s < a) { b = a; a = s; ia = q; } else if (s < b) b = s;
    }
    ratio[i] = Math.sqrt(a / b); idx[i] = ia; d1s[i] = Math.sqrt(a);
  }
  return { ratio, idx, d1: d1s };
}

// ---------- p_data as a T, so the same code gives its exact field ----------
export const DATA2 = { y: Float64Array.from(MIX2.means.flat()), k: Uint8Array.from([0, 1, 2]), n: 3, d: 2, logw: MIX2.w.map(Math.log) };
export const H_DATA = MIX2.std;
// log p_data of a 2D point.
export function logp2(x, y) {
  const v = MIX2.std ** 2; let s = 0;
  MIX2.means.forEach(([mx, my], k) => { s += MIX2.w[k] * Math.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * v)) / (2 * Math.PI * v); });
  return Math.log(s);
}
// Mean of log p_data over the first two coordinates of m points in dimension d.
export function meanLogp(X, m, d) { let s = 0; for (let i = 0; i < m; i++) s += logp2(X[i * d], X[i * d + 1]) / m; return s; }
// Reference value: the same average over fresh samples of p_data (the negative entropy of the 2D mixture).
export const LOGP_REF = (() => { const r = makeRng(77); let s = 0; const M = 20000; for (let i = 0; i < M; i++) { const q = gmm2.sample(r); s += logp2(q.x, q.y) / M; } return s; })();

// ---------- Steps 2 to 4: cached runs ----------
export const NS2 = 200, NS3 = 100, NS4 = 64;
export const NS_LIST = [8, 16, 32, 64, 128, 256, 512], D_LIST = [2, 4, 8, 16];
const memo = (fn, size = 24) => { const m = new Map(); return (...a) => { const k = a.join('|'); if (!m.has(k)) { if (m.size >= size) m.delete(m.keys().next().value); m.set(k, fn(...a)); } return m.get(k); }; };
const Z2 = noise(401, 2, NS2);
// Step 2: paths of the empirical field and of the p_data field from the same noise.
export const run2 = memo((n) => {
  const T = train(SEED_A, 2, n), F = flow(T, 0, Z2, NS2, { keep: true }), near = nearest(T, F.X, NS2);
  const copies = near.ratio.reduce((a, r) => a + (r < 1 / 3), 0), hit = new Set(near.idx).size;
  return { T, F, near, copies, hit };
});
export const DATA_RUN = (() => { const F = flow(DATA2, H_DATA, Z2, NS2, { keep: true }); return { F }; })();
// Copies of the p_data samples, measured against the training set of size n (for comparison in the readout).
export const dataCopies = memo((n) => { const T = train(SEED_A, 2, n), near = nearest(T, DATA_RUN.F.X, NS2); return near.ratio.reduce((a, r) => a + (r < 1 / 3), 0); });

// Step 3: speciation and collapse times from the mean class certainty and the mean largest weight.
export const run3 = memo((n, d) => {
  const T = train(SEED_A, d, n), F = flow(T, 0, noise(402, d, NS3), NS3, { keep: true });
  const nk = F.nk, cls = new Float64Array(nk), pt = new Float64Array(nk);
  for (let k = 0; k < nk; k++) for (let i = 0; i < NS3; i++) { cls[k] += F.cmax[i * nk + k] / NS3; pt[k] += F.wmax[i * nk + k] / NS3; }
  const cross = (c, lev = 0.9) => { for (let k = 1; k < nk; k++) if (c[k] >= lev && c[k - 1] < lev) { const f = (lev - c[k - 1]) / (c[k] - c[k - 1]); return T_GRID[k - 1] + f * (T_GRID[k] - T_GRID[k - 1]); } return c[0] >= lev ? 0 : null; };
  const end = Uint8Array.from(nearest(T, F.X, NS3).idx, (i) => T.k[i]);
  return { T, F, cls, pt, end, tS: cross(cls), tC: cross(pt) };
});

// Step 4: two models trained on disjoint sets A and B, from the same noise.
// Gaussian model: mean and covariance of the training set; its flow map is x_1 = mu + Sigma^{1/2} x_0.
function symSqrt(S, d) {
  // Jacobi eigenvalue iterations on a copy of S (row major), then V diag(sqrt(max(l, 0))) V^T.
  const A = Float64Array.from(S), V = new Float64Array(d * d); for (let i = 0; i < d; i++) V[i * d + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) off += A[i * d + j] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) {
      const apq = A[p * d + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * d + q] - A[p * d + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < d; k++) { const akp = A[k * d + p], akq = A[k * d + q]; A[k * d + p] = c * akp - s * akq; A[k * d + q] = s * akp + c * akq; }
      for (let k = 0; k < d; k++) { const apk = A[p * d + k], aqk = A[q * d + k]; A[p * d + k] = c * apk - s * aqk; A[q * d + k] = s * apk + c * aqk; }
      for (let k = 0; k < d; k++) { const vkp = V[k * d + p], vkq = V[k * d + q]; V[k * d + p] = c * vkp - s * vkq; V[k * d + q] = s * vkp + c * vkq; }
    }
  }
  const R = new Float64Array(d * d);
  for (let k = 0; k < d; k++) { const l = Math.sqrt(Math.max(A[k * d + k], 0)); for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) R[i * d + j] += V[i * d + k] * l * V[j * d + k]; }
  return R;
}
export function gaussFlow(T, x0, m) {
  const { y, n, d } = T, mu = new Float64Array(d), S = new Float64Array(d * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mu[j] += y[i * d + j] / n;
  for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a * d + b] += (y[i * d + a] - mu[a]) * (y[i * d + b] - mu[b]) / n;
  const R = symSqrt(S, d), X = new Float64Array(m * d);
  for (let i = 0; i < m; i++) for (let a = 0; a < d; a++) { let v = mu[a]; for (let b = 0; b < d; b++) v += R[a * d + b] * x0[i * d + b]; X[i * d + a] = v; }
  return { X };
}
export const H_SMOOTH = 0.3, MODELS = ['empirical', 'smoothed', 'gaussian'];
export const run4 = memo((model, n, d) => {
  const z = noise(403, d, NS4), A = train(SEED_A, d, n), B = train(SEED_B, d, n);
  const go = (T) => (model === 'gaussian' ? gaussFlow(T, z, NS4) : flow(T, model === 'smoothed' ? H_SMOOTH : 0, z, NS4));
  const XA = go(A).X, XB = go(B).X, near = nearest(A, XA, NS4);
  let dist = 0, scale = 0;
  for (let i = 0; i < NS4; i++) { let s = 0; for (let j = 0; j < d; j++) s += (XA[i * d + j] - XB[i * d + j]) ** 2; dist += Math.sqrt(s) / NS4; }
  // Scale for the distance: the mean distance between two independent samples of p_data in dimension d.
  return { XA, XB, A, B, dist, copies: near.ratio.reduce((a, r) => a + (r < 1 / 3), 0), logp: meanLogp(XA, NS4, d) };
});
// Mean distance between two independent draws of p_data in dimension d, estimated once per d.
export const PAIR_DIST = memo((d) => { const s = stream(990, d, 4000).y; let m = 0; for (let i = 0; i < 2000; i++) { let q = 0; for (let j = 0; j < d; j++) q += (s[i * d + j] - s[(i + 2000) * d + j]) ** 2; m += Math.sqrt(q) / 2000; } return m; });

// ---------- Step 5: local memorization ----------
// 160 points of p_data, five isolated points far from the modes and one point repeated ten times.
export const NS5 = 400;
export const SET5 = (() => {
  const r = makeRng(305), pts = [], grp = [];
  for (let i = 0; i < 160; i++) { const q = gmm2.sample(r); pts.push(q.x, q.y); grp.push(0); }
  [[-3.1, 1.9], [3.0, 1.6], [2.9, -2.4], [-3.0, -2.3], [-0.2, -2.6]].forEach(([x, y]) => { pts.push(x, y); grp.push(1); });
  for (let i = 0; i < 10; i++) { pts.push(2.7, 0.7); grp.push(2); }
  const n = grp.length;
  return { y: Float64Array.from(pts), k: new Uint8Array(n), grp: Uint8Array.from(grp), n, d: 2 };
})();
// Distinct points of SET5, for the copy test (a duplicate must not count as its own second neighbour).
const DISTINCT5 = (() => { const keep = []; for (let i = 0; i < SET5.n; i++) if (SET5.grp[i] !== 2 || i === SET5.n - 10) keep.push(i); return { y: Float64Array.from(keep.flatMap((i) => [SET5.y[2 * i], SET5.y[2 * i + 1]])), grp: Uint8Array.from(keep.map((i) => SET5.grp[i])), n: keep.length, d: 2 }; })();
const Z5 = noise(405, 2, NS5);
// Share of fresh draws of p_data that pass the copy test against the distinct points of SET5, for comparison.
export const BASE5 = (() => { const r = makeRng(406), m = 2000, X = new Float64Array(2 * m); for (let i = 0; i < m; i++) { const q = gmm2.sample(r); X[2 * i] = q.x; X[2 * i + 1] = q.y; } return nearest(DISTINCT5, X, m).ratio.reduce((a, v) => a + (v < 1 / 3), 0) / m; })();
export const H5_LIST = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4];
export const run5 = memo((h) => {
  const F = flow(SET5, h, Z5, NS5), near = nearest(DISTINCT5, F.X, NS5);
  // Each sample is assigned to the group of its nearest distinct training point.
  const tot = [0, 0, 0], cp = [0, 0, 0], copy = new Uint8Array(NS5), grp = new Uint8Array(NS5);
  for (let i = 0; i < NS5; i++) { const g = DISTINCT5.grp[near.idx[i]]; grp[i] = g; tot[g]++; if (near.ratio[i] < 1 / 3) { cp[g]++; copy[i] = 1; } }
  return { X: F.X, copy, grp, tot, cp };
}, 12);

// ---------- Step 6: a patch denoiser on 12 x 12 digits ----------
// Eight digits drawn as polylines, rendered with 4 x 4 supersampling; pixels in [-1, 1], background -1.
export const N_IMG = 12, BG = -1;
const arc = (cx, cy, rx, ry, a0, a1, k = 16) => Array.from({ length: k + 1 }, (_, i) => { const a = Math.PI * (a0 + (a1 - a0) * i / k); return [cx + rx * Math.cos(a), cy - ry * Math.sin(a)]; });
const STROKES = [
  [arc(6, 6, 3, 4.2, 0, 2, 28)],
  [[[4.2, 3.4], [6.4, 1.8], [6.4, 10.2]]],
  [[...arc(6, 4.6, 3, 2.8, 0.9, -0.2, 12), [3, 10.2], [9.2, 10.2]]],
  [arc(5.8, 3.9, 2.8, 2.1, 0.85, -0.5, 12), arc(5.8, 8.1, 3, 2.1, 0.5, -0.85, 12)],
  [[[7.6, 10.2], [7.6, 1.8], [2.6, 7.8], [9.6, 7.8]]],
  [[[8.8, 1.8], [3.8, 1.8], [3.4, 5.6], ...arc(5.8, 7.6, 3.1, 2.6, 0.75, -0.8, 14)]],
  [[...arc(6.6, 6.4, 3.4, 4.6, 0.45, 1.0, 6), ...arc(6, 7.6, 2.9, 2.6, 1.0, 3.0, 22)]],
  [[[2.8, 1.8], [9.2, 1.8], [5.2, 10.2]]],
];
function render(polys) {
  const N = N_IMG, segs = [];
  for (const p of polys) for (let i = 0; i + 1 < p.length; i++) segs.push([...p[i], ...p[i + 1]]);
  const ds = (px, py, [x0, y0, x1, y1]) => { const dx = x1 - x0, dy = y1 - y0, t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy || 1))); return Math.hypot(px - x0 - t * dx, py - y0 - t * dy); };
  const out = new Float64Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    let hit = 0;
    for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) { const px = i + (a + 0.5) / 4, py = j + (b + 0.5) / 4; if (segs.some((sg) => ds(px, py, sg) < 0.85)) hit++; }
    out[j * N + i] = 2 * hit / 16 - 1;
  }
  return out;
}
export const DIGITS = STROKES.map(render);
// Equivariant local score machine (Kamb and Ganguli), with a local shift window: pixel p of xhat_1 is the
// average of training pixels y_i[q] over images i and positions q within s of p, weighted by how well the
// P x P patch of x_t around p matches alpha_t times the patch of y_i around q. Pixels outside are background.
// P = 0 means the whole image (the empirical optimum when s = 0).
export function els(x, t, P, s, out) {
  const N = N_IMG, a = alpha(t), v2 = 2 * sigma(t) ** 2, r = P ? (P - 1) / 2 : N;
  const get = (im, i, j) => (i < 0 || j < 0 || i >= N || j >= N ? BG : im[j * N + i]);
  const L = [], Y = [];
  for (let pj = 0; pj < N; pj++) for (let pi = 0; pi < N; pi++) {
    let mx = -Infinity, c = 0;
    const r0 = P ? r : 0;
    for (const im of DIGITS) for (let qj = pj - s; qj <= pj + s; qj++) for (let qi = pi - s; qi <= pi + s; qi++) {
      let q = 0;
      if (P) { for (let dj = -r0; dj <= r0; dj++) for (let di = -r0; di <= r0; di++) { const e = get(x, pi + di, pj + dj) - a * get(im, qi + di, qj + dj); q += e * e; } }
      else { for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const e = x[j * N + i] - a * get(im, i + qi - pi, j + qj - pj); q += e * e; } }
      L[c] = -q / v2; Y[c] = get(im, qi, qj); if (L[c] > mx) mx = L[c]; c++;
    }
    let z = 0, m = 0; for (let k = 0; k < c; k++) { const e = Math.exp(L[k] - mx); z += e; m += e * Y[k]; }
    out[pj * N + pi] = m / z;
  }
  return out;
}
// Sampling in K Euler steps of dx/ds = x - xhat_1 on a uniform grid of s = log sigma, down to sigma = 2e-3.
export const K_ELS = 30, M_ELS = 6;
const S_ELS = Float64Array.from({ length: K_ELS + 1 }, (_, k) => Math.log(2e-3) * k / K_ELS);
export function elsStart(seed = 600) {
  const r = makeRng(seed);
  return { k: 0, xs: Array.from({ length: M_ELS }, () => Float64Array.from({ length: N_IMG * N_IMG }, () => r.gauss())), xh: Array.from({ length: M_ELS }, () => new Float64Array(N_IMG * N_IMG)) };
}
// One Euler step for all samples; returns false when done.
export function elsStep(st, P, s) {
  if (st.k >= K_ELS) return false;
  const t = 1 - Math.exp(S_ELS[st.k]), ds = S_ELS[st.k + 1] - S_ELS[st.k];
  st.xs.forEach((x, m) => { const xh = els(x, t, P, s, st.xh[m]); for (let i = 0; i < x.length; i++) x[i] += ds * (x[i] - xh[i]); });
  st.k++; st.t = 1 - Math.exp(S_ELS[st.k]);
  return st.k < K_ELS;
}
// Nearest training digit of an image, and the copy test d1/d2 < 1/3 of step 2.
export function nearestDigit(x) {
  const d = DIGITS.map((im) => Math.sqrt(im.reduce((a, v, i) => a + (v - x[i]) ** 2, 0))).map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  return { idx: d[0][1], d1: d[0][0], ratio: d[0][0] / d[1][0], copy: d[0][0] / d[1][0] < 1 / 3 };
}
