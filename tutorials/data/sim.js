// Data, fits and objects of the §2.0 primer. No DOM access, so the numbers can be checked in Node.
// The 1D data are the mixture MIX1 of the later tutorials and the 2D data the mixture MIX2 of §2.2,
// so every density here is exact. Only the fits of steps 4, 5 and 7 run on samples.
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, p1, MIX2 } from '../ebm-langevin/sim.js';

export { MIX1, MIX2 };
export const pdata = p1;
export const phi = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (Math.sqrt(2 * Math.PI) * s);
function erf(x) { const s = Math.sign(x), a = Math.abs(x), k = 1 / (1 + 0.3275911 * a); return s * (1 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * Math.exp(-a * a)); }
const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// ---------- Samples from p_data (steps 2 to 7) ----------
// One fixed stream of draws; a step that needs n samples takes the first n, so the sets are nested.
export const NMAX = 10000;
export const SAMPLES = (() => {
  const rng = makeRng(20), out = new Float64Array(NMAX);
  for (let i = 0; i < NMAX; i++) {
    const u = rng.uniform();
    const k = u < MIX1.w[0] ? 0 : u < MIX1.w[0] + MIX1.w[1] ? 1 : 2;
    out[i] = MIX1.mu[k] + MIX1.sd[k] * rng.gauss();
  }
  return out;
})();
// Exact probability of (a, b) under p_data.
export const probInterval = (a, b) => MIX1.mu.reduce((s, m, k) => s + MIX1.w[k] * (Phi((b - m) / MIX1.sd[k]) - Phi((a - m) / MIX1.sd[k])), 0);
export function histogram(n, x0, x1, width) {
  const nb = Math.max(1, Math.round((x1 - x0) / width)), c = new Float64Array(nb);
  for (let i = 0; i < n; i++) { const j = Math.floor((SAMPLES[i] - x0) / width); if (j >= 0 && j < nb) c[j]++; }
  return Array.from(c, (v) => v / (n * width));   // density units, comparable with p_data
}

// ---------- Step 4: a Gaussian and its average log-likelihood ----------
export const NFIT = 200;   // the samples the fits of steps 4 and 7 use
export const logN = (x, m, s) => -0.5 * Math.log(2 * Math.PI * s * s) - (x - m) ** 2 / (2 * s * s);
export function avgLogLik(m, s, n = NFIT) { let a = 0; for (let i = 0; i < n; i++) a += logN(SAMPLES[i], m, s); return a / n; }
export function sampleMoments(n = NFIT) {
  let m = 0; for (let i = 0; i < n; i++) m += SAMPLES[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (SAMPLES[i] - m) ** 2; v /= n;
  return { m, s: Math.sqrt(v) };
}

// ---------- Quadrature on [-7, 7] (steps 6 and 7) ----------
const QN = 2801, QX = Array.from({ length: QN }, (_, i) => -7 + 14 * i / (QN - 1)), QH = 14 / (QN - 1);
const QP = QX.map(pdata);
export const H_DATA = -QP.reduce((a, p, i) => a + (p > 0 ? p * Math.log(p) : 0), 0) * QH;
// KL(p_data || q) for a density q given as a function.
export const klTo = (q) => QP.reduce((a, p, i) => a + (p > 1e-300 ? p * (Math.log(p) - Math.log(Math.max(q(QX[i]), 1e-300))) : 0), 0) * QH;

// ---------- Step 6: the best single Gaussian ----------
// Maximum likelihood with infinitely many samples matches the mean and variance of p_data.
export const POP = (() => {
  const m = MIX1.mu.reduce((a, mu, k) => a + MIX1.w[k] * mu, 0);
  const v = MIX1.mu.reduce((a, mu, k) => a + MIX1.w[k] * (mu * mu + MIX1.sd[k] ** 2), 0) - m * m;
  return { m, s: Math.sqrt(v) };
})();
export const KL_GAUSS = klTo((x) => phi(x, POP.m, POP.s));

// ---------- Step 7: a mixture of K Gaussians fitted by EM ----------
export const mixDens = (c) => (x) => c.w.reduce((a, w, k) => a + w * phi(x, c.mu[k], c.sd[k]), 0);
export const EM_ITERS = 80;
// Runs EM from a seeded start and returns the parameters after every iteration (index 0 is the start).
export function emRun(K, n = NFIT, seed = 3) {
  const rng = makeRng(seed + 17 * K), xs = SAMPLES.subarray(0, n), all = sampleMoments(n);
  // Start: K distinct data points as means, the overall spread as every width, equal weights.
  const pick = new Set();
  while (pick.size < K) pick.add(Math.floor(rng.uniform() * n));
  let c = { mu: [...pick].map((i) => xs[i]).sort((a, b) => a - b), sd: Array(K).fill(all.s), w: Array(K).fill(1 / K) };
  const hist = [c], r = new Float64Array(n * K);
  for (let it = 0; it < EM_ITERS; it++) {
    // E step: responsibilities.
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let k = 0; k < K; k++) { const v = c.w[k] * phi(xs[i], c.mu[k], c.sd[k]); r[i * K + k] = v; z += v; }
      for (let k = 0; k < K; k++) r[i * K + k] /= z || 1;
    }
    // M step: weighted moments, with a floor on the width so that no component collapses onto one point.
    const nc = { mu: [], sd: [], w: [] };
    for (let k = 0; k < K; k++) {
      let nk = 0, m = 0, v = 0;
      for (let i = 0; i < n; i++) { nk += r[i * K + k]; m += r[i * K + k] * xs[i]; }
      m /= nk;
      for (let i = 0; i < n; i++) v += r[i * K + k] * (xs[i] - m) ** 2;
      nc.mu.push(m); nc.sd.push(Math.max(0.05, Math.sqrt(v / nk))); nc.w.push(nk / n);
    }
    c = nc; hist.push(c);
  }
  return hist.map((p) => ({ ...p, ll: avgLogLikMix(p, n), kl: klTo(mixDens(p)) }));
}
function avgLogLikMix(c, n) { let a = 0; for (let i = 0; i < n; i++) a += Math.log(mixDens(c)(SAMPLES[i])); return a / n; }

// ---------- Step 5: samples of the 2D mixture and their moments ----------
export const N2 = 600;
export const SAMPLES2 = (() => {
  const rng = makeRng(21), W = MIX2.w.reduce((a, b) => a + b, 0), out = [];
  for (let i = 0; i < N2; i++) {
    let u = rng.uniform() * W, k = 0;
    while (k < 2 && u > MIX2.w[k]) { u -= MIX2.w[k]; k++; }
    out.push([MIX2.means[k][0] + MIX2.std * rng.gauss(), MIX2.means[k][1] + MIX2.std * rng.gauss()]);
  }
  return out;
})();
export const FIT2 = (() => {
  const m = [0, 1].map((d) => SAMPLES2.reduce((a, p) => a + p[d], 0) / N2);
  const c = (a, b) => SAMPLES2.reduce((s, p) => s + (p[a] - m[a]) * (p[b] - m[b]), 0) / N2;
  const s1 = Math.sqrt(c(0, 0)), s2 = Math.sqrt(c(1, 1));
  return { m, s1, s2, r: c(0, 1) / (s1 * s2) };
})();
// Average log-likelihood of the 2D samples under N(m, Σ) with Σ = [[s1², r s1 s2], [r s1 s2, s2²]].
export function avgLogLik2({ m, s1, s2, r }) {
  const det = s1 * s1 * s2 * s2 * (1 - r * r);
  let a = 0;
  for (const [x, y] of SAMPLES2) {
    const u = (x - m[0]) / s1, v = (y - m[1]) / s2;
    a += -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - (u * u - 2 * r * u * v + v * v) / (2 * (1 - r * r));
  }
  return a / N2;
}
// Number of parameters of one Gaussian in d dimensions with a full covariance.
export const nParams = (d) => d + d * (d + 1) / 2;

// ---------- Step 1: four objects and their vectors ----------
// A 16 x 16 grayscale digit "2", drawn from a distance field so that it needs no image file.
export const IMG = (() => {
  const n = 16, out = [];
  const segs = [];
  // Upper arc, then the diagonal, then the bottom bar, in pixel coordinates.
  for (let i = 0; i < 24; i++) {
    const a0 = Math.PI * (0.92 - 1.17 * i / 24), a1 = Math.PI * (0.92 - 1.17 * (i + 1) / 24);
    segs.push([7.5 + 4 * Math.cos(a0), 6.9 - 3.6 * Math.sin(a0), 7.5 + 4 * Math.cos(a1), 6.9 - 3.6 * Math.sin(a1)]);
  }
  const e = segs[segs.length - 1];
  segs.push([e[2], e[3], 3.4, 13], [3.4, 13, 12.4, 13]);
  const dseg = (px, py, [x0, y0, x1, y1]) => {
    const dx = x1 - x0, dy = y1 - y0, t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(px - x0 - t * dx, py - y0 - t * dy);
  };
  // Each pixel is the share of a 4 x 4 grid of sample points that lie within the pen width of the stroke.
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    let hit = 0;
    for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) hit += Math.min(...segs.map((s) => dseg(i + (a + 0.5) / 4, j + (b + 0.5) / 4, s))) < 1.15 ? 1 : 0;
    out.push(Math.round(100 * hit / 16) / 100);
  }
  return out;
})();

// Caffeine, heavy atoms only: 3D conformer of PubChem CID 2519 (coordinates in Å; the molecule is
// planar, so z is close to 0). Bonds [i, j, order] are from the same record.
export const MOL = {
  name: 'Caffeine, C8H10N4O2 (PubChem CID 2519, heavy atoms)',
  atoms: [[0.47, 2.569, 0.001, 'O'], [-3.127, -0.444, 0, 'O'], [-0.969, -1.312, 0, 'N'], [2.218, 0.141, 0, 'N'], [-1.348, 1.08, 0, 'N'], [1.412, -1.937, 0, 'N'], [0.858, 0.259, -0.001, 'C'], [0.39, -1.026, 0, 'C'], [0.031, 1.422, -0.001, 'C'], [-1.906, -0.249, 0, 'C'], [2.503, -1.2, 0, 'C'], [-1.428, -2.696, 0.001, 'C'], [3.193, 1.206, 0, 'C'], [-2.297, 2.188, 0.001, 'C']],
  bonds: [[0, 8, 2], [1, 9, 2], [2, 7, 1], [2, 9, 1], [2, 11, 1], [3, 6, 1], [3, 10, 1], [3, 12, 1], [4, 8, 1], [4, 9, 1], [4, 13, 1], [5, 7, 1], [5, 10, 2], [6, 7, 2], [6, 8, 1]],
  types: ['C', 'N', 'O'],
};
// Vector of a molecule: per atom its three coordinates and a one-hot atom type over C, N, O.
export const molVector = () => MOL.atoms.flatMap(([x, y, z, el]) => [x, y, z, ...MOL.types.map((t) => (t === el ? 1 : 0))]);

// Ubiquitin, residues 1 to 16 (the first β-hairpin): backbone atoms N, Cα and C of PDB entry 1UBQ, in Å.
export const PROT = {
  name: 'Ubiquitin, residues 1 to 16 (PDB 1UBQ)',
  // β-strands from the SHEET records of 1UBQ (residues 1 to 7 and 10 to 17), as 0-based ranges cut at residue 16.
  strands: [[0, 6], [9, 15]],
  res: ['MET', 'GLN', 'ILE', 'PHE', 'VAL', 'LYS', 'THR', 'LEU', 'THR', 'GLY', 'LYS', 'THR', 'ILE', 'THR', 'LEU', 'GLU'],
  bb: [[[27.34, 24.43, 2.614], [26.266, 25.413, 2.842], [26.913, 26.639, 3.531]], [[26.335, 27.77, 3.258], [26.85, 29.021, 3.898], [26.1, 29.253, 5.202]], [[26.849, 29.656, 6.217], [26.235, 30.058, 7.497], [26.882, 31.428, 7.862]], [[26.214, 32.097, 8.771], [26.772, 33.436, 9.197], [27.151, 33.362, 10.65]], [[28.26, 33.943, 11.096], [28.605, 33.965, 12.503], [28.638, 35.461, 12.9]], [[27.751, 35.867, 13.74], [27.691, 37.315, 14.143], [28.469, 37.475, 15.42]], [[29.426, 38.43, 15.446], [30.225, 38.643, 16.662], [29.664, 39.839, 17.434]], [[30.132, 40.069, 18.642], [29.607, 41.18, 19.467], [30.075, 42.538, 18.984]], [[30.991, 42.571, 17.998], [31.422, 43.94, 17.553], [30.755, 44.351, 16.277]], [[29.721, 43.673, 15.885], [28.978, 43.96, 14.678], [29.604, 43.507, 13.393]], [[30.563, 42.623, 13.495], [31.191, 42.012, 12.331], [30.459, 40.666, 12.13]], [[30.163, 40.338, 10.886], [29.542, 39.02, 10.653], [30.494, 38.261, 9.729]], [[30.795, 37.015, 10.095], [31.72, 36.289, 9.176], [30.955, 35.211, 8.459]], [[31.244, 34.986, 7.197], [30.505, 33.884, 6.512], [31.409, 32.68, 6.446]], [[30.884, 31.485, 6.666], [31.677, 30.275, 6.639], [31.022, 29.288, 5.665]], [[31.834, 28.412, 5.125], [31.22, 27.341, 4.275], [31.44, 26.079, 5.08]]],
};
const sub = (a, b) => a.map((v, i) => v - b[i]), dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
// Per residue: the Cα position relative to the centroid, and the rotation of its N, Cα, C frame as an
// axis-angle vector (axis times angle in radians). Six numbers per residue.
export const protVector = () => {
  const ca = PROT.bb.map((r) => r[1]), cen = [0, 1, 2].map((d) => ca.reduce((s, p) => s + p[d], 0) / ca.length);
  return PROT.bb.flatMap(([n, a, c]) => {
    const e1 = norm(sub(c, a)), u = sub(n, a), e2 = norm(sub(u, e1.map((v) => v * dot(u, e1)))), e3 = cross(e1, e2);
    const R = [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]];
    const th = Math.acos(Math.max(-1, Math.min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)));
    const ax = norm([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]).map((v) => v * th);
    return [...sub(a, cen), ...ax];
  });
};
// A 2D view of the backbone: the plane of largest spread of the Cα trace, by power iteration.
export const PROT2D = (() => {
  const pts = PROT.bb.flat(), cen = [0, 1, 2].map((d) => pts.reduce((s, p) => s + p[d], 0) / pts.length);
  const C = [0, 1, 2].map((i) => [0, 1, 2].map((j) => pts.reduce((s, p) => s + (p[i] - cen[i]) * (p[j] - cen[j]), 0)));
  const top = (M) => { let v = norm([1, 0.3, 0.2]); for (let k = 0; k < 200; k++) v = norm(M.map((r) => dot(r, v))); return v; };
  const v1 = top(C), l1 = dot(v1, C.map((r) => dot(r, v1)));
  const v2 = top(C.map((r, i) => r.map((c, j) => c - l1 * v1[i] * v1[j])));
  return PROT.bb.map((r) => r.map((p) => [dot(sub(p, cen), v1), dot(sub(p, cen), v2)]));
})();

// A short sentence as token ids. The ids are illustrative: real tokenizers split words into pieces.
// V is the vocabulary size of GPT-2 and E the embedding width of GPT-2 small; the embedding values
// are seeded Gaussian draws with the spread of a trained table, not learned ones.
export const TEXT = {
  sentence: 'noise slowly turns into data.',
  tokens: ['noise', 'slowly', 'turns', 'into', 'data', '.'],
  ids: [9214, 15872, 7731, 1061, 2305, 13],
  V: 50257, E: 768,
};
export const EMB = (() => {
  const rng = makeRng(23);
  return TEXT.ids.map(() => Float64Array.from({ length: TEXT.E }, () => Math.round(1000 * 0.12 * rng.gauss()) / 1000));
})();

// A 12 x 12 colour picture (sky, sun and two hills), stored as three channels of 8-bit values in
// the order R, G, B, each row after row. Each pixel averages a 4 x 4 grid of sample points.
export const CIMG = (() => {
  const n = 12, R = [], G = [], B = [];
  const at = (x, y) => {
    if (Math.hypot(x - 8.6, y - 3.1) < 1.9) return [255, 200, 60];
    if (y > 9.6 + 0.9 * Math.cos((x - 9) / 2.1)) return [46, 122, 58];
    if (y > 7.4 + 1.4 * Math.cos((x - 3.2) / 2.6)) return [104, 176, 84];
    const u = Math.min(1, y / 8);
    return [72 + 100 * u, 140 + 70 * u, 226 + 22 * u];
  };
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const c = [0, 0, 0];
    for (let b = 0; b < 4; b++) for (let a = 0; a < 4; a++) at(i + (a + 0.5) / 4, j + (b + 0.5) / 4).forEach((v, k) => (c[k] += v / 16));
    R.push(Math.round(c[0])); G.push(Math.round(c[1])); B.push(Math.round(c[2]));
  }
  return { n, ch: [R, G, B] };
})();
