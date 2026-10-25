// Numerics of the §7 tutorial. Every toy is exact (closed form or ODE on an exact field) or Monte Carlo
// over seeded draws, and no function here touches the DOM. Convention: t = 0 noise, t = 1 data,
// x_t = t x_1 + (1 - t) x_0.
import { makeRng } from '../../animations/lib/rng.js';
import { flowMap, Phi } from '../lib/mix1d.js';

const sum = (a) => a.reduce((s, v) => s + v, 0);
const mean = (a) => sum(a) / a.length;
// E|Z| for Z ~ N(mu, v).
export const absMoment = (mu, v) => (v <= 0 ? Math.abs(mu) : Math.sqrt(v) * Math.sqrt(2 / Math.PI) * Math.exp(-mu * mu / (2 * v)) + mu * (1 - 2 * Phi(-mu / Math.sqrt(v))));
// Fixed-step RK4 from t = 0 to 1 that keeps every node, for a field f(x, t) on arrays.
function rk4Path(f, x, n) {
  const out = [x.slice()], h = 1 / n, ax = (a, b, s) => a.map((v, i) => v + s * b[i]);
  for (let k = 0; k < n; k++) {
    const t = k * h, k1 = f(x, t), k2 = f(ax(x, k1, h / 2), t + h / 2), k3 = f(ax(x, k2, h / 2), t + h / 2), k4 = f(ax(x, k3, h), t + h);
    x = x.map((v, i) => v + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    out.push(x.slice());
  }
  return out;
}
// The state of a stored path at time t, linear between the nodes.
export function pathAt(path, t) {
  const n = path.length - 1, u = Math.min(Math.max(t, 0), 1) * n, k = Math.min(Math.floor(u), n - 1), a = u - k;
  return path[k].map((v, i) => (1 - a) * v + a * path[k + 1][i]);
}

// ---------- Step 1: video ----------
// A clip is F frames of a dot on a line: x^f = c + s A tau_f + eps_f with c ~ N(0, C²), s = ±1 and
// eps ~ N(0, E²). It is a two-component Gaussian in F dimensions with covariance C² 11ᵀ + E² I, whose
// eigenvalues are F C² + E² along 1 and E² on the rest, so the joint field is exact in closed form.
export const VF = 12, VA = 1.5, VC = 0.6, VE = 0.05, NV = 400, NVS = 8, VSTEPS = 50;
export const TAU = Array.from({ length: VF }, (_, f) => f / (VF - 1) - 0.5);
const LAM1 = VF * VC * VC + VE * VE, LAM0 = VE * VE, STT = sum(TAU.map((q) => q * q));
const MV = [1, -1].map((s) => TAU.map((q) => s * VA * q));
function vidField(x, t) {
  const v1 = t * t * LAM1 + (1 - t) ** 2, v0 = t * t * LAM0 + (1 - t) ** 2;
  const c1 = (t * LAM1 - (1 - t)) / v1, c0 = (t * LAM0 - (1 - t)) / v0;
  const parts = MV.map((m) => {
    const d = x.map((v, i) => v - t * m[i]), d1 = mean(d), dp = d.map((v) => v - d1);
    return { m, d1, dp, q: VF * d1 * d1 / v1 + sum(dp.map((v) => v * v)) / v0 };
  });
  const qm = Math.min(parts[0].q, parts[1].q), w = parts.map((p) => Math.exp(-(p.q - qm) / 2)), Z = w[0] + w[1];
  return x.map((_, i) => parts.reduce((a, p, k) => a + w[k] / Z * (p.m[i] + c1 * p.d1 + c0 * p.dp[i]), 0));
}
// The per-frame marginal: a 1D mixture with modes at ±A tau_f and standard deviation sqrt(C² + E²).
const frameMix = (f) => ({ mu: [-VA * TAU[f], VA * TAU[f]], sd: [Math.hypot(VC, VE), Math.hypot(VC, VE)], w: [0.5, 0.5] });
const FMIX = TAU.map((_, f) => frameMix(f));
// Paths of one clip under each model: a list of VSTEPS + 1 states of F frames.
function vidPath(mode, z) {
  if (mode === 'joint') return rk4Path(vidField, z, VSTEPS);
  const zz = mode === 'shared' ? TAU.map(() => z[0]) : z;
  return Array.from({ length: VSTEPS + 1 }, (_, k) => zz.map((v, f) => flowMap(FMIX[f], v, 0, k / VSTEPS)));
}
const rngV = makeRng(71);
const VZ = Array.from({ length: NV }, () => TAU.map(() => rngV.gauss()));
const VDATA = Array.from({ length: NV }, () => { const c = VC * rngV.gauss(), s = rngV.uniform() < 0.5 ? 1 : -1; return TAU.map((q) => c + s * VA * q + VE * rngV.gauss()); });
export const VMODES = ['data', 'joint', 'independent', 'shared'];
const vidCache = new Map();
export function video(mode) {
  if (vidCache.has(mode)) return vidCache.get(mode);
  const show = mode === 'data' ? VDATA.slice(0, NVS).map((x) => [x, x]) : VZ.slice(0, NVS).map((z) => vidPath(mode, z));
  const ends = mode === 'data' ? VDATA : VZ.map((z, i) => (i < NVS ? show[i][show[i].length - 1] : endOf(mode, z)));
  const r = { show, ends, ...vidStats(ends) };
  vidCache.set(mode, r);
  return r;
}
const endOf = (mode, z) => (mode === 'joint' ? rk4Path(vidField, z, VSTEPS)[VSTEPS] : (mode === 'shared' ? TAU.map(() => z[0]) : z).map((v, f) => flowMap(FMIX[f], v, 0, 1)));
// Fitted speed (least squares on tau, the shared offset drops out because tau sums to zero), jitter
// (RMS second difference) and the share of clips whose speed is within 0.5 of ±A.
export const slope = (x) => sum(x.map((v, i) => v * TAU[i])) / STT;
function vidStats(ends) {
  const sl = ends.map(slope);
  const jit = mean(ends.map((x) => Math.sqrt(mean(x.slice(1, -1).map((v, i) => (x[i + 2] - 2 * v + x[i]) ** 2)))));
  return { slopes: sl, jitter: jit, coherent: mean(sl.map((s) => (Math.abs(Math.abs(s) - VA) < 0.5 ? 1 : 0))) };
}
// The data's fitted speed is ±A plus N(0, E²/sum tau²) noise.
export const SLOPE_SD = VE / Math.sqrt(STT);

// ---------- Step 2: world model ----------
// A ring of P cells, one-hot frames, and a seeded sequence of key presses in {-1, 0, +1}. The
// next-frame model moves 1 - eps of the intensity correctly, 0.6 eps one cell too far and 0.4 eps one
// cell too short (for a no-op, one cell right and one cell left). Variant B first sharpens its
// context to the cell of largest intensity, which leaves real frames unchanged.
export const WP = 32, WK = 64;
export const WACT = (() => {
  const rng = makeRng(15), out = []; let a = 1;
  while (out.length < WK) { const n = 2 + Math.floor(rng.uniform() * 5); for (let i = 0; i < n && out.length < WK; i++) out.push(a); a = [-1, 0, 1].filter((v) => v !== a)[rng.uniform() < 0.5 ? 0 : 1]; }
  return out;
})();
export const WPOS = WACT.reduce((p, a) => [...p, (p[p.length - 1] + a + WP) % WP], [WP / 2]);
const roll = (x, d) => x.map((_, i) => x[(i - d + 2 * WP) % WP]);
const onehot = (p) => Array.from({ length: WP }, (_, i) => (i === p ? 1 : 0));
function wStep(x, a, eps) {
  const s = roll(x, a), d = a || 1, far = roll(s, d), short = roll(s, -d);
  return s.map((v, i) => (1 - eps) * v + 0.6 * eps * far[i] + 0.4 * eps * short[i]);
}
const argmax = (x) => x.reduce((b, v, i) => (v > x[b] ? i : b), 0);
// Frames 0..K of the rollout and the error TV(frame, true frame) = 1 - mass on the true cell.
export function worldRollout(eps, sharpen) {
  const frames = [onehot(WPOS[0])];
  for (let k = 0; k < WK; k++) { const prev = frames[k]; frames.push(wStep(sharpen ? onehot(argmax(prev)) : prev, WACT[k], eps)); }
  return { frames, err: frames.map((x, k) => 1 - x[WPOS[k]]) };
}
// The bound of eq. (rollout-tv) with eps_k = eps, and the drift bound of eq. (lipschitz-drift).
export const tvBound = (eps, k) => k * eps;
export const lipBound = (eps, L, k) => (Math.abs(L - 1) < 1e-9 ? k * eps : eps * (L ** k - 1) / (L - 1));

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

// ---------- Step 4: molecules and symmetry ----------
// Three atoms in the plane (O, H, H), centered. Coordinates live in the 4D subspace of zero center:
// c[a][d] = sum_i e_a[i] X[i][d] with e_1 = (1, -1, 0)/√2 and e_2 = (1, 1, -2)/√6. A rotation acts
// on d, so it commutes with this basis. The data are the reference geometry at all orientations: a
// uniform mixture over NROT rotations, each with isotropic noise MS in the subspace. With NROT = 360
// the mixture is invariant under rotations by 1° and, at this noise level, under every rotation to
// within the printed discrepancy.
export const NROT = 360, MS = 0.08, NM = 300, MSTEPS = 40;
const E1 = [1 / Math.SQRT2, -1 / Math.SQRT2, 0], E2 = [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)];
export const toSub = (X) => [E1, E2].flatMap((e) => [0, 1].map((d) => e[0] * X[0][d] + e[1] * X[1][d] + e[2] * X[2][d]));
export const fromSub = (c) => [0, 1, 2].map((i) => [0, 1].map((d) => E1[i] * c[d] + E2[i] * c[2 + d]));
export const rot2 = (c, th) => { const C = Math.cos(th), S = Math.sin(th); return [C * c[0] - S * c[1], S * c[0] + C * c[1], C * c[2] - S * c[3], S * c[2] + C * c[3]]; };
// Water: O-H 0.96 Å, angle 104.5°, oxygen up; then centered.
const WATER = (() => { const a = 104.5 / 2 * Math.PI / 180, X = [[0, 0], [-0.96 * Math.sin(a), -0.96 * Math.cos(a)], [0.96 * Math.sin(a), -0.96 * Math.cos(a)]]; const cy = (X[1][1] + X[2][1]) / 3; return X.map(([x, y]) => [x, y - cy]); })();
const CREF = toSub(WATER);
const MOLMEANS = { all: Array.from({ length: NROT }, (_, j) => rot2(CREF, 2 * Math.PI * j / NROT)), one: [CREF] };
function molField(means) {
  const s2 = MS * MS;
  return (x, t) => {
    const v = t * t * s2 + (1 - t) ** 2, c = (t * s2 - (1 - t)) / v;
    let qm = Infinity; const q = means.map((m) => { let d = 0; for (let i = 0; i < 4; i++) d += (x[i] - t * m[i]) ** 2; qm = Math.min(qm, d); return d; });
    const out = [0, 0, 0, 0]; let Z = 0;
    means.forEach((m, k) => { const w = Math.exp(-(q[k] - qm) / (2 * v)); Z += w; for (let i = 0; i < 4; i++) out[i] += w * (m[i] + c * (x[i] - t * m[i])); });
    return out.map((u) => u / Z);
  };
}
const MF = { all: molField(MOLMEANS.all), one: molField(MOLMEANS.one) };
export const molPath = (which, z) => rk4Path(MF[which], z, MSTEPS);
// Orientation of a molecule: the angle of the vector from the midpoint of the hydrogens to the oxygen,
// measured from straight up.
export const orient = (c) => { const X = fromSub(c), vx = X[0][0] - (X[1][0] + X[2][0]) / 2, vy = X[0][1] - (X[1][1] + X[2][1]) / 2; return Math.atan2(-vx, vy); };
const rngM = makeRng(23);
export const MZ = Array.from({ length: NM }, () => [0, 0, 0, 0].map(() => rngM.gauss()));
export const MZ0 = MZ[0];
const molCache = new Map();
export function molEnds(which) {
  if (!molCache.has(which)) molCache.set(which, MZ.map((z) => molPath(which, z)[MSTEPS]).map(orient));
  return molCache.get(which);
}
// Largest atom distance between Φ(g x0) and g Φ(x0), in Å.
export function equivGap(pa, pb2, th) {
  const A = fromSub(pa), B = fromSub(rot2(pb2, th));
  return Math.max(...A.map((p, i) => Math.hypot(p[0] - B[i][0], p[1] - B[i][1])));
}

// ---------- Step 5: protein frames on SE(3) ----------
export const N_LOC = [-0.525, 1.363, 0], C_LOC = [1.526, 0, 0];
export const NR = 12, SIG0 = 4;
const v3 = { sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], scl: (a, s) => [a[0] * s, a[1] * s, a[2] * s], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] };
export const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const nrm = (a) => v3.scl(a, 1 / len3(a));
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
export const qrot = (q, v) => { const p = qmul(qmul(q, [0, ...v]), [q[0], -q[1], -q[2], -q[3]]); return [p[1], p[2], p[3]]; };
// Geodesic on SO(3) through unit quaternions: the same path as R0 exp(t log(R0ᵀ R1)).
export function slerp(a, b, u) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) { b = b.map((v) => -v); d = -d; }
  if (d > 0.9995) { const r = a.map((v, i) => v + u * (b[i] - v)); const n = Math.hypot(...r); return r.map((v) => v / n); }
  const th = Math.acos(d), s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - u) * th) * v + Math.sin(u * th) * b[i]) / s);
}
export const qangle = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));
export const qmat = (q) => [qrot(q, [1, 0, 0]), qrot(q, [0, 1, 0]), qrot(q, [0, 0, 1])];   // columns
function matToQuat(e1, e2, e3) {
  const m = [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]], tr = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]; }
  const n = Math.hypot(...q); return q.map((v) => v / n);
}
// Target: an ideal α-helix (radius 2.3 Å, rise 1.5 Å, 100° per residue) along z, centered, with
// frames from the Cα trace (e1 along the chain, e2 toward the local curvature).
export const CA1 = (() => { const p = Array.from({ length: NR }, (_, i) => { const a = i * 100 * Math.PI / 180; return [2.3 * Math.cos(a), 2.3 * Math.sin(a), (i - (NR - 1) / 2) * 1.5]; }); return p; })();
export const Q1 = CA1.map((p, i) => {
  const a = CA1[Math.max(i - 1, 0)], b = CA1[Math.min(i + 1, NR - 1)], e1 = nrm(v3.sub(b, a));
  let c = v3.sub(v3.add(a, b), v3.scl(p, 2));
  if (i === 0 || i === NR - 1) c = v3.scl([p[0], p[1], 0], -1);
  const e2 = nrm(v3.sub(c, v3.scl(e1, v3.dot(c, e1))));
  return matToQuat(e1, e2, v3.cross(e1, e2));
});
const rngF = makeRng(16);
export const CA0 = (() => { const x = CA1.map(() => [0, 0, 0].map(() => SIG0 * rngF.gauss())); const c = [0, 1, 2].map((d) => mean(x.map((p) => p[d]))); return x.map((p) => v3.sub(p, c)); })();
// Uniform rotations: a normalized 4D Gaussian is uniform on SO(3).
export const Q0 = CA1.map(() => { const q = [0, 0, 0, 0].map(() => rngF.gauss()), n = Math.hypot(...q); return q.map((v) => v / n); });
const atoms = (x, q) => [v3.add(x, qrot(q, N_LOC)), x, v3.add(x, qrot(q, C_LOC))];
export const ATOMS0 = CA0.map((x, i) => atoms(x, Q0[i])), ATOMS1 = CA1.map((x, i) => atoms(x, Q1[i]));
// Backbone atoms (N, Cα, C) of every residue at time t, on the frame path or by straight lines between atoms.
export function backbone(t, linear) {
  if (linear) return ATOMS0.map((r, i) => r.map((p, j) => v3.add(v3.scl(p, 1 - t), v3.scl(ATOMS1[i][j], t))));
  return CA0.map((x, i) => atoms(v3.add(v3.scl(x, 1 - t), v3.scl(CA1[i], t)), slerp(Q0[i], Q1[i], t)));
}
export const BOND_N = len3(N_LOC), BOND_C = len3(C_LOC);
// Largest relative error of the two bonds inside a residue.
export const bondErr = (bb) => Math.max(...bb.map(([n, ca, c]) => Math.max(Math.abs(len3(v3.sub(n, ca)) / BOND_N - 1), Math.abs(len3(v3.sub(c, ca)) / BOND_C - 1))));
export const ANG = Q0.map((q, i) => qangle(q, Q1[i]));
// The residue shown on its own: the one whose rotation angle is closest to 120°.
export const RSHOW = ANG.reduce((b, a, i) => (Math.abs(a - 2.1) < Math.abs(ANG[b] - 2.1) ? i : b), 0);
// For one residue: the naive matrix average (1 - t) R0 + t R1 and its defects.
export function naiveRot(i, t) {
  const A = qmat(Q0[i]), B = qmat(Q1[i]), M = A.map((col, j) => col.map((v, k) => (1 - t) * v + t * B[j][k]));
  const det = v3.dot(M[0], v3.cross(M[1], M[2]));
  let orth = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) orth += (v3.dot(M[a], M[b]) - (a === b ? 1 : 0)) ** 2;
  return { M, det, orth: Math.sqrt(orth) };
}
export const rmsd = (bb) => Math.sqrt(mean(bb.map((r, i) => len3(v3.sub(r[1], CA1[i])) ** 2)));

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
