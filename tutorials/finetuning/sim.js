// Numerics of the §3 tutorial. The model is linear in its weights: one 2D velocity per Euler step,
// v_k(x) = W_k b(x), where b are 256 Gaussian bumps on a 16 × 16 grid plus a constant. Every fit is
// least squares against the exact velocity of a Gaussian mixture on one fixed design set, so the
// fine-tuning methods of the page reduce to linear algebra with closed-form answers.
import { makeRng } from '../../animations/lib/rng.js';

export const K = 16, GRID = 16, BOX = 3.2, WID = 0.45, D = GRID * GRID + 1, M = 2 * K;
export const NZ = 2000;
const TS = Float64Array.from({ length: K }, (_, k) => k / K);

// ---------- Data: the six-mode ring and its edits ----------
export const RAD = 1.6, SD = 0.13, SSD = 0.3;
export const RING = Array.from({ length: 6 }, (_, k) => [RAD * Math.cos(2 * Math.PI * k / 6 + Math.PI / 6), RAD * Math.sin(2 * Math.PI * k / 6 + Math.PI / 6)]);
const mix = (mu, sd = SD) => ({ mu, sd: mu.map(() => sd) });
const rot = (deg) => { const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a); return ([x, y]) => [c * x - s * y, s * x + c * y]; };
const scale = (f) => ([x, y]) => [f * x, f * y];
// A subject at the origin replaces the first mode of a ring.
const withSubject = (ring) => [[0, 0], ...ring.slice(1)];
export const MIX = {
  base: mix(RING),
  new: mix(withSubject(RING)),
  subj: mix([[0, 0]], SSD),
  rot: mix(RING.map(rot(20))),
  rotJ: mix(withSubject(RING.map(rot(20)))),
  shr: mix(RING.map(scale(0.6))),
  shrJ: mix(withSubject(RING.map(scale(0.6)))),
};

// Exact velocity E[x1 - x0 | x_t = x] of an equal-weight mixture on the linear path, written to out.
function vel(m, x, y, t, out, o = 0) {
  const C = m.mu.length, lw = new Float64Array(C), vt = new Float64Array(C);
  let mx = -Infinity;
  for (let c = 0; c < C; c++) {
    const s2 = m.sd[c] * m.sd[c]; vt[c] = t * t * s2 + (1 - t) * (1 - t);
    const dx = x - t * m.mu[c][0], dy = y - t * m.mu[c][1];
    lw[c] = -Math.log(vt[c]) - (dx * dx + dy * dy) / (2 * vt[c]); if (lw[c] > mx) mx = lw[c];
  }
  let Z = 0, ux = 0, uy = 0;
  for (let c = 0; c < C; c++) {
    const r = Math.exp(lw[c] - mx), s2 = m.sd[c] * m.sd[c], a = (t * s2 - (1 - t)) / vt[c];
    const dx = x - t * m.mu[c][0], dy = y - t * m.mu[c][1];
    Z += r; ux += r * (m.mu[c][0] + a * dx); uy += r * (m.mu[c][1] + a * dy);
  }
  out[o] = ux / Z; out[o + 1] = uy / Z;
}
function sampleMix(m, n, rng) {
  const out = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { const c = Math.min(m.mu.length - 1, Math.floor(rng.uniform() * m.mu.length)); out[2 * i] = m.mu[c][0] + m.sd[c] * rng.gauss(); out[2 * i + 1] = m.mu[c][1] + m.sd[c] * rng.gauss(); }
  return out;
}

// ---------- Features ----------
const CEN = Float64Array.from({ length: 2 * GRID * GRID }, (_, i) => { const j = i >> 1, g = -BOX + 2 * BOX * ((i & 1 ? Math.floor(j / GRID) : j % GRID) / (GRID - 1)); return g; });
export function feats(x, y, out, o = 0) {
  const k = 1 / (2 * WID * WID);
  for (let j = 0; j < GRID * GRID; j++) { const dx = x - CEN[2 * j], dy = y - CEN[2 * j + 1]; out[o + j] = Math.exp(-k * (dx * dx + dy * dy)); }
  out[o + D - 1] = 1;
}

// ---------- Design set: draws of p_t for each mixture at each Euler time ----------
const NPER = 64, DESIGN = ['base', 'new', 'subj', 'rot', 'rotJ', 'shr', 'shrJ'];
export const N = DESIGN.length * K * NPER;
const X = new Float64Array(2 * N), F = new Float64Array(N * D);
{
  const rng = makeRng(1); let i = 0;
  for (const name of DESIGN) for (const t of TS) {
    const x1 = sampleMix(MIX[name], NPER, rng);
    for (let n = 0; n < NPER; n++, i++) { X[2 * i] = t * x1[2 * n] + (1 - t) * rng.gauss(); X[2 * i + 1] = t * x1[2 * n + 1] + (1 - t) * rng.gauss(); }
  }
  for (let n = 0; n < N; n++) feats(X[2 * n], X[2 * n + 1], F, n * D);
}
// Gram matrix of the features over the design set, Gm = F^T F / N.
export const GM = new Float64Array(D * D);
for (let n = 0; n < N; n++) {
  const o = n * D;
  for (let a = 0; a < D; a++) { const fa = F[o + a]; if (fa < 1e-12) continue; const ra = a * D; for (let b = a; b < D; b++) GM[ra + b] += fa * F[o + b]; }
}
for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { GM[a * D + b] /= N; GM[b * D + a] = GM[a * D + b]; }

// Cholesky factor of Gm + 1e-6 I for the least-squares fits.
const CH = new Float64Array(D * D);
for (let i = 0; i < D; i++) for (let j = 0; j <= i; j++) {
  let s = GM[i * D + j] + (i === j ? 1e-6 : 0);
  for (let k = 0; k < j; k++) s -= CH[i * D + k] * CH[j * D + k];
  CH[i * D + j] = i === j ? Math.sqrt(s) : s / CH[j * D + j];
}
function cholSolve(b) {   // b: D x M, solved in place
  for (let c = 0; c < M; c++) {
    for (let i = 0; i < D; i++) { let s = b[i * M + c]; for (let k = 0; k < i; k++) s -= CH[i * D + k] * b[k * M + c]; b[i * M + c] = s / CH[i * D + i]; }
    for (let i = D - 1; i >= 0; i--) { let s = b[i * M + c]; for (let k = i + 1; k < D; k++) s -= CH[k * D + i] * b[k * M + c]; b[i * M + c] = s / CH[i * D + i]; }
  }
  return b;
}

// Right-hand side F^T Y / N for the exact velocities of a mixture (D x M), and the fitted W (M x D).
const memo = (f) => { const m = new Map(); return (k) => { if (!m.has(k)) m.set(k, f(k)); return m.get(k); }; };
export const rhs = memo((name) => {
  const b = new Float64Array(D * M), u = new Float64Array(M);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k++) vel(MIX[name], X[2 * n], X[2 * n + 1], TS[k], u, 2 * k);
    const o = n * D;
    for (let a = 0; a < D; a++) { const fa = F[o + a]; if (fa < 1e-12) continue; for (let c = 0; c < M; c++) b[a * M + c] += fa * u[c]; }
  }
  for (let i = 0; i < D * M; i++) b[i] /= N;
  return b;
});
const transpose = (A, r, c) => { const T = new Float64Array(r * c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j * r + i] = A[i * c + j]; return T; };
export const fit = memo((name) => transpose(cholSolve(rhs(name).slice()), D, M));
export const W0 = () => fit('base');
export const sub = (A, B) => A.map((v, i) => v - B[i]);
export const comb = (W, ...terms) => { const out = W.slice(); for (const [a, U] of terms) for (let i = 0; i < out.length; i++) out[i] += a * U[i]; return out; };

// Inner product of two updates in function space, E_rho[(U b)·(V b)] = tr(U Gm V^T).
export function ip(U, V) {
  let s = 0; const tmp = new Float64Array(D);
  for (let c = 0; c < M; c++) {
    tmp.fill(0); const uo = c * D;
    for (let a = 0; a < D; a++) { const ua = U[uo + a]; if (!ua) continue; const ga = a * D; for (let b = 0; b < D; b++) tmp[b] += ua * GM[ga + b]; }
    for (let b = 0; b < D; b++) s += tmp[b] * V[uo + b];
  }
  return s;
}

// ---------- Sampling ----------
export const Z0 = (() => { const r = makeRng(5), z = new Float64Array(2 * NZ); for (let i = 0; i < 2 * NZ; i++) z[i] = r.gauss(); return z; })();
export function run(W, z = Z0) {
  const n = z.length / 2, x = z.slice(), b = new Float64Array(D);
  for (let i = 0; i < n; i++) {
    let px = x[2 * i], py = x[2 * i + 1];
    for (let k = 0; k < K; k++) {
      feats(px, py, b);
      const ox = 2 * k * D, oy = ox + D; let vx = 0, vy = 0;
      for (let a = 0; a < D; a++) { vx += W[ox + a] * b[a]; vy += W[oy + a] * b[a]; }
      px += vx / K; py += vy / K;
    }
    x[2 * i] = px; x[2 * i + 1] = py;
  }
  return x;
}
export const ref = memo((name) => sampleMix(MIX[name], NZ, makeRng(7)));
// Sliced Wasserstein-1 distance over 32 directions.
export function sw(a, b) {
  const n = Math.min(a.length, b.length) / 2, pa = new Float64Array(n), pb = new Float64Array(n); let s = 0;
  for (let q = 0; q < 32; q++) {
    const c = Math.cos(Math.PI * q / 32), d = Math.sin(Math.PI * q / 32);
    for (let i = 0; i < n; i++) { pa[i] = c * a[2 * i] + d * a[2 * i + 1]; pb[i] = c * b[2 * i] + d * b[2 * i + 1]; }
    pa.sort(); pb.sort(); let t = 0; for (let i = 0; i < n; i++) t += Math.abs(pa[i] - pb[i]); s += t / n;
  }
  return s / 32;
}
export const share = (s, cx = 0, cy = 0, r = 0.5) => { let k = 0; const n = s.length / 2; for (let i = 0; i < n; i++) if (Math.hypot(s[2 * i] - cx, s[2 * i + 1] - cy) < r) k++; return k / n; };

// ---------- Symmetric eigendecomposition (cyclic Jacobi), eigenvalues descending ----------
export function eigSym(A0, n) {
  const A = Float64Array.from(A0), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0, tot = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = A[i * n + j] * A[i * n + j]; tot += v; if (i !== j) off += v; }
    if (off <= 1e-26 * tot) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = A[k * n + p], akq = A[k * n + q]; A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = A[p * n + k], aqk = A[q * n + k]; A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k * n + p], vkq = V[k * n + q]; V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq; }
    }
  }
  const ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b * n + b] - A[a * n + a]);
  // vals[j] and the column vecs[:, j] (stored as rows: vecs[j * n + k]).
  return { vals: ord.map((i) => A[i * n + i]), vecs: Float64Array.from({ length: n * n }, (_, x) => V[(x % n) * n + ord[Math.floor(x / n)]]) };
}

// ---------- Step 1: LoRA as the best rank-r update ----------
// The update of full fine-tuning, its function-space spectrum (eigen of U Gm U^T) and its
// parameter-space spectrum (eigen of U U^T), both M x M.
const outerM = (U, G) => {
  const R = new Float64Array(M * M), tmp = new Float64Array(D);
  for (let i = 0; i < M; i++) {
    if (G) { tmp.fill(0); for (let a = 0; a < D; a++) { const ua = U[i * D + a]; if (!ua) continue; for (let b = 0; b < D; b++) tmp[b] += ua * G[a * D + b]; } }
    else tmp.set(U.subarray(i * D, i * D + D));
    for (let j = 0; j < M; j++) { let s = 0; for (let b = 0; b < D; b++) s += tmp[b] * U[j * D + b]; R[i * M + j] = s; }
  }
  return R;
};
export const lora = memo((name) => {
  const U = sub(fit(name), W0()), fn = eigSym(outerM(U, GM), M), par = eigSym(outerM(U), M);
  const cum = (v) => { const p = v.map((q) => Math.max(q, 0)), tot = p.reduce((a, b) => a + b, 0); let s = 0; return p.map((q) => (s += q) / tot); };
  // The rank-r update V_r V_r^T U.
  const rank = memo((r) => {
    if (r >= M) return U;
    const P = new Float64Array(M * M);
    for (let j = 0; j < r; j++) for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) P[a * M + b] += fn.vecs[j * M + a] * fn.vecs[j * M + b];
    const out = new Float64Array(M * D);
    for (let a = 0; a < M; a++) for (let b = 0; b < M; b++) { const p = P[a * M + b]; if (!p) continue; for (let c = 0; c < D; c++) out[a * D + c] += p * U[b * D + c]; }
    return out;
  });
  return { U, fnCum: cum(fn.vals), parCum: cum(par.vals), rank };
});

// ---------- Step 2: DreamBooth with and without prior preservation ----------
// Features [b, S b] for the subject prompt and [b, 0] for the class prompt, weights [Ws, Wid]. The loss
//   E|(Ws + S Wid) b - u_subj|^2 + lam E|(Ws - W0) b|^2
// is minimized by k steps of gradient descent from (W0, 0). Per eigenvector of Gm the two blocks are
// coupled by the 2 x 2 matrix [[1 + lam, S], [S, S^2]], so k steps have a closed form.
let EIG = null;
export const gramEig = () => (EIG ??= eigSym(GM, D));
export function dreambooth(S, lam, k) {
  const { vals, vecs } = gramEig(), gmax = vals[0], eta = 1 / (gmax * (12 + S * S));
  // Project the subject right-hand side and the base weights onto the eigenvectors: (D x M).
  const B = rhs('subj'), W = W0(), Bs = new Float64Array(D * M), C0 = new Float64Array(D * M);
  for (let i = 0; i < D; i++) for (let a = 0; a < D; a++) { const q = vecs[i * D + a]; if (!q) continue; for (let c = 0; c < M; c++) { Bs[i * M + c] += q * B[a * M + c]; C0[i * M + c] += q * W[c * D + a]; } }
  // Eigen of the 2 x 2 coupling.
  const a = 1 + lam, b = S, d = S * S, tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const mu = [tr / 2 + disc, tr / 2 - disc], V2 = mu.map((m) => { const vx = b, vy = m - a, nn = Math.hypot(vx, vy); return nn > 1e-12 ? [vx / nn, vy / nn] : [1, 0]; });
  if (Math.abs(b) < 1e-12) { V2[0] = a >= d ? [1, 0] : [0, 1]; V2[1] = a >= d ? [0, 1] : [1, 0]; mu[0] = Math.max(a, d); mu[1] = Math.min(a, d); }
  const Ws = new Float64Array(D * M), Wi = new Float64Array(D * M);
  for (let i = 0; i < D; i++) {
    const g = Math.max(vals[i], 0);
    for (let c = 0; c < M; c++) {
      const r0 = Bs[i * M + c] + lam * g * C0[i * M + c], r1 = S * Bs[i * M + c], w0 = C0[i * M + c];
      let ws = 0, wi = 0;
      for (let j = 0; j < 2; j++) {
        const [px, py] = V2[j], a0 = px * w0, rr = px * r0 + py * r1, lj = g * mu[j];
        const aj = lj > 1e-12 ? rr / lj + (1 - eta * lj) ** k * (a0 - rr / lj) : a0 + eta * k * rr;
        ws += px * aj; wi += py * aj;
      }
      Ws[i * M + c] = ws; Wi[i * M + c] = wi;
    }
  }
  // Back from the eigenbasis: W = (Q w)^T, M x D.
  const back = (A) => { const out = new Float64Array(M * D); for (let i = 0; i < D; i++) for (let a = 0; a < D; a++) { const q = vecs[i * D + a]; if (!q) continue; for (let c = 0; c < M; c++) out[c * D + a] += q * A[i * M + c]; } return out; };
  const ws = back(Ws), wi = back(Wi);
  return { cls: ws, subj: comb(ws, [S, wi]) };
}

// ---------- Step 3: merging two adapters ----------
export const STYLES = {
  rot: { name: 'rotate by 20°', style: 'rot', joint: 'rotJ' },
  shr: { name: 'shrink to 0.6', style: 'shr', joint: 'shrJ' },
};
export const merge = memo((key) => {
  const st = STYLES[key], W = W0(), A = sub(fit('new'), W), B = sub(fit(st.style), W), J = fit(st.joint);
  // The part of the joint change that is not the sum of the two changes, in function space.
  const R = sub(sub(J, W), comb(A, [1, B])), DJ = sub(J, W);
  return { A, B, J, cos: ip(A, B) / Math.sqrt(ip(A, A) * ip(B, B)), nonadd: Math.sqrt(ip(R, R) / ip(DJ, DJ)) };
});
