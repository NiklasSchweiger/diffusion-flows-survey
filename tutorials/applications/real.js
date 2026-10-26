// Numerics of the steps of the §7 tutorial that run on real data: Muybridge's photographs (video),
// a small maze game (world model), PubChem conformers (molecules) and PDB backbones (proteins). The
// files are in data/, with their sources in data/SOURCES.md. Every model is exact for its data set:
// the empirical denoiser of the frames, a hand-specified next-frame distribution, the posterior over
// rotations of one conformer, and the conditional paths on SE(3). Convention: t = 0 noise, t = 1 data,
// x_t = t x_1 + (1 - t) x_0. Only the two loaders touch the DOM.
import { makeRng } from '../../animations/lib/rng.js';

const DATA = new URL('./data/', import.meta.url);
let framesP = null, structP = null;
// Frames as Float32Arrays in [-1, 1], one row of the sprite sheet per clip.
export function loadFrames() {
  return (framesP ??= (async () => {
    const meta = await (await fetch(new URL('frames.json', DATA))).json();
    const img = new Image(); img.src = new URL('frames.png', DATA).href; await img.decode();
    const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
    const cx = cv.getContext('2d', { willReadFrequently: true }); cx.drawImage(img, 0, 0);
    const px = cx.getImageData(0, 0, cv.width, cv.height).data, { W, H } = meta, frames = [];
    meta.clips.forEach((c, j) => {
      c.start = frames.length;
      for (let i = 0; i < c.frames; i++) {
        const v = new Float32Array(W * H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) v[y * W + x] = px[4 * ((j * H + y) * cv.width + i * W + x)] / 127.5 - 1;
        frames.push({ clip: j, idx: i, v });
      }
    });
    return { ...meta, D: W * H, frames };
  })());
}
export const loadStructures = () => (structP ??= fetch(new URL('structures.json', DATA)).then((r) => r.json()));

const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function softmax(l) { const m = Math.max(...l); let z = 0; const w = l.map((v) => { const e = Math.exp(v - m); z += e; return e; }); return w.map((v) => v / z); }

// ---------- Step 1: video from Muybridge's photographs ----------
// The data are all windows of VF consecutive frames of the six clips. Every model is the exact
// empirical denoiser of its data, E[x_1 | x_t] = sum_k w_k m_k with softmax weights, so it memorizes.
// A state x_t = a z + sum_k c_k (frame k) stays in the span of the noise z and the frames under the
// update x <- x + (t' - t)/(1 - t) (E - x), so the sampler only tracks a and the coefficients c, with
// the Gram matrix of the frames; the pixels are rebuilt for drawing.
export const VF = 8, VSTEPS = 25, VN = 12, VSHOW = 6;
export const VMODES = ['data', 'joint', 'independent', 'shared'];
export function videoModel(fr) {
  const N = fr.frames.length, D = fr.D;
  const G = Array.from({ length: N }, (_, i) => Float64Array.from({ length: N }, (_, j) => dotv(fr.frames[i].v, fr.frames[j].v)));
  const wins = [];
  fr.clips.forEach((c) => { for (let i = 0; i + VF <= c.frames; i++) wins.push(c.start + i); });
  const cache = {};
  function sample(mode, s) {
    const rng = makeRng(9100 + 37 * s), z = [];
    for (let f = 0; f < VF; f++) {
      if (mode === 'shared' && f > 0) { z.push(z[0]); continue; }
      const v = new Float32Array(D); for (let i = 0; i < D; i++) v[i] = rng.gauss(); z.push(v);
    }
    if (mode === 'data') { const w = wins[Math.floor(rng.uniform() * wins.length)]; return { z, path: [{ a: 0, c: Array.from({ length: VF }, (_, f) => { const c = new Float64Array(N); c[w + f] = 1; return c; }) }] }; }
    const zf = z.map((v) => fr.frames.map((m) => dotv(v, m.v)));
    let a = 1, c = Array.from({ length: VF }, () => new Float64Array(N));
    const path = [{ a, c: c.map((q) => q.slice()) }];
    for (let k = 0; k < VSTEPS; k++) {
      const t = k / VSTEPS, t2 = (k + 1) / VSTEPS, s2 = (1 - t) ** 2;
      // x_f . m_j for every frame f of the clip and every training frame j.
      const dots = c.map((cf, f) => Float64Array.from({ length: N }, (_, j) => { let d = a * zf[f][j]; for (let i = 0; i < N; i++) if (cf[i]) d += cf[i] * G[i][j]; return d; }));
      const lg = (f, j) => (t * dots[f][j] - 0.5 * t * t * G[j][j]) / s2;
      const E = Array.from({ length: VF }, () => new Float64Array(N));
      if (mode === 'joint') {
        const w = softmax(wins.map((st) => { let l = 0; for (let f = 0; f < VF; f++) l += lg(f, st + f); return l; }));
        wins.forEach((st, q) => { if (w[q] > 1e-15) for (let f = 0; f < VF; f++) E[f][st + f] += w[q]; });
      } else {
        for (let f = 0; f < VF; f++) { const w = softmax(Array.from({ length: N }, (_, j) => lg(f, j))); w.forEach((v, j) => { if (v > 1e-15) E[f][j] = v; }); }
      }
      const r = (t2 - t) / (1 - t);
      a *= 1 - r;
      c = c.map((cf, f) => cf.map((v, j) => v + r * (E[f][j] - v)));
      path.push({ a, c: c.map((q) => q.slice()) });
    }
    return { z, path };
  }
  // Source of each generated frame: the training frame with the largest coefficient at t = 1.
  const source = (c) => { let b = 0; for (let j = 1; j < c.length; j++) if (c[j] > c[b]) b = j; return { k: b, w: c[b] }; };
  function run(mode) {
    if (cache[mode]) return cache[mode];
    const S = Array.from({ length: VN }, (_, s) => sample(mode, s));
    const src = S.map((q) => q.path[q.path.length - 1].c.map(source));
    const coherent = src.map((r) => r.every((q, f) => !f || (fr.frames[q.k].clip === fr.frames[r[f - 1].k].clip && q.k === r[f - 1].k + 1)));
    const clips = src.map((r) => new Set(r.map((q) => fr.frames[q.k].clip)).size);
    const frozen = src.map((r) => r.every((q) => q.k === r[0].k));
    const share = (b) => b.filter(Boolean).length / b.length;
    return (cache[mode] = { S, src, coherent: share(coherent), frozen: share(frozen), clips: clips.reduce((x, y) => x + y, 0) / VN, minW: Math.min(...src.flat().map((q) => q.w)) });
  }
  // Pixels of frame f of a sample at grid index k, into a Uint8ClampedArray of grey values.
  function pixels(smp, k, f, out) {
    const p = smp.path[Math.min(k, smp.path.length - 1)], cf = p.c[f], z = smp.z[f], acc = new Float32Array(D);
    if (p.a) for (let i = 0; i < D; i++) acc[i] = p.a * z[i];
    cf.forEach((v, j) => { if (Math.abs(v) > 1e-6) { const m = fr.frames[j].v; for (let i = 0; i < D; i++) acc[i] += v * m[i]; } });
    for (let i = 0; i < D; i++) out[i] = (acc[i] + 1) * 127.5;
    return out;
  }
  return { run, pixels, wins: wins.length, N };
}

// ---------- Step 2: a world model of a maze game ----------
// The frame is fully determined by the player's cell, so a model's next frame is a distribution q over
// cells and its picture is the average of the frames, the player drawn with opacity q. The true game
// moves the player one cell in the direction of the key unless a wall is in the way. Both models move
// a share 1 - eps of the mass correctly, 0.6 eps one cell too far and 0.4 eps not at all; without a
// key, eps spreads evenly to the open neighbours. Model B first replaces its context by the frame of
// its most likely cell, which changes nothing on a real frame.
export const MAZE = [
  '################',
  '#......#.......#',
  '#.####.#.#####.#',
  '#.#..........#.#',
  '#.#.##.###.#.#.#',
  '#...#....#.#...#',
  '###.#.##.#.###.#',
  '#.....#......#.#',
  '#.###...####...#',
  '################',
];
export const MW = MAZE[0].length, MH = MAZE.length, START = 1 * MW + 1;
export const open = (c) => MAZE[Math.floor(c / MW)][c % MW] !== '#';
const DIRS = { left: -1, right: 1, up: -MW, down: MW };
const move = (c, a) => (a && open(c + DIRS[a]) ? c + DIRS[a] : c);
function modelStep(q, a, eps) {
  const r = new Float64Array(q.length);
  q.forEach((m, c) => {
    if (m < 1e-12) return;
    if (a) { const c1 = move(c, a); r[c1] += (1 - eps) * m; r[move(c1, a)] += 0.6 * eps * m; r[c] += 0.4 * eps * m; return; }
    const nb = Object.keys(DIRS).map((d) => c + DIRS[d]).filter(open);
    r[c] += (1 - eps) * m + (nb.length ? 0 : eps * m);
    nb.forEach((n) => { r[n] += eps * m / nb.length; });
  });
  return r;
}
const delta = (c) => { const q = new Float64Array(MW * MH); q[c] = 1; return q; };
// Rollout of the game and both models for a list of keys ('left', 'right', 'up', 'down' or null).
export function worldPlay(keys, eps) {
  let pos = START, A = delta(START), B = delta(START);
  const out = [{ pos, A, B, errA: 0, errB: 0 }];
  for (const a of keys) {
    pos = move(pos, a);
    A = modelStep(A, a, eps);
    let mx = 0; B.forEach((v, c) => { if (v > B[mx]) mx = c; });
    B = modelStep(delta(mx), a, eps);
    out.push({ pos, A, B, errA: 1 - A[pos], errB: 1 - B[pos] });
  }
  return out;
}
// A fixed run through the maze, shown before the reader presses a key.
export const DEMO_KEYS = [...'rrrrrddrrrrrrdd...rrdd..uulluullllll..dd'].map((k) => ({ r: 'right', l: 'left', u: 'up', d: 'down', '.': null })[k]);

// ---------- Rotations ----------
const v3 = { sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], scl: (a, s) => [a[0] * s, a[1] * s, a[2] * s], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] };
export { v3 };
export const len3 = (a) => Math.hypot(a[0], a[1], a[2]);
const nrm = (a) => v3.scl(a, 1 / len3(a));
// 3 × 3 matrices as arrays of rows.
export const mv = (R, p) => [v3.dot(R[0], p), v3.dot(R[1], p), v3.dot(R[2], p)];
export const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
export const mt = (A) => [0, 1, 2].map((i) => [A[0][i], A[1][i], A[2][i]]);
export const qmat = (q) => { const [w, x, y, z] = q; return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]; };
export const axisRot = (u, th) => { const s = Math.sin(th / 2), n = len3(u); return qmat([Math.cos(th / 2), s * u[0] / n, s * u[1] / n, s * u[2] / n]); };
export const rotAngle = (R) => Math.acos(Math.max(-1, Math.min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)));
function matToQuat(m) {
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]; }
  const n = Math.hypot(...q); return q.map((v) => v / n);
}
// Geodesic on SO(3) through unit quaternions: the same path as R0 exp(t log(R0ᵀ R1)).
export function slerp(a, b, u) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) { b = b.map((v) => -v); d = -d; }
  if (d > 0.9995) { const r = a.map((v, i) => v + u * (b[i] - v)); const n = Math.hypot(...r); return r.map((v) => v / n); }
  const th = Math.acos(d), s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - u) * th) * v + Math.sin(u * th) * b[i]) / s);
}
export const qangle = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));
// Uniform rotations: a normalized 4D Gaussian is uniform on SO(3).
const randQuat = (rng) => { const q = [0, 0, 0, 0].map(() => rng.gauss()), n = Math.hypot(...q); return q.map((v) => v / n); };
// Largest eigenvector of a symmetric 4 × 4 matrix, by Jacobi rotations.
function topEig4(A) {
  const a = A.map((r) => r.slice()), V = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => +(i === j)));
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0; for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] ** 2;
    if (off < 1e-26) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), tt = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
      for (let k = 0; k < 4; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 4; k++) { const x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
    }
  }
  let b = 0; for (let i = 1; i < 4; i++) if (a[i][i] > a[b][b]) b = i;
  return [V[0][b], V[1][b], V[2][b], V[3][b]];
}
// The rotation R that maximizes sum_i x_iᵀ R m_i (Horn's quaternion method), from C = sum_i x_i m_iᵀ.
function bestRot(C) {
  const S = mt(C);   // S[a][b] = sum_i m_ia x_ib
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  return qmat(topEig4([[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz], [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]]));
}

// ---------- Step 4: molecules ----------
// The data of one molecule are its conformer m (centred) at a random rotation R, with Gaussian noise of
// standard deviation MS per coordinate: x_1 = R m + MS ξ. Given x_t, the posterior over R has the density
// exp((t / v) <R, C>) with C = sum_i x_i m_iᵀ and v = t² MS² + (1 - t)² MSIG². It is evaluated on NQ
// rotations R* Q_q around its mode R*, which makes the field exactly equivariant: rotating x_t by g
// rotates R* and with it every quadrature point. With 'one', the data contain only R = I.
export const MS = 0.1, MSIG = 2, NQ = 4096, MSTEPS = 60, NMOL = 32;
const QUAD = (() => { const rng = makeRng(77), Q = [qmat([1, 0, 0, 0])]; for (let i = 1; i < NQ; i++) Q.push(qmat(randQuat(rng))); return Q; })();
export const CPK = { H: '#e8e8e8', C: '#5a5f69', N: '#3b6fe0', O: '#e0412f', S: '#e5c33a' };
export const VDW = { H: 0.55, C: 0.8, N: 0.78, O: 0.76, S: 0.95 };
export function moleculeModel(mol) {
  const c0 = [0, 1, 2].map((d) => mol.atoms.reduce((s, a) => s + a[1 + d], 0) / mol.atoms.length);
  const m = mol.atoms.map((a) => [a[1] - c0[0], a[2] - c0[1], a[3] - c0[2]]), n = m.length;
  function denoise(x, t, one) {
    const v = t * t * MS * MS + (1 - t) ** 2 * MSIG * MSIG, cs = t * MS * MS / v;
    let Rbar;
    if (one) Rbar = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    else {
      const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      x.forEach((p, i) => { for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += p[a] * m[i][b]; });
      const Rs = bestRot(C), Cp = mm(mt(Rs), C), k = t / v;
      const l = QUAD.map((Q) => { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += Q[a][b] * Cp[a][b]; return k * s; });
      const w = softmax(l), Qb = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      QUAD.forEach((Q, q) => { if (w[q] > 1e-14) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) Qb[a][b] += w[q] * Q[a][b]; });
      Rbar = mm(Rs, Qb);
    }
    return m.map((mi, i) => { const r = mv(Rbar, mi); return [0, 1, 2].map((d) => (1 - cs * t) * r[d] + cs * x[i][d]); });
  }
  // Path from a noise x_0 at the grid times k / MSTEPS, with the update x <- x + (t' - t)/(1 - t)(E - x).
  function flow(x0, one) {
    let x = x0.map((p) => p.slice());
    const path = [x];
    for (let k = 0; k < MSTEPS; k++) {
      const t = k / MSTEPS, r = 1 / (MSTEPS - k), E = denoise(x, t, one);
      x = x.map((p, i) => p.map((v, d) => v + r * (E[i][d] - v)));
      path.push(x);
    }
    return path;
  }
  const noise = (seed) => { const rng = makeRng(seed), z = m.map(() => [0, 0, 0].map(() => MSIG * rng.gauss())), c = [0, 1, 2].map((d) => z.reduce((s, p) => s + p[d], 0) / n); return z.map((p) => v3.sub(p, c)); };
  // Rotation of a generated molecule relative to the conformer, from the best superposition.
  const orient = (x) => { const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; x.forEach((p, i) => { for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += p[a] * m[i][b]; }); return bestRot(C); };
  const cache = {};
  function ends(one) {
    const key = one ? 'one' : 'all';
    return (cache[key] ??= Array.from({ length: NMOL }, (_, s) => { const p = flow(noise(500 + s), one); return p[MSTEPS]; }));
  }
  return { m, n, bonds: mol.bonds, el: mol.atoms.map((a) => a[0]), flow, noise, ends, orient };
}
export const pathAtK = (path, t) => { const u = Math.min(Math.max(t, 0), 1) * (path.length - 1), k = Math.min(Math.floor(u), path.length - 2), a = u - k; return path[k].map((p, i) => p.map((v, d) => (1 - a) * v + a * path[k + 1][i][d])); };

// ---------- Step 5: protein backbones on SE(3) ----------
// Frames from the real N, Cα and C atoms: e1 along Cα→C, e2 in the N–Cα–C plane, e3 = e1 × e2. Each
// residue keeps its own N and C coordinates in its frame, so the frame path keeps every bond exact.
// The noise draws the Cα positions from a centred Gaussian that matches the protein's radius of
// gyration and the rotations uniformly from SO(3).
export function proteinModel(prot, seed = 16) {
  const bb = prot.bb, n = bb.length;
  const c = [0, 1, 2].map((d) => bb.reduce((s, r) => s + r[1][d], 0) / n);
  const atoms1 = bb.map((r) => r.map((p) => v3.sub(p, c)));
  const frameOf = ([N, CA, C]) => { const e1 = nrm(v3.sub(C, CA)), u = v3.sub(N, CA), e2 = nrm(v3.sub(u, v3.scl(e1, v3.dot(u, e1)))); return [e1, e2, v3.cross(e1, e2)]; };
  const R1 = atoms1.map((r) => mt(frameOf(r)));   // columns e1, e2, e3
  const Q1 = R1.map(matToQuat), CA1 = atoms1.map((r) => r[1]);
  const loc = atoms1.map((r, i) => [mv(mt(R1[i]), v3.sub(r[0], r[1])), mv(mt(R1[i]), v3.sub(r[2], r[1]))]);
  const rg = Math.sqrt(CA1.reduce((s, p) => s + v3.dot(p, p), 0) / n), sig = rg / Math.sqrt(3);
  const rng = makeRng(seed);
  const CA0 = (() => { const x = CA1.map(() => [0, 0, 0].map(() => sig * rng.gauss())), m = [0, 1, 2].map((d) => x.reduce((s, p) => s + p[d], 0) / n); return x.map((p) => v3.sub(p, m)); })();
  const Q0 = CA1.map(() => randQuat(rng));
  const place = (x, R, i) => [v3.add(x, mv(R, loc[i][0])), x, v3.add(x, mv(R, loc[i][1]))];
  const atoms0 = CA0.map((x, i) => place(x, qmat(Q0[i]), i));
  const bond = loc.map(([a, b]) => [len3(a), len3(b)]);
  // Atoms and frames at time t, on the frame path or with every atom on a straight line.
  function at(t, linear) {
    if (linear) {
      const A = atoms0.map((r, i) => r.map((p, j) => v3.add(v3.scl(p, 1 - t), v3.scl(atoms1[i][j], t))));
      return { atoms: A, R: A.map((r) => mt(frameOf(r))) };
    }
    const R = Q0.map((q, i) => qmat(slerp(q, Q1[i], t)));
    return { atoms: CA0.map((x, i) => place(v3.add(v3.scl(x, 1 - t), v3.scl(CA1[i], t)), R[i], i)), R };
  }
  const bondErrs = (A) => A.map(([N, CA, C], i) => Math.max(Math.abs(len3(v3.sub(N, CA)) / bond[i][0] - 1), Math.abs(len3(v3.sub(C, CA)) / bond[i][1] - 1)));
  const rmsd = (A) => Math.sqrt(A.reduce((s, r, i) => s + v3.dot(v3.sub(r[1], CA1[i]), v3.sub(r[1], CA1[i])), 0) / n);
  const ang = Q0.map((q, i) => qangle(q, Q1[i]));
  const show = ang.reduce((b, a, i) => (Math.abs(a - 2.1) < Math.abs(ang[b] - 2.1) ? i : b), 0);
  // For one residue: the naive matrix average (1 - t) R0 + t R1 and its defects.
  function naive(i, t) {
    const A = qmat(Q0[i]), B = R1[i], M = A.map((r, a) => r.map((v, b) => (1 - t) * v + t * B[a][b]));
    const cols = mt(M), det = v3.dot(cols[0], v3.cross(cols[1], cols[2]));
    let orth = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) orth += (v3.dot(cols[a], cols[b]) - (a === b ? 1 : 0)) ** 2;
    return { M, det, orth: Math.sqrt(orth) };
  }
  return { n, ss: prot.ss, at, bondErrs, rmsd, ang, show, naive, R1, Q0, Q1, rg, CA1 };
}
