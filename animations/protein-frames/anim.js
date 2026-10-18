// A16 · Protein backbone generation with SE(3) frames.
// Each residue is a rigid body: a rotation R_i and a translation x_i (the Cα position). The backbone atoms N, Cα and C
// sit at fixed ideal coordinates in that frame, so bond lengths and the N–Cα–C angle hold at every step. The target is
// one fixed three-helix bundle. Noise draws translations from a centered Gaussian and rotations uniformly from SO(3).
// The path is the conditional path that SE(3) flow matching regresses on: translations interpolate linearly,
// x_t = t x1 + (1 − t) x0, and rotations follow the geodesic from R0 to R1. Nothing here is learned.
import { makeRng } from '../lib/rng.js';
import { panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const VX = 1010, VY = 440, PXA = 13;   // view centre and pixels per Å
const SIGMA0 = 10;                     // Å, noise scale for translations
// Ideal local atom coordinates (Å) in the residue frame, Cα at the origin.
const N_LOC = [-0.525, 1.363, 0], C_LOC = [1.526, 0, 0];

const T = {
  setup: [0, 11], zoom: [3.5, 10.5],
  noise: [11, 19], noiseRun: [12, 16],
  flow: [19, 41], flowRun: [20, 38],
  fold: [41, 54],
};

const STILLS = [
  { t: 10.0, name: 'frames', label: 'Residues as rigid frames' },
  { t: 18.5, name: 'noise', label: 'Noise on SE(3)ᴺ' },
  { t: 29.0, name: 'midway', label: 'Halfway along the path' },
  { t: 53.2, name: 'fold', label: 'Generated backbone' },
];

export default {
  id: 'protein-frames',
  title: 'Protein backbone generation with SE(3) frames',
  duration: 54,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Frames' },
    { t: T.noise[0], label: 'I · Noise' },
    { t: T.flow[0], label: 'II · The flow' },
    { t: T.fold[0], label: 'III · Geometry' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 990, h: 900 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    ctx.fillStyle = th.bg; ctx.fillRect(0, 0, W, H);
    const tau = interpTime(t);
    const frames = st.frames(tau);
    const yaw = 0.16 * t + 0.4, pitch = 0.32;
    const zoom = t < T.noise[0] ? ease.inOut(window_(t, T.zoom[0], T.zoom[1], 1.2)) : 0;
    drawBackbone(ctx, th, frames, yaw, pitch, zoom, st, tau);
    if (t >= T.fold[0] - 0.5) drawChart(ctx, th, st, window_(t, T.fold[0], T.fold[1] + 1, 0.6));
    if (!env.clean) drawPanel(ctx, t, env, st, tau, frames);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// Interpolation time: 1 (data) in the setup, falls to 0 while the noise is drawn, then runs 0 → 1.
function interpTime(t) {
  if (t < T.noise[0]) return 1;
  if (t < T.flow[0]) return 1 - ease.inOut(seg(t, ...T.noiseRun));
  return ease.inOut(seg(t, ...T.flowRun));
}

// ---------- Quaternions and vectors ----------
const qmul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
];
function qrot(q, v) {
  const [w, x, y, z] = q, p = qmul(qmul(q, [0, ...v]), [w, -x, -y, -z]);
  return [p[1], p[2], p[3]];
}
function slerp(a, b, u) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (d < 0) { b = b.map((v) => -v); d = -d; }
  if (d > 0.9995) { const r = a.map((v, i) => v + u * (b[i] - v)); const n = Math.hypot(...r); return r.map((v) => v / n); }
  const th = Math.acos(d), s = Math.sin(th);
  return a.map((v, i) => (Math.sin((1 - u) * th) * v + Math.sin(u * th) * b[i]) / s);
}
const qangle = (a, b) => 2 * Math.acos(Math.min(1, Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])));
// Rotation matrix with columns e1, e2, e3 → quaternion.
function matToQuat(e1, e2, e3) {
  const m = [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]];
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]; }
  else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]; }
  else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]; }
  const n = Math.hypot(...q);
  return q.map((v) => v / n);
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dotp = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => scl(a, 1 / Math.hypot(...a));

// ---------- Target fold: a three-helix bundle ----------
function helix(n, axis, dir, phase) {
  // α-helix: radius 2.3 Å, rise 1.5 Å, 100° per residue. Axis along z through (axis[0], axis[1]).
  return Array.from({ length: n }, (_, i) => {
    const a = phase + dir * i * (100 * Math.PI / 180);
    return [axis[0] + 2.3 * Math.cos(a), axis[1] + 2.3 * Math.sin(a), dir * (i - (n - 1) / 2) * 1.5];
  });
}
function loop(p, q, n, bulge) {
  // Quadratic Bézier from p to q bulging along z, with n interior residues.
  const m = add(scl(add(p, q), 0.5), [0, 0, bulge]);
  return Array.from({ length: n }, (_, i) => {
    const u = (i + 1) / (n + 1);
    return add(add(scl(p, (1 - u) ** 2), scl(m, 2 * u * (1 - u))), scl(q, u * u));
  });
}
function targetCA() {
  const L = 15, R = 5.6;
  const ax = [0, 1, 2].map((k) => [R * Math.cos(2 * Math.PI * k / 3 + 0.3), R * Math.sin(2 * Math.PI * k / 3 + 0.3)]);
  const h1 = helix(L, ax[0], 1, 0), h2 = helix(L, ax[1], -1, 2.2), h3 = helix(L, ax[2], 1, 4.1);
  const l1 = loop(h1[L - 1], h2[0], 3, 5), l2 = loop(h2[L - 1], h3[0], 3, -5);
  const ca = [...h1, ...l1, ...h2, ...l2, ...h3];
  const c = ca.reduce((s, p) => add(s, p), [0, 0, 0]).map((v) => v / ca.length);
  return ca.map((p) => sub(p, c));
}
// Frame from the Cα trace: e1 along the chain, e2 toward the local curvature, e3 = e1 × e2.
function framesFromTrace(ca) {
  return ca.map((p, i) => {
    const a = ca[Math.max(i - 1, 0)], b = ca[Math.min(i + 1, ca.length - 1)];
    const e1 = norm(sub(b, a));
    let c = sub(add(a, b), scl(p, 2));
    if (i === 0 || i === ca.length - 1 || Math.hypot(...c) < 1e-6) c = cross(e1, [0.3, 0.5, 0.8]);
    const e2 = norm(sub(c, scl(e1, dotp(c, e1))));
    return matToQuat(e1, e2, cross(e1, e2));
  });
}

export function buildData() {
  const ca1 = targetCA(), q1 = framesFromTrace(ca1), N = ca1.length;
  const rng = makeRng(16);
  let x0 = ca1.map(() => [rng.gauss() * SIGMA0, rng.gauss() * SIGMA0, rng.gauss() * SIGMA0]);
  const c0 = x0.reduce((s, p) => add(s, p), [0, 0, 0]).map((v) => v / N);
  x0 = x0.map((p) => sub(p, c0));
  // Uniform rotations: a normalized 4D Gaussian is Haar-distributed on SO(3).
  const q0 = ca1.map(() => norm4([rng.gauss(), rng.gauss(), rng.gauss(), rng.gauss()]));
  const frames = (tau) => ca1.map((p, i) => ({ x: add(scl(x0[i], 1 - tau), scl(p, tau)), q: slerp(q0[i], q1[i], tau) }));
  const stats = (tau) => {
    const f = frames(tau);
    let rmsd = 0, ang = 0, cc = 0;
    f.forEach((r, i) => { rmsd += dotp(sub(r.x, ca1[i]), sub(r.x, ca1[i])); ang += qangle(r.q, q1[i]); });
    for (let i = 0; i < N - 1; i++) cc += Math.hypot(...sub(f[i + 1].x, f[i].x));
    return { rmsd: Math.sqrt(rmsd / N), ang: (ang / N) * 180 / Math.PI, cc: cc / (N - 1) };
  };
  const curve = Array.from({ length: 101 }, (_, k) => stats(k / 100));
  return { N, frames, stats, curve, ca1 };
}
function norm4(q) { const n = Math.hypot(...q); return q.map((v) => v / n); }

// ---------- Drawing ----------
function project(p, yaw, pitch, scale, cx, cy) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const x1 = c * p[0] + s * p[1], y1 = -s * p[0] + c * p[1];
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y2 = cp * p[2] - sp * y1, z2 = sp * p[2] + cp * y1;
  return [cx + x1 * scale, cy - y2 * scale, z2];
}

function drawBackbone(ctx, th, frames, yaw, pitch, zoom, st, tau) {
  const N = frames.length;
  // During the zoom we magnify around residue FOCUS and show its local axes.
  const FOCUS = 7;
  const scale = PXA * (1 + 0.9 * zoom);
  const f0 = frames[FOCUS].x;
  const cx = VX, cy = VY;
  const P = (p) => project(sub(p, scl(f0, zoom)), yaw, pitch, scale, cx, cy);
  const atoms = frames.map((f) => ({ n: P(add(f.x, qrot(f.q, N_LOC))), ca: P(f.x), c: P(add(f.x, qrot(f.q, C_LOC))) }));
  const zs = atoms.map((a) => a.ca[2]), zmin = Math.min(...zs), zmax = Math.max(...zs);
  const depth = (z) => (zmax > zmin ? (z - zmin) / (zmax - zmin) : 0.5);
  // Trace through Cα: faint in noise, solid near data.
  const items = [];
  for (let i = 0; i < N - 1; i++) items.push({ z: (zs[i] + zs[i + 1]) / 2, kind: 'trace', i });
  for (let i = 0; i < N; i++) items.push({ z: zs[i] + 0.01, kind: 'res', i });
  items.sort((a, b) => a.z - b.z);
  // Peptide bond C(i) → N(i+1) is drawn with the trace.
  for (const it of items) {
    const i = it.i, col = rampColor(th, i / (N - 1), 1), d = depth(it.z), fogA = 0.35 + 0.65 * d;
    if (it.kind === 'trace') {
      ctx.save();
      ctx.globalAlpha = fogA * (0.18 + 0.5 * tau);
      ctx.strokeStyle = col; ctx.lineWidth = 2 + 2.5 * tau;
      ctx.beginPath(); ctx.moveTo(atoms[i].c[0], atoms[i].c[1]); ctx.lineTo(atoms[i + 1].n[0], atoms[i + 1].n[1]); ctx.stroke();
      ctx.restore();
    } else {
      const a = atoms[i];
      ctx.save();
      ctx.globalAlpha = fogA;
      ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 3.2;
      ctx.beginPath(); ctx.moveTo(...a.n.slice(0, 2)); ctx.lineTo(...a.ca.slice(0, 2)); ctx.lineTo(...a.c.slice(0, 2)); ctx.stroke();
      ctx.globalAlpha = fogA * 0.28;
      ctx.beginPath(); ctx.moveTo(...a.n.slice(0, 2)); ctx.lineTo(...a.ca.slice(0, 2)); ctx.lineTo(...a.c.slice(0, 2)); ctx.closePath(); ctx.fill();
      ctx.globalAlpha = fogA;
      ctx.beginPath(); ctx.arc(a.ca[0], a.ca[1], 3 + 2 * d, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    }
  }
  if (zoom > 0.02) {
    // Local axes of the focus residue and atom labels.
    const f = frames[FOCUS], o = P(f.x), L = 2.4;
    const axes = [[[1, 0, 0], th.ramp[3], 'e₁'], [[0, 1, 0], th.accent, 'e₂'], [[0, 0, 1], th.fg, 'e₃']];
    for (const [v, col, lab] of axes) {
      const e = P(add(f.x, qrot(f.q, scl(v, L))));
      ctx.save(); ctx.globalAlpha = zoom; ctx.strokeStyle = col; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(e[0], e[1]); ctx.stroke(); ctx.restore();
      text(ctx, lab, e[0] + 8, e[1] - 6, { size: 22, weight: 600, color: col, alpha: zoom, font: 'serif' });
    }
    const a = atoms[FOCUS];
    for (const [p, lab] of [[a.n, 'N'], [a.ca, 'Cα'], [a.c, 'C']]) text(ctx, lab, p[0] - 10, p[1] + 30, { size: 19, weight: 700, color: th.fg, alpha: zoom });
  }
}

// Act III: how the geometry emerges along t.
function drawChart(ctx, th, st, a) {
  if (a <= 0) return;
  const B = { x: 1230, y: 620, w: 300, h: 170 };
  ctx.save();
  ctx.globalAlpha = a * 0.93; ctx.fillStyle = th.bg; ctx.fillRect(B.x - 56, B.y - 64, B.w + 90, B.h + 112);
  ctx.globalAlpha = a; ctx.strokeStyle = th.faint; ctx.strokeRect(B.x - 56, B.y - 64, B.w + 90, B.h + 112);
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(B.x, B.y); ctx.lineTo(B.x, B.y + B.h); ctx.lineTo(B.x + B.w, B.y + B.h); ctx.stroke();
  ctx.restore();
  const ymax = 20, X = (u) => B.x + u * B.w, Y = (v) => B.y + B.h - (Math.min(v, ymax) / ymax) * B.h;
  for (const v of [0, 3.8, 10, 20]) text(ctx, v === 3.8 ? '3.8' : String(v), B.x - 8, Y(v) + 5, { size: 13, color: th.muted, align: 'right', alpha: a });
  text(ctx, '0', X(0), B.y + B.h + 20, { size: 13, color: th.muted, align: 'center', alpha: a });
  text(ctx, '1', X(1), B.y + B.h + 20, { size: 13, color: th.muted, align: 'center', alpha: a });
  text(ctx, 'interpolation time t', B.x + B.w / 2, B.y + B.h + 38, { size: 14, color: th.muted, align: 'center', alpha: a });
  const series = [['rmsd', th.fg, 'Cα RMSD to target (Å)', -44], ['cc', th.ramp[3], 'neighbour Cα–Cα distance (Å)', -26], ['bond', th.accent, 'N–Cα bond length (Å)', -8]];
  for (const [key, col, name, dy] of series) {
    ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2.4;
    ctx.beginPath();
    st.curve.forEach((s, k) => { const v = key === 'bond' ? Math.hypot(...N_LOC) : s[key]; (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k / 100), Y(v)); });
    ctx.stroke(); ctx.restore();
    text(ctx, name, B.x - 40, B.y + dy, { size: 14, weight: 600, color: col, alpha: a });
  }
}

// ---------- Text panel ----------
const COPY = [
  { span: T.setup, act: 'Residues as rigid frames',
    body: 'A protein backbone is a chain of residues. Each one is treated as a rigid body with a rotation and a translation, and its N, Cα and C atoms sit at fixed ideal positions inside that frame. A structure with N residues is a point in SE(3)ᴺ.',
    eq: 'T_i = (R_i, x_i) ∈ SE(3)' },
  { span: T.noise, act: 'Act I · Noise on frames',
    body: 'The noise distribution places the translations as a centered Gaussian cloud and draws each rotation uniformly from SO(3). Every residue keeps its own internal geometry, but the chain is scattered and the orientations carry no information.',
    eq: 'x_i ~ N(0, σ² I),  R_i ~ Uniform(SO(3))' },
  { span: T.flow, act: 'Act II · The flow on SE(3)',
    body: 'Flow matching on frames moves each translation along a straight line and each rotation along the shortest rotation, a geodesic on SO(3). A network is trained to predict the velocity of this path from the noisy frames. Here we show the path toward one fixed target fold.',
    eq: 'x_t = t x₁ + (1 − t) x₀,  R_t = R₀ exp(t log(R₀ᵀ R₁))' },
  { span: T.fold, act: 'Act III · What frames buy',
    body: 'Because atoms are placed inside each frame, bond lengths and angles within a residue are exact at every step, which a free point cloud would have to learn. What remains to be generated is the arrangement of the frames: the spacing between neighbours and the fold.',
    eq: '' },
];

function drawPanel(ctx, t, env, st, tau, frames) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §7 APPLICATIONS', 'Protein backbones', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: c.eq.length > 40 ? 20 : 23, color: th.fg, alpha: a, font: 'serif' });
  }
  const s = st.stats(tau);
  const ra = seg(t, T.noise[0], T.noise[0] + 0.8);
  if (ra > 0) {
    const read = (lab, v, col, xx) => {
      text(ctx, lab, xx, 650, { size: 15, weight: 600, color: th.muted, alpha: ra, spacing: 1.5 });
      text(ctx, v, xx, 706, { size: 44, weight: 700, color: col, alpha: ra, font: 'mono' });
    };
    read('Cα RMSD (Å)', s.rmsd.toFixed(1), th.fg, x);
    read('ROTATION ERROR', `${Math.round(s.ang)}°`, th.ramp[3], x + 170);
    read('N–Cα BOND', `${Math.hypot(...N_LOC).toFixed(2)}`, th.accent, x + 340);
  }
  text(ctx, `t = ${tau.toFixed(2)}`, x, 768, { size: 22, color: th.muted, font: 'mono' });
  paragraph(ctx, `Target: an idealized three-helix bundle with ${st.N} residues. The frames follow the exact conditional path of SE(3) flow matching toward that target (linear in x, geodesic in R). A trained model would predict this target from the noisy frames. Nothing here is learned.`, x, 806, w, { size: 14, color: th.muted, lineHeight: 20 });
}
