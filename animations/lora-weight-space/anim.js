// A6 · LoRA in weight space.
// A small flow-matching MLP (hidden width 64) is pretrained on six Gaussians on a ring. The fine-tuning data is the same
// ring with one mode moved to the centre. Only the middle 64×64 weight matrix W2 is adapted, either fully (4,096
// numbers) or through a rank-r update W2 + B A (rank 1: 128 numbers). The weights come from tools/train_lora.mjs (seeded,
// Adam). Samples are drawn here with the same noise for every model, so differences come from the weights alone.
import { makeRng } from '../lib/rng.js';
import { makePlot, drawGrid, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp } from '../lib/theme.js';
import WT from './weights.js';

const W = 1600, H = 900;
const CX = 930, CY = 420, S = 118;
const plot = makePlot(CX, CY, S);
const px = plot.px;
const HID = 64, STD = 0.13;
const RING = Array.from({ length: 6 }, (_, k) => [1.6 * Math.cos((2 * Math.PI * k) / 6 + Math.PI / 6), 1.6 * Math.sin((2 * Math.PI * k) / 6 + Math.PI / 6)]);
const NEW = [0, 0];
const NS = 360, STEPS = 32, NP = 36;

const T = {
  setup: [0, 9], paths: [1, 6],
  full: [9, 22], fullTarget: [9.6, 11], fullW: [12, 13.5], fullMorph: [14, 18.5],
  lora: [22, 36], loraBA: [23, 25], loraMorph: [26.5, 31],
  spec: [36, 50], specDraw: [37, 41],
};

const STILLS = [
  { t: 8.5, name: 'pretrained', crop: { x: 638, y: 12, w: 952, h: 668 }, label: 'Pretrained flow' },
  { t: 21.5, name: 'full', crop: { x: 638, y: 12, w: 952, h: 668 }, label: 'Full update of W₂' },
  { t: 35.5, name: 'lora', crop: { x: 638, y: 12, w: 952, h: 792 }, label: 'Rank-1 LoRA update' },
  { t: 49.4, name: 'spectrum', crop: { x: 638, y: 12, w: 952, h: 792 }, label: 'Rank of the two updates' },
];

export default {
  id: 'lora-weight-space',
  title: 'LoRA in weight space',
  duration: 50,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Pretrained' },
    { t: T.full[0], label: 'I · Full fine-tuning' },
    { t: T.lora[0], label: 'II · LoRA' },
    { t: T.spec[0], label: 'III · Rank' },
  ],
  stills: STILLS,  // Paper crops trimmed to the drawn content, one height per figure row.
  crop: { x: 600, y: 0, w: 990, h: 900 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    drawTargets(ctx, th, t, env);

    // Which samples to show: pretrained → full, then pretrained → LoRA rank 1.
    let A = st.samples.pre, B = st.samples.pre, u = 0, col = th.accent;
    if (t >= T.full[0] && t < T.lora[0]) { B = st.samples.full; u = ease.inOut(seg(t, ...T.fullMorph)); }
    else if (t >= T.lora[0]) { B = st.samples.r1; u = ease.inOut(seg(t, ...T.loraMorph)); }
    if (u > 0) col = th.ramp[3];
    if (t < T.full[0]) drawPaths(ctx, th, st, seg(t, ...T.paths), env.clean ? 1 : window_(t, -1, T.full[0], 0.6));
    for (let i = 0; i < NS; i++) {
      const [X, Y] = px(lerp(A[i][0], B[i][0], u), lerp(A[i][1], B[i][1], u));
      dot(ctx, th, X, Y, u > 0.5 ? th.ramp[3] : th.accent, 0.9, 2.3);
    }

    drawWeights(ctx, th, st, t);
    if (t >= T.spec[0] - 0.5) drawSpectrum(ctx, th, st, t);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Model ----------
const feat = (x, y, t) => [x, y, t, Math.sin(Math.PI * t), Math.cos(Math.PI * t), Math.sin(2 * Math.PI * t), Math.cos(2 * Math.PI * t)];
function velocity(p, W2, x, y, t) {
  const f = feat(x, y, t), h1 = new Float64Array(HID), out = [p.b3[0], p.b3[1]];
  for (let i = 0; i < HID; i++) { let s = p.b1[i]; const r = p.W1[i]; for (let j = 0; j < 7; j++) s += r[j] * f[j]; h1[i] = Math.tanh(s); }
  for (let i = 0; i < HID; i++) {
    let s = p.b2[i]; const r = W2[i];
    for (let j = 0; j < HID; j++) s += r[j] * h1[j];
    const h = Math.tanh(s);
    out[0] += p.W3[0][i] * h; out[1] += p.W3[1][i] * h;
  }
  return out;
}
function sample(p, W2, x, y, keep) {
  const h = 1 / STEPS, path = keep ? [[x, y]] : null;
  for (let i = 0; i < STEPS; i++) {
    const t = i * h, a = velocity(p, W2, x, y, t), b = velocity(p, W2, x + h * a[0], y + h * a[1], t + h);
    x += 0.5 * h * (a[0] + b[0]); y += 0.5 * h * (a[1] + b[1]);
    if (keep) path.push([x, y]);
  }
  return keep ? path : [x, y];
}

// Singular values by Jacobi eigenvalues of MᵀM.
function singularValues(M) {
  const n = M[0].length;
  const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => M.reduce((s, r) => s + r[i] * r[j], 0)));
  for (let sw = 0; sw < 12; sw++) for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
    if (Math.abs(A[p][q]) < 1e-12) continue;
    const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
    const c = 1 / Math.sqrt(t * t + 1), s = t * c;
    for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
    for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
  }
  return A.map((r, i) => Math.sqrt(Math.max(0, r[i]))).sort((a, b) => b - a);
}

function buildData() {
  const p = WT.pre;
  const add = (D) => p.W2.map((r, i) => r.map((v, j) => v + D[i][j]));
  const dFull = WT.full.W2.map((r, i) => r.map((v, j) => v - p.W2[i][j]));
  const W2 = { pre: p.W2, full: WT.full.W2, r1: add(WT[1].dW), r4: add(WT[4].dW) };
  const rng = makeRng(9), noise = Array.from({ length: NS }, () => [rng.gauss(), rng.gauss()]);
  const samples = {}, mass = {};
  for (const k of Object.keys(W2)) {
    samples[k] = noise.map(([x, y]) => sample(p, W2[k], x, y, false));
    const near = (c) => samples[k].filter(([x, y]) => Math.hypot(x - c[0], y - c[1]) < 3 * STD).length / NS;
    mass[k] = { new: near(NEW), old: near(RING[0]), rest: RING.slice(1).reduce((s, c) => s + near(c), 0) };
  }
  const paths = noise.slice(0, NP).map(([x, y]) => sample(p, W2.pre, x, y, true));
  const sv = { full: singularValues(dFull), r1: singularValues(WT[1].dW), r4: singularValues(WT[4].dW) };
  const en = {}; for (const k in sv) { const z = sv[k].reduce((s, v) => s + v * v, 0); en[k] = sv[k].map((v) => (v * v) / z); }
  const maxAbs = (M) => Math.max(...M.flat().map(Math.abs));
  return { samples, mass, paths, dFull, dR1: WT[1].dW, A: WT[1].A[0], B: WT[1].B.map((r) => r[0]), W2pre: p.W2, sv, en,
    scaleW: maxAbs(p.W2), scaleD: Math.max(maxAbs(dFull), maxAbs(WT[1].dW)) };
}

// ---------- Drawing ----------
function drawTargets(ctx, th, t, env) {
  const a = env.clean ? (t >= T.full[0] ? 1 : 0) : seg(t, ...T.fullTarget);
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a; ctx.lineWidth = 1.6; ctx.setLineDash([5, 5]);
  const r = 3 * STD * S;
  for (const c of [NEW, ...RING.slice(1)]) {
    const [X, Y] = px(...c); ctx.strokeStyle = th.ramp[3];
    ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.stroke();
  }
  const [X, Y] = px(...RING[0]); ctx.strokeStyle = th.muted;
  ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(X - r * 0.7, Y - r * 0.7); ctx.lineTo(X + r * 0.7, Y + r * 0.7); ctx.stroke();
  // Arrow from the old mode to the new one.
  const [X2, Y2] = px(...NEW), ang = Math.atan2(Y2 - Y, X2 - X);
  const sx = X + Math.cos(ang) * (r + 6), sy = Y + Math.sin(ang) * (r + 6), ex = X2 - Math.cos(ang) * (r + 6), ey = Y2 - Math.sin(ang) * (r + 6);
  ctx.strokeStyle = th.ramp[3]; ctx.fillStyle = th.ramp[3]; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 11 * Math.cos(ang - 0.4), ey - 11 * Math.sin(ang - 0.4)); ctx.lineTo(ex - 11 * Math.cos(ang + 0.4), ey - 11 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
  ctx.restore();
  text(ctx, 'fine-tuning data', X2 - 8, Y2 + r + 26, { size: 16, color: th.ramp[3], alpha: a, align: 'center' });
}

function drawPaths(ctx, th, st, p, a) {
  if (a <= 0 || p <= 0) return;
  const n = Math.floor(ease.inOut(p) * STEPS);
  ctx.save(); ctx.globalAlpha = a * 0.4; ctx.strokeStyle = th.accent; ctx.lineWidth = 1.2;
  for (const path of st.paths) {
    ctx.beginPath();
    for (let i = 0; i <= n; i++) { const [X, Y] = px(...path[i]); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
    ctx.stroke();
  }
  ctx.restore();
}

const MX = 1290, MY = 150, CELL = 4;   // heatmap origin and cell size (64 × 4 = 256 px)
function heat(ctx, th, M, x0, y0, cell, scale, a) {
  const pos = th.accent, neg = th.ramp[3];
  ctx.save();
  for (let i = 0; i < M.length; i++) for (let j = 0; j < M[i].length; j++) {
    const v = M[i][j] / scale;
    ctx.globalAlpha = a * clamp(Math.abs(v) ** 0.5, 0, 1);
    ctx.fillStyle = v > 0 ? pos : neg;
    ctx.fillRect(x0 + j * cell, y0 + i * cell, cell, cell);
  }
  ctx.globalAlpha = a; ctx.strokeStyle = th.faint; ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, M[0].length * cell, M.length * cell);
  ctx.restore();
}

function drawWeights(ctx, th, st, t) {
  const size = HID * CELL;
  // Pretrained W2 in the setup, the full update in Act I, B A in Act II and III.
  const pa = window_(t, 2, T.full[0] + 0.5, 0.6);
  if (pa > 0) {
    heat(ctx, th, st.W2pre, MX, MY, CELL, st.scaleW, pa);
    text(ctx, 'W₂, pretrained (64 × 64)', MX, MY - 18, { size: 17, weight: 600, color: th.fg, alpha: pa });
  }
  const fa = window_(t, T.fullW[0], T.lora[0] + 0.3, 0.6) * (t < T.lora[0] ? 1 : 1);
  if (fa > 0 && t < T.lora[0] + 0.3) {
    heat(ctx, th, st.dFull, MX, MY, CELL, st.scaleD, fa);
    text(ctx, 'ΔW₂, full update', MX, MY - 18, { size: 17, weight: 600, color: th.fg, alpha: fa });
    text(ctx, '4,096 trainable numbers', MX, MY + size + 26, { size: 14, color: th.muted, alpha: fa });
  }
  const la = t >= T.lora[0] ? seg(t, ...T.loraBA) : 0;
  if (la > 0) {
    const bmax = Math.max(...st.B.map(Math.abs)), amax = Math.max(...st.A.map(Math.abs));
    heat(ctx, th, st.B.map((v) => [v]), MX - 26, MY, CELL, bmax, la);
    heat(ctx, th, [st.A], MX, MY - 26, CELL, amax, la);
    const ma = seg(t, T.loraBA[0] + 0.8, T.loraBA[1] + 0.8);
    heat(ctx, th, st.dR1, MX, MY, CELL, st.scaleD, ma);
    text(ctx, 'B', MX - 24, MY + size + 24, { size: 17, weight: 700, color: th.fg, alpha: la, font: 'serif' });
    text(ctx, 'A', MX + size + 10, MY - 18, { size: 17, weight: 700, color: th.fg, alpha: la, font: 'serif' });
    text(ctx, 'ΔW₂ = B A, rank 1', MX, MY - 44, { size: 17, weight: 600, color: th.fg, alpha: la });
    text(ctx, '64 + 64 = 128 trainable numbers', MX, MY + size + 26, { size: 14, color: th.muted, alpha: la });
  }
}

// Act III: σ_k² / Σσ² for the two updates, i.e. the share of the squared norm in each singular direction.
function drawSpectrum(ctx, th, st, t) {
  const a = window_(t, T.spec[0], T.spec[1] + 1, 0.6), p = ease.inOut(seg(t, ...T.specDraw));
  if (a <= 0) return;
  const B = { x: 1290, y: 520, w: 256, h: 170 }, K = 12;
  ctx.save();
  ctx.globalAlpha = a * 0.93; ctx.fillStyle = th.bg; ctx.fillRect(B.x - 30, B.y - 60, B.w + 50, B.h + 130);
  ctx.globalAlpha = a; ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(B.x, B.y + B.h); ctx.lineTo(B.x + B.w, B.y + B.h); ctx.stroke();
  const bw = B.w / K;
  for (let k = 0; k < K; k++) {
    const hf = st.en.full[k] * B.h * p, hr = st.en.r1[k] * B.h * p;
    ctx.fillStyle = th.muted; ctx.globalAlpha = a * 0.7;
    ctx.fillRect(B.x + k * bw + 2, B.y + B.h - hf, bw / 2 - 2, hf);
    ctx.fillStyle = th.ramp[3]; ctx.globalAlpha = a;
    ctx.fillRect(B.x + k * bw + bw / 2, B.y + B.h - hr, bw / 2 - 2, hr);
  }
  ctx.restore();
  text(ctx, 'share of ‖ΔW₂‖²', B.x, B.y - 36, { size: 17, weight: 600, color: th.fg, alpha: a });
  text(ctx, 'full', B.x, B.y - 14, { size: 15, weight: 600, color: th.muted, alpha: a });
  text(ctx, 'LoRA rank 1', B.x + 46, B.y - 14, { size: 15, weight: 600, color: th.ramp[3], alpha: a });
  const e8 = Math.round(st.en.full.slice(0, 8).reduce((x, y) => x + y, 0) * 100);
  text(ctx, `full: top direction ${Math.round(st.en.full[0] * 100)}%`, B.x, B.y + B.h + 46, { size: 14, color: th.fg, alpha: a * p });
  text(ctx, `full: top eight ${e8}%`, B.x, B.y + B.h + 66, { size: 14, color: th.fg, alpha: a * p });
  text(ctx, 'rank 1: top direction 100%', B.x, B.y + B.h + 86, { size: 14, color: th.ramp[3], alpha: a * p });
  text(ctx, `largest ${K} of 64 directions`, B.x + B.w, B.y + B.h + 22, { size: 14, color: th.muted, alpha: a, align: 'right' });
}

// ---------- Text panel ----------
const COPY = [
  { span: T.setup, act: 'A pretrained flow',
    body: 'A small velocity network with hidden width 64 has been trained by flow matching on six Gaussians arranged on a ring. Its middle layer is a 64 × 64 weight matrix W₂. We will adapt only this matrix.',
    eq: 'v(x, t) = W₃ tanh(W₂ tanh(W₁ φ(x, t)))' },
  { span: T.full, act: 'Act I · Full fine-tuning',
    body: 'The new data moves one mode of the ring to the centre and keeps the other five. Fine-tuning every entry of W₂ moves that mode while the rest of the distribution stays in place. The update ΔW₂ touches all 4,096 entries.',
    eq: 'W₂ ← W₂ + ΔW₂' },
  { span: T.lora, act: 'Act II · Low-rank adaptation',
    body: 'LoRA freezes W₂ and learns the update as a product of two thin matrices, B with one column and A with one row. With rank 1 this is 128 numbers instead of 4,096, and the samples move in the same way.',
    eq: 'W₂ + B A,  B is 64 × r,  A is r × 64' },
  { span: T.spec, act: 'Act III · Rank',
    body: 'The full update is spread over many directions, because nothing in training asks it to be compact. The rank-1 update concentrates on one direction and does the same job. The change to the data is simple, so a few directions of weight space are enough to express it.',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §3 FINE-TUNING', 'LoRA in weight space', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: c.eq.length > 34 ? 21 : 23, color: th.fg, alpha: a, font: 'serif' });
  }
  const read = (lab, v, col, xx, al) => {
    text(ctx, lab, xx, 690, { size: 15, weight: 600, color: th.muted, alpha: al, spacing: 1.5 });
    text(ctx, v, xx, 746, { size: 44, weight: 700, color: col, alpha: al, font: 'mono' });
  };
  let key = 'pre', params = null, ra = window_(t, -1, T.full[0], 0.4);
  if (t >= T.full[0] && t < T.lora[0]) { key = seg(t, ...T.fullMorph) > 0.5 ? 'full' : 'pre'; params = '4,096'; ra = window_(t, T.full[0], T.lora[0], 0.4); }
  if (t >= T.lora[0]) { key = t >= T.spec[0] || seg(t, ...T.loraMorph) > 0.5 ? 'r1' : 'pre'; params = '128'; ra = window_(t, T.lora[0], 51, 0.4); }
  read('MASS AT CENTRE', st.mass[key].new.toFixed(2), th.ramp[3], x, ra);
  read('OTHER FIVE MODES', st.mass[key].rest.toFixed(2), th.fg, x + 210, ra);
  if (params) text(ctx, `${params} trainable numbers`, x, 784, { size: 18, color: th.muted, alpha: ra });
  paragraph(ctx, `Trained weights from tools/train_lora.mjs. ${NS} samples from the same noise for every model. Mass counts samples within 3σ of a mode; the fine-tuning data puts 0.17 at the centre.`, x, 812, w, { size: 14, color: th.muted, lineHeight: 20 });
}
