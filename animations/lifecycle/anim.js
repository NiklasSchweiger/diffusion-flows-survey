// A0 · The lifecycle of a generative model, on one 2D toy.
// Every stage is an exact Gaussian mixture, so the flows are exact rather than learned. The same 400 noise draws are
// carried through all five stages: pretrain on a ring, fine-tune to add a mode, align by tilting with a reward,
// distill the ODE into one step, and spend test-time compute with best-of-N against a verifier.
import { GMM, trajAt } from '../lib/gmm.js';
import { makeRng } from '../lib/rng.js';
import { makePlot, drawGrid, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1090, CY = 395, S = 132;
const plot = makePlot(CX, CY, S);
const px = plot.px;
const STD = 0.16, LAMBDA = 1.2, N = 400, STEPS = 40, NP = 32, GROUP = 8;
const RING = Array.from({ length: 6 }, (_, k) => [1.6 * Math.cos(Math.PI / 6 + (k * Math.PI) / 3), 1.6 * Math.sin(Math.PI / 6 + (k * Math.PI) / 3)]);
const TARGET = RING[2];   // the verifier prefers the upper-left mode

const STAGES = [
  { key: 'pre', span: [0, 10], name: 'Pretrain', sec: '§2' },
  { key: 'ft', span: [10, 19], name: 'Fine-tune', sec: '§3' },
  { key: 'al', span: [19, 28], name: 'Align', sec: '§4' },
  { key: 'di', span: [28, 38], name: 'Distill', sec: '§5' },
  { key: 'se', span: [38, 50], name: 'Search', sec: '§6' },
];
const STILLS = [
  { t: 9.5, name: 'pretrain', crop: { x: 626, y: 0, w: 954, h: 690 }, label: '1 · Pretrain' },
  { t: 18.5, name: 'finetune', crop: { x: 626, y: 0, w: 954, h: 690 }, label: '2 · Fine-tune' },
  { t: 27.5, name: 'align', crop: { x: 626, y: 0, w: 954, h: 690 }, label: '3 · Align' },
  { t: 37.5, name: 'distill', label: '4 · Distill' },
  { t: 49.4, name: 'search', label: '5 · Search' },
];

export default {
  id: 'lifecycle',
  title: 'The lifecycle of a generative model',
  duration: 50,
  width: W, height: H,
  chapters: STAGES.map((s, i) => ({ t: s.span[0], label: `${i + 1} · ${s.name}` })),
  stills: STILLS,  // Paper crops trimmed to the drawn content, one height per figure row.
  crop: { x: 620, y: 0, w: 960, h: 720 },

  setup() {
    // Pretrained ring; fine-tuned ring plus a centre mode; aligned = fine-tuned tilted by exp(λ y).
    const pre = new GMM(RING, STD);
    const ftMeans = [...RING, [0, 0]], ftW = [1, 1, 1, 1, 1, 1, 1.5];
    const ft = new GMM(ftMeans, STD, { weights: ftW });
    // exp(λ y) times a Gaussian is a Gaussian with mean shifted by λσ² and weight scaled by exp(λ μ_y + λ²σ²/2).
    const alMeans = ftMeans.map(([x, y]) => [x, y + LAMBDA * STD * STD]);
    const alW = ftMeans.map(([, y], k) => ftW[k] * Math.exp(LAMBDA * y));
    const al = new GMM(alMeans, STD, { weights: alW });

    const rng = makeRng(17), noise = Array.from({ length: N }, () => [rng.gauss(), rng.gauss()]);
    const trajPre = noise.map(([x, y]) => pre.integrateODE(x, y, STEPS));
    const end = (g) => noise.map(([x, y]) => { const tr = g.integrateODE(x, y, STEPS); return [tr[2 * STEPS], tr[2 * STEPS + 1]]; });
    const P = { pre: trajPre.map((tr) => [tr[2 * STEPS], tr[2 * STEPS + 1]]), ft: end(ft), al: end(al) };
    const trajAl = noise.slice(0, NP).map(([x, y]) => al.integrateODE(x, y, STEPS));

    // Best-of-8 against a verifier that scores closeness to TARGET.
    const score = ([x, y]) => -Math.hypot(x - TARGET[0], y - TARGET[1]);
    const chosen = [];
    for (let g = 0; g < N / GROUP; g++) {
      let b = g * GROUP;
      for (let i = b + 1; i < (g + 1) * GROUP; i++) if (score(P.al[i]) > score(P.al[b])) b = i;
      chosen.push(b);
    }
    const near = (pts, c) => pts.filter(([x, y]) => Math.hypot(x - c[0], y - c[1]) < 3 * STD).length / pts.length;
    const meanY = (pts) => pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const m = {
      centrePre: near(P.pre, [0, 0]), centreFt: near(P.ft, [0, 0]),
      yFt: meanY(P.ft), yAl: meanY(P.al),
      targetOne: near(P.al, TARGET), targetBest: near(chosen.map((i) => P.al[i]), TARGET),
    };
    return { noise, trajPre, trajAl, P, chosen: new Set(chosen), m };
  },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    drawReward(ctx, th, t, env);
    drawDistill(ctx, th, st, t, env);
    drawVerifier(ctx, th, t, env);

    const sa = env.clean ? (t >= 38 ? 1 : 0) : ease.inOut(seg(t, 39, 41));
    for (let i = 0; i < N; i++) {
      const [x, y] = position(st, i, t);
      const pick = st.chosen.has(i);
      const [X, Y] = px(x, y);
      if (pick && sa > 0) dot(ctx, th, X, Y, th.ramp[3], 1, lerp(2.3, 3.6, sa));
      else dot(ctx, th, X, Y, th.accent, lerp(0.9, 0.25, sa), 2.3);
    }

    // The stage strip repeats the still titles, so paper stills leave it out.
    if (!env.clean) drawStrip(ctx, th, t);
    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// Same noise, a new exact model at each stage: samples flow out of noise, then move as the model changes.
function position(st, i, t) {
  if (t < 7.5) return trajAt(st.trajPre[i], ease.inOut(seg(t, 1.5, 7.5)));
  const pre = st.P.pre[i], ft = st.P.ft[i], al = st.P.al[i];
  if (t < 12) return pre;
  if (t < 21) { const u = ease.inOut(seg(t, 12, 15.5)); return [lerp(pre[0], ft[0], u), lerp(pre[1], ft[1], u)]; }
  const u = ease.inOut(seg(t, 21, 24.5));
  return [lerp(ft[0], al[0], u), lerp(ft[1], al[1], u)];
}

function drawReward(ctx, th, t, env) {
  const a = env.clean ? (t >= 19 && t < 28 ? 1 : 0) : window_(t, 19.5, 28, 0.8);
  if (a <= 0) return;
  ctx.save(); ctx.fillStyle = th.ramp[3];
  for (let y = -3; y < 3; y += 0.1) {
    ctx.globalAlpha = a * 0.2 * Math.exp(LAMBDA * y) / Math.exp(LAMBDA * 3);
    const [, Y0] = px(0, y + 0.1), [, Y1] = px(0, y);
    ctx.fillRect(650, Y0, 950, Y1 - Y0 + 0.5);
  }
  ctx.restore();
  text(ctx, 'reward r(x) = y', 1560, 60, { size: 18, weight: 600, color: th.ramp[3], alpha: a, align: 'right' });
  text(ctx, 'higher is better ↑', 1560, 84, { size: 15, color: th.muted, alpha: a, align: 'right' });
}

// Distill: the teacher's curved ODE paths, then the straight one-step map to the same endpoints.
function drawDistill(ctx, th, st, t, env) {
  const a = env.clean ? (t >= 28 && t < 38 ? 1 : 0) : window_(t, 28.5, 38, 0.6);
  if (a <= 0) return;
  const pc = env.clean ? 1 : ease.inOut(seg(t, 29, 32)), ps = env.clean ? 1 : ease.inOut(seg(t, 32.5, 35));
  ctx.save(); ctx.lineWidth = 1.3;
  for (const tr of st.trajAl) {
    const n = Math.floor(pc * STEPS);
    ctx.globalAlpha = a * 0.45; ctx.strokeStyle = th.muted; ctx.beginPath();
    for (let k = 0; k <= n; k++) { const [X, Y] = px(tr[2 * k], tr[2 * k + 1]); if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
    ctx.stroke();
    const [X0, Y0] = px(tr[0], tr[1]);
    ctx.globalAlpha = a * 0.6; ctx.fillStyle = th.muted; ctx.beginPath(); ctx.arc(X0, Y0, 2.4, 0, 2 * Math.PI); ctx.fill();
    if (ps > 0) {
      const [X1, Y1] = px(tr[2 * STEPS], tr[2 * STEPS + 1]);
      ctx.globalAlpha = a * 0.85; ctx.strokeStyle = th.ramp[3]; ctx.beginPath();
      ctx.moveTo(X0, Y0); ctx.lineTo(lerp(X0, X1, ps), lerp(Y0, Y1, ps)); ctx.stroke();
    }
  }
  ctx.restore();
  text(ctx, '40-step teacher', 1560, 60, { size: 17, weight: 600, color: th.muted, alpha: a * (pc > 0 ? 1 : 0), align: 'right' });
  text(ctx, 'one-step student', 1560, 86, { size: 17, weight: 600, color: th.ramp[3], alpha: a * ps, align: 'right' });
}

function drawVerifier(ctx, th, t, env) {
  const a = env.clean ? (t >= 38 ? 1 : 0) : seg(t, 38.5, 39.5);
  if (a <= 0) return;
  const [X, Y] = px(...TARGET), r = 3 * STD * S;
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = th.ramp[3]; ctx.lineWidth = 1.6; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
  text(ctx, 'verifier prefers this mode', X, Y - r - 14, { size: 16, color: th.ramp[3], alpha: a, align: 'center' });
}

// Lifecycle strip under the plot: past stages filled, current stage highlighted.
function drawStrip(ctx, th, t) {
  const y = 822, x0 = 720, dx = 200;
  ctx.save(); ctx.strokeStyle = th.faint; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 4 * dx, y); ctx.stroke(); ctx.restore();
  STAGES.forEach((s, i) => {
    const x = x0 + i * dx, cur = t >= s.span[0] && t < s.span[1], past = t >= s.span[1];
    ctx.save();
    ctx.fillStyle = cur ? th.accent : past ? th.muted : th.bg; ctx.strokeStyle = cur ? th.accent : th.muted; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, cur ? 11 : 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); ctx.restore();
    text(ctx, s.name, x, y + 38, { size: 17, weight: cur ? 700 : 500, color: cur ? th.fg : th.muted, align: 'center' });
    text(ctx, s.sec, x, y - 22, { size: 14, color: th.muted, align: 'center' });
  });
}

// ---------- Text panel ----------
const COPY = [
  { act: '1 · Pretrain',
    body: 'A flow model learns to carry Gaussian noise to the data along an ODE. Here the data are six Gaussians on a ring, and the velocity field is the exact one for that mixture. Every later stage starts from these 400 noise draws.',
    eq: 'dx/dt = v(x, t),  x₀ ~ N(0, I)' },
  { act: '2 · Fine-tune',
    body: 'Fine-tuning moves the model to new data from the same family. The new data add a mode at the centre and keep the ring. The same noise now lands on the new distribution, and most samples stay on their old mode.',
    eq: 'p(x) → p′(x)' },
  { act: '3 · Align',
    body: 'Alignment reweights the model by a reward instead of matching new data. With a KL penalty the optimum is the old distribution tilted by exp(r/β). The reward here is the height y, so the upper modes gain mass and the lower ones lose it.',
    eq: 'p*(x) ∝ p(x) exp(r(x)/β)' },
  { act: '4 · Distill',
    body: 'Sampling the ODE takes many network evaluations. Distillation trains a student that jumps from noise to the endpoint in one step. An ideal student reproduces the teacher’s flow map, which is what the straight segments show.',
    eq: 'x₁ = f(x₀) in one step' },
  { act: '5 · Search',
    body: 'At test time one can spend extra compute on each output. Best-of-N draws eight samples and keeps the one a verifier scores highest. Without changing any weights, almost every kept sample lands on the mode the verifier prefers.',
    eq: 'x* = argmax over N draws of r′(x)' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460, m = st.m;
  panelHeader(ctx, th, H, 'SURVEY · §1 INTRODUCTION', 'The lifecycle of a model', x);
  STAGES.forEach((s, i) => {
    const a = window_(t, s.span[0], s.span[1], 0.4);
    if (a <= 0) return;
    const c = COPY[i];
    text(ctx, `${c.act}  (${s.sec})`, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    text(ctx, c.eq, x, yEnd + 34, { size: 23, color: th.fg, alpha: a, font: 'serif' });
    const R = [
      [['ODE STEPS', '40', th.fg], ['MASS AT CENTRE', m.centrePre.toFixed(2), th.ramp[3]]],
      [['CENTRE BEFORE', m.centrePre.toFixed(2), th.fg], ['CENTRE AFTER', m.centreFt.toFixed(2), th.ramp[3]]],
      [['MEAN REWARD BEFORE', m.yFt.toFixed(2), th.fg], ['AFTER', m.yAl.toFixed(2), th.ramp[3]]],
      [['TEACHER STEPS', '40', th.fg], ['STUDENT STEPS', '1', th.ramp[3]]],
      [['AT TARGET, ONE DRAW', m.targetOne.toFixed(2), th.fg], ['BEST OF 8', m.targetBest.toFixed(2), th.ramp[3]]],
    ][i];
    R.forEach(([lab, v, col], j) => {
      text(ctx, lab, x + j * 250, 690, { size: 15, weight: 600, color: th.muted, alpha: a, spacing: 1.5 });
      text(ctx, v, x + j * 250, 746, { size: 44, weight: 700, color: col, alpha: a, font: 'mono' });
    });
  });
  paragraph(ctx, 'Every stage is an exact Gaussian mixture, so the flows, the reward tilt and the one-step map are computed rather than learned. The same 400 noise draws are used throughout.', x, 806, w, { size: 14, color: th.muted, lineHeight: 20 });
}
