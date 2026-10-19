// A12 · Self Forcing vs teacher forcing: exposure bias in an autoregressive video model.
// Each "video frame" is a ball on a ring, described by its position p, size s and brightness c. In real videos the
// ball moves 15° per frame and s = c = 1. Both per-frame models below are written down, not trained. They make the
// same small systematic error on every real frame (a little too fast, too far out, too large, too bright), so their
// teacher-forced one-step loss is identical. The self-forced model also pulls a drifted frame back toward the
// nearest real frame. That term is zero on real contexts, so only a loss computed on the model's own rollouts can
// see it, which is the point of Self Forcing.
import { makeRng } from '../lib/rng.js';
import { seg, ease, window_, text, paragraph, lerp, clamp } from '../lib/theme.js';
import { panelHeader } from '../lib/scene.js';

const W = 1600, H = 900;
const N = 96;                         // frames per rollout
const OMEGA = (2 * Math.PI) / 24;     // 15° per frame
const TAU = 0.015;                    // per-frame sampling noise
const ERR = { w: 0.03, r: 0.003, s: 0.007, c: 0.008 };  // shared one-step model error
const KAPPA = 0.25;                   // learned pull-back of the self-forced model
const NR = 400;                       // rollouts for the error curves

const T = {
  data: [0, 10], dataPlay: [1.4, 8.6],
  tf: [10, 22], tfStep: [11.2, 20.4],
  roll: [22, 43], rollRun: [23, 39.5],
  chart: [43, 58], chartDraw: [44, 49],
};

const STILLS = [
  { t: 9.5, name: 'data', label: 'A real video' },
  { t: 18.7, name: 'one-step', label: 'Same one-step error on real frames' },
  { t: 42.4, name: 'rollout', label: 'Rollouts of 96 frames' },
  { t: 57.4, name: 'curves', label: 'Distance from the data' },
];

export default {
  id: 'self-forcing',
  title: 'Self Forcing vs teacher forcing',
  duration: 58,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.tf[0], label: 'I · Teacher forcing' },
    { t: T.roll[0], label: 'II · Rollout' },
    { t: T.chart[0], label: 'III · Error growth' },
  ],
  stills: STILLS,
  crop: { x: 620, y: 0, w: 970, h: 800 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    const rowA = 1 - seg(t, T.chart[0], T.chart[0] + 0.8);
    if (t < T.tf[0]) drawDataRow(ctx, th, st, t, window_(t, T.data[0] - 1, T.data[1], 0.5), env.clean);
    else if (t < T.roll[0]) drawOneStep(ctx, th, st, t, window_(t, T.tf[0], T.tf[1], 0.5));
    else if (rowA > 0) drawRollouts(ctx, th, st, t, Math.min(seg(t, T.roll[0], T.roll[0] + 0.5), rowA));
    if (t >= T.chart[0]) drawChart(ctx, th, st, t, seg(t, T.chart[0], T.chart[0] + 0.8));

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Dynamics ----------

const real = (z) => {
  const c = Math.cos(OMEGA), s = Math.sin(OMEGA);
  return [c * z[0] - s * z[1], s * z[0] + c * z[1], 1, 1];
};

// One-step mean of a model. sf = true adds the pull-back toward the nearest real frame (zero on real frames).
function step(z, sf) {
  const om = OMEGA * (1 + ERR.w), co = Math.cos(om), si = Math.sin(om), g = 1 + ERR.r;
  const out = [g * (co * z[0] - si * z[1]), g * (si * z[0] + co * z[1]), z[2] * (1 + ERR.s), z[3] + ERR.c];
  if (sf) {
    const r = Math.hypot(z[0], z[1]);
    const fz = [g * (co * z[0] - si * z[1]), g * (si * z[0] + co * z[1])];
    // Pull toward the continuation of the nearest real frame.
    out[0] += KAPPA * (g * (co * z[0] / r - si * z[1] / r) - fz[0]);
    out[1] += KAPPA * (g * (si * z[0] / r + co * z[1] / r) - fz[1]);
    out[2] += KAPPA * (1 - z[2]);
    out[3] += KAPPA * (1 - z[3]);
  }
  return out;
}

const dist = (z) => Math.hypot(Math.hypot(z[0], z[1]) - 1, z[2] - 1, z[3] - 1);

function rollout(sf, rng, z0) {
  const zs = [z0];
  let z = z0;
  for (let n = 0; n < N; n++) {
    const m = step(z, sf);
    z = [m[0] + TAU * rng.gauss(), m[1] + TAU * rng.gauss(), m[2] + TAU * rng.gauss(), m[3] + TAU * rng.gauss()];
    zs.push(z);
  }
  return zs;
}

export function buildData() {
  const z0 = [Math.cos(0.4), Math.sin(0.4), 1, 1];
  const data = [z0];
  for (let n = 0; n < N; n++) data.push(real(data[n]));
  // Same noise for both models, so the only difference is the pull-back term.
  const tf = rollout(false, makeRng(12), z0), sf = rollout(true, makeRng(12), z0);
  // One-step predictions from real contexts (frames 0..11) for the teacher-forcing act.
  const rng = makeRng(31);
  const one = data.slice(0, 12).map((z) => {
    const e = [rng.gauss(), rng.gauss(), rng.gauss(), rng.gauss()];
    const a = step(z, false), b = step(z, true);
    return { tf: a.map((v, i) => v + TAU * e[i]), sf: b.map((v, i) => v + TAU * e[i]) };
  });
  // Teacher-forced loss on real frames: E‖f(z) + τε − z'‖², identical for both models.
  let l1 = 0, l2 = 0;
  for (let n = 0; n < N; n++) {
    const a = step(data[n], false), b = step(data[n], true), y = data[n + 1];
    l1 += a.reduce((s, v, i) => s + (v - y[i]) ** 2, 0);
    l2 += b.reduce((s, v, i) => s + (v - y[i]) ** 2, 0);
  }
  const tfLoss = l1 / N + 4 * TAU * TAU, sfLoss = l2 / N + 4 * TAU * TAU;
  // Mean distance from the data per frame index, over many rollouts.
  const curve = { tf: new Float64Array(N + 1), sf: new Float64Array(N + 1) };
  for (let i = 0; i < NR; i++) {
    const a = rollout(false, makeRng(1000 + i), z0), b = rollout(true, makeRng(1000 + i), z0);
    for (let n = 0; n <= N; n++) { curve.tf[n] += dist(a[n]) / NR; curve.sf[n] += dist(b[n]) / NR; }
  }
  return { data, tf, sf, one, tfLoss, sfLoss, curve };
}

// ---------- Frame rendering ----------

const ballColor = (th, c, alpha = 1) => {
  // Brightness above 1 washes the ball out toward white, as overexposed AR video does.
  const base = th.name === 'dark' ? [123, 108, 255] : [91, 69, 224];
  const hot = th.name === 'dark' ? [255, 255, 255] : [255, 196, 120];
  const w = clamp((c - 1) / 0.8);
  return `rgba(${base.map((v, i) => Math.round(lerp(v, hot[i], w))).join(',')},${alpha})`;
};

function drawTile(ctx, th, x, y, size, z, a, { ghost = null, label = null, ring = true } = {}) {
  if (a <= 0) return;
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.name === 'dark' ? '#121827' : '#f1f4f9';
  ctx.strokeStyle = th.faint;
  ctx.lineWidth = 1;
  ctx.fillRect(x, y, size, size);
  ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
  ctx.beginPath(); ctx.rect(x, y, size, size); ctx.clip();
  const cx = x + size / 2, cy = y + size / 2, R = 0.3 * size;
  if (ring) {
    ctx.strokeStyle = th.faint; ctx.lineWidth = Math.max(1, size / 120); ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
  if (ghost) {
    ctx.strokeStyle = th.muted; ctx.lineWidth = Math.max(1.2, size / 110);
    ctx.beginPath(); ctx.arc(cx + R * ghost[0], cy - R * ghost[1], 0.09 * size, 0, 2 * Math.PI); ctx.stroke();
  }
  const bx = cx + R * z[0], by = cy - R * z[1], br = 0.09 * size * Math.max(z[2], 0.2), c = z[3];
  const glow = ctx.createRadialGradient(bx, by, 0, bx, by, br * (1.6 + 1.6 * clamp(c - 1, 0, 1.5)));
  glow.addColorStop(0, ballColor(th, c, th.name === 'dark' ? 0.55 : 0.4));
  glow.addColorStop(1, ballColor(th, c, 0));
  ctx.fillStyle = glow;
  ctx.beginPath(); ctx.arc(bx, by, br * 3.5, 0, 2 * Math.PI); ctx.fill();
  ctx.fillStyle = ballColor(th, c, clamp(0.5 + 0.5 * c));
  ctx.beginPath(); ctx.arc(bx, by, br, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
  if (label) text(ctx, label, x + 6, y + size + 18, { size: 14, color: th.muted, alpha: a, font: 'mono' });
}

// A row: one large current frame and a 2 × 6 strip of every 8th frame so far.
const ROWS = [{ y: 130, name: 'tf' }, { y: 492, name: 'sf' }];
const BIG = { x: 660, size: 270 }, THUMB = { x: 960, size: 96, gap: 8 };

function drawRow(ctx, th, y, zs, n, a, title, color) {
  text(ctx, title, BIG.x, y - 20, { size: 22, weight: 700, color, alpha: a });
  const i = Math.min(Math.floor(n), N);
  drawTile(ctx, th, BIG.x, y, BIG.size, zs[i], a, { label: `frame ${i}` });
  for (let q = 0; q < 12; q++) {
    const f = 8 * (q + 1), col = q % 6, rr = Math.floor(q / 6);
    const x = THUMB.x + col * (THUMB.size + THUMB.gap), yy = y + rr * (THUMB.size + 34);
    const shown = n >= f - 0.5;
    if (!shown) {
      ctx.save(); ctx.globalAlpha = a * 0.5; ctx.strokeStyle = th.faint; ctx.setLineDash([3, 4]);
      ctx.strokeRect(x + 0.5, yy + 0.5, THUMB.size - 1, THUMB.size - 1); ctx.restore();
      continue;
    }
    drawTile(ctx, th, x, yy, THUMB.size, zs[f], a * clamp((n - f + 0.5) * 2), { label: `${f}` });
  }
}

function drawDataRow(ctx, th, st, t, a, clean) {
  const n = N * seg(t, ...T.dataPlay);
  // In paper stills the panel title already says this.
  drawRow(ctx, th, ROWS[0].y, st.data, n, a, clean ? '' : 'Real video', th.fg);
  const b = a * seg(t, 3, 4);
  if (b > 0) {
    const y = ROWS[1].y;
    text(ctx, 'One frame is described by four numbers', BIG.x, y + 10, { size: 20, weight: 600, color: th.fg, alpha: b });
    const rows = [
      ['position p', 'moves 15° along the ring per frame'],
      ['size s', 'stays at 1'],
      ['brightness c', 'stays at 1'],
    ];
    rows.forEach(([k, v], i) => {
      text(ctx, k, BIG.x, y + 56 + 36 * i, { size: 19, color: th.accent, alpha: b, font: 'mono' });
      text(ctx, v, BIG.x + 190, y + 56 + 36 * i, { size: 19, color: th.fg, alpha: b });
    });
    text(ctx, 'Distance from the data = ‖(|p| − 1, s − 1, c − 1)‖', BIG.x, y + 200, { size: 19, color: th.muted, alpha: b, font: 'serif' });
  }
}

// Teacher forcing: each model sees real frame n and predicts frame n + 1. Ghost circle = the real next frame.
function drawOneStep(ctx, th, st, t, a) {
  const k = Math.min(Math.floor(12 * seg(t, ...T.tfStep)), 11);
  for (const [r, key, title, color] of [[ROWS[0], 'tf', 'Teacher-forced model', th.fg], [ROWS[1], 'sf', 'Self-forced model', th.accent]]) {
    const y = r.y;
    text(ctx, title, BIG.x, y - 20, { size: 22, weight: 700, color, alpha: a });
    drawTile(ctx, th, BIG.x, y, BIG.size, st.data[k], a, { label: `real frame ${k}` });
    text(ctx, '→', BIG.x + BIG.size + 22, y + BIG.size / 2 + 12, { size: 40, color: th.muted, alpha: a });
    const pred = st.one[k][key];
    drawTile(ctx, th, BIG.x + BIG.size + 80, y, BIG.size, pred, a, { ghost: st.data[k + 1], label: `predicted frame ${k + 1}` });
    const e = Math.hypot(...pred.map((v, i) => v - st.data[k + 1][i]));
    text(ctx, 'ONE-STEP ERROR', BIG.x + 2 * BIG.size + 120, y + 110, { size: 15, weight: 600, color: th.muted, alpha: a, spacing: 1.5 });
    text(ctx, e.toFixed(3), BIG.x + 2 * BIG.size + 120, y + 166, { size: 52, weight: 700, color, alpha: a, font: 'mono' });
  }
}

function drawRollouts(ctx, th, st, t, a) {
  const n = N * ease.inOut(seg(t, ...T.rollRun));
  drawRow(ctx, th, ROWS[0].y, st.tf, n, a, 'Teacher-forced model, own rollout', th.fg);
  drawRow(ctx, th, ROWS[1].y, st.sf, n, a, 'Self-forced model, own rollout', th.accent);
}

function drawChart(ctx, th, st, t, a) {
  const L = 760, R = 1300, top = 150, bot = 700, grow = ease.inOut(seg(t, ...T.chartDraw));
  const ymax = 1.4, X = (n) => L + (n / N) * (R - L), Y = (v) => bot - (v / ymax) * (bot - top);
  ctx.save();
  ctx.globalAlpha = a;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L, top); ctx.lineTo(L, bot); ctx.lineTo(R, bot); ctx.stroke();
  ctx.restore();
  text(ctx, 'mean distance from the data', L, top - 26, { size: 18, color: th.muted, alpha: a });
  for (const v of [0, 0.4, 0.8, 1.2]) text(ctx, v.toFixed(1), L - 14, Y(v) + 6, { size: 16, color: th.muted, align: 'right', alpha: a });
  for (const n of [0, 24, 48, 72, 96]) text(ctx, String(n), X(n), bot + 28, { size: 16, color: th.muted, align: 'center', alpha: a });
  text(ctx, 'frame index', (L + R) / 2, bot + 62, { size: 18, color: th.muted, align: 'center', alpha: a });
  const series = [
    { c: st.curve.tf, color: th.fg, dash: [9, 7], name: 'teacher forcing' },
    { c: st.curve.sf, color: th.accent, dash: [], name: 'self forcing' },
  ];
  for (const s of series) {
    const m = Math.floor(grow * N);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.setLineDash(s.dash);
    ctx.beginPath();
    for (let n = 0; n <= m; n++) (n ? ctx.lineTo : ctx.moveTo).call(ctx, X(n), Y(s.c[n]));
    ctx.stroke();
    ctx.restore();
    const la = a * seg(grow, 0.95, 1);
    text(ctx, `${s.name}: ${s.c[N].toFixed(2)}`, R + 12, Y(s.c[N]) + 6, { size: 18, weight: 700, color: s.color, alpha: la });
  }
}

// ---------- Text panel ----------

const COPY = [
  { span: T.data, act: 'Setup',
    body: 'An autoregressive video model generates one frame at a time, each conditioned on the frames before it. In this toy video a ball moves around a ring at constant speed, size and brightness.',
    eq: 'x_{n+1} ~ p_θ(x_{n+1} | x_{≤n})' },
  { span: T.tf, act: 'Act I · Teacher forcing',
    body: 'Teacher forcing trains each step on real context frames. Both models below make the same small error on every real frame: slightly too fast, too far out, too large and too bright. The self-forced model also pulls drifted frames back toward the data, but that term is zero on real frames, so their teacher-forced losses are equal.',
    eq: 'L_TF = E ‖f_θ(x_n) − x_{n+1}‖²,  x_n real' },
  { span: T.roll, act: 'Act II · Rolling out',
    body: 'At inference the model conditions on its own frames, which teacher forcing never showed it. The teacher-forced model repeats its error on contexts that are already wrong, so the ball drifts outward, grows and saturates. The self-forced model sees the same noise and stays near the data.',
    eq: 'x̂_{n+1} = f_θ(x̂_n) + τ ε' },
  { span: T.chart, act: 'Act III · Error growth',
    body: 'Averaged over 400 rollouts, the teacher-forced error grows with every frame, while the self-forced error levels off near 0.06, close to the noise of a single frame. Self Forcing trains on the model\'s own rollouts with a video-level loss, so drift is visible to the loss and is corrected.',
    eq: 'L_SF = D( p_θ(x̂_{1:N}), p_data(x_{1:N}) )' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §5 DISTILLATION', 'Self Forcing', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 24, color: th.fg, alpha: a, font: 'serif' });
  }
  const la = window_(t, T.tf[0], T.tf[1], 0.4);
  if (la > 0) {
    text(ctx, 'TEACHER-FORCED LOSS ON REAL FRAMES', x, 690, { size: 15, weight: 600, color: th.muted, alpha: la, spacing: 1.5 });
    text(ctx, `TF model ${st.tfLoss.toFixed(4)}   SF model ${st.sfLoss.toFixed(4)}`, x, 726, { size: 22, weight: 700, color: th.fg, alpha: la, font: 'mono' });
  }
  const ra = window_(t, T.roll[0], T.roll[1], 0.4);
  if (ra > 0) {
    const n = Math.min(Math.floor(N * ease.inOut(seg(t, ...T.rollRun))), N);
    text(ctx, `frame ${n} of ${N}`, x, 650, { size: 18, weight: 600, color: th.muted, alpha: ra });
    text(ctx, 'DISTANCE FROM THE DATA', x, 692, { size: 15, weight: 600, color: th.muted, alpha: ra, spacing: 1.5 });
    text(ctx, dist(st.tf[n]).toFixed(2), x, 756, { size: 56, weight: 700, color: th.fg, alpha: ra, font: 'mono' });
    text(ctx, dist(st.sf[n]).toFixed(2), x + 220, 756, { size: 56, weight: 700, color: th.accent, alpha: ra, font: 'mono' });
    text(ctx, 'teacher forcing', x, 784, { size: 15, color: th.muted, alpha: ra });
    text(ctx, 'self forcing', x + 220, 784, { size: 15, color: th.muted, alpha: ra });
  }
  paragraph(ctx, `Both per-frame models are written down, not trained. Shared one-step error: speed +${(ERR.w * 100).toFixed(0)}%, radius +${(ERR.r * 100).toFixed(1)}%, size +${(ERR.s * 100).toFixed(1)}%, brightness +${ERR.c}. Self-forced pull-back κ = ${KAPPA}. Sampling noise τ = ${TAU} per coordinate.`, x, 818, w, { size: 14, color: th.muted, lineHeight: 20 });
}
