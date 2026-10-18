// A8 · Flow-GRPO: group rollouts, group-relative advantages and the policy update.
// The policy is the spiral mixture with trainable component logits ℓ (means and widths fixed). Rollouts use the exact
// SDE dx = b_θ(x, t) dt + g(t) dW with b = u + ½g²∇log p_t. For a mixture, b = Σ γ_k b_k, and the gradient of each
// Euler–Maruyama step's log-likelihood has the closed form ∂ log p / ∂ℓ_j = (√h / g) ε · γ_j (b_j − b), where ε is the
// step's standard normal noise. So the GRPO update below is the real policy gradient, not an illustration of one.
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1000, CY = 455, S = 150;
const plot = makePlot(CX, CY, S);
const px = plot.px;

const { means: MEANS, us: US, std: STD } = spiralMixture();
const K = MEANS.length, S2 = STD * STD;
const STEPS = 20;            // SDE steps per rollout
const G = 16;                // group size
const ITERS = 80;            // policy updates
const LR = 1;                // step size on the logits
const BETA = 0.3;            // KL weight in the regularized run
const RHO = 0.5;             // reward width
const NS = 360;              // samples drawn from the current policy in the training acts
const g = (t) => 1.15 * Math.sqrt(1 - t);
const C = MEANS[Math.round(0.55 * (K - 1))];
const reward = (x, y) => Math.exp(-((x - C[0]) ** 2 + (y - C[1]) ** 2) / (2 * RHO * RHO));
// E[r(x)] for x ~ N(μ, σ²I), in closed form.
const compReward = (m) => (RHO * RHO / (RHO * RHO + S2)) * Math.exp(-((m[0] - C[0]) ** 2 + (m[1] - C[1]) ** 2) / (2 * (RHO * RHO + S2)));
const CR = MEANS.map(compReward);

const T = {
  setup: [0, 5], reward: [1.4, 3],
  group: [5, 17.4], groupRun: [5.6, 11.6], groupScore: [11.8, 13], groupAdv: [13.4, 15.2],
  update: [17.4, 27], updateShow: [18.4, 20.4],
  train: [27, 40.4], trainRun: [28, 38.6],
  kl: [40.4, 53.8], klRun: [41.4, 52],
  chart: [53.8, 66], chartDraw: [54.8, 59.6],
};

const STILLS = [
  { t: 16.8, name: 'group', label: 'A group of 16 rollouts' },
  { t: 26.4, name: 'update', label: 'One policy update' },
  { t: 39.8, name: 'no-kl', label: 'GRPO without KL' },
  { t: 53.2, name: 'kl', label: `GRPO with KL (β = ${BETA})` },
  { t: 65.4, name: 'curves', label: 'Reward and diversity' },
];

const POS = (th) => (th.name === 'dark' ? '#9dff8a' : '#15803d');
const NEG = (th) => (th.name === 'dark' ? '#ff6b6b' : '#dc2626');

export default {
  id: 'flow-grpo',
  title: 'Flow-GRPO: group rollouts and relative advantages',
  duration: 66,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.group[0], label: 'I · A group' },
    { t: T.update[0], label: 'II · The update' },
    { t: T.train[0], label: 'III · Training' },
    { t: T.kl[0], label: 'IV · With KL' },
    { t: T.chart[0], label: 'V · Curves' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 990, h: 900 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    const chartA = seg(t, T.chart[0], T.chart[0] + 0.8);
    const dim = 1 - 0.85 * chartA;
    const rA = (env.clean ? 1 : ease.out(seg(t, ...T.reward))) * dim;
    drawReward(ctx, th, rA);

    // Policy weights: one bead per component, area growing with its weight.
    const { w, samples, sampA } = policyAt(st, t);
    drawBeads(ctx, th, w, (env.clean ? 1 : seg(t, 0.3, 1.6)) * dim);
    if (sampA > 0) for (const [x, y] of samples) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, rewardMix(th, reward(x, y), 1), sampA * dim, 2.3); }

    const gA = window_(t, T.group[0], T.update[1], 0.5);
    if (gA > 0) drawGroup(ctx, th, st, t, gA);
    const uA = window_(t, T.update[0], T.update[1], 0.5);
    if (uA > 0) drawDelta(ctx, th, st, t, uA);
    if (chartA > 0) drawChart(ctx, th, st, t, chartA);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 70, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Exact policy and GRPO ----------

// Per-component drifts b_k, responsibilities γ_k and the mixture drift b at (x, t) for logits l.
function drift(l, x, y, t, gt) {
  const a = t, sg = 1 - t, v = a * a * S2 + sg * sg;
  const gam = new Float64Array(K), bx = new Float64Array(K), by = new Float64Array(K);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) {
    const rx = x - a * MEANS[k][0], ry = y - a * MEANS[k][1];
    gam[k] = l[k] - (rx * rx + ry * ry) / (2 * v);
    if (gam[k] > mx) mx = gam[k];
    // b_k = E[x1|x,k] − E[x0|x,k] + ½g² ∇log N_k   (α = t, σ = 1 − t)
    bx[k] = MEANS[k][0] + ((a * S2) / v) * rx - (sg / v) * rx - (0.5 * gt * gt * rx) / v;
    by[k] = MEANS[k][1] + ((a * S2) / v) * ry - (sg / v) * ry - (0.5 * gt * gt * ry) / v;
  }
  let z = 0, Bx = 0, By = 0;
  for (let k = 0; k < K; k++) z += (gam[k] = Math.exp(gam[k] - mx));
  for (let k = 0; k < K; k++) { gam[k] /= z; Bx += gam[k] * bx[k]; By += gam[k] * by[k]; }
  return { gam, bx, by, Bx, By };
}

const ZERO = new Float64Array(K);

// One Euler–Maruyama rollout. Accumulates ∇ℓ log p(trajectory) into `grad` and, with beta, the gradient of the
// per-step KL to the reference policy, Σ h ‖b_θ − b_ref‖² / 2g², into `kgrad`.
function rollout(l, rng, grad, kgrad, beta, keep) {
  let x = rng.gauss(), y = rng.gauss();
  const h = 1 / STEPS, sq = Math.sqrt(h), path = keep ? [[x, y]] : null;
  for (let s = 0; s < STEPS; s++) {
    const t = s * h, gt = g(t), c = drift(l, x, y, t, gt), ex = rng.gauss(), ey = rng.gauss();
    for (let k = 0; k < K; k++) grad[k] += (sq / gt) * c.gam[k] * (ex * (c.bx[k] - c.Bx) + ey * (c.by[k] - c.By));
    if (beta) {
      const r0 = drift(ZERO, x, y, t, gt), dx = c.Bx - r0.Bx, dy = c.By - r0.By;
      for (let k = 0; k < K; k++) kgrad[k] += (h / (gt * gt)) * c.gam[k] * ((c.bx[k] - c.Bx) * dx + (c.by[k] - c.By) * dy);
    }
    x += h * c.Bx + gt * sq * ex;
    y += h * c.By + gt * sq * ey;
    if (path) path.push([x, y]);
  }
  return { x: [x, y], r: reward(x, y), path };
}

function stats(l) {
  const mx = Math.max(...l), w = Array.from(l, (v) => Math.exp(v - mx)), z = w.reduce((a, b) => a + b);
  let er = 0, ent = 0;
  for (let k = 0; k < K; k++) { const p = (w[k] /= z); er += p * CR[k]; if (p > 0) ent -= p * Math.log(p); }
  return { w, er, modes: Math.exp(ent) };
}

export function train(beta, seed = 10) {
  const rng = makeRng(seed), l = new Float64Array(K), hist = [stats(l)];
  let first = null;
  for (let it = 0; it < ITERS; it++) {
    const grads = [], outs = [], kg = new Float64Array(K);
    for (let i = 0; i < G; i++) { const gr = new Float64Array(K); outs.push(rollout(l, rng, gr, kg, beta, it === 0)); grads.push(gr); }
    const rs = outs.map((o) => o.r), m = rs.reduce((a, b) => a + b) / G;
    const sd = Math.sqrt(rs.reduce((a, b) => a + (b - m) ** 2, 0) / G);
    const adv = rs.map((r) => (r - m) / (sd + 1e-4));
    const delta = new Float64Array(K);
    for (let i = 0; i < G; i++) for (let k = 0; k < K; k++) delta[k] += (LR * adv[i] * grads[i][k]) / G;
    for (let k = 0; k < K; k++) delta[k] -= (LR * beta * kg[k]) / G;
    for (let k = 0; k < K; k++) l[k] += delta[k];
    if (it === 0) first = { outs, adv, mean: m, delta };
    hist.push(stats(l));
  }
  return { hist, first };
}

export function buildData() {
  const st = { plain: train(0), kl: train(BETA) };
  // Fixed draws for showing the current policy: component by inverse CDF along the spiral, then a fixed offset.
  const rng = makeRng(77);
  st.draws = Array.from({ length: NS }, () => [(rng.uniform() + 0.0001) * 0.9998, rng.gauss(), rng.gauss()]);
  return st;
}

function samplePolicy(st, w) {
  const cum = [];
  let c = 0;
  for (const v of w) cum.push((c += v));
  return st.draws.map(([u, gx, gy]) => {
    let k = cum.findIndex((v) => u <= v);
    if (k < 0) k = K - 1;
    return [MEANS[k][0] + STD * gx, MEANS[k][1] + STD * gy];
  });
}

// The run and fractional iteration shown at time t.
function iterAt(t) {
  if (t < T.kl[0]) return { run: 'plain', f: ITERS * ease.inOut(seg(t, ...T.trainRun)) };
  return { run: 'kl', f: ITERS * ease.inOut(seg(t, ...T.klRun)) };
}

function policyAt(st, t) {
  const { run, f } = iterAt(t), hist = st[run].hist;
  const i = Math.min(Math.floor(f), ITERS - 1), u = f - i;
  const w = hist[i].w.map((v, k) => lerp(v, hist[i + 1].w[k], u));
  const sampA = Math.max(window_(t, T.train[0], T.train[1], 0.5), window_(t, T.kl[0], T.kl[1], 0.5));
  return { w, samples: sampA > 0 ? samplePolicy(st, w) : [], sampA, er: lerp(hist[i].er, hist[i + 1].er, u), modes: lerp(hist[i].modes, hist[i + 1].modes, u), f };
}

// ---------- Drawing ----------

function drawReward(ctx, th, a) {
  if (a <= 0) return;
  const [x, y] = px(...C), r = RHO * S, col = POS(th);
  ctx.save();
  ctx.globalAlpha = a;
  const grad = ctx.createRadialGradient(x, y, 0, x, y, 2 * r);
  grad.addColorStop(0, th.name === 'dark' ? 'rgba(157,255,138,0.16)' : 'rgba(21,128,61,0.12)');
  grad.addColorStop(1, th.name === 'dark' ? 'rgba(157,255,138,0)' : 'rgba(21,128,61,0)');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(x, y, 2 * r, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();
  text(ctx, 'reward r(x)', x, y - r - 14, { size: 20, weight: 600, color: col, alpha: a, align: 'center', font: 'serif' });
}

function drawBeads(ctx, th, w, a) {
  if (a <= 0) return;
  for (let k = 0; k < K; k++) {
    const [x, y] = px(...MEANS[k]), r = 2.4 * Math.pow(w[k] * K, 0.38);
    ctx.save();
    ctx.globalAlpha = a * 0.85;
    ctx.fillStyle = rampColor(th, US[k], 0.75);
    ctx.beginPath(); ctx.arc(x, y, Math.max(r, 0.6), 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
}

function rewardMix(th, r, alpha) {
  const a = hexRgb(th.name === 'dark' ? '#5c6680' : '#9aa3b5'), b = hexRgb(POS(th));
  const w = clamp(r * 1.3);
  return `rgba(${a.map((v, i) => Math.round(lerp(v, b[i], w))).join(',')},${alpha})`;
}
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function advColor(th, A, alpha) {
  const c = hexRgb(A >= 0 ? POS(th) : NEG(th));
  return `rgba(${c.join(',')},${alpha})`;
}

// Bars of the group: reward and advantage per rollout, sorted by reward.
const BAR = { x: 1430, y: 150, row: 34, scale: 52 };

function drawGroup(ctx, th, st, t, a) {
  const { outs, adv } = st.plain.first;
  const sf = STEPS * ease.inOut(seg(t, ...T.groupRun)), score = seg(t, ...T.groupScore), advA = seg(t, ...T.groupAdv);
  const fade = 1 - 0.6 * seg(t, T.update[0], T.update[0] + 0.8);
  for (let i = 0; i < G; i++) {
    const p = outs[i].path, n = Math.min(Math.ceil(sf), STEPS);
    if (n < 1) continue;
    const base = rewardMix(th, score * outs[i].r, 1), A = adv[i];
    ctx.save();
    ctx.globalAlpha = a * fade * (advA > 0 ? lerp(0.85, 0.25 + 0.6 * clamp(Math.abs(A) / 1.6), advA) : 0.85);
    ctx.strokeStyle = advA > 0.5 ? advColor(th, A, 1) : base;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(...px(...p[0]));
    for (let s = 1; s <= n; s++) {
      const w = s === n ? sf - (s - 1) : 1;
      ctx.lineTo(...px(lerp(p[s - 1][0], p[s][0], w), lerp(p[s - 1][1], p[s][1], w)));
    }
    ctx.stroke();
    ctx.restore();
    const k = Math.min(Math.floor(sf), STEPS - 1), u = sf - k;
    const [x, y] = px(lerp(p[k][0], p[k + 1][0], u), lerp(p[k][1], p[k + 1][1], u));
    dot(ctx, th, x, y, advA > 0.5 ? advColor(th, A, 1) : base, a * fade, 4);
  }

  // Bar chart of the group.
  const ba = a * seg(t, T.groupScore[0], T.groupScore[0] + 0.6) * fade;
  if (ba <= 0) return;
  const order = [...Array(G).keys()].sort((i, j) => outs[j].r - outs[i].r);
  text(ctx, 'r', BAR.x - 74, BAR.y - 22, { size: 17, color: th.muted, alpha: ba, align: 'center', font: 'serif' });
  text(ctx, 'advantage A', BAR.x + 30, BAR.y - 22, { size: 17, color: th.muted, alpha: ba * advA, align: 'center' });
  ctx.save();
  ctx.strokeStyle = th.muted; ctx.globalAlpha = ba * advA * 0.6; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(BAR.x, BAR.y - 8); ctx.lineTo(BAR.x, BAR.y + G * BAR.row - 14); ctx.stroke();
  ctx.restore();
  order.forEach((i, q) => {
    const y = BAR.y + q * BAR.row;
    text(ctx, outs[i].r.toFixed(2), BAR.x - 74, y + 6, { size: 16, color: th.fg, alpha: ba, align: 'center', font: 'mono' });
    const A = adv[i], len = A * BAR.scale * ease.out(advA);
    ctx.save();
    ctx.globalAlpha = ba;
    ctx.fillStyle = advColor(th, A, 0.85);
    ctx.fillRect(Math.min(BAR.x, BAR.x + len), y - 8, Math.abs(len), 16);
    ctx.restore();
  });
}

// The first update: Δℓ per component, shown as a ring that is green where the logit rises and red where it falls.
function drawDelta(ctx, th, st, t, a) {
  const { delta } = st.plain.first, show = a * ease.out(seg(t, ...T.updateShow));
  const mx = Math.max(...delta.map(Math.abs));
  for (let k = 0; k < K; k++) {
    const d = delta[k] / mx, [x, y] = px(...MEANS[k]);
    if (Math.abs(d) < 0.03) continue;
    ctx.save();
    ctx.globalAlpha = show * (0.25 + 0.75 * Math.abs(d));
    ctx.strokeStyle = advColor(th, d, 1);
    ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.arc(x, y, 4 + 9 * Math.abs(d), 0, 2 * Math.PI); ctx.stroke();
    ctx.restore();
  }
}

function drawChart(ctx, th, st, t, a) {
  const L = CX - 290, R = CX + 400, grow = ease.inOut(seg(t, ...T.chartDraw));
  ctx.save();
  ctx.globalAlpha = a * 0.92;
  ctx.fillStyle = th.bg;
  ctx.fillRect(L - 100, 50, R - L + 160, H - 70);
  ctx.restore();
  const panels = [
    { top: 110, bot: 400, key: 'er', lab: 'expected reward', ticks: [0, 0.5, 1], Y: (v, tp, bt) => bt - v * (bt - tp) },
    { top: 500, bot: 790, key: 'modes', lab: 'effective number of modes', ticks: [1, 10, 100], Y: (v, tp, bt) => bt - (Math.log10(v) / Math.log10(K)) * (bt - tp) },
  ];
  const X = (i) => L + (i / ITERS) * (R - L);
  for (const p of panels) {
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(L, p.top); ctx.lineTo(L, p.bot); ctx.lineTo(R, p.bot); ctx.stroke();
    ctx.restore();
    text(ctx, p.lab, L, p.top - 22, { size: 18, color: th.muted, alpha: a });
    for (const v of p.ticks) text(ctx, String(v), L - 14, p.Y(v, p.top, p.bot) + 6, { size: 16, color: th.muted, align: 'right', alpha: a });
    for (const i of [0, 20, 40, 60, 80]) text(ctx, String(i), X(i), p.bot + 26, { size: 16, color: th.muted, align: 'center', alpha: a });
    const series = [
      { h: st.plain.hist, color: th.fg, dash: [9, 7], name: 'no KL' },
      { h: st.kl.hist, color: th.accent, dash: [], name: `KL, β = ${BETA}` },
    ];
    for (const s of series) {
      const n = grow * ITERS;
      ctx.save();
      ctx.globalAlpha = a;
      ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.setLineDash(s.dash); ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i <= Math.floor(n); i++) {
        const y = p.Y(s.h[i][p.key], p.top, p.bot);
        if (i === 0) ctx.moveTo(X(i), y); else ctx.lineTo(X(i), y);
      }
      ctx.stroke();
      ctx.restore();
      const la = a * seg(grow, 0.95, 1), v = s.h[ITERS][p.key];
      text(ctx, `${s.name}: ${p.key === 'er' ? v.toFixed(2) : v.toFixed(1)}`, R + 10, p.Y(v, p.top, p.bot) + (s.color === th.fg ? -10 : 22), { size: 18, weight: 700, color: s.color, alpha: la });
    }
  }
  text(ctx, 'policy updates (16 rollouts each)', (L + R) / 2, 790 + 60, { size: 18, color: th.muted, align: 'center', alpha: a });
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'The policy is the spiral mixture from A1 with one trainable logit per component. A reward model prefers a region of the outer arm. Flow-GRPO samples with an SDE instead of the ODE, so each step has a Gaussian likelihood that a policy gradient can use.',
    eq: 'p_θ = Σₖ softmax(ℓ)ₖ N(μₖ, σ²I)' },
  { span: [T.group[0], T.groupAdv[0]], act: 'Act I · A group of rollouts',
    body: 'For one prompt, the policy draws a group of 16 SDE rollouts and the reward model scores each final sample.',
    eq: 'x_{t+h} = x_t + h b_θ(x_t, t) + g(t)√h ε' },
  { span: [T.groupAdv[0], T.group[1]], act: 'Act I · A group of rollouts',
    body: 'GRPO needs no value network. Each rollout is compared with the rest of its group: the advantage is its reward minus the group mean, divided by the group standard deviation. Green rollouts did better than their siblings, red ones worse.',
    eq: 'Aᵢ = (rᵢ − mean r) / std r' },
  { span: T.update, act: 'Act II · The policy update',
    body: 'Every SDE step of every rollout contributes the gradient of its log-likelihood, weighted by the rollout\'s advantage. Rings show the resulting change in each logit: green components gain weight, red ones lose it. For this policy the step gradient is exact.',
    eq: 'Δℓ ∝ Σᵢ Aᵢ Σₜ ∇ℓ log p_θ(x_{t+h} | x_t)' },
  { span: T.train, act: 'Act III · Training without KL',
    body: 'Repeating the update 80 times moves the weight onto the components under the reward. Without a penalty nothing stops it, and the policy collapses onto a single mode. Dots are samples from the current policy.',
    eq: '' },
  { span: T.kl, act: `Act IV · Training with KL`,
    body: 'Flow-GRPO adds a KL penalty to the reference model, which for two SDEs is a sum over steps of the squared drift difference. The policy still moves toward the reward but keeps several modes under it.',
    eq: 'KL = Σₜ h ‖b_θ − b_ref‖² / 2g(t)²' },
  { span: T.chart, act: 'Act V · Reward and diversity',
    body: 'Without KL the expected reward approaches its maximum while the effective number of modes falls to about one. With β = 0.3 the reward ends near 0.73 and the policy keeps about ten modes. A7 shows where this trade-off leads in the limit.',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §4 POST-TRAINING', 'Flow-GRPO', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 24, color: th.fg, alpha: a, font: 'serif' });
  }

  const ga = window_(t, T.group[0], T.update[1], 0.4);
  if (ga > 0) {
    const sc = seg(t, ...T.groupScore);
    text(ctx, `group of G = ${G} · ${STEPS} SDE steps each`, x, 640, { size: 18, weight: 600, color: th.muted, alpha: ga });
    if (sc > 0) text(ctx, `group mean reward ${st.plain.first.mean.toFixed(2)}`, x, 670, { size: 18, weight: 600, color: th.muted, alpha: ga * sc });
  }
  for (const [span, run] of [[T.train, st.plain], [T.kl, st.kl]]) {
    const a = window_(t, span[0], span[1], 0.4);
    if (a <= 0) continue;
    const p = policyAt(st, t);
    text(ctx, `update ${Math.round(p.f)} of ${ITERS}`, x, 640, { size: 18, weight: 600, color: th.muted, alpha: a });
    text(ctx, 'EXPECTED REWARD', x, 690, { size: 15, weight: 600, color: th.muted, alpha: a, spacing: 1.5 });
    text(ctx, p.er.toFixed(2), x, 756, { size: 60, weight: 700, color: POS(th), alpha: a, font: 'mono' });
    text(ctx, 'EFFECTIVE MODES', x + 240, 690, { size: 15, weight: 600, color: th.muted, alpha: a, spacing: 1.5 });
    text(ctx, p.modes.toFixed(1), x + 240, 756, { size: 60, weight: 700, color: th.fg, alpha: a, font: 'mono' });
  }

  paragraph(ctx, `Exact mixture policy, ${K} trainable logits. Euler–Maruyama with g(t) = 1.15√(1−t), one on-policy update per group (ratio 1, so clipping is inactive), step size ${LR}. Effective modes = exp(entropy of the weights).`, x, 806, w, { size: 14, color: th.muted, lineHeight: 20 });
}
