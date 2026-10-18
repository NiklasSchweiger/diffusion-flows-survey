// A1 · Flow matching: conditional paths -> marginal ODE -> SDE with the same marginals.
// Everything is exact for a Gaussian-mixture target (see ../lib/gmm.js); nothing is learned.
import { trajAt } from '../lib/gmm.js';
import { spiralMixture, makePlot, drawGrid as grid, renderGlow, dot as glowDot, panelHeader } from '../lib/scene.js';
import { makeRng } from '../lib/rng.js';
import { seg, ease, window_, rampColor, text, paragraph, arrow, lerp } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1050, CY = 455, S = 150; // plot centre (px) and px per world unit
const plot = makePlot(CX, CY, S);
const px = plot.px;

const N = 900;          // particles
const N_TRAIL = 380;    // particles that also draw their path (ODE)
const N_TRAIL_SDE = 200; // fewer for the SDE: jittery paths saturate quickly
const N_LINES = 360;    // conditional lines shown in Act I
const ODE_STEPS = 240, SDE_STEPS = 320;
const P_HI = [-0.7, -0.9], R_HI = 0.24; // highlighted point for "the average of many slopes"

const STILLS = [
  { t: 9.6, name: 'conditional', label: 'Conditional paths' },
  { t: 22.4, name: 'ode', label: 'Marginal ODE' },
  { t: 31.4, name: 'sde', label: 'SDE, same marginals' },
];

// Timeline (seconds)
const T = {
  intro: [0, 2.6],
  a1: [2.6, 13.4], a1Lines: 2.6, a1Go: [3.6, 6.0], a1Hi: [6.0, 10.2], a1Go2: [10.2, 12.2],
  a2: [13.4, 23.2], a2Go: [14.6, 21.0],
  a3: [23.2, 32.0], a3Go: [24.2, 30.4],
};

export default {
  id: 'flow-matching',
  title: 'Flow matching: conditional paths, marginal ODE and SDE',
  duration: 32,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.a1[0], label: 'I · Conditional paths' },
    { t: T.a1Hi[0], label: 'Averaging slopes' },
    { t: T.a2[0], label: 'II · Marginal ODE' },
    { t: T.a3[0], label: 'III · SDE' },
  ],
  stills: STILLS,
  crop: { x: CX - 430, y: CY - 430, w: 860, h: 860 },

  setup(env) {
    const mix = spiralMixture();
    const { gmm, means, us } = mix;
    const rng = makeRng(7);

    const x0 = [], x1 = [], uCond = [];
    for (let i = 0; i < N; i++) {
      x0.push([rng.gauss(), rng.gauss()]);
      const s = gmm.sample(rng);
      x1.push([s.x, s.y]);
      uCond.push(us[s.k]);
    }
    const ode = x0.map(([x, y]) => gmm.integrateODE(x, y, ODE_STEPS));
    const g = (t) => 1.15 * Math.sqrt(Math.max(1 - t, 0));
    const sdeRng = makeRng(11);
    const sde = x0.map(([x, y]) => gmm.integrateSDE(x, y, SDE_STEPS, g, sdeRng));
    const dest = (tr) => { const [x, y] = trajAt(tr, 1); return us[gmm.argmaxComponent(x, y, 1)]; };
    const uOde = ode.map(dest), uSde = sde.map(dest);

    // Pairs whose straight path passes near P_HI at t = 0.5.
    const near = [];
    for (let i = 0; i < N; i++) {
      const mx = 0.5 * (x0[i][0] + x1[i][0]), my = 0.5 * (x0[i][1] + x1[i][1]);
      const d = Math.hypot(mx - P_HI[0], my - P_HI[1]);
      if (d < R_HI) near.push({ i, d });
    }
    near.sort((a, b) => a.d - b.d);
    const hiIdx = near.slice(0, 16).map((n) => n.i);
    const uHi = gmm.velocity(P_HI[0], P_HI[1], 0.5);

    const st = { mix, gmm, means, us, x0, x1, uCond, ode, sde, uOde, uSde, hiIdx, hiSet: new Set(hiIdx), uHi };
    st.bg = renderGlow(env, plot, mix);
    return st;
  },

  onTheme(env, st) { st.bg = renderGlow(env, plot, st.mix); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    grid(ctx, th, plot, W, H);

    // Faint target density, always present so the eye knows where particles must go.
    const targetA = env.clean ? 1 : seg(t, 0.4, 2.0);
    ctx.globalAlpha = targetA;
    ctx.drawImage(st.bg, 0, 0, W, H);
    ctx.globalAlpha = 1;

    if (t < T.a1[1]) drawAct1(ctx, t, env, st);
    else if (t < T.a2[1]) drawAct2(ctx, t, env, st);
    else drawAct3(ctx, t, env, st);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else drawStillLabel(ctx, t, env);
  },
};

// ---------- Acts ----------

function drawAct1(ctx, t, env, st) {
  const th = env.theme;
  // tau: interpolation time shown in the plot.
  const tau = 0.5 * ease.inOut(seg(t, ...T.a1Go)) + 0.5 * ease.inOut(seg(t, ...T.a1Go2));
  const fadeOut = 1 - seg(t, T.a1[1] - 0.7, T.a1[1]);
  const hiIn = seg(t, T.a1Hi[0], T.a1Hi[0] + 0.6) * (1 - seg(t, T.a1Hi[1] - 0.4, T.a1Hi[1]));
  const dim = 1 - 0.75 * hiIn;

  // Straight conditional lines.
  const linesA = seg(t, T.a1Lines, T.a1Lines + 1.0) * fadeOut * dim;
  if (linesA > 0) {
    ctx.save();
    ctx.globalCompositeOperation = th.blend;
    ctx.lineWidth = 1;
    for (let i = 0; i < N_LINES; i++) {
      const [ax, ay] = px(...st.x0[i]), [bx, by] = px(...st.x1[i]);
      ctx.strokeStyle = rampColor(th, st.uCond[i], 0.16 * linesA);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    ctx.restore();
  }

  // Particles on their straight lines.
  const pA = seg(t, 0.2, 1.2) * fadeOut;
  const colorA = seg(t, T.a1Lines, T.a1Lines + 1.0);
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  for (let i = 0; i < N; i++) {
    const x = lerp(st.x0[i][0], st.x1[i][0], tau), y = lerp(st.x0[i][1], st.x1[i][1], tau);
    const a = pA * (st.hiSet.has(i) ? 1 : dim);
    dot(ctx, th, ...px(x, y), st.uCond[i], a, colorA);
  }
  ctx.restore();

  // Highlight: many slopes through one point, their average is the marginal velocity.
  if (hiIn > 0) {
    const [cx, cy] = px(...P_HI);
    ctx.save();
    ctx.strokeStyle = th.fg;
    ctx.globalAlpha = 0.6 * hiIn;
    ctx.setLineDash([5, 6]);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, R_HI * S + 10, 0, 2 * Math.PI); ctx.stroke();
    ctx.restore();
    const grow = ease.out(seg(t, T.a1Hi[0] + 0.3, T.a1Hi[0] + 1.6));
    const L = 0.42 * S; // same scale for every arrow, so the white one really is their mean
    for (const i of st.hiIdx) {
      const vx = st.x1[i][0] - st.x0[i][0], vy = st.x1[i][1] - st.x0[i][1];
      const [sx, sy] = px(0.5 * (st.x0[i][0] + st.x1[i][0]), 0.5 * (st.x0[i][1] + st.x1[i][1]));
      arrow(ctx, sx, sy, sx + L * vx * grow, sy - L * vy * grow, { color: rampColor(th, st.uCond[i], 1), width: 1.6, head: 7, alpha: 0.9 * hiIn });
    }
    const avg = ease.out(seg(t, T.a1Hi[0] + 2.0, T.a1Hi[0] + 2.9));
    if (avg > 0) {
      arrow(ctx, cx, cy, cx + L * st.uHi[0] * avg, cy - L * st.uHi[1] * avg, { color: th.highlight, width: 4, head: 14, alpha: hiIn });
      text(ctx, 'uₜ(x)', cx + L * st.uHi[0] * 1.2 - 64, cy - L * st.uHi[1] * 1.2 + 6, { size: 20, weight: 600, color: th.fg, alpha: avg * hiIn, font: 'serif' });
    }
  }
}

function drawAct2(ctx, t, env, st) {
  const th = env.theme;
  const tau = ease.inOut(seg(t, ...T.a2Go));
  const a = window_(t, T.a2[0], T.a2[1], 0.8);
  drawField(ctx, th, st.gmm, tau, a * (1 - seg(tau, 0.82, 1)) * 0.9);
  drawTrails(ctx, th, st.ode, st.uOde, tau, a, 0.34, N_TRAIL);
  drawHeads(ctx, th, st.ode, st.uOde, tau, a);
}

function drawAct3(ctx, t, env, st) {
  const th = env.theme;
  const tau = ease.inOut(seg(t, ...T.a3Go));
  const a = seg(t, T.a3[0], T.a3[0] + 0.8);
  drawTrails(ctx, th, st.sde, st.uSde, tau, a, th.name === 'dark' ? 0.11 : 0.16, N_TRAIL_SDE);
  drawHeads(ctx, th, st.sde, st.uSde, tau, a);
}

// ---------- Drawing helpers ----------

function dot(ctx, th, x, y, u, alpha, colorMix = 1) {
  glowDot(ctx, th, x, y, colorMix < 0.5 ? th.muted : rampColor(th, u, 1), alpha);
}

function drawTrails(ctx, th, trajs, us, tau, alpha, lineA, count) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  ctx.lineWidth = 1.15;
  ctx.lineJoin = 'round';
  for (let i = 0; i < count; i++) {
    const tr = trajs[i], steps = tr.length / 2 - 1, end = Math.floor(tau * steps);
    if (end < 1) continue;
    ctx.strokeStyle = rampColor(th, us[i], lineA * alpha);
    ctx.beginPath();
    let [x, y] = px(tr[0], tr[1]);
    ctx.moveTo(x, y);
    for (let j = 1; j <= end; j++) { [x, y] = px(tr[2 * j], tr[2 * j + 1]); ctx.lineTo(x, y); }
    [x, y] = px(...trajAt(tr, tau));
    ctx.lineTo(x, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHeads(ctx, th, trajs, us, tau, alpha) {
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  for (let i = 0; i < trajs.length; i++) dot(ctx, th, ...px(...trajAt(trajs[i], tau)), us[i], alpha);
  ctx.restore();
}

function drawField(ctx, th, gmm, tau, alpha) {
  if (alpha <= 0) return;
  for (let gx = -3.6; gx <= 3.61; gx += 0.4) {
    for (let gy = -2.8; gy <= 2.81; gy += 0.4) {
      const [vx, vy] = gmm.velocity(gx, gy, tau);
      const m = Math.hypot(vx, vy);
      if (m < 1e-6) continue;
      const len = Math.min(m * 0.12, 0.28) * S;
      const [sx, sy] = px(gx, gy);
      arrow(ctx, sx, sy, sx + (len * vx) / m, sy - (len * vy) / m, { color: th.muted, width: 1.2, head: 5, alpha: alpha * 0.55 });
    }
  }
}

// ---------- Text panel ----------

const COPY = [
  {
    span: T.intro, act: 'Setup',
    body: 'Gaussian noise at t = 0, data at t = 1. We want a velocity field whose flow carries one distribution onto the other.',
    eq: 'dx/dt = uₜ(x)',
  },
  {
    span: [T.a1[0], T.a1Hi[0]], act: 'Act I · Conditional paths',
    body: 'Pair each noise sample x₀ with a random data sample x₁ and join them by a straight line. The network regresses onto the slope x₁ − x₀. These paths are simple, but they cross.',
    eq: 'xₜ = (1 − t) x₀ + t x₁',
  },
  {
    span: [T.a1Hi[0], T.a1[1]], act: 'Act I · Averaging slopes',
    body: 'Many lines pass through the same point with different slopes. Squared error is minimized by their average, the marginal velocity, so that is what the network learns.',
    eq: 'uₜ(x) = E[ x₁ − x₀ | xₜ = x ]',
  },
  {
    span: T.a2, act: 'Act II · Marginal ODE',
    body: 'Following the averaged field gives a deterministic ODE. Paths bend and never cross, yet still land on the data. Colours mark each particle’s destination, so the noise-to-data map is smooth.',
    eq: 'dx/dt = uₜ(x)',
  },
  {
    span: T.a3, act: 'Act III · SDE, same marginals',
    body: 'Adding noise together with a score correction leaves every marginal pₜ unchanged. Paths turn jittery and the destination colours scramble, but the final distribution is the same.',
    eq: 'dx = [uₜ + ½g²∇log pₜ] dt + g dW',
  },
];

function drawPanel(ctx, t, env) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §2 FUNDAMENTALS', 'Flow matching', x);

  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.45);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    text(ctx, c.eq, x, yEnd + 30, { size: 25, color: th.fg, alpha: a, font: 'serif' });
  }

  // Interpolation-time readout.
  const tau = currentTau(t);
  text(ctx, 't', x, 720, { size: 22, color: th.muted, font: 'serif' });
  text(ctx, tau.toFixed(2), x + 22, 720, { size: 22, weight: 500, color: th.fg, font: 'mono' });
  ctx.fillStyle = th.faint; ctx.fillRect(x + 100, 712, w - 100, 4);
  ctx.fillStyle = th.accent; ctx.fillRect(x + 100, 712, (w - 100) * tau, 4);
  text(ctx, 'noise', x + 100, 742, { size: 13, color: th.muted });
  text(ctx, 'data', x + w, 742, { size: 13, color: th.muted, align: 'right' });

  text(ctx, 'Exact fields for a 110-component Gaussian mixture. No neural network involved.', x, 838, { size: 14, color: th.muted });
}

function currentTau(t) {
  if (t < T.a1[1]) return 0.5 * ease.inOut(seg(t, ...T.a1Go)) + 0.5 * ease.inOut(seg(t, ...T.a1Go2));
  if (t < T.a2[1]) return ease.inOut(seg(t, ...T.a2Go));
  return ease.inOut(seg(t, ...T.a3Go));
}

function drawStillLabel(ctx, t, env) {
  const s = [...STILLS].reverse().find((s) => t >= s.t - 3);
  if (s) text(ctx, s.label, CX - 404, CY - 376, { size: 40, weight: 600, color: env.theme.fg });
}
