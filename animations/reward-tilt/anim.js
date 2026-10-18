// A7 · Reward tilting and reward hacking.
// The base model is the exact spiral mixture. KL-regularized fine-tuning with reward r has the optimum
// p*(x) ∝ p(x) exp(r(x) / β). For a linear reward r(x) = x₁ and isotropic components this is again a Gaussian
// mixture: each weight is multiplied by exp(μ_k,1 / β) and each mean moves by σ² / β along the reward direction.
// So the fine-tuned model is exact for every β, and we sweep β from ∞ down to 0.004.
import { GMM } from '../lib/gmm.js';
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1000, CY = 450, S = 150;
const plot = makePlot(CX, CY, S);
const px = plot.px;

const { means: MEANS, us: US, std: STD } = spiralMixture();
const XS = MEANS.map((m) => m[0]);
const XMIN = Math.min(...XS), XMAX = Math.max(...XS);
const N = 600, NT = 60, STEPS = 32;
const NU = 26;                      // grid of dial positions for the particle simulation
const NC = 61;                      // finer grid for the metric curves
const MC = 8000;                    // exact samples per point of the metric curves
const LMAX = 250, KK = 6;           // 1/β runs from 0 to LMAX on a log-like dial
const lam = (u) => (LMAX * (Math.exp(KK * u) - 1)) / (Math.exp(KK) - 1);
const uOf = (l) => Math.log(1 + (l * (Math.exp(KK) - 1)) / LMAX) / KK;
const ON2 = (2 * STD) ** 2;          // "looks like data": within 2σ of some data mean
const norm = (x) => (x - XMIN) / (XMAX - XMIN); // reward in units where the data span [0, 1]

const U1 = 0.3, U2 = 0.55;           // dial positions at the end of acts I and II

const T = {
  setup: [0, 6], flow: [1.4, 5],
  tilt: [6, 19], tiltUp: [7.4, 15],
  collapse: [19, 29.5], collapseUp: [20.2, 26.5],
  hack: [29.5, 42], hackUp: [30.6, 39],
  curve: [42, 54], curveDraw: [43, 48.5],
};

const STILLS = [
  { t: 18.6, name: 'tilted', label: 'Strong KL (β = 0.3)' },
  { t: 29.0, name: 'collapsed', label: 'Weak KL (β = 0.06)' },
  { t: 41.4, name: 'hacked', label: 'Almost no KL (β = 0.004)' },
  { t: 53.4, name: 'overoptimization', label: 'Proxy vs true reward' },
];

const OFF = (th) => (th.name === 'dark' ? '#ff6b6b' : '#dc2626');
const PROXY = (th) => rampColor(th, 0, 1);
const TRUE = (th) => rampColor(th, 1, 1);

export default {
  id: 'reward-tilt',
  title: 'Reward tilting and reward hacking',
  duration: 54,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.tilt[0], label: 'I · KL-regularized tilt' },
    { t: T.collapse[0], label: 'II · Collapse' },
    { t: T.hack[0], label: 'III · Reward hacking' },
    { t: T.curve[0], label: 'IV · Overoptimization' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 1000, h: 900 },

  setup(env) {
    const st = buildData();
    renderAll(env, st);
    return st;
  },
  onTheme(env, st) { renderAll(env, st); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    const curveA = seg(t, T.curve[0], T.curve[0] + 0.8);
    const dim = 1 - 0.85 * curveA;

    ctx.globalAlpha = (env.clean ? 1 : seg(t, 0.2, 1.4)) * dim;
    ctx.drawImage(st.glow, 0, 0, W, H);
    ctx.globalAlpha = 1;
    drawReward(ctx, th, t, env, dim * (1 - curveA));

    const u = dialAt(t);
    drawSamples(ctx, th, st, t, u, dim * (1 - curveA));
    if (curveA > 0) drawCurve(ctx, th, st, t, curveA);

    if (!env.clean) drawPanel(ctx, t, env, st, u);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 70, { size: 36, weight: 600, color: th.fg });
    }
  },
};

function renderAll(env, st) {
  st.glow = renderGlow(env, plot, { means: MEANS, us: US, std: STD });
}

// Dial position u in [0, 1] at time t; 1/β = lam(u).
function dialAt(t) {
  return U1 * ease.inOut(seg(t, ...T.tiltUp))
    + (U2 - U1) * ease.inOut(seg(t, ...T.collapseUp))
    + (1 - U2) * ease.inOut(seg(t, ...T.hackUp));
}

// ---------- Exact fine-tuned model ----------

// The tilted mixture for reward strength l = 1/β, with closed-form expected reward, KL and effective modes.
export function tilted(l) {
  const lw = MEANS.map((m) => l * m[0]), mx = Math.max(...lw);
  const w = lw.map((v) => Math.exp(v - mx)), z = w.reduce((a, b) => a + b);
  const keep = w.map((v, k) => (v / z > 1e-10 ? k : -1)).filter((k) => k >= 0);
  const shift = STD * STD * l;
  const gmm = new GMM(keep.map((k) => [MEANS[k][0] + shift, MEANS[k][1]]), STD, { weights: keep.map((k) => w[k]) });
  let er = 0, ent = 0;
  w.forEach((v, k) => { const p = v / z; er += p * (MEANS[k][0] + shift); if (p > 0) ent -= p * Math.log(p); });
  const logZ = mx + Math.log(z) - Math.log(MEANS.length) + (STD * STD * l * l) / 2;
  return { gmm, kl: l * er - logZ, er, eff: Math.exp(ent) };
}

function nearest(x, y) {
  let best = Infinity, k0 = 0;
  for (let k = 0; k < MEANS.length; k++) {
    const d = (x - MEANS[k][0]) ** 2 + (y - MEANS[k][1]) ** 2;
    if (d < best) { best = d; k0 = k; }
  }
  return { k: k0, on: best < ON2 };
}

export function buildData() {
  const st = { grid: [], curve: [] };
  const rng = makeRng(11), x0 = [];
  for (let i = 0; i < N; i++) x0.push([rng.gauss(), rng.gauss()]);

  for (let j = 0; j < NU; j++) {
    const u = j / (NU - 1), { gmm } = tilted(lam(u));
    const keep = j === 0 ? N : NT; // full trajectories of the base model for the opening flow
    const end = new Float32Array(2 * N), on = new Uint8Array(N), col = new Float32Array(N), traj = [];
    for (let i = 0; i < N; i++) {
      let [x, y] = x0[i];
      const path = i < keep ? new Float32Array(2 * (STEPS + 1)) : null;
      if (path) { path[0] = x; path[1] = y; }
      const h = 1 / STEPS;
      for (let s = 0; s < STEPS; s++) {
        const t = s * h, a = gmm.velocity(x, y, t), b = gmm.velocity(x + h * a[0], y + h * a[1], Math.min(t + h, 0.999));
        x += 0.5 * h * (a[0] + b[0]); y += 0.5 * h * (a[1] + b[1]);
        if (path) { path[2 * s + 2] = x; path[2 * s + 3] = y; }
      }
      end[2 * i] = x; end[2 * i + 1] = y;
      const nb = nearest(x, y);
      on[i] = nb.on ? 1 : 0; col[i] = US[nb.k];
      if (path) traj.push(path);
    }
    st.grid.push({ u, end, on, col, traj });
  }

  // Metric curves from exact samples of each tilted mixture.
  for (let j = 0; j < NC; j++) {
    const u = j / (NC - 1), m = tilted(lam(u)), mrng = makeRng(23); // common random numbers keep the curve smooth
    let good = 0, tr = 0;
    for (let i = 0; i < MC; i++) {
      const s = m.gmm.sample(mrng);
      if (nearest(s.x, s.y).on) { good++; tr += norm(s.x); }
    }
    st.curve.push({ u, kl: Math.max(m.kl, 0), proxy: norm(m.er), true: tr / MC, onData: good / MC, eff: m.eff });
  }
  return st;
}

function gridAt(st, u) {
  const f = clamp(u) * (NU - 1), i = Math.min(Math.floor(f), NU - 2);
  return { a: st.grid[i], b: st.grid[i + 1], w: f - i };
}

function curveAt(st, u) {
  const f = clamp(u) * (NC - 1), i = Math.min(Math.floor(f), NC - 2), w = f - i, a = st.curve[i], b = st.curve[i + 1];
  const o = {};
  for (const k of Object.keys(a)) o[k] = lerp(a[k], b[k], w);
  return o;
}

// ---------- Drawing ----------

// The reward model: iso-lines of r(x) = x₁ and the edge of the data.
function drawReward(ctx, th, t, env, alpha) {
  const a = (env.clean ? 1 : seg(t, 1.6, 3)) * alpha;
  if (a <= 0) return;
  const [, yTop] = px(0, 1.95), [, yBot] = px(0, -2.55);
  ctx.save();
  ctx.globalAlpha = a * 0.55;
  ctx.strokeStyle = PROXY(th); ctx.lineWidth = 1.2; ctx.setLineDash([4, 7]);
  for (let x = -1; x <= 3; x++) {
    const [X] = px(x, 0);
    ctx.beginPath(); ctx.moveTo(X, yTop); ctx.lineTo(X, yBot); ctx.stroke();
  }
  ctx.globalAlpha = a * 0.9;
  ctx.strokeStyle = th.fg; ctx.lineWidth = 1.6; ctx.setLineDash([10, 6]);
  const [XE] = px(XMAX, 0);
  ctx.beginPath(); ctx.moveTo(XE, yTop + 30); ctx.lineTo(XE, yBot); ctx.stroke();
  ctx.restore();
  for (let x = -1; x <= 3; x++) {
    const [X] = px(x, 0);
    text(ctx, `r = ${x < 0 ? '−' : ''}${Math.abs(x)}`, X, yBot + 24, { size: 15, color: PROXY(th), align: 'center', alpha: a * 0.9 });
  }
  text(ctx, 'best data sample', XE + 10, yTop + 52, { size: 16, weight: 600, color: th.fg, alpha: a * 0.9 });
  // Reward direction arrow along the top.
  const [x0] = px(-1.2, 0), [x1] = px(2.6, 0), y = yTop - 4;
  ctx.save();
  ctx.globalAlpha = a; ctx.strokeStyle = ctx.fillStyle = PROXY(th); ctx.lineWidth = 2.4;
  ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1 + 12, y); ctx.lineTo(x1 - 2, y - 7); ctx.lineTo(x1 - 2, y + 7); ctx.fill();
  ctx.restore();
  text(ctx, 'reward model r(x) = x₁ increases to the right', (x0 + x1) / 2, y - 14, { size: 18, weight: 600, color: PROXY(th), align: 'center', alpha: a });
}

function drawSamples(ctx, th, st, t, u, dim) {
  const flow = seg(t, ...T.flow);
  const { a, b, w } = gridAt(st, u);
  const lineA = dim * seg(t, T.flow[1], T.flow[1] + 0.8) * (th.name === 'dark' ? 0.2 : 0.28);

  if (lineA > 0) {
    ctx.save();
    ctx.globalAlpha = lineA; ctx.lineWidth = 1.1;
    ctx.strokeStyle = th.ramp[3];
    for (let i = 0; i < NT; i++) {
      const p = a.traj[i], q = b.traj[i];
      ctx.beginPath();
      for (let k = 0; k <= STEPS; k++) {
        const [x, y] = px(lerp(p[2 * k], q[2 * k], w), lerp(p[2 * k + 1], q[2 * k + 1], w));
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  const off = OFF(th), g = w < 0.5 ? a : b;
  for (let i = 0; i < N; i++) {
    let x, y;
    if (flow < 1) {
      const p = st.grid[0].traj[i], f = ease.inOut(flow) * STEPS, k = Math.min(Math.floor(f), STEPS - 1), v = f - k;
      x = lerp(p[2 * k], p[2 * k + 2], v); y = lerp(p[2 * k + 1], p[2 * k + 3], v);
      const c = rampColor(th, st.grid[0].col[i], 1);
      dot(ctx, th, ...px(x, y), flow < 0.6 ? th.muted : c, dim * lerp(0.7, 0.95, ease.in(flow)), 2.1);
      continue;
    }
    x = lerp(a.end[2 * i], b.end[2 * i], w); y = lerp(a.end[2 * i + 1], b.end[2 * i + 1], w);
    const on = g.on[i];
    dot(ctx, th, ...px(x, y), on ? rampColor(th, g.col[i], 1) : off, dim * (on ? 0.95 : 0.85), 2.1);
  }
}

// Proxy and true reward against the KL divergence from the base model.
function drawCurve(ctx, th, st, t, alpha) {
  const L = 760, Rr = 1480, Tp = 170, B = 690;
  const KX = (kl) => Math.log10(1 + kl) / Math.log10(1 + 200);
  const X = (kl) => L + KX(kl) * (Rr - L), Y = (r) => B - (r / 1.45) * (B - Tp);
  ctx.save();
  ctx.globalAlpha = alpha * 0.9;
  ctx.fillStyle = th.bg;
  ctx.fillRect(L - 110, Tp - 80, Rr - L + 170, B - Tp + 180);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L, Tp); ctx.lineTo(L, B); ctx.lineTo(Rr, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const r of [0.5, 1]) { ctx.beginPath(); ctx.moveTo(L, Y(r)); ctx.lineTo(Rr, Y(r)); ctx.stroke(); }
  ctx.strokeStyle = th.fg; ctx.globalAlpha = alpha * 0.7; ctx.setLineDash([10, 6]); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L, Y(1)); ctx.lineTo(Rr, Y(1)); ctx.stroke();
  ctx.restore();
  text(ctx, 'best data sample', Rr, Y(1) - 10, { size: 15, color: th.fg, align: 'right', alpha: alpha * 0.85 });
  for (const r of [0, 0.5, 1]) text(ctx, r.toFixed(1), L - 14, Y(r) + 6, { size: 17, color: th.muted, align: 'right', alpha });
  for (const k of [0, 1, 3, 10, 30, 100]) text(ctx, String(k), X(k), B + 28, { size: 17, color: th.muted, align: 'center', alpha });
  text(ctx, 'KL divergence from the base model (nats, log scale)', (L + Rr) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha });
  text(ctx, 'reward (data span 0 to 1)', L, Tp - 24, { size: 18, color: th.muted, alpha });

  const grow = ease.inOut(seg(t, ...T.curveDraw));
  const n = st.curve.length, f = grow * (n - 1);
  const series = [
    { key: 'proxy', color: PROXY(th), name: 'proxy reward' },
    { key: 'true', color: TRUE(th), name: 'true reward' },
  ];
  for (const s of series) {
    ctx.save();
    ctx.globalAlpha = alpha; ctx.strokeStyle = s.color; ctx.lineWidth = 3.4; ctx.lineJoin = 'round';
    ctx.beginPath();
    const c0 = st.curve[0];
    ctx.moveTo(X(c0.kl), Y(c0[s.key]));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const v = Math.min(f - (i - 1), 1), p = st.curve[i - 1], q = st.curve[i];
      ctx.lineTo(lerp(X(p.kl), X(q.kl), v), lerp(Y(p[s.key]), Y(q[s.key]), v));
    }
    ctx.stroke();
    ctx.restore();
  }
  // Labels and the three act markers.
  const la = alpha * seg(grow, 0.85, 1);
  const end = st.curve[n - 1];
  text(ctx, series[0].name, X(end.kl) - 20, Y(end.proxy) - 12, { size: 20, weight: 700, color: PROXY(th), align: 'right', alpha: la });
  text(ctx, series[1].name, X(end.kl) - 18, Y(end.true) - 16, { size: 20, weight: 700, color: TRUE(th), align: 'right', alpha: la });
  const marks = [[U1, 'I'], [U2, 'II'], [1, 'III']];
  for (const [u, lbl] of marks) {
    if (u * (n - 1) > f) continue;
    const c = curveAt(st, u), x = X(c.kl);
    ctx.save();
    ctx.globalAlpha = alpha * 0.5; ctx.strokeStyle = th.muted; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.moveTo(x, Y(c.proxy)); ctx.lineTo(x, B); ctx.stroke();
    ctx.restore();
    ctx.save();
    ctx.globalAlpha = alpha;
    for (const [k, col] of [['proxy', PROXY(th)], ['true', TRUE(th)]]) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, Y(c[k]), 6, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
    text(ctx, lbl, x, Tp + 4, { size: 18, weight: 700, color: th.fg, align: 'center', alpha: la, font: 'serif' });
  }
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'A base flow model samples the spiral. A reward model scores each sample by how far right it is. It was fit on data, so away from the spiral its scores are extrapolation.',
    eq: 'r(x) = x₁' },
  { span: T.tilt, act: 'Act I · KL-regularized fine-tuning',
    body: 'Fine-tuning a model q maximizes reward minus β times its KL divergence to the base model p. The optimum reweights p by exp(r / β). With a strong KL term, mass moves right along the spiral and keeps its shape.',
    eq: 'max over q:  E[r(x)] − β KL(q ‖ p)' },
  { span: T.collapse, act: 'Act II · A weaker KL term',
    body: 'As β shrinks, the tilt concentrates on the few beads with the highest reward. Diversity collapses, but every sample still looks like data.',
    eq: 'q*(x) ∝ p(x) exp(r(x) / β)' },
  { span: T.hack, act: 'Act III · Reward hacking',
    body: 'The tilt also moves every component by σ² / β toward higher reward. For small β the samples leave the data. The reward model scores them above every real sample. Red marks samples that are not within 2σ of a data mode.',
    eq: 'μₖ  ↦  μₖ + (σ² / β) ∇r' },
  { span: T.curve, act: 'Act IV · Overoptimization',
    body: 'As the model moves away from the base, the proxy reward keeps rising. The true reward, which counts only samples that look like data, peaks at a moderate KL and then falls to zero.',
    eq: '' },
];

function drawPanel(ctx, t, env, st, u) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §4 POST-TRAINING', 'Reward tilting', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 30, { size: c.eq.length > 30 ? 21 : 25, color: th.fg, alpha: a, font: 'serif' });
  }

  // The KL dial and three metrics.
  const da = window_(t, 0.8, T.curve[0] + 0.6, 0.6);
  if (da > 0) {
    const y0 = 632, x0 = x, x1 = x + 430, X = (v) => lerp(x0, x1, v);
    ctx.save();
    ctx.globalAlpha = da;
    ctx.strokeStyle = th.muted; ctx.lineWidth = 3; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    ctx.strokeStyle = u > U2 ? OFF(th) : TRUE(th); ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(X(u), y0); ctx.stroke();
    ctx.fillStyle = th.fg;
    ctx.beginPath(); ctx.arc(X(u), y0, 9, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
    for (const [b, lbl] of [[Infinity, '∞'], [1, '1'], [0.1, '0.1'], [0.01, '0.01']]) {
      text(ctx, lbl, X(b === Infinity ? 0 : uOf(1 / b)), y0 + 30, { size: 15, color: th.muted, align: 'center', alpha: da });
    }
    text(ctx, 'KL WEIGHT β', x0, y0 - 26, { size: 14, weight: 600, color: th.muted, alpha: da, spacing: 1.5 });
    const l = lam(u), bs = l < 1e-3 ? '∞' : 1 / l >= 0.1 ? (1 / l).toFixed(2) : (1 / l).toFixed(3);
    text(ctx, `β = ${bs}`, x1, y0 - 22, { size: 22, weight: 700, color: th.fg, alpha: da, align: 'right', font: 'mono' });

    const c = curveAt(st, u);
    bar(ctx, th, x0, 690, 'proxy reward', c.proxy, c.proxy.toFixed(2), PROXY(th), da);
    bar(ctx, th, x0, 742, 'true reward', c.true, c.true.toFixed(2), TRUE(th), da);
    const em = Math.log(c.eff) / Math.log(MEANS.length);
    bar(ctx, th, x0, 794, 'effective modes', em * 1.45, c.eff < 10 ? c.eff.toFixed(1) : c.eff.toFixed(0), th.muted, da, false);
  }

  text(ctx, 'Exact fields. The fine-tuned model is the exact tilted mixture, which is again a Gaussian mixture.', x, 862, { size: 14, color: th.muted });
}

// Reward bars span 0 to 1.45 so the proxy can run past the best data sample (tick at 1).
function bar(ctx, th, x, y, label, v, vs, color, a, tick = true) {
  const w = 300;
  text(ctx, label, x, y, { size: 17, weight: 600, color: th.fg, alpha: a });
  text(ctx, vs, x + w + 130, y + 26, { size: 20, weight: 700, color: th.fg, alpha: a, align: 'right', font: 'mono' });
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.faint;
  ctx.fillRect(x, y + 12, w, 14);
  ctx.fillStyle = color;
  ctx.fillRect(x, y + 12, w * clamp(v / 1.45), 14);
  if (tick) {
    ctx.strokeStyle = th.fg; ctx.lineWidth = 2;
    const xt = x + w / 1.45;
    ctx.beginPath(); ctx.moveTo(xt, y + 6); ctx.lineTo(xt, y + 32); ctx.stroke();
  }
  ctx.restore();
}
