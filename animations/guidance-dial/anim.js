// A5 · The guidance dial: classifier-free guidance vs autoguidance on exact mixtures.
// Two classes live on overlapping arcs of one ring. The "trained" conditional model is the true class-A mixture
// with every component blurred (an imperfect network), the unconditional model is the same blur over both
// classes, and the weak model for autoguidance blurs class A further. All velocities are exact.
import { GMM } from '../lib/gmm.js';
import { makeRng } from '../lib/rng.js';
import { makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1060, CY = 455, S = 165;
const plot = makePlot(CX, CY, S);
const px = plot.px;

const K = 40;            // components per class
const R = 1.7;           // ring radius
const OVERLAP = 30;      // degrees by which each class arc runs past the horizontal into the other
const STD = 0.075;       // data
const STD_GOOD = 0.11;   // the conditional and unconditional models
const STD_BAD = 0.18;    // the weak model used by autoguidance
const N = 500, NT = 70, STEPS = 32;
const WS = Array.from({ length: 11 }, (_, i) => 1 + 0.5 * i); // guidance weights 1 ... 6
const DATA_PREC = 1 - Math.exp(-2); // precision of exact data samples: within 2σ of their own mean

const T = {
  setup: [0, 5.2], flow: [1.6, 4.6],
  cfg: [5.2, 21], cfgUp: [6.4, 15], cfgHold: 20.4,
  auto: [21, 37.4], bad: [21.6, 24], autoUp: [24.6, 29], autoMore: [31.6, 35],
  trade: [37.4, 48], tradeDraw: [38.4, 43],
};

const STILLS = [
  { t: 5.0, name: 'conditional', label: 'No guidance (w = 1)' },
  { t: 19.6, name: 'cfg', label: 'Classifier-free guidance, w = 6' },
  { t: 30.6, name: 'autoguidance', label: 'Autoguidance, w = 3' },
  { t: 47.4, name: 'tradeoff', label: 'Precision vs coverage' },
];

const AMBER = (th, a = 1) => rampColor(th, 1, a);
const CYAN = (th, a = 1) => rampColor(th, 0, a);
const OFF = (th) => (th.name === 'dark' ? '#ff6b6b' : '#dc2626');

export default {
  id: 'guidance-dial',
  title: 'The guidance dial: classifier-free guidance vs autoguidance',
  duration: 48,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.cfg[0], label: 'I · CFG' },
    { t: T.auto[0], label: 'II · Autoguidance' },
    { t: T.trade[0], label: 'III · Trade-off' },
  ],
  stills: STILLS,
  crop: { x: CX - 430, y: CY - 430, w: 860, h: 860 },

  setup(env) {
    const st = buildData();
    renderAll(env, st);
    return st;
  },
  onTheme(env, st) { renderAll(env, st); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    const tradeA = seg(t, T.trade[0], T.trade[0] + 0.8);
    const dim = 1 - 0.85 * tradeA;

    // Target densities: class A (amber) and class B (cyan), and the weak model during act II.
    ctx.globalAlpha = (env.clean ? 1 : seg(t, 0.2, 1.4)) * dim;
    ctx.drawImage(st.glowA, 0, 0, W, H);
    ctx.globalAlpha = (env.clean ? 1 : seg(t, 0.6, 1.8)) * dim * 0.85;
    ctx.drawImage(st.glowB, 0, 0, W, H);
    const badA = window_(t, T.auto[0], T.auto[1], 0.5) * (0.35 + 0.65 * window_(t, T.bad[0], T.autoUp[0] + 0.6, 0.6));
    ctx.globalAlpha = badA * dim;
    ctx.drawImage(st.glowBad, 0, 0, W, H);
    ctx.globalAlpha = 1;
    labels(ctx, th, t, env, dim * (1 - tradeA), badA);

    const s = stateAt(t);
    if (s) drawSamples(ctx, th, st, t, s, dim);
    if (tradeA > 0) drawTradeoff(ctx, th, st, t, tradeA);

    if (!env.clean) drawPanel(ctx, t, env, st, s);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, CX - 404, CY - 376, { size: 36, weight: 600, color: th.fg });
    }
  },
};

function renderAll(env, st) {
  st.glowA = renderGlow(env, plot, { means: st.A, us: st.A.map(() => 1), std: STD });
  st.glowB = renderGlow(env, plot, { means: st.B, us: st.B.map(() => 0), std: STD });
  st.glowBad = renderGlow(env, plot, { means: st.A, us: st.A.map(() => 1), std: STD_BAD });
}

// Which method and guidance weight are on screen at time t.
function stateAt(t) {
  if (t < T.cfg[0]) return { mode: 'cfg', w: 1, flow: seg(t, ...T.flow) };
  if (t < T.auto[0]) return { mode: 'cfg', w: 1 + 5 * ease.inOut(seg(t, ...T.cfgUp)), flow: 1 };
  if (t < T.trade[0] + 0.8) {
    const w = 1 + 2 * ease.inOut(seg(t, ...T.autoUp)) + 3 * ease.inOut(seg(t, ...T.autoMore));
    return { mode: 'auto', w, flow: 1 };
  }
  return null;
}

// ---------- Simulation ----------

export function buildData() {
  const arc = (a0, a1) => Array.from({ length: K }, (_, k) => {
    const a = ((a0 + ((a1 - a0) * k) / (K - 1)) * Math.PI) / 180;
    return [R * Math.cos(a), R * Math.sin(a)];
  });
  const A = arc(-OVERLAP, 180 + OVERLAP), B = arc(180 - OVERLAP, 360 + OVERLAP);
  const cond = new GMM(A, STD_GOOD), uncond = new GMM([...A, ...B], STD_GOOD), bad = new GMM(A, STD_BAD);
  const st = { A, B, cfg: [], auto: [] };

  const rng = makeRng(5), x0 = [];
  for (let i = 0; i < N; i++) x0.push([rng.gauss(), rng.gauss()]);

  for (const mode of ['cfg', 'auto']) {
    const other = mode === 'cfg' ? uncond : bad;
    for (const w of WS) {
      const v = (x, y, t) => {
        const c = cond.velocity(x, y, t), o = other.velocity(x, y, t);
        return [o[0] + w * (c[0] - o[0]), o[1] + w * (c[1] - o[1])];
      };
      const keep = mode === 'cfg' && w === 1 ? N : NT; // full trajectories for the opening flow
      const end = new Float32Array(2 * N), traj = [];
      for (let i = 0; i < N; i++) {
        let [x, y] = x0[i];
        const path = i < keep ? new Float32Array(2 * (STEPS + 1)) : null;
        if (path) { path[0] = x; path[1] = y; }
        const h = 1 / STEPS;
        for (let s = 0; s < STEPS; s++) {
          const t = s * h, a = v(x, y, t), b = v(x + h * a[0], y + h * a[1], Math.min(t + h, 0.999));
          x += 0.5 * h * (a[0] + b[0]); y += 0.5 * h * (a[1] + b[1]);
          if (path) { path[2 * s + 2] = x; path[2 * s + 3] = y; }
        }
        end[2 * i] = x; end[2 * i + 1] = y;
        if (path) traj.push(path);
      }
      st[mode].push({ w, end, traj, ...metrics(A, end) });
    }
  }
  return st;
}

// Precision: share of samples within 2σ of a class-A mean (exact data scores DATA_PREC).
// Coverage: share of class-A components with a sample within 2σ of their mean.
export function metrics(A, end) {
  const n = end.length / 2, r2 = (2 * STD) ** 2, hit = new Uint8Array(A.length), on = new Uint8Array(n);
  let p = 0;
  for (let i = 0; i < n; i++) {
    const x = end[2 * i], y = end[2 * i + 1];
    for (let k = 0; k < A.length; k++) {
      if ((x - A[k][0]) ** 2 + (y - A[k][1]) ** 2 < r2) { hit[k] = 1; on[i] = 1; }
    }
    p += on[i];
  }
  return { prec: p / n, cov: hit.reduce((a, b) => a + b) / A.length, on };
}

// Linear interpolation between the two nearest precomputed guidance weights.
function at(st, mode, w) {
  const f = clamp((w - 1) / 0.5, 0, WS.length - 1), i = Math.min(Math.floor(f), WS.length - 2), u = f - i;
  return { a: st[mode][i], b: st[mode][i + 1], u };
}

// ---------- Drawing ----------

function labels(ctx, th, t, env, dim, badA) {
  const a = (env.clean ? 1 : seg(t, 0.8, 2)) * dim;
  // In paper stills the still title occupies the top, so the class A label moves inside the ring.
  const [ax, ay] = px(0, env.clean ? R - 0.42 : R + 0.42), [bx, by] = px(0, -R - 0.5);
  text(ctx, 'class A (prompt)', ax, ay, { size: 21, weight: 600, color: AMBER(th), alpha: a, align: 'center' });
  text(ctx, 'class B', bx, by, { size: 21, weight: 600, color: CYAN(th), alpha: a, align: 'center' });
  if (badA > 0.4 && !env.clean) {
    const [x, y] = px(R + 0.32, 0.55);
    text(ctx, 'weak model', x, y, { size: 19, weight: 600, color: AMBER(th), alpha: (badA - 0.35) / 0.65 * dim, font: 'serif' });
  }
}

function drawSamples(ctx, th, st, t, s, dim) {
  const { a, b, u } = at(st, s.mode, s.w);
  const lineA = dim * (s.flow < 1 ? 0 : 1) * (th.name === 'dark' ? 0.22 : 0.3);

  // Trajectories of a subset, for the current weight.
  if (lineA > 0) {
    ctx.save();
    ctx.strokeStyle = AMBER(th, 1);
    ctx.globalAlpha = lineA;
    ctx.lineWidth = 1.1;
    for (let i = 0; i < NT; i++) {
      const p = a.traj[i], q = b.traj[i];
      ctx.beginPath();
      for (let k = 0; k <= STEPS; k++) {
        const [x, y] = px(lerp(p[2 * k], q[2 * k], u), lerp(p[2 * k + 1], q[2 * k + 1], u));
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  // Samples. During the opening they flow from noise along the w = 1 trajectories.
  const amber = AMBER(th, 1), off = OFF(th);
  for (let i = 0; i < N; i++) {
    let x, y;
    if (s.flow < 1) {
      const p = st.cfg[0].traj[i], f = ease.inOut(s.flow) * STEPS, k = Math.min(Math.floor(f), STEPS - 1), v = f - k;
      x = lerp(p[2 * k], p[2 * k + 2], v); y = lerp(p[2 * k + 1], p[2 * k + 3], v);
    } else {
      x = lerp(a.end[2 * i], b.end[2 * i], u); y = lerp(a.end[2 * i + 1], b.end[2 * i + 1], u);
    }
    const on = (u < 0.5 ? a : b).on[i];
    const col = s.flow < 1 ? th.muted : on ? amber : off;
    dot(ctx, th, ...px(x, y), s.flow < 1 ? mixHex(th.muted, th.ramp[3], ease.in(s.flow)) : col, dim * (s.flow < 1 ? 0.85 : on ? 0.95 : 0.8), 2.1);
  }
}

function mixHex(c1, c2, w) {
  const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const a = h(c1), b = h(c2);
  return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], w))).join(',')})`;
}

// Precision vs coverage, one curve per method, parametrised by w.
function drawTradeoff(ctx, th, st, t, alpha) {
  const L = CX - 290, Rr = CX + 330, Tp = CY - 250, B = CY + 250;
  const X = (c) => L + ((c - 0.3) / 0.75) * (Rr - L), Y = (p) => B - ((p - 0.45) / 0.6) * (B - Tp);
  ctx.save();
  ctx.globalAlpha = alpha * 0.9;
  ctx.fillStyle = th.bg;
  ctx.fillRect(L - 100, Tp - 70, Rr - L + 150, B - Tp + 160);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L, Tp); ctx.lineTo(L, B); ctx.lineTo(Rr, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const p of [0.6, 0.8, 1]) { ctx.beginPath(); ctx.moveTo(L, Y(p)); ctx.lineTo(Rr, Y(p)); ctx.stroke(); }
  ctx.restore();
  for (const p of [0.6, 0.8, 1]) text(ctx, p.toFixed(1), L - 14, Y(p) + 6, { size: 17, color: th.muted, align: 'right', alpha });
  for (const c of [0.4, 0.6, 0.8, 1]) text(ctx, c.toFixed(1), X(c), B + 28, { size: 17, color: th.muted, align: 'center', alpha });
  text(ctx, 'coverage (share of class-A modes reached)', (L + Rr) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha });
  text(ctx, 'precision (share of samples on a class-A mode)', L, Tp - 24, { size: 18, color: th.muted, alpha });

  // Exact data samples.
  const [dx, dy] = [X(1), Y(DATA_PREC)];
  ctx.save();
  ctx.globalAlpha = alpha; ctx.strokeStyle = th.fg; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(dx - 8, dy); ctx.lineTo(dx + 8, dy); ctx.moveTo(dx, dy - 8); ctx.lineTo(dx, dy + 8); ctx.stroke();
  ctx.restore();
  text(ctx, 'data', dx - 14, dy + 6, { align: 'right', size: 18, weight: 600, color: th.fg, alpha });

  const grow = ease.inOut(seg(t, ...T.tradeDraw));
  const series = [
    { pts: st.cfg, color: OFF(th), name: 'classifier-free guidance', at: [0.62, 0.56], align: 'left' },
    { pts: st.auto, color: AMBER(th), name: 'autoguidance', lx: -16, ly: -6, align: 'right' },
  ];
  for (const s of series) {
    const f = grow * (s.pts.length - 1);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = ctx.fillStyle = s.color;
    ctx.lineWidth = 3.2; ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(X(s.pts[0].cov), Y(s.pts[0].prec));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1), p = s.pts[i - 1], q = s.pts[i];
      ctx.lineTo(lerp(X(p.cov), X(q.cov), w), lerp(Y(p.prec), Y(q.prec), w));
    }
    ctx.stroke();
    for (let i = 0; i <= Math.floor(f); i += 2) { ctx.beginPath(); ctx.arc(X(s.pts[i].cov), Y(s.pts[i].prec), 5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
    const la = alpha * seg(grow, 0.9, 1);
    const last = s.pts[s.pts.length - 1];
    // The CFG name sits below its curve, since its end point is at the left edge of the plot.
    const [nx, ny] = s.at ? [X(s.at[0]), Y(s.at[1])] : [X(last.cov) + s.lx, Y(last.prec) + s.ly];
    text(ctx, s.name, nx, ny, { size: 21, weight: 700, color: s.color, align: s.align, alpha: la });
    // Weight labels at w = 3 and 6 on each curve; w = 1 is shared.
    const isCfg = s.pts === st.cfg;
    for (const i of isCfg ? [0, 4, 10] : [4]) {
      if (i > f) continue;
      const p = s.pts[i];
      if (isCfg && i === 0) {
        text(ctx, 'w = 1 (both)', X(p.cov) + 14, Y(p.prec) + 6, { size: 15, color: th.muted, alpha: la });
      } else if (isCfg) {
        text(ctx, `w = ${p.w}`, X(p.cov) + 10, Y(p.prec) + 26, { size: 15, color: th.muted, alpha: la });
      } else {
        text(ctx, 'w = 3 to 6', X(p.cov) + 14, Y(p.prec) + 6, { size: 15, color: th.muted, alpha: la });
      }
    }
  }
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'Two classes share one ring and overlap at its two ends. We sample class A from a conditional model that is slightly too blurry, as a trained network would be.',
    eq: 'u(x | c),   u(x | ∅)' },
  { span: [T.cfg[0], T.cfg[1]], act: 'Act I · Classifier-free guidance',
    body: 'Guidance extrapolates from the unconditional velocity past the conditional one. Samples move away from everything class B could also explain: the overlapping ends empty out, mass piles up at the top, and some samples overshoot off the ring. Red marks samples that are not on any class-A mode.',
    eq: 'u_w = u(x | ∅) + w [u(x | c) − u(x | ∅)]' },
  { span: [T.auto[0], T.autoMore[0] - 0.4], act: 'Act II · Autoguidance',
    body: 'Autoguidance extrapolates away from a weaker version of the same conditional model instead. The two models make the same kind of error, so the difference points from blur to detail. Samples sharpen onto the modes and the whole arc stays covered.',
    eq: 'u_w = u_weak(x | c) + w [u(x | c) − u_weak(x | c)]' },
  { span: [T.autoMore[0] - 0.4, T.auto[1]], act: 'Act II · Autoguidance',
    body: 'Pushed further, autoguidance over-sharpens: samples become narrower than the data. The useful range of w is where precision matches the data.',
    eq: 'u_w = u_weak(x | c) + w [u(x | c) − u_weak(x | c)]' },
  { span: T.trade, act: 'Act III · The trade-off',
    body: 'Each curve traces w from 1 to 6. Classifier-free guidance loses coverage, and past w ≈ 3 it loses precision too. Autoguidance raises precision at full coverage. In practice CFG also improves prompt alignment, which this toy does not measure.',
    eq: '' },
];

function drawPanel(ctx, t, env, st, s) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §2 CONDITIONING AND GUIDANCE', 'The guidance dial', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 30, { size: c.eq.length > 40 ? 20 : 25, color: th.fg, alpha: a, font: 'serif' });
  }

  // The dial and the two metrics.
  const da = s ? window_(t, 0.8, T.trade[0] + 0.6, 0.6) : 0;
  if (da > 0) {
    const { a, b, u } = at(st, s.mode, s.w), y0 = 612, x0 = x, x1 = x + 430;
    const X = (v) => lerp(x0, x1, (v - 1) / 5);
    ctx.save();
    ctx.globalAlpha = da;
    ctx.strokeStyle = th.muted; ctx.lineWidth = 3; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    ctx.strokeStyle = s.mode === 'cfg' ? OFF(th) : AMBER(th); ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(X(s.w), y0); ctx.stroke();
    ctx.fillStyle = th.fg;
    ctx.beginPath(); ctx.arc(X(s.w), y0, 9, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
    for (const v of [1, 2, 3, 4, 5, 6]) text(ctx, String(v), X(v), y0 + 30, { size: 15, color: th.muted, align: 'center', alpha: da });
    text(ctx, s.mode === 'cfg' ? 'CFG WEIGHT' : 'AUTOGUIDANCE WEIGHT', x0, y0 - 26, { size: 14, weight: 600, color: th.muted, alpha: da, spacing: 1.5 });
    text(ctx, `w = ${s.w.toFixed(2)}`, x1, y0 - 22, { size: 22, weight: 700, color: th.fg, alpha: da, align: 'right', font: 'mono' });

    const prec = lerp(a.prec, b.prec, u), cov = lerp(a.cov, b.cov, u);
    bar(ctx, th, x0, 690, 300, 'precision', prec, AMBER(th), da, DATA_PREC);
    bar(ctx, th, x0, 752, 300, 'coverage', cov, AMBER(th), da, 1);
  }

  text(ctx, `Exact fields. Data components σ = ${STD}; the models blur them to ${STD_GOOD} (conditional) and ${STD_BAD} (weak).`, x, 838, { size: 14, color: th.muted });
}

function bar(ctx, th, x, y, w, label, v, color, a, ref) {
  text(ctx, label, x, y, { size: 17, weight: 600, color: th.fg, alpha: a });
  text(ctx, v.toFixed(2), x + w + 130, y + 26, { size: 20, weight: 700, color: th.fg, alpha: a, align: 'right', font: 'mono' });
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.faint;
  ctx.fillRect(x, y + 12, w, 14);
  ctx.fillStyle = color;
  ctx.fillRect(x, y + 12, w * clamp(v), 14);
  ctx.strokeStyle = th.fg; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x + w * ref, y + 6); ctx.lineTo(x + w * ref, y + 32); ctx.stroke();
  ctx.restore();
  if (ref < 1) text(ctx, 'data', x + w * ref, y + 4, { size: 13, color: th.muted, alpha: a, align: 'center' });
}
