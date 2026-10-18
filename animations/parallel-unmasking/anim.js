// A4 · Parallel unmasking: autoregressive vs masked-diffusion decoding.
// A tiny language of 64 eight-token sentences with three coupled token pairs. Its distribution is known, so every
// conditional p(x_i | unmasked tokens) is exact. The only error source is sampling several tokens independently
// in one step, which is what parallel decoding in masked diffusion models does.
import { makeRng } from '../lib/rng.js';
import { drawGrid, makePlot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const plot = makePlot(1060, 450, 150);

// ---------- The language ----------

const WTS = [0.4, 0.3, 0.2, 0.1];
const ANIMAL = [['cat', 'meows'], ['dog', 'barks'], ['owl', 'hoots'], ['cow', 'moos']];
const CITY = [['new', 'york'], ['san', 'francisco'], ['los', 'angeles'], ['hong', 'kong']];
const TIME = [['at', 'night'], ['in', 'spring'], ['on', 'sunday'], ['by', 'noon']];
const L = 8;
// Coupled position pairs and the group of each position (-1: fixed token).
const PAIRS = [[1, 5], [3, 4], [6, 7]];
const GROUP = [-1, 0, -1, 1, 1, 0, 2, 2];
const SENT = [], PS = [];
ANIMAL.forEach((a, i) => CITY.forEach((c, j) => TIME.forEach((m, k) => {
  SENT.push(['the', a[0], 'from', c[0], c[1], a[1], m[0], m[1]]);
  PS.push(WTS[i] * WTS[(j + 1) % 4] * WTS[(k + 2) % 4]);
})));
// Options per position, used for slot widths.
const OPTIONS = Array.from({ length: L }, (_, i) => [...new Set(SENT.map((s) => s[i]))]);

// Exact conditional marginals of every position given the revealed tokens x (null = mask).
// Once a sequence is inconsistent it can never become valid again; for display we then condition on the
// sentences that agree with the most revealed tokens.
function marginals(x) {
  let ok = SENT.map((s) => s.every((w, i) => x[i] === null || x[i] === w));
  if (!ok.some(Boolean)) {
    const m = SENT.map((s) => s.reduce((a, w, i) => a + (x[i] === w ? 1 : 0), 0)), mx = Math.max(...m);
    ok = m.map((v) => v === mx);
  }
  const M = Array.from({ length: L }, () => new Map());
  let z = 0;
  SENT.forEach((s, j) => {
    if (!ok[j]) return;
    z += PS[j];
    s.forEach((w, i) => M[i].set(w, (M[i].get(w) || 0) + PS[j]));
  });
  for (const m of M) for (const [k, v] of m) m.set(k, v / z);
  return M;
}

function draw(m, rng) {
  let u = rng.uniform(), c = 0, last;
  for (const [w, p] of m) { c += p; last = [w, p]; if (u <= c) return last; }
  return last;
}

export const isValid = (x) => SENT.some((s) => s.every((w, i) => w === x[i]));
const pairOk = (x, a, b) => SENT.some((s) => s[a] === x[a] && s[b] === x[b]);

// One decoding run. rule: {ar} | {k} (random positions) | {tau} (confidence threshold, at least one).
export function decode(rule, rng) {
  const x = Array(L).fill(null), steps = [];
  while (x.includes(null)) {
    const M = marginals(x), masked = x.map((v, i) => (v === null ? i : -1)).filter((i) => i >= 0);
    let preds, keep;
    if (rule.ar) {
      const i = masked[0], [w, p] = draw(M[i], rng);
      preds = [{ i, w, p }]; keep = [i];
    } else {
      preds = masked.map((i) => { const [w, p] = draw(M[i], rng); return { i, w, p, j: rng.uniform() }; });
      if (rule.k) keep = [...preds].sort((a, b) => b.j - a.j).slice(0, rule.k).map((p) => p.i);
      else {
        keep = preds.filter((p) => p.p > rule.tau).map((p) => p.i);
        if (!keep.length) keep = [[...preds].sort((a, b) => b.p + 1e-6 * b.j - a.p - 1e-6 * a.j)[0].i];
      }
    }
    for (const p of preds) if (keep.includes(p.i)) x[p.i] = p.w;
    steps.push({ preds, keep });
  }
  const bad = PAIRS.filter(([a, b]) => !pairOk(x, a, b));
  return { x, steps, ok: bad.length === 0, bad };
}

// Share of valid sentences and mean number of steps for a rule.
export function evaluate(rule, n, seed) {
  const rng = makeRng(seed);
  let ok = 0, st = 0;
  for (let r = 0; r < n; r++) { const o = decode(rule, rng); ok += o.ok; st += o.steps.length; }
  return { steps: st / n, valid: ok / n };
}

const TAU = 0.35, KPAR = 4;
const TAUS = [0.9, 0.35, 0.25, 0.15, 0.05];
// Seeds of the runs on screen (picked for typical outcomes: see the notes on the page).
const SEED_AR = 3, SEED_ONE = 4, SEEDS_K = [101, 102, 103, 104], SEEDS_TAU = [201, 203, 205, 209];

// ---------- Timeline ----------

const T = {
  setup: [0, 7.5],
  one: [7.5, 19.5], oneRun: [8.8, 17.6],
  par: [19.5, 31], parRun: [20.8, 27.4],
  tau: [31, 43.5], tauRun: [32.3, 41.3],
  chart: [43.5, 55], chartDraw: [44.5, 49.5],
};

const STILLS = [
  { t: 7.2, name: 'language', label: 'A language with coupled tokens' },
  { t: 19.2, name: 'one-per-step', label: 'One token per step (8 steps)' },
  { t: 30.7, name: 'fixed-parallel', label: 'Four tokens per step (2 steps)' },
  { t: 43.2, name: 'threshold', label: `Confidence threshold τ = ${TAU}` },
  { t: 54.6, name: 'tradeoff', label: 'Valid sentences vs steps' },
];

export default {
  id: 'parallel-unmasking',
  title: 'Parallel unmasking: autoregressive vs masked-diffusion decoding',
  duration: 55,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.one[0], label: 'I · One token per step' },
    { t: T.par[0], label: 'II · Fixed parallel' },
    { t: T.tau[0], label: 'III · Confidence threshold' },
    { t: T.chart[0], label: 'IV · Trade-off' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 1000, h: 900 },

  setup() {
    const st = {
      ar: decode({ ar: true }, makeRng(SEED_AR)),
      one: decode({ k: 1 }, makeRng(SEED_ONE)),
      par: SEEDS_K.map((s) => decode({ k: KPAR }, makeRng(s))),
      tau: SEEDS_TAU.map((s) => decode({ tau: TAU }, makeRng(s))),
      curveK: [1, 2, 4, 8].map((k) => ({ k, ...evaluate({ k }, 3000, 7) })),
      curveTau: TAUS.map((tau) => ({ tau, ...evaluate({ tau }, 3000, 9) })),
      samples: [0, 1, 2].map((s) => decode({ ar: true }, makeRng(50 + s)).x),
    };
    return st;
  },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    const sa = window_(t, T.setup[0], T.setup[1], 0.5), oa = window_(t, ...T.one, 0.5);
    const pa = window_(t, ...T.par, 0.5), ta = window_(t, ...T.tau, 0.5), ca = seg(t, T.chart[0], T.chart[0] + 0.6);
    const clean = env.clean;
    if (sa > 0) drawSetup(ctx, th, t, clean ? 1 : sa, st);
    if (oa > 0) drawOne(ctx, th, t, clean ? 1 : oa, st);
    if (pa > 0) drawBatch(ctx, th, t, clean ? 1 : pa, st.par, T.parRun, 2, `masked diffusion, ${KPAR} random tokens per step`);
    if (ta > 0) drawBatch(ctx, th, t, clean ? 1 : ta, st.tau, T.tauRun, 4, `masked diffusion, keep tokens with p > ${TAU}`, TAU);
    if (ca > 0) drawChart(ctx, th, t, ca, st);

    if (!clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 70, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Token rows ----------

const CH = 14.6, PAD = 30, GAP = 12, BH = 56;
const SLOT = OPTIONS.map((o) => Math.max(...o.map((w) => w.length)) * CH + PAD);
const ROWW = SLOT.reduce((a, b) => a + b) + GAP * (L - 1);
const X0 = 1090 - ROWW / 2;
const SX = SLOT.map((_, i) => X0 + SLOT.slice(0, i).reduce((a, b) => a + b, 0) + GAP * i);
const GCOL = (th, g) => (g < 0 ? th.muted : rampColor(th, [0, 0.36, 0.68][g], 1));
const OFF = (th) => (th.name === 'dark' ? '#ff6b6b' : '#dc2626');
const OK = (th) => (th.name === 'dark' ? '#4ade80' : '#16a34a');

function box(ctx, th, i, y, a, { fill = null, stroke = null, dash = false, width = 1.5 } = {}) {
  ctx.save();
  ctx.globalAlpha = a;
  ctx.beginPath(); ctx.roundRect(SX[i], y - BH / 2, SLOT[i], BH, 8);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; if (dash) ctx.setLineDash([5, 5]); ctx.stroke(); }
  ctx.restore();
  // Group colour strip along the bottom edge.
  ctx.save();
  ctx.globalAlpha = a * (GROUP[i] < 0 ? 0.35 : 0.9);
  ctx.fillStyle = GCOL(th, GROUP[i]);
  ctx.fillRect(SX[i] + 8, y + BH / 2 - 4, SLOT[i] - 16, 3);
  ctx.restore();
}

// Draws one decoding run at fractional step f (step s in progress with phase f - s).
// Returns the number of finished steps.
function drawRun(ctx, th, run, y, f, a, { bars = false, tau = null, showCheck = true } = {}) {
  const x = Array(L).fill(null);
  const s = Math.floor(f), ph = f - s, n = run.steps.length;
  for (let k = 0; k < Math.min(s, n); k++) for (const i of run.steps[k].keep) x[i] = run.x[i];
  const cur = s < n ? run.steps[s] : null;
  const appear = ease.out(seg(ph, 0, 0.35)), commit = ease.inOut(seg(ph, 0.5, 0.8));

  for (let i = 0; i < L; i++) {
    const cx = SX[i] + SLOT[i] / 2;
    const pred = cur && cur.preds.find((p) => p.i === i);
    const kept = cur && cur.keep.includes(i);
    if (x[i] !== null) {
      box(ctx, th, i, y, a, { fill: th.faint, stroke: th.muted });
      text(ctx, x[i], cx, y + 9, { size: 25, weight: 600, color: th.fg, align: 'center', alpha: a });
      continue;
    }
    // Masked slot.
    const solid = kept ? commit : 0;
    box(ctx, th, i, y, a, { stroke: th.muted, dash: true, width: 1.2 });
    if (solid > 0) box(ctx, th, i, y, a * solid, { fill: th.faint, stroke: th.accent, width: 2.2 });
    if (pred) {
      const ga = appear * (kept ? 1 : 1 - commit);
      text(ctx, pred.w, cx, y + 9, { size: 25, weight: solid > 0.5 ? 600 : 400, color: solid > 0.5 ? th.fg : th.muted, align: 'center', alpha: a * lerp(ga * 0.75, 1, solid) });
      if (bars) {
        const bw = SLOT[i] - 16, by = y + BH / 2 + 9;
        ctx.save();
        ctx.globalAlpha = a * (1 - commit * (kept ? 0 : 1)) * appear;
        ctx.fillStyle = th.faint; ctx.fillRect(SX[i] + 8, by, bw, 6);
        ctx.fillStyle = kept ? th.accent : th.muted; ctx.fillRect(SX[i] + 8, by, bw * pred.p * appear, 6);
        if (tau !== null) {
          ctx.strokeStyle = th.fg; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(SX[i] + 8 + bw * tau, by - 3); ctx.lineTo(SX[i] + 8 + bw * tau, by + 9); ctx.stroke();
        }
        ctx.restore();
      }
    } else {
      text(ctx, '?', cx, y + 8, { size: 20, color: th.muted, align: 'center', alpha: a * 0.6 });
    }
  }

  // Outcome: red links between disagreeing coupled tokens, or a check mark.
  if (f >= n && showCheck) {
    const oa = a * seg(f - n, 0, 0.3);
    for (const [p, q] of run.bad) {
      const x1 = SX[p] + SLOT[p] / 2, x2 = SX[q] + SLOT[q] / 2, yt = y - BH / 2;
      box(ctx, th, p, y, oa, { stroke: OFF(th), width: 2.6 });
      box(ctx, th, q, y, oa, { stroke: OFF(th), width: 2.6 });
      ctx.save();
      ctx.globalAlpha = oa; ctx.strokeStyle = OFF(th); ctx.lineWidth = 2.2;
      const h = 14 + 0.06 * (x2 - x1);
      ctx.beginPath(); ctx.moveTo(x1, yt - 2); ctx.bezierCurveTo(x1, yt - h, x2, yt - h, x2, yt - 2); ctx.stroke();
      ctx.restore();
    }
    const tx = X0 + ROWW + 22;
    if (run.ok) text(ctx, '✓', tx, y + 10, { size: 28, weight: 700, color: OK(th), alpha: oa });
    else text(ctx, '✗', tx, y + 10, { size: 28, weight: 700, color: OFF(th), alpha: oa });
  }
  return Math.min(s, n);
}

function rowLabel(ctx, th, label, y, a, right = '') {
  text(ctx, label, X0, y - BH / 2 - 16, { size: 17, weight: 600, color: th.muted, alpha: a });
  if (right) text(ctx, right, X0 + ROWW, y - BH / 2 - 16, { size: 17, weight: 600, color: th.fg, alpha: a, align: 'right', font: 'mono' });
}

// ---------- Scenes ----------

function drawSetup(ctx, th, t, a, st) {
  // Template row with options per slot and coupling arcs.
  const y = 300;
  text(ctx, 'sentence template', X0, y - 140, { size: 17, weight: 600, color: th.muted, alpha: a });
  for (let i = 0; i < L; i++) {
    const cx = SX[i] + SLOT[i] / 2;
    box(ctx, th, i, y, a, { stroke: th.muted, dash: GROUP[i] >= 0, width: 1.2 });
    if (GROUP[i] < 0) text(ctx, SENT[0][i], cx, y + 9, { size: 25, weight: 600, color: th.fg, align: 'center', alpha: a });
    else {
      const opts = OPTIONS[i];
      opts.forEach((w, k) => {
        const p = marginals(Array(L).fill(null))[i].get(w);
        text(ctx, w, cx, y + BH / 2 + 32 + 28 * k, { size: 19, color: GCOL(th, GROUP[i]), align: 'center', alpha: a * lerp(0.55, 1, p / 0.4) });
      });
      text(ctx, '?', cx, y + 8, { size: 20, color: th.muted, align: 'center', alpha: a * 0.6 });
    }
  }
  const la = a * seg(t, 1.5, 2.6);
  const names = ['animal and its sound', 'two halves of a city', 'preposition and time'];
  PAIRS.forEach(([p, q], g) => {
    const x1 = SX[p] + SLOT[p] / 2, x2 = SX[q] + SLOT[q] / 2, yt = y - BH / 2, h = [118, 52, 52][g];
    ctx.save();
    ctx.globalAlpha = la; ctx.strokeStyle = GCOL(th, g); ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(x1, yt - 3); ctx.bezierCurveTo(x1, yt - h, x2, yt - h, x2, yt - 3); ctx.stroke();
    ctx.restore();
    text(ctx, names[g], (x1 + x2) / 2, yt - 0.75 * h - 10, { size: 15, weight: 600, color: GCOL(th, g), align: 'center', alpha: la });
  });
  // A few sentences sampled from the language.
  const sa = a * seg(t, 3.2, 4.4);
  text(ctx, 'samples from the language', X0, 600, { size: 17, weight: 600, color: th.muted, alpha: sa });
  st.samples.forEach((x, r) => {
    const yy = 650 + 62 * r, ra = sa * seg(t, 3.4 + 0.5 * r, 4.2 + 0.5 * r);
    for (let i = 0; i < L; i++) {
      box(ctx, th, i, yy, ra, { fill: th.faint, stroke: th.muted });
      text(ctx, x[i], SX[i] + SLOT[i] / 2, yy + 9, { size: 25, weight: 600, color: th.fg, align: 'center', alpha: ra });
    }
  });
}

function drawOne(ctx, th, t, a, st) {
  const f = 8.6 * seg(t, ...T.oneRun);
  const y1 = 330, y2 = 560;
  const s1 = drawRun(ctx, th, st.ar, y1, f, a);
  rowLabel(ctx, th, 'autoregressive: left to right', y1, a, `step ${Math.min(Math.ceil(f), 8)} / 8`);
  const s2 = drawRun(ctx, th, st.one, y2, f, a, { bars: true });
  rowLabel(ctx, th, 'masked diffusion: one token per step, any order', y2, a, `step ${Math.min(Math.ceil(f), 8)} / 8`);
  void s1; void s2;
}

// Four runs in lockstep. Each step takes dur seconds; runs that finish early show their step count.
function drawBatch(ctx, th, t, a, runs, span, maxSteps, label, tau = null) {
  const f = (maxSteps + 0.6) * seg(t, ...span);
  const ys = [270, 410, 550, 690];
  text(ctx, label, X0, ys[0] - BH / 2 - 52, { size: 19, weight: 600, color: th.fg, alpha: a });
  runs.forEach((run, r) => {
    const n = run.steps.length, done = f >= n;
    drawRun(ctx, th, run, ys[r], Math.min(f, n + 0.999), a, { bars: true, tau });
    const stepTxt = done ? `${n} step${n > 1 ? 's' : ''}` : `step ${Math.min(Math.ceil(f) || 1, n)}`;
    rowLabel(ctx, th, `run ${r + 1}`, ys[r], a, stepTxt);
  });
  const allDone = f > maxSteps + 0.2;
  if (allDone) {
    const v = runs.filter((r) => r.ok).length;
    text(ctx, `${v} of ${runs.length} sentences valid`, X0, 800, { size: 21, weight: 700, color: v === runs.length ? OK(th) : OFF(th), alpha: a * seg(f, maxSteps + 0.2, maxSteps + 0.5) });
  }
}

function drawChart(ctx, th, t, alpha, st) {
  const L0 = 760, R0 = 1480, Tp = 170, B = 690;
  const X = (s) => L0 + ((s - 0.5) / 8) * (R0 - L0), Y = (v) => B - v * (B - Tp);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = th.bg;
  ctx.fillRect(620, 90, 980, 760);
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L0, Tp - 10); ctx.lineTo(L0, B); ctx.lineTo(R0, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const v of [0.25, 0.5, 0.75, 1]) { ctx.beginPath(); ctx.moveTo(L0, Y(v)); ctx.lineTo(R0, Y(v)); ctx.stroke(); }
  ctx.restore();
  for (const v of [0, 0.25, 0.5, 0.75, 1]) text(ctx, v.toFixed(2), L0 - 14, Y(v) + 6, { size: 17, color: th.muted, align: 'right', alpha });
  for (const s of [1, 2, 3, 4, 5, 6, 7, 8]) text(ctx, String(s), X(s), B + 28, { size: 17, color: th.muted, align: 'center', alpha });
  text(ctx, 'decoding steps per sentence (mean)', (L0 + R0) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha });
  text(ctx, 'share of valid sentences', L0, Tp - 30, { size: 18, color: th.muted, alpha });

  const grow = ease.inOut(seg(t, ...T.chartDraw));
  const series = [
    { pts: [...st.curveK].reverse(), color: OFF(th), name: 'fixed tokens per step', label: (p) => `${p.k}/step` },
    { pts: [...st.curveTau].reverse(), color: rampColor(th, 1, 1), name: 'confidence threshold', label: (p) => `τ = ${p.tau}` },
  ];
  for (const s of series) {
    const f = grow * (s.pts.length - 1);
    ctx.save();
    ctx.globalAlpha = alpha; ctx.strokeStyle = ctx.fillStyle = s.color; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(X(s.pts[0].steps), Y(s.pts[0].valid));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1), p = s.pts[i - 1], q = s.pts[i];
      ctx.lineTo(lerp(X(p.steps), X(q.steps), w), lerp(Y(p.valid), Y(q.valid), w));
    }
    ctx.stroke();
    for (let i = 0; i <= Math.floor(f); i++) { ctx.beginPath(); ctx.arc(X(s.pts[i].steps), Y(s.pts[i].valid), 5.5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
  }
  const la = alpha * seg(grow, 0.85, 1);
  // Series names and selected point labels.
  const k4 = st.curveK.find((p) => p.k === 4), k2 = st.curveK.find((p) => p.k === 2);
  text(ctx, 'fixed tokens per step (random positions)', X(k2.steps) + 16, Y(k2.valid) + 34, { size: 19, weight: 700, color: OFF(th), alpha: la });
  text(ctx, '4 per step', X(k4.steps) + 14, Y(k4.valid) + 6, { size: 15, color: th.muted, alpha: la });
  text(ctx, '2 per step', X(k2.steps) + 14, Y(k2.valid) + 6, { size: 15, color: th.muted, alpha: la });
  const t35 = st.curveTau.find((p) => p.tau === TAU);
  text(ctx, `confidence threshold, τ = ${TAU}`, X(t35.steps) + 4, Y(t35.valid) - 22, { size: 19, weight: 700, color: rampColor(th, 1, 1), alpha: la });
  text(ctx, 'one token per step', X(8) + 12, Y(0.86), { size: 15, color: th.fg, align: 'right', alpha: la });
  text(ctx, '(either order, 8 steps)', X(8) + 12, Y(0.86) + 21, { size: 15, color: th.muted, align: 'right', alpha: la });
  ctx.save();
  ctx.globalAlpha = la; ctx.strokeStyle = th.fg; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(X(8), Y(1), 10, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup · A tiny language',
    body: 'Every sentence has eight tokens. Three pairs are coupled: the animal fixes its sound, the two halves of a city name belong together, and the preposition fixes the time. The language has 64 sentences with known probabilities, so every conditional distribution is exact.',
    eq: '' },
  { span: T.one, act: 'Act I · One token per step',
    body: 'An autoregressive model reveals tokens from left to right. A masked diffusion model starts from all masks and can reveal them in any order. With one token per step, both sample the exact joint distribution, and both need eight steps.',
    eq: 'p(x) = ∏ᵢ p(xᵢ | tokens revealed before xᵢ)' },
  { span: T.par, act: 'Act II · Several tokens per step',
    body: 'To go faster, the masked model predicts every masked token at once and keeps four of them. Each token is drawn from its own marginal, independently of the others. Coupled tokens decoded in the same step can disagree, as in "san york" or "cat barks".',
    eq: '∏ᵢ p(xᵢ | xᵁ)  ≠  p(xᴹ | xᵁ)' },
  { span: T.tau, act: 'Act III · A confidence threshold',
    body: `Keep only predictions with probability above τ = ${TAU}, at least one per step, and mask the rest again. In this language a probability above ${TAU} means the most likely option, so tokens kept in the same step agree. The price is diversity: the decoder now favours the most probable sentences.`,
    eq: 'keep i  if  p(x̂ᵢ | xᵁ) > τ' },
  { span: T.chart, act: 'Act IV · Speed versus correctness',
    body: 'Each point averages 3000 decoded sentences. With a fixed number of tokens per step, validity drops quickly as the step count shrinks. The threshold rule stays valid at about three steps. It fails once τ drops below 0.3, the probability of the second most likely option.',
    eq: '' },
];

function drawPanel(ctx, t, env) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §2 DISCRETE AND LANGUAGE DIFFUSION', 'Parallel unmasking', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: c.eq.length > 30 ? 21 : 25, color: th.fg, alpha: a, font: 'serif' });
  }
  text(ctx, 'U: unmasked positions. M: masked positions.', x, 800, { size: 15, color: th.muted, alpha: window_(t, T.par[0], T.tau[1], 0.4) });
  text(ctx, 'Exact conditionals. The model is the true distribution, so errors come only from independent sampling.', x, 862, { size: 14, color: th.muted });
}
