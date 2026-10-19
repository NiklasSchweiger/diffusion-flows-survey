// A2 · Score fields and annealed Langevin dynamics on the exact spiral mixture.
// Smoothing the data with Gaussian noise of scale σ gives another Gaussian mixture (component std √(s² + σ²)), so its
// score ∇log p_σ is exact at every noise level. Langevin dynamics at one small σ samples the right density only in
// the limit of very long runs: particles settle on the arm nearest to where they started, so the mass along the
// spiral follows the starting layout, not the data. Annealing σ from large to small first moves particles at a coarse
// scale and then refines them, which recovers the correct mass. This is the sampler of NCSN (Song and Ermon, 2019).
import { GMM } from '../lib/gmm.js';
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 960, CY = 450, S = 128;
const plot = makePlot(CX, CY, S);
const px = plot.px;
const mix = spiralMixture();
const { gmm, means: MEANS, us: US, std: STD } = mix;

const N = 500, BOX = 3;
const SIG_SINGLE = 0.05, STEPS_SINGLE = 160;
const L = 10, SIG_HI = 3, SIG_LO = 0.05, STEPS_LEVEL = 20, C_STEP = 0.3;   // step h = C_STEP σ²
const SIGS = Array.from({ length: L }, (_, i) => SIG_HI * (SIG_LO / SIG_HI) ** (i / (L - 1)));
const NB = 5;   // bins along the arm, inner to outer

const T = {
  setup: [0, 10], field: [1, 3], init: [6.5, 8],
  single: [10, 25], singleRun: [11, 21], singleHist: [21, 22.5],
  anneal: [25, 44], annealRun: [26, 40], annealHist: [40, 41.5],
  compare: [44, 56],
};

const STILLS = [
  { t: 9.5, name: 'field', label: 'Score field' },
  { t: 24.5, name: 'single', label: 'Langevin at one noise level' },
  { t: 31.5, name: 'anneal-mid', label: 'Annealed, mid-way' },
  { t: 55.4, name: 'compare', label: 'Mass along the spiral' },
];

export default {
  id: 'score-langevin',
  title: 'Score fields and annealed Langevin dynamics',
  duration: 56,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'The score' },
    { t: T.single[0], label: 'I · One noise level' },
    { t: T.anneal[0], label: 'II · Annealing' },
    { t: T.compare[0], label: 'III · Mass along the arm' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 990, h: 900 },

  setup(env) { const st = buildData(); st.glow = renderGlow(env, plot, mix); return st; },
  onTheme(env, st) { st.glow = renderGlow(env, plot, mix); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    ctx.save(); ctx.globalAlpha = 0.55; ctx.drawImage(st.glow, 0, 0, W, H); ctx.restore();

    const { sigma, pos, phase } = state(t, st);
    const fa = env.clean ? 1 : seg(t, ...T.field);
    drawField(ctx, th, st, sigma, fa * (phase === 'compare' ? 0.35 : 1));
    if (pos) {
      const pa = env.clean ? 1 : seg(t, ...T.init);
      for (let i = 0; i < N; i++) {
        const x = pos[2 * i], y = pos[2 * i + 1], k = gmm.argmaxComponent(x, y, 1);
        const [X, Y] = px(x, y);
        dot(ctx, th, X, Y, rampColor(th, US[k], 1), pa * 0.9, 2.3);
      }
    }
    if (phase !== 'compare') text(ctx, `σ = ${sigma.toFixed(2)}`, CX, 868, { size: 24, color: th.fg, font: 'mono', align: 'center' });

    const ha1 = phase === 'compare' ? 1 : window_(t, T.single[0], T.single[1], 0.5) * seg(t, ...T.singleHist);
    const ha2 = phase === 'compare' ? 1 : window_(t, T.anneal[0], T.anneal[1], 0.5) * seg(t, ...T.annealHist);
    const ca = window_(t, T.compare[0], T.compare[1] + 1, 0.6);
    if (ca > 0) drawCompare(ctx, th, st, ca);
    else {
      if (ha1 > 0) drawHist(ctx, th, st.single.hist, 1230, 640, ha1, 'one level');
      if (ha2 > 0) drawHist(ctx, th, st.anneal.hist, 1230, 640, ha2, 'annealed');
    }

    if (!env.clean) drawPanel(ctx, t, env, st, sigma, phase);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// Current noise level and particle positions as a pure function of t.
function state(t, st) {
  const at = (run, f) => {
    const n = run.frames.length - 1, x = clamp(f, 0, 1) * n, i = Math.min(Math.floor(x), n - 1), u = x - i;
    const a = run.frames[i], b = run.frames[i + 1], out = new Float32Array(2 * N);
    for (let j = 0; j < 2 * N; j++) out[j] = a[j] + u * (b[j] - a[j]);
    return { pos: out, step: x };
  };
  if (t < T.single[0]) return { sigma: 0.1, pos: t >= T.init[0] ? st.init : null, phase: 'setup' };
  if (t < T.anneal[0]) {
    const f = ease.inOut(seg(t, ...T.singleRun));
    return { sigma: SIG_SINGLE, ...at(st.single, f), phase: 'single' };
  }
  if (t < T.compare[0]) {
    const f = seg(t, ...T.annealRun), r = at(st.anneal, f);
    const lvl = Math.min(L - 1, Math.floor(r.step / STEPS_LEVEL));
    return { sigma: t < T.annealRun[0] ? SIGS[0] : SIGS[lvl], ...r, level: lvl, phase: 'anneal' };
  }
  return { sigma: SIG_LO, pos: st.anneal.frames[st.anneal.frames.length - 1], phase: 'compare' };
}

// ---------- Simulation ----------
const levelGMM = (s) => new GMM(MEANS, Math.sqrt(STD * STD + s * s));

function langevin(init, levels, stepsPer, rng) {
  const P = Float32Array.from(init), frames = [Float32Array.from(P)];
  for (const s of levels) {
    const g = levelGMM(s), h = C_STEP * s * s, sq = Math.sqrt(h);
    for (let k = 0; k < stepsPer; k++) {
      for (let i = 0; i < N; i++) {
        const sc = g.fields(P[2 * i], P[2 * i + 1], 1).score;
        P[2 * i] += 0.5 * h * sc[0] + sq * rng.gauss();
        P[2 * i + 1] += 0.5 * h * sc[1] + sq * rng.gauss();
      }
      frames.push(Float32Array.from(P));
    }
  }
  return frames;
}

function armHist(P) {
  const b = new Array(NB).fill(0);
  for (let i = 0; i < N; i++) b[Math.min(NB - 1, Math.floor(US[gmm.argmaxComponent(P[2 * i], P[2 * i + 1], 1)] * NB))] += 1 / N;
  return b;
}
const tvU = (b) => 0.5 * b.reduce((s, v) => s + Math.abs(v - 1 / NB), 0);

export function buildData() {
  const rng = makeRng(2);
  const init = new Float32Array(2 * N);
  for (let j = 0; j < 2 * N; j++) init[j] = (rng.uniform() * 2 - 1) * BOX;
  const single = { frames: langevin(init, [SIG_SINGLE], STEPS_SINGLE, makeRng(5)) };
  const anneal = { frames: langevin(init, SIGS, STEPS_LEVEL, makeRng(5)) };
  for (const r of [single, anneal]) { r.hist = armHist(r.frames[r.frames.length - 1]); r.tv = tvU(r.hist); }
  return { init, single, anneal, initHist: armHist(init) };
}

// ---------- Drawing ----------
const GRID = 23;
function drawField(ctx, th, st, sigma, a) {
  if (a <= 0) return;
  const g = levelGMM(sigma), span = 3.1, L0 = 22;
  ctx.save();
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.5;
  for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) {
    const x = -span + (2 * span * i) / (GRID - 1), y = -span + (2 * span * j) / (GRID - 1);
    const sc = g.fields(x, y, 1).score, m = Math.hypot(...sc);
    if (m < 1e-9) continue;
    // Length grows with the noise-scaled score σ‖∇log p_σ‖ and saturates.
    const len = L0 * Math.tanh(0.6 * m * Math.sqrt(STD * STD + sigma * sigma)), ux = sc[0] / m, uy = sc[1] / m;
    const [X, Y] = px(x, y), X2 = X + ux * len, Y2 = Y - uy * len;
    ctx.globalAlpha = a * (0.25 + 0.5 * len / L0);
    ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X2, Y2); ctx.stroke();
    const ang = Math.atan2(Y2 - Y, X2 - X), hl = 5;
    ctx.beginPath(); ctx.moveTo(X2, Y2);
    ctx.lineTo(X2 - hl * Math.cos(ang - 0.45), Y2 - hl * Math.sin(ang - 0.45));
    ctx.lineTo(X2 - hl * Math.cos(ang + 0.45), Y2 - hl * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fillStyle = th.muted; ctx.fill();
  }
  ctx.restore();
}

// Bar chart of the share of particles in each fifth of the spiral (inner to outer). The data puts 0.2 in each.
function drawHist(ctx, th, hist, x0, y0, a, label, w = 300, h = 150) {
  ctx.save();
  ctx.globalAlpha = a * 0.93; ctx.fillStyle = th.bg; ctx.fillRect(x0 - 40, y0 - 50, w + 64, h + 100);
  ctx.globalAlpha = a; ctx.strokeStyle = th.faint; ctx.strokeRect(x0 - 40, y0 - 50, w + 64, h + 100);
  const ymax = 0.4, bw = w / NB;
  for (let i = 0; i < NB; i++) {
    const bh = (hist[i] / ymax) * h;
    ctx.fillStyle = rampColor(th, (i + 0.5) / NB, 1);
    ctx.globalAlpha = a * 0.85;
    ctx.fillRect(x0 + i * bw + 6, y0 + h - bh, bw - 12, bh);
  }
  ctx.globalAlpha = a; ctx.strokeStyle = th.fg; ctx.lineWidth = 1.6; ctx.setLineDash([5, 5]);
  const yt = y0 + h - (0.2 / ymax) * h;
  ctx.beginPath(); ctx.moveTo(x0, yt); ctx.lineTo(x0 + w, yt); ctx.stroke();
  ctx.setLineDash([]); ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(x0, y0 + h); ctx.lineTo(x0 + w, y0 + h); ctx.stroke();
  ctx.restore();
  text(ctx, label, x0, y0 - 24, { size: 16, weight: 600, color: th.fg, alpha: a });
  text(ctx, 'data', x0 + w + 4, yt + 5, { size: 13, color: th.fg, alpha: a });
  text(ctx, 'inner', x0 + 6, y0 + h + 22, { size: 13, color: th.muted, alpha: a });
  text(ctx, 'outer', x0 + w - 6, y0 + h + 22, { size: 13, color: th.muted, alpha: a, align: 'right' });
  text(ctx, `TV ${tvU(hist).toFixed(2)}`, x0 + w, y0 - 24, { size: 16, weight: 600, color: th.muted, alpha: a, align: 'right', font: 'mono' });
}

function drawCompare(ctx, th, st, a) {
  drawHist(ctx, th, st.single.hist, 1230, 120, a, 'one level, σ = 0.05');
  drawHist(ctx, th, st.anneal.hist, 1230, 640, a, 'annealed, 10 levels');
}

// ---------- Text panel ----------
const COPY = [
  { span: T.setup, act: 'The score',
    body: 'The score of a density is the gradient of its log, so it points toward regions of higher probability. Langevin dynamics follows the score and adds noise, and for small enough steps it samples the density. The arrows show the score of the data smoothed with noise of scale σ.',
    eq: 'x ← x + (h/2) ∇log p_σ(x) + √h z' },
  { span: T.single, act: 'Act I · One noise level',
    body: 'Started from a uniform layout, Langevin dynamics at a single small σ makes each particle settle on the arm nearest to its start. Crossing to another arm requires passing through regions of almost zero density, which takes far longer than any practical run. The mass along the spiral follows the starting layout.',
    eq: '' },
  { span: T.anneal, act: 'Act II · Annealing the noise',
    body: 'Annealed Langevin dynamics runs a few steps at each of a decreasing sequence of noise levels. At large σ the smoothed density has one broad bump and particles spread over it in proportion to the data. Each smaller level refines the previous one, so the particles end on the arms with the correct mass.',
    eq: 'σ₁ > σ₂ > … > σ_L,  h_i = c σ_i²' },
  { span: T.compare, act: 'Act III · The link to diffusion',
    body: 'Up to a rescaling, the smoothed densities p_σ are the marginals of a diffusion model at different times, and the score at each level follows from the denoiser by Tweedie\'s formula. Diffusion sampling is a principled version of this annealing, with the noise level decreasing continuously.',
    eq: '∇log p_t(x) = (α_t E[x₁ | x_t = x] − x) / σ_t²' },
];

function drawPanel(ctx, t, env, st, sigma, phase) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §2 FUNDAMENTALS', 'Scores and Langevin', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: c.eq.length > 34 ? 21 : 24, color: th.fg, alpha: a, font: 'serif' });
  }
  const read = (lab, v, col, xx, al) => {
    text(ctx, lab, xx, 690, { size: 15, weight: 600, color: th.muted, alpha: al, spacing: 1.5 });
    text(ctx, v, xx, 750, { size: 48, weight: 700, color: col, alpha: al, font: 'mono' });
  };
  const sa = window_(t, T.single[0], T.single[1], 0.4);
  if (sa > 0) {
    const step = Math.round(ease.inOut(seg(t, ...T.singleRun)) * STEPS_SINGLE);
    read('STEP', String(step), th.fg, x, sa);
    read('NOISE LEVEL σ', SIG_SINGLE.toFixed(2), th.ramp[3], x + 200, sa);
  }
  const aa = window_(t, T.anneal[0], T.anneal[1], 0.4);
  if (aa > 0) {
    const f = seg(t, ...T.annealRun), lvl = Math.min(L - 1, Math.floor(f * L));
    read('LEVEL', `${lvl + 1} / ${L}`, th.fg, x, aa);
    read('NOISE LEVEL σ', sigma.toFixed(2), th.ramp[3], x + 200, aa);
  }
  paragraph(ctx, `Exact Gaussian-mixture scores at every noise level. ${N} particles from a uniform start, step h = ${C_STEP} σ². One level: ${STEPS_SINGLE} steps at σ = ${SIG_SINGLE}. Annealed: ${L} levels from ${SIG_HI} to ${SIG_LO}, ${STEPS_LEVEL} steps each. Colour marks the nearest spiral component.`, x, 806, w, { size: 14, color: th.muted, lineHeight: 20 });
}
