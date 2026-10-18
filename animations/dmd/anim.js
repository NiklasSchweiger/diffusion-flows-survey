// A11 · Distribution matching distillation.
// The real score is the exact score of the noised spiral mixture. The fake score is the exact score of a Gaussian
// kernel density over the current generator samples (one component per sample), so it is also closed form at every
// noise level. The generator is 500 free particles. Each DMD step draws t, noises each particle, and moves it by the
// difference of the two denoisers at the noisy point, which is the score difference times sigma_t^2 / alpha_t.
import { makeRng } from '../lib/rng.js';
import { GMM } from '../lib/gmm.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, arrow, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const plot = makePlot(1070, 450, 150);
const px = (x, y) => plot.px(x, y);

const N = 500;          // generator particles
const IT = 300;         // DMD iterations
const ETA = 0.5;        // step size on the denoiser difference
const BW = 0.06;        // kernel bandwidth of the fake density
const TLO = 0.02, THI = 0.9;
const IT_REAL = 300;    // iterations of the real-score-only baseline
const SIG_D = 0.2;      // noise level (in data units) at which the fields are drawn
const EVERY = 10;       // metric cadence
const NDATA = 500;

const T = {
  setup: [0, 8], blobIn: [1.0, 2.6],
  fields: [8, 22], realIn: [9.0, 10.6], fakeIn: [12.6, 14.2], diffIn: [16.4, 18.0],
  real: [22, 31], collapse: [23.2, 27.4],
  dmd: [31, 50], run: [32.2, 47.6],
  chart: [50, 63], chartDraw: [51.2, 56.4],
};

const STILLS = [
  { t: 21.4, name: 'score-fields', label: 'Real and fake scores' },
  { t: 30.4, name: 'real-score-only', label: 'Real score only' },
  { t: 49.4, name: 'dmd', label: 'Distribution matching' },
  { t: 62.4, name: 'mmd', label: 'Distance to the data' },
];

// ---------- Dynamics ----------

function nearestStats(mix, X, n) {
  let on = 0;
  const hit = new Uint8Array(mix.means.length);
  for (let i = 0; i < n; i++) {
    const x = X[2 * i], y = X[2 * i + 1];
    let best = 1e9, bk = 0;
    mix.means.forEach((m, k) => { const d = (x - m[0]) ** 2 + (y - m[1]) ** 2; if (d < best) { best = d; bk = k; } });
    if (Math.sqrt(best) < 2 * mix.std) { on++; hit[bk] = 1; }
  }
  return { on: on / n, cov: hit.reduce((a, b) => a + b, 0) };
}

const KM = 0.15; // MMD kernel width
function kmean(A, na, B, nb) {
  let s = 0;
  for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++) {
    const dx = A[2 * i] - B[2 * j], dy = A[2 * i + 1] - B[2 * j + 1];
    s += Math.exp(-(dx * dx + dy * dy) / (2 * KM * KM));
  }
  return s / (na * nb);
}
const mmd = (A, B, kBB) => Math.sqrt(Math.max(0, kmean(A, N, A, N) + kBB - 2 * kmean(A, N, B, NDATA)));

const toPts = (F, n) => Array.from({ length: n }, (_, i) => [F[2 * i], F[2 * i + 1]]);

export default {
  id: 'dmd',
  title: 'Distribution matching distillation',
  duration: 63,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.fields[0], label: 'I · Two score fields' },
    { t: T.real[0], label: 'II · Real score alone' },
    { t: T.dmd[0], label: 'III · Distribution matching' },
    { t: T.chart[0], label: 'IV · Distance to the data' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 1000, h: 900 },

  setup(env) {
    const mix = spiralMixture(), real = mix.gmm;
    let mx = 0, my = 0;
    mix.means.forEach((m) => { mx += m[0] / mix.means.length; my += m[1] / mix.means.length; });

    const rng = makeRng(1);
    const X0 = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) { X0[2 * i] = mx + 0.35 * rng.gauss(); X0[2 * i + 1] = my + 0.35 * rng.gauss(); }

    // DMD: noise each sample at a random level and move it by the denoiser difference.
    const D = new Float32Array((IT + 1) * 2 * N);
    D.set(X0, 0);
    for (let it = 0; it < IT; it++) {
      const cur = D.subarray(it * 2 * N, (it + 1) * 2 * N), nxt = D.subarray((it + 1) * 2 * N, (it + 2) * 2 * N);
      const fake = new GMM(toPts(cur, N), BW);
      for (let i = 0; i < N; i++) {
        const x = cur[2 * i], y = cur[2 * i + 1], t = TLO + (THI - TLO) * rng.uniform();
        const xt = t * x + (1 - t) * rng.gauss(), yt = t * y + (1 - t) * rng.gauss();
        const dr = real.fields(xt, yt, t).d1, df = fake.fields(xt, yt, t).d1;
        nxt[2 * i] = x + ETA * (dr[0] - df[0]); nxt[2 * i + 1] = y + ETA * (dr[1] - df[1]);
      }
    }

    // Baseline: follow the real score only (a denoiser step at a fixed small noise level).
    const R = new Float32Array((IT_REAL + 1) * 2 * N);
    R.set(X0, 0);
    for (let it = 0; it < IT_REAL; it++) {
      const o = it * 2 * N;
      for (let i = 0; i < N; i++) {
        const x = R[o + 2 * i], y = R[o + 2 * i + 1], d = real.fields(0.9 * x, 0.9 * y, 0.9).d1;
        R[o + 2 * N + 2 * i] = x + ETA * (d[0] - x); R[o + 2 * N + 2 * i + 1] = y + ETA * (d[1] - y);
      }
    }

    // Metrics against held-out data samples.
    const dr = makeRng(7), DATA = new Float32Array(2 * NDATA), DATA2 = new Float32Array(2 * N);
    for (let i = 0; i < NDATA; i++) { const s = real.sample(dr); DATA[2 * i] = s.x; DATA[2 * i + 1] = s.y; }
    for (let i = 0; i < N; i++) { const s = real.sample(dr); DATA2[2 * i] = s.x; DATA2[2 * i + 1] = s.y; }
    const kBB = kmean(DATA, NDATA, DATA, NDATA);
    const metric = (S, it) => {
      const A = S.subarray(it * 2 * N, (it + 1) * 2 * N);
      return { it, mmd: mmd(A, DATA, kBB), ...nearestStats(mix, A, N) };
    };
    const curveD = [], curveR = [];
    for (let it = 0; it <= IT; it += EVERY) curveD.push(metric(D, it));
    for (let it = 0; it <= IT_REAL; it += EVERY) curveR.push(metric(R, it));
    const floor = mmd(DATA2, DATA, kBB);

    // Real field on a grid, smoothed at the display noise level (data coordinates).
    const realS = new GMM(mix.means, Math.hypot(mix.std, SIG_D));
    const grid = [];
    for (let gx = -2.7; gx <= 2.71; gx += 0.36) for (let gy = -2.52; gy <= 1.81; gy += 0.36) {
      grid.push({ x: gx, y: gy, r: disp(realS, gx, gy) });
    }

    return { mix, D, R, curveD, curveR, floor, realS, grid, cache: new Map(), glow: renderGlow(env, plot, mix) };
  },

  onTheme(env, st) { st.glow = renderGlow(env, plot, st.mix); },

  draw(ctx, t, env, st) {
    const th = env.theme, clean = env.clean;
    drawGrid(ctx, th, plot, W, H);
    const chartA = seg(t, T.chart[0], T.chart[0] + 0.7);
    if (chartA < 1) {
      ctx.save(); ctx.globalAlpha = 1 - chartA; ctx.drawImage(st.glow, 0, 0, W, H); ctx.restore();
      drawScene(ctx, th, t, st, 1 - chartA, clean);
    }
    if (chartA > 0) drawChart(ctx, th, t, chartA, st);
    if (!clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 70, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Scenes ----------

const RED = (th) => (th.name === 'dark' ? '#ff6b6b' : '#dc2626');

// Iteration shown at time t in the DMD act (slow start, where most of the motion happens).
const dmdIter = (t) => IT * ease.inOut(seg(t, ...T.run)) ** 1.6;
const realIter = (t) => 40 * ease.inOut(seg(t, ...T.collapse));

function posAt(S, itf, i) {
  const a = Math.floor(itf), b = Math.min(a + 1, S.length / (2 * N) - 1), w = itf - a;
  return [lerp(S[a * 2 * N + 2 * i], S[b * 2 * N + 2 * i], w), lerp(S[a * 2 * N + 2 * i + 1], S[b * 2 * N + 2 * i + 1], w)];
}

// Fake field of iteration `it`, smoothed at the display noise level. Cached; it depends only on `it`.
function fakeAt(st, it) {
  let f = st.cache.get(it);
  if (!f) {
    f = new GMM(toPts(st.D.subarray(it * 2 * N, (it + 1) * 2 * N), N), Math.hypot(BW, SIG_D));
    if (st.cache.size > 400) st.cache.clear();
    st.cache.set(it, f);
  }
  return f;
}

// Tweedie displacement at the display noise level: E[clean | noisy = x] − x = σ² ∇log p_σ(x).
// The smoothed densities are evaluated at t = 1 of a mixture whose width already includes σ.
function disp(g, x, y) { const s = g.fields(x, y, 1).score; return [SIG_D * SIG_D * s[0], SIG_D * SIG_D * s[1]]; }

function fieldArrow(ctx, x, y, d, k, color, alpha, width = 1.6, cap = 0.3) {
  const L = Math.hypot(d[0], d[1]) * k, c = L > cap ? cap / L : 1;
  const [x0, y0] = px(x, y), [x1, y1] = px(x + c * k * d[0], y + c * k * d[1]);
  if (Math.hypot(x1 - x0, y1 - y0) < 3) return;
  arrow(ctx, x0, y0, x1, y1, { color, width, head: 6, alpha });
}

function drawScene(ctx, th, t, st, A, clean) {
  const sA = window_(t, T.setup[0], T.setup[1], 0.5), fA = window_(t, ...T.fields, 0.5);
  const rA = window_(t, ...T.real, 0.5), dA = window_(t, ...T.dmd, 0.5);
  const cR = rampColor(th, 0.0, 1), cF = rampColor(th, 1, 1), cD = RED(th);

  // Setup and act I: the initial blob.
  if (sA > 0 || fA > 0) {
    const a = A * Math.max(clean ? sA : sA * seg(t, ...T.blobIn), fA);
    for (let i = 0; i < N; i++) { const [x, y] = px(st.D[2 * i], st.D[2 * i + 1]); dot(ctx, th, x, y, th.fg, a * 0.8); }
  }

  if (fA > 0) {
    const ra = A * fA * seg(t, ...T.realIn), fa = A * fA * seg(t, ...T.fakeIn), da = A * fA * seg(t, ...T.diffIn);
    const fake = fakeAt(st, 0), K = 0.9;
    for (const g of st.grid) {
      if (ra > 0) fieldArrow(ctx, g.x, g.y, g.r, K, cR, ra * (0.75 - 0.45 * da), 1.6);
      if (fa > 0) {
        fieldArrow(ctx, g.x, g.y, disp(fake, g.x, g.y), K, cF, fa * (0.75 - 0.45 * da), 1.6);
      }
    }
    if (da > 0) drawDiff(ctx, th, st, 0, 0, da, cD);
    legend(ctx, th, [[cR, 'real score (teacher)', ra], [cF, 'fake score (generator samples)', fa], [cD, 'difference, the update', da]]);
  }

  // Act II: real score only.
  if (rA > 0) {
    const a = A * rA, itf = realIter(t);
    for (let i = 0; i < N; i++) { const [x, y] = px(...posAt(st.R, itf, i)); dot(ctx, th, x, y, th.fg, a * 0.85); }
    const ca = a * seg(t, T.collapse[1], T.collapse[1] + 0.8);
    const last = st.curveR[4];
    text(ctx, `${last.cov} of 110 modes reached`, 1430, 820, { size: 20, weight: 600, color: th.fg, align: 'right', alpha: ca });
  }

  // Act III: DMD iterations.
  if (dA > 0) {
    const a = A * dA, itf = dmdIter(t), it = Math.round(itf);
    for (let i = 0; i < N; i++) { const [x, y] = px(...posAt(st.D, itf, i)); dot(ctx, th, x, y, th.fg, a * 0.85); }
    drawDiff(ctx, th, st, it, itf, a * 0.9, cD);
    text(ctx, `DMD step ${String(it).padStart(3, ' ')} / ${IT}`, 1430, 820, { size: 20, weight: 600, color: th.fg, align: 'right', font: 'mono', alpha: a });
    legend(ctx, th, [[cD, 'score difference at each sample', a]]);
  }
}

// Difference of the two smoothed denoisers at each sample: where the update pushes it.
function drawDiff(ctx, th, st, it, itf, a, color) {
  const fake = fakeAt(st, it), K = 1.3;
  for (let i = 0; i < N; i += 2) {
    const [x, y] = posAt(st.D, itf, i), r = disp(st.realS, x, y), f = disp(fake, x, y);
    fieldArrow(ctx, x, y, [r[0] - f[0], r[1] - f[1]], K, color, a, 1.4);
  }
}

function legend(ctx, th, rows) {
  rows.forEach(([c, lab, a], k) => {
    if (a <= 0) return;
    const y = 48 + 30 * k;
    arrow(ctx, 1160, y - 6, 1196, y - 6, { color: c, width: 2.4, head: 7, alpha: a });
    text(ctx, lab, 1208, y, { size: 17, color: th.fg, alpha: a });
  });
}

function drawChart(ctx, th, t, alpha, st) {
  const L0 = 780, R0 = 1460, Tp = 170, B = 690, ymax = 0.5;
  const X = (it) => L0 + (it / IT) * (R0 - L0), Y = (v) => B - (v / ymax) * (B - Tp);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L0, Tp - 10); ctx.lineTo(L0, B); ctx.lineTo(R0 + 10, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const v of [0.1, 0.2, 0.3, 0.4, 0.5]) { ctx.beginPath(); ctx.moveTo(L0, Y(v)); ctx.lineTo(R0, Y(v)); ctx.stroke(); }
  ctx.setLineDash([6, 6]); ctx.strokeStyle = th.muted;
  ctx.beginPath(); ctx.moveTo(L0, Y(st.floor)); ctx.lineTo(R0, Y(st.floor)); ctx.stroke();
  ctx.restore();
  text(ctx, 'two independent data sets of 500', L0 + 40, Y(st.floor) + 26, { size: 15, color: th.muted, alpha });
  for (const v of [0, 0.1, 0.2, 0.3, 0.4, 0.5]) text(ctx, v.toFixed(1), L0 - 14, Y(v) + 6, { size: 17, color: th.muted, align: 'right', alpha });
  for (const it of [0, 100, 200, 300]) text(ctx, String(it), X(it), B + 28, { size: 17, color: th.muted, align: 'center', alpha });
  text(ctx, 'update steps', (L0 + R0) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha });
  text(ctx, 'MMD between the 500 samples and the data', L0 - 60, Tp - 34, { size: 18, color: th.muted, alpha });

  const grow = ease.inOut(seg(t, ...T.chartDraw));
  const series = [
    { pts: st.curveR, color: rampColor(th, 0.05, 1) },
    { pts: st.curveD, color: rampColor(th, 1, 1) },
  ];
  for (const s of series) {
    const f = grow * (s.pts.length - 1);
    ctx.save();
    ctx.globalAlpha = alpha; ctx.strokeStyle = ctx.fillStyle = s.color; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(X(s.pts[0].it), Y(s.pts[0].mmd));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1), p = s.pts[i - 1], q = s.pts[i];
      ctx.lineTo(lerp(X(p.it), X(q.it), w), lerp(Y(p.mmd), Y(q.mmd), w));
    }
    ctx.stroke();
    for (let i = 0; i <= Math.floor(f); i += 5) { ctx.beginPath(); ctx.arc(X(s.pts[i].it), Y(s.pts[i].mmd), 5.5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
  }
  const la = alpha * seg(grow, 0.85, 1), r = st.curveR.at(-1), d = st.curveD.at(-1);
  text(ctx, 'real score only', X(170), Y(st.curveR[17].mmd) + 34, { size: 19, weight: 700, color: rampColor(th, 0.05, 1), alpha: la });
  text(ctx, 'distribution matching (DMD)', X(120), Y(st.curveD[12].mmd) - 30, { size: 19, weight: 700, color: rampColor(th, 1, 1), alpha: la });
  text(ctx, r.mmd.toFixed(3), X(IT) + 14, Y(r.mmd) + 6, { size: 15, color: th.fg, alpha: la, font: 'mono' });
  text(ctx, d.mmd.toFixed(3), X(IT) + 14, Y(d.mmd) + 6, { size: 15, color: th.fg, alpha: la, font: 'mono' });
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup · A one-step generator',
    body: 'Distribution matching distillation trains a generator that turns noise into a sample in a single step. Its samples should follow the data, shown as the glow. Here the generator is 500 free particles that start as a blob, so every particle is its own parameter.',
    eq: 'x = G(z; θ)' },
  { span: T.fields, act: 'Act I · Two score fields',
    body: 'Blur both distributions with the same Gaussian noise. The real score (cyan) points toward the data and comes from the pretrained teacher. The fake score (amber) points toward the generator samples, and a second diffusion model learns it online. Their difference (red) is the direction each sample should move.',
    eq: 'Δx ∝ E[ w(t) (sʳᵉᵃˡ − sᶠᵃᵏᵉ)(xₜ, t) ]' },
  { span: T.real, act: 'Act II · Real score alone',
    body: 'Without the fake score, each sample simply climbs the teacher density, as in score distillation sampling. Every sample lands on the spiral, but only on the peaks closest to where it started. Most of the spiral stays empty. This is mode collapse.',
    eq: 'Δx ∝ sʳᵉᵃˡ(x)' },
  { span: T.dmd, act: 'Act III · Distribution matching',
    body: 'Each step draws a noise level, noises every sample, and moves it by the score difference at the noisy point. The fake score is then refit to the moved samples. Where samples pile up, the fake score pushes them apart, so they spread along the whole arm. The arrows shrink as the two distributions agree.',
    eq: 'xₜ = t x + (1 − t) ε,  t ~ U(0.02, 0.9)' },
  { span: T.chart, act: 'Act IV · Distance to the data',
    body: 'Maximum mean discrepancy between the samples and held-out data. Climbing the real score alone ends further from the data than the starting blob, because collapse removes the spread the data has. Distribution matching reaches the level of two independent data sets of the same size, which is as close as 500 samples can get.',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §5 DISTILLATION', 'Distribution matching', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 24, color: th.fg, alpha: a, font: 'serif' });
  }

  // Readouts: real score only vs DMD, at the iteration on screen.
  const ra = seg(t, T.real[0] + 0.6, T.real[0] + 1.4);
  if (ra > 0) {
    const y0 = 660;
    const dA = seg(t, T.dmd[0], T.dmd[0] + 0.8);
    const r = t >= T.chart[0] ? st.curveR.at(-1) : st.curveR[Math.round(realIter(t) / EVERY)];
    const d = st.curveD[Math.round((t >= T.chart[0] ? IT : dmdIter(t)) / EVERY)];
    text(ctx, 'real score only', x + 270, y0, { size: 15, weight: 600, color: th.muted, align: 'right', alpha: ra });
    text(ctx, 'DMD', x + 400, y0, { size: 15, weight: 600, color: th.muted, align: 'right', alpha: dA });
    const rows = [
      ['samples on the spiral', `${Math.round(100 * r.on)}%`, `${Math.round(100 * d.on)}%`],
      ['modes covered (of 110)', String(r.cov), String(d.cov)],
      ['MMD to data', r.mmd.toFixed(3), d.mmd.toFixed(3)],
    ];
    rows.forEach(([lab, a1, a2], k) => {
      const y = y0 + 36 + 34 * k;
      text(ctx, lab, x, y, { size: 16, color: th.fg, alpha: ra });
      text(ctx, a1, x + 270, y, { size: 17, color: th.fg, align: 'right', font: 'mono', alpha: ra });
      text(ctx, a2, x + 400, y, { size: 17, weight: 700, color: th.accent, align: 'right', font: 'mono', alpha: dA });
    });
  }
  text(ctx, 'Exact spiral-mixture score. The fake score is a kernel density over the samples; the generator is free particles.', x, 862, { size: 14, color: th.muted });
}
