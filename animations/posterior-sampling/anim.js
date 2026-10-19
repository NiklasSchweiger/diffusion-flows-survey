// A14 · Posterior sampling for an inverse problem: exact conditional flow vs DPS.
// Prior: the spiral mixture. Measurement: y = x₁ + n with n ~ N(0, σ_y²), i.e. we observe only the horizontal
// coordinate. For a Gaussian mixture prior and a linear Gaussian measurement the posterior is again a Gaussian
// mixture (reweighted components, shrunk along x₁), and since x_t = t x1 + (1 − t) x0 with x0 independent of y, the
// conditional probability path is that posterior's own path. Its velocity is exact, so the reference samples are
// true posterior samples. DPS instead adds −ζ ∇‖y − A x̂₁(x_t)‖ to the unconditional flow, where x̂₁ = E[x1 | x_t] is
// the exact denoiser of the prior and its Jacobian is taken by finite differences.
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp, rampColor } from '../lib/theme.js';
import { trajAt } from '../lib/gmm.js';

const W = 1600, H = 900;
const CX = 1000, CY = 455, S = 150;
const plot = makePlot(CX, CY, S);
const px = plot.px;
const mix = spiralMixture();
const { gmm, means: MEANS, us: US, std: STD } = mix;
const K = MEANS.length, S2 = STD * STD;

const Y0 = -1.2, SY = 0.06;          // observed x₁ and its noise
const ZETA = 10;                      // DPS step size in Acts II
const ZS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30];
const STEPS = 100, NS = 300, NT = 36, NSWEEP = 200;

const T = {
  setup: [0, 9], band: [3, 4.4],
  exact: [9, 21], exactRun: [10, 16.5], exactHist: [16.8, 18],
  dps: [21, 34], dpsRun: [22, 28.5], dpsHist: [28.8, 30],
  sweep: [34, 52], sweepRun: [35.5, 49],
};

const STILLS = [
  { t: 8.5, name: 'measurement', label: 'Prior and measurement' },
  { t: 20.5, name: 'exact', label: 'Exact posterior samples' },
  { t: 33.5, name: 'dps', label: `DPS, ζ = ${ZETA}` },
  { t: 51.4, name: 'sweep', label: 'DPS across step sizes' },
];

export default {
  id: 'posterior-sampling',
  title: 'Posterior sampling for an inverse problem',
  duration: 52,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.exact[0], label: 'I · Exact posterior' },
    { t: T.dps[0], label: 'II · DPS' },
    { t: T.sweep[0], label: 'III · Step size' },
  ],
  stills: STILLS,
  crop: { x: 620, y: 0, w: 970, h: 900 },

  setup(env) { const st = buildData(); st.glow = renderGlow(env, plot, mix); return st; },
  onTheme(env, st) { st.glow = renderGlow(env, plot, mix); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    ctx.save();
    ctx.globalAlpha = 1 - 0.55 * seg(t, T.exact[0], T.exact[0] + 1);
    ctx.drawImage(st.glow, 0, 0, W, H);
    ctx.restore();

    // Prior samples in the setup.
    const pa = window_(t, -1, T.exact[0] + 0.6, 0.6) * (env.clean ? 1 : seg(t, 0.3, 1.5));
    if (pa > 0) for (const [x, y, k] of st.prior) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, rampColor(th, US[k], 1), pa * 0.7, 1.8); }

    const ba = env.clean ? 1 : ease.out(seg(t, ...T.band));
    drawBand(ctx, th, ba);
    drawPosteriorGlow(ctx, th, st, seg(t, T.exact[0], T.exact[0] + 1.2));

    const ea = window_(t, T.exact[0], T.exact[1], 0.5);
    if (ea > 0) drawRun(ctx, th, st.exact, seg(t, ...T.exactRun), ea, th.accent);
    const da = window_(t, T.dps[0], T.dps[1], 0.5);
    if (da > 0) drawRun(ctx, th, st.dps, seg(t, ...T.dpsRun), da, th.ramp[3]);

    if (t >= T.exact[0] && t < T.sweep[0]) {
      const ha = t < T.dps[0] ? ea * seg(t, ...T.exactHist) : da;
      const samples = t < T.dps[0] ? null : st.dps.end;
      drawHist(ctx, th, st, samples, ha, t < T.dps[0] ? 0 : seg(t, ...T.dpsHist));
    }
    const sa = window_(t, T.sweep[0], T.sweep[1] + 1, 0.5);
    if (sa > 0) drawSweep(ctx, th, st, t, sa);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Exact posterior and DPS ----------

// Posterior components: weight ∝ w_k N(y; μ_k1, s² + σ_y²), x₁ mean and variance updated, x₂ unchanged.
const POST = (() => {
  const lw = MEANS.map((m) => -((Y0 - m[0]) ** 2) / (2 * (S2 + SY * SY)));
  const mx = Math.max(...lw), w = lw.map((l) => Math.exp(l - mx)), z = w.reduce((a, b) => a + b);
  const g = S2 / (S2 + SY * SY);
  return MEANS.map((m, k) => ({ w: w[k] / z, mu: [m[0] + g * (Y0 - m[0]), m[1]], v: [S2 * SY * SY / (S2 + SY * SY), S2] }));
})();

function postVel(x, y, t) {
  const a = t, sg = 1 - t, P = [x, y], lg = new Float64Array(K);
  let mx = -Infinity;
  for (let k = 0; k < K; k++) {
    const p = POST[k];
    if (p.w < 1e-9) { lg[k] = -Infinity; continue; }
    let l = Math.log(p.w);
    for (let d = 0; d < 2; d++) { const V = a * a * p.v[d] + sg * sg; l -= (P[d] - a * p.mu[d]) ** 2 / (2 * V) + 0.5 * Math.log(V); }
    lg[k] = l;
    if (l > mx) mx = l;
  }
  let z = 0;
  for (let k = 0; k < K; k++) z += (lg[k] = Math.exp(lg[k] - mx));
  const u = [0, 0];
  for (let k = 0; k < K; k++) {
    const gk = lg[k] / z;
    if (gk < 1e-10) continue;
    const p = POST[k];
    for (let d = 0; d < 2; d++) {
      const V = a * a * p.v[d] + sg * sg, r = P[d] - a * p.mu[d];
      u[d] += gk * (p.mu[d] + (a * p.v[d] / V) * r - (sg / V) * r);
    }
  }
  return u;
}

function exactPath(x, y) {
  const out = new Float32Array((STEPS + 1) * 2), h = 1 / STEPS;
  out[0] = x; out[1] = y;
  for (let i = 0; i < STEPS; i++) {
    const t = i * h, a = postVel(x, y, t), b = postVel(x + h * a[0], y + h * a[1], t + h);
    x += 0.5 * h * (a[0] + b[0]); y += 0.5 * h * (a[1] + b[1]);
    out[2 * i + 2] = x; out[2 * i + 3] = y;
  }
  return out;
}

// DPS with the normalized gradient: x ← x + h u(x, t) − ζ h ∇‖y − x̂₁,₁(x)‖.
function dpsPath(x, y, zeta) {
  const out = new Float32Array((STEPS + 1) * 2), h = 1 / STEPS, e = 1e-4;
  out[0] = x; out[1] = y;
  for (let i = 0; i < STEPS; i++) {
    const t = i * h, f = gmm.fields(x, y, t), r = Y0 - f.d1[0];
    const jx = (gmm.fields(x + e, y, t).d1[0] - f.d1[0]) / e, jy = (gmm.fields(x, y + e, t).d1[0] - f.d1[0]) / e;
    const s = Math.sign(r);
    x += h * f.u[0] + zeta * h * s * jx;
    y += h * f.u[1] + zeta * h * s * jy;
    out[2 * i + 2] = x; out[2 * i + 3] = y;
  }
  return out;
}

const NB = 36, YLO = -2.7, YHI = 2.7;
const binOf = (y) => Math.floor(((y - YLO) / (YHI - YLO)) * NB);
function hist(end) {
  const b = new Float64Array(NB);
  for (const [, y] of end) { const i = binOf(y); if (i >= 0 && i < NB) b[i] += 1 / end.length; }
  return b;
}
// Exact posterior mass per x₂ bin (the x₂ marginal of each component is N(μ_k2, s²)).
function exactHist() {
  const b = new Float64Array(NB), erf = (x) => { const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)); };
  const cdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  for (let i = 0; i < NB; i++) {
    const lo = YLO + (i / NB) * (YHI - YLO), hi = lo + (YHI - YLO) / NB;
    for (const p of POST) b[i] += p.w * (cdf((hi - p.mu[1]) / STD) - cdf((lo - p.mu[1]) / STD));
  }
  return b;
}
const tv = (a, b) => 0.5 * a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
const inBand = (end) => end.filter(([x]) => Math.abs(x - Y0) < 2 * SY).length / end.length;

export function buildData() {
  const rng = makeRng(3);
  const noise = Array.from({ length: NS }, () => [rng.gauss(), rng.gauss()]);
  const prng = makeRng(8);
  const prior = Array.from({ length: 700 }, () => { const s = gmm.sample(prng); return [s.x, s.y, s.k]; });
  const run = (f) => {
    const paths = noise.map(([x, y], i) => (i < NT ? f(x, y) : null));
    const end = noise.map(([x, y], i) => { const p = paths[i] ?? f(x, y); return [p[2 * STEPS], p[2 * STEPS + 1]]; });
    return { paths: paths.filter(Boolean), end };
  };
  const exact = run(exactPath), dps = run((x, y) => dpsPath(x, y, ZETA));
  const ref = exactHist();
  exact.tv = tv(hist(exact.end), ref); exact.band = inBand(exact.end);
  dps.tv = tv(hist(dps.end), ref); dps.band = inBand(dps.end);
  const sweep = ZS.map((z) => {
    const end = noise.slice(0, NSWEEP).map(([x, y]) => { const p = z === ZETA ? null : dpsPath(x, y, z); return p ? [p[2 * STEPS], p[2 * STEPS + 1]] : null; });
    const e = z === ZETA ? dps.end.slice(0, NSWEEP) : end;
    return { z, end: e, tv: tv(hist(e), ref), band: inBand(e) };
  });
  // Mass the exact posterior and DPS put on the middle arm (|x₂| < 1).
  const mid = (end) => end.filter(([, y]) => Math.abs(y) < 1).length / end.length;
  let midExact = 0;
  for (const p of POST) if (Math.abs(p.mu[1]) < 1) midExact += p.w;
  return { prior, exact, dps, sweep, ref, midExact, midDps: mid(dps.end) };
}

// ---------- Drawing ----------

function drawBand(ctx, th, a) {
  if (a <= 0) return;
  const [x] = px(Y0, 0), w = 2 * SY * S;
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.name === 'dark' ? 'rgba(255,181,71,0.10)' : 'rgba(217,119,6,0.09)';
  ctx.fillRect(x - w, 40, 2 * w, H - 80);
  ctx.strokeStyle = th.ramp[3]; ctx.lineWidth = 1.4; ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x, H - 40); ctx.stroke();
  ctx.restore();
  text(ctx, `observed x₁ = ${Y0}`, x + 12, 112, { size: 19, weight: 600, color: th.ramp[3], alpha: a, font: 'serif' });
}

function drawPosteriorGlow(ctx, th, st, a) {
  if (a <= 0) return;
  for (const p of POST) {
    if (p.w < 0.004) continue;
    const [x, y] = px(...p.mu), r = 14 + 60 * Math.sqrt(p.w);
    ctx.save();
    ctx.globalAlpha = a * 0.9;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, th.name === 'dark' ? 'rgba(125,211,252,0.35)' : 'rgba(3,105,161,0.22)');
    g.addColorStop(1, th.name === 'dark' ? 'rgba(125,211,252,0)' : 'rgba(3,105,161,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
}

function drawRun(ctx, th, run, p, a, color) {
  const tau = ease.inOut(p);
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 1.3; ctx.globalAlpha = a * 0.45;
  for (const path of run.paths) {
    const n = Math.floor(tau * STEPS);
    ctx.beginPath();
    for (let i = 0; i <= n; i++) { const [X, Y] = px(path[2 * i], path[2 * i + 1]); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
    const [X, Y] = px(...trajAt(path, tau)); ctx.lineTo(X, Y);
    ctx.stroke();
  }
  ctx.restore();
  // Endpoints of all samples appear as the run finishes; heads of the shown paths move.
  for (const path of run.paths) { const [X, Y] = px(...trajAt(path, tau)); dot(ctx, th, X, Y, color, a, 2.6); }
  const ea = a * seg(p, 0.9, 1);
  if (ea > 0) for (const [x, y] of run.end) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, color, ea * 0.85, 2.2); }
}

// Histogram of x₂ along the right edge, aligned with the plot's vertical axis.
const HX = 1430, HW = 150;
function drawHist(ctx, th, st, samples, a, dpsA) {
  if (a <= 0) return;
  const max = Math.max(...st.ref) * 1.15, bh = ((YHI - YLO) / NB) * S;
  text(ctx, 'posterior over x₂', HX + HW, 112, { size: 17, color: th.muted, alpha: a, align: 'right' });
  ctx.save();
  ctx.globalAlpha = a;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(HX, px(0, YHI)[1]); ctx.lineTo(HX, px(0, YLO)[1]); ctx.stroke();
  for (let i = 0; i < NB; i++) {
    const y = px(0, YLO + ((i + 1) / NB) * (YHI - YLO))[1];
    ctx.fillStyle = th.name === 'dark' ? 'rgba(125,211,252,0.55)' : 'rgba(3,105,161,0.45)';
    ctx.fillRect(HX, y + 1, (st.ref[i] / max) * HW, bh - 2);
  }
  if (samples && dpsA > 0) {
    const b = hist(samples);
    ctx.globalAlpha = a * dpsA;
    ctx.strokeStyle = th.ramp[3]; ctx.lineWidth = 2.4;
    for (let i = 0; i < NB; i++) {
      if (b[i] <= 0) continue;
      const y = px(0, YLO + ((i + 1) / NB) * (YHI - YLO))[1];
      ctx.strokeRect(HX, y + 1, (b[i] / max) * HW, bh - 2);
    }
  }
  ctx.restore();
  text(ctx, 'exact', HX + 4, px(0, YLO)[1] + 24, { size: 15, color: th.accent, alpha: a, weight: 600 });
  if (samples) text(ctx, 'DPS', HX + 64, px(0, YLO)[1] + 24, { size: 15, color: th.ramp[3], alpha: a * dpsA, weight: 600 });
}

// Act III: DPS samples for each ζ in turn, with a small chart of TV distance and in-band fraction.
function drawSweep(ctx, th, st, t, a) {
  const f = ease.inOut(seg(t, ...T.sweepRun)) * (ZS.length - 1), i = Math.min(Math.floor(f), ZS.length - 2), u = f - i;
  const cur = u < 0.5 ? i : i + 1, s = st.sweep[cur];
  const fade = 1 - Math.abs(u - 0.5) * 2 < 0.25 ? 0.4 : 1;
  for (const [x, y] of s.end) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, th.ramp[3], a * 0.85 * fade, 2.2); }
  // Inset chart.
  const B = { x: 1250, y: 610, w: 280, h: 200 };
  ctx.save();
  ctx.globalAlpha = a * 0.92; ctx.fillStyle = th.bg; ctx.fillRect(B.x - 50, B.y - 46, B.w + 80, B.h + 94);
  ctx.globalAlpha = a; ctx.strokeStyle = th.faint; ctx.strokeRect(B.x - 50, B.y - 46, B.w + 80, B.h + 94);
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(B.x, B.y); ctx.lineTo(B.x, B.y + B.h); ctx.lineTo(B.x + B.w, B.y + B.h); ctx.stroke();
  ctx.restore();
  const X = (j) => B.x + (j / (ZS.length - 1)) * B.w, Y = (v) => B.y + B.h - v * B.h;
  text(ctx, 'DPS step size ζ', B.x + B.w / 2, B.y + B.h + 40, { size: 15, color: th.muted, align: 'center', alpha: a });
  for (const j of [0, 4, 6, 9]) text(ctx, String(ZS[j]), X(j), B.y + B.h + 20, { size: 13, color: th.muted, align: 'center', alpha: a });
  for (const v of [0, 0.5, 1]) text(ctx, String(v), B.x - 10, Y(v) + 5, { size: 13, color: th.muted, align: 'right', alpha: a });
  for (const [key, color, name, dy] of [['tv', th.fg, 'TV to exact posterior', -26], ['band', th.ramp[3], 'fraction in measurement band', -8]]) {
    ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = color; ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let j = 0; j <= cur; j++) (j ? ctx.lineTo : ctx.moveTo).call(ctx, X(j), Y(st.sweep[j][key]));
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(X(cur), Y(st.sweep[cur][key]), 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
    text(ctx, name, B.x, B.y + dy, { size: 14, weight: 600, color, alpha: a });
  }
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'The prior is the spiral mixture. We observe only the horizontal coordinate of an unknown sample, with a little noise. Several arms of the spiral cross the measurement band, so the posterior has several modes with different weights.',
    eq: 'y = A x + n,  A = [1 0],  n ~ N(0, σ_y²)' },
  { span: T.exact, act: 'Act I · The exact posterior',
    body: 'For a Gaussian mixture prior and a linear Gaussian measurement the posterior is again a mixture, and its probability path has an exact velocity. Following it from noise gives true posterior samples. The bars on the right show their distribution along the vertical axis.',
    eq: 'p(x | y) ∝ p(x) N(y; A x, σ_y²)' },
  { span: T.dps, act: 'Act II · Diffusion posterior sampling',
    body: 'DPS keeps the unconditional flow and adds a gradient step that pulls the denoised estimate x̂₁ toward the measurement. Its samples satisfy the measurement but not the posterior: the middle arm, which the exact posterior almost excludes, receives a large share of the samples.',
    eq: 'x ← x + h u_t(x) − ζ h ∇‖y − A x̂₁(x)‖' },
  { span: T.sweep, act: 'Act III · The step size',
    body: 'With a small step DPS ignores the measurement. Larger steps satisfy it but distort the weights between modes, and very large steps overshoot and scatter the samples. No single ζ recovers the exact posterior, because the guidance uses p(y | x̂₁) in place of the intractable p(y | x_t).',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §6 INFERENCE-TIME SCALING', 'Inverse problems', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 23, color: th.fg, alpha: a, font: 'serif' });
  }
  const read = (lab, v, col, xx, al) => {
    text(ctx, lab, xx, 690, { size: 15, weight: 600, color: th.muted, alpha: al, spacing: 1.5 });
    text(ctx, v, xx, 750, { size: 52, weight: 700, color: col, alpha: al, font: 'mono' });
  };
  const ea = window_(t, T.exact[0], T.exact[1], 0.4) * seg(t, ...T.exactHist);
  if (ea > 0) { read('MASS ON MIDDLE ARM', st.midExact.toFixed(2), th.accent, x, ea); read('IN BAND', st.exact.band.toFixed(2), th.fg, x + 250, ea); }
  const da = window_(t, T.dps[0], T.dps[1], 0.4) * seg(t, ...T.dpsHist);
  if (da > 0) { read('MASS ON MIDDLE ARM', st.midDps.toFixed(2), th.ramp[3], x, da); read('TV TO EXACT', st.dps.tv.toFixed(2), th.fg, x + 250, da); }
  const sa = window_(t, T.sweep[0], T.sweep[1], 0.4);
  if (sa > 0) {
    const f = ease.inOut(seg(t, ...T.sweepRun)) * (ZS.length - 1), i = Math.min(Math.floor(f), ZS.length - 2), cur = f - i < 0.5 ? i : i + 1;
    read('STEP SIZE ζ', String(ZS[cur]), th.ramp[3], x, sa);
    read('TV TO EXACT', st.sweep[cur].tv.toFixed(2), th.fg, x + 250, sa);
  }
  paragraph(ctx, `Exact Gaussian-mixture prior and posterior. σ_y = ${SY}, ${STEPS} steps, ${NS} samples (${NSWEEP} per ζ in Act III). TV distance is over ${NB} bins of x₂. DPS uses the exact denoiser of the prior, with its Jacobian by finite differences.`, x, 812, w, { size: 14, color: th.muted, lineHeight: 20 });
}
