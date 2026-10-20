// A13 · Best-of-N vs sequential Monte Carlo: two ways to spend test-time compute on a reward.
// The base model is the exact SDE of a Gaussian mixture. SMC weights particles by the reward of their
// look-ahead estimate E[x1 | x_t] (a difference potential, so the weights telescope to exp(λ r(x1))) and
// resamples when the effective sample size drops below N/2.
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, arrow, lerp, clamp } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1050, CY = 455, S = 150;
const plot = makePlot(CX, CY, S);
const px = plot.px;

const N = 64;            // particles in the animated runs
const STEPS = 50;        // SDE steps, so a run costs N x 50 network evaluations
const LAMBDA = 20;       // reward tilt
const RHO = 0.1;         // reward width
const OFFSET = 0.16;     // reward centre sits this far outside the arm, in the tail of one bead
const RUN_SEED = 127;    // a typical run: best-of-N near its expected best, SMC succeeds
const CURVE_NS = [4, 8, 16, 32, 64, 128, 256];
const CURVE_SEEDS = 160;
// Expected best reward of SMC and of trust-region search for CURVE_NS, from smcCurve(buildData(), 160) and
// trsCurve(buildData(), 160). Too slow to run at page load.
const SMC_CURVE = [0.076, 0.182, 0.251, 0.426, 0.734, 0.919, 0.993];
const TRS_CURVE = [0.016, 0.024, 0.062, 0.259, 0.651, 0.89, 0.993];
const TRS_SEED = 112;    // a typical successful search: the best reward climbs over several batches
const g = (t) => 1.15 * Math.sqrt(1 - t);
// Trust-region noise search. Shared settings from the paper (Table 4); k and the batch size are chosen for a 2D noise space.
const TRS = { k: 3, per: 2, warm: 0.2, l0: 0.8, lmin: 0.05, lmax: 2.4, alpha: 1.5, cs: 3, cf: 3, pmin: 0.1, pmax: 0.9 };
const ODE_STEPS = 25;    // Heun steps, so one candidate costs 50 network evaluations, like one particle

const T = {
  setup: [0, 4.4], reward: [1.6, 3.2],
  bon: [4.4, 15.2], bonRun: [5.0, 11.6], bonScore: [11.8, 13.2],
  smc: [15.2, 31.6], smcRun: [16.0, 28.0], smcScore: [28.2, 29.6],
  trs: [31.6, 48.4], trsWarm: [32.4, 35.0], trsZoom: [35.0, 36.0], trsIter: [36.0, 45.6], trsScore: [45.8, 47.2],
  curve: [48.4, 60.8], curveDraw: [49.4, 54.6],
};

const STILLS = [
  { t: 14.6, name: 'best-of-n', crop: { x: 645, y: 15, w: 947, h: 782 }, label: 'Best-of-N' },
  { t: 23.4, name: 'smc-resampling', crop: { x: 645, y: 15, w: 947, h: 782 }, label: 'SMC: look ahead and resample' },
  { t: 30.9, name: 'smc', crop: { x: 645, y: 15, w: 947, h: 782 }, label: 'SMC' },
  { t: 40.9, name: 'trs-search', label: 'Trust-region noise search' },
  { t: 47.9, name: 'trs', label: 'Trust-region search: result' },
  { t: 60.2, name: 'scaling', label: 'Reward vs compute' },
];

const reward = (st, x, y) => Math.exp(-((x - st.c[0]) ** 2 + (y - st.c[1]) ** 2) / (2 * RHO * RHO));
const rewardColor = (th) => (th.name === 'dark' ? '#9dff8a' : '#15803d');

export default {
  id: 'best-of-n-vs-smc',
  title: 'Best-of-N, SMC and trust-region noise search at test time',
  duration: 60.8,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.bon[0], label: 'I · Best-of-N' },
    { t: T.smc[0], label: 'II · SMC' },
    { t: T.trs[0], label: 'III · Trust regions' },
    { t: T.curve[0], label: 'IV · Scaling' },
  ],
  stills: STILLS,  // Paper crops trimmed to the drawn content, one height per figure row.
  crop: { x: 640, y: 0, w: 952, h: 900 },

  setup(env) {
    const st = buildData();
    st.glow = renderGlow(env, plot, st.mix);
    return st;
  },

  onTheme(env, st) {
    st.glow = renderGlow(env, plot, st.mix);
    st.heatCanvas = null;
  },

  draw(ctx, t, env, st) {
    const th = env.theme;
    drawGrid(ctx, th, plot, W, H);
    const curveA = seg(t, T.curve[0], T.curve[0] + 0.8);
    ctx.globalAlpha = (env.clean ? 1 : seg(t, 0.3, 1.8)) * (1 - 0.85 * curveA);
    ctx.drawImage(st.glow, 0, 0, W, H);
    ctx.globalAlpha = 1;
    drawReward(ctx, th, st, (env.clean ? 1 : ease.out(seg(t, ...T.reward))) * (1 - curveA), t);

    const bonA = window_(t, T.bon[0], T.bon[1], 0.5);
    if (bonA > 0) drawRun(ctx, th, st, st.bon, t, T.bonRun, T.bonScore, bonA, false);
    const smcA = window_(t, T.smc[0], T.smc[1], 0.5);
    if (smcA > 0) drawRun(ctx, th, st, st.smc, t, T.smcRun, T.smcScore, smcA, true);
    const trsA = window_(t, T.trs[0], T.trs[1], 0.5);
    if (trsA > 0) drawTRS(ctx, th, st, t, trsA);
    if (curveA > 0) drawCurve(ctx, th, st, t, curveA);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const s = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, s.label, 668, 70, { size: 40, weight: 600, color: th.fg });
    }
  },
};

// ---------- Simulation ----------

export function buildData() {
  const mix = spiralMixture();
  const { gmm, means } = mix;
  const [mx, my] = means[Math.round(0.82 * (means.length - 1))], r0 = Math.hypot(mx, my);
  const st = { mix, gmm, c: [mx * (1 + OFFSET / r0), my * (1 + OFFSET / r0)] };

  st.bon = simulate(st, N, RUN_SEED, false);
  st.smc = simulate(st, N, RUN_SEED, true);

  // Expected best reward of best-of-N from a pool of exact samples: E[max of N draws] in closed form.
  const pr = makeRng(1), pool = [];
  for (let i = 0; i < 20000; i++) { const s = gmm.sample(pr); pool.push(reward(st, s.x, s.y)); }
  pool.sort((a, b) => a - b);
  const M = pool.length;
  st.curveBon = CURVE_NS.map((n) => pool.reduce((acc, r, i) => acc + r * (((i + 1) / M) ** n - (i / M) ** n), 0));
  st.curveSmc = SMC_CURVE;
  st.curveTrs = TRS_CURVE;
  buildTRS(st);
  return st;
}

export function smcCurve(st, seeds = CURVE_SEEDS) {
  return CURVE_NS.map((n) => {
    let b = 0;
    for (let s = 0; s < seeds; s++) b += simulate(st, n, 1000 + s, true, false).best / seeds;
    return b;
  });
}

// Euler-Maruyama particles with optional SMC resampling. With `record`, keeps every step for drawing:
//   P[k]   positions at step k (before resampling at k), D1[k] look-ahead estimates, RH[k] their rewards,
//   W[k]   normalized weights at k, A[k] ancestor index for each particle after step k (identity if none),
//   RES[k] whether step k resampled.
export function simulate(st, n, seed, smc, record = true) {
  const { gmm } = st, rng = makeRng(seed), h = 1 / STEPS, sq = Math.sqrt(h);
  let X = [], RH = new Float64Array(n), logw = new Float64Array(n), d1 = [], F = [];
  for (let i = 0; i < n; i++) {
    X.push([rng.gauss(), rng.gauss()]);
    F.push(gmm.fields(X[i][0], X[i][1], 0));
    d1.push(F[i].d1);
    RH[i] = reward(st, ...d1[i]);
  }
  const rec = { P: [], D1: [], RH: [], W: [], A: [], RES: [], nres: 0 };
  const ident = Int32Array.from({ length: n }, (_, i) => i);
  for (let k = 0; k <= STEPS; k++) {
    const mxl = Math.max(...logw), w = Array.from(logw, (l) => Math.exp(l - mxl)), z = w.reduce((a, b) => a + b);
    for (let i = 0; i < n; i++) w[i] /= z;
    const ess = 1 / w.reduce((a, b) => a + b * b, 0);
    let A = ident;
    if (smc && k > 0 && k < STEPS && ess < n / 2) {
      A = new Int32Array(n);
      const u0 = rng.uniform() / n;
      let cum = w[0], j = 0;
      for (let i = 0; i < n; i++) {
        const u = u0 + i / n;
        while (cum < u && j < n - 1) cum += w[++j];
        A[i] = j;
      }
    }
    if (record) {
      rec.P.push(X.map((p) => [...p])); rec.D1.push(d1.map((p) => [...p]));
      rec.RH.push(Float64Array.from(RH)); rec.W.push(Float64Array.from(w));
      rec.A.push(A); rec.RES.push(A !== ident);
    }
    if (A !== ident) {
      rec.nres++;
      X = Array.from(A, (j) => [...X[j]]); d1 = Array.from(A, (j) => d1[j]); F = Array.from(A, (j) => F[j]);
      RH = Float64Array.from(A, (j) => RH[j]); logw.fill(0);
    }
    if (k === STEPS) break;
    const t = k * h, gt = g(t);
    for (let i = 0; i < n; i++) {
      // One field evaluation per step: the look-ahead at (x, t + h) is reused as the drift of the next step.
      const f = F[i];
      X[i][0] += h * (f.u[0] + 0.5 * gt * gt * f.score[0]) + gt * sq * rng.gauss();
      X[i][1] += h * (f.u[1] + 0.5 * gt * gt * f.score[1]) + gt * sq * rng.gauss();
      const tn = t + h;
      if (tn < 1 - 1e-9) { F[i] = gmm.fields(X[i][0], X[i][1], tn); d1[i] = F[i].d1; } else d1[i] = [X[i][0], X[i][1]];
      const rn = reward(st, ...d1[i]);
      logw[i] += LAMBDA * (rn - RH[i]);
      RH[i] = rn;
    }
  }
  const final = X.map((p) => reward(st, p[0], p[1]));
  rec.final = final;
  rec.best = Math.max(...final);
  rec.bestIdx = final.indexOf(rec.best);
  rec.mean = final.reduce((a, b) => a + b) / n;
  return rec;
}

// Trust-region noise search (Schweiger et al., 2026) on the deterministic ODE x1 = F(x0). Only r(F(x0)) is observed.
// A random warm-up spends 20% of the budget; then each batch proposes PER candidates in each of K boxes centred on the
// K best noises so far. A box grows by 1.5 after 3 successes in a row, shrinks by 1.5 after 3 failures, and is reset
// when it fails at the minimum size. With `record`, keeps the batches and the boxes for drawing.
export function searchTRS(st, budget, seed, record = false) {
  const { gmm } = st, rng = makeRng(seed), pts = [];
  const evalz = (z, region) => {
    const tr = gmm.integrateODE(z[0], z[1], ODE_STEPS), x = [tr[2 * ODE_STEPS], tr[2 * ODE_STEPS + 1]];
    const p = { z, x, r: reward(st, ...x), region, tr: record ? tr : null };
    pts.push(p);
    return p;
  };
  const nw = Math.min(budget, Math.max(TRS.k, Math.round(TRS.warm * budget)));
  const batches = [{ cand: Array.from({ length: nw }, () => evalz([rng.gauss(), rng.gauss()], -1)), boxes: null }];
  const reg = Array.from({ length: TRS.k }, () => ({ l: TRS.l0, s: 0, f: 0 }));
  const top = () => [...pts].sort((a, b) => b.r - a.r).slice(0, TRS.k);
  while (pts.length < budget) {
    const T = top(), cand = [], boxes = [];
    for (let j = 0; j < TRS.k; j++) {
      const R = reg[j], ctr = T[j % T.length];
      const box = { c: ctr.z, l: R.l, best: ctr.r, outcome: 0 };
      let improved = false;
      for (let b = 0; b < TRS.per && pts.length < budget; b++) {
        let m;
        do { const p = TRS.pmin + (TRS.pmax - TRS.pmin) * rng.uniform(); m = [rng.uniform() < p, rng.uniform() < p]; } while (!m[0] && !m[1]);
        const z = ctr.z.map((v, d) => v + (m[d] ? (rng.uniform() - 0.5) * R.l : 0));
        const p = evalz(z, j);
        cand.push(p);
        if (p.r > ctr.r) improved = true;
      }
      if (improved) { R.s++; R.f = 0; } else { R.f++; R.s = 0; }
      if (R.s >= TRS.cs) { R.l = Math.min(R.l * TRS.alpha, TRS.lmax); R.s = 0; box.outcome = 1; }
      if (R.f >= TRS.cf) { R.l = R.l <= TRS.lmin + 1e-9 ? TRS.l0 : Math.max(R.l / TRS.alpha, TRS.lmin); R.f = 0; box.outcome = -1; }
      box.lNext = R.l;
      boxes.push(box);
    }
    batches.push({ cand, boxes });
  }
  const best = pts.reduce((a, b) => (b.r > a.r ? b : a));
  return { best: best.r, bestPt: best, pts, batches };
}

export function trsCurve(st, seeds = CURVE_SEEDS) {
  return CURVE_NS.map((n) => {
    let b = 0;
    for (let s = 0; s < seeds; s++) b += searchTRS(st, n, 1000 + s).best / seeds;
    return b;
  });
}

const INSET = { x: 1286, y: 534, w: 296, h: 296 };

// The animated search, the inset cameras, and r(F(x0)) on a grid (ground truth that the search never sees).
function buildTRS(st) {
  const run = searchTRS(st, N, TRS_SEED, true);
  // The inset camera frames the boxes of each batch and of the next one, and moves along with the boxes.
  const fit = (batches) => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const b of batches) for (const bx of b.boxes) {
      const h = Math.max(bx.l, bx.lNext) / 2;
      x0 = Math.min(x0, bx.c[0] - h); x1 = Math.max(x1, bx.c[0] + h);
      y0 = Math.min(y0, bx.c[1] - h); y1 = Math.max(y1, bx.c[1] + h);
    }
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: Math.max(0.62 * Math.max(x1 - x0, y1 - y0), 0.75) };
  };
  run.cams = run.batches.map((b, i) => (i === 0 ? null : fit(run.batches.slice(i, i + 2))));
  st.trs = run;
  st.camFull = { cx: 0, cy: 0, h: 3.3 };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const c of run.cams.slice(1)) {
    x0 = Math.min(x0, c.cx - c.h); x1 = Math.max(x1, c.cx + c.h); y0 = Math.min(y0, c.cy - c.h); y1 = Math.max(y1, c.cy + c.h);
  }
  const G = 72, cam = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, h: 0.5 * Math.max(x1 - x0, y1 - y0) + 0.2 }, hm = new Float32Array(G * G);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const zx = cam.cx - cam.h + ((i + 0.5) * 2 * cam.h) / G, zy = cam.cy - cam.h + ((j + 0.5) * 2 * cam.h) / G;
    const tr = st.gmm.integrateODE(zx, zy, ODE_STEPS);
    hm[i * G + j] = reward(st, tr[2 * ODE_STEPS], tr[2 * ODE_STEPS + 1]);
  }
  st.heat = { G, hm, cam };
  // When each candidate is proposed (tA) and when its ODE solve lands (tB).
  const nb = run.batches.length - 1, D = (T.trsIter[1] - T.trsIter[0]) / nb, [w0, w1] = T.trsWarm;
  run.D = D;
  run.batches.forEach((b, i) => b.cand.forEach((p, q) => {
    if (i === 0) { p.tA = w0 + (q / b.cand.length) * (w1 - w0 - 0.9); p.tB = p.tA + 0.9; }
    else { p.tA = T.trsIter[0] + (i - 1) * D + D * (0.04 + (0.36 * q) / b.cand.length); p.tB = p.tA + 0.42 * D; }
  }));
}

// r(F(x0)) as a G x G image in the reward colour, alpha proportional to the reward. Rebuilt on theme change.
function heatCanvas(th, { G, hm }) {
  const c = document.createElement('canvas');
  c.width = c.height = G;
  const g = c.getContext('2d'), img = g.createImageData(G, G), [R, Gc, B] = hexRgb(rewardColor(th));
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const o = 4 * ((G - 1 - j) * G + i);
    img.data[o] = R; img.data[o + 1] = Gc; img.data[o + 2] = B; img.data[o + 3] = Math.round(255 * 0.32 * hm[i * G + j]);
  }
  g.putImageData(img, 0, 0);
  return c;
}

// Batch index i >= 1 and the fraction u of its slot at time t.
function trsSlot(run, t) {
  const nb = run.batches.length - 1, f = (t - T.trsIter[0]) / run.D;
  if (f < 0) return [1, 0];
  if (f >= nb) return [nb, 1];
  return [Math.floor(f) + 1, f - Math.floor(f)];
}

// Box j at time t: the size changes in [0.6, 0.78] of the slot, then the box moves to its next centre.
function trsBox(run, t, j) {
  const [i, u] = trsSlot(run, t), b = run.batches[i].boxes[j], nx = run.batches[i + 1]?.boxes[j];
  const gs = ease.inOut(seg(u, 0.6, 0.78)), mv = nx ? ease.inOut(seg(u, 0.8, 0.98)) : 0;
  return {
    c: [lerp(b.c[0], nx ? nx.c[0] : b.c[0], mv), lerp(b.c[1], nx ? nx.c[1] : b.c[1], mv)],
    l: lerp(b.l, b.lNext, gs),
    flash: b.outcome * window_(u, 0.58, 0.98, 0.08) * (t <= T.trsIter[1] ? 1 : 0),
  };
}

function drawTRS(ctx, th, st, t, alpha) {
  const run = st.trs, amber = th.ramp[3];
  const score = seg(t, ...T.trsScore);
  const landed = run.pts.filter((p) => p.tB <= t);

  // Main plot: each candidate's ODE path from x0 to x1 = F(x0), then its sample.
  ctx.save();
  ctx.lineWidth = 1.4;
  for (const p of run.pts) {
    if (t < p.tA) continue;
    const f = seg(t, p.tA, p.tB), fade = 1 - seg(t, p.tB + 0.2, p.tB + 0.9);
    if (fade <= 0) continue;
    const n = Math.max(1, Math.ceil(f * ODE_STEPS));
    ctx.globalAlpha = alpha * fade * 0.75;
    ctx.strokeStyle = mixColor(th, f >= 1 ? p.r : 0, 1);
    ctx.beginPath();
    ctx.moveTo(...px(p.tr[0], p.tr[1]));
    for (let k = 1; k <= n; k++) {
      const w = k === n ? f * ODE_STEPS - (k - 1) : 1;
      ctx.lineTo(...px(lerp(p.tr[2 * k - 2], p.tr[2 * k], w), lerp(p.tr[2 * k - 1], p.tr[2 * k + 1], w)));
    }
    ctx.stroke();
    const [sx, sy] = px(p.tr[0], p.tr[1]);
    ctx.strokeStyle = th.muted;
    ctx.globalAlpha = alpha * fade * 0.6;
    ctx.strokeRect(sx - 3, sy - 3, 6, 6);
  }
  ctx.restore();
  const parts = landed.map((p) => [p.x[0], p.x[1], mixColor(th, p.r, 1), 3.4]);
  for (const [x, y, c, r] of parts) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, c, alpha, r); }
  if (score > 0) {
    const [x, y] = px(...run.bestPt.x);
    ring(ctx, th, x, y, 16 - 6 * ease.out(score), alpha * score);
  }
  const lA = alpha * seg(t, T.trsScore[0] - 0.6, T.trsScore[0] + 0.4);
  if (lA > 0) {
    const view = { P: { [STEPS]: run.pts.map((p) => p.x) }, bestIdx: run.pts.indexOf(run.bestPt), best: run.best };
    drawLoupe(ctx, th, st, view, parts, score, lA);
  }

  // Inset: the search itself, in noise space.
  const { x: IX, y: IY, w: IW, h: IH } = INSET;
  const zA = ease.inOut(seg(t, ...T.trsZoom)), [si, su] = trsSlot(run, t);
  const ca = run.cams[si], cb = run.cams[si + 1] ?? ca, mv = ease.inOut(seg(su, 0.8, 0.98));
  const cz = { cx: lerp(ca.cx, cb.cx, mv), cy: lerp(ca.cy, cb.cy, mv), h: lerp(ca.h, cb.h, mv) };
  const cam = { cx: lerp(st.camFull.cx, cz.cx, zA), cy: lerp(st.camFull.cy, cz.cy, zA), h: lerp(st.camFull.h, cz.h, zA) };
  const sc = IW / (2 * cam.h), ip = (zx, zy) => [IX + IW / 2 + (zx - cam.cx) * sc, IY + IH / 2 - (zy - cam.cy) * sc];
  const inA = alpha * seg(t, T.trs[0] + 0.2, T.trs[0] + 1);
  ctx.save();
  ctx.globalAlpha = inA;
  ctx.fillStyle = th.bg;
  ctx.fillRect(IX, IY, IW, IH);
  ctx.beginPath(); ctx.rect(IX, IY, IW, IH); ctx.clip();
  // Faint unit grid in noise space.
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (let v = -4; v <= 4; v++) {
    const [gx] = ip(v, 0), [, gy] = ip(0, v);
    ctx.beginPath(); ctx.moveTo(gx, IY); ctx.lineTo(gx, IY + IH); ctx.moveTo(IX, gy); ctx.lineTo(IX + IW, gy); ctx.stroke();
  }
  // Ground truth r(F(x0)).
  const { cam: hc } = st.heat, hA = seg(t, T.trsWarm[1] - 1, T.trsWarm[1]);
  if (!st.heatCanvas) st.heatCanvas = heatCanvas(th, st.heat);
  const [hx, hy] = ip(hc.cx - hc.h, hc.cy + hc.h);
  ctx.globalAlpha = inA * hA;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(st.heatCanvas, hx, hy, 2 * hc.h * sc, 2 * hc.h * sc);
  // Trust regions.
  const bA = inA * seg(t, T.trsWarm[1], T.trsWarm[1] + 0.5);
  if (bA > 0) for (let j = 0; j < TRS.k; j++) {
    const b = trsBox(run, t, j), [bx, by] = ip(...b.c), side = b.l * sc;
    ctx.globalAlpha = bA * 0.1;
    ctx.fillStyle = amber;
    ctx.fillRect(bx - side / 2, by - side / 2, side, side);
    ctx.globalAlpha = bA;
    ctx.strokeStyle = amber;
    ctx.lineWidth = 1.8 + 1.6 * Math.abs(b.flash);
    ctx.strokeRect(bx - side / 2, by - side / 2, side, side);
  }
  // Candidates, coloured by the reward of their sample once it has landed.
  for (const p of run.pts) {
    if (t < p.tA) continue;
    const [x, y] = ip(...p.z), f = seg(t, p.tA, p.tB);
    const pop = 1 - seg(t, p.tA, p.tA + 0.35);
    ctx.globalAlpha = 1;
    dot(ctx, th, x, y, mixColor(th, f >= 1 ? p.r : 0, 1), inA, 3.2 + 3 * pop);
  }
  if (score > 0) {
    const [x, y] = ip(...run.bestPt.z);
    ctx.restore();
    ctx.save();
    ctx.beginPath(); ctx.rect(IX, IY, IW, IH); ctx.clip();
    ring(ctx, th, x, y, 12 - 4 * ease.out(score), inA * score, 2);
  }
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4; ctx.globalAlpha = inA;
  ctx.strokeRect(IX, IY, IW, IH);
  ctx.restore();
  text(ctx, 'noise space  x₀', IX, IY - 14, { size: 18, weight: 600, color: th.fg, alpha: inA });
  const noteA = inA * seg(t, T.trsWarm[1] - 1, T.trsWarm[1]);
  text(ctx, 'shaded: r(F(x₀)),', IX, IY + IH + 24, { size: 14, color: th.muted, alpha: noteA });
  text(ctx, 'never seen by the search', IX, IY + IH + 44, { size: 14, color: th.muted, alpha: noteA });
  // Size changes, written next to the box that changed.
  if (bA > 0) for (let j = 0; j < TRS.k; j++) {
    const b = trsBox(run, t, j);
    if (!b.flash) continue;
    const [bx, by] = ip(...b.c), side = b.l * sc;
    if (bx < IX || bx > IX + IW || by < IY || by > IY + IH) continue;
    text(ctx, b.flash > 0 ? 'ℓ × 1.5' : 'ℓ ÷ 1.5', bx + side / 2 + 6, by - side / 2 - 4, { size: 15, weight: 600, color: amber, alpha: bA * Math.abs(b.flash) });
  }
}

// ---------- Drawing ----------

function drawReward(ctx, th, st, a, t, label = true, lw = 1) {
  if (a <= 0) return;
  const [x, y] = px(...st.c), col = rewardColor(th), r = RHO * S;
  ctx.save();
  ctx.globalAlpha = a;
  const grad = ctx.createRadialGradient(x, y, 0, x, y, 2.6 * r);
  grad.addColorStop(0, th.name === 'dark' ? 'rgba(157,255,138,0.32)' : 'rgba(21,128,61,0.22)');
  grad.addColorStop(1, th.name === 'dark' ? 'rgba(157,255,138,0)' : 'rgba(21,128,61,0)');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.arc(x, y, 2.6 * r, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = col; ctx.lineWidth = 1.6 * lw; ctx.setLineDash([5 * lw, 5 * lw]);
  ctx.beginPath(); ctx.arc(x, y, 2 * r, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();
  if (label) text(ctx, 'reward r(x)', x, y - 58, { size: 20, weight: 600, color: col, alpha: a, align: 'center', font: 'serif' });
}

// Fractional step at time t. Steps 0-30 take a quarter of the run and steps 37-50, where the look-ahead
// sharpens and resampling happens, take the last 60%.
const STEP_KNOTS = [[0, 0], [0.25, 30], [0.4, 37], [1, STEPS]];
function stepAt(t, span) {
  const u = seg(t, ...span);
  for (let i = 1; i < STEP_KNOTS.length; i++) {
    const [u0, s0] = STEP_KNOTS[i - 1], [u1, s1] = STEP_KNOTS[i];
    if (u <= u1) return lerp(s0, s1, (u - u0) / (u1 - u0));
  }
  return STEPS;
}

// Particle i at fractional step sf: interpolates from its parent at step k to itself at step k + 1.
function particleAt(run, sf, i) {
  const k = Math.min(Math.floor(sf), STEPS - 1), f = sf - k;
  const p = run.P[k][run.A[k][i]], q = run.P[k + 1][i];
  return [lerp(p[0], q[0], f), lerp(p[1], q[1], f)];
}

const LOUPE = { x: 1450, y: 196, r: 118, z: 2.5 };

function drawRun(ctx, th, st, run, t, span, scoreSpan, alpha, smc) {
  const sf = stepAt(t, span), k = Math.min(Math.floor(sf), STEPS - 1), kn = k + 1;
  const score = seg(t, ...scoreSpan);
  const started = seg(t, span[0] - 0.4, span[0]);
  const rNow = (i) => run.final[i] * score + (1 - score) * run.RH[kn][i];

  // Genealogy. Edge (s, i) joins particle i at step s + 1 to its parent A[s][i] at step s. Every edge is drawn
  // faintly; edges on an ancestral line of a particle that is still alive are drawn on top, once each.
  const edge = (s, i, f = 1) => {
    const p = run.P[s][run.A[s][i]], q = run.P[s + 1][i];
    ctx.moveTo(...px(...p));
    ctx.lineTo(...px(lerp(p[0], q[0], f), lerp(p[1], q[1], f)));
  };
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = th.muted;
  ctx.globalAlpha = (smc ? 0.16 : 0) * alpha * started;
  if (smc) {
    ctx.beginPath();
    for (let s = 0; s < k; s++) for (let i = 0; i < N; i++) edge(s, i);
    ctx.stroke();
  }
  ctx.globalAlpha = alpha * started;
  ctx.lineWidth = smc ? 1.7 : 1.1;
  let alive = Array.from({ length: N }, (_, i) => i);
  for (let s = k; s >= 0; s--) {
    for (const i of alive) {
      const r = s === k ? rNow(i) : run.RH[s + 1][i];
      ctx.strokeStyle = mixColor(th, smc ? Math.max(r, 0.35) : r, smc ? 0.6 : th.name === 'dark' ? 0.3 : 0.4);
      ctx.beginPath();
      edge(s, i, s === k ? sf - k : 1);
      ctx.stroke();
    }
    alive = [...new Set(alive.map((i) => run.A[s][i]))];
  }
  ctx.restore();

  // Look-ahead estimates x̂1 = E[x1 | x_t] (SMC only).
  if (smc) {
    const la = alpha * (1 - score) * seg(sf, 1, 4);
    const kk = Math.min(Math.round(sf), STEPS);
    ctx.save();
    ctx.strokeStyle = th.muted;
    ctx.lineWidth = 1.2;
    ctx.globalAlpha = la * 0.8;
    for (let i = 0; i < N; i++) {
      const [x, y] = px(...run.D1[kk][i]);
      ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
  }

  // Particles dropped at a recent resampling step: a red cross where they died.
  if (smc) {
    for (let s = Math.max(1, k - 3); s <= k; s++) {
      if (!run.RES[s]) continue;
      const fa = alpha * (1 - clamp((sf - s) / 2.5));
      if (fa <= 0) continue;
      const kept = new Set(run.A[s]);
      ctx.save();
      ctx.strokeStyle = th.name === 'dark' ? '#ff6b6b' : '#dc2626';
      ctx.lineWidth = 2;
      ctx.globalAlpha = fa;
      ctx.beginPath();
      for (let j = 0; j < N; j++) {
        if (kept.has(j)) continue;
        const [x, y] = px(...run.P[s][j]), r = 5;
        ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  // Particles: colour by (look-ahead) reward, size by weight.
  const parts = [];
  for (let i = 0; i < N; i++) {
    const w = smc && score < 1 ? run.W[kn][i] * N : 1;
    parts.push([...particleAt(run, sf, i), mixColor(th, rNow(i), 1), 2.4 + 1.6 * Math.sqrt(clamp(w, 0, 3))]);
  }
  for (const [x, y, c, r] of parts) { const [X, Y] = px(x, y); dot(ctx, th, X, Y, c, alpha * started, r); }

  // The returned sample.
  if (score > 0) {
    const [x, y] = px(...run.P[STEPS][run.bestIdx]);
    ring(ctx, th, x, y, 16 - 6 * ease.out(score), alpha * score);
  }

  // Loupe: the reward region magnified, so the final samples can be told apart.
  const lA = alpha * (smc ? seg(sf, 33, 37) : seg(t, scoreSpan[0] - 0.6, scoreSpan[0] + 0.4));
  if (lA > 0) drawLoupe(ctx, th, st, run, parts, score, lA);
}

function ring(ctx, th, x, y, r, a, w = 2.4) {
  ctx.save();
  ctx.strokeStyle = th.highlight;
  ctx.lineWidth = w;
  ctx.globalAlpha = a;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();
}

function drawLoupe(ctx, th, st, run, parts, score, a) {
  const { x: LX, y: LY, r: LR, z: Z } = LOUPE, [cx, cy] = px(...st.c);
  const src = LR / Z;
  ctx.save();
  // Connector from the source circle to the loupe.
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2; ctx.globalAlpha = a * 0.7;
  const ang = Math.atan2(LY - cy, LX - cx);
  ctx.beginPath(); ctx.arc(cx, cy, src, 0, 2 * Math.PI);
  ctx.moveTo(cx + src * Math.cos(ang), cy + src * Math.sin(ang));
  ctx.lineTo(LX - LR * Math.cos(ang), LY - LR * Math.sin(ang));
  ctx.stroke();
  ctx.globalAlpha = a;
  ctx.beginPath(); ctx.arc(LX, LY, LR, 0, 2 * Math.PI);
  ctx.fillStyle = th.bg; ctx.fill();
  ctx.clip();
  ctx.translate(LX, LY); ctx.scale(Z, Z); ctx.translate(-cx, -cy);
  ctx.drawImage(st.glow, 0, 0, W, H);
  drawReward(ctx, th, st, a, 0, false, 1 / Z);
  for (const [x, y, c, r] of parts) {
    const [X, Y] = px(x, y);
    if (Math.hypot(X - cx, Y - cy) > src + 4) continue;
    dot(ctx, th, X, Y, c, a, (r * 0.9) / Z);
  }
  if (score > 0) {
    const [x, y] = px(...run.P[STEPS][run.bestIdx]);
    ring(ctx, th, x, y, 10 / Z, a * score, 2 / Z);
  }
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.6; ctx.globalAlpha = a;
  ctx.beginPath(); ctx.arc(LX, LY, LR, 0, 2 * Math.PI); ctx.stroke();
  ctx.restore();
  text(ctx, `×${Z}`, LX + LR * 0.72, LY - LR * 0.78, { size: 16, color: th.muted, alpha: a });
  if (score > 0) text(ctx, `best r = ${run.best.toFixed(2)}`, LX, LY + LR + 30, { size: 20, weight: 600, color: th.fg, alpha: a * score, align: 'center', font: 'serif' });
}

function mixColor(th, r, alpha) {
  const a = hexRgb(th.name === 'dark' ? '#5c6680' : '#9aa3b5'), b = hexRgb(rewardColor(th));
  const w = clamp(r * 1.6);
  return `rgba(${a.map((v, i) => Math.round(lerp(v, b[i], w))).join(',')},${alpha})`;
}
const hexRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// Expected best reward vs N (log axis). Both methods spend N x 50 network evaluations.
function drawCurve(ctx, th, st, t, a) {
  const L = CX - 300, R = CX + 360, Tp = CY - 250, B = CY + 260;
  const X = (n) => L + ((Math.log2(n) - 2) / 6) * (R - L), Y = (v) => B - v * (B - Tp);
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.bg;
  ctx.globalAlpha = a * 0.9;
  ctx.fillRect(L - 90, Tp - 70, R - L + 140, B - Tp + 150);
  ctx.globalAlpha = a;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L, Tp); ctx.lineTo(L, B); ctx.lineTo(R, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const v of [0.25, 0.5, 0.75, 1]) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); }
  ctx.restore();
  for (const v of [0, 0.5, 1]) text(ctx, v.toFixed(1), L - 14, Y(v) + 6, { size: 17, color: th.muted, align: 'right', alpha: a });
  CURVE_NS.forEach((n) => text(ctx, String(n), X(n), B + 28, { size: 17, color: th.muted, align: 'center', alpha: a }));
  text(ctx, 'samples per run N   (compute = N × 50 NFE)', (L + R) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha: a });
  text(ctx, 'expected best reward', L, Tp - 24, { size: 18, color: th.muted, alpha: a });

  // N = 64 guide: the two runs shown above.
  ctx.save();
  ctx.setLineDash([4, 6]); ctx.strokeStyle = th.muted; ctx.globalAlpha = a * 0.6; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(X(64), Tp); ctx.lineTo(X(64), B); ctx.stroke();
  ctx.restore();

  const grow = ease.inOut(seg(t, ...T.curveDraw));
  const series = [
    { v: st.curveBon, color: th.fg, name: 'best-of-N', dash: [9, 7] },
    { v: st.curveSmc, color: rewardColor(th), name: 'SMC', dash: [] },
    { v: st.curveTrs, color: th.ramp[3], name: 'trust-region search', dash: [2, 8] },
  ];
  for (const s of series) {
    const f = grow * (CURVE_NS.length - 1);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = ctx.fillStyle = s.color;
    ctx.lineWidth = 3.4; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.setLineDash(s.dash);
    ctx.beginPath();
    ctx.moveTo(X(CURVE_NS[0]), Y(s.v[0]));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1);
      ctx.lineTo(lerp(X(CURVE_NS[i - 1]), X(CURVE_NS[i]), w), lerp(Y(s.v[i - 1]), Y(s.v[i]), w));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    for (let i = 0; i <= Math.floor(f); i++) { ctx.beginPath(); ctx.arc(X(CURVE_NS[i]), Y(s.v[i]), 5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
    const la = a * seg(grow, 0.95, 1);
    if (s.v === st.curveTrs) {
      text(ctx, s.name, X(CURVE_NS[3]) + 16, Y(s.v[3]) + 30, { size: 22, weight: 700, color: s.color, alpha: la });
      continue;
    }
    const last = s.v.length - (s.name === 'SMC' ? 1 : 2);
    text(ctx, s.name, X(CURVE_NS[last]) - 14, Y(s.v[last]) - 18, { size: 22, weight: 700, color: s.color, align: 'right', alpha: la });
  }
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'A reward model scores samples, and we want samples it likes. Here the reward is a narrow bump just outside the outer arm. Fewer than 1 in 400 samples from the base model score above 0.5.',
    eq: 'r(x) = exp(−‖x − c‖² / 2ρ²)' },
  { span: T.bon, act: 'Act I · Best-of-N',
    body: 'Best-of-N draws N independent samples and keeps the one with the highest reward. It spends the same compute on every trajectory, including the ones that were hopeless early on.',
    eq: 'x* = argmaxᵢ r(x₁⁽ⁱ⁾)' },
  { span: [T.smc[0], T.smcScore[0]], act: 'Act II · Sequential Monte Carlo',
    body: 'SMC scores particles while they are still noisy, using the look-ahead x̂₁ = E[x₁ | xₜ] (hollow circles). Weights grow with the improvement in reward. When they become uneven, weak particles are dropped (red crosses) and strong ones are copied.',
    eq: 'wᵢ ← wᵢ · exp(λ Δr(x̂₁⁽ⁱ⁾))' },
  { span: [T.smcScore[0], T.smc[1]], act: 'Act II · Sequential Monte Carlo',
    body: 'With the same 64 particles and the same number of network evaluations, most final samples sit on the reward bump. The weights telescope, so SMC samples from the reward-tilted distribution.',
    eq: 'p̃(x) ∝ p(x) · exp(λ r(x))' },
  { span: [T.trs[0], T.trsZoom[1]], act: 'Act III · Trust-region noise search',
    body: 'Trust-region search treats the sampler as a black box. It uses the ODE, so each initial noise x₀ (inset) gives one sample x₁ = F(x₀), and it only sees the reward of finished samples. It starts with random noise on 20% of the budget.',
    eq: 'x₀* = argmax r(F(x₀))' },
  { span: [T.trsZoom[1], T.trsScore[0]], act: 'Act III · Trust-region noise search',
    body: 'It keeps k = 3 boxes centred on the best noises so far and draws new candidates inside them. A box grows after 3 improvements in a row and shrinks after 3 failures. After every batch, the boxes move to the current top 3.',
    eq: 'ℓ ← 1.5 ℓ   or   ℓ ← ℓ / 1.5' },
  { span: [T.trsScore[0], T.trs[1]], act: 'Act III · Trust-region noise search',
    body: 'With 64 candidates, the compute of 64 particles, the search ends on the reward bump. It used no gradients and no rewards of intermediate states. It only learned where good noise lies.',
    eq: '' },
  { span: T.curve, act: 'Act IV · Spending compute',
    body: `Best-of-N improves slowly, because it can only wait for a lucky sample. SMC and trust-region search reach the narrow region with far fewer samples. SMC steers each trajectory with look-ahead rewards. Trust-region search only scores finished samples, so it needs a few batches before it finds a signal. Curves average ${CURVE_SEEDS} runs per N; best-of-N is in closed form.`,
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §6 INFERENCE-TIME SCALING', 'Test-time search', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 30, { size: 25, color: th.fg, alpha: a, font: 'serif' });
  }

  // Readouts for the two runs.
  for (const [span, runSpan, scoreSpan, run, smc] of [[T.bon, T.bonRun, T.bonScore, st.bon, false], [T.smc, T.smcRun, T.smcScore, st.smc, true]]) {
    const a = window_(t, span[0], span[1], 0.4);
    if (a <= 0) continue;
    const sf = stepAt(t, runSpan), k = Math.floor(sf);
    text(ctx, `N = ${N} particles · ${Math.round(sf) * N} NFE`, x, 610, { size: 18, weight: 600, color: th.muted, alpha: a });
    if (smc) {
      const nres = run.RES.slice(0, k + 1).filter(Boolean).length;
      text(ctx, `resampling steps: ${nres}`, x, 640, { size: 18, weight: 600, color: th.muted, alpha: a });
    }
    const sc = seg(t, ...scoreSpan);
    if (sc > 0) {
      text(ctx, 'BEST REWARD', x, 680, { size: 15, weight: 600, color: th.muted, alpha: a * sc, spacing: 1.5 });
      text(ctx, run.best.toFixed(2), x, 752, { size: 66, weight: 700, color: smc ? rewardColor(th) : th.fg, alpha: a * sc, font: 'mono' });
      const hits = run.final.filter((r) => r > 0.5).length;
      text(ctx, `${hits} of ${N} samples with r > 0.5`, x + 190, 740, { size: 18, color: th.fg, alpha: a * sc });
    }
  }

  // Readouts for trust-region search.
  const ta = window_(t, T.trs[0], T.trs[1], 0.4);
  if (ta > 0) {
    const run = st.trs, landed = run.pts.filter((p) => p.tB <= t), [i] = trsSlot(run, t);
    const nb = run.batches.length - 1;
    text(ctx, `${landed.length} of ${N} noise candidates · ${landed.length * 50} NFE`, x, 610, { size: 18, weight: 600, color: th.muted, alpha: ta });
    const phase = t < T.trsWarm[1] ? 'random warm-up' : `batch ${i} of ${nb} · ${TRS.k} trust regions`;
    text(ctx, phase, x, 640, { size: 18, weight: 600, color: th.muted, alpha: ta });
    if (landed.length) {
      const best = Math.max(...landed.map((p) => p.r)), sc = seg(t, ...T.trsScore);
      text(ctx, sc > 0 ? 'BEST REWARD' : 'BEST REWARD SO FAR', x, 680, { size: 15, weight: 600, color: th.muted, alpha: ta, spacing: 1.5 });
      text(ctx, best.toFixed(2), x, 752, { size: 66, weight: 700, color: sc > 0 ? th.ramp[3] : th.fg, alpha: ta, font: 'mono' });
      if (sc > 0) {
        const hits = run.pts.filter((p) => p.r > 0.5).length;
        text(ctx, `${hits} of ${N} samples with r > 0.5`, x + 190, 740, { size: 18, color: th.fg, alpha: ta * sc });
      }
    }
  }

  const tf = window_(t, T.trs[0], T.trs[1], 0.4);
  text(ctx, `Base model: exact SDE for a 110-component Gaussian mixture, ${STEPS} Euler–Maruyama steps. λ = ${LAMBDA}, resample when ESS < N/2.`, x, 838, { size: 14, color: th.muted, alpha: 1 - tf });
  paragraph(ctx, `Trust-region search: exact ODE, ${ODE_STEPS} Heun steps (50 NFE). k = 3 boxes, 2 candidates per box per batch, ℓ₀ = 0.8, ℓ ∈ [0.05, 2.4], random coordinate masks. Other settings follow Schweiger et al. (2026).`, x, 818, w, { size: 14, color: th.muted, alpha: tf, lineHeight: 20 });
}
