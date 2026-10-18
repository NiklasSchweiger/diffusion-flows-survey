// A10 · Reflow straightens paths.
// 1-rectified flow: with independent noise-data pairs, the flow-matching optimum is the exact velocity of the spiral
// mixture, so its ODE is exact. Reflow pairs each noise point with its own ODE endpoint and fits a new velocity to
// the straight lines between them. We fit that velocity by kernel regression on 8000 such pairs (no network), which
// is the least-squares flow-matching optimum smoothed with a Gaussian kernel.
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, renderGlow, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, arrow, lerp, clamp, rampColor } from '../lib/theme.js';

const W = 1600, H = 900;
const plot = makePlot(1070, 450, 150);
const px = (x, y) => plot.px(x, y);

const N = 360;          // particles on screen
const NSEG = 150;       // segments used for the crossing and cost readouts
const NPAIR = 8000;     // reflow training pairs
const BW = 0.07;        // kernel bandwidth
const TS = 32;          // time grid of the reflow field
const ODE = 48;         // Heun steps for the exact 1-RF ODE
const NTEST = 300;
const NFES = [1, 2, 4, 8, 16, 32];

const T = {
  setup: [0, 6], noiseIn: [1.2, 2.8],
  ind: [6, 18], indLines: [6.6, 8.6], indMove: [9.4, 13.2], cross: [13.8, 17.6],
  rf1: [18, 30], rf1Flow: [18.8, 23.4], rf1Jump: [24.6, 27.6],
  re: [30, 43], reLines: [30.6, 33.2], reFlow: [34.0, 37.4], reJump: [38.4, 41.4],
  chart: [43, 56], chartDraw: [44.2, 49.5],
};

const STILLS = [
  { t: 17.4, name: 'independent', label: 'Independent coupling' },
  { t: 29.6, name: 'one-step-1rf', label: 'One Euler step, 1-rectified flow' },
  { t: 33.8, name: 'reflow-coupling', label: 'Reflow coupling' },
  { t: 42.6, name: 'one-step-reflow', label: 'One Euler step after reflow' },
  { t: 55.6, name: 'error-vs-nfe', label: 'Error vs number of steps' },
];

// ---------- Reflow field by kernel regression ----------

function fitReflow(pairs) {
  const C = 3 * BW, grids = [];
  for (let s = 0; s <= TS; s++) {
    const t = s / TS, cells = new Map();
    for (let i = 0; i < pairs.n; i++) {
      const zx = pairs.x0[2 * i] + t * pairs.d[2 * i], zy = pairs.x0[2 * i + 1] + t * pairs.d[2 * i + 1];
      const k = Math.floor(zx / C) * 100003 + Math.floor(zy / C);
      let a = cells.get(k);
      if (!a) cells.set(k, (a = []));
      a.push(i);
    }
    grids.push(cells);
  }
  // Nadaraya-Watson estimate of E[x1 - x0 | x_t = z] on the time grid (s = 0..TS).
  return (x, y, s) => {
    const t = s / TS, cells = grids[s], cx = Math.floor(x / C), cy = Math.floor(y / C);
    let sw = 0, vx = 0, vy = 0;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
      const arr = cells.get((cx + a) * 100003 + cy + b);
      if (!arr) continue;
      for (const i of arr) {
        const dx = x - pairs.x0[2 * i] - t * pairs.d[2 * i], dy = y - pairs.x0[2 * i + 1] - t * pairs.d[2 * i + 1];
        const w = Math.exp(-(dx * dx + dy * dy) / (2 * BW * BW));
        sw += w; vx += w * pairs.d[2 * i]; vy += w * pairs.d[2 * i + 1];
      }
    }
    return sw > 1e-12 ? [vx / sw, vy / sw] : [0, 0];
  };
}

// Euler with n steps of a field given on the grid s = 0..TS (n divides TS).
function eulerGrid(v, x, y, n) {
  const out = [[x, y]];
  for (let k = 0; k < n; k++) {
    const [vx, vy] = v(x, y, (k * TS) / n);
    x += vx / n; y += vy / n;
    out.push([x, y]);
  }
  return out;
}
function eulerExact(gmm, x, y, n) {
  for (let k = 0; k < n; k++) { const [vx, vy] = gmm.velocity(x, y, k / n); x += vx / n; y += vy / n; }
  return [x, y];
}

function segCross(a, b) {
  const o = (p, q, r) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return o(a[0], a[1], b[0]) !== o(a[0], a[1], b[1]) && o(b[0], b[1], a[0]) !== o(b[0], b[1], a[1]);
}
function couplingStats(segs) {
  let c = 0, tot = 0, cost = 0;
  segs.forEach((s, i) => {
    cost += (s[1][0] - s[0][0]) ** 2 + (s[1][1] - s[0][1]) ** 2;
    for (let j = i + 1; j < segs.length; j++) { tot++; c += segCross(s, segs[j]); }
  });
  return { cross: c / tot, cost: cost / segs.length };
}

export default {
  id: 'reflow',
  title: 'Reflow straightens paths',
  duration: 56,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.ind[0], label: 'I · Independent coupling' },
    { t: T.rf1[0], label: 'II · 1-rectified flow' },
    { t: T.re[0], label: 'III · Reflow' },
    { t: T.chart[0], label: 'IV · Error vs steps' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 1000, h: 900 },

  setup(env) {
    const mix = spiralMixture(), { gmm } = mix;
    const ode = (x, y) => gmm.integrateODE(x, y, ODE);
    const end = (tr) => [tr[2 * ODE], tr[2 * ODE + 1]];

    // Reflow training pairs (x0, Phi(x0)).
    const rp = makeRng(11), pairs = { n: NPAIR, x0: new Float64Array(2 * NPAIR), d: new Float64Array(2 * NPAIR) };
    for (let i = 0; i < NPAIR; i++) {
      const x = rp.gauss(), y = rp.gauss(), [ex, ey] = end(ode(x, y));
      pairs.x0[2 * i] = x; pairs.x0[2 * i + 1] = y; pairs.d[2 * i] = ex - x; pairs.d[2 * i + 1] = ey - y;
    }
    const v2 = fitReflow(pairs);

    // Particles on screen.
    const rng = makeRng(5), P = [];
    for (let i = 0; i < N; i++) {
      const x = rng.gauss(), y = rng.gauss(), dat = gmm.sample(rng);
      const tr = ode(x, y), e = end(tr);
      const v0 = gmm.velocity(x, y, 0);
      const re = eulerGrid(v2, x, y, TS);
      P.push({
        x0: [x, y], x1: [dat.x, dat.y], uInd: mix.us[dat.k],
        tr, e, u: mix.us[gmm.argmaxComponent(e[0], e[1])],
        j1: [x + v0[0], y + v0[1]], re, j2: eulerGrid(v2, x, y, 1)[1],
      });
    }
    const statInd = couplingStats(P.slice(0, NSEG).map((p) => [p.x0, p.x1]));
    const statRe = couplingStats(P.slice(0, NSEG).map((p) => [p.x0, p.e]));

    // A crossing pair of independent segments that meet at nearly the same time, at a wide angle.
    let best = null;
    for (let i = 0; i < 120; i++) for (let j = i + 1; j < 120; j++) {
      const a = P[i], b = P[j];
      const da = [a.x1[0] - a.x0[0], a.x1[1] - a.x0[1]], db = [b.x1[0] - b.x0[0], b.x1[1] - b.x0[1]];
      for (let k = 10; k <= 80; k++) {
        const t = k / 100;
        const dx = a.x0[0] + t * da[0] - b.x0[0] - t * db[0], dy = a.x0[1] + t * da[1] - b.x0[1] - t * db[1];
        const dist = Math.hypot(dx, dy);
        const cos = (da[0] * db[0] + da[1] * db[1]) / (Math.hypot(...da) * Math.hypot(...db));
        const z = [a.x0[0] + t * da[0], a.x0[1] + t * da[1]];
        const score = dist + 0.5 * Math.abs(cos) + 0.15 * Math.hypot(z[0] + 0.6, z[1] - 0.2);
        if (dist < 0.04 && (!best || score < best.score)) best = { i, j, t, score, z };
      }
    }

    // Error vs NFE against the exact 1-RF endpoint, on held-out noise.
    const rt = makeRng(77), tests = [];
    for (let i = 0; i < NTEST; i++) { const x = rt.gauss(), y = rt.gauss(); const tr = gmm.integrateODE(x, y, 120); tests.push({ x, y, e: [tr[240], tr[241]] }); }
    const errOf = (f) => tests.reduce((a, s) => { const [x, y] = f(s); return a + Math.hypot(x - s.e[0], y - s.e[1]); }, 0) / NTEST;
    const curve1 = NFES.map((n) => ({ n, err: errOf((s) => eulerExact(gmm, s.x, s.y, n)) }));
    const curve2 = NFES.map((n) => ({ n, err: errOf((s) => eulerGrid(v2, s.x, s.y, n)[n]) }));

    return { mix, P, statInd, statRe, cross: best, curve1, curve2, glow: renderGlow(env, plot, mix) };
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

const P2 = (p) => px(p[0], p[1]);
function line(ctx, a, b, color, alpha, width = 1) {
  const [x0, y0] = P2(a), [x1, y1] = P2(b);
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.restore();
}
function polyline(ctx, pts, f, color, alpha, width = 1) {
  // pts: array of [x,y] or flat Float32Array; draws the first fraction f.
  const flat = !Array.isArray(pts), n = flat ? pts.length / 2 : pts.length;
  const at = (i) => (flat ? [pts[2 * i], pts[2 * i + 1]] : pts[i]);
  const m = f * (n - 1);
  if (m <= 0) return;
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
  ctx.beginPath();
  let [x, y] = P2(at(0)); ctx.moveTo(x, y);
  for (let i = 1; i <= Math.ceil(m); i++) {
    const w = Math.min(m - (i - 1), 1), a = at(i - 1), b = at(i);
    [x, y] = px(lerp(a[0], b[0], w), lerp(a[1], b[1], w)); ctx.lineTo(x, y);
  }
  ctx.stroke(); ctx.restore();
}
const pathAt = (pts, f) => {
  const flat = !Array.isArray(pts), n = flat ? pts.length / 2 : pts.length;
  const at = (i) => (flat ? [pts[2 * i], pts[2 * i + 1]] : pts[i]);
  const m = clamp(f) * (n - 1), i = Math.min(Math.floor(m), n - 2), w = m - i, a = at(i), b = at(i + 1);
  return [lerp(a[0], b[0], w), lerp(a[1], b[1], w)];
};
const mix2 = (a, b, w) => [lerp(a[0], b[0], w), lerp(a[1], b[1], w)];

function drawScene(ctx, th, t, st, A, clean) {
  const { P } = st;
  const sA = window_(t, T.setup[0], T.setup[1], 0.5);
  const iA = window_(t, ...T.ind, 0.5), rA = window_(t, ...T.rf1, 0.5), eA = window_(t, ...T.re, 0.5);

  // Setup: noise cloud.
  if (sA > 0) {
    const a = A * (clean ? 1 : sA) * seg(t, ...T.noiseIn);
    for (const p of P) { const [x, y] = P2(p.x0); dot(ctx, th, x, y, th.muted, a); }
  }

  // Act I: independent pairs as straight lines, particles moving along them.
  if (iA > 0) {
    const a = A * iA, lf = ease.inOut(seg(t, ...T.indLines));
    const mv = ease.inOut(seg(t, ...T.indMove)) * (1 - ease.inOut(seg(t, T.cross[0], T.cross[0] + 0.8)));
    const hi = st.cross, hA = seg(t, T.cross[0] + 0.6, T.cross[0] + 1.4);
    P.forEach((p, i) => {
      const c = rampColor(th, p.uInd, 1), isH = hi && (i === hi.i || i === hi.j);
      if (i < NSEG) line(ctx, p.x0, mix2(p.x0, p.x1, lf), c, a * (0.32 - (isH ? 0 : 0.2 * hA)), 1);
      const pos = mix2(p.x0, p.x1, isH ? lerp(mv, hi.t, hA) : mv);
      const [x, y] = P2(pos);
      dot(ctx, th, x, y, c, a * (isH ? 1 : 1 - 0.65 * hA));
    });
    if (hi && hA > 0) drawCrossing(ctx, th, t, st, a * hA);
  }

  // Act II: the 1-RF ODE paths, then one Euler step.
  if (rA > 0) {
    const a = A * rA, ff = ease.inOut(seg(t, ...T.rf1Flow)), jf = ease.inOut(seg(t, ...T.rf1Jump));
    const jA = seg(t, T.rf1Jump[0] - 0.6, T.rf1Jump[0]);
    P.forEach((p, i) => {
      const c = rampColor(th, p.u, 1);
      if (i < NSEG) polyline(ctx, p.tr, ff, c, a * (0.34 - 0.16 * jA), 1);
      if (jA > 0) line(ctx, p.x0, mix2(p.x0, p.j1, jf), c, a * jA * 0.3, 1);
      const pos = jA > 0 ? mix2(p.x0, p.j1, jf) : pathAt(p.tr, ff);
      const [x, y] = P2(pos);
      dot(ctx, th, x, y, c, a);
    });
  }

  // Act III: reflow coupling, reflowed paths, one Euler step.
  if (eA > 0) {
    const a = A * eA, lf = ease.inOut(seg(t, ...T.reLines)), ff = ease.inOut(seg(t, ...T.reFlow)), jf = ease.inOut(seg(t, ...T.reJump));
    const fA = seg(t, T.reFlow[0] - 0.4, T.reFlow[0]), jA = seg(t, T.reJump[0] - 0.6, T.reJump[0]);
    P.forEach((p, i) => {
      const c = rampColor(th, p.u, 1);
      if (i < NSEG) {
        line(ctx, p.x0, mix2(p.x0, p.e, lf), c, a * (0.34 - 0.2 * fA), 1);
        if (fA > 0) polyline(ctx, p.re, ff, c, a * fA * (0.34 - 0.14 * jA), 1);
      }
      let pos = p.x0;
      if (jA > 0) pos = mix2(p.x0, p.j2, jf);
      else if (fA > 0) pos = pathAt(p.re, ff);
      const [x, y] = P2(pos);
      dot(ctx, th, x, y, c, a);
    });
  }
}

// Two independent pairs pass through the same point at the same time with different directions.
function drawCrossing(ctx, th, t, st, a) {
  const { i, j, z } = st.cross, A = st.P[i], B = st.P[j];
  const [cx, cy] = px(z[0], z[1]);
  for (const p of [A, B]) {
    line(ctx, p.x0, p.x1, rampColor(th, p.uInd, 1), a * 0.95, 2.4);
    const d = [p.x1[0] - p.x0[0], p.x1[1] - p.x0[1]], s = 0.32;
    const [ex, ey] = px(z[0] + s * d[0], z[1] + s * d[1]);
    arrow(ctx, cx, cy, ex, ey, { color: rampColor(th, p.uInd, 1), width: 2.6, head: 11, alpha: a });
  }
  const m = [(A.x1[0] - A.x0[0] + B.x1[0] - B.x0[0]) / 2, (A.x1[1] - A.x0[1] + B.x1[1] - B.x0[1]) / 2];
  const mA = a * seg(t, T.cross[0] + 2.0, T.cross[0] + 2.8);
  const [mx, my] = px(z[0] + 0.32 * m[0], z[1] + 0.32 * m[1]);
  arrow(ctx, cx, cy, mx, my, { color: th.fg, width: 3, head: 12, alpha: mA });
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = th.fg; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.arc(cx, cy, 16, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
  text(ctx, 'same place, same time', cx + 34, cy + 44, { size: 17, weight: 600, color: th.fg, alpha: a });
  text(ctx, 'the field can only point to the average', cx + 34, cy + 68, { size: 15, color: th.muted, alpha: mA });
}

function drawChart(ctx, th, t, alpha, st) {
  const L0 = 780, R0 = 1460, Tp = 170, B = 690;
  const lo = -2, hi = 0.5; // log10 error range
  const X = (n) => L0 + (Math.log2(n) / 5) * (R0 - L0), Y = (e) => B - ((Math.log10(e) - lo) / (hi - lo)) * (B - Tp);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(L0, Tp - 10); ctx.lineTo(L0, B); ctx.lineTo(R0 + 10, B); ctx.stroke();
  ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
  for (const e of [0.01, 0.1, 1]) { ctx.beginPath(); ctx.moveTo(L0, Y(e)); ctx.lineTo(R0, Y(e)); ctx.stroke(); }
  // Mode width reference.
  ctx.setLineDash([6, 6]); ctx.strokeStyle = th.muted;
  ctx.beginPath(); ctx.moveTo(L0, Y(st.mix.std)); ctx.lineTo(R0, Y(st.mix.std)); ctx.stroke();
  ctx.restore();
  text(ctx, 'width of one data mode (σ = 0.075)', L0 + 150, Y(st.mix.std) + 22, { size: 15, color: th.muted, alpha });
  for (const e of [0.01, 0.1, 1]) text(ctx, String(e), L0 - 14, Y(e) + 6, { size: 17, color: th.muted, align: 'right', alpha });
  for (const n of NFES) text(ctx, String(n), X(n), B + 28, { size: 17, color: th.muted, align: 'center', alpha });
  text(ctx, 'Euler steps (log scale)', (L0 + R0) / 2, B + 62, { size: 18, color: th.muted, align: 'center', alpha });
  text(ctx, 'mean distance to the exact ODE sample (log scale)', L0 - 60, Tp - 34, { size: 18, color: th.muted, alpha });

  const grow = ease.inOut(seg(t, ...T.chartDraw));
  const series = [
    { pts: st.curve1, color: rampColor(th, 0.05, 1), name: '1-rectified flow' },
    { pts: st.curve2, color: rampColor(th, 1, 1), name: 'after one reflow' },
  ];
  for (const s of series) {
    const f = grow * (s.pts.length - 1);
    ctx.save();
    ctx.globalAlpha = alpha; ctx.strokeStyle = ctx.fillStyle = s.color; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(X(s.pts[0].n), Y(s.pts[0].err));
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1), p = s.pts[i - 1], q = s.pts[i];
      ctx.lineTo(lerp(X(p.n), X(q.n), w), Y(10 ** lerp(Math.log10(p.err), Math.log10(q.err), w)));
    }
    ctx.stroke();
    for (let i = 0; i <= Math.floor(f); i++) { ctx.beginPath(); ctx.arc(X(s.pts[i].n), Y(s.pts[i].err), 5.5, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
  }
  const la = alpha * seg(grow, 0.85, 1), c1 = st.curve1, c2 = st.curve2;
  text(ctx, '1-rectified flow', X(1) + 16, Y(c1[0].err) - 14, { size: 19, weight: 700, color: rampColor(th, 0.05, 1), alpha: la });
  text(ctx, 'after one reflow', X(4), Y(c2[2].err) - 16, { size: 19, weight: 700, color: rampColor(th, 1, 1), alpha: la });
  text(ctx, `${c1[0].err.toFixed(2)}`, X(1) - 12, Y(c1[0].err) + 6, { size: 15, color: th.fg, align: 'right', alpha: la, font: 'mono' });
  text(ctx, `${c2[0].err.toFixed(3)}`, X(1) - 12, Y(c2[0].err) + 6, { size: 15, color: th.fg, align: 'right', alpha: la, font: 'mono' });
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup · Noise and data',
    body: 'Grey dots are Gaussian noise, and the glow is the spiral mixture. A flow model learns a velocity field whose ODE carries every noise point to a data point. The curvature of those paths decides how many solver steps sampling needs.',
    eq: '' },
  { span: T.ind, act: 'Act I · Independent coupling',
    body: 'Flow matching pairs each noise point with a random data point and regresses the velocity onto the straight line between them. The lines cross. Where two of them pass through the same point at the same time, the field cannot follow both and learns their average.',
    eq: 'v(x, t) = E[x₁ − x₀ | xₜ = x]' },
  { span: T.rf1, act: 'Act II · 1-rectified flow',
    body: 'The averaged field has curved paths that never cross. They are exact here. At t = 0 the field points from each noise sample to the mean of the data, so a single Euler step sends every sample to that same point.',
    eq: 'x₁ ≈ x₀ + v(x₀, 0)' },
  { span: T.re, act: 'Act III · Reflow',
    body: 'Reflow keeps the noise and replaces its partner by the endpoint of its own ODE path. These new pairs are much straighter and almost never cross, so the refitted field moves along nearly straight lines. One Euler step now lands on the spiral.',
    eq: '(x₀, x₁)  ↦  (x₀, ODE₁(x₀))' },
  { span: T.chart, act: 'Act IV · Steps versus error',
    body: 'Error of n Euler steps, measured against the exact 1-rectified sample from the same noise. The original flow needs about 16 steps to get within one mode width. After reflow, one step is enough, and the remaining error comes from the finite fit.',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §5 DISTILLATION', 'Reflow', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 24, color: th.fg, alpha: a, font: 'serif' });
  }

  // Coupling readouts: independent vs reflow.
  const ra = seg(t, T.ind[0] + 1, T.ind[0] + 2), rb = seg(t, T.reLines[1] - 0.4, T.reLines[1] + 0.4);
  const jb = seg(t, T.rf1Jump[1] - 0.2, T.rf1Jump[1] + 0.4), jc = seg(t, T.reJump[1] - 0.2, T.reJump[1] + 0.4);
  if (ra > 0) {
    const y0 = 660;
    text(ctx, 'before', x + 250, y0, { size: 15, weight: 600, color: th.muted, align: 'right', alpha: ra });
    text(ctx, 'after reflow', x + 400, y0, { size: 15, weight: 600, color: th.muted, align: 'right', alpha: rb });
    const rows = [
      ['crossing segments', `${(100 * st.statInd.cross).toFixed(1)}%`, `${(100 * st.statRe.cross).toFixed(1)}%`, ra, rb],
      ['transport cost E‖x₁ − x₀‖²', st.statInd.cost.toFixed(2), st.statRe.cost.toFixed(2), ra, rb],
      ['one-step error', st.curve1[0].err.toFixed(2), st.curve2[0].err.toFixed(3), jb, jc],
    ];
    rows.forEach(([lab, a1, a2, al, bl], k) => {
      const y = y0 + 36 + 34 * k;
      text(ctx, lab, x, y, { size: 16, color: th.fg, alpha: ra });
      text(ctx, a1, x + 250, y, { size: 17, color: th.fg, align: 'right', font: 'mono', alpha: al });
      text(ctx, a2, x + 400, y, { size: 17, weight: 700, color: th.accent, align: 'right', font: 'mono', alpha: bl });
    });
  }
  text(ctx, 'Exact spiral-mixture flow. The reflowed field is a kernel regression on 8000 pairs (no network).', x, 862, { size: 14, color: th.muted });
}
