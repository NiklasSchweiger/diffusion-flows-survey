// Live scenes of the §5 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { XMIN, XMAX, quantile, kde } from '../lib/mix1d.js';
import {
  MIX1, XG, vel, flowMap, pData, W1_EXACT, w1, eulerPath, KS2, eulerW1, teacher3, distilRound, runModel, modelW1,
  MC, NPAIR4, consistency, cmSamples, EXACT_TWO, ubar, meanflowSides, jumpsW1, NR5, NSHOW5, KRK, BW5, MIX2,
  reflowData, reflowVel, euler2, straightness, reflowPaths, endErr, teacherVel2, NP7, BOUND, TR7, DATA_MASS, makeDMD,
} from './sim.js';

// ---------- Helpers ----------
// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const WIDE = 640, MU = MIX1.mu;
function panels(w, h, k, gap = 26) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * 14) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + 14), w, ph]);
}
const aspect = (k) => (w) => (w >= WIDE ? (k === 3 ? 16 / 5.6 : 16 / 6.4) : 0.6 / (k === 3 ? 1.5 : 1));
const listSlider = (id, list, value, on, fmt = String, tex) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt, tex }, on);
function axes(ctx, pal, [x0, y0, w, h], { xmin = XMIN, xmax = XMAX, ymin, ymax, xt = MU, yt, xl = '', title, yfmt = String, xfmt = String, titleColor, right = 10 }) {
  const L = x0 + 44, R = x0 + w - right, T = y0 + 30, B = y0 + h - 30;
  const X = (v) => L + (R - L) * (v - xmin) / (xmax - xmin), Y = (v) => B - (B - T) * (v - ymin) / (ymax - ymin);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
  yt.forEach((q) => label(ctx, yfmt(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'));
  xt.forEach((q) => label(ctx, xfmt(q), X(q), B + 16, pal.muted, 11, 'center'));
  if (xl) label(ctx, xl, R, B + 28, pal.muted, 11.5, 'right');
  if (title) label(ctx, title, L, T - 12, titleColor ?? pal.fg, 12.5);
  return { X, Y, L, R, T, B };
}
function polyline(ctx, xs, ys, color, lw = 2, dash = []) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x, ys[i]));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}
// A curve given on the abscissae xs, optionally filled down to y = 0, clipped to the plot.
function curve(ctx, ax, ys, color, { lw = 2, dash = [], fill = 0, xs = XG } = {}) {
  clip(ctx, ax.L, ax.T - 1, ax.R - ax.L, ax.B - ax.T + 2);
  if (fill) {
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ax.X(x), ax.Y(ys[i])));
    ctx.lineTo(ax.X(xs[xs.length - 1]), ax.Y(0)); ctx.lineTo(ax.X(xs[0]), ax.Y(0)); ctx.closePath();
    ctx.fillStyle = color; ctx.globalAlpha = fill; ctx.fill(); ctx.globalAlpha = 1;
  }
  if (lw) polyline(ctx, Array.from(xs, ax.X), Array.from(ys, ax.Y), color, lw, dash);
  ctx.restore();
}
function legend(ctx, ax, items, left = false, bottom = false) {
  items.forEach(([text, color, dash], i) => {
    const y = bottom ? ax.B - 12 - 16 * (items.length - 1 - i) : ax.T + 14 + 16 * i;
    if (left) { hdash(ctx, color, ax.L + 8, ax.L + 28, y, 2, dash ?? []); label(ctx, text, ax.L + 32, y + 4, color, 11.5); return; }
    const x = ax.R - 8;
    label(ctx, text, x - 24, y + 4, color, 11.5, 'right');
    hdash(ctx, color, x - 20, x, y, 2, dash ?? []);
  });
}
const niceMax = (v) => { const s = [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30]; return s.find((q) => q >= v) ?? v; };
const dmax = (...arrs) => niceMax(1.08 * Math.max(...arrs.map((a) => Math.max(...a))));
// Room above the curves for a legend in the top-left corner.
const dmaxL = (...arrs) => niceMax(1.45 * Math.max(...arrs.map((a) => Math.max(...a))));
const densAxes = (ctx, pal, rect, ymax, title, titleColor) => axes(ctx, pal, rect, { ymin: 0, ymax, yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(ymax < 1 ? 2 : 1) : '0'), xl: 'x', title, titleColor });
const memo = (f) => { const m = new Map(); return (k, ...a) => { if (!m.has(k)) m.set(k, f(...a)); return m.get(k); }; };
const colJump = (pal) => pal.ramp[3];
const tGrid = (n, a = 0, b = 1) => Float64Array.from({ length: n + 1 }, (_, i) => a + (b - a) * i / n);
const pathAxes = (ctx, pal, rect, title, ymin = -3.5, ymax = 3.5) => axes(ctx, pal, rect, { xmin: 0, xmax: 1, ymin, ymax, xt: [0, 0.25, 0.5, 0.75, 1], yt: [-3, -2, -1, 0, 1, 2, 3].filter((q) => q >= ymin && q <= ymax), xl: 't', title });
// Teacher paths: in 1D the flow map keeps quantiles, so the path of quantile q is t ↦ F_t^{-1}(q).
const TP = tGrid(60), QP = Array.from({ length: 40 }, (_, i) => (i + 0.5) / 40), QMARK = 24;
let paths1 = null;
const teacherPaths = () => (paths1 ??= QP.map((q) => Float64Array.from(TP, (t) => quantile(MIX1, q, t))));
function drawPaths(ctx, ax, color, alpha = 0.35, lw = 1) {
  clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T); ctx.globalAlpha = alpha;
  teacherPaths().forEach((p) => polyline(ctx, Array.from(TP, ax.X), Array.from(p, ax.Y), color, lw));
  ctx.globalAlpha = 1; ctx.restore();
}

// ---------- Step 1: the flow map ----------
function flowmapScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let s = 0.2, t = 0.7;
  const tv = () => t;   // t below s is read as t = s
  viz.append(c.cv, controls(slider('s', { min: 0, max: 0.95, step: 0.05, value: s }, (v) => { s = v; draw(); }), slider('t', { min: 0, max: 1, step: 0.05, value: t }, (v) => { t = v; draw(); })), ro,
    note('Exact: the paths are the quantile curves t ↦ F<sub>t</sub><sup>−1</sup>(q) of the noised mixture, and every jump is F<sub>t</sub><sup>−1</sup>(F<sub>s</sub>(x)) with the inverse found by bisection.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), q = QP[QMARK], t = Math.max(tv(), s);
    const ax = pathAxes(ctx, pal, P1, '40 teacher paths and one jump');
    drawPaths(ctx, ax, hlc('u', pal, pal.ramp[0]));
    vdash(ctx, hlc('s', pal, pal.muted), ax.X(s), ax.T, ax.B); vdash(ctx, hlc('t', pal, pal.muted), ax.X(t), ax.T, ax.B);
    const xs = quantile(MIX1, q, s), xt = flowMap(MIX1, xs, s, t);
    if (t > s) {
      const X0 = ax.X(s), Y0 = ax.Y(xs), X1 = ax.X(t), Y1 = ax.Y(xt), mx = (X0 + X1) / 2, my = Math.min(Y0, Y1) - 40;
      ctx.strokeStyle = hlc('X', pal, colJump(pal)); ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.quadraticCurveTo(mx, my, X1, Y1); ctx.stroke();
    }
    dot(ctx, ax.X(s), ax.Y(xs), 4.5, pal.fg); dot(ctx, ax.X(t), ax.Y(xt), 4.5, hlc('X', pal, colJump(pal)));
    label(ctx, 's', ax.X(s) + 4, ax.B - 6, pal.muted, 11); label(ctx, 't', ax.X(t) + 4, ax.B - 6, pal.muted, 11);
    // The map x ↦ X_{s→t}(x) on the central 99% of p_s.
    const qs = Array.from({ length: 121 }, (_, i) => 0.005 + 0.99 * i / 120), a = qs.map((q2) => quantile(MIX1, q2, s)), b = qs.map((q2) => quantile(MIX1, q2, t));
    const ax2 = axes(ctx, pal, P2, { xmin: -3.5, xmax: 3.5, ymin: -3.5, ymax: 3.5, xt: [-3, -2, -1, 0, 1, 2, 3], yt: [-3, -2, -1, 0, 1, 2, 3], xl: 'x at time s', title: `The map X_{s→t}, s = ${f2(s)}, t = ${f2(t)}` });
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    polyline(ctx, [ax2.X(-3.5), ax2.X(3.5)], [ax2.Y(-3.5), ax2.Y(3.5)], pal.line, 1, [4, 4]);
    polyline(ctx, a.map(ax2.X), b.map(ax2.Y), hlc('X', pal, colJump(pal)), 2.4);
    ctx.restore();
    dot(ctx, ax2.X(xs), ax2.Y(xt), 4.5, hlc('X', pal, colJump(pal)));
    legend(ctx, ax2, [['X_{s→t}(x)', hlc('X', pal, colJump(pal))], ['identity', pal.line, [4, 4]]], true);
    // Composition through t to 1, and the time derivative at t = s.
    const viaT = flowMap(MIX1, xt, t, 1), direct = flowMap(MIX1, xs, s, 1), d = 1e-4, dt = (flowMap(MIX1, xs, s, s + d) - xs) / d;
    ro.innerHTML = `start x = <b>${f3(xs)}</b> at s = ${f2(s)} · X<sub>s→t</sub>(x) = <b>${f3(xt)}</b><br>` +
      `X<sub>t→1</sub>(X<sub>s→t</sub>(x)) = <b>${f3(viaT)}</b> and X<sub>s→1</sub>(x) = <b>${f3(direct)}</b> · ∂<sub>t</sub>X<sub>s→t</sub>(x) at t = s: <b>${f3(dt)}</b>, u<sub>s</sub>(x) = <b>${f3(vel(MIX1, xs, s))}</b>`;
  }
  return { draw };
}

// ---------- Step 2: the cost of the ODE ----------
function eulerScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let K = 4;
  viz.append(c.cv, controls(listSlider('K', KS2, K, (v) => { K = v; draw(); })), ro,
    note('Exact field; the W1 of 2000 stratified starts to p<sub>data</sub> is computed from sorted samples. The red bars are the local errors: the distance after one step between the Euler point and the exact path through the previous Euler point.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const ax = pathAxes(ctx, pal, P1, `Euler with K = ${K} against the exact paths`);
    drawPaths(ctx, ax, pal.muted, 0.25);
    const starts = [6, 14, 24, 33].map((i) => quantile(MIX1, QP[i], 0));
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    starts.forEach((x0, n) => {
      const P = eulerPath(x0, K), Xs = P.map((_, k) => ax.X(k / K));
      if (n === 2) teacherPaths()[24] && polyline(ctx, Array.from(TP, ax.X), Array.from(teacherPaths()[24], ax.Y), pal.fg, 1.6, [6, 4]);
      polyline(ctx, Xs, P.map(ax.Y), hlc(['K', 'h'], pal, colJump(pal)), n === 2 ? 2.2 : 1.2);
      for (let k = 0; k < K; k++) {
        const ex = flowMap(MIX1, P[k], k / K, (k + 1) / K);
        ctx.strokeStyle = pal.bad; ctx.lineWidth = n === 2 ? 3 : 2; ctx.beginPath(); ctx.moveTo(ax.X((k + 1) / K), ax.Y(ex)); ctx.lineTo(ax.X((k + 1) / K), ax.Y(P[k + 1])); ctx.stroke();
        if (K <= 16) dot(ctx, Xs[k + 1], ax.Y(P[k + 1]), 2.5, hlc(['K', 'h'], pal, colJump(pal)));
      }
    });
    ctx.restore();
    legend(ctx, ax, [['exact path', pal.fg, [6, 4]], ['Euler', hlc(['K', 'h'], pal, colJump(pal))], ['local error', pal.bad]], true);
    const E = eulerW1(), L2 = Math.log2, L10 = Math.log10;
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: 7, ymin: -3.3, ymax: 0.5, xt: [0, 1, 2, 3, 4, 5, 6, 7], xfmt: (q) => String(2 ** q), yt: [-3, -2, -1, 0], yfmt: (q) => (q === 0 ? '1' : `1e${q}`), xl: 'K (NFE)', title: 'W1 to p_{data} against the number of steps' });
    hdash(ctx, pal.ok, ax2.L, ax2.R, ax2.Y(L10(W1_EXACT)), 1.4, [5, 4]);
    polyline(ctx, [ax2.X(0), ax2.X(7)], [ax2.Y(L10(E[0])), ax2.Y(L10(E[0] / 128))], pal.line, 1, [2, 4]);
    polyline(ctx, KS2.map((k) => ax2.X(L2(k))), E.map((e) => ax2.Y(L10(e))), colJump(pal), 2.2);
    KS2.forEach((k, i) => dot(ctx, ax2.X(L2(k)), ax2.Y(L10(E[i])), k === K ? 6 : 3.5, k === K ? pal.mark : colJump(pal)));
    legend(ctx, ax2, [['Euler on the exact field', colJump(pal)], ['slope 1/K', pal.line, [2, 4]], ['exact flow map, one jump', pal.ok, [5, 4]]], false, false);
    const i = KS2.indexOf(K);
    ro.innerHTML = `K = <b>${K}</b> steps, NFE = <b>${K}</b> · W1 to p<sub>data</sub> = <b>${f3(E[i])}</b>` + (i ? ` (${f2(E[i - 1] / E[i])} times smaller than with K = ${K / 2})` : '') + ` · one exact jump: <b>${W1_EXACT.toFixed(4)}</b>, the floor of 2000 stratified samples`;
  }
  return { draw };
}

// ---------- Step 3: progressive distillation ----------
const M3 = [4, 8, 16, 32];
const rounds3 = memo((M) => { const out = [teacher3()]; while (out[out.length - 1].n > 1) out.push(distilRound(out[out.length - 1], M)); out.forEach((Mo) => { Mo.w1 = modelW1(Mo); }); return out; });
const ZE = Float64Array.from({ length: 161 }, (_, i) => -3 + 6 * i / 160), EX3 = ZE.map((z) => flowMap(MIX1, z, 0, 1));
function progressiveScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let M = 16, round = 0;
  const next = button('Next round', () => { if (round < 6) round++; draw(); });
  viz.append(c.cv, controls(listSlider('M', M3, M, (v) => { M = v; draw(); }), next, button('Reset', () => { round = 0; draw(); })), ro,
    note('Each round fits, for every new step, a piecewise-linear map with M knots by least squares to two steps of the previous model, on 2000 stratified points of the exact marginal p<sub>t</sub>. The teacher is 64 Euler steps of the exact field.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), R = rounds3(M, M), cur = R[round], T0m = R[0];
    next.disabled = round >= 6;
    const err = ZE.map((z, i) => runModel(cur, z) - EX3[i]), errT = ZE.map((z, i) => runModel(T0m, z) - EX3[i]);
    const ym = niceMax(1.15 * Math.max(0.05, ...err.map(Math.abs), ...errT.map(Math.abs)));
    const ax = axes(ctx, pal, P1, { xmin: -3, xmax: 3, ymin: -ym, ymax: ym, xt: [-3, -2, -1, 0, 1, 2, 3], yt: [-ym, 0, ym], yfmt: (q) => q.toFixed(ym < 0.1 ? 2 : 1), xl: 'noise x_0', title: `Error of the ${cur.n}-step model: X(x_0) − X_{0→1}(x_0)` });
    hdash(ctx, pal.line, ax.L, ax.R, ax.Y(0), 1, []);
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    polyline(ctx, Array.from(ZE, ax.X), errT.map(ax.Y), pal.muted, 1.6, [5, 4]);
    polyline(ctx, Array.from(ZE, ax.X), err.map(ax.Y), hlc('M', pal, pal.ok), 2.2);
    ctx.restore();
    legend(ctx, ax, [[`student, ${cur.n} step${cur.n > 1 ? 's' : ''}`, hlc('M', pal, pal.ok)], ['teacher, 64 Euler steps', pal.muted, [5, 4]]], true);
    const E = eulerW1(), L2 = Math.log2, L10 = Math.log10;
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: 6, ymin: -3.3, ymax: 0.5, xt: [0, 1, 2, 3, 4, 5, 6], xfmt: (q) => String(2 ** q), yt: [-3, -2, -1, 0], yfmt: (q) => (q === 0 ? '1' : `1e${q}`), xl: 'steps (NFE)', title: 'W1 to p_{data} against the number of steps' });
    hdash(ctx, pal.ramp[0], ax2.L, ax2.R, ax2.Y(L10(W1_EXACT)), 1.4, [5, 4]);
    polyline(ctx, KS2.slice(0, 7).map((k) => ax2.X(L2(k))), E.slice(0, 7).map((e) => ax2.Y(L10(e))), colJump(pal), 2);
    const done = R.slice(0, round + 1);
    polyline(ctx, done.map((Mo) => ax2.X(L2(Mo.n))), done.map((Mo) => ax2.Y(L10(Mo.w1))), hlc('M', pal, pal.ok), 2.2);
    done.forEach((Mo, i) => dot(ctx, ax2.X(L2(Mo.n)), ax2.Y(L10(Mo.w1)), i === round ? 6 : 3.5, i === round ? pal.mark : pal.ok));
    legend(ctx, ax2, [['Euler on the exact field', colJump(pal)], [`distilled, M = ${M}`, hlc('M', pal, pal.ok)], ['exact flow map', pal.ramp[0], [5, 4]]], false, true);
    ro.innerHTML = `M = <b>${M}</b> knots per step · round <b>${round}</b> of 6 · <b>${cur.n}</b> step${cur.n > 1 ? 's' : ''} · W1 = <b>${f3(cur.w1)}</b> · teacher (64 Euler steps) <b>${f3(T0m.w1)}</b> · Euler with ${cur.n} step${cur.n > 1 ? 's' : ''}: <b>${f3(E[KS2.indexOf(cur.n)])}</b>`;
  }
  return { draw };
}

// ---------- Step 4: consistency models ----------
const N4 = [2, 4, 8, 16, 32, 64];
const cm4 = memo((N, kind) => { const C = consistency(N, kind), S = cmSamples(C); return { C, S, one: w1(S.one), two: w1(S.two) }; });
function consistencyScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let kind = 'cd', N = 16;
  const bD = button('Distillation (CD)', () => { kind = 'cd'; sync(); }), bT = button('Training (CT)', () => { kind = 'ct'; sync(); });
  const sync = () => { bD.classList.toggle('on', kind === 'cd'); bT.classList.toggle('on', kind === 'ct'); draw(); };
  bD.classList.add('on');
  viz.append(c.cv, controls(bD, bT, listSlider('N', N4, N, (v) => { N = v; draw(); })), ro,
    note(`One piecewise-linear map with ${MC} knots per grid time t<sub>j</sub> = j/N, fitted backwards from f(x, 1) = x by least squares. CD targets come from one Euler step of the exact field on 2000 points of p<sub>t</sub>; CT targets from the same seeded pair (x<sub>0</sub>, x<sub>1</sub>) at the next time, with ${NPAIR4} pairs.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), R = cm4(kind + N, N, kind), C = R.C;
    const ax = axes(ctx, pal, P1, { xmin: -3.5, xmax: 3.5, ymin: -3.5, ymax: 3.5, xt: [-3, -2, -1, 0, 1, 2, 3], yt: [-3, -2, -1, 0, 1, 2, 3], xl: 'x at time t', title: 'f_θ(x, t) against the exact X_{t→1}(x)' });
    const js = [...new Set([0, N / 4, N / 2, (3 * N) / 4].map((j) => Math.floor(j)))];
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    js.forEach((j, n) => {
      const t = j / N, qs = Array.from({ length: 81 }, (_, i) => 0.005 + 0.99 * i / 80), xs = qs.map((q) => quantile(MIX1, q, t));
      polyline(ctx, xs.map(ax.X), xs.map((x) => ax.Y(flowMap(MIX1, x, t, 1))), pal.muted, 1.4, [5, 4]);
      polyline(ctx, xs.map(ax.X), xs.map((x) => ax.Y(C.f[j](x))), hlc('f', pal, pal.ramp[n]), 2.2);
    });
    ctx.restore();
    legend(ctx, ax, [...js.map((j, n) => [`f_θ at t = ${f2(j / N)}`, hlc('f', pal, pal.ramp[n])]), ['exact X_{t→1}', pal.muted, [5, 4]]], false, true);
    const k1 = kde(Array.from(R.S.one), 0.08), k2 = kde(Array.from(R.S.two), 0.08);
    const ax2 = densAxes(ctx, pal, P2, dmaxL(k1, k2, pData), 'One-step and two-step samples');
    curve(ctx, ax2, pData, hlc('pd', pal, pal.muted), { lw: 1.4, fill: 0.14 });
    curve(ctx, ax2, k1, hlc('f', pal, pal.ok), { lw: 2.2 });
    curve(ctx, ax2, k2, colJump(pal), { lw: 1.8, dash: [5, 4] });
    legend(ctx, ax2, [['p_{data}', hlc('pd', pal, pal.muted)], ['one step', hlc('f', pal, pal.ok)], ['two steps, re-noised at t = 0.5', colJump(pal), [5, 4]]], true);
    const iE = KS2.indexOf(N);
    ro.innerHTML = `${kind === 'cd' ? 'Consistency distillation' : 'Consistency training'}, N = <b>${N}</b>, Δ = ${f3(1 / N)} · W1 one step <b>${f3(R.one)}</b>, two steps <b>${f3(R.two)}</b><br>` +
      `Euler with N = ${N} steps: <b>${f3(eulerW1()[iE])}</b> · two steps of the exact map with the same re-noising: <b>${f3(EXACT_TWO)}</b>`;
  }
  return { draw };
}

// ---------- Step 5: reflow in 2D ----------
const K5 = [1, 2, 4];
let rp5 = null;
const reflowed = () => (rp5 ??= reflowPaths(24));
const ends5 = memo((field, K) => euler2(field, K));
const L5 = 3.4;
function sqAxes(ctx, pal, [x0, y0, w, h], title) {
  const s = Math.min(w - 54, h - 44), L = x0 + 44 + (w - 54 - s) / 2, T = y0 + 30;
  const X = (v) => L + s * (v + L5) / (2 * L5), Y = (v) => T + s * (L5 - v) / (2 * L5);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, s, s);
  label(ctx, title, L, T - 12, pal.fg, 12.5);
  [-3, 0, 3].forEach((q) => { label(ctx, String(q), X(q), T + s + 14, pal.muted, 11, 'center'); label(ctx, String(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'); });
  return { X, Y, L, T, R: L + s, B: T + s };
}
function dataBlobs(ctx, ax, pal) {
  MIX2.means.forEach(([mx, my]) => [2, 1].forEach((k) => { ctx.beginPath(); ctx.arc(ax.X(mx), ax.Y(my), (ax.X(k * MIX2.std) - ax.X(0)), 0, 7); ctx.fillStyle = pal.muted; ctx.globalAlpha = 0.12; ctx.fill(); ctx.globalAlpha = 1; }));
}
function reflowScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.78 }), ro = readout();
  let view = 'teacher', K = 1;
  const bT = button('Teacher', () => { view = 'teacher'; sync(); }), bR = button('After one reflow', () => { view = 'reflow'; sync(); });
  const sync = () => { bT.classList.toggle('on', view === 'teacher'); bR.classList.toggle('on', view === 'reflow'); draw(); };
  bT.classList.add('on');
  viz.append(c.cv, controls(bT, bR, listSlider('K', K5, K, (v) => { K = v; draw(); })), ro,
    note(`Teacher exact: RK4 with ${KRK} steps on the exact field of the 2D mixture. Estimated: the reflowed field v is a kernel regression (bandwidth ${BW5}) of x<sub>1</sub> − x<sub>0</sub> on x<sub>t</sub> over ${NR5} seeded teacher pairs. ${NSHOW5} of the pairs are drawn.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), D = reflowData();
    const P = view === 'teacher' ? D.paths : reflowed(), col = view === 'teacher' ? pal.ramp[0] : pal.ok;
    const ax = sqAxes(ctx, pal, P1, view === 'teacher' ? 'Teacher ODE paths' : 'Paths of the reflowed field');
    dataBlobs(ctx, ax, pal);
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    ctx.globalAlpha = 0.35;
    P.slice(0, 150).forEach((p) => polyline(ctx, p.map((q) => ax.X(q[0])), p.map((q) => ax.Y(q[1])), hlc(view === 'teacher' ? 'u' : 'v', pal, col), 1));
    ctx.globalAlpha = 1; ctx.restore();
    const field = view === 'teacher' ? teacherVel2 : (x, y, t) => reflowVel(x, y, t, D);
    const E = ends5(view + K, field, K);
    const ax2 = sqAxes(ctx, pal, P2, `${K} Euler step${K > 1 ? 's' : ''} from ${NSHOW5} starts`);
    dataBlobs(ctx, ax2, pal);
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    E.forEach(([x, y], i) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(ax2.X(x), ax2.Y(y)); ctx.lineTo(ax2.X(D.x1[2 * i]), ax2.Y(D.x1[2 * i + 1])); ctx.stroke(); });
    E.forEach(([x, y]) => dot(ctx, ax2.X(x), ax2.Y(y), 2.2, hlc(['K', view === 'teacher' ? 'u' : 'v'], pal, col)));
    ctx.restore();
    const st = view === 'teacher' ? straightness(D.paths) : straightness(reflowed());
    ro.innerHTML = `${view === 'teacher' ? 'Teacher' : 'After one reflow'} · straightness E|u(x<sub>t</sub>, t) − (x<sub>1</sub> − x<sub>0</sub>)|² = <b>${f3(st)}</b> · K = <b>${K}</b> · mean distance of the samples to the teacher end points of the same noise: <b>${f3(endErr(E, D))}</b> (grey lines)`;
  }
  return { draw };
}

// ---------- Step 6: average velocities ----------
const Q6 = [0.1, 0.25, 0.4, 0.61, 0.75, 0.9], J6 = [1, 2, 4];
let jw6 = null;
function meanflowScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let s = 0.2, t = 0.7;
  viz.append(c.cv, controls(slider('s', { min: 0, max: 0.9, step: 0.05, value: s }, (v) => { s = v; draw(); }), slider('t', { min: 0.05, max: 1, step: 0.05, value: t }, (v) => { t = v; draw(); })), ro,
    note('Exact: ū is computed from the exact flow map, and the total derivative along the path is a finite difference with step 10<sup>−4</sup> along the exact path.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), tt = Math.max(t, s + 0.05), q = QP[QMARK];
    const ax = pathAxes(ctx, pal, P1, 'Tangent at s and chord from s to t', -1.5, 1.5);
    drawPaths(ctx, ax, pal.muted, 0.2);
    const path = teacherPaths()[QMARK];
    polyline(ctx, Array.from(TP, ax.X), Array.from(path, ax.Y), pal.fg, 2);
    const xs = quantile(MIX1, q, s), xt = flowMap(MIX1, xs, s, tt), u0 = vel(MIX1, xs, s), ub = (xt - xs) / (tt - s);
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    polyline(ctx, [ax.X(s - 0.15), ax.X(s + 0.35)], [ax.Y(xs - 0.15 * u0), ax.Y(xs + 0.35 * u0)], hlc('u', pal, pal.ramp[0]), 2, [5, 4]);
    polyline(ctx, [ax.X(s), ax.X(tt)], [ax.Y(xs), ax.Y(xt)], hlc('ub', pal, colJump(pal)), 2.4);
    ctx.restore();
    dot(ctx, ax.X(s), ax.Y(xs), 4.5, pal.fg); dot(ctx, ax.X(tt), ax.Y(xt), 4.5, hlc('ub', pal, colJump(pal)));
    legend(ctx, ax, [['tangent, slope u_s', hlc('u', pal, pal.ramp[0]), [5, 4]], ['chord, slope ū', hlc('ub', pal, colJump(pal))]], true);
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: 1, ymin: -2, ymax: 2, xt: [0, 0.25, 0.5, 0.75, 1], yt: [-2, -1, 0, 1, 2], xl: 'end time t', title: `ū(x_s, s, t) along six paths, s = ${f2(s)}` });
    hdash(ctx, pal.line, ax2.L, ax2.R, ax2.Y(0), 1, []); vdash(ctx, hlc('s', pal, pal.muted), ax2.X(s), ax2.T, ax2.B);
    const TT = tGrid(60, s, 1);
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    Q6.forEach((q2, n) => {
      const x = quantile(MIX1, q2, s), ys = Array.from(TT, (t2) => ubar(x, s, t2));
      polyline(ctx, Array.from(TT, ax2.X), ys.map(ax2.Y), q2 === q ? hlc('ub', pal, colJump(pal)) : pal.muted, q2 === q ? 2.4 : 1.2);
    });
    ctx.restore();
    dot(ctx, ax2.X(s), ax2.Y(u0), 4.5, hlc('u', pal, pal.ramp[0])); dot(ctx, ax2.X(tt), ax2.Y(ub), 4.5, hlc('ub', pal, colJump(pal)));
    const M = meanflowSides(xs, s, tt);
    jw6 ??= J6.map((k) => jumpsW1(k));
    ro.innerHTML = `x<sub>s</sub> = <b>${f3(xs)}</b>, s = ${f2(s)}, t = ${f2(tt)} · ū = <b>${f3(M.lhs)}</b> · u<sub>s</sub> + (t − s) dū/ds = ${f3(M.u)} + ${f2(tt - s)} × ${f3(M.dds)} = <b>${f3(M.rhs)}</b><br>` +
      `samples with K exact jumps of ū: W1 = ${J6.map((k, i) => `<b>${jw6[i].toFixed(4)}</b> (K = ${k})`).join(', ')}`;
  }
  return { draw };
}

// ---------- Step 7: distribution matching ----------
const NIT7 = 400;
function dmdScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let range = 'low', lam = 0, D = null;
  const reset = () => { D = makeDMD({ range, lam }); };
  reset();
  const r = runner({ step: () => D.step(), done: () => D.it >= NIT7, draw: () => draw(), interval: 30, perFrame: 2 });
  const bL = button('Low noise only', () => { range = 'low'; sync(); }), bW = button('Wide range of t', () => { range = 'wide'; sync(); });
  const b0 = button('λ = 0', () => { lam = 0; sync(); }), b1 = button('λ = 0.02', () => { lam = 0.02; sync(); });
  const sync = () => { bL.classList.toggle('on', range === 'low'); bW.classList.toggle('on', range === 'wide'); b0.classList.toggle('on', lam === 0); b1.classList.toggle('on', lam > 0); reset(); draw(); r.kick(); };
  bL.classList.add('on'); b0.classList.add('on');
  viz.append(c.cv, controls(bL, bW, b0, b1, r.play, r.one, button('Reset', () => { reset(); draw(); r.kick(); })), ro,
    note(`Trained in the page from a fixed seed. The generator is ${NP7} particles at fixed noise quantiles, each a free parameter, started on the right mode. s<sub>real</sub> is the exact score of the data; s<sub>fake</sub> is the exact score of the noised particle cloud. Each iteration draws 4 times per particle, t uniform in [${TR7.low}] (low) or [${TR7.wide}] (wide), and takes a gradient step of size 0.5. The density is the cloud smoothed with width 0.08.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), cur = D.cur;
    const ax = densAxes(ctx, pal, P1, dmaxL(cur.p, pData), `Generator after ${D.it} iterations`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.4, fill: 0.14 });
    curve(ctx, ax, cur.p, hlc('gth', pal, pal.ok), { lw: 2.2 });
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    ctx.strokeStyle = hlc('gth', pal, pal.ok); ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
    D.g.forEach((y, i) => { if (i % 4 === 0) { ctx.beginPath(); ctx.moveTo(ax.X(y), ax.B); ctx.lineTo(ax.X(y), ax.B - 8); ctx.stroke(); } });
    ctx.globalAlpha = 1; ctx.restore();
    BOUND.forEach((b) => vdash(ctx, pal.line, ax.X(b), ax.T, ax.B));
    legend(ctx, ax, [['p_{data}', hlc('pd', pal, pal.muted)], ['generator', hlc('gth', pal, pal.ok)]], true);
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: NIT7, ymin: 0, ymax: 1, xt: [0, 100, 200, 300, 400], yt: [0, 0.25, 0.5, 0.75, 1], yfmt: (q) => String(q), xl: 'iteration', title: 'Share of the samples in each mode' });
    const names = ['left', 'middle', 'right'];
    names.forEach((_, k) => {
      hdash(ctx, pal.ramp[k], ax2.L, ax2.R, ax2.Y(DATA_MASS[k]), 1, [2, 3]);
      polyline(ctx, D.hist.map((q) => ax2.X(q.it)), D.hist.map((q) => ax2.Y(q.mass[k])), pal.ramp[k], 2);
    });
    legend(ctx, ax2, names.map((n, k) => [`${n} mode`, pal.ramp[k]]), false, false);
    ro.innerHTML = `${range === 'low' ? 'low noise only' : 'wide range of t'}, λ = <b>${lam}</b>, iteration <b>${D.it}</b> · shares ${cur.mass.map((m) => `<b>${f2(m)}</b>`).join(' / ')} (data ${DATA_MASS.map(f2).join(' / ')}) · W1 to p<sub>data</sub> <b>${f3(cur.w1)}</b> · KL(generator ‖ p<sub>data</sub>) <b>${f3(cur.kl)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

export const SCENES = { flowmap: flowmapScene, euler: eulerScene, progressive: progressiveScene, consistency: consistencyScene, reflow: reflowScene, meanflow: meanflowScene, dmd: dmdScene };
