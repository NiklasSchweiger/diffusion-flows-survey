// Live scenes of the §2.11 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, readout, note, runner } from '../lib/tutorial.js';
import {
  MU, SD, PA, NM, XMIN, XMAX, DATA, COND, XG, NG, logp, score, logClassA, dlogClassA, logh, densAt, piW, sharedMass,
  flow, trails, guidedVel, tiltVel, targetQ, w1, kde, quality, iidW1, PK_A, MODEL_C, MODEL_U, MODEL_W, TG, NK,
  smc, NRUN, N6,
} from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const WIDE = 640;
function panels(w, h, k, gap = 26) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * 14) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + 14), w, ph]);
}
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
// A curve on the grid XG, optionally filled down to y = 0.
function curve(ctx, ax, ys, color, { lw = 2, dash = [], fill = 0 } = {}) {
  clip(ctx, ax.L, ax.T - 1, ax.R - ax.L, ax.B - ax.T + 2);
  if (fill) {
    ctx.beginPath(); XG.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ax.X(x), ax.Y(ys[i])));
    ctx.lineTo(ax.X(XMAX), ax.Y(0)); ctx.lineTo(ax.X(XMIN), ax.Y(0)); ctx.closePath();
    ctx.fillStyle = color; ctx.globalAlpha = fill; ctx.fill(); ctx.globalAlpha = 1;
  }
  if (lw) polyline(ctx, Array.from(XG, ax.X), Array.from(ys, ax.Y), color, lw, dash);
  ctx.restore();
}
// Class colours: A cyan, B pink; the shared mode carries both.
const colA = (pal) => pal.ramp[0], colB = (pal) => pal.ramp[2];
function modeMarks(ctx, pal, ax) {
  MU.forEach((m, k) => {
    const X = ax.X(m), Y = ax.B - 5;
    if (PA[k] === 1 || PA[k] === 0) { ctx.fillStyle = PA[k] ? colA(pal) : colB(pal); ctx.fillRect(X - 7, Y, 14, 4); }
    else { ctx.fillStyle = colA(pal); ctx.fillRect(X - 7, Y, 7, 4); ctx.fillStyle = colB(pal); ctx.fillRect(X, Y, 7, 4); }
  });
}
// Legend lines in the top right corner of a plot.
function legend(ctx, ax, items) {
  items.forEach(([text, color, dash], i) => {
    const y = ax.T + 14 + 16 * i, x = ax.R - 8;
    label(ctx, text, x - 24, y + 4, color, 11.5, 'right');
    hdash(ctx, color, x - 20, x, y, 2, dash ?? []);
  });
}
const niceMax = (v) => { const s = [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30]; return s.find((q) => q >= v) ?? v; };
const dmax = (...arrs) => niceMax(1.08 * Math.max(...arrs.map((a) => Math.max(...a))));
const densAxes = (ctx, pal, rect, ymax, title, titleColor) => axes(ctx, pal, rect, { ymin: 0, ymax, yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(ymax < 1 ? 2 : 1) : '0'), xl: 'x', title, titleColor });
const tSlider = (value, on) => slider('t', { min: 0, max: 1, step: 0.01, value, fmt: f2 }, on);
const memo = (f) => { const m = new Map(); return (k, ...a) => { if (!m.has(k)) m.set(k, f(...a)); return m.get(k); }; };

// ---------- Step 1: Bayes' rule on scores ----------
function bayesScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.4 : 0.6), { maxH: 0.72 }), ro = readout();
  let t = 0.5, xq = -0.75;
  viz.append(c.cv, controls(tSlider(t, (v) => { t = v; draw(); }), slider('x', { min: -4, max: 4, step: 0.05, value: xq, fmt: f2 }, (v) => { xq = v; draw(); })), ro,
    note('Exact closed forms: the densities, the classifier p_t(A | x) = Σ_k p_t(k | x) p(A | k) and the scores are computed from the five mixture components.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const p = densAt(DATA, t), pc = densAt(COND, t), cl = Float64Array.from(XG, (x) => Math.exp(logClassA(x, t)));
    const ymax = dmax(p, pc), ax = densAxes(ctx, pal, P1, ymax, `Densities and classifier at t = ${f2(t)}`);
    curve(ctx, ax, p, hlc('pt', pal, pal.muted), { lw: 1.6, fill: 0.14 });
    curve(ctx, ax, pc, hlc('ptc', pal, colA(pal)), { lw: 2.2 });
    const Yc = (v) => ax.B - (ax.B - ax.T) * v * 0.92;
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    polyline(ctx, Array.from(XG, ax.X), Array.from(cl, Yc), hlc('cls', pal, pal.ramp[1]), 2, [6, 4]);
    ctx.restore();
    [[1, '1'], [0.5, '½']].forEach(([v, s]) => label(ctx, s, ax.R + 4, Yc(v) + 4, pal.ramp[1], 11));
    modeMarks(ctx, pal, ax);
    vdash(ctx, pal.mark, ax.X(xq), ax.T, ax.B, 1.4);
    legend(ctx, ax, [['p_t(x)', hlc('pt', pal, pal.muted)], ['p_t(x | A)', hlc('ptc', pal, colA(pal))], ['p_t(A | x)', hlc('cls', pal, pal.ramp[1]), [6, 4]]]);
    // Scores.
    const su = Float64Array.from(XG, (x) => score(DATA, x, t)), sc = Float64Array.from(XG, (x) => score(COND, x, t)), g = Float64Array.from(XG, (x) => dlogClassA(x, t));
    const sm = niceMax(Math.min(30, 1.05 * Math.max(...[su, sc, g].map((a) => Math.max(...a.map(Math.abs)))))), ax2 = axes(ctx, pal, P2, { ymin: -sm, ymax: sm, yt: [-sm, 0, sm], xl: 'x', title: 'Scores at the same t' });
    hdash(ctx, pal.line, ax2.L, ax2.R, ax2.Y(0), 1, []);
    curve(ctx, ax2, su, hlc('s', pal, pal.muted), { lw: 1.6 });
    curve(ctx, ax2, g, hlc('cls', pal, pal.ramp[1]), { lw: 2, dash: [6, 4] });
    curve(ctx, ax2, sc, hlc('ptc', pal, colA(pal)), { lw: 2.2 });
    modeMarks(ctx, pal, ax2);
    vdash(ctx, pal.mark, ax2.X(xq), ax2.T, ax2.B, 1.4);
    const a = score(DATA, xq, t), b = dlogClassA(xq, t), s = score(COND, xq, t);
    [[a, pal.muted], [b, pal.ramp[1]], [s, colA(pal)]].forEach(([v, col]) => { if (Math.abs(v) <= sm) dot(ctx, ax2.X(xq), ax2.Y(v), 4, col); });
    legend(ctx, ax2, [['∇log p_t(x)', hlc('s', pal, pal.muted)], ['∇log p_t(A | x)', hlc('cls', pal, pal.ramp[1]), [6, 4]], ['∇log p_t(x | A)', hlc('ptc', pal, colA(pal))]]);
    ro.innerHTML = `t = <b>${f2(t)}</b>, x = <b>${f2(xq)}</b> · p_t(A | x) = <b>${f3(Math.exp(logClassA(xq, t)))}</b><br>` +
      `∇log p_t(x) = <b>${f3(a)}</b>, ∇log p_t(A | x) = <b>${f3(b)}</b>, sum <b>${f3(a + b)}</b> · ∇log p_t(x | A) = <b>${f3(s)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: CFG and the density it follows ----------
const W2 = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
function cfgScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.4 : 0.6), { maxH: 0.72 }), ro = readout();
  let wg = 3, t = 1;
  viz.append(c.cv, controls(listSlider('w', W2, wg, (v) => { wg = v; draw(); }), tSlider(t, (v) => { t = v; draw(); })), ro,
    note('Exact: π^w_t is p_t(x) p_t(A | x)^w normalized on a grid of 361 points; the conditional density and the classifier are closed forms.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const pw = piW(wg, t), pc = densAt(COND, t), p = densAt(DATA, t);
    const ax = densAxes(ctx, pal, P1, dmax(pw, pc), `π^w_t at w = ${wg}, t = ${f2(t)}`, hlc('pi', pal, pal.fg));
    curve(ctx, ax, p, pal.muted, { lw: 1.2, dash: [2, 3] });
    curve(ctx, ax, pw, hlc('pi', pal, pal.ramp[3]), { lw: 2.2, fill: 0.2 });
    curve(ctx, ax, pc, hlc('ptc', pal, colA(pal)), { lw: 2, dash: [6, 4] });
    modeMarks(ctx, pal, ax);
    legend(ctx, ax, [['π^w_t', hlc('pi', pal, pal.ramp[3])], ['p_t(x | A) = π^1_t', hlc('ptc', pal, colA(pal)), [6, 4]], ['p_t(x)', pal.muted, [2, 3]]]);
    const cl = Float64Array.from(XG, (x) => logClassA(x, t)), ax2 = axes(ctx, pal, P2, { ymin: 0, ymax: 1.1, yt: [0, 0.5, 1], xl: 'x', title: 'The factor that CFG multiplies by' });
    curve(ctx, ax2, cl.map((q) => Math.exp(q)), hlc('cls', pal, pal.ramp[1]), { lw: 1.8, dash: [6, 4] });
    curve(ctx, ax2, cl.map((q) => Math.exp(wg * q)), hlc('w', pal, pal.ramp[3]), { lw: 2.2, fill: 0.12 });
    modeMarks(ctx, pal, ax2);
    legend(ctx, ax2, [[`p_t(A | x)^w, w = ${wg}`, hlc('w', pal, pal.ramp[3])], ['p_t(A | x)', hlc('cls', pal, pal.ramp[1]), [6, 4]]]);
    ro.innerHTML = `w = <b>${wg}</b>, t = <b>${f2(t)}</b> · mass within 0.6 of the shared mode: π^w_t <b>${f3(sharedMass(pw))}</b>, p_t(x | A) <b>${f3(sharedMass(pc))}</b><br>` +
      `at t = 1 the shared mode has weight (1/2)^w / (2 + (1/2)^w) = <b>${f3(0.5 ** wg / (2 + 0.5 ** wg))}</b> in π^w_1`;
  }
  return { draw };
}

// ---------- Step 3: CFG does not sample its target ----------
const W3 = [1, 1.5, 2, 3, 4, 6, 8];
const runs3 = memo((w) => { const cfg = flow(guidedVel(COND, DATA, w)), tilt = flow(tiltVel(w)), Q = targetQ(w); return { cfg, tilt, kc: kde(cfg), kt: kde(tilt), w1c: w1(cfg, Q), w1t: w1(tilt, Q), qc: quality(cfg), qt: quality(tilt) }; });
function jensenScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.4 : 0.6), { maxH: 0.72 }), ro = readout();
  let wg = 3, t = 0.4;
  viz.append(c.cv, controls(listSlider('w', W3, wg, (v) => { wg = v; draw(); }), tSlider(t, (v) => { t = Math.min(v, 0.99); draw(); })), ro,
    note('Exact scores and an exact h_t (a quadrature over the Gaussian posterior of x_1 in each mode). Both samplers are RK4 on 120 steps from the same 400 noise quantiles; the curves are kernel densities of their end points, and W1 is computed from the sorted end points against the quantiles of π^w_1.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), R = runs3(wg, wg);
    const tg = piW(wg, 1), ax = densAxes(ctx, pal, P1, dmax(tg, R.kc, R.kt), `End points at w = ${wg}`);
    curve(ctx, ax, tg, hlc('pi', pal, pal.fg), { lw: 0, fill: 0.16 });
    curve(ctx, ax, R.kt, hlc('h', pal, pal.ok), { lw: 2.2 });
    curve(ctx, ax, R.kc, hlc('sw', pal, pal.ramp[3]), { lw: 2.2 });
    modeMarks(ctx, pal, ax);
    legend(ctx, ax, [['target π^w_1', pal.muted], ['exact tilt h_t', hlc('h', pal, pal.ok)], ['CFG', hlc('sw', pal, pal.ramp[3])]]);
    // The two log factors at time t.
    const lh = Float64Array.from(XG, (x) => logh(x, t, wg)[0]), lc = Float64Array.from(XG, (x) => wg * logClassA(x, t));
    const ymin = -6, ax2 = axes(ctx, pal, P2, { ymin, ymax: 0.5, yt: [-6, -3, 0], xl: 'x', title: `log h_t and w log p_t(A | x) at t = ${f2(t)}` });
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    ctx.beginPath(); XG.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ax2.X(x), ax2.Y(lh[i])));
    for (let i = NG - 1; i >= 0; i--) ctx.lineTo(ax2.X(XG[i]), ax2.Y(Math.max(lc[i], ymin - 1)));
    ctx.closePath(); ctx.fillStyle = pal.ok; ctx.globalAlpha = 0.14; ctx.fill(); ctx.globalAlpha = 1;
    ctx.restore();
    curve(ctx, ax2, lh, hlc('h', pal, pal.ok), { lw: 2.2 });
    curve(ctx, ax2, lc, hlc('cls', pal, pal.ramp[3]), { lw: 2.2, dash: [6, 4] });
    modeMarks(ctx, pal, ax2);
    legend(ctx, ax2, [['log h_t', hlc('h', pal, pal.ok)], ['w log p_t(A | x)', hlc('cls', pal, pal.ramp[3]), [6, 4]]]);
    const i0 = Math.round((0 - XMIN) / (XMAX - XMIN) * (NG - 1));
    ro.innerHTML = `w = <b>${wg}</b> · W1 to π^w_1: CFG <b>${f3(R.w1c)}</b>, exact tilt <b>${f3(R.w1t)}</b> · share of the shared mode: target <b>${f3(targetShare(wg))}</b>, CFG <b>${f3(R.qc.share[2])}</b>, tilt <b>${f3(R.qt.share[2])}</b><br>` +
      `at t = ${f2(t)} and x = 0 the gap log h_t − w log p_t(A | x) is <b>${f3(lh[i0] - lc[i0])}</b>, a ratio of <b>${f2(Math.exp(lh[i0] - lc[i0]))}</b>`;
  }
  return { draw };
}
const targetShare = (w) => quality(targetQ(w)).share[2];

// ---------- Step 4: where to guide ----------
const W4 = [1, 1.5, 2, 3, 4, 6], AB = Array.from({ length: 11 }, (_, i) => i / 10);
const runs4 = memo((wg, a, b) => { const v = guidedVel(MODEL_C, MODEL_U, wg, a, b), end = flow(v); return { end, k: kde(end), q: quality(end), tr: trails(v), w1: w1(end, targetQ(1)) }; });
// Colour of a path by the class of the mode it ends nearest to.
function endColor(pal, x) { let k = 0; for (let j = 1; j < NM; j++) if (Math.abs(x - MU[j]) < Math.abs(x - MU[k])) k = j; return PA[k] === 1 ? colA(pal) : PA[k] === 0 ? colB(pal) : pal.ramp[1]; }
function shareLabels(ctx, pal, ax, share) {
  MU.forEach((m, k) => {
    const good = PK_A[k] > 0, d = share[k] - PK_A[k];
    label(ctx, f2(share[k]), ax.X(m), ax.T + 14, good ? (Math.abs(d) < 0.03 ? pal.fg : pal.mark) : (share[k] > 0.005 ? colB(pal) : pal.muted), 11, 'center', 600);
  });
  label(ctx, 'share', ax.L + 4, ax.T + 14, pal.muted, 10.5);
}
function intervalScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.6 : 0.58), { maxH: 0.74 }), ro = readout();
  let wg = 3, a = 0, b = 1;
  viz.append(c.cv, controls(listSlider('w', W4, wg, (v) => { wg = v; draw(); }), listSlider('ab', AB, a, (v) => { a = v; draw(); }, f2, 'a'), listSlider('ab', AB, b, (v) => { b = v; draw(); }, f2, 'b')), ro,
    note('The model is exact but wrong on purpose: its modes are 0.45 wide instead of 0.3 and 15% of its conditional mass ignores the condition. All fields are exact mixture fields; 400 paths from noise quantiles, RK4 on 120 steps. Paths are coloured by the mode they end on.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), R = runs4(`${wg}|${a}|${b}`, wg, a, b), R1 = runs4('1|0|1', 1, 0, 1);
    const ax = axes(ctx, pal, P1, { xmin: 0, xmax: 1, ymin: XMIN, ymax: XMAX, xt: [0, 0.5, 1], yt: [-3, 0, 3], xl: 't', title: a < b && wg !== 1 ? `Paths, guided with w = ${wg} on [${f2(a)}, ${f2(b)}]` : 'Paths, no guidance' });
    if (a < b && wg !== 1) { ctx.fillStyle = hlc('ab', pal, pal.ramp[3]); ctx.globalAlpha = 0.13; ctx.fillRect(ax.X(a), ax.T, ax.X(b) - ax.X(a), ax.B - ax.T); ctx.globalAlpha = 1; }
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    R.tr.forEach((tr) => { ctx.globalAlpha = 0.75; polyline(ctx, Array.from(TG, ax.X), Array.from(tr, ax.Y), endColor(pal, tr[NK]), 1.3); });
    ctx.globalAlpha = 1; ctx.restore();
    const pc = densAt(COND, 1), ax2 = densAxes(ctx, pal, P2, dmax(pc, R.k, R1.k), 'Generated density and the shares of the modes');
    curve(ctx, ax2, R1.k, pal.muted, { lw: 1.2, dash: [2, 3] });
    curve(ctx, ax2, pc, colA(pal), { lw: 2, dash: [6, 4] });
    curve(ctx, ax2, R.k, pal.ramp[3], { lw: 2.2, fill: 0.15 });
    modeMarks(ctx, pal, ax2); shareLabels(ctx, pal, ax2, R.q.share);
    legend(ctx, Object.assign({}, ax2, { T: ax2.T + 14 }), [['generated', pal.ramp[3]], ['p(x | A)', colA(pal), [6, 4]], ['model, w = 1', pal.muted, [2, 3]]]);
    ro.innerHTML = `${a < b && wg !== 1 ? `w = <b>${wg}</b> on [<b>${f2(a)}</b>, <b>${f2(b)}</b>]` : 'no guidance (w = 1 or empty interval)'} · precision <b>${f2(R.q.prec)}</b>, coverage <b>${f2(R.q.cov)}</b>, W1 to p(x | A) <b>${f3(R.w1)}</b><br>` +
      `shares of the five modes <b>${Array.from(R.q.share, f2).join(' · ')}</b> against 0.40 · 0.40 · 0.20 · 0 · 0 · without guidance precision ${f2(R1.q.prec)}, coverage ${f2(R1.q.cov)}`;
  }
  return { draw };
}

// ---------- Step 5: autoguidance ----------
const W5 = [1, 1.5, 2, 2.5, 3, 4, 5, 6];
let all5 = null;
const runs5 = () => all5 ??= W5.map((wg) => {
  const cf = flow(guidedVel(MODEL_C, MODEL_U, wg)), ag = flow(guidedVel(MODEL_C, MODEL_W, wg));
  return { cf: { k: kde(cf), q: quality(cf) }, ag: { k: kde(ag), q: quality(ag) } };
});
function autoguidanceScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.6 : 0.52), { maxH: 0.74 }), ro = readout();
  let wg = 4;
  viz.append(c.cv, controls(listSlider('w', W5, wg, (v) => { wg = v; draw(); })), ro,
    note('Same model as in step 4. The weak model has modes 0.65 wide and the same weights as the good conditional model. Precision and coverage from the 400 end points of each sampler.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, all = runs5(), i = W5.indexOf(wg), R = all[i];
    let P1, P2, P3;
    if (wide) { const pw = (w - 26) * 0.56; P1 = [0, 0, pw, h]; P2 = [pw + 26, 0, w - pw - 26, h / 2 - 4]; P3 = [pw + 26, h / 2 + 4, w - pw - 26, h / 2 - 4]; }
    else { const ph = h * 0.46; P1 = [0, 0, w, ph]; P2 = [0, ph + 10, w, (h - ph - 20) / 2]; P3 = [0, ph + 20 + (h - ph - 20) / 2, w, (h - ph - 20) / 2]; }
    const pc = densAt(COND, 1), ax = densAxes(ctx, pal, P1, dmax(pc, R.cf.k, R.ag.k), `Generated densities at w = ${wg}`);
    curve(ctx, ax, pc, colA(pal), { lw: 0, fill: 0.16 });
    curve(ctx, ax, R.cf.k, pal.ramp[3], { lw: 2.2 });
    curve(ctx, ax, R.ag.k, hlc('sweak', pal, pal.ok), { lw: 2.2 });
    modeMarks(ctx, pal, ax);
    legend(ctx, ax, [['p(x | A)', colA(pal)], ['CFG', pal.ramp[3]], ['autoguidance', hlc('sweak', pal, pal.ok)]]);
    [[P2, 'prec', 'Precision against w', 0.5], [P3, 'cov', 'Coverage against w', 0.3]].forEach(([P, key, title, lo]) => {
      const a = axes(ctx, pal, P, { xmin: 1, xmax: 6, ymin: lo, ymax: 1, xt: [1, 2, 3, 4, 5, 6], yt: [lo, 1], xl: 'w', title, yfmt: f2 });
      [['cf', pal.ramp[3]], ['ag', hlc('sweak', pal, pal.ok)]].forEach(([m, col]) => {
        polyline(ctx, W5.map(a.X), all.map((r) => a.Y(r[m].q[key])), col, 2);
        dot(ctx, a.X(wg), a.Y(R[m].q[key]), 4.5, col);
      });
      vdash(ctx, pal.mark, a.X(wg), a.T, a.B, 1.2);
    });
    ro.innerHTML = `w = <b>${wg}</b> · CFG: precision <b>${f2(R.cf.q.prec)}</b>, coverage <b>${f2(R.cf.q.cov)}</b>, shared mode <b>${f2(R.cf.q.share[2])}</b><br>` +
      `autoguidance: precision <b>${f2(R.ag.q.prec)}</b>, coverage <b>${f2(R.ag.q.cov)}</b>, shared mode <b>${f2(R.ag.q.share[2])}</b>, modes of B <b>${f2(R.ag.q.share[3] + R.ag.q.share[4])}</b>`;
  }
  return { draw };
}

// ---------- Step 6: Feynman–Kac SMC ----------
const W6 = [2, 3, 5];
const res6 = new Map(), cfg6 = memo((w) => { const x = flow(guidedVel(COND, DATA, w)); return { k: kde(x), w1: w1(x, targetQ(w)) }; });
const floor6 = memo((w) => iidW1(w, NRUN));
function smcScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.4 : 0.6), { maxH: 0.72 }), ro = readout();
  let wg = 3, N = 8;
  const key = (n) => `${wg}|${n}`, missing = () => N6.find((n) => !res6.has(key(n)));
  const r = runner({
    step: () => { const n = missing(); const s = smc(n, wg); res6.set(key(n), { ...s, k: kde(s.out), w1: w1(s.out, targetQ(wg, NRUN)), q: quality(s.out) }); },
    done: () => missing() === undefined, draw: () => draw(), interval: 30,
  });
  viz.append(c.cv, controls(listSlider('N', N6, N, (v) => { N = v; draw(); }), listSlider('w', W6, wg, (v) => { wg = v; draw(); r.kick(); })), ro,
    note(`Exact conditional scores and classifier; Euler–Maruyama on 100 steps from t = 0.02 to 0.995, the last weight with p(A | x_1) itself. ${NRUN} seeded runs per setting, one output each.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), R = res6.get(key(N)), C = cfg6(wg, wg), fl = floor6(wg, wg);
    const tg = piW(wg, 1), ax = densAxes(ctx, pal, P1, dmax(tg, C.k, R ? R.k : tg), `Outputs at w = ${wg}, N = ${N}`);
    curve(ctx, ax, tg, hlc('pi', pal, pal.fg), { lw: 0, fill: 0.16 });
    curve(ctx, ax, C.k, pal.ramp[3], { lw: 1.8, dash: [6, 4] });
    if (R) curve(ctx, ax, R.k, hlc('N', pal, pal.ok), { lw: 2.2 });
    else label(ctx, 'computing…', (ax.L + ax.R) / 2, (ax.T + ax.B) / 2, pal.muted, 12, 'center');
    modeMarks(ctx, pal, ax);
    legend(ctx, ax, [['target π^w_1', pal.muted], ['SMC', hlc('N', pal, pal.ok)], ['CFG', pal.ramp[3], [6, 4]]]);
    const L = (v) => Math.log10(v), ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: Math.log2(N6[N6.length - 1]), ymin: L(0.02), ymax: L(2), xt: N6.map(Math.log2), xfmt: (q) => String(2 ** q), yt: [0.03, 0.1, 0.3, 1].map(L), yfmt: (q) => String(+(10 ** q).toPrecision(1)), xl: 'N', title: 'W1 to π^w_1 against the number of particles' });
    hdash(ctx, pal.ramp[3], ax2.L, ax2.R, ax2.Y(L(C.w1)), 1.6, [6, 4]);
    label(ctx, `CFG ${f2(C.w1)}`, ax2.R - 4, ax2.Y(L(C.w1)) - 6, pal.ramp[3], 11.5, 'right');
    hdash(ctx, pal.muted, ax2.L, ax2.R, ax2.Y(L(fl)), 1.2, [2, 3]);
    label(ctx, `${NRUN} exact draws ${f2(fl)}`, ax2.L + 6, ax2.Y(L(fl)) + 15, pal.muted, 11.5);
    const done = N6.filter((n) => res6.has(key(n)));
    polyline(ctx, done.map((n) => ax2.X(Math.log2(n))), done.map((n) => ax2.Y(L(res6.get(key(n)).w1))), hlc('N', pal, pal.ok), 2);
    done.forEach((n) => dot(ctx, ax2.X(Math.log2(n)), ax2.Y(L(res6.get(key(n)).w1)), n === N ? 5.5 : 3.5, n === N ? pal.mark : pal.ok));
    ro.innerHTML = R
      ? `w = <b>${wg}</b>, N = <b>${N}</b> · W1 to π^w_1 <b>${f3(R.w1)}</b> (CFG ${f3(C.w1)}, ${NRUN} exact draws ${f3(fl)}) · share of the shared mode <b>${f3(R.q.share[2])}</b> against ${f3(targetShare(wg))}<br>resampling steps per run <b>${f2(R.resamples)}</b>`
      : `w = <b>${wg}</b>, N = <b>${N}</b> · computing ${NRUN} runs…`;
  }
  return { draw, active: (on) => { r.setActive(on); } };
}

export const SCENES = { bayes: bayesScene, cfg: cfgScene, jensen: jensenScene, interval: intervalScene, autoguidance: autoguidanceScene, smc: smcScene };
