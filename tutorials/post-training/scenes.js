// Live scenes of the §4 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { XMIN, XMAX } from '../lib/mix1d.js';
import {
  MIX1, XG, A0, B0, reward, pData, pStar, densGrid, normalize, klGrid, meanGrid, tiltQuad, fitProxy, overopt, gold, BETAS2,
  LAMS, sdeDemo, TG, KS, T0, makePG, gradEstimate, gradAllK, KTR, BATCH, NAMES, makePairs, makeDPO, control, PXG, w1Samples, sampleData,
} from './sim.js';
import { makeRng } from '../../animations/lib/rng.js';

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
const colStar = (pal) => pal.ramp[3], colModel = (pal) => pal.ok;
// The reward drawn on its own scale in the top part of a density plot: value lo maps to the bottom.
function rewardCurve(ctx, ax, f, lo, hi, color, dash = [6, 4]) {
  const Yr = (v) => ax.B - (ax.B - ax.T) * 0.92 * (v - lo) / (hi - lo);
  clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
  polyline(ctx, Array.from(XG, ax.X), Array.from(XG, (x) => Yr(f(x))), color, 1.8, dash);
  ctx.restore();
  return Yr;
}

// ---------- Step 1: the tilted target ----------
const B1 = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
const BS1 = Float64Array.from({ length: 61 }, (_, i) => 10 ** (-1.7 + 3 * i / 60));
const sweep1 = memo((a) => Array.from(BS1, (beta) => { const p = normalize(densGrid(tiltQuad(MIX1, a, beta))), er = meanGrid(p, (x) => reward(x, a)), kl = klGrid(p, pData); return { er, kl, obj: er - beta * kl }; }));
function tiltScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let beta = 0.5, a = A0;
  viz.append(c.cv, controls(listSlider('beta', B1, beta, (v) => { beta = v; draw(); }), slider('a', { min: -2, max: 2.5, step: 0.1, value: a, fmt: (v) => v.toFixed(1) }, (v) => { a = v; draw(); })), ro,
    note('Exact: for the quadratic reward the tilted target is a mixture of three Gaussians with closed-form weights, means and widths. Expectations and the KL are integrals on a grid of 361 points.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const ps = normalize(densGrid(tiltQuad(MIX1, a, beta)));
    const ax = densAxes(ctx, pal, P1, dmaxL(ps, pData), `Base model and tilted target at β = ${beta}`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.6, fill: 0.14 });
    curve(ctx, ax, ps, hlc('ps', pal, colStar(pal)), { lw: 2.2, fill: 0.2 });
    rewardCurve(ctx, ax, (x) => reward(x, a), -6, 0, hlc(['r', 'a'], pal, pal.ramp[1]));
    vdash(ctx, hlc('a', pal, pal.ramp[1]), ax.X(a), ax.T, ax.B, 1);
    legend(ctx, ax, [['p_{data}', hlc('pd', pal, pal.muted)], ['p*', hlc('ps', pal, colStar(pal))], ['r(x)', hlc('r', pal, pal.ramp[1]), [6, 4]]], true);
    const S = sweep1(a, a), L = Math.log10;
    const ax2 = axes(ctx, pal, P2, { xmin: L(BS1[0]), xmax: L(BS1[BS1.length - 1]), ymin: -3, ymax: 3, xt: [-1, 0, 1].map((q) => q), xfmt: (q) => String(10 ** q), yt: [-3, -2, -1, 0, 1, 2, 3], xl: 'β (log scale)', title: 'What p* gains and what it costs' });
    hdash(ctx, pal.line, ax2.L, ax2.R, ax2.Y(0), 1, []);
    const Xs = Array.from(BS1, (b) => ax2.X(L(b)));
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    polyline(ctx, Xs, S.map((q) => ax2.Y(q.er)), hlc('r', pal, pal.ramp[1]), 2);
    polyline(ctx, Xs, S.map((q) => ax2.Y(q.kl)), hlc('ps', pal, colStar(pal)), 2);
    polyline(ctx, Xs, S.map((q) => ax2.Y(q.obj)), pal.fg, 1.4, [5, 4]);
    ctx.restore();
    const er = meanGrid(ps, (x) => reward(x, a)), kl = klGrid(ps, pData);
    vdash(ctx, hlc('beta', pal, pal.mark), ax2.X(L(beta)), ax2.T, ax2.B, 1.4);
    [[er, pal.ramp[1]], [kl, colStar(pal)], [er - beta * kl, pal.fg]].forEach(([v, col]) => { if (v > -3) dot(ctx, ax2.X(L(beta)), ax2.Y(v), 4, col); });
    legend(ctx, ax2, [['E_{p*}[r]', hlc('r', pal, pal.ramp[1])], ['KL(p* ‖ p_{data})', hlc('ps', pal, colStar(pal))], ['objective', pal.fg, [5, 4]]]);
    ro.innerHTML = `β = <b>${beta}</b>, a = <b>${a.toFixed(1)}</b> · E<sub>p*</sub>[r] = <b>${f3(er)}</b> (base model ${f3(meanGrid(pData, (x) => reward(x, a)))}) · KL(p* ‖ p<sub>data</sub>) = <b>${f3(kl)}</b> · objective E[r] − β KL = <b>${f3(er - beta * kl)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: proxy rewards and over-optimization ----------
function proxyScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let seed = 1, bi = 24, P = fitProxy(seed), O = overopt(P);
  const fmtB = (i) => (BETAS2[i] >= 1 ? BETAS2[i].toFixed(1) : BETAS2[i].toPrecision(2));
  viz.append(c.cv, controls(slider('beta', { min: 0, max: BETAS2.length - 1, step: 1, value: bi, fmt: fmtB }, (v) => { bi = v; draw(); }),
    button('New labels', () => { seed++; P = fitProxy(seed); O = overopt(P); draw(); })), ro,
    note('Exact given the labels: the 20 labelled points are seeded draws from the base model with label noise 0.1, the proxy is their least-squares line, and every p*_β is the base model tilted by the proxy on a grid.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), cur = O[bi];
    const ax = densAxes(ctx, pal, P1, dmaxL(cur.p, pData), `Tilted by the proxy at β = ${fmtB(bi)}`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.6, fill: 0.14 });
    curve(ctx, ax, cur.p, hlc('ps', pal, colStar(pal)), { lw: 2.2, fill: 0.2 });
    const Yr = rewardCurve(ctx, ax, gold, -0.2, 1.6, hlc('r', pal, pal.ramp[1]));
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    polyline(ctx, [ax.X(XMIN), ax.X(XMAX)], [Yr(P.f(XMIN)), Yr(P.f(XMAX))], hlc('rh', pal, pal.ramp[0]), 2);
    P.xs.forEach((x, i) => dot(ctx, ax.X(x), Yr(P.ys[i]), 3, hlc('rh', pal, pal.ramp[0])));
    ctx.restore();
    legend(ctx, ax, [['p_{data}', hlc('pd', pal, pal.muted)], ['p*_β for the proxy', hlc('ps', pal, colStar(pal))], ['gold reward', hlc('r', pal, pal.ramp[1]), [6, 4]], ['proxy, 20 labels', hlc('rh', pal, pal.ramp[0])]], true);
    const kmax = Math.ceil(Math.sqrt(O[O.length - 1].kl) * 2) / 2;
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: kmax, ymin: 0, ymax: 1, xt: Array.from({ length: kmax * 2 }, (_, i) => i / 2), yt: [0, 0.5, 1], xl: '√KL(p*_β ‖ p_{data})', title: 'Reward of p*_β as β falls' });
    const Xs = O.map((q) => ax2.X(Math.sqrt(q.kl)));
    polyline(ctx, Xs, O.map((q) => ax2.Y(q.proxy)), hlc('rh', pal, pal.ramp[0]), 2);
    polyline(ctx, Xs, O.map((q) => ax2.Y(q.gold)), hlc('r', pal, pal.ramp[1]), 2.2);
    const X = ax2.X(Math.sqrt(cur.kl));
    vdash(ctx, hlc('beta', pal, pal.mark), X, ax2.T, ax2.B, 1.2);
    dot(ctx, X, ax2.Y(cur.proxy), 4.5, pal.ramp[0]); dot(ctx, X, ax2.Y(cur.gold), 4.5, pal.ramp[1]);
    const best = O.reduce((b, q) => (q.gold > b.gold ? q : b), O[0]);
    legend(ctx, ax2, [['proxy reward E[r̂]', hlc('rh', pal, pal.ramp[0])], ['gold reward E[r]', hlc('r', pal, pal.ramp[1])]], true);
    ro.innerHTML = `β = <b>${fmtB(bi)}</b> · √KL = <b>${f2(Math.sqrt(cur.kl))}</b> · proxy reward <b>${f3(cur.proxy)}</b> · gold reward <b>${f3(cur.gold)}</b><br>` +
      `best gold reward <b>${f3(best.gold)}</b> at β = ${best.beta.toPrecision(2)}; base model ${f3(O[0].gold)} · labels set ${seed}`;
  }
  return { draw };
}

// ---------- Step 3: a sampler with a likelihood ----------
const demo3 = memo((lam) => sdeDemo(lam));
function sdeScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.72 }), ro = readout();
  let lam = 1;
  viz.append(c.cv, controls(listSlider('lam', LAMS, lam, (v) => { lam = v; draw(); })), ro,
    note(`Exact fields of the base mixture; Euler–Maruyama with ${KS} steps from t = ${T0} to 1. The histogram has 2000 seeded paths started from N(0, 1); the 40 paths share the start x = 0.15.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3), D = demo3(lam, lam);
    const ax = axes(ctx, pal, P1, { xmin: 0, xmax: 1, ymin: -4, ymax: 4, xt: [0, 0.5, 1], yt: [-4, -2, 0, 2, 4], xl: 't', title: `40 paths from one start, λ = ${lam}` });
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    ctx.globalAlpha = 0.45;
    D.paths.forEach((p, i) => { if (i) polyline(ctx, Array.from(TG, ax.X), Array.from(p.xs, ax.Y), hlc('eps', pal, pal.ramp[0]), 1); });
    ctx.globalAlpha = 1;
    if (lam > 0) polyline(ctx, Array.from(TG, ax.X), Array.from(D.paths[0].xs, ax.Y), hlc('pi', pal, pal.ramp[3]), 2);
    polyline(ctx, Array.from(TG, ax.X), Array.from(D.ode.xs, ax.Y), pal.fg, 2, [6, 4]);
    ctx.restore();
    legend(ctx, ax, [['ODE path', pal.fg, [6, 4]], ['SDE paths', hlc('eps', pal, pal.ramp[0])]], true);
    // End histogram against the base density.
    const bins = 60, lo = XMIN, bw = (XMAX - XMIN) / bins, cnt = new Float64Array(bins);
    D.xs.forEach((x) => { const i = Math.floor((x - lo) / bw); if (i >= 0 && i < bins) cnt[i]++; });
    const dens = cnt.map((q) => q / (D.xs.length * bw)), ym = dmax(dens, pData);
    const ax2 = densAxes(ctx, pal, P2, ym, '2000 end points against p_{data}');
    ctx.fillStyle = hlc('eps', pal, pal.ramp[0]); ctx.globalAlpha = 0.45;
    dens.forEach((d, i) => ctx.fillRect(ax2.X(lo + i * bw) + 0.5, ax2.Y(d), ax2.X(lo + bw) - ax2.X(lo) - 1, ax2.B - ax2.Y(d)));
    ctx.globalAlpha = 1;
    curve(ctx, ax2, pData, hlc('pd', pal, pal.fg), { lw: 1.8 });
    // Per-step log-probabilities of the highlighted path.
    const ax3 = axes(ctx, pal, P3, { xmin: 0, xmax: 1, ymin: -4, ymax: 4, xt: [0, 0.5, 1], yt: [-4, 0, 4], xl: 't', title: 'Log-density of each orange step' });
    hdash(ctx, pal.line, ax3.L, ax3.R, ax3.Y(0), 1, []);
    if (lam > 0) {
      ctx.fillStyle = hlc('pi', pal, pal.ramp[3]);
      D.lp.forEach((v, k) => { const x0 = ax3.X(TG[k]) + 0.5, x1 = ax3.X(TG[k + 1]) - 0.5, y = ax3.Y(Math.max(-4, Math.min(4, v))); ctx.fillRect(x0, Math.min(y, ax3.Y(0)), x1 - x0, Math.abs(y - ax3.Y(0))); });
    } else label(ctx, 'λ = 0: every step is a point mass,', (ax3.L + ax3.R) / 2, (ax3.T + ax3.B) / 2 - 8, pal.muted, 12, 'center'),
      label(ctx, 'so the path has no density', (ax3.L + ax3.R) / 2, (ax3.T + ax3.B) / 2 + 10, pal.muted, 12, 'center');
    ro.innerHTML = `λ = <b>${lam}</b> · W1 of 2000 end points to p<sub>data</sub> <b>${f3(D.w1)}</b> (2000 exact draws: 0.038 on average, 0.02 to 0.06 over 30 seeds) · ` +
      (lam > 0 ? `log-density of the orange path <b>${f2(D.lpSum)}</b>` : 'the paths from one start coincide, so the path has no density to differentiate');
  }
  return { draw };
}

// ---------- Step 4: policy gradient ----------
const NIT4 = 150, G4 = [4, 8, 16, 32], B4 = [0.25, 0.5, 1];
function pgScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.72 }), ro = readout();
  let algo = 'grpo', G = 16, beta = B0, S = null;
  const reset = () => { S = makePG({ algo, G, beta, lr: 0.02 }); };
  reset();
  const r = runner({ step: () => S.step(), done: () => S.it >= NIT4, draw: () => draw(), interval: 30 });
  const bG = button('GRPO', () => { algo = 'grpo'; sync(); }), bD = button('DDPO', () => { algo = 'ddpo'; sync(); });
  const sync = () => { bG.classList.toggle('on', algo === 'grpo'); bD.classList.toggle('on', algo === 'ddpo'); reset(); draw(); r.kick(); };
  bG.classList.add('on');
  viz.append(c.cv, controls(bG, bD, listSlider('G', G4, G, (v) => { G = v; reset(); draw(); r.kick(); }), listSlider('beta', B4, beta, (v) => { beta = v; reset(); draw(); r.kick(); }),
    r.play, r.one, button('Reset', () => { reset(); draw(); r.kick(); })), ro,
    note('Trained in the page from a fixed seed: 4 groups of G rollouts per iteration on the memoryless SDE, Adam with step 0.02. The model is a mixture, so p_θ and both KLs are exact; the path KL is the average log-ratio of the batch.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3), cur = S.cur, ps = pStar(A0, beta);
    const ax = axes(ctx, pal, P1, { xmin: 0, xmax: 1, ymin: -4, ymax: 4, xt: [0, 0.5, 1], yt: [-4, -2, 0, 2, 4], xl: 't', title: S.group ? `One group of ${G} rollouts, coloured by A` : 'Press Play to train' });
    hdash(ctx, hlc('a', pal, pal.ramp[1]), ax.L, ax.R, ax.Y(A0), 1);
    if (S.group) {
      clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
      S.group.forEach((q) => { ctx.globalAlpha = 0.3 + 0.7 * Math.min(1, Math.abs(q.A) / 2); polyline(ctx, Array.from(TG, ax.X), Array.from(q.xs, ax.Y), hlc('A', pal, q.A >= 0 ? pal.ok : pal.bad), 1.3); });
      ctx.globalAlpha = 1; ctx.restore();
    }
    label(ctx, 'a', ax.R - 6, ax.Y(A0) - 5, pal.ramp[1], 11, 'right');
    const ax2 = densAxes(ctx, pal, P2, dmaxL(ps, cur.p, pData), `p_θ after ${S.it} iterations`);
    curve(ctx, ax2, pData, hlc('pd', pal, pal.muted), { lw: 1.4, dash: [2, 3] });
    curve(ctx, ax2, ps, hlc('ps', pal, colStar(pal)), { lw: 0, fill: 0.2 });
    curve(ctx, ax2, cur.p, hlc('pth', pal, colModel(pal)), { lw: 2.2 });
    legend(ctx, ax2, [['p*', hlc('ps', pal, colStar(pal))], ['p_θ', hlc('pth', pal, colModel(pal))], ['p_{data}', hlc('pd', pal, pal.muted), [2, 3]]], true);
    const er0 = meanGrid(ps, reward), kl0 = klGrid(ps, pData);
    const ax3 = axes(ctx, pal, P3, { xmin: 0, xmax: NIT4, ymin: -2.2, ymax: 1.4, xt: [0, 50, 100, 150], yt: [-2, -1, 0, 1], xl: 'iteration', title: 'Training curves' });
    hdash(ctx, pal.ramp[1], ax3.L, ax3.R, ax3.Y(er0), 1, [2, 3]); hdash(ctx, colStar(pal), ax3.L, ax3.R, ax3.Y(kl0), 1, [2, 3]);
    const Xs = S.hist.map((q) => ax3.X(q.it));
    polyline(ctx, Xs, S.hist.map((q) => ax3.Y(q.er)), hlc('r', pal, pal.ramp[1]), 2);
    polyline(ctx, Xs, S.hist.map((q) => ax3.Y(q.klD)), hlc('pd', pal, pal.muted), 2);
    polyline(ctx, Xs, S.hist.map((q) => ax3.Y(Math.min(q.klS, 1.4))), hlc('ps', pal, colStar(pal)), 2);
    legend(ctx, ax3, [['E[r]', hlc('r', pal, pal.ramp[1])], ['KL(p_θ ‖ p_{data})', hlc('pd', pal, pal.muted)], ['KL(p_θ ‖ p*)', hlc('ps', pal, colStar(pal))]], false, true);
    const last = S.hist[S.hist.length - 1], pk = S.hist.slice(-10).filter((q) => q.it > 0);
    const pkm = pk.length ? pk.reduce((s, q) => s + q.pathKL, 0) / pk.length : 0;
    ro.innerHTML = `${algo.toUpperCase()}, G = <b>${G}</b>, β = <b>${beta}</b>, iteration <b>${S.it}</b> · E[r] = <b>${f3(last.er)}</b> (p*: ${f3(er0)}) · KL(p<sub>θ</sub> ‖ p<sub>data</sub>) = <b>${f3(last.klD)}</b> (p*: ${f3(kl0)}) · KL(p<sub>θ</sub> ‖ p*) = <b>${f3(last.klS)}</b>` +
      (pk.length ? `<br>path KL, mean of the last ${pk.length} batches: <b>${f3(pkm)}</b>` : '');
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: pathwise gradients ----------
let true5 = null;
const getTrue5 = () => (true5 ??= gradAllK(4000, makeRng(99)));
const SHORT = ['w₁', 'w₂', 'w₃', 'μ₁', 'μ₂', 'μ₃', 's₁', 's₂', 's₃'], NREP5 = 200, err5 = new Map();
function pathwiseScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let K = 10;
  const missing = () => BATCH.find((n) => !err5.has(n));
  const r = runner({
    step: () => {
      const n = missing(), T = getTrue5(), gt = T[T.length - 1], nrm = Math.hypot(...gt), rng = makeRng(1000 + n);
      const sf = [], pw = KTR.map(() => []);
      for (let i = 0; i < NREP5; i++) {
        const g = gradEstimate('sf', n, rng); sf.push(g.reduce((s, v, j) => s + (v - gt[j]) ** 2, 0));
        gradAllK(n, rng).forEach((q, ki) => pw[ki].push(q.reduce((s, v, j) => s + (v - gt[j]) ** 2, 0)));
      }
      const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v, 0) / a.length) / nrm;
      err5.set(n, { sf: rms(sf), pw: pw.map(rms) });
    },
    done: () => missing() === undefined, draw: () => draw(), interval: 30,
  });
  viz.append(c.cv, controls(listSlider('K', KTR, K, (v) => { K = v; draw(); })), ro,
    note(`Gradient of E[r(x_1)] at the base model, in its nine parameters. The reference gradient uses 4000 seeded paths; each error is the root mean square over ${NREP5} seeded batches, relative to the length of the reference.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), T = getTrue5(), gt = T[T.length - 1], ki = KTR.indexOf(K), gk = T[ki];
    const L = Math.log2, L10 = Math.log10;
    const ax = axes(ctx, pal, P1, { xmin: 0, xmax: L(32), ymin: L10(0.1), ymax: L10(10), xt: BATCH.map(L), xfmt: (q) => String(2 ** q), yt: [0.1, 0.3, 1, 3, 10].map(L10), yfmt: (q) => String(+(10 ** q).toPrecision(1)), xl: 'batch size', title: 'Relative error of the gradient estimate' });
    const done = BATCH.filter((n) => err5.has(n)), Xs = done.map((n) => ax.X(L(n)));
    const lines = [['score function', pal.ramp[1], (e) => e.sf], ['pathwise, all 40 steps', pal.ok, (e) => e.pw[KTR.length - 1]], [`pathwise, last ${K}`, hlc('K', pal, pal.ramp[3]), (e) => e.pw[ki]]];
    lines.forEach(([, col, f]) => { polyline(ctx, Xs, done.map((n) => ax.Y(L10(f(err5.get(n))))), col, 2); done.forEach((n, i) => dot(ctx, Xs[i], ax.Y(L10(f(err5.get(n)))), 3, col)); });
    if (!done.length) label(ctx, 'computing…', (ax.L + ax.R) / 2, (ax.T + ax.B) / 2, pal.muted, 12, 'center');
    legend(ctx, ax, lines.map(([t, col]) => [t, col]), true, true);
    // Bars: reference gradient and the truncated one.
    const gm = Math.max(...gt.map(Math.abs)) * 1.15;
    const ax2 = axes(ctx, pal, P2, { xmin: -0.5, xmax: 8.5, ymin: -gm, ymax: gm, xt: [], yt: [-1, 0, 1].map((q) => q * Math.floor(gm * 10) / 10), yfmt: f2, title: `Gradient by parameter: full against last ${K} steps` });
    label(ctx, 'w: weight logits, μ: means, s: log widths', ax2.L + 6, ax2.T + 14, pal.muted, 11);
    hdash(ctx, pal.line, ax2.L, ax2.R, ax2.Y(0), 1, []);
    const bw = (ax2.X(1) - ax2.X(0)) * 0.36;
    gt.forEach((v, j) => {
      ctx.fillStyle = pal.ok; ctx.fillRect(ax2.X(j) - bw, Math.min(ax2.Y(v), ax2.Y(0)), bw, Math.abs(ax2.Y(v) - ax2.Y(0)));
      ctx.fillStyle = hlc('K', pal, pal.ramp[3]); ctx.fillRect(ax2.X(j), Math.min(ax2.Y(gk[j]), ax2.Y(0)), bw, Math.abs(ax2.Y(gk[j]) - ax2.Y(0)));
      label(ctx, SHORT[j], ax2.X(j), ax2.B + 16, pal.muted, 10.5, 'center');
    });
    legend(ctx, ax2, [['all 40 steps', pal.ok], [`last ${K} steps`, hlc('K', pal, pal.ramp[3])]], false, true);
    const bias = Math.hypot(...gk.map((v, j) => v - gt[j])) / Math.hypot(...gt), e8 = err5.get(8);
    ro.innerHTML = `K = <b>${K}</b> · bias of the truncated gradient <b>${f3(bias)}</b> of the full one` +
      (e8 ? ` · relative error at batch 8: score function <b>${f2(e8.sf)}</b>, pathwise <b>${f2(e8.pw[KTR.length - 1])}</b>, last ${K} steps <b>${f2(e8.pw[ki])}</b>` : '');
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 6: Diffusion-DPO ----------
const N6 = [100, 300, 1000], B6 = [0.25, 0.5, 1], NIT6 = 300;
function dpoScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let n = 300, beta = B0, pairs, E, L;
  const reset = () => { pairs = makePairs(n, 3); E = makeDPO({ pairs, beta, kind: 'exact' }); L = makeDPO({ pairs, beta, kind: 'elbo' }); };
  reset();
  const r = runner({ step: () => { E.step(); L.step(); }, done: () => E.it >= NIT6, draw: () => draw(), interval: 30, perFrame: 2 });
  viz.append(c.cv, controls(listSlider('npairs', N6, n, (v) => { n = v; reset(); draw(); r.kick(); }, String, '\\#\\text{pairs}'), listSlider('beta', B6, beta, (v) => { beta = v; reset(); draw(); r.kick(); }), r.play, r.one), ro,
    note('Trained in the page from fixed seeds, 300 full-batch Adam steps. The exact fit uses the mixture log-density; the ELBO fit uses 6 seeded draws of (t, x_0) per sample. Pairs are seeded draws from the base model labelled by a Bradley–Terry draw on r. The band below the axis shows the first 24 pairs, the winner in green and the loser in red.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), ps = pStar(A0, beta);
    // Density axes with a band below zero for the pair links.
    const ym = dmaxL(ps, E.cur.p, L.cur.p, pData);
    const ax = axes(ctx, pal, P1, { ymin: -0.42 * ym, ymax: ym, yt: [0, ym / 2, ym], yfmt: (q) => (q ? q.toFixed(ym < 1 ? 2 : 1) : '0'), xl: 'x', title: `DPO fits from ${n} pairs, β = ${beta}` });
    hdash(ctx, pal.line, ax.L, ax.R, ax.Y(0), 1, []);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.4, dash: [2, 3] });
    curve(ctx, ax, ps, hlc('ps', pal, colStar(pal)), { lw: 0, fill: 0.2 });
    curve(ctx, ax, E.cur.p, hlc('pth', pal, colModel(pal)), { lw: 2.2 });
    curve(ctx, ax, L.cur.p, pal.ramp[0], { lw: 2, dash: [6, 4] });
    // The first 24 pairs as links between winner and loser.
    pairs.slice(0, 24).forEach(([xw, xl], i) => {
      const y = ax.Y(0) + 7 + i * (ax.B - ax.Y(0) - 12) / 23;
      ctx.globalAlpha = 0.6; polyline(ctx, [ax.X(xw), ax.X(xl)], [y, y], pal.line, 1); ctx.globalAlpha = 1;
      dot(ctx, ax.X(xw), y, 2.2, hlc('xw', pal, pal.ok)); dot(ctx, ax.X(xl), y, 2.2, hlc('xl', pal, pal.bad));
    });
    legend(ctx, ax, [['p*', hlc('ps', pal, colStar(pal))], ['exact DPO', hlc('pth', pal, colModel(pal))], ['ELBO DPO', pal.ramp[0], [6, 4]], ['p_{data}', hlc('pd', pal, pal.muted), [2, 3]]], true);
    const km = niceMax(Math.max(0.3, ...E.hist.map((q) => q.klS), ...L.hist.map((q) => q.klS)));
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: NIT6, ymin: 0, ymax: km, xt: [0, 100, 200, 300], yt: [0, km / 2, km], yfmt: f2, xl: 'iteration', title: 'KL(p_θ ‖ p*)' });
    polyline(ctx, E.hist.map((q) => ax2.X(q.it)), E.hist.map((q) => ax2.Y(q.klS)), hlc('pth', pal, colModel(pal)), 2);
    polyline(ctx, L.hist.map((q) => ax2.X(q.it)), L.hist.map((q) => ax2.Y(q.klS)), pal.ramp[0], 2, [6, 4]);
    legend(ctx, ax2, [['exact DPO', hlc('pth', pal, colModel(pal))], ['ELBO DPO', pal.ramp[0], [6, 4]]]);
    ro.innerHTML = `${n} pairs, β = <b>${beta}</b>, iteration <b>${E.it}</b> · KL(p<sub>θ</sub> ‖ p*): exact <b>${f3(E.cur.klS)}</b>, ELBO <b>${f3(L.cur.klS)}</b> · KL(p<sub>data</sub> ‖ p*) ${f3(klGrid(pData, ps))}`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 7: memoryless noise ----------
const L7 = [0.1, 0.25, 0.5, 0.75, 1], B7 = [0.25, 0.5, 1], A7 = [-0.5, 0.3, 1.2, 2];
const ctl7 = memo((lam, beta, a) => control(lam, a, beta));
function controlScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let lam = 1, beta = B0, a = A0;
  viz.append(c.cv, controls(listSlider('lam', L7, lam, (v) => { lam = v; draw(); }), listSlider('beta', B7, beta, (v) => { beta = v; draw(); }), listSlider('a', A7, a, (v) => { a = v; draw(); })), ro,
    note('Grid solution, not sampled: h_t from the backward equation of the base SDE and the controlled density from its Fokker–Planck equation, both implicit finite differences on 301 points and 500 time steps from t = 0.01, started from the exact marginal there.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), C = ctl7(`${lam}|${beta}|${a}`, lam, beta, a);
    const ax = densAxes(ctx, pal, P1, dmaxL(C.ps, C.q.filter((_, i) => Math.abs(PXG[i]) < 4.5), pData), `End of the controlled SDE at λ = ${lam}`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.4, dash: [2, 3] });
    curve(ctx, ax, C.ps, hlc('ps', pal, colStar(pal)), { lw: 0, fill: 0.22, xs: PXG });
    curve(ctx, ax, C.q, hlc('pth', pal, colModel(pal)), { lw: 2.2, xs: PXG });
    legend(ctx, ax, [['p*', hlc('ps', pal, colStar(pal))], ['controlled', hlc('pth', pal, colModel(pal))], ['p_{data}', hlc('pd', pal, pal.muted), [2, 3]]], true);
    const lh = Array.from(C.h0, Math.log), mean = lh.reduce((s, v, i) => s + v * C.p0[i], 0) / C.p0.reduce((s, v) => s + v, 0);
    const d = lh.map((v) => v - mean), ym = niceMax(Math.max(0.1, ...d.filter((_, i) => Math.abs(PXG[i]) < 3).map(Math.abs)));
    const ax2 = axes(ctx, pal, P2, { xmin: -3.5, xmax: 3.5, ymin: -ym, ymax: ym, xt: [-3, -2, -1, 0, 1, 2, 3], yt: [-ym, 0, ym], yfmt: f2, xl: 'x at t = 0.01', title: 'log h at the start, minus its mean' });
    const pm = Math.max(...C.p0);
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    ctx.beginPath(); PXG.forEach((x, i) => ctx.lineTo(ax2.X(x), ax2.B - 0.3 * (ax2.B - ax2.T) * C.p0[i] / pm));
    ctx.lineTo(ax2.R, ax2.B); ctx.lineTo(ax2.L, ax2.B); ctx.closePath(); ctx.fillStyle = pal.muted; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1; ctx.restore();
    hdash(ctx, pal.line, ax2.L, ax2.R, ax2.Y(0), 1, []);
    curve(ctx, ax2, d, hlc('h', pal, pal.ramp[1]), { lw: 2.2, xs: PXG });
    label(ctx, 'start density', ax2.L + 6, ax2.B - 6, pal.muted, 11);
    ro.innerHTML = `λ = <b>${lam}</b>, β = <b>${beta}</b>, a = <b>${a}</b> · W1 of the controlled end density to p* <b>${f3(C.w1)}</b> · spread of log h over the start <b>${f3(C.sdh)}</b>`;
  }
  return { draw };
}

export const SCENES = { tilt: tiltScene, proxy: proxyScene, sde: sdeScene, pg: pgScene, pathwise: pathwiseScene, dpo: dpoScene, control: controlScene };
