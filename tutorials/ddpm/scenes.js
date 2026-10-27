// Live scenes of the DDPM tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, label, hlc, isHL, controls, slider, button, toggle, readout, note } from '../lib/tutorial.js';
import { makeRng } from '../../animations/lib/rng.js';
import { MIX1, fields as fmFields } from '../lib/mix1d.js';
import {
  abarC, schedule, postCoef, fields, dens, reverseKernel, klToGauss, posteriorX0,
  elbo, ENTROPY, lambdas, PAIRS, runSampler, W1_FLOOR, KS, NP, fmTime,
} from './sim.js';

// ---------- Helpers ----------
const NARROW = 560;
const XLO = -3.6, XHI = 3.6;
const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const XS = lin(XLO, XHI, 361);
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function curve(ctx, pts, color, width = 2, dash = null) {
  ctx.beginPath(); pts.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
  if (dash) ctx.setLineDash(dash);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]);
}
function fillUnder(ctx, pts, base, color, alpha) {
  ctx.beginPath(); ctx.moveTo(pts[0][0], base); pts.forEach(([X, Y]) => ctx.lineTo(X, Y)); ctx.lineTo(pts[pts.length - 1][0], base); ctx.closePath();
  ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.fill(); ctx.globalAlpha = 1;
}
const hline = (ctx, pal, x0, x1, y) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };
const vline = (ctx, color, x, y0, y1, width = 1, dash = null) => { if (dash) ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, y0); ctx.lineTo(Math.round(x) + 0.5, y1); ctx.stroke(); ctx.setLineDash([]); };
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const fmtE = (v) => (v === 0 ? '0' : Math.abs(v) >= 0.01 ? v.toFixed(3) : v.toExponential(1).replace('e', '·10^').replace(/\^(-?\d+)/, '<sup>$1</sup>'));
// Number of steps on a log slider: T = round(10^x).
const kSlider = (value, min, on) => slider('K', { min: Math.log10(min), max: 3, step: 0.01, value: Math.log10(value), map: (x) => Math.round(10 ** x), fmt: (v) => String(v) }, on);
const levelSlider = (value, on) => slider('k', { min: 0.01, max: 1, step: 0.005, value, tex: 't/T' }, on);
const levelOf = (frac, K) => Math.min(K, Math.max(1, Math.round(frac * K)));
// Log axis ticks 10^a .. 10^b, with superscript digits because label() only knows subscripts.
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function logTicks(ctx, pal, Y, a, b, x, align = 'right') {
  for (let e = a; e <= b; e++) label(ctx, e === 0 ? '1' : e === 1 ? '10' : `10${String(e).replace(/./g, (ch) => SUP[ch])}`, x, Y(e) + 4, pal.muted, 10.5, align);
}

// ---------- Step 1: the forward chain ----------
// Heat map of p_t over (level, x). The marginal depends on the level only through abar, so the
// continuous schedule abar(t/T) draws it for every T.
const heatCache = new Map();
function heatMap(w, h) {
  const key = `${w}x${h}|${live.theme}`;
  if (heatCache.has(key)) return heatCache.get(key);
  if (heatCache.size > 12) heatCache.clear();
  const res = 2, W = Math.max(1, Math.ceil(w / res)), H = Math.max(1, Math.ceil(h / res));
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H);
  const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.55 : 0.38;
  for (let i = 0; i < W; i++) {
    const ab = abarC(i / (W - 1)), col = new Float64Array(H);
    let mx = 0;
    for (let j = 0; j < H; j++) { col[j] = dens(XHI - (XHI - XLO) * j / (H - 1), ab); mx = Math.max(mx, col[j]); }
    for (let j = 0; j < H; j++) {
      const a = Math.pow(col[j] / mx, 0.7), o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * a * amax);
    }
  }
  g.putImageData(im, 0, 0);
  heatCache.set(key, off);
  return off;
}
const NPATH = 24;
const pathCache = new Map();
function chainPaths(K) {
  if (pathCache.has(K)) return pathCache.get(K);
  const S = schedule(K), r = makeRng(17), out = [];
  for (let i = 0; i < NPATH; i++) {
    const p = new Float64Array(K + 1); p[0] = PAIRS.x0[i];
    for (let k = 1; k <= K; k++) p[k] = Math.sqrt(S.alpha[k]) * p[k - 1] + Math.sqrt(S.beta[k]) * r.gauss();
    out.push(p);
  }
  if (pathCache.size > 30) pathCache.clear();
  pathCache.set(K, out);
  return out;
}
function forward(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.62 : 16 / 9), { maxH: 0.7 }), ro = readout();
  let K = 50, frac = 0.3;
  viz.append(c.cv, controls(kSlider(K, 5, (v) => { K = v; draw(); }), levelSlider(frac, (v) => { frac = v; draw(); })), ro,
    note(`The heat map is the exact density p<sub>t</sub> of each level, normalized per column. The ${NPATH} paths run the chain with T Gaussian steps from data points drawn from the mixture, with a fixed seed.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), narrow = w < NARROW;
    const S = schedule(K), k = levelOf(frac, K), ab = S.abar[k];
    // Main panel and slice panel, side by side or stacked.
    const L = 40, T = 26, mainW = narrow ? w - L - 12 : (w - L) * 0.72 - 16, mainH = narrow ? (h - T - 30) * 0.64 : h - T - 34;
    const sx = narrow ? L : L + mainW + 22, sy = narrow ? T + mainH + 30 : T, sw = narrow ? mainW : w - sx - 12, sh = narrow ? h - sy - 12 : mainH;
    const X = (tau) => L + mainW * tau, Y = (x) => T + mainH * (XHI - x) / (XHI - XLO);
    ctx.globalAlpha = isHL('pk') ? 1 : 0.9; ctx.drawImage(heatMap(Math.round(mainW), Math.round(mainH)), L, T, mainW, mainH); ctx.globalAlpha = 1;
    frame(ctx, pal, L, T, mainW, mainH);
    clip(ctx, L, T, mainW, mainH);
    chainPaths(K).forEach((p, i) => {
      const pts = Array.from(p, (x, j) => [X(j / K), Y(x)]);
      curve(ctx, pts, hlc(['xk', 'q'], pal, pal.fg), isHL('xk', 'q') ? 1.6 : 1, null);
      ctx.globalAlpha = 1;
      dot(ctx, X(k / K), Y(p[k]), 3, hlc('xk', pal, pal.ramp[3]));
      if (i === 0) dot(ctx, X(0), Y(p[0]), 3.5, hlc('x0', pal, pal.ramp[0]));
    });
    ctx.restore();
    vline(ctx, hlc('k', pal, pal.acc), X(k / K), T, T + mainH, 2);
    label(ctx, `level ${k} of ${K}`, Math.min(X(k / K) + 6, L + mainW - 90), T + 16, hlc('k', pal, pal.acc), 12.5);
    [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 6, Y(v) + 4, pal.muted, 11, 'right'));
    label(ctx, 't = 0, data', L, T + mainH + 18, hlc('x0', pal, pal.muted), 12);
    label(ctx, 't = T, noise', L + mainW, T + mainH + 18, pal.muted, 12, 'right');
    label(ctx, 'x_t', L - 6, T - 10, hlc('xk', pal, pal.muted), 12.5, 'right');
    // Slice: p_t against N(0, 1), with the path positions as ticks.
    frame(ctx, pal, sx, sy, sw, sh);
    const pmax = 0.9, sliceP = (x) => dens(x, ab);
    if (narrow) {
      const Xs = (x) => sx + sw * (x - XLO) / (XHI - XLO), Ys = (p) => sy + sh - 6 - (sh - 26) * Math.min(1.1, p / pmax);
      const pts = XS.map((x) => [Xs(x), Ys(sliceP(x))]);
      fillUnder(ctx, pts, sy + sh - 6, hlc('pk', pal, pal.acc2), 0.25); curve(ctx, pts, hlc('pk', pal, pal.acc2), 2);
      curve(ctx, XS.map((x) => [Xs(x), Ys(Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI))]), pal.muted, 1.4, [5, 4]);
      label(ctx, `p_t at t = ${k}; dashed: N(0, 1)`, sx + 8, sy + 16, pal.muted, 12);
    } else {
      const Ys = (x) => sy + sh * (XHI - x) / (XHI - XLO), Xs = (p) => sx + 6 + (sw - 16) * Math.min(1.1, p / pmax);
      const pts = XS.map((x) => [Xs(sliceP(x)), Ys(x)]);
      ctx.beginPath(); ctx.moveTo(sx + 6, Ys(XS[0])); pts.forEach(([a, b]) => ctx.lineTo(a, b)); ctx.lineTo(sx + 6, Ys(XS[XS.length - 1])); ctx.closePath();
      ctx.fillStyle = hlc('pk', pal, pal.acc2); ctx.globalAlpha = 0.25; ctx.fill(); ctx.globalAlpha = 1;
      curve(ctx, pts, hlc('pk', pal, pal.acc2), 2);
      curve(ctx, XS.map((x) => [Xs(Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI)), Ys(x)]), pal.muted, 1.4, [5, 4]);
      label(ctx, `p_t at t = ${k}`, sx + 8, sy - 8, hlc('pk', pal, pal.acc2), 12.5);
      label(ctx, 'dashed: N(0, 1)', sx + sw, sy + sh + 18, pal.muted, 12, 'right');
    }
    ro.innerHTML = `T = <b>${K}</b> · t = <b>${k}</b> · β<sub>t</sub> = <b>${fmtE(S.beta[k])}</b> · ᾱ<sub>t</sub> = <b>${fmtE(ab)}</b> · √ᾱ<sub>t</sub> = <b>${Math.sqrt(ab).toFixed(3)}</b> · √(1 − ᾱ<sub>t</sub>) = <b>${Math.sqrt(1 - ab).toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: one reverse step ----------
function kernel(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.8 : 16 / 9), { maxH: 0.7 }), ro = readout();
  let K = 5, frac = 0.4, xk = 0.4;
  viz.append(c.cv, controls(kSlider(K, 2, (v) => { K = v; draw(); }), levelSlider(frac, (v) => { frac = v; draw(); }),
    slider('xk', { min: -3, max: 3, step: 0.01, value: xk }, (v) => { xk = v; draw(); })), ro,
    note('Exact. For the mixture data the reverse kernel is a mixture of three Gaussians. The thin curves use nine data points at evenly spaced quantiles of q(x<sub>0</sub> | x<sub>t</sub>), each scaled by 1/9, so they add up to roughly the filled curve.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const S = schedule(K), k = levelOf(frac, K), ker = reverseKernel(S, k, xk), bt = S.btil[k];
    const { c0, ck } = postCoef(S, k);
    const L = 40, R = w - 14;
    // Upper strip: p_t with the chosen point.
    const t1 = 22, b1 = t1 + Math.max(50, h * 0.2);
    const Xa = (x) => L + (R - L) * (x - XLO) / (XHI - XLO);
    const pk = XS.map((x) => dens(x, S.abar[k])), pm = Math.max(...pk);
    const pts1 = XS.map((x, i) => [Xa(x), b1 - (b1 - t1 - 4) * pk[i] / pm]);
    fillUnder(ctx, pts1, b1, hlc('pk', pal, pal.acc2), 0.2); curve(ctx, pts1, hlc('pk', pal, pal.acc2), 1.6);
    hline(ctx, pal, L, R, b1);
    vline(ctx, hlc('xk', pal, pal.ramp[3]), Xa(xk), t1, b1 + 4, 2.4);
    label(ctx, `p_t at level t = ${k} of ${K}, and the point x_t`, L, t1 - 6, pal.muted, 12.5);
    // Lower panel: kernel, posteriors and the Gaussian model on a window around the kernel.
    const narrow = w < NARROW, t2 = b1 + (narrow ? 62 : 44), b2 = h - 26;
    const sd = Math.sqrt(Math.max(ker.var, bt)), lo = ker.mean - 4 * sd, hi = ker.mean + 4 * sd;
    const X = (y) => L + (R - L) * (y - lo) / (hi - lo), ys = lin(lo, hi, 400);
    const kp = ys.map(ker.pdf), gp = ys.map((y) => Math.exp(-((y - ker.mean) ** 2) / (2 * bt)) / Math.sqrt(2 * Math.PI * bt));
    const ymax = Math.max(...kp, ...gp) * 1.08, Y = (p) => b2 - (b2 - t2) * p / ymax;
    hline(ctx, pal, L, R, b2);
    const n = 9, x0s = posteriorX0(xk, S.abar[k], n);
    x0s.forEach((x0) => {
      const m = c0 * x0 + ck * xk;
      curve(ctx, ys.map((y) => [X(y), Y(Math.exp(-((y - m) ** 2) / (2 * bt)) / Math.sqrt(2 * Math.PI * bt) / n)]), hlc(['qpost', 'mut', 'btil'], pal, pal.ramp[0]), isHL('qpost', 'mut', 'btil') ? 1.8 : 1.1);
    });
    const kpts = ys.map((y, i) => [X(y), Y(kp[i])]);
    fillUnder(ctx, kpts, b2, hlc('qrev', pal, pal.acc), isHL('qrev') ? 0.4 : 0.2); curve(ctx, kpts, hlc('qrev', pal, pal.acc), 2.2);
    curve(ctx, ys.map((y, i) => [X(y), Y(gp[i])]), hlc(['ptheta', 'mutheta'], pal, pal.fg), 2, [7, 5]);
    vline(ctx, pal.muted, X(ker.mean), t2, b2, 1, [2, 3]);
    // Ticks of the window.
    [lo + sd, ker.mean, hi - sd].forEach((v) => { vline(ctx, pal.line, X(v), b2, b2 + 4); label(ctx, v.toFixed(Math.abs(hi - lo) < 0.2 ? 3 : 2), X(v), b2 + 17, pal.muted, 11, 'center'); });
    label(ctx, 'x_{t−1}', R, b2 + 17, hlc('xkm', pal, pal.muted), 12.5, 'right');
    label(ctx, 'filled: exact q(x_{t−1} | x_t)', L, t2 - 22, hlc('qrev', pal, pal.acc), 12.5);
    label(ctx, `dashed: Gaussian with variance β̃_t`, narrow ? L : R, narrow ? t2 - 40 : t2 - 22, hlc(['ptheta', 'btil'], pal, pal.fg), 12.5, narrow ? 'left' : 'right');
    label(ctx, 'thin: q(x_{t−1} | x_t, x_0) for nine data points, ÷ 9', L, t2 - 5, hlc('qpost', pal, pal.ramp[0]), 12);
    // Dashed connector from the strip to the window: the window is zoomed.
    ctx.globalAlpha = 0.5; vline(ctx, pal.muted, Xa(xk), b1 + 4, b1 + 14, 1, [2, 3]); ctx.globalAlpha = 1;
    const kl = klToGauss(ker, ker.mean, bt);
    ro.innerHTML = `T = <b>${K}</b> · t = <b>${k}</b> · β<sub>t</sub> = <b>${fmtE(S.beta[k])}</b> · β̃<sub>t</sub> = <b>${fmtE(bt)}</b> · variance of the exact kernel <b>${fmtE(ker.var)}</b><br>` +
      `mean of the exact kernel <b>${ker.mean.toFixed(4)}</b> = μ̃<sub>t</sub>(x<sub>t</sub>, E[x<sub>0</sub> | x<sub>t</sub>]) · KL(exact ‖ dashed) = <b>${fmtE(kl)}</b> nats`;
  }
  return { draw };
}

// ---------- Step 3: the bound ----------
function elboScene(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.85 : 16 / 9), { maxH: 0.7 }), ro = readout();
  let K = 20, vch = 'tilde';
  const bT = button('\\(\\sigma_t^2 = \\tilde\\beta_t\\)', () => setV('tilde'), 'on'), bB = button('\\(\\sigma_t^2 = \\beta_t\\)', () => setV('beta'));
  const setV = (v) => { vch = v; bT.classList.toggle('on', v === 'tilde'); bB.classList.toggle('on', v === 'beta'); draw(); };
  viz.append(c.cv, controls(kSlider(K, 2, (v) => { K = v; draw(); }), bT, bB), ro,
    note('Exact, by quadrature. The model mean is the best possible one, E[μ̃<sub>t</sub> | x<sub>t</sub>], so the bound shows what the Gaussian steps themselves cost. The decoder is a Gaussian with variance β<sub>1</sub>.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), narrow = w < NARROW;
    const e = elbo(K, vch);
    const L = 50, T = 30, lw = narrow ? w - L - 14 : (w - L) * 0.58 - 20, lh = narrow ? (h - T) * 0.5 - 30 : h - T - 36;
    // Left: T L_{t-1} over the levels, log scale.
    const X = (tau) => L + lw * tau, ya = -3, yb = 3, Y = (lg) => T + lh * (yb - Math.max(ya, Math.min(yb, lg))) / (yb - ya);
    frame(ctx, pal, L, T, lw, lh);
    for (let g = ya; g <= yb; g++) { ctx.globalAlpha = 0.5; hline(ctx, pal, L, L + lw, Y(g)); ctx.globalAlpha = 1; }
    logTicks(ctx, pal, Y, ya, yb, L - 6);
    clip(ctx, L, T, lw, lh);
    ctx.fillStyle = hlc(['Lk', 'L'], pal, pal.acc); ctx.globalAlpha = isHL('Lk', 'L') ? 0.55 : 0.3;
    for (let k = 2; k <= K; k++) { const v = e.terms[k] * K; if (v > 0) { const x0 = X((k - 1) / K), x1 = X(k / K); ctx.fillRect(x0, Y(Math.log10(v)), Math.max(1, x1 - x0 - (K < 60 ? 1 : 0)), T + lh - Y(Math.log10(v))); } }
    ctx.globalAlpha = 1;
    ctx.restore();
    label(ctx, 'T · L_{t−1} over the levels (log scale)', L, T - 10, hlc('Lk', pal, pal.acc), 12.5);
    label(ctx, 't/T = 0', L, T + lh + 17, pal.muted, 11.5); label(ctx, '1', L + lw, T + lh + 17, pal.muted, 11.5, 'right');
    // Right: the total against T for both choices.
    const rx = narrow ? L : L + lw + 56, ry = narrow ? T + lh + 56 : T, rw = narrow ? lw : w - rx - 14, rh = narrow ? h - ry - 30 : lh;
    const Xk = (K_) => rx + rw * Math.log10(K_ / 2) / Math.log10(500), vlo = 1.4, vhi = 3.4, Yv = (v) => ry + rh * (vhi - Math.max(vlo, Math.min(vhi, v))) / (vhi - vlo);
    frame(ctx, pal, rx, ry, rw, rh);
    [1.5, 2, 2.5, 3].forEach((v) => label(ctx, v.toFixed(1), rx - 6, Yv(v) + 4, pal.muted, 10.5, 'right'));
    [2, 10, 100, 1000].forEach((K_) => label(ctx, String(K_), Xk(K_), ry + rh + 16, pal.muted, 10.5, 'center'));
    ctx.setLineDash([5, 4]); hline(ctx, { line: hlc('Ent', pal, pal.ok) }, rx, rx + rw, Yv(ENTROPY)); ctx.setLineDash([]);
    label(ctx, `H(p_{data}) = ${ENTROPY.toFixed(2)}`, rx + rw - 6, Yv(ENTROPY) + 16, hlc('Ent', pal, pal.ok), 11.5, 'right');
    [['tilde', pal.acc], ['beta', pal.ramp[3]]].forEach(([v, col]) => {
      const pts = KS.map((K_) => [Xk(K_), Yv(elbo(K_, v).total)]);
      curve(ctx, pts, col, v === vch ? 2.4 : 1.2, v === vch ? null : [4, 3]);
      pts.forEach(([a, b]) => dot(ctx, a, b, v === vch ? 3 : 2, col));
    });
    ring(ctx, Xk(K), Yv(e.total), 6, pal.fg, 2);
    label(ctx, 'total L against T', rx, ry - 10, hlc('L', pal, pal.muted), 12.5);
    label(ctx, 'β̃_t', Xk(2) + 6, Yv(elbo(2, 'tilde').total) - 8, pal.acc, 12); label(ctx, 'β_t', Xk(3) + 2, Yv(elbo(3, 'beta').total) + 18, pal.ramp[3], 12);
    ro.innerHTML = `T = <b>${K}</b> · L<sub>T</sub> = <b>${e.LK.toFixed(4)}</b> · Σ L<sub>t−1</sub> = <b>${e.sum.toFixed(3)}</b> · L<sub>0</sub> = <b>${e.L0.toFixed(3)}</b><br>` +
      `total L = <b>${e.total.toFixed(3)}</b> nats · H(p<sub>data</sub>) = <b>${ENTROPY.toFixed(3)}</b> · gap <b>${(e.total - ENTROPY).toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: weights of the noise loss ----------
function eps(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.75 : 16 / 8), { maxH: 0.6 }), ro = readout();
  const K = 1000, lt = lambdas(K, 'tilde'), lb = lambdas(K, 'beta');
  const mean = (l) => { let s = 0; for (let k = 2; k <= K; k++) s += l[k]; return s / (K - 1); };
  const mt = mean(lt), mb = mean(lb);
  let frac = 0.05;
  viz.append(c.cv, controls(levelSlider(frac, (v) => { frac = v; draw(); })), ro,
    note('T = 1000 with the schedule of step 1. Each weight is divided by its average over the levels, so 1 means the same emphasis as L<sub>simple</sub>.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const L = 50, T = 30, R = w - 14, B = h - 30, ya = -1, yb = 2;
    const X = (tau) => L + (R - L) * tau, Y = (lg) => T + (B - T) * (yb - Math.max(ya, Math.min(yb, lg))) / (yb - ya);
    frame(ctx, pal, L, T, R - L, B - T);
    for (let g = ya; g <= yb; g++) { ctx.globalAlpha = 0.5; hline(ctx, pal, L, R, Y(g)); ctx.globalAlpha = 1; }
    logTicks(ctx, pal, Y, ya, yb, L - 6);
    clip(ctx, L, T, R - L, B - T);
    const pts = (l, m) => Array.from({ length: K - 1 }, (_, i) => [X((i + 2) / K), Y(Math.log10(l[i + 2] / m))]);
    curve(ctx, pts(lt, mt), hlc(['lam', 'btil'], pal, pal.acc), 2.4);
    curve(ctx, pts(lb, mb), hlc(['lam', 'beta'], pal, pal.ramp[3]), 2.4);
    curve(ctx, [[L, Y(0)], [R, Y(0)]], hlc('Ls', pal, pal.fg), 2, [7, 5]);
    ctx.restore();
    const k = levelOf(frac, K), kk = Math.max(2, k);
    vline(ctx, hlc('k', pal, pal.muted), X(kk / K), T, B, 1.2, [3, 3]);
    dot(ctx, X(kk / K), Y(Math.log10(lt[kk] / mt)), 4, pal.acc); dot(ctx, X(kk / K), Y(Math.log10(lb[kk] / mb)), 4, pal.ramp[3]);
    label(ctx, 'λ_t / average, σ_t² = β̃_t', R - 8, T + 18, hlc('lam', pal, pal.acc), 12.5, 'right');
    label(ctx, 'λ_t / average, σ_t² = β_t', R - 8, T + 36, hlc('lam', pal, pal.ramp[3]), 12.5, 'right');
    label(ctx, 'dashed: L_{simple}, weight 1 at every level', R - 8, Y(0) - 8, hlc('Ls', pal, pal.fg), 12.5, 'right');
    label(ctx, 't/T = 0, little noise', L, B + 18, pal.muted, 11.5); label(ctx, '1, pure noise', R, B + 18, pal.muted, 11.5, 'right');
    ro.innerHTML = `t = <b>${kk}</b> of ${K} · ᾱ<sub>t</sub> = <b>${fmtE(schedule(K).abar[kk])}</b> · relative ELBO weight: <b>${(lt[kk] / mt).toFixed(2)}</b> with β̃<sub>t</sub>, <b>${(lb[kk] / mb).toFixed(2)}</b> with β<sub>t</sub>, against <b>1</b> in L<sub>simple</sub>`;
  }
  return { draw };
}

// ---------- Step 5: the noise predictor is a score ----------
function score(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.9 : 16 / 9), { maxH: 0.7 }), ro = readout();
  let frac = 0.35, showScore = false;
  viz.append(c.cv, controls(levelSlider(frac, (v) => { frac = v; draw(); }),
    toggle('Show \\(-\\sqrt{1-\\bar\\alpha_t}\\,\\nabla\\log p_t\\)', showScore, (on) => { showScore = on; draw(); })), ro,
    note(`${PAIRS.n} pairs (x<sub>0</sub>, ε) of the mixture, drawn once; the level uses the schedule of step 1 with T = 1000. The circles are bin means of the labels; bins with fewer than 15 dots are left out. Both curves are exact.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const K = 1000, k = levelOf(frac, K), ab = schedule(K).abar[k], a = Math.sqrt(ab), s = Math.sqrt(1 - ab);
    const L = 40, T = 30, R = w - 14, B = h - 34, LO = -4, HI = 4, YM = 3.4;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO), Y = (y) => (T + B) / 2 - (B - T) / 2 * y / YM;
    frame(ctx, pal, L, T, R - L, B - T); hline(ctx, pal, L, R, Y(0));
    [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 6, Y(v) + 4, pal.muted, 11, 'right'));
    const NB = 48, bw = (HI - LO) / NB, sum = new Float64Array(NB), cnt = new Float64Array(NB);
    let lOpt = 0, l0 = 0;
    ctx.fillStyle = isHL('eps', 'xk') ? pal.mark : pal.fg;
    ctx.globalAlpha = isHL('eps', 'xk') ? 0.45 : 0.22;
    for (let i = 0; i < PAIRS.n; i++) {
      const x = a * PAIRS.x0[i] + s * PAIRS.eps[i], y = PAIRS.eps[i];
      const e = fields(x, ab).eps; lOpt += (y - e) ** 2 / PAIRS.n; l0 += y * y / PAIRS.n;
      const b = Math.floor((x - LO) / bw); if (b >= 0 && b < NB) { sum[b] += y; cnt[b]++; }
      ctx.fillRect(X(x) - 1.2, Y(y) - 1.2, 2.4, 2.4);
    }
    ctx.globalAlpha = 1;
    clip(ctx, L, T, R - L, B - T);
    const xs = lin(LO, HI, 321);
    curve(ctx, xs.map((x) => [X(x), Y(fields(x, ab).eps)]), hlc(['epsstar', 'epsth'], pal, pal.acc), isHL('epsstar', 'epsth') ? 3.6 : 2.8);
    let md = 0;
    if (showScore) {
      const pts = xs.map((x) => { const f = fields(x, ab), v = -s * f.score; md = Math.max(md, Math.abs(v - f.eps)); return [X(x), Y(v)]; });
      curve(ctx, pts, hlc('score', pal, pal.fg), 2, [7, 6]);
    }
    ctx.restore();
    for (let b = 0; b < NB; b++) if (cnt[b] >= 15) ring(ctx, X(LO + (b + 0.5) * bw), Y(sum[b] / cnt[b]), 4, pal.ramp[3], 1.8);
    label(ctx, 'label ε of each pair (vertical) against its input x_t (horizontal)', L + 4, T - 10, hlc(['eps', 'xk'], pal, pal.muted), 12.5);
    label(ctx, 'solid: ε*(x) = E[ε | x_t = x]', L, B + 20, hlc('epsstar', pal, pal.acc), 12.5);
    if (showScore) label(ctx, 'dashed: −√(1 − ᾱ_t) ∇log p_t(x)', R, B + 20, hlc('score', pal, pal.fg), 12.5, 'right');
    ro.innerHTML = `t = <b>${k}</b> of ${K} · ᾱ<sub>t</sub> = <b>${fmtE(ab)}</b> · √(1 − ᾱ<sub>t</sub>) = <b>${s.toFixed(3)}</b><br>` +
      `L<sub>simple</sub> at this level: <b>${l0.toFixed(3)}</b> for ε<sub>θ</sub> = 0, <b>${lOpt.toFixed(3)}</b> for ε*` +
      (showScore ? ` · largest gap between the two curves <b>${md.toExponential(1)}</b>` : '');
  }
  return { draw };
}

// ---------- Step 6: two samplers ----------
function samplers(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.72 : 16 / 9), { maxH: 0.72 }), ro = readout();
  let ki = KS.indexOf(20), eta = 1, pending = null;
  const bA = button('Ancestral, \\(\\eta = 1\\)', () => setEta(1), 'on'), bD = button('DDIM, \\(\\eta = 0\\)', () => setEta(0));
  const setEta = (v) => { eta = v; bA.classList.toggle('on', v === 1); bD.classList.toggle('on', v === 0); draw(); };
  const kSl = slider('K', { min: 0, max: KS.length - 1, step: 1, value: ki, map: (i) => KS[i], fmt: (v) => String(v) }, (v) => { ki = KS.indexOf(v); draw(); });
  viz.append(c.cv, controls(kSl, bA, bD), ro,
    note(`Both samplers use the exact ε* of the mixture, so the only errors are the step size and, for η = 1, the sampling noise. ${NP} start points at evenly spaced quantiles of N(0, 1), the same for every run; one in 40 is drawn as a path. The curves of the lower panel fill in while the page computes them.`));
  // W1 of every (T, eta) pair, computed one per frame so the page stays responsive.
  const W1 = { 0: new Map(), 1: new Map() };
  const pump = () => {
    pending = null;
    const todo = [1, 0].flatMap((e) => KS.filter((K) => !W1[e].has(K)).map((K) => [e, K]));
    if (!todo.length) return;
    const [e, K] = todo[0]; W1[e].set(K, runSampler(K, e).w1); draw();
    pending = requestAnimationFrame(pump);
  };
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), narrow = w < NARROW;
    const K = KS[ki], run = runSampler(K, eta); W1[eta].set(K, run.w1);
    const col = eta === 1 ? pal.acc : pal.ramp[3];
    const L = 40, T = 26, top = narrow ? h * 0.56 : h * 0.6, pw = narrow ? w - L - 14 : (w - L) * 0.66 - 14, ph = top - T - 22;
    // Paths over the levels, data at the left as in step 1.
    const X = (tau) => L + pw * tau, Y = (x) => T + ph * (XHI - x) / (XHI - XLO);
    frame(ctx, pal, L, T, pw, ph);
    clip(ctx, L, T, pw, ph);
    run.paths.forEach((p) => curve(ctx, Array.from(p, (x, j) => [X(j / K), Y(x)]), hlc(['eta', 'z'], pal, col), 1));
    ctx.restore();
    label(ctx, 't = 0, data', L, T + ph + 16, pal.muted, 11.5); label(ctx, 't = T, noise', L + pw, T + ph + 16, pal.muted, 11.5, 'right');
    label(ctx, `${eta === 1 ? 'ancestral' : 'DDIM'} paths, T = ${K}`, L, T - 9, col, 12.5);
    [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 6, Y(v) + 4, pal.muted, 11, 'right'));
    // Histogram of the samples against the data density, on the same vertical axis.
    const hx = L + pw + 10, hw = narrow ? 0 : w - hx - 14;
    if (!narrow) {
      frame(ctx, pal, hx, T, hw, ph);
      const NB = 60, bw = (XHI - XLO) / NB, cnt = new Float64Array(NB);
      run.x.forEach((x) => { const b = Math.floor((x - XLO) / bw); if (b >= 0 && b < NB) cnt[b]++; });
      const pmax = 0.6, Wd = (p) => hx + 4 + (hw - 10) * Math.min(1, p / pmax);
      ctx.fillStyle = col; ctx.globalAlpha = 0.35;
      for (let b = 0; b < NB; b++) { const p = cnt[b] / (NP * bw); ctx.fillRect(hx + 4, Y(XLO + (b + 1) * bw), Wd(p) - hx - 4, Y(XLO + b * bw) - Y(XLO + (b + 1) * bw) - 0.5); }
      ctx.globalAlpha = 1;
      curve(ctx, XS.map((x) => [Wd(dens(x, 1)), Y(x)]), hlc('x0', pal, pal.fg), 1.8, [5, 4]);
      label(ctx, 'samples; dashed: data', hx, T - 9, pal.muted, 12);
    }
    // W1 against T for both samplers, log-log.
    const by = top + 24, bh = h - by - 26, bx = L + 10, bwid = w - bx - 14;
    const Xk = (K_) => bx + bwid * Math.log10(K_ / 2) / Math.log10(500), ya = -3, yb = 0;
    const Yl = (lg) => by + bh * (yb - Math.max(ya, Math.min(yb, lg))) / (yb - ya), Yw = (v) => Yl(Math.log10(v));
    frame(ctx, pal, bx, by, bwid, bh);
    logTicks(ctx, pal, Yl, ya, yb, bx - 6);
    [2, 10, 100, 1000].forEach((K_) => label(ctx, String(K_), Xk(K_), by + bh + 15, pal.muted, 10.5, 'center'));
    ctx.globalAlpha = 0.8; ctx.setLineDash([2, 3]); hline(ctx, { line: pal.muted }, bx, bx + bwid, Yw(W1_FLOOR)); ctx.setLineDash([]); ctx.globalAlpha = 1;
    label(ctx, narrow ? 'exact draws' : `W1 of ${NP} exact data draws`, bx + 6, Yw(W1_FLOOR) + 14, pal.muted, 11);
    [[1, pal.acc], [0, pal.ramp[3]]].forEach(([e, cc]) => {
      const pts = KS.filter((K_) => W1[e].has(K_)).map((K_) => [Xk(K_), Yw(W1[e].get(K_))]);
      if (pts.length) { curve(ctx, pts, cc, e === eta ? 2.4 : 1.4, e === eta ? null : [4, 3]); pts.forEach(([a, b]) => dot(ctx, a, b, 2.6, cc)); }
    });
    ring(ctx, Xk(K), Yw(run.w1), 6, pal.fg, 2);
    label(ctx, 'W1 to the data against T: ancestral and DDIM', bx, by - 8, pal.muted, 12);
    ro.innerHTML = `T = <b>${K}</b> · ${eta === 1 ? 'ancestral, η = 1' : 'DDIM, η = 0'} · W1 to the data <b>${run.w1.toFixed(4)}</b> · floor of ${NP} exact draws <b>${W1_FLOOR.toFixed(4)}</b>`;
  }
  return { draw, active: (on) => { if (on && !pending) pending = requestAnimationFrame(pump); if (!on && pending) { cancelAnimationFrame(pending); pending = null; } } };
}

// ---------- Step 7: one path, two clocks ----------
function clocks(viz) {
  const c = makeCanvas((w) => (w < NARROW ? 0.85 : 16 / 9), { maxH: 0.72 }), ro = readout();
  let frac = 0.4;
  viz.append(c.cv, controls(levelSlider(frac, (v) => { frac = v; draw(); })), ro,
    note('Exact. The solid curve uses the DDPM formula at level t. The dots use the flow-matching posterior of mix1d at time s and the rescaled input c<sub>t</sub>x, an independent computation.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const K = 1000, k = levelOf(frac, K), ab = schedule(K).abar[k], { t, c: ck } = fmTime(ab);
    const L = 44, R = w - 14, T = 26, B1 = T + (h - T) * 0.38, T2 = B1 + 48, B2 = h - 28;
    // Upper: coefficients and the flow-matching time against the level.
    const X = (tau) => L + (R - L) * tau, Y = (v) => B1 - (B1 - T) * v;
    frame(ctx, pal, L, T, R - L, B1 - T);
    const taus = lin(0, 1, 201);
    curve(ctx, taus.map((tau) => [X(tau), Y(Math.sqrt(abarC(tau)))]), hlc('abar', pal, pal.acc2), 1.6, [6, 4]);
    curve(ctx, taus.map((tau) => [X(tau), Y(Math.sqrt(1 - abarC(tau)))]), pal.muted, 1.6, [2, 3]);
    curve(ctx, taus.map((tau) => [X(tau), Y(fmTime(abarC(tau)).t)]), hlc('srv', pal, pal.acc), 2.6);
    vline(ctx, pal.muted, X(k / K), T, B1, 1, [3, 3]); dot(ctx, X(k / K), Y(t), 5, hlc('srv', pal, pal.acc));
    ['0', '1'].forEach((s, i) => label(ctx, s, L - 6, Y(i) + 4, pal.muted, 11, 'right'));
    label(ctx, 'solid: flow-matching time s', R - 6, T + 16, hlc('srv', pal, pal.acc), 12, 'right');
    label(ctx, 'dashed: √ᾱ_t   dotted: √(1 − ᾱ_t)', R - 6, T + 33, pal.muted, 12, 'right');
    label(ctx, 't/T = 0', L, B1 + 16, pal.muted, 11.5); label(ctx, '1', R, B1 + 16, pal.muted, 11.5, 'right');
    // Lower: the two noise predictors against x.
    const LO = -3.6, HI = 3.6, YM = 2.6, Xx = (x) => L + (R - L) * (x - LO) / (HI - LO), Yx = (y) => (T2 + B2) / 2 - (B2 - T2) / 2 * y / YM;
    frame(ctx, pal, L, T2, R - L, B2 - T2); hline(ctx, pal, L, R, Yx(0));
    [-2, 0, 2].forEach((v) => label(ctx, String(v), L - 6, Yx(v) + 4, pal.muted, 11, 'right'));
    clip(ctx, L, T2, R - L, B2 - T2);
    const xs = lin(LO, HI, 301);
    curve(ctx, xs.map((x) => [Xx(x), Yx(fields(x, ab).eps)]), hlc('epsstar', pal, pal.acc), 2.6);
    let md = 0;
    const fm = (x) => { const y = ck * x, f = fmFields(MIX1, y, t); return (y - t * f.xhat) / (1 - t); };
    xs.forEach((x) => { md = Math.max(md, Math.abs(fm(x) - fields(x, ab).eps)); });
    lin(LO + 0.15, HI - 0.15, 25).forEach((x) => ring(ctx, Xx(x), Yx(fm(x)), 4, hlc(['srv', 'c'], pal, pal.ramp[3]), 1.8));
    ctx.restore();
    label(ctx, 'solid: ε*_{DDPM}(x, t)   circles: ε*_{FM}(c_t x, s)', L, T2 - 10, pal.muted, 12.5);
    label(ctx, 'x', R, B2 + 18, pal.muted, 12, 'right');
    ro.innerHTML = `t = <b>${k}</b> of ${K} · ᾱ<sub>t</sub> = <b>${fmtE(ab)}</b> · s = <b>${t.toFixed(3)}</b> · c<sub>t</sub> = <b>${ck.toFixed(3)}</b> · largest difference between the two predictors <b>${md.toExponential(1)}</b>`;
  }
  return { draw };
}

export const SCENES = { forward, kernel, elbo: elboScene, eps, score, samplers, clocks };
