// Live scenes of the §6 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { XMIN, XMAX, kde } from '../lib/mix1d.js';
import {
  MIX1, MIX2, XG, pData, NS, Z0, w1, w1ToGrid, w1Dens, LSNR, grid, SOLVERS, NFE1, solverW1, W1_EXACT,
  RA, reward, rGrid, tilt, bestOfN, klBoN, NS2, betaForKL, meanR, klGrid,
  p2data, K3, REW3, W3, NFE_EVAL3, LIM3, NG3, pullback3, search3, TRS3, tail2,
  ZG, NFE_ITG, KOC, dflowG, ocflowG, path2,
  posterior4, NP4, K4, sample4, score4,
  K5, RUNS5, runBoN, runSMC, runGuided, NS5, BETAS5,
  SPUR, spur, goldBoNSystematic, RUNS6, goldBoNNoise,
} from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const fw = (v) => (v < 0.01 ? v.toFixed(4) : v.toFixed(3));
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
// Room above the curves for a legend in the top-left corner.
const dmaxL = (...arrs) => niceMax(1.45 * Math.max(...arrs.map((a) => Math.max(...a))));
const densAxes = (ctx, pal, rect, ymax, title, titleColor) => axes(ctx, pal, rect, { ymin: 0, ymax, yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(ymax < 1 ? 2 : 1) : '0'), xl: 'x', title, titleColor });
const memo = (f) => { const m = new Map(); return (k, ...a) => { if (!m.has(k)) m.set(k, f(...a)); return m.get(k); }; };
// The reward drawn on a density plot: scaled so that its maximum 1 sits at 85% of the axis.
const rewardCurve = (ctx, ax, pal, ymax) => curve(ctx, ax, Float64Array.from(rGrid, (r) => 0.85 * ymax * r), hlc('r', pal, pal.muted), { lw: 1.4, dash: [5, 4] });
// Square axes for the plane, [-lim, lim]².
function sqAxes(ctx, pal, [x0, y0, w, h], title, lim) {
  const s = Math.min(w - 54, h - 44), L = x0 + 44 + (w - 54 - s) / 2, T = y0 + 30;
  const X = (v) => L + s * (v + lim) / (2 * lim), Y = (v) => T + s * (lim - v) / (2 * lim);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, s, s);
  label(ctx, title, L, T - 12, pal.fg, 12.5);
  [-3, 0, 3].forEach((q) => { label(ctx, String(q), X(q), T + s + 14, pal.muted, 11, 'center'); label(ctx, String(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'); });
  return { X, Y, L, T, R: L + s, B: T + s, s };
}
function dataBlobs(ctx, ax, pal, color = pal.muted) {
  MIX2.means.forEach(([mx, my]) => [2, 1].forEach((k) => { ctx.beginPath(); ctx.arc(ax.X(mx), ax.Y(my), (ax.X(k * MIX2.std) - ax.X(0)), 0, 7); ctx.fillStyle = color; ctx.globalAlpha = 0.12; ctx.fill(); ctx.globalAlpha = 1; }));
}
function circle(ctx, ax, r, color, dash = [4, 4]) { ctx.beginPath(); ctx.arc(ax.X(0), ax.Y(0), ax.X(r) - ax.X(0), 0, 7); ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]); }
const log2 = Math.log2, log10 = Math.log10;

// ---------- Step 1: better solvers first ----------
const SOLV = [['euler', 'Euler'], ['heun', 'Heun'], ['dpm', 'DPM-Solver++(2M)']];
const solvCol = (pal, i) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][i];
const solvSamples = memo((name, kind, nfe) => { const [f, per] = SOLVERS[name], ts = grid(nfe / per, kind); return Z0.map((z) => f(z, ts)); });
function solversScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let kind = 'uniform', nfe = 8;
  const bU = button('Uniform grid', () => { kind = 'uniform'; sync(); }), bL = button('Uniform in log-SNR', () => { kind = 'logsnr'; sync(); });
  const sync = () => { bU.classList.toggle('on', kind === 'uniform'); bL.classList.toggle('on', kind === 'logsnr'); draw(); };
  bU.classList.add('on');
  viz.append(c.cv, controls(bU, bL, listSlider('nfe', NFE1, nfe, (v) => { nfe = v; draw(); })), ro,
    note(`Exact field. Each solver maps ${NS} stratified quantiles of the noise, and W1 to p<sub>data</sub> is computed from the sorted end points. The floor is the W1 of ${NS} stratified quantiles of p<sub>data</sub> itself (${W1_EXACT.toFixed(4)}). Heun spends two evaluations per step, so at equal NFE it takes half as many steps. The log-SNR grid is uniform in λ = log(t/(1 − t)) on [−${LSNR}, ${LSNR}], with its end points moved to t = 0 and t = 1.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const ax = axes(ctx, pal, P1, { xmin: 0.8, xmax: 6.2, ymin: -3.3, ymax: 0.3, xt: [1, 2, 3, 4, 5, 6], xfmt: (q) => String(2 ** q), yt: [-3, -2, -1, 0], yfmt: (q) => (q === 0 ? '1' : `1e${q}`), xl: 'NFE (log scale)', title: 'W1 to p_{data} against NFE' });
    hdash(ctx, pal.muted, ax.L, ax.R, ax.Y(log10(W1_EXACT)), 1, [2, 3]);
    label(ctx, 'floor of the sample set', ax.L + 6, ax.Y(log10(W1_EXACT)) - 5, pal.muted, 11);
    vdash(ctx, hlc('nfe', pal, pal.line), ax.X(log2(nfe)), ax.T, ax.B);
    SOLV.forEach(([name], i) => {
      const W = solverW1(name, kind), col = solvCol(pal, i);
      polyline(ctx, NFE1.map((n) => ax.X(log2(n))), W.map((v) => ax.Y(log10(v))), col, 2);
      NFE1.forEach((n, j) => dot(ctx, ax.X(log2(n)), ax.Y(log10(W[j])), n === nfe ? 4.5 : 2.8, col));
    });
    legend(ctx, ax, SOLV.map(([, n], i) => [n, solvCol(pal, i)]), false, false);
    // Right: the end points of each solver at this NFE, and its time grid.
    const [x0, y0, pw, ph] = P2, rows = 3, strip = 18 * rows + 22, D = [x0, y0, pw, ph - strip];
    const dens = SOLV.map(([name]) => kde(solvSamples(name + kind + nfe, name, kind, nfe), 0.07));
    const ax2 = densAxes(ctx, pal, D, dmaxL(pData, ...dens), `End points at NFE = ${nfe}`);
    curve(ctx, ax2, pData, hlc('pd', pal, pal.muted), { lw: 1.4, fill: 0.14 });
    dens.forEach((d, i) => curve(ctx, ax2, d, solvCol(pal, i), { lw: 1.8 }));
    legend(ctx, ax2, [['p_{data}', hlc('pd', pal, pal.muted)]], true);
    const L = ax2.L, R = ax2.R, top = y0 + ph - strip + 18;
    SOLV.forEach(([name], i) => {
      const per = SOLVERS[name][1], ts = grid(nfe / per, kind), y = top + 18 * i;
      hdash(ctx, pal.line, L, R, y, 1, []);
      ts.forEach((tt) => { ctx.strokeStyle = hlc(['t', 'h', 'lam'], pal, solvCol(pal, i)); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(L + (R - L) * tt, y - 5); ctx.lineTo(L + (R - L) * tt, y + 5); ctx.stroke(); });
    });
    label(ctx, 't = 0', L, top + 18 * rows + 2, pal.muted, 11); label(ctx, 't = 1', R, top + 18 * rows + 2, pal.muted, 11, 'right');
    const order = (W) => log2(W[1] / W[3]) / 2;
    ro.innerHTML = `${kind === 'uniform' ? 'uniform grid' : 'log-SNR grid'}, NFE = <b>${nfe}</b> · ` +
      SOLV.map(([name, n]) => { const W = solverW1(name, kind); return `${n}: W1 <b>${fw(W[NFE1.indexOf(nfe)])}</b>`; }).join(' · ') +
      `<br>slope between NFE 4 and 16 (the estimated order): ${SOLV.map(([name, n]) => `${n} <b>${f2(order(solverW1(name, kind)))}</b>`).join(', ')}`;
  }
  return { draw };
}

// ---------- Step 2: best-of-N ----------
const FRONT = (() => { const out = []; for (let i = 0; i <= 80; i++) { const b = 10 ** (-1.8 + 3 * i / 80), p = tilt(b); out.push([klGrid(p, pData), meanR(p)]); } return out; })();
const bonInfo = memo((N) => { const p = bestOfN(N), kl = klBoN(N), beta = betaForKL(kl), q = Number.isFinite(beta) ? tilt(beta) : pData; return { p, kl, klg: klGrid(p, pData), beta, q, rN: meanR(p), rS: meanR(q), w: w1Dens(p, q) }; });
function bestofnScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let N = 8;
  viz.append(c.cv, controls(listSlider('N', NS2, N, (v) => { N = v; draw(); })), ro,
    note('Exact: p<sub>N</sub> is N p<sub>data</sub> F<sub>r</sub>(r)<sup>N − 1</sup> on a grid, with F<sub>r</sub> from the CDF of the mixture, and p* is the tilted density at the β whose KL to p<sub>data</sub> equals log N − (N − 1)/N. The grid has 361 points, which limits N to 128.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), I = bonInfo(N, N);
    const ym = dmaxL(I.p, I.q, pData), ax = densAxes(ctx, pal, P1, ym, `Best of N = ${N}`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.4, fill: 0.14 });
    rewardCurve(ctx, ax, pal, ym);
    curve(ctx, ax, I.q, hlc(['ps', 'beta'], pal, pal.ok), { lw: 2 });
    curve(ctx, ax, I.p, hlc(['pN', 'N', 'Fr'], pal, pal.ramp[3]), { lw: 2.2 });
    legend(ctx, ax, [['p_{data}', hlc('pd', pal, pal.muted)], ['reward r (scaled)', hlc('r', pal, pal.muted), [5, 4]], ['p_N', hlc(['pN', 'N'], pal, pal.ramp[3])], ['p* at the same KL', hlc('ps', pal, pal.ok)]], true);
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: 4, ymin: 0, ymax: 1, xt: [0, 1, 2, 3, 4], yt: [0, 0.25, 0.5, 0.75, 1], xl: 'KL(p ‖ p_{data})', title: 'Mean reward against KL' });
    clip(ctx, ax2.L, ax2.T, ax2.R - ax2.L, ax2.B - ax2.T);
    polyline(ctx, FRONT.map((q) => ax2.X(q[0])), FRONT.map((q) => ax2.Y(q[1])), hlc(['ps', 'beta'], pal, pal.ok), 2);
    const B = NS2.map((n) => bonInfo(n, n));
    polyline(ctx, B.map((q) => ax2.X(q.kl)), B.map((q) => ax2.Y(q.rN)), hlc(['pN', 'N'], pal, pal.ramp[3]), 2);
    B.forEach((q, i) => dot(ctx, ax2.X(q.kl), ax2.Y(q.rN), NS2[i] === N ? 4.5 : 2.8, hlc(['pN', 'N'], pal, pal.ramp[3])));
    ctx.restore();
    dot(ctx, ax2.X(I.kl), ax2.Y(I.rS), 4.5, hlc('ps', pal, pal.ok));
    legend(ctx, ax2, [['p* over β', hlc('ps', pal, pal.ok)], ['best-of-N', hlc(['pN', 'N'], pal, pal.ramp[3])]], false, true);
    ro.innerHTML = `N = <b>${N}</b> · E[r] of the winner <b>${f3(I.rN)}</b> (E[r] under p<sub>data</sub> ${f3(meanR(pData))}) · KL(p<sub>N</sub> ‖ p<sub>data</sub>) on the grid <b>${f3(I.klg)}</b>, log N − (N − 1)/N = <b>${f3(I.kl)}</b>` +
      `<br>p* at β = <b>${Number.isFinite(I.beta) ? f3(I.beta) : '∞'}</b>: E[r] <b>${f3(I.rS)}</b> · W1(p<sub>N</sub>, p*) <b>${f3(I.w)}</b>`;
  }
  return { draw };
}

// ---------- Step 3: searching the noise ----------
const NN3 = [5, 10, 20, 40, 80, 160], RHO3 = [0.1, 0.2, 0.4, 0.8];
const REWN3 = [['far', 'Reward beyond the data'], ['rare', 'Rare reward in the data']];
const search = memo((kind, n, rho, rk) => {
  const S = search3(kind, n, rho, 5, rk), chain = [S.pts[0]];
  S.pts.forEach((p) => { if (p.r > chain[chain.length - 1].r) chain.push(p); });
  return { ...S, chain };
});
// Reward toggle shared by steps 3 and 4.
function rewardButtons(get, set) {
  const bs = REWN3.map(([k, n]) => button(n, () => { set(k); sync(); }));
  const sync = () => bs.forEach((b, i) => b.classList.toggle('on', REWN3[i][0] === get()));
  sync(); return bs;
}
// The noise plane shaded by the reward of the end point, with circles at |x_0| = √2 and 3.
function noisePlane(ctx, pal, P, rk, title) {
  const ax = sqAxes(ctx, pal, P, title, LIM3), PB = pullback3(rk);
  let lo = Infinity, hi = -Infinity; PB.r.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
  const cs = ax.s / NG3, pw = rk === 'far' ? 3 : 0.5;
  ctx.fillStyle = hlc('r', pal, pal.acc2);
  // Cell edges rounded to whole pixels, so that neighbouring cells neither overlap nor leave seams.
  const e = (k) => Math.round(k * cs);
  for (let j = 0; j < NG3; j++) for (let i = 0; i < NG3; i++) {
    ctx.globalAlpha = 0.04 + 0.5 * ((PB.r[j * NG3 + i] - lo) / (hi - lo)) ** pw;
    const y0 = e(NG3 - 1 - j); ctx.fillRect(Math.round(ax.L) + e(i), Math.round(ax.T) + y0, e(i + 1) - e(i), e(NG3 - j) - y0);
  }
  ctx.globalAlpha = 1;
  circle(ctx, ax, Math.SQRT2, hlc('z', pal, pal.fg)); circle(ctx, ax, 3, pal.muted, [2, 4]);
  label(ctx, '|x_0| = √2', ax.X(0) + 4, ax.Y(Math.SQRT2) - 4, hlc('z', pal, pal.fg), 11);
  label(ctx, '|x_0| = 3', ax.X(0) + 4, ax.Y(3) - 4, pal.muted, 11);
  return ax;
}
// The reward in data space: a cross at a, or the circle r = 1/2 around b.
function rewardMark(ctx, pal, ax, rk) {
  const ac = hlc('r', pal, pal.acc2), [px, py] = REW3[rk].at; ctx.strokeStyle = ac; ctx.lineWidth = 2;
  if (rk === 'far') {
    ctx.beginPath(); ctx.moveTo(ax.X(px) - 6, ax.Y(py) - 6); ctx.lineTo(ax.X(px) + 6, ax.Y(py) + 6); ctx.moveTo(ax.X(px) - 6, ax.Y(py) + 6); ctx.lineTo(ax.X(px) + 6, ax.Y(py) - 6); ctx.stroke();
    label(ctx, 'a', ax.X(px) + 8, ax.Y(py) - 6, ac, 12);
  } else {
    ctx.beginPath(); ctx.arc(ax.X(px), ax.Y(py), ax.X(1.1774 * W3) - ax.X(0), 0, 7); ctx.stroke();
    label(ctx, 'b', ax.X(px) + 12, ax.Y(py) - 8, ac, 12);
  }
}
const bigDot = (ctx, pal, X, Y, col) => { dot(ctx, X, Y, 6, pal.fg); dot(ctx, X, Y, 4.5, col); };
function searchScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.78 }), ro = readout();
  let n = 40, rho = 0.4, rk = 'far';
  viz.append(c.cv, controls(...rewardButtons(() => rk, (v) => { rk = v; draw(); })), controls(listSlider('n', NN3, n, (v) => { n = v; draw(); }, (v) => `${v} (${v * NFE_EVAL3} NFE)`), listSlider('rho', RHO3, rho, (v) => { rho = v; draw(); })), ro,
    note(`Exact field of the 2D mixture; each evaluation solves the ODE with RK4 in ${K3} steps (${NFE_EVAL3} NFE). The shading is the reward of the end point on a ${NG3} × ${NG3} grid of noises. All three searches use the same seed, so their first noise is the same. Trust-region search spends ${TRS3.warm * 100}% of the budget on random draws, then keeps ${TRS3.k} boxes around the ${TRS3.k} best noises; the squares are the boxes after the last round. The searches are single seeded runs, so other seeds give other numbers.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const R = search('random' + n + rk, 'random', n, rho, rk), Lc = search('local' + n + ':' + rho + rk, 'local', n, rho, rk), T = search('trs' + n + rk, 'trs', n, rho, rk);
    const cR = pal.ramp[0], cL = pal.ramp[3], cT = pal.ramp[1];
    const ax = noisePlane(ctx, pal, P1, rk, 'Noise x_0, shaded by the reward of X_{0→1}(x_0)');
    clip(ctx, ax.L, ax.T, ax.s, ax.s);
    R.pts.forEach((p) => dot(ctx, ax.X(p.z[0]), ax.Y(p.z[1]), 2.4, cR));
    T.pts.forEach((p) => dot(ctx, ax.X(p.z[0]), ax.Y(p.z[1]), 1.8, cT));
    Lc.pts.forEach((p) => dot(ctx, ax.X(p.z[0]), ax.Y(p.z[1]), 1.8, cL));
    polyline(ctx, Lc.chain.map((p) => ax.X(p.z[0])), Lc.chain.map((p) => ax.Y(p.z[1])), cL, 1.6);
    ctx.strokeStyle = cT; ctx.lineWidth = 2;
    (T.boxes ?? []).forEach(({ c: [bx, by], l }) => { const s = ax.X(l / 2) - ax.X(0); ctx.strokeRect(ax.X(bx) - s, ax.Y(by) - s, 2 * s, 2 * s); });
    [[R.best, cR], [T.best, cT], [Lc.best, cL]].forEach(([b, col]) => bigDot(ctx, pal, ax.X(b.z[0]), ax.Y(b.z[1]), col));
    ctx.restore();
    legend(ctx, ax, [['random search', cR], ['trust-region search', cT], ['local search', cL]], true, true);
    const ax2 = sqAxes(ctx, pal, P2, 'End points X_{0→1}(x_0) in data space', 3.5);
    dataBlobs(ctx, ax2, pal, hlc('pd', pal, pal.muted));
    clip(ctx, ax2.L, ax2.T, ax2.s, ax2.s);
    R.pts.forEach((p) => dot(ctx, ax2.X(p.x[0]), ax2.Y(p.x[1]), 2.4, cR));
    T.pts.forEach((p) => dot(ctx, ax2.X(p.x[0]), ax2.Y(p.x[1]), 1.8, cT));
    Lc.pts.forEach((p) => dot(ctx, ax2.X(p.x[0]), ax2.Y(p.x[1]), 1.8, cL));
    [[R.best, cR], [T.best, cT], [Lc.best, cL]].forEach(([b, col]) => bigDot(ctx, pal, ax2.X(b.x[0]), ax2.Y(b.x[1]), col));
    ctx.restore();
    rewardMark(ctx, pal, ax2, rk);
    const row = (name, S) => { const b = S.best, nz = Math.hypot(b.z[0], b.z[1]); return `${name}: best r <b>${fr(b.r)}</b>, |x<sub>0</sub>| = <b>${f2(nz)}</b>, P(|Z| > |x<sub>0</sub>|) = <b>${tail2(b.z).toExponential(1)}</b>, p<sub>data</sub> at the end point <b>${p2data(b.x[0], b.x[1]).toExponential(1)}</b>`; };
    ro.innerHTML = `n = <b>${n}</b> evaluations, <b>${n * NFE_EVAL3}</b> NFE<br>${row('random search', R)}<br>${row('trust-region search', T)}<br>${row(`local search, ρ = ${rho}`, Lc)}`;
  }
  return { draw };
}
// Rewards span many orders of magnitude on the rare bump.
const fr = (v) => (v !== 0 && Math.abs(v) < 1e-3 ? v.toExponential(1) : f3(v));

// ---------- Step 4: optimizing through the flow ----------
const IT4 = [0, 5, 10, 20, 40, 80], BETA4 = [0, 0.1, 0.3, 1];
const grad4 = memo((it, beta, rk) => ({ D: dflowG(it, beta, rk), O: ocflowG(it, beta, rk) }));
const free4 = memo(() => path2(ZG));
function flowgradScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.78 }), ro = readout();
  let it = 20, beta = 0, rk = 'far';
  viz.append(c.cv, controls(...rewardButtons(() => rk, (v) => { rk = v; draw(); })), controls(listSlider('it', IT4, it, (v) => { it = v; draw(); }, (v) => `${v} (${v * NFE_ITG} NFE)`), listSlider('beta', BETA4, beta, (v) => { beta = v; draw(); })), ro,
    note(`Exact field and its exact Jacobian. Both methods start from the first noise of step 3. D-Flow differentiates the RK4 solve of ${NFE_EVAL3} NFE and takes gradient steps of size 0.1 on the noise. OC-Flow uses Euler with ${KOC} steps and one control per step, updated with step size 0.5 from the costate of a backward pass. An iteration of either is counted as ${NFE_ITG} NFE: one forward solve and one backward pass of about the same cost.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), { D, O } = grad4(`${it}:${beta}:${rk}`, it, beta, rk), F = free4('f');
    const cD = pal.ramp[0], cO = pal.ramp[2];
    const ax = noisePlane(ctx, pal, P1, rk, 'Noise x_0: the iterates of D-Flow');
    clip(ctx, ax.L, ax.T, ax.s, ax.s);
    polyline(ctx, D.zs.map((q) => ax.X(q[0])), D.zs.map((q) => ax.Y(q[1])), cD, 1.6);
    D.zs.forEach((q) => dot(ctx, ax.X(q[0]), ax.Y(q[1]), 2, cD));
    dot(ctx, ax.X(ZG[0]), ax.Y(ZG[1]), 4.5, pal.fg); bigDot(ctx, pal, ax.X(D.z[0]), ax.Y(D.z[1]), cD);
    ctx.restore();
    legend(ctx, ax, [['start x_0', pal.fg], ['D-Flow iterates', cD]], true, true);
    const ax2 = sqAxes(ctx, pal, P2, 'Paths in data space', 3.5);
    dataBlobs(ctx, ax2, pal, hlc('pd', pal, pal.muted));
    clip(ctx, ax2.L, ax2.T, ax2.s, ax2.s);
    const pl = (xs, col, lw, dash) => polyline(ctx, xs.map((q) => ax2.X(q[0])), xs.map((q) => ax2.Y(q[1])), col, lw, dash);
    pl(F, pal.muted, 1.6, [4, 4]); pl(D.xs, cD, 2); pl(O.xs, hlc('th', pal, cO), 2);
    dot(ctx, ax2.X(ZG[0]), ax2.Y(ZG[1]), 4.5, pal.fg);
    bigDot(ctx, pal, ax2.X(D.x[0]), ax2.Y(D.x[1]), cD); bigDot(ctx, pal, ax2.X(O.x[0]), ax2.Y(O.x[1]), hlc('th', pal, cO));
    ctx.restore();
    rewardMark(ctx, pal, ax2, rk);
    legend(ctx, ax2, [['without optimization', pal.muted, [4, 4]], ['D-Flow, from its last noise', cD], ['OC-Flow, from the start', hlc('th', pal, cO)]], true, true);
    const nz = Math.hypot(D.z[0], D.z[1]);
    ro.innerHTML = `<b>${it}</b> iterations, <b>${it * NFE_ITG}</b> NFE, β = <b>${beta}</b><br>` +
      `D-Flow: r <b>${fr(D.r)}</b>, |x<sub>0</sub>| = <b>${f2(nz)}</b>, P(|Z| > |x<sub>0</sub>|) = <b>${tail2(D.z).toExponential(1)}</b>, p<sub>data</sub> at the end point <b>${D.p.toExponential(1)}</b><br>` +
      `OC-Flow: r <b>${fr(O.r)}</b>, control energy <b>${O.E < 1e-3 ? O.E.toExponential(1) : f3(O.E)}</b>, p<sub>data</sub> at the end point <b>${O.p.toExponential(1)}</b>`;
  }
  return { draw };
}

// ---------- Step 5: inverse problems ----------
const YS4 = [-1.2, -0.6, 0, 0.4, 0.8, 1.4], SY4 = [0.1, 0.3, 0.5, 1], ZE4 = [0.5, 1, 2, 4];
const METH4 = [['exact', 'exact posterior field'], ['pigdm', 'ΠGDM'], ['dps', 'DPS']];
const col4 = (pal, k) => ({ exact: pal.ok, pigdm: pal.ramp[1], dps: pal.ramp[3] }[k]);
const run4 = memo((kind, y, sy, ze) => { const S = sample4(kind, y, sy, ze); return { S, sc: score4(S, y, sy) }; });
function dpsScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.78 }), ro = readout();
  let y = 0, sy = 0.1, ze = 1, show = 'dps';
  const bs = METH4.map(([k, n]) => button(n, () => { show = k; sync(); }));
  const sync = () => { bs.forEach((b, i) => b.classList.toggle('on', METH4[i][0] === show)); draw(); };
  bs[2].classList.add('on');
  viz.append(c.cv, controls(...bs), controls(listSlider('y', YS4, y, (v) => { y = v; draw(); }), listSlider('sy', SY4, sy, (v) => { sy = v; draw(); }), listSlider('zeta', ZE4, ze, (v) => { ze = v; draw(); })), ro,
    note(`Exact: the posterior p(x ∣ y) is again a mixture of Gaussians, and the exact posterior field follows from it. Each method moves the same ${NP4} seeded noises with ${K4} Euler steps. DPS and ΠGDM add a likelihood gradient computed from the exact E[x<sub>1</sub> ∣ x<sub>t</sub>] and its exact Jacobian; ζ acts on DPS only. W1 is measured per coordinate against the exact posterior marginals.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), P = posterior4(y, sy);
    const all = Object.fromEntries(METH4.map(([k]) => [k, run4(`${k}:${y}:${sy}:${k === 'dps' ? ze : 1}`, k, y, sy, k === 'dps' ? ze : 1)]));
    const ax = sqAxes(ctx, pal, P1, `Samples of ${METH4.find((m) => m[0] === show)[1]}, y = ${y}`, 3.5);
    dataBlobs(ctx, ax, pal, hlc('pd', pal, pal.muted));
    clip(ctx, ax.L, ax.T, ax.s, ax.s);
    ctx.fillStyle = hlc(['y', 'sy'], pal, pal.acc2); ctx.globalAlpha = 0.14;
    ctx.fillRect(ax.X(y - sy), ax.T, ax.X(y + sy) - ax.X(y - sy), ax.s); ctx.globalAlpha = 1;
    vdash(ctx, hlc('y', pal, pal.acc2), ax.X(y), ax.T, ax.B, 1.4);
    // The posterior components, at one and two standard deviations, drawn with their weights.
    P.mu.forEach(([mx, my], k) => [1, 2].forEach((m) => {
      ctx.beginPath(); ctx.ellipse(ax.X(mx), ax.Y(my), m * (ax.X(Math.sqrt(P.var[k][0])) - ax.X(0)), m * (ax.X(Math.sqrt(P.var[k][1])) - ax.X(0)), 0, 0, 7);
      ctx.strokeStyle = pal.ok; ctx.globalAlpha = Math.min(1, 0.15 + P.w[k]); ctx.lineWidth = 1.4; ctx.stroke(); ctx.globalAlpha = 1;
    }));
    all[show].S.forEach(([a, b]) => dot(ctx, ax.X(a), ax.Y(b), 1.7, col4(pal, show)));
    ctx.restore();
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: 3, ymin: 0, xt: [0.5, 1.5, 2.5], ymax: 1.45, xfmt: (q) => ['left mode', 'right mode', 'top mode'][Math.floor(q)], yt: [0, 0.25, 0.5, 0.75, 1], title: 'Share of the samples nearest to each mode' });
    const groups = [['posterior weight', pal.muted, P.w], ...METH4.map(([k, n]) => [n, col4(pal, k), all[k].sc.shares])], bw = 0.8 / groups.length;
    groups.forEach(([, col, v], g) => v.forEach((q, k) => { ctx.fillStyle = col; ctx.globalAlpha = g === 0 ? 0.5 : 0.85; const X0 = ax2.X(k + 0.1 + g * bw), X1 = ax2.X(k + 0.1 + (g + 1) * bw) - 2; ctx.fillRect(X0, ax2.Y(q), X1 - X0, ax2.Y(0) - ax2.Y(q)); ctx.globalAlpha = 1; }));
    legend(ctx, ax2, groups.map(([n, col]) => [n, col]), false, false);
    ro.innerHTML = `y = <b>${y}</b>, σ<sub>y</sub> = <b>${sy}</b>, ζ = <b>${ze}</b> · posterior weights ${P.w.map((q) => `<b>${f2(q)}</b>`).join(' / ')}<br>` +
      METH4.map(([k, n]) => { const s = all[k].sc; return `${n}: shares ${s.shares.map(f2).join(' / ')}, W1 per coordinate <b>${f3(s.w1x)}</b> and <b>${f3(s.w1y)}</b>`; }).join('<br>');
  }
  return { draw };
}

// ---------- Step 6: SMC against best-of-N at equal compute ----------
// Results fill in progressively: one chunk of seeded runs per animation frame.
const CH5 = 50, bon5 = new Map(), smc5 = new Map(), guided5 = new Map();
const key5 = (beta, N) => `${beta}:${N}`;
function nextJob(beta) {
  for (const N of NS5) {
    if ((bon5.get(N)?.length ?? 0) < RUNS5) return ['bon', N];
    if ((smc5.get(key5(beta, N))?.length ?? 0) < RUNS5) return ['smc', N];
  }
  return null;
}
function doJob(beta) {
  const j = nextJob(beta); if (!j) return;
  const [kind, N] = j, map = kind === 'bon' ? bon5 : smc5, k = kind === 'bon' ? N : key5(beta, N);
  const arr = map.get(k) ?? []; map.set(k, arr);
  for (let r = arr.length, e = Math.min(RUNS5, arr.length + CH5); r < e; r++) arr.push(kind === 'bon' ? runBoN(N, r) : runSMC(N, beta, r));
}
const ready = (map, k) => (map.get(k)?.length ?? 0) >= RUNS5;
const tilt5 = memo((beta) => tilt(beta));
const ess5 = memo((N, beta) => { const tr = []; runSMC(N, beta, 0, tr); return tr; });
const stat5 = memo((xs, p) => ({ w: w1ToGrid(xs, p), r: xs.reduce((a, x) => a + reward(x), 0) / xs.length }));
function smcScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.72 }), ro = readout();
  let beta = 0.2, N = 16;
  const r = runner({ step: () => doJob(beta), done: () => !nextJob(beta), draw: () => draw(), interval: 0, perFrame: 1 });
  viz.append(c.cv, controls(listSlider('beta', BETAS5, beta, (v) => { beta = v; draw(); r.kick(); }), listSlider('N', NS5, N, (v) => { N = v; draw(); }, (v) => `${v} (${v * K5} NFE)`)), ro,
    note(`Monte Carlo: ${RUNS5} seeded runs per method and N, one output per run, computed in the page and filled in as they finish. All methods simulate the exact SDE of the mixture with noise ε<sub>t</sub><sup>2</sup> = 2(1 − t) in ${K5} Euler–Maruyama steps, so one particle or one sample costs ${K5} NFE. p* is exact on a grid; W1 compares the ${RUNS5} outputs with it. Guidance adds β<sup>−1</sup>∇r(x̂<sub>1</sub>) through the exact Jacobian of x̂<sub>1</sub>, which in a network costs a backward pass per step.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3), q = tilt5(String(beta), beta);
    if (!guided5.has(beta)) guided5.set(beta, Array.from({ length: RUNS5 }, (_, i) => runGuided(beta, i)));
    const G = guided5.get(beta), sG = stat5('g' + beta, G, q);
    const cS = hlc(['N', 'G', 'ess'], pal, pal.ramp[1]), cB = pal.ramp[3], cG = pal.ramp[2];
    const kS = key5(beta, N), haveS = ready(smc5, kS), haveB = ready(bon5, N);
    const dS = haveS ? kde(smc5.get(kS), 0.08) : null, dB = haveB ? kde(bon5.get(N), 0.08) : null;
    const ym = dmaxL(q, pData, ...[dS, dB].filter(Boolean)), ax = densAxes(ctx, pal, P1, ym, `Outputs at N = ${N}, β = ${beta}`);
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.2, fill: 0.12 });
    rewardCurve(ctx, ax, pal, ym);
    curve(ctx, ax, q, hlc(['ps', 'beta'], pal, pal.ok), { lw: 2 });
    if (dB) curve(ctx, ax, dB, cB, { lw: 1.8 });
    if (dS) curve(ctx, ax, dS, cS, { lw: 2 });
    if (!haveS || !haveB) label(ctx, 'computing…', ax.R - 8, ax.T + 16, pal.muted, 11.5, 'right');
    legend(ctx, ax, [['p*', hlc('ps', pal, pal.ok)], ['SMC', cS], ['best-of-N', cB]], true);
    // Middle: W1 to p* against NFE.
    const ax2 = axes(ctx, pal, P2, { xmin: log2(K5) - 0.3, xmax: log2(64 * K5) + 0.3, ymin: 0, ymax: 1.5, xt: NS5.map((n) => log2(n * K5)), xfmt: (v) => String(Math.round(2 ** v)), yt: [0, 0.5, 1, 1.5], xl: 'NFE per output (log scale)', title: 'W1 to p* against compute' });
    vdash(ctx, hlc('N', pal, pal.line), ax2.X(log2(N * K5)), ax2.T, ax2.B);
    [[smc5, (n) => key5(beta, n), cS, 'S'], [bon5, (n) => n, cB, 'B']].forEach(([map, kf, col, tag]) => {
      const pts = NS5.filter((n) => ready(map, kf(n))).map((n) => [ax2.X(log2(n * K5)), ax2.Y(stat5(`${tag}${kf(n)}:${beta}`, map.get(kf(n)), q).w), n]);
      polyline(ctx, pts.map((p) => p[0]), pts.map((p) => p[1]), col, 2);
      pts.forEach((p) => dot(ctx, p[0], p[1], p[2] === N ? 4.5 : 2.8, col));
    });
    dot(ctx, ax2.X(log2(K5)), ax2.Y(sG.w), 4.5, cG);
    legend(ctx, ax2, [['SMC', cS], ['best-of-N', cB], ['guidance (one sample)', cG]], false, false);
    // Right: effective sample size of one SMC run.
    const tr = ess5(kS, N, beta), ax3 = axes(ctx, pal, P3, { xmin: 0, xmax: 1, ymin: 0, ymax: 1, xt: [0, 0.5, 1], yt: [0, 0.5, 1], xl: 't', title: `ESS / N of one run, N = ${N}` });
    hdash(ctx, pal.muted, ax3.L, ax3.R, ax3.Y(0.5), 1, [2, 3]);
    label(ctx, 'resample below', ax3.R - 6, ax3.Y(0.5) - 5, pal.muted, 11, 'right');
    polyline(ctx, tr.map((_, i) => ax3.X((i + 1) / K5)), tr.map((v) => ax3.Y(v)), hlc('ess', pal, cS), 2);
    const res = tr.filter((v) => v < 0.5).length;
    const sS = haveS ? stat5(`S${kS}:${beta}`, smc5.get(kS), q) : null, sB = haveB ? stat5(`B${N}:${beta}`, bon5.get(N), q) : null;
    const fmt = (s) => (s ? `W1 to p* <b>${f3(s.w)}</b>, E[r] <b>${f3(s.r)}</b>` : 'computing…');
    ro.innerHTML = `β = <b>${beta}</b>, N = <b>${N}</b>, ${N * K5} NFE per output · E[r] under p* <b>${f3(meanR(q))}</b><br>SMC: ${fmt(sS)} · best-of-N: ${fmt(sB)}<br>guidance (${K5} NFE and ${K5} gradients): W1 to p* <b>${f3(sG.w)}</b>, E[r] <b>${f3(sG.r)}</b> · resampling steps in the shown run: <b>${res}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 7: scaling against an imperfect verifier ----------
const NS6 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], SIG6 = [0.1, 0.3, 1], B6 = [0, 0.5, 1, 1.2, 1.5, 2];
const sys6 = memo((b) => goldBoNSystematic(b, NS6)), noise6 = memo((s) => goldBoNNoise(s, NS6));
function verifierScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let sig = 0.3, b = 1.5;
  viz.append(c.cv, controls(listSlider('sig', SIG6, sig, (v) => { sig = v; draw(); }), listSlider('b', B6, b, (v) => { b = v; draw(); })), ro,
    note(`The systematic curve is exact: under a deterministic verifier the winner has density N p<sub>data</sub> F<sub>r̂</sub>(r̂)<sup>N − 1</sup>, integrated on a fine grid. The noisy curve is Monte Carlo over ${RUNS6} seeded runs per N, with draws from p<sub>data</sub> and independent Gaussian verifier noise. Both curves report the gold reward r of the selected sample.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const ym = dmaxL(pData), ax = densAxes(ctx, pal, P1, ym, 'Gold reward and the verifier');
    curve(ctx, ax, pData, hlc('pd', pal, pal.muted), { lw: 1.4, fill: 0.14 });
    const sc = 0.85 * ym / Math.max(1, b);
    curve(ctx, ax, Float64Array.from(XG, (x) => sc * (reward(x) + b * spur(x))), hlc(['rh', 'b'], pal, pal.bad), { lw: 1.8, dash: [5, 4] });
    curve(ctx, ax, Float64Array.from(rGrid, (r) => sc * r), hlc('r', pal, pal.ok), { lw: 2 });
    legend(ctx, ax, [['p_{data}', hlc('pd', pal, pal.muted)], ['gold reward r', hlc('r', pal, pal.ok)], ['r + b × spurious bump', hlc(['rh', 'b'], pal, pal.bad), [5, 4]]], true);
    const S0 = sys6('0', 0), Sb = sys6(String(b), b), Ns = noise6(String(sig), sig);
    const ax2 = axes(ctx, pal, P2, { xmin: -0.3, xmax: 10.3, ymin: 0, ymax: 1, xt: [0, 2, 4, 6, 8, 10], xfmt: (v) => String(2 ** v), yt: [0, 0.25, 0.5, 0.75, 1], xl: 'N (log scale)', title: 'Gold reward of the selected sample' });
    const xs = NS6.map((n) => ax2.X(log2(n)));
    polyline(ctx, xs, S0.map(ax2.Y), pal.muted, 1.4, [2, 3]);
    polyline(ctx, xs, Ns.map(ax2.Y), hlc(['sig', 'rh'], pal, pal.ramp[1]), 2);
    polyline(ctx, xs, Sb.map(ax2.Y), hlc(['b', 'rh'], pal, pal.bad), 2);
    [[Ns, hlc(['sig', 'rh'], pal, pal.ramp[1])], [Sb, hlc(['b', 'rh'], pal, pal.bad)]].forEach(([v, col]) => v.forEach((q, i) => dot(ctx, xs[i], ax2.Y(q), 2.8, col)));
    legend(ctx, ax2, [['perfect verifier', pal.muted, [2, 3]], [`noise σ = ${sig}`, hlc(['sig', 'rh'], pal, pal.ramp[1])], [`systematic error b = ${b}`, hlc(['b', 'rh'], pal, pal.bad)]], true);
    const at = (v) => [4, 6, 8, 10].map((i) => `<b>${f3(v[i])}</b>`).join(', ');
    const peak = Sb.reduce((m, q, i) => (q > Sb[m] ? i : m), 0);
    ro.innerHTML = `gold reward at N = 16, 64, 256 and 1024 · perfect verifier ${at(S0)}<br>noise σ = <b>${sig}</b>: ${at(Ns)}<br>systematic error b = <b>${b}</b>: ${at(Sb)} · best N = <b>${NS6[peak]}</b> with <b>${f3(Sb[peak])}</b>`;
  }
  return { draw };
}

export const SCENES = { solvers: solversScene, bestofn: bestofnScene, search: searchScene, flowgrad: flowgradScene, dps: dpsScene, smc: smcScene, verifier: verifierScene };
