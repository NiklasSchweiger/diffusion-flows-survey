// Live scenes of the §2.6 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { dens1 } from '../sde/sim.js';
import {
  MIX1, MIX2, gmm2, LIN, pt1, uExact1, pt2, pdata1, BATCHES, NS, NV, couple, T, uOT, ptOT, W22, COST_IND1, NQ, Z0, euler1, eulerPaths,
  rk4Paths, w1Data, KE_T, KE_IND, NR4, NSTEP4, RECT, NG, GX, sinkhorn, bridges, NBR, NBS, T_S, NK6, TG, dlogh, dloghGuid, tilted, w1To,
  NTR, simulate, startTilted, hSpread, NRUN, smc,
} from './sim.js';

// ---------- Helpers ----------
const LO = -4, HI = 4, XS = Array.from({ length: 321 }, (_, i) => LO + (HI - LO) * i / 320);
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
// Mode a quantile belongs to, from the cumulative mixture weights, and the nearest mode of a data point.
const modeOfQ = (q) => (q < MIX1.w[0] ? 0 : q < MIX1.w[0] + MIX1.w[1] ? 1 : 2);
const ORDER = MIX1.mu.map((m, k) => [m, k]).sort((a, b) => a[0] - b[0]);
const modeOfX = (x) => { let k = ORDER[0][1]; for (let i = 1; i < ORDER.length; i++) if (x > (ORDER[i - 1][0] + ORDER[i][0]) / 2) k = ORDER[i][1]; return k; };
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_\{([^}]*)\}|_/g, '$1')).width + 12, x0 = align === 'right' ? x - w + 6 : align === 'center' ? x - w / 2 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const hline = (ctx, pal, x0, x1, y) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };
function vdash(ctx, color, x, y0, y1) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1.4, dash = [6, 5]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
function curve(ctx, xs, X, Y, f, color, lw, dash) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(f(x, i))));
  if (dash) ctx.setLineDash(dash);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]);
}
function shade(ctx, xs, X, Y, f, base, color, alpha) {
  ctx.beginPath(); ctx.moveTo(X(xs[0]), base); xs.forEach((x, i) => ctx.lineTo(X(x), Y(f(x, i)))); ctx.lineTo(X(xs[xs.length - 1]), base); ctx.closePath();
  ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.fill(); ctx.globalAlpha = 1;
}
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
const sd = (xs) => { let m = 0; for (const v of xs) m += v / xs.length; let s = 0; for (const v of xs) s += (v - m) ** 2 / xs.length; return Math.sqrt(s); };
const tSlider = (value, on, min = 0, max = 1) => slider('t', { min, max, step: 0.01, value }, on);
// Small cache for expensive per-parameter computations.
function memo(fn, size = 8) {
  const m = new Map();
  return (key, ...rest) => { if (!m.has(key)) { if (m.size >= size) m.delete(m.keys().next().value); m.set(key, fn(key, ...rest)); } return m.get(key); };
}
const heatRGB = () => (live.theme === 'dark' ? [[123, 108, 255], 0.55] : [[91, 69, 224], 0.42]);

// World box [-bx, bx] x [-by, by] fitted into a rectangle of the canvas.
function view(x0, y0, w, h, bx = 3.4, by = 2.6) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y], wx: (X, Y) => [(X - cx) / s, (cy - Y) / s] };
}
// Heat map of the 2D data density, cached by size and theme. Alpha follows p^0.7.
const heatCache = new Map();
function heat2(ctx, w, h, v, alphaMul = 0.85) {
  const k = `${w}x${h}|${v.s.toFixed(3)}|${v.px(0, 0).join(',')}|${live.theme}`;
  if (!heatCache.has(k)) {
    if (heatCache.size > 20) heatCache.clear();
    const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res), off = document.createElement('canvas'); off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H);
    const pmax = Math.max(...MIX2.means.map(([mx, my]) => pt2(mx, my, 1))) * 1.05, [rgb, amax] = heatRGB();
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const [x, y] = v.wx(i * res, j * res), o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(Math.min(1, pt2(x, y, 1) / pmax), 0.7));
    }
    g.putImageData(im, 0, 0);
    heatCache.set(k, off);
  }
  ctx.globalAlpha = alphaMul; ctx.drawImage(heatCache.get(k), 0, 0, w, h); ctx.globalAlpha = 1;
}
// Heat map of a 1D density over (t, x), one column per time, each column scaled to its own maximum.
const stripCache = new Map();
function strip(key, dens, t0 = 0, t1 = 1) {
  const k = `${key}|${live.theme}`;
  if (stripCache.has(k)) return stripCache.get(k);
  const W = 161, H = 161, off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H), [rgb, amax] = heatRGB();
  for (let i = 0; i < W; i++) {
    const t = t0 + (t1 - t0) * i / (W - 1), col = Array.from({ length: H }, (_, j) => dens(HI - (HI - LO) * j / (H - 1), t)), m = Math.max(...col);
    for (let j = 0; j < H; j++) {
      const o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(col[j] / m, 0.7));
    }
  }
  g.putImageData(im, 0, 0);
  stripCache.set(k, off);
  return off;
}
// Histogram of samples on [LO, HI] as a density, drawn as bars from the baseline.
function bars(ctx, xs, X, Y, base, color, alpha, nbin = 64) {
  const bw = (HI - LO) / nbin, cnt = new Float64Array(nbin);
  for (const x of xs) { const b = Math.floor((x - LO) / bw); if (b >= 0 && b < nbin) cnt[b]++; }
  ctx.fillStyle = color; ctx.globalAlpha = alpha;
  cnt.forEach((q, b) => { const d = q / (xs.length * bw); if (d > 0) ctx.fillRect(X(LO + b * bw) + 0.5, Y(d), X(LO + (b + 1) * bw) - X(LO + b * bw) - 1, base - Y(d)); });
  ctx.globalAlpha = 1;
  return Math.max(...cnt) / (xs.length * bw);
}
// Sideways histogram: density along a vertical x axis, bars growing to the right from x = L.
function sideBars(ctx, xs, Yx, L, wid, pmax, color, alpha, nbin = 64) {
  const bw = (HI - LO) / nbin, cnt = new Float64Array(nbin);
  for (const x of xs) { const b = Math.floor((x - LO) / bw); if (b >= 0 && b < nbin) cnt[b]++; }
  ctx.fillStyle = color; ctx.globalAlpha = alpha;
  cnt.forEach((q, b) => { const d = q / (xs.length * bw); if (d > 0) { const y0 = Yx(LO + (b + 1) * bw), y1 = Yx(LO + b * bw); ctx.fillRect(L, y0 + 0.5, wid * Math.min(1, d / pmax), y1 - y0 - 1); } });
  ctx.globalAlpha = 1;
}
function sideCurve(ctx, Yx, L, wid, pmax, f, color, lw) {
  ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, L + wid * Math.min(1.02, f(x) / pmax), Yx(x)));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
}

// ---------- Step 1: minibatch OT pairs in 2D ----------
function couplings(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let bi = 0;
  viz.append(c.cv, controls(slider('B', { min: 0, max: BATCHES.length - 1, step: 1, value: bi, map: (v) => BATCHES[v], fmt: (v) => String(v) }, (v) => { bi = BATCHES.indexOf(v); draw(); })), ro,
    note(`${NS} noise points (open circles) and ${NS} points of the 2D mixture of §2.2 (dots, coloured by component), drawn once. Batches are consecutive blocks of B points; inside each the assignment is solved exactly with the Hungarian algorithm. The mean squared length and the conditional spread are estimated from ${NV} further pairs; the spread uses the pairs whose midpoints fall in the same square of side 0.35. The heat map is p<sub>data</sub>.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h, 4.6, 2.6), B = BATCHES[bi], C = couple(B), D = C.D;
    heat2(ctx, w, h, v, 0.55);
    clip(ctx, 0, 0, w, h);
    ctx.lineWidth = isHL('pi', 'sigma') ? 1.5 : 1.1; ctx.globalAlpha = isHL('pi', 'sigma') ? 0.8 : 0.5;
    for (let i = 0; i < D.n; i++) {
      const j = C.m[i];
      ctx.beginPath(); ctx.moveTo(...v.px(D.x0[2 * i], D.x0[2 * i + 1])); ctx.lineTo(...v.px(D.x1[2 * j], D.x1[2 * j + 1]));
      ctx.strokeStyle = hlc(['pi', 'sigma'], pal, modeColor(pal, D.k[j])); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < D.n; i++) ring(ctx, ...v.px(D.x0[2 * i], D.x0[2 * i + 1]), isHL('x0') ? 3.6 : 2.8, hlc('x0', pal, pal.fg), 1.2);
    for (let i = 0; i < D.n; i++) dot(ctx, ...v.px(D.x1[2 * i], D.x1[2 * i + 1]), isHL('x1') ? 3.8 : 3, hlc('x1', pal, modeColor(pal, D.k[i])));
    ctx.restore();
    frame(ctx, pal, 0, 0, w, h);
    tag(ctx, pal, B === 1 ? 'B = 1: independent pairs' : B === NS ? `B = ${B}: one batch, the OT pairing of all ${NS} points` : `B = ${B}: ${NS / B} batches of ${B}`, 16, 26, hlc('B', pal, pal.fg), 14);
    ro.innerHTML = `B = <b>${B}</b> · pairs of lines that cross: <b>${C.cross}</b> of ${NS * (NS - 1) / 2}<br>` +
      `mean squared length E‖x<sub>1</sub> − x<sub>0</sub>‖² <b>${C.len.toFixed(2)}</b> · share of Var[x<sub>1</sub> − x<sub>0</sub>] left once x<sub>0.5</sub> is known <b>${C.cvar.toFixed(2)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: the two fields in 1D ----------
const STARTS = Array.from({ length: 25 }, (_, i) => 20 + 40 * i);   // indices into the 1000 noise quantiles
const ST_X = Float64Array.from(STARTS, (i) => Z0[i]), ST_C = STARTS.map((i) => modeOfQ((i + 0.5) / NQ));
let RK_IND = null;
const eulInd = memo((K) => ({ end: euler1(uExact1, K), paths: eulerPaths(uExact1, K, ST_X) }), 40);
const eulOT = memo((K) => ({ end: euler1(uOT, K), paths: eulerPaths(uOT, K, ST_X) }), 40);
// Velocity-score formula of §2.4 evaluated with each density, against the actual field: mean |gap| under p_t.
function scoreGap(t, ot) {
  const lp = (x) => Math.log(ot ? ptOT(x, t) : pt1(x, t)), c = 0.5 * LIN.g2(t), dx = 0.02;
  let s = 0, z = 0;
  for (let x = -3.5; x <= 3.5; x += 0.05) {
    const p = ot ? ptOT(x, t) : pt1(x, t), score = ot ? (lp(x + 1e-3) - lp(x - 1e-3)) / 2e-3 : dens1(x, t).score;
    const u = ot ? uOT(x, t) : uExact1(x, t);
    s += p * Math.abs(u - (x / t + c * score)) * dx; z += p * dx;
  }
  return s / z;
}
let GAPS = null;
function field(viz) {
  const c = makeCanvas(16 / 11, { maxH: 0.78 }), ro = readout();
  let K = 2;
  viz.append(c.cv, controls(slider('K', { min: 1, max: 32, step: 1, value: K, fmt: (v) => String(v) }, (v) => { K = v; draw(); })), ro,
    note('Exact fields of the 1D mixture. Upper panel: the independent coupling, its p<sub>t</sub> as a heat map (each column scaled to its maximum) and 25 RK4 paths of its marginal velocity (thin). Lower panel: the optimal coupling, its p<sub>t</sub> and the same 25 starting points on their straight lines. Bold: K-step Euler polygons from the same starts. The strips on the right show where 1000 stratified noise draws end after K Euler steps (bars), against p<sub>data</sub> (line).'));
  function draw() {
    c.fit();
    if (!RK_IND) RK_IND = rk4Paths(ST_X);
    if (!GAPS) GAPS = [0.3, 0.5, 0.8].map((t) => [t, scoreGap(t, false), scoreGap(t, true)]);
    const { ctx, w, h } = c, pal = palette(), L = 46, R = Math.round(w * 0.78), sL = R + 14, sR = w - 12;
    const panels = [
      { key: 'ind', top: 30, bot: h / 2 - 22, dens: pt1, E: eulInd(K), thin: (i, s) => RK_IND[i][Math.round(s * 100)], name: 'independent coupling' },
      { key: 'ot', top: h / 2 + 26, bot: h - 30, dens: ptOT, E: eulOT(K), thin: (i, s) => (1 - s) * ST_X[i] + s * T(ST_X[i]), name: 'optimal coupling' },
    ];
    const pd = XS.map(pdata1), pmax = Math.max(...pd) * 1.15;
    for (const P of panels) {
      const X = (s) => L + (R - L) * s, Y = (x) => P.top + (P.bot - P.top) * (HI - x) / (HI - LO);
      ctx.globalAlpha = isHL('pt') ? 1 : 0.85; ctx.drawImage(strip(P.key, P.dens), L, P.top, R - L, P.bot - P.top); ctx.globalAlpha = 1;
      clip(ctx, L, P.top, R - L, P.bot - P.top);
      ctx.lineWidth = 1; ctx.globalAlpha = isHL('vel', 'T') ? 0.9 : 0.55;
      ST_X.forEach((_, i) => {
        ctx.beginPath(); for (let k = 0; k <= 100; k += 2) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k / 100), Y(P.thin(i, k / 100)));
        ctx.strokeStyle = hlc(P.key === 'ot' ? ['vel', 'T'] : 'vel', pal, pal.muted); ctx.stroke();
      });
      ctx.globalAlpha = 1;
      P.E.paths.forEach((p, i) => {
        const ec = hlc('K', pal, modeColor(pal, ST_C[i]));
        ctx.beginPath(); p.forEach((x, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, X(j / K), Y(x)));
        ctx.strokeStyle = ec; ctx.lineWidth = isHL('K') ? 2.2 : 1.5; ctx.stroke();
        if (K <= 16) p.forEach((x, j) => dot(ctx, X(j / K), Y(x), 1.8, ec));
      });
      ctx.restore();
      frame(ctx, pal, L, P.top, R - L, P.bot - P.top);
      [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 8, Y(v) + 4, pal.muted, 11, 'right'));
      tag(ctx, pal, P.name, L + 10, P.top + 20, pal.fg, 13.5);
      // End points after K Euler steps against p_data.
      frame(ctx, pal, sL, P.top, sR - sL, P.bot - P.top);
      sideBars(ctx, P.E.end, Y, sL + 1, sR - sL - 4, pmax, hlc('K', pal, pal.fg), 0.35);
      sideCurve(ctx, Y, sL + 1, sR - sL - 4, pmax, pdata1, pal.acc2, 1.8);
      P.w1 = w1Data(P.E.end);
    }
    [0, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), L + (R - L) * s, h - 14, pal.muted, 11, 'center'));
    label(ctx, 't', R + 8, h - 14, hlc('t', pal, pal.muted), 12);
    label(ctx, `end after ${K} step${K > 1 ? 's' : ''}`, sR, 18, pal.muted, 11.5, 'right');
    ro.innerHTML = `K = <b>${K}</b> · W<sub>1</sub> of the 1000 Euler end points to p<sub>data</sub>: independent <b>${panels[0].w1.toFixed(3)}</b> · optimal <b>${panels[1].w1.toFixed(3)}</b> (the stratified sample itself has about 0.002)<br>` +
      `mean gap between u<sub>t</sub> and (α̇<sub>t</sub>/α<sub>t</sub>)x + ½σ<sub>t</sub>²λ̇<sub>t</sub>∇log p<sub>t</sub>: ` +
      GAPS.map(([t, a, b]) => `t = ${t}: independent <b>${a.toExponential(0)}</b>, optimal <b>${b.toFixed(1)}</b>`).join(' · ');
  }
  return { draw };
}

// ---------- Step 3: kinetic energy ----------
function energy(viz) {
  const c = makeCanvas(16 / 11, { maxH: 0.74 }), ro = readout();
  let t = 0.5;
  viz.append(c.cv, controls(tSlider(t, (v) => { t = v; draw(); })), ro,
    note(`Upper panel: the exact p<sub>t</sub> of the two couplings at the time of the slider. Lower panel: kinetic energy per unit time, ∫u<sub>t</sub>² p<sub>t</sub> dx. For the independent coupling it is a quadrature of the exact field on 1401 points, at 101 times; its integral uses Simpson’s rule. The dashed line is E|x<sub>1</sub> − x<sub>0</sub>|² of the random pairs, which is exact.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), L = 52, R = w - 18, top1 = 30, bot1 = h * 0.44, top2 = h * 0.56, bot2 = h - 34;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    const ind = XS.map((x) => pt1(x, t)), ot = XS.map((x) => ptOT(x, t)), pm = Math.max(...ind, ...ot) * 1.1;
    const Y1 = (p) => bot1 - (bot1 - top1) * p / pm;
    shade(ctx, XS, X, Y1, (x, i) => ind[i], bot1, pal.ramp[1], isHL('pt') ? 0.3 : 0.14);
    curve(ctx, XS, X, Y1, (x, i) => ind[i], hlc('pt', pal, pal.ramp[1]), 2.2);
    curve(ctx, XS, X, Y1, (x, i) => ot[i], hlc(['pt', 'T'], pal, pal.acc), 2.6);
    curve(ctx, XS, X, Y1, (x) => pdata1(x), pal.muted, 1, [3, 4]);
    hline(ctx, pal, L, R, bot1);
    [-3, 0, 3].forEach((x) => label(ctx, String(x), X(x), bot1 + 15, pal.muted, 11, 'center'));
    label(ctx, `p_t at t = ${t.toFixed(2)}: independent (shaded), optimal (bold) · dotted: p_{data}`, L + 4, top1 - 12, hlc('pt', pal, pal.fg), 13);
    label(ctx, 'x', R, bot1 + 15, hlc('x', pal, pal.muted), 12, 'right');
    // Lower panel.
    const EM = 3.6, X2 = (s) => L + (R - L) * s, Y2 = (e) => bot2 - (bot2 - top2) * e / EM, TT = Array.from({ length: 101 }, (_, j) => j / 100);
    shade(ctx, TT, X2, Y2, (s, j) => KE_T[j], bot2, pal.ramp[1], isHL('KE') ? 0.3 : 0.14);
    curve(ctx, TT, X2, Y2, (s, j) => KE_T[j], hlc(['KE', 'vel'], pal, pal.ramp[1]), 2.2);
    hdash(ctx, hlc(['W2', 'T'], pal, pal.acc), L, R, Y2(W22), 2.6, []);
    hdash(ctx, pal.muted, L, R, Y2(COST_IND1));
    hline(ctx, pal, L, R, bot2);
    vdash(ctx, hlc('t', pal, pal.fg), X2(t), top2, bot2);
    dot(ctx, X2(t), Y2(KE_T[Math.round(t * 100)]), 4, pal.ramp[1]); dot(ctx, X2(t), Y2(W22), 4, pal.acc);
    [0, 1, 2, 3].forEach((e) => label(ctx, String(e), L - 8, Y2(e) + 4, pal.muted, 11, 'right'));
    [0, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), X2(s), bot2 + 16, pal.muted, 11, 'center'));
    label(ctx, 't', R, bot2 + 30, hlc('t', pal, pal.muted), 12, 'right');
    label(ctx, 'kinetic energy per unit time: independent field (shaded), optimal (bold, flat at W₂²)', L + 4, top2 - 12, hlc('KE', pal, pal.fg), 13);
    tag(ctx, pal, 'E|x_1 − x_0|² of the random pairs', R - 6, Y2(COST_IND1) - 8, pal.muted, 12, 'right');
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · energy rate: independent <b>${KE_T[Math.round(t * 100)].toFixed(3)}</b>, optimal <b>${W22.toFixed(3)}</b><br>` +
      `W<sub>2</sub>² = <b>${W22.toFixed(3)}</b> ≤ energy of the independent field 𝓔 = <b>${KE_IND.toFixed(3)}</b> ≤ E|x<sub>1</sub> − x<sub>0</sub>|² of its random pairs = <b>${COST_IND1.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: rectification in 2D ----------
const SHOW4 = Array.from({ length: 48 }, (_, i) => 7 + 31 * i);
function rectify(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let t = 1;
  const r = runner({ step: () => { t = t >= 1 ? 0 : Math.min(1, t + 0.01); tSl.set(t); }, done: () => false, draw: () => {}, interval: 30 });
  r.playing = false; r.play.textContent = 'Play';
  const tSl = tSlider(t, (v) => { t = v; draw(); });
  viz.append(c.cv, controls(tSl, r.play), ro,
    note(`${NR4} noise points, integrated with ${NSTEP4} RK4 steps along the exact marginal velocity of the independent coupling of the 2D mixture; 48 of them are drawn. Faint: their random data partners. Bold: the ODE paths up to time t. Dashed: the chords from start to end, the rectified pairs. The bars are averages over all ${NR4} points; the middle one integrates |u<sub>t</sub>(X<sub>t</sub>)|² by the trapezoid rule.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), pw = Math.round(w * 0.74), v = view(0, 0, pw, h, 4.6, 2.6), D = RECT.D, P = RECT.path, j = Math.round(t * NSTEP4);
    heat2(ctx, w, h, v, 0.5);
    clip(ctx, 0, 0, pw, h);
    const kOf = (i) => modeOfQ(0) * 0 + D.k[i];
    ctx.lineWidth = 1; ctx.globalAlpha = isHL('x1', 'x0') ? 0.5 : 0.22;
    for (const i of SHOW4) { ctx.beginPath(); ctx.moveTo(...v.px(D.x0[2 * i], D.x0[2 * i + 1])); ctx.lineTo(...v.px(D.x1[2 * i], D.x1[2 * i + 1])); ctx.strokeStyle = hlc(['x0', 'x1'], pal, pal.muted); ctx.stroke(); }
    ctx.globalAlpha = 1;
    const last = P[NSTEP4];
    for (const i of SHOW4) {
      ctx.beginPath(); ctx.moveTo(...v.px(D.x0[2 * i], D.x0[2 * i + 1])); ctx.lineTo(...v.px(last[2 * i], last[2 * i + 1]));
      ctx.setLineDash([4, 4]); ctx.strokeStyle = hlc('c', pal, pal.fg); ctx.globalAlpha = 0.55; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.beginPath(); for (let k = 0; k <= j; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, ...v.px(P[k][2 * i], P[k][2 * i + 1]));
      ctx.strokeStyle = hlc(['X', 'vel'], pal, pal.acc); ctx.lineWidth = isHL('X', 'vel') ? 2.4 : 1.7; ctx.stroke();
      ring(ctx, ...v.px(D.x0[2 * i], D.x0[2 * i + 1]), 2.6, pal.fg, 1.1);
      dot(ctx, ...v.px(P[j][2 * i], P[j][2 * i + 1]), isHL('X') ? 4.2 : 3.2, hlc('X', pal, pal.acc));
    }
    void kOf;
    ctx.restore();
    frame(ctx, pal, 0, 0, pw, h);
    tag(ctx, pal, `t = ${(j / NSTEP4).toFixed(2)}`, pw - 14, 26, hlc('t', pal, pal.muted), 13.5, 'right');
    // The three terms of the chain.
    const bL = pw + 26, bR = w - 12, top = 54, bot = h - 46, M = Math.ceil(RECT.old + 0.5), Yb = (e) => bot - (bot - top) * e / M;
    const items = [[RECT.rect, 'new pairs', hlc(['c', 'X'], pal, pal.acc)], [RECT.mid, 'field', hlc(['vel', 'KE'], pal, pal.ramp[1])], [RECT.old, 'old pairs', hlc(['x0', 'x1'], pal, pal.muted)]];
    const bw = (bR - bL) / 3;
    items.forEach(([val, name, col], k) => {
      const x0 = bL + k * bw + 6;
      ctx.fillStyle = col; ctx.globalAlpha = 0.75; ctx.fillRect(x0, Yb(val), bw - 12, bot - Yb(val)); ctx.globalAlpha = 1;
      label(ctx, val.toFixed(2), x0 + (bw - 12) / 2, Yb(val) - 6, pal.fg, 12, 'center');
      label(ctx, name, x0 + (bw - 12) / 2, bot + 16, pal.muted, 11, 'center');
    });
    hline(ctx, pal, bL, bR, bot);
    label(ctx, 'E|·|², three terms', (bL + bR) / 2, 26, pal.fg, 12.5, 'center');
    ro.innerHTML = `t = <b>${(j / NSTEP4).toFixed(2)}</b> · E|X<sub>1</sub> − X<sub>0</sub>|² = <b>${RECT.rect.toFixed(3)}</b> ≤ ∫E|u<sub>t</sub>(X<sub>t</sub>)|²dt = <b>${RECT.mid.toFixed(3)}</b> ≤ E|x<sub>1</sub> − x<sub>0</sub>|² = <b>${RECT.old.toFixed(3)}</b> (exact value for the independent coupling ${RECT.exact.toFixed(3)})`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: Sinkhorn and the bridges ----------
const sinkImg = new Map();
function couplingImage(S) {
  const k = `${S.eps.toFixed(3)}|${live.theme}`;
  if (sinkImg.has(k)) return sinkImg.get(k);
  if (sinkImg.size > 30) sinkImg.clear();
  const off = document.createElement('canvas'); off.width = NG; off.height = NG;
  const g = off.getContext('2d'), im = g.createImageData(NG, NG), [rgb] = heatRGB(), m = Math.max(...S.P);
  for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) {
    const o = 4 * ((NG - 1 - j) * NG + i);   // x0 to the right, x1 upwards
    im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * 0.9 * Math.pow(S.P[i * NG + j] / m, 0.45));
  }
  g.putImageData(im, 0, 0);
  sinkImg.set(k, off);
  return off;
}
function bridge(viz) {
  const c = makeCanvas(16 / 8.5), ro = readout();
  let eps = 0.5;
  viz.append(c.cv, controls(slider('eps', { min: -1, max: Math.log10(3), step: 0.02, value: Math.log10(eps), map: (v) => 10 ** v, fmt: (v) => v.toFixed(2) }, (v) => { eps = v; draw(); })), ro,
    note(`Both marginals live on a grid of ${NG} points in [−4, 4]. The coupling minimizes E<sub>π</sub>|x<sub>1</sub> − x<sub>0</sub>|² − 2ε²H(π), solved with log-domain Sinkhorn iterations to a marginal error below 1e-7, warm-started from the previous ε. Left: π as a heat map (x<sub>0</sub> to the right, x<sub>1</sub> upwards, intensity ∝ π<sup>0.45</sup>) and the optimal map T (thin). Right: ${NBR} bridges, endpoints drawn from π with fixed uniforms plus ε times a fixed Brownian bridge.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), S = sinkhorn(eps), Bs = bridges(S);
    const top = 30, bot = h - 34, side = bot - top, L = 46, R = L + side;
    const PX = (x) => L + side * (x - LO) / (HI - LO), PY = (y) => bot - side * (y - LO) / (HI - LO);
    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = isHL('pi', 'H') ? 1 : 0.9; ctx.drawImage(couplingImage(S), L, top, side, side); ctx.globalAlpha = 1;
    clip(ctx, L, top, side, side);
    curve(ctx, XS, PX, PY, (x) => T(x), hlc('T', pal, pal.fg), isHL('T') ? 2.4 : 1.2, [4, 3]);
    ctx.restore();
    frame(ctx, pal, L, top, side, side);
    [-3, 0, 3].forEach((v) => { label(ctx, String(v), PX(v), bot + 16, pal.muted, 11, 'center'); label(ctx, String(v), L - 8, PY(v) + 4, pal.muted, 11, 'right'); });
    label(ctx, 'x_0', R, bot + 30, hlc('x0', pal, pal.muted), 12, 'right');
    label(ctx, 'x_1', L - 8, top - 10, hlc('x1', pal, pal.muted), 12, 'right');
    label(ctx, 'coupling π', L + 4, top - 10, hlc('pi', pal, pal.fg), 13);
    // Bridges over (t, x).
    const bL = R + 70, bR = w - 14, X = (s) => bL + (bR - bL) * s, Y = (x) => top + (bot - top) * (HI - x) / (HI - LO);
    clip(ctx, bL, top, bR - bL, bot - top);
    ctx.lineWidth = isHL('eps', 'z') ? 1.6 : 1.1; ctx.globalAlpha = 0.75;
    for (const b of Bs) {
      ctx.beginPath(); b.path.forEach((x, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(k / NBS), Y(x)));
      ctx.strokeStyle = hlc(['eps', 'z', 'x'], pal, modeColor(pal, modeOfX(b.x1))); ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.restore();
    frame(ctx, pal, bL, top, bR - bL, bot - top);
    [-3, 0, 3].forEach((v) => label(ctx, String(v), bL - 8, Y(v) + 4, pal.muted, 11, 'right'));
    [0, 0.5, 1].forEach((s) => label(ctx, String(s), X(s), bot + 16, pal.muted, 11, 'center'));
    label(ctx, 't', bR, bot + 30, hlc('t', pal, pal.muted), 12, 'right');
    label(ctx, `${NBR} bridges, coloured by the component nearest to x_1`, bL + 4, top - 10, pal.fg, 13);
    ro.innerHTML = `ε = <b>${eps.toFixed(2)}</b> · transport cost E<sub>π</sub>|x<sub>1</sub> − x<sub>0</sub>|² <b>${S.cost.toFixed(3)}</b> (optimal W<sub>2</sub>² = ${W22.toFixed(3)}, independent ${COST_IND1.toFixed(3)})<br>` +
      `mutual information of x<sub>0</sub> and x<sub>1</sub> <b>${S.mi.toFixed(2)}</b> nats · Sinkhorn iterations <b>${S.it}</b>`;
  }
  return { draw };
}

// ---------- Step 6: the exact control ----------
let BASE = null;
const baseRun = () => (BASE ??= simulate(null));
const exactRun = memo((key, beta, a) => {
  const st = startTilted(beta, a), run = simulate((x, s) => dlogh(x, s, beta, a), st.x0), tg = tilted(beta, a);
  return { ...run, startKL: st.kl, tg, spread: hSpread(beta, a), w1: w1To(run.end, tg.Q), sd: sd(run.end) };
}, 12);
const BETA_DEF = 0.3, A_DEF = 1;
// Paths over (t, x) on the time grid of the sampler, plus end histograms on the right.
function pathsPanel(ctx, pal, { L, R, top, bot }, runs, a) {
  const X = (s) => L + (R - L) * (s - T_S) / (1 - T_S), Y = (x) => top + (bot - top) * (HI - x) / (HI - LO);
  clip(ctx, L, top, R - L, bot - top);
  for (const { trails, color, alpha, lw } of runs) {
    ctx.globalAlpha = alpha; ctx.lineWidth = lw; ctx.strokeStyle = color;
    for (const tr of trails) { ctx.beginPath(); for (let k = 0; k <= NK6; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(TG[k]), Y(tr[k])); ctx.stroke(); }
  }
  ctx.globalAlpha = 1; ctx.restore();
  hdash(ctx, hlc(['a', 'r'], pal, pal.acc2), L, R, Y(a), isHL('a', 'r') ? 2 : 1.2, [4, 4]);
  frame(ctx, pal, L, top, R - L, bot - top);
  [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 8, Y(v) + 4, pal.muted, 11, 'right'));
  [T_S, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), X(s), bot + 16, pal.muted, 11, 'center'));
  label(ctx, 't', R, bot + 30, hlc('t', pal, pal.muted), 12, 'right');
  return Y;
}
function histPanel(ctx, pal, { L, R, top, bot }, Y, xs, pstar, color) {
  const pd = XS.map(pdata1), pm = Math.max(...pd, ...XS.map(pstar)) * 1.1;
  frame(ctx, pal, L, top, R - L, bot - top);
  sideBars(ctx, xs, Y, L + 1, R - L - 4, pm, color, 0.4);
  sideCurve(ctx, Y, L + 1, R - L - 4, pm, pdata1, pal.muted, 1);
  sideCurve(ctx, Y, L + 1, R - L - 4, pm, pstar, hlc('pstar', pal, pal.fg), isHL('pstar') ? 3 : 2);
}
function control(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let beta = BETA_DEF, a = A_DEF;
  viz.append(c.cv, controls(
    slider('beta', { min: Math.log10(0.05), max: Math.log10(3), step: 0.02, value: Math.log10(beta), map: (v) => 10 ** v, fmt: (v) => v.toFixed(2) }, (v) => { beta = v; draw(); }),
    slider('a', { min: -3, max: 3, step: 0.1, value: a, fmt: (v) => v.toFixed(1) }, (v) => { a = v; draw(); })), ro,
  note(`1000 particles from stratified quantiles of the exact p<sub>${T_S}</sub>, ${NK6} Euler–Maruyama steps uniform in log-SNR up to t = 0.995, memoryless noise ε<sub>t</sub>² = σ<sub>t</sub>²λ̇<sub>t</sub>, exact score of the 1D mixture. The controlled run adds ε<sub>t</sub>²∇log h<sub>t</sub> with the exact h<sub>t</sub> and starts from p<sub>${T_S}</sub> tilted by h<sub>${T_S}</sub>. Base and controlled runs use the same Gaussian increments. ${NTR} paths of each are drawn. Right: end points (bars), p<sup>⋆</sup> (bold) and p<sub>data</sub> (thin). The dashed line marks a.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), B = baseRun(), E = exactRun(`${beta.toFixed(3)}|${a.toFixed(2)}`, beta, a);
    const box = { L: 46, R: Math.round(w * 0.72), top: 30, bot: h - 34 };
    const Y = pathsPanel(ctx, pal, box, [
      { trails: B.trails, color: pal.muted, alpha: 0.45, lw: 1 },
      { trails: E.trails, color: hlc(['u', 'h', 'x'], pal, pal.acc), alpha: 0.85, lw: isHL('u', 'h') ? 1.9 : 1.4 },
    ], a);
    tag(ctx, pal, 'grey: base sampler · bold: controlled, same noise', box.L + 10, box.top + 20, pal.fg, 13);
    histPanel(ctx, pal, { L: box.R + 18, R: w - 12, top: box.top, bot: box.bot }, Y, E.end, E.tg.pdf, hlc('u', pal, pal.acc));
    label(ctx, 'end points', w - 12, 18, pal.muted, 11.5, 'right');
    ro.innerHTML = `β = <b>${beta.toFixed(2)}</b> · a = <b>${a.toFixed(1)}</b> · KL(p<sup>⋆</sup> ‖ p<sub>data</sub>) = <b>${E.tg.kl.toFixed(3)}</b> · control cost E∫½|c<sub>t</sub>|²dt + start tilt = <b>${E.cost.toFixed(3)}</b> + <b>${E.startKL.toFixed(4)}</b><br>` +
      `spread of h at t = ${T_S}: max/min − 1 = <b>${E.spread.toFixed(2)}</b> · W<sub>1</sub> to p<sup>⋆</sup>: controlled <b>${E.w1.toFixed(3)}</b> · base to p<sub>data</sub> <b>${w1Data(B.end).toFixed(3)}</b> (time discretization)`;
  }
  return { draw };
}

// ---------- Step 7: guidance and SMC against the exact control ----------
const NS7 = [1, 2, 4, 8, 16, 32, 64];
let GUID = null;
const guidRun = () => (GUID ??= (() => { const run = simulate((x, s) => dloghGuid(x, s, BETA_DEF, A_DEF)), tg = tilted(BETA_DEF, A_DEF); return { ...run, w1: w1To(run.end, tg.Q), sd: sd(run.end) }; })());
const smcRun = memo((N) => { const end = smc(N, BETA_DEF, A_DEF); return { end, w1: w1To(end, tilted(BETA_DEF, A_DEF).Q), sd: sd(end) }; }, 8);
function estimate(viz) {
  const c = makeCanvas(16 / 8, { maxH: 0.66 }), ro = readout();
  let N = 8;
  viz.append(c.cv, controls(slider('N', { min: 0, max: NS7.length - 1, step: 1, value: NS7.indexOf(N), map: (v) => NS7[v], fmt: (v) => String(v) }, (v) => { N = v; draw(); })), ro,
    note(`β = ${BETA_DEF} and a = ${A_DEF}, the defaults of step 6. Exact control and guidance: the sampler of step 6 with 1000 particles and the same noise; guidance uses the plain start and replaces log h<sub>t</sub> by r(x̂<sub>1</sub>)/β, differentiated through the exact denoiser. SMC: ${NRUN} independent runs of N particles on ${NK6 / 2} steps, weighted by the ratio of successive guidance estimates of h and by e<sup>r/β</sup> at the end, resampled when the effective sample size falls below N/2; each run returns one particle drawn by its weight. These histograms are Monte Carlo estimates. Bold curve: p<sup>⋆</sup>; thin: p<sub>data</sub>.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), E = exactRun(`${BETA_DEF.toFixed(3)}|${A_DEF.toFixed(2)}`, BETA_DEF, A_DEF), G = guidRun(), M = smcRun(N), tg = E.tg;
    const top = 34, bot = h - 34, gap = 22, L0 = 40, pw = (w - L0 - 12 - 2 * gap) / 3;
    const Yx = (x) => top + (bot - top) * (HI - x) / (HI - LO), pm = Math.max(...XS.map(pdata1), ...XS.map(tg.pdf)) * 1.1;
    const cols = [[E, 'exact control', hlc(['u', 'h'], pal, pal.acc)], [G, 'guidance', hlc('hx1', pal, pal.ramp[3])], [M, `SMC, N = ${N}`, hlc('N', pal, pal.ramp[1])]];
    cols.forEach(([run, name, col], k) => {
      const L = L0 + k * (pw + gap);
      frame(ctx, pal, L, top, pw, bot - top);
      clip(ctx, L, top, pw, bot - top);
      sideBars(ctx, run.end, Yx, L + 1, pw - 4, pm, col, 0.45);
      sideCurve(ctx, Yx, L + 1, pw - 4, pm, pdata1, pal.muted, 1);
      sideCurve(ctx, Yx, L + 1, pw - 4, pm, tg.pdf, hlc('pstar', pal, pal.fg), isHL('pstar') ? 3 : 2);
      ctx.restore();
      hdash(ctx, hlc(['a', 'r'], pal, pal.acc2), L, L + pw, Yx(A_DEF), 1.2, [4, 4]);
      label(ctx, name, L + 4, top - 12, col, 13);
      label(ctx, `W₁ ${run.w1.toFixed(3)}`, L + pw - 6, bot - 8, pal.muted, 11.5, 'right');
      if (k === 0) [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 8, Yx(v) + 4, pal.muted, 11, 'right'));
    });
    let psd = 0; for (const q of tg.comp) psd += q.w * (q.v + q.m * q.m); let pm1 = 0; for (const q of tg.comp) pm1 += q.w * q.m; psd = Math.sqrt(psd - pm1 * pm1);
    ro.innerHTML = `standard deviation of the end points: p<sup>⋆</sup> <b>${psd.toFixed(2)}</b> · exact control <b>${E.sd.toFixed(2)}</b> · guidance <b>${G.sd.toFixed(2)}</b> · SMC <b>${M.sd.toFixed(2)}</b><br>` +
      `control cost: exact <b>${E.cost.toFixed(2)}</b> (KL(p<sup>⋆</sup> ‖ p<sub>data</sub>) = ${tg.kl.toFixed(2)}) · guidance <b>${G.cost.toFixed(2)}</b> · SMC uses no control, it spends <b>${N}</b> sampler${N > 1 ? 's' : ''} per sample`;
  }
  return { draw };
}

export const SCENES = { couplings, field, energy, rectify, bridge, control, estimate };
