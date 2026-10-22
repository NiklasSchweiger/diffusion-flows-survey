// Live scenes of the §2.4 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, arrow, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import {
  MIX1, MIX2, gmm2, W2, LIN, SCHEDULES, T_MIN, dens1, pt1, NR, TR, reverse1, quality2, LL_EXACT, pt2, PAIRS2, posteriorSamples,
  uExact1, idParts, NC, NP, swirlVel, swirlPaths, meanLogPt, exactLogPt, uModel, makeTrainer, LOSS_FLOOR, evalLoss, labelsAt,
  sampleField, meanLogP1, LL1_EXACT, rankCorr, euler2, straightness, DATA_MEAN2,
} from './sim.js';

// ---------- Helpers ----------
const LO = -4, HI = 4, XS = Array.from({ length: 321 }, (_, i) => LO + (HI - LO) * i / 320);
const DATA_MEAN1 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * m, 0);
const DATA_VAR1 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * (MIX1.sd[k] ** 2 + m * m), 0) - DATA_MEAN1 ** 2;
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
// Mode a quantile belongs to, from the cumulative mixture weights.
const modeOfQ = (q) => (q < MIX1.w[0] ? 0 : q < MIX1.w[0] + MIX1.w[1] ? 1 : 2);
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function cross(ctx, X, Y, r, color, lw = 2) {
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X - r, Y + r); ctx.lineTo(X + r, Y - r); ctx.stroke();
}
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_/g, '')).width + 12, x0 = align === 'right' ? x - w + 6 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const hline = (ctx, pal, x0, x1, y) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };
function vdash(ctx, color, x, y0, y1) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
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
const fmt2 = ([a, b]) => `(${a.toFixed(2)}, ${b.toFixed(2)})`;
const tSlider = (value, on, min = 0, max = 1) => slider('t', { min, max, step: 0.01, value }, on);
const mm = (q) => q.mass.map((x) => x.toFixed(2)).join(' / ');

// World box [-bx, bx] x [-by, by] fitted into a rectangle of the canvas.
function view(x0, y0, w, h, bx = 3.4, by = 2.6) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y], wx: (X, Y) => [(X - cx) / s, (cy - Y) / s] };
}
const track = (c, onMove, mkView = () => view(0, 0, c.w, c.h)) => c.cv.addEventListener('pointermove', (e) => {
  const r = c.cv.getBoundingClientRect(); onMove(mkView().wx(e.clientX - r.left, e.clientY - r.top));
});
// Heat map of p_t on the 2D mixture, cached by size, time and theme. Alpha follows p^0.7.
const heatCache = new Map();
function heatT(ctx, w, h, v, t, alphaMul = 0.85) {
  const k = `${w}x${h}|${t.toFixed(2)}|${live.theme}`;
  if (!heatCache.has(k)) {
    if (heatCache.size > 60) heatCache.clear();
    const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res), off = document.createElement('canvas'); off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H);
    const pmax = Math.max(...MIX2.means.map(([mx, my]) => pt2(t * mx, t * my, t))) * 1.05;
    const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.5 : 0.32;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const [x, y] = v.wx(i * res, j * res), o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(Math.min(1, pt2(x, y, t) / pmax), 0.7));
    }
    g.putImageData(im, 0, 0);
    heatCache.set(k, off);
  }
  ctx.globalAlpha = alphaMul; ctx.drawImage(heatCache.get(k), 0, 0, w, h); ctx.globalAlpha = 1;
}
// Small cache for expensive per-parameter computations.
function memo(fn, size = 8) {
  const m = new Map();
  return (key) => { if (!m.has(key)) { if (m.size >= size) m.delete(m.keys().next().value); m.set(key, fn(key)); } return m.get(key); };
}
const TRAILS = Array.from({ length: 40 }, (_, i) => 12 + 25 * i);   // the 40 of 1000 particles drawn with a trail

// ---------- Step 1: the continuity equation and swirls that leave p_t unchanged ----------
const swirl = memo(swirlPaths, 9);
const exactLL = memo(exactLogPt, 40);
const ARROWS = (() => { const a = []; for (let i = -6; i <= 6; i++) for (let j = -4; j <= 4; j++) a.push([i * 0.5, j * 0.55]); return a; })();
function continuity(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let cc = 2, t = 0.5;
  const r = runner({ step: () => { t = Math.min(1, t + 0.01); tSl.set(t); }, done: () => t >= 1, draw: () => {}, interval: 40 });
  r.playing = false;
  const tSl = tSlider(t, (v) => { t = v; draw(); });
  viz.append(c.cv, controls(
    slider('c', { min: -4, max: 4, step: 1, value: cc, fmt: (v) => String(v) }, (v) => { cc = v; draw(); }), tSl,
    button('Play from t = 0', () => { t = 0; tSl.set(0); r.playing = true; r.kick(); })), ro,
  note(`1000 points from N(0, I), moved with ${NC} RK2 steps along the exact u<sub>t</sub> + w<sub>t</sub> of the 2D mixture of §2.2. The heat map is the exact p<sub>t</sub>; the grey arrows are u<sub>t</sub> + w<sub>t</sub>, compressed in length. 40 points carry a trail from t = 0.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h, 4.6, 2.6), P = swirl(cc), j = Math.round(t * NC);
    heatT(ctx, w, h, v, t, isHL('pt', 'dtp') ? 1 : 0.85);
    // Velocity field u + w at the current t.
    const tv = Math.min(t, 0.995), ac = hlc(['vel', 'w', 'c', 'J'], pal, pal.muted);
    ctx.globalAlpha = isHL('vel', 'w', 'c', 'J') ? 0.9 : 0.45;
    for (const [x, y] of ARROWS) {
      const [ux, uy] = swirlVel(x, y, tv, cc), n = Math.hypot(ux, uy), L = 0.32 * Math.tanh(n / 2) / Math.max(n, 1e-9);
      const [X0, Y0] = v.px(x, y), [X1, Y1] = v.px(x + L * ux, y + L * uy);
      arrow(ctx, X0, Y0, X1, Y1, ac, 1.2, 5);
    }
    ctx.globalAlpha = 1;
    clip(ctx, 0, 0, w, h);
    ctx.strokeStyle = hlc(['x', 'w', 'c'], pal, pal.fg); ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
    for (const i of TRAILS) {
      ctx.beginPath(); ctx.moveTo(...v.px(P[0][2 * i], P[0][2 * i + 1]));
      for (let k = 2; k <= j; k += 2) ctx.lineTo(...v.px(P[k][2 * i], P[k][2 * i + 1]));
      ctx.lineTo(...v.px(P[j][2 * i], P[j][2 * i + 1])); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = hlc('x', pal, pal.fg);
    for (let i = 0; i < NP; i++) { const [X, Y] = v.px(P[j][2 * i], P[j][2 * i + 1]); ctx.fillRect(X - 1.2, Y - 1.2, 2.4, 2.4); }
    ctx.restore();
    frame(ctx, pal, 0, 0, w, h);
    tag(ctx, pal, cc === 0 ? 'c = 0: the marginal velocity of step 3' : `c = ${cc}: a swirl is added`, 16, 26, hlc('c', pal, pal.fg), 14);
    tag(ctx, pal, `t = ${(j / NC).toFixed(2)}`, w - 14, 26, hlc('t', pal, pal.muted), 13.5, 'right');
    const ll = meanLogPt(P[j], j / NC), ex = exactLL((j / NC).toFixed(2));
    let line2 = '';
    if (j === NC) line2 = `<br>at t = 1: mass per mode <b>${mm(quality2(P[NC]))}</b> (data ${W2.map((x) => x.toFixed(2)).join(' / ')})`;
    ro.innerHTML = `c = <b>${cc}</b> · t = <b>${(j / NC).toFixed(2)}</b> · mean log p<sub>t</sub> of the 1000 points <b>${ll.toFixed(3)}</b> · of 3000 exact samples of p<sub>t</sub> <b>${ex.toFixed(3)}</b>` + line2;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 2: the conditional paths ----------
const NPAIR = 120, HL_PAIR = 7;
const CROSSINGS = (() => {
  const seg = (i) => [PAIRS2.x0[2 * i], PAIRS2.x0[2 * i + 1], PAIRS2.x1[2 * i], PAIRS2.x1[2 * i + 1]];
  const side = (ax, ay, bx, by, cx, cy) => Math.sign((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
  let n = 0;
  for (let i = 0; i < NPAIR; i++) for (let j = i + 1; j < NPAIR; j++) {
    const [a, b, c, d] = seg(i), [e, f, g, hh] = seg(j);
    if (side(a, b, c, d, e, f) !== side(a, b, c, d, g, hh) && side(e, f, g, hh, a, b) !== side(e, f, g, hh, c, d)) n++;
  }
  return n;
})();
function conditional(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let t = 0.4;
  const r = runner({ step: () => { t = t >= 1 ? 0 : Math.min(1, t + 0.008); tSl.set(t); }, done: () => false, draw: () => {}, interval: 30 });
  r.playing = false; r.play.textContent = 'Play';
  const tSl = tSlider(t, (v) => { t = v; draw(); });
  viz.append(c.cv, controls(tSl, r.play), ro,
    note(`${NPAIR} pairs (x<sub>0</sub>, x<sub>1</sub>) drawn independently, x<sub>0</sub> from N(0, I) and x<sub>1</sub> from the 2D mixture; lines are coloured by the component of x<sub>1</sub>. The arrow on the highlighted pair is its velocity x<sub>1</sub> − x<sub>0</sub>, drawn at a third of its length.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h, 4.6, 2.6);
    heatT(ctx, w, h, v, 1, 0.45);
    clip(ctx, 0, 0, w, h);
    const P = (i, s) => v.px(s * PAIRS2.x1[2 * i] + (1 - s) * PAIRS2.x0[2 * i], s * PAIRS2.x1[2 * i + 1] + (1 - s) * PAIRS2.x0[2 * i + 1]);
    ctx.lineWidth = 1; ctx.globalAlpha = isHL('cv', 'x0', 'x1') ? 0.6 : 0.3;
    for (let i = 0; i < NPAIR; i++) { ctx.beginPath(); ctx.moveTo(...P(i, 0)); ctx.lineTo(...P(i, 1)); ctx.strokeStyle = hlc('cv', pal, modeColor(pal, PAIRS2.k[i])); ctx.stroke(); }
    ctx.globalAlpha = 1;
    for (let i = 0; i < NPAIR; i++) { const [X, Y] = P(i, t); dot(ctx, X, Y, isHL('x') ? 3.6 : 2.6, hlc('x', pal, modeColor(pal, PAIRS2.k[i]))); }
    // The highlighted pair.
    const i = HL_PAIR, [A, B] = P(i, 0), [C, D] = P(i, 1), [X, Y] = P(i, t);
    ctx.beginPath(); ctx.moveTo(A, B); ctx.lineTo(C, D); ctx.strokeStyle = pal.fg; ctx.lineWidth = 2; ctx.stroke();
    ring(ctx, A, B, isHL('x0') ? 8 : 6, hlc('x0', pal, pal.fg), 2);
    dot(ctx, C, D, isHL('x1') ? 8 : 6, hlc('x1', pal, pal.fg));
    const vx = PAIRS2.x1[2 * i] - PAIRS2.x0[2 * i], vy = PAIRS2.x1[2 * i + 1] - PAIRS2.x0[2 * i + 1];
    const xt = [t * PAIRS2.x1[2 * i] + (1 - t) * PAIRS2.x0[2 * i], t * PAIRS2.x1[2 * i + 1] + (1 - t) * PAIRS2.x0[2 * i + 1]];
    const [E, F] = v.px(xt[0] + vx / 3, xt[1] + vy / 3);
    arrow(ctx, X, Y, E, F, hlc(['cv', 'adot', 'sdot'], pal, pal.acc), 3, 11);
    dot(ctx, X, Y, isHL('x') ? 7 : 5.5, hlc('x', pal, pal.fg));
    tag(ctx, pal, 'x_0', A - 12, B + 4, hlc('x0', pal, pal.fg), 14, 'right');
    tag(ctx, pal, 'x_1', C + 12, D + 4, hlc('x1', pal, pal.fg), 14);
    ctx.restore();
    frame(ctx, pal, 0, 0, w, h);
    tag(ctx, pal, `t = ${t.toFixed(2)}`, w - 14, 26, hlc('t', pal, pal.muted), 13.5, 'right');
    tag(ctx, pal, 'heat map: the data density p_1', 16, h - 14, pal.muted, 12.5);
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · highlighted pair: x<sub>t</sub> = <b>${fmt2(xt)}</b>, velocity x<sub>1</sub> − x<sub>0</sub> = <b>${fmt2([vx, vy])}</b> at every t<br>` +
      `pairs of lines among the ${NPAIR} that cross: <b>${CROSSINGS}</b> of ${NPAIR * (NPAIR - 1) / 2}`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 3: the marginal velocity as a posterior average ----------
function marginal(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let t = 0.4, probe = [-0.4, 0.3];
  const mk = () => view(0, 0, c.w, c.h, 4, 2.5);
  track(c, (p) => { probe = p; draw(); }, mk);
  viz.append(c.cv, controls(tSlider(t, (x) => { t = x; draw(); }, 0, 0.95)), ro,
    note('Move the cursor to place the probe x, and click to pin it there. Coloured dots: 60 posterior draws of x<sub>1</sub> given x<sub>t</sub> = x. Thin arrows: their conditional velocities, scaled by σ<sub>t</sub> so that each ends at its dot. The heat map is p<sub>t</sub>; the cross is the data mean.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = mk(), s = 1 - t;
    heatT(ctx, w, h, v, t, isHL('pt') ? 1 : 0.75);
    const [X, Y] = v.px(...probe), { samples, resp } = posteriorSamples(...probe, t);
    let mx = 0, my = 0;
    const cvHL = isHL('cv', 'x0');
    samples.forEach(({ k, x, y }) => {
      // Conditional velocity x1 - x0 with x0 = (x - t x1) / (1 - t); probe + sigma_t (x1 - x0) = x1.
      const vx = (x - probe[0]) / s, vy = (y - probe[1]) / s; mx += vx / samples.length; my += vy / samples.length;
      const [Xe, Ye] = v.px(x, y);
      ctx.globalAlpha = 0.35; arrow(ctx, X, Y, Xe, Ye, cvHL ? pal.mark : modeColor(pal, k), cvHL ? 1.6 : 1.1, 5); ctx.globalAlpha = 1;
      dot(ctx, Xe, Ye, isHL('x1') ? 4 : 3, hlc('x1', pal, modeColor(pal, k)));
    });
    const u = gmm2.fields(...probe, t).u;
    const [Xm, Ym] = v.px(probe[0] + s * mx, probe[1] + s * my), [Xu, Yu] = v.px(probe[0] + s * u[0], probe[1] + s * u[1]);
    arrow(ctx, X, Y, Xm, Ym, hlc('cv', pal, pal.acc), 3.4, 12);
    ctx.setLineDash([6, 4]); arrow(ctx, X, Y, Xu, Yu, hlc('vel', pal, pal.fg), 2, 10); ctx.setLineDash([]);
    cross(ctx, ...v.px(...DATA_MEAN2), 6, pal.muted, 1.6);
    dot(ctx, X, Y, isHL('x') ? 7 : 5.5, hlc('x', pal, pal.fg));
    tag(ctx, pal, 'x', X - 10, Y - 10, hlc('x', pal, pal.fg), 14, 'right');
    frame(ctx, pal, 0, 0, w, h);
    tag(ctx, pal, `t = ${t.toFixed(2)}`, w - 14, 26, hlc('t', pal, pal.muted), 13.5, 'right');
    tag(ctx, pal, 'bold: mean of the thin arrows · dashed: exact u_t(x) · both scaled by σ_t', 16, h - 14, pal.muted, 12.5);
    ro.innerHTML = `x = <b>${fmt2(probe)}</b> · t = <b>${t.toFixed(2)}</b> · posterior weight of the components: <b>${resp.map((q) => q.toFixed(2)).join(', ')}</b><br>` +
      `mean of 60 conditional velocities = <b>${fmt2([mx, my])}</b> · exact u<sub>t</sub>(x) = <b>${fmt2(u)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: flow matching on the 1D mixture ----------
const MAX_STEPS = 3000, PER = 20;
function fm(viz) {
  const c = makeCanvas(16 / 12, { maxH: 0.78 }), ro = readout();
  let tr = makeTrainer(), t = 0.5, hist = [], samp = null;
  const record = () => { const e = evalLoss(tr.th); hist.push([tr.step, e.loss, e.gap]); samp = null; };
  record();
  const r = runner({ step: () => { tr.train(PER); record(); }, done: () => tr.step >= MAX_STEPS, draw: () => draw(), interval: 30 });
  r.playing = false;
  const trainBtn = button('Train', () => { r.playing = !r.playing; trainBtn.textContent = r.playing ? 'Pause' : 'Train'; r.kick(); });
  viz.append(c.cv, controls(tSlider(t, (v) => { t = v; draw(); }, 0.05, 0.95), trainBtn,
    button('Reset', () => { r.playing = false; trainBtn.textContent = 'Train'; tr = makeTrainer(); hist = []; record(); draw(); })), ro,
  note(`The model is linear in its 297 parameters (bumps in x times hat functions in t), so every curve here is cheap to evaluate. Adam, minibatches of 128, ${PER} steps per frame up to ${MAX_STEPS}. The loss is measured on 4000 fixed triples. Samples: 1000 stratified noise draws, 40 Heun steps with the current model.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), L = 52, R = w - 18;
    const top1 = 28, bot1 = h * 0.42, top2 = h * 0.5, bot2 = h * 0.66, top3 = h * 0.75, bot3 = h - 30;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    // Upper panel: labels, exact marginal velocity, model.
    const VM = 7, mid = (top1 + bot1) / 2, Y1 = (u) => mid - (bot1 - top1) / 2 * u / VM;
    hline(ctx, pal, L, R, mid);
    clip(ctx, L, top1 - 4, R - L, bot1 - top1 + 8);
    ctx.fillStyle = hlc(['cv', 'x0', 'x1'], pal, pal.muted); ctx.globalAlpha = isHL('cv', 'x0', 'x1') ? 0.8 : 0.45;
    for (const [x, y] of labelsAt(t)) ctx.fillRect(X(x) - 1.5, Y1(y) - 1.5, 3, 3);
    ctx.globalAlpha = 1;
    curve(ctx, XS, X, Y1, (x) => uExact1(x, t), hlc('vel', pal, pal.fg), isHL('vel') ? 3 : 1.8, [6, 5]);
    curve(ctx, XS, X, Y1, (x) => uModel(tr.th, x, t), hlc(['utheta', 'theta'], pal, pal.acc), isHL('utheta', 'theta') ? 3.6 : 2.6);
    ctx.restore();
    [-VM + 1, 0, VM - 1].forEach((u) => label(ctx, String(u), L - 8, Y1(u) + 4, pal.muted, 11, 'right'));
    label(ctx, `labels x_1 − x_0 at t = ${t.toFixed(2)} (dots)`, L + 4, top1 - 10, hlc('cv', pal, pal.muted), 13);
    label(ctx, 'bold: model u_θ · dashed: exact u_t', R, top1 - 10, hlc(['utheta', 'vel'], pal, pal.fg), 13, 'right');
    label(ctx, 'x_t', R, bot1 + 14, hlc('x', pal, pal.muted), 12.5, 'right');
    // Middle panel: loss against step.
    const lmax = Math.max(hist[0][1], LOSS_FLOOR * 1.6), lmin = LOSS_FLOOR * 0.8;
    const XS2 = (k) => L + (R - L) * k / MAX_STEPS, Y2 = (l) => bot2 - (bot2 - top2) * (Math.min(l, lmax) - lmin) / (lmax - lmin);
    hline(ctx, pal, L, R, bot2);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = pal.muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, Y2(LOSS_FLOOR)); ctx.lineTo(R, Y2(LOSS_FLOOR)); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); hist.forEach(([k, l], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, XS2(k), Y2(l)));
    ctx.strokeStyle = hlc(['LFM', 'Et'], pal, pal.acc); ctx.lineWidth = isHL('LFM', 'Et') ? 3 : 2; ctx.stroke();
    label(ctx, f2(lmax), L - 8, top2 + 8, pal.muted, 11, 'right'); label(ctx, f2(LOSS_FLOOR), L - 8, Y2(LOSS_FLOOR) + 4, pal.muted, 11, 'right');
    label(ctx, 'loss L_{FM}(θ) against training step; dashed: the floor', L + 4, top2 - 8, hlc('LFM', pal, pal.fg), 13);
    label(ctx, String(MAX_STEPS), R, bot2 + 14, pal.muted, 11, 'right');
    // Lower panel: samples of the current model at t = 1 against the data density.
    if (!samp) samp = sampleField((x, s) => uModel(tr.th, x, s));
    const NBIN = 64, bw = (HI - LO) / NBIN, cnt = new Float64Array(NBIN);
    for (const x of samp) { const b = Math.floor((x - LO) / bw); if (b >= 0 && b < NBIN) cnt[b]++; }
    const hst = Array.from(cnt, (q) => q / (samp.length * bw)), ex = XS.map((x) => pt1(x, 1)), pm = Math.max(...ex, ...hst) * 1.08;
    const Y3 = (p) => bot3 - (bot3 - top3) * p / pm;
    ctx.fillStyle = hlc('x', pal, pal.fg); ctx.globalAlpha = 0.3;
    hst.forEach((q, b) => ctx.fillRect(X(LO + b * bw) + 0.5, Y3(q), X(LO + (b + 1) * bw) - X(LO + b * bw) - 1, bot3 - Y3(q)));
    ctx.globalAlpha = 1;
    curve(ctx, XS, X, Y3, (x, i) => ex[i], hlc('pt', pal, pal.acc2), 2);
    hline(ctx, pal, L, R, bot3);
    label(ctx, 'samples of the model at t = 1 (bars) and the data density p_1 (line)', L + 4, top3 - 8, hlc('pt', pal, pal.fg), 13);
    [-3, 0, 3].forEach((x) => label(ctx, String(x), X(x), bot3 + 16, pal.muted, 11, 'center'));
    const [k, l, gap] = hist[hist.length - 1];
    ro.innerHTML = `step <b>${k}</b> · loss <b>${l.toFixed(3)}</b> · floor (no function of x<sub>t</sub> does better) <b>${LOSS_FLOOR.toFixed(3)}</b> · E‖u<sub>θ</sub> − u<sub>t</sub>‖² = loss − floor = <b>${gap.toFixed(3)}</b><br>` +
      `mean log p<sub>1</sub> of the model samples <b>${meanLogP1(samp).toFixed(3)}</b> · of exact data samples <b>${LL1_EXACT.toFixed(3)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: the velocity-score identity ----------
const DOTS = Array.from({ length: 41 }, (_, i) => -3.6 + 7.2 * i / 40);
const PATHS = { linear: 'Linear path', cosine: 'Cosine path' };
function identity(viz) {
  const c = makeCanvas(16 / 11, { maxH: 0.72 }), ro = readout();
  let key = 'linear', t = 0.5;
  const btns = Object.keys(PATHS).map((k) => { const b = button(PATHS[k], () => { key = k; btns.forEach((x) => x.classList.toggle('on', x === b)); draw(); }, k === key ? 'on' : ''); return b; });
  viz.append(c.cv, controls(...btns, tSlider(t, (v) => { t = v; draw(); }, 0.05, 0.95)), ro,
    note('All curves are exact for the 1D mixture. The bold curve is computed from the posterior means E[x<sub>1</sub> | x<sub>t</sub>] and E[x<sub>0</sub> | x<sub>t</sub>], the two parts and the dots from the score. The vertical scale of the upper panel adapts to t; the lower panel is clipped at 6.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), S = SCHEDULES[key], L = 52, R = w - 18;
    const top1 = 28, bot1 = h * 0.56, top2 = h * 0.66, bot2 = h - 32;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    const P = XS.map((x) => idParts(x, t, S)), U = XS.map((x) => uExact1(x, t, S));
    const inner = XS.map((x, i) => i).filter((i) => Math.abs(XS[i]) < 3.4);
    const M = Math.max(...inner.map((i) => Math.max(Math.abs(U[i]), Math.abs(P[i].lin), Math.abs(P[i].sc)))) * 1.1;
    const mid = (top1 + bot1) / 2, Y1 = (u) => mid - (bot1 - top1) / 2 * Math.max(-1.05, Math.min(1.05, u / M));
    const D = XS.map((x) => pt1(x, t, S)), Dm = Math.max(...D);
    shade(ctx, XS, X, (p) => bot1 - (bot1 - top1) * 0.45 * p / Dm, (x, i) => D[i], bot1, hlc('pt', pal, pal.acc2), isHL('pt') ? 0.35 : 0.12);
    hline(ctx, pal, L, R, mid);
    clip(ctx, L, top1 - 4, R - L, bot1 - top1 + 8);
    curve(ctx, XS, X, Y1, (x, i) => P[i].lin, hlc(['alpha', 'adot'], pal, pal.ramp[0]), isHL('alpha', 'adot') ? 3.2 : 2);
    curve(ctx, XS, X, Y1, (x, i) => P[i].sc, hlc(['glp', 'lamdot', 'sig'], pal, pal.ramp[3]), isHL('glp', 'lamdot', 'sig') ? 3.2 : 2);
    curve(ctx, XS, X, Y1, (x, i) => U[i], hlc(['vel', 'hx1', 'hx0'], pal, pal.fg), isHL('vel', 'hx1', 'hx0') ? 4 : 3);
    ctx.restore();
    let res = 0;
    DOTS.forEach((x) => { const q = idParts(x, t, S), s = q.lin + q.sc; res = Math.max(res, Math.abs(s - uExact1(x, t, S))); dot(ctx, X(x), Y1(s), 3.4, pal.acc); });
    label(ctx, '0', L - 8, mid + 4, pal.muted, 11, 'right'); label(ctx, f2(M / 1.1), L - 8, Y1(M / 1.1) + 4, pal.muted, 11, 'right');
    label(ctx, '(α̇_t/α_t) x', L + 4, top1 - 10, hlc(['alpha', 'adot'], pal, pal.ramp[0]), 13);
    label(ctx, '½ σ_t² λ̇_t ∇log p_t(x)', L + 110, top1 - 10, hlc(['glp', 'lamdot'], pal, pal.ramp[3]), 13);
    label(ctx, 'bold: u_t from the denoisers · dots: the sum of the two parts', R, top1 - 10, hlc('vel', pal, pal.fg), 13, 'right');
    label(ctx, 'x', R, bot1 + 14, hlc('x', pal, pal.muted), 12.5, 'right');
    // Lower panel: the coefficients against t.
    const TT = Array.from({ length: 193 }, (_, i) => 0.02 + 0.96 * i / 192), CM = 6;
    const X2 = (s) => L + (R - L) * s, Y2 = (v) => bot2 - (bot2 - top2) * Math.min(v, CM * 1.04) / CM;
    hline(ctx, pal, L, R, bot2);
    clip(ctx, L, top2 - 6, R - L, bot2 - top2 + 8);
    curve(ctx, TT, X2, Y2, (s) => S.dalpha(s) / S.alpha(s), hlc(['alpha', 'adot'], pal, pal.ramp[0]), 2);
    curve(ctx, TT, X2, Y2, (s) => 0.5 * S.g2(s), hlc(['lamdot', 'sig'], pal, pal.ramp[3]), 2);
    ctx.restore();
    vdash(ctx, hlc('t', pal, pal.fg), X2(t), top2, bot2);
    const q = idParts(0, t, S);
    dot(ctx, X2(t), Y2(q.ca), 4, pal.ramp[0]); dot(ctx, X2(t), Y2(q.cs), 4, pal.ramp[3]);
    [0, 2, 4, 6].forEach((v) => label(ctx, String(v), L - 8, Y2(v) + 4, pal.muted, 11, 'right'));
    [0, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), X2(s), bot2 + 16, pal.muted, 11, 'center'));
    label(ctx, 'coefficients against t: α̇_t/α_t and ½ σ_t² λ̇_t', L + 4, top2 - 10, pal.fg, 13);
    label(ctx, 't', R, bot2 + 28, hlc('t', pal, pal.muted), 12.5, 'right');
    ro.innerHTML = `${PATHS[key]} · t = <b>${t.toFixed(2)}</b> · α̇<sub>t</sub>/α<sub>t</sub> = <b>${f2(q.ca)}</b> · ½σ<sub>t</sub>²λ̇<sub>t</sub> = <b>${f2(q.cs)}</b><br>` +
      `largest gap between the sum of the two parts and u<sub>t</sub> at the 41 dots: <b>${res.toExponential(1)}</b>`;
  }
  return { draw };
}

// ---------- Step 6: the family of samplers ----------
const rev = memo(reverse1, 7);
const SHOW = Array.from({ length: 40 }, (_, k) => 12 + 25 * k);
const stripCache = new Map();
function strip(times) {
  const k = live.theme;
  if (stripCache.has(k)) return stripCache.get(k);
  const W = times.length, H = 161, off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H);
  const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.55 : 0.34;
  for (let i = 0; i < W; i++) {
    const col = Array.from({ length: H }, (_, j) => pt1(HI - (HI - LO) * j / (H - 1), times[i])), m = Math.max(...col);
    for (let j = 0; j < H; j++) {
      const o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(col[j] / m, 0.7));
    }
  }
  g.putImageData(im, 0, 0);
  stripCache.set(k, off);
  return off;
}
function family(viz) {
  const c = makeCanvas(16 / 8.5), ro = readout();
  let eta = 0;
  viz.append(c.cv, controls(slider('eta', { min: 0, max: 1.5, step: 0.25, value: eta }, (v) => { eta = v; draw(); })), ro,
    note(`1000 particles from stratified quantiles of the exact p<sub>${T_MIN}</sub>, ${NR} Euler–Maruyama steps with the exact velocity and score of the 1D mixture. The noise draws are fixed, so changing η changes only the equation. 40 paths are drawn; the right panel shows all 1000.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), Rv = rev(eta.toFixed(2) * 1);
    const L = 46, R = Math.round(w * 0.62), top = 28, bot = h - 36;
    const X = (s) => L + (R - L) * (s - T_MIN) / (1 - T_MIN), Y = (y) => top + (bot - top) * (HI - y) / (HI - LO);
    const color = (i) => modeColor(pal, modeOfQ((i + 0.5) / Rv.n));
    ctx.globalAlpha = isHL('pt') ? 1 : 0.85;
    ctx.drawImage(strip(TR.filter((_, i) => i % 2 === 0)), L, top, R - L, bot - top); ctx.globalAlpha = 1;
    clip(ctx, L, top, R - L, bot - top);
    ctx.globalAlpha = isHL('x', 'eta', 'eps', 'W') ? 0.9 : 0.6; ctx.lineWidth = 1.1;
    for (const i of SHOW) {
      ctx.beginPath(); for (let k = 0; k <= NR; k += 2) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(TR[k]), Y(Rv.x[i][k]));
      ctx.strokeStyle = hlc(['x', 'eta', 'eps', 'W'], pal, color(i)); ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.restore();
    hline(ctx, pal, L, R, bot);
    [-3, 0, 3].forEach((v) => label(ctx, String(v), L - 8, Y(v) + 4, pal.muted, 11, 'right'));
    label(ctx, 'x', L - 8, top - 10, hlc('x', pal, pal.muted), 13, 'right');
    [0.03, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), X(s), bot + 16, pal.muted, 11, 'center'));
    label(ctx, 't', R, bot + 30, hlc('t', pal, pal.muted), 12.5, 'right');
    tag(ctx, pal, eta === 0 ? 'η = 0: the ODE' : eta === 1 ? 'η = 1: the reverse SDE' : `η = ${eta.toFixed(2)}`, L + 10, top + 20, hlc('eta', pal, pal.fg), 13.5);
    // Right panel: end point against starting point.
    const sL = R + 70, sR = w - 16, sTop = top, sBot = bot, A0 = -3, A1 = 3;
    const SX = (x) => sL + (sR - sL) * (x - A0) / (A1 - A0), SY = (y) => sTop + (sBot - sTop) * (HI - y) / (HI - LO);
    frame(ctx, pal, sL, sTop, sR - sL, sBot - sTop);
    clip(ctx, sL, sTop, sR - sL, sBot - sTop);
    for (let i = 0; i < Rv.n; i++) { ctx.fillStyle = color(i); ctx.globalAlpha = 0.7; ctx.fillRect(SX(Rv.x[i][0]) - 1.3, SY(Rv.x[i][NR]) - 1.3, 2.6, 2.6); }
    ctx.globalAlpha = 1; ctx.restore();
    [-2, 0, 2].forEach((v) => label(ctx, String(v), SX(v), sBot + 16, pal.muted, 11, 'center'));
    [-3, 0, 3].forEach((v) => label(ctx, String(v), sL - 8, SY(v) + 4, pal.muted, 11, 'right'));
    label(ctx, 'end at t = 1 against start', sL, sTop - 10, pal.fg, 13);
    label(ctx, `start at t = ${T_MIN}`, sR, sBot + 30, pal.muted, 12.5, 'right');
    const end = Rv.x.map((a) => a[NR]); let m = 0; for (const v of end) m += v / end.length; let s2 = 0; for (const v of end) s2 += (v - m) ** 2 / end.length;
    ro.innerHTML = `η = <b>${eta.toFixed(2)}</b> · rank correlation of start and end: <b>${rankCorr(Rv).toFixed(2)}</b><br>` +
      `1000 particles at t = 1: mean <b>${m.toFixed(2)}</b>, variance <b>${s2.toFixed(3)}</b> · exact p<sub>data</sub>: mean <b>${DATA_MEAN1.toFixed(2)}</b>, variance <b>${DATA_VAR1.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 7: few-step Euler on curved marginal paths ----------
const REF = euler2(200), REF_S = straightness(REF);
const eul = memo((K) => { const p = euler2(K); return { path: p, q: quality2(p[K]) }; }, 24);
function straight(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let K = 4;
  viz.append(c.cv, controls(slider('K', { min: 1, max: 20, step: 1, value: K, fmt: (v) => String(v) }, (v) => { K = v; draw(); })), ro,
    note('Exact marginal velocity of the 2D mixture. The same 1000 starting points as in step 1; 40 of them are drawn with their accurate path (200 steps, faint), its chord (dashed) and the K-step Euler polygon (bold). The small dots are the final points of all 1000 Euler runs, the cross is the data mean, the heat map is p<sub>data</sub>.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h, 4.6, 2.6), E = eul(K), P = E.path;
    heatT(ctx, w, h, v, 1, isHL('pt') ? 0.9 : 0.6);
    clip(ctx, 0, 0, w, h);
    const pt = (path, k, i) => v.px(path[k][2 * i], path[k][2 * i + 1]);
    for (const i of TRAILS) {
      ctx.beginPath(); ctx.moveTo(...pt(REF, 0, i)); for (let k = 4; k <= 200; k += 4) ctx.lineTo(...pt(REF, k, i));
      ctx.strokeStyle = pal.muted; ctx.globalAlpha = 0.6; ctx.lineWidth = 1; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...pt(REF, 0, i)); ctx.lineTo(...pt(REF, 200, i)); ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.moveTo(...pt(P, 0, i)); for (let k = 1; k <= K; k++) ctx.lineTo(...pt(P, k, i));
      ctx.strokeStyle = hlc(['K', 'h', 'vel'], pal, pal.acc); ctx.lineWidth = isHL('K', 'h', 'vel') ? 2.4 : 1.6; ctx.stroke();
      for (let k = 0; k < K; k++) dot(ctx, ...pt(P, k, i), 1.8, hlc(['K', 'h'], pal, pal.acc));
    }
    ctx.fillStyle = hlc('x', pal, pal.fg);
    for (let i = 0; i < NP; i++) { const [X, Y] = pt(P, K, i); ctx.fillRect(X - 1.2, Y - 1.2, 2.4, 2.4); }
    ctx.restore();
    cross(ctx, ...v.px(...DATA_MEAN2), 7, pal.fg, 2);
    frame(ctx, pal, 0, 0, w, h);
    tag(ctx, pal, `K = ${K} Euler step${K > 1 ? 's' : ''}, h = 1/${K}`, 16, 26, hlc(['K', 'h'], pal, pal.fg), 14);
    const sE = straightness(P);
    ro.innerHTML = `K = <b>${K}</b> · mass per mode after the last step (data ${W2.map((x) => x.toFixed(2)).join(' / ')}): <b>${mm(E.q)}</b> · mean log p<sub>data</sub> <b>${f2(E.q.ll)}</b> (exact samples ${LL_EXACT.toFixed(2)})<br>` +
      `straightness, chord length over path length averaged over the 1000 points: accurate paths <b>${REF_S.toFixed(2)}</b> · Euler polygons <b>${sE.toFixed(2)}</b> (1 for straight paths)`;
  }
  return { draw };
}

export const SCENES = { continuity, conditional, marginal, fm, identity, family, straight };
