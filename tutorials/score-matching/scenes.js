// Live scenes of the §2.2 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, arrow, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { makeRng } from '../../animations/lib/rng.js';
import {
  alpha, sigma, p1, score1, scoret1, SAMPLES1, PAIRS1, model, objectives, ismSample,
  MIX2, W2, pt2, PAIRS2, posteriorSamples, fields2, WEIGHTS,
} from './sim.js';

// ---------- Helpers ----------
// World box [-bx, bx] x [-by, by] fitted into a rectangle of the canvas.
function view(x0, y0, w, h, bx = 3.2, by = 2.4) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y], wx: (X, Y) => [(X - cx) / s, (cy - Y) / s] };
}
// Heat map of a 2D density, cached by key. The colour is the second accent, alpha follows p^0.7.
const heatCache = new Map();
function heat(key, w, h, v, dens, pmax) {
  const k = `${key}|${w}x${h}|${live.theme}`;
  if (heatCache.has(k)) return heatCache.get(k);
  if (heatCache.size > 80) heatCache.clear();
  const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res);
  const off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H);
  const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.5 : 0.32;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const [x, y] = v.wx(i * res, j * res), a = Math.pow(Math.min(1, dens(x, y) / pmax), 0.7), o = 4 * (j * W + i);
    im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * a * amax);
  }
  g.putImageData(im, 0, 0);
  heatCache.set(k, off);
  return off;
}
// Heat map of p_t (t = 1 is the data density p).
function heatT(ctx, w, h, v, t, alphaMul = 0.85) {
  const dens = (x, y) => pt2(x, y, t), pmax = Math.max(...MIX2.means.map(([mx, my]) => pt2(alpha(t) * mx, alpha(t) * my, t))) * 1.05;
  ctx.globalAlpha = alphaMul; ctx.drawImage(heat(`pt${t.toFixed(3)}`, w, h, v, dens, pmax), 0, 0, w, h); ctx.globalAlpha = 1;
}
// A label on a translucent box, for text that sits on top of a vector field.
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_/g, '')).width + 12, x0 = align === 'right' ? x - w + 6 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function cross(ctx, X, Y, r, color, lw = 2) {
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X - r, Y + r); ctx.lineTo(X + r, Y - r); ctx.stroke();
}
// Pointer position in world coordinates of a 2D scene.
const track = (c, onMove, mkView = () => view(0, 0, c.w, c.h)) => c.cv.addEventListener('pointermove', (e) => {
  const r = c.cv.getBoundingClientRect(); onMove(mkView().wx(e.clientX - r.left, e.clientY - r.top));
});
const fmt2 = ([a, b]) => `(${a.toFixed(2)}, ${b.toFixed(2)})`;
const DATA_MEAN = MIX2.means.reduce(([a, b], [mx, my], k) => [a + W2[k] * mx, b + W2[k] * my], [0, 0]);
const tSlider = (value, on, min = 0.05, max = 0.95) => slider('t', { min, max, step: 0.01, value }, on);

// ---------- Steps 1 and 2: a Gaussian score model on the 1D mixture ----------
const LO = -4, HI = 4, XS = Array.from({ length: 401 }, (_, i) => LO + (HI - LO) * i / 400);
const PMAX1 = Math.max(...XS.map(p1));
const SAMPLE_MEAN = SAMPLES1.reduce((a, b) => a + b, 0) / SAMPLES1.length;
const SAMPLE_VAR = SAMPLES1.reduce((a, b) => a + (b - SAMPLE_MEAN) ** 2, 0) / SAMPLES1.length;

function oneD(viz, ism) {
  const c = makeCanvas(16 / 10), ro = readout();
  let m = 0.5, v = 1, probe = null, k = 0;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = LO + (HI - LO) * (e.clientX - r.left - 44) / (r.width - 62); draw(); });
  c.cv.addEventListener('pointerleave', () => { probe = null; draw(); });
  const mSl = slider('theta', { min: -3, max: 3, step: 0.01, value: m, tex: 'm' }, (x) => { m = x; draw(); });
  const vSl = slider('theta', { min: -1, max: Math.log10(5), step: 0.01, value: Math.log10(v), map: (x) => 10 ** x, tex: 'v' }, (x) => { v = x; draw(); });
  const items = [mSl, vSl];
  let r = null;
  if (ism) {
    // Gradient descent on the sample estimate of J_ISM in (m, log v). It never evaluates p.
    const LR = 0.1, KMAX = 300;
    const step = () => {
      const g = ismSample(m, v);
      // Preconditioned by the inverse curvature, v^2 in m and v in log v. The sliders round to their step,
      // so the exact values are restored after moving them.
      const m1 = m - LR * v * v * g.gm, v1 = v * Math.exp(-LR * v * g.gl);
      mSl.set(m1); vSl.set(Math.log10(v1)); m = m1; v = v1;
      k++;
    };
    r = runner({ step, done: () => k >= KMAX || Math.abs(ismSample(m, v).gm) + Math.abs(ismSample(m, v).gl) < 1e-5, draw: () => draw(), interval: 30, perFrame: 2 });
    r.playing = false;
    items.push(button('Fit from samples', () => { k = 0; r.playing = true; r.kick(); }),
      button('Reset', () => { r.playing = false; k = 0; mSl.set(0.5); vSl.set(0); }));
  }
  viz.append(c.cv, controls(...items), ro,
    note(ism ? `The dots under the axis are the 200 samples the fit uses. The minimizer of the sample objective is their mean ${SAMPLE_MEAN.toFixed(2)} and variance ${SAMPLE_VAR.toFixed(2)}.`
      : 'The dots under the axis are 200 samples from p, all a learner has in practice. The data are the one-dimensional mixture of §2.1 with three components, and the curves are exact so that the objective can be shown.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const L = 44, R = w - 18, top1 = 30, bot1 = h * 0.55, top2 = h * 0.64, bot2 = h - 30, mid1 = (top1 + bot1) / 2;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    const SMAX = 6, Y1 = (s) => mid1 - (bot1 - top1) / 2 * Math.max(-1.1, Math.min(1.1, s / SMAX));
    const Yp = (p) => bot1 - (bot1 - top1) * 0.9 * p / PMAX1;
    const { esm, ism: jism } = objectives(m, v);
    // Upper panel: p shaded, true score dashed, model line.
    ctx.save(); ctx.beginPath(); ctx.rect(L, top1 - 6, R - L, bot1 - top1 + 12); ctx.clip();
    ctx.beginPath(); ctx.moveTo(X(LO), bot1); XS.forEach((x) => ctx.lineTo(X(x), Yp(p1(x)))); ctx.lineTo(X(HI), bot1); ctx.closePath();
    ctx.fillStyle = hlc(['p', 'Ep'], pal, pal.acc2); ctx.globalAlpha = isHL('p', 'Ep') ? 0.4 : 0.16; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, mid1 + 0.5); ctx.lineTo(R, mid1 + 0.5); ctx.stroke();
    ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y1(score1(x))));
    ctx.setLineDash([6, 5]); ctx.strokeStyle = hlc('glp', pal, pal.fg); ctx.lineWidth = isHL('glp') ? 3 : 1.8; ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y1(model(x, m, v))));
    ctx.strokeStyle = hlc(['stheta', 'theta', 'div'], pal, pal.acc); ctx.lineWidth = isHL('stheta', 'theta', 'div') ? 3.4 : 2.4; ctx.stroke();
    ctx.restore();
    const narrow = w < 560;
    label(ctx, narrow ? 'model s_θ(x)' : 'model s_θ(x) = −(x − m)/v', L + 4, top1 - 10, hlc(['stheta', 'theta'], pal, pal.acc), 13.5);
    label(ctx, narrow ? '∇log p(x), dashed' : 'true score ∇log p(x), dashed', R, top1 - 10, hlc('glp', pal, pal.muted), 13.5, 'right');
    label(ctx, 'p(x), shaded', L + 6, bot1 - 8, hlc(['p', 'Ep'], pal, pal.acc2), 12.5);
    label(ctx, '0', L - 8, mid1 + 4, pal.muted, 11, 'right');
    label(ctx, String(SMAX), L - 8, Y1(SMAX) + 4, pal.muted, 11, 'right'); label(ctx, String(-SMAX), L - 8, Y1(-SMAX) + 4, pal.muted, 11, 'right');
    ctx.fillStyle = hlc(['Ep', 'p'], pal, pal.fg); ctx.globalAlpha = 0.55;
    SAMPLES1.forEach((x, i) => { if (x > LO && x < HI) ctx.fillRect(X(x) - 1, bot1 + 4 + (i % 4) * 3, 2, 2.5); });
    ctx.globalAlpha = 1;
    // Lower panel: the integrand whose area is the objective.
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
    const f = ism ? (x) => p1(x) * (0.5 * model(x, m, v) ** 2 - 1 / v) : (x) => 0.5 * p1(x) * (model(x, m, v) - score1(x)) ** 2;
    const zero = ism ? top2 + (bot2 - top2) * 0.62 : bot2, FMAX = 2.5;
    const Y2 = (y) => zero - (ism ? (bot2 - top2) * 0.6 : bot2 - top2 - 20) * Math.max(-1.05, Math.min(1.05, y / FMAX));
    ctx.beginPath(); ctx.moveTo(L, zero + 0.5); ctx.lineTo(R, zero + 0.5); ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(L, top2 + 16, R - L, bot2 - top2 - 6); ctx.clip();
    ctx.beginPath(); ctx.moveTo(X(LO), zero); XS.forEach((x) => ctx.lineTo(X(x), Y2(f(x)))); ctx.lineTo(X(HI), zero); ctx.closePath();
    const jc = hlc('J', pal, ism ? pal.ramp[3] : pal.acc);
    ctx.fillStyle = jc; ctx.globalAlpha = isHL('J') ? 0.5 : 0.28; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = jc; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.restore();
    label(ctx, ism ? 'p(x) (½ s_θ(x)² + s_θ′(x)): signed area = J_{ESM} − C' : '½ p(x) (s_θ(x) − ∇log p(x))²: area = J_{ESM}', L + 4, top2 + 6, jc, 13.5);
    // Probe.
    let pr = '';
    if (probe !== null && probe > LO && probe < HI) {
      const Xp = X(probe);
      ctx.strokeStyle = hlc('x', pal, pal.muted); ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(Xp, top1); ctx.lineTo(Xp, bot2); ctx.stroke(); ctx.setLineDash([]);
      pr = `x = <b>${probe.toFixed(2)}</b> · s<sub>θ</sub>(x) = <b>${model(probe, m, v).toFixed(2)}</b> · ∇log p(x) = <b>${score1(probe).toFixed(2)}</b><br>`;
    }
    label(ctx, 'x', R, bot2 + 18, hlc('x', pal, pal.muted), 13, 'right');
    ro.innerHTML = pr + `m = <b>${m.toFixed(2)}</b> · v = <b>${v.toFixed(2)}</b> · J<sub>ESM</sub> = <b>${esm.toFixed(3)}</b>` +
      (ism ? ` · E<sub>p</sub>[s<sub>θ</sub>²/2 + ∇·s<sub>θ</sub>] = <b>${jism.toFixed(3)}</b> · difference C = <b>${(esm - jism).toFixed(3)}</b>${k ? ` · fit step <b>${k}</b>` : ''}` : '');
  }
  return { draw, active: (on) => r?.setActive(on) };
}
const esm = (viz) => oneD(viz, false);
const ism = (viz) => oneD(viz, true);

// ---------- Step 3: the conditional score around one data point ----------
// Fixed noise draws. Draw 0 has length exactly 1, so the check can be read off the readout.
const NOISE = (() => { const r = makeRng(51); const a = [[0.6, 0.8]]; for (let i = 1; i < 48; i++) a.push([r.gauss(), r.gauss()]); return a; })();
function cond(viz) {
  const c = makeCanvas(4 / 3), ro = readout();
  let t = 0.5, x1 = [1.35, -0.85];
  c.cv.addEventListener('click', (e) => { const r = c.cv.getBoundingClientRect(); x1 = view(0, 0, c.w, c.h).wx(e.clientX - r.left, e.clientY - r.top); draw(); });
  viz.append(c.cv, controls(tSlider(t, (x) => { t = x; draw(); })), ro,
    note('Click to move the data point. 48 fixed noise draws. Arrow lengths are compressed; the readout gives the exact value for the highlighted draw, which has ‖x<sub>0</sub>‖ = 1.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h), a = alpha(t), s = sigma(t);
    heatT(ctx, w, h, v, 1, 0.45);
    const [Cx, Cy] = v.px(a * x1[0], a * x1[1]), [X1, Y1] = v.px(...x1), [O0, O1] = v.px(0, 0);
    // Shrinkage of the data point toward the origin.
    ctx.setLineDash([2, 4]); ctx.strokeStyle = hlc('alpha', pal, pal.muted); ctx.lineWidth = isHL('alpha') ? 2.2 : 1.2;
    ctx.beginPath(); ctx.moveTo(O0, O1); ctx.lineTo(X1, Y1); ctx.stroke(); ctx.setLineDash([]);
    cross(ctx, O0, O1, 4, pal.muted, 1.2);
    // Level sets of p_t(x | x1): radius sigma and 2 sigma.
    const pcHL = isHL('pcond');
    [1, 2].forEach((q) => ring(ctx, Cx, Cy, q * s * v.s, hlc(['pcond', 'sig'], pal, pal.acc2), (pcHL || isHL('sig')) && q === 1 ? 2.6 : 1.2));
    // Noisy points and their conditional scores.
    const csHL = isHL('cs', 'x0');
    NOISE.forEach(([gx, gy], i) => {
      const xt = [a * x1[0] + s * gx, a * x1[1] + s * gy], [X, Y] = v.px(...xt);
      const sc = [-gx / s, -gy / s], mm = Math.hypot(...sc) || 1, len = (i ? 34 : 60) * Math.tanh(0.12 * mm);
      ctx.globalAlpha = i ? 0.75 : 1;
      arrow(ctx, X, Y, X + len * sc[0] / mm, Y - len * sc[1] / mm, i ? (csHL ? pal.mark : pal.muted) : hlc(['cs', 'x0'], pal, pal.acc), i ? 1.3 : 2.8, i ? 5 : 10);
      dot(ctx, X, Y, i ? 2.4 : 5, i ? hlc(['xt', 'x'], pal, pal.fg) : hlc(['xt', 'x'], pal, pal.acc));
      ctx.globalAlpha = 1;
    });
    dot(ctx, Cx, Cy, 4, hlc('alpha', pal, pal.acc2));
    ring(ctx, X1, Y1, isHL('x1') ? 10 : 8, hlc('x1', pal, pal.fg), isHL('x1') ? 3 : 2);
    tag(ctx, pal, 'x_1', X1 + 12, Y1 - 10, hlc('x1', pal, pal.fg), 14);
    tag(ctx, pal, 'α_t x_1', Cx + 10, Cy + 22, hlc('alpha', pal, pal.acc2), 13);
    tag(ctx, pal, `t = ${t.toFixed(2)}, σ_t = ${s.toFixed(2)}`, w - 12, 22, hlc(['t', 'sig'], pal, pal.muted), 13.5, 'right');
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · α<sub>t</sub> = <b>${a.toFixed(2)}</b> · σ<sub>t</sub> = <b>${s.toFixed(2)}</b> · x<sub>1</sub> = <b>${fmt2(x1)}</b><br>highlighted draw: ‖x<sub>0</sub>‖ = <b>1.00</b> · ‖∇log p<sub>t</sub>(x<sub>t</sub> | x<sub>1</sub>)‖ = ‖x<sub>0</sub>‖/σ<sub>t</sub> = <b>${(1 / s).toFixed(2)}</b>`;
  }
  return { draw };
}

// ---------- Step 5: regression on the noisy target ----------
function dsm(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let t = 0.5, probe = 0.6;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = LO + (HI - LO) * (e.clientX - r.left - 44) / (r.width - 62); draw(); });
  viz.append(c.cv, controls(tSlider(t, (x) => { t = x; draw(); }, 0.1, 0.9)), ro,
    note(`${PAIRS1.n} pairs (x<sub>1</sub>, x<sub>0</sub>) of the one-dimensional mixture, drawn once. The open circles are bin means of the labels; bins with fewer than 15 dots are left out.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), a = alpha(t), s = sigma(t);
    const L = 44, R = w - 18, top = 30, bot = h - 30, mid = (top + bot) / 2, YM = 3 / s;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO), Y = (y) => mid - (bot - top) / 2 * y / YM;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, mid + 0.5); ctx.lineTo(R, mid + 0.5); ctx.stroke();
    label(ctx, '0', L - 8, mid + 4, pal.muted, 11, 'right');
    label(ctx, (YM * 0.9).toFixed(1), L - 8, Y(YM * 0.9) + 4, pal.muted, 11, 'right'); label(ctx, (-YM * 0.9).toFixed(1), L - 8, Y(-YM * 0.9) + 4, pal.muted, 11, 'right');
    // Dots and statistics.
    const NB = 48, bw = (HI - LO) / NB, sum = new Float64Array(NB), cnt = new Float64Array(NB);
    let l0 = 0, lS = 0, sq = 0, pn = 0, ps = 0, pss = 0;
    const dotHL = isHL('Y', 'cs', 'X', 'xt');
    ctx.fillStyle = dotHL ? pal.mark : pal.fg;
    for (let i = 0; i < PAIRS1.n; i++) {
      const x = a * PAIRS1.x1[i] + s * PAIRS1.x0[i], y = -PAIRS1.x0[i] / s, sc = scoret1(x, t);
      l0 += 0.5 * y * y / PAIRS1.n; lS += 0.5 * (sc - y) ** 2 / PAIRS1.n; sq += 0.5 * sc * sc / PAIRS1.n;
      const b = Math.floor((x - LO) / bw); if (b >= 0 && b < NB) { sum[b] += y; cnt[b]++; }
      const near = Math.abs(x - probe) < 0.12;
      if (near) { pn++; ps += y; pss += y * y; }
      ctx.globalAlpha = near ? 0.85 : dotHL ? 0.4 : 0.2;
      ctx.fillRect(X(x) - 1.2, Y(y) - 1.2, 2.4, 2.4);
    }
    ctx.globalAlpha = 1;
    // Exact marginal score.
    ctx.save(); ctx.beginPath(); ctx.rect(L, top, R - L, bot - top); ctx.clip();
    ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(scoret1(x, t))));
    ctx.strokeStyle = hlc(['ms', 'EY'], pal, pal.acc); ctx.lineWidth = isHL('ms', 'EY') ? 3.6 : 2.6; ctx.stroke();
    ctx.restore();
    for (let b = 0; b < NB; b++) if (cnt[b] >= 15) ring(ctx, X(LO + (b + 0.5) * bw), Y(sum[b] / cnt[b]), 4, hlc('EY', pal, pal.ramp[3]), 1.8);
    // Probe slice: mean and one standard deviation of the labels near the probe.
    if (probe > LO && probe < HI) {
      const Xp = X(probe);
      ctx.fillStyle = pal.fg; ctx.globalAlpha = 0.07; ctx.fillRect(X(probe - 0.12), top, X(probe + 0.12) - X(probe - 0.12), bot - top); ctx.globalAlpha = 1;
      if (pn > 4) {
        const mu = ps / pn, sd = Math.sqrt(Math.max(0, pss / pn - mu * mu));
        ctx.strokeStyle = pal.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(Xp, Y(mu - sd)); ctx.lineTo(Xp, Y(mu + sd)); ctx.stroke();
        [mu - sd, mu + sd].forEach((q) => { ctx.beginPath(); ctx.moveTo(Xp - 6, Y(q)); ctx.lineTo(Xp + 6, Y(q)); ctx.stroke(); });
        dot(ctx, Xp, Y(mu), 5, pal.ramp[3]);
        tag(ctx, pal, `labels near x: mean ${mu.toFixed(2)}, spread ${sd.toFixed(2)}`, Math.min(Xp + 12, R - 250), top + 18, pal.fg, 12.5);
      }
    }
    label(ctx, 'label −x_0/σ_t of each pair (vertical) against its input x_t (horizontal)', L + 4, top - 12, hlc(['Y', 'X'], pal, pal.muted), 13);
    label(ctx, 'solid: ∇log p_t(x)', R, bot + 20, hlc(['ms', 'EY'], pal, pal.acc), 13, 'right');
    label(ctx, 'circles: bin means of the labels', L, bot + 20, hlc('EY', pal, pal.ramp[3]), 13);
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · σ<sub>t</sub> = <b>${s.toFixed(2)}</b> · probe x = <b>${probe.toFixed(2)}</b>, ∇log p<sub>t</sub>(x) = <b>${scoret1(probe, t).toFixed(2)}</b><br>` +
      `J<sub>DSM</sub> of the exact score: <b>${lS.toFixed(3)}</b> · J<sub>DSM</sub> of f = 0: <b>${l0.toFixed(3)}</b> · difference <b>${(l0 - lS).toFixed(3)}</b> ≈ J<sub>ESM</sub> of f = 0, E‖∇log p<sub>t</sub>‖²/2 = <b>${sq.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: the marginal score as a posterior average ----------
function marginal(viz) {
  const c = makeCanvas(4 / 3), ro = readout();
  let t = 0.45, probe = [-0.75, -0.05];
  track(c, (p) => { probe = p; draw(); });
  viz.append(c.cv, controls(tSlider(t, (x) => { t = x; draw(); }, 0.1, 0.9)), ro,
    note('Move the cursor to place the probe x, and click to pin it there. The dots mark α<sub>t</sub>x<sub>1</sub> for 60 posterior draws of x<sub>1</sub>. All arrows are scaled by σ<sub>t</sub>², so each thin arrow ends at its dot. The heat map is p<sub>t</sub>.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h), a = alpha(t), s = sigma(t), s2 = s * s;
    heatT(ctx, w, h, v, t, isHL('pt') ? 1 : 0.8);
    const [X, Y] = v.px(...probe), { samples, resp } = posteriorSamples(...probe, t);
    let mx = 0, my = 0;
    const csHL = isHL('cs');
    samples.forEach(({ k, x, y }) => {
      const cs = [-(probe[0] - a * x) / s2, -(probe[1] - a * y) / s2]; mx += cs[0] / samples.length; my += cs[1] / samples.length;
      const [Xe, Ye] = v.px(probe[0] + s2 * cs[0], probe[1] + s2 * cs[1]);
      ctx.globalAlpha = 0.35; arrow(ctx, X, Y, Xe, Ye, csHL ? pal.mark : modeColor(pal, k), csHL ? 1.6 : 1.1, 5); ctx.globalAlpha = 1;
      dot(ctx, Xe, Ye, isHL('x1') ? 4 : 3, hlc('x1', pal, modeColor(pal, k)));
    });
    const sc = fields2(...probe, t).score;
    const [Xm, Ym] = v.px(probe[0] + s2 * mx, probe[1] + s2 * my), [Xs, Ys] = v.px(probe[0] + s2 * sc[0], probe[1] + s2 * sc[1]);
    arrow(ctx, X, Y, Xm, Ym, hlc(['cs', 'Ex0'], pal, pal.acc), 3.4, 12);
    ctx.setLineDash([6, 4]); arrow(ctx, X, Y, Xs, Ys, hlc('ms', pal, pal.fg), 2, 10); ctx.setLineDash([]);
    dot(ctx, X, Y, isHL('x', 'xt') ? 7 : 5.5, hlc(['x', 'xt'], pal, pal.fg));
    tag(ctx, pal, 'x', X - 10, Y - 10, hlc(['x', 'xt'], pal, pal.fg), 14, 'right');
    tag(ctx, pal, `t = ${t.toFixed(2)}`, w - 12, 22, hlc('t', pal, pal.muted), 13.5, 'right');
    tag(ctx, pal, 'bold: mean of the thin arrows · dashed: exact ∇log p_t(x)', 14, h - 12, pal.muted, 12.5);
    ro.innerHTML = `x = <b>${fmt2(probe)}</b> · t = <b>${t.toFixed(2)}</b> · posterior weight of the components: <b>${resp.map((q) => q.toFixed(2)).join(', ')}</b><br>` +
      `mean of 60 conditional scores = <b>${fmt2([mx, my])}</b> · exact ∇log p<sub>t</sub>(x) = <b>${fmt2(sc)}</b>`;
  }
  return { draw };
}

// ---------- Step 6: Tweedie's formula ----------
// Mean squared distance of the data points of the pairs from the data mean.
const SPREAD2 = Array.from({ length: PAIRS2.n }, (_, i) => (PAIRS2.x1[2 * i] - DATA_MEAN[0]) ** 2 + (PAIRS2.x1[2 * i + 1] - DATA_MEAN[1]) ** 2).reduce((a, b) => a + b, 0) / PAIRS2.n;
function tweedie(viz) {
  const c = makeCanvas(4 / 3), ro = readout();
  let t = 0.5, probe = [-1.9, 0.8];
  track(c, (p) => { probe = p; draw(); });
  viz.append(c.cv, controls(tSlider(t, (x) => { t = x; draw(); }, 0.02, 0.98)), ro,
    note(`Move the cursor to place the probe x. ${PAIRS2.n} fixed pairs: open circles are the noisy points x<sub>t</sub>, filled dots their denoised positions E[x<sub>1</sub> | x<sub>t</sub>], coloured by the component that produced x<sub>1</sub>. The heat map is the data density p, the bold cross the data mean and the small cross the origin.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h), a = alpha(t), s = sigma(t);
    heatT(ctx, w, h, v, 1, 0.55);
    const n = PAIRS2.n, xt = [], d1 = [];
    for (let i = 0; i < n; i++) {
      const p = [a * PAIRS2.x1[2 * i] + s * PAIRS2.x0[2 * i], a * PAIRS2.x1[2 * i + 1] + s * PAIRS2.x0[2 * i + 1]];
      xt.push(p); d1.push(fields2(...p, t).d1);
    }
    ctx.lineWidth = 0.8; ctx.strokeStyle = pal.muted; ctx.globalAlpha = 0.3;
    ctx.beginPath(); for (let i = 0; i < n; i++) { const [X0, Y0] = v.px(...xt[i]), [X1, Y1] = v.px(...d1[i]); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); } ctx.stroke();
    ctx.globalAlpha = isHL('xt', 'x') ? 0.95 : 0.55;
    for (let i = 0; i < n; i++) { const [X0, Y0] = v.px(...xt[i]); ring(ctx, X0, Y0, 2.6, hlc(['xt', 'x'], pal, pal.muted), 1); }
    ctx.globalAlpha = 1;
    for (let i = 0; i < n; i++) { const [X1, Y1] = v.px(...d1[i]); dot(ctx, X1, Y1, isHL('Ex1') ? 3.4 : 2.6, hlc('Ex1', pal, modeColor(pal, PAIRS2.k[i]))); }
    const [Mx, My] = v.px(...DATA_MEAN); cross(ctx, Mx, My, 7, pal.fg, 2.4);
    // The two moves of the formula at the probe: the step by sigma^2 times the score lands at
    // alpha E[x1 | x]; the division by alpha moves it away from the origin along the dotted ray.
    const f = fields2(...probe, t), step = [s * s * f.score[0], s * s * f.score[1]], mid = [probe[0] + step[0], probe[1] + step[1]];
    const [X, Y] = v.px(...probe), [Xm, Ym] = v.px(...mid), [Xd, Yd] = v.px(...f.d1), [O0, O1] = v.px(0, 0);
    cross(ctx, O0, O1, 4, pal.muted, 1.2);
    ctx.setLineDash([2, 4]); ctx.strokeStyle = hlc('alpha', pal, pal.acc2); ctx.lineWidth = isHL('alpha') ? 2.4 : 1.4;
    ctx.beginPath(); ctx.moveTo(O0, O1); ctx.lineTo(Xm, Ym); ctx.stroke(); ctx.setLineDash([]);
    arrow(ctx, Xm, Ym, Xd, Yd, hlc('alpha', pal, pal.acc2), 2.2, 9);
    arrow(ctx, X, Y, Xm, Ym, hlc(['ms', 'sig'], pal, pal.acc), 2.8, 10);
    dot(ctx, X, Y, 5.5, hlc('x', pal, pal.fg)); dot(ctx, Xm, Ym, 4, hlc('alpha', pal, pal.acc2)); dot(ctx, Xd, Yd, 6, hlc('Ex1', pal, pal.ramp[2]));
    tag(ctx, pal, 'x', X - 10, Y - 10, hlc('x', pal, pal.fg), 14, 'right');
    tag(ctx, pal, '1. add σ_t² ∇log p_t(x)', (X + Xm) / 2 + 10, (Y + Ym) / 2 - 8, hlc(['ms', 'sig'], pal, pal.acc), 12.5);
    tag(ctx, pal, '2. divide by α_t', Xd + 10, Yd + 18, hlc('alpha', pal, pal.acc2), 12.5);
    const pr = `x = <b>${fmt2(probe)}</b> · σ<sub>t</sub>²∇log p<sub>t</sub>(x) = <b>${fmt2(step)}</b> · x + σ<sub>t</sub>²∇log p<sub>t</sub>(x) = α<sub>t</sub>E[x<sub>1</sub> | x] = <b>${fmt2(mid)}</b> · divided by α<sub>t</sub>: E[x<sub>1</sub> | x] = <b>${fmt2(f.d1)}</b><br>`;
    tag(ctx, pal, `t = ${t.toFixed(2)}, α_t = ${a.toFixed(2)}, σ_t = ${s.toFixed(2)}`, w - 12, 22, hlc(['t', 'alpha', 'sig'], pal, pal.muted), 13.5, 'right');
    let sp = 0; for (const [x, y] of d1) sp += ((x - DATA_MEAN[0]) ** 2 + (y - DATA_MEAN[1]) ** 2) / n;
    ro.innerHTML = pr + `t = <b>${t.toFixed(2)}</b> · spread of the denoised points around the data mean, as a fraction of the spread of the data: <b>${(sp / SPREAD2).toFixed(2)}</b> (1 at t = 1, 0 at t = 0)`;
  }
  return { draw };
}

// ---------- Step 7: four prediction targets and their loss weights ----------
function targets(viz) {
  const c = makeCanvas(16 / 10, { maxH: 0.42 }), c1 = makeCanvas(16 / 6, { maxH: 0.22 }), ro = readout();
  let t = 0.5, probe = [-1.3, 0.6];
  const BX = 2.4, BY = 1.5;
  track(c, (p) => { probe = p; draw(); }, () => view(0, 0, c.w, c.h, BX, BY));
  viz.append(c.cv, c1.cv, controls(tSlider(t, (x) => { t = x; draw(); }, 0.05, 0.95)), ro,
    note('Move the cursor to place the probe x. Exact posterior means of the 2D mixture; the heat map is p<sub>t</sub> and the small cross the origin. Lower plot: the factor by which each loss multiplies an error ‖x̂<sub>1</sub> − x<sub>1</sub>‖² in the data prediction, against t. The dashed line marks the current t.'));
  function draw() {
    c.fit(); c1.fit();
    const pal = palette(), a = alpha(t), s = sigma(t), f = fields2(...probe, t);
    { const { ctx, w, h } = c, v = view(0, 0, w, h, BX, BY);
      heatT(ctx, w, h, v, t, 0.6);
      const O = v.px(0, 0), A = v.px(a * f.d1[0], a * f.d1[1]), P = v.px(...probe), H0 = v.px(...f.d0), H1 = v.px(...f.d1);
      cross(ctx, ...O, 4, pal.muted, 1.2);
      // The ray from the origin through x̂1, on which alpha x̂1 lies.
      ctx.setLineDash([2, 4]); ctx.strokeStyle = hlc('alpha', pal, pal.muted); ctx.lineWidth = isHL('alpha') ? 2.2 : 1;
      ctx.beginPath(); ctx.moveTo(...O); ctx.lineTo(...H1); ctx.stroke(); ctx.setLineDash([]);
      dot(ctx, ...A, 3.5, hlc('alpha', pal, pal.muted));
      dot(ctx, ...H0, isHL('hx0') ? 7 : 5.5, hlc('hx0', pal, pal.ramp[3]));
      dot(ctx, ...H1, isHL('hx1') ? 7 : 5.5, hlc('hx1', pal, pal.ramp[0]));
      // Velocity: the segment from x̂0 to x̂1, which passes through x at fraction t.
      arrow(ctx, ...H0, ...H1, hlc(['hu', 'adot', 'sdot'], pal, pal.ramp[2]), isHL('hu') ? 4 : 3, 12);
      // Score scaled by sigma^2: from x to alpha x̂1.
      arrow(ctx, ...P, ...A, hlc('st', pal, pal.fg), isHL('st') ? 3 : 2, 9);
      dot(ctx, ...P, 5.5, hlc('x', pal, pal.fg));
      // x̂0 can leave the plot for t near 1; its label then stays at the edge.
      const inside = H0[0] > 0 && H0[0] < w && H0[1] > 0 && H0[1] < h;
      const cl = (q, lo, hi) => Math.max(lo, Math.min(hi, q));
      tag(ctx, pal, inside ? 'x̂_0' : 'x̂_0 is off the plot', cl(H0[0] + 10, 14, w - 150), cl(H0[1] + 18, 40, h - 34), hlc('hx0', pal, pal.ramp[3]), 13);
      tag(ctx, pal, 'x̂_1', H1[0] + 10, H1[1] - 8, hlc('hx1', pal, pal.ramp[0]), 13);
      tag(ctx, pal, 'x', P[0] - 10, P[1] - 10, hlc('x', pal, pal.fg), 14, 'right');
      tag(ctx, pal, 'α_t x̂_1', A[0] + 10, A[1] + 18, hlc('alpha', pal, pal.muted), 12.5);
      tag(ctx, pal, 'σ_t² s_t', (P[0] + A[0]) / 2 + 10, (P[1] + A[1]) / 2 - 8, hlc('st', pal, pal.fg), 12.5);
      tag(ctx, pal, 'û_t = x̂_1 − x̂_0', 14, h - 12, hlc('hu', pal, pal.ramp[2]), 12.5);
      tag(ctx, pal, `t = ${t.toFixed(2)}`, w - 12, 22, hlc('t', pal, pal.muted), 13.5, 'right');
    }
    { const { ctx, w, h } = c1, L = 48, R = w - 150, top = 14, bot = h - 24, LMIN = -2, LMAX = 4;
      frame(ctx, pal, 0, 0, w, h);
      const T = (u) => L + (R - L) * u, Y = (q) => bot - (bot - top) * (Math.log10(Math.max(q, 10 ** LMIN)) - LMIN) / (LMAX - LMIN);
      ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, bot + 0.5); ctx.lineTo(R, bot + 0.5); ctx.moveTo(L + 0.5, top); ctx.lineTo(L + 0.5, bot); ctx.stroke();
      [[0.01, '0.01'], [1, '1'], [100, '100'], [1e4, '10⁴']].forEach(([q, s]) => label(ctx, s, L - 6, Y(q) + 4, pal.muted, 10.5, 'right'));
      [0, 0.5, 1].forEach((u) => label(ctx, String(u), T(u), bot + 16, pal.muted, 10.5, 'center'));
      const curves = [['data', 'x̂_1 (data)', pal.ramp[0], 'hx1'], ['noise', 'x̂_0 (noise)', pal.ramp[3], 'hx0'], ['score', 's_t (score)', pal.fg, 'st'], ['velocity', 'û_t (velocity)', pal.ramp[2], 'hu']];
      curves.forEach(([key, name, col, id], j) => {
        const wf = WEIGHTS[key], c0 = hlc(id, pal, col);
        ctx.save(); ctx.beginPath(); ctx.rect(L, top - 2, R - L, bot - top + 4); ctx.clip();
        ctx.beginPath(); for (let i = 0; i <= 200; i++) { const u = 0.005 + 0.99 * i / 200; (i ? ctx.lineTo : ctx.moveTo).call(ctx, T(u), Y(wf(u))); }
        ctx.strokeStyle = c0; ctx.lineWidth = isHL(id) ? 3.2 : 2; ctx.stroke(); ctx.restore();
        dot(ctx, T(t), Y(wf(t)), 4, c0);
        label(ctx, `${name}: ${wf(t) < 100 ? wf(t).toFixed(2) : wf(t).toFixed(0)}`, R + 12, top + 12 + j * 18, c0, 12);
      });
      ctx.strokeStyle = hlc('t', pal, pal.muted); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(T(t), top); ctx.lineTo(T(t), bot); ctx.stroke(); ctx.setLineDash([]);
      label(ctx, w < 560 ? 'log scale' : 'weight on ‖x̂_1 − x_1‖², log scale', L + 6, top + 10, pal.muted, 11.5);
    }
    ro.innerHTML = `x = <b>${fmt2(probe)}</b> · t = <b>${t.toFixed(2)}</b> · x̂<sub>1</sub> = <b>${fmt2(f.d1)}</b> · x̂<sub>0</sub> = <b>${fmt2(f.d0)}</b><br>` +
      `s<sub>t</sub> = −x̂<sub>0</sub>/σ<sub>t</sub> = <b>${fmt2(f.score)}</b> · û<sub>t</sub> = x̂<sub>1</sub> − x̂<sub>0</sub> = <b>${fmt2(f.u)}</b> · check t x̂<sub>1</sub> + (1 − t) x̂<sub>0</sub> = <b>${fmt2([a * f.d1[0] + s * f.d0[0], a * f.d1[1] + s * f.d0[1]])}</b> = x`;
  }
  return { draw };
}

export const SCENES = { esm, ism, cond, dsm, marginal, tweedie, targets };
