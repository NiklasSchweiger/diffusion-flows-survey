// Live scenes of the §2.1 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, arrow, label, hlc, isHL, controls, slider, button, toggle, readout, note, runner } from '../lib/tutorial.js';
import { makeRng } from '../../animations/lib/rng.js';
import {
  p1, MIX2, score2, p2, spread2, Cloud, FP, fokkerPlanckRun, ulaVar,
  MIX3, ANNEAL, sigmas, levelGMM, annealRuns, massPerMode, nearest3,
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
const mixDensity = (means, std, w) => (x, y) => p2(x, y, means, std, w);
const pmax2 = p2(...MIX2.means[0]) * 1.05;
const dens2 = mixDensity(MIX2.means, MIX2.std, MIX2.w);
// A label on a translucent box, for text that sits on top of a vector field.
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_/g, '')).width + 12, x0 = align === 'right' ? x - w + 6 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
// Mass per mode (filled) against the target weights (outlined), for steps 6 and 7.
function massBars(ctx, pal, x, y, w, h, mass, target, title) {
  label(ctx, title, x, y + 12, pal.muted, 12.5);
  const top = y + 24, bh = h - 44, gw = w / 3;
  ctx.strokeStyle = pal.line; ctx.beginPath(); ctx.moveTo(x, top + bh + 0.5); ctx.lineTo(x + w, top + bh + 0.5); ctx.stroke();
  target.forEach((t, k) => {
    const cx = x + gw * (k + 0.5), bw = Math.min(46, gw * 0.5);
    const hm = bh * Math.min(1, mass[k] / 0.7), ht = bh * t / 0.7;
    ctx.fillStyle = modeColor(pal, k); ctx.globalAlpha = 0.85; ctx.fillRect(cx - bw / 2, top + bh - hm, bw, hm); ctx.globalAlpha = 1;
    ctx.setLineDash([4, 3]); ctx.strokeStyle = pal.fg; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(cx - bw / 2 - 6, top + bh - ht); ctx.lineTo(cx + bw / 2 + 6, top + bh - ht); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, gw < 100 ? mass[k].toFixed(2) : `${mass[k].toFixed(2)} / ${t.toFixed(2)}`, cx, top + bh + 15, pal.muted, 11.5, 'center');
  });
}
const W3 = (() => { const z = MIX3.w.reduce((a, b) => a + b, 0); return MIX3.w.map((x) => x / z); })();

// ---------- Step 1: energy and density on a 1D slice ----------
function energy(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let probe = 0.3, shift = 0;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = -4 + 8 * (e.clientX - r.left - 44) / (r.width - 62); draw(); });
  viz.append(c.cv, controls(slider('c', { min: -1, max: 1, step: 0.01, value: 0 }, (v) => { shift = v; draw(); })), ro,
    note('A one-dimensional Gaussian mixture with three components. All curves are exact.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const L = 44, R = w - 18, top1 = 26, bot1 = h * 0.34, top2 = h * 0.42, bot2 = h - 30;
    const X = (x) => L + (R - L) * (x + 4) / 8;
    const E = (x) => -Math.log(p1(x)) + shift;
    const Y1 = (e) => bot1 - (bot1 - top1) * (e + 2) / 9, Y2 = (v) => bot2 - (bot2 - top2 - 34) * v / 1.15;
    const xs = Array.from({ length: 401 }, (_, i) => -4 + 8 * i / 400);
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
    [bot1, bot2].forEach((y) => { ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(R, y); ctx.stroke(); });
    // E(x), clipped at the top of its panel.
    ctx.save(); ctx.beginPath(); ctx.rect(L, top1 - 12, R - L, bot1 - top1 + 12); ctx.clip();
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y1(E(x))));
    ctx.strokeStyle = hlc('E', pal, pal.fg); ctx.lineWidth = isHL('E') ? 3.2 : 2; ctx.stroke(); ctx.restore();
    label(ctx, 'E(x)', L + 4, top1 + 4, hlc('E', pal, pal.muted), 14);
    label(ctx, '0', L - 8, Y1(0) + 4, pal.muted, 11, 'right');
    // exp(−E) with its area Z, clipped to its panel.
    ctx.save(); ctx.beginPath(); ctx.rect(L, top2 + 30, R - L, bot2 - top2 - 30); ctx.clip();
    const Zc = Math.exp(-shift);
    ctx.beginPath(); ctx.moveTo(X(-4), Y2(0));
    xs.forEach((x) => ctx.lineTo(X(x), Y2(p1(x) * Zc)));
    ctx.lineTo(X(4), Y2(0)); ctx.closePath();
    ctx.fillStyle = hlc('Z', pal, pal.acc2); ctx.globalAlpha = isHL('Z') ? 0.35 : 0.16; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.acc2; ctx.lineWidth = 2; ctx.stroke();
    // p(x), dashed.
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y2(p1(x))));
    ctx.setLineDash([6, 5]); ctx.strokeStyle = hlc('p', pal, pal.fg); ctx.lineWidth = isHL('p') ? 3 : 1.8; ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    label(ctx, 'exp(−E(x)), shaded area = Z', L + 4, top2 + 2, hlc('Z', pal, pal.acc2), 14);
    label(ctx, 'p(x) = exp(−E(x)) / Z, dashed', L + 4, top2 + 20, hlc('p', pal, pal.muted), 14);
    // Probe.
    const xp = Math.max(-4, Math.min(4, probe)), Xp = X(xp);
    ctx.strokeStyle = hlc('x', pal, pal.muted); ctx.lineWidth = isHL('x') ? 2 : 1; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(Xp, top1); ctx.lineTo(Xp, bot2); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'x', Xp, bot2 + 18, hlc('x', pal, pal.muted), 14, 'center');
    ro.innerHTML = `x = <b>${xp.toFixed(2)}</b> · E(x) = <b>${E(xp).toFixed(2)}</b> · exp(−E(x)) = <b>${(p1(xp) * Zc).toFixed(3)}</b> · Z = <b>${Zc.toFixed(3)}</b> · p(x) = <b>${p1(xp).toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: the score field ----------
function score(viz) {
  const c = makeCanvas(4 / 3), ro = readout();
  let probe = [0.4, 0.35], shift = 0;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = view(0, 0, c.w, c.h).wx(e.clientX - r.left, e.clientY - r.top); draw(); });
  viz.append(c.cv, controls(slider('c', { min: -1.5, max: 1.5, step: 0.01, value: 0 }, (v) => { shift = v; draw(); })), ro,
    note('Exact score of a two-dimensional Gaussian mixture with three components. No network is trained.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h), fieldHL = isHL('s', 'nabla');
    ctx.globalAlpha = isHL('p') ? 1 : 0.85; ctx.drawImage(heat('mix2', w, h, v, dens2, pmax2), 0, 0, w, h); ctx.globalAlpha = 1;
    for (let i = 0; i < 21; i++) for (let j = 0; j < 15; j++) {
      const x = -3 + 6 * i / 20, y = -2.2 + 4.4 * j / 14, sc = score2(x, y), m = Math.hypot(...sc);
      const len = 16 * Math.tanh(0.25 * m), [X, Y] = v.px(x, y);
      ctx.globalAlpha = 0.35 + 0.5 * len / 16;
      arrow(ctx, X, Y, X + len * sc[0] / m, Y - len * sc[1] / m, fieldHL ? pal.mark : pal.muted, 1.4, 5);
    }
    ctx.globalAlpha = 1;
    const sc = score2(...probe), m = Math.hypot(...sc), len = 60 * Math.tanh(0.18 * m), [X, Y] = v.px(...probe);
    arrow(ctx, X, Y, X + len * sc[0] / m, Y - len * sc[1] / m, fieldHL ? pal.mark : pal.fg, 3.2, 12);
    ctx.beginPath(); ctx.arc(X, Y, isHL('x') ? 7 : 5, 0, 7); ctx.fillStyle = hlc('x', pal, pal.fg); ctx.fill();
    label(ctx, 's(x)', X + len * sc[0] / m + 8, Y - len * sc[1] / m - 6, hlc('s', pal, pal.fg), 14);
    label(ctx, 'Arrow length is compressed for long vectors; the readout gives the true value.', 10, h - 10, pal.muted, 11.5);
    const E = -Math.log(p2(...probe)) + shift;
    ro.innerHTML = `x = <b>(${probe[0].toFixed(2)}, ${probe[1].toFixed(2)})</b> · s(x) = <b>(${sc[0].toFixed(2)}, ${sc[1].toFixed(2)})</b> · ‖s‖ = <b>${m.toFixed(2)}</b> · E(x) = <b>${E.toFixed(2)}</b> · Z = <b>${Math.exp(-shift).toFixed(3)}</b>`;
  }
  return { draw };
}

// Draws a 2D particle cloud with the target heat map. Particle 0 is skipped (drawn by the caller).
function drawCloud(ctx, pal, v, w, h, cloud, alpha = 0.75) {
  ctx.globalAlpha = isHL('p') ? 1 : 0.7; ctx.drawImage(heat('mix2', w, h, v, dens2, pmax2), 0, 0, w, h);
  ctx.globalAlpha = alpha; ctx.fillStyle = pal.fg;
  for (let i = 1; i < cloud.n; i++) {
    const [X, Y] = v.px(cloud.P[2 * i], cloud.P[2 * i + 1]);
    if (X < -5 || X > w + 5 || Y < -5 || Y > h + 5) continue;
    ctx.beginPath(); ctx.arc(X, Y, 2.1, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ---------- Step 3: the Langevin SDE ----------
function langevin(viz) {
  const c = makeCanvas(4 / 3), ro = readout(), DT = 0.01, KMAX = 1500;
  const cloud = new Cloud(400, 5);
  let drift = true, noise = true;
  const restart = () => { cloud.reset(); draw(); r.kick(); };
  const r = runner({ step: () => cloud.step(DT, score2, { drift, noise }), done: () => cloud.k >= KMAX, draw: () => draw(), interval: 33, perFrame: 3 });
  viz.append(c.cv, controls(
    toggle('drift \\(\\nabla\\log p\\,\\dd\\tau\\)', true, (on) => { drift = on; restart(); }),
    toggle('noise \\(\\sqrt2\\,\\dd\\W_\\tau\\)', true, (on) => { noise = on; restart(); }),
    r.play, button('Reset', restart),
  ), ro, note(`400 particles from a uniform start, exact mixture score, Euler–Maruyama with Δτ = ${DT}. A switch restarts the run with the same random numbers.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), v = view(0, 0, w, h);
    drawCloud(ctx, pal, v, w, h, cloud, isHL('xtau') ? 0.95 : 0.75);
    if (isHL('xtau')) { ctx.globalAlpha = 0.5; ctx.fillStyle = pal.mark; for (let i = 1; i < cloud.n; i++) { const [X, Y] = v.px(cloud.P[2 * i], cloud.P[2 * i + 1]); ctx.beginPath(); ctx.arc(X, Y, 2.4, 0, 7); ctx.fill(); } ctx.globalAlpha = 1; }
    // Trail of the tracked particle.
    ctx.beginPath(); cloud.trail.forEach(([x, y], i) => { const [X, Y] = v.px(x, y); (i ? ctx.lineTo : ctx.moveTo).call(ctx, X, Y); });
    ctx.strokeStyle = hlc(['dW', 'W'], pal, pal.acc2); ctx.lineWidth = 1.6; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
    const [X, Y] = v.px(cloud.P[0], cloud.P[1]);
    ctx.beginPath(); ctx.arc(X, Y, 5.5, 0, 7); ctx.fillStyle = hlc('xtau', pal, pal.acc2); ctx.fill();
    // Score at the tracked particle: the drift direction.
    const sc = score2(cloud.P[0], cloud.P[1]), m = Math.hypot(...sc) || 1, len = 40 * Math.tanh(0.25 * m);
    if (drift) arrow(ctx, X, Y, X + len * sc[0] / m, Y - len * sc[1] / m, hlc(['nabla', 'p'], pal, pal.acc), 2.4, 9);
    label(ctx, `τ = ${(cloud.k * DT).toFixed(2)}`, w - 12, 22, hlc(['tau', 'dtau'], pal, pal.muted), 14, 'right');
    const sp = cloud.spread();
    ro.innerHTML = `τ = <b>${(cloud.k * DT).toFixed(2)}</b> · drift <b>${drift ? 'on' : 'off'}</b> · noise <b>${noise ? 'on' : 'off'}</b> · spread of the cloud = <b>${Math.sqrt(sp).toFixed(2)}</b> (target ${Math.sqrt(spread2).toFixed(2)})`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 4: Fokker–Planck, histogram of ρ_τ against p and KL over τ ----------
function stationary(viz) {
  const c = makeCanvas(4 / 3), ro = readout();
  let run = null, f = 0;
  const get = () => (run ??= fokkerPlanckRun());
  const n = Math.round(FP.T / (FP.dtau * 5));
  const tauSl = slider('tau', { min: 0, max: n, step: 1, value: 0, fmt: (v) => (v * FP.dtau * 5).toFixed(2) }, (v) => { f = v; draw(); });
  const r = runner({ step: () => { f++; tauSl.querySelector('input').value = f; tauSl.querySelector('.val').textContent = (f * FP.dtau * 5).toFixed(2); }, done: () => f >= n, draw: () => draw(), interval: 50 });
  r.playing = false; r.play.textContent = 'Play';
  const play = r.play.onclick;
  r.play.onclick = () => { if (!r.playing && f >= n) f = 0; play(); };
  viz.append(c.cv, controls(tauSl, r.play), ro,
    note(`${FP.N} particles of the one-dimensional mixture of step 1, exact score, Δτ = ${FP.dtau}. The histogram has ${FP.bins} bins, so its KL estimate stays slightly above zero even for exact samples.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), R = get(), hist = R.frames[Math.min(f, R.frames.length - 1)];
    const L = 48, Rr = w - 18, top1 = 30, bot1 = h * 0.58, top2 = h * 0.68, bot2 = h - 30;
    const X = (x) => L + (Rr - L) * (x - FP.lo) / (FP.hi - FP.lo), Y1 = (v) => bot1 - (bot1 - top1) * Math.min(v, 1.6) / 1.6;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(L, bot1 + 0.5); ctx.lineTo(Rr, bot1 + 0.5); ctx.stroke();
    // Histogram of ρ_τ.
    const bw = (Rr - L) / FP.bins;
    ctx.fillStyle = hlc('rho', pal, pal.acc2); ctx.globalAlpha = isHL('rho') ? 0.75 : 0.5;
    for (let b = 0; b < FP.bins; b++) { const y = Y1(hist[b]); ctx.fillRect(L + b * bw + 0.5, y, bw - 1, bot1 - y); }
    ctx.globalAlpha = 1;
    // Target p.
    ctx.beginPath();
    for (let i = 0; i <= 300; i++) { const x = FP.lo + (FP.hi - FP.lo) * i / 300; (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y1(p1(x))); }
    ctx.setLineDash([6, 5]); ctx.strokeStyle = hlc('p', pal, pal.fg); ctx.lineWidth = isHL('p') ? 3 : 1.8; ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'histogram of the particles: ρ_τ', L + 4, top1 - 8, hlc('rho', pal, pal.acc2), 13.5);
    label(ctx, 'target p, dashed', Rr, top1 - 8, hlc('p', pal, pal.muted), 13.5, 'right');
    // KL over τ, log scale from 0.01 to 10.
    const T = (tau) => L + (Rr - L) * tau / FP.T, Y2 = (k) => bot2 - (bot2 - top2) * (Math.log10(Math.max(k, 0.01)) + 2) / 3;
    ctx.strokeStyle = pal.line; ctx.beginPath(); ctx.moveTo(L, bot2 + 0.5); ctx.lineTo(Rr, bot2 + 0.5); ctx.moveTo(L + 0.5, top2); ctx.lineTo(L + 0.5, bot2); ctx.stroke();
    [0.01, 0.1, 1, 10].forEach((k) => label(ctx, String(k), L - 6, Y2(k) + 4, pal.muted, 10.5, 'right'));
    ctx.beginPath(); R.kl.forEach((k, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, T(i * R.dtauFrame), Y2(k)));
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.beginPath(); R.kl.slice(0, f + 1).forEach((k, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, T(i * R.dtauFrame), Y2(k)));
    ctx.strokeStyle = hlc('KL', pal, pal.acc); ctx.lineWidth = isHL('KL') ? 3 : 2.2; ctx.stroke();
    const kf = R.kl[Math.min(f, R.kl.length - 1)];
    ctx.beginPath(); ctx.arc(T(f * R.dtauFrame), Y2(kf), isHL('tau') ? 6 : 4.5, 0, 7); ctx.fillStyle = hlc('tau', pal, pal.acc); ctx.fill();
    label(ctx, 'KL(ρ_τ ‖ p), log scale', L + 8, top2 - 6, hlc('KL', pal, pal.acc), 13.5);
    label(ctx, 'τ', Rr, bot2 + 18, hlc('tau', pal, pal.muted), 13, 'right');
    ro.innerHTML = `τ = <b>${(f * R.dtauFrame).toFixed(2)}</b> · KL(ρ<sub>τ</sub> ‖ p) ≈ <b>${kf.toFixed(3)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: ULA, one step by hand and the bias on N(0, 1) ----------
function ula(viz) {
  const c = makeCanvas(16 / 10, { maxH: 0.45 }), c1 = makeCanvas(16 / 5, { maxH: 0.2 }), ro = readout(), warn = note('', 'warn');
  const KMAX = 800, N1 = 3000, BOX = 20;
  const cloud = new Cloud(400, 7), X1 = new Float64Array(N1);
  let eta = 0.05, rng1, vHist = [];
  const reset = () => { cloud.reset(); rng1 = makeRng(13); X1.fill(0); vHist = []; };
  const step = () => {
    cloud.step(eta, score2);
    // Clamp diverging particles so the picture stays finite; the warning explains why.
    for (let i = 0; i < 2 * cloud.n; i++) cloud.P[i] = Math.max(-BOX, Math.min(BOX, cloud.P[i]));
    const sq = Math.sqrt(2 * eta);
    let v = 0;
    for (let i = 0; i < N1; i++) { X1[i] += -eta * X1[i] + sq * rng1.gauss(); v += X1[i] * X1[i] / N1; }
    vHist.push(v);
  };
  reset();
  const r = runner({ step, done: () => cloud.k >= KMAX, draw: () => draw(), interval: 40 });
  const etaSl = slider('eta', { min: -2.7, max: Math.log10(1.95), step: 0.01, value: Math.log10(eta), map: (v) => 10 ** v, fmt: (v) => v.toFixed(3) }, (v) => { eta = v; reset(); draw(); r.kick(); });
  viz.append(c.cv, c1.cv, controls(etaSl, r.play, r.one, button('Reset', () => { reset(); draw(); r.kick(); })), warn, ro,
    note('Upper plot: 400 particles on the mixture of steps 2 and 3, exact score. Lower plot: 3000 particles on N(0, 1), started at 0. Changing η restarts both runs with the same random numbers.'));
  function draw() {
    c.fit(); c1.fit();
    const pal = palette(), vU = ulaVar(eta, 1);
    // Upper: the cloud and one step of the tracked particle.
    { const { ctx, w, h } = c, v = view(0, 0, w, h);
      drawCloud(ctx, pal, v, w, h, cloud);
      const dCol = hlc(['eta', 's'], pal, pal.acc), kCol = hlc('xi', pal, pal.ramp[3]);
      const pr = cloud.prev, dr = cloud.drift, cur = [cloud.P[0], cloud.P[1]];
      const [X0, Y0] = v.px(...pr), [X1p, Y1p] = v.px(pr[0] + dr[0], pr[1] + dr[1]), [X2, Y2] = v.px(...cur);
      if (cloud.k > 0) {
        arrow(ctx, X0, Y0, X1p, Y1p, dCol, 2.4, 8); arrow(ctx, X1p, Y1p, X2, Y2, kCol, 2.4, 8);
        ctx.beginPath(); ctx.arc(X0, Y0, 6, 0, 7); ctx.strokeStyle = hlc('xk', pal, pal.fg); ctx.lineWidth = isHL('xk') ? 2.6 : 1.6; ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(X2, Y2, isHL('xk1') ? 7 : 5.5, 0, 7); ctx.fillStyle = hlc('xk1', pal, pal.acc2); ctx.fill();
      // Zoomed inset of the last step, scaled so the longer arrow is 52 px. Hidden when that is not a magnification.
      const zoom = Math.min(52 / Math.max(Math.hypot(...dr), Math.hypot(cur[0] - pr[0], cur[1] - pr[1]), 1e-9), 400);
      if (cloud.k > 0 && zoom >= 1.5 * v.s) {
        const iw = 168, ih = 132, ix = w - iw - 10, iy = 10;
        ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.92; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1; frame(ctx, pal, ix, iy, iw, ih);
        const kick = [cur[0] - pr[0] - dr[0], cur[1] - pr[1] - dr[1]], Lmax = Math.max(Math.hypot(...dr), Math.hypot(...kick), Math.hypot(cur[0] - pr[0], cur[1] - pr[1]), 1e-9);
        const z = Math.max(Math.min(52 / Lmax, v.s * 400), v.s), ox = ix + iw / 2 - z * (dr[0] + kick[0]) / 2, oy = iy + ih / 2 + 8 + z * (dr[1] + kick[1]) / 2;
        const a = [ox, oy], b = [ox + z * dr[0], oy - z * dr[1]], e = [b[0] + z * kick[0], b[1] - z * kick[1]];
        arrow(ctx, ...a, ...b, dCol, 2.6, 9); arrow(ctx, ...b, ...e, kCol, 2.6, 9);
        ctx.beginPath(); ctx.arc(...a, 5, 0, 7); ctx.strokeStyle = hlc('xk', pal, pal.fg); ctx.lineWidth = 1.6; ctx.stroke();
        ctx.beginPath(); ctx.arc(...e, 4.5, 0, 7); ctx.fillStyle = hlc('xk1', pal, pal.acc2); ctx.fill();
        label(ctx, `last step, magnified ×${(z / v.s).toFixed(z / v.s < 10 ? 1 : 0)}`, ix + 8, iy + 16, pal.muted, 11.5);
        label(ctx, 'η s(x_k)', ix + 8, iy + ih - 22, dCol, 11.5); label(ctx, '√(2η) ξ_k', ix + 8, iy + ih - 8, kCol, 11.5);
      }
    }
    // Lower: histogram on N(0, 1) against N(0, 1) and N(0, 1/(1 − η/2)).
    { const { ctx, w, h } = c1, L = 10, R = w - 10, top = 22, bot = h - 18, lo = -5, hi = 5, bins = 50, dx = (hi - lo) / bins;
      const X = (x) => L + (R - L) * (x - lo) / (hi - lo), Y = (d) => bot - (bot - top) * Math.min(d, 0.5) / 0.5;
      frame(ctx, pal, 0, 0, w, h);
      const hist = new Float64Array(bins);
      for (let i = 0; i < N1; i++) { const b = Math.floor((X1[i] - lo) / dx); if (b >= 0 && b < bins) hist[b] += 1 / (N1 * dx); }
      ctx.fillStyle = pal.acc2; ctx.globalAlpha = 0.45;
      for (let b = 0; b < bins; b++) { const y = Y(hist[b]); ctx.fillRect(X(lo + b * dx) + 0.5, y, (R - L) / bins - 1, bot - y); }
      ctx.globalAlpha = 1;
      const curve = (v, dash, col, wd) => {
        ctx.beginPath();
        for (let i = 0; i <= 200; i++) { const x = lo + (hi - lo) * i / 200; (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(Math.exp(-x * x / (2 * v)) / Math.sqrt(2 * Math.PI * v))); }
        ctx.setLineDash(dash); ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.stroke(); ctx.setLineDash([]);
      };
      curve(1, [6, 5], hlc('p', pal, pal.fg), 1.6);
      curve(vU, [], hlc('eta', pal, pal.acc), 2);
      label(ctx, 'N(0, 1), dashed', L + 4, 15, hlc('p', pal, pal.muted), 12);
      label(ctx, `N(0, 1/(1 − η/2)), predicted ULA limit`, R - 4, 15, hlc('eta', pal, pal.acc), 12, 'right');
    }
    const tail = vHist.slice(-50), vEmp = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0;
    warn.textContent = eta > 0.5 ? `η = ${eta.toFixed(2)} exceeds 2 × 0.25 = 0.5, twice the component variance of the 2D mixture, so ULA diverges in the upper plot. Particles are held at the edge of a large box.` : '';
    ro.innerHTML = `k = <b>${cloud.k}</b> · η = <b>${eta.toFixed(3)}</b> · √(2η) = <b>${Math.sqrt(2 * eta).toFixed(3)}</b> · 2D spread = <b>${Math.sqrt(cloud.spread()).toFixed(2)}</b> (target ${Math.sqrt(spread2).toFixed(2)})<br>` +
      `N(0, 1): variance <b>${cloud.k ? vEmp.toFixed(3) : '0'}</b>${cloud.k >= 50 ? ' (mean over the last 50 steps)' : ''} · predicted 1/(1 − η/2) = <b>${vU.toFixed(3)}</b> · target 1`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// Uniform start shared by steps 6 and 7 (the same layout annealRuns uses).
function uniformStart(N) {
  const P = new Float64Array(2 * N), r0 = makeRng(3);
  for (let i = 0; i < N; i++) { P[2 * i] = (r0.uniform() * 2 - 1) * 3; P[2 * i + 1] = (r0.uniform() * 2 - 1) * 2.3; }
  return P;
}
const smoothDensity = (s) => mixDensity(MIX3.means, Math.sqrt(MIX3.std ** 2 + s * s), MIX3.w);
const pmaxSig = (s) => smoothDensity(s)(...MIX3.means[0]) * 1.05;
function drawParticles(ctx, pal, v, P, r = 2.1) {
  for (let i = 0; i < P.length / 2; i++) {
    const [X, Y] = v.px(P[2 * i], P[2 * i + 1]);
    ctx.fillStyle = modeColor(pal, nearest3(P[2 * i], P[2 * i + 1])); ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ---------- Step 6: one noise level ----------
function smoothing(viz) {
  const c = makeCanvas(4 / 3), ro = readout(), N = 300, KMAX = 400;
  let sig = 0.25, P, k, rng, g, lab0;
  const reset = () => { P = uniformStart(N); k = 0; rng = makeRng(9); g = levelGMM(sig); lab0 = Array.from({ length: N }, (_, i) => nearest3(P[2 * i], P[2 * i + 1])); };
  const step = () => {
    const eta = ANNEAL.gamma * sig * sig, sq = Math.sqrt(2 * eta);
    for (let i = 0; i < N; i++) {
      const sc = g.fields(P[2 * i], P[2 * i + 1], 1).score;
      P[2 * i] += eta * sc[0] + sq * rng.gauss(); P[2 * i + 1] += eta * sc[1] + sq * rng.gauss();
    }
    k++;
  };
  reset();
  const r = runner({ step, done: () => k >= KMAX, draw: () => draw(), interval: 30, perFrame: 2 });
  viz.append(c.cv, controls(
    slider('sigma', { min: Math.log10(0.1), max: Math.log10(3), step: 0.01, value: Math.log10(sig), map: (v) => 10 ** v }, (v) => { sig = v; reset(); draw(); r.kick(); }),
    r.play, button('Reset', () => { reset(); draw(); r.kick(); }),
  ), ro, note(`Three narrow modes (standard deviation ${MIX3.std}), exact score of p_σ. Colour marks the nearest mode of p.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), ph = h - 118, v = view(0, 0, w, ph, 3.4, 2.5);
    ctx.globalAlpha = isHL('psig') ? 1 : 0.85; ctx.drawImage(heat(`s${sig.toFixed(3)}`, w, ph, v, smoothDensity(sig), pmaxSig(sig)), 0, 0, w, ph); ctx.globalAlpha = 1;
    // Score field of p_σ.
    const fieldHL = isHL('ssig', 'nabla');
    for (let i = 0; i < 19; i++) for (let j = 0; j < 12; j++) {
      const x = -3.2 + 6.4 * i / 18, y = -2.3 + 4.6 * j / 11, sc = g.fields(x, y, 1).score, m = Math.hypot(...sc) || 1;
      const len = 14 * Math.tanh(0.6 * m * Math.sqrt(MIX3.std ** 2 + sig * sig)), [X, Y] = v.px(x, y);
      ctx.globalAlpha = 0.25 + 0.45 * len / 14;
      arrow(ctx, X, Y, X + len * sc[0] / m, Y - len * sc[1] / m, fieldHL ? pal.mark : pal.muted, 1.2, 4.5);
    }
    ctx.globalAlpha = 1;
    // Rings at the modes of p, sized by weight.
    MIX3.means.forEach(([mx, my], j) => { const [X, Y] = v.px(mx, my); ctx.beginPath(); ctx.arc(X, Y, 14 + 18 * W3[j], 0, 7); ctx.strokeStyle = hlc('p', pal, modeColor(pal, j)); ctx.lineWidth = isHL('p') ? 2.4 : 1.2; ctx.stroke(); });
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, w, ph); ctx.clip(); drawParticles(ctx, pal, v, P); ctx.restore();
    tag(ctx, pal, w < 520 ? `σ = ${sig.toFixed(2)}, ${k}/${KMAX}` : `σ = ${sig.toFixed(2)}, step ${k} of ${KMAX}`, w - 12, 22, hlc('sigma', pal, pal.muted), 13.5, 'right');
    tag(ctx, pal, w < 520 ? 'p_σ' : 'heat map: p_σ', 14, 22, hlc('psig', pal, pal.muted), 13.5);
    frame(ctx, pal, 0, ph, w, h - ph);
    const mass = massPerMode(P);
    massBars(ctx, pal, 16, ph + 8, w - 32, 108, mass, W3, w < 520 ? 'mass (filled), target (dashed)' : 'mass per mode (filled) against the target weight (dashed line)');
    // Mixing: share of particles whose nearest mode changed since the start. Shape: mean distance to the nearest mode.
    let moved = 0, dist = 0;
    for (let i = 0; i < N; i++) { const j = nearest3(P[2 * i], P[2 * i + 1]), [mx, my] = MIX3.means[j]; if (j !== lab0[i]) moved++; dist += Math.hypot(P[2 * i] - mx, P[2 * i + 1] - my) / N; }
    ro.innerHTML = `σ = <b>${sig.toFixed(2)}</b> · η = γσ² = <b>${(ANNEAL.gamma * sig * sig).toFixed(4)}</b> · mass = <b>${mass.map((m) => m.toFixed(2)).join(', ')}</b> (target ${W3.map((m) => m.toFixed(2)).join(', ')}) · changed mode: <b>${Math.round(100 * moved / N)}%</b> · mean distance to nearest mode: <b>${dist.toFixed(2)}</b> (samples of p: ${(MIX3.std * Math.sqrt(Math.PI / 2)).toFixed(2)})`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 7: annealed against single-level ULA ----------
function annealing(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let L = 8, runs = null, f = 0;
  const get = () => (runs ??= annealRuns(L));
  const total = () => get().anneal.frames.length - 1;
  const r = runner({ step: () => { f++; }, done: () => f >= total(), draw: () => draw(), interval: 30 });
  viz.append(c.cv, controls(
    slider('L', { min: 1, max: ANNEAL.Lmax, step: 1, value: L, fmt: (v) => String(v) }, (v) => { L = v; runs = null; f = 0; draw(); r.kick(); }),
    r.play, button('Reset', () => { f = 0; draw(); r.kick(); }),
  ), ro, note(`${ANNEAL.N} particles, K = ${ANNEAL.K} steps per level, η = γσ² with γ = ${ANNEAL.gamma}, exact scores at every level. Colour marks the nearest mode of p.`));
  function panel(ctx, pal, x, y, w, h, run, title, annealed) {
    const fr = Math.min(f, run.frames.length - 1), li = run.lev[fr], s = run.levels[li];
    const v = view(x, y + 22, w, h - 22, 3.4, 2.5);
    ctx.save(); ctx.beginPath(); ctx.rect(x, y + 22, w, h - 22); ctx.clip();
    ctx.globalAlpha = isHL('psig') ? 1 : 0.85;
    const hm = heat(`s${s.toFixed(3)}`, w, h - 22, view(0, 0, w, h - 22, 3.4, 2.5), smoothDensity(s), pmaxSig(s));
    ctx.drawImage(hm, x, y + 22, w, h - 22); ctx.globalAlpha = 1;
    drawParticles(ctx, pal, v, run.frames[fr], 1.9);
    ctx.restore(); frame(ctx, pal, x, y + 22, w, h - 22);
    label(ctx, title, x, y + 14, pal.fg, 13.5, 'left', 600);
    label(ctx, `σ = ${s.toFixed(2)}`, x + w - 8, y + 40, hlc('sigma', pal, pal.muted), 12.5, 'right');
    if (annealed) {
      // Ladder of levels σ_1 > … > σ_L; the current one is filled.
      const n = run.levels.length, lx = x + 10, ly = y + 34;
      for (let j = 0; j < n; j++) {
        ctx.beginPath(); ctx.arc(lx + j * 11, ly, 3.6, 0, 7);
        ctx.fillStyle = j === li ? hlc(['ell', 'L'], pal, pal.acc) : pal.line; ctx.fill();
      }
      label(ctx, `level ℓ = ${li + 1} of ${n}`, lx, ly + 18, hlc(['ell', 'L'], pal, pal.muted), 12);
    }
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), R = get(), gap = 18, pw = (w - gap) / 2, ph = h - 120;
    panel(ctx, pal, 0, 0, pw, ph, R.single, `One level, σ = ${ANNEAL.sigLo}`, false);
    panel(ctx, pal, pw + gap, 0, pw, ph, R.anneal, `Annealed, ${L} level${L > 1 ? 's' : ''}`, true);
    const fr = Math.min(f, total()), ms = massPerMode(R.single.frames[fr]), ma = massPerMode(R.anneal.frames[fr]);
    massBars(ctx, pal, 8, ph + 8, pw - 16, 110, ms, W3, 'mass per mode / target');
    massBars(ctx, pal, pw + gap + 8, ph + 8, pw - 16, 110, ma, W3, 'mass per mode / target');
    ro.innerHTML = `step <b>${fr}</b> of ${total()} · L = <b>${L}</b> · one level: <b>${ms.map((m) => m.toFixed(2)).join(', ')}</b> · annealed: <b>${ma.map((m) => m.toFixed(2)).join(', ')}</b> · target ${W3.map((m) => m.toFixed(2)).join(', ')}`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

export const SCENES = { energy, score, langevin, stationary, ula, smoothing, annealing };
