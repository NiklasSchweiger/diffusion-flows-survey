// Live scenes of the §2.7 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner, arrow } from '../lib/tutorial.js';
import {
  alpha, sigma, train, denoise, SEED_A, NSTEP, T_GRID, NS_LIST, D_LIST, NS2, NS3, NS4, run2, DATA_RUN, dataCopies, run3, logp2,
  LOGP_REF, MODELS, H_SMOOTH, run4, PAIR_DIST, SET5, NS5, H5_LIST, run5, BASE5, N_IMG, DIGITS, K_ELS, M_ELS, elsStart, elsStep, nearestDigit,
} from './sim.js';

// ---------- Helpers ----------
const classColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_\{([^}]*)\}|_/g, '$1')).width + 12, x0 = align === 'right' ? x - w + 6 : align === 'center' ? x - w / 2 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2);
const pct = (a, b) => `${Math.round(100 * a / Math.max(b, 1))}%`;
// World box [-bx, bx] x [-by, by] fitted into a rectangle of the canvas.
function view(x0, y0, w, h, bx = 3.2, by = 2.9) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y], wx: (X, Y) => [(X - cx) / s, (cy - Y) / s] };
}
// Panels side by side on wide canvases and stacked on narrow ones.
const WIDE = 640;
function panels(w, h, k, gap = 14) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + gap), w, ph]);
}
// Heat map of p_data inside a rectangle, cached by size, view and theme. Alpha follows p^0.7.
const heatCache = new Map();
function heat(ctx, [x0, y0, w, h], v, alphaMul = 0.8) {
  const k = `${x0},${y0},${w}x${h}|${v.s.toFixed(3)}|${live.theme}`;
  if (!heatCache.has(k)) {
    if (heatCache.size > 30) heatCache.clear();
    const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res), off = document.createElement('canvas'); off.width = W; off.height = H;
    const g = off.getContext('2d'), im = g.createImageData(W, H), pmax = Math.exp(logp2(-1.4, -0.55)) * 1.05;
    const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.5 : 0.34;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const [x, y] = v.wx(x0 + i * res, y0 + j * res), o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(Math.min(1, Math.exp(logp2(x, y)) / pmax), 0.7));
    }
    g.putImageData(im, 0, 0);
    heatCache.set(k, off);
  }
  ctx.globalAlpha = alphaMul; ctx.drawImage(heatCache.get(k), x0, y0, w, h); ctx.globalAlpha = 1;
}
const listSlider = (id, list, value, on, fmt = String) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt }, on);
// Time slider over the ODE grid, which is uniform in log sigma.
const kSlider = (value, on) => slider('t', { min: 0, max: NSTEP, step: 1, value, fmt: (k) => T_GRID[k].toFixed(3) }, on);
function trainingDots(ctx, pal, v, T, r = 2.4, alphaV = 1, ids = ['y']) {
  ctx.globalAlpha = alphaV;
  for (let i = 0; i < T.n; i++) dot(ctx, ...v.px(T.y[i * T.d], T.y[i * T.d + 1]), isHL(...ids) ? r + 1 : r, hlc(ids, pal, classColor(pal, T.k[i])));
  ctx.globalAlpha = 1;
}
// Paths of the first two coordinates up to grid index k; F.path holds nk = NSTEP + 1 points per path.
function paths(ctx, v, F, m, k, colorOf, alphaV, lw) {
  const nk = F.nk; ctx.lineWidth = lw; ctx.globalAlpha = alphaV;
  for (let i = 0; i < m; i++) {
    ctx.beginPath();
    for (let q = 0; q <= k; q++) (q ? ctx.lineTo : ctx.moveTo).call(ctx, ...v.px(F.path[(i * nk + q) * 2], F.path[(i * nk + q) * 2 + 1]));
    ctx.strokeStyle = colorOf(i); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
const at = (F, i, k) => [F.path[(i * F.nk + k) * 2], F.path[(i * F.nk + k) * 2 + 1]];
// Runner that sweeps the grid index k of a slider and then stops.
function sweep(get, set) {
  const r = runner({ step: () => set(get() >= NSTEP ? 0 : get() + 1), done: () => get() >= NSTEP, draw: () => {}, interval: 45 });
  return r;
}

// ---------- Step 1: the empirical denoiser at one point ----------
function empirical(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8.4 : 0.62)), ro = readout();
  let t = 0.7, n = 32, probe = [0.42, 0.12];
  const out = new Float64Array(2), w = new Float64Array(512);
  viz.append(c.cv, controls(
    slider('t', { min: 0, max: 0.99, step: 0.01, value: t }, (v) => { t = v; draw(); }),
    listSlider('n', NS_LIST, n, (v) => { n = v; draw(); })), ro,
  note('Exact: the training set is the first n points of a fixed sample of the 2D mixture (dots, coloured by component; the heat map is p<sub>data</sub>), and the weights and x̂<sub>1</sub> are the closed form of the formula. Drag in the plot to move x<sub>t</sub>. Halo areas are proportional to the weights.'));
  let L = null;
  const layout = () => { const { w: W, h: H } = c; return W >= WIDE ? { plot: [0, 0, Math.round(W * 0.7), H], bars: [Math.round(W * 0.7) + 24, 30, W - Math.round(W * 0.7) - 36, H - 64] } : { plot: [0, 0, W, Math.round(H * 0.7)], bars: [44, Math.round(H * 0.7) + 30, W - 56, H - Math.round(H * 0.7) - 52] }; };
  const move = (e) => {
    if (!L || !(e.buttons & 1) && e.type !== 'pointerdown') return;
    const r = c.cv.getBoundingClientRect(), [x0, y0, pw, ph] = L.plot, X = e.clientX - r.left, Y = e.clientY - r.top;
    if (X < x0 || X > x0 + pw || Y < y0 || Y > y0 + ph) return;
    const p = view(...L.plot).wx(X, Y); probe = [Math.max(-3, Math.min(3, p[0])), Math.max(-2.8, Math.min(2.8, p[1]))]; draw();
  };
  c.cv.addEventListener('pointerdown', move); c.cv.addEventListener('pointermove', move);
  function draw() {
    c.fit(); L = layout();
    const { ctx } = c, pal = palette(), T = train(SEED_A, 2, n), v = view(...L.plot), a = alpha(t), s = sigma(t);
    denoise(T, probe, t, 0, out, w);
    heat(ctx, L.plot, v, 0.5);
    clip(ctx, ...L.plot);
    // Halos: area proportional to the weight.
    for (let i = 0; i < n; i++) if (w[i] > 0.002) { const [X, Y] = v.px(T.y[2 * i], T.y[2 * i + 1]); ctx.globalAlpha = isHL('w') ? 0.5 : 0.3; dot(ctx, X, Y, 3 + 26 * Math.sqrt(w[i]), hlc('w', pal, classColor(pal, T.k[i]))); }
    ctx.globalAlpha = 1;
    trainingDots(ctx, pal, v, T, 2.6);
    // The circle of radius sigma/alpha around x/alpha in data space.
    if (a > 0.02) {
      const [cx, cy] = v.px(probe[0] / a, probe[1] / a), R = Math.min(2000, v.s * s / a);
      ctx.strokeStyle = hlc(['sig', 'alpha'], pal, pal.muted); ctx.lineWidth = isHL('sig', 'alpha') ? 2 : 1.2; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cx - 4, cy - 4); ctx.lineTo(cx + 4, cy + 4); ctx.moveTo(cx - 4, cy + 4); ctx.lineTo(cx + 4, cy - 4); ctx.stroke();
    }
    const [PX, PY] = v.px(...probe), [HX, HY] = v.px(out[0], out[1]);
    arrow(ctx, PX, PY, HX, HY, hlc('hx1', pal, pal.fg), isHL('hx1') ? 2.4 : 1.6, 9);
    ring(ctx, PX, PY, isHL('x') ? 8 : 6.5, hlc('x', pal, pal.acc), 2.2);
    // Star at the denoiser output.
    ctx.beginPath(); for (let q = 0; q < 10; q++) { const r = q % 2 ? 3 : 7.5, ang = -Math.PI / 2 + q * Math.PI / 5; (q ? ctx.lineTo : ctx.moveTo).call(ctx, HX + r * Math.cos(ang), HY + r * Math.sin(ang)); }
    ctx.closePath(); ctx.fillStyle = hlc('hx1', pal, pal.fg); ctx.fill();
    ctx.restore();
    frame(ctx, pal, ...L.plot);
    tag(ctx, pal, `t = ${t.toFixed(2)} · σ_t/α_t = ${a > 0.005 ? (s / a).toFixed(2) : '∞'}`, L.plot[0] + 14, L.plot[1] + 24, pal.fg, 13);
    // Largest weights in decreasing order.
    const [bx, by, bw, bh] = L.bars, K = 10, ord = Array.from({ length: n }, (_, i) => i).sort((p, q) => w[q] - w[p]).slice(0, K), gw = bw / K;
    ord.forEach((i, j) => {
      const hh = bh * w[i];
      ctx.fillStyle = hlc('w', pal, classColor(pal, T.k[i])); ctx.globalAlpha = 0.85; ctx.fillRect(bx + j * gw + 3, by + bh - hh, gw - 6, hh); ctx.globalAlpha = 1;
    });
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, by + bh + 0.5); ctx.lineTo(bx + bw, by + bh + 0.5); ctx.stroke();
    label(ctx, '1', bx - 8, by + 5, pal.muted, 11, 'right'); label(ctx, '0', bx - 8, by + bh + 4, pal.muted, 11, 'right');
    hdash(ctx, pal.line, bx, bx + bw, by);
    label(ctx, `the ${Math.min(K, n)} largest weights w_i`, bx, by - 12, pal.fg, 12.5);
    let s2 = 0, mx = 0, ia = 0; for (let i = 0; i < n; i++) { s2 += w[i] * w[i]; if (w[i] > mx) { mx = w[i]; ia = i; } }
    const dn = Math.hypot(out[0] - T.y[2 * ia], out[1] - T.y[2 * ia + 1]);
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · n = <b>${n}</b> · effective number of points 1/Σw<sub>i</sub>² = <b>${(1 / s2).toFixed(1)}</b> of ${n} · largest weight <b>${mx.toFixed(3)}</b><br>` +
      `x̂<sub>1</sub> = (${f2(out[0])}, ${f2(out[1])}) · distance of x̂<sub>1</sub> to the training point with the largest weight <b>${dn.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: paths of the empirical field and of the p_data field ----------
function copies(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 2.15 : 0.54)), ro = readout();
  let n = 64, k = NSTEP;
  const kS = kSlider(k, (v) => { k = v; draw(); });
  const r = sweep(() => k, (v) => kS.set(v));
  r.playing = false; r.play.textContent = 'Play';
  viz.append(c.cv, controls(kS, r.play, listSlider('n', NS_LIST, n, (v) => { n = v; draw(); })), ro,
    note(`${NS2} fixed noise draws, integrated with ${NSTEP} RK4 steps uniform in log σ<sub>t</sub> down to σ = 0.001, with the exact empirical field (left) and the exact field of the mixture (right). Copies are counted with the test d<sub>1</sub>/d<sub>2</sub> &lt; 1/3 against the training set on the left; on the right the same test is applied to the samples of p<sub>data</sub>, which never saw the training set.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), R = run2(n), D = DATA_RUN.F, P = panels(w, h, 2);
    const copyCol = hlc(['X', 'ratio'], pal, pal.acc);
    [[R.F, P[0], 'empirical optimum, n = ' + n], [D, P[1], 'exact field of p_{data}']].forEach(([F, box, name], j) => {
      const v = view(...box);
      if (j === 1) heat(ctx, box, v, 0.55);
      clip(ctx, ...box);
      if (j === 0) trainingDots(ctx, pal, v, R.T, n > 128 ? 1.8 : 2.6, 0.9);
      paths(ctx, v, F, NS2, k, () => hlc(['X', 'vel'], pal, pal.muted), isHL('X', 'vel') ? 0.4 : 0.18, 1);
      for (let i = 0; i < NS2; i++) {
        const [X, Y] = v.px(...at(F, i, k)), isCopy = j === 0 && k === NSTEP && R.near.ratio[i] < 1 / 3;
        if (isCopy) ring(ctx, X, Y, 4.2, copyCol, 1.4); else dot(ctx, X, Y, 2.2, j === 0 ? pal.fg : hlc('X', pal, pal.fg));
      }
      ctx.restore();
      frame(ctx, pal, ...box);
      tag(ctx, pal, name, box[0] + 14, box[1] + 24, pal.fg, 13);
    });
    tag(ctx, pal, `t = ${T_GRID[k].toFixed(3)}`, P[1][0] + P[1][2] - 14, P[1][1] + 24, hlc('t', pal, pal.muted), 13, 'right');
    ro.innerHTML = `empirical optimum: copies <b>${R.copies}</b> of ${NS2} (${pct(R.copies, NS2)}) · distinct training points reached <b>${R.hit}</b> of ${n}<br>` +
      `exact field of p<sub>data</sub>: samples that pass the same copy test <b>${dataCopies(n)}</b> of ${NS2} (${pct(dataCopies(n), NS2)}), by chance` + (k < NSTEP ? ' · the copy test is applied at t = 1' : '');
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 3: speciation and collapse ----------
function collapse(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 2.15 : 0.54)), ro = readout();
  let n = 64, d = 2, k = NSTEP;
  const kS = kSlider(k, (v) => { k = v; draw(); });
  const r = sweep(() => k, (v) => kS.set(v));
  r.playing = false; r.play.textContent = 'Play';
  viz.append(c.cv, controls(kS, r.play, listSlider('n', NS_LIST, n, (v) => { n = v; draw(); }), listSlider('d', D_LIST, d, (v) => { d = v; draw(); })), ro,
    note(`${NS3} fixed noise draws in dimension d, integrated with ${NSTEP} RK4 steps of the exact empirical field. The paths show the first two coordinates, coloured by the class of the training point they end on. The curves average max<sub>k</sub> P(k | x<sub>t</sub>) and max<sub>i</sub> w<sub>i</sub> over the paths; t<sub>S</sub> and t<sub>C</sub> are where they cross 0.9, interpolated between grid times. Both are estimates from ${NS3} paths.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), R = run3(n, d), P = panels(w, h, 2, 26), box = P[0], v = view(...box);
    clip(ctx, ...box);
    trainingDots(ctx, pal, v, R.T, n > 128 ? 1.6 : 2.4, 0.55);
    paths(ctx, v, R.F, NS3, k, (i) => hlc(['X', 'k'], pal, classColor(pal, R.end[i])), isHL('X', 'k') ? 0.6 : 0.32, 1);
    for (let i = 0; i < NS3; i++) dot(ctx, ...v.px(...at(R.F, i, k)), 2.6, classColor(pal, R.end[i]));
    ctx.restore();
    frame(ctx, pal, ...box);
    tag(ctx, pal, `n = ${n} · d = ${d}${d > 2 ? ' (first two coordinates)' : ''}`, box[0] + 14, box[1] + 24, pal.fg, 13);
    // Curves over t.
    const [cx0, cy0, cw, ch] = P[1], L = cx0 + 34, Rr = cx0 + cw - 10, top = cy0 + 34, bot = cy0 + ch - 34;
    const X = (tt) => L + (Rr - L) * tt, Y = (q) => bot - (bot - top) * q;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, top + 0.5, Rr - L, bot - top);
    hdash(ctx, pal.line, L, Rr, Y(0.9));
    [0, 0.5, 1].forEach((q) => { label(ctx, String(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'); label(ctx, String(q), X(q), bot + 16, pal.muted, 11, 'center'); });
    label(ctx, 't', Rr, bot + 30, pal.muted, 11.5, 'right');
    const line = (arr, col, lw) => { ctx.beginPath(); arr.forEach((q, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, X(T_GRID[j]), Y(q))); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); };
    const cS = hlc(['Pk', 'tS'], pal, pal.ramp[2]), cC = hlc(['wmax', 'tC'], pal, pal.acc);
    line(R.cls, cS, isHL('Pk', 'tS') ? 3 : 2); line(R.pt, cC, isHL('wmax', 'tC') ? 3 : 2);
    if (R.tS != null) { vdash(ctx, cS, X(R.tS), top, bot, 1.3); tag(ctx, pal, `t_S ≈ ${R.tS.toFixed(2)}`, X(R.tS) - 6, top + 20, cS, 12, 'right'); }
    if (R.tC != null) { vdash(ctx, cC, X(R.tC), top, bot, 1.3); tag(ctx, pal, `t_C ≈ ${R.tC.toFixed(2)}`, X(R.tC) - 6, top + 40, cC, 12, 'right'); }
    vdash(ctx, hlc('t', pal, pal.muted), X(T_GRID[k]), top, bot, 1, [1, 3]);
    label(ctx, 'max_k P(k | x_t)', L + 8, Y(0.12), cS, 12); label(ctx, 'max_i w_i', L + 8, Y(0.12) + 16, cC, 12);
    label(ctx, 'averages over the paths', L, top - 12, pal.fg, 12.5);
    const fmt = (q) => (q == null ? 'not reached' : q.toFixed(2));
    ro.innerHTML = `n = <b>${n}</b> · d = <b>${d}</b> · speciation t<sub>S</sub> ≈ <b>${fmt(R.tS)}</b> · collapse t<sub>C</sub> ≈ <b>${fmt(R.tC)}</b> · gap <b>${R.tS != null && R.tC != null ? (R.tC - R.tS).toFixed(2) : '–'}</b><br>` +
      `at t = ${T_GRID[k].toFixed(2)}: mean class certainty <b>${R.cls[k].toFixed(2)}</b> · mean largest weight <b>${R.pt[k].toFixed(2)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 4: two models trained on disjoint sets ----------
const MODEL_NAME = { empirical: 'empirical', smoothed: `smoothed, h = ${H_SMOOTH}`, gaussian: 'Gaussian' };
function agree(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8.4 : 0.48)), ro = readout();
  let n = 64, d = 2, model = 'empirical';
  const btns = MODELS.map((m) => button(MODEL_NAME[m], () => { model = m; btns.forEach((b, j) => b.classList.toggle('on', MODELS[j] === m)); draw(); }, m === model ? 'on' : ''));
  viz.append(c.cv, controls(...btns, listSlider('n', NS_LIST, n, (v) => { n = v; draw(); }), listSlider('d', D_LIST, d, (v) => { d = v; draw(); })), ro,
    note(`Training sets A and B: the first n points of two independent samples of the data in dimension d. ${NS4} shared noise draws. The empirical and smoothed models are integrated with ${NSTEP} RK4 steps of their exact fields; the Gaussian model uses its exact flow map. The plot shows the first two coordinates. Disagreement is in all d coordinates, divided by the mean distance between two independent draws of the data. Log density: the mean of log p<sub>data</sub> over the first two coordinates of the samples of A; for real data it is ${LOGP_REF.toFixed(2)} (dashed tick). These numbers are averages over ${NS4} samples.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE;
    const plot = wide ? [0, 0, Math.round(w * 0.64), h] : [0, 0, w, Math.round(h * 0.56)], v = view(...plot);
    const all = Object.fromEntries(MODELS.map((m) => [m, run4(m, n, d)])), R = all[model], pd = PAIR_DIST(d);
    const cA = hlc(['Phi'], pal, pal.acc), cB = hlc(['Phi'], pal, pal.ramp[2]);
    heat(ctx, plot, v, 0.45);
    clip(ctx, ...plot);
    ctx.globalAlpha = 0.35;
    for (let i = 0; i < n; i++) { dot(ctx, ...v.px(R.A.y[i * d], R.A.y[i * d + 1]), 1.8, pal.acc); dot(ctx, ...v.px(R.B.y[i * d], R.B.y[i * d + 1]), 1.8, pal.ramp[2]); }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.muted; ctx.lineWidth = isHL('Phi') ? 1.6 : 1; ctx.globalAlpha = 0.7;
    for (let i = 0; i < NS4; i++) { ctx.beginPath(); ctx.moveTo(...v.px(R.XA[i * d], R.XA[i * d + 1])); ctx.lineTo(...v.px(R.XB[i * d], R.XB[i * d + 1])); ctx.stroke(); }
    ctx.globalAlpha = 1;
    for (let i = 0; i < NS4; i++) { dot(ctx, ...v.px(R.XA[i * d], R.XA[i * d + 1]), 3.4, cA); ring(ctx, ...v.px(R.XB[i * d], R.XB[i * d + 1]), 3.8, cB, 1.5); }
    ctx.restore();
    frame(ctx, pal, ...plot);
    tag(ctx, pal, `${MODEL_NAME[model]} · filled: model A · open: model B`, plot[0] + 14, plot[1] + 24, pal.fg, 13);
    // Three metrics for all three models.
    const bx = wide ? plot[2] + 30 : 16, bw = wide ? w - plot[2] - 42 : w - 32, by = wide ? 40 : plot[3] + 40, bh = (wide ? h - 60 : h - plot[3] - 52) / 3;
    const metrics = [
      ['disagreement / distance of two data points', (q) => q.dist / pd, 1, (q) => (q.dist / pd).toFixed(2), null],
      ['share of copies of set A', (q) => q.copies / NS4, 1, (q) => pct(q.copies, NS4), null],
      ['mean log p_{data} of the samples', (q) => (q.logp + 6) / 4, 1, (q) => q.logp.toFixed(2), (LOGP_REF + 6) / 4],
    ];
    metrics.forEach(([name, f, , fmt, ref], j) => {
      const y0 = by + j * bh, rowH = Math.min(15, (bh - 26) / 3);
      label(ctx, name, bx, y0, pal.fg, 12);
      MODELS.forEach((m, q) => {
        const yy = y0 + 10 + q * (rowH + 4), val = Math.max(0, Math.min(1, f(all[m]))), lw = 70;
        label(ctx, MODEL_NAME[m].split(',')[0], bx, yy + rowH - 3, m === model ? pal.fg : pal.muted, 11);
        ctx.fillStyle = pal.line; ctx.fillRect(bx + lw, yy, bw - lw - 44, rowH);
        ctx.fillStyle = [pal.ramp[0], pal.ramp[1], pal.ramp[3]][q]; ctx.globalAlpha = m === model ? 0.95 : 0.55; ctx.fillRect(bx + lw, yy, (bw - lw - 44) * val, rowH); ctx.globalAlpha = 1;
        label(ctx, fmt(all[m]), bx + bw, yy + rowH - 3, m === model ? pal.fg : pal.muted, 11, 'right');
      });
      if (ref != null) vdash(ctx, pal.fg, bx + 70 + (bw - 70 - 44) * ref, y0 + 6, y0 + 10 + 3 * (rowH + 4), 1.4, [3, 3]);
    });
    ro.innerHTML = `${MODEL_NAME[model]} · n = <b>${n}</b> per set · d = <b>${d}</b> · mean ‖Φ<sup>A</sup>(x<sub>0</sub>) − Φ<sup>B</sup>(x<sub>0</sub>)‖ = <b>${R.dist.toFixed(2)}</b> (two independent data points: ${pd.toFixed(2)})<br>` +
      `copies of set A <b>${R.copies}</b> of ${NS4} · mean log p<sub>data</sub> of the samples <b>${R.logp.toFixed(2)}</b> (real data ${LOGP_REF.toFixed(2)})`;
  }
  return { draw };
}

// ---------- Step 5: memorization of isolated and duplicated points ----------
const GROUP = ['in the modes', 'isolated', 'duplicated'];
function local(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8.4 : 0.62)), ro = readout();
  let h = 0.15;
  viz.append(c.cv, controls(listSlider('h', H5_LIST, h, (v) => { h = v; draw(); }, (v) => v.toFixed(2))), ro,
    note(`${NS5} fixed noise draws integrated with ${NSTEP} RK4 steps of the exact denoiser of the smoothed training set. Each sample is assigned to the group of its nearest distinct training point and counts as a copy when d<sub>1</sub>/d<sub>2</sub> &lt; 1/3. For comparison, ${Math.round(100 * BASE5)}% of fresh draws of p<sub>data</sub> pass the test against this training set (the dashed tick). The circles have radius h.`));
  function draw() {
    c.fit();
    const { ctx, w, h: H } = c, pal = palette(), wide = w >= WIDE, R = run5(h);
    const plot = wide ? [0, 0, Math.round(w * 0.68), H] : [0, 0, w, Math.round(H * 0.68)], v = view(...plot, 3.5, 3.0);
    heat(ctx, plot, v, 0.35);
    clip(ctx, ...plot);
    const ty = SET5.y;
    for (let i = 0; i < SET5.n; i++) {
      const [X, Y] = v.px(ty[2 * i], ty[2 * i + 1]), g = SET5.grp[i];
      if (g === 0) dot(ctx, X, Y, isHL('y') ? 2.6 : 1.8, hlc('y', pal, pal.muted));
      else if (g === 1 || i === SET5.n - 1) {
        if (h > 0) { ctx.globalAlpha = 0.6; ring(ctx, X, Y, Math.max(2, h * v.s), hlc('h', pal, pal.muted), isHL('h') ? 2 : 1); ctx.globalAlpha = 1; }
        dot(ctx, X, Y, 3.4, hlc('y', pal, g === 1 ? pal.ramp[3] : pal.ramp[2]));
        if (g === 2) tag(ctx, pal, '×10', X + 10, Y - 6, pal.ramp[2], 12);
      }
    }
    for (let i = 0; i < NS5; i++) {
      const [X, Y] = v.px(R.X[2 * i], R.X[2 * i + 1]);
      if (R.copy[i]) dot(ctx, X, Y, 2.6, hlc('ratio', pal, pal.acc)); else ring(ctx, X, Y, 2.6, hlc('phath', pal, pal.fg), 1);
    }
    ctx.restore();
    frame(ctx, pal, ...plot);
    tag(ctx, pal, `h = ${h.toFixed(2)} · filled: copy · open: not a copy`, plot[0] + 14, plot[1] + 24, pal.fg, 13);
    // Share of copies per group.
    const bx = wide ? plot[2] + 28 : 20, bw = wide ? w - plot[2] - 40 : w - 40, by = wide ? 60 : plot[3] + 36, rowH = 20, gap = wide ? 46 : 30, lw = wide ? 0 : 100;
    label(ctx, 'share of samples that are copies', bx, by - (wide ? 22 : 14), pal.fg, 12.5);
    GROUP.forEach((name, g) => {
      const y0 = by + g * (rowH + gap), share = R.cp[g] / Math.max(R.tot[g], 1), col = [pal.muted, pal.ramp[3], pal.ramp[2]][g], x0 = bx + lw, ww = bw - lw - 44;
      label(ctx, `${name}${wide ? ` (${R.cp[g]} of ${R.tot[g]})` : ''}`, bx, wide ? y0 - 6 : y0 + 14, pal.muted, 11.5);
      const yb = wide ? y0 : y0;
      ctx.fillStyle = pal.line; ctx.fillRect(x0, yb, ww, rowH);
      ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.fillRect(x0, yb, ww * share, rowH); ctx.globalAlpha = 1;
      label(ctx, pct(R.cp[g], R.tot[g]), x0 + ww + 40, yb + 15, pal.fg, 12, 'right');
      if (g === 0) vdash(ctx, pal.fg, x0 + ww * BASE5, yb - 4, yb + rowH + 4, 1.4, [3, 3]);
    });
    ro.innerHTML = `h = <b>${h.toFixed(2)}</b> · copies in the modes <b>${R.cp[0]}</b> of ${R.tot[0]} (${pct(R.cp[0], R.tot[0])}, fresh data ${Math.round(100 * BASE5)}%) · ` +
      `near isolated points <b>${R.cp[1]}</b> of ${R.tot[1]} · near the duplicated point <b>${R.cp[2]}</b> of ${R.tot[2]}`;
  }
  return { draw };
}

// ---------- Step 6: patch denoiser on digits ----------
const P_LIST = [3, 5, 7, 9, 0];
const imgCache = document.createElement('canvas'); imgCache.width = imgCache.height = N_IMG;
const hexRGB = (s) => { const m = s.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i); return m ? m.slice(1).map((q) => parseInt(q, 16)) : [128, 128, 128]; };
function drawImg(ctx, pal, im, x, y, size) {
  const g = imgCache.getContext('2d'), d = g.createImageData(N_IMG, N_IMG), bg = hexRGB(pal.bg), fg = hexRGB(pal.fg);
  for (let i = 0; i < im.length; i++) {
    const u = Math.max(0, Math.min(1, (im[i] + 1) / 2));
    for (let c = 0; c < 3; c++) d.data[4 * i + c] = Math.round(bg[c] + (fg[c] - bg[c]) * u);
    d.data[4 * i + 3] = 255;
  }
  g.putImageData(d, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(imgCache, x, y, size, size); ctx.imageSmoothingEnabled = true;
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}
function patches(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.9 : 0.78)), ro = readout();
  let P = 5, s = 1, st = elsStart();
  const r = runner({ step: () => elsStep(st, P, s), done: () => st.k >= K_ELS, draw: () => draw(), interval: 30 });
  const restart = () => { st = elsStart(); draw(); r.kick(); };
  viz.append(c.cv, controls(
    listSlider('P', P_LIST, P, (v) => { P = v; restart(); }, (v) => (v ? `${v} × ${v}` : 'full')),
    slider('s', { min: 0, max: 2, step: 1, value: s, fmt: String }, (v) => { s = v; restart(); }),
    r.play, button('Restart', restart)), ro,
  note(`Exact closed form of the patch denoiser for the eight training digits (12 × 12 pixels, values in [−1, 1], pixels outside the image count as background). ${M_ELS} fixed noise images, ${K_ELS} Euler steps uniform in log σ<sub>t</sub> down to σ = 0.002. A sample counts as a copy when its distance to the nearest training digit is below a third of the distance to the second nearest. A shifted copy does not count as a copy under this test.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE;
    // Training digits.
    const nTop = DIGITS.length, ts = Math.min(wide ? 56 : 34, (w - 24 - (nTop - 1) * 8) / nTop), tx0 = (w - (nTop * ts + (nTop - 1) * 8)) / 2;
    label(ctx, 'training set', tx0, 18, pal.muted, 12);
    DIGITS.forEach((im, i) => drawImg(ctx, pal, im, tx0 + i * (ts + 8), 26, ts));
    if (isHL('y')) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(tx0 - 3, 23, nTop * (ts + 8) - 2, ts + 6); }
    // Samples: x_t above, xhat_1 below.
    const cols = wide ? M_ELS : 3, rows = M_ELS / cols, top = 26 + ts + 34, gap = 14;
    const cw = (w - 24 - (cols - 1) * gap) / cols, rowH = (h - top - 8) / rows, big = Math.min(cw * 0.95, (rowH - 36) / 1.5), small = big * 0.5;
    st.xs.forEach((x, m) => {
      const col = m % cols, row = Math.floor(m / cols), cx = 12 + col * (cw + gap) + cw / 2, y0 = top + row * rowH;
      drawImg(ctx, pal, x, cx - small / 2, y0, small);
      drawImg(ctx, pal, st.k ? st.xh[m] : new Float64Array(N_IMG * N_IMG).fill(-1), cx - big / 2, y0 + small + 6, big);
      if (isHL('hx1')) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(cx - big / 2 - 2, y0 + small + 4, big + 4, big + 4); }
      if (isHL('x')) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(cx - small / 2 - 2, y0 - 2, small + 4, small + 4); }
      // Patch around the centre pixel on the first sample.
      if (m === 0 && P) { const px = big / N_IMG, q = Math.floor(N_IMG / 2) - (P - 1) / 2; ctx.strokeStyle = hlc(['P', 'patch'], pal, pal.acc); ctx.lineWidth = isHL('P', 'patch') ? 2.6 : 1.6; ctx.strokeRect(cx - big / 2 + q * px, y0 + small + 6 + q * px, P * px, P * px); }
      if (st.k >= K_ELS) {
        const nd = nearestDigit(x);
        label(ctx, nd.copy ? `copy of ${nd.idx}` : `new · nearest ${nd.idx}`, cx, y0 + small + big + 24, nd.copy ? pal.muted : pal.acc, 12, 'center');
      }
    });
    if (!wide) label(ctx, 'upper: x_t · lower: x̂_1', w - 12, 18, pal.muted, 11.5, 'right');
    else label(ctx, 'each column: x_t (small) and x̂_1 (large)', w - 12, 18, pal.muted, 12, 'right');
    const t = st.k ? st.t : 0, done = st.k >= K_ELS, res = done ? st.xs.map(nearestDigit) : null, nc = res ? res.filter((q) => q.copy).length : 0;
    ro.innerHTML = `P = <b>${P ? `${P} × ${P}` : 'full image'}</b> · s = <b>${s}</b> · step <b>${st.k}</b> of ${K_ELS} · t = <b>${t.toFixed(3)}</b>` +
      (done ? `<br>copies <b>${nc}</b> of ${M_ELS} · new samples <b>${M_ELS - nc}</b> · nearest training digits ${res.map((q) => q.idx).join(', ')}` : '<br>running…');
  }
  return { draw, active: (on) => r.setActive(on) };
}

export const SCENES = { empirical, copies, collapse, agree, local, patches };
