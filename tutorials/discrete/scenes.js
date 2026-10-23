// Live scenes of the §2.9 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { makeRng } from '../../animations/lib/rng.js';
import {
  WORDS1, K1, M1, P1, marg1, flows1, hist1, K2, P2, marg2, dlog2, ratio2, hist2, T2, entropy3,
  L, PAIRS, GROUP, SENT, probOf, H_LANG, SENT4, maskCosts, bound4, nllModel, avg4, H_LANG_NATS, TAU4_LIST, SCHED4, integrand4, area4,
  K5_LIST, stats5, runs5, K6_LIST, TAU6_LIST, stats6, runs6,
} from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const pct = (q) => `${Math.round(100 * q)}%`;
// Panels side by side on wide canvases and stacked on narrow ones.
const WIDE = 640;
function panels(w, h, k, gap = 14) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + gap), w, ph]);
}
const listSlider = (id, list, value, on, fmt = String) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt }, on);
// Axes of a small plot; returns the maps from data to canvas.
function axes(ctx, pal, [x0, y0, w, h], { xmin = 0, xmax = 1, ymin, ymax, xt = [0, 0.5, 1], yt, xl = 't', title, yfmt = String, xfmt = String, titleColor }) {
  const L = x0 + 40, R = x0 + w - 10, T = y0 + 30, B = y0 + h - 30;
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
function setOn(btns, keys, k) { btns.forEach((b, j) => b.classList.toggle('on', keys[j] === k)); }
const tSlider = (t, on, min = 0, max = 1) => slider('t', { min, max, step: 0.01, value: t, fmt: f2 }, on);
// Canvas coordinates of a pointer event.
const at = (c, e) => { const r = c.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
// A word chip: a rounded box with the word, or a grey [M] when the token is masked.
function chip(ctx, pal, x, y, w, h, word, { color = pal.fg, fill = pal.panel, stroke = pal.line, size = 12 } = {}) {
  ctx.beginPath(); ctx.roundRect(x + 1, y + 1, w - 2, h - 2, 5);
  ctx.fillStyle = word === null ? pal.line : fill; ctx.fill();
  ctx.strokeStyle = stroke; ctx.lineWidth = stroke === pal.line ? 1 : 1.8; ctx.stroke();
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  let s = word ?? '[M]';
  while (s.length > 2 && ctx.measureText(s).width > w - 6) s = s.slice(0, -2) + '…';
  label(ctx, s, x + w / 2, y + h / 2 + size * 0.36, word === null ? pal.muted : color, size, 'center');
}
// Positions of the three coupled pairs whose words do not belong together.
function badPositions(x) {
  const bad = new Set();
  for (const [a, b] of PAIRS) if (x[a] !== null && x[b] !== null && !SENT.some((s) => s[a] === x[a] && s[b] === x[b])) { bad.add(a); bad.add(b); }
  return bad;
}

// ---------- Step 1: a Markov chain on a finite set ----------
const NAMES1 = [...WORDS1, '[M]'];
function chainScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7 : 0.62)), ro = readout();
  let kind = 'uniform', t = 0.5, y = 0, geo = null;
  const keys = ['uniform', 'absorbing'];
  const btns = keys.map((k) => button(k === 'uniform' ? 'Uniform kernel' : 'Absorbing kernel', () => { kind = k; setOn(btns, keys, k); if (k === 'uniform' && y === M1) y = 0; draw(); }, k === kind ? 'on' : ''));
  c.cv.addEventListener('pointerdown', (e) => {
    if (!geo) return;
    const [X, Y] = at(c, e), i = Math.floor((X - geo.x0) / geo.bw);
    if (Y >= geo.y0 && Y <= geo.y1 && i >= 0 && i <= K1 && !(kind === 'uniform' && i === M1)) { y = i; draw(); }
  });
  viz.append(c.cv, controls(...btns, tSlider(t, (v) => { t = v; draw(); })), ro,
    note('Exact closed forms for the bars and the flows; the dots are the histogram of 400 seeded chains, each with its own replacement time. The flows use the generative rates \\(R_t(x, y) = Q_t(y, x)\\,p_t(y)/p_t(x)\\) derived in step 2. Click a bar to choose the word whose flows are shown.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [A, B] = panels(w, h, 2, 36);
    const p = marg1(kind, t), hs = hist1(kind, t), n = K1 + 1;
    const ax = axes(ctx, pal, A, { xmin: 0, xmax: n, ymin: 0, ymax: 1, xt: [], yt: [0, 0.25, 0.5, 0.75, 1], yfmt: f2, xl: '', title: `p_t and the histogram of 400 chains, t = ${f2(t)}`, titleColor: hlc('pt', pal, pal.fg) });
    const bw = (ax.R - ax.L) / n;
    geo = { x0: ax.L, bw, y0: ax.T, y1: ax.B + 22 };
    for (let i = 0; i < n; i++) {
      const xc = ax.L + (i + 0.5) * bw, bwi = bw * 0.56;
      if (i === y) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 1.2; ctx.strokeRect(xc - bw / 2 + 3.5, ax.T + 3.5, bw - 7, ax.B - ax.T - 6); }
      ctx.fillStyle = i === M1 ? hlc('M', pal, pal.muted) : hlc('pt', pal, pal.acc); ctx.globalAlpha = 0.75;
      ctx.fillRect(xc - bwi / 2, ax.Y(p[i]), bwi, ax.B - ax.Y(p[i])); ctx.globalAlpha = 1;
      if (i < K1) { ctx.strokeStyle = hlc('p1', pal, pal.fg); ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]); ctx.strokeRect(xc - bwi / 2, ax.Y(P1[i]), bwi, ax.B - ax.Y(P1[i])); ctx.setLineDash([]); }
      dot(ctx, xc, ax.Y(hs[i]), 4.5, pal.ramp[3]);
      label(ctx, NAMES1[i], xc, ax.B + 17, i === y ? pal.fg : pal.muted, 12, 'center', i === y ? 600 : 500);
    }
    // Flows into and out of the chosen word.
    const [bx, by, bwid, bh] = B, nm = NAMES1[y];
    label(ctx, `Flows of ${nm} under the generative chain`, bx + 10, by + 18, hlc('R', pal, pal.fg), 12.5);
    if (t < 0.01) {
      label(ctx, 'The rates grow like 1/t and are infinite at t = 0.', bx + 10, by + 50, pal.muted, 12);
      ro.innerHTML = `${kind} kernel · t = <b>0.00</b> · p<sub>t</sub>(${nm}) = <b>${f3(p[y])}</b>`;
      return;
    }
    const F = flows1(kind, t, y), rows = [];
    for (let x = 0; x < n; x++) if (x !== y && !(kind === 'uniform' && x === M1)) rows.push([`in from ${NAMES1[x]}`, F.inflow[x], pal.acc]);
    rows.push(['total in', F.inTot, pal.ok], ['out', -F.out, pal.bad], ['in − out', F.inTot - F.out, pal.fg], ['∂_t p_t (closed form)', F.dpdt, pal.ramp[3]]);
    const mx = Math.max(1e-9, ...rows.map((r) => Math.abs(r[1]))), lw = Math.min(150, bwid * 0.36), x0 = bx + 10 + lw, half = (bwid - lw - 80) / 2, zc = x0 + half;
    const rh = Math.min(26, (bh - 50) / rows.length);
    rows.forEach(([name, v, col], j) => {
      const yy = by + 36 + j * rh, sep = j === rows.length - 4;
      if (sep) hdash(ctx, pal.line, bx + 10, bx + bwid - 10, yy - 3, 1, []);
      label(ctx, name, bx + 10, yy + rh * 0.62, j >= rows.length - 4 ? pal.fg : pal.muted, 11.5);
      ctx.fillStyle = col; ctx.globalAlpha = 0.8;
      const len = half * v / mx; ctx.fillRect(Math.min(zc, zc + len), yy + 3, Math.abs(len), rh - 8); ctx.globalAlpha = 1;
      label(ctx, (v >= 0 ? '+' : '−') + f3(Math.abs(v)), bx + bwid - 8, yy + rh * 0.62, j >= rows.length - 4 ? pal.fg : pal.muted, 11.5, 'right');
    });
    vdash(ctx, pal.muted, zc, by + 34, by + 36 + rows.length * rh, 1, []);
    ro.innerHTML = `${kind} kernel · t = <b>${f2(t)}</b> · p<sub>t</sub>(${nm}) = <b>${f3(p[y])}</b>, histogram ${f3(hs[y])}<br>` +
      `inflow <b>${f3(F.inTot)}</b> − outflow <b>${f3(F.out)}</b> = <b>${f3(F.inTot - F.out)}</b>, and the derivative of the closed form is <b>${f3(F.dpdt)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: time reversal and the concrete score ----------
function reverseScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7 : 0.62)), ro = readout();
  let t = 0.5;
  const r = runner({ step: () => { t = Math.min(1, (Math.round(t * T2) + 1) / T2); tSl.set(t); }, done: () => t >= 1, draw: () => {}, interval: 50 });
  r.playing = false;
  const tSl = tSlider(t, (v) => { t = v; draw(); });
  viz.append(c.cv, controls(tSl, button('Play from t = 0', () => { t = 0; tSl.set(0); r.playing = true; r.kick(); })), ro,
    note('Exact closed forms for \\(p_t\\) and the ratios. The dots in the left panel are the histogram of 400 seeded chains simulated with the rates \\(R_t\\) from \\(t = 0.001\\), with 2200 substeps. Nothing is fitted: the chains use the true concrete score.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [A, B] = panels(w, h, 2, 30), j = Math.round(t * T2), p = marg2(t), hs = hist2(j);
    const ax = axes(ctx, pal, A, { xmin: -0.5, xmax: K2 - 0.5, ymin: 0, ymax: 0.13, xt: [0, 8, 16, 24, 31], yt: [0, 0.05, 0.1], yfmt: f2, xl: 'x', title: `p_t and the histogram of 400 chains, t = ${f2(t)}`, titleColor: hlc('pt', pal, pal.fg) });
    const bw = (ax.R - ax.L) / K2;
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    for (let x = 0; x < K2; x++) {
      const X = ax.X(x);
      ctx.fillStyle = hlc('pt', pal, pal.acc); ctx.globalAlpha = 0.6; ctx.fillRect(X - bw * 0.4, ax.Y(p[x]), bw * 0.8, ax.B - ax.Y(p[x])); ctx.globalAlpha = 1;
    }
    polyline(ctx, P2.map((_, x) => ax.X(x)), P2.map((q) => ax.Y(q)), hlc('p1', pal, pal.fg), 1.4, [4, 3]);
    for (let x = 0; x < K2; x++) dot(ctx, ax.X(x), ax.Y(hs[x]), 3.2, pal.ramp[3]);
    ctx.restore();
    label(ctx, 'dashed: p_1', ax.R - 6, ax.T + 16, hlc('p1', pal, pal.muted), 11.5, 'right');
    // Concrete score towards the right neighbour and 1 + d/dx log p_t of the smooth density.
    const bx = axes(ctx, pal, B, { xmin: 0, xmax: K2 - 1, ymin: 0, ymax: 4, xt: [0, 8, 16, 24, 31], yt: [0, 1, 2, 3, 4], xl: 'x', title: 'p_t(x + 1)/p_t(x) and 1 + ∂_x log p_t', titleColor: hlc('cs', pal, pal.fg) });
    clip(ctx, bx.L, bx.T, bx.R - bx.L, bx.B - bx.T);
    hdash(ctx, pal.muted, bx.L, bx.R, bx.Y(1));
    const us = Array.from({ length: 311 }, (_, i) => i / 10);
    polyline(ctx, us.map(bx.X), us.map((u) => bx.Y(1 + dlog2(Math.max(t, 1e-3), u))), pal.muted, 1.8);
    let gap = 0;
    for (let x = 0; x < K2 - 1; x++) {
      const q = ratio2(t, x);
      gap = Math.max(gap, Math.abs(q - 1 - dlog2(Math.max(t, 1e-3), x + 0.5)));
      dot(ctx, bx.X(x + 0.5), bx.Y(q), 3.4, hlc('cs', pal, pal.ramp[2]));
    }
    ctx.restore();
    let tv = 0;
    for (let x = 0; x < K2; x++) tv += 0.5 * Math.abs(hs[x] - p[x]);
    ro.innerHTML = `t = <b>${f2(t)}</b> · total variation between the histogram and p<sub>t</sub> <b>${f3(tv)}</b> (about 0.1 is the sampling noise of 400 chains on 32 states)<br>` +
      `largest difference between the ratio and 1 + ∂<sub>x</sub> log p<sub>t</sub> at x + ½: <b>${f3(gap)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 3: score entropy ----------
function entropyScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7 : 0.62)), ro = readout();
  let t = 0.6, x = 12, s = 1.5, geo = null;
  const drag = (e) => {
    if (!geo || !(e.buttons & 1) && e.type !== 'pointerdown') return;
    const [X, Y] = at(c, e);
    if (X < geo.L - 10 || X > geo.R + 10 || Y < geo.T || Y > geo.B) return;
    s = Math.max(geo.smin, Math.min(geo.smax, geo.inv(X))); draw();
  };
  c.cv.addEventListener('pointerdown', drag); c.cv.addEventListener('pointermove', drag);
  viz.append(c.cv, controls(tSlider(t, (v) => { t = v; draw(); }, 0.05, 0.95), slider('x', { min: 0, max: K2 - 2, step: 1, value: x, fmt: String }, (v) => { x = v; draw(); })), ro,
    note('Exact. The posterior and the expected loss are sums over the 32 clean values under the uniform kernel of step 2. The curve is \\(\\E[s - r\\log s]\\) minus its minimum. Drag along the curve to move s.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [A, B] = panels(w, h, 2, 30), y = x + 1, E = entropy3(t, x);
    // Posterior mass of the three cases of r.
    const groups = [[`x_1 = ${y} (= y)`, E.post[y], E.r[y]], [`x_1 = ${x} (= x)`, E.post[x], E.r[x]], ['any other x_1', 1 - E.post[x] - E.post[y], 1]];
    const ax = axes(ctx, pal, A, { xmin: 0, xmax: 3, ymin: 0, ymax: 1, xt: [], yt: [0, 0.5, 1], yfmt: f2, xl: '', title: `posterior p(x_1 | x = ${x}) at t = ${f2(t)}`, titleColor: hlc('post', pal, pal.fg) });
    const bw = (ax.R - ax.L) / 3;
    groups.forEach(([name, m, rv], i) => {
      const xc = ax.L + (i + 0.5) * bw;
      ctx.fillStyle = hlc('post', pal, pal.acc); ctx.globalAlpha = 0.75; ctx.fillRect(xc - bw * 0.28, ax.Y(m), bw * 0.56, ax.B - ax.Y(m)); ctx.globalAlpha = 1;
      label(ctx, f3(m), xc, ax.Y(m) - 6, pal.fg, 11.5, 'center');
      label(ctx, name, xc, ax.B + 15, pal.muted, 11, 'center');
      label(ctx, `r = ${rv >= 100 ? rv.toFixed(0) : rv >= 10 ? rv.toFixed(1) : f3(rv)}`, xc, ax.B + 28, hlc('r', pal, pal.ramp[2]), 11.5, 'center', 600);
    });
    // Expected excess loss as a function of s.
    const smax = Math.max(2.5, 2.2 * E.target), smin = 0.03 * smax, ex = (q) => E.loss(q) - E.minLoss;
    const ymax = 1.15 * Math.max(ex(smax), ex(Math.max(smin, 0.3 * E.target)));
    const bx = axes(ctx, pal, B, { xmin: 0, xmax: smax, ymin: 0, ymax, xt: [0, E.target, ...[1, 2, 3, 4, 5, 6, 8, 10].filter((q) => q < 0.95 * smax && Math.abs(q - E.target) > 0.12 * smax)], yt: [0, ymax / 2, ymax].map((q) => +q.toPrecision(2)), xfmt: (q) => q.toFixed(2), yfmt: (q) => String(+q.toPrecision(2)), xl: 's', title: 'E[s − r log s] − min' });
    geo = { L: bx.L, R: bx.R, T: bx.T, B: bx.B + 10, smin, smax, inv: (X) => smax * (X - bx.L) / (bx.R - bx.L) };
    clip(ctx, bx.L, bx.T, bx.R - bx.L, bx.B - bx.T);
    const ss = Array.from({ length: 300 }, (_, i) => smin + (smax - smin) * i / 299);
    polyline(ctx, ss.map(bx.X), ss.map((q) => bx.Y(ex(q))), pal.fg, 2);
    vdash(ctx, hlc('cs', pal, pal.ramp[2]), bx.X(E.target), bx.T, bx.B, 1.6);
    ctx.restore();
    label(ctx, 'p_t(y)/p_t(x)', bx.X(E.target) + 6, bx.T + 16, hlc('cs', pal, pal.ramp[2]), 11.5);
    dot(ctx, bx.X(s), bx.Y(Math.min(ymax, ex(s))), 7, hlc('s', pal, pal.ramp[3]));
    ring(ctx, bx.X(s), bx.Y(Math.min(ymax, ex(s))), 10, hlc('s', pal, pal.ramp[3]), 1.2);
    ro.innerHTML = `t = <b>${f2(t)}</b> · x = <b>${x}</b>, y = ${y} · s = <b>${f3(s)}</b> with excess loss <b>${f3(ex(s))}</b><br>` +
      `posterior mean of r <b>${f3(E.target)}</b> · marginal ratio p<sub>t</sub>(y)/p<sub>t</sub>(x) <b>${f3(E.marginal)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: masked diffusion and its ELBO ----------
const U4 = (() => { const r = makeRng(9404); return Array.from({ length: L }, () => r.uniform()); })();
function elboScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.4 : 0.6)), ro = readout();
  let si = 0, tau = 1, sk = 'linear', t = 0.5;
  const costs = new Map(), nll = new Map();
  const getS = () => { const k = `${si}:${tau}`; if (!costs.has(k)) costs.set(k, maskCosts(SENT4[si], tau).S); return costs.get(k); };
  const getNll = () => { const k = `${si}:${tau}`; if (!nll.has(k)) nll.set(k, nllModel(SENT4[si], tau)); return nll.get(k); };
  const short = (x) => `${x[1]}, ${x[3]} ${x[4]}, ${x[6]} ${x[7]}`;
  const sb = SENT4.map((x, i) => button(short(x), () => { si = i; setOn(sb, [0, 1, 2], i); draw(); }, i === si ? 'on' : ''));
  const keys = Object.keys(SCHED4), kb = keys.map((k) => button(SCHED4[k].name, () => { sk = k; setOn(kb, keys, k); draw(); }, k === sk ? 'on' : ''));
  viz.append(c.cv, controls(...sb), controls(...kb), controls(listSlider('tau', TAU4_LIST, tau, (v) => { tau = v; draw(); }), tSlider(t, (v) => { t = v; draw(); })), ro,
    note('Exact. The expectation over masks sums all 2⁸ patterns, the conditionals are those of the 64 sentences, and the areas are computed with 2000 midpoints. The chips show one mask drawn at time t with the selected schedule, from fixed uniforms per position.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), x = SENT4[si], S = getS(), sc = SCHED4[sk];
    // The sentence with the mask of time t.
    const m = 1 - sc.a(t), cw = Math.min(110, (w - 20) / L), ch = 30, cx0 = (w - cw * L) / 2;
    for (let i = 0; i < L; i++) chip(ctx, pal, cx0 + i * cw, 6, cw, ch, U4[i] < m ? null : x[i], { size: w >= WIDE ? 12.5 : 10.5, fill: pal.panel });
    // Integrands of the bound for the three schedules.
    const ts = Array.from({ length: 401 }, (_, i) => i / 400), curves = keys.map((k) => ts.map((q) => integrand4(SCHED4[k], S, q)));
    const ymax = 1.12 * Math.max(...curves.flat()), yt = [0, ymax / 2, ymax].map((q) => +q.toPrecision(2));
    const ax = axes(ctx, pal, [0, ch + 16, w, h - ch - 16], { ymin: 0, ymax, yt, yfmt: String, xt: [0, 0.25, 0.5, 0.75, 1], title: w >= WIDE ? 'integrand  (α′/(1 − α)) E[Σ over masked tokens of −log p_θ]' : 'integrand of the bound', titleColor: hlc('ptheta', pal, pal.fg) });
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    keys.forEach((k, j) => {
      const on = k === sk, col = [pal.acc, pal.ramp[0], pal.ramp[3]][j];
      if (on) {
        ctx.beginPath(); ctx.moveTo(ax.X(0), ax.Y(0)); ts.forEach((q, i) => ctx.lineTo(ax.X(q), ax.Y(curves[j][i]))); ctx.lineTo(ax.X(1), ax.Y(0)); ctx.closePath();
        ctx.fillStyle = col; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
      }
      polyline(ctx, ts.map(ax.X), curves[j].map(ax.Y), on ? col : pal.muted, on ? 2.4 : 1.4, on ? [] : [5, 4]);
    });
    vdash(ctx, hlc('t', pal, pal.mark), ax.X(t), ax.T, ax.B, 1.2);
    ctx.restore();
    const areas = keys.map((k) => area4(SCHED4[k], S)), b = bound4(S), nlp = -Math.log(probOf(x)), nm = getNll(), A = avg4(tau);
    ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(ax.R - 190, ax.T + 4, 186, 58); ctx.globalAlpha = 1;
    keys.forEach((k, j) => label(ctx, `${SCHED4[k].name}: area ${f3(areas[j])}`, ax.R - 10, ax.T + 20 + 17 * j, k === sk ? [pal.acc, pal.ramp[0], pal.ramp[3]][j] : pal.muted, 12, 'right', k === sk ? 600 : 500));
    ro.innerHTML = `τ = <b>${tau}</b> · bound <b>${f3(b)}</b> nats (the three areas agree to quadrature error) · −log p<sub>data</sub>(x<sub>1</sub>) = <b>${f3(nlp)}</b> · gap <b>${f3(b - nlp)}</b> · −log p<sub>θ</sub>(x<sub>1</sub>) of the chain <b>${f3(nm)}</b><br>` +
      `averaged over the language: bound <b>${f3(A.bound)}</b> against the entropy <b>${f3(H_LANG_NATS)}</b> nats, a KL divergence of <b>${f3(A.bound - H_LANG_NATS)}</b>`;
  }
  return { draw };
}

// ---------- Step 5: discrete flow matching ----------
function dfmScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8.4 : 0.64), { maxH: 0.7 }), ro = readout();
  let source = 'mask', N = 8, sel = 0, geo = null;
  const runs = new Map(), getRuns = () => { const k = `${source}:${N}`; if (!runs.has(k)) runs.set(k, runs5(source, N)); return runs.get(k); };
  const keys = ['mask', 'uniform'], sb = keys.map((k) => button(k === 'mask' ? 'Masked source' : 'Uniform source', () => { source = k; setOn(sb, keys, k); draw(); }, k === source ? 'on' : ''));
  c.cv.addEventListener('pointerdown', (e) => {
    if (!geo) return;
    const [X, Y] = at(c, e), i = Math.floor((Y - geo.y0) / geo.rh);
    if (X >= geo.x0 && X <= geo.x1 && i >= 0 && i < 8) { sel = i; draw(); }
  });
  viz.append(c.cv, controls(...sb, listSlider('Ksteps', K5_LIST, N, (v) => { N = v; draw(); })), ro,
    note('Exact posterior over the 64 sentences at every step, κ<sub>t</sub> = t, and seeded draws. The statistics average 1000 runs per setting. Click a sentence in the list to show its run.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, R = getRuns(), run = R[sel];
    const G = wide ? [0, 0, w * 0.54, h] : [0, 0, w, h * 0.56], Lst = wide ? [w * 0.54 + 24, 0, w * 0.46 - 24, h] : [0, h * 0.56 + 10, w, h * 0.44 - 10];
    // The selected run, one row per Euler step.
    const [gx, gy, gw, gh] = G, lw = 62, cw = (gw - lw) / L, rows = run.path.length, rh = Math.min(28, (gh - 24) / rows);
    label(ctx, `run ${sel + 1}: one row per step, t from 0 (top) to 1`, gx, gy + 13, pal.fg, 12);
    const seen = Array(L).fill(0);
    run.path.forEach((x, k) => {
      const y0 = gy + 22 + k * rh;
      if (rh >= 12 && (k % Math.ceil(14 / rh) === 0 || k === rows - 1)) label(ctx, `t = ${f2(k / N)}`, gx + lw - 6, y0 + rh * 0.68, pal.muted, Math.min(11, rh * 0.75), 'right');
      const bad = k === rows - 1 ? badPositions(x) : new Set();
      for (let i = 0; i < L; i++) {
        const ch = k > 0 && x[i] !== run.path[k - 1][i];
        if (ch) seen[i]++;
        const stroke = ch ? (seen[i] > 1 ? pal.ramp[2] : pal.mark) : pal.line;
        if (rh >= 15) chip(ctx, pal, gx + lw + i * cw, y0, cw, rh, x[i], { size: Math.min(11.5, rh * 0.5), stroke, color: bad.has(i) ? pal.bad : pal.fg });
        else {
          ctx.fillStyle = x[i] === null ? pal.line : bad.has(i) ? pal.bad : GROUP[i] < 0 ? pal.muted : pal.ramp[[0, 1, 3][GROUP[i]]];
          ctx.globalAlpha = x[i] === null ? 1 : 0.55; ctx.fillRect(gx + lw + i * cw + 1, y0 + 0.5, cw - 2, Math.max(1, rh - 1)); ctx.globalAlpha = 1;
          if (ch) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.2; ctx.strokeRect(gx + lw + i * cw + 1.5, y0 + 0.5, cw - 3, Math.max(1, rh - 1)); }
        }
      }
    });
    // The final sentences of the eight runs.
    if (!wide) { const top = gy + 22 + rows * rh + 24; Lst[3] += Lst[1] - top; Lst[1] = top; }
    const [lx, ly, lwid, lh] = Lst, lrh = Math.min(30, (lh - 24) / 8);
    label(ctx, `final sentences of 8 runs, N = ${N}`, lx, ly + 13, hlc('Ksteps', pal, pal.fg), 12);
    geo = { x0: lx, x1: lx + lwid, y0: ly + 22, rh: lrh };
    R.forEach((o, i) => {
      const y0 = ly + 22 + i * lrh, bad = badPositions(o.x);
      if (i === sel) { ctx.fillStyle = pal.line; ctx.fillRect(lx, y0, lwid, lrh - 2); }
      label(ctx, o.ok ? '✓' : '✗', lx + 8, y0 + lrh * 0.66, o.ok ? pal.ok : pal.bad, 13, 'left', 600);
      let xx = lx + 26;
      const size = Math.min(12.5, lwid / 30);
      o.x.forEach((wd, j) => { xx += label(ctx, wd + ' ', xx, y0 + lrh * 0.66, bad.has(j) ? pal.bad : i === sel ? pal.fg : pal.muted, size); });
      if (source === 'uniform') label(ctx, `${o.revisions} rev.`, lx + lwid - 4, y0 + lrh * 0.66, pal.muted, 11, 'right');
    });
    const st = stats5(source, N);
    ro.innerHTML = `${source === 'mask' ? 'masked' : 'uniform'} source · N = <b>${N}</b> · over 1000 runs: valid sentences <b>${pct(st.valid)}</b> · changes per token <b>${f2(st.changes)}</b> · revisions per sentence <b>${f2(st.revisions)}</b><br>` +
      `run ${sel + 1}: ${run.ok ? 'valid' : 'not a sentence of the language'}, ${run.changes} changes, ${run.revisions} revisions. Outlined cells changed in that step; pink outlines are revisions.`;
  }
  return { draw };
}

// ---------- Step 6: parallel decoding ----------
function parallelScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.6 : 0.5), { maxH: 0.68 }), ro = readout();
  let rule = 'k', k = 2, tau = 0.35;
  const keys = ['k', 'tau'], rb = keys.map((q) => button(q === 'k' ? 'k tokens per step' : 'Confidence threshold', () => { rule = q; setOn(rb, keys, q); draw(); }, q === rule ? 'on' : ''));
  const cur = () => (rule === 'k' ? { k } : { tau });
  viz.append(c.cv, controls(...rb), controls(listSlider('k', K6_LIST, k, (v) => { k = v; if (rule !== 'k') rb[0].click(); else draw(); }), listSlider('tau', TAU6_LIST, tau, (v) => { tau = v; if (rule !== 'tau') rb[1].click(); else draw(); })), ro,
    note('Exact conditionals of the 64 sentences and seeded draws; each point averages 1000 runs. The threshold rule compares the probability of the drawn value with τ and reveals at least one token per step.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE;
    const A = wide ? [0, 0, w * 0.5, h] : [0, 0, w, h * 0.5], B = wide ? [w * 0.5 + 24, 0, w * 0.5 - 24, h] : [0, h * 0.5 + 10, w, h * 0.5 - 10];
    const ax = axes(ctx, pal, A, { xmin: 0.5, xmax: 8.5, ymin: 0, ymax: 1, xt: [1, 2, 4, 6, 8], yt: [0, 0.5, 1], yfmt: pct, xl: 'network evaluations (steps)', title: 'valid sentences against steps' });
    const series = [['k', K6_LIST.map((q) => stats6({ k: q })), pal.acc, K6_LIST], ['tau', TAU6_LIST.map((q) => stats6({ tau: q })), pal.ramp[3], TAU6_LIST]];
    series.forEach(([nm, S, col, vals]) => {
      polyline(ctx, S.map((q) => ax.X(q.steps)), S.map((q) => ax.Y(q.valid)), col, 1.4, nm === 'tau' ? [4, 3] : []);
      S.forEach((q, i) => {
        dot(ctx, ax.X(q.steps), ax.Y(q.valid), 4, col);
        if (nm === 'k' && vals[i] <= 4) label(ctx, `k=${vals[i]}`, ax.X(q.steps) + 7, ax.Y(q.valid) + 14, pal.muted, 10.5);
      });
    });
    const now = stats6(cur());
    ring(ctx, ax.X(now.steps), ax.Y(now.valid), 9, pal.mark, 2.2);
    label(ctx, '— k tokens per step', ax.R - 8, ax.B - 30, pal.acc, 11.5, 'right');
    label(ctx, '- - threshold τ', ax.R - 8, ax.B - 13, pal.ramp[3], 11.5, 'right');
    // Entropy of the output against the entropy of the language, and six runs.
    const [bx, by, bw, bh] = B, scale = (bw - 150) / 10;
    label(ctx, 'entropy of 1000 sentences (bits)', bx, by + 14, pal.fg, 12.5);
    [['language', H_LANG, pal.muted], ['this sampler', now.H, rule === 'k' ? pal.acc : pal.ramp[3]]].forEach(([nm, v, col], i) => {
      const yy = by + 26 + i * 24;
      label(ctx, nm, bx, yy + 13, pal.muted, 11.5);
      ctx.fillStyle = col; ctx.globalAlpha = 0.75; ctx.fillRect(bx + 100, yy + 2, scale * v, 16); ctx.globalAlpha = 1;
      label(ctx, f2(v), bx + 106 + scale * v, yy + 14, pal.fg, 11.5);
    });
    const R = runs6(cur()), ry = by + 90, rrh = Math.min(28, (bh - 100) / 6), size = Math.min(12.5, bw / 31);
    label(ctx, 'six runs', bx, ry, pal.fg, 12.5);
    R.forEach((o, i) => {
      const y0 = ry + 8 + i * rrh, bad = new Set(o.bad.flat());
      label(ctx, o.ok ? '✓' : '✗', bx + 4, y0 + rrh * 0.66, o.ok ? pal.ok : pal.bad, 13, 'left', 600);
      let xx = bx + 22;
      o.x.forEach((wd, j) => { xx += label(ctx, wd + ' ', xx, y0 + rrh * 0.66, bad.has(j) ? pal.bad : pal.fg, size); });
      label(ctx, `${o.steps.length} st.`, bx + bw - 4, y0 + rrh * 0.66, pal.muted, 11, 'right');
    });
    ro.innerHTML = `${rule === 'k' ? `k = <b>${k}</b> tokens per step` : `threshold τ = <b>${tau}</b>`} · over 1000 runs: valid <b>${pct(now.valid)}</b> · steps <b>${f2(now.steps)}</b> · entropy <b>${f2(now.H)}</b> bits against <b>${f2(H_LANG)}</b> for the language · <b>${now.distinct}</b> distinct sentences`;
  }
  return { draw };
}

export const SCENES = { chain: chainScene, reverse: reverseScene, entropy: entropyScene, elbo: elboScene, dfm: dfmScene, parallel: parallelScene };
