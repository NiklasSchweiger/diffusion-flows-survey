// Live scenes of the §2.10 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import {
  DIGITS, N_IMG, D, K, S0, V0, QLO_HI, typ, posterior, RUN1, runs2, stats2, SCALES2, NS2,
  SAMPLERS3, settings3, runs3, stats3, NS3, K4_LIST, TEST4, codebook, quantize, stats4, NT4,
  NT5, tokPix, RUN5, IMGS5, field5,
} from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f1 = (v) => v.toFixed(1), f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3);
const pct = (q) => `${Math.round(100 * q)}%`;
const WIDE = 640;
function panels(w, h, k, gap = 14) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + gap), w, ph]);
}
const listSlider = (id, list, value, on, fmt = String, tex) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt, tex }, on);
function axes(ctx, pal, [x0, y0, w, h], { xmin = 0, xmax = 1, ymin, ymax, xt = [0, 0.5, 1], yt, xl = '', title, yfmt = String, xfmt = String, titleColor }) {
  const L = x0 + 44, R = x0 + w - 10, T = y0 + 30, B = y0 + h - 30;
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
const at = (c, e) => { const r = c.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

// A greyscale image with values in [-1, 1] from the background to the foreground colour. NaN pixels are
// not generated yet and are drawn in the line colour.
const hexRGB = (s) => { const m = s.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i); return m ? m.slice(1).map((q) => parseInt(q, 16)) : [128, 128, 128]; };
const imgCanvas = document.createElement('canvas');
function drawImg(ctx, pal, im, x, y, size, border = pal.line) {
  const n = Math.round(Math.sqrt(im.length));
  imgCanvas.width = imgCanvas.height = n;
  const g = imgCanvas.getContext('2d'), d = g.createImageData(n, n), bg = hexRGB(pal.bg), fg = hexRGB(pal.fg), ln = hexRGB(pal.muted).map((q, c) => Math.round(bg[c] + 0.22 * (q - bg[c])));
  for (let i = 0; i < im.length; i++) {
    const v = im[i], u = Math.max(0, Math.min(1, (v + 1) / 2));
    for (let c = 0; c < 3; c++) d.data[4 * i + c] = Number.isNaN(v) ? ln[c] : Math.round(bg[c] + (fg[c] - bg[c]) * u);
    d.data[4 * i + 3] = 255;
  }
  g.putImageData(d, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(imgCanvas, x, y, size, size); ctx.imageSmoothingEnabled = true;
  ctx.strokeStyle = border; ctx.lineWidth = border === pal.line ? 1 : 2.2; ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}
// Border colour of a sample from the typicality test: too unlikely, typical or too likely.
const typColor = (pal, c) => (c < 0 ? pal.bad : c > 0 ? pal.acc2 : pal.line);
// A horizontal bar split into the shares below, inside and above the typical band.
function typBar(ctx, pal, x, y, w, h, st) {
  const parts = [[st.low, pal.bad], [st.typical, pal.ok], [st.high, pal.acc2]];
  let X = x;
  for (const [q, col] of parts) { ctx.fillStyle = col; ctx.globalAlpha = 0.8; ctx.fillRect(X, y, w * q, h); X += w * q; }
  ctx.globalAlpha = 1; ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  label(ctx, `too unlikely ${pct(st.low)}`, x, y + h + 15, pal.bad, 11.5);
  label(ctx, `typical ${pct(st.typical)}`, x + w / 2, y + h + 15, pal.ok, 11.5, 'center', 600);
  label(ctx, `too likely ${pct(st.high)}`, x + w, y + h + 15, pal.acc2, 11.5, 'right');
}
// Eight bars of a posterior over the digits, labelled 0 to 7.
function postBars(ctx, pal, [x, y, w, h], pi, { color = pal.acc, title, titleColor, labels = true } = {}) {
  if (title) label(ctx, title, x, y + 12, titleColor ?? pal.fg, 12.5);
  const top = y + (title ? 22 : 2), bot = y + h - (labels ? 16 : 2), bw = w / K;
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, bot + 0.5); ctx.lineTo(x + w, bot + 0.5); ctx.stroke();
  pi.forEach((p, j) => {
    ctx.fillStyle = color; ctx.globalAlpha = 0.8;
    ctx.fillRect(x + j * bw + bw * 0.18, bot - (bot - top) * p, bw * 0.64, (bot - top) * p); ctx.globalAlpha = 1;
    if (labels) label(ctx, String(j), x + (j + 0.5) * bw, bot + 13, p > 0.5 ? pal.fg : pal.muted, 11, 'center', p > 0.5 ? 600 : 500);
  });
}
const gauss = (x, m, v) => Math.exp(-0.5 * (x - m) ** 2 / v) / Math.sqrt(2 * Math.PI * v);
const argmax = (a) => a.indexOf(Math.max(...a));

// ---------- Step 1: the chain rule in any order ----------
function chainScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.6 : 0.62), { maxH: 0.75 }), ro = readout();
  let order = 'raster', m = 40;
  const keys = ['raster', 'random'];
  const cum = Object.fromEntries(keys.map((o) => { const a = [0]; RUN1[o].logc.forEach((l) => a.push(a[a.length - 1] + l)); return [o, a]; }));
  const all = keys.flatMap((o) => cum[o]), ymin = Math.floor(Math.min(...all) / 10) * 10, ymax = Math.ceil(Math.max(...all, 0) / 10) * 10;
  const r = runner({ step: () => { m = Math.min(D, m + 1); sl.set(m); }, done: () => m >= D, draw: () => {}, interval: 45 });
  r.playing = false;
  const sl = slider('xi', { min: 0, max: D, step: 1, value: m, fmt: String, tex: 'i' }, (v) => { m = v; draw(); });
  const btns = keys.map((k) => button(k === 'raster' ? 'Raster order' : 'Random order', () => { order = k; setOn(btns, keys, k); draw(); }, k === order ? 'on' : ''));
  const play = button('Play from i = 0', () => { m = 0; sl.set(0); r.playing = true; r.kick(); });
  viz.append(c.cv, controls(...btns, sl, play), ro,
    note('Exact closed forms: the posterior is computed from the eight mixture components, the curve is the exact conditional density of the next pixel, and log p(x) is computed directly from the mixture. The two images are seeded draws, one per order.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, R = RUN1[order];
    let A, P, Q, C;
    if (wide) {
      const s = Math.min(h - 30, w * 0.3); A = [0, 0, s, h]; const rest = w - s - 28, bx = s + 28, bw = rest * 0.45;
      P = [bx, 0, bw, h * 0.46]; Q = [bx - 34, h * 0.5, bw + 34, h * 0.5]; C = [bx + bw + 20, 0, rest * 0.55 - 20, h];
    } else {
      const s = w * 0.5, rest = h - s - 40; A = [0, 0, s, s + 30]; P = [s + 16, 0, w - s - 16, s + 24];
      Q = [0, s + 36, w, rest * 0.45]; C = [0, s + 46 + rest * 0.45, w, rest * 0.55 - 6];
    }
    // The image so far.
    const sz = Math.min(A[2], A[3] - 30), im = new Float64Array(D).fill(NaN);
    for (let q = 0; q < m; q++) im[R.ord[q]] = R.x[R.ord[q]];
    label(ctx, wide ? `${m} of ${D} pixels, ${order} order` : `${m} of ${D} pixels`, A[0], A[1] + 14, hlc('x', pal, pal.fg), 12.5);
    drawImg(ctx, pal, im, A[0], A[1] + 24, sz, hlc('x', pal, pal.line));
    const cell = sz / N_IMG, nxt = m < D ? R.ord[m] : -1;
    if (nxt >= 0) { ctx.strokeStyle = hlc('xi', pal, pal.mark); ctx.lineWidth = 2.2; ctx.strokeRect(A[0] + (nxt % N_IMG) * cell + 1, A[1] + 24 + Math.floor(nxt / N_IMG) * cell + 1, cell - 2, cell - 2); }
    // Posterior over the digits and the density of the next pixel.
    const pi = R.post[m];
    postBars(ctx, pal, P, pi, { color: hlc('pi', pal, pal.acc), title: wide ? 'Posterior π_j over the digits' : 'Posterior π_j', titleColor: hlc('pi', pal, pal.fg) });
    const ax = axes(ctx, pal, Q, { xmin: -2, xmax: 2, ymin: 0, ymax: 1.4, xt: [-1, 0, 1], yt: [0, 1], xl: '', title: nxt >= 0 ? 'Density of the next pixel' : 'All pixels generated', titleColor: hlc('xi', pal, pal.fg) });
    if (nxt >= 0) {
      const xs = Array.from({ length: 161 }, (_, q) => -2 + q / 40), ys = xs.map((v) => pi.reduce((s, p, j) => s + p * gauss(v, DIGITS[j][nxt], V0), 0));
      clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
      ctx.beginPath(); xs.forEach((v, q) => (q ? ctx.lineTo : ctx.moveTo).call(ctx, ax.X(v), ax.Y(ys[q]))); ctx.lineTo(ax.R, ax.B); ctx.lineTo(ax.L, ax.B);
      ctx.fillStyle = pal.acc; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
      polyline(ctx, xs.map(ax.X), ys.map(ax.Y), pal.acc, 2);
      vdash(ctx, hlc('xi', pal, pal.mark), ax.X(R.x[nxt]), ax.T, ax.B, 1.6, []);
      ctx.restore();
      label(ctx, `drawn x_i = ${f2(R.x[nxt])}`, ax.R, ax.T + 14, hlc('xi', pal, pal.mark), 11.5, 'right');
    }
    // Running sum of log conditionals.
    const cs = cum[order], ax2 = axes(ctx, pal, C, { xmin: 0, xmax: D, ymin, ymax, xt: [0, 36, 72, 108, 144], yt: [ymin, (ymin + ymax) / 2, ymax].map(Math.round), xl: 'pixels generated', title: 'Sum of log conditionals', titleColor: hlc('logp', pal, pal.fg) });
    hdash(ctx, hlc('logp', pal, pal.muted), ax2.L, ax2.R, ax2.Y(R.logp), 1.4, [5, 4]);
    label(ctx, `log p(x) = ${f2(R.logp)}`, ax2.R - 4, ax2.Y(R.logp) + (R.logp > (ymin + ymax) / 2 ? 16 : -6), hlc('logp', pal, pal.muted), 11.5, 'right');
    polyline(ctx, cs.slice(0, m + 1).map((_, q) => ax2.X(q)), cs.slice(0, m + 1).map(ax2.Y), pal.acc, 2);
    dot(ctx, ax2.X(m), ax2.Y(cs[m]), 4.5, pal.mark);
    const best = argmax(pi);
    ro.innerHTML = `${order} order · <b>${m}</b> of 144 pixels · most likely digit <b>${best}</b> with π = <b>${f2(pi[best])}</b><br>` +
      `sum of log conditionals <b>${f3(cs[m])}</b>${m === D ? ` = log p(x) <b>${f3(R.logp)}</b> computed directly` : `, and the finished image has log p(x) = ${f3(R.logp)}`}`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 2: next-scale prediction ----------
function scaleScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8.2 : 0.5), { maxH: 0.78 }), ro = readout();
  let joint = true, idx = 0;
  const keys = [true, false];
  const btns = keys.map((k) => button(k ? 'Joint draw per scale' : 'Independent draw per token', () => { joint = k; setOn(btns, keys, k); draw(); }, k === joint ? 'on' : ''));
  const next = button('Next sample', () => { idx = (idx + 1) % NS2; draw(); });
  viz.append(c.cv, controls(...btns, next), ro,
    note(`Exact conditionals of the block-average maps given the coarser ones, computed from the mixture. The typicality test uses the exact log density; the shares are over ${NS2} seeded samples per mode.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, run = runs2(joint)[idx], st = stats2(joint);
    // The maps of one run, with the posterior that each scale drew its digit from.
    const top = wide ? h * 0.56 : h * 0.6, ncol = wide ? 4 : 2, gap = wide ? 36 : 18;
    const cw = (w - (ncol - 1) * gap) / ncol, rows = wide ? 1 : 2, rh = top / rows;
    const sz = Math.min(cw, rh - 70);
    label(ctx, 'One sample, scale by scale', 0, 14, hlc('r', pal, pal.fg), 12.5);
    run.maps.forEach((mp, q) => {
      const cx = (q % ncol) * (cw + gap), cy = 26 + Math.floor(q / ncol) * rh, x0 = cx + (cw - sz) / 2;
      drawImg(ctx, pal, mp.map, x0, cy, sz, hlc('r', pal, pal.line));
      label(ctx, `scale ${mp.r}: ${mp.r * mp.r} token${mp.r > 1 ? 's' : ''}`, cx + cw / 2, cy + sz + 15, pal.muted, 11.5, 'center');
      postBars(ctx, pal, [x0, cy + sz + 20, sz, 30], mp.pi, { color: hlc('pi', pal, pal.acc), labels: false });
      if (wide && q < run.maps.length - 1) { ctx.fillStyle = pal.muted; label(ctx, '→', cx + cw + gap / 2, cy + sz / 2 + 5, pal.muted, 16, 'center'); }
    });
    // Eight final samples and the typicality shares.
    const y0 = top + 8, n = 8, per = wide ? 8 : 4, s2 = Math.min((w - (per - 1) * 10) / per, (h - y0 - 70) / (wide ? 1 : 2));
    label(ctx, `Final samples (${joint ? 'joint' : 'independent'} draws), border by the typicality test`, 0, y0 + 12, pal.fg, 12.5);
    const samples = runs2(joint);
    for (let q = 0; q < n; q++) {
      const s = samples[(idx + q) % NS2], X = (q % per) * (s2 + 10), Y = y0 + 22 + Math.floor(q / per) * (s2 + 10);
      drawImg(ctx, pal, s.x, X, Y, s2, typColor(pal, typ(s.x)));
    }
    const yb = y0 + 22 + (wide ? 1 : 2) * (s2 + 10) + 4;
    typBar(ctx, pal, 0, yb, Math.min(w, 520), 12, st);
    const pj = posterior(run.x), bj = argmax(pj);
    ro.innerHTML = `${joint ? 'Joint draw of each scale (exact)' : 'Independent draw per token'} · 4 sequential steps for 190 tokens<br>` +
      `typical <b>${pct(st.typical)}</b>, too unlikely <b>${pct(st.low)}</b>, too likely <b>${pct(st.high)}</b> of ${NS2} samples · the shown sample is closest to digit ${bj} (posterior ${f2(pj[bj])})`;
  }
  return { draw };
}

// ---------- Step 3: one axis, steps against dependence ----------
const NAMES3 = { raster: 'Raster, b pixels per step', random: 'Random order, b per step', scale: 'Next scale, per-token draws', flow: 'Flow, N Euler steps' };
function axisScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.6 : 0.55), { maxH: 0.78 }), ro = readout();
  let samp = 'raster';
  const sel = { raster: 6, random: 6, scale: 3, flow: 2 };
  const sliders = Object.fromEntries(SAMPLERS3.map((s) => {
    const L = settings3(s), list = L.map((_, j) => j);
    const id = s === 'flow' ? 'N' : s === 'scale' ? 'r' : 'b';
    const tex = s === 'scale' ? '\\text{scales}' : undefined;
    const fmt = (j) => (s === 'scale' ? L[j].label.replace('scales ', '') : s === 'flow' ? String(L[j].steps) : L[j].label.replace('k = ', ''));
    const el = listSlider(id, list, sel[s], (j) => { sel[s] = j; draw(); }, fmt, tex);
    el.style.display = s === samp ? '' : 'none';
    return [s, el];
  }));
  const btns = SAMPLERS3.map((s) => button({ raster: 'Raster', random: 'Random order', scale: 'Next scale', flow: 'Flow (Euler)' }[s], () => {
    samp = s; setOn(btns, SAMPLERS3, s); SAMPLERS3.forEach((q) => { sliders[q].style.display = q === s ? '' : 'none'; }); draw();
  }, s === samp ? 'on' : ''));
  viz.append(c.cv, controls(...btns, ...SAMPLERS3.map((s) => sliders[s])), ro,
    note(`Every sampler uses the exact conditionals or the exact probability-flow field of the mixture, so the only error is the parallelism of each step. Shares are over ${NS3} seeded samples per setting.`));
  const colors = (pal) => ({ raster: pal.ramp[1], random: pal.ramp[3], scale: pal.ramp[0], flow: pal.ok });
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, col = colors(pal);
    const [A, B] = wide ? [[0, 0, w * 0.56, h], [w * 0.56 + 24, 0, w * 0.44 - 24, h]] : [[0, 0, w, h * 0.55], [0, h * 0.55 + 14, w, h * 0.45 - 14]];
    const lx = (v) => Math.log2(v);
    const ax = axes(ctx, pal, A, { xmin: 0, xmax: lx(144), ymin: 0, ymax: 1, xt: [1, 2, 4, 8, 16, 32, 64, 144].map(lx), xfmt: (v) => String(Math.round(2 ** v)), yt: [0, 0.25, 0.5, 0.75, 1], yfmt: pct, xl: 'sequential steps (log scale)', title: 'Typical share of the samples' });
    for (const s of SAMPLERS3) {
      const pts = settings3(s).map((_, j) => stats3(s, j)).sort((a, b) => a.steps - b.steps);
      const lw = s === samp ? 2.6 : 1.4;
      ctx.globalAlpha = s === samp ? 1 : 0.55;
      polyline(ctx, pts.map((p) => ax.X(lx(p.steps))), pts.map((p) => ax.Y(p.typical)), col[s], lw);
      pts.forEach((p) => dot(ctx, ax.X(lx(p.steps)), ax.Y(p.typical), s === samp ? 3.5 : 2.5, col[s]));
      ctx.globalAlpha = 1;
    }
    const cur = stats3(samp, sel[samp]);
    ring(ctx, ax.X(lx(cur.steps)), ax.Y(cur.typical), 8, pal.mark, 2);
    // Legend.
    SAMPLERS3.forEach((s, q) => {
      const Y = ax.B - 14 - (SAMPLERS3.length - 1 - q) * 17;
      ctx.fillStyle = col[s]; ctx.fillRect(ax.R - 190, Y - 8, 14, 3.5);
      label(ctx, NAMES3[s], ax.R - 170, Y - 3, s === samp ? pal.fg : pal.muted, 11.5, 'left', s === samp ? 600 : 500);
    });
    // Eight samples of the current setting.
    const [bx, by, bw0, bh] = B, bw = bw0 - 8, xs = runs3(samp, sel[samp]), per = 4, s2 = Math.min((bw - 3 * 10) / per, (bh - 90) / 2);
    const lab = settings3(samp)[sel[samp]].label.replace('k = ', 'b = ');
    label(ctx, `${NAMES3[samp].split(',')[0]}, ${lab}: ${cur.steps} step${cur.steps > 1 ? 's' : ''}`, bx, by + 14, pal.fg, 12.5);
    for (let q = 0; q < 8; q++) drawImg(ctx, pal, xs[q], bx + (q % per) * (s2 + 10), by + 26 + Math.floor(q / per) * (s2 + 10), s2, typColor(pal, typ(xs[q])));
    typBar(ctx, pal, bx, by + 26 + 2 * (s2 + 10) + 6, Math.min(bw, 4 * s2 + 30), 12, cur);
    ro.innerHTML = `${NAMES3[samp]} · ${lab} · <b>${cur.steps}</b> sequential steps<br>` +
      `typical <b>${pct(cur.typical)}</b>, too unlikely (mixed digits) <b>${pct(cur.low)}</b>, too likely (smoothed) <b>${pct(cur.high)}</b> of ${NS3} samples`;
  }
  return { draw };
}

// ---------- Step 4: discrete tokens lose information ----------
function vqScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.4 : 0.72), { maxH: 0.78 }), ro = readout();
  let Kc = 16;
  const sl = listSlider('K', K4_LIST, Kc, (v) => { Kc = v; draw(); });
  viz.append(c.cv, controls(sl), ro,
    note('The codebooks are found by k-means (k-means++ start, 15 iterations, fixed seeds) on the 2 × 2 patches of 400 draws, and the error is measured on 200 other draws. The images are exact draws of the mixture.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE;
    const [A, B] = wide ? [[0, 0, w * 0.5, h], [w * 0.5 + 24, 0, w * 0.5 - 24, h]] : [[0, 0, w, h * 0.5], [0, h * 0.5 + 14, w, h * 0.5 - 14]];
    const x = TEST4[0], q = quantize(x, Kc), err = x.map((v, i) => Math.min(1, Math.abs(q[i] - v)) * 2 - 1);
    const [ax0, ay0, aw, ah] = A, s = Math.min((aw - 2 * 14) / 3, ah * 0.5);
    [['draw x', x], [`${NT4} tokens, K = ${Kc}`, q], ['|error|', err]].forEach(([t, im], j) => {
      label(ctx, t, ax0 + j * (s + 14), ay0 + 14, j === 1 ? hlc('K', pal, pal.fg) : pal.fg, 12.5);
      drawImg(ctx, pal, im, ax0 + j * (s + 14), ay0 + 24, s);
    });
    // The codebook as a grid of 2 x 2 patches.
    const C = codebook(Kc), gy = ay0 + 24 + s + 30, per = wide ? 8 : 16, cs = Math.min(34, (aw - (per - 1) * 4) / per);
    label(ctx, `the ${Kc} codes, each a 2 × 2 patch`, ax0, gy - 8, hlc('K', pal, pal.muted), 11.5);
    const shown = C.slice(0, wide ? 32 : 64);
    shown.forEach((v, j) => drawImg(ctx, pal, v, ax0 + (j % per) * (cs + 4), gy + Math.floor(j / per) * (cs + 4), cs));
    if (C.length > shown.length) label(ctx, `and ${C.length - shown.length} more`, ax0, gy + Math.ceil(shown.length / per) * (cs + 4) + 12, pal.muted, 11.5);
    // Error per pixel against K on log axes.
    const lg = Math.log10, pts = K4_LIST.map((k) => ({ k, ...stats4(k) }));
    const ax = axes(ctx, pal, B, { xmin: 1, xmax: 8, ymin: lg(0.015), ymax: lg(0.6), xt: [1, 2, 3, 4, 5, 6, 7, 8], xfmt: (v) => String(2 ** v), yt: [0.02, 0.05, 0.1, 0.2, 0.5].map(lg), yfmt: (v) => String(+(10 ** v).toFixed(2)), xl: 'K (log scale)', title: 'Error per pixel over 200 test draws', titleColor: hlc('K', pal, pal.fg) });
    const ref = (k) => pts[3].mse * Math.sqrt(16 / k);
    polyline(ctx, [ax.X(1), ax.X(8)], [ax.Y(lg(ref(2))), ax.Y(lg(ref(256)))], hlc('d', pal, pal.muted), 1.3, [5, 4]);
    label(ctx, '∝ 1/√K', ax.X(1.5), ax.Y(lg(ref(2 ** 1.5))) + 18, hlc('d', pal, pal.muted), 11.5, 'center');
    hdash(ctx, hlc('eta', pal, pal.acc2), ax.L, ax.R, ax.Y(lg(V0)), 1.3, [2, 3]);
    label(ctx, 'pixel noise η² = 0.09', ax.L + 6, ax.Y(lg(V0)) - 6, hlc('eta', pal, pal.acc2), 11.5);
    polyline(ctx, pts.map((p) => ax.X(Math.log2(p.k))), pts.map((p) => ax.Y(lg(p.mse))), pal.acc, 2);
    pts.forEach((p) => dot(ctx, ax.X(Math.log2(p.k)), ax.Y(lg(p.mse)), 3.5, pal.acc));
    const cur = stats4(Kc);
    ring(ctx, ax.X(Math.log2(Kc)), ax.Y(lg(cur.mse)), 8, pal.mark, 2);
    ro.innerHTML = `K = <b>${Kc}</b> codes · <b>${cur.bits}</b> bits per image (${NT4} tokens × ${Math.log2(Kc)} bits)<br>` +
      `error per pixel <b>${f3(cur.mse)}</b>, against the pixel noise η² = ${f2(V0)}`;
  }
  return { draw };
}

// ---------- Step 5: a diffusion head for one continuous token ----------
function marScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.6 : 0.52), { maxH: 0.8 }), ro = readout();
  let m = 0, j = 43, geo = null;
  const sl = slider('m', { min: 0, max: 16, step: 1, value: m, fmt: String }, (v) => { m = v; draw(); });
  c.cv.addEventListener('pointerdown', (e) => {
    if (!geo) return;
    const [X, Y] = at(c, e), col = Math.floor((X - geo.x) / geo.cell), row = Math.floor((Y - geo.y) / geo.cell);
    if (col >= 0 && col < N_IMG && row >= 0 && row < N_IMG) { j = Math.floor((row * N_IMG + col) / 2); draw(); }
  });
  viz.append(c.cv, controls(sl), ro,
    note('Exact: the density of the token is the mixture of the eight digits weighted by the posterior given the visible tokens, the paths follow the exact probability-flow field of that density (what an ideal head learns), and the cross is the exact conditional mean. Click a pixel of the image to choose the token.'));
  let dens = null;
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, F = field5(j, m);
    let A, B, C;
    if (wide) { const s = Math.min(h - 40, w * 0.25), bw = Math.min(h, w * 0.4); A = [0, 0, s, h]; B = [s + 20, 0, bw, h]; C = [s + 20 + bw + 10, 0, w - (s + 30 + bw), h]; }
    else { const s = w * 0.44; A = [0, 0, s, s + 30]; C = [s + 16, 0, w - s - 16, s + 30]; B = [0, s + 44, w, h - s - 44]; }
    // The image with the visible tokens and the token to predict.
    const sz = Math.min(A[2], A[3] - 30), im = new Float64Array(D).fill(NaN);
    for (const q of RUN5.ord.slice(0, m)) for (const i of tokPix(q)) im[i] = RUN5.x[i];
    label(ctx, `${m} visible token${m === 1 ? '' : 's'}`, A[0], A[1] + 14, hlc('m', pal, pal.fg), 12.5);
    drawImg(ctx, pal, im, A[0], A[1] + 24, sz);
    const cell = sz / N_IMG, [pa] = tokPix(j);
    geo = { x: A[0], y: A[1] + 24, cell };
    ctx.strokeStyle = hlc('z', pal, pal.mark); ctx.lineWidth = 2.4;
    ctx.strokeRect(A[0] + (pa % N_IMG) * cell + 1, A[1] + 24 + Math.floor(pa / N_IMG) * cell + 1, 2 * cell - 2, cell - 2);
    // The 2D density of the token, the flow paths and the regression prediction.
    const [bx, by, bw, bh] = B, side = Math.min(bw - 50, bh - 92), L = bx + 40, T = by + 26, lim = 2.2;
    const X = (v) => L + side * (v + lim) / (2 * lim), Y = (v) => T + side * (lim - v) / (2 * lim);
    label(ctx, `Token k: pixels (${Math.floor(pa / N_IMG)}, ${pa % N_IMG}) and (${Math.floor(pa / N_IMG)}, ${pa % N_IMG + 1})`, L, by + 14, hlc('z', pal, pal.fg), 12.5);
    const n = 72;
    dens = dens ?? document.createElement('canvas');
    dens.width = dens.height = n;
    const g = dens.getContext('2d'), d = g.createImageData(n, n), ac = hexRGB(pal.acc), [a, b] = tokPix(j);
    let mx = 0; const vals = new Float64Array(n * n);
    for (let q = 0; q < n; q++) for (let p = 0; p < n; p++) {
      const u = -lim + 2 * lim * (p + 0.5) / n, v = lim - 2 * lim * (q + 0.5) / n;
      let s = 0; for (let k = 0; k < K; k++) if (F.pi[k] > 1e-9) s += F.pi[k] * gauss(u, DIGITS[k][a], V0) * gauss(v, DIGITS[k][b], V0);
      vals[q * n + p] = s; mx = Math.max(mx, s);
    }
    for (let i = 0; i < n * n; i++) { const u = Math.sqrt(vals[i] / mx); for (let ch = 0; ch < 3; ch++) d.data[4 * i + ch] = ac[ch]; d.data[4 * i + 3] = Math.round(220 * u); }
    g.putImageData(d, 0, 0);
    ctx.drawImage(dens, L, T, side, side);
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, side, side);
    [-2, -1, 0, 1, 2].forEach((v) => { label(ctx, String(v), X(v), T + side + 15, pal.muted, 11, 'center'); label(ctx, String(v), L - 6, Y(v) + 4, pal.muted, 11, 'right'); });
    label(ctx, 'first pixel', L + side, T + side + 28, pal.muted, 11.5, 'right');
    clip(ctx, L, T, side, side);
    for (const P of F.paths) {
      polyline(ctx, P.map((p) => X(p[0])), P.map((p) => Y(p[1])), hlc('epsth', pal, pal.fg), isHL('epsth') ? 1.6 : 1.1);
      dot(ctx, X(P[0][0]), Y(P[0][1]), 2.5, pal.muted);
      dot(ctx, X(P[P.length - 1][0]), Y(P[P.length - 1][1]), 3.5, hlc('epsth', pal, pal.fg));
    }
    const [cx, cy] = [X(F.mean[0]), Y(F.mean[1])];
    ctx.strokeStyle = pal.bad; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(cx - 8, cy - 8); ctx.lineTo(cx + 8, cy + 8); ctx.moveTo(cx - 8, cy + 8); ctx.lineTo(cx + 8, cy - 8); ctx.stroke();
    ctx.restore();
    // Legend for the plane.
    const ly = T + side + 46;
    if (ly + 22 < by + bh) {
      dot(ctx, L + 4, ly - 4, 3.5, pal.fg); label(ctx, 'diffusion head: ends of flow paths', L + 14, ly, pal.muted, 11.5);
      label(ctx, '×', L + 4, ly + 19, pal.bad, 14, 'center', 600); label(ctx, 'regression head: the conditional mean', L + 14, ly + 18, pal.muted, 11.5);
    }
    // Posterior given the context, and finished images from both heads.
    const [cx0, cy0, cw, ch] = C;
    postBars(ctx, pal, [cx0, cy0, cw, Math.min(110, ch * 0.3)], F.pi, { color: hlc(['c', 'pi'], pal, pal.acc), title: 'Context: posterior given m tokens', titleColor: hlc('c', pal, pal.fg) });
    const iy = cy0 + Math.min(110, ch * 0.3) + 22, s2 = Math.min((cw - 30) / 4, (ch - (iy - cy0) - 50) / 2);
    [['diff', 'Diffusion head'], ['mse', 'Regression head']].forEach(([hd, nm], r) => {
      const Y0 = iy + r * (s2 + 30);
      label(ctx, nm, cx0, Y0 + 12, pal.fg, 12);
      IMGS5(hd).forEach((im2, q) => drawImg(ctx, pal, im2, cx0 + q * (s2 + 10), Y0 + 20, s2, typColor(pal, typ(im2))));
    });
    const best = argmax(F.pi);
    ro.innerHTML = `token at pixels ${pa} and ${pa + 1} · m = <b>${m}</b> visible tokens${F.visible ? ' (this token is already among them)' : ''} · ${F.pi[best] > 0.5 ? `most likely digit <b>${best}</b> (π = ${f2(F.pi[best])})` : `no digit has π above 0.5 yet`}<br>` +
      `regression prediction (<b>${f2(F.mean[0])}</b>, <b>${f2(F.mean[1])}</b>) with expected squared error <b>${f3(F.mse)}</b>, against <b>${f3(2 * V0)}</b> for the pixel noise alone`;
  }
  return { draw };
}

// ---------- Step 6: one transformer, two losses ----------
const SEQ6 = [...Array(4).fill('t'), ...Array(9).fill('a'), ...Array(3).fill('t'), ...Array(9).fill('b'), ...Array(2).fill('t')];
const MODES6 = ['causal', 'transfusion', 'bidir'];
const allowed = (mode, i, k) => mode === 'bidir' || k <= i || (mode === 'transfusion' && SEQ6[i] !== 't' && SEQ6[i] === SEQ6[k]);
function transfusionScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 8 : 0.62), { maxH: 0.8 }), ro = readout();
  let mode = 'transfusion', hov = -1, geo = null;
  const btns = MODES6.map((k) => button({ causal: 'Causal', transfusion: 'Transfusion', bidir: 'Bidirectional' }[k], () => { mode = k; setOn(btns, MODES6, k); draw(); }, k === mode ? 'on' : ''));
  const pick = (e) => { if (!geo) return; const [, Y] = at(c, e), i = Math.floor((Y - geo.y) / geo.cs); const nh = i >= 0 && i < SEQ6.length ? i : -1; if (nh !== hov) { hov = nh; draw(); } };
  c.cv.addEventListener('pointermove', pick); c.cv.addEventListener('pointerdown', pick);
  c.cv.addEventListener('pointerleave', () => { hov = -1; draw(); });
  viz.append(c.cv, controls(...btns), ro, note('The sequence has 9 text tokens (grey), image A with 9 patches and image B with 9 patches. Hover or tap a row of the matrix to see which positions that token attends to.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, n = SEQ6.length;
    const typeCol = (q) => (q === 't' ? pal.muted : q === 'a' ? pal.acc : pal.acc2);
    const size = wide ? Math.min(h - 40, w * 0.5) : Math.min(w - 30, h * 0.62), cs = size / n, x0 = 22, y0 = 26;
    geo = { y: y0, cs };
    label(ctx, 'Attention mask: row = query, column = key', x0, 14, pal.fg, 12.5);
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = typeCol(SEQ6[i]); ctx.fillRect(x0 - 12, y0 + i * cs + 1, 8, cs - 2); ctx.fillRect(x0 + i * cs + 1, y0 + size + 4, cs - 2, 8);
      for (let k = 0; k < n; k++) {
        if (!allowed(mode, i, k)) continue;
        const extra = k > i, inImg = SEQ6[i] !== 't' && SEQ6[i] === SEQ6[k];
        ctx.fillStyle = !extra ? pal.fg : inImg ? hlc('n', pal, pal.acc) : pal.bad;
        ctx.globalAlpha = !extra ? (hov === i ? 0.75 : 0.42) : (hov === i ? 0.95 : 0.7);
        ctx.fillRect(x0 + k * cs + 0.5, y0 + i * cs + 0.5, cs - 1, cs - 1);
      }
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, size, size);
    for (const [s, e] of [[4, 13], [16, 25]]) { ctx.strokeStyle = pal.muted; ctx.setLineDash([3, 3]); ctx.strokeRect(x0 + s * cs, y0 + s * cs, (e - s) * cs, (e - s) * cs); ctx.setLineDash([]); }
    if (hov >= 0) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(x0, y0 + hov * cs, size, cs); }
    // Counts and legend.
    let pairs = 0, extraImg = 0, extraFut = 0;
    for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) if (allowed(mode, i, k)) { pairs++; if (k > i) { if (SEQ6[i] !== 't' && SEQ6[i] === SEQ6[k]) extraImg++; else extraFut++; } }
    const lx = wide ? x0 + size + 36 : x0, ly = wide ? y0 + 6 : y0 + size + 30;
    const rows = [
      [pal.fg, 0.42, 'earlier or same position (causal)'],
      [pal.acc, 0.7, 'later patch of the same image'],
      [pal.bad, 0.7, 'later token of another segment'],
    ];
    rows.forEach(([col, al, t], q) => { ctx.fillStyle = col; ctx.globalAlpha = al; ctx.fillRect(lx, ly + q * 20, 12, 12); ctx.globalAlpha = 1; label(ctx, t, lx + 20, ly + q * 20 + 10, pal.muted, 11.5); });
    const sy = ly + 76;
    const seqW = wide ? w - lx - 4 : w - lx - 4, tw = seqW / n;
    label(ctx, hov >= 0 ? `Position ${hov + 1} (${SEQ6[hov] === 't' ? 'text' : `image ${SEQ6[hov].toUpperCase()}`}) reads:` : 'The sequence', lx, sy, pal.fg, 12);
    for (let k = 0; k < n; k++) {
      const on = hov < 0 || allowed(mode, hov, k);
      ctx.fillStyle = typeCol(SEQ6[k]); ctx.globalAlpha = on ? 0.85 : 0.15;
      ctx.fillRect(lx + k * tw + 1, sy + 10, tw - 2, 16); ctx.globalAlpha = 1;
      if (k === hov) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(lx + k * tw + 0.5, sy + 9.5, tw - 1, 17); }
    }
    if (wide) {
      const ty = sy + 56;
      [['Text tokens', '9 terms of the language-model loss', pal.muted], ['Image A and image B', '18 patches in the diffusion loss', pal.acc]].forEach(([a, bb, col], q) => {
        label(ctx, a, lx, ty + q * 40, col, 12, 'left', 600); label(ctx, bb, lx, ty + q * 40 + 17, pal.muted, 11.5);
      });
    }
    const total = n * n;
    ro.innerHTML = `${{ causal: 'Causal mask', transfusion: 'Transfusion mask', bidir: 'Bidirectional mask' }[mode]} · <b>${pairs}</b> of ${total} query–key pairs allowed` +
      ` (causal ${n * (n + 1) / 2}${extraImg ? `, plus <b>${extraImg}</b> within the images` : ''}${extraFut ? `, plus <b>${extraFut}</b> that read later segments` : ''})<br>` +
      `loss terms: <b>9</b> next-token terms on the text and <b>18</b> denoising terms on the patches${mode === 'bidir' ? ' · text can no longer be generated left to right' : ''}`;
  }
  return { draw };
}

export const SCENES = { chain: chainScene, scale: scaleScene, axis: axisScene, vq: vqScene, mar: marScene, transfusion: transfusionScene };
