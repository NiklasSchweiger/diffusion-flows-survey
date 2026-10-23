// Live scenes of the §2.8 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, label, hlc, isHL, controls, slider, button, toggle, readout, note, arrow } from '../lib/tutorial.js';
import {
  alpha, sigma, logp2, LOGP_REF, DIGITS, N_IMG, tokens, PRESETS, B_LIST, snrEff, shiftT, upsample, blockAvg, noisy, blockNoiseSD,
  L3, K3_LIST, H3_LIST, permutation, N_TRAIN3, NS3, Q01, run3, encoderF, repaTarget, denoiser, T4, M4, R2,
  D5, BASIS5, TARGETS, linPred, xt5, predImage, N6, TRAIN6, attention, ARCHS, imgTokens,
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
const pct = (q) => `${Math.round(100 * q)}%`;
// 10 to an integer power with Unicode superscripts, for log axes.
const pow10 = (e) => `10${String(e).replace(/-/g, '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c])}`;
// Large counts in a compact form: 4096, 1.7 million, 3.1 billion.
const big = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(1)} billion` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} million` : v >= 1e4 ? `${Math.round(v / 1e3)} thousand` : String(Math.round(v)));
const short = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e4 ? `${Math.round(v / 1e3)}k` : String(Math.round(v)));
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
// Square greyscale image of any side n (values in [-1, 1]) drawn with the theme's background and foreground.
const hexRGB = (s) => { const m = s.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i); return m ? m.slice(1).map((q) => parseInt(q, 16)) : [128, 128, 128]; };
const imgCanvas = document.createElement('canvas');
function drawImg(ctx, pal, im, x, y, size, border = pal.line) {
  const n = Math.round(Math.sqrt(im.length));
  imgCanvas.width = imgCanvas.height = n;
  const g = imgCanvas.getContext('2d'), d = g.createImageData(n, n), bg = hexRGB(pal.bg), fg = hexRGB(pal.fg);
  for (let i = 0; i < im.length; i++) {
    const u = Math.max(0, Math.min(1, (im[i] + 1) / 2));
    for (let c = 0; c < 3; c++) d.data[4 * i + c] = Math.round(bg[c] + (fg[c] - bg[c]) * u);
    d.data[4 * i + 3] = 255;
  }
  g.putImageData(d, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(imgCanvas, x, y, size, size); ctx.imageSmoothingEnabled = true;
  ctx.strokeStyle = border; ctx.lineWidth = border === pal.mark ? 2 : 1; ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}
function star(ctx, X, Y, color, R = 7.5) {
  ctx.beginPath(); for (let q = 0; q < 10; q++) { const r = q % 2 ? R * 0.4 : R, ang = -Math.PI / 2 + q * Math.PI / 5; (q ? ctx.lineTo : ctx.moveTo).call(ctx, X + r * Math.cos(ang), Y + r * Math.sin(ang)); }
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}
// A draggable probe inside the rectangle returned by box(); on() receives world coordinates.
function dragProbe(c, box, on, lim = [3, 2.8]) {
  const move = (e) => {
    const B = box();
    if (!B || !(e.buttons & 1) && e.type !== 'pointerdown') return;
    const r = c.cv.getBoundingClientRect(), [x0, y0, pw, ph] = B.rect, X = e.clientX - r.left, Y = e.clientY - r.top;
    if (X < x0 || X > x0 + pw || Y < y0 || Y > y0 + ph) return;
    const p = B.v.wx(X, Y); on([Math.max(-lim[0], Math.min(lim[0], p[0])), Math.max(-lim[1], Math.min(lim[1], p[1]))]);
  };
  c.cv.addEventListener('pointerdown', move); c.cv.addEventListener('pointermove', move);
}
// Axes of a small line plot; returns the maps from data to canvas.
function axes(ctx, pal, [x0, y0, w, h], { xmin = 0, xmax = 1, ymin, ymax, xt = [0, 0.5, 1], yt, xl = 't', title, yfmt = String, xfmt = String }) {
  const L = x0 + 40, R = x0 + w - 10, T = y0 + 30, B = y0 + h - 30;
  const X = (v) => L + (R - L) * (v - xmin) / (xmax - xmin), Y = (v) => B - (B - T) * (v - ymin) / (ymax - ymin);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
  yt.forEach((q) => label(ctx, yfmt(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'));
  xt.forEach((q) => label(ctx, xfmt(q), X(q), B + 16, pal.muted, 11, 'center'));
  label(ctx, xl, R, B + 28, pal.muted, 11.5, 'right');
  if (title) label(ctx, title, L, T - 12, pal.fg, 12.5);
  return { X, Y, L, R, T, B };
}
function polyline(ctx, xs, ys, color, lw = 2, dash = []) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x, ys[i]));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}
function setOn(btns, keys, k) { btns.forEach((b, j) => b.classList.toggle('on', keys[j] === k)); }

// ---------- Step 1: from pixels to tokens ----------
const H_LIST = [256, 512, 704, 1024, 1536, 2048], F_LIST = [4, 8, 16, 32], P_LIST = [1, 2, 4], C_LIST = [4, 8, 16, 32, 64, 128];
function tokenScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.6 : 0.62)), ro = readout();
  let s = { ...PRESETS.sd }, preset = 'sd';
  const keys = Object.keys(PRESETS);
  const sl = {};
  const setPreset = (k) => { preset = k; s = { ...PRESETS[k] }; setOn(btns, keys, k); sl.HW.set(H_LIST.indexOf(s.H)); sl.f.set(F_LIST.indexOf(s.f)); sl.p.set(P_LIST.indexOf(s.p)); sl.C.set(C_LIST.indexOf(s.C)); draw(); };
  const btns = keys.map((k) => button(PRESETS[k].name, () => setPreset(k), k === preset ? 'on' : ''));
  const free = () => { preset = null; setOn(btns, keys, null); };
  sl.HW = listSlider('HW', H_LIST, s.H, (v) => { if (v !== s.H) free(); s.H = v; draw(); }, (v) => `H = ${v}`);
  sl.f = listSlider('f', F_LIST, s.f, (v) => { if (v !== s.f) free(); s.f = v; draw(); });
  sl.p = listSlider('p', P_LIST, s.p, (v) => { if (v !== s.p) free(); s.p = v; draw(); });
  sl.C = listSlider('C', C_LIST, s.C, (v) => { if (v !== s.C) free(); s.C = v; draw(); });
  viz.append(c.cv, controls(...btns), controls(sl.HW, sl.f, sl.p, sl.C), ro,
    note('Exact arithmetic. The width follows the aspect ratio of the preset (square for images, 1216 × 704 for the video) and is rounded to a multiple of f·p. The picture is a stand-in; only the grids matter. The bars use a logarithmic scale.'));
  function geom() {
    const asp = s.T > 1 ? 1216 / 704 : 1, q = s.f * s.p, W = Math.max(q, Math.round(s.H * asp / q) * q), H = Math.max(q, Math.round(s.H / q) * q);
    return { ...s, H, W, ...tokens({ ...s, H, W }) };
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, G = geom();
    const left = wide ? [0, 0, Math.round(w * 0.52), h] : [0, 0, w, Math.round(h * 0.55)];
    const right = wide ? [left[2] + 30, 0, w - left[2] - 30, h] : [0, left[3] + 16, w, h - left[3] - 16];
    // The image with its token grid.
    const [lx, ly, lw, lh] = left, asp = G.W / G.H, iw = Math.min(lw - 150, (lh - 60) * asp), ih = iw / asp, ix = lx + 8, iy = ly + 36;
    tag(ctx, pal, `${G.W} × ${G.H}${G.T > 1 ? ` × ${G.T} frames` : ''} pixels`, ix + 6, ly + 20, pal.fg, 13);
    const grad = ctx.createLinearGradient(ix, iy, ix + iw, iy + ih);
    grad.addColorStop(0, pal.ramp[0]); grad.addColorStop(0.5, pal.ramp[1]); grad.addColorStop(1, pal.ramp[2]);
    ctx.globalAlpha = 0.28; ctx.fillStyle = grad; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
    drawImg(ctx, pal, DIGITS[3].map((v) => v * 0.6 - 0.2), ix + (iw - ih * 0.8) / 2, iy + ih * 0.1, ih * 0.8, 'rgba(0,0,0,0)');
    ctx.globalAlpha = 0.28; ctx.fillStyle = grad; ctx.fillRect(ix, iy, iw, ih); ctx.globalAlpha = 1;
    const nx = G.W / (G.f * G.p), ny = G.H / (G.f * G.p), every = Math.max(1, Math.ceil(nx / 48));
    ctx.strokeStyle = hlc(['p', 'N'], pal, pal.acc); ctx.lineWidth = isHL('p', 'N') ? 1.4 : 0.8; ctx.globalAlpha = 0.75; ctx.beginPath();
    for (let i = 0; i <= nx; i += every) { const X = ix + iw * i / nx; ctx.moveTo(X, iy); ctx.lineTo(X, iy + ih); }
    for (let j = 0; j <= ny; j += every) { const Y = iy + ih * j / ny; ctx.moveTo(ix, Y); ctx.lineTo(ix + iw, Y); }
    ctx.stroke(); ctx.globalAlpha = 1;
    frame(ctx, pal, ix, iy, iw, ih);
    label(ctx, every > 1 ? `one in ${every} token boundaries drawn` : `${nx} × ${ny} tokens${G.T > 1 ? ` × ${Math.ceil(G.T / G.fT)} latent frames` : ''}`, ix, iy + ih + 18, hlc(['p', 'N'], pal, pal.acc), 12);
    // Inset: one token = p x p latent positions, each f x f pixels.
    const S = Math.min(120, lw - iw - 34), sx = ix + iw + 22, sy = iy + 18;
    if (S > 50) {
      label(ctx, 'one token', sx, sy - 8, pal.fg, 12);
      ctx.fillStyle = pal.panel; ctx.fillRect(sx, sy, S, S);
      if (G.f * G.p <= 32) { ctx.strokeStyle = pal.line; ctx.lineWidth = 0.5; ctx.beginPath(); const n = G.f * G.p; for (let i = 1; i < n; i++) { ctx.moveTo(sx + S * i / n, sy); ctx.lineTo(sx + S * i / n, sy + S); ctx.moveTo(sx, sy + S * i / n); ctx.lineTo(sx + S, sy + S * i / n); } ctx.stroke(); }
      ctx.strokeStyle = hlc('f', pal, pal.ramp[2]); ctx.lineWidth = isHL('f') ? 2.4 : 1.4; ctx.beginPath();
      for (let i = 1; i < G.p; i++) { ctx.moveTo(sx + S * i / G.p, sy); ctx.lineTo(sx + S * i / G.p, sy + S); ctx.moveTo(sx, sy + S * i / G.p); ctx.lineTo(sx + S, sy + S * i / G.p); }
      ctx.stroke();
      ctx.strokeStyle = hlc(['p', 'N'], pal, pal.acc); ctx.lineWidth = 2; ctx.strokeRect(sx + 1, sy + 1, S - 2, S - 2);
      label(ctx, wide ? `${G.p} × ${G.p} latent positions` : `${G.p} × ${G.p} latents`, sx, sy + S + 18, hlc('p', pal, pal.ramp[2]), 11.5);
      label(ctx, wide ? `of ${G.f} × ${G.f} pixels, ${G.C} channels` : `of ${G.f} × ${G.f} px, C = ${G.C}`, sx, sy + S + 34, hlc(['f', 'C'], pal, pal.muted), 11.5);
      if (G.f * G.p > 32) label(ctx, '(pixel grid not drawn)', sx, sy + S + 50, pal.muted, 11);
    }
    // Bars on a logarithmic scale.
    const [rx, ry, rw, rh] = wide ? right : [right[0] + 10, right[1], right[2] - 10, right[3]], rows = [['pixel values', G.pixels, pal.muted, 'HW'], ['latent values', G.latent, pal.ramp[2], 'C'], ['tokens N', G.N, pal.acc, 'N'], ['attention pairs N²', G.pairs, pal.ramp[3], 'N']];
    const lmax = 11, bx = rx + (wide ? 0 : 0), bw = rw - 70, top = ry + (wide ? 50 : 26), gap = (rh - (wide ? 80 : 40)) / rows.length;
    label(ctx, 'count per image or video (log scale)', bx, top - 20, pal.fg, 12.5);
    for (let e = 0; e <= lmax; e += wide ? 3 : 3) { const X = bx + bw * e / lmax; vdash(ctx, pal.line, X, top - 6, top + gap * rows.length - 8, 1, [2, 4]); label(ctx, pow10(e), X, top + gap * rows.length + 6, pal.muted, 10.5, 'center'); }
    rows.forEach(([name, v, col, id], j) => {
      const y0 = top + j * gap, bh = Math.min(20, gap - 22), len = bw * Math.log10(Math.max(1, v)) / lmax;
      label(ctx, name, bx, y0 + 10, pal.muted, 11.5);
      ctx.fillStyle = hlc(id, pal, col); ctx.globalAlpha = 0.85; ctx.fillRect(bx, y0 + 16, len, bh); ctx.globalAlpha = 1;
      label(ctx, short(v), bx + len + 6, y0 + 16 + bh - 4, pal.fg, 12);
    });
    ro.innerHTML = `${preset ? `${PRESETS[preset].name} · ` : ''}latent ${G.H / G.f} × ${G.W / G.f}${G.T > 1 ? ` × ${Math.ceil(G.T / G.fT)}` : ''} × ${G.C} · dimension reduced by a factor of <b>${G.reduction.toFixed(G.reduction % 1 ? 1 : 0)}</b><br>` +
      `tokens N = <b>${G.N.toLocaleString('en-US')}</b> · attention pairs N² = <b>${big(G.pairs)}</b> per layer and head`;
  }
  return { draw };
}

// ---------- Step 2: the schedule shift ----------
function shiftScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 5.2 : 0.8)), ro = readout();
  let b = 4, t = 0.5, shift = false, d = 3;
  const dBtns = DIGITS.map((_, i) => button(String(i), () => { d = i; setOn(dBtns, DIGITS.map((_, j) => j), i); draw(); }, i === d ? 'on' : ''));
  viz.append(c.cv, controls(listSlider('b', B_LIST, b, (v) => { b = v; draw(); }), slider('t', { min: 0.02, max: 0.98, step: 0.01, value: t }, (v) => { t = v; draw(); }),
    toggle('Shift the schedule', shift, (on) => { shift = on; draw(); })), controls(...dBtns), ro,
  note('Exact: each resolution has its own fixed draw of independent N(0, 1) noise in every pixel, and the images are x<sub>t</sub> = α<sub>t</sub>x<sub>1</sub> + σ<sub>t</sub>x<sub>0</sub> on the linear path. The block average divides the noise standard deviation by b; the readout gives the value measured on the fixed noise.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, tt = shift ? shiftT(t, b) : t, im = DIGITS[d];
    const base = noisy(im, 1, t), hi = noisy(im, b, tt), avg = blockAvg(hi, b);
    const boxes = wide ? panels(w, h, 4, 18) : [[0, 0, w / 2 - 7, h * 0.48], [w / 2 + 7, 0, w / 2 - 7, h * 0.48], [0, h * 0.52, w / 2 - 7, h * 0.48], [w / 2 + 7, h * 0.52, w / 2 - 7, h * 0.48]];
    const imgs = [[base, `base, 12 × 12, t = ${f2(t)}`, 'x'], [hi, `${12 * b} × ${12 * b}, t${shift ? "'" : ''} = ${f2(tt)}`, shift ? 'ts' : 'x'], [avg, `block average of ${b} × ${b}`, 'b']];
    imgs.forEach(([I, name, id], j) => {
      const [x0, y0, pw, ph] = boxes[j], S = Math.min(pw - 8, ph - 34);
      drawImg(ctx, pal, I, x0 + (pw - S) / 2, y0 + 26, S, isHL(id) ? pal.mark : pal.line);
      label(ctx, name, x0 + pw / 2, y0 + 16, isHL(id) ? pal.mark : pal.fg, wide ? 12 : 11, 'center');
    });
    // log SNR of the coarse content against t.
    const P = boxes[3], lo = -3, hiE = 3;
    const A = axes(ctx, pal, P, { ymin: lo, ymax: hiE, yt: [-2, 0, 2], yfmt: pow10, title: 'coarse SNR' });
    const ts = Array.from({ length: 99 }, (_, i) => (i + 1) / 100), cl = (v) => Math.max(lo, Math.min(hiE, Math.log10(v)));
    polyline(ctx, ts.map(A.X), ts.map((q) => A.Y(cl(snrEff(q, 1)))), pal.muted, 2);
    polyline(ctx, ts.map(A.X), ts.map((q) => A.Y(cl(snrEff(q, b)))), hlc(['b', 'snr'], pal, pal.acc), 2);
    const Y0 = A.Y(cl(snrEff(t, 1)));
    dot(ctx, A.X(t), Y0, 4, pal.muted);
    dot(ctx, A.X(tt), A.Y(cl(snrEff(tt, b))), 4.5, hlc(shift ? 'ts' : 'b', pal, pal.acc));
    if (shift) { hdash(ctx, hlc('ts', pal, pal.fg), A.X(tt), A.X(t), Y0, 1.2); }
    label(ctx, 'b = 1', A.X(0.7) + 6, A.Y(cl(snrEff(0.7, 1))) + 14, pal.muted, 11.5);
    label(ctx, `b = ${b}`, A.X(0.3), A.Y(cl(snrEff(0.3, b))) - 8, hlc('b', pal, pal.acc), 11.5, 'right');
    const sd = blockNoiseSD(b) * sigma(tt);
    ro.innerHTML = `b = <b>${b}</b> · t = <b>${f2(t)}</b>${shift ? ` · shifted time t′ = <b>${tt.toFixed(3)}</b>` : ''} · SNR of the base image <b>${snrEff(t, 1).toFixed(2)}</b> · SNR of the block averages <b>${snrEff(tt, b).toFixed(2)}</b><br>` +
      `noise standard deviation in the block average: σ<sub>t</sub>/b = <b>${(sigma(tt) / b).toFixed(3)}</b>, measured on the fixed noise <b>${sd.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 3: an encoder that reconstructs perfectly ----------
function latentScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 5.6 : 0.5)), ro = readout();
  let k = 6, h = 0.3;
  viz.append(c.cv, controls(listSlider('k', K3_LIST, k, (v) => { k = v; draw(); }), listSlider('h', H3_LIST, h, (v) => { h = v; draw(); }, f2)), ro,
    note(`Exact: ${N_TRAIN3} training points from the 2D mixture, and ${NS3} samples of the smoothed model, each a training latent plus h times a fixed noise draw. The decoder inverts the encoder to rounding error. A sample counts as far from the data when log p<sub>data</sub> is below the value that only 1% of real data fall under (${Q01.toFixed(2)}). Mean log p<sub>data</sub> of real data: ${LOGP_REF.toFixed(2)}.`));
  // Band colours of the k strips, by their position in data space.
  const bandCol = (pal, j) => pal.ramp[j % 4];
  function bands(ctx, v, box, kk, where) {
    if (kk === 1) return;
    const W = 2 * L3 / kk, p = permutation(kk);
    for (let j = 0; j < kk; j++) {
      const pos = where === 'latent' ? p[j] : j, [X0] = v.px(-L3 + pos * W, 0), [X1] = v.px(-L3 + (pos + 1) * W, 0);
      ctx.fillStyle = bandCol(palette(), j); ctx.globalAlpha = isHL('k', 'Enc', 'Dec') ? 0.2 : 0.1; ctx.fillRect(X0, box[1], X1 - X0, box[3]); ctx.globalAlpha = 1;
      if (pos > 0) vdash(ctx, palette().line, X0, box[1], box[1] + box[3], 1, [2, 3]);
    }
  }
  function draw() {
    c.fit();
    const { ctx, w, h: H } = c, pal = palette(), B = panels(w, H, 3), I = run3(1, h), R = run3(k, h);
    const specs = [[B[0], I.X, I, 'identity encoder, decoded samples', 'data', null], [B[1], R.Z, R, `strip encoder, k = ${k}: latent space`, 'latent', R.lat], [B[2], R.X, R, 'strip encoder, decoded samples', 'data', null]];
    specs.forEach(([box, P, run, name, where, lat], j) => {
      const v = view(...box, 3.75, 3.0);
      if (where === 'data') heat(ctx, box, v, 0.45);
      clip(ctx, ...box);
      bands(ctx, v, box, j === 0 ? 1 : k, where);
      if (lat) { ctx.globalAlpha = 0.5; for (let i = 0; i < N_TRAIN3; i++) dot(ctx, ...v.px(lat[2 * i], lat[2 * i + 1]), 1.5, hlc('z', pal, pal.muted)); ctx.globalAlpha = 1; }
      for (let i = 0; i < NS3; i++) {
        const [X, Y] = v.px(P[2 * i], P[2 * i + 1]);
        const far = where === 'data' && logp2(P[2 * i], P[2 * i + 1]) < Q01;
        if (far) dot(ctx, X, Y, 2.8, pal.bad); else ring(ctx, X, Y, 2.2, hlc(where === 'latent' ? 'z' : 'Dec', pal, pal.fg), 1);
      }
      ctx.restore();
      frame(ctx, pal, ...box);
      tag(ctx, pal, name, box[0] + 12, box[1] + 22, pal.fg, w >= WIDE ? 12 : 11.5);
      if (where === 'data') tag(ctx, pal, `far from the data ${pct(run.far)}`, box[0] + box[2] - 12, box[1] + box[3] - 10, run.far > I.far + 0.01 && j ? pal.bad : pal.muted, 12, 'right');
    });
    ro.innerHTML = `k = <b>${k}</b> · h = <b>${f2(h)}</b> · largest reconstruction error ${R.rec.toExponential(0)}<br>` +
      `identity encoder: mean log p<sub>data</sub> <b>${I.logp.toFixed(2)}</b>, far from the data <b>${pct(I.far)}</b> · strip encoder: mean log p<sub>data</sub> <b>${R.logp.toFixed(2)}</b>, far from the data <b>${pct(R.far)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: representation alignment ----------
function repaScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7 : 0.5)), ro = readout();
  let t = 0.5, probe = [-0.2, 0.3], L = null;
  const tgt = new Float64Array(3), fx = new Float64Array(3);
  viz.append(c.cv, controls(slider('t', { min: 0.02, max: 0.98, step: 0.01, value: t }, (v) => { t = v; draw(); })), ro,
    note(`Exact: P(k | x<sub>t</sub>) and x̂<sub>1</sub> are closed-form quantities of the Gaussian mixture, and f(x<sub>1</sub>) = P(k | x<sub>1</sub>). The curves are R² = 1 − E‖g − f(x<sub>1</sub>)‖² / Var f(x<sub>1</sub>) over ${M4} fixed pairs (x<sub>1</sub>, x<sub>0</sub>) at ${T4.length} times; a negative value means the prediction is worse than the constant mean. The heat map is p<sub>data</sub>. Drag in the plot to move x<sub>t</sub>.`));
  dragProbe(c, () => L && { rect: L.plot, v: view(...L.plot) }, (p) => { probe = p; draw(); });
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE;
    L = wide ? { plot: [0, 0, Math.round(w * 0.42), h], bars: [Math.round(w * 0.42) + 30, 0, Math.round(w * 0.2), h], curve: [Math.round(w * 0.62) + 40, 0, w - Math.round(w * 0.62) - 40, h] }
      : { plot: [0, 0, w, Math.round(h * 0.5)], bars: [10, Math.round(h * 0.5) + 10, Math.round(w * 0.4), Math.round(h * 0.5) - 10], curve: [Math.round(w * 0.4) + 6, Math.round(h * 0.5) + 10, w - Math.round(w * 0.4) - 6, Math.round(h * 0.5) - 10] };
    const v = view(...L.plot), [hx, hy] = denoiser(probe[0], probe[1], t);
    repaTarget(probe[0], probe[1], t, tgt); encoderF(hx, hy, fx);
    heat(ctx, L.plot, v, 0.5);
    clip(ctx, ...L.plot);
    const [PX, PY] = v.px(...probe), [HX, HY] = v.px(hx, hy);
    arrow(ctx, PX, PY, HX, HY, hlc('hx1', pal, pal.fg), 1.6, 9);
    ring(ctx, PX, PY, isHL('x') ? 8 : 6.5, hlc('x', pal, pal.acc), 2.2);
    star(ctx, HX, HY, hlc('hx1', pal, pal.fg));
    ctx.restore();
    frame(ctx, pal, ...L.plot);
    tag(ctx, pal, `t = ${f2(t)} · ring: x_t · star: x̂_1`, L.plot[0] + 12, L.plot[1] + 22, pal.fg, 12.5);
    // Bars for the three classes.
    const [bx, by, bw, bh] = L.bars, top = by + 44, bot = by + bh - 62, gw = bw / 3;
    label(ctx, 'class probabilities', bx, by + 22, pal.fg, 12.5);
    hdash(ctx, pal.line, bx, bx + bw, top); label(ctx, '1', bx + bw + 4, top + 4, pal.muted, 10.5);
    for (let q = 0; q < 3; q++) {
      const X = bx + q * gw + 6, ww = gw - 12, col = classColor(pal, q);
      ctx.fillStyle = hlc('Pk', pal, col); ctx.globalAlpha = 0.8; ctx.fillRect(X, bot - (bot - top) * tgt[q], ww, (bot - top) * tgt[q]); ctx.globalAlpha = 1;
      ctx.strokeStyle = hlc('feat', pal, pal.fg); ctx.lineWidth = isHL('feat') ? 2.6 : 1.6; ctx.strokeRect(X + 3, bot - (bot - top) * fx[q], ww - 6, (bot - top) * fx[q]);
      label(ctx, `k = ${q + 1}`, X + ww / 2, bot + 16, pal.muted, 11, 'center');
    }
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, bot + 0.5); ctx.lineTo(bx + bw, bot + 0.5); ctx.stroke();
    label(ctx, 'filled: P(k | x_{t})', bx, bot + 36, hlc('Pk', pal, pal.muted), 11);
    label(ctx, 'outlined: f(x̂_{1})', bx, bot + 52, hlc('feat', pal, pal.muted), 11);
    // R^2 curves along t.
    const A = axes(ctx, pal, L.curve, { ymin: -0.4, ymax: 1, yt: [0, 0.5, 1], title: 'variance of f(x_{1}) explained' });
    hdash(ctx, pal.line, A.L, A.R, A.Y(0));
    const tx = Array.from(T4, A.X);
    polyline(ctx, tx, R2.sem.map((q) => A.Y(Math.max(-0.4, q))), hlc('Pk', pal, pal.acc), isHL('Pk') ? 3 : 2);
    polyline(ctx, tx, R2.fx.map((q) => A.Y(Math.max(-0.4, q))), hlc('feat', pal, pal.fg), isHL('feat') ? 2.6 : 1.6, [5, 4]);
    vdash(ctx, hlc('t', pal, pal.muted), A.X(t), A.T, A.B, 1, [1, 3]);
    label(ctx, 'P(k | x_{t})', A.X(0.35), A.Y(0.62), hlc('Pk', pal, pal.acc), 11.5, 'right');
    label(ctx, 'f(x̂_{1})', A.X(0.42) + 4, A.Y(0.05), hlc('feat', pal, pal.fg), 11.5);
    const i = Math.round(t * 40), gap = Math.max(...[0, 1, 2].map((q) => Math.abs(tgt[q] - fx[q])));
    ro.innerHTML = `t = <b>${f2(t)}</b> · x<sub>t</sub> = (${f2(probe[0])}, ${f2(probe[1])}) · x̂<sub>1</sub> = (${f2(hx)}, ${f2(hy)})<br>` +
      `target P(k | x<sub>t</sub>) = (${[...tgt].map(f2).join(', ')}) · f(x̂<sub>1</sub>) = (${[...fx].map(f2).join(', ')}) · largest difference <b>${f2(gap)}</b><br>` +
      `variance explained at t ≈ ${(i / 40).toFixed(3)}: by the target <b>${f2(R2.sem[i])}</b>, by f(x̂<sub>1</sub>) <b>${f2(R2.fx[i])}</b>`;
  }
  return { draw };
}

// ---------- Step 5: which target a narrow network can learn ----------
const TNAME = { x1: 'predict x_1', x0: 'predict x_0', v: 'predict u' }, TSYM = { x1: 'x1', x0: 'x0', v: 'vel' };
function pixelScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.4 : 0.66)), ro = readout();
  let d = 3, t = 0.5, r = 16;
  const curves = new Map();
  const curveAt = (tt) => {
    if (!curves.has(tt)) { if (curves.size > 40) curves.clear(); curves.set(tt, Object.fromEntries(TARGETS.map((q) => [q, Float64Array.from({ length: D5 }, (_, j) => linPred(q, tt, j + 1).err)]))); }
    return curves.get(tt);
  };
  const dBtns = DIGITS.map((_, i) => button(String(i), () => { d = i; setOn(dBtns, DIGITS.map((_, j) => j), i); draw(); }, i === d ? 'on' : ''));
  viz.append(c.cv, controls(slider('t', { min: 0.05, max: 0.95, step: 0.05, value: t }, (v) => { t = Math.round(v * 100) / 100; draw(); }),
    slider('r', { min: 1, max: D5, step: 1, value: r, fmt: String }, (v) => { r = v; draw(); })), controls(...dBtns), ro,
  note(`Exact: the eight digits from §2.7 are the data, equally likely, and the noise has a fixed draw per digit. Each predictor is the best linear map of rank r from x<sub>t</sub> to its target (with a free bias), computed from the mean and covariance in closed form, and its prediction is converted to x̂<sub>1</sub>. The numbers are the expected squared error E‖x̂<sub>1</sub> − x<sub>1</sub>‖² over data and noise. A linear map stands in for a network whose narrowest layer has r units.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, x = xt5(d, t), Cv = curveAt(t);
    const preds = TARGETS.map((q) => { const P = linPred(q, t, r); return { q, P, im: predImage(x, t, P) }; });
    const imgArea = wide ? [0, 0, Math.round(w * 0.6), h] : [0, 0, w, Math.round(h * 0.55)], plot = wide ? [imgArea[2] + 24, 0, w - imgArea[2] - 24, h] : [0, imgArea[3] + 10, w, h - imgArea[3] - 10];
    // Images: x_1 and x_t on the first row, the three predictions below.
    const [ax, ay, aw, ah] = imgArea, S = Math.min((aw - 40) / 3, (ah - 90) / 2), g = (aw - 3 * S) / 2;
    const cell = (I, col, row, name, id, sub, subCol) => {
      const X = ax + col * (S + g / 1), Y = ay + 22 + row * (S + 46);
      drawImg(ctx, pal, I, X, Y, S, isHL(id) ? pal.mark : pal.line);
      label(ctx, name, X + S / 2, Y - 7, isHL(id) ? pal.mark : pal.fg, 12, 'center');
      if (sub) label(ctx, sub, X + S / 2, Y + S + 16, subCol, 11.5, 'center');
    };
    cell(DIGITS[d], 0, 0, 'clean x_1', 'x1');
    cell(x, 1, 0, `x_t, t = ${f2(t)}`, 'x');
    const errs = preds.map((p) => p.P.err), best = Math.min(...errs);
    preds.forEach((p, j) => cell(p.im, j, 1, `x̂_1, ${TNAME[p.q]}`, TSYM[p.q], `${wide ? 'expected ' : ''}error ${p.P.err < 100 ? p.P.err.toFixed(2) : Math.round(p.P.err)}`, p.P.err <= best * 1.0001 ? pal.fg : pal.muted));
    // Expected error against r on a log scale.
    const all = TARGETS.flatMap((q) => [...Cv[q]]), emin = Math.floor(Math.log10(Math.max(1e-3, Math.min(...all)))), emax = Math.ceil(Math.log10(Math.max(...all)));
    const A = axes(ctx, pal, plot, { xmin: 1, xmax: D5, xt: [1, 7, 72, 144], xl: 'r', ymin: emin, ymax: emax, yt: Array.from({ length: emax - emin + 1 }, (_, i) => emin + i).filter((q, i, arr) => arr.length <= 5 || i % 2 === 0), yfmt: pow10, title: 'expected error of x̂_1 against the width r' });
    const rs = Array.from({ length: D5 }, (_, j) => j + 1), cols = { x1: pal.acc, x0: pal.ramp[2], v: pal.ramp[3] };
    TARGETS.forEach((q) => polyline(ctx, rs.map(A.X), [...Cv[q]].map((e) => A.Y(Math.log10(Math.max(1e-3, e)))), hlc(TSYM[q], pal, cols[q]), isHL(TSYM[q]) ? 3 : 2));
    vdash(ctx, hlc('r', pal, pal.muted), A.X(r), A.T, A.B, 1.2, [2, 3]);
    TARGETS.forEach((q, j) => { dot(ctx, A.X(r), A.Y(Math.log10(Math.max(1e-3, Cv[q][r - 1]))), 4, hlc(TSYM[q], pal, cols[q])); label(ctx, TNAME[q], A.R - 6, A.T + 16 + 15 * j, hlc(TSYM[q], pal, cols[q]), 11.5, 'right'); });
    const kept = (q) => { const P = preds.find((p) => p.q === q).P; let n = 0; for (let j = 0; j < BASIS5.rank; j++) n += P.keep[j]; return n; };
    ro.innerHTML = `t = <b>${f2(t)}</b> · width r = <b>${r}</b> of D = ${D5} · expected error of x̂<sub>1</sub>: predict x<sub>1</sub> <b>${errs[0].toFixed(2)}</b>, predict x<sub>0</sub> <b>${errs[1].toFixed(2)}</b>, predict u <b>${errs[2].toFixed(2)}</b><br>` +
      `data directions kept: x<sub>1</sub> ${kept('x1')} of ${BASIS5.rank}, x<sub>0</sub> ${kept('x0')} of ${BASIS5.rank}, u ${kept('v')} of ${BASIS5.rank} · (D − r)σ<sub>t</sub>²/α<sub>t</sub>² = ${((D5 - r) * sigma(t) ** 2 / alpha(t) ** 2).toFixed(2)}`;
  }
  return { draw };
}

// ---------- Step 6: attention is the empirical denoiser ----------
function attentionScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.4 : 0.6)), ro = readout();
  let t = 0.6, probe = [0.5, 0.2], L = null;
  viz.append(c.cv, controls(slider('t', { min: 0.02, max: 0.98, step: 0.01, value: t }, (v) => { t = v; draw(); })), ro,
    note(`Exact: ${N6} training points from the 2D mixture. The weights are computed twice, once as softmax(⟨q, k<sub>i</sub>⟩/τ + b<sub>i</sub>) and once as the distance softmax of §2.7. Dots are the values y<sub>i</sub>, small rings the keys α<sub>t</sub>y<sub>i</sub>, and the halo area is proportional to the weight. Drag in the plot to move x<sub>t</sub>.`));
  dragProbe(c, () => L && { rect: L.plot, v: view(...L.plot) }, (p) => { probe = p; draw(); });
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, a = alpha(t), A = attention(probe, t);
    L = wide ? { plot: [0, 0, Math.round(w * 0.55), h], bars: [Math.round(w * 0.55) + 40, 40, w - Math.round(w * 0.55) - 52, h - 80] } : { plot: [0, 0, w, Math.round(h * 0.6)], bars: [36, Math.round(h * 0.6) + 40, w - 48, h - Math.round(h * 0.6) - 66] };
    const v = view(...L.plot);
    heat(ctx, L.plot, v, 0.35);
    clip(ctx, ...L.plot);
    for (let i = 0; i < N6; i++) {
      const [VX, VY] = v.px(TRAIN6[2 * i], TRAIN6[2 * i + 1]), [KX, KY] = v.px(a * TRAIN6[2 * i], a * TRAIN6[2 * i + 1]);
      if (A.wa[i] > 0.002) { ctx.globalAlpha = 0.3; dot(ctx, VX, VY, 3 + 24 * Math.sqrt(A.wa[i]), hlc('val', pal, pal.acc)); ctx.globalAlpha = 1; }
      ctx.strokeStyle = pal.line; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(KX, KY); ctx.lineTo(VX, VY); ctx.stroke();
      ring(ctx, KX, KY, isHL('key') ? 4 : 3, hlc('key', pal, pal.muted), 1.2);
      dot(ctx, VX, VY, isHL('val', 'y') ? 3.6 : 2.8, hlc(['val', 'y'], pal, pal.fg));
    }
    const [PX, PY] = v.px(...probe), [HX, HY] = v.px(...A.out);
    arrow(ctx, PX, PY, HX, HY, hlc('hx1', pal, pal.fg), 1.6, 9);
    ring(ctx, PX, PY, isHL('x', 'q') ? 8 : 6.5, hlc(['x', 'q'], pal, pal.acc), 2.2);
    star(ctx, HX, HY, hlc('hx1', pal, pal.fg));
    ctx.restore();
    frame(ctx, pal, ...L.plot);
    tag(ctx, pal, `t = ${f2(t)} · τ = σ_t² = ${(sigma(t) ** 2).toFixed(3)}`, L.plot[0] + 12, L.plot[1] + 22, hlc('tau', pal, pal.fg), 12.5);
    // Weights of the training points, both ways.
    const [bx, by, bw, bh] = L.bars, gw = bw / N6, mx = Math.max(...A.wa, 0.05);
    label(ctx, 'weights of the 24 training points', bx, by - 16, pal.fg, 12.5);
    for (let i = 0; i < N6; i++) {
      const X = bx + i * gw;
      ctx.fillStyle = hlc('key', pal, pal.acc); ctx.globalAlpha = 0.8; ctx.fillRect(X + 1.5, by + bh * (1 - A.wa[i] / mx), gw - 3, bh * A.wa[i] / mx); ctx.globalAlpha = 1;
      const Y = by + bh * (1 - A.wd[i] / mx); ctx.strokeStyle = pal.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X + 1, Y); ctx.lineTo(X + gw - 1, Y); ctx.stroke();
    }
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, by + bh + 0.5); ctx.lineTo(bx + bw, by + bh + 0.5); ctx.stroke();
    label(ctx, mx.toFixed(2), bx - 6, by + 5, pal.muted, 11, 'right'); label(ctx, '0', bx - 6, by + bh + 4, pal.muted, 11, 'right');
    label(ctx, 'bars: attention · lines: distance softmax', bx, by + bh + 20, pal.muted, 11.5);
    ro.innerHTML = `t = <b>${f2(t)}</b> · x<sub>t</sub> = (${f2(probe[0])}, ${f2(probe[1])}) · output Σ<sub>i</sub> w<sub>i</sub>v<sub>i</sub> = x̂<sub>1</sub> = (${f2(A.out[0])}, ${f2(A.out[1])})<br>` +
      `largest weight <b>${Math.max(...A.wa).toFixed(3)}</b> · largest difference between the two sets of weights <b>${A.diff.toExponential(1)}</b>`;
  }
  return { draw };
}

// ---------- Step 7: joint attention ----------
const NT_LIST = [77, 154, 256, 512], R_LIST = [256, 512, 1024, 2048], D_LIST = [768, 1152, 1536, 2048, 3072];
function mmditScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 7.4 : 0.66)), ro = readout();
  let arch = 'mmdit', nt = 154, R = 512, d = 1536;
  const keys = Object.keys(ARCHS), btns = keys.map((k) => button(ARCHS[k].name, () => { arch = k; setOn(btns, keys, k); draw(); }, k === arch ? 'on' : ''));
  viz.append(c.cv, controls(...btns), controls(listSlider('Nt', NT_LIST, nt, (v) => { nt = v; draw(); }), listSlider('Ni', R_LIST, R, (v) => { R = v; draw(); }, (v) => `${imgTokens(v)} (${v}²)`), listSlider('d', D_LIST, d, (v) => { d = v; draw(); })), ro,
    note('Exact counts. Image tokens are those of a square image of side R with f = 8 and p = 2. Parameters per layer count the query, key, value and output projections (4d² per weight set) and an MLP of width 4d (8d²); layer norms and the modulation of the time embedding are left out. The matrix is drawn to scale.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), wide = w >= WIDE, ni = imgTokens(R), A = ARCHS[arch], n = ni + nt;
    const M = wide ? Math.min(h - 66, w * 0.48) : Math.min(w - 60, h * 0.58), mx = wide ? 40 : (w - M) / 2 + 10, my = 34;
    const tw = Math.max(6, M * nt / n), ix = mx + tw, iw = M - tw;
    label(ctx, 'keys (read) →', mx + M, my - 10, pal.muted, 11.5, 'right');
    label(ctx, 'text', mx + tw / 2, my - 10, hlc('Nt', pal, pal.ramp[2]), 11.5, 'center');
    label(ctx, 'image', ix + iw / 2, my - 10, hlc('Ni', pal, pal.acc), 11.5, 'center');
    ctx.save(); ctx.translate(mx - 12, my + M / 2); ctx.rotate(-Math.PI / 2); label(ctx, 'queries (updated)', 0, 0, pal.muted, 11.5, 'center'); ctx.restore();
    // Blocks: [image reads image, image reads text, text reads image, text reads text].
    const cell = [[ix, my + tw, iw, iw], [mx, my + tw, tw, iw], [ix, my, iw, tw], [mx, my, tw, tw]];
    const rowCol = (j) => (j < 2 ? pal.acc : arch === 'single' ? pal.acc : pal.ramp[2]);
    cell.forEach(([x, y, cw, ch], j) => {
      if (A.blocks[j]) {
        ctx.fillStyle = arch === 'cross' && j === 1 ? pal.ramp[1] : rowCol(j); ctx.globalAlpha = (j === 1 || j === 2) ? 0.55 : 0.8; ctx.fillRect(x, y, cw, ch); ctx.globalAlpha = 1;
      } else {
        ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.strokeRect(x + 1, y + 1, cw - 2, ch - 2); ctx.setLineDash([]);
      }
    });
    frame(ctx, pal, mx, my, M, M);
    ctx.strokeStyle = pal.bg; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(ix, my); ctx.lineTo(ix, my + M); ctx.moveTo(mx, my + tw); ctx.lineTo(mx + M, my + tw); ctx.stroke();
    tag(ctx, pal, 'image reads image', ix + iw / 2, my + tw + iw / 2, pal.fg, 12, 'center');
    tag(ctx, pal, arch === 'cross' ? 'image reads text (cross-attention)' : 'text reads image', arch === 'cross' ? ix + iw / 2 : ix + iw / 2, arch === 'cross' ? my + tw + iw / 2 + 24 : my + tw / 2 + 5, arch === 'cross' ? pal.ramp[1] : pal.fg, 11.5, 'center');
    if (arch === 'cross') tag(ctx, pal, 'text rows: not computed', ix + iw / 2, my + tw / 2 + 5, pal.muted, 11.5, 'center');
    label(ctx, `${nt} text and ${ni} image tokens`, mx, my + M + 18, pal.muted, 11.5);
    // Comparison of the three designs.
    const bx = wide ? mx + M + 50 : 20, bw = wide ? w - bx - 20 : w - 40, by = wide ? 40 : my + M + 46, rowsH = wide ? (h - 70) / 2 : (h - by - 10) / 2;
    const metrics = [['attention pairs per layer', (k) => ARCHS[k].pairs(ni, nt), 'Ni'], ['parameters per layer', (k) => ARCHS[k].params(d), 'd']];
    metrics.forEach(([name, f, id], m) => {
      const y0 = by + m * rowsH, mxv = Math.max(...keys.map(f)), rh = Math.min(18, (rowsH - 30) / 3 - 4);
      label(ctx, name, bx, y0, isHL(id) ? pal.mark : pal.fg, 12.5);
      keys.forEach((k, q) => {
        const yy = y0 + 10 + q * (rh + 6), lw = wide ? 150 : 120, val = f(k);
        label(ctx, ARCHS[k].name.split(' (')[0], bx, yy + rh - 4, k === arch ? pal.fg : pal.muted, 11);
        ctx.fillStyle = pal.line; ctx.fillRect(bx + lw, yy, bw - lw - 60, rh);
        ctx.fillStyle = k === arch ? pal.acc : pal.muted; ctx.globalAlpha = k === arch ? 0.9 : 0.5; ctx.fillRect(bx + lw, yy, (bw - lw - 60) * val / mxv, rh); ctx.globalAlpha = 1;
        label(ctx, short(val), bx + bw, yy + rh - 4, k === arch ? pal.fg : pal.muted, 11, 'right');
      });
    });
    const pr = A.pairs(ni, nt), self = ni * ni;
    ro.innerHTML = `${A.name} · N<sub>txt</sub> = <b>${nt}</b> · N<sub>img</sub> = <b>${ni}</b> · d = <b>${d}</b><br>` +
      `attention pairs <b>${big(pr)}</b>, ${(pr / self).toFixed(2)} times those of self-attention over the image alone · parameters per layer <b>${big(A.params(d))}</b>`;
  }
  return { draw };
}

export const SCENES = { tokens: tokenScene, shift: shiftScene, latent: latentScene, repa: repaScene, pixels: pixelScene, attention: attentionScene, mmdit: mmditScene };
