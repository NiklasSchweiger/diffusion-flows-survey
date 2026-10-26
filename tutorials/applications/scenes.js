// Live scenes of the §7 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw() }. Colours of hovered symbols come from live.hl
// through hlc(). Steps 1, 2, 4 and 5 run on real data through real.js; steps 3 and 6 use sim.js.
import { palette, makeCanvas, label, hlc, controls, slider, button, toggle, readout, note, runner } from '../lib/tutorial.js';
import {
  loadFrames, loadStructures, VF, VSTEPS, VN, VSHOW, VMODES, videoModel,
  MW, MH, open, worldPlay, DEMO_KEYS,
  MS, MSIG, NQ, MSTEPS, NMOL, CPK, VDW, moleculeModel, pathAtK,
  proteinModel, v3, len3, mv, mm, mt, qmat, axisRot, rotAngle, slerp,
} from './real.js';
import { PK, PH, NP, OBS, DEMOS, policy, crps, crpsPlain, argminK, MS6, YMEM, YOBS, ensembleScore, gDens } from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), pct = (v) => ((v > 0.995 && v < 1) || (v > 0 && v < 0.005) ? `${(100 * v).toFixed(1)}%` : `${Math.round(100 * v)}%`);
const WIDE = 640;
function panels(w, h, k, gap = 26) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * 14) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + 14), w, ph]);
}
const aspect = (k) => (w) => (w >= WIDE ? (k === 3 ? 16 / 5.6 : 16 / 6.4) : 0.6 / (k === 3 ? 1.5 : 1));
const listSlider = (id, list, value, on, fmt = String, tex) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt, tex }, on);
function axes(ctx, pal, [x0, y0, w, h], { xmin, xmax, ymin, ymax, xt, yt, xl = '', yl = '', title, yfmt = String, xfmt = String, titleColor, right = 10 }) {
  const L = x0 + 44, R = x0 + w - right, T = y0 + 30, B = y0 + h - 30;
  const X = (v) => L + (R - L) * (v - xmin) / (xmax - xmin), Y = (v) => B - (B - T) * (v - ymin) / (ymax - ymin);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
  yt.forEach((q) => label(ctx, yfmt(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'));
  xt.forEach((q) => label(ctx, xfmt(q), X(q), B + 16, pal.muted, 11, 'center'));
  if (xl) label(ctx, xl, R, B + 28, pal.muted, 11.5, 'right');
  if (yl) label(ctx, yl, R - 6, T + 14, pal.muted, 11.5, 'right');
  if (title) label(ctx, title, L, T - 12, titleColor ?? pal.fg, 12.5);
  return { X, Y, L, R, T, B };
}
function polyline(ctx, xs, ys, color, lw = 2, dash = []) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x, ys[i]));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}
// bg: fill a box of that colour behind the entries, for legends that curves may cross.
function legend(ctx, ax, items, left = false, bottom = false, bg = null) {
  if (bg) {
    ctx.font = '11.5px sans-serif';
    const tw = Math.max(...items.map(([text]) => ctx.measureText(text.replace(/_/g, '')).width)) + 40;
    const y0 = bottom ? ax.B - 12 - 16 * (items.length - 1) - 10 : ax.T + 4, x0 = left ? ax.L + 2 : ax.R - 2 - tw;
    ctx.save(); ctx.globalAlpha = 0.9; ctx.fillStyle = bg; ctx.fillRect(x0, y0, tw, 16 * items.length + 4); ctx.restore();
  }
  items.forEach(([text, color, dash], i) => {
    const y = bottom ? ax.B - 12 - 16 * (items.length - 1 - i) : ax.T + 14 + 16 * i;
    if (left) { hdash(ctx, color, ax.L + 8, ax.L + 28, y, 2, dash ?? []); label(ctx, text, ax.L + 32, y + 4, color, 11.5); return; }
    const x = ax.R - 8;
    label(ctx, text, x - 24, y + 4, color, 11.5, 'right');
    hdash(ctx, color, x - 20, x, y, 2, dash ?? []);
  });
}
const niceMax = (v) => { const s = [0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30]; return s.find((q) => q >= v) ?? v; };
const memo = (f) => { const m = new Map(); return (k, ...a) => { if (!m.has(k)) m.set(k, f(...a)); return m.get(k); }; };
// Share of the values in each of nb bins on [lo, hi].
function hist(vals, lo, hi, nb) {
  const c = new Array(nb).fill(0);
  vals.forEach((v) => { const i = Math.floor((v - lo) / (hi - lo) * nb); if (i >= 0 && i < nb) c[i]++; });
  return c.map((n) => n / vals.length);
}
function bars(ctx, ax, h, lo, hi, color, alpha = 0.55) {
  const bw = (hi - lo) / h.length;
  ctx.fillStyle = color; ctx.globalAlpha = alpha;
  h.forEach((v, i) => { if (v > 0) { const x0 = ax.X(lo + i * bw), x1 = ax.X(lo + (i + 1) * bw); ctx.fillRect(x0 + 0.5, ax.Y(Math.min(v, ax.ymax ?? v)), x1 - x0 - 1, ax.B - ax.Y(Math.min(v, ax.ymax ?? v))); } });
  ctx.globalAlpha = 1;
}
// A group of buttons of which exactly one is on.
function choice(items, value, on) {
  const bs = items.map(([v, html]) => button(html, () => { set(v); on(v); }));
  const set = (v) => bs.forEach((b, i) => b.classList.toggle('on', items[i][0] === v));
  set(value);
  return bs;
}

// ---------- 3D drawing for steps 4 and 5 ----------
// Orthographic view: V rotates the coordinates, x goes right, y up, and larger z is nearer.
function dragRotate(cv, view, redraw) {
  cv.addEventListener('pointerdown', (e) => {
    let x = e.clientX, y = e.clientY;
    const mvh = (ev) => { const dx = ev.clientX - x, dy = ev.clientY - y; x = ev.clientX; y = ev.clientY; view.V = mm(mm(axisRot([0, 1, 0], dx * 0.01), axisRot([1, 0, 0], dy * 0.01)), view.V); redraw(); };
    const up = () => { removeEventListener('pointermove', mvh); removeEventListener('pointerup', up); };
    addEventListener('pointermove', mvh); addEventListener('pointerup', up);
  });
}
const hexRGB = (h) => { const s = h.trim().replace('#', ''), f = s.length === 3 ? [...s].map((c) => c + c).join('') : s; return [0, 2, 4].map((i) => parseInt(f.slice(i, i + 2), 16)); };
const mixRGB = (a, b, u) => `rgb(${a.map((v, i) => Math.round(v + u * (b[i] - v))).join(',')})`;
// Ball-and-stick model, depth sorted. With ghost, only outlines and dashed bonds, drawn on top.
function drawMol(ctx, pal, P, el, bonds, V, cx, cy, sc, ghost = null) {
  const q = P.map((p) => { const v = mv(V, p); return [cx + sc * v[0], cy - sc * v[1], v[2]]; });
  const rad = (i) => Math.max(1.6, 0.3 * VDW[el[i]] * sc);
  if (ghost) {
    ctx.strokeStyle = ghost; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
    bonds.forEach(([i, j]) => { ctx.beginPath(); ctx.moveTo(q[i][0], q[i][1]); ctx.lineTo(q[j][0], q[j][1]); ctx.stroke(); });
    ctx.setLineDash([]);
    q.forEach((p, i) => ring(ctx, p[0], p[1], rad(i) + 2, ghost, 1.2));
    return;
  }
  const items = [];
  bonds.forEach(([i, j, o]) => items.push([(q[i][2] + q[j][2]) / 2 - 0.01, () => {
    const dx = q[j][0] - q[i][0], dy = q[j][1] - q[i][1], L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L, lw = Math.max(1.4, 0.16 * sc);
    const offs = o >= 2 ? [-0.55 * lw, 0.55 * lw] : [0];
    ctx.lineCap = 'round';
    offs.forEach((d) => [[i, 0, 0.5], [j, 0.5, 1]].forEach(([a, u0, u1]) => {
      ctx.strokeStyle = CPK[el[a]] ?? '#999'; ctx.lineWidth = offs.length > 1 ? lw * 0.6 : lw;
      ctx.beginPath(); ctx.moveTo(q[i][0] + u0 * dx + d * nx, q[i][1] + u0 * dy + d * ny); ctx.lineTo(q[i][0] + u1 * dx + d * nx, q[i][1] + u1 * dy + d * ny); ctx.stroke();
    }));
    ctx.lineCap = 'butt';
  }]));
  q.forEach(([X, Y, z], i) => items.push([z, () => {
    const r = rad(i), c = hexRGB(CPK[el[i]] ?? '#999999'), g = ctx.createRadialGradient(X - 0.35 * r, Y - 0.4 * r, 0.1 * r, X, Y, r);
    g.addColorStop(0, mixRGB(c, [255, 255, 255], 0.55)); g.addColorStop(0.55, mixRGB(c, c, 0)); g.addColorStop(1, mixRGB(c, [0, 0, 0], 0.45));
    ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 0.8; ctx.stroke();
  }]));
  items.sort((a, b) => a[0] - b[0]).forEach(([, f]) => f());
}
// Waits for a promise of the data before the first real draw; draws a line of text meanwhile.
function whenLoaded(promise, c, onReady) {
  let state = 'loading';
  promise.then((d) => { state = 'ready'; onReady(d); }, (e) => { state = `Could not load the data (${e.message}).`; });
  return (pal) => { if (state === 'ready') return true; label(c.ctx, state === 'loading' ? 'Loading the data…' : state, 20, 30, pal.muted, 13); return false; };
}

// ---------- Step 1: video ----------
const VNAMES = { data: 'Training clips', joint: 'Joint model', independent: 'Per frame, own noise', shared: 'Per frame, shared noise' };
// Six clearly distinct hues, the same in both themes.
const clipCols = () => ['#22b8cf', '#7950f2', '#f2b01e', '#40c057', '#f0506e', '#a87b55'];
// Sepia tone curve for the grey frames, as an RGB lookup table: dark brown to warm paper.
const SEPIA = (() => {
  const st = [[0, [22, 15, 10]], [0.45, [122, 88, 58]], [0.8, [214, 186, 146]], [1, [252, 242, 222]]], lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const u = v / 255, i = st.findIndex(([p]) => p >= u), [p0, c0] = st[Math.max(0, i - 1)], [p1, c1] = st[Math.max(i, 1)], s = p1 > p0 ? (u - p0) / (p1 - p0) : 0;
    for (let k = 0; k < 3; k++) lut[3 * v + k] = c0[k] + s * (c1[k] - c0[k]);
  }
  return lut;
})();
function rrect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
function videoScene(viz) {
  const c = makeCanvas((w) => (w >= WIDE ? 16 / 6.6 : w / (w * 1.13 + 440)), { maxH: 0.78 }), ro = readout();
  let mode = 'joint', t = 1, ph = 0, M = null, fr = null, buf = null;
  const ready = whenLoaded(loadFrames(), c, (d) => { fr = d; M = videoModel(d); buf = new Uint8ClampedArray(d.D); draw(); });
  const r = runner({ step: () => { ph = (ph + 1) % VF; }, done: () => false, draw: () => draw(), interval: 140 });
  viz.append(c.cv, controls(...choice(VMODES.map((m) => [m, VNAMES[m]]), mode, (v) => { mode = v; draw(); }),
    slider('t', { min: 0, max: 1, step: 1 / VSTEPS, value: t }, (v) => { t = v; draw(); }), r.play), ro,
    note(`Exact and memorizing. The data are the 62 windows of ${VF} consecutive frames from six photographic sequences by Eadweard Muybridge (1878 to 1887, public domain), at 128 × 80 grey pixels, shown in sepia. Every model is the exact empirical denoiser of its training set, the softmax-weighted average of the training items, integrated with ${VSTEPS} steps. The joint model's items are whole windows; the per-frame models only know the 104 single frames, with independent noise per frame or one noise for all frames. Each tile names the training sequence its current frame converges to; the grid lists, for ${VN} generated clips, the training frame behind every generated frame.`));
  const cache = new Map();
  function frameImg(smp, key, k, f) {
    const id = `${key}|${k}|${f}`;
    if (cache.has(id)) return cache.get(id);
    if (cache.size > 1500) cache.clear();
    M.pixels(smp, k, f, buf);
    const cv = document.createElement('canvas'); cv.width = fr.W; cv.height = fr.H;
    const cx = cv.getContext('2d'), im = cx.createImageData(fr.W, fr.H);
    for (let i = 0; i < buf.length; i++) { const v = 3 * buf[i]; im.data[4 * i] = SEPIA[v]; im.data[4 * i + 1] = SEPIA[v + 1]; im.data[4 * i + 2] = SEPIA[v + 2]; im.data[4 * i + 3] = 255; }
    cx.putImageData(im, 0, 0); cache.set(id, cv);
    return cv;
  }
  // One playing clip: the current frame large, with a vignette, the source of that frame as a tag, and the filmstrip below.
  function tile(ctx, pal, cols, smp, key, k, src, x, y, tw) {
    const th = tw * fr.H / fr.W, mg = 2, mw = (tw - (VF - 1) * mg) / VF, mh = mw * fr.H / fr.W;
    ctx.save(); rrect(ctx, x, y, tw, th, 7); ctx.clip();
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(frameImg(smp, key, k, ph), x, y, tw, th);
    const g = ctx.createRadialGradient(x + tw / 2, y + th / 2, 0.3 * tw, x + tw / 2, y + th / 2, 0.75 * tw);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(10,6,2,0.45)');
    ctx.fillStyle = g; ctx.fillRect(x, y, tw, th);
    ctx.restore();
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; rrect(ctx, x + 0.5, y + 0.5, tw - 1, th - 1, 7); ctx.stroke();
    if (src) {
      const F = fr.frames[src[ph].k], name = `${fr.clips[F.clip].name} · ${F.idx + 1}`;
      ctx.font = '600 10.5px Inter, system-ui, sans-serif';
      const lw = ctx.measureText(name).width + 22;
      ctx.fillStyle = 'rgba(12,8,4,0.62)'; rrect(ctx, x + 6, y + 6, lw, 18, 9); ctx.fill();
      dot(ctx, x + 15, y + 15, 3.6, cols[F.clip]);
      label(ctx, name, x + 23, y + 19, '#f6ecd9', 10.5, 'left', 600);
    }
    const sy = y + th + 5;
    for (let f = 0; f < VF; f++) {
      const mx = x + f * (mw + mg);
      ctx.save(); rrect(ctx, mx, sy, mw, mh, 2); ctx.clip();
      ctx.globalAlpha = f === ph ? 1 : 0.7; ctx.drawImage(frameImg(smp, key, k, f), mx, sy, mw, mh); ctx.restore();
    }
    ctx.strokeStyle = pal.mark; ctx.lineWidth = 1.6; rrect(ctx, x + ph * (mw + mg) - 1, sy - 1, mw + 2, mh + 2, 3); ctx.stroke();
    return th + 5 + mh;
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    if (!ready(pal)) return;
    const R = M.run(mode), k = mode === 'data' ? 0 : Math.round(t * VSTEPS), cols = clipCols(pal), wide = w >= WIDE, fin = mode === 'data' || k === VSTEPS;
    // Tiles: VSHOW clips in a 3 x 2 grid (2 x 3 on narrow screens).
    const nc = wide ? 3 : 2, nr = VSHOW / nc, gap = 12, AW = wide ? 0.62 * w : w;
    const tileH = (tw) => tw * fr.H / fr.W + 5 + ((tw - (VF - 1) * 2) / VF) * fr.H / fr.W;
    let tw = (AW - (nc - 1) * gap) / nc;
    const availH = (wide ? h : h - 360) - 34 - 22;
    while (tw > 40 && nr * tileH(tw) + (nr - 1) * gap > availH) tw -= 2;
    label(ctx, mode === 'data' ? `${VSHOW} training clips` : `${VSHOW} generated clips at t = ${f2(t)}`, 2, 18, pal.fg, 12.5);
    for (let s = 0; s < VSHOW; s++) {
      const i = s % nc, j = Math.floor(s / nc);
      tile(ctx, pal, cols, R.S[s], `${mode}${s}`, k, fin ? R.src[s] : null, 2 + i * (tw + gap), 32 + j * (tileH(tw) + gap), tw);
    }
    const tb = 32 + nr * tileH(tw) + (nr - 1) * gap;
    label(ctx, fin ? 'tag: training sequence and frame of the playing frame' : 'tags appear at t = 1, when every frame has converged to a training frame', 2, tb + 16, pal.muted, 11);
    // Grid: source of every frame of VN generated clips.
    const legH = 3 * 16 + 6, gx = wide ? AW + 40 : 0, gy = wide ? 0 : tb + 34, gw = wide ? w - gx : w, gh = wide ? h : h - gy;
    const gl = gx + 26, gt = gy + 32, cw = Math.min(30, (gw - 30) / VF), rh = Math.min(20, (gh - 40 - legH) / VN);
    label(ctx, 'Training frame behind each generated frame', gx, gy + 18, pal.fg, 12.5);
    R.src.forEach((row, s) => {
      label(ctx, String(s + 1), gl - 8, gt + s * rh + rh / 2 + 4, s < VSHOW ? pal.fg : pal.muted, 10.5, 'right');
      row.forEach((q, f) => {
        const F = fr.frames[q.k], x = gl + f * cw, y = gt + s * rh;
        ctx.fillStyle = cols[F.clip]; ctx.globalAlpha = fin ? 0.85 : 0.25; rrect(ctx, x + 1.5, y + 1.5, cw - 3, rh - 3, 4); ctx.fill(); ctx.globalAlpha = 1;
        if (fin) label(ctx, String(F.idx + 1), x + cw / 2, y + rh / 2 + 4, '#fff', 10.5, 'center', 600);
      });
    });
    ctx.strokeStyle = pal.mark; ctx.lineWidth = 1.8; rrect(ctx, gl + ph * cw, gt - 1, cw, VN * rh + 2, 5); ctx.stroke();
    fr.clips.forEach((cl, j) => {
      const x = gx + (j % 2) * gw / 2, y = gt + VN * rh + 20 + Math.floor(j / 2) * 16;
      dot(ctx, x + 5, y - 4, 4.5, cols[j]); label(ctx, cl.name, x + 15, y, pal.muted, 11);
    });
    ro.innerHTML = `${VNAMES[mode]} · clips that are consecutive frames of one training clip: <b>${pct(R.coherent)}</b> · training clips mixed per generated clip: <b>${R.clips.toFixed(1)}</b> · clips that repeat one frame: <b>${pct(R.frozen)}</b> · every generated frame is a copy of a training frame (weight ≥ ${f3(R.minW)})`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 2: world model ----------
const EPS = [0.02, 0.05, 0.1, 0.2];
const KEYS = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down', ' ': null };
function worldScene(viz) {
  const narrowH = (w) => 3 * (w * MH / MW + 34) + 200;
  const c = makeCanvas((w) => (w >= WIDE ? 2.35 : w / narrowH(w)), { maxH: 0.8 }), ro = readout();
  let eps = 0.05, keys = [...DEMO_KEYS], focused = false;
  const cv = c.cv;
  cv.tabIndex = 0; cv.setAttribute('aria-label', 'Maze game: select it and press the arrow keys to move, space to wait');
  const press = (a) => { if (keys.length < 400) keys.push(a); draw(); };
  cv.addEventListener('keydown', (e) => { if (!(e.key in KEYS)) return; e.preventDefault(); press(KEYS[e.key]); });
  cv.addEventListener('focus', () => { focused = true; draw(); });
  cv.addEventListener('blur', () => { focused = false; draw(); });
  cv.addEventListener('pointerdown', () => cv.focus());
  viz.append(cv, controls(button('←', () => press('left')), button('↑', () => press('up')), button('↓', () => press('down')), button('→', () => press('right')), button('Wait', () => press(null)),
    button('Start over', () => { keys = []; draw(); }), button('Demo run', () => { keys = [...DEMO_KEYS]; draw(); }),
    listSlider('eps', EPS, eps, (v) => { eps = v; draw(); })), ro,
    note(`Exact. A frame of this game is fully determined by the player's cell, so a model's next frame is a distribution over cells, and its picture is the average of those frames: the player drawn in every cell with opacity equal to its probability. The true game moves the player one cell unless a wall is in the way. Both models move a share 1 − ε of the probability correctly, 0.6 ε one cell too far and 0.4 ε not at all; when no key is pressed, ε leaks to the open neighbours. Both condition on their own last frame. Model B first replaces it by the frame of its most likely cell, which changes nothing on a real frame, so both have the same one-step error ε. The error is 1 − (probability of the true cell).`));
  function maze(ctx, pal, x0, y0, cs, q, col, pos, title, trail) {
    label(ctx, title, x0, y0 - 8, pal.fg, 12.5);
    ctx.fillStyle = pal.bg; ctx.fillRect(x0, y0, cs * MW, cs * MH);
    for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) if (!open(y * MW + x)) { ctx.fillStyle = pal.line; ctx.fillRect(x0 + x * cs, y0 + y * cs, cs + 0.3, cs + 0.3); }
    if (trail) trail.forEach((p, i) => dot(ctx, x0 + (p % MW + 0.5) * cs, y0 + (Math.floor(p / MW) + 0.5) * cs, cs * 0.12, pal.faint));
    q.forEach((v, cell) => {
      if (v < 0.003) return;
      ctx.globalAlpha = v; ctx.fillStyle = col;
      const r = cs * 0.16, X = x0 + (cell % MW) * cs + r, Y = y0 + Math.floor(cell / MW) * cs + r, s = cs - 2 * r;
      ctx.beginPath(); ctx.roundRect(X, Y, s, s, cs * 0.12); ctx.fill();
    });
    ctx.globalAlpha = 1;
    if (q[pos] < 1) { ctx.strokeStyle = pal.fg; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.strokeRect(x0 + (pos % MW) * cs + 2, y0 + Math.floor(pos / MW) * cs + 2, cs - 4, cs - 4); ctx.setLineDash([]); }
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, cs * MW, cs * MH);
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), run = worldPlay(keys, eps), last = run[run.length - 1], n = keys.length;
    const ca = hlc('L', pal, pal.ramp[2]), cb = pal.ramp[0];
    let cs, spots, plot;
    if (w >= WIDE) {
      const gap = 26; cs = Math.min((w - 2 * gap) / 3 / MW, (h * 0.55 - 24) / MH);
      const mw = cs * MW, off = (w - 3 * mw - 2 * gap) / 2;
      spots = [0, 1, 2].map((i) => [off + i * (mw + gap), 24]);
      plot = [0, 24 + cs * MH + 10, w, h - (24 + cs * MH + 10)];
    } else {
      cs = w / MW; spots = [0, 1, 2].map((i) => [0, 24 + i * (cs * MH + 34)]);
      plot = [0, 3 * (cs * MH + 34), w, h - 3 * (cs * MH + 34)];
    }
    const delta = new Float64Array(MW * MH); delta[last.pos] = 1;
    const trail = run.slice(Math.max(0, run.length - 14), -1).map((s) => s.pos);
    maze(ctx, pal, ...spots[0], cs, delta, pal.fg, last.pos, `The game, frame ${n}`, trail);
    maze(ctx, pal, ...spots[1], cs, last.A, ca, last.pos, `Model A, from its own frames: error ${f2(last.errA)}`);
    maze(ctx, pal, ...spots[2], cs, last.B, cb, last.pos, `Model B, snaps to its likeliest cell: ${f2(last.errB)}`);
    if (!focused) { const X = spots[0][0] + cs * MW / 2, Y = spots[0][1] + cs * MH - cs / 2; ctx.globalAlpha = 0.85; ctx.fillStyle = pal.panel; ctx.fillRect(X - 130, Y - 13, 260, 22); ctx.globalAlpha = 1; label(ctx, 'Click here, then use the arrow keys', X, Y + 3, pal.mark, 12.5, 'center', 600); }
    const K = Math.max(20, n), ax = axes(ctx, pal, plot, { xmin: 0, xmax: K, ymin: 0, ymax: 1, xt: [0, Math.round(K / 2), K], yt: [0, 0.5, 1], xl: 'frame k', title: 'Error of each generated frame' });
    const ks = run.map((_, k) => k);
    clip(ctx, ax.L, ax.T - 1, ax.R - ax.L, ax.B - ax.T + 2);
    polyline(ctx, [ax.X(0), ax.X(Math.min(K, 1 / eps))], [ax.Y(0), ax.Y(Math.min(1, K * eps))], hlc('eps', pal, pal.muted), 1.4, [5, 4]);
    if (K * eps > 1) polyline(ctx, [ax.X(1 / eps), ax.X(K)], [ax.Y(1), ax.Y(1)], hlc('eps', pal, pal.muted), 1.4, [5, 4]);
    polyline(ctx, ks.map(ax.X), run.map((s) => ax.Y(s.errA)), ca, 2);
    polyline(ctx, ks.map(ax.X), run.map((s) => ax.Y(s.errB)), cb, 2);
    ctx.restore();
    legend(ctx, ax, [['bound k ε', hlc('eps', pal, pal.muted), [5, 4]], ['model A', ca], ['model B', cb]], true, false, pal.panel);
    ro.innerHTML = `ε = <b>${eps}</b> · frame <b>${n}</b> · error of model A <b>${f3(last.errA)}</b>, of model B <b>${f3(last.errB)}</b> · the bound k ε = ${f2(n * eps)}${n * eps >= 1 ? ' <b>is vacuous</b>' : ''}`;
  }
  return { draw };
}

// ---------- Step 3: robot policy ----------
const PNAMES = { demo: 'Demonstrations', reg: 'Regression policy', flow: 'Flow policy' };
function policyScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let kind = 'flow', H = 8;
  viz.append(c.cv, controls(...choice(Object.entries(PNAMES), kind, (v) => { kind = v; draw(); }), listSlider('H', PH, H, (v) => { H = v; draw(); })), ro,
    note(`Exact conditional of the demonstrations, Monte Carlo over ${NP} seeded rollouts per policy and chunk length. Each demonstration bends around the obstacle on the left or on the right with equal probability and has a smooth random deviation. The flow policy samples the next H waypoints from their exact conditional given the current position, executes all of them and replans. The regression policy outputs the conditional mean of the same chunk. A rollout fails when a waypoint lies inside the red box.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const ax = axes(ctx, pal, P1, { xmin: 0, xmax: 1, ymin: -0.6, ymax: 0.6, xt: [0, 0.5, 1], yt: [-0.5, 0, 0.5], xl: 'forward position s', title: kind === 'demo' ? '30 demonstrations' : `30 rollouts, chunk H = ${H}` });
    ctx.fillStyle = pal.bad; ctx.globalAlpha = 0.25;
    ctx.fillRect(ax.X(OBS.s - OBS.hs), ax.Y(OBS.hy), ax.X(OBS.s + OBS.hs) - ax.X(OBS.s - OBS.hs), ax.Y(-OBS.hy) - ax.Y(OBS.hy));
    ctx.globalAlpha = 1;
    const paths = kind === 'demo' ? DEMOS : policy(kind, H).show, col = kind === 'reg' ? pal.ramp[3] : kind === 'flow' ? hlc('A', pal, pal.ramp[0]) : pal.muted;
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    paths.forEach((ys) => {
      const hit = ys.some((v, i) => Math.abs(i / PK - OBS.s) <= OBS.hs && Math.abs(v) < OBS.hy);
      polyline(ctx, ys.map((_, i) => ax.X(i / PK)), ys.map(ax.Y), hit ? pal.bad : col, 1.3);
      if (kind !== 'demo' && H < PK) for (let i = 0; i < ys.length; i += H) dot(ctx, ax.X(i / PK), ax.Y(ys[i]), 1.8, hit ? pal.bad : col);
    });
    ctx.restore();
    const lg = (v) => Math.log2(v);
    const ax2 = axes(ctx, pal, P2, { xmin: -0.3, xmax: lg(40) + 0.3, ymin: 0, ymax: 1, xt: PH.map(lg), xfmt: (q) => String(Math.round(2 ** q)), yt: [0, 0.5, 1], yfmt: (q) => pct(q), xl: 'chunk length H (log scale)', title: 'Share of rollouts that pass' });
    vdash(ctx, hlc('H', pal, pal.line), ax2.X(lg(H)), ax2.T, ax2.B);
    [['reg', pal.ramp[3]], ['flow', hlc('A', pal, pal.ramp[0])]].forEach(([k, cl]) => {
      const sv = PH.map((q) => policy(k, q).success);
      polyline(ctx, PH.map((q) => ax2.X(lg(q))), sv.map(ax2.Y), cl, 2);
      PH.forEach((q, i) => dot(ctx, ax2.X(lg(q)), ax2.Y(sv[i]), q === H ? 4.5 : 2.8, cl));
    });
    legend(ctx, ax2, [['flow policy', hlc('A', pal, pal.ramp[0])], ['regression', pal.ramp[3]]], false, true);
    const R = policy('reg', H), F = policy('flow', H);
    ro.innerHTML = `H = <b>${H}</b> · passes: regression <b>${pct(R.success)}</b>, flow <b>${pct(F.success)}</b> · mode switches of the flow policy near the obstacle: <b>${f2(F.switches)}</b> per rollout`;
  }
  return { draw };
}

// ---------- Step 4: molecules and symmetry ----------
const deg = (r) => r * 180 / Math.PI;
const G_AXIS = [1, 2, 0.5];
// Plays t from 0 to 1 on its slider.
function tPlayer(get, set, sl) {
  const r = runner({ step: () => { set(Math.min(1, get() + 0.01)); sl.set(get()); }, done: () => get() >= 1, draw: () => {}, interval: 30 });
  r.playing = false;
  r.start = button('Play from t = 0', () => { set(0); sl.set(0); r.playing = true; r.kick(); });
  return r;
}
function moleculeScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout(), view = { V: mm(axisRot([1, 0, 0], 0.5), axisRot([0, 1, 0], 0.4)) };
  let th = 2, t = 1, one = false, mi = 0, models = null;
  const pick = controls();
  const ready = whenLoaded(loadStructures(), c, (d) => { models = d.molecules.map((m) => ({ name: m.name, cid: m.cid, M: moleculeModel(m) })); pick.append(...choice(models.map((m, i) => [i, m.name]), mi, (v) => { mi = v; draw(); })); draw(); });
  const tSl = slider('t', { min: 0, max: 1, step: 0.01, value: t }, (v) => { t = v; draw(); }), r = tPlayer(() => t, (v) => { t = v; }, tSl);
  viz.append(c.cv, pick, controls(slider('th', { min: 0, max: 6.28, step: 0.02, value: th, fmt: (v) => `${Math.round(deg(v))}°` }, (v) => { th = v; draw(); }), tSl, r.start,
    toggle('Train on one orientation', one, (v) => { one = v; draw(); })), ro,
    note(`Exact. The conformers are PubChem's 3D structures (public domain), hydrogens included, centred at the origin. The data are one conformer m at a uniformly random rotation R with Gaussian noise of ${MS} Å per coordinate, and the noise x<sub>0</sub> is a centred Gaussian with ${MSIG} Å per coordinate. Given x<sub>t</sub>, the posterior over R is proportional to exp((t / v) tr(Rᵀ C)) with C = Σ<sub>i</sub> x<sub>t,i</sub> m<sub>i</sub>ᵀ; the field averages it over ${NQ} seeded rotations around its mode and is integrated with ${MSTEPS} steps. The toggle replaces the data by the single orientation R = I. Drag the molecule to turn the view.`));
  dragRotate(c.cv, view, () => draw());
  const flows = memo((M, one, th) => M.flow(M.noise(3).map((p) => mv(axisRot(G_AXIS, th), p)), one));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    if (!ready(pal)) return;
    const [P1, P2] = panels(w, h, 2), { M, name } = models[mi], key = `${mi}${one}`;
    const g = axisRot(G_AXIS, th), A = pathAtK(flows(`${key}|${th}`, M, one, th), t), B = pathAtK(flows(`${key}|0`, M, one, 0), t).map((p) => mv(g, p));
    const [x0, y0, pw, ph] = P1, sc = Math.min(pw, ph - 50) / 11, cPhi = hlc('Phi', pal, pal.fg), cG = hlc('g', pal, pal.mark);
    label(ctx, `${name}: two paths at t = ${f2(t)}`, x0 + 6, y0 + 18, pal.fg, 12.5);
    drawMol(ctx, pal, A, M.el, M.bonds, view.V, x0 + pw / 2, y0 + 24 + (ph - 50) / 2, sc);
    drawMol(ctx, pal, B, M.el, M.bonds, view.V, x0 + pw / 2, y0 + 24 + (ph - 50) / 2, sc, cG);
    label(ctx, 'solid: flow of the rotated noise', x0 + 6, y0 + ph - 22, cPhi, 11.5);
    label(ctx, 'dashed rings: rotated flow of the noise', x0 + 6, y0 + ph - 6, cG, 11.5);
    // Right: samples at t = 1 in the same view.
    const ends = M.ends(one), angs = ends.map((x) => deg(rotAngle(M.orient(x)))), [u0, v0, uw, uh] = P2;
    const nc = 4, nr = 2, cw = uw / nc, chh = (uh - 30) / nr, rmax = Math.max(...M.m.map(len3)) + 1, s2 = Math.min(cw, chh - 16) / (2 * rmax);
    label(ctx, `8 of ${NMOL} samples at t = 1, in the same view`, u0 + 6, v0 + 18, pal.fg, 12.5);
    for (let i = 0; i < nc * nr; i++) {
      const X = u0 + (i % nc + 0.5) * cw, Y = v0 + 26 + Math.floor(i / nc) * chh + (chh - 16) / 2;
      drawMol(ctx, pal, ends[i], M.el, M.bonds, view.V, X, Y, s2);
      label(ctx, `${Math.round(angs[i])}° from m`, X, Y + (chh - 16) / 2 + 12, pal.muted, 11, 'center');
    }
    let gap = 0; A.forEach((p, i) => { gap = Math.max(gap, len3(v3.sub(p, B[i]))); });
    const mean = angs.reduce((a, b) => a + b, 0) / angs.length;
    ro.innerHTML = `rotation g by <b>${Math.round(deg(th))}°</b> · largest distance between the two paths' atoms: <b>${gap < 1e-3 ? gap.toExponential(1) : f3(gap)} Å</b> · mean rotation of the ${NMOL} samples from the conformer: <b>${Math.round(mean)}°</b> (uniform rotations: 126°)`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: protein frames ----------
const SSW = { H: 1.2, E: 1.4, C: 0.3 };
const unit = (a) => v3.scl(a, 1 / len3(a));
const cr = (a, b, c, d, u) => b.map((_, k) => 0.5 * (2 * b[k] + (c[k] - a[k]) * u + (2 * a[k] - 5 * b[k] + 4 * c[k] - d[k]) * u * u + (3 * b[k] - a[k] - 3 * c[k] + d[k]) * u * u * u));
// Cartoon of the Cα trace: a Catmull-Rom ribbon whose width follows the secondary structure of the
// target and lies across the local curvature, shaded by how much each piece faces the viewer.
function ribbon(ctx, pal, CA, ss, V, cx, cy, sc) {
  const n = CA.length, p = CA.map((x) => mv(V, x));
  const side = p.map((x, i) => { const a = p[Math.max(0, i - 1)], b = p[Math.min(n - 1, i + 1)], s = v3.cross(v3.sub(b, a), v3.sub(v3.scl(v3.add(a, b), 0.5), x)); return len3(s) > 1e-6 ? unit(s) : null; });
  for (let i = 1; i < n; i++) side[i] ??= side[i - 1];
  for (let i = n - 2; i >= 0; i--) side[i] ??= side[i + 1];
  for (let i = 1; i < n; i++) if (side[i] && side[i - 1] && v3.dot(side[i], side[i - 1]) < 0) side[i] = v3.scl(side[i], -1);
  const base = { H: hexRGB(pal.ramp[1]), E: hexRGB(pal.ramp[3]), C: hexRGB(pal.ramp[0]) }, S = 8, pts = [];
  for (let i = 0; i < n - 1; i++) for (let k = 0; k < S + (i === n - 2); k++) {
    const u = k / S, s = unit(v3.add(v3.scl(side[i] ?? [0, 1, 0], 1 - u), v3.scl(side[i + 1] ?? [0, 1, 0], u)));
    pts.push({ x: cr(p[Math.max(0, i - 1)], p[i], p[i + 1], p[Math.min(n - 1, i + 2)], u), s, hw: (1 - u) * SSW[ss[i]] + u * SSW[ss[i + 1]], c: ss[u < 0.5 ? i : i + 1] });
  }
  const quads = [];
  for (let j = 0; j + 1 < pts.length; j++) {
    const a = pts[j], b = pts[j + 1], nz = Math.abs(unit(v3.cross(v3.sub(b.x, a.x), a.s))[2] || 0);
    const cs = [v3.add(a.x, v3.scl(a.s, a.hw)), v3.add(b.x, v3.scl(b.s, b.hw)), v3.sub(b.x, v3.scl(b.s, b.hw)), v3.sub(a.x, v3.scl(a.s, a.hw))];
    quads.push([(a.x[2] + b.x[2]) / 2, cs, mixRGB(base[a.c], [0, 0, 0], 0.55 * (1 - nz))]);
  }
  quads.sort((a, b) => a[0] - b[0]).forEach(([, cs, col]) => {
    ctx.beginPath(); cs.forEach((q, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, cx + sc * q[0], cy - sc * q[1])); ctx.closePath();
    ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke();
  });
}
function proteinScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout(), view = { V: axisRot([1, 0, 0], -0.3) };
  let t = 1, linear = false, pi = 0, models = null;
  const pick = controls();
  const ready = whenLoaded(loadStructures(), c, (d) => { models = d.proteins.map((p) => ({ id: p.id, name: p.name, M: proteinModel(p) })); pick.append(...choice(models.map((m, i) => [i, `${m.name} (${m.id})`]), pi, (v) => { pi = v; draw(); })); draw(); });
  const tSl = slider('t', { min: 0, max: 1, step: 0.01, value: t }, (v) => { t = v; draw(); }), r = tPlayer(() => t, (v) => { t = v; }, tSl);
  viz.append(c.cv, pick, controls(tSl, r.start, toggle('Move the atoms on straight lines', linear, (v) => { linear = v; draw(); })), ro,
    note(`Exact paths. The targets are the backbones (N, Cα and C atoms) of three PDB entries (CC0), with the secondary structure from the files. The noise draws the Cα positions from a centred Gaussian with the protein's radius of gyration and the rotations uniformly from SO(3). Each residue's N and C atoms sit at fixed coordinates in its frame. On the frame path the Cα positions move on straight lines and the rotations on SO(3) geodesics; the toggle moves all three atoms of every residue on straight lines instead. The ribbon is drawn through the Cα positions at time t. Drag it to turn the view.`));
  dragRotate(c.cv, view, () => draw());
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    if (!ready(pal)) return;
    const [P1, P2] = panels(w, h, 2), { M, id, name } = models[pi], st = M.at(t, linear), V = view.V;
    const [x0, y0, pw, ph] = P1, cx = x0 + pw / 2, cy = y0 + 22 + (ph - 22) / 2, sc = Math.min(pw, ph - 30) / (3.4 * M.rg);
    label(ctx, `${name} (${id}) at t = ${f2(t)}`, x0 + 6, y0 + 18, pal.fg, 12.5);
    const P = (q) => { const v = mv(V, q); return [cx + sc * v[0], cy - sc * v[1]]; };
    const tr = M.CA1.map(P);
    polyline(ctx, tr.map((q) => q[0]), tr.map((q) => q[1]), pal.faint, 5);
    ribbon(ctx, pal, st.atoms.map((r) => r[1]), M.ss, V, cx, cy, sc);
    const errs = M.bondErrs(st.atoms), cR = hlc('R', pal, pal.fg);
    st.atoms.forEach(([N, CA, C], i) => {
      const [a, b, d] = [N, CA, C].map(P), col = errs[i] > 0.02 ? pal.bad : cR;
      ctx.globalAlpha = errs[i] > 0.02 ? 0.9 : 0.35; ctx.strokeStyle = col; ctx.lineWidth = errs[i] > 0.02 ? 1.6 : 1.1;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(d[0], d[1]); ctx.stroke(); ctx.globalAlpha = 1;
      if (i === M.show) ring(ctx, b[0], b[1], 8, pal.mark, 1.6);
    });
    label(ctx, 'helix', x0 + 6, y0 + ph - 6, pal.ramp[1], 11.5); label(ctx, 'strand', x0 + 46, y0 + ph - 6, pal.ramp[3], 11.5); label(ctx, 'coil', x0 + 96, y0 + ph - 6, pal.ramp[0], 11.5);
    label(ctx, 'faint: target Cα trace', x0 + pw - 6, y0 + ph - 6, pal.muted, 11.5, 'right');
    // One residue's rotation: geodesic, naive matrix average, target. Rows of mt(R) are the frame axes.
    const i = M.show, [u0, v0, uw, uh] = P2, ux = u0 + uw / 2, uy = v0 + 16 + (uh - 16) / 2, us = Math.min(uw, uh - 40) * 0.36;
    label(ctx, `Rotation of residue ${i + 1} (ringed), θ = ${Math.round(deg(M.ang[i]))}°`, u0 + 6, v0 + 18, pal.fg, 12.5);
    const nv = M.naive(i, t), Rt = qmat(slerp(M.Q0[i], M.Q1[i], t));
    const Q = (e) => { const v = mv(V, e); return [ux + us * v[0], uy - us * v[1]]; };
    const axc = [pal.ramp[0], pal.ramp[1], pal.ramp[2]];
    ring(ctx, ux, uy, us, pal.faint, 1);
    mt(M.R1[i]).forEach((e, j) => { const q = Q(e); ctx.strokeStyle = axc[j]; ctx.globalAlpha = 0.35; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.globalAlpha = 1; });
    mt(nv.M).forEach((e, j) => { const q = Q(e); ctx.strokeStyle = axc[j]; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.setLineDash([]); });
    mt(Rt).forEach((e, j) => { const q = Q(e), col = hlc('R', pal, axc[j]); ctx.strokeStyle = col; ctx.lineWidth = 2.6; ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); dot(ctx, q[0], q[1], 3.5, col); });
    legend(ctx, { L: u0 + 4, R: u0 + uw, T: v0 + 30, B: v0 + uh - 4 }, [['geodesic R_t', pal.fg], ['(1 − t) R_0 + t R_1', pal.fg, [4, 3]], ['target R_1', pal.muted]], true, true);
    const e = Math.max(...errs), th = M.ang[i];
    ro.innerHTML = `t = <b>${f2(t)}</b> · largest bond-length error: <b>${e < 1e-6 ? '0' : pct(e)}</b> ${linear ? '(straight atoms)' : '(frames)'} · Cα RMSD to the target <b>${f2(M.rmsd(st.atoms))} Å</b><br>residue ${i + 1}: angle left to turn (1 − t)θ = <b>${Math.round(deg((1 - t) * th))}°</b>, ‖Ω‖<sub>F</sub> = √2 θ = <b>${f2(Math.SQRT2 * th)}</b> · naive average: det <b>${f2(nv.det)}</b>, ‖MᵀM − I‖<sub>F</sub> <b>${f2(nv.orth)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 6: weather and the CRPS ----------
const KMAX = 2;
function crpsScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let kap = 0.7, M = 5, fair = false;
  viz.append(c.cv, controls(slider('kap', { min: 0, max: KMAX, step: 0.02, value: kap }, (v) => { kap = v; draw(); }), listSlider('M', MS6, M, (v) => { M = v; draw(); }),
    toggle('Fair estimator', fair, (v) => { fair = v; draw(); })), ro,
    note('Exact. The true distribution of tomorrow\'s value is the two-component mixture in grey, with mean 0. The forecast F<sub>κ</sub> scales it about its mean by κ, so κ = 0 is a single deterministic value and κ = 1 is the truth. The expected scores on the right are sums of E|N(μ, v)| over pairs of mixture components. The ensemble below the left plot is one seeded draw of M members, and the dashed line is one observation.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const xs = Array.from({ length: 241 }, (_, i) => -4 + i / 30), gd = xs.map((x) => gDens(x)), fd = xs.map((x) => gDens(x, kap));
    const ymax = niceMax(1.2 * Math.max(...gd, ...fd.map((v) => Math.min(v, 1.2))));
    const ax = axes(ctx, pal, P1, { xmin: -4, xmax: 4, ymin: 0, ymax, xt: [-4, -2, 0, 2, 4], yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(2) : '0'), xl: 'value', title: `Forecast with spread κ = ${f2(kap)}` });
    clip(ctx, ax.L, ax.T - 1, ax.R - ax.L, ax.B - ax.T + 2);
    ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ax.X(x), ax.Y(gd[i]))); ctx.lineTo(ax.X(4), ax.B); ctx.lineTo(ax.X(-4), ax.B); ctx.closePath();
    ctx.fillStyle = pal.muted; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
    polyline(ctx, xs.map(ax.X), gd.map(ax.Y), pal.muted, 1.4);
    if (kap > 0.02) polyline(ctx, xs.map(ax.X), fd.map(ax.Y), hlc('F', pal, pal.ramp[1]), 2);
    else vdash(ctx, hlc('F', pal, pal.ramp[1]), ax.X(0), ax.T, ax.B, 2.5, []);
    ctx.restore();
    const ens = YMEM.slice(0, M).map((v) => kap * v), cE = hlc('M', pal, pal.ramp[1]);
    ens.forEach((v) => { ctx.strokeStyle = cE; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(ax.X(v), ax.B - 12); ctx.lineTo(ax.X(v), ax.B); ctx.stroke(); });
    vdash(ctx, hlc('y', pal, pal.fg), ax.X(YOBS), ax.T, ax.B, 1.4);
    legend(ctx, ax, [['truth G', pal.muted], ['forecast F_κ', hlc('F', pal, pal.ramp[1])], ['observation y', hlc('y', pal, pal.fg), [3, 4]]], true);
    const ks = Array.from({ length: 101 }, (_, i) => KMAX * i / 100), ex = ks.map(crps), pl = ks.map((k) => crpsPlain(k, M));
    const lo = Math.floor(10 * Math.min(...pl, ...ex)) / 10, hi = Math.ceil(10 * Math.max(...ex, ...pl)) / 10;
    const ax2 = axes(ctx, pal, P2, { xmin: 0, xmax: KMAX, ymin: lo, ymax: hi, xt: [0, 0.5, 1, 1.5, 2], yt: [lo, (lo + hi) / 2, hi], yfmt: f2, xl: 'spread κ', title: 'Expected score against the spread' });
    vdash(ctx, hlc('kap', pal, pal.line), ax2.X(kap), ax2.T, ax2.B);
    const cFair = fair ? pal.ramp[0] : pal.muted, cPlain = fair ? pal.muted : pal.ramp[2];
    polyline(ctx, ks.map(ax2.X), ex.map(ax2.Y), cFair, fair ? 2.4 : 1.4);
    polyline(ctx, ks.map(ax2.X), pl.map(ax2.Y), cPlain, fair ? 1.4 : 2.4, [6, 4]);
    const kp = argminK((k) => crpsPlain(k, M));
    dot(ctx, ax2.X(1), ax2.Y(crps(1)), 4.5, cFair); dot(ctx, ax2.X(kp), ax2.Y(crpsPlain(kp, M)), 4.5, cPlain);
    legend(ctx, { ...ax2, R: ax2.X(1.45) }, [['CRPS, and the fair estimator', cFair], [`plain estimator, M = ${M}`, cPlain, [6, 4]]], false, false, pal.panel);
    ro.innerHTML = `κ = <b>${f2(kap)}</b>, M = <b>${M}</b> · expected CRPS <b>${f3(crps(kap))}</b> (minimum ${f3(crps(1))} at κ = 1) · expected plain estimate <b>${f3(crpsPlain(kap, M))}</b>, minimized at κ = <b>${f2(kp)}</b><br>this ensemble against this observation: ${fair ? 'fair' : 'plain'} score <b>${f3(ensembleScore(ens, YOBS, fair))}</b> · error of the forecast mean (y − 0)² = ${f3(YOBS * YOBS)}, the same for every κ`;
  }
  return { draw };
}

export const SCENES = { video: videoScene, world: worldScene, policy: policyScene, molecule: moleculeScene, protein: proteinScene, crps: crpsScene };
