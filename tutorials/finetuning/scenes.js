// Live scenes of the §3 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw() }. Colours of hovered symbols come from live.hl
// through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, button, toggle, readout, note } from '../lib/tutorial.js';
import { M, D, Z0, RING, SSD, MIX, W0, comb, ip, run, ref, sw, share, lora, dreambooth, STYLES, merge } from './sim.js';

// ---------- Helpers ----------
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.4, dash = []) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]); };
function cross(ctx, X, Y, r, color, lw = 1.6) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X - r, Y + r); ctx.lineTo(X + r, Y - r); ctx.stroke(); }
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
function vdash(ctx, color, x, y0, y1, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y, lw = 1, dash = [3, 4]) { ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => v.toFixed(2), f3 = (v) => v.toFixed(3), pct = (q) => `${Math.round(100 * q)}%`;
// World box [-bx, bx] x [-by, by] fitted into a rectangle of the canvas.
function view(x0, y0, w, h, bx = 2.5, by = 2.3) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y] };
}
// Panels side by side on wide canvases and stacked on narrow ones.
const WIDE = 640;
function panels(w, h, k, gap = 18) {
  if (w >= WIDE) { const pw = (w - (k - 1) * gap) / k; return Array.from({ length: k }, (_, i) => [i * (pw + gap), 0, pw, h]); }
  const ph = (h - (k - 1) * 14) / k; return Array.from({ length: k }, (_, i) => [0, i * (ph + 14), w, ph]);
}
const aspect = (k) => (w) => (w >= WIDE ? (k === 3 ? 16 / 5.8 : 16 / 7) : 0.62 / (k === 3 ? 1.5 : 1));
const listSlider = (id, list, value, on, fmt = String, tex) => slider(id, { min: 0, max: list.length - 1, step: 1, value: list.indexOf(value), map: (v) => list[v], fmt, tex }, on);
const memo = (f) => { const m = new Map(); return (key, ...a) => { if (!m.has(key)) m.set(key, f(...a)); return m.get(key); }; };
function axes(ctx, pal, [x0, y0, w, h], { xmin, xmax, ymin, ymax, xt, yt, xl = '', title, yfmt = String, xfmt = String }) {
  const L = x0 + 44, R = x0 + w - 10, T = y0 + 30, B = y0 + h - 30;
  const X = (v) => L + (R - L) * (v - xmin) / (xmax - xmin), Y = (v) => B - (B - T) * (v - ymin) / (ymax - ymin);
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
  yt.forEach((q) => label(ctx, yfmt(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'));
  xt.forEach((q) => label(ctx, xfmt(q), X(q), B + 16, pal.muted, 11, 'center'));
  if (xl) label(ctx, xl, R, B + 28, pal.muted, 11.5, 'right');
  if (title) label(ctx, title, L, T - 12, pal.fg, 12.5);
  return { X, Y, L, R, T, B };
}
function polyline(ctx, xs, ys, color, lw = 2, dash = []) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x, ys[i]));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
}
function legend(ctx, ax, items) {
  items.forEach(([text, color, dash], i) => {
    const y = ax.T + 14 + 16 * i, x = ax.R - 8;
    label(ctx, text, x - 24, y + 4, color, 11.5, 'right');
    hdash(ctx, color, x - 20, x, y, 2, dash ?? []);
  });
}
// A panel of samples: a frame, a title, the modes of a mixture as dashed circles, and the points.
function cloud(ctx, pal, rect, title, xs, color, { modes = [], modeColor = pal.muted, base = true, marks = [] } = {}) {
  const [x0, y0, w, h] = rect, T = 26, v = view(x0, y0 + T, w, h - T);
  frame(ctx, pal, x0, y0 + T, w, h - T);
  label(ctx, title, x0 + 2, y0 + 16, pal.fg, 12.5);
  if (base) { ctx.globalAlpha = 0.35; RING.forEach(([a, b]) => ring(ctx, ...v.px(a, b), 0.32 * v.s, pal.muted, 1.2)); ctx.globalAlpha = 1; }
  modes.forEach(([a, b, r]) => ring(ctx, ...v.px(a, b), (r ?? 0.32) * v.s, modeColor, 1.4, [4, 4]));
  clip(ctx, x0, y0 + T, w, h - T);
  ctx.fillStyle = color; ctx.globalAlpha = live_alpha(pal);
  for (let i = 0; i < xs.length / 2; i++) { const [X, Y] = v.px(xs[2 * i], xs[2 * i + 1]); ctx.fillRect(X - 1.1, Y - 1.1, 2.2, 2.2); }
  ctx.globalAlpha = 1;
  ctx.restore();
  marks.forEach(([a, b, col]) => cross(ctx, ...v.px(a, b), 5, col));
  return v;
}
const live_alpha = (pal) => (pal.name === 'dark' ? 0.55 : 0.5);
const modesOf = (name, r) => MIX[name].mu.map(([a, b]) => [a, b, r]);
const SUBJ = [0, 0], OLD = RING[0];

// ---------- Step 1: LoRA ----------
const RANKS = [1, 2, 3, 4, 8, M];
const samples1 = memo((r, a) => run(comb(W0(), [a, lora('new').rank(r)])));
function loraScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let r = 1, a = 1;
  viz.append(c.cv, controls(
    listSlider('r', RANKS, r, (v) => { r = v; draw(); }, (v) => (v === M ? '32 (full)' : String(v))),
    slider('alpha', { min: 0, max: 1.25, step: 0.05, value: a }, (v) => { a = v; draw(); })), ro,
    note('Exact up to the model: every target is the velocity of a Gaussian mixture, computed in closed form, and the rank-r update is the best one in function space, which is where training of the low-rank factors converges. Samples are 2000 seeded noise draws pushed through 16 Euler steps.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), L = lora('new');
    const xs = samples1(`${r},${a}`, r, a), col = hlc(['r', 'alpha', 'dW'], pal, pal.ramp[0]);
    cloud(ctx, pal, P1, `Samples of W₀ + α ΔW at rank ${r === M ? '32 (full)' : r}`, xs, col, { modes: modesOf('new'), modeColor: hlc('u', pal, pal.muted), marks: [[...OLD, hlc('W0', pal, pal.bad)]] });
    const J = 8, ax = axes(ctx, pal, P2, { xmin: 0.4, xmax: J + 0.6, ymin: 0, ymax: 1.3, xt: Array.from({ length: J }, (_, i) => i + 1), yt: [0, 0.5, 1], yfmt: (q) => pct(q), xl: 'directions kept', title: 'Share of the update captured by the top directions' });
    const bw = (ax.X(1) - ax.X(0)) * 0.34;
    for (let j = 1; j <= J; j++) {
      const on = j <= r, X = ax.X(j);
      ctx.globalAlpha = on ? 0.95 : 0.3;
      ctx.fillStyle = hlc('mu', pal, pal.ramp[0]); ctx.fillRect(X - bw - 1, ax.Y(L.fnCum[j - 1]), bw, ax.Y(0) - ax.Y(L.fnCum[j - 1]));
      ctx.fillStyle = pal.ramp[3]; ctx.fillRect(X + 1, ax.Y(L.parCum[j - 1]), bw, ax.Y(0) - ax.Y(L.parCum[j - 1]));
      ctx.globalAlpha = 1;
    }
    legend(ctx, ax, [['in function space, Σ μ_{j}', hlc('mu', pal, pal.ramp[0])], ['in parameter space', pal.ramp[3]]]);
    const n = r >= M ? M * D : r * (M + D);
    ro.innerHTML = `r = <b>${r}</b>, α = <b>${f2(a)}</b> · distance to the target (sliced W₁) = <b>${f3(sw(xs, ref('new')))}</b> · mass at the new centre mode = <b>${pct(share(xs))}</b>, at the replaced mode = <b>${pct(share(xs, ...OLD))}</b> · trainable numbers = <b>${n}</b> of ${M * D}`;
  }
  return { draw };
}

// ---------- Step 2: DreamBooth and prior preservation ----------
const LAMS = [0, 0.03, 0.1, 0.3, 1, 3, 10], KAPS = [0.3, 0.5, 1], KS = [300, 1000, 3000, 10000, 30000];
const ZS = Z0.subarray(0, 1000);
const db = memo((kap, lam, k) => dreambooth(kap, lam, k));
const dbRun = memo((kap, lam, k) => { const m = db(`${kap},${lam},${k}`, kap, lam, k); return { cls: run(m.cls), subj: run(m.subj) }; });
const dbCurve = memo((kap, k) => LAMS.map((lam) => { const m = db(`${kap},${lam},${k}`, kap, lam, k), c = run(m.cls, ZS), s = run(m.subj, ZS); return { cls: sw(c, ref('base')), subj: sw(s, ref('subj')), mid: share(c, 0, 0, 0.6) }; }));
function dreamboothScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.7 }), ro = readout();
  let lam = 0, kap = 0.5, k = 3000;
  viz.append(c.cv, controls(
    listSlider('lam', LAMS, lam, (v) => { lam = v; draw(); }),
    listSlider('kap', KAPS, kap, (v) => { kap = v; draw(); }),
    listSlider('k', KS, k, (v) => { k = v; draw(); })), ro,
    note('Exact up to the model: k full-batch gradient steps on the DreamBooth loss have a closed form in the eigenbasis of G, so no training loop runs in the page. The curves on the right use the first 1000 of the 2000 noise draws.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3);
    const R = dbRun(`${kap},${lam},${k}`, kap, lam, k), col = hlc(['lam', 'kap', 'k'], pal, pal.ramp[0]);
    cloud(ctx, pal, P1, 'Class prompt c_{cls}', R.cls, hlc('cc', pal, col), { modes: modesOf('base'), modeColor: pal.muted, base: false });
    cloud(ctx, pal, P2, 'Subject prompt c_{subj}', R.subj, hlc('cs', pal, pal.ramp[2]), { modes: [[...SUBJ, 2 * SSD]], modeColor: pal.muted });
    const C = dbCurve(`${kap},${k}`, kap, k), ymax = 0.8;
    const ax = axes(ctx, pal, P3, { xmin: -0.3, xmax: LAMS.length - 0.7, ymin: 0, ymax, xt: LAMS.map((_, i) => i), xfmt: (i) => String(LAMS[i]), yt: [0, 0.2, 0.4, 0.6, 0.8], yfmt: (q) => q.toFixed(1), xl: 'λ', title: 'Distance to the target, by λ' });
    const Xs = LAMS.map((_, i) => ax.X(i));
    polyline(ctx, Xs, C.map((q) => ax.Y(Math.min(q.cls, ymax))), hlc('cc', pal, pal.ramp[0]), 2);
    polyline(ctx, Xs, C.map((q) => ax.Y(Math.min(q.subj, ymax))), hlc('cs', pal, pal.ramp[2]), 2);
    const i = LAMS.indexOf(lam);
    vdash(ctx, hlc('lam', pal, pal.mark), Xs[i], ax.T, ax.B, 1.3);
    dot(ctx, Xs[i], ax.Y(Math.min(C[i].cls, ymax)), 4.5, pal.ramp[0]); dot(ctx, Xs[i], ax.Y(Math.min(C[i].subj, ymax)), 4.5, pal.ramp[2]);
    legend(ctx, ax, [['class prompt to the ring', hlc('cc', pal, pal.ramp[0])], ['subject prompt to the subject', hlc('cs', pal, pal.ramp[2])]]);
    ro.innerHTML = `λ = <b>${lam}</b>, κ = <b>${kap}</b>, k = <b>${k}</b> · class prompt: distance to the ring = <b>${f3(sw(R.cls, ref('base')))}</b>, mass near the centre = <b>${pct(share(R.cls, 0, 0, 0.6))}</b> · subject prompt: distance to the subject = <b>${f3(sw(R.subj, ref('subj')))}</b> · share of the update the class receives at λ = 0: 1/(1 + κ²) = <b>${f2(1 / (1 + kap * kap))}</b>`;
  }
  return { draw };
}

// ---------- Step 3: merging adapters ----------
const runW = memo((W) => run(W));
function mergeScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.7 }), ro = readout();
  let key = 'rot', a = 1, orth = false, joint = false;
  const styleBtn = button('', () => { key = key === 'rot' ? 'shr' : 'rot'; label_(); draw(); });
  const label_ = () => { styleBtn.innerHTML = `Style: ${STYLES[key].name}`; };
  label_();
  viz.append(c.cv, controls(styleBtn,
    slider('alpha', { min: 0, max: 1.25, step: 0.05, value: a }, (v) => { a = v; draw(); }),
    toggle('Orthogonalize', orth, (v) => { orth = v; draw(); }),
    toggle('Joint fit', joint, (v) => { joint = v; draw(); })), ro,
    note('Exact up to the model: both adapters and the joint fit are least-squares fits to exact mixture velocities on the same design set. Orthogonalize removes from the style adapter its component along the subject adapter, in the inner product of G.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3), st = STYLES[key], m = merge(key), W = W0();
    const B = orth ? comb(m.B, [-ip(m.A, m.B) / ip(m.A, m.A), m.A]) : m.B;
    const xa = runW(`A${a}`, comb(W, [a, m.A])), xb = runW(`B${key}${a}`, comb(W, [a, m.B]));
    const xm = joint ? runW(`J${key}`, m.J) : runW(`M${key}${a}${orth}`, comb(W, [a, m.A], [a, B]));
    const stray = MIX[st.style].mu[0];
    cloud(ctx, pal, P1, 'W₀ + α ΔW_{subj}', xa, hlc('dS', pal, pal.ramp[2]), { modes: modesOf('new'), modeColor: pal.muted });
    cloud(ctx, pal, P2, 'W₀ + α ΔW_{style}', xb, hlc('dT', pal, pal.ramp[0]), { modes: modesOf(st.style), modeColor: pal.muted });
    cloud(ctx, pal, P3, joint ? 'Joint fit W_{J}' : `W₀ + α ΔW_{subj} + α ΔW_{style}${orth ? ' (orthogonalized)' : ''}`, xm, hlc(joint ? 'WJ' : ['dS', 'dT', 'alpha'], pal, pal.ramp[3]),
      { modes: modesOf(st.joint), modeColor: hlc('WJ', pal, pal.muted), marks: [[...stray, pal.bad]] });
    const tgt = ref(st.joint);
    ro.innerHTML = `${st.name}, α = <b>${f2(a)}</b> · cosine of the two adapters in function space = <b>${f2(m.cos)}</b> · non-additive part of the joint change = <b>${pct(m.nonadd)}</b> · right plot: distance to the joint target = <b>${f3(sw(xm, tgt))}</b>, mass at the centre = <b>${pct(share(xm))}</b>, at the replaced mode × = <b>${pct(share(xm, ...stray, 0.4))}</b> (joint fit: ${f3(sw(runW(`J${key}`, m.J), tgt))})`;
  }
  return { draw };
}

export const SCENES = { lora: loraScene, dreambooth: dreamboothScene, merge: mergeScene };
