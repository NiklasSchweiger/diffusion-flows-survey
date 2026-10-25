// Live scenes of the §7 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw() }. Colours of hovered symbols come from live.hl
// through hlc().
import { palette, makeCanvas, label, hlc, controls, slider, button, toggle, readout, note } from '../lib/tutorial.js';
import {
  VF, VA, NV, NVS, VMODES, video, pathAt, SLOPE_SD,
  WP, WK, WPOS, worldRollout,
  PK, PH, NP, OBS, DEMOS, policy,
  NROT, MS, NM, MSTEPS, MZ0, molPath, molEnds, rot2, fromSub, equivGap,
  NR, CA1, Q0, Q1, backbone, bondErr, rmsd, ANG, RSHOW, slerp, qmat, naiveRot,
  G, crps, crpsPlain, argminK, MS6, YMEM, YOBS, ensembleScore, gDens,
} from './sim.js';

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

// ---------- Step 1: video ----------
const VNAMES = { data: 'Data', joint: 'Joint model', independent: 'Per frame, own noise', shared: 'Per frame, shared noise' };
function videoScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let mode = 'joint', t = 1;
  viz.append(c.cv, controls(...choice(VMODES.map((m) => [m, VNAMES[m]]), mode, (v) => { mode = v; draw(); }),
    slider('t', { min: 0, max: 1, step: 0.02, value: t }, (v) => { t = v; draw(); })), ro,
    note(`Exact. A clip is ${VF} frames of a dot on a line, x<sup>f</sup> = c + s·${VA}·τ<sub>f</sub> + noise, with a random offset c, a random direction s = ±1 and τ<sub>f</sub> from −0.5 to 0.5. The joint model integrates the exact velocity of this two-component Gaussian in ${VF} dimensions with RK4 (50 steps). The per-frame models use the exact one-dimensional flow map of each frame's marginal, which is all a model of single frames can learn. The histogram is over ${NV} clips.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), r = video(mode);
    const tt = mode === 'data' ? 1 : t;
    const ax = axes(ctx, pal, P1, { xmin: -0.5, xmax: VF - 0.5, ymin: -3, ymax: 3, xt: [0, 3, 6, 9, 11], yt: [-3, 0, 3], xl: 'frame f', title: `${NVS} clips${mode === 'data' ? ' from the data' : ` at t = ${f2(tt)}`}` });
    clip(ctx, ax.L, ax.T, ax.R - ax.L, ax.B - ax.T);
    r.show.forEach((p, i) => {
      const x = pathAt(p, tt), col = pal.ramp[i % 4];
      polyline(ctx, x.map((_, f) => ax.X(f)), x.map(ax.Y), col, 1.6);
      x.forEach((v, f) => dot(ctx, ax.X(f), ax.Y(v), 2.6, col));
    });
    ctx.restore();
    const lo = -3.5, hi = 3.5, nb = 40, hs = hist(r.slopes, lo, hi, nb), ymax = niceMax(1.15 * Math.max(...hs));
    const ax2 = axes(ctx, pal, P2, { xmin: lo, xmax: hi, ymin: 0, ymax, xt: [-3, -1.5, 0, 1.5, 3], yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(2) : '0'), xl: 'fitted speed', title: `Speed of ${NV} generated clips${mode === 'data' ? '' : ' at t = 1'}` });
    ax2.ymax = ymax;
    [-VA, VA].forEach((v) => vdash(ctx, pal.muted, ax2.X(v), ax2.T, ax2.B, 1.2));
    bars(ctx, ax2, hs, lo, hi, mode === 'data' ? pal.muted : pal.ramp[1]);
    legend(ctx, ax2, [['speeds in the data', pal.muted, [3, 4]]], true);
    const d = video('data');
    ro.innerHTML = `${VNAMES[mode]} · jitter <b>${f3(r.jitter)}</b> (data ${f3(d.jitter)}) · clips with a speed within 0.5 of ±${VA}: <b>${pct(r.coherent)}</b> · mean |speed| <b>${f2(r.slopes.reduce((a, s) => a + Math.abs(s), 0) / r.slopes.length)}</b>`;
  }
  return { draw };
}

// ---------- Step 2: world model ----------
const EPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3], KS = [8, 16, 32, 64];
const wMemo = memo((eps, sharpen) => worldRollout(eps, sharpen));
function worldScene(viz) {
  const c = makeCanvas(aspect(3), { maxH: 0.72 }), ro = readout();
  let eps = 0.05, K = 64;
  viz.append(c.cv, controls(listSlider('eps', EPS, eps, (v) => { eps = v; draw(); }), listSlider('K', KS, K, (v) => { K = v; draw(); })), ro,
    note(`Exact. The world is a ring of ${WP} cells with one bright cell, and the ${WK} key presses (left, none, right) are fixed. Both models move a share 1 − ε of the intensity one cell in the direction of the key, 0.6 ε one cell too far and 0.4 ε one cell too short. Model B first replaces its context by the one-hot frame at its brightest cell, which changes nothing on a real frame, so the two have the same one-step error ε on real contexts. The error is the total variation 1 − (intensity on the true cell).`));
  function heat(ctx, pal, rect, r, title, col) {
    const [x0, y0, pw, ph] = rect, L = x0 + 44, R = x0 + pw - 10, T = y0 + 30, B = y0 + ph - 30, cw = (R - L) / WP, rh = (B - T) / (K + 1);
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
    label(ctx, title, L, T - 12, pal.fg, 12.5);
    for (let k = 0; k <= K; k++) r.frames[k].forEach((v, i) => { if (v > 0.004) { ctx.globalAlpha = Math.min(1, Math.sqrt(v)); ctx.fillStyle = col; ctx.fillRect(L + i * cw, T + k * rh, cw + 0.3, rh + 0.3); } });
    ctx.globalAlpha = 1;
    for (let k = 0; k <= K; k++) dot(ctx, L + (WPOS[k] + 0.5) * cw, T + (k + 0.5) * rh, Math.min(2, rh / 3), pal.fg);
    [0, K].forEach((k) => label(ctx, `k = ${k}`, L - 6, T + (k + 0.5) * rh + 4, pal.muted, 11, 'right'));
    label(ctx, 'cell', R, B + 16, pal.muted, 11.5, 'right');
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2, P3] = panels(w, h, 3), A = wMemo(`${eps}a`, eps, false), Bm = wMemo(`${eps}b`, eps, true);
    const ca = hlc('L', pal, pal.ramp[2]), cb = pal.ramp[0];
    heat(ctx, pal, P1, A, 'Model A, its own frames', ca);
    heat(ctx, pal, P2, Bm, 'Model B, sharpens its context', cb);
    const ax = axes(ctx, pal, P3, { xmin: 0, xmax: K, ymin: 0, ymax: 1, xt: [0, K / 2, K], yt: [0, 0.5, 1], xl: 'frame k', title: 'Error of the rollout' });
    const ks = Array.from({ length: K + 1 }, (_, k) => k);
    clip(ctx, ax.L, ax.T - 1, ax.R - ax.L, ax.B - ax.T + 2);
    polyline(ctx, ks.map(ax.X), ks.map((k) => ax.Y(Math.min(1, k * eps))), hlc('eps', pal, pal.muted), 1.4, [5, 4]);
    polyline(ctx, ks.map(ax.X), ks.map((k) => ax.Y(A.err[k])), ca, 2);
    polyline(ctx, ks.map(ax.X), ks.map((k) => ax.Y(Bm.err[k])), cb, 2);
    ctx.restore();
    legend(ctx, { ...ax, B: ax.Y(0.3) }, [['bound k ε', hlc('eps', pal, pal.muted), [5, 4]], ['model A', ca], ['model B', cb]], false, true, pal.panel);
    const kv = Math.ceil(1 / eps);
    ro.innerHTML = `ε = <b>${eps}</b> · error at frame ${K}: model A <b>${f3(A.err[K])}</b>, model B <b>${f3(Bm.err[K])}</b> · the bound Σε<sub>k</sub> = ${f2(K * eps)} ${K * eps >= 1 ? `<b>is vacuous</b> from frame ${kv} on` : 'is not yet vacuous'}`;
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
const molRot = memo((which, th) => molPath(which, rot2(MZ0, th)));
const deg = (r) => r * 180 / Math.PI;
function moleculeScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let th = 2, t = 1, one = false;
  viz.append(c.cv, controls(slider('th', { min: 0, max: 6.28, step: 0.02, value: th, fmt: (v) => `${Math.round(deg(v))}°` }, (v) => { th = v; draw(); }),
    slider('t', { min: 0, max: 1, step: 0.02, value: t }, (v) => { t = v; draw(); }),
    toggle('Train on one orientation', one, (v) => { one = v; draw(); })), ro,
    note(`Exact. The molecule is water, three atoms in the plane with the centre fixed at the origin, so it has four coordinates. The data are the reference geometry at ${NROT} orientations, 1° apart, each with Gaussian noise of standard deviation ${MS} Å, and the noise is N(0, I) in the same four coordinates. The field is the exact velocity of this mixture, integrated with RK4 (${MSTEPS} steps). The toggle replaces the data by the single reference orientation. The histogram is over ${NM} noises.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2), which = one ? 'one' : 'all';
    const lim = 2.4, [x0, y0, pw, ph] = P1, s = Math.min(pw - 54, ph - 44), L = x0 + 44 + (pw - 54 - s) / 2, T = y0 + 30;
    const X = (v) => L + s * (v + lim) / (2 * lim), Y = (v) => T + s * (lim - v) / (2 * lim);
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(L + 0.5, T + 0.5, s, s);
    label(ctx, `Two paths at t = ${f2(t)}`, L, T - 12, pal.fg, 12.5);
    [-2, 0, 2].forEach((q) => { label(ctx, String(q), X(q), T + s + 14, pal.muted, 11, 'center'); label(ctx, String(q), L - 6, Y(q) + 4, pal.muted, 11, 'right'); });
    label(ctx, 'Å', L + s, T + s + 26, pal.muted, 11.5, 'right');
    const base = molPath(which, MZ0), cA = pathAt(molRot(`${which}${th}`, which, th), t), cB = rot2(pathAt(base, t), th);
    const A = fromSub(cA), B = fromSub(cB), cPhi = hlc('Phi', pal, pal.ramp[2]), cG = hlc('g', pal, pal.ramp[0]);
    const mol = (P, col, filled) => {
      [1, 2].forEach((i) => { ctx.strokeStyle = col; ctx.lineWidth = filled ? 3 : 1.4; ctx.setLineDash(filled ? [] : [4, 3]); ctx.beginPath(); ctx.moveTo(X(P[0][0]), Y(P[0][1])); ctx.lineTo(X(P[i][0]), Y(P[i][1])); ctx.stroke(); ctx.setLineDash([]); });
      P.forEach((p, i) => (filled ? dot : ring)(ctx, X(p[0]), Y(p[1]), i ? 6 : 9, col));
    };
    mol(A, cPhi, true);
    mol(B, cG, false);
    label(ctx, 'O', X(A[0][0]) + 12, Y(A[0][1]) - 8, cPhi, 11.5);
    legend(ctx, { L, R: L + s, T, B: T + s }, [['flow of the rotated noise', cPhi], ['rotated flow of the noise', cG, [4, 3]]], true, true);
    const ends = molEnds(which).map(deg), lo = -180, hi = 180, nb = 36, hs = hist(ends, lo, hi, nb), ymax = niceMax(1.2 * Math.max(...hs, 1 / nb));
    const ax2 = axes(ctx, pal, P2, { xmin: lo, xmax: hi, ymin: 0, ymax, xt: [-180, -90, 0, 90, 180], xfmt: (q) => `${q}°`, yt: [0, ymax / 2, ymax], yfmt: (q) => (q ? q.toFixed(2) : '0'), xl: 'orientation of the generated molecule', title: `Orientations of ${NM} samples` });
    ax2.ymax = ymax;
    bars(ctx, ax2, hs, lo, hi, pal.ramp[1]);
    hdash(ctx, pal.muted, ax2.L, ax2.R, ax2.Y(1 / nb), 1.2, [5, 4]);
    legend(ctx, ax2, [['uniform', pal.muted, [5, 4]]], true);
    const gap = equivGap(cA, pathAt(base, t), th), sd = Math.sqrt(ends.reduce((a, v) => a + v * v, 0) / ends.length);
    ro.innerHTML = `rotation <b>${Math.round(deg(th))}°</b> · largest distance between the two paths' atoms: <b>${gap < 1e-3 ? gap.toExponential(1) : f3(gap)} Å</b> · spread of the orientations <b>${Math.round(sd)}°</b> (uniform ${Math.round(180 / Math.sqrt(3))}°)`;
  }
  return { draw };
}

// ---------- Step 5: protein frames ----------
function proteinScene(viz) {
  const c = makeCanvas(aspect(2), { maxH: 0.72 }), ro = readout();
  let t = 0.5, az = 0.6, linear = false;
  viz.append(c.cv, controls(slider('t', { min: 0, max: 1, step: 0.02, value: t }, (v) => { t = v; draw(); }),
    slider('az', { min: 0, max: 6.28, step: 0.02, value: az, fmt: (v) => `${Math.round(deg(v))}°` }, (v) => { az = v; draw(); }),
    toggle('Move the atoms on straight lines', linear, (v) => { linear = v; draw(); })), ro,
    note(`Exact paths. The target is an ideal α-helix of ${NR} residues (radius 2.3 Å, rise 1.5 Å, 100° per residue). The noise draws the Cα positions from a centred Gaussian with a standard deviation of 4 Å and the rotations uniformly from SO(3). Each residue's N, Cα and C atoms sit at fixed coordinates in its frame. On the frame path the Cα positions move on straight lines and the rotations on SO(3) geodesics; the toggle moves the three atoms of every residue on straight lines instead.`));
  // View: rotate about the helix axis z by az, then draw z to the right and the rotated y upward.
  const view = (p) => { const C = Math.cos(az), S = Math.sin(az); return [p[2], S * p[0] + C * p[1], C * p[0] - S * p[1]]; };
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), [P1, P2] = panels(w, h, 2);
    const [x0, y0, pw, ph] = P1, cx = x0 + pw / 2, cy = y0 + 16 + (ph - 16) / 2, sc = Math.min(pw / 20, (ph - 40) / 18);
    label(ctx, `Backbone at t = ${f2(t)}`, x0 + 10, y0 + 18, pal.fg, 12.5);
    const P = (p) => { const v = view(p); return [cx + sc * v[0], cy - sc * v[1], v[2]]; };
    // Target trace, faint.
    polyline(ctx, CA1.map((p) => P(p)[0]), CA1.map((p) => P(p)[1]), pal.faint, 6);
    const bb = backbone(t, linear), cR = hlc('R', pal, pal.ramp[1]), cX = hlc('xi', pal, pal.ramp[0]);
    polyline(ctx, bb.map((r) => P(r[1])[0]), bb.map((r) => P(r[1])[1]), pal.muted, 1, [3, 4]);
    const order = bb.map((r, i) => [P(r[1])[2], i]).sort((a, b) => a[0] - b[0]);
    order.forEach(([, i]) => {
      const [n, ca, cc] = bb[i].map(P), err = bondErr([bb[i]]), col = err > 0.02 ? pal.bad : cR;
      [n, cc].forEach((q) => { ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(ca[0], ca[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); });
      dot(ctx, n[0], n[1], 3, col); dot(ctx, cc[0], cc[1], 3, col); dot(ctx, ca[0], ca[1], 4.2, cX);
      if (i === RSHOW) ring(ctx, ca[0], ca[1], 8, pal.fg, 1.2);
    });
    // One residue's rotation: geodesic, naive matrix average, target.
    const [u0, v0, uw, uh] = P2, ux = u0 + uw / 2, uy = v0 + 16 + (uh - 16) / 2, us = Math.min(uw, uh - 40) * 0.36;
    label(ctx, `Rotation of residue ${RSHOW + 1} (ringed), θ = ${Math.round(deg(ANG[RSHOW]))}°`, u0 + 10, v0 + 18, pal.fg, 12.5);
    const nv = naiveRot(RSHOW, t), R1 = qmat(Q1[RSHOW]), Rt = qmat(slerp(Q0[RSHOW], Q1[RSHOW], t));
    const Q = (p) => { const v = view(p); return [ux + us * v[0], uy - us * v[1]]; };
    const axc = [pal.ramp[0], pal.ramp[1], pal.ramp[2]];
    ring(ctx, ux, uy, us, pal.faint, 1);
    R1.forEach((e, j) => { const q = Q(e); ctx.strokeStyle = axc[j]; ctx.globalAlpha = 0.35; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.globalAlpha = 1; });
    nv.M.forEach((e, j) => { const q = Q(e); ctx.strokeStyle = axc[j]; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.setLineDash([]); });
    Rt.forEach((e, j) => { const q = Q(e); ctx.strokeStyle = hlc('R', pal, axc[j]); ctx.lineWidth = 2.6; ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(q[0], q[1]); ctx.stroke(); dot(ctx, q[0], q[1], 3.5, hlc('R', pal, axc[j])); });
    const lb = { L: u0 + 4, R: u0 + uw, T: v0 + 30, B: v0 + uh - 4 };
    legend(ctx, lb, [['geodesic R_t', pal.fg], ['(1 − t) R_0 + t R_1', pal.fg, [4, 3]], ['target R_1', pal.muted]], true, true);
    const e = bondErr(bb), th = ANG[RSHOW];
    ro.innerHTML = `t = <b>${f2(t)}</b> · largest bond-length error: <b>${e < 1e-6 ? '0' : pct(e)}</b> ${linear ? '(straight atoms)' : '(frames)'} · Cα RMSD to the target <b>${f2(rmsd(bb))} Å</b><br>residue ${RSHOW + 1}: angle left to turn (1 − t)θ = <b>${Math.round(deg((1 - t) * th))}°</b>, ‖Ω‖<sub>F</sub> = √2 θ = <b>${f2(Math.SQRT2 * th)}</b> · naive average: det <b>${f2(nv.det)}</b>, ‖MᵀM − I‖<sub>F</sub> <b>${f2(nv.orth)}</b>`;
  }
  return { draw };
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
