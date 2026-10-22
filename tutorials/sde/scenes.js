// Live scenes of the §2.3 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, arrow, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import {
  MIX1, MIX2, W2, p2, LIN, SCHEDULES, R_MAX, R_MIN, T_MIN, dens1, pt1, dtau1, fpTerms,
  NT, TAU_MAX, TAUS, FWD, PF, NR, TR, reverse1, T0, sample2, quality2, LL_EXACT, LOGR, EDM, tOfRatio,
} from './sim.js';

// ---------- Helpers ----------
const LO = -4, HI = 4, XS = Array.from({ length: 321 }, (_, i) => LO + (HI - LO) * i / 320);
const DATA_MEAN1 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * m, 0);
const DATA_VAR1 = MIX1.mu.reduce((a, m, k) => a + MIX1.w[k] * (MIX1.sd[k] ** 2 + m * m), 0) - DATA_MEAN1 ** 2;
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
// Mode a quantile path ends in, from the cumulative mixture weights.
const modeOfQ = (q) => (q < MIX1.w[0] ? 0 : q < MIX1.w[0] + MIX1.w[1] ? 1 : 2);
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_/g, '')).width + 12, x0 = align === 'right' ? x - w + 6 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const hline = (ctx, pal, x0, x1, y) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };
function vdash(ctx, color, x, y0, y1) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function curve(ctx, xs, X, Y, f, color, lw) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(f(x, i))));
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke();
}
function shade(ctx, xs, X, Y, f, base, color, alpha) {
  ctx.beginPath(); ctx.moveTo(X(xs[0]), base); xs.forEach((x, i) => ctx.lineTo(X(x), Y(f(x, i)))); ctx.lineTo(X(xs[xs.length - 1]), base); ctx.closePath();
  ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.fill(); ctx.globalAlpha = 1;
}
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const nearest = (arr, v) => { let j = 0; for (let i = 1; i < arr.length; i++) if (Math.abs(arr[i] - v) < Math.abs(arr[j] - v)) j = i; return j; };
const f2 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));

// Density strip behind a path plot: column i is time times[i], rows run from HI (top) to LO. Each column
// is normalized by its own maximum, so the shading shows the shape of the density at every time.
const stripCache = new Map();
function strip(key, times, dens) {
  const k = `${key}|${live.theme}`;
  if (stripCache.has(k)) return stripCache.get(k);
  const W = times.length, H = 161, off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H);
  const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.55 : 0.34;
  for (let i = 0; i < W; i++) {
    const col = Array.from({ length: H }, (_, j) => dens(HI - (HI - LO) * j / (H - 1), times[i])), m = Math.max(...col);
    for (let j = 0; j < H; j++) {
      const o = 4 * (j * W + i);
      im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(col[j] / m, 0.7));
    }
  }
  g.putImageData(im, 0, 0);
  stripCache.set(k, off);
  return off;
}
// Layout of a path plot with a histogram inset on the right.
function pframe(w, h) {
  const L = 46, R = Math.round(w * 0.74), top = 28, bot = h - 36, iL = R + 34, iR = w - 16;
  return { L, R, top, bot, iL, iR, Y: (y) => top + (bot - top) * (HI - y) / (HI - LO) };
}
function yAxis(ctx, pal, fr, name, id) {
  [-3, 0, 3].forEach((v) => label(ctx, String(v), fr.L - 8, fr.Y(v) + 4, pal.muted, 11, 'right'));
  label(ctx, name, fr.L - 8, fr.top - 10, hlc(id, pal, pal.muted), 13, 'right');
}
// Histogram of vals against the exact density, drawn sideways in the inset. ticks are optional extra marks.
function inset(ctx, pal, fr, vals, dens, { color, hl, ticks } = {}) {
  const NB = 48, bw = (HI - LO) / NB, cnt = new Float64Array(NB);
  for (const v of vals) { const b = Math.floor((v - LO) / bw); if (b >= 0 && b < NB) cnt[b]++; }
  const ex = XS.map(dens), hist = Array.from(cnt, (c) => c / (vals.length * bw));
  const pm = Math.max(...ex, ...hist) * 1.08, X = (p) => fr.iL + (fr.iR - fr.iL) * p / pm;
  ctx.fillStyle = color ?? pal.fg; ctx.globalAlpha = 0.3;
  hist.forEach((q, b) => { const y0 = fr.Y(LO + (b + 1) * bw), y1 = fr.Y(LO + b * bw); ctx.fillRect(fr.iL, y0 + 0.5, X(q) - fr.iL, y1 - y0 - 1); });
  ctx.globalAlpha = 1;
  ctx.beginPath(); XS.forEach((y, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(ex[i]), fr.Y(y)));
  ctx.strokeStyle = hl ?? pal.fg; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(fr.iL - 0.5, fr.top); ctx.lineTo(fr.iL - 0.5, fr.bot); ctx.stroke();
  (ticks ?? []).forEach(([y, c]) => { ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(fr.iL - 10, fr.Y(y)); ctx.lineTo(fr.iL - 2, fr.Y(y)); ctx.stroke(); });
}
const meanVar = (vals) => { let m = 0; for (const v of vals) m += v / vals.length; let s = 0; for (const v of vals) s += (v - m) ** 2 / vals.length; return [m, s]; };
const exactMV = (t) => [LIN.alpha(t) * DATA_MEAN1, LIN.alpha(t) ** 2 * DATA_VAR1 + LIN.sigma(t) ** 2];
const SHOW = Array.from({ length: 40 }, (_, k) => 12 + 25 * k);   // the 40 of 1000 paths that are drawn
const column = (rows, j) => rows.map((r) => r[j]);

// ---------- Steps 1 and 4: forward paths in tau ----------
function forwardPlot(viz, withPF) {
  const c = makeCanvas(16 / 9), ro = readout();
  let tau = withPF ? 0.6 : 0.5;
  viz.append(c.cv, controls(slider('tau', { min: 0, max: TAU_MAX, step: 0.01, value: tau }, (v) => { tau = v; draw(); })), ro,
    note(withPF
      ? 'Grey: 40 of the 1000 SDE paths of step 1. Coloured: 49 probability-flow paths, started at the 2%, 4%, …, 98% quantiles of the data and integrated with the exact score; the colour marks the mode each one starts in. The ticks beside the histogram are where the coloured paths are at the chosen τ.'
      : 'The paths use the exact Gaussian transition of the SDE between grid points, so they carry no discretization error. The shading is the exact density ρ<sub>τ</sub>, normalized at each τ so its shape stays visible at the noise end.'));
  const X = (fr, s) => fr.L + (fr.R - fr.L) * s / TAU_MAX;
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), fr = pframe(w, h), t = 1 - tau, j = Math.round(tau / TAU_MAX * NT), tj = 1 - TAUS[j];
    ctx.imageSmoothingEnabled = true; ctx.globalAlpha = isHL('rho') ? 1 : 0.85;
    ctx.drawImage(strip('fwd', TAUS, (y, s) => pt1(y, 1 - s)), fr.L, fr.top, fr.R - fr.L, fr.bot - fr.top); ctx.globalAlpha = 1;
    clip(ctx, fr.L, fr.top, fr.R - fr.L, fr.bot - fr.top);
    const pc = withPF ? pal.muted : hlc(['y', 'W', 'g'], pal, pal.fg);
    ctx.globalAlpha = withPF ? 0.28 : isHL('y', 'W', 'g') ? 0.75 : 0.42; ctx.lineWidth = 1;
    for (const i of SHOW) { ctx.beginPath(); for (let k = 0; k <= NT; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(fr, TAUS[k]), fr.Y(FWD.y[i][k])); ctx.strokeStyle = pc; ctx.stroke(); }
    ctx.globalAlpha = 1;
    if (withPF) PF.y.forEach((a, n) => {
      ctx.beginPath(); for (let k = 0; k <= NT; k++) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(fr, TAUS[k]), fr.Y(a[k]));
      ctx.strokeStyle = hlc(['vel', 'y'], pal, modeColor(pal, modeOfQ(PF.qs[n]))); ctx.lineWidth = isHL('vel') ? 2.2 : 1.5; ctx.stroke();
    });
    ctx.restore();
    hline(ctx, pal, fr.L, fr.R, fr.bot); vdash(ctx, hlc('tau', pal, pal.fg), X(fr, TAUS[j]), fr.top, fr.bot);
    yAxis(ctx, pal, fr, 'y', 'y');
    [0, 0.25, 0.5, 0.75].forEach((s) => label(ctx, String(s), X(fr, s), fr.bot + 16, pal.muted, 11, 'center'));
    label(ctx, '0.97', X(fr, TAU_MAX), fr.bot + 16, pal.muted, 11, 'center');
    label(ctx, 'τ (data at 0, noise towards the right)', fr.R, fr.bot + 32, hlc('tau', pal, pal.muted), 12.5, 'right');
    const vals = column(FWD.y, j);
    inset(ctx, pal, fr, vals, (y) => pt1(y, tj), { color: hlc(['y'], pal, pal.fg), hl: hlc('rho', pal, pal.acc2),
      ticks: withPF ? PF.y.map((a, n) => [a[j], hlc('vel', pal, modeColor(pal, modeOfQ(PF.qs[n])))]) : [] });
    label(ctx, `histogram at τ = ${TAUS[j].toFixed(2)}`, fr.iL, fr.top - 10, pal.muted, 12);
    label(ctx, 'line: exact ρ_τ', fr.iR, fr.bot + 16, hlc('rho', pal, pal.acc2), 12, 'right');
    const [m, v] = meanVar(vals), [me, ve] = exactMV(tj);
    if (withPF) {
      const sorted = Float64Array.from(vals).sort(), q = (p) => sorted[Math.floor(p * sorted.length)];
      ro.innerHTML = `τ = <b>${TAUS[j].toFixed(2)}</b> · position of the ODE path started at the 10%, 50% and 90% quantile: <b>${[4, 24, 44].map((n) => PF.y[n][j].toFixed(2)).join(', ')}</b><br>` +
        `the same quantiles of the 1000 SDE paths at this τ: <b>${[0.1, 0.5, 0.9].map((p) => q(p).toFixed(2)).join(', ')}</b> · of the exact ρ<sub>τ</sub>: <b>${[0.1, 0.5, 0.9].map((p) => quantileT(p, tj).toFixed(2)).join(', ')}</b>`;
    } else {
      ro.innerHTML = `τ = <b>${tau.toFixed(2)}</b> · t = 1 − τ = <b>${t.toFixed(2)}</b> · α<sub>t</sub> = <b>${LIN.alpha(t).toFixed(2)}</b> · σ<sub>t</sub> = <b>${LIN.sigma(t).toFixed(2)}</b> · f<sub>τ</sub> = <b>${f2(LIN.f(t))}</b> · g<sub>τ</sub>² = <b>${f2(LIN.g2(t))}</b><br>` +
        `1000 paths at τ = ${TAUS[j].toFixed(2)}: mean <b>${m.toFixed(2)}</b>, variance <b>${v.toFixed(3)}</b> · exact: mean <b>${me.toFixed(2)}</b>, variance α<sub>t</sub>² Var(data) + σ<sub>t</sub>² = <b>${ve.toFixed(3)}</b>`;
    }
  }
  return { draw };
}
// Quantile of p_t by bisection on the exact CDF, via the density integrated on a fine grid.
function quantileT(q, t) {
  const n = 2000, a = -8, b = 8, d = (b - a) / n; let c = 0;
  for (let i = 0; i < n; i++) { const p = pt1(a + (i + 0.5) * d, t) * d; if (c + p >= q) return a + (i + (q - c) / p) * d; c += p; }
  return b;
}
const forward = (viz) => forwardPlot(viz, false);
const pfode = (viz) => forwardPlot(viz, true);

// ---------- Steps 2 and 3: the terms of Fokker-Planck ----------
const DOTS = Array.from({ length: 41 }, (_, i) => -3.6 + 7.2 * i / 40);
function fpPlot(viz, asTransport) {
  const c = makeCanvas(16 / 10), ro = readout();
  let tau = asTransport ? 0.4 : 0.15, probe = null;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = LO + (HI - LO) * (e.clientX - r.left - 46) / (r.width - 64); draw(); });
  c.cv.addEventListener('pointerleave', () => { probe = null; draw(); });
  viz.append(c.cv, controls(slider('tau', { min: asTransport ? 0.05 : 0.02, max: 0.9, step: 0.01, value: tau }, (v) => { tau = v; draw(); })), ro,
    note(asTransport
      ? 'All curves are exact. The vertical scales adapt to τ; the numbers at the left edge give the range. The arrows under the upper panel show the bold velocity at 25 points, with their lengths compressed.'
      : 'All curves are exact, and the dots are an independent finite difference of the exact density in τ. The vertical scale of the lower panel adapts to τ.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), t = 1 - tau, L = 46, R = w - 18;
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    const T = XS.map((x) => fpTerms(x, tau)), pm = Math.max(...T.map((q) => q.p)) * 1.1;
    const dts = DOTS.map((x) => dtau1(x, tau));
    // Continuity form: -d/dy (v rho), with v = b - 1/2 g^2 score, by a central difference in y.
    const vel = (x) => { const q = fpTerms(x, tau); return q.b + q.vdiff; };
    const cont = (x) => { const e = 1e-3; return -((vel(x + e) * pt1(x + e, t)) - (vel(x - e) * pt1(x - e, t))) / (2 * e); };
    let top1, bot1, top2, bot2;
    if (asTransport) { top1 = 24; bot1 = h * 0.5; top2 = h * 0.64; bot2 = h - 30; } else { top1 = 24; bot1 = h * 0.33; top2 = h * 0.42; bot2 = h - 30; }
    // Upper panel.
    if (!asTransport) {
      const Yp = (p) => bot1 - (bot1 - top1) * p / pm;
      shade(ctx, XS, X, Yp, (x, i) => T[i].p, bot1, hlc('rho', pal, pal.acc2), isHL('rho') ? 0.5 : 0.24);
      curve(ctx, XS, X, Yp, (x, i) => T[i].p, hlc('rho', pal, pal.acc2), 1.6);
      hline(ctx, pal, L, R, bot1);
      label(ctx, `ρ_τ at τ = ${tau.toFixed(2)}`, L + 4, top1 - 6, hlc('rho', pal, pal.acc2), 13);
    } else {
      const inner = XS.filter((x) => Math.abs(x) < 3.4), vs = inner.map((x) => fpTerms(x, tau));
      const M = Math.max(...vs.map((q) => Math.max(Math.abs(q.b), Math.abs(q.vdiff), Math.abs(q.b + q.vdiff)))) * 1.1;
      const mid = (top1 + bot1) / 2, Yv = (v) => mid - (bot1 - top1) / 2 * Math.max(-1.05, Math.min(1.05, v / M));
      const Yp = (p) => bot1 - (bot1 - top1) * 0.55 * p / pm;
      shade(ctx, XS, X, Yp, (x, i) => T[i].p, bot1, hlc('rho', pal, pal.acc2), isHL('rho') ? 0.4 : 0.14);
      hline(ctx, pal, L, R, mid);
      clip(ctx, L, top1 - 4, R - L, bot1 - top1 + 8);
      curve(ctx, XS, X, Yv, (x, i) => T[i].b, hlc(['drift', 'f'], pal, pal.ramp[0]), isHL('drift', 'f') ? 3.2 : 2);
      curve(ctx, XS, X, Yv, (x, i) => T[i].vdiff, hlc(['glr', 'g'], pal, pal.ramp[3]), isHL('glr', 'g') ? 3.2 : 2);
      curve(ctx, XS, X, Yv, (x, i) => T[i].b + T[i].vdiff, hlc('vel', pal, pal.fg), isHL('vel') ? 4 : 3);
      ctx.restore();
      label(ctx, '0', L - 8, mid + 4, pal.muted, 11, 'right');
      label(ctx, f2(M / 1.1), L - 8, Yv(M / 1.1) + 4, pal.muted, 11, 'right');
      label(ctx, 'b = f_τ y', L + 4, top1 - 6, hlc(['drift', 'f'], pal, pal.ramp[0]), 13);
      label(ctx, '−½ g_τ² ∇log ρ_τ', L + 100, top1 - 6, hlc(['glr', 'g'], pal, pal.ramp[3]), 13);
      label(ctx, 'bold: their sum, the velocity', L + 260, top1 - 6, hlc('vel', pal, pal.fg), 13);
      // Arrow strip.
      const ay = bot1 + (top2 - bot1) * 0.42;
      for (let i = 0; i < 25; i++) {
        const x = -3.6 + 7.2 * i / 24, v = vel(x), len = 26 * Math.tanh(2 * v / M);
        arrow(ctx, X(x), ay, X(x) + len, ay, hlc('vel', pal, pal.fg), 1.8, 6);
      }
    }
    // Lower panel: the rate of change of the density.
    const sum = (i) => T[i].transport + T[i].diffusion;
    const M2 = Math.max(...T.map((q, i) => Math.max(Math.abs(q.transport), Math.abs(q.diffusion), Math.abs(sum(i)))), ...dts.map(Math.abs)) * 1.1;
    const mid2 = (top2 + bot2) / 2, Y2 = (v) => mid2 - (bot2 - top2) / 2 * v / (asTransport ? Math.max(...dts.map(Math.abs)) * 1.25 : M2);
    hline(ctx, pal, L, R, mid2);
    clip(ctx, L, top2 - 4, R - L, bot2 - top2 + 8);
    if (asTransport) curve(ctx, XS, X, Y2, (x) => cont(x), hlc(['vel', 'div'], pal, pal.fg), 2.6);
    else {
      curve(ctx, XS, X, Y2, (x, i) => T[i].transport, hlc(['drift', 'f', 'div'], pal, pal.ramp[0]), isHL('drift', 'f', 'div') ? 3.2 : 2);
      curve(ctx, XS, X, Y2, (x, i) => T[i].diffusion, hlc(['lap', 'g'], pal, pal.ramp[3]), isHL('lap', 'g') ? 3.2 : 2);
      curve(ctx, XS, X, Y2, (x, i) => sum(i), pal.fg, 3);
    }
    ctx.restore();
    DOTS.forEach((x, i) => { dot(ctx, X(x), Y2(dts[i]), isHL('dt') ? 4.5 : 3.4, hlc('dt', pal, pal.acc)); });
    label(ctx, '0', L - 8, mid2 + 4, pal.muted, 11, 'right');
    if (asTransport) label(ctx, '−∇·((b − ½ g_τ² ∇log ρ_τ) ρ_τ), dots: ∂_τ ρ_τ', L + 4, top2 - 8, hlc(['vel', 'dt'], pal, pal.fg), 13);
    else {
      label(ctx, 'transport −∇·(b ρ_τ)', L + 4, top2 - 8, hlc(['drift', 'f', 'div'], pal, pal.ramp[0]), 13);
      label(ctx, 'diffusion ½ g_τ² Δρ_τ', L + 170, top2 - 8, hlc(['lap', 'g'], pal, pal.ramp[3]), 13);
      label(ctx, 'bold: sum · dots: ∂_τ ρ_τ', R, top2 - 8, hlc('dt', pal, pal.fg), 13, 'right');
    }
    label(ctx, 'y', R, h - 10, hlc('y', pal, pal.muted), 13, 'right');
    // Residual of the equation at the dots.
    let res = 0; const dmax = Math.max(...dts.map(Math.abs));
    DOTS.forEach((x, i) => { const q = fpTerms(x, tau), rhs = asTransport ? cont(x) : q.transport + q.diffusion; res = Math.max(res, Math.abs(rhs - dts[i])); });
    let pr = '';
    if (probe !== null && probe > LO && probe < HI) {
      vdash(ctx, pal.muted, X(probe), top1, bot2);
      const q = fpTerms(probe, tau);
      pr = asTransport
        ? `y = <b>${probe.toFixed(2)}</b> · b = <b>${q.b.toFixed(2)}</b> · −½g²∇log ρ = <b>${q.vdiff.toFixed(2)}</b> · velocity = <b>${(q.b + q.vdiff).toFixed(2)}</b><br>`
        : `y = <b>${probe.toFixed(2)}</b> · transport <b>${q.transport.toFixed(3)}</b> · diffusion <b>${q.diffusion.toFixed(3)}</b> · sum <b>${(q.transport + q.diffusion).toFixed(3)}</b> · ∂<sub>τ</sub>ρ<sub>τ</sub> = <b>${dtau1(probe, tau).toFixed(3)}</b><br>`;
    }
    ro.innerHTML = pr + `τ = <b>${tau.toFixed(2)}</b> · f<sub>τ</sub> = <b>${f2(LIN.f(t))}</b> · g<sub>τ</sub>² = <b>${f2(LIN.g2(t))}</b> · largest gap between the ${asTransport ? 'continuity form' : 'sum'} and the dots, relative to the largest dot: <b>${(res / dmax).toExponential(1)}</b>`;
  }
  return { draw };
}
const fp = (viz) => fpPlot(viz, false);
const transport = (viz) => fpPlot(viz, true);

// ---------- Step 5: the reverse family ----------
const revCache = new Map();
function rev(eta) {
  const k = eta.toFixed(2);
  if (!revCache.has(k)) { if (revCache.size > 6) revCache.delete(revCache.keys().next().value); revCache.set(k, reverse1(eta)); }
  return revCache.get(k);
}
function reverse(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let eta = 1, t = 0.6;
  viz.append(c.cv, controls(
    slider('eta', { min: 0, max: 1, step: 0.05, value: eta }, (v) => { eta = v; draw(); }),
    slider('t', { min: T_MIN, max: 1, step: 0.01, value: t }, (v) => { t = v; draw(); })), ro,
  note(`1000 particles, started at stratified quantiles of the exact p<sub>${T_MIN}</sub>, ${NR} Euler–Maruyama steps with the exact score. The noise draws are fixed, so changing η changes only the equation. 40 paths are drawn.`));
  const X = (fr, s) => fr.L + (fr.R - fr.L) * (s - T_MIN) / (1 - T_MIN);
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), fr = pframe(w, h), j = Math.round((t - T_MIN) / (1 - T_MIN) * NR), tj = TR[j], R = rev(eta);
    ctx.globalAlpha = isHL('pt') ? 1 : 0.85;
    ctx.drawImage(strip('rev', TR.filter((_, i) => i % 2 === 0), (y, s) => pt1(y, s)), fr.L, fr.top, fr.R - fr.L, fr.bot - fr.top); ctx.globalAlpha = 1;
    clip(ctx, fr.L, fr.top, fr.R - fr.L, fr.bot - fr.top);
    ctx.globalAlpha = isHL('x', 'eta', 'W') ? 0.8 : 0.5; ctx.lineWidth = 1;
    for (const i of SHOW) {
      ctx.beginPath(); for (let k = 0; k <= NR; k += 2) (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(fr, TR[k]), fr.Y(R.x[i][k]));
      ctx.strokeStyle = hlc(['x', 'eta', 'W'], pal, modeColor(pal, R.x[i][NR] < -0.65 ? 0 : R.x[i][NR] < 1.1 ? 1 : 2)); ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.restore();
    hline(ctx, pal, fr.L, fr.R, fr.bot); vdash(ctx, hlc('t', pal, pal.fg), X(fr, tj), fr.top, fr.bot);
    yAxis(ctx, pal, fr, 'x', 'x');
    [0.03, 0.25, 0.5, 0.75, 1].forEach((s) => label(ctx, String(s), X(fr, s), fr.bot + 16, pal.muted, 11, 'center'));
    label(ctx, 't (noise at the left, data at 1)', fr.R, fr.bot + 32, hlc('t', pal, pal.muted), 12.5, 'right');
    const vals = column(R.x, j);
    inset(ctx, pal, fr, vals, (y) => pt1(y, tj), { color: hlc('x', pal, pal.fg), hl: hlc('pt', pal, pal.acc2) });
    label(ctx, `histogram at t = ${tj.toFixed(2)}`, fr.iL, fr.top - 10, pal.muted, 12);
    label(ctx, 'line: exact p_t', fr.iR, fr.bot + 16, hlc('pt', pal, pal.acc2), 12, 'right');
    tag(ctx, pal, eta === 0 ? 'η = 0: probability-flow ODE' : eta === 1 ? 'η = 1: reverse SDE' : `η = ${eta.toFixed(2)}`, fr.L + 10, fr.top + 20, hlc('eta', pal, pal.fg), 13.5);
    const [m, v] = meanVar(vals), [me, ve] = exactMV(tj);
    ro.innerHTML = `η = <b>${eta.toFixed(2)}</b> · t = <b>${tj.toFixed(2)}</b> · 1000 particles: mean <b>${m.toFixed(2)}</b>, variance <b>${v.toFixed(3)}</b> · exact p<sub>t</sub>: mean <b>${me.toFixed(2)}</b>, variance <b>${ve.toFixed(3)}</b>`;
  }
  return { draw };
}

// ---------- Step 6: ODE and SDE samplers on the 2D mixture ----------
function view(x0, y0, w, h, bx = 3.2, by = 2.6) {
  const s = Math.min(w / (2 * bx), h / (2 * by)), cx = x0 + w / 2, cy = y0 + h / 2;
  return { s, px: (x, y) => [cx + s * x, cy - s * y] };
}
const heatCache = new Map();
function heat(w, h, v, x0, y0) {
  const k = `${w}x${h}|${live.theme}`;
  if (heatCache.has(k)) return heatCache.get(k);
  const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res), off = document.createElement('canvas'); off.width = W; off.height = H;
  const g = off.getContext('2d'), im = g.createImageData(W, H), pmax = Math.max(...MIX2.means.map(([a, b]) => p2(a, b))) * 1.05;
  const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224], amax = live.theme === 'dark' ? 0.5 : 0.32;
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const x = (x0 + i * res - (x0 + w / 2)) / v.s, y = ((y0 + h / 2) - (y0 + j * res)) / v.s, o = 4 * (j * W + i);
    im.data[o] = rgb[0]; im.data[o + 1] = rgb[1]; im.data[o + 2] = rgb[2]; im.data[o + 3] = Math.round(255 * amax * Math.pow(Math.min(1, p2(x, y) / pmax), 0.7));
  }
  g.putImageData(im, 0, 0);
  if (heatCache.size > 20) heatCache.clear();
  heatCache.set(k, off);
  return off;
}
const sampCache = new Map();
function samp(K, sde) {
  const k = `${K}|${sde}`;
  if (!sampCache.has(k)) {
    if (sampCache.size > 12) sampCache.delete(sampCache.keys().next().value);
    const r = sample2(K, sde); sampCache.set(k, { ...r, q: quality2(r.path[K]) });
  }
  return sampCache.get(k);
}
const TRAILS = Array.from({ length: 36 }, (_, i) => 7 + 13 * i), TRAIL_T = 0.3;
function ours(viz) {
  const c = makeCanvas(16 / 8.4), ro = readout();
  let K = 20, u = 0;
  const r = runner({ step: () => { u = Math.min(1, u + 1 / 80); }, done: () => u >= 1, draw: () => draw(), interval: 30 });
  const kSl = slider('K', { min: Math.log10(5), max: Math.log10(200), step: 0.01, value: Math.log10(K), map: (x) => Math.round(10 ** x), fmt: (v) => String(v) }, (v) => { K = v; u = 1; draw(); });
  viz.append(c.cv, controls(kSl, button('Run', () => { u = 0; r.playing = true; r.kick(); })), ro,
    note(`500 particles from N(0, I) at t = ${T0}, the same draws for both samplers and every K. Exact score of the 2D mixture of §2.2. The heat map is p<sub>data</sub>. Trails start at t = ${TRAIL_T}, for 36 particles of the ODE and 12 of the SDE, whose early noisy steps would otherwise cover the panel. Particles that leave the frame are not drawn but still count in the readout.`));
  function panel(ctx, pal, x0, w, h, run, title, id) {
    const v = view(x0, 0, w, h), ts = run.ts, Kk = ts.length - 1;
    ctx.globalAlpha = isHL('pdata') ? 1 : 0.75; ctx.drawImage(heat(w, h, v, x0, 0), x0, 0, w, h); ctx.globalAlpha = 1;
    clip(ctx, x0, 0, w, h);
    const pos = u * Kk, k = Math.min(Kk - 1, Math.floor(pos)), fr = pos - k;
    const at = (i, d) => run.path[k][2 * i + d] * (1 - fr) + run.path[k + 1][2 * i + d] * fr;
    ctx.strokeStyle = hlc(['x'], pal, pal.muted); ctx.lineWidth = 0.9; ctx.globalAlpha = 0.55;
    let q0 = 0; while (q0 < Kk && ts[q0] < TRAIL_T) q0++;
    for (const i of (id === 'W' ? TRAILS.filter((_, j) => j % 3 === 0) : TRAILS)) {
      if (k < q0) break;
      ctx.beginPath(); ctx.moveTo(...v.px(run.path[q0][2 * i], run.path[q0][2 * i + 1]));
      for (let q = q0 + 1; q <= k; q++) ctx.lineTo(...v.px(run.path[q][2 * i], run.path[q][2 * i + 1]));
      ctx.lineTo(...v.px(at(i, 0), at(i, 1))); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const col = hlc(['x', id], pal, pal.fg);
    for (let i = 0; i < 500; i++) { const [X, Y] = v.px(at(i, 0), at(i, 1)); ctx.fillStyle = col; ctx.fillRect(X - 1.3, Y - 1.3, 2.6, 2.6); }
    ctx.restore();
    frame(ctx, pal, x0, 0, w, h);
    tag(ctx, pal, title, x0 + 14, 24, hlc(id, pal, pal.fg), 14);
    tag(ctx, pal, `t = ${(ts[k] + (ts[k + 1] - ts[k]) * fr).toFixed(3)}`, x0 + w - 12, h - 12, hlc('t', pal, pal.muted), 12.5, 'right');
  }
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), gap = 14, pw = (w - gap) / 2;
    const ode = samp(K, false), sde = samp(K, true);
    panel(ctx, pal, 0, pw, h, ode, `ODE, ${K} Euler steps`, 'lamdot');
    panel(ctx, pal, pw + gap, pw, h, sde, `SDE, ${K} Euler–Maruyama steps`, 'W');
    const mm = (q) => q.mass.map((x) => x.toFixed(2)).join(' / ');
    ro.innerHTML = `K = <b>${K}</b> · mass per mode after the last step (exact ${W2.map((x) => x.toFixed(2)).join(' / ')}): ODE <b>${mm(ode.q)}</b>, SDE <b>${mm(sde.q)}</b><br>` +
      `mean log p<sub>data</sub> of the samples (exact samples: ${LL_EXACT.toFixed(2)} ± 0.05): ODE <b>${f2(ode.q.ll)}</b>, SDE <b>${f2(sde.q.ll)}</b>`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 7: schedules as reparametrizations ----------
const SNAMES = { linear: 'Linear', cosine: 'Cosine (variance preserving)', ve: 'Variance exploding' };
// t at every point of the EDM grid in log sigma, per schedule.
const TOF = Object.fromEntries(Object.entries(SCHEDULES).map(([k, S]) => [k, Float64Array.from(LOGR, (l) => tOfRatio(S, Math.exp(l)))]));
const RANGE = Object.fromEntries(Object.entries(SCHEDULES).map(([k, S]) => [k, [tOfRatio(S, R_MAX), tOfRatio(S, R_MIN)]]));
const gridT = (key, K) => { const [a, b] = RANGE[key]; return Array.from({ length: K + 1 }, (_, k) => a + (b - a) * k / K); };
const ratioAt = (key, t) => { const S = SCHEDULES[key]; return S.sigma(t) / S.alpha(t); };
const below = (key, K, r0 = 0.3) => gridT(key, K).slice(1).filter((t) => ratioAt(key, t) < r0).length;
function schedules(viz) {
  const c = makeCanvas(16 / 11, { maxH: 0.7 }), ro = readout();
  let key = 'linear', K = 12;
  const btns = Object.keys(SNAMES).map((k) => { const b = button(SNAMES[k], () => { key = k; btns.forEach((x) => x.classList.toggle('on', x === b)); draw(); }, k === key ? 'on' : ''); return b; });
  viz.append(c.cv, controls(...btns, slider('K', { min: 4, max: 30, step: 1, value: K, fmt: (v) => String(v) }, (v) => { K = v; draw(); })), ro,
    note(`31 probability-flow paths of the 1D mixture, started at quantiles of p(y; σ) at σ = ${R_MAX} and integrated in log σ down to σ = ${R_MIN} with the exact score. Every schedule is run over this same range of σ.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), S = SCHEDULES[key], L = 52, R = w - 20, gT = gridT(key, K), tt = TOF[key];
    const top1 = 26, bot1 = h * 0.44, top2 = h * 0.56, bot2 = h - 34;
    // Upper panel: x_t = alpha_t y against t.
    let M = 0; EDM.y.forEach((a) => a.forEach((y, j) => { M = Math.max(M, Math.abs(S.alpha(tt[j]) * y)); }));
    M *= 1.08;
    const X1 = (t) => L + (R - L) * t, Y1 = (x) => (top1 + bot1) / 2 - (bot1 - top1) / 2 * x / M;
    hline(ctx, pal, L, R, (top1 + bot1) / 2); hline(ctx, pal, L, R, bot1);
    gT.forEach((t) => { ctx.fillStyle = pal.faint; ctx.fillRect(Math.round(X1(t)), top1, 1, bot1 - top1); });
    EDM.y.forEach((a, n) => {
      ctx.beginPath(); a.forEach((y, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, X1(tt[j]), Y1(S.alpha(tt[j]) * y)));
      ctx.strokeStyle = hlc(['x', 'alpha'], pal, modeColor(pal, modeOfQ(EDM.qs[n]))); ctx.lineWidth = 1.5; ctx.stroke();
    });
    gT.forEach((t) => dot(ctx, X1(t), bot1, isHL('K') ? 4.5 : 3.4, hlc('K', pal, pal.acc)));
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => label(ctx, String(t), X1(t), bot1 + 16, pal.muted, 11, 'center'));
    label(ctx, f2(M / 1.08), L - 8, Y1(M / 1.08) + 4, pal.muted, 11, 'right'); label(ctx, '0', L - 8, Y1(0) + 4, pal.muted, 11, 'right');
    label(ctx, `x_t = α_t y against t, ${SNAMES[key].toLowerCase()} schedule`, L + 4, top1 - 8, hlc(['x', 'alpha'], pal, pal.fg), 13);
    label(ctx, 't', R, bot1 + 30, hlc('t', pal, pal.muted), 13, 'right');
    // Lower panel: y against log sigma, the same for every schedule.
    const lA = Math.log(R_MAX), lB = Math.log(R_MIN), YM = 8.2;
    const X2 = (l) => L + (R - L) * (l - lA) / (lB - lA), Y2 = (y) => (top2 + bot2) / 2 - (bot2 - top2) / 2 * y / YM;
    hline(ctx, pal, L, R, (top2 + bot2) / 2); hline(ctx, pal, L, R, bot2);
    ctx.fillStyle = hlc('r', pal, pal.muted); ctx.globalAlpha = 0.08; ctx.fillRect(X2(Math.log(0.3)), top2, R - X2(Math.log(0.3)), bot2 - top2); ctx.globalAlpha = 1;
    vdash(ctx, pal.muted, X2(Math.log(0.3)), top2, bot2);
    EDM.y.forEach((a, n) => {
      ctx.beginPath(); a.forEach((y, j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, X2(LOGR[j]), Y2(y)));
      ctx.strokeStyle = hlc(['yr', 'psig'], pal, modeColor(pal, modeOfQ(EDM.qs[n]))); ctx.lineWidth = 1.5; ctx.stroke();
    });
    gT.forEach((t) => dot(ctx, X2(Math.log(ratioAt(key, t))), bot2, isHL('K') ? 4.5 : 3.4, hlc('K', pal, pal.acc)));
    [4, 1, 0.3, 0.1, 0.03].forEach((r) => label(ctx, String(r), X2(Math.log(r)), bot2 + 16, pal.muted, 11, 'center'));
    [-6, 0, 6].forEach((y) => label(ctx, String(y), L - 8, Y2(y) + 4, pal.muted, 11, 'right'));
    label(ctx, 'y = x_t/α_t against σ = σ_t/α_t (log scale): identical for every schedule', L + 4, top2 - 8, hlc(['yr', 'r'], pal, pal.fg), 13);
    label(ctx, 'σ', R, bot2 + 30, hlc('r', pal, pal.muted), 13, 'right');
    label(ctx, 'dots: K equal steps in t', L, bot2 + 30, hlc('K', pal, pal.acc), 12.5);
    const [a, b] = RANGE[key];
    ro.innerHTML = `${SNAMES[key]}: t runs from <b>${a.toFixed(3)}</b> to <b>${b.toFixed(3)}</b> to cover σ from ${R_MAX} to ${R_MIN} · α<sub>t</sub> at the start <b>${S.alpha(a).toFixed(2)}</b><br>` +
      `steps of the ${K} that end below σ = 0.3: ` + Object.keys(SNAMES).map((k) => `${SNAMES[k].split(' ')[0].toLowerCase()} <b>${below(k, K)}</b>`).join(' · ');
  }
  return { draw };
}

export const SCENES = { forward, fp, transport, pfode, reverse, ours, schedules };
