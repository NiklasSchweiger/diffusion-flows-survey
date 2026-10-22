// Live scenes of the §2.5 tutorial, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import {
  LAM_MIN, LAM_MAX, pdata, ELL, LAMS, NL, ell, L_INF, levels, bound, H_DATA, stepPost, pLam, NDRAW, postDraws,
  SCHED, fSched, NMC, RUN, WNAMES, W, odeLogp, NQ, XQ,
} from './sim.js';

// ---------- Helpers ----------
const LO = -3.6, HI = 3.8, XS = Array.from({ length: 297 }, (_, i) => LO + (HI - LO) * i / 296);
const modeColor = (pal, k) => [pal.ramp[0], pal.ramp[1], pal.ramp[3]][k];
const dot = (ctx, X, Y, r, color) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = color; ctx.fill(); };
const ring = (ctx, X, Y, r, color, lw = 1.6) => { ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); };
function tag(ctx, pal, s, x, y, color, size = 13, align = 'left') {
  ctx.font = `500 ${size}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(s.replace(/_/g, '')).width + 12, x0 = align === 'right' ? x - w + 6 : x - 6;
  ctx.fillStyle = pal.bg; ctx.globalAlpha = 0.85; ctx.fillRect(x0, y - size - 3, w, size + 10); ctx.globalAlpha = 1;
  label(ctx, s, x, y, color, size, align);
}
function frame(ctx, pal, x, y, w, h) { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
const hline = (ctx, pal, x0, x1, y) => { ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); };
function vdash(ctx, color, x, y0, y1) { ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke(); ctx.setLineDash([]); }
function hdash(ctx, color, x0, x1, y) { ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); }
function curve(ctx, xs, X, Y, f, color, lw, dash) {
  ctx.beginPath(); xs.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y(f(x, i))));
  if (dash) ctx.setLineDash(dash);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]);
}
function shade(ctx, xs, X, Y, f, base, color, alpha) {
  ctx.beginPath(); ctx.moveTo(X(xs[0]), base); xs.forEach((x, i) => ctx.lineTo(X(x), Y(f(x, i)))); ctx.lineTo(X(xs[xs.length - 1]), base); ctx.closePath();
  ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.fill(); ctx.globalAlpha = 1;
}
const clip = (ctx, x, y, w, h) => { ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); };
const f2 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
const f3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
// Small cache for expensive per-parameter computations.
function memo(fn, size = 16) {
  const m = new Map();
  return (key) => { if (!m.has(key)) { if (m.size >= size) m.delete(m.keys().next().value); m.set(key, fn(key)); } return m.get(key); };
}
// The probe follows the cursor along the horizontal axis X of a plot; inv maps a canvas x back to data.
const track1 = (c, inv, onMove) => c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); onMove(inv(e.clientX - r.left)); });
// Horizontal axis in log-SNR, shared by steps 3, 5 and 6.
function lamAxis(ctx, pal, X, y, color = pal.muted) {
  hline(ctx, pal, X(LAM_MIN), X(LAM_MAX), y);
  for (let l = LAM_MIN; l <= LAM_MAX; l += 3) label(ctx, String(l), X(l), y + 15, color, 11, 'center');
  label(ctx, 'λ', X(LAM_MAX), y + 30, color, 13, 'right');
}
// One button per key, of which exactly one is on.
function choice(names, key, on) {
  const btns = Object.keys(names).map((k) => { const b = button(names[k], () => { btns.forEach((x) => x.classList.toggle('on', x === b)); on(k); }, k === key ? 'on' : ''); return b; });
  return btns;
}
// Values of N offered by the sliders of steps 1 and 3.
const NS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024];
const nSlider = (i0, on) => slider('Nl', { min: 0, max: NS.length - 1, step: 1, value: i0, map: (i) => NS[i], fmt: (v) => String(v) }, on);
const boundN = memo(bound, 20);

// ---------- Step 1: the latent levels and the bound against N ----------
const GAPS = NS.map((n) => bound(n).total - H_DATA);
function vae(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let N = 8;
  viz.append(c.cv, controls(nSlider(NS.indexOf(N), (v) => { N = v; draw(); })), ro,
    note('Every density is exact. The latent levels are equally spaced in log-SNR on the variance-preserving path, \\(\\alpha_t^2 = 1/(1 + e^{-\\lambda})\\). Each decoder step uses the exact posterior mean of \\(\\x_1\\) given the noisier level; the reconstruction decoder is a Gaussian with the posterior scale of the last level. At most 16 rows are drawn.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), b = boundN(N);
    // Left: ridgeline of p(x_{t_i}), noise at the bottom.
    const L = 24, R = w * 0.56, top = 62, bot = h - 34;
    const show = N <= 16 ? [...b.lams.keys()] : Array.from({ length: 16 }, (_, j) => Math.round(j * (N - 1) / 15));
    // One row per latent level, plus the data density x_1 on top (it is not a latent).
    const rows = show.length + 1, hrow = Math.min((bot - top) / Math.max(rows - 1, 1) * 2, (bot - top) * 0.45);
    const X = (x) => L + (R - L) * (x - LO) / (HI - LO);
    const row = (j, f, col, a, lw) => {
      const base = bot - j * (bot - top - hrow * 0.6) / (rows - 1);
      const pmax = Math.max(...XS.map(f)), Y = (p) => base - hrow * p / Math.max(pmax, 0.42);
      shade(ctx, XS, X, Y, f, base, col, a); curve(ctx, XS, X, Y, f, col, lw);
      return base;
    };
    const lcol = hlc(['xt', 'q', 'Nl'], pal, pal.ramp[1]);
    show.forEach((i, j) => {
      const lam = b.lams[i], base = row(j, (x) => pLam(x, lam), lcol, 0.14, 1.3);
      if (j === 0 || j === show.length - 1 || show.length <= 8) label(ctx, `λ = ${lam.toFixed(1)}`, R + 6, base - 3, pal.muted, 11);
    });
    const dbase = row(rows - 1, pdata, hlc('x1', pal, pal.acc), 0.35, 2.2);
    label(ctx, 'data x_1', R + 6, dbase - 3, hlc('x1', pal, pal.fg), 12);
    label(ctx, '(not a latent)', R + 6, dbase + 12, pal.muted, 11);
    label(ctx, `latents x_t at N = ${N} levels, noisiest at the bottom: x_{t₀}`, L, h - 10, hlc('xt', pal, pal.muted), 12.5);
    // Right: excess of the bound over the entropy, log-log.
    const PL = w * 0.69, PR = w - 18, PT = 40, PB = h - 46;
    const lx = (n) => PL + (PR - PL) * Math.log(n / 2) / Math.log(512), ly = (g) => PB - (PB - PT) * (Math.log10(g) + 2) / 6;
    hline(ctx, pal, PL, PR, PB); ctx.strokeStyle = pal.line; ctx.beginPath(); ctx.moveTo(PL + 0.5, PT); ctx.lineTo(PL + 0.5, PB); ctx.stroke();
    [-2, 0, 2, 4].forEach((e) => label(ctx, `10${e < 0 ? '⁻' : ''}${'⁰¹²³⁴'[Math.abs(e)]}`, PL - 6, ly(10 ** e) + 4, pal.muted, 11, 'right'));
    [2, 8, 32, 128, 1024].forEach((n) => label(ctx, String(n), lx(n), PB + 15, pal.muted, 11, 'center'));
    label(ctx, 'N', PR, PB + 30, hlc('Nl', pal, pal.muted), 13, 'right');
    ctx.beginPath(); NS.forEach((n, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, lx(n), ly(GAPS[i])));
    ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.4; ctx.stroke();
    NS.forEach((n, i) => dot(ctx, lx(n), ly(GAPS[i]), n === N ? 6 : 3, n === N ? hlc('Nl', pal, pal.acc) : pal.muted));
    label(ctx, 'bound − entropy of the data (nats)', PL, PT - 16, hlc('ptheta', pal, pal.fg), 13);
    frame(ctx, pal, 0, 0, w, h);
    const big = b.terms.reduce((a, v) => Math.max(a, v), 0);
    ro.innerHTML = `N = <b>${N}</b> · prior <b>${b.prior.toExponential(1)}</b> · denoising terms <b>${f3(b.den)}</b> (largest single term <b>${f3(big)}</b>) · reconstruction <b>${f3(b.recon)}</b><br>` +
      `bound on −log p averaged over the data <b>${f3(b.total)}</b> · entropy of the data <b>${H_DATA.toFixed(3)}</b> · excess <b>${f3(b.total - H_DATA)}</b> nats`;
  }
  return { draw };
}

// ---------- Step 2: one KL term at a probe x_s ----------
function kl(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let ls = -1, dl = 1, xs = 0.35;
  const L = 24, R = () => c.w - 24, X = (x) => L + (R() - L) * (x - LO) / (HI - LO);
  track1(c, (px) => LO + (HI - LO) * (px - L) / (R() - L), (x) => { xs = Math.max(LO, Math.min(HI, x)); draw(); });
  viz.append(c.cv, controls(
    slider('lam', { min: -6, max: 4, step: 0.1, value: ls, tex: '\\lambda_s', fmt: (v) => v.toFixed(1) }, (v) => { ls = v; draw(); }),
    slider('dlam', { min: 0.05, max: 2, step: 0.05, value: dl }, (v) => { dl = v; draw(); })), ro,
  note(`Move the cursor to place the probe x<sub>s</sub>, and click to pin it. Exact for the 1D mixture on the variance-preserving path. The ${NDRAW} thin Gaussians are q(x<sub>t</sub> | x<sub>s</sub>, x<sub>1</sub>) for stratified draws of x<sub>1</sub> from its exact posterior, coloured by mixture component; the bold one is the decoder. All have the same width ς and are drawn at the same height.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), lt = ls + dl, P = stepPost(ls, lt), D = postDraws(xs, ls);
    // Upper panel: density of x_s and the probe.
    const t1 = 34, b1 = h * 0.38, pm = Math.max(...XS.map((x) => pLam(x, ls))), Y1 = (p) => b1 - (b1 - t1) * p / pm;
    shade(ctx, XS, X, Y1, (x) => pLam(x, ls), b1, pal.ramp[1], 0.16);
    curve(ctx, XS, X, Y1, (x) => pLam(x, ls), hlc('lam', pal, pal.ramp[1]), 1.8);
    hline(ctx, pal, L, R(), b1);
    vdash(ctx, hlc('xs', pal, pal.fg), X(xs), t1 - 6, h - 30);
    dot(ctx, X(xs), b1, isHL('xs') ? 7 : 5.5, hlc('xs', pal, pal.fg));
    tag(ctx, pal, 'x_s', X(xs) + 10, b1 - 8, hlc('xs', pal, pal.fg), 14);
    label(ctx, `density of x_s at λ_s = ${ls.toFixed(1)}`, L, t1 - 12, hlc('lam', pal, pal.muted), 13);
    // Lower panel: encoder posteriors for draws of x1, and the decoder.
    const t2 = h * 0.5, b2 = h - 30, sd = Math.sqrt(P.v), Y2 = (g) => b2 - (b2 - t2) * 0.9 * g;
    const gs = (m) => (x) => Math.exp(-((x - m) ** 2) / (2 * P.v));
    const xs2 = Array.from({ length: 401 }, (_, i) => LO + (HI - LO) * i / 400);
    hline(ctx, pal, L, R(), b2);
    clip(ctx, L, t2 - 10, R() - L, b2 - t2 + 10);
    ctx.globalAlpha = isHL('x1', 'mu') ? 0.9 : 0.55;
    for (const d of D.draws) curve(ctx, xs2, X, Y2, gs(P.cs * xs + P.c1 * d.x1), hlc(['x1', 'mu'], pal, modeColor(pal, d.k)), 1.1);
    ctx.globalAlpha = 1;
    const mDec = P.cs * xs + P.c1 * D.e1;
    curve(ctx, xs2, X, Y2, gs(mDec), hlc(['hx1', 'ptheta'], pal, pal.fg), isHL('hx1', 'ptheta') ? 3.4 : 2.6);
    // Width marker: plus and minus one ς around the decoder mean.
    const yw = Y2(Math.exp(-0.5)), cw = hlc('vs', pal, pal.acc);
    ctx.strokeStyle = cw; ctx.lineWidth = isHL('vs') ? 3 : 2; ctx.beginPath(); ctx.moveTo(X(mDec - sd), yw); ctx.lineTo(X(mDec + sd), yw); ctx.stroke();
    ctx.restore();
    label(ctx, `λ_t = λ_s + Δλ = ${lt.toFixed(2)}`, L, t2 - 12, hlc('dlam', pal, pal.muted), 13);
    label(ctx, 'thin: q(x_t | x_s, x_1) for 16 draws of x_1 · bold: decoder p_θ(x_t | x_s)', R(), t2 - 12, hlc(['q', 'ptheta'], pal, pal.muted), 12.5, 'right');
    frame(ctx, pal, 0, 0, w, h);
    const resp = D.comps.map((q) => q[0].toFixed(2)).join(', ');
    ro.innerHTML = `x<sub>s</sub> = <b>${xs.toFixed(2)}</b> · posterior weight of the components <b>${resp}</b> · E[x<sub>1</sub> | x<sub>s</sub>] = <b>${D.e1.toFixed(3)}</b> · Var(x<sub>1</sub> | x<sub>s</sub>) = <b>${D.var1.toFixed(3)}</b><br>` +
      `ς² = <b>${P.v.toFixed(4)}</b> · weight ½(SNR(t) − SNR(s)) = <b>${(0.5 * P.dsnr).toFixed(4)}</b> · expected KL with the best decoder = weight × variance = <b>${(0.5 * P.dsnr * D.var1).toFixed(4)}</b> nats`;
  }
  return { draw };
}

// ---------- Step 3: the Riemann sum and the integral ----------
function vdm(viz) {
  const c = makeCanvas(16 / 8), ro = readout();
  let N = 16;
  viz.append(c.cv, controls(nSlider(NS.indexOf(N), (v) => { N = v; draw(); })), ro,
    note('The curve is the exact integrand ℓ(λ) = ½ e<sup>λ</sup> E[Var(x<sub>1</sub> | x<sub>λ</sub>)] for the 1D mixture, by quadrature. Each bar has the width of one step and the area of its KL term.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), b = boundN(N);
    const L = 52, R = w - 20, T = 34, B = h - 46, YM = 1.2;
    const X = (l) => L + (R - L) * (l - LAM_MIN) / (LAM_MAX - LAM_MIN), Y = (v) => B - (B - T) * Math.min(v, YM) / YM;
    const lams = Array.from(LAMS);
    shade(ctx, lams, X, Y, (l, i) => ELL[i], B, hlc(['ell', 'Linf'], pal, pal.ramp[1]), isHL('ell', 'Linf') ? 0.35 : 0.18);
    // Bars of the discrete bound.
    const dl = (LAM_MAX - LAM_MIN) / (N - 1), cb = hlc(['Nl', 'dlam'], pal, pal.acc);
    for (let i = 1; i < N; i++) {
      const hgt = b.terms[i - 1] / dl, x0 = X(b.lams[i - 1]), x1 = X(b.lams[i]);
      ctx.fillStyle = cb; ctx.globalAlpha = 0.18; ctx.fillRect(x0, Y(hgt), x1 - x0, B - Y(hgt)); ctx.globalAlpha = 1;
      ctx.strokeStyle = cb; ctx.lineWidth = N > 200 ? 0.6 : 1.2; ctx.strokeRect(x0 + 0.5, Y(hgt) + 0.5, x1 - x0 - 1, B - Y(hgt));
      if (hgt > YM) { ctx.fillStyle = cb; ctx.beginPath(); const m = (x0 + x1) / 2; ctx.moveTo(m - 5, T + 7); ctx.lineTo(m + 5, T + 7); ctx.lineTo(m, T - 1); ctx.fill(); }
    }
    curve(ctx, lams, X, Y, (l, i) => ELL[i], hlc('ell', pal, pal.fg), isHL('ell') ? 3.2 : 2.4);
    hdash(ctx, pal.muted, L, R, Y(0.5));
    label(ctx, 'd/2 = 0.5', L + 6, Y(0.5) - 6, pal.muted, 11.5);
    lamAxis(ctx, pal, X, B, hlc('lam', pal, pal.muted));
    [0, 0.5, 1].forEach((v) => label(ctx, v.toFixed(1), L - 8, Y(v) + 4, pal.muted, 11, 'right'));
    label(ctx, 'ℓ(λ)', L + 4, T - 14, hlc('ell', pal, pal.fg), 13);
    label(ctx, 'bars: discrete bound (cut at the top, marked ▲)', R, T - 14, cb, 12.5, 'right');
    frame(ctx, pal, 0, 0, w, h);
    ro.innerHTML = `N = <b>${N}</b> · sum of the ${N - 1} denoising terms (area of the bars) <b>${f3(b.den)}</b> · L<sub>∞</sub> (area under the curve) <b>${L_INF.toFixed(3)}</b> · difference <b>${f3(b.den - L_INF)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: four schedules, one integral ----------
const SKEYS = Object.keys(SCHED);
const sColor = (pal, i) => pal.ramp[i];
function schedule(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let n = NMC;
  const PER = 20;
  const r = runner({ step: () => { n = Math.min(NMC, n + PER); nSl.set(n); }, done: () => n >= NMC, draw: () => {}, interval: 30 });
  r.playing = false;
  const nSl = slider('n', { min: 1, max: NMC, step: 1, value: n, tex: 'n', fmt: (v) => String(v) }, (v) => { n = v; draw(); });
  viz.append(c.cv, controls(nSl, button('Play from n = 1', () => { n = 1; nSl.set(1); r.playing = true; r.kick(); })), ro,
    note(`The same ${NMC} uniform draws of t for every schedule. For each draw the expectation over data and noise is exact, \\(\\dot\\lambda_t\\,\\ell(\\lambda_t)\\) by quadrature, so the only randomness is the draw of t. All schedules run from λ = −9 to 9.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), L = 52, R = w - 20;
    // Upper panel: integrand in t.
    const T1 = 34, B1 = h * 0.47, YM = 16, X = (t) => L + (R - L) * t, Y1 = (v) => B1 - (B1 - T1) * Math.min(v, YM) / YM;
    const ts = Array.from({ length: 401 }, (_, i) => i / 400);
    hline(ctx, pal, L, R, B1);
    clip(ctx, L, T1 - 6, R - L, B1 - T1 + 6);
    SKEYS.forEach((k, i) => curve(ctx, ts, X, Y1, (t) => fSched(k, t), hlc(['sched', 'lamdot'], pal, sColor(pal, i)), isHL('sched', 'lamdot') ? 2.8 : 2));
    ctx.restore();
    [0, 8, 16].forEach((v) => label(ctx, String(v), L - 8, Y1(v) + 4, pal.muted, 11, 'right'));
    [0, 0.5].forEach((t) => label(ctx, t.toFixed(1), X(t), B1 + 15, pal.muted, 11, 'center'));
    label(ctx, 't = 1', R, B1 + 15, pal.muted, 11, 'right');
    label(ctx, 'λ̇_t ℓ(λ_t), cut at 16 · every curve encloses the area L_∞', L + 4, T1 - 14, hlc(['lamdot', 'ell'], pal, pal.fg), 13);
    // Lower panel: running means.
    const T2 = h * 0.6, B2 = h - 34, lx = (i) => L + (R - L) * Math.log(i + 1) / Math.log(NMC), Y2 = (v) => B2 - (B2 - T2) * (Math.max(1.5, Math.min(v, 8)) - 1.5) / 6.5;
    hline(ctx, pal, L, R, B2);
    hdash(ctx, hlc('Linf', pal, pal.muted), L, R, Y2(L_INF));
    SKEYS.forEach((k, j) => {
      const run = RUN[k].run; ctx.beginPath();
      for (let i = 0; i < n; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, lx(i), Y2(run[i]));
      ctx.strokeStyle = sColor(pal, j); ctx.lineWidth = 1.6; ctx.stroke();
    });
    [2, 4, 6, 8].forEach((v) => label(ctx, String(v), L - 8, Y2(v) + 4, pal.muted, 11, 'right'));
    label(ctx, 'L_∞', R - 4, Y2(L_INF) + 16, hlc('Linf', pal, pal.fg), 12.5, 'right');
    [1, 10, 100, 1000].forEach((i) => label(ctx, String(i), lx(i - 1), B2 + 15, pal.muted, 11, 'center'));
    label(ctx, `running mean of the first n one-sample estimates (log scale in n)`, L + 4, T2 - 12, pal.fg, 13);
    // Legend.
    SKEYS.forEach((k, i) => { const x = L + 8 + i * Math.min(190, (R - L) / 4); ctx.fillStyle = sColor(pal, i); ctx.fillRect(x, B1 + 26, 14, 3); label(ctx, SCHED[k].name, x + 20, B1 + 32, pal.fg, 12); });
    frame(ctx, pal, 0, 0, w, h);
    ro.innerHTML = `n = <b>${n}</b> · running means ` + SKEYS.map((k, i) => `<span style="color:${sColor(pal, i)}">${SCHED[k].name}</span> <b>${RUN[k].run[n - 1].toFixed(3)}</b>`).join(' · ') +
      `<br>exact mean of every schedule <b>${L_INF.toFixed(3)}</b> · standard deviation of one estimate: ` + SKEYS.map((k) => `${SCHED[k].name} <b>${RUN[k].sd.toFixed(2)}</b>`).join(' · ');
  }
  return { draw, active: (on) => r.setActive(on) };
}

// ---------- Step 5: weightings ----------
const WKEYS = Object.keys(WNAMES);
function weights(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let key = 'fm';
  viz.append(c.cv, controls(...choice(WNAMES, key, (k) => { key = k; draw(); })), ro,
    note('Each weight is scaled so that its loss for the exact denoiser is L<sub>∞</sub>. Flow matching: π uniform on the linear path. SD3: logit-normal π(t) with mean 0 and scale 1 on the linear path. EDM: log-normal density of the noise scale σ = e<sup>−λ/2</sup> with P<sub>mean</sub> = −1.2, P<sub>std</sub> = 1.2 and the EDM loss weight with σ<sub>data</sub> = 0.5.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), L = 52, R = w - 20, lams = Array.from(LAMS);
    const X = (l) => L + (R - L) * (l - LAM_MIN) / (LAM_MAX - LAM_MIN);
    // Upper: log10 w.
    const T1 = 34, B1 = h * 0.44, Y1 = (v) => B1 - (B1 - T1) * (Math.log10(Math.max(v, 1e-4)) + 4) / 7;
    hline(ctx, pal, L, R, B1);
    clip(ctx, L, T1 - 4, R - L, B1 - T1 + 4);
    WKEYS.forEach((k, i) => { if (k !== key) { ctx.globalAlpha = 0.35; curve(ctx, lams, X, Y1, (l, j) => W[k].w[j], pal.muted, 1.2); ctx.globalAlpha = 1; } });
    curve(ctx, lams, X, Y1, (l, j) => W[key].w[j], hlc(['w', 'pi'], pal, pal.acc), isHL('w', 'pi') ? 3.4 : 2.6);
    ctx.restore();
    [-4, -2, 0, 2].forEach((e) => label(ctx, `10${e < 0 ? '⁻' : ''}${'⁰¹²³⁴'[Math.abs(e)]}`, L - 8, Y1(10 ** e) + 4, pal.muted, 11, 'right'));
    label(ctx, `w(λ), log scale · bold: ${WNAMES[key]} · grey: the others`, L + 4, T1 - 14, hlc('w', pal, pal.fg), 13);
    // Lower: w l and the ELBO integrand l.
    const T2 = h * 0.56, B2 = h - 46, wl = W[key].w.map((v, j) => v * ELL[j]), ym = Math.max(0.6, ...wl) * 1.08;
    const Y2 = (v) => B2 - (B2 - T2) * v / ym;
    shade(ctx, lams, X, Y2, (l, j) => wl[j], B2, hlc('w', pal, pal.acc), 0.2);
    curve(ctx, lams, X, Y2, (l, j) => wl[j], hlc('w', pal, pal.acc), 2.4);
    curve(ctx, lams, X, Y2, (l, j) => ELL[j], hlc('ell', pal, pal.fg), isHL('ell') ? 2.6 : 1.6, [6, 5]);
    lamAxis(ctx, pal, X, B2, hlc('lam', pal, pal.muted));
    label(ctx, f2(ym / 1.08), L - 8, Y2(ym / 1.08) + 4, pal.muted, 11, 'right'); label(ctx, '0', L - 8, B2 + 4, pal.muted, 11, 'right');
    label(ctx, 'w(λ) ℓ(λ), area = loss · dashed: ℓ(λ), the ELBO integrand', L + 4, T2 - 12, hlc(['w', 'ell'], pal, pal.fg), 13);
    frame(ctx, pal, 0, 0, w, h);
    let im = 0; wl.forEach((v, j) => { if (v > wl[im]) im = j; });
    const ww = W[key].w, ratio = ww[0] / ww[NL];
    ro.innerHTML = `${WNAMES[key]} · w(−9) / w(9) = <b>${ratio >= 100 ? ratio.toExponential(1) : ratio.toFixed(2)}</b> · weighted integrand largest at λ = <b>${LAMS[im].toFixed(1)}</b> · ` +
      `share of the loss from λ &lt; 0: <b>${(100 * wl.slice(0, NL / 2).reduce((a, b) => a + b, 0) / wl.reduce((a, b) => a + b, 0)).toFixed(0)}%</b> (ELBO <b>${(100 * ELL.slice(0, NL / 2).reduce((a, b) => a + b, 0) / ELL.reduce((a, b) => a + b, 0)).toFixed(0)}%</b>)`;
  }
  return { draw };
}

// ---------- Step 6: -w' as a density of noise levels ----------
function monotone(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let key = 'fm';
  viz.append(c.cv, controls(...choice(WNAMES, key, (k) => { key = k; draw(); })), ro,
    note('Same weights as in step 5. −w′ is computed by central differences on the grid of 721 levels, and both sides of the identity by the trapezoid rule.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), L = 52, R = w - 20, lams = Array.from(LAMS), Wk = W[key];
    const X = (l) => L + (R - L) * (l - LAM_MIN) / (LAM_MAX - LAM_MIN);
    // Upper: w on a linear scale.
    const T1 = 34, B1 = h * 0.4, wm = Math.max(...Wk.w), Y1 = (v) => B1 - (B1 - T1) * v / wm;
    hline(ctx, pal, L, R, B1);
    shade(ctx, lams, X, Y1, (l, j) => Wk.w[j], B1, hlc('w', pal, pal.acc), 0.15);
    curve(ctx, lams, X, Y1, (l, j) => Wk.w[j], hlc('w', pal, pal.acc), isHL('w') ? 3.2 : 2.4);
    dot(ctx, X(LAM_MAX), Y1(Wk.end), 4.5, hlc('w', pal, pal.acc));
    label(ctx, f2(wm), L - 8, T1 + 4, pal.muted, 11, 'right'); label(ctx, '0', L - 8, B1 + 4, pal.muted, 11, 'right');
    label(ctx, `w(λ), linear scale · ${WNAMES[key]}`, L + 4, T1 - 14, hlc('w', pal, pal.fg), 13);
    // Lower: -w' with its negative part shaded.
    const T2 = h * 0.52, B2 = h - 46, dw = Wk.dw, dmax = Math.max(...dw.map(Math.abs)) * 1.05, mid = dw.some((v) => v < -1e-9) ? (T2 + B2) / 2 : B2 - 4;
    const Y2 = (v) => mid - (mid - T2) * v / dmax;
    hline(ctx, pal, L, R, mid);
    shade(ctx, lams, X, Y2, (l, j) => Math.max(dw[j], 0), mid, hlc('lprime', pal, pal.ok), 0.22);
    shade(ctx, lams, X, Y2, (l, j) => Math.min(dw[j], 0), mid, pal.bad, 0.35);
    curve(ctx, lams, X, Y2, (l, j) => dw[j], hlc('lprime', pal, pal.fg), 2);
    hline(ctx, pal, L, R, B2);
    for (let l = LAM_MIN; l <= LAM_MAX; l += 3) label(ctx, String(l), X(l), B2 + 15, pal.muted, 11, 'center');
    label(ctx, 'λ′', X(LAM_MAX), B2 + 30, hlc('lprime', pal, pal.muted), 13, 'right');
    label(ctx, '−w′(λ′): density of the noise level of the data', L + 4, T2 - 12, hlc(['lprime', 'w'], pal, pal.fg), 13);
    if (!Wk.monotone) label(ctx, 'red: −w′ < 0, not a density', R, T2 - 12, pal.bad, 12.5, 'right');
    frame(ctx, pal, 0, 0, w, h);
    ro.innerHTML = `${WNAMES[key]} · left side ∫ w ℓ dλ = <b>${Wk.lhs.toFixed(4)}</b> · right side = <b>${Wk.rhs.toFixed(4)}</b><br>` +
      `mass of −w′ where it is positive <b>${f3(Wk.pos)}</b> · where it is negative <b>${f3(Wk.neg)}</b> · w(λ<sub>max</sub>) = <b>${Wk.end.toFixed(4)}</b> · average of ELBOs: <b>${Wk.monotone ? 'yes' : 'no'}</b>`;
  }
  return { draw };
}

// ---------- Step 7: exact likelihood through the ODE ----------
const XP = Array.from({ length: 19 }, (_, i) => -3.15 + 6.5 * i / 18);
const odeAll = memo((K) => XP.map((x) => odeLogp(x, K)), 40);
const avgNLL = memo((K) => { let s = 0; for (const x of XQ) s -= odeLogp(x, K).logp / NQ; return s; }, 40);
const AVG_EXACT = (() => { let s = 0; for (const x of XQ) s -= Math.log(pdata(x)) / NQ; return s; })();
function exact(viz) {
  const c = makeCanvas(16 / 10), ro = readout();
  let K = 4, x1 = 0.9;
  const L = 52, R = () => c.w - 20, X = (x) => L + (R() - L) * (x - LO) / (HI - LO);
  track1(c, (px) => LO + (HI - LO) * (px - L) / (R() - L), (x) => { x1 = Math.max(-3.4, Math.min(3.6, x)); draw(); });
  viz.append(c.cv, controls(slider('K', { min: 1, max: 32, step: 1, value: K, fmt: (v) => String(v) }, (v) => { K = v; draw(); })), ro,
    note(`Exact marginal velocity of the 1D mixture on the linear path. Backward RK4 from t = 1 − 10<sup>−6</sup> to t = 0 on the pair (x, ∫∇·u dt); the divergence is a central difference of u. The average over the data uses ${NQ} stratified quantiles of p<sub>data</sub>.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), all = odeAll(K), pr = odeLogp(x1, K);
    // Upper: exact log p_data and the ODE estimates.
    const T1 = 34, B1 = h * 0.42, Y1 = (v) => B1 - (B1 - T1) * (Math.max(v, -7) + 7) / 7;
    hline(ctx, pal, L, R(), B1);
    curve(ctx, XS, X, Y1, (x) => Math.log(pdata(x)), hlc('pt', pal, pal.fg), isHL('pt') ? 3 : 2);
    all.forEach((r, i) => dot(ctx, X(XP[i]), Y1(r.logp), 4, hlc(['div', 'K'], pal, pal.acc)));
    ring(ctx, X(x1), Y1(pr.logp), 7.5, hlc('x1', pal, pal.acc), 2);
    [-6, -3, 0].forEach((v) => label(ctx, String(v), L - 8, Y1(v) + 4, pal.muted, 11, 'right'));
    label(ctx, `line: exact log p_{data}(x_1) · dots: ODE estimate with K = ${K}`, L + 4, T1 - 14, pal.fg, 13);
    // Lower: backward paths, t from 1 at the top to 0 at the bottom.
    const T2 = h * 0.54, B2 = h - 34, Y2 = (t) => B2 - (B2 - T2) * t;
    const pathCol = hlc(['u', 'xt'], pal, pal.muted);
    const drawPath = (p, col, lw) => { ctx.beginPath(); p.forEach((x, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, X(x), Y2(1 - k / K))); ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.stroke(); };
    clip(ctx, L, T2 - 6, R() - L, B2 - T2 + 12);
    ctx.globalAlpha = 0.55; all.forEach((r) => drawPath(r.path, pathCol, 1.1)); ctx.globalAlpha = 1;
    drawPath(pr.path, hlc('u', pal, pal.acc), 2.6);
    pr.path.forEach((x, k) => dot(ctx, X(x), Y2(1 - k / K), 2.6, hlc('K', pal, pal.acc)));
    ctx.restore();
    dot(ctx, X(x1), Y2(1), isHL('x1') ? 7 : 5.5, hlc('x1', pal, pal.fg));
    ring(ctx, X(pr.x0), Y2(0), isHL('x0') ? 8 : 6, hlc('x0', pal, pal.fg), 2);
    tag(ctx, pal, 'x_1', X(x1) + 10, Y2(1) + 4, hlc('x1', pal, pal.fg), 14);
    tag(ctx, pal, 'x_0', X(pr.x0) + 10, Y2(0) - 2, hlc('x0', pal, pal.fg), 14);
    label(ctx, 't = 1', L - 8, Y2(1) + 4, pal.muted, 11, 'right'); label(ctx, 't = 0', L - 8, Y2(0) + 4, pal.muted, 11, 'right');
    label(ctx, 'backward paths of the probability-flow ODE', L + 4, T2 - 12, hlc('u', pal, pal.fg), 13);
    frame(ctx, pal, 0, 0, w, h);
    const ex = Math.log(pdata(x1));
    ro.innerHTML = `x<sub>1</sub> = <b>${x1.toFixed(2)}</b> · x<sub>0</sub> = <b>${pr.x0.toFixed(3)}</b> · log p<sub>0</sub>(x<sub>0</sub>) = <b>${pr.lp0.toFixed(3)}</b> · ∫∇·u dt = <b>${pr.div.toFixed(3)}</b> · ` +
      `log p<sub>1</sub>(x<sub>1</sub>) by the ODE <b>${pr.logp.toFixed(3)}</b> · exact <b>${ex.toFixed(3)}</b><br>` +
      `average −log p over the data: ODE with K = ${K} <b>${avgNLL(K).toFixed(4)}</b> · exact at the same points <b>${AVG_EXACT.toFixed(4)}</b> · entropy of the data <b>${H_DATA.toFixed(4)}</b>`;
  }
  return { draw };
}

export const SCENES = { vae, kl, vdm, schedule, weights, monotone, exact };
