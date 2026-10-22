// Live scenes of the §2.0 primer, one per step. Each factory gets the empty .viz element, fills it
// with canvases and controls, and returns { draw(), active?(on) }. Colours of hovered symbols come
// from live.hl through hlc() and isHL().
import { live, palette, makeCanvas, label, hlc, isHL, controls, slider, button, readout, note, runner } from '../lib/tutorial.js';
import { makeRng } from '../../animations/lib/rng.js';
import {
  MIX2, pdata, phi, SAMPLES, NMAX, probInterval, histogram, NFIT, avgLogLik, sampleMoments,
  POP, KL_GAUSS, H_DATA, mixDens, emRun, EM_ITERS, SAMPLES2, N2, FIT2, avgLogLik2, nParams,
  IMG, MOL, molVector, PROT, protVector, PROT2D, TEXT,
} from './sim.js';
import { p2 } from '../ebm-langevin/sim.js';

// ---------- Helpers ----------
const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const XS = lin(-4.5, 4.5, 451);
// Axes of a 1D density panel: x in [x0, x1] mapped to [L, R], density in [0, ymax] mapped to [bot, top].
function axes1(L, R, top, bot, x0 = -4.5, x1 = 4.5, ymax = 0.45) {
  return { X: (x) => L + (R - L) * (x - x0) / (x1 - x0), Y: (v) => bot - (bot - top) * Math.min(v, ymax * 1.08) / ymax, wx: (px) => x0 + (x1 - x0) * (px - L) / (R - L), L, R, top, bot, x0, x1 };
}
function baseline(ctx, pal, a, ticks = [-4, -2, 0, 2, 4]) {
  ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(a.L, a.bot + 0.5); ctx.lineTo(a.R, a.bot + 0.5); ctx.stroke();
  ticks.forEach((t) => label(ctx, String(t).replace('-', '−'), a.X(t), a.bot + 16, pal.muted, 11, 'center'));
}
function curve(ctx, a, f, color, width = 2, dash = null) {
  ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, a.X(x), a.Y(f(x))));
  if (dash) ctx.setLineDash(dash);
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.setLineDash([]);
}
const dataCurve = (ctx, pal, a) => curve(ctx, a, pdata, hlc('pdata', pal, pal.fg), isHL('pdata') ? 3 : 1.8, [6, 5]);
// Vertical jitter of the dot strips, fixed per sample.
const JIT = (() => { const r = makeRng(5); return Float64Array.from({ length: NMAX }, () => r.uniform()); })();
function dots(ctx, pal, a, n, y0, hgt, color, inside = null) {
  const r = n > 2000 ? 1.1 : n > 300 ? 1.6 : 2.4;
  ctx.globalAlpha = n > 2000 ? 0.35 : n > 300 ? 0.55 : 0.8;
  for (let i = 0; i < n; i++) {
    const x = SAMPLES[i];
    if (x < a.x0 || x > a.x1) continue;
    ctx.fillStyle = inside && inside(x) ? inside.color : color;
    ctx.beginPath(); ctx.arc(a.X(x), y0 + hgt * JIT[i], r, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;
}
const sup = (n) => String(n).replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);
const fmt = (v, k = 2) => (v.toFixed(k).replace(/^-(0\.0*)$/, '$1')).replace('-', '−');

// ---------- Step 1: four objects and their vectors ----------
const ELC = (pal) => ({ C: pal.muted, N: pal.ramp[1], O: pal.bad });
const OBJECTS = {
  image: { name: 'Image', d: 256, groups: 256, vec: () => IMG },
  molecule: { name: 'Molecule', d: 84, groups: 14, vec: molVector, cols: ['x', 'y', 'z', 'C', 'N', 'O'] },
  protein: { name: 'Protein', d: 96, groups: 16, vec: protVector, cols: ['x', 'y', 'z', 'ω₁', 'ω₂', 'ω₃'] },
  text: { name: 'Sentence', d: 6, groups: 6, vec: () => TEXT.ids, cols: ['id'] },
};
function objects(viz) {
  const c = makeCanvas(16 / 9, { maxH: 0.66 }), ro = readout();
  let kind = 'image', hov = null;
  const vecs = Object.fromEntries(Object.entries(OBJECTS).map(([k, o]) => [k, o.vec()]));
  const btns = Object.entries(OBJECTS).map(([k, o]) => button(o.name, () => { kind = k; hov = null; btns.forEach((b) => b.classList.toggle('on', b === btns[Object.keys(OBJECTS).indexOf(kind)])); draw(); }, k === kind ? 'on' : ''));
  let hit = () => null;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); hov = hit(e.clientX - r.left, e.clientY - r.top); draw(); });
  c.cv.addEventListener('pointerleave', () => { hov = null; draw(); });
  viz.append(c.cv, controls(...btns), ro, note('Caffeine: 3D conformer of PubChem CID 2519, heavy atoms only. Protein: residues 1 to 16 of ubiquitin, PDB entry 1UBQ, drawn in the plane of largest spread. The image is drawn by code, and the token ids are illustrative.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), o = OBJECTS[kind], v = vecs[kind];
    const split = w * 0.46, objR = { x: 16, y: 34, w: split - 32, h: h - 50 }, vecR = { x: split + 10, y: 34, w: w - split - 22, h: h - 50 };
    label(ctx, o.name, objR.x, 20, pal.fg, 14, 'left', 600);
    label(ctx, `x ∈ ℝ${sup(o.d)}`, vecR.x, 20, hlc(['x', 'd'], pal, pal.fg), 14, 'left', 600);
    label(ctx, `d = ${o.d}`, w - 12, 20, hlc('d', pal, pal.muted), 13, 'right');
    ctx.strokeStyle = pal.line; ctx.beginPath(); ctx.moveTo(split, 30); ctx.lineTo(split, h - 12); ctx.stroke();
    const hits = [];
    const mark = (g) => g === hov;
    // ---- left: the object ----
    if (kind === 'image') {
      const s = Math.floor(Math.min(objR.w, objR.h) / 16), x0 = objR.x + (objR.w - 16 * s) / 2, y0 = objR.y + (objR.h - 16 * s) / 2;
      for (let j = 0; j < 16; j++) for (let i = 0; i < 16; i++) {
        const val = IMG[16 * j + i], g = Math.round(255 * (live.theme === 'dark' ? val : 1 - val));
        ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x0 + i * s, y0 + j * s, s, s);
      }
      ctx.strokeStyle = pal.line; ctx.strokeRect(x0 - 0.5, y0 - 0.5, 16 * s + 1, 16 * s + 1);
      if (hov != null) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2.5; ctx.strokeRect(x0 + (hov % 16) * s, y0 + Math.floor(hov / 16) * s, s, s); ctx.lineWidth = 1; }
      hits.push((px, py) => (px >= x0 && px < x0 + 16 * s && py >= y0 && py < y0 + 16 * s ? 16 * Math.floor((py - y0) / s) + Math.floor((px - x0) / s) : null));
    } else if (kind === 'molecule' || kind === 'protein') {
      const pts = kind === 'molecule' ? MOL.atoms.map((a) => [a[0], a[1]]) : PROT2D.flat();
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const sc = Math.min((objR.w - 40) / (Math.max(...xs) - Math.min(...xs)), (objR.h - 40) / (Math.max(...ys) - Math.min(...ys)));
      const P = ([x, y]) => [objR.x + objR.w / 2 + sc * (x - cx), objR.y + objR.h / 2 - sc * (y - cy)];
      if (kind === 'molecule') {
        MOL.bonds.forEach(([i, j, ord]) => {
          const [X1, Y1] = P(pts[i]), [X2, Y2] = P(pts[j]), L = Math.hypot(X2 - X1, Y2 - Y1), nx = -(Y2 - Y1) / L * 3, ny = (X2 - X1) / L * 3;
          ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.6;
          for (const o2 of ord === 2 ? [-1, 1] : [0]) { ctx.beginPath(); ctx.moveTo(X1 + o2 * nx, Y1 + o2 * ny); ctx.lineTo(X2 + o2 * nx, Y2 + o2 * ny); ctx.stroke(); }
        });
        const col = ELC(pal);
        MOL.atoms.forEach((a, k) => {
          const [X, Y] = P(pts[k]), r = mark(k) ? 15 : 12;
          ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = pal.bg; ctx.fill();
          ctx.strokeStyle = mark(k) ? pal.mark : col[a[3]]; ctx.lineWidth = mark(k) ? 3 : 2; ctx.stroke();
          label(ctx, a[3], X, Y + 5, mark(k) ? pal.mark : col[a[3]], 13, 'center', 600);
        });
        hits.push((px, py) => { let b = null, bd = 18; pts.forEach((p, k) => { const [X, Y] = P(p), dd = Math.hypot(px - X, py - Y); if (dd < bd) { bd = dd; b = k; } }); return b; });
      } else {
        ctx.beginPath(); pts.forEach((p, i) => { const [X, Y] = P(p); (i ? ctx.lineTo : ctx.moveTo).call(ctx, X, Y); });
        ctx.strokeStyle = pal.muted; ctx.lineWidth = 1.6; ctx.stroke();
        PROT2D.forEach((r3, k) => {
          // The frame of a residue: Cα, with short strokes to its N and C.
          const [Xa, Ya] = P(r3[1]);
          ctx.beginPath(); ctx.arc(Xa, Ya, mark(k) ? 8 : 5.5, 0, 7); ctx.fillStyle = mark(k) ? pal.mark : pal.acc; ctx.fill();
          if (mark(k)) {
            [[r3[0], pal.ramp[1]], [r3[2], pal.bad]].forEach(([p, cl]) => { const [X, Y] = P(p); ctx.strokeStyle = cl; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(Xa, Ya); ctx.lineTo(X, Y); ctx.stroke(); });
            label(ctx, `${PROT.res[k]} ${k + 1}`, Xa + 12, Ya - 10, pal.mark, 13, 'left', 600);
          }
        });
        label(ctx, 'N-terminus', P(PROT2D[0][0])[0], P(PROT2D[0][0])[1] + 18, pal.muted, 11, 'center');
        hits.push((px, py) => { let b = null, bd = 16; PROT2D.forEach((r3, k) => { const [X, Y] = P(r3[1]), dd = Math.hypot(px - X, py - Y); if (dd < bd) { bd = dd; b = k; } }); return b; });
      }
    } else {
      const n = TEXT.tokens.length, bw = Math.min(110, (objR.w - 10) / n - 8), x0 = objR.x + (objR.w - n * (bw + 8)) / 2, y0 = objR.y + objR.h / 2 - 20;
      TEXT.tokens.forEach((t, k) => {
        const X = x0 + k * (bw + 8);
        ctx.strokeStyle = mark(k) ? pal.mark : pal.line; ctx.lineWidth = mark(k) ? 2.5 : 1; ctx.strokeRect(X, y0, bw, 40);
        label(ctx, t, X + bw / 2, y0 + 25, mark(k) ? pal.mark : pal.fg, 14, 'center');
      });
      hits.push((px, py) => (py >= y0 && py <= y0 + 40 && px >= x0 ? (Math.floor((px - x0) / (bw + 8)) < n ? Math.floor((px - x0) / (bw + 8)) : null) : null));
    }
    // ---- right: the vector ----
    const sel = isHL('xj') || isHL('x');
    if (kind === 'image') {
      // 256 cells in 4 lines of 64: four image rows per line, one after another.
      const per = 64, rows = 4, sw = vecR.w / per, sh = Math.min(3 * sw, (vecR.h - 70) / rows - 6), gap = 6, x0 = vecR.x, y0 = vecR.y + 30;
      const CX = (i) => x0 + (i % per) * sw, CY = (i) => y0 + Math.floor(i / per) * (sh + gap);
      for (let i = 0; i < 256; i++) {
        const val = IMG[i], g = Math.round(255 * (live.theme === 'dark' ? val : 1 - val));
        ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(CX(i), CY(i), sw - (sw > 4 ? 1 : 0), sh);
      }
      if (hov != null) {
        // The image row of the hovered pixel is 16 consecutive entries.
        const r0 = 16 * Math.floor(hov / 16);
        ctx.strokeStyle = pal.acc; ctx.lineWidth = 1.5; ctx.strokeRect(CX(r0) - 1, CY(r0) - 1, 16 * sw + 1, sh + 2);
        ctx.strokeStyle = pal.mark; ctx.lineWidth = 2.5; ctx.strokeRect(CX(hov) - 1, CY(hov) - 2, sw + 1, sh + 4); ctx.lineWidth = 1;
      }
      if (sel) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(x0 - 3, y0 - 3, per * sw + 5, rows * (sh + gap) - gap + 6); ctx.lineWidth = 1; }
      label(ctx, 'x_{1}', x0, y0 - 8, pal.muted, 11.5);
      label(ctx, 'x_{64}', x0 + per * sw, y0 - 8, pal.muted, 11.5, 'right');
      label(ctx, 'x_{256}', x0 + per * sw, CY(255) + sh + 16, pal.muted, 11.5, 'right');
      label(ctx, 'Four image rows per line, 16 entries each.', x0, CY(255) + sh + 40, pal.muted, 11.5);
      hits.push((px, py) => { const j = Math.floor((py - y0) / (sh + gap)), i = Math.floor((px - x0) / sw); return j >= 0 && j < rows && py - y0 - j * (sh + gap) <= sh && i >= 0 && i < per ? j * per + i : null; });
    } else {
      const cols = o.cols, n = o.groups, rowH = Math.min(22, (vecR.h - 30) / n), cw = Math.min(62, (vecR.w - 90) / cols.length), x0 = vecR.x + 80, y0 = vecR.y + 26;
      const names = kind === 'molecule' ? MOL.atoms.map((a, k) => `${a[3]}${k + 1}`) : kind === 'protein' ? PROT.res.map((r, k) => `${r} ${k + 1}`) : TEXT.tokens;
      cols.forEach((cl, j) => label(ctx, cl, x0 + cw * (j + 0.5), y0 - 8, pal.muted, 11.5, 'center'));
      const font = Math.max(9, Math.min(12.5, rowH * 0.62));
      for (let k = 0; k < n; k++) {
        const Y = y0 + k * rowH;
        if (mark(k)) { ctx.fillStyle = pal.mark; ctx.globalAlpha = 0.16; ctx.fillRect(vecR.x, Y, 80 + cw * cols.length, rowH); ctx.globalAlpha = 1; }
        label(ctx, names[k], vecR.x + 2, Y + rowH * 0.72, mark(k) ? pal.mark : pal.muted, font);
        cols.forEach((_, j) => {
          const val = v[k * cols.length + j], txt = kind === 'text' ? String(val) : kind === 'molecule' && j >= 3 ? String(val) : fmt(val);
          label(ctx, txt, x0 + cw * (j + 0.85), Y + rowH * 0.72, mark(k) || sel ? pal.mark : (kind === 'molecule' && j >= 3 && val === 0 ? pal.faint : pal.fg), font, 'right');
        });
      }
      if (kind === 'protein') label(ctx, 'Cα position in Å and the frame rotation as axis × angle', vecR.x, y0 + n * rowH + 16, pal.muted, 11.5);
      if (kind === 'molecule') label(ctx, 'Coordinates in Å and a one-hot code of the element', vecR.x, y0 + n * rowH + 16, pal.muted, 11.5);
      hits.push((px, py) => (px >= vecR.x && py >= y0 && py < y0 + n * rowH ? Math.floor((py - y0) / rowH) : null));
    }
    hit = (px, py) => { for (const f of hits) { const r = f(px, py); if (r != null) return r; } return null; };
    // ---- readout ----
    const per = kind === 'image' ? 1 : o.cols.length;
    if (hov == null) ro.innerHTML = `${o.name} · d = <b>${o.d}</b> numbers · move the pointer over the object or the vector`;
    else {
      const i0 = hov * per + 1, vals = v.slice(hov * per, hov * per + per).map((x) => (Number.isInteger(x) ? String(x) : fmt(x)));
      const what = kind === 'image' ? `pixel in row ${Math.floor(hov / 16) + 1}, column ${(hov % 16) + 1}` : kind === 'molecule' ? `atom ${hov + 1} (${MOL.atoms[hov][3]})` : kind === 'protein' ? `residue ${hov + 1} (${PROT.res[hov]})` : `token “${TEXT.tokens[hov]}”`;
      ro.innerHTML = `${what} · entries <b>x<sub>${i0}</sub>${per > 1 ? ` … x<sub>${i0 + per - 1}</sub>` : ''}</b> = <b>${vals.join(', ')}</b> · d = <b>${o.d}</b>`;
    }
  }
  return { draw };
}

// ---------- Step 2: samples and their histogram ----------
function samples(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let n = 300, bw = 0.3;
  viz.append(c.cv, controls(
    slider('n', { min: 1, max: 4, step: 0.01, value: Math.log10(300), map: (v) => Math.round(10 ** v), fmt: (v) => String(v) }, (v) => { n = v; draw(); }),
    slider('h', { min: 0.05, max: 1, step: 0.01, value: 0.3 }, (v) => { bw = v; draw(); }),
  ), ro, note('The first n draws of one fixed stream from the three-component mixture, so increasing n only adds points. The dashed curve is the exact density.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), a = axes1(44, w - 18, 30, h - 92);
    const hist = histogram(n, -4.5, 4.5, bw), nb = hist.length, wpx = (a.R - a.L) / nb;
    ctx.fillStyle = hlc(['phat', 'h'], pal, pal.acc); ctx.globalAlpha = isHL('phat', 'h') ? 0.55 : 0.35;
    hist.forEach((v, j) => { const Y = a.Y(v); ctx.fillRect(a.L + j * wpx + 0.5, Y, wpx - 1, a.bot - Y); });
    ctx.globalAlpha = 1;
    dataCurve(ctx, pal, a);
    baseline(ctx, pal, a);
    dots(ctx, pal, a, n, h - 62, 40, hlc(['xi', 'n', 'phat'], pal, pal.fg));
    label(ctx, 'samples', a.L, h - 38, pal.muted, 11.5);
    label(ctx, 'p_{data}, dashed', a.R, 22, hlc('pdata', pal, pal.muted), 13, 'right');
    label(ctx, 'histogram of the samples', a.L, 22, hlc(['phat', 'h'], pal, pal.acc), 13);
    let err = 0; hist.forEach((v, j) => { const x = -4.5 + (j + 0.5) * bw; err += Math.abs(v - pdata(x)) * bw; });
    ro.innerHTML = `n = <b>${n}</b> · h = <b>${bw.toFixed(2)}</b> · ${nb} bins · area between histogram and p<sub>data</sub> ≈ <b>${err.toFixed(2)}</b>`;
  }
  return { draw };
}

// ---------- Step 3: probability as area, and a density above 1 ----------
function interval(viz) {
  const c = makeCanvas(16 / 9), ro = readout(), N = 1000;
  let A = -1, B = 0.8, sg = 0.5;
  viz.append(c.cv, controls(
    slider('a', { min: -4, max: 4, step: 0.05, value: A }, (v) => { A = v; draw(); }),
    slider('b', { min: -4, max: 4, step: 0.05, value: B }, (v) => { B = v; draw(); }),
    slider('sigma', { min: 0.05, max: 1.5, step: 0.01, value: sg }, (v) => { sg = v; draw(); }),
  ), ro, note('Left: the exact probability of (a, b) and the fraction of 1000 samples in it. Right: N(0, σ²), whose area is 1 for every σ.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), split = w * 0.64;
    const lo = Math.min(A, B), hi = Math.max(A, B), a = axes1(40, split - 20, 54, h - 80);
    ctx.beginPath(); ctx.moveTo(a.X(lo), a.bot);
    lin(lo, hi, 200).forEach((x) => ctx.lineTo(a.X(x), a.Y(pdata(x))));
    ctx.lineTo(a.X(hi), a.bot); ctx.closePath();
    ctx.fillStyle = hlc('P', pal, pal.acc); ctx.globalAlpha = isHL('P') ? 0.5 : 0.3; ctx.fill(); ctx.globalAlpha = 1;
    [[A, 'a'], [B, 'b']].forEach(([x, id]) => { ctx.strokeStyle = hlc(id, pal, pal.acc); ctx.lineWidth = isHL(id) ? 3 : 1.5; ctx.beginPath(); ctx.moveTo(a.X(x), a.top); ctx.lineTo(a.X(x), a.bot); ctx.stroke(); label(ctx, id, a.X(x), a.top - 6, hlc(id, pal, pal.acc), 14, 'center', 600); });
    dataCurve(ctx, pal, a);
    baseline(ctx, pal, a);
    const inside = (x) => x > lo && x < hi; inside.color = pal.acc;
    dots(ctx, pal, a, N, h - 52, 30, pal.muted, inside);
    let cnt = 0; for (let i = 0; i < N; i++) if (inside(SAMPLES[i])) cnt++;
    // Right panel: a Gaussian of adjustable width on its own vertical scale.
    const b2 = { L: split + 24, R: w - 16, top: 40, bot: h - 80 }, ymax = 4.2;
    const X2 = (x) => b2.L + (b2.R - b2.L) * (x + 2) / 4, Y2 = (v) => b2.bot - (b2.bot - b2.top) * Math.min(v, ymax) / ymax;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(b2.L, b2.bot + 0.5); ctx.lineTo(b2.R, b2.bot + 0.5); ctx.moveTo(b2.L - 0.5, b2.top); ctx.lineTo(b2.L - 0.5, b2.bot); ctx.stroke();
    [1, 2, 3, 4].forEach((v) => { ctx.setLineDash(v === 1 ? [4, 4] : [1, 5]); ctx.strokeStyle = v === 1 ? pal.bad : pal.line; ctx.beginPath(); ctx.moveTo(b2.L, Y2(v)); ctx.lineTo(b2.R, Y2(v)); ctx.stroke(); ctx.setLineDash([]); label(ctx, String(v), b2.L - 6, Y2(v) + 4, v === 1 ? pal.bad : pal.muted, 11, 'right'); });
    ctx.beginPath(); ctx.moveTo(X2(-2), b2.bot);
    lin(-2, 2, 300).forEach((x) => ctx.lineTo(X2(x), Y2(phi(x, 0, sg))));
    ctx.lineTo(X2(2), b2.bot); ctx.closePath();
    ctx.fillStyle = pal.acc2; ctx.globalAlpha = 0.22; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = hlc('sigma', pal, pal.acc2); ctx.lineWidth = 2; ctx.stroke();
    const peak = phi(0, 0, sg);
    label(ctx, `N(0, σ²), area 1`, b2.L, 22, pal.acc2, 13);
    // Keep the peak label off the dashed line at density 1.
    const py = peak > ymax ? Y2(ymax) + 18 : Math.abs(Y2(peak) - Y2(1)) < 18 ? Y2(1) + 16 : Y2(peak) - 8;
    label(ctx, `peak ${peak.toFixed(2)}`, X2(0) + 8, py, peak > 1 ? pal.bad : pal.muted, 12.5);
    label(ctx, 'density 1', b2.R, Y2(1) - 6, pal.bad, 11, 'right');
    [-2, 0, 2].forEach((t) => label(ctx, String(t).replace('-', '−'), X2(t), b2.bot + 16, pal.muted, 11, 'center'));
    label(ctx, 'p_{data}, dashed; shaded area = P(a < x < b)', a.L, 22, hlc('pdata', pal, pal.muted), 13);
    ro.innerHTML = `P(a &lt; x &lt; b) = <b>${probInterval(lo, hi).toFixed(3)}</b> exact · <b>${cnt}</b> of ${N} samples = <b>${(cnt / N).toFixed(3)}</b> · Gaussian with σ = ${sg.toFixed(2)}: peak density <b>${peak.toFixed(2)}</b>`;
  }
  return { draw };
}

// ---------- Step 4: Gaussian fit by maximum likelihood ----------
const MU = [-2.5, 2.5], SG = [0.3, 3];
const LLGRID = (() => {
  const nx = 70, ny = 60, g = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) g[j * nx + i] = avgLogLik(MU[0] + (MU[1] - MU[0]) * i / (nx - 1), Math.exp(Math.log(SG[0]) + Math.log(SG[1] / SG[0]) * j / (ny - 1)));
  return { nx, ny, g };
})();
const FIT1 = sampleMoments();
const llCache = new Map();
function llImage(theme) {
  if (llCache.has(theme)) return llCache.get(theme);
  const { nx, ny, g } = LLGRID, off = document.createElement('canvas'); off.width = nx; off.height = ny;
  const o = off.getContext('2d'), im = o.createImageData(nx, ny), top = avgLogLik(FIT1.m, FIT1.s), bot = top - 2.5;
  const rgb = theme === 'dark' ? [123, 108, 255] : [91, 69, 224];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    // Rows go up in σ, so image row 0 is the largest σ.
    const v = g[(ny - 1 - j) * nx + i], a = Math.max(0, Math.min(1, (v - bot) / (top - bot))), q = 4 * (j * nx + i);
    im.data[q] = rgb[0]; im.data[q + 1] = rgb[1]; im.data[q + 2] = rgb[2]; im.data[q + 3] = Math.round(255 * (0.05 + 0.85 * a ** 1.5));
  }
  o.putImageData(im, 0, 0); llCache.set(theme, off);
  return off;
}
function fit1d(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let mu = 1, sg = 0.6, map = null;
  const muSl = slider('mu', { min: -2.5, max: 2.5, step: 0.01, value: mu }, (v) => { mu = v; draw(); });
  const sgSl = slider('sigma', { min: Math.log(0.3), max: Math.log(3), step: 0.005, value: Math.log(sg), map: Math.exp }, (v) => { sg = v; draw(); });
  c.cv.addEventListener('click', (e) => {
    if (!map) return;
    const r = c.cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < map.x || px > map.x + map.w || py < map.y || py > map.y + map.h) return;
    muSl.set(MU[0] + (MU[1] - MU[0]) * (px - map.x) / map.w);
    sgSl.set(Math.log(SG[1]) - Math.log(SG[1] / SG[0]) * (py - map.y) / map.h);
  });
  viz.append(c.cv, controls(muSl, sgSl, button('Fit', () => { muSl.set(FIT1.m); sgSl.set(Math.log(FIT1.s)); })), ro,
    note(`${NFIT} samples from the mixture of step 2. Click the map to set μ and σ there; σ runs on a log scale from 0.3 to 3.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), split = w * 0.58;
    const a = axes1(40, split - 16, 34, h - 70, -4.5, 4.5, 0.75);
    const hist = histogram(NFIT, -4.5, 4.5, 0.3), wpx = (a.R - a.L) / hist.length;
    ctx.fillStyle = pal.faint; ctx.globalAlpha = 0.5; hist.forEach((v, j) => { const Y = a.Y(v); ctx.fillRect(a.L + j * wpx + 0.5, Y, wpx - 1, a.bot - Y); }); ctx.globalAlpha = 1;
    curve(ctx, a, (x) => phi(x, mu, sg), hlc(['Nx', 'mu', 'sigma'], pal, pal.acc), isHL('Nx') ? 3.4 : 2.4);
    baseline(ctx, pal, a);
    // Each sample as a tick whose height is its log-density under the model, clipped below.
    for (let i = 0; i < NFIT; i++) {
      const x = SAMPLES[i], ld = Math.log(phi(x, mu, sg)), t = Math.max(0, Math.min(1, (ld + 8) / 8));
      ctx.strokeStyle = t < 0.35 ? pal.bad : hlc('xi', pal, pal.fg); ctx.globalAlpha = 0.35 + 0.5 * t;
      ctx.beginPath(); ctx.moveTo(a.X(x), h - 30); ctx.lineTo(a.X(x), h - 30 - 18 * t - 2); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    label(ctx, 'samples, tick height = log-density under the model (red: below −5)', a.L, h - 12, pal.muted, 11);
    label(ctx, 'N(x; μ, σ²) and a histogram of the samples', a.L, 22, hlc('Nx', pal, pal.acc), 13);
    // Right: the map of ℓ(μ, σ).
    map = { x: split + 40, y: 34, w: w - split - 56, h: h - 80 };
    ctx.imageSmoothingEnabled = true; ctx.drawImage(llImage(live.theme), map.x, map.y, map.w, map.h);
    ctx.strokeStyle = hlc('ell', pal, pal.line); ctx.lineWidth = isHL('ell') ? 2.5 : 1; ctx.strokeRect(map.x - 0.5, map.y - 0.5, map.w + 1, map.h + 1);
    const MX = (m) => map.x + map.w * (m - MU[0]) / (MU[1] - MU[0]), SY = (s) => map.y + map.h * (Math.log(SG[1]) - Math.log(s)) / Math.log(SG[1] / SG[0]);
    ctx.strokeStyle = pal.fg; ctx.lineWidth = 2;
    const fx = MX(FIT1.m), fy = SY(FIT1.s);
    ctx.beginPath(); ctx.moveTo(fx - 7, fy - 7); ctx.lineTo(fx + 7, fy + 7); ctx.moveTo(fx + 7, fy - 7); ctx.lineTo(fx - 7, fy + 7); ctx.stroke();
    ctx.beginPath(); ctx.arc(MX(mu), SY(sg), 6, 0, 7); ctx.fillStyle = hlc(['mu', 'sigma'], pal, pal.acc); ctx.fill();
    label(ctx, 'ℓ(μ, σ), stronger colour is larger', map.x, 22, hlc('ell', pal, pal.fg), 13);
    label(ctx, 'μ', map.x + map.w / 2, map.y + map.h + 30, hlc('mu', pal, pal.muted), 13, 'center');
    [-2, 0, 2].forEach((t) => label(ctx, String(t).replace('-', '−'), MX(t), map.y + map.h + 15, pal.muted, 11, 'center'));
    [0.5, 1, 2].forEach((t) => label(ctx, String(t), map.x - 6, SY(t) + 4, pal.muted, 11, 'right'));
    label(ctx, 'σ', map.x - 26, map.y + map.h / 2, hlc('sigma', pal, pal.muted), 13, 'center');
    const ll = avgLogLik(mu, sg), best = avgLogLik(FIT1.m, FIT1.s);
    ro.innerHTML = `ℓ(μ, σ) = <b>${fmt(ll, 3)}</b> · maximum <b>${fmt(best, 3)}</b> at μ̂ = <b>${fmt(FIT1.m)}</b>, σ̂ = <b>${fmt(FIT1.s)}</b> (the sample mean and standard deviation)`;
  }
  return { draw };
}

// ---------- Step 5: 2D Gaussian with ellipses ----------
const heat2 = new Map();
function gauss2d(viz) {
  const c = makeCanvas(4 / 3, { maxH: 0.6 }), ro = readout();
  let s1 = 0.7, s2 = 0.7, r = 0;
  const sl1 = slider('s1', { min: 0.3, max: 2, step: 0.01, value: s1 }, (v) => { s1 = v; draw(); });
  const sl2 = slider('s2', { min: 0.3, max: 2, step: 0.01, value: s2 }, (v) => { s2 = v; draw(); });
  const slr = slider('rho', { min: -0.95, max: 0.95, step: 0.01, value: r }, (v) => { r = v; draw(); });
  viz.append(c.cv, controls(sl1, sl2, slr, button('Fit', () => { sl1.set(FIT2.s1); sl2.set(FIT2.s2); slr.set(FIT2.r); })), ro,
    note(`${N2} samples from the two-dimensional mixture of §2.1, shaded by its exact density. The centre is fixed at the sample mean.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), bx = 3.4, by = 2.6, s = Math.min(w / (2 * bx), h / (2 * by)), cx = w / 2, cy = h / 2;
    const P = (x, y) => [cx + s * x, cy - s * y];
    const k = `${w}x${h}|${live.theme}`;
    if (!heat2.has(k)) {
      heat2.clear();
      const res = 3, W = Math.ceil(w / res), H = Math.ceil(h / res), off = document.createElement('canvas'); off.width = W; off.height = H;
      const g = off.getContext('2d'), im = g.createImageData(W, H), pm = p2(...MIX2.means[0], MIX2.means, MIX2.std, MIX2.w);
      const rgb = live.theme === 'dark' ? [123, 108, 255] : [91, 69, 224];
      for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
        const x = (i * res - cx) / s, y = (cy - j * res) / s, a = Math.pow(Math.min(1, p2(x, y, MIX2.means, MIX2.std, MIX2.w) / pm), 0.7), q = 4 * (j * W + i);
        im.data[q] = rgb[0]; im.data[q + 1] = rgb[1]; im.data[q + 2] = rgb[2]; im.data[q + 3] = Math.round(255 * a * (live.theme === 'dark' ? 0.4 : 0.25));
      }
      g.putImageData(im, 0, 0); heat2.set(k, off);
    }
    ctx.globalAlpha = isHL('pdata') ? 1 : 0.8; ctx.drawImage(heat2.get(k), 0, 0, w, h); ctx.globalAlpha = 1;
    ctx.fillStyle = hlc('xi', pal, pal.fg); ctx.globalAlpha = 0.6;
    SAMPLES2.forEach(([x, y]) => { const [X, Y] = P(x, y); ctx.beginPath(); ctx.arc(X, Y, 2, 0, 7); ctx.fill(); });
    ctx.globalAlpha = 1;
    // Ellipse axes from the eigen-decomposition of Σ.
    const A = s1 * s1, B = r * s1 * s2, C = s2 * s2, m = (A + C) / 2, dlt = Math.sqrt(((A - C) / 2) ** 2 + B * B);
    const l1 = m + dlt, l2 = Math.max(1e-9, m - dlt), ang = 0.5 * Math.atan2(2 * B, A - C), [MX, MY] = P(...FIT2.m);
    [1, 2].forEach((kk) => {
      ctx.beginPath(); ctx.ellipse(MX, MY, kk * s * Math.sqrt(l1), kk * s * Math.sqrt(l2), -ang, 0, 7);
      ctx.strokeStyle = hlc(['Sigma', 's1', 's2', 'rho', 'Nx'], pal, pal.acc); ctx.lineWidth = kk === 1 ? 2.6 : 1.6; ctx.setLineDash(kk === 1 ? [] : [6, 4]); ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.beginPath(); ctx.arc(MX, MY, 5, 0, 7); ctx.fillStyle = hlc('mu', pal, pal.acc); ctx.fill();
    label(ctx, '1σ (solid) and 2σ (dashed) contours', 12, 22, pal.muted, 13);
    label(ctx, `parameters: d + d(d+1)/2 = ${nParams(2)}`, w - 12, 22, hlc(['Sigma', 'd'], pal, pal.muted), 13, 'right');
    const ll = avgLogLik2({ m: FIT2.m, s1, s2, r }), best = avgLogLik2(FIT2);
    ro.innerHTML = `Σ = [[<b>${fmt(A)}</b>, <b>${fmt(B)}</b>], [<b>${fmt(B)}</b>, <b>${fmt(C)}</b>]] · average log-likelihood <b>${fmt(ll, 3)}</b> · maximum <b>${fmt(best, 3)}</b> at σ₁ = ${fmt(FIT2.s1)}, σ₂ = ${fmt(FIT2.s2)}, ρ = ${fmt(FIT2.r)}`;
  }
  return { draw };
}

// ---------- Step 6: the best single Gaussian and its KL ----------
function onegauss(viz) {
  const c = makeCanvas(16 / 9), ro = readout();
  let probe = POP.m;
  const q = (x) => phi(x, POP.m, POP.s);
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); probe = Math.max(-4.5, Math.min(4.5, -4.5 + 9 * (e.clientX - r.left - 44) / (r.width - 62))); draw(); });
  viz.append(c.cv, ro, note('The Gaussian with the mean and variance of the data distribution, which is the maximum-likelihood fit in the limit of infinitely many samples. KL and entropy are computed by quadrature on [−7, 7].'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette();
    const a = axes1(44, w - 18, 34, h * 0.56), b = axes1(44, w - 18, h * 0.68, h - 30, -4.5, 4.5, 0.32);
    // Upper panel: where q exceeds p_data, shaded.
    ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, a.X(x), a.Y(Math.max(q(x), pdata(x)))));
    [...XS].reverse().forEach((x) => ctx.lineTo(a.X(x), a.Y(pdata(x)))); ctx.closePath();
    ctx.fillStyle = pal.bad; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
    dataCurve(ctx, pal, a);
    curve(ctx, a, q, hlc(['q', 'Nx'], pal, pal.acc), isHL('q', 'Nx') ? 3.4 : 2.4);
    baseline(ctx, pal, a, []);
    label(ctx, 'p_{data}, dashed', a.L, 22, hlc('pdata', pal, pal.muted), 13);
    label(ctx, `q = N(${fmt(POP.m)}, ${POP.s.toFixed(2)}²), bold; red where q > p_{data}`, a.R, 22, hlc('q', pal, pal.acc), 13, 'right');
    // Lower panel: the KL integrand, which can be negative where q > p_data.
    const f = (x) => { const p = pdata(x); return p > 1e-12 ? p * Math.log(p / q(x)) : 0; }, Yb = (v) => b.bot - (b.bot - b.top) * (v + 0.1) / 0.42;
    ctx.strokeStyle = pal.line; ctx.beginPath(); ctx.moveTo(b.L, Yb(0)); ctx.lineTo(b.R, Yb(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(b.X(-4.5), Yb(0)); XS.forEach((x) => ctx.lineTo(b.X(x), Yb(f(x)))); ctx.lineTo(b.X(4.5), Yb(0)); ctx.closePath();
    ctx.fillStyle = hlc('KL', pal, pal.acc2); ctx.globalAlpha = isHL('KL') ? 0.45 : 0.25; ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); XS.forEach((x, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, b.X(x), Yb(f(x)))); ctx.strokeStyle = pal.acc2; ctx.lineWidth = 1.8; ctx.stroke();
    label(ctx, 'p_{data} log(p_{data} / q); its signed area is KL(p_{data} ‖ q)', b.L, b.top - 8, hlc('KL', pal, pal.acc2), 13);
    [-4, -2, 0, 2, 4].forEach((t) => label(ctx, String(t).replace('-', '−'), b.X(t), b.bot + 16, pal.muted, 11, 'center'));
    // Probe.
    const X = a.X(probe);
    ctx.strokeStyle = pal.muted; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(X, a.top); ctx.lineTo(X, b.bot); ctx.stroke(); ctx.setLineDash([]);
    [[pdata(probe), pal.fg], [q(probe), pal.acc]].forEach(([v, cl]) => { ctx.beginPath(); ctx.arc(X, a.Y(v), 4.5, 0, 7); ctx.fillStyle = cl; ctx.fill(); });
    ro.innerHTML = `x = <b>${fmt(probe)}</b> · p<sub>data</sub>(x) = <b>${pdata(probe).toFixed(3)}</b> · q(x) = <b>${q(probe).toFixed(3)}</b> · KL(p<sub>data</sub> ‖ q) = <b>${KL_GAUSS.toFixed(3)}</b> nats · H(p<sub>data</sub>) = <b>${H_DATA.toFixed(3)}</b> nats`;
  }
  return { draw };
}

// ---------- Step 7: EM for a mixture ----------
function mixture(viz) {
  const c = makeCanvas(16 / 9), ro = readout(), runs = {};
  let K = 3, it = 0;
  const get = () => (runs[K] ??= emRun(K));
  const itSl = slider('it', { min: 0, max: EM_ITERS, step: 1, value: 0, fmt: (v) => String(v) }, (v) => { it = v; draw(); });
  const r = runner({ step: () => { itSl.set(it + 1); }, done: () => it >= EM_ITERS, draw: () => {}, interval: 120 });
  r.playing = false; r.play.textContent = 'Play';
  const play = r.play.onclick;
  // At the end, Play restarts from the seeded start.
  r.play.onclick = () => { if (it >= EM_ITERS) { itSl.set(0); if (r.playing) { r.kick(); return; } } play(); };
  viz.append(c.cv, controls(
    slider('K', { min: 1, max: 5, step: 1, value: K, fmt: (v) => String(v) }, (v) => { K = v; itSl.set(0); }),
    itSl, r.play,
  ), ro, note(`EM on the first ${NFIT} samples from a seeded start: K distinct samples as means, the overall spread as every width, equal weights. Widths are kept above 0.05. KL is measured against the exact data distribution, by quadrature.`));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), R = get(), p = R[Math.min(it, R.length - 1)];
    const a = axes1(44, w - 18, 34, h * 0.6, -4.5, 4.5, 0.5);
    const hist = histogram(NFIT, -4.5, 4.5, 0.25), wpx = (a.R - a.L) / hist.length;
    ctx.fillStyle = pal.faint; ctx.globalAlpha = 0.45; hist.forEach((v, j) => { const Y = a.Y(v); ctx.fillRect(a.L + j * wpx + 0.5, Y, wpx - 1, a.bot - Y); }); ctx.globalAlpha = 1;
    dataCurve(ctx, pal, a);
    p.mu.forEach((m, k) => curve(ctx, a, (x) => p.w[k] * phi(x, m, p.sd[k]), hlc(['pik', 'muk', 'sk'], pal, pal.ramp[k % 4]), 1.4));
    curve(ctx, a, mixDens(p), hlc('q', pal, pal.acc), isHL('q') ? 3.4 : 2.4);
    baseline(ctx, pal, a);
    label(ctx, 'p_{data}, dashed · mixture q, bold · its weighted components, thin', a.L, 22, pal.muted, 13);
    // Lower panel: KL per iteration.
    const b = { L: 44, R: w - 18, top: h * 0.72, bot: h - 22 }, kmax = 0.3;
    const X = (i) => b.L + (b.R - b.L) * i / EM_ITERS, Y = (v) => b.bot - (b.bot - b.top) * Math.min(v, kmax) / kmax;
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(b.L, b.bot + 0.5); ctx.lineTo(b.R, b.bot + 0.5); ctx.stroke();
    ctx.setLineDash([4, 4]); ctx.strokeStyle = pal.muted; ctx.beginPath(); ctx.moveTo(b.L, Y(KL_GAUSS)); ctx.lineTo(b.R, Y(KL_GAUSS)); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, `best single Gaussian, ${KL_GAUSS.toFixed(2)}`, b.R, Y(KL_GAUSS) - 5, pal.muted, 11, 'right');
    ctx.beginPath(); R.forEach((q, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(q.kl))); ctx.strokeStyle = hlc('KL', pal, pal.acc2); ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.arc(X(it), Y(p.kl), 5, 0, 7); ctx.fillStyle = hlc('it', pal, pal.acc2); ctx.fill();
    label(ctx, 'KL(p_{data} ‖ q) per EM iteration m', b.L + 14, b.top - 8, hlc('KL', pal, pal.acc2), 13);
    label(ctx, '0', b.L - 6, b.bot + 4, pal.muted, 11, 'right');
    const list = p.mu.map((m, k) => `π = ${p.w[k].toFixed(2)}, μ = ${fmt(m)}, σ = ${p.sd[k].toFixed(2)}`).join(' | ');
    ro.innerHTML = `K = <b>${K}</b> · m = <b>${it}</b> · ${list} · ℓ = <b>${fmt(p.ll, 3)}</b> · KL = <b>${p.kl.toFixed(3)}</b> nats`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

export const SCENES = { objects, samples, interval, fit1d, gauss2d, onegauss, mixture };
