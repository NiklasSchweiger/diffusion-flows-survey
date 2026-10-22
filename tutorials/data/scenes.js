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
// Drawing helpers shared by step 1 and the preview of step 8. Colours are mixed in RGB.
const rgb = (c) => { const m = /^#([0-9a-f]{6})$/i.exec(c); return m ? [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16)) : [128, 128, 128]; };
const mix = (a, b, f) => { const A = rgb(a), B = rgb(b); return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * f).toString(16).padStart(2, '0')).join(''); };
// Colour map of the image: background at 0, the theme ramp towards 1. One table of 101 colours per theme.
const CMAP = {};
function cmap(pal, v) {
  const lut = CMAP[live.theme] ??= (() => {
    const st = live.theme === 'dark' ? [mix(pal.bg, pal.ramp[1], 0.07), mix(pal.bg, pal.ramp[1], 0.6), pal.ramp[2], pal.ramp[3], mix(pal.ramp[3], '#ffffff', 0.5)]
      : [mix(pal.bg, pal.ramp[0], 0.05), mix(pal.bg, pal.ramp[0], 0.45), pal.ramp[1], mix(pal.ramp[1], pal.fg, 0.6)];
    return Array.from({ length: 101 }, (_, i) => { const u = i / 100 * (st.length - 1), k = Math.min(st.length - 2, Math.floor(u)); return mix(st[k], st[k + 1], u - k); });
  })();
  return lut[Math.round(100 * Math.max(0, Math.min(1, v)))];
}
// A 16 x 16 grid of rounded tiles with a hairline gap; val(k) is the value of pixel k in [0, 1].
function tiles(ctx, pal, x0, y0, q, val) {
  ctx.fillStyle = cmap(pal, 0); ctx.beginPath(); ctx.roundRect(x0 - 3, y0 - 3, 16 * q + 6, 16 * q + 6, 6); ctx.fill();
  const gap = q > 8 ? 1 : 0, rr = q > 12 ? 2 : 0;
  for (let k = 0; k < 256; k++) {
    ctx.fillStyle = cmap(pal, val(k)); ctx.beginPath(); ctx.roundRect(x0 + (k % 16) * q + gap / 2, y0 + Math.floor(k / 16) * q + gap / 2, q - gap, q - gap, rr); ctx.fill();
  }
}
// A shaded sphere: a highlight towards the upper left and a darker rim.
function sphere(ctx, X, Y, r, col) {
  const g = ctx.createRadialGradient(X - 0.35 * r, Y - 0.4 * r, 0.1 * r, X, Y, r);
  g.addColorStop(0, mix(col, '#ffffff', 0.6)); g.addColorStop(0.5, col); g.addColorStop(1, mix(col, '#000000', 0.45));
  ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fillStyle = g; ctx.fill();
}
const ATOM_R = { C: 0.36, N: 0.34, O: 0.33 };   // drawn radii in Å, about a third of the van der Waals radii
const atomCol = (pal, el) => (el === 'C' ? mix(pal.muted, pal.fg, live.theme === 'dark' ? 0.15 : 0) : ELC(pal)[el]);
// Ball and stick: bonds split in the colours of their two atoms, double bonds as two thinner sticks.
function ballStick(ctx, pal, P, atoms, bonds, sc, { letters = true, ring = null } = {}) {
  const sw = Math.max(2.5, 0.15 * sc), out = live.theme === 'dark' ? '#000000' : mix(pal.fg, pal.bg, 0.35);
  ctx.lineCap = 'round';
  bonds.forEach(([i, j, ord]) => {
    const [X1, Y1] = P(atoms[i].p), [X2, Y2] = P(atoms[j].p), L = Math.hypot(X2 - X1, Y2 - Y1) || 1, nx = -(Y2 - Y1) / L, ny = (X2 - X1) / L;
    const offs = ord === 2 ? [-0.55 * sw, 0.55 * sw] : [0], wd = ord === 2 ? 0.62 * sw : sw;
    for (const o of offs) {
      const ax = X1 + o * nx, ay = Y1 + o * ny, bx = X2 + o * nx, by = Y2 + o * ny, mx = (ax + bx) / 2, my = (ay + by) / 2;
      ctx.strokeStyle = out; ctx.globalAlpha = 0.35; ctx.lineWidth = wd + 2; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.globalAlpha = 1;
      [[ax, ay, atoms[i].el], [bx, by, atoms[j].el]].forEach(([x, y, el]) => { ctx.strokeStyle = atomCol(pal, el); ctx.lineWidth = wd; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(mx, my); ctx.stroke(); });
    }
  });
  ctx.lineCap = 'butt';
  atoms.forEach(({ p, el }, k) => {
    const [X, Y] = P(p), r = ATOM_R[el] * sc;
    sphere(ctx, X, Y, r, atomCol(pal, el));
    if (letters && r > 9) label(ctx, el, X, Y + 0.36 * r, '#ffffff', Math.round(r * 0.95), 'center', 700);
    if (ring === k) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X, Y, r + 4, 0, 7); ctx.stroke(); ctx.lineWidth = 1; }
  });
}
// Colour along the chain, from the N- to the C-terminus.
const chainCol = (pal, u) => { const st = pal.ramp, x = Math.max(0, Math.min(1, u)) * (st.length - 1), k = Math.min(st.length - 2, Math.floor(x)); return mix(st[k], st[k + 1], x - k); };
// Catmull-Rom spline through pts, m samples per segment; each sample carries its residue coordinate u.
function spline(pts, m = 10) {
  const out = [], n = pts.length, at = (i) => pts[Math.max(0, Math.min(n - 1, i))];
  for (let i = 0; i < n - 1; i++) for (let j = 0; j < (i === n - 2 ? m + 1 : m); j++) {
    const s = j / m, [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    const f = (d) => 0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * s + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * s * s + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * s * s * s);
    out.push({ x: f(0), y: f(1), u: i + s });
  }
  return out;
}
// Cartoon of the backbone in screen coordinates: β-strands as flat arrows, the rest as a round tube.
// The guide points are the Cα positions, smoothed inside strands so that the pleat does not show.
function cartoon(ctx, pal, ca, strands, wS, wC, fade = 1) {
  const inS = (k) => strands.some(([a, b]) => k >= a && k <= b);
  let g = ca.map((p) => p.slice());
  for (let it = 0; it < 2; it++) g = g.map((p, k) => (k > 0 && k < ca.length - 1 && inS(k) ? [0, 1].map((d) => 0.25 * g[k - 1][d] + 0.5 * g[k][d] + 0.25 * g[k + 1][d]) : p));
  const sp = spline(g), n = ca.length;
  const half = (u) => {
    for (const [a, b] of strands) if (u >= a - 0.01 && u <= b) { const head = b - 0.9; return u < head ? wS : 1.8 * wS * (b - u) / 0.9; }
    return wC;
  };
  ctx.lineJoin = 'round';
  for (let i = 0; i < sp.length - 1; i++) {
    const A = sp[i], B = sp[i + 1], L = Math.hypot(B.x - A.x, B.y - A.y) || 1, nx = -(B.y - A.y) / L, ny = (B.x - A.x) / L;
    const ha = half(A.u), hb = half(B.u), col = mix(pal.bg, chainCol(pal, A.u / (n - 1)), fade);
    if (ha === wC && hb === wC) { ctx.strokeStyle = col; ctx.lineWidth = 2 * wC; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke(); ctx.lineCap = 'butt'; continue; }
    const q = [[A.x + ha * nx, A.y + ha * ny], [B.x + hb * nx, B.y + hb * ny], [B.x - hb * nx, B.y - hb * ny], [A.x - ha * nx, A.y - ha * ny]];
    ctx.beginPath(); q.forEach(([x, y], j) => (j ? ctx.lineTo : ctx.moveTo).call(ctx, x, y)); ctx.closePath();
    ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 0.8; ctx.stroke();
    ctx.strokeStyle = mix(col, '#000000', 0.4); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(...q[0]); ctx.lineTo(...q[1]); ctx.moveTo(...q[2]); ctx.lineTo(...q[3]);
    if (Math.abs(ha - hb) > 0.3 * wS) { ctx.moveTo(...q[0]); ctx.lineTo(...q[3]); }
    ctx.stroke();
  }
  return sp;
}
// The frame of one residue as a triangle N, Cα, C in screen coordinates.
function frameTri(ctx, pal, [N, A, C], col, strong = false, alpha = 0.8) {
  ctx.beginPath(); ctx.moveTo(...N); ctx.lineTo(...A); ctx.lineTo(...C); ctx.closePath();
  ctx.fillStyle = col; ctx.globalAlpha = strong ? 0.95 : alpha; ctx.fill(); ctx.globalAlpha = 1;
  ctx.strokeStyle = mix(col, live.theme === 'dark' ? '#ffffff' : '#000000', 0.35); ctx.lineWidth = 1; ctx.lineJoin = 'round'; ctx.stroke();
}
// A rounded token chip; state is 'on', 'mark' or 'masked'.
function chip(ctx, pal, X, Y, w, h, text, state, size = 14) {
  ctx.beginPath(); ctx.roundRect(X, Y, w, h, Math.min(10, h / 3));
  if (state === 'masked') {
    ctx.fillStyle = pal.panel; ctx.fill(); ctx.setLineDash([4, 3]); ctx.strokeStyle = pal.faint; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    label(ctx, text, X + w / 2, Y + h / 2 + size * 0.32, pal.faint, size * 0.8, 'center', 600);
    return;
  }
  const c = state === 'mark' ? pal.mark : pal.acc;
  ctx.fillStyle = mix(pal.bg, c, live.theme === 'dark' ? 0.16 : 0.1); ctx.fill();
  ctx.strokeStyle = mix(pal.bg, c, state === 'mark' ? 1 : 0.5); ctx.lineWidth = state === 'mark' ? 2 : 1.2; ctx.stroke();
  label(ctx, text, X + w / 2, Y + h / 2 + size * 0.34, state === 'mark' ? pal.mark : pal.fg, size, 'center', 600);
}
const OBJECTS = {
  image: { name: 'Image', d: 256, groups: 256, vec: () => IMG },
  molecule: { name: 'Molecule', d: 84, groups: 14, vec: molVector, cols: ['x', 'y', 'z', 'C', 'N', 'O'] },
  protein: { name: 'Protein', d: 96, groups: 16, vec: protVector, cols: ['x', 'y', 'z', 'ω₁', 'ω₂', 'ω₃'] },
  text: { name: 'Sentence', d: 6, groups: 6, vec: () => TEXT.ids, cols: ['id'] },
};
function objects(viz) {
  // Object and vector side by side, or stacked on narrow screens.
  const NARROW = 560, c = makeCanvas((w) => (w < NARROW ? 0.5 : 16 / 9), { maxH: 0.66 }), ro = readout();
  let kind = 'image', hov = null;
  const vecs = Object.fromEntries(Object.entries(OBJECTS).map(([k, o]) => [k, o.vec()]));
  const btns = Object.entries(OBJECTS).map(([k, o]) => button(o.name, () => { kind = k; hov = null; btns.forEach((b) => b.classList.toggle('on', b === btns[Object.keys(OBJECTS).indexOf(kind)])); draw(); }, k === kind ? 'on' : ''));
  let hit = () => null;
  c.cv.addEventListener('pointermove', (e) => { const r = c.cv.getBoundingClientRect(); hov = hit(e.clientX - r.left, e.clientY - r.top); draw(); });
  c.cv.addEventListener('pointerleave', () => { hov = null; draw(); });
  viz.append(c.cv, controls(...btns), ro, note('Caffeine: 3D conformer of PubChem CID 2519, heavy atoms only. Protein: residues 1 to 16 of ubiquitin, PDB entry 1UBQ, drawn in the plane of largest spread, with the two β-strands of its SHEET records as arrows and the colour running from the N- to the C-terminus. The image is drawn by code, and the token ids are illustrative.'));
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), o = OBJECTS[kind], v = vecs[kind];
    const one = w < NARROW, split = one ? Math.round(h * 0.4) : w * 0.46;
    const objR = one ? { x: 8, y: 30, w: w - 16, h: split - 36 } : { x: 16, y: 34, w: split - 32, h: h - 50 };
    const vecR = one ? { x: 8, y: split + 30, w: w - 16, h: h - split - 36 } : { x: split + 10, y: 34, w: w - split - 22, h: h - 50 };
    label(ctx, o.name, objR.x, 20, pal.fg, 14, 'left', 600);
    label(ctx, `x ∈ ℝ${sup(o.d)}`, vecR.x, vecR.y - 14, hlc(['x', 'd'], pal, pal.fg), 14, 'left', 600);
    label(ctx, `d = ${o.d}`, w - 12, vecR.y - 14, hlc('d', pal, pal.muted), 13, 'right');
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath();
    if (one) { ctx.moveTo(8, split + 0.5); ctx.lineTo(w - 8, split + 0.5); } else { ctx.moveTo(split, 30); ctx.lineTo(split, h - 12); }
    ctx.stroke();
    const hits = [];
    const mark = (g) => g === hov;
    // ---- left: the object ----
    if (kind === 'image') {
      const s = Math.floor(Math.min(objR.w, objR.h - 10) / 16), x0 = objR.x + (objR.w - 16 * s) / 2, y0 = objR.y + (objR.h - 16 * s) / 2;
      tiles(ctx, pal, x0, y0, s, (k) => IMG[k]);
      if (hov != null) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2.5; ctx.strokeRect(x0 + (hov % 16) * s - 1, y0 + Math.floor(hov / 16) * s - 1, s + 2, s + 2); ctx.lineWidth = 1; }
      hits.push((px, py) => (px >= x0 && px < x0 + 16 * s && py >= y0 && py < y0 + 16 * s ? 16 * Math.floor((py - y0) / s) + Math.floor((px - x0) / s) : null));
    } else if (kind === 'molecule' || kind === 'protein') {
      const pts = kind === 'molecule' ? MOL.atoms.map((a) => [a[0], a[1]]) : PROT2D.flat();
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const sc = Math.min((objR.w - 48) / (Math.max(...xs) - Math.min(...xs)), (objR.h - 48) / (Math.max(...ys) - Math.min(...ys)));
      const P = ([x, y]) => [objR.x + objR.w / 2 + sc * (x - cx), objR.y + objR.h / 2 - sc * (y - cy)];
      if (kind === 'molecule') {
        ballStick(ctx, pal, P, MOL.atoms.map((at) => ({ p: [at[0], at[1]], el: at[3] })), MOL.bonds, sc, { ring: hov });
        hits.push((px, py) => { let b = null, bd = 18; pts.forEach((p, k) => { const [X, Y] = P(p), dd = Math.hypot(px - X, py - Y); if (dd < bd) { bd = dd; b = k; } }); return b; });
      } else {
        const sp = cartoon(ctx, pal, PROT2D.map((r3) => P(r3[1])), PROT.strands, Math.max(5, 0.75 * sc), Math.max(2.5, 0.3 * sc));
        // One dot per residue on the centre line of the cartoon.
        const at = PROT2D.map((_, k) => sp[Math.min(sp.length - 1, Math.round(k * 10))]);
        at.forEach(({ x, y }, k) => { ctx.beginPath(); ctx.arc(x, y, mark(k) ? 4 : 2.2, 0, 7); ctx.fillStyle = mark(k) ? pal.mark : live.theme === 'dark' ? '#ffffff' : pal.bg; ctx.fill(); });
        if (hov != null) {
          const r3 = PROT2D[hov].map(P);
          frameTri(ctx, pal, r3, pal.mark, true);
          label(ctx, `${PROT.res[hov]} ${hov + 1}`, r3[1][0] + 12, r3[1][1] - 12, pal.mark, 13, 'left', 600);
        }
        // Terminus labels just beyond the two ends, along the direction of the chain.
        const end = (i, j, s) => { const L = Math.hypot(sp[i].x - sp[j].x, sp[i].y - sp[j].y) || 1; return [sp[i].x + s * (sp[i].x - sp[j].x) / L, sp[i].y + s * (sp[i].y - sp[j].y) / L + 4]; };
        label(ctx, 'N', ...end(0, 3, 14), pal.muted, 12, 'center', 600);
        label(ctx, 'C', ...end(sp.length - 1, sp.length - 4, 14), pal.muted, 12, 'center', 600);
        hits.push((px, py) => { let b = null, bd = 16; at.forEach(({ x, y }, k) => { const dd = Math.hypot(px - x, py - y); if (dd < bd) { bd = dd; b = k; } }); return b; });
      }
    } else {
      const n = TEXT.tokens.length, gp = 8, bw = Math.min(104, (objR.w - 10) / n - gp), ch = Math.min(42, Math.max(30, 0.75 * bw)), x0 = objR.x + (objR.w - n * (bw + gp) + gp) / 2, y0 = objR.y + objR.h / 2 - 30;
      TEXT.tokens.forEach((tk, k) => {
        const X = x0 + k * (bw + gp);
        chip(ctx, pal, X, y0, bw, ch, tk, mark(k) ? 'mark' : 'on', Math.min(14, bw / 4.4));
        label(ctx, String(TEXT.ids[k]), X + bw / 2, y0 + ch + 22, mark(k) ? pal.mark : pal.muted, 11.5, 'center');
      });
      hits.push((px, py) => (py >= y0 && py <= y0 + 70 && px >= x0 ? (Math.floor((px - x0) / (bw + gp)) < n ? Math.floor((px - x0) / (bw + gp)) : null) : null));
    }
    // ---- right: the vector ----
    const sel = isHL('xj') || isHL('x');
    if (kind === 'image') {
      // 256 cells in 4 lines of 64: four image rows per line, one after another.
      const per = 64, rows = 4, sw = vecR.w / per, sh = Math.min(3 * sw, (vecR.h - 70) / rows - 6), gap = 6, x0 = vecR.x, y0 = vecR.y + 30;
      const CX = (i) => x0 + (i % per) * sw, CY = (i) => y0 + Math.floor(i / per) * (sh + gap);
      for (let i = 0; i < 256; i++) {
        ctx.fillStyle = cmap(pal, IMG[i]); ctx.fillRect(CX(i), CY(i), sw - (sw > 4 ? 1 : 0), sh);
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

// ---------- Step 8: preview, from noise to data ----------
// One fixed noise draw per object, blended with the data point by x_t = t x1 + (1 - t) x0. Rotations
// and tokens are not blended: frames turn along the shortest arc and tokens are unmasked at random times.
const BOND = 1.6;   // Å, the distance below which two atoms are drawn bonded
function noiseDraw(seed) {
  const r = makeRng(seed), g = () => r.gauss();
  const ca = PROT2D.map((r3) => r3[1]), mc = MOL.atoms.map((a) => a.slice(0, 3));
  const cen = (P) => P[0].map((_, d) => P.reduce((s, p) => s + p[d], 0) / P.length);
  const rms = (P) => { const c = cen(P); return Math.sqrt(P.reduce((s, p) => s + p.reduce((a, v, d) => a + (v - c[d]) ** 2, 0), 0) / (P.length * P[0].length)); };
  return {
    img: Float64Array.from({ length: 256 }, g),
    molC: cen(mc), molS: rms(mc), mol: MOL.atoms.map(() => [g(), g(), g()]), typ: MOL.atoms.map(() => [g(), g(), g()]),
    protC: cen(ca), protS: rms(ca), prot: ca.map(() => [g(), g()]), ang: ca.map(() => Math.PI * (2 * r.uniform() - 1)),
    tok: TEXT.tokens.map(() => r.uniform()),
  };
}
function preview(viz) {
  // Two by two panels, or one column on narrow screens.
  const NARROW = 560, c = makeCanvas((w) => (w < NARROW ? 0.6 : 16 / 9), { maxH: 0.7 }), ro = readout();
  let t = 0, seed = 11, Z = noiseDraw(seed);
  const tSl = slider('t', { min: 0, max: 1, step: 0.01, value: 0 }, (v) => { t = v; draw(); });
  // Play runs t from its current value to 1 and stops there; at t = 1 it starts again from noise.
  const r = runner({ step: () => { tSl.set(Math.min(1, Math.round(100 * t + 1) / 100)); if (t >= 1) { r.playing = false; r.play.textContent = 'Play'; } }, done: () => t >= 1, draw: () => {}, interval: 30 });
  r.playing = false; r.play.textContent = 'Play';
  const play = r.play.onclick;
  r.play.onclick = () => { if (t >= 1 && !r.playing) tSl.set(0); play(); };
  viz.append(c.cv, controls(tSl, r.play, button('New noise', () => { seed += 1; Z = noiseDraw(seed); draw(); })), ro,
    note('No model is involved: each panel blends one fixed noise draw with the data point of step 1. The image is mapped to [−1, 1] first; the noise of the molecule and the protein has the spread of their coordinates. Bonds are drawn between atoms closer than 1.6 Å.'));
  function panel(ctx, pal, R, title, sub) {
    ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.strokeRect(R.x + 0.5, R.y + 0.5, R.w - 1, R.h - 1);
    label(ctx, title, R.x + 12, R.y + 20, pal.fg, 14, 'left', 600);
    label(ctx, sub, R.x + R.w - 12, R.y + 20, pal.muted, 11.5, 'right');
    return { x: R.x + 12, y: R.y + 30, w: R.w - 24, h: R.h - 40 };
  }
  // Maps 2D points into a box, with the scale fixed by the data so that the noise can spread beyond it.
  const fitter = (pts, B, pad) => {
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const sc = Math.min((B.w - 2 * pad) / (Math.max(...xs) - Math.min(...xs)), (B.h - 2 * pad) / (Math.max(...ys) - Math.min(...ys)));
    return Object.assign(([x, y]) => [B.x + B.w / 2 + sc * (x - cx), B.y + B.h / 2 - sc * (y - cy)], { sc });
  };
  function draw() {
    c.fit();
    const { ctx, w, h } = c, pal = palette(), a = t, s = 1 - t, g = 10;
    const one = w < NARROW, pw = one ? w : (w - g) / 2, ph = one ? (h - 3 * g) / 4 : (h - g) / 2;
    const rects = (one ? [[0, 0], [0, 1], [0, 2], [0, 3]] : [[0, 0], [1, 0], [0, 1], [1, 1]]).map(([i, j]) => ({ x: i * (pw + g), y: j * (ph + g), w: pw, h: ph }));
    const clip = (B, f) => { ctx.save(); ctx.beginPath(); ctx.rect(B.x - 10, B.y - 4, B.w + 20, B.h + 12); ctx.clip(); f(); ctx.restore(); };
    // ---- image ----
    let B = panel(ctx, pal, rects[0], 'Image', 'Gaussian noise on every pixel');
    clip(B, () => {
      const q = Math.floor(Math.min(B.w, B.h - 8) / 16), x0 = B.x + (B.w - 16 * q) / 2, y0 = B.y + (B.h - 16 * q) / 2;
      tiles(ctx, pal, x0, y0, q, (k) => (a * (2 * IMG[k] - 1) + s * Z.img[k] + 1) / 2);
      if (isHL('xt')) { ctx.strokeStyle = pal.mark; ctx.lineWidth = 2; ctx.strokeRect(x0 - 4, y0 - 4, 16 * q + 8, 16 * q + 8); ctx.lineWidth = 1; }
    });
    // ---- molecule ----
    B = panel(ctx, pal, rects[1], 'Molecule', 'coordinates and atom types blended');
    const mol = MOL.atoms.map((at, k) => {
      const p = [0, 1, 2].map((d) => a * at[d] + s * (Z.molC[d] + Z.molS * Z.mol[k][d]));
      const ty = MOL.types.map((e, j) => a * (e === at[3] ? 1 : 0) + s * Z.typ[k][j]);
      return { p, el: MOL.types[ty.indexOf(Math.max(...ty))] };
    });
    const bonds = [];
    for (let i = 0; i < mol.length; i++) for (let j = i + 1; j < mol.length; j++) if (Math.hypot(...mol[i].p.map((v, d) => v - mol[j].p[d])) < BOND) bonds.push([i, j, 1]);
    const nb = bonds.length;
    clip(B, () => {
      const P = fitter(MOL.atoms.map((at) => [at[0], at[1]]), B, 24);
      ballStick(ctx, pal, P, mol, bonds, P.sc);
    });
    // ---- protein ----
    B = panel(ctx, pal, rects[2], 'Protein', 'positions blended, frames rotated');
    clip(B, () => {
      const P = fitter(PROT2D.flat(), B, 18);
      const res = PROT2D.map(([n, ca, cc], k) => {
        const q = [0, 1].map((d) => a * ca[d] + s * (Z.protC[d] + Z.protS * Z.prot[k][d]));
        // The frame turns by the remaining fraction of the angle between the noise and the data frame.
        const th = s * Z.ang[k], cs = Math.cos(th), sn = Math.sin(th);
        const rot = (u) => [q[0] + cs * (u[0] - ca[0]) - sn * (u[1] - ca[1]), q[1] + sn * (u[0] - ca[0]) + cs * (u[1] - ca[1])];
        return [rot(n), q, rot(cc)].map(P);
      });
      // Near t = 1 the cartoon of step 1 fades in under the frames.
      const fade = Math.max(0, Math.min(1, (t - 0.85) / 0.15));
      if (fade > 0) cartoon(ctx, pal, res.map((r3) => r3[1]), PROT.strands, Math.max(4, 0.75 * P.sc), Math.max(2, 0.3 * P.sc), 0.6 * fade);
      const sp = spline(res.map((r3) => r3[1]), 8), n = res.length;
      ctx.lineCap = 'round'; ctx.lineWidth = 2;
      for (let i = 0; i < sp.length - 1; i++) { ctx.strokeStyle = chainCol(pal, sp[i].u / (n - 1)); ctx.beginPath(); ctx.moveTo(sp[i].x, sp[i].y); ctx.lineTo(sp[i + 1].x, sp[i + 1].y); ctx.stroke(); }
      ctx.lineCap = 'butt';
      res.forEach((r3, k) => frameTri(ctx, pal, r3, chainCol(pal, k / (n - 1)), false, 0.85 - 0.25 * fade));
    });
    // ---- sentence ----
    B = panel(ctx, pal, rects[3], 'Sentence', 'tokens masked, not blended');
    const shown = Z.tok.map((u) => u < t);
    {
      // Chips on top; below them a time axis with the reveal time of each token and the current t.
      const n = TEXT.tokens.length, gp = 8, bw = Math.min(96, (B.w - 10) / n - gp), x0 = B.x + (B.w - n * (bw + gp) + gp) / 2;
      const ch = Math.min(38, B.h * 0.22), y0 = B.y + Math.max(4, B.h * 0.12), ax0 = x0, ax1 = x0 + n * (bw + gp) - gp, ay = y0 + ch + Math.min(56, B.h * 0.36);
      const T = (u) => ax0 + (ax1 - ax0) * u;
      TEXT.tokens.forEach((tk, k) => {
        const X = x0 + k * (bw + gp), sz = Math.min(14, bw / 4.4);
        ctx.strokeStyle = shown[k] ? mix(pal.bg, pal.acc, 0.5) : pal.faint; ctx.globalAlpha = shown[k] ? 1 : 0.5; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X + bw / 2, y0 + ch); ctx.bezierCurveTo(X + bw / 2, ay - 16, T(Z.tok[k]), y0 + ch + 10, T(Z.tok[k]), ay); ctx.stroke(); ctx.globalAlpha = 1;
        chip(ctx, pal, X, y0, bw, ch, shown[k] ? tk : 'MASK', shown[k] ? 'on' : 'masked', sz);
      });
      ctx.strokeStyle = pal.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(ax0, ay + 0.5); ctx.lineTo(ax1, ay + 0.5); ctx.stroke();
      ctx.strokeStyle = mix(pal.bg, pal.acc, 0.8); ctx.lineWidth = 2.5; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(ax0, ay + 0.5); ctx.lineTo(T(t), ay + 0.5); ctx.stroke(); ctx.lineCap = 'butt';
      Z.tok.forEach((u, k) => { ctx.beginPath(); ctx.arc(T(u), ay + 0.5, 3.5, 0, 7); ctx.fillStyle = shown[k] ? pal.acc : pal.bg; ctx.fill(); ctx.strokeStyle = shown[k] ? pal.acc : pal.faint; ctx.lineWidth = 1.2; ctx.stroke(); });
      label(ctx, '0', ax0, ay + 17, pal.muted, 11, 'center'); label(ctx, '1', ax1, ay + 17, pal.muted, 11, 'center');
      label(ctx, 't', T(t), ay - 8, pal.acc, 12, 'center', 600);
      if (B.h > 150) label(ctx, 'Each token is revealed once t passes its own random time.', B.x + B.w / 2, Math.min(B.y + B.h - 4, ay + 38), pal.muted, 11.5, 'center');
    }
    ro.innerHTML = `t = <b>${t.toFixed(2)}</b> · α<sub>t</sub> = <b>${a.toFixed(2)}</b>, σ<sub>t</sub> = <b>${s.toFixed(2)}</b> · molecule: <b>${nb}</b> bonds within ${BOND} Å (caffeine has ${MOL.bonds.length}) · sentence: <b>${shown.filter(Boolean).length}</b> of ${shown.length} tokens unmasked`;
  }
  return { draw, active: (on) => r.setActive(on) };
}

export const SCENES = { objects, samples, interval, fit1d, gauss2d, onegauss, mixture, preview };
