// A3 · From pixels to tokens: latent encoding, patchify, and joint text-image attention (DiT / MMDiT).
// The image is procedural. The encoder is illustrative: 8× average pooling into a 4-channel latent (three colour
// channels and one brightness-gradient channel), where a real autoencoder learns its channels. Patchify and the token
// counts are exact. The attention weights come from a hand-set similarity between colours, not from a trained model;
// they show what one row of the joint attention matrix means.
import { panelHeader } from '../lib/scene.js';
import { seg, ease, window_, text, paragraph, lerp, clamp } from '../lib/theme.js';

const W = 1600, H = 900;
const IMG = 96, F = 8, LAT = IMG / F, CH = 4, P = 2, G = LAT / P, NI = G * G;   // 96 px, 12×12 latent, 6×6 tokens
const WORDS = ['a', 'red', 'sun', 'above', 'green', 'hills', 'under', 'blue', 'sky'];
const NT = WORDS.length, NTOK = NT + NI;
const WORD_COL = {
  red: [1, 0.25, 0.15], sun: [1, 0.45, 0.1], green: [0.2, 0.75, 0.25], hills: [0.25, 0.6, 0.2],
  blue: [0.3, 0.55, 0.95], sky: [0.45, 0.7, 1],
};
const BETA = 7;

const T = {
  setup: [0, 7],
  latent: [7, 18], latMove: [7.6, 10], latShow: [10, 12],
  patch: [18, 31], grid: [18.6, 20.4], fly: [21, 26.5],
  attn: [31, 50], words: [31.6, 33.6], matrix: [34, 36], q1: [36.5, 41], q2: [41, 45.5], q3: [45.5, 50],
};
const QUERIES = [{ span: T.q1, w: 2 }, { span: T.q2, w: 5 }, { span: T.q3, w: 8 }];

const STILLS = [
  { t: 6.5, name: 'image', label: 'Pixels' },
  { t: 17.5, name: 'latent', label: 'Latent, 8× smaller per side', crop: { x: 600, y: 0, w: 990, h: 500 } },
  { t: 30.5, name: 'tokens', label: '2×2 patches become tokens' },
  { t: 39.5, name: 'attention', label: 'Joint attention, query "sun"' },
];

export default {
  id: 'patchify-attend',
  title: 'Patchify and joint text-image attention',
  duration: 50,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Image' },
    { t: T.latent[0], label: 'I · Latent' },
    { t: T.patch[0], label: 'II · Patchify' },
    { t: T.attn[0], label: 'III · Joint attention' },
  ],
  stills: STILLS,
  crop: { x: 600, y: 0, w: 990, h: 720 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    ctx.fillStyle = th.bg; ctx.fillRect(0, 0, W, H);

    // Image: centered and large in the setup, then moves left and shrinks.
    const m = ease.inOut(seg(t, ...T.latMove));
    const imgA = 1 - seg(t, T.patch[0], T.patch[0] + 1);
    if (imgA > 0) {
      const size = lerp(384, 288, m), x0 = lerp(1060 - 192, 640, m), y0 = lerp(258, 156, m);
      drawPixels(ctx, st.img, IMG, x0, y0, size / IMG, imgA);
      text(ctx, `image  ${IMG} × ${IMG} × 3`, x0, y0 + size + 30, { size: 17, color: th.muted, alpha: imgA, font: 'mono' });
    }
    // Latent grid: appears at the right, then slides into the image's place.
    const la = seg(t, ...T.latShow);
    const slide = ease.inOut(seg(t, T.patch[0], T.patch[0] + 1.2));
    const LX = lerp(1150, 640, slide), LY = 156, LS = 288 / LAT;
    if (la > 0) {
      if (slide < 0.5) {
        arrowRight(ctx, th, 960, 300, 1120, 300, la * (1 - 2 * slide));
        text(ctx, 'encoder', 1040, 282, { size: 17, color: th.muted, align: 'center', alpha: la * (1 - 2 * slide) });
      }
      drawLatent(ctx, th, st, LX, LY, LS, la, t);
      text(ctx, `latent  ${LAT} × ${LAT} × ${CH}`, LX, LY + 288 + 30, { size: 17, color: th.muted, alpha: la, font: 'mono' });
    }
    if (t >= T.patch[0]) drawTokens(ctx, th, st, t, LX, LY, LS);
    if (t >= T.attn[0]) drawAttention(ctx, th, st, t);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Data ----------
function pixel(u, v) {
  // Sky gradient, a sun, and two hills. u to the right, v downward, both in [0, 1].
  let c = [lerp(0.33, 0.78, v / 0.7), lerp(0.52, 0.86, v / 0.7), lerp(0.93, 1, v / 0.7)];
  if (Math.hypot(u - 0.7, v - 0.27) < 0.13) c = [1, 0.42, 0.12];
  if (v > 0.6 + 0.08 * Math.sin(6 * u)) c = [0.22, 0.64, 0.27];
  if (v > 0.76 + 0.05 * Math.sin(4 * u + 1)) c = [0.13, 0.47, 0.2];
  return c;
}

function buildData() {
  const img = [];
  for (let j = 0; j < IMG; j++) for (let i = 0; i < IMG; i++) {
    // 2×2 supersampling for soft edges.
    const s = [0, 0, 0];
    for (const [a, b] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
      const c = pixel((i + a) / IMG, (j + b) / IMG);
      s[0] += c[0] / 4; s[1] += c[1] / 4; s[2] += c[2] / 4;
    }
    img.push(s);
  }
  // Illustrative encoder: 8×8 average pooling; channel 4 is the vertical brightness gradient.
  const lat = [];
  for (let j = 0; j < LAT; j++) for (let i = 0; i < LAT; i++) {
    const s = [0, 0, 0, 0];
    for (let b = 0; b < F; b++) for (let a = 0; a < F; a++) {
      const c = img[(j * F + b) * IMG + i * F + a], lum = (c[0] + c[1] + c[2]) / 3;
      s[0] += c[0]; s[1] += c[1]; s[2] += c[2]; s[3] += (b < F / 2 ? 1 : -1) * lum;
    }
    lat.push(s.map((v) => v / (F * F)));
  }
  // Tokens: 2×2 latent patches in raster order. Colour feature = mean of the patch's colour channels.
  const tokens = [];
  for (let gj = 0; gj < G; gj++) for (let gi = 0; gi < G; gi++) {
    const cells = [], c = [0, 0, 0];
    for (let b = 0; b < P; b++) for (let a = 0; a < P; a++) {
      const l = lat[(gj * P + b) * LAT + gi * P + a];
      cells.push(l);
      for (let d = 0; d < 3; d++) c[d] += l[d] / (P * P);
    }
    tokens.push({ gi, gj, cells, col: c });
  }
  // Features for attention: colour minus neutral grey, normalized. Function words are neutral.
  const feat = (c) => { const v = c.map((x) => x - 0.55), n = Math.hypot(...v); return n < 1e-6 ? [0, 0, 0] : v.map((x) => x / n); };
  const F_ = [...WORDS.map((w) => feat(WORD_COL[w] ?? [0.55, 0.55, 0.55])), ...tokens.map((k) => feat(k.col))];
  const A = F_.map((q) => {
    const s = F_.map((k) => BETA * (q[0] * k[0] + q[1] * k[1] + q[2] * k[2]));
    const mx = Math.max(...s), e = s.map((v) => Math.exp(v - mx)), z = e.reduce((a, b) => a + b);
    return e.map((v) => v / z);
  });
  return { img, lat, tokens, A };
}

// ---------- Drawing ----------
const rgb = (c, a = 1) => `rgba(${Math.round(255 * clamp(c[0], 0, 1))},${Math.round(255 * clamp(c[1], 0, 1))},${Math.round(255 * clamp(c[2], 0, 1))},${a})`;

function drawPixels(ctx, img, n, x0, y0, cell, a) {
  ctx.save(); ctx.globalAlpha = a;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    ctx.fillStyle = rgb(img[j * n + i]);
    ctx.fillRect(x0 + i * cell, y0 + j * cell, cell + 0.6, cell + 0.6);
  }
  ctx.restore();
}

function latentColor(l) {
  // Show the colour channels slightly desaturated and shade by the fourth channel.
  const g = (l[0] + l[1] + l[2]) / 3, k = 1 + 1.4 * l[3];
  return [0, 1, 2].map((d) => (0.7 * l[d] + 0.3 * g) * k);
}

function drawLatent(ctx, th, st, x0, y0, cs, a, t) {
  ctx.save(); ctx.globalAlpha = a;
  for (let j = 0; j < LAT; j++) for (let i = 0; i < LAT; i++) {
    ctx.fillStyle = rgb(latentColor(st.lat[j * LAT + i]));
    ctx.fillRect(x0 + i * cs + 1, y0 + j * cs + 1, cs - 2, cs - 2);
  }
  // Patch grid in Act II and later.
  const ga = seg(t, ...T.grid);
  if (ga > 0) {
    ctx.globalAlpha = a * ga; ctx.strokeStyle = th.fg; ctx.lineWidth = 2.2;
    for (let k = 0; k <= G; k++) {
      ctx.beginPath(); ctx.moveTo(x0 + k * P * cs, y0); ctx.lineTo(x0 + k * P * cs, y0 + LAT * cs); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0, y0 + k * P * cs); ctx.lineTo(x0 + LAT * cs, y0 + k * P * cs); ctx.stroke();
    }
  }
  ctx.restore();
}

// Token row: text tokens (Act III) followed by image tokens, at y = ROW_Y.
const ROW_Y = 640, TOK = 18, GAP = 2.4, ROW_X = 640;
const slotX = (i) => ROW_X + i * (TOK + GAP);

function drawTokens(ctx, th, st, t, LX, LY, LS) {
  const fly = seg(t, ...T.fly);
  if (fly <= 0) return;
  const wa = seg(t, ...T.words);
  const q = activeQuery(t);
  st.tokens.forEach((tk, i) => {
    // Stagger the flights in raster order.
    const u = ease.inOut(clamp(fly * 1.6 - (i / NI) * 0.6, 0, 1));
    const sx = LX + tk.gi * P * LS, sy = LY + tk.gj * P * LS, sz = P * LS;
    const x = lerp(sx, slotX(NT + i), u), y = lerp(sy, ROW_Y, u), sz2 = lerp(sz, TOK, u);
    const sub = sz2 / P;
    ctx.save();
    ctx.globalAlpha = 0.95;
    tk.cells.forEach((l, c) => { ctx.fillStyle = rgb(latentColor(l)); ctx.fillRect(x + (c % P) * sub, y + Math.floor(c / P) * sub, sub, sub); });
    if (q && q.a > 0) {
      // Highlight by the attention weight from the query word.
      const w = st.A[q.w][NT + i] / Math.max(...st.A[q.w].slice(NT));
      ctx.globalAlpha = q.a * (1 - w) * 0.75; ctx.fillStyle = th.bg; ctx.fillRect(x, y, sz2, sz2);
    }
    ctx.restore();
  });
  const la = seg(fly, 0.85, 1);
  text(ctx, `${NI} image tokens, each ${P}×${P}×${CH} = ${P * P * CH} numbers`, slotX(NT), ROW_Y + 50, { size: 16, color: th.muted, alpha: la * (1 - wa) });
  if (wa > 0) {
    WORDS.forEach((w, i) => {
      const x = slotX(i), y = ROW_Y - 40 * (1 - ease.out(wa));
      const hl = q && q.w === i ? q.a : 0;
      ctx.save(); ctx.globalAlpha = wa;
      ctx.fillStyle = hl > 0 ? th.ramp[3] : th.accent; ctx.globalAlpha = wa * (0.55 + 0.45 * hl);
      ctx.fillRect(x, y, TOK, TOK);
      ctx.restore();
      ctx.save(); ctx.translate(x + TOK / 2 + 5, y - 8); ctx.rotate(-Math.PI / 2);
      text(ctx, w, 0, 0, { size: 15, weight: hl > 0.5 ? 700 : 500, color: hl > 0.5 ? th.ramp[3] : th.fg, alpha: wa });
      ctx.restore();
    });
    text(ctx, 'text tokens', slotX(0), ROW_Y + 50, { size: 16, color: th.accent, alpha: wa });
    text(ctx, 'image tokens', slotX(NT), ROW_Y + 50, { size: 16, color: th.muted, alpha: wa });
    text(ctx, 'one sequence', slotX(NTOK - 1) + TOK, ROW_Y + 50, { size: 16, color: th.fg, alpha: wa, align: 'right' });
  }
}

function activeQuery(t) {
  for (const qq of QUERIES) {
    const a = window_(t, qq.span[0], qq.span[1], 0.4);
    if (a > 0) return { w: qq.w, a: t > T.q3[0] && qq === QUERIES[2] ? Math.min(1, seg(t, qq.span[0], qq.span[0] + 0.4)) : a };
  }
  return null;
}

// Attention matrix (rows: queries, columns: keys), text block first. Overlay on the latent grid for the active query.
function drawAttention(ctx, th, st, t) {
  const ma = seg(t, ...T.matrix);
  const q = activeQuery(t);
  if (ma > 0) {
    const X0 = 1100, Y0 = 130, C = 360 / NTOK;
    const max = 0.25;
    ctx.save();
    for (let i = 0; i < NTOK; i++) for (let j = 0; j < NTOK; j++) {
      const v = clamp(st.A[i][j] / max, 0, 1) ** 0.6;
      ctx.globalAlpha = ma * (0.06 + 0.94 * v);
      ctx.fillStyle = th.accent;
      ctx.fillRect(X0 + j * C, Y0 + i * C, C - 0.5, C - 0.5);
    }
    ctx.globalAlpha = ma; ctx.strokeStyle = th.fg; ctx.lineWidth = 1.4;
    ctx.strokeRect(X0, Y0, NTOK * C, NTOK * C);
    ctx.beginPath(); ctx.moveTo(X0 + NT * C, Y0); ctx.lineTo(X0 + NT * C, Y0 + NTOK * C);
    ctx.moveTo(X0, Y0 + NT * C); ctx.lineTo(X0 + NTOK * C, Y0 + NT * C); ctx.stroke();
    if (q) {
      ctx.globalAlpha = ma * q.a; ctx.strokeStyle = th.ramp[3]; ctx.lineWidth = 2.6;
      ctx.strokeRect(X0 - 2, Y0 + q.w * C - 1, NTOK * C + 4, C + 2);
    }
    ctx.restore();
    text(ctx, 'text', X0 + (NT * C) / 2, Y0 - 12, { size: 15, color: th.muted, align: 'center', alpha: ma });
    text(ctx, 'image', X0 + NT * C + (NI * C) / 2, Y0 - 12, { size: 15, color: th.muted, align: 'center', alpha: ma });
    text(ctx, 'keys →', X0 + NTOK * C, Y0 - 12, { size: 15, color: th.muted, align: 'right', alpha: ma });
    ctx.save(); ctx.translate(X0 - 14, Y0 + NTOK * C); ctx.rotate(-Math.PI / 2);
    text(ctx, 'queries →', 0, 0, { size: 15, color: th.muted, alpha: ma });
    ctx.restore();
    text(ctx, 'attention weights softmax(QKᵀ/√d)', X0, Y0 + NTOK * C + 30, { size: 16, color: th.muted, alpha: ma });
  }
  // Overlay on the latent grid: weight from the query word to each image patch.
  if (q && q.a > 0) {
    const LX = 640, LY = 156, LS = 288 / LAT, row = st.A[q.w].slice(NT), mx = Math.max(...row);
    ctx.save();
    st.tokens.forEach((tk, i) => {
      ctx.globalAlpha = q.a * 0.8 * (1 - row[i] / mx);
      ctx.fillStyle = th.bg;
      ctx.fillRect(LX + tk.gi * P * LS, LY + tk.gj * P * LS, P * LS, P * LS);
    });
    ctx.restore();
    text(ctx, `where "${WORDS[q.w]}" attends`, LX, LY - 18, { size: 18, weight: 600, color: th.ramp[3], alpha: q.a });
  }
}

function arrowRight(ctx, th, x1, y1, x2, y2, a) {
  if (a <= 0) return;
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = th.muted; ctx.fillStyle = th.muted; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - 10, y2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 12, y2 - 7); ctx.lineTo(x2 - 12, y2 + 7); ctx.closePath(); ctx.fill();
  ctx.restore();
}

// ---------- Text panel ----------
const COPY = [
  { span: T.setup, act: 'An image',
    body: 'Modern image and video generators do not denoise pixels directly. The image is first compressed by an autoencoder, then cut into patches that a transformer treats as tokens, in the same sequence as the text prompt.',
    eq: '' },
  { span: T.latent, act: 'Act I · The latent space',
    body: 'A pretrained autoencoder maps the image to a latent grid that is 8 times smaller per side, with a few channels per cell. Diffusion or flow matching then runs entirely on these latents, and the decoder maps the result back to pixels.',
    eq: 'z = E(x),  x̂ = D(z)' },
  { span: T.patch, act: 'Act II · Patchify',
    body: 'A diffusion transformer cuts the latent into non-overlapping 2×2 patches. Each patch is flattened, projected linearly to the model width and given a position embedding. The number of tokens, and with it the cost of attention, grows with the image area.',
    eq: 'tokens = (H / 8p) · (W / 8p)' },
  { span: T.attn, act: 'Act III · Joint attention',
    body: 'In an MMDiT the text tokens and image tokens form one sequence. Each modality keeps its own weights, but attention runs over the whole sequence, so every image patch can read the prompt and every word can read the image. One row of the matrix shows where a word attends.',
    eq: 'Attn([c; z]) = softmax(QKᵀ / √d) V' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §2 FUNDAMENTALS', 'Patchify and attend', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 23, color: th.fg, alpha: a, font: 'serif' });
  }
  const read = (lab, v, col, xx, al) => {
    text(ctx, lab, xx, 690, { size: 15, weight: 600, color: th.muted, alpha: al, spacing: 1.5 });
    text(ctx, v, xx, 746, { size: 44, weight: 700, color: col, alpha: al, font: 'mono' });
  };
  const la = window_(t, T.latent[0], T.latent[1], 0.4) * seg(t, ...T.latShow);
  if (la > 0) { read('PIXEL VALUES', (IMG * IMG * 3).toLocaleString('en-US'), th.fg, x, la); read('LATENT VALUES', (LAT * LAT * CH).toLocaleString('en-US'), th.accent, x + 250, la); }
  const pa = window_(t, T.patch[0], T.patch[1], 0.4) * seg(t, ...T.fly);
  if (pa > 0) { read('IMAGE TOKENS', String(NI), th.accent, x, pa); read('AT 1024 × 1024', '4,096', th.fg, x + 250, pa); }
  const aa = window_(t, T.attn[0], T.attn[1], 0.4);
  if (aa > 0) { read('SEQUENCE LENGTH', `${NT} + ${NI}`, th.fg, x, aa); const q = activeQuery(t); if (q) read('QUERY', `"${WORDS[q.w]}"`, th.ramp[3], x + 250, aa * q.a); }
  paragraph(ctx, 'Procedural image. The encoder here is 8× average pooling into four channels, standing in for a learned autoencoder. Attention weights come from a hand-set colour similarity, not from a trained model. The 1024 × 1024 count assumes an 8× autoencoder and 2×2 patches.', x, 806, w, { size: 14, color: th.muted, lineHeight: 20 });
}
