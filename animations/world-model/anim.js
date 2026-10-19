// A15 · An action-conditioned world model on a tiny grid game.
// The game engine moves an agent one cell per key press, blocks it at walls and removes coins it touches. The world
// model predicts the next frame from the current frame and the action. Its frame is a field of agent intensity per
// cell, and its one-step prediction moves each unit of intensity with a slightly imperfect kernel (84% one cell,
// 9% two cells, 3% none, 2% to each side), then lets it erode coins it covers. Teacher forcing only ever feeds it real,
// sharp frames. The self-forced variant first re-sharpens its context (A ↦ A^γ / Σ A^γ), which is the identity on
// real frames, so the two variants have the same one-step loss and differ only on their own rollouts.
import { seg, ease, window_, text, paragraph, lerp, clamp } from '../lib/theme.js';
import { panelHeader } from '../lib/scene.js';

const W = 1600, H = 900;
const MAP = [
  '######################',
  '#....#...........o...#',
  '#.o..#..####.........#',
  '#....#.....#....o....#',
  '#....#..o..#.........#',
  '#.......#..#..######.#',
  '#.......#.......o....#',
  '###.#####............#',
  '#.......#...o...###..#',
  '#..o....#.......#....#',
  '#.......####....#..o.#',
  '#..........#....#....#',
  '#....o.....#.........#',
  '#...........o........#',
  '######################',
];
const GW = MAP[0].length, GH = MAP.length;
const WALL = MAP.map((r) => [...r].map((ch) => ch === '#'));
const START = [6, 3];                              // (col, row)
const KER = { fwd: 0.84, two: 0.09, stay: 0.03, side: 0.02 };
const GAMMA = 3;
const NA = 64;                                     // actions in the rollout
const DIRS = { U: [0, -1], D: [0, 1], L: [-1, 0], R: [1, 0] };
const ARROW = { U: '↑', D: '↓', L: '←', R: '→' };

const T = {
  setup: [0, 9], setupPlay: [1.5, 8.4],
  one: [9, 19], oneStep: [10, 18.4],
  roll: [19, 41], rollRun: [20, 38.5],
  chart: [41, 54], chartDraw: [42, 46.5],
};
const SETUP_N = 12;

const STILLS = [
  { t: 8.6, name: 'engine', label: 'The game engine' },
  { t: 18.3, name: 'one-step', label: 'One-step predictions from real frames' },
  { t: 40.4, name: 'rollout', label: `After ${NA} actions` },
  { t: 53.4, name: 'curves', label: 'Error along the rollout' },
];

export default {
  id: 'world-model',
  title: 'An action-conditioned world model',
  duration: 54,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.one[0], label: 'I · One step' },
    { t: T.roll[0], label: 'II · Rollout' },
    { t: T.chart[0], label: 'III · Error' },
  ],
  stills: STILLS,
  crop: { x: 620, y: 0, w: 970, h: 845 },

  setup() { return buildData(); },

  draw(ctx, t, env, st) {
    const th = env.theme;
    const pa = 1 - 0.85 * seg(t, T.chart[0], T.chart[0] + 0.8);
    const { n, mode } = frameAt(t);
    drawKeys(ctx, th, st, n, pa);
    if (mode === 'one') {
      const k = Math.min(Math.floor(n), SETUP_N + 5);
      drawGrid(ctx, th, PANELS.gt, st.gt[k + 1], pa, 'Game engine, frame ' + (k + 1), th.fg);
      drawGrid(ctx, th, PANELS.tf, st.oneTF[k], pa, 'Teacher-forced, real frame ' + k, th.fg);
      drawGrid(ctx, th, PANELS.sf, st.oneSF[k], pa, 'Self-forced, real frame ' + k, th.accent);
    } else {
      const k = Math.min(Math.floor(n), NA), u = mode === 'setup' ? 0 : n - k;
      drawGrid(ctx, th, PANELS.gt, mixFrame(st.gt, k, u), pa, `Game engine, frame ${k}`, th.fg);
      const ma = mode === 'setup' ? seg(t, 4, 5.5) * pa : pa;
      if (ma > 0) {
        const m = mode === 'setup' ? 0 : k;
        drawGrid(ctx, th, PANELS.tf, mode === 'setup' ? st.gt[0] : mixFrame(st.tf, m, u), ma, mode === 'setup' ? 'Teacher-forced model' : `Teacher-forced, own frame ${m}`, th.fg);
        drawGrid(ctx, th, PANELS.sf, mode === 'setup' ? st.gt[0] : mixFrame(st.sf, m, u), ma, mode === 'setup' ? 'Self-forced model' : `Self-forced, own frame ${m}`, th.accent);
      }
    }
    if (t >= T.chart[0]) drawChart(ctx, th, st, t, seg(t, T.chart[0], T.chart[0] + 0.8));

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const c = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, c.label, 660, 64, { size: 36, weight: 600, color: th.fg });
    }
  },
};

// ---------- Game engine and world model ----------

const idx = (c, r) => r * GW + c;

function coinsInit() {
  const C = new Float64Array(GW * GH);
  MAP.forEach((row, r) => [...row].forEach((ch, c) => { if (ch === 'o') C[idx(c, r)] = 1; }));
  return C;
}

// Engine: the agent field is one-hot.
function engineStep(f, a) {
  const [c, r] = f.pos, [dc, dr] = DIRS[a];
  const pos = WALL[r + dr][c + dc] ? [c, r] : [c + dc, r + dr];
  const C = f.C.slice();
  C[idx(...pos)] = 0;
  const A = new Float64Array(GW * GH);
  A[idx(...pos)] = 1;
  return { pos, A, C };
}

// Move intensity at (c, r) by k cells along (dc, dr), stopping before walls.
function slide(c, r, dc, dr, k) {
  for (let s = 0; s < k; s++) {
    if (WALL[r + dr][c + dc]) break;
    c += dc; r += dr;
  }
  return idx(c, r);
}

function modelStep(f, a, sharpen) {
  let A = f.A;
  if (sharpen) {
    let z = 0;
    A = A.map((v) => { const w = Math.pow(v, GAMMA); z += w; return w; });
    A = A.map((v) => v / z);
  }
  const [dc, dr] = DIRS[a], out = new Float64Array(GW * GH);
  for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
    const m = A[idx(c, r)];
    if (m < 1e-9) continue;
    out[slide(c, r, dc, dr, 1)] += KER.fwd * m;
    out[slide(c, r, dc, dr, 2)] += KER.two * m;
    out[idx(c, r)] += KER.stay * m;
    out[slide(c, r, dr, dc, 1)] += KER.side * m;
    out[slide(c, r, -dr, -dc, 1)] += KER.side * m;
  }
  const C = f.C.map((v, i) => v * (1 - clamp(out[i])));
  return { A: out, C };
}

function centroid(A) {
  let x = 0, y = 0, z = 0;
  for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) { const m = A[idx(c, r)]; x += m * c; y += m * r; z += m; }
  return [x / z, y / z];
}

// Greedy route: two moves up, then walk to the nearest remaining coin by BFS until NA actions.
function route() {
  const C = coinsInit(), acts = ['U', 'U'];
  let pos = [...START];
  for (const a of acts) { const [dc, dr] = DIRS[a]; if (!WALL[pos[1] + dr][pos[0] + dc]) pos = [pos[0] + dc, pos[1] + dr]; }
  while (acts.length < NA) {
    const prev = new Map(), q = [pos], key = (p) => p[0] + ',' + p[1];
    prev.set(key(pos), null);
    let goal = null;
    while (q.length) {
      const p = q.shift();
      if (C[idx(...p)] > 0 && key(p) !== key(pos)) { goal = p; break; }
      for (const a of ['U', 'R', 'D', 'L']) {
        const [dc, dr] = DIRS[a], n = [p[0] + dc, p[1] + dr];
        if (WALL[n[1]][n[0]] || prev.has(key(n))) continue;
        prev.set(key(n), [p, a]); q.push(n);
      }
    }
    if (!goal) break;
    const path = [];
    for (let p = goal; prev.get(key(p)); p = prev.get(key(p))[0]) path.unshift(prev.get(key(p))[1]);
    for (const a of path) {
      if (acts.length >= NA) break;
      acts.push(a);
      const [dc, dr] = DIRS[a];
      pos = [pos[0] + dc, pos[1] + dr];
      C[idx(...pos)] = 0;
    }
  }
  return acts;
}

export function buildData() {
  const acts = route();
  const A0 = new Float64Array(GW * GH);
  A0[idx(...START)] = 1;
  const gt = [{ pos: [...START], A: A0, C: coinsInit() }];
  for (const a of acts) gt.push(engineStep(gt.at(-1), a));
  const tf = [gt[0]], sf = [gt[0]];
  for (const a of acts) { tf.push(modelStep(tf.at(-1), a, false)); sf.push(modelStep(sf.at(-1), a, true)); }
  // One-step predictions from real frames.
  const oneTF = acts.map((a, k) => modelStep(gt[k], a, false));
  const oneSF = acts.map((a, k) => modelStep(gt[k], a, true));
  const err = (fr, k) => { const [x, y] = centroid(fr.A), [c, r] = gt[k].pos; return Math.hypot(x - c, y - r); };
  const coinErr = (fr, k) => fr.C.reduce((s, v, i) => s + Math.abs(v - gt[k].C[i]), 0);
  const peak = (fr) => Math.max(...fr.A);
  const curve = {
    tf: tf.map(err), sf: sf.map(err),
    tfC: tf.map(coinErr), sfC: sf.map(coinErr),
    tfP: tf.map(peak), sfP: sf.map(peak),
  };
  const oneLoss = (P) => P.reduce((s, fr, k) => s + fr.A.reduce((q, v, i) => q + (v - gt[k + 1].A[i]) ** 2, 0), 0) / P.length;
  return { acts, gt, tf, sf, oneTF, oneSF, curve, lossTF: oneLoss(oneTF), lossSF: oneLoss(oneSF) };
}

function frameAt(t) {
  if (t < T.one[0]) return { mode: 'setup', n: SETUP_N * seg(t, ...T.setupPlay) };
  if (t < T.roll[0]) return { mode: 'one', n: SETUP_N + 6 * seg(t, ...T.oneStep) - 0.001 };
  return { mode: 'roll', n: NA * ease.inOut(seg(t, ...T.rollRun)) };
}

// Linear blend between consecutive frames, so motion reads smoothly.
function mixFrame(fs, k, u) {
  if (u <= 0 || k >= fs.length - 1) return fs[Math.min(k, fs.length - 1)];
  const a = fs[k], b = fs[k + 1];
  return { A: a.A.map((v, i) => lerp(v, b.A[i], u)), C: a.C.map((v, i) => lerp(v, b.C[i], u)) };
}

// ---------- Drawing ----------

const CELL = 20;
const PANELS = {
  gt: { x: 660, y: 118 },
  tf: { x: 660, y: 500 },
  sf: { x: 1130, y: 500 },
};
const KEYS = { x: 1130, y: 118 };

function drawGrid(ctx, th, P, fr, a, title, color) {
  if (a <= 0) return;
  text(ctx, title, P.x, P.y - 18, { size: 19, weight: 700, color, alpha: a });
  ctx.save();
  ctx.globalAlpha = a;
  ctx.fillStyle = th.name === 'dark' ? '#111725' : '#f1f4f9';
  ctx.fillRect(P.x, P.y, GW * CELL, GH * CELL);
  const agent = th.name === 'dark' ? [125, 211, 252] : [3, 105, 161];
  for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
    const x = P.x + c * CELL, y = P.y + r * CELL;
    if (WALL[r][c]) {
      ctx.fillStyle = th.name === 'dark' ? '#2a3348' : '#c5cbd6';
      ctx.fillRect(x, y, CELL, CELL);
      continue;
    }
    const cv = fr.C[idx(c, r)];
    if (cv > 0.02) {
      ctx.fillStyle = th.ramp[3];
      ctx.globalAlpha = a * clamp(cv);
      ctx.beginPath(); ctx.arc(x + CELL / 2, y + CELL / 2, CELL * 0.26, 0, 2 * Math.PI); ctx.fill();
      ctx.globalAlpha = a;
    }
    const m = fr.A[idx(c, r)];
    if (m > 0.003) {
      ctx.fillStyle = `rgba(${agent.join(',')},${clamp(1.3 * Math.pow(m, 0.4))})`;
      ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
    }
  }
  ctx.restore();
}

function drawKeys(ctx, th, st, n, a) {
  if (a <= 0) return;
  const k = Math.min(Math.floor(n), NA - 1), cur = st.acts[k], fresh = 1 - (n - Math.floor(n));
  text(ctx, 'Actions from the keyboard', KEYS.x, KEYS.y - 18, { size: 19, weight: 700, color: th.fg, alpha: a });
  const S = 64, gap = 10, x0 = KEYS.x + 90, y0 = KEYS.y + 20;
  const pos = { U: [1, 0], L: [0, 1], D: [1, 1], R: [2, 1] };
  for (const key of ['U', 'L', 'D', 'R']) {
    const [i, j] = pos[key], x = x0 + i * (S + gap), y = y0 + j * (S + gap), on = key === cur;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.fillStyle = on ? th.accent : th.name === 'dark' ? '#161d2d' : '#eef1f6';
    if (on) ctx.globalAlpha = a * (0.45 + 0.55 * fresh);
    ctx.fillRect(x, y, S, S);
    ctx.globalAlpha = a;
    ctx.strokeStyle = th.muted; ctx.lineWidth = 1.2;
    ctx.strokeRect(x + 0.5, y + 0.5, S - 1, S - 1);
    ctx.restore();
    text(ctx, ARROW[key], x + S / 2, y + S / 2 + 12, { size: 32, weight: 700, color: on ? th.bg : th.fg, alpha: a, align: 'center' });
  }
  const hist = st.acts.slice(Math.max(0, k - 10), k + 1).map((v) => ARROW[v]).join(' ');
  text(ctx, 'history', KEYS.x, KEYS.y + 214, { size: 15, weight: 600, color: th.muted, alpha: a, spacing: 1.5 });
  text(ctx, hist, KEYS.x, KEYS.y + 248, { size: 22, color: th.fg, alpha: a, font: 'mono' });
}

function drawChart(ctx, th, st, t, a) {
  const L = 740, R = 1300, grow = ease.inOut(seg(t, ...T.chartDraw));
  ctx.save();
  ctx.globalAlpha = a * 0.9;
  ctx.fillStyle = th.bg;
  ctx.fillRect(640, 40, W - 640, H - 60);
  ctx.restore();
  const panels = [
    { top: 130, bot: 380, key: '', lab: 'agent position error (cells)', max: Math.ceil(Math.max(...st.curve.tf)) },
    { top: 500, bot: 770, key: 'C', lab: 'coins that changed but should not have', max: Math.ceil(Math.max(...st.curve.tfC)) },
  ];
  const X = (n) => L + (n / NA) * (R - L);
  for (const p of panels) {
    const Y = (v) => p.bot - (v / p.max) * (p.bot - p.top);
    ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = th.muted; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(L, p.top); ctx.lineTo(L, p.bot); ctx.lineTo(R, p.bot); ctx.stroke(); ctx.restore();
    text(ctx, p.lab, L, p.top - 22, { size: 18, color: th.muted, alpha: a });
    for (const v of [0, p.max / 2, p.max]) text(ctx, String(+v.toFixed(1)), L - 14, Y(v) + 6, { size: 16, color: th.muted, align: 'right', alpha: a });
    for (const v of [0, 16, 32, 48, 64]) text(ctx, String(v), X(v), p.bot + 26, { size: 16, color: th.muted, align: 'center', alpha: a });
    for (const [s, color, dash, name] of [['tf', th.fg, [9, 7], 'teacher forcing'], ['sf', th.accent, [], 'self forcing']]) {
      const c = st.curve[s + p.key], m = Math.floor(grow * NA);
      ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.setLineDash(dash); ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i <= m; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(c[i]));
      ctx.stroke(); ctx.restore();
      text(ctx, `${name}: ${c[NA].toFixed(1)}`, R + 12, Y(c[NA]) + (s === 'sf' ? 8 : -4), { size: 18, weight: 700, color, alpha: a * seg(grow, 0.95, 1) });
    }
  }
  text(ctx, 'actions', (L + R) / 2, 770 + 58, { size: 18, color: th.muted, align: 'center', alpha: a });
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'A world model is a video generator conditioned on actions. Here the "game" is a grid: each key press moves the agent one cell, walls block it, and coins disappear when collected. The model sees the current frame and the key, and generates the next frame.',
    eq: 'x_{n+1} ~ p_θ(x_{n+1} | x_{≤n}, a_n)' },
  { span: T.one, act: 'Act I · One step from real frames',
    body: 'From a real frame both models predict the next one well: most of the agent moves one cell, a little overshoots, lags or slips sideways. The self-forced model first re-sharpens its context, which changes nothing on a real frame, so both models have exactly the same one-step loss.',
    eq: '' },
  { span: T.roll, act: 'Act II · Playing inside the model',
    body: 'Now each model conditions on its own previous frame. The teacher-forced model was never shown a blurred frame and blurs it further: the agent smears out, drifts from where the engine has it, and partly erases coins it never reached. The self-forced model cleans its context before each step and keeps the game state intact.',
    eq: '' },
  { span: T.chart, act: 'Act III · Error along the rollout',
    body: 'The teacher-forced errors compound with every action. Interactive world models therefore borrow the Self Forcing recipe and train on their own long rollouts, because a frame that is slightly off becomes the context for every later frame.',
    eq: '' },
];

function drawPanel(ctx, t, env, st) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §7 APPLICATIONS', 'World models', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 34, { size: 24, color: th.fg, alpha: a, font: 'serif' });
  }
  const oa = window_(t, T.one[0], T.one[1], 0.4);
  if (oa > 0) {
    text(ctx, 'ONE-STEP LOSS ON REAL FRAMES', x, 690, { size: 15, weight: 600, color: th.muted, alpha: oa, spacing: 1.5 });
    text(ctx, `TF ${st.lossTF.toFixed(4)}   SF ${st.lossSF.toFixed(4)}`, x, 728, { size: 24, weight: 700, color: th.fg, alpha: oa, font: 'mono' });
  }
  const ra = window_(t, T.roll[0], T.roll[1], 0.4);
  if (ra > 0) {
    const k = Math.min(Math.round(NA * ease.inOut(seg(t, ...T.rollRun))), NA);
    text(ctx, `action ${k} of ${NA}`, x, 640, { size: 18, weight: 600, color: th.muted, alpha: ra });
    text(ctx, 'POSITION ERROR (CELLS)', x, 684, { size: 15, weight: 600, color: th.muted, alpha: ra, spacing: 1.5 });
    text(ctx, st.curve.tf[k].toFixed(1), x, 748, { size: 56, weight: 700, color: th.fg, alpha: ra, font: 'mono' });
    text(ctx, st.curve.sf[k].toFixed(1), x + 220, 748, { size: 56, weight: 700, color: th.accent, alpha: ra, font: 'mono' });
    text(ctx, 'teacher forcing', x, 776, { size: 15, color: th.muted, alpha: ra });
    text(ctx, 'self forcing', x + 220, 776, { size: 15, color: th.muted, alpha: ra });
  }
  paragraph(ctx, `Hand-specified models, no training. Move kernel: ${Math.round(KER.fwd * 100)}% one cell, ${Math.round(KER.two * 100)}% two cells, ${Math.round(KER.stay * 100)}% none, ${Math.round(KER.side * 100)}% to each side, blocked by walls. Self-forced context sharpening A^γ with γ = ${GAMMA}.`, x, 818, w, { size: 14, color: th.muted, lineHeight: 20 });
}
