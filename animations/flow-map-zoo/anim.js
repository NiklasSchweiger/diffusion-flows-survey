// A9 · The flow-map zoo: Euler steps, consistency models, flow maps and MeanFlow on one exact teacher trajectory.
// The teacher is the exact probability-flow ODE of a Gaussian mixture, so every "learned" object shown here is
// its ideal target: f(x_s, s) = x_1, X_{s,t}(x_s) = x_t, and the average velocity (x_t - x_s) / (t - s).
import { trajAt } from '../lib/gmm.js';
import { makeRng } from '../lib/rng.js';
import { spiralMixture, makePlot, drawGrid, dot, panelHeader } from '../lib/scene.js';
import { seg, ease, window_, rampColor, text, paragraph, arrow, lerp } from '../lib/theme.js';

const W = 1600, H = 900;
const CX = 1050, CY = 455, S = 150;

// Camera: the full spiral for the cloud acts, zoomed onto the hero trajectory for the single-path acts.
// It is set from t at the start of draw(), so every helper below stays a pure function of t.
const VIEWS = {
  full: { z: 1, c: [0, 0] },
  euler: { z: 1.9, c: [-0.08, -1.12] },   // fits the hero curve and the one-step endpoint at the data mean
  hero: { z: 3, c: [-0.1, -1.86] },
};
const CAM_KEYS = [
  [0, 'full'], [2.0, 'full'], [3.4, 'euler'], [10.2, 'euler'], [11.0, 'full'], [16.5, 'full'],
  [17.4, 'hero'], [39.5, 'hero'], [40.4, 'full'],
];
let plot = makePlot(CX, CY, S);
const px = (x, y) => plot.px(x, y);

function setCamera(t) {
  let i = 0;
  while (i < CAM_KEYS.length - 2 && t >= CAM_KEYS[i + 1][0]) i++;
  const [t0, a] = CAM_KEYS[i], [t1, b] = CAM_KEYS[i + 1];
  const w = ease.inOut(seg(t, t0, t1)), A = VIEWS[a], B = VIEWS[b];
  const z = Math.exp(lerp(Math.log(A.z), Math.log(B.z), w));
  // Interpolate the world point under the screen centre so the zoom looks like one continuous move.
  const k = (1 / z - 1 / A.z) / (1 / B.z - 1 / A.z || 1);
  const c = A.z === B.z ? [lerp(A.c[0], B.c[0], w), lerp(A.c[1], B.c[1], w)] : [lerp(A.c[0], B.c[0], k), lerp(A.c[1], B.c[1], k)];
  const sc = S * z;
  plot = { cx: CX, cy: CY, s: sc, px: (x, y) => [CX + sc * (x - c[0]), CY - sc * (y - c[1])] };
}

const N = 900;              // cloud particles
const N_FLOW_BG = 220;      // faint teacher trajectories in the background
const N_NEAR = 70;          // extra faint trajectories starting near the hero
const TEACHER_STEPS = 400;  // Heun steps for the teacher ODE (the ground truth)
const EULER_NS = [1, 2, 4, 8, 32];
const HERO_NS = [1, 2, 4, 8, 32];
const HERO_X0 = [0.38, -2.09];

// Timeline (seconds)
const T = {
  setup: [0, 3],
  euler: [3, 10.6], eulerN: [3.6, 4.9, 6.2, 7.5, 8.8], eulerDur: 1.1,
  cloud: [10.6, 16.8], cloudMoves: [[10.9, 11.9], [12.6, 13.2], [13.6, 14.2], [14.6, 15.2], [15.6, 16.2]],
  cm: [16.8, 24.4], cmProbes: [17.4, 18.0, 18.6, 19.2], cmSweep: [20.4, 23.6],
  fm: [24.4, 32.4], fmSweep: [25.2, 28.0], fmComp: [28.4, 32.0],
  mf: [32.4, 40.0], mfSweep: [33.4, 38.2],
  one: [40.0, 48.0], oneEuler: [40.6, 42.8], oneFm: [43.8, 46.4],
};
const FM_S = 0.12, FM_T = 0.45, FM_R = 0.9;   // composition example
const MF_S = 0.25;                             // MeanFlow anchor time
const CM_PROBES = [0, 0.3, 0.55, 0.8];

const STILLS = [
  { t: 10.0, name: 'euler', label: 'Euler steps' },
  { t: 13.4, name: 'euler-cloud', label: 'Two Euler steps, all samples' },
  { t: 20.2, name: 'consistency', label: 'Consistency model' },
  { t: 31.6, name: 'flow-map', crop: { x: 623, y: 25, w: 831, h: 820 }, label: 'Flow map' },
  { t: 34.6, name: 'meanflow', crop: { x: 623, y: 25, w: 831, h: 820 }, label: 'MeanFlow: chord vs tangent' },
  { t: 47.6, name: 'one-step', crop: { x: 623, y: 25, w: 831, h: 820 }, label: 'One flow-map step' },
];

export default {
  id: 'flow-map-zoo',
  title: 'The flow-map zoo: Euler, consistency models, flow maps and MeanFlow',
  duration: 48,
  width: W, height: H,
  chapters: [
    { t: 0, label: 'Setup' },
    { t: T.euler[0], label: 'I · Euler steps' },
    { t: T.cloud[0], label: 'NFE vs quality' },
    { t: T.cm[0], label: 'II · Consistency' },
    { t: T.fm[0], label: 'III · Flow map' },
    { t: T.mf[0], label: 'IV · MeanFlow' },
    { t: T.one[0], label: 'V · One step' },
  ],
  stills: STILLS,  // Paper crops trimmed to the drawn content, one height per figure row.
  crop: { x: CX - 430, y: CY - 430, w: 860, h: 860 },

  setup() {
    return buildData();
  },

  draw(ctx, t, env, st) {
    const th = env.theme;
    setCamera(t);
    drawGrid(ctx, th, plot, W, H);
    drawGlow(ctx, th, st.mix, env.clean ? 1 : seg(t, 0.3, 1.8));
    // Background flow lines: present whenever we talk about a single trajectory.
    const bgA = (1 - window_(t, T.cloud[0], T.cloud[1], 0.5)) * (1 - window_(t, T.one[0], T.one[1] + 1, 0.5));
    drawFlowLines(ctx, th, st, bgA * (env.clean ? 1 : seg(t, 0.8, 2.6)));

    if (t < T.cloud[0]) drawEuler(ctx, t, env, st);
    else if (t < T.cm[0]) drawCloud(ctx, t, env, st);
    else if (t < T.fm[0]) drawConsistency(ctx, t, env, st);
    else if (t < T.mf[0]) drawFlowMap(ctx, t, env, st);
    else if (t < T.one[0]) drawMeanFlow(ctx, t, env, st);
    else drawOneStep(ctx, t, env, st);

    if (!env.clean) drawPanel(ctx, t, env, st);
    else {
      const s = STILLS.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
      text(ctx, s.label, CX - 404, CY - 376, { size: 40, weight: 600, color: th.fg });
    }
  },
};

// ---------- Data ----------

export function buildData() {
  const mix = spiralMixture();
  const { gmm, us } = mix;
  const dest = (x, y) => us[gmm.argmaxComponent(x, y, 1)];

  // Cloud: teacher endpoints and N-step Euler endpoints for every particle.
  const rng = makeRng(21);
  const x0 = [], teach = [], uDest = [];
  for (let i = 0; i < N; i++) {
    const p = [rng.gauss(), rng.gauss()];
    x0.push(p);
    const tr = gmm.integrateODE(p[0], p[1], 120);
    teach.push(tr);
    uDest.push(dest(...trajAt(tr, 1)));
  }
  const eulerEnds = EULER_NS.map((n) => x0.map(([x, y]) => eulerPath(gmm, x, y, n).at(-1)));

  // Hero: chosen by scanning 600 noise draws for the largest bow away from the chord (0.67 units) among
  // trajectories that end on the outer arm. Strongly curved, so every few-step method visibly differs.
  const tr = gmm.integrateODE(HERO_X0[0], HERO_X0[1], TEACHER_STEPS);
  const hero = { x0: HERO_X0, traj: tr, u: dest(...trajAt(tr, 1)) };
  hero.euler = HERO_NS.map((n) => eulerPath(gmm, hero.x0[0], hero.x0[1], n));
  // Neighbours of the hero: they split between the inner and outer arm, which is why its path bends late.
  const nrng = makeRng(8), near = [], nearU = [];
  for (let i = 0; i < N_NEAR; i++) {
    const tr = gmm.integrateODE(HERO_X0[0] + 0.22 * nrng.gauss(), HERO_X0[1] + 0.22 * nrng.gauss(), 120);
    near.push(tr);
    nearU.push(dest(...trajAt(tr, 1)));
  }
  return { mix, gmm, x0, teach, uDest, eulerEnds, hero, near, nearU };
}

function eulerPath(gmm, x, y, n) {
  const pts = [[x, y]];
  for (let i = 0; i < n; i++) {
    const [vx, vy] = gmm.velocity(x, y, i / n);
    x += vx / n; y += vy / n;
    pts.push([x, y]);
  }
  return pts;
}

function drawGlow(ctx, th, mix, alpha) {
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  ctx.globalAlpha = alpha;
  mix.means.forEach(([mx, my], i) => {
    const [x, y] = px(mx, my), r = mix.std * plot.s * 2.6;
    const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, rampColor(th, mix.us[i], th.name === 'dark' ? 0.16 : 0.12));
    grad.addColorStop(1, rampColor(th, mix.us[i], 0));
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
  });
  ctx.restore();
}

function drawFlowLines(ctx, th, st, alpha) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  ctx.lineWidth = 1;
  const a = th.name === 'dark' ? 0.09 : 0.11;
  const path = (tr, u, k) => {
    ctx.strokeStyle = rampColor(th, u, a * k * alpha);
    ctx.beginPath();
    for (let j = 0; j < tr.length / 2; j++) {
      const [x, y] = px(tr[2 * j], tr[2 * j + 1]);
      j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  };
  for (let i = 0; i < N_FLOW_BG; i++) path(st.teach[i], st.uDest[i], 1);
  for (let i = 0; i < N_NEAR; i++) path(st.near[i], st.nearU[i], 2.2);
  ctx.restore();
}

// ---------- Hero helpers ----------

const EULER_LABEL = [[14, -10, 'left'], [-14, -6, 'right'], [-14, 22, 'right'], [14, 12, 'left'], [16, 30, 'left']];
const C = (th) => ({ euler: th.name === 'dark' ? '#9dff8a' : '#15803d', cm: th.ramp[0], fm: th.ramp[1], mf: th.ramp[2] });

function heroAt(st, s) { return px(...trajAt(st.hero.traj, s)); }

function drawTeacher(ctx, th, st, upTo = 1, alpha = 1) {
  const tr = st.hero.traj, end = Math.floor(upTo * TEACHER_STEPS);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = th.fg;
  ctx.lineWidth = 2.6;
  ctx.lineCap = ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(...px(tr[0], tr[1]));
  for (let j = 1; j <= end; j++) ctx.lineTo(...px(tr[2 * j], tr[2 * j + 1]));
  ctx.lineTo(...heroAt(st, upTo));
  ctx.stroke();
  ctx.restore();
  // start (noise) and end (data) markers
  const [ax, ay] = heroAt(st, 0);
  marker(ctx, th, ax, ay, th.fg, alpha, false);
  if (upTo >= 1) { const [bx, by] = heroAt(st, 1); marker(ctx, th, bx, by, th.fg, alpha, true); }
}

function marker(ctx, th, x, y, color, alpha, ring) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.arc(x, y, ring ? 9 : 6, 0, 2 * Math.PI);
  ring ? ctx.stroke() : ctx.fill();
  if (ring) { ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill(); }
  ctx.restore();
}

// Label placed `d` px outward from the middle of the hero arch, so it never sits on the curve itself.
const ARCH_CENTER = [-0.1, -1.95];
function tag(ctx, th, str, x, y, color, alpha, { d = 24, size = 21 } = {}) {
  const [cx, cy] = px(...ARCH_CENTER), n = Math.hypot(x - cx, y - cy) || 1;
  const ux = (x - cx) / n, uy = (y - cy) / n;
  const align = ux > 0.35 ? 'left' : ux < -0.35 ? 'right' : 'center';
  label(ctx, th, str, x + d * ux, y + d * uy + size * 0.35, color, alpha, { size, align });
}

function label(ctx, th, str, x, y, color, alpha, opts = {}) {
  text(ctx, str, x, y, { size: 19, weight: 600, color, alpha, font: 'serif', ...opts });
}

// ---------- Acts ----------

function drawEuler(ctx, t, env, st) {
  const th = env.theme, col = C(th);
  const out = 1 - seg(t, T.euler[1] - 0.5, T.euler[1]);
  drawTeacher(ctx, th, st, ease.inOut(seg(t, 1.0, 3.2)), seg(t, 0.8, 1.6) * out);
  T.eulerN.forEach((start, k) => {
    const g = seg(t, start, start + T.eulerDur);
    if (g <= 0) return;
    const pts = st.hero.euler[k], n = pts.length - 1;
    const newest = k === T.eulerN.length - 1 || t < T.eulerN[k + 1];
    const a = out * (newest ? 1 : 0.4);
    const f = ease.inOut(g) * n;
    ctx.save();
    ctx.globalAlpha = a;
    ctx.strokeStyle = col.euler;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(...px(...pts[0]));
    let last = pts[0];
    for (let i = 1; i <= Math.ceil(f); i++) {
      const w = Math.min(f - (i - 1), 1);
      last = [lerp(pts[i - 1][0], pts[i][0], w), lerp(pts[i - 1][1], pts[i][1], w)];
      ctx.lineTo(...px(...last));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    for (let i = 1; i <= Math.floor(f); i++) {
      const [x, y] = px(...pts[i]);
      ctx.fillStyle = col.euler;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
    if (g >= 1) {
      const [x, y] = px(...pts[n]);
      const [dx, dy, align] = EULER_LABEL[k];
      label(ctx, th, `N = ${n}`, x + dx, y + dy, col.euler, a, { align });
    }
  });
}

function drawCloud(ctx, t, env, st) {
  const th = env.theme;
  const a = window_(t, T.cloud[0], T.cloud[1], 0.5);
  // stage: 0 = noise, k = after the k-th Euler budget in EULER_NS.
  let stage = 0;
  T.cloudMoves.forEach(([s0, s1]) => { stage += ease.inOut(seg(t, s0, s1)); });
  const k = Math.floor(stage), w = stage - k;
  const posAt = (j, i) => (j === 0 ? st.x0[i] : st.eulerEnds[j - 1][i]);
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  for (let i = 0; i < N; i++) {
    const p = posAt(Math.min(k, EULER_NS.length), i), q = posAt(Math.min(k + 1, EULER_NS.length), i);
    const [x, y] = px(lerp(p[0], q[0], w), lerp(p[1], q[1], w));
    dot(ctx, th, x, y, rampColor(th, st.uDest[i], 1), a);
  }
  ctx.restore();
}

function drawConsistency(ctx, t, env, st) {
  const th = env.theme, col = C(th);
  const a = window_(t, T.cm[0], T.cm[1], 0.5);
  drawTeacher(ctx, th, st, 1, a);
  const [ex, ey] = heroAt(st, 1);
  const sweepOn = seg(t, T.cmSweep[0], T.cmSweep[0] + 0.4);
  CM_PROBES.forEach((s, k) => {
    const g = ease.out(seg(t, T.cmProbes[k], T.cmProbes[k] + 0.6));
    if (g <= 0) return;
    const [x, y] = heroAt(st, s);
    const pa = a * (1 - 0.6 * sweepOn);
    arrow(ctx, x, y, lerp(x, ex, g), lerp(y, ey, g), { color: col.cm, width: 2.4, head: 11, alpha: pa });
    marker(ctx, th, x, y, col.cm, pa, false);
    tag(ctx, th, `s = ${s.toFixed(2)}`, x, y, col.cm, pa * g, { size: 18, d: 22 });
  });
  if (sweepOn > 0) {
    const s = 0.95 * ease.inOut(seg(t, ...T.cmSweep));
    const [x, y] = heroAt(st, s);
    arrow(ctx, x, y, ex, ey, { color: col.cm, width: 3.4, head: 14, alpha: a * sweepOn });
    marker(ctx, th, x, y, th.highlight, a * sweepOn, false);
  }
  tag(ctx, th, 'x₁', ex, ey, th.fg, a, { size: 23 });
}

function drawFlowMap(ctx, t, env, st) {
  const th = env.theme, col = C(th);
  const a = window_(t, T.fm[0], T.fm[1], 0.5);
  drawTeacher(ctx, th, st, 1, a * 0.85);
  const comp = seg(t, T.fmComp[0], T.fmComp[0] + 0.5);
  const sweep = 1 - comp;
  if (sweep > 0) {
    const tt = lerp(FM_S + 0.12, 0.95, ease.inOut(seg(t, ...T.fmSweep)));
    chord(ctx, th, st, FM_S, tt, col.fm, a * sweep, 's', 't');
  }
  if (comp > 0) {
    const g1 = ease.out(seg(t, T.fmComp[0] + 0.2, T.fmComp[0] + 1.0));
    const g2 = ease.out(seg(t, T.fmComp[0] + 1.0, T.fmComp[0] + 1.8));
    const g3 = ease.out(seg(t, T.fmComp[0] + 2.2, T.fmComp[0] + 3.0));
    chord(ctx, th, st, FM_S, FM_T, col.fm, a * comp, 's', 't', g1);
    if (g2 > 0) chord(ctx, th, st, FM_T, FM_R, col.fm, a * comp, null, 'r', g2);
    if (g3 > 0) chord(ctx, th, st, FM_S, FM_R, th.fg, a * comp, null, null, g3, true);
  }
}

// Straight jump from x_s to x_t on the hero trajectory.
function chord(ctx, th, st, s, t, color, alpha, ls, lt, grow = 1, dashed = false) {
  const [x0, y0] = heroAt(st, s), [x1, y1] = heroAt(st, t);
  ctx.save();
  if (dashed) ctx.setLineDash([8, 6]);
  arrow(ctx, x0, y0, lerp(x0, x1, grow), lerp(y0, y1, grow), { color, width: dashed ? 2.4 : 3, head: 13, alpha });
  ctx.restore();
  marker(ctx, th, x0, y0, color, alpha, false);
  if (grow >= 1) marker(ctx, th, x1, y1, color, alpha, false);
  if (ls) tag(ctx, th, `x${sub(ls)}`, x0, y0, color, alpha);
  if (lt && grow >= 1) tag(ctx, th, `x${sub(lt)}`, x1, y1, color, alpha);
}
const sub = (c) => ({ s: 'ₛ', t: 'ₜ', r: 'ᵣ' })[c];

function drawMeanFlow(ctx, t, env, st) {
  const th = env.theme, col = C(th);
  const a = window_(t, T.mf[0], T.mf[1], 0.5);
  drawTeacher(ctx, th, st, 1, a * 0.85);
  const tt = lerp(0.95, MF_S + 0.004, ease.inOut(seg(t, ...T.mfSweep)));
  const [xs, ys] = trajAt(st.hero.traj, MF_S), [xt, yt] = trajAt(st.hero.traj, tt);
  const [sx, sy] = px(xs, ys), [ex, ey] = px(xt, yt);
  // chord, faint
  ctx.save();
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = col.mf; ctx.lineWidth = 1.6; ctx.globalAlpha = a * 0.7;
  ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.restore();
  marker(ctx, th, ex, ey, col.mf, a, false);
  tag(ctx, th, 'xₜ', ex, ey, col.mf, a * seg(tt, MF_S + 0.06, MF_S + 0.12));
  // velocities drawn from x_s on a shared scale
  const L = 0.24 * plot.s;
  const ub = [(xt - xs) / (tt - MF_S), (yt - ys) / (tt - MF_S)];
  const u = st.gmm.velocity(xs, ys, MF_S);
  arrow(ctx, sx, sy, sx + L * u[0], sy - L * u[1], { color: th.fg, width: 3, head: 13, alpha: a });
  arrow(ctx, sx, sy, sx + L * ub[0], sy - L * ub[1], { color: col.mf, width: 3.4, head: 14, alpha: a });
  marker(ctx, th, sx, sy, th.highlight, a, false);
  // Tip labels sit past the arrowhead, pushed to opposite sides so they stay apart when the arrows coincide.
  const tipLabel = (str, v, side, color, size) => {
    const n = Math.hypot(v[0], v[1]) || 1, dx = v[0] / n, dy = -v[1] / n;
    const x = sx + L * v[0] + 20 * dx + side * 16 * -dy, y = sy - L * v[1] + 20 * dy + side * 16 * dx;
    label(ctx, th, str, x, y + size * 0.35, color, a, { size, align: 'center' });
  };
  tipLabel('uₛ', u, 1, th.fg, 21);
  tipLabel('ū', ub, -1, col.mf, 24);
  tag(ctx, th, 'xₛ', sx, sy, th.fg, a);
}

function drawOneStep(ctx, t, env, st) {
  const th = env.theme;
  const eA = window_(t, T.oneEuler[0] - 0.4, T.oneFm[0] - 0.3, 0.4);
  const fA = seg(t, T.oneFm[0] - 0.4, T.oneFm[0]);
  ctx.save();
  ctx.globalCompositeOperation = th.blend;
  if (eA > 0) {
    const g = ease.inOut(seg(t, ...T.oneEuler));
    for (let i = 0; i < N; i++) {
      const p = st.x0[i], q = st.eulerEnds[0][i];
      dot(ctx, th, ...px(lerp(p[0], q[0], g), lerp(p[1], q[1], g)), rampColor(th, st.uDest[i], 1), eA);
    }
  }
  if (fA > 0) {
    const g = ease.inOut(seg(t, ...T.oneFm));
    ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
      const p = st.x0[i], q = trajAt(st.teach[i], 1);
      const c = [lerp(p[0], q[0], g), lerp(p[1], q[1], g)];
      if (i < 260 && g > 0 && g < 1) {
        ctx.strokeStyle = rampColor(th, st.uDest[i], 0.14 * fA * (1 - g));
        ctx.beginPath(); ctx.moveTo(...px(...p)); ctx.lineTo(...px(...c)); ctx.stroke();
      }
      dot(ctx, th, ...px(...c), rampColor(th, st.uDest[i], 1), fA);
    }
  }
  ctx.restore();
  const [mx, my] = px(...st.eulerEnds[0][0]);
  text(ctx, '1 Euler step: everything lands on the data mean', mx, my + 52, { size: 18, weight: 600, color: th.fg, align: 'center', alpha: eA * seg(t, T.oneEuler[1] - 0.4, T.oneEuler[1]) });
}

// ---------- Text panel ----------

const COPY = [
  { span: T.setup, act: 'Setup',
    body: 'A flow model defines an ODE whose trajectories carry noise to data. Every few-step method in this chapter learns some piece of these trajectories. The white curve is one of them, solved exactly.',
    eq: 'dx/dt = uₜ(x)' },
  { span: T.euler, act: 'Act I · Euler steps',
    body: 'A sampler follows the tangent and cuts corners. One step lands on the data mean. Eight steps reach a sharp sample on the wrong arm, because this path only turns outward late. It takes about 32.',
    eq: 'xₜ₊ₕ = xₜ + h · uₜ(xₜ)' },
  { span: T.cloud, act: 'Act I · All samples',
    body: 'Across the whole distribution, one step collapses every sample onto one blurry average. A sharp spiral needs dozens of network evaluations (NFE). Colours mark where the exact ODE would have sent each sample.',
    eq: '' },
  { span: T.cm, act: 'Act II · Consistency models',
    body: 'A consistency model learns the endpoint directly. Every point on a trajectory maps to the same sample. This self-consistency is the property used to train it.',
    eq: 'f(xₛ, s) = x₁   for all s' },
  { span: [T.fm[0], T.fmComp[0]], act: 'Act III · Flow maps',
    body: 'A flow map jumps between any two times on the trajectory. Consistency models are the special case t = 1, and the instantaneous velocity is the limit t → s.',
    eq: 'Xₛ,ₜ(xₛ) = xₜ' },
  { span: [T.fmComp[0], T.fm[1]], act: 'Act III · Jumps compose',
    body: 'Two short jumps equal one long jump. Shortcut models and flow-map self-distillation use this identity to train long jumps from short ones without a teacher.',
    eq: 'Xₜ,ᵣ ∘ Xₛ,ₜ = Xₛ,ᵣ' },
  { span: T.mf, act: 'Act IV · MeanFlow',
    body: 'MeanFlow parametrizes the jump by its average velocity, the slope of the chord. As t approaches s the chord turns into the tangent, so the ordinary velocity is a special case.',
    eq: 'ū(xₛ, s, t) = (xₜ − xₛ) / (t − s)' },
  { span: [T.one[0], T.oneFm[0] - 0.2], act: 'Act V · One step',
    body: 'One Euler step collapses everything onto the mean, because the initial slope only knows the average destination.',
    eq: 'x₀ + u₀(x₀) = E[x₁]' },
  { span: [T.oneFm[0] - 0.2, T.one[1] + 1], act: 'Act V · One step',
    body: 'One flow-map step lands every sample on the data. It learned where the curved path ends, so the path itself does not need to be straight.',
    eq: 'x₁ = X₀,₁(x₀)' },
];

function drawPanel(ctx, t, env) {
  const th = env.theme, x = 72, w = 460;
  panelHeader(ctx, th, H, 'SURVEY · §5 DISTILLATION', 'The flow-map zoo', x);
  for (const c of COPY) {
    const a = window_(t, c.span[0], c.span[1], 0.4);
    if (a <= 0) continue;
    text(ctx, c.act, x, 214, { size: 22, weight: 600, color: th.accent, alpha: a });
    const yEnd = paragraph(ctx, c.body, x, 256, w, { size: 20, color: th.fg, alpha: a * 0.92, lineHeight: 31 });
    if (c.eq) text(ctx, c.eq, x, yEnd + 30, { size: 25, color: th.fg, alpha: a, font: 'serif' });
  }

  // NFE readout during the Euler acts, the flow-map plane during II to IV.
  const nfeA = Math.max(window_(t, T.cloud[0], T.cloud[1], 0.4), window_(t, T.one[0], T.one[1] + 1, 0.4));
  if (nfeA > 0) {
    const stage = cloudStage(t), nfe = t >= T.one[0] ? 1 : stage ? EULER_NS[stage - 1] : 0;
    text(ctx, 'NFE', x, 640, { size: 18, weight: 600, color: th.muted, alpha: nfeA, spacing: 1.5 });
    text(ctx, String(nfe), x, 718, { size: 72, weight: 700, color: th.fg, alpha: nfeA, font: 'mono' });
  }
  const planeA = window_(t, T.cm[0], T.mf[1], 0.5);
  if (planeA > 0) drawPlane(ctx, t, th, x, 560, 170, planeA);

  text(ctx, 'Teacher trajectories are exact ODE solutions for a 110-component Gaussian mixture. The maps shown are their ideal targets.', x, 838, { size: 14, color: th.muted });
}

function cloudStage(t) {
  return T.cloudMoves.filter(([, s1]) => t >= s1 - 0.3).length;
}

// The (s, t) plane: the diagonal is the velocity field, the top edge the consistency model, the interior flow maps.
function drawPlane(ctx, t, th, x, y, size, alpha) {
  const col = C(th);
  const P = (s, tt) => [x + s * size, y + (1 - tt) * size];
  const act = t < T.fm[0] ? 'cm' : t < T.mf[0] ? 'fm' : 'mf';
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = th.faint;
  ctx.beginPath(); ctx.moveTo(...P(0, 0)); ctx.lineTo(...P(0, 1)); ctx.lineTo(...P(1, 1)); ctx.closePath(); ctx.fill();
  if (act === 'fm') { ctx.fillStyle = col.fm; ctx.globalAlpha = alpha * 0.16; ctx.fill(); ctx.globalAlpha = alpha; }
  const edge = (a, b, color, wdt) => { ctx.strokeStyle = color; ctx.lineWidth = wdt; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); };
  edge(P(0, 1), P(1, 1), act === 'cm' ? col.cm : th.muted, act === 'cm' ? 4 : 1.5);
  edge(P(0, 0), P(1, 1), act === 'mf' ? th.fg : th.muted, act === 'mf' ? 3 : 1.5);
  edge(P(0, 0), P(0, 1), th.muted, 1);
  ctx.restore();
  text(ctx, 's', x + size / 2, y + size + 24, { size: 16, color: th.muted, font: 'serif', alpha, align: 'center' });
  text(ctx, 't', x - 16, y + size / 2, { size: 16, color: th.muted, font: 'serif', alpha });
  text(ctx, 'consistency (t = 1)', x + size + 14, y + 6, { size: 14, weight: 600, color: act === 'cm' ? col.cm : th.muted, alpha });
  text(ctx, 'flow map Xₛ,ₜ', x + size + 14, y + size * 0.42, { size: 14, weight: 600, color: act === 'fm' ? col.fm : th.muted, alpha });
  text(ctx, 'velocity (t = s)', x + size + 14, y + size, { size: 14, weight: 600, color: act === 'mf' ? th.fg : th.muted, alpha });

  // Current (s, t) marks.
  const marks = [];
  if (act === 'cm') {
    CM_PROBES.forEach((s, k) => { if (t >= T.cmProbes[k]) marks.push([s, 1, col.cm]); });
    if (t >= T.cmSweep[0]) marks.push([0.95 * ease.inOut(seg(t, ...T.cmSweep)), 1, th.fg]);
  } else if (act === 'fm') {
    if (t < T.fmComp[0]) marks.push([FM_S, lerp(FM_S + 0.12, 0.95, ease.inOut(seg(t, ...T.fmSweep))), col.fm]);
    else marks.push([FM_S, FM_T, col.fm], [FM_T, FM_R, col.fm], [FM_S, FM_R, th.fg]);
  } else {
    marks.push([MF_S, lerp(0.95, MF_S + 0.004, ease.inOut(seg(t, ...T.mfSweep))), col.mf]);
  }
  for (const [s, tt, c] of marks) {
    const [mx, my] = P(s, tt);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = c;
    ctx.beginPath(); ctx.arc(mx, my, 5.5, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
}
