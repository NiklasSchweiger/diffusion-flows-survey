// Shared scene pieces for the 2D toy animations: the spiral target, plot grid, glow, particles, text panel.
import { GMM } from './gmm.js';
import { rampColor, text } from './theme.js';

// Gaussian beads along a spiral. us[k] in [0,1] is the position along the arm and drives colour.
export function spiralMixture({ K = 110, std = 0.075 } = {}) {
  const means = [], us = [];
  for (let k = 0; k < K; k++) {
    const u = k / (K - 1), th = 0.35 + u * 2 * Math.PI * 1.65, r = 0.42 + 1.98 * u;
    means.push([r * Math.cos(th), r * Math.sin(th)]);
    us.push(u);
  }
  return { gmm: new GMM(means, std), means, us, std };
}

// World -> pixel mapping for a plot centred at (cx, cy) with s px per world unit.
export function makePlot(cx, cy, s) {
  return { cx, cy, s, px: (x, y) => [cx + s * x, cy - s * y] };
}

export function drawGrid(ctx, th, plot, W, H) {
  ctx.save();
  ctx.strokeStyle = th.grid;
  ctx.lineWidth = 1;
  for (let x = -6; x <= 6; x++) { const [a] = plot.px(x, 0); ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, H); ctx.stroke(); }
  for (let y = -4; y <= 4; y++) { const [, b] = plot.px(0, y); ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(W, b); ctx.stroke(); }
  ctx.restore();
}

// Offscreen canvas with a soft glow per mixture component (the target density).
export function renderGlow(env, plot, mix) {
  const { W, H } = env, k = env.dpr || 1;
  const c = document.createElement('canvas');
  c.width = W * k; c.height = H * k;
  const g = c.getContext('2d');
  g.scale(k, k);
  g.globalCompositeOperation = env.theme.blend;
  mix.means.forEach(([mx, my], i) => {
    const [x, y] = plot.px(mx, my), r = mix.std * plot.s * 2.6;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, rampColor(env.theme, mix.us[i], env.theme.name === 'dark' ? 0.16 : 0.12));
    grad.addColorStop(1, rampColor(env.theme, mix.us[i], 0));
    g.fillStyle = grad;
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
  });
  return c;
}

// Glowing particle: a soft halo plus a bright core.
export function dot(ctx, th, x, y, color, alpha, r = 2.1) {
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha * (th.name === 'dark' ? 0.22 : 0.12);
  ctx.beginPath(); ctx.arc(x, y, r * 2.6, 0, 2 * Math.PI); ctx.fill();
  ctx.globalAlpha = alpha * 0.95;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
  ctx.globalAlpha = 1;
}

// Opaque-to-transparent backdrop behind the left text panel, plus kicker and title.
export function panelHeader(ctx, th, H, kicker, title, x = 72) {
  ctx.save();
  const grad = ctx.createLinearGradient(0, 0, 640, 0);
  grad.addColorStop(0, th.bg); grad.addColorStop(0.78, th.bg);
  grad.addColorStop(1, th.name === 'dark' ? 'rgba(10,14,23,0)' : 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 640, H);
  ctx.restore();
  text(ctx, kicker, x, 92, { size: 14, weight: 600, color: th.muted, spacing: 2 });
  text(ctx, title, x, 146, { size: 46, weight: 700, color: th.fg });
}
