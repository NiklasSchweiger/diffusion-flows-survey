// Shared visual language for all survey animations.
export const themes = {
  dark: {
    name: 'dark', bg: '#0a0e17', fg: '#e8ecf4', muted: '#8b95a9', faint: 'rgba(232,236,244,0.10)',
    grid: 'rgba(255,255,255,0.045)', accent: '#7dd3fc', highlight: '#ffffff', blend: 'lighter',
    ramp: ['#38e1ff', '#7b6cff', '#ff4fa3', '#ffb547'],
  },
  light: {
    name: 'light', bg: '#ffffff', fg: '#151a24', muted: '#5d6678', faint: 'rgba(21,26,36,0.10)',
    grid: 'rgba(0,0,0,0.05)', accent: '#0369a1', highlight: '#111111', blend: 'source-over',
    ramp: ['#0891b2', '#5b45e0', '#d61f78', '#d97706'],
  },
};

export const fonts = {
  sans: '"Inter", "Helvetica Neue", system-ui, sans-serif',
  mono: '"JetBrains Mono", "SF Mono", Menlo, monospace',
  serif: '"STIX Two Text", "Times New Roman", serif',
};

export const clamp = (x, a = 0, b = 1) => Math.min(Math.max(x, a), b);
export const lerp = (a, b, w) => a + (b - a) * w;
// Progress of t through the window [a, b], clamped to [0, 1].
export const seg = (t, a, b) => clamp((t - a) / (b - a));
export const ease = {
  inOut: (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2),
  out: (x) => 1 - Math.pow(1 - x, 3),
  in: (x) => x * x * x,
};
// Fade in over [a, a+f], fade out over [b-f, b].
export const window_ = (t, a, b, f = 0.5) => Math.min(seg(t, a, a + f), 1 - seg(t, b - f, b));

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
export function rampColor(theme, u, alpha = 1) {
  const stops = theme.ramp.map(hex);
  const f = clamp(u) * (stops.length - 1);
  const i = Math.min(Math.floor(f), stops.length - 2), w = f - i;
  const c = stops[i].map((v, j) => Math.round(lerp(v, stops[i + 1][j], w)));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

export function text(ctx, str, x, y, { size = 20, weight = 400, color = '#fff', align = 'left', font = 'sans', alpha = 1, spacing = 0 } = {}) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.font = `${weight} ${size}px ${fonts[font]}`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  if (spacing) ctx.letterSpacing = `${spacing}px`;
  ctx.fillText(str, x, y);
  ctx.restore();
}

// Word-wrapped paragraph; returns the y below the last line.
export function paragraph(ctx, str, x, y, maxWidth, opts = {}) {
  const size = opts.size ?? 20, lh = opts.lineHeight ?? size * 1.45;
  ctx.save();
  ctx.font = `${opts.weight ?? 400} ${size}px ${fonts[opts.font ?? 'sans']}`;
  const words = str.split(' ');
  let line = '';
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (ctx.measureText(test).width > maxWidth && line) {
      text(ctx, line, x, y, { ...opts, size });
      line = w; y += lh;
    } else line = test;
  }
  if (line) { text(ctx, line, x, y, { ...opts, size }); y += lh; }
  ctx.restore();
  return y;
}

export function arrow(ctx, x0, y0, x1, y1, { color = '#fff', width = 2, head = 8, alpha = 1 } = {}) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 1e-3) return;
  const ux = dx / L, uy = dy / L, h = Math.min(head, L * 0.6);
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - ux * h * 0.8, y1 - uy * h * 0.8);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * h - uy * h * 0.5, y1 - uy * h + ux * h * 0.5);
  ctx.lineTo(x1 - ux * h + uy * h * 0.5, y1 - uy * h - ux * h * 0.5);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
