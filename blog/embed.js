// Multi-instance player for the blog companion. Unlike lib/player.js it has no global key
// handler and no window.__anim, so many animations can live on one page.
// Markup: <figure class="anim" data-anim="<id>"></figure>. The module is imported, set up and
// drawn only when the figure comes near the viewport, and it plays only while visible.
import { themes } from '../animations/lib/theme.js';

const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
// On narrow screens the text panel inside the canvas would be scaled to a few pixels, so draw the
// clean variant (no panel, as in the paper stills) cropped to the plot. The caption carries the text.
const narrowQuery = matchMedia('(max-width: 640px)');
const darkQuery = matchMedia('(prefers-color-scheme: dark)');
const players = new Set();

function themeName() {
  const forced = document.documentElement.dataset.theme;
  if (forced === 'light' || forced === 'dark') return forced;
  return darkQuery.matches ? 'dark' : 'light';
}

function onThemeChange() {
  const name = themeName();
  for (const p of players) p.setTheme(name);
}
new MutationObserver(onThemeChange).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
darkQuery.addEventListener('change', onThemeChange);
narrowQuery.addEventListener('change', () => { for (const p of players) p.relayout(); });

async function create(fig) {
  const id = fig.dataset.anim;
  const anim = (await import(new URL(`../animations/${id}/anim.js`, import.meta.url))).default;
  const W = anim.width ?? 1600, H = anim.height ?? 900;
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5);

  const stage = fig.querySelector('.anim-stage');
  const canvas = document.createElement('canvas');
  canvas.className = 'anim-canvas';
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', anim.title);
  stage.replaceChildren(canvas);
  const ctx = canvas.getContext('2d');

  const env = { W, H, dpr, clean: false, params: new URLSearchParams(), theme: themes[themeName()] };
  let state, view;
  // view is the visible region of the 1600x900 frame: the whole frame, or anim.crop in compact mode.
  function layout() {
    env.clean = narrowQuery.matches && !!anim.crop;
    view = env.clean ? anim.crop : { x: 0, y: 0, w: W, h: H };
    stage.style.aspectRatio = `${view.w} / ${view.h}`;
    canvas.width = view.w * dpr;
    canvas.height = view.h * dpr;
    state = anim.setup ? anim.setup(env) : {};
  }
  layout();

  let t = 0, playing = false, visible = false, wantPlay = !reduceMotion, last = null, raf = 0, released = false;

  const bar = document.createElement('div');
  bar.className = 'anim-controls';
  bar.innerHTML = `
    <button class="play" type="button" aria-label="Play or pause"></button>
    <input class="scrub" type="range" min="0" max="${anim.duration}" step="0.01" aria-label="Time in ${anim.title}">
    <span class="time"></span>`;
  const chips = document.createElement('div');
  chips.className = 'anim-chapters';
  for (const c of anim.chapters ?? []) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = c.label;
    b.onclick = () => { t = c.t; render(); };
    chips.appendChild(b);
  }
  fig.insertBefore(bar, fig.querySelector('figcaption'));
  fig.insertBefore(chips, fig.querySelector('figcaption'));
  const $ = (s) => bar.querySelector(s);

  function sync() {
    $('.play').textContent = playing ? '❚❚' : '▶';
    $('.play').setAttribute('aria-pressed', String(playing));
    $('.scrub').value = t;
    $('.time').textContent = `${t.toFixed(1)} / ${anim.duration.toFixed(0)} s`;
    const active = [...(anim.chapters ?? [])].reverse().find((c) => t >= c.t);
    chips.querySelectorAll('button').forEach((b) => b.classList.toggle('on', !!active && b.textContent === active.label));
  }

  function render() {
    if (released) return;
    ctx.setTransform(dpr, 0, 0, dpr, -view.x * dpr, -view.y * dpr);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = env.theme.bg;
    ctx.fillRect(0, 0, W, H);
    anim.draw(ctx, t, env, state);
    sync();
  }

  function frame(now) {
    raf = 0;
    if (!playing) return;
    if (last !== null) t += Math.min((now - last) / 1000, 0.1);
    if (t >= anim.duration) t = 0;
    last = now;
    render();
    raf = requestAnimationFrame(frame);
  }

  function updatePlaying() {
    const should = wantPlay && visible && !document.hidden;
    if (should === playing) return;
    playing = should;
    last = null;
    if (playing && !raf) raf = requestAnimationFrame(frame);
    sync();
  }

  $('.play').onclick = () => { wantPlay = !(wantPlay && playing); if (wantPlay && !visible) visible = true; updatePlaying(); sync(); };
  $('.scrub').oninput = (e) => { t = Number(e.target.value); wantPlay = false; updatePlaying(); render(); };

  // Keyboard shortcuts apply only while focus is inside this figure.
  fig.tabIndex = -1;
  fig.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.code === 'Space' && e.target.tagName !== 'BUTTON') { e.preventDefault(); $('.play').click(); }
    if (e.target.type === 'range') return;
    if (e.code === 'ArrowRight') { t = Math.min(t + 1, anim.duration); render(); }
    if (e.code === 'ArrowLeft') { t = Math.max(t - 1, 0); render(); }
  });
  canvas.addEventListener('click', () => $('.play').click());

  const player = {
    setVisible(v) { visible = v; updatePlaying(); },
    setTheme(name) {
      if (env.theme === themes[name]) return;
      env.theme = themes[name];
      if (anim.onTheme) anim.onTheme(env, state);
      render();
    },
    // Free the backing store of far-away canvases to bound memory with 17 players on one page.
    release() { if (!released) { released = true; canvas.width = 1; canvas.height = 1; } },
    restore() { if (released) { released = false; canvas.width = view.w * dpr; canvas.height = view.h * dpr; render(); } },
    relayout() { if (env.clean !== (narrowQuery.matches && !!anim.crop)) { layout(); if (released) { canvas.width = 1; canvas.height = 1; } render(); } },
    pauseHidden() { updatePlaying(); },
    render,
  };
  players.add(player);
  fig.classList.add('ready');
  render();
  document.fonts?.ready.then(render);
  return player;
}

export function mountAll(root = document) {
  const figs = [...root.querySelectorAll('figure.anim[data-anim]')];
  const made = new Map();
  const near = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const fig = e.target;
      if (e.isIntersecting) {
        if (!made.has(fig)) {
          made.set(fig, create(fig).catch((err) => {
            fig.querySelector('.anim-stage').textContent = `This animation failed to load (${err.message}).`;
            console.error(err);
          }));
        } else made.get(fig).then((p) => p?.restore());
      } else made.get(fig)?.then((p) => p?.release());
    }
  }, { rootMargin: '150% 0px' });
  const seen = new IntersectionObserver((entries) => {
    for (const e of entries) made.get(e.target)?.then((p) => p?.setVisible(e.isIntersecting));
  }, { threshold: 0.35 });
  // The visibility observer may fire before setup finishes, so re-check once a player exists.
  for (const fig of figs) {
    near.observe(fig);
    seen.observe(fig);
  }
  const recheck = () => {
    for (const [fig, pr] of made) pr.then((p) => {
      if (!p) return;
      const r = fig.getBoundingClientRect();
      const vis = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0)) / Math.max(1, r.height);
      p.setVisible(vis >= 0.35);
    });
  };
  addEventListener('scroll', () => { clearTimeout(recheck.id); recheck.id = setTimeout(recheck, 120); }, { passive: true });
  document.addEventListener('visibilitychange', () => { for (const p of players) p.pauseHidden(); });
  setTimeout(recheck, 400);
}
