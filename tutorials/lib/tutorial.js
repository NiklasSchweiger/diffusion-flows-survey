// Shared runtime for the interactive tutorials. A tutorial page calls
//   mount({ symbols, steps, scenes })
// where symbols maps an id to its glossary entry, steps is the ordered list of steps (prose, one
// formula, an optional derivation or check, and the name of a scene), and scenes maps a scene name
// to a factory (viz, api) => { draw(), active?(on) }. Scenes are functions of their controls, not of
// wall-clock time: simulations use makeRng(seed) and restart from the same seed on every change.
import { themes } from '../../animations/lib/theme.js';
import { RULES, GROUPS } from './rules.js';

// ---------- Shared state ----------
export const live = {
  hl: null,          // id of the highlighted symbol, read by every scene
  theme: 'dark',
  reduced: matchMedia('(prefers-reduced-motion: reduce)').matches,
  scenes: [],        // { el, scene, active }
  symbols: {},
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export function palette() {
  return { ...themes[live.theme], mark: css('--mark'), acc: css('--accent'), acc2: css('--accent-2'), line: css('--line'), panel: css('--panel'), ok: css('--ok'), bad: css('--bad'), faint: css('--faint') };
}
export const redrawAll = () => live.scenes.forEach((s) => s.active && s.scene.draw());

// \class tags make each symbol a DOM node inside the MathJax SVG: S('eta') -> \class{sym-eta}{\eta}.
export const S = (id, tex) => `\\class{sym-${id}}{${tex ?? live.symbols[id].tex}}`;
export const CHG = (tex) => `\\class{chg}{${tex}}`;   // the part of a derivation line that changed

// ---------- Canvas helpers ----------
export function makeCanvas(aspect, { maxH = 0.62 } = {}) {
  const cv = document.createElement('canvas'), ctx = cv.getContext('2d');
  const obj = { cv, ctx, w: 0, h: 0 };
  obj.fit = () => {
    // Height follows the aspect ratio but never exceeds a share of the viewport, so canvas and controls fit on one screen.
    let w = cv.clientWidth || 600, h = Math.round(w / aspect);
    const hmax = Math.round(innerHeight * maxH);
    if (h > hmax && innerWidth > 960) h = hmax;
    const d = Math.min(devicePixelRatio || 1, 2);
    cv.style.height = `${h}px`;
    if (cv.width !== Math.round(w * d) || cv.height !== Math.round(h * d)) { cv.width = Math.round(w * d); cv.height = Math.round(h * d); }
    obj.w = w; obj.h = h;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    ctx.clearRect(0, 0, w, h);
  };
  pinnable(obj);
  return obj;
}
// A canvas that follows the cursor can be pinned: a click fixes the probe where it is, so the reader
// can move the sliders or hover the formula without losing the position. A second click releases it.
// The scenes register plain pointermove and pointerleave listeners; while pinned they are not called.
function pinnable(obj) {
  const { cv } = obj, add = cv.addEventListener.bind(cv), moves = [], leaves = [];
  let badge = null;
  obj.pinned = false;
  cv.addEventListener = (type, fn, opts) => {
    if (type === 'pointermove') { if (!moves.length) arm(); moves.push(fn); }
    else if (type === 'pointerleave') leaves.push(fn);
    else add(type, fn, opts);
  };
  add('pointermove', (e) => { if (!obj.pinned) moves.forEach((f) => f(e)); });
  add('pointerleave', (e) => { if (!obj.pinned) leaves.forEach((f) => f(e)); });
  function arm() {
    cv.classList.add('probe');
    add('click', (e) => {
      // Releasing moves the probe to the click; pinning first places it there, which also serves taps.
      moves.forEach((f) => f(e));
      obj.pinned = !obj.pinned;
      cv.classList.toggle('pinned', obj.pinned);
      if (!badge) { badge = document.createElement('div'); badge.className = 'pinbadge'; badge.textContent = 'Pinned. Click the plot to release.'; }
      if (obj.pinned) { cv.after(badge); place(); } else badge.remove();
    });
  }
  function place() {
    const p = cv.offsetParent;
    if (!p || !badge.isConnected) return;
    badge.style.top = `${cv.offsetTop + 8}px`;
    badge.style.right = `${p.clientWidth - cv.offsetLeft - cv.offsetWidth + 8}px`;
  }
  addEventListener('resize', () => badge && place());
}
export function arrow(ctx, x0, y0, x1, y1, color, width = 2, head = 8) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 0.5) return;
  const ux = dx / L, uy = dy / L, h = Math.min(head, L * 0.6);
  ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * h * 0.7, y1 - uy * h * 0.7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * h - uy * h * 0.5, y1 - uy * h + ux * h * 0.5);
  ctx.lineTo(x1 - ux * h + uy * h * 0.5, y1 - uy * h - ux * h * 0.5);
  ctx.closePath(); ctx.fill();
}
// Canvas text with subscripts: "ρ_τ" or "σ_{ℓ+1}" draws the part after the underscore smaller and lower.
export function label(ctx, s, x, y, color, size = 13, align = 'left', weight = 500) {
  const parts = [];
  for (const m of s.matchAll(/_\{([^}]*)\}|_(.)|([^_]+)/gu)) parts.push(m[3] !== undefined ? [m[3], false] : [m[1] ?? m[2], true]);
  const font = (sub) => `${weight} ${sub ? Math.round(size * 0.72) : size}px Inter, system-ui, sans-serif`;
  const widths = parts.map(([t, sub]) => { ctx.font = font(sub); return ctx.measureText(t).width; });
  const total = widths.reduce((a, b) => a + b, 0);
  let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
  ctx.fillStyle = color; ctx.textAlign = 'left';
  parts.forEach(([t, sub], i) => { ctx.font = font(sub); ctx.fillText(t, cx, sub ? y + size * 0.28 : y); cx += widths[i]; });
  return total;
}
// Colour of a symbol's canvas object: the mark colour while that symbol is hovered.
export const hlc = (ids, pal, base) => ([].concat(ids).includes(live.hl) ? pal.mark : base);
export const isHL = (...ids) => ids.includes(live.hl);

// ---------- Controls ----------
export function controls(...items) {
  const el = document.createElement('div');
  el.className = 'controls'; el.append(...items.filter(Boolean));
  return el;
}
// A slider named by its symbol. map turns the raw input value into the parameter (e.g. a log scale).
export function slider(id, { min, max, step, value, fmt = (v) => v.toFixed(2), map = (v) => v, tex }, oninput) {
  const el = document.createElement('label');
  el.className = 'ctl';
  el.innerHTML = `<span class="lab">\\(${S(id, tex)}\\)</span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}"><span class="val"></span>`;
  const inp = el.querySelector('input'), val = el.querySelector('.val');
  inp.setAttribute('aria-label', live.symbols[id]?.name ?? id);
  const upd = () => { const v = map(Number(inp.value)); val.textContent = fmt(v); oninput(v); };
  inp.addEventListener('input', upd);
  val.textContent = fmt(map(Number(value)));
  el.set = (raw) => { inp.value = raw; upd(); };
  return el;
}
export function button(html, onclick, cls = '') {
  const b = document.createElement('button');
  b.type = 'button'; b.className = `btn ${cls}`; b.innerHTML = html; b.onclick = onclick;
  return b;
}
export function toggle(html, on, onchange) {
  const b = button(html, () => { on = !on; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); onchange(on); }, on ? 'on' : '');
  b.setAttribute('aria-pressed', on);
  return b;
}
export const readout = () => Object.assign(document.createElement('div'), { className: 'readout' });
export const note = (html, cls = 'exact') => Object.assign(document.createElement('div'), { className: cls, innerHTML: html });

// Steps a simulation on animation frames while its scene is active and playing. With reduced motion
// it runs to the end at once. step() advances one unit, done() says when to stop.
export function runner({ step, done, draw, interval = 40, perFrame = 1 }) {
  const r = { playing: !live.reduced, active: false, last: null, raf: null };
  const loop = (now) => {
    r.raf = null;
    if (!r.playing || !r.active || done()) return;
    if (r.last === null || now - r.last >= interval) {
      for (let i = 0; i < perFrame && !done(); i++) step();
      draw(); r.last = now;
    }
    r.raf = requestAnimationFrame(loop);
  };
  r.kick = () => {
    if (live.reduced && r.active && r.playing) { while (!done()) step(); draw(); return; }
    if (!r.raf && r.playing && r.active) { r.last = null; r.raf = requestAnimationFrame(loop); }
  };
  r.setActive = (on) => { r.active = on; r.kick(); };
  r.play = button(r.playing ? 'Pause' : 'Play', () => { r.playing = !r.playing; r.play.textContent = r.playing ? 'Pause' : 'Play'; r.kick(); });
  r.one = button('One step', () => { if (r.playing) r.play.click(); if (!done()) step(); draw(); });
  return r;
}

// ---------- Step text, derivations and checks ----------
function stepText(st, i, n) {
  const el = document.createElement('section');
  el.className = 'step'; el.dataset.step = i; el.id = st.id;
  el.innerHTML = `<div class="kicker">Step ${i + 1} of ${n} · ${st.kicker}</div><h2>${st.title}</h2>
    <div class="prose">${st.prose.map((p) => `<p>${p}</p>`).join('')}</div>
    <div class="formula">\\[${st.tex}\\]</div>
    <div class="hint">Hover or tap a symbol to see what it is and where it is in the plot.</div>
    ${st.after ? `<div class="prose">${st.after.map((p) => `<p>${p}</p>`).join('')}</div>` : ''}`;
  if (st.deriv) el.append(derivation(st.deriv, st.tex));
  for (const c of [].concat(st.check ?? [])) el.append(check(c));
  if (st.closing) el.insertAdjacentHTML('beforeend', `<div class="closing">${st.closing}</div>`);
  return el;
}
function derivation(d, stepTex) {
  const goal = d.goal === 'step' ? stepTex : d.goal;   // 'step': the goal is the formula of the step
  const box = document.createElement('div');
  box.className = 'deriv';
  box.innerHTML = `<div class="head">${d.title ?? 'Derivation'} <span class="spacer"></span><button class="btn back" type="button">Back</button><button class="btn next" type="button">Next line</button></div>`;
  // The task stays visible above the lines: what we start from, what we want to show and how.
  if (goal) box.querySelector('.head').insertAdjacentHTML('afterend', `<div class="task">
    ${d.have?.length ? `<div class="tl">We have</div><ul class="have">${d.have.map((h) => `<li>${glue(h)}</li>`).join('')}</ul>` : ''}
    <div class="tl">We want to show</div><div class="goal">\\[${goal}\\]</div>
    ${d.plan ? `<div class="tl">Plan</div><p class="plan">${glue(d.plan)}</p>` : ''}</div>`);
  const rows = d.lines.map((l, i) => {
    const r = document.createElement('div');
    r.className = 'line' + (i ? ' hidden' : '');
    // Lines that use a rule or an earlier result get a hint, closed until the reader asks for it.
    const hint = l.uses?.length ? `<button class="tipbtn" type="button" aria-expanded="false">Hint</button>` : '';
    r.innerHTML = `<div>\\[${l.tex}\\]</div><div class="why">${glue(l.why)} ${hint}</div>${l.uses?.length ? `<div class="tips" hidden>${l.uses.map(tip).join('')}</div>` : ''}`;
    r.querySelector('.tipbtn')?.addEventListener('click', (e) => {
      const b = e.currentTarget, t = r.querySelector('.tips'), open = t.hidden;
      t.hidden = !open; b.setAttribute('aria-expanded', open); b.textContent = open ? 'Hide hint' : 'Hint';
      if (open && !t.dataset.typeset && window.MathJax?.typesetPromise) { t.dataset.typeset = '1'; MathJax.typesetPromise([t]).then(() => wireSymbols(t)); }
    });
    box.append(r);
    return r;
  });
  // Done folds the derivation to its last line, the result; a further click shows all lines again.
  let n = 1, done = false;
  const next = box.querySelector('.next'), back = box.querySelector('.back');
  const state = document.createElement('span');
  state.className = 'state';
  box.querySelector('.head .spacer').before(state);
  const upd = () => {
    const last = n === rows.length;
    rows.forEach((r, i) => { r.classList.toggle('hidden', i >= n || (done && i < n - 1)); r.classList.toggle('new', !done && i === n - 1 && n > 1); });
    box.classList.toggle('done', done);
    state.textContent = done ? 'Complete' : `Line ${n} of ${rows.length}`;
    back.disabled = n === 1 || done;
    next.textContent = done ? 'Show all lines' : last ? 'Done' : 'Next line';
  };
  next.onclick = () => {
    if (done) done = false;
    else if (n === rows.length) done = true;
    else n++;
    upd();
  };
  back.onclick = () => { n = Math.max(1, n - 1); upd(); };
  upd();
  return box;
}
// One hint: the rule or earlier result a line uses, its statement and a link to the full explanation.
function tip(id) {
  const r = RULES[id];
  if (!r) throw new Error(`Unknown rule ${id}`);
  const label = r.kind === 'result' ? `Result from ${r.at}` : `Rule · ${Object.fromEntries(GROUPS)[r.group]}`;
  const link = r.kind === 'result' ? `<a href="${r.href}">Go to the step</a>` : `<a href="../toolbox/#${id}">Full explanation in the toolbox</a>`;
  return `<div class="tip ${r.kind}"><div class="tn">${r.name} <span class="ty">${label}</span></div><div class="st">\\[${r.tex}\\]</div><div class="sh">${glue(r.short)} ${link}</div></div>`;
}
// Multiple choice: { q, opts: [[label, correct]], fb: { true, false } }.
// Numeric: { q, answer, tol, unit?, fb: { true, false } }.
function check(c) {
  const box = document.createElement('div');
  box.className = 'check';
  box.innerHTML = `<div class="q">Check: ${c.q}</div><div class="opts"></div><div class="fb" aria-live="polite"></div>`;
  const fb = box.querySelector('.fb'), opts = box.querySelector('.opts');
  const say = (ok) => { fb.innerHTML = c.fb[ok]; fb.classList.toggle('ok', ok); if (window.MathJax?.typesetPromise) MathJax.typesetPromise([fb]).then(() => wireSymbols(fb)); };
  if (c.opts) {
    c.opts.forEach(([t, ok]) => {
      const b = button(t, () => {
        opts.querySelectorAll('button').forEach((x) => x.classList.remove('right', 'wrong'));
        b.classList.add(ok ? 'right' : 'wrong'); say(ok);
      });
      opts.append(b);
    });
  } else {
    opts.innerHTML = `<span class="num"><input type="text" inputmode="decimal" aria-label="Your answer"><button class="btn" type="button">Check</button></span>`;
    const inp = opts.querySelector('input');
    const go = () => {
      const v = Number(inp.value.replace(',', '.'));
      if (!inp.value.trim() || Number.isNaN(v)) { fb.textContent = 'Enter a number.'; return; }
      const ok = Math.abs(v - c.answer) <= c.tol;
      inp.classList.toggle('right', ok); inp.classList.toggle('wrong', !ok); say(ok);
    };
    opts.querySelector('button').onclick = go;
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  }
  return box;
}

// ---------- Glossary and symbol linking ----------
let card, sticky = null;
function glossary() {
  const g = document.createElement('aside');
  g.className = 'gloss'; g.setAttribute('aria-label', 'Symbols');
  g.innerHTML = '<h3>Symbols, in order of appearance</h3>';
  for (const [id, s] of Object.entries(live.symbols)) {
    const e = document.createElement('div');
    e.className = 'ent'; e.dataset.sym = id; e.tabIndex = 0;
    e.innerHTML = `<div class="t">\\(${s.tex}\\)</div><div><div>${s.name}</div><div class="ty">${s.type}</div>${s.macro ? `<code>paper: ${s.macro}</code>` : ''}</div>`;
    e.onmouseenter = () => setHL(id); e.onmouseleave = () => setHL(null);
    e.onclick = () => {
      const first = document.querySelector(`.steps .formula .sym-${id}`)?.closest('.step');
      first?.scrollIntoView({ behavior: live.reduced ? 'auto' : 'smooth', block: 'start' });
    };
    g.append(e);
  }
  document.body.append(g);
  return g;
}
// Keeps punctuation after inline math on the same line as the math.
const glue = (html) => (html || '').replace(/(\\\((?:[^\\]|\\(?!\)))*\\\)[.,;:)]+)/g, '<span class="nw">$1</span>');
export function setHL(id, anchor) {
  if (live.hl === id && !anchor) return;
  live.hl = id;
  document.querySelectorAll('mjx-container .hl, .gloss .ent.hl').forEach((n) => n.classList.remove('hl'));
  if (id) {
    document.querySelectorAll(`mjx-container .sym-${id}`).forEach((n) => n.classList.add('hl'));
    document.querySelectorAll(`.gloss .ent[data-sym="${id}"]`).forEach((n) => n.classList.add('hl'));
  }
  if (id && anchor) {
    const s = live.symbols[id], r = anchor.getBoundingClientRect();
    const place = () => {
      const left = Math.min(window.scrollX + r.left, window.scrollX + innerWidth - 316);
      card.style.left = `${Math.max(8, left)}px`;
      // Below the symbol, or above it when there is no room underneath.
      const below = r.bottom + 10, h = card.offsetHeight;
      card.style.top = `${window.scrollY + (below + h > innerHeight && r.top - h - 10 > 54 ? r.top - h - 10 : below)}px`;
    };
    // The card text contains \( \) math, typeset on every show; the old math is cleared first.
    if (window.MathJax?.typesetClear) MathJax.typesetClear([card]);
    card.innerHTML = `<div class="nm">${s.name}</div><div class="ty">${s.type} · ${s.shape}</div><div class="df">${glue(s.def)}</div>${s.canvas ? `<div class="cv">In the plot: ${glue(s.canvas)}</div>` : ''}`;
    card.style.display = 'block';
    place();
    if (window.MathJax?.typesetPromise) MathJax.typesetPromise([card]).then(() => { if (live.hl === id) place(); });
  } else card.style.display = 'none';
  redrawAll();
}
export function wireSymbols(root) {
  root.querySelectorAll('mjx-container [class*="sym-"]').forEach((n) => {
    const id = [...n.classList].find((c) => c.startsWith('sym-'))?.slice(4);
    if (!id || !live.symbols[id] || n.closest('.gloss') || n.dataset.wired) return;
    n.dataset.wired = '1';
    n.setAttribute('tabindex', '0'); n.setAttribute('role', 'button'); n.setAttribute('aria-label', live.symbols[id].name);
    n.addEventListener('mouseenter', () => { if (!sticky) setHL(id, n); });
    n.addEventListener('mouseleave', () => { if (!sticky) setHL(null); });
    n.addEventListener('focus', () => setHL(id, n));
    n.addEventListener('blur', () => { if (!sticky) setHL(null); });
    // Tap or click pins the card until the next tap anywhere.
    n.addEventListener('click', (e) => { e.stopPropagation(); sticky = sticky === n ? null : n; setHL(sticky ? id : null, sticky ? n : undefined); });
  });
}

// ---------- Mount ----------
export async function mount({ symbols, steps, scenes, intro }) {
  live.symbols = symbols;
  const root = document.documentElement;
  try { const t = localStorage.getItem('survey-theme'); if (t === 'light' || t === 'dark') root.dataset.theme = t; } catch {}
  live.theme = root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.getElementById('theme').onclick = () => {
    live.theme = live.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = live.theme;
    try { localStorage.setItem('survey-theme', live.theme); } catch {}
    redrawAll();
  };
  card = document.createElement('div'); card.className = 'symcard'; card.setAttribute('role', 'tooltip'); card.style.display = 'none';
  document.body.append(card);
  document.addEventListener('click', () => { if (sticky) { sticky = null; setHL(null); } });
  const gl = glossary();
  document.getElementById('gloss').onclick = (e) => { gl.classList.toggle('open'); e.currentTarget.classList.toggle('on'); };

  // One column, as in the paper: each canvas sits directly under its step's formula.
  const main = document.querySelector('main');
  const col = document.createElement('div');
  col.className = 'steps';
  if (intro) col.insertAdjacentHTML('beforeend', `<div class="intro">${intro}</div>`);
  const texts = steps.map((st, i) => { const t = stepText(st, i, steps.length); col.append(t); return t; });
  main.append(col);
  steps.forEach((st, i) => {
    const el = document.createElement('div');
    el.className = 'viz'; el.dataset.scene = st.scene; el.dataset.step = i;
    texts[i].querySelector('.hint').after(el);
    live.scenes.push({ el, scene: scenes[st.scene](el), active: false });
  });

  // A scene is active while its canvas is on screen, so simulations only run where the reader looks.
  const setScene = (s, on) => { if (s.active !== on) { s.active = on; s.scene.active?.(on); if (on) s.scene.draw(); } };
  const vio = new IntersectionObserver((es) => es.forEach((e) => setScene(live.scenes[Number(e.target.dataset.step)], e.isIntersecting)), { rootMargin: '80px 0px' });
  live.scenes.forEach((s) => vio.observe(s.el));
  // The step counter in the bar follows the step that crosses the middle of the screen.
  const pos = document.querySelector('.bar .pos');
  const tio = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting && pos) pos.textContent = `Step ${Number(e.target.dataset.step) + 1} of ${steps.length}`; }), { rootMargin: '-45% 0px -45% 0px' });
  texts.forEach((t) => tio.observe(t));
  if (pos) pos.textContent = `Step 1 of ${steps.length}`;

  while (!window.MathJax?.typesetPromise) await new Promise((r) => setTimeout(r, 30));
  await MathJax.startup.promise;
  await MathJax.typesetPromise([document.body]);
  wireSymbols(document.body);
  addEventListener('resize', redrawAll);
  redrawAll();
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}
