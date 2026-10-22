// The math toolbox: one entry per rule in lib/rules.js. "Used in" is collected from the derivation
// lines of every tutorial, so it stays in sync with the steps without a hand-kept list.
import { RULES, GROUPS } from '../lib/rules.js';

const TUTORIALS = [
  ['data', '§2.0', 'Data and distributions'],
  ['ebm-langevin', '§2.1', 'Energies, scores and Langevin dynamics'],
  ['score-matching', '§2.2', 'Learning the score'],
  ['sde', '§2.3', 'Diffusion as an SDE'],
  ['flow-matching', '§2.4', 'Flows and flow matching'],
  ['likelihood', '§2.5', 'The likelihood view'],
];

const root = document.documentElement;
document.getElementById('theme').onclick = () => {
  const dark = (root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('survey-theme', root.dataset.theme); } catch {}
};

// Each steps.js registers its own symbols on import, so the tutorials are loaded one after another.
const used = {};
for (const [dir, sec, name] of TUTORIALS) {
  const { STEPS } = await import(`../${dir}/steps.js`);
  STEPS.forEach((st, i) => (st.deriv?.lines ?? []).forEach((l, j) => (l.uses ?? []).forEach((id) => {
    const key = `${dir}#${st.id}`;
    const list = (used[id] ??= []);
    let e = list.find((x) => x.key === key);
    if (!e) list.push(e = { key, href: `../${dir}/#${st.id}`, sec, tut: name, step: i + 1, title: st.title, lines: [] });
    e.lines.push(j + 1);
  })));
}

const usedIn = (id) => {
  const list = used[id] ?? [];
  if (!list.length) return '';
  const items = list.map((e) => `<li><a href="${e.href}"><span class="sec">${e.sec}, step ${e.step}</span> ${e.title}</a> <span class="ln">${e.lines.length > 1 ? 'lines' : 'line'} ${e.lines.join(', ')}</span></li>`);
  return `<div class="used"><div class="lab">Used in</div><ul>${items.join('')}</ul></div>`;
};

const entry = (id, r) => `<article class="rule-entry" id="${id}">
    <h3>${r.name}</h3>
    <div class="formula">\\[${r.tex}\\]</div>
    <p class="short">${r.short}</p>
    ${(r.body ?? []).map((p) => `<p>${p}</p>`).join('')}
    ${r.example ? `<div class="ex"><div class="lab">As it appears in a tutorial</div>\\[${r.example}\\]</div>` : ''}
    ${usedIn(id)}
  </article>`;

const rules = Object.entries(RULES).filter(([, r]) => r.kind === 'rule');
const results = Object.entries(RULES).filter(([, r]) => r.kind === 'result');
const count = (g) => rules.filter(([, r]) => r.group === g).length;

document.querySelector('main').innerHTML = `
  <div class="intro">
    <div class="kicker">Tutorials · Reference</div>
    <h1>Math toolbox</h1>
    <p>Every line of a derivation in the tutorials uses one or two rules from calculus, probability or stochastic calculus. This page states each rule once, explains why it holds and shows where it is used. In a tutorial, the Hint button next to a derivation line names the rule that line uses and links here.</p>
    <p>Some lines instead use a result that an earlier step derived. Their hints are marked in purple and lead back to that step. They are listed at the end of this page.</p>
    <nav class="tb-toc">${GROUPS.map(([g, name]) => `<div><div class="lab">${name}</div>${rules.filter(([, r]) => r.group === g).map(([id, r]) => `<a href="#${id}">${r.name}</a>`).join('')}</div>`).join('')}
      <div><div class="lab">Earlier results</div>${results.map(([id, r]) => `<a href="#${id}">${r.name}</a>`).join('')}</div>
    </nav>
  </div>
  ${GROUPS.map(([g, name]) => `<section class="group"><h2>${name} <span class="n">${count(g)} rules</span></h2>${rules.filter(([, r]) => r.group === g).map(([id, r]) => entry(id, r)).join('')}</section>`).join('')}
  <section class="group"><h2>Results from earlier steps</h2>
    <p class="gintro">These are derived in the tutorials themselves. The link on each name goes to the step that derives it.</p>
    ${results.map(([id, r]) => `<article class="rule-entry result" id="${id}">
      <h3><a href="${r.href}">${r.name}</a> <span class="at">${r.at}</span></h3>
      <div class="formula">\\[${r.tex}\\]</div>
      <p class="short">${r.short}</p>
      ${usedIn(id)}
    </article>`).join('')}
  </section>`;

while (!window.MathJax?.typesetPromise) await new Promise((r) => setTimeout(r, 30));
await MathJax.startup.promise;
await MathJax.typesetPromise([document.body]);
if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
