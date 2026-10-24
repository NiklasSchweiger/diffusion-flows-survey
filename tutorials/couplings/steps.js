// Content of the §2.6 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex,
// section "Couplings, optimal transport, Schrödinger bridges and control"; S(id) tags a symbol from
// symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), pt = S('pt'), u = S('vel'), pi = S('pi'), x0 = S('x0'), x1 = S('x1'), T = S('T');
const xd = S('x', '\\x_{t+\\dd t}');
const X = S('X'), eps = S('eps'), H = S('H'), P = S('P'), Q = S('Q'), et = S('epst'), c = S('u'), h = S('h'), r = S('r'), be = S('beta');
const ps = S('pstar'), Z = S('Z'), W22 = S('W2'), KE = S('KE');
const d10 = `${x1} - ${x0}`;

export const INTRO = `
  <div class="kicker">Survey §2.6 · Interactive tutorial</div>
  <h1>Couplings, transport and control</h1>
  <p>Every earlier tutorial paired noise and data independently and sampled towards the data distribution itself. This tutorial changes both choices. Changing the pairing of noise and data, the coupling, changes the velocity field that flow matching learns, and the optimal coupling gives straight paths that one Euler step integrates exactly. Optimal transport, rectification and Schrödinger bridges are three views of this idea. Changing the target at the end of the path leads to stochastic optimal control, the mathematics behind reward fine-tuning and guided sampling in the later chapters of the survey. Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>The convention is that of the survey: \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data, on the linear path \\(\\alpha_t = t\\), \\(\\sigma_t = 1 - t\\). The data are the Gaussian mixtures of the earlier tutorials. In one dimension the optimal coupling is a quantile map, and for a quadratic reward the tilted target and the function \\(h_t\\) are again Gaussian mixtures, so almost everything on this page is exact. The exceptions are the minibatch statistics of step 1 and the Monte Carlo comparisons of step 7, which are estimated from fixed samples, and the page says so where they appear.</p>
  <ol class="toc">
    <li><a href="#couplings">Couplings are a free choice</a></li>
    <li><a href="#field">The coupling changes the field</a></li>
    <li><a href="#energy">Optimal transport as least kinetic energy</a></li>
    <li><a href="#rectify">Rectification never increases cost</a></li>
    <li><a href="#bridge">Schrödinger bridges are entropic optimal transport</a></li>
    <li><a href="#control">Changing the target: control and Doob’s h-transform</a></li>
    <li><a href="#estimate">Estimating h while sampling</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'couplings', kicker: 'Pairing noise and data', title: 'Couplings are a free choice', scene: 'couplings',
    prose: [
      'Flow matching needs pairs \\((\\x_0, \\x_1)\\) with \\(\\x_0\\sim\\N(0, I)\\) and \\(\\x_1\\sim\\pdata\\). Nothing requires the two to be independent. Any joint distribution π with these two marginals is called a coupling, and each coupling gives an interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) whose ends are noise and data. Step 2 shows that flow matching still works with every one of them.',
      'Minibatch optimal transport is the coupling used in practice. Split the noise and the data samples into batches of size \\(B\\), and inside each batch pair them by the permutation with the smallest total squared distance, the second line of the formula. The plot joins 128 noise points (open circles) to 128 data points (dots) by straight lines. Move \\(B\\) from 1 to 128. The lines get shorter, fewer of them cross, and the readout reports how much of the regression target \\(\\x_1 - \\x_0\\) remains random once \\(\\x_t\\) is known.',
    ],
    tex: `\\begin{aligned} ${x} &= ${S('x', '\\alpha_t')}\\,${x1} + \\sigma_t\\,${x0}, \\qquad (${x0}, ${x1})\\sim${pi}, \\quad ${pi}(\\cdot, \\R^d) = \\N(0, I),\\ ${pi}(\\R^d, \\cdot) = \\pdata \\\\ ${S('sigma')} &= \\arg\\min_{\\varsigma}\\ \\sum_{i=1}^{${S('B')}} \\big\\|${S('x1', '\\x_1^{\\varsigma(i)}')} - ${S('x0', '\\x_0^{i}')}\\big\\|^2 \\end{aligned}`,
    after: ['Even the optimal pairing for the squared distance leaves a few crossing lines, because two lines can cross at different times. Step 3 shows that the particles of the optimal coupling never meet at the same time, which is what matters for a velocity field. The share of the target that stays random given \\(\\x_t\\) is the part of the flow-matching loss that no model can remove, the floor of §2.4. Lower variance means less noisy gradients and a field that is easier to fit.'],
    check: {
      q: 'What is minibatch OT with \\(B = 1\\)?',
      opts: [['The independent coupling: each batch has one pair, so nothing is reassigned', true], ['The optimal coupling, because one point is always matched optimally', false], ['It is undefined', false]],
      fb: { true: 'Right. A batch of one noise point and one data point has a single permutation, so the pairs stay as they were drawn, independently. Set B = 1 and compare the readout with the independent pairing of §2.4.', false: 'A batch of size one contains one noise point and one data point, and the only permutation of one element is the identity. The pairs stay as they were drawn, independently.' },
    },
  },
  {
    id: 'field', kicker: 'The same proof, another field', title: 'The coupling changes the field', scene: 'field',
    prose: [
      'The marginal velocity of §2.4 averages the conditional velocities of all pairs through a point. With a coupling π the average is taken under π, the first line of the formula, and the derivation below shows that it still generates the marginal path of that coupling. The proof never used that \\(\\x_0\\) and \\(\\x_1\\) are independent. Flow matching on pairs drawn from π therefore learns this field and samples from \\(\\pdata\\).',
      'The field does change. The upper panel shows the independent coupling in one dimension: the heat map is its \\(p_t\\) over \\((t, x)\\), and the curves are paths of its exact velocity. The lower panel shows the optimal coupling, which pairs each noise quantile with the same data quantile through the map \\(T\\). Its paths are the straight conditional lines themselves, the second line of the formula. Move \\(K\\): the bold polygons are \\(K\\)-step Euler runs, and the strips on the right compare their end points with \\(\\pdata\\). On the optimal coupling every \\(K\\) is exact.',
    ],
    tex: `\\begin{aligned} ${u}(\\x) &= \\E_{${pi}}\\big[\\,${d10}\\;\\big|\\;${x} = \\x\\,\\big] \\\\ \\text{optimal } ${pi}:\\quad ${u}\\big((1 - ${t})${x0} + ${t}\\,${T}(${x0})\\big) &= ${T}(${x0}) - ${x0} \\end{aligned}`,
    deriv: {
      have: ['A coupling π of \\(\\N(0, I)\\) and \\(\\pdata\\), and \\(\\x_t = (1 - t)\\x_0 + t\\x_1\\) with \\((\\x_0, \\x_1)\\sim\\pi\\).', 'The field \\(\\bm u_t(\\x) = \\E_\\pi[\\x_1 - \\x_0\\mid\\x_t = \\x]\\).'],
      goal: `\\partial_t ${pt} + \\diver\\big(${pt}\\,${u}\\big) = 0`,
      plan: 'Repeat the derivation of §2.4 with test functions, and note at which line the coupling enters. Nowhere is independence needed.',
      title: 'Why every coupling gives a valid field', lines: [
      { tex: `\\frac{\\dd}{\\dd ${t}}\\,\\E\\,\\varphi(${x}) = \\E\\big[\\nabla\\varphi(${x})\\cdot(${d10})\\big]`, why: 'Differentiate along each straight line, then average over the pairs. The pairs are drawn from π, and this is the only place where π appears.', uses: ['chain-rule', 'expectation'] },
      { tex: `= \\E\\big[\\nabla\\varphi(${x})\\cdot${CHG(`\\E_{${pi}}[${d10}\\mid ${x}]`)}\\big] = \\int\\nabla\\varphi\\cdot${u}\\;${pt}\\,\\dd\\x`, why: 'Condition on \\(\\x_t\\): \\(\\nabla\\varphi(\\x_t)\\) is a function of \\(\\x_t\\), so it can be taken out of the inner expectation. The inner expectation is the field.', uses: ['cond-exp'] },
      { tex: `\\int\\varphi\\,\\partial_t ${pt}\\,\\dd\\x = -\\int\\varphi\\,\\diver\\big(${pt}\\,${u}\\big)\\,\\dd\\x`, why: 'Integrate by parts; the boundary term vanishes because φ vanishes at infinity.', uses: ['ibp'] },
      { tex: `\\partial_t ${pt} + \\diver\\big(${pt}\\,${u}\\big) = 0`, why: 'The identity holds for every test function. This is the continuity equation, for the marginal path of π.', uses: ['test-functions', 'continuity'] },
    ] },
    after: ['What does not carry over is the link to the score. The velocity–score formula of §2.4 used that \\(\\x_t\\) given \\(\\x_1\\) is Gaussian with mean \\(\\alpha_t\\x_1\\), which holds only when \\(\\x_0\\) is independent of \\(\\x_1\\). A model trained on OT pairs is a velocity model and nothing else.'],
    check: {
      q: 'Does the formula \\(\\bm u_t = (\\dot\\alpha_t/\\alpha_t)\\x + \\tfrac12\\sigma_t^2\\dot\\lambda_t\\nabla\\log p_t\\) of §2.4 hold for the optimal coupling, with its own \\(p_t\\)?',
      opts: [['No: it requires \\(\\x_t\\) given \\(\\x_1\\) to be \\(\\N(\\alpha_t\\x_1, \\sigma_t^2 I)\\)', true], ['Yes, it holds for every coupling', false], ['Only at \\(t = 0.5\\)', false]],
      fb: { true: 'Right. Under the optimal coupling \\(\\x_0 = T^{-1}(\\x_1)\\) is a function of \\(\\x_1\\), so \\(\\x_t\\) given \\(\\x_1\\) is a single point. The readout evaluates the formula with the exact OT density and finds errors of order 1 or more, against about 1e-15 for the independent coupling.', false: 'The formula comes from Tweedie’s identity, which needs the conditional law of \\(\\x_t\\) given \\(\\x_1\\) to be Gaussian with variance \\(\\sigma_t^2\\). Under the optimal coupling \\(\\x_0\\) is a function of \\(\\x_1\\) and this law is a point mass. The readout shows the size of the mismatch.' },
    },
  },
  {
    id: 'energy', kicker: 'Benamou–Brenier', title: 'Optimal transport as least kinetic energy', scene: 'energy',
    prose: [
      'Why does the optimal coupling give straight paths? Benamou and Brenier showed that \\(W_2^2\\), the smallest mean squared distance between paired noise and data, is also the least kinetic energy a velocity field needs to carry \\(\\N(0, I)\\) to \\(\\pdata\\), the formula below. The constraint is the continuity equation, so this is flow matching with one extra criterion that selects a single field among all fields that move the mass.',
      'The derivation shows that no particle can do better than a straight line at constant speed, and that the optimal map achieves this for every particle at once. The upper panel shows the densities \\(p_t\\) of the two couplings at the time of the slider. The lower panel shows the kinetic energy per unit time, \\(\\int u_t^2 p_t\\,\\dd x\\). For the optimal coupling it is flat at \\(W_2^2\\); for the independent coupling it is computed by quadrature from the exact field, and the area under it is the energy \\(\\mathcal E\\) in the readout.',
    ],
    tex: `${W22} = \\min_{(${pt}, ${u})}\\ ${KE},\\qquad ${KE} = \\int_0^1\\!\\!\\int\\|${u}(\\x)\\|^2\\,${pt}(\\x)\\,\\dd\\x\\,\\dd ${t}\\quad\\text{s.t.}\\quad \\partial_t ${pt} + \\diver(${pt}\\,${u}) = 0`,
    deriv: {
      have: ['A field \\(\\bm u_t\\) that generates a path \\(p_t\\) from \\(\\N(0, I)\\) to \\(\\pdata\\), and its ODE paths \\(\\bm X_t\\).', 'The definition \\(W_2^2 = \\min_\\pi\\E_\\pi\\|\\x_1 - \\x_0\\|^2\\).'],
      goal: `${KE} \\ge ${W22}, \\text{ with equality for the straight lines of } ${T}`,
      plan: 'Write the energy as an average over particles. For one particle, the energy of its path is at least the squared length of its chord. The chords define a coupling, so their mean is at least \\(W_2^2\\).',
      title: 'Why straight lines at constant speed are optimal', lines: [
      { tex: `${KE} = \\int_0^1\\E\\,\\|${u}(${X})\\|^2\\,\\dd ${t} = \\E\\int_0^1\\big\\|\\dot{\\bm X}_t\\big\\|^2\\,\\dd ${t}`, why: 'The ODE paths have law \\(p_t\\), so the integral against \\(p_t\\) is an average over particles, and \\(\\bm u_t(\\bm X_t)\\) is the speed of each particle.', uses: ['transport', 'expectation'] },
      { tex: `\\int_0^1\\big\\|\\dot{\\bm X}_t\\big\\|^2\\dd ${t} \\;\\ge\\; \\Big\\|\\int_0^1\\dot{\\bm X}_t\\,\\dd ${t}\\Big\\|^2 = ${CHG(`\\big\\|${S('X', '\\bm X_1')} - ${S('X', '\\bm X_0')}\\big\\|^2`)}`, why: 'Jensen’s inequality for the convex function \\(\\|\\cdot\\|^2\\) and the uniform average over t. Equality holds exactly when the speed is constant, a straight line at constant speed.', uses: ['jensen', 'ftc'] },
      { tex: `${KE} \\ge \\E\\,\\big\\|${S('X', '\\bm X_1')} - ${S('X', '\\bm X_0')}\\big\\|^2 \\ge ${W22}`, why: 'The pair \\((\\bm X_0, \\bm X_1)\\) has the marginals \\(\\N(0, I)\\) and \\(\\pdata\\), so it is a coupling, and no coupling has a smaller mean squared distance than \\(W_2^2\\).', uses: ['expectation'] },
      { tex: `\\bm X_t = (1 - ${t})${x0} + ${t}\\,${T}(${x0}):\\quad ${KE} = \\E\\,\\|${T}(${x0}) - ${x0}\\|^2 = ${W22}`, why: 'The OT paths are straight at constant speed (equality in Jensen) and their chords form the optimal coupling (equality in the last step). In 1D they also keep their order, so two particles are never at the same place at the same time.', uses: ['transport'] },
    ] },
    after: ['The readout shows the chain \\(W_2^2 \\le \\mathcal E \\le \\E\\|\\x_1 - \\x_0\\|^2\\) for the independent coupling. The field is cheaper than the random lines it averages, and step 4 turns this observation into an algorithm.'],
    check: {
      q: 'The independent field needs energy 1.02, while its random lines have mean squared length 3.27. Why is the field cheaper than the lines it averages?',
      opts: [['Averaging the velocities through a point cancels opposite directions, and \\(\\E\\|\\E[\\bm v\\mid\\x_t]\\|^2\\le\\E\\|\\bm v\\|^2\\)', true], ['The field moves fewer particles', false], ['The quadrature misses the tails', false]],
      fb: { true: 'Right. At every t, \\(\\E\\|\\bm u_t(\\x_t)\\|^2 = \\E\\|\\E[\\x_1 - \\x_0\\mid\\x_t]\\|^2\\le\\E\\|\\x_1 - \\x_0\\|^2\\), by Jensen for the conditional expectation. Lines that cross in opposite directions cancel in the average.', false: 'The field carries the same mass. Through a point, lines arrive in many directions, and their average is shorter than a typical line: \\(\\E\\|\\E[\\bm v\\mid\\x_t]\\|^2\\le\\E\\|\\bm v\\|^2\\) by Jensen. Compare the dashed line of the random lines with the curve in the lower panel.' },
    },
  },
  {
    id: 'rectify', kicker: 'Reflow', title: 'Rectification never increases cost', scene: 'rectify',
    prose: [
      'Integrating the marginal ODE from each noise point gives a new pairing \\((\\bm X_0, \\bm X_1)\\) of noise and data, a deterministic coupling. Rectified flow retrains flow matching on these pairs. The formula shows, for every convex cost, that the new pairs are at least as close as the old ones, with the kinetic energy of the field in between.',
      'The plot uses the 2D mixture of §2.2. Faint lines are random pairs of the independent coupling. Bold curves are paths of its exact marginal velocity, integrated by RK4 from the same noise points, and the dashed chords join their ends: these are the rectified pairs. Press "Play" or move \\(t\\) to see the particles travel. The three bars on the right are the three terms of the chain, estimated from 1500 noise points.',
    ],
    tex: `\\E\\,${S('c')}\\big(${S('X', '\\bm X_1')} - ${S('X', '\\bm X_0')}\\big)\\;\\le\\;\\int_0^1\\E\\,${S('c')}\\big(${u}(${X})\\big)\\,\\dd ${t}\\;\\le\\;\\E\\,${S('c')}\\big(${d10}\\big)`,
    deriv: {
      have: ['The independent coupling, its marginal velocity \\(\\bm u_t(\\x) = \\E[\\x_1 - \\x_0\\mid\\x_t = \\x]\\) and its ODE paths \\(\\bm X_t\\) from \\(\\bm X_0 = \\x_0\\).', 'A convex cost c.'],
      goal: `\\E\\,${S('c')}(\\bm X_1 - \\bm X_0) \\le \\E\\,${S('c')}(${d10})`,
      plan: 'Write the new displacement as the time integral of the velocity and apply Jensen twice: once over time, once over the pairs through a point.',
      title: 'Two applications of Jensen’s inequality', lines: [
      { tex: `${S('X', '\\bm X_1')} - ${S('X', '\\bm X_0')} = \\int_0^1 ${u}(${X})\\,\\dd ${t}`, why: 'The fundamental theorem of calculus along one ODE path.', uses: ['ftc'] },
      { tex: `${S('c')}\\Big(\\int_0^1 ${u}(${X})\\,\\dd ${t}\\Big) \\le ${CHG(`\\int_0^1 ${S('c')}\\big(${u}(${X})\\big)\\,\\dd ${t}`)}`, why: 'Jensen’s inequality for the convex cost and the uniform average over t. Take expectations of both sides.', uses: ['jensen'] },
      { tex: `\\E\\,${S('c')}\\big(${u}(${X})\\big) = \\E\\,${S('c')}\\big(${u}(${x})\\big)`, why: '\\(\\bm X_t\\) and \\(\\x_t\\) have the same law \\(p_t\\), because the marginal velocity generates the marginal path.', uses: ['continuity'] },
      { tex: `${S('c')}\\big(\\E[${d10}\\mid ${x}]\\big) \\le ${CHG(`\\E\\big[${S('c')}(${d10})\\mid ${x}\\big]`)}`, why: 'Jensen’s inequality for the conditional expectation. Average over \\(\\x_t\\) and integrate over t; the right side does not depend on t.', uses: ['jensen', 'cond-exp'] },
    ] },
    after: ['Repeating the procedure straightens the paths further, which is reflow in §6 of the survey. Each round needs a full training run and a dataset of ODE solutions, and the errors of the learned field accumulate, so in practice one or two rounds are used.'],
    check: {
      q: 'In one dimension, what coupling does a single rectification of the independent coupling give?',
      opts: [['The optimal coupling, because a 1D flow map keeps the order of the particles', true], ['A coupling halfway between the independent and the optimal one', false], ['The independent coupling again', false]],
      fb: { true: 'Right. Paths of a 1D ODE cannot cross, so the flow map is increasing. An increasing map that pushes \\(\\N(0, 1)\\) to \\(\\pdata\\) is unique, and it is the quantile map T of step 2. In 2D the flow map is not the optimal map, which is why the chords here still cross a little.', false: 'Two paths of a 1D ODE cannot cross, so \\(\\bm X_1\\) is an increasing function of \\(\\bm X_0\\). The only increasing map from \\(\\N(0, 1)\\) to \\(\\pdata\\) is the quantile map of step 2, the optimal one.' },
    },
  },
  {
    id: 'bridge', kicker: 'Stochastic transport', title: 'Schrödinger bridges are entropic optimal transport', scene: 'bridge',
    prose: [
      'Optimal transport is deterministic: each noise point has one partner. Its stochastic counterpart replaces kinetic energy by relative entropy with respect to Brownian motion \\(Q\\) with noise level ε, and asks for the path measure \\(P\\) closest to \\(Q\\) that starts at \\(\\N(0, I)\\) and ends at \\(\\pdata\\). This is the Schrödinger bridge problem. The derivation splits it into a choice of endpoints and a choice of paths between them. The paths are Brownian bridges, and the endpoints solve optimal transport with an entropy penalty, the formula below.',
      'In one dimension this problem can be solved exactly on a grid by Sinkhorn iterations. The left panel shows the coupling π over \\((x_0, x_1)\\) as a heat map, with the optimal map \\(T\\) as a thin curve. The right panel shows 40 bridges drawn from it. Move ε: at small ε the coupling concentrates on the curve of \\(T\\) and the bridges become the straight lines of step 2, while at large ε the coupling spreads towards the independent one.',
    ],
    tex: `\\begin{aligned} &\\min_{${pi}}\\ \\E_{${pi}}\\|${d10}\\|^2 - 2${eps}^2\\,${H} \\\\ &${x} = (1 - ${t})${x0} + ${t}\\,${x1} + ${eps}\\sqrt{${t}(1 - ${t})}\\;${S('z')} \\end{aligned}`,
    deriv: {
      have: ['The reference \\(Q\\): Brownian motion \\(\\dd\\x_t = \\varepsilon\\,\\dd\\W_t\\) from \\(\\N(0, I)\\), so its endpoint law is \\(Q_{01}(\\x_0, \\x_1) = \\N(\\x_0; 0, I)\\,\\N(\\x_1; \\x_0, \\varepsilon^2 I)\\).', 'Path measures \\(P\\) with ends \\(P_0 = \\N(0, I)\\) and \\(P_1 = \\pdata\\).'],
      goal: `\\min_{${P}}\\KL(${P}\\,\\|\\,${Q}) \\;\\Leftrightarrow\\; \\min_{${pi}}\\ \\E_{${pi}}\\|${d10}\\|^2 - 2${eps}^2\\,${H}`,
      plan: 'Split the KL divergence of path measures into endpoints and bridges, choose the bridges of Q, and expand the endpoint term.',
      title: 'From the bridge problem to entropic OT', lines: [
      { tex: `\\KL(${P}\\,\\|\\,${Q}) = \\KL(${P}_{01}\\,\\|\\,${Q}_{01}) + \\E_{${P}_{01}}\\KL\\big(${P}^{\\x_0\\x_1}\\,\\big\\|\\,${Q}^{\\x_0\\x_1}\\big)`, why: 'The chain rule of the KL divergence: first the joint law of the two ends, then the law of the path given its ends.', uses: ['kl'] },
      { tex: `${P}^{\\x_0\\x_1} = ${Q}^{\\x_0\\x_1}\\;\\Rightarrow\\;\\text{second term} = ${CHG('0')}`, why: 'The second term is at least zero and vanishes when the paths given their ends are the Brownian bridges of Q. These are the bridges in the formula: the straight line plus \\(\\varepsilon\\sqrt{t(1-t)}\\,\\bm z\\).', uses: ['kl', 'brownian'] },
      { tex: `\\KL(${pi}\\,\\|\\,${Q}_{01}) = -${H} - \\E\\log\\N(${x0}; 0, I) + \\frac{\\E_{${pi}}\\|${d10}\\|^2}{2${eps}^2} + \\text{const}`, why: 'Expand \\(\\log Q_{01}\\) with the two Gaussian densities. The term in \\(\\x_0\\) alone is fixed by the marginal \\(\\N(0, I)\\).', uses: ['log-rules', 'gauss'] },
      { tex: `2${eps}^2\\,\\KL(${pi}\\,\\|\\,${Q}_{01}) = ${CHG(`\\E_{${pi}}\\|${d10}\\|^2 - 2${eps}^2\\,${H}`)} + \\text{const}`, why: 'Multiply by \\(2\\varepsilon^2\\) and drop the constants. The remaining problem is optimal transport with an entropy bonus.', uses: ['expectation'] },
    ] },
    after: ['As ε → 0 the Schrödinger bridge recovers the Benamou–Brenier solution of step 3. Diffusion Schrödinger bridges solve the problem for images by alternately fitting forward and backward SDEs to the two marginals, which is the same alternation as the Sinkhorn iterations of this plot, with learned fields in place of the grid.'],
    check: {
      q: 'What does the Sinkhorn coupling become as ε grows large?',
      opts: [['The independent coupling: the entropy term dominates and the mutual information goes to 0', true], ['The optimal coupling', false], ['A coupling with all mass on the diagonal \\(x_1 = x_0\\)', false]],
      fb: { true: 'Right. For large ε the transport cost hardly matters, and the coupling with the largest entropy among those with fixed marginals is the independent one. Set ε = 3: the mutual information in the readout is close to 0, and the transport cost approaches that of the random pairs.', false: 'The weight of the entropy grows with \\(\\varepsilon^2\\). With fixed marginals, maximal entropy means zero mutual information, which is the independent coupling. Move ε to its largest value and read the mutual information.' },
    },
  },
  {
    id: 'control', kicker: 'Reward tilting', title: 'Changing the target: control and Doob’s h-transform', scene: 'control',
    prose: [
      'Now keep the pretrained sampler but change where it should end. A base process \\(\\dd\\x_t = \\bm b_t\\,\\dd t + \\varepsilon_t\\,\\dd\\W_t\\) samples \\(\\pdata\\). We want samples that score highly under a reward \\(r\\) without moving far from the base model, measured by the KL divergence between path measures. By the Girsanov theorem this divergence is the quadratic cost of an extra drift \\(\\varepsilon_t\\bm c_t\\), so the problem is one of stochastic optimal control. The derivation shows that its solution samples the tilted target \\(p^\\star\\propto\\pdata\\,e^{r/\\beta}\\), and that the optimal control is the gradient of \\(\\log h_t\\), the formula below.',
      'The plot uses the 1D mixture as \\(\\pdata\\), the reverse SDE with the exact score as base process and the reward \\(r(x) = -(x - a)^2/2\\). For this reward \\(h_t\\) and \\(p^\\star\\) are again Gaussian mixtures, so the control is exact. Grey paths are base samples, bold paths are controlled samples driven by the same noise. On the right, the histogram of 1000 controlled end points is compared with \\(p^\\star\\) (bold) and \\(\\pdata\\) (thin). Move β and \\(a\\).',
    ],
    tex: `\\begin{aligned} \\dd${x} &= \\big(${S('b')} + ${et}\\,${c}\\big)\\dd ${t} + ${et}\\,\\dd${S('W')}, \\qquad ${c}^\\star = ${et}\\,\\nabla\\log ${h} \\\\ ${h}(\\x) &= \\E\\big[e^{${r}(${x1})/${be}}\\,\\big|\\,${x} = \\x\\big], \\qquad ${ps} \\propto \\pdata\\,e^{${r}/${be}} \\end{aligned}`,
    deriv: {
      have: ['The base path measure \\(P\\) and the controlled one \\(P^{\\bm c}\\), whose divergence is \\(\\KL(P^{\\bm c}\\|P) = \\E_{P^{\\bm c}}\\int_0^1\\tfrac12\\|\\bm c_t\\|^2\\dd t\\) by the Girsanov theorem.', 'The objective \\(J(\\bm c) = \\E_{P^{\\bm c}}\\,r(\\x_1) - \\beta\\,\\KL(P^{\\bm c}\\|P)\\).'],
      goal: `J(${c}) = -${be}\\,\\KL(P^{${c}}\\,\\|\\,P^\\star) + ${be}\\log ${Z}, \\qquad ${c}^\\star = ${et}\\nabla\\log ${h}`,
      plan: 'Combine the reward and the KL term into one KL divergence to a reweighted path measure \\(P^\\star\\). Then find the drift of \\(P^\\star\\) from its transition densities.',
      title: 'From the control objective to Doob’s h-transform', lines: [
      { tex: `J(${c}) = -${be}\\,\\E_{P^{${c}}}\\Big[\\log\\frac{\\dd P^{${c}}}{\\dd P} - \\frac{${r}(${x1})}{${be}}\\Big] = -${be}\\,\\E_{P^{${c}}}\\Big[\\log\\frac{\\dd P^{${c}}}{${CHG(`\\dd P\\,e^{${r}(${x1})/${be}}/${Z}`)}}\\Big] + ${be}\\log ${Z}`, why: 'Write the KL divergence as an expectation of a log ratio, pull \\(r/\\beta\\) inside the logarithm and normalize with \\(Z = \\E_P\\,e^{r(\\x_1)/\\beta}\\).', uses: ['kl', 'log-rules'] },
      { tex: `\\frac{\\dd P^\\star}{\\dd P} = \\frac{e^{${r}(${x1})/${be}}}{${Z}}:\\qquad J(${c}) = -${be}\\,${CHG(`\\KL(P^{${c}}\\,\\|\\,P^\\star)`)} + ${be}\\log ${Z}`, why: 'The bracket is the log ratio to the reweighted path measure \\(P^\\star\\). J is largest when the KL divergence is zero, so the controlled process should have the law \\(P^\\star\\). Its end has density \\(\\propto\\pdata\\,e^{r/\\beta} = p^\\star\\).', uses: ['kl'] },
      { tex: `P^\\star(${xd}\\mid ${x}) = P(${xd}\\mid ${x})\\;\\frac{${S('h', 'h_{t+\\dd t}')}(${xd})}{${h}(${x})}`, why: 'Bayes’ rule for the reweighted path measure. The weight of the future given the present is \\(h_t\\), the expected reweighting factor, so each transition is reweighted by the ratio of two values of h.', uses: ['bayes', 'cond-exp'] },
      { tex: `\\text{drift of } P^\\star = ${S('b')} + ${CHG(`${et}^2\\nabla\\log ${h}`)}`, why: 'A Gaussian transition with mean \\(\\x + \\bm b_t\\dd t\\) and variance \\(\\varepsilon_t^2\\dd t\\), multiplied by \\(e^{\\log h}\\) expanded to first order around \\(\\x\\), is a Gaussian whose mean is shifted by \\(\\varepsilon_t^2\\nabla\\log h_t\\,\\dd t\\). In units of \\(\\varepsilon_t\\) this is \\(\\bm c^\\star_t\\).', uses: ['gauss'] },
    ] },
    after: ['The readout compares the control cost \\(\\E\\int\\tfrac12\\|\\bm c_t\\|^2\\dd t\\) with \\(\\KL(p^\\star\\|\\pdata)\\). They agree up to the time discretization, because \\(P^\\star\\) and \\(P\\) differ only through the end point. One detail of the plot: the sampler starts at t = 0.03 rather than at pure noise, where \\(h_t\\) is close to, but not exactly, constant. The plot therefore also tilts the starting points by \\(h\\), and the readout reports how much that costs.'],
    check: {
      q: 'The control cannot change the initial noise. Why does it still reach \\(p^\\star\\) in the survey’s argument?',
      opts: [['With memoryless noise \\(\\x_1\\) is independent of \\(\\x_0\\), so \\(h_0\\) is constant and \\(P^\\star\\) starts from \\(\\N(0, I)\\)', true], ['Because the control is large near t = 0', false], ['Because the reward does not depend on \\(\\x_0\\)', false]],
      fb: { true: 'Right. \\(P^\\star\\) starts from \\(\\N(0, I)\\,h_0/Z\\). This equals the noise law only if \\(h_0\\) is constant, which holds when the end of the base process does not depend on its start, the memoryless property of §2.4. Here h at t = 0.03 still varies by the amount in the readout, but the KL of the tilt is tiny.', false: 'The start of \\(P^\\star\\) has density \\(\\N(0, I)\\,h_0/Z\\). For the uncontrolled start to be right, \\(h_0\\) must be constant, which requires \\(\\x_1\\) to be independent of \\(\\x_0\\) under the base process. The memoryless noise level is chosen for exactly this.' },
    },
  },
  {
    id: 'estimate', kicker: 'Inference-time control', title: 'Estimating h while sampling', scene: 'estimate',
    prose: [
      'In practice \\(h_t\\) is unknown. Fine-tuning methods learn the control with a network, as §4 of the survey describes. Inference-time methods estimate \\(h_t\\) during sampling instead. Guidance replaces the expectation in \\(h_t\\) by its value at the denoiser output \\(\\hat\\x_1\\), the first formula. Sequential Monte Carlo runs \\(N\\) base samplers side by side, weights them by the ratio of successive estimates of \\(h\\), resamples when the weights become uneven and multiplies by the exact factor \\(e^{r/\\beta}\\) at the end. The weights correct whatever the estimate gets wrong.',
      'The three panels compare the exact control of step 6, guidance and SMC on the same reward, each with its histogram of end points against \\(p^\\star\\). The control and guidance panels use 1000 samples. The SMC panel uses 600 independent runs with \\(N\\) particles each, returning one particle per run. Move \\(N\\): one particle is the base sampler, and from a few particles on SMC approaches \\(p^\\star\\). Guidance does not improve with any setting; its error is a bias of the estimate.',
    ],
    tex: `\\begin{aligned} \\text{guidance:}\\quad ${h}(\\x) &\\approx e^{${r}(${S('hx1')}(\\x, ${t}))/${be}} \\\\ \\text{SMC:}\\quad w^i &\\leftarrow w^i\\,\\frac{\\tilde h_{t+\\dd t}(\\x^i_{t+\\dd t})}{\\tilde h_{t}(\\x^i_{t})}, \\quad i = 1, \\dots, ${S('N')} \\end{aligned}`,
    after: ['Guidance concentrates the samples too much and pays several times the necessary control cost, because it treats \\(\\x_1\\) as known once \\(\\x_t\\) is known. The exact \\(h_t\\) of this page averages the reward over the spread of \\(\\x_1\\) given \\(\\x_t\\), which flattens it at high noise. SMC pays for its accuracy with N network evaluations per step. §6 of the survey treats both families and their combinations.'],
    check: {
      q: 'Compare the readouts of guidance and the exact control. What is the bias of guidance on this example?',
      opts: [['Its samples are too concentrated and its control cost is several times too high', true], ['Its samples are too spread out', false], ['It has no bias; the difference is Monte Carlo noise', false]],
      fb: { true: 'Right. For β = 0.3 and a = 1 the guided samples have a smaller standard deviation than \\(p^\\star\\), and the guided drift costs about three and a half times the KL divergence it needs. Replacing \\(\\E[e^{r(\\x_1)/\\beta}\\mid\\x_t]\\) by \\(e^{r(\\hat\\x_1)/\\beta}\\) ignores the uncertainty about \\(\\x_1\\), so the pull towards a is too strong at high noise.', false: 'Read the standard deviations and control costs in the readout. Guidance gives a narrower histogram than \\(p^\\star\\) and a much larger cost than the exact control, at every number of samples. The cause is that \\(e^{r(\\hat\\x_1)/\\beta}\\) ignores the spread of \\(\\x_1\\) given \\(\\x_t\\).' },
    },
    closing: `<h3>Where this leads</h3>
      <p>§4 of the survey builds reward fine-tuning on step 6: Adjoint Matching and related methods learn the control \\(\\bm c^\\star\\) with the memoryless noise level. §5 returns to step 4 with reflow and distillation, which straighten paths until few steps suffice. §6 develops the guidance and Feynman–Kac methods of step 7 and compares them with best-of-N at equal compute. Each of the three chapters has its own tutorial: <a href="../post-training/">Reward fine-tuning and preference alignment</a>, <a href="../distillation/">Distillation and few-step generation</a> and <a href="../inference-time/">Inference-time scaling</a>. The reward-tilted target \\(p^\\star\\propto\\pdata\\,e^{r/\\beta}\\) is an energy-based model, so the chapter ends where §2.1 began.</p>
      <p>The earlier tutorials, <a href="../flow-matching/">Flows: the continuity equation and flow matching</a> and <a href="../sde/">Diffusion as an SDE</a>, derive the marginal velocity, the memoryless sampler and the reverse SDE used here. <a href="../ebm-langevin/">Energies, scores and Langevin dynamics</a> starts the chapter with energy-based models.</p>`,
  },
];
