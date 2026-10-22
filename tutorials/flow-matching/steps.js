// Content of the §2.4 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const x = S('x'), t = S('t'), pt = S('pt'), u = S('vel'), div = S('div'), phi = S('phi');
const x0 = S('x0'), x1 = S('x1'), al = S('alpha'), sg = S('sig'), ad = S('adot'), sd = S('sdot');
const ratio = `\\frac{${ad}}{${al}}`, glp = `${S('glp')}(${x})`, lamd = S('lamdot'), eps2 = `${S('eps')}^2`;
const ux = `${u}(${x})`;

export const INTRO = `
  <div class="kicker">Survey §2.4 · Interactive tutorial</div>
  <h1>Flows: the continuity equation and flow matching</h1>
  <p>The probability-flow ODE of the previous tutorial showed that a deterministic velocity field can carry noise to data. Flow matching starts from this end. It fixes the path of densities directly through the interpolation and learns the velocity that generates it, without any SDE. This tutorial goes from the continuity equation to the flow-matching loss, shows that the learned velocity and the score are the same object, and ends with the family of samplers that one trained velocity supports. Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>The convention is that of the survey: \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), where \\(t = 0\\) is noise and \\(t = 1\\) is data. The path is linear, \\(\\alpha_t = t\\) and \\(\\sigma_t = 1 - t\\), as in rectified flow, except where step 5 offers the cosine path. The data are the Gaussian mixtures of the earlier tutorials, so every density, score and velocity on this page is exact; only the model of step 4 is learned.</p>
  <ol class="toc">
    <li><a href="#continuity">The continuity equation</a></li>
    <li><a href="#conditional">One pair, one straight line</a></li>
    <li><a href="#marginal">The marginal velocity</a></li>
    <li><a href="#fm">Flow matching: regression on the conditional velocity</a></li>
    <li><a href="#identity">Velocity and score are the same object</a></li>
    <li><a href="#family">One marginal path, a family of samplers</a></li>
    <li><a href="#straight">Straight lines, curved paths and few steps</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'continuity', kicker: 'Transport without noise', title: 'The continuity equation', scene: 'continuity',
    prose: [
      'A velocity field \\(\\bm u_t\\) moves every point by the ODE \\(\\dot\\x_t = \\bm u_t(\\x_t)\\). If the starting points are drawn from \\(p_0\\), the density \\(p_t\\) of the moving points changes by the continuity equation, the first line below. It is the Fokker–Planck equation of §2.3 without the diffusion term: mass is neither created nor destroyed, it only flows, and the density at a point falls where more flows out than in. We say that \\(\\bm u_t\\) generates the path \\(p_t\\).',
      'The equation does not determine \\(\\bm u_t\\). Any extra velocity \\(\\bm w_t\\) whose flux \\(p_t\\bm w_t\\) has zero divergence can be added without changing a single density. The plot uses \\(\\bm w_t = c\\,\\sigma_t^2 J\\nabla\\log p_t\\), which moves points along the contour lines of \\(p_t\\). It starts 1000 points from \\(\\N(0, I)\\) and moves them with \\(\\bm u_t + \\bm w_t\\). Move \\(c\\): the paths swirl differently, but the cloud at every \\(t\\) stays on the heat map of the same \\(p_t\\), as the readout confirms.',
    ],
    tex: `\\begin{aligned} ${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) &= 0 \\\\ ${div}\\big(${pt}\\,${S('w')}\\big) = 0 \\;&\\Rightarrow\\; ${u} + ${S('w')} \\text{ generates the same } ${pt}, \\qquad ${S('w')} = ${S('c')}\\,${sg}^2\\,${S('J')}\\,${S('glp')} \\end{aligned}`,
    deriv: {
      have: ['An ODE \\(\\dot\\x_t = \\bm u_t(\\x_t)\\) whose state at time t has density \\(p_t\\).', 'Smooth test functions φ that vanish at infinity.'],
      goal: `${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) = 0, \\qquad ${div}\\big(${pt}\\,${S('w')}\\big) = 0`,
      plan: 'Follow φ along one path, average over paths and integrate by parts, as for Fokker–Planck in §2.3 but without noise. Then check that the extra flux of the plot has zero divergence.',
      title: 'Derivation of the continuity equation', lines: [
      { tex: `\\frac{\\dd}{\\dd ${t}}\\,${phi}(${x}) = \\nabla${phi}(${x})\\cdot${u}(${x})`, why: 'The chain rule for a smooth test function φ along one path of the ODE. There is no second-order term, because there is no noise.', uses: ['chain-rule'] },
      { tex: `\\frac{\\dd}{\\dd ${t}}\\int ${phi}\\,${pt}\\,\\dd\\x = \\int \\nabla${phi}\\cdot${u}\\;${CHG(pt)}\\,\\dd\\x`, why: 'Take expectations over the starting point. An expectation at time t is an integral against \\(p_t\\).', uses: ['expectation'] },
      { tex: `\\int ${phi}\\,${S('dtp')}\\,\\dd\\x = ${CHG(`-\\int ${phi}\\;${div}\\big(${pt}\\,${u}\\big)\\,\\dd\\x`)}`, why: 'Integrate by parts to move the derivative from φ to \\(p_t\\bm u_t\\). The boundary term vanishes because φ vanishes at infinity.', uses: ['ibp'] },
      { tex: `${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) = 0`, why: 'Both sides agree for every test function, so the integrands agree.', uses: ['test-functions'] },
      { tex: `${div}\\big(${pt}\\,${S('c')}\\,${sg}^2${S('J')}\\nabla\\log ${pt}\\big) = ${S('c')}\\,${sg}^2\\,${div}\\big(${S('J')}\\nabla ${pt}\\big) = ${S('c')}\\,${sg}^2\\big(${CHG('-\\partial_1\\partial_2 p_t + \\partial_2\\partial_1 p_t')}\\big) = 0`, why: 'The extra flux of the plot. \\(p_t\\nabla\\log p_t = \\nabla p_t\\), and \\(J\\) turns \\((\\partial_1 p_t, \\partial_2 p_t)\\) into \\((-\\partial_2 p_t, \\partial_1 p_t)\\). Mixed partial derivatives commute, so the divergence is zero.', uses: ['log-deriv', 'div-grad-lap', 'mixed-partials'] },
    ] },
    after: ['So a path \\(p_t\\) alone does not determine a velocity, and a model cannot simply be asked for "the" velocity of a path. Step 3 picks a canonical one, and step 4 shows that it is the one a regression learns.'],
    check: {
      q: 'In one dimension, can a nonzero extra velocity \\(w_t\\) leave every \\(p_t\\) unchanged?',
      opts: [['Yes, any odd function will do', false], ['No: \\(p_t w_t\\) would have to be constant, and it vanishes at infinity', true], ['Only at \\(t = 0\\)', false]],
      fb: { true: 'Right. In 1D the divergence is the derivative, so \\(p_t w_t\\) is constant in x, and it must be zero far out. The velocity that generates a 1D path is unique, which is why the 1D flow of §2.3 keeps every quantile.', false: 'In 1D the condition is \\(\\frac{\\dd}{\\dd x}(p_t w_t) = 0\\), so \\(p_t w_t\\) is constant. Far out in the tails both \\(p_t\\) and the flux vanish, so the constant is zero and \\(w_t = 0\\) wherever \\(p_t > 0\\). Swirls need at least two dimensions.' },
    },
  },
  {
    id: 'conditional', kicker: 'The conditional path', title: 'One pair, one straight line', scene: 'conditional',
    prose: [
      'Instead of a velocity, flow matching starts from the interpolation of the survey. Draw a noise sample \\(\\x_0\\) and a data point \\(\\x_1\\) independently and connect them by \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\). The marginal law of \\(\\x_t\\) over all pairs is \\(p_t\\), the path we want to generate. Differentiating in \\(t\\) for a single pair gives its velocity, the conditional velocity of the second line.',
      'For the linear path, each pair moves along the straight line from \\(\\x_0\\) to \\(\\x_1\\) at the constant velocity \\(\\x_1 - \\x_0\\). The plot shows 120 such pairs and moves the points along their lines; press "Play" or move \\(t\\). One pair is highlighted with its velocity. The lines cross everywhere, and the readout counts the crossings. The conditional velocity at a point is therefore not a function of the point: different pairs pass through the same place in different directions.',
    ],
    tex: `\\begin{aligned} ${x} &= ${al}\\,${x1} + ${sg}\\,${x0}, \\qquad ${x0}\\sim\\N(0, I),\\ ${x1}\\sim\\pdata \\\\ \\dot{\\x}_t &= ${ad}\\,${x1} + ${sd}\\,${x0} \\;=\\; ${x1} - ${x0} \\quad\\text{for } \\alpha_t = t,\\ \\sigma_t = 1 - t \\end{aligned}`,
    after: ['A velocity field must assign one velocity to each point and time, so it cannot follow these lines. The next step averages them.'],
    check: {
      q: 'On the linear path, how does the velocity of one pair change as \\(t\\) goes from 0 to 1?',
      opts: [['It is the same at every \\(t\\)', true], ['It grows, because the point accelerates towards the data', false], ['It shrinks to zero at \\(t = 1\\)', false]],
      fb: { true: 'Right. \\(\\dot\\alpha_t = 1\\) and \\(\\dot\\sigma_t = -1\\) are constants, so \\(\\dot\\x_t = \\x_1 - \\x_0\\) does not depend on t. Each point moves at constant speed along a straight line.', false: 'Insert \\(\\dot\\alpha_t = 1\\) and \\(\\dot\\sigma_t = -1\\): the velocity is \\(\\x_1 - \\x_0\\), with no t in it. Watch the highlighted pair: its arrow does not change while it moves.' },
    },
  },
  {
    id: 'marginal', kicker: 'Averaging the lines', title: 'The marginal velocity', scene: 'marginal',
    prose: [
      'Many lines pass through any point \\(\\x\\) at time \\(t\\). The velocity that generates \\(p_t\\) is their average, weighted by how likely each pair is to be at \\(\\x\\) at that time: the conditional expectation in the formula. It is called the marginal velocity, and the derivation below proves that it satisfies the continuity equation of step 1 for the path of step 2.',
      'Move the cursor to place the probe \\(\\x\\), and click to pin it there. The coloured dots are 60 draws of \\(\\x_1\\) from its posterior given \\(\\x_t = \\x\\), coloured by mixture component. Each thin arrow is the conditional velocity \\(\\x_1 - \\x_0\\) of one such pair, scaled by \\(\\sigma_t\\) so that it ends at its dot. The bold arrow is their mean, and the dashed arrow is the exact marginal velocity, with the same scaling. Early in time the posterior spreads over all three components; late in time it concentrates on one.',
    ],
    tex: `${ux} = \\E\\big[\\,${S('cv')}\\;\\big|\\;${x} = \\x\\,\\big]`,
    deriv: {
      have: ['Conditional paths \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), each with its own velocity \\(\\dot\\x_t = \\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\).', 'The marginal velocity, defined as the average \\(\\bm u_t(\\x) = \\E[\\dot\\x_t\\mid\\x_t = \\x]\\).'],
      goal: `${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) = 0`,
      plan: 'Show that the averaged field satisfies the continuity equation of step 1 for the marginals \\(p_t\\). Then its ODE has the same marginals as the random lines.',
      title: 'Why the average generates the marginal path', lines: [
      { tex: `\\frac{\\dd}{\\dd ${t}}\\E\\,${phi}(${x}) = \\E\\big[\\nabla${phi}(${x})\\cdot\\dot\\x_t\\big]`, why: 'The chain rule along each conditional path, for a smooth test function φ. Here \\(\\dot\\x_t = \\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\).', uses: ['chain-rule'] },
      { tex: `= \\E\\big[\\nabla${phi}(${x})\\cdot${CHG(`\\E[\\dot\\x_t\\mid ${x}]`)}\\big] = \\E\\big[\\nabla${phi}(${x})\\cdot${CHG(`${u}(${x})`)}\\big]`, why: 'Condition on \\(\\x_t\\). The factor \\(\\nabla\\varphi(\\x_t)\\) is a function of \\(\\x_t\\), so only the velocity is averaged.', uses: ['cond-exp'] },
      { tex: `\\int ${phi}\\,${S('dtp')}\\,\\dd\\x = \\int \\nabla${phi}\\cdot${u}\\;${pt}\\,\\dd\\x = ${CHG(`-\\int ${phi}\\;${div}\\big(${pt}\\,${u}\\big)\\,\\dd\\x`)}`, why: 'Write both sides as integrals against \\(p_t\\) and integrate by parts, as in step 1.', uses: ['expectation', 'ibp'] },
      { tex: `${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) = 0`, why: 'This holds for every test function, so the marginal velocity generates \\(p_t\\).', uses: ['test-functions'] },
    ] },
    after: ['The marginal velocity is a property of the path, not of individual pairs. It can be computed here because the data are a Gaussian mixture, where the posterior of \\(\\x_1\\) is known. For real data it is not available, and the next step learns it.'],
    check: {
      q: 'At \\(t = 0\\), what is the marginal velocity \\(\\bm u_0(\\x)\\) on the linear path?',
      opts: [['\\(\\E[\\x_1] - \\x\\): every point heads towards the data mean', true], ['Zero, because nothing has moved yet', false], ['\\(\\x_1 - \\x\\) for the data point nearest to \\(\\x\\)', false]],
      fb: { true: 'Right. At t = 0, \\(\\x_t = \\x_0\\) carries no information about \\(\\x_1\\), so the posterior of \\(\\x_1\\) is the data distribution and \\(\\E[\\x_1 - \\x_0 \\mid \\x_0 = \\x] = \\E[\\x_1] - \\x\\). Set t = 0 in the plot: the dots spread over all three components and the bold arrow points to the data mean.', false: 'At t = 0, \\(\\x_t = \\x_0\\) says nothing about which data point the pair ends at. Average \\(\\x_1 - \\x_0\\) given \\(\\x_0 = \\x\\): the result is \\(\\E[\\x_1] - \\x\\). Set t = 0 in the plot and look where the bold arrow points.' },
    },
  },
  {
    id: 'fm', kicker: 'Learning', title: 'Flow matching: regression on the conditional velocity', scene: 'fm',
    prose: [
      'The marginal velocity is a conditional expectation, and a conditional expectation is what least-squares regression learns (the regression lemma of §2.2). The flow-matching loss therefore regresses a model \\(\\bm u_\\theta\\) onto the conditional velocity, which is easy to compute, and its minimizer is the marginal velocity, which is not. For the linear path the label is simply \\(\\x_1 - \\x_0\\).',
      'The plot trains a small model on the 1D mixture with Adam and minibatches of 128 random triples. The upper panel shows labels at the chosen \\(t\\): at each \\(\\x_t\\) they scatter widely, because many pairs pass through it. The bold curve is the model, the dashed curve the exact marginal velocity. The middle panel shows the loss, which approaches a floor above zero, the part of the label that no function of \\(\\x_t\\) can predict. The lower panel integrates the current model from noise to \\(t = 1\\) and compares the samples with the data density. Press "Train".',
    ],
    tex: `${S('LFM')}(${S('theta')}) = ${S('Et')}\\,\\big\\|\\,${S('utheta')}(${x}, ${t}) - \\big(${ad}\\,${x1} + ${sd}\\,${x0}\\big)\\big\\|^2`,
    deriv: {
      have: ['The flow-matching loss, which regresses the model on the random label \\(\\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\).', 'The marginal velocity of step 3, the conditional mean of that label given \\(\\x_t\\).'],
      goal: `${S('LFM')}(${S('theta')}) = \\E\\,\\big\\|${S('utheta')} - ${u}\\big\\|^2 + \\text{const}`,
      plan: 'Repeat the regression lemma of §2.2: add and subtract the conditional mean of the label, and show that the cross term vanishes.',
      title: 'Why the minimizer is the marginal velocity', lines: [
      { tex: `\\bm v := ${ad}\\,${x1} + ${sd}\\,${x0}, \\qquad ${S('LFM')} = \\E\\,\\big\\|${S('utheta')}(${x}) - \\bm v\\big\\|^2`, why: 'Call the label \\(\\bm v\\). It is random given \\(\\x_t\\).' },
      { tex: `= \\E\\,\\big\\|${S('utheta')}(${x}) - ${CHG(`${u}(${x})`)}\\big\\|^2 + \\E\\,\\big\\|${CHG(`${u}(${x})`)} - \\bm v\\big\\|^2 + 2\\,\\E\\big[(${S('utheta')} - ${u})\\cdot(${u} - \\bm v)\\big]`, why: 'Add and subtract the marginal velocity \\(\\bm u_t(\\x_t) = \\E[\\bm v\\mid\\x_t]\\) and expand the square.', uses: ['expand-square'] },
      { tex: `2\\,\\E\\Big[(${S('utheta')} - ${u})\\cdot\\big(${u} - ${CHG(`\\E[\\bm v\\mid ${x}]`)}\\big)\\Big] = 0`, why: 'The first factor of the cross term is a function of \\(\\x_t\\). Conditioning on \\(\\x_t\\) replaces \\(\\bm v\\) by its mean, which is \\(\\bm u_t\\).', uses: ['cond-exp'] },
      { tex: `${S('LFM')}(${S('theta')}) = \\E\\,\\big\\|${S('utheta')} - ${u}\\big\\|^2 + ${CHG('\\text{const}')}`, why: 'The second term does not depend on θ. It is the floor of the loss curve. Minimizing the loss is minimizing the distance to the marginal velocity.', uses: ['regression-lemma'] },
    ] },
    after: ['Training is simulation-free. A step needs one time, one noise draw and one data point per example, and it never integrates the ODE; only sampling does. This is the main practical difference from earlier continuous normalizing flows, which had to solve the ODE during training to evaluate a likelihood.'],
    check: {
      q: 'What does one training example of flow matching require?',
      opts: [['A time, a noise draw and a data point', true], ['An ODE solve from noise to data', false], ['The score \\(\\nabla\\log p_t\\) at \\(\\x_t\\)', false]],
      fb: { true: 'Right. The label \\(\\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\) is computed from the triple directly. Neither the marginal velocity nor the score is needed, and nothing is integrated.', false: 'Look at the loss: the expectation is over \\(t\\), \\(\\x_0\\) and \\(\\x_1\\), and the label is computed from them directly. No ODE is solved and no score appears.' },
    },
  },
  {
    id: 'identity', kicker: 'Two names for one field', title: 'Velocity and score are the same object', scene: 'identity',
    prose: [
      'The marginal velocity is a combination of the two denoisers of §2.2, \\(\\hat\\x_1 = \\E[\\x_1\\mid\\x_t]\\) and \\(\\hat\\x_0 = \\E[\\x_0\\mid\\x_t]\\), and Tweedie’s formula writes both through the score. Substituting gives the velocity–score identity below. Its right-hand side is exactly the probability-flow ODE of §2.3, step 6. For Gaussian paths, a flow-matching model and a score model learn the same field up to a known affine map, and what remains between "diffusion" and "flow" models is the choice of schedule, of network output and of weighting over time.',
      'The upper panel draws the exact marginal velocity of the 1D mixture, computed from the posterior means, as the bold curve, and the two terms of the identity, a linear part and a score part. The dots are their sum. Move \\(t\\) or switch to the cosine path: the dots stay on the bold curve. The lower panel shows the two coefficients as functions of \\(t\\).',
    ],
    tex: `${ux} = ${ratio}\\,${x} + \\tfrac12\\,${sg}^2\\,${lamd}\\,${glp}`,
    deriv: {
      have: ['The marginal velocity \\(\\bm u_t(\\x) = \\E[\\dot\\alpha_t\\x_1 + \\dot\\sigma_t\\x_0\\mid\\x_t = \\x]\\) of step 3.', 'Tweedie’s formula of §2.2 for the noise prediction, \\(\\hat\\x_0 = -\\sigma_t\\nabla\\log p_t(\\x)\\).'],
      goal: 'step',
      plan: 'Write the velocity with the two predictions \\(\\hat\\x_1\\) and \\(\\hat\\x_0\\), eliminate \\(\\hat\\x_1\\) with the interpolation, and insert Tweedie for \\(\\hat\\x_0\\).',
      lines: [
      { tex: `${ux} = ${ad}\\,${S('hx1')} + ${sd}\\,${S('hx0')}`, why: 'Take the conditional expectation of the conditional velocity given \\(\\x_t = \\x\\).', uses: ['cond-exp', 'expectation'] },
      { tex: `${x} = ${al}\\,${S('hx1')} + ${sg}\\,${S('hx0')} \\;\\Rightarrow\\; ${S('hx1')} = ${CHG(`\\frac{${x} - ${sg}\\,${S('hx0')}}{${al}}`)}`, why: 'Take the conditional expectation of the interpolation itself. Given \\(\\x_t = \\x\\), the left side is \\(\\x\\).', uses: ['cond-exp'] },
      { tex: `${ux} = ${ratio}\\,${x} + \\Big(${sd} - \\frac{${ad}\\,${sg}}{${al}}\\Big)${S('hx0')}`, why: 'Insert into the first line and collect the terms in \\(\\hat\\x_0\\).' },
      { tex: `${S('hx0')} = ${CHG(`-${sg}\\,${glp}`)}`, why: 'Tweedie’s formula of §2.2 for the noise prediction.', uses: ['tweedie'] },
      { tex: `${ux} = ${ratio}\\,${x} + \\Big(${sg}^2${ratio} - ${sg}${sd}\\Big)${glp} = ${ratio}\\,${x} + ${CHG(`\\tfrac12\\,${sg}^2\\,${lamd}`)}\\,${glp}`, why: 'Multiply out. Since \\(\\dot\\lambda_t = 2(\\dot\\alpha_t/\\alpha_t - \\dot\\sigma_t/\\sigma_t)\\), the bracket is \\(\\tfrac12\\sigma_t^2\\dot\\lambda_t\\).', uses: ['log-rules', 'log-deriv'] },
    ] },
    after: ['The identity can be read in both directions. A trained score model gives a velocity, so it can be sampled with any ODE solver, and a trained velocity model gives a score, which the next step needs. The two terms grow large near \\(t = 0\\), where \\(\\alpha_t\\) vanishes, while their sum stays bounded; this is one reason to let the network output the velocity rather than the score.'],
    check: {
      q: 'On the linear path at \\(t = 0.5\\), what is the coefficient \\(\\tfrac12\\sigma_t^2\\dot\\lambda_t\\) of the score?',
      answer: 1, tol: 0.01,
      fb: { true: 'Right. \\(\\dot\\lambda_t = 2/(t(1-t)) = 8\\) at t = 0.5, and \\(\\tfrac12 \\cdot 0.25 \\cdot 8 = 1\\). In general it is \\((1-t)/t\\) for the linear path. Read it off the lower panel at t = 0.5.', false: 'For the linear path \\(\\dot\\lambda_t = 2/(t(1-t))\\), which is 8 at t = 0.5, and \\(\\sigma_t^2 = 0.25\\). Multiply by one half.' },
    },
  },
  {
    id: 'family', kicker: 'Adding noise back', title: 'One marginal path, a family of samplers', scene: 'family',
    prose: [
      'The identity also shows how to add noise to a flow without changing its marginals. Add Brownian noise of size \\(\\varepsilon_t\\), and add to the drift the score term that undoes its spreading. The result, the formula below, has the marginals \\(p_t\\) for every choice of \\(\\varepsilon_t\\). \\(\\varepsilon = 0\\) is the ODE; \\(\\varepsilon_t^2 = \\sigma_t^2\\dot\\lambda_t\\) is the reverse SDE of §2.3. A single trained velocity, converted to a score by step 5, supports all of them.',
      'The plot sets \\(\\varepsilon_t = \\eta\\,\\sigma_t\\sqrt{\\dot\\lambda_t}\\) and runs 1000 particles of the 1D mixture with the exact fields, from stratified starting points at \\(t = 0.03\\). The colour of a path is fixed by its starting point: the lowest 45% start in one colour, the middle 30% and the top 25% in the others, matching the weights of the three modes. Move \\(\\eta\\). At \\(\\eta = 0\\) every colour ends in its own mode, because the ODE keeps quantiles. With more noise the colours mix, and the scatter on the right, end point against starting point, loses its order. The final histogram does not change.',
    ],
    tex: `\\dd${x} = \\Big(${ux} + \\tfrac12\\,${eps2}\\,${glp}\\Big)\\dd${t} + ${S('eps')}\\,\\dd${S('W')}, \\qquad ${S('eps')} = ${S('eta')}\\,${sg}\\sqrt{${lamd}}`,
    deriv: {
      have: ['The marginal velocity \\(\\bm u_t\\), which generates \\(p_t\\) by the continuity equation of step 1.', 'The Fokker–Planck equation of §2.3 for an SDE with noise level \\(\\varepsilon_t\\).'],
      goal: `${S('dtp')} + ${div}\\big(${pt}\\,${u}\\big) = 0 \\quad \\text{for every } ${S('eps')}`,
      plan: 'Write the Fokker–Planck equation of the SDE and show that its extra drift cancels the diffusion term exactly.',
      title: 'Why the marginals do not change', lines: [
      { tex: `${S('dtp')} = -${div}\\Big(\\big(${u} + \\tfrac12${eps2}\\nabla\\log ${pt}\\big)${pt}\\Big) + \\tfrac12${eps2}\\,\\Delta ${pt}`, why: 'The Fokker–Planck equation of §2.3 for this SDE: transport along the drift plus diffusion with coefficient \\(\\varepsilon_t^2\\).', uses: ['fokker-planck'] },
      { tex: `-${div}\\big(\\tfrac12${eps2}\\,${pt}\\nabla\\log ${pt}\\big) = -\\tfrac12${eps2}\\,${div}\\nabla ${pt} = ${CHG(`-\\tfrac12${eps2}\\,\\Delta ${pt}`)}`, why: 'The extra drift, using \\(p_t\\nabla\\log p_t = \\nabla p_t\\). Its transport term is exactly minus the diffusion term.', uses: ['log-deriv', 'div-grad-lap'] },
      { tex: `${S('dtp')} = -${div}\\big(${pt}\\,${u}\\big)`, why: 'The two terms cancel, and the continuity equation of step 1 remains. The marginal velocity generates \\(p_t\\), so the SDE has the marginals \\(p_t\\).', uses: ['continuity'] },
    ] },
    after: ['Larger \\(\\varepsilon\\) pulls particles towards high density through the score term and so corrects errors of a learned field, at the cost of more steps. \\(\\varepsilon = 0\\) allows high-order ODE solvers and the few-step methods of §6. The value \\(\\eta = 1\\), the reverse SDE, is the "memoryless" schedule: it injects just enough noise to make the final sample independent of the initial noise. Adjoint Matching builds reward fine-tuning on this property, which §3 and §4 of the survey return to.'],
    check: {
      q: 'Set \\(\\eta = 1\\). How strongly does the final sample depend on the starting point?',
      opts: [['Hardly at all: the rank correlation is close to 0', true], ['Completely: the order of the particles is kept', false], ['About as much as at \\(\\eta = 0.5\\)', false]],
      fb: { true: 'Right. The readout shows a rank correlation of about 0 at η = 1, against 1 at η = 0 and about 0.4 at η = 0.5. This is the memoryless property of the reverse SDE.', false: 'Read the rank correlation in the readout at η = 0, 0.5 and 1. It falls from 1 to about 0.4 and then to about 0: at η = 1 the end point no longer remembers where the particle started.' },
    },
  },
  {
    id: 'straight', kicker: 'Sampling cost', title: 'Straight lines, curved paths and few steps', scene: 'straight',
    prose: [
      'Sampling integrates \\(\\dot\\x = \\bm u_t(\\x)\\) from \\(t = 0\\) to \\(t = 1\\), and each step costs one evaluation of the network. The simplest solver is Euler’s method, the formula below. It is exact in one step if every path is a straight line traversed at constant speed. The conditional paths of step 2 are straight, but they cross, and the marginal velocity of step 3 averages the directions of all lines through a point. The marginal paths therefore curve.',
      'The plot runs Euler with \\(K\\) steps on the exact marginal velocity of the 2D mixture, from the same 1000 starting points as step 1. The faint curves are accurate paths, computed with 200 steps, and the dashed lines join their start and end points. Lower \\(K\\): the Euler polygons cut the corners of the curved paths, and the samples collapse towards the middle. At \\(K = 1\\) all of them land on one point.',
    ],
    tex: `${S('x', `\\x_{t+${S('h')}}`)} =${x} + ${S('h')}\\,${u}(${x}), \\qquad ${S('h')} = \\frac{1}{${S('K')}}`,
    after: ['Curvature is what makes few-step sampling inaccurate. Reducing it is the purpose of the couplings of §3, which pair noise and data so that fewer lines cross, and of reflow and the distillation methods of §6, which train on the paths of a previous model until they are nearly straight.'],
    check: {
      q: 'With \\(K = 1\\), where does Euler’s method put every sample?',
      opts: [['At the data mean \\(\\E[\\x_1]\\)', true], ['At the nearest mode of the data', false], ['It leaves them at their starting points', false]],
      fb: { true: 'Right. One step goes from t = 0 with \\(h = 1\\), and step 3 showed that \\(\\bm u_0(\\x) = \\E[\\x_1] - \\x\\). So \\(\\x + \\bm u_0(\\x) = \\E[\\x_1]\\) for every starting point. Set K = 1: all samples sit on the cross.', false: 'One Euler step from t = 0 with h = 1 gives \\(\\x + \\bm u_0(\\x)\\), and step 3 showed \\(\\bm u_0(\\x) = \\E[\\x_1] - \\x\\). Set K = 1 in the plot.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>§2.5 of the survey compares the training targets (data, noise, score and velocity) and their weightings over time, which step 5 showed to be the same field in different coordinates. §3 treats couplings and control, including the memoryless schedule of step 6, and §6 the distillation and reflow methods that straighten the paths of step 7.</p>
      <p>The earlier tutorials, <a href="../ebm-langevin/">Energies, scores and Langevin dynamics</a>, <a href="../score-matching/">Learning the score</a> and <a href="../sde/">Diffusion as an SDE</a>, derive the score, its regression loss and the SDEs used here. The animation <a href="../../animations/flow-matching/">A1, flow matching</a> shows the straight conditional paths, the curved marginal paths and an SDE with the same marginals side by side.</p>`,
  },
];
