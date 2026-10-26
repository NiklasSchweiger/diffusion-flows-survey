// Content of the §2.2 tutorial: one entry per step. Formulas follow paper/sections/02_fundamentals.tex;
// S(id) tags a symbol from symbols.js, CHG marks the part of a derivation line that changed.
import { live, S, CHG } from '../lib/tutorial.js';
import { SYMBOLS } from './symbols.js';

// S() looks up the default tex of a symbol, so the glossary must be registered before the steps are built.
live.symbols = SYMBOLS;

const sx = `${S('stheta')}(${S('x')})`;
const pc = `${S('pcond')}`;

export const INTRO = `
  <div class="kicker">Survey §2.2 · Interactive tutorial</div>
  <h1>Learning the score: from Hyvärinen to Tweedie</h1>
  <p>The first tutorial assumed that the score was known. Annealed Langevin dynamics then turned noise into samples of \\(p\\), given the score of the smoothed density \\(p_\\sigma\\) at every level of a noise ladder. In practice \\(p\\) is unknown, and all we have is a dataset of samples from it. This tutorial shows how a model learns those scores from the dataset alone.</p>
  <p>A dataset allows one operation: averaging over it. A training objective must therefore be an expectation under \\(p\\), and the quantity inside the expectation must be computable from one sample and the model. Steps 1, 2 and 5 test three objectives against these two requirements. Explicit score matching fails the second. Implicit score matching passes, but at a cost that grows with the dimension. Denoising score matching passes cheaply and learns all noise levels with one network. Steps 3 and 4 build the pieces it needs, and steps 6 and 7 show what else the learned score gives.</p>
  <p>Each step states one idea, shows one formula and gives you a plot that the formula controls. Every symbol can be hovered or tapped, and the button "Symbols" in the top bar opens the full list.</p>
  <p>The data are Gaussian mixtures, so every score, posterior and denoiser on this page is exact. The interpolation is \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with \\(\\alpha_t = t\\) and \\(\\sigma_t = 1 - t\\): \\(t = 0\\) is noise and \\(t = 1\\) is data.</p>
  <ol class="toc">
    <li><a href="#esm">Matching the score directly</a></li>
    <li><a href="#ism">Integration by parts removes the unknown score</a></li>
    <li><a href="#cond">The score of one noisy data point</a></li>
    <li><a href="#marginal">The marginal score is an average of conditional scores</a></li>
    <li><a href="#dsm">Regressing on a noisy target</a></li>
    <li><a href="#tweedie">Tweedie's formula: a denoiser is a score</a></li>
    <li><a href="#targets">Four interchangeable prediction targets</a></li>
  </ol>`;

export const STEPS = [
  {
    id: 'esm', kicker: 'Explicit score matching', title: 'Matching the score directly', scene: 'esm',
    prose: [
      'In §2.1 the score was given by a formula. Here we pretend it is not, and keep only the 200 samples drawn as dots under the axis. The two requirements of the introduction apply: the objective must be an average over these samples, and each term of the average must be computable.',
      'The most direct candidate compares the model score \\(\\s_\\theta\\) with the true score at every point. The squared distance is the natural way to compare two vector fields. Averaging it under \\(p\\) is what turns it into an expectation, and it also means that errors count only where the data lie. The result is the Fisher divergence, called explicit score matching. The derivation below shows why driving it to zero is enough: a model with the right score has the right density.',
      'On this page the model is the score of a single Gaussian \\(\\N(m, v)\\), a straight line \\(\\s_\\theta(x) = -(x-m)/v\\). Move \\(m\\) and \\(v\\). The lower panel shades the integrand, which is large where the line misses the dashed true score and \\(p\\) has mass. Its area is \\(J_{\\mathrm{ESM}}\\).',
    ],
    tex: `${S('J')}(${S('theta')}) = \\tfrac12\\,${S('Ep')}\\big\\|${sx} - ${S('glp')}(${S('x')})\\big\\|^2`,
    deriv: {
      title: 'Why matching the score is enough',
      have: ['A model density \\(p_\\theta\\) whose score is \\(\\s_\\theta = \\nabla\\log p_\\theta\\). As in §2.1, its normalizing constant does not enter the score.', 'A data density \\(p\\) that is positive everywhere, as the mixture here is.'],
      goal: `${S('J')}(${S('theta')}) = 0 \\;\\Longrightarrow\\; p_${S('theta')} = ${S('p')}`,
      plan: 'A zero average of a non-negative quantity forces the quantity to vanish. Equal scores mean equal log densities up to a constant, and normalization fixes the constant.',
      lines: [
      { tex: `${S('J')}(${S('theta')}) = 0 \\;\\Longrightarrow\\; ${sx} = ${S('glp')}(${S('x')}) \\text{ for all } ${S('x')}`, why: 'The integrand is a square times \\(p > 0\\). Its average is zero only if it vanishes everywhere, for continuous scores.', uses: ['expectation'] },
      { tex: `\\nabla\\big(${CHG(`\\log p_${S('theta')}(${S('x')}) - \\log ${S('p')}(${S('x')})`)}\\big) = 0`, why: 'Write \\(\\s_\\theta = \\nabla\\log p_\\theta\\) and move both terms under one gradient.', uses: ['grad-linear'] },
      { tex: `\\log p_${S('theta')}(${S('x')}) = \\log ${S('p')}(${S('x')}) + ${CHG('c')}`, why: 'A function with zero gradient everywhere is constant.' },
      { tex: `p_${S('theta')} = ${CHG('e^{c}')}\\,${S('p')}, \\qquad 1 = \\int p_${S('theta')}\\,\\dd${S('x')} = e^{c}\\!\\int ${S('p')}\\,\\dd${S('x')} = e^{c} \\;\\Longrightarrow\\; ${CHG('c = 0')}`, why: 'Exponentiate, then integrate both sides. Both densities integrate to one, so \\(e^c = 1\\) and the densities agree.' },
    ] },
    after: ['So \\(J_{\\mathrm{ESM}}\\) is the right target, and it passes the first requirement: it is an average under \\(p\\). It fails the second, because each term contains \\(\\nabla\\log p\\), the very function we want to learn, and the samples do not tell us its value. (A straight line cannot follow a three-mode score, so \\(J_{\\mathrm{ESM}}\\) never reaches zero on this page. The best line is still well defined.)'],
    check: {
      q: 'Where does a missed score cost the most in \\(J_{\\mathrm{ESM}}\\)?',
      opts: [['Where the gap between the two curves is largest', false], ['Where the gap is large and p has mass', true], ['In the tails, where p is small', false]],
      fb: { true: 'Right. The squared gap is multiplied by p. In the tails the gap can be huge and still cost almost nothing, which is why score models are poor far from the data.', false: 'Look at the lower panel. The squared gap is multiplied by p(x), so a large miss in the tails, where p is nearly zero, adds almost nothing to the area.' },
    },
  },
  {
    id: 'ism', kicker: 'Implicit score matching', title: 'Integration by parts removes the unknown score', scene: 'ism',
    prose: [
      'Hyvärinen (2005) showed that the unknown score can be removed, which fixes the second requirement. Expanding the square gives a cross term \\(\\E_p[\\s_\\theta\\cdot\\nabla\\log p]\\), and integration by parts turns it into \\(-\\E_p[\\diver\\s_\\theta]\\), which needs only the model. Step through the derivation below.',
      'The result differs from \\(J_{\\mathrm{ESM}}\\) by a constant \\(C\\) that does not depend on \\(\\theta\\). Move \\(m\\) and \\(v\\) and watch both readouts: they change together, and their difference stays fixed. Then press "Fit from samples". It runs gradient descent on an estimate of the new objective from the 200 samples on the axis, and never evaluates \\(p\\) or its score.',
    ],
    tex: `${S('J')}(${S('theta')}) = ${S('Ep')}\\Big[\\tfrac12\\|${sx}\\|^2 + ${S('div')}${sx}\\Big] + ${S('C')}`,
    deriv: {
      have: ['The explicit loss \\(J(\\theta) = \\tfrac12\\E_p\\|\\s_\\theta(\\x) - \\nabla\\log p(\\x)\\|^2\\) of step 1. It contains the unknown score.', 'Samples from p, but not p itself.'],
      goal: 'step',
      plan: 'Expand the square. Only the cross term contains both the model and the unknown score, and integration by parts moves the derivative from p onto the model.',
      lines: [
      { tex: `${S('J')} = \\tfrac12${S('Ep')}\\|${S('stheta')}\\|^2 - ${S('Ep')}\\big[${S('stheta')}\\cdot${S('glp')}\\big] + \\tfrac12${S('Ep')}\\|${S('glp')}\\|^2`, why: 'Expand the square \\(\\|a-b\\|^2 = \\|a\\|^2 - 2a\\cdot b + \\|b\\|^2\\) and take the expectation term by term.', uses: ['expand-square', 'expectation'] },
      { tex: `${S('Ep')}\\big[${S('stheta')}\\cdot${S('glp')}\\big] = \\int ${S('p')}\\,${S('stheta')}\\cdot${CHG(`\\frac{\\nabla ${S('p')}}{${S('p')}}`)}\\,\\dd${S('x')} = \\int ${S('stheta')}\\cdot\\nabla ${S('p')}\\,\\dd${S('x')}`, why: 'Write the expectation as an integral and use \\(\\nabla\\log p = \\nabla p/p\\). The density cancels.', uses: ['expectation', 'log-deriv'] },
      { tex: `= ${CHG(`-\\int ${S('p')}\\,${S('div')}${S('stheta')}\\,\\dd${S('x')}`)} = -${S('Ep')}\\big[${S('div')}${S('stheta')}\\big]`, why: 'Integrate by parts. The boundary term vanishes when \\(p\\,\\s_\\theta\\) goes to zero at infinity.', uses: ['ibp'] },
      { tex: `${S('J')} = ${S('Ep')}\\Big[\\tfrac12\\|${S('stheta')}\\|^2 ${CHG(`+ ${S('div')}${S('stheta')}`)}\\Big] + ${CHG(`${S('C')}`)},\\quad ${S('C')} = \\tfrac12${S('Ep')}\\|${S('glp')}\\|^2`, why: 'Insert the cross term into the first line. The last term of the first line does not contain θ, so it becomes the constant C.' },
    ] },
    after: ['For the Gaussian model, \\(\\diver\\s_\\theta = -1/v\\) everywhere, and the minimizer is the sample mean and variance. The fit lands there. Implicit score matching therefore passes both requirements. Its cost is the divergence: for a neural network in \\(d\\) dimensions, it is the trace of a \\(d\\times d\\) Jacobian, needed again for every sample.'],
    check: {
      q: 'Computing \\(\\diver\\s_\\theta\\) exactly for one image with \\(d = 10^5\\) pixels takes how many backward passes through the network?',
      opts: [['One', false], ['About d', true], ['About d²', false]],
      fb: { true: 'Right. Each backward pass gives one row of the Jacobian, and the trace needs all d diagonal entries. This cost is what denoising score matching avoids.', false: 'One backward pass gives the gradient of one output coordinate, which is one row of the Jacobian. The trace needs d diagonal entries from d different rows, so it costs about d passes.' },
    },
  },
  {
    id: 'cond', kicker: 'Adding noise', title: 'The score of one noisy data point', scene: 'cond',
    prose: [
      'Vincent (2011) avoided the Jacobian by adding noise to the data. There are two reasons to do so. This step shows the first: with noise, the target becomes computable. The second comes from §2.1. Annealed Langevin dynamics never used the score of \\(p\\) itself. It used the scores of the smoothed densities \\(p_\\sigma\\) at a whole ladder of noise levels, so those are the scores we need to learn anyway.',
      'From here on the noisy point is written \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\), with a data point \\(\\x_1\\) and noise \\(\\x_0 \\sim \\N(0, I)\\). Dividing by \\(\\alpha_t\\) gives \\(\\x_t/\\alpha_t = \\x_1 + (\\sigma_t/\\alpha_t)\\,\\x_0\\), which is the noisy point of §2.1 with \\(\\sigma = \\sigma_t/\\alpha_t\\). Each time \\(t\\) is therefore one rung of the noise ladder, and \\(p_t\\) is \\(p_\\sigma\\) shrunk by the factor \\(\\alpha_t\\). Here \\(t = 1\\) is the data and \\(t \\to 0\\) is pure noise.',
      'Now fix one data point \\(\\x_1\\). The noisy point \\(\\x_t\\) is then Gaussian around \\(\\alpha_t\\x_1\\) with standard deviation \\(\\sigma_t\\), and the score of a Gaussian is known in closed form.',
      'Click in the plot to choose \\(\\x_1\\), and move \\(t\\). Every arrow is the conditional score at one noisy point. It points back to \\(\\alpha_t\\x_1\\), and its length is \\(\\|\\x_0\\|/\\sigma_t\\): it depends only on the noise draw. As \\(t \\to 1\\) the circles shrink and the arrows grow without bound.',
    ],
    tex: `\\begin{aligned} ${pc} &= \\N\\big(${S('x')};\\,${S('alpha')}${S('x1')},\\,${S('sig')}^2 I\\big) \\\\ ${S('cs')} &= -\\frac{${S('xt')} - ${S('alpha')}${S('x1')}}{${S('sig')}^2} = -\\frac{${S('x0')}}{${S('sig')}} \\end{aligned}`,
    deriv: {
      have: ['The interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) with \\(\\x_0\\sim\\N(0, I)\\). Given \\(\\x_1\\), it is Gaussian with mean \\(\\alpha_t\\x_1\\) and covariance \\(\\sigma_t^2 I\\).'],
      goal: `${S('cs')} = -\\frac{${S('x0')}}{${S('sig')}}`,
      plan: 'Write the log density of this Gaussian, take its gradient in \\(\\x\\), and evaluate it at the noisy sample \\(\\x_t\\).',
      lines: [
      { tex: `\\log ${pc} = -\\frac{\\|${S('x')} - ${S('alpha')}${S('x1')}\\|^2}{2${S('sig')}^2} + \\text{const}`, why: 'Log of an isotropic Gaussian density. The constant collects the normalization, which does not depend on x.', uses: ['gauss', 'log-rules'] },
      { tex: `\\nabla_{${S('x')}}\\log ${pc} = ${CHG(`-\\frac{${S('x')} - ${S('alpha')}${S('x1')}}{${S('sig')}^2}`)}`, why: 'The gradient of \\(\\|\\x-\\bm c\\|^2/2\\) is \\(\\x - \\bm c\\).', uses: ['gauss'] },
      { tex: `${S('cs')} = -\\frac{${CHG(`${S('sig')}${S('x0')}`)}}{${S('sig')}^2} = -\\frac{${S('x0')}}{${S('sig')}}`, why: 'Evaluate at \\(\\x = \\x_t\\) and use \\(\\x_t - \\alpha_t\\x_1 = \\sigma_t\\x_0\\).' },
    ] },
    check: {
      q: 'At \\(t = 0.75\\), so \\(\\sigma_t = 0.25\\), a noise draw has \\(\\|\\x_0\\| = 1\\). How long is the conditional score at \\(\\x_t\\)?',
      answer: 4, tol: 0.01,
      fb: { true: 'Right. ‖x_0‖/σ_t = 1/0.25 = 4. Set t = 0.75: the readout gives the same value for the highlighted draw, which has ‖x_0‖ = 1.', false: 'Use the last form of the formula: the length is ‖x_0‖ divided by σ_t. With ‖x_0‖ = 1 and σ_t = 0.25 this is 1/0.25.' },
    },
  },
  {
    id: 'marginal', kicker: 'Marginal score', title: 'The marginal score is an average of conditional scores', scene: 'marginal',
    prose: [
      'Fix a noisy point \\(\\x\\). Many data points \\(\\x_1\\) could have produced it, each with a posterior probability given \\(\\x_t = \\x\\). Draw 60 such data points from that posterior. Each has its own conditional score, drawn as a thin arrow from \\(\\x\\) toward \\(\\alpha_t\\x_1\\), and a dot at \\(\\alpha_t\\x_1\\) coloured by its mixture component.',
      'The bold arrow is the average of the thin arrows. The dashed arrow is the exact marginal score \\(\\nabla\\log p_t(\\x)\\). Move the probe and \\(t\\): the two agree up to the error of averaging 60 draws. Near a mode at large \\(t\\), the posterior is narrow and the arrows agree. Between modes, they point in different directions and partly cancel.',
    ],
    tex: `\\begin{aligned} ${S('ms')} &= \\E\\big[${S('cs')}\\,\\big|\\,${S('xt')} = ${S('x')}\\big] \\\\ &= -\\frac{1}{${S('sig')}}\\,${S('Ex0')} \\end{aligned}`,
    deriv: {
      have: ['The marginal density as a mixture of conditionals, \\(p_t(\\x) = \\int p(\\x\\mid\\x_1)\\,\\pdata(\\x_1)\\,\\dd\\x_1\\).', 'The conditional score \\(-\\x_0/\\sigma_t\\) of step 3.'],
      goal: 'step',
      plan: 'Differentiate the mixture under the integral and divide by \\(p_t\\). The weights that appear are the posterior of \\(\\x_1\\) given \\(\\x_t\\), by Bayes’ rule.',
      lines: [
      { tex: `${S('pt')}(${S('x')}) = \\int ${pc}\\,p_{\\mathrm{data}}(${S('x1')})\\,\\dd${S('x1')}`, why: 'The marginal density averages the conditional densities over the data.', uses: ['bayes'] },
      { tex: `\\nabla ${S('pt')}(${S('x')}) = \\int ${CHG(`\\nabla\\log ${pc}\\;${pc}`)}\\,p_{\\mathrm{data}}(${S('x1')})\\,\\dd${S('x1')}`, why: 'Differentiate under the integral and write \\(\\nabla q = q\\nabla\\log q\\).', uses: ['diff-under-int', 'log-deriv'] },
      { tex: `${S('ms')} = \\int \\nabla\\log ${pc}\\;${CHG(`\\frac{${pc}\\,p_{\\mathrm{data}}(${S('x1')})}{${S('pt')}(${S('x')})}`)}\\,\\dd${S('x1')}`, why: 'Divide by \\(p_t\\)(x). The fraction is the posterior density of \\(\\x_1\\) given \\(\\x_t = \\x\\), by Bayes’ rule.', uses: ['bayes'] },
      { tex: `= ${CHG(`\\E\\big[${S('cs')}\\,\\big|\\,${S('xt')} = ${S('x')}\\big]`)} = -\\frac{1}{${S('sig')}}\\,${S('Ex0')}`, why: 'An integral against the posterior is a conditional expectation. Insert the conditional score \\(-\\x_0/\\sigma_t\\) from step 3.', uses: ['cond-exp', 'cond-score'] },
    ] },
    after: ['The marginal score at \\(\\x\\) is the average of the conditional scores of all data points that could have produced \\(\\x\\). The conditional scores are known, but the average needs the posterior, which needs \\(p_t\\). Step 5 shows that a regression computes this average for us, from samples alone.'],
  },
  {
    id: 'dsm', kicker: 'Denoising score matching', title: 'Regressing on a noisy target', scene: 'dsm',
    prose: [
      'Denoising score matching trains the model to predict the conditional score \\(-\\x_0/\\sigma_t\\) from \\(\\x_t\\). The plot shows 2500 training pairs from the one-dimensional data of steps 1 and 2: each dot is one \\(\\x_t\\) on the horizontal axis and its label \\(-\\x_0/\\sigma_t\\) on the vertical axis. The labels are very noisy, because many different data points can produce the same \\(\\x_t\\).',
      'The lemma below says what a squared-error regression on noisy labels learns: the conditional mean of the label at each input. The open circles are the mean of the dots in narrow bins. They follow the solid curve, which is the exact marginal score \\(\\nabla\\log p_t\\), as step 4 predicts. Move the probe and \\(t\\) to compare them.',
    ],
    tex: `\\begin{aligned} ${S('Jd')}(${S('theta')}) &= \\tfrac12\\,\\E_{${S('x1')},${S('x0')}}\\Big\\|${S('stheta')}(${S('xt')},${S('t')}) + \\frac{${S('x0')}}{${S('sig')}}\\Big\\|^2 \\\\ \\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 &= \\E\\|${S('f')}(${S('X')}) - ${S('EY')}\\|^2 \\\\ &\\quad + \\E\\|${S('Y')} - ${S('EY')}\\|^2 \\end{aligned}`,
    deriv: {
      have: ['Two random variables X and Y, and any function f of X.', 'The conditional mean \\(\\E[Y\\mid X]\\), which is itself a function of X.'],
      goal: `\\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 = \\E\\|${S('f')}(${S('X')}) - ${S('EY')}\\|^2 + \\E\\|${S('Y')} - ${S('EY')}\\|^2`,
      plan: 'Add and subtract the conditional mean and expand the square. The cross term has mean zero, which conditioning on X shows.',
      title: 'Proof of the lemma', lines: [
      { tex: `${S('f')}(${S('X')}) - ${S('Y')} = \\big(${S('f')}(${S('X')}) - ${S('EY')}\\big) + \\big(${S('EY')} - ${S('Y')}\\big)`, why: 'Add and subtract the conditional mean.', uses: ['expand-square'] },
      { tex: `\\E\\|${S('f')} - ${S('Y')}\\|^2 = \\E\\|${S('f')} - ${S('EY')}\\|^2 + \\E\\|${S('Y')} - ${S('EY')}\\|^2 + ${CHG(`2\\,\\E\\big[(${S('f')} - ${S('EY')})\\cdot(${S('EY')} - ${S('Y')})\\big]`)}`, why: 'Expand the square of the sum.', uses: ['expand-square', 'expectation'] },
      { tex: `\\E\\big[(${S('f')} - ${S('EY')})\\cdot(${S('EY')} - ${S('Y')})\\big] = \\E\\big[(${S('f')} - ${S('EY')})\\cdot ${CHG(`\\E[${S('EY')} - ${S('Y')}\\mid ${S('X')}]`)}\\big] = ${CHG('0')}`, why: 'Condition on X. Given X, the first factor is fixed, and the second has conditional mean zero.', uses: ['cond-exp'] },
      { tex: `\\E\\|${S('f')}(${S('X')}) - ${S('Y')}\\|^2 = \\E\\|${S('f')}(${S('X')}) - ${S('EY')}\\|^2 + \\E\\|${S('Y')} - ${S('EY')}\\|^2`, why: 'Only the first term depends on f. It is zero for \\(f = \\E[Y\\mid X]\\), which is therefore the minimizer.' },
    ] },
    after: ['Put \\(X = \\x_t\\) and \\(Y = -\\x_0/\\sigma_t\\). By step 4, \\(\\E[Y\\mid X]\\) is the marginal score \\(\\nabla\\log p_t\\), so the first term of the lemma is the explicit objective of step 1 for \\(p_t\\), and the second term is a constant. Like the implicit objective, \\(J_{\\mathrm{DSM}}\\) equals \\(J_{\\mathrm{ESM}}\\) plus a constant, now for the noised density \\(p_t\\).', 'Denoising score matching passes both requirements of step 1. It is an average over data points and noise draws, and each term needs only the model and the noise that was added, with no divergence. Sampling \\(t\\) at random as well trains one network \\(\\s_\\theta(\\x, t)\\) for every noise level at once. The same pattern of a tractable conditional target with an intractable marginal minimizer returns in flow matching (§2.4), in distillation and in reward fine-tuning.'],
    check: {
      q: 'A perfect score model is trained with \\(J_{\\mathrm{DSM}}\\). What is its loss?',
      opts: [['Zero', false], ['The second term of the lemma, the spread of the labels around their mean', true], ['It cannot be computed', false]],
      fb: { true: 'Right. The readout shows it: the exact score still has a large loss, equal to the variance of the dots around the curve. This is why DSM loss values say little about model quality on their own.', false: 'Look at the readout "loss of the exact score". Even the exact marginal score cannot predict which noise draw produced a given x_t, so the spread of the labels, the second term of the lemma, remains.' },
    },
  },
  {
    id: 'tweedie', kicker: 'Tweedie’s formula', title: 'Tweedie’s formula: a denoiser is a score', scene: 'tweedie',
    prose: [
      'The best guess of the clean data point behind a noisy point is the posterior mean \\(\\E[\\x_1\\mid\\x_t]\\), the denoiser. Tweedie’s formula computes it from the score. Starting at \\(\\x\\), step \\(\\sigma_t^2\\) along the score and divide by \\(\\alpha_t\\).',
      'The plot shows 300 noisy points, each joined to its denoised position. Move \\(t\\). Close to \\(t = 1\\) little noise has been added and each point barely moves. Further toward \\(t = 0\\) the denoised points are pulled to the centres of the modes, and close to \\(t = 0\\) they all collapse onto the data mean, because the noisy point says almost nothing about which data point produced it. Hover the plot to see the two terms of the formula at the probe.',
    ],
    tex: `${S('Ex1')} = \\frac{${S('x')} + ${S('sig')}^2\\,${S('ms')}}{${S('alpha')}}`,
    deriv: {
      have: ['The interpolation \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\).', 'The noise prediction of step 4, \\(\\E[\\x_0\\mid\\x_t = \\x] = -\\sigma_t\\nabla\\log p_t(\\x)\\).'],
      goal: 'step',
      plan: 'Take the conditional expectation of the interpolation given \\(\\x_t\\), insert the noise prediction, and solve for the data prediction.',
      lines: [
      { tex: `${S('x')} = ${S('alpha')}\\,${S('Ex1')} + ${S('sig')}\\,${S('Ex0')}`, why: 'Take the conditional expectation of \\(\\x_t = \\alpha_t\\x_1 + \\sigma_t\\x_0\\) given \\(\\x_t = \\x\\). The left side is known given itself.', uses: ['cond-exp', 'expectation'] },
      { tex: `${S('x')} = ${S('alpha')}\\,${S('Ex1')} ${CHG(`- ${S('sig')}^2\\,${S('ms')}`)}`, why: 'Insert \\(\\E[\\x_0\\mid\\x_t=\\x] = -\\sigma_t\\nabla\\log p_t(\\x)\\) from step 4.', uses: ['marginal-score'] },
      { tex: `${S('Ex1')} = ${CHG(`\\frac{${S('x')} + ${S('sig')}^2\\,${S('ms')}}{${S('alpha')}}`)}`, why: 'Solve for the posterior mean.' },
    ] },
    after: ['The denoiser and the score carry the same information. A network trained to denoise is a score model, and a score model is a denoiser. The survey uses this identity for posterior sampling in inverse problems (§6), for distribution-matching distillation (§5) and for representation alignment (§2.8).'],
    check: {
      q: 'As \\(t\\) approaches 0, where does \\(\\E[\\x_1\\mid\\x_t]\\) go for every noisy point?',
      opts: [['It stays at the noisy point', false], ['To the nearest mode', false], ['To the mean of the data', true]],
      fb: { true: 'Right. At t = 0 the noisy point is pure noise and carries no information about x_1, so the best guess is the unconditional mean. Move t toward 0 and the filled dots meet at the cross.', false: 'At t = 0 the noisy point is pure noise, independent of x_1. The posterior is the prior, so the best guess in squared error is the data mean. Move t toward 0 and watch the filled dots.' },
    },
  },
  {
    id: 'targets', kicker: 'Parametrizations', title: 'Four interchangeable prediction targets', scene: 'targets',
    prose: [
      'Write \\(\\hx_1\\) for the denoiser and \\(\\hx_0\\) for the posterior mean of the noise. Because \\(\\x = \\alpha_t\\hx_1 + \\sigma_t\\hx_0\\), either one determines the other, and both determine the score and the velocity of flow matching. A network can therefore be trained to output any of them. In the plot the two legs \\(\\alpha_t\\hx_1\\) and \\(\\sigma_t\\hx_0\\) add up to the probe \\(\\x\\).',
      'At the optimum all four give the same model, but their losses weight time differently. The same error in \\(\\hx_1\\) costs a factor \\(\\alpha_t^2/\\sigma_t^2\\) more as a noise error, and the velocity loss of the linear path multiplies it by \\((1-t)^{-2}\\). The lower plot shows these weights on a log scale.',
    ],
    tex: `\\begin{aligned} ${S('hx0')} &= -${S('sig')}\\,${S('st')}, \\qquad ${S('hx1')} = \\frac{${S('x')} + ${S('sig')}^2\\,${S('st')}}{${S('alpha')}} \\\\ ${S('hu')} &= ${S('adot')}\\,${S('hx1')} + ${S('sdot')}\\,${S('hx0')} \\\\ \\|${S('hu')} - \\vel_t\\|^2 &= \\Big(${S('adot')} - ${S('sdot')}\\frac{${S('alpha')}}{${S('sig')}}\\Big)^2\\,\\|${S('hx1')} - ${S('x1')}\\|^2 \\end{aligned}`,
    after: ['Near \\(t = 0\\) the input is almost pure noise, so predicting \\(\\hx_0\\) is easy and \\(\\hx_1\\) is hard. Near \\(t = 1\\) the opposite holds. §2.5 of the survey returns to the choice of weighting.'],
    check: {
      q: 'For the linear path at \\(t = 0.5\\), by what factor does the velocity loss weight an error in \\(\\hx_1\\)?',
      answer: 4, tol: 0.01,
      fb: { true: 'Right. With α̇ = 1, σ̇ = −1 the weight is (1 + t/(1 − t))² = (1 − t)^{−2} = 4 at t = 0.5. The lower plot shows the same value at t = 0.5.', false: 'Insert α̇_t = 1, σ̇_t = −1, α_t = σ_t = 0.5 into the bracket: 1 − (−1)(0.5/0.5) = 2, and its square is 4. You can also read the velocity curve at t = 0.5.' },
    },
    closing: `<h3>Where this leads</h3>
      <p>This closes the loop with the first tutorial. Annealed Langevin dynamics needed the score of \\(p_\\sigma\\) at every level of a noise ladder. One network \\(\\s_\\theta(\\x, t)\\), trained with \\(J_{\\mathrm{DSM}}\\) at random times, provides all of them from data alone. With \\(\\sigma = \\sigma_t/\\alpha_t\\) and the rescaling by \\(\\alpha_t\\) undone, it can replace the exact mixture score in the last step of <a href="../ebm-langevin/#annealing">Energies, scores and Langevin dynamics</a>.</p>
      <p><a href="../sde/">§2.3 of the survey</a> makes the noise level continuous and shows that the score learned here drives both a reverse-time SDE and a deterministic probability-flow ODE. §2.4 replaces the score target by the velocity target of the last step. The tutorial <a href="../ddpm/">DDPM: a chain of small Gaussian steps</a> reaches the same denoising loss from a variational bound and shows that its noise predictor is a rescaled score.</p>
      <p>The animation <a href="../../animations.html#a14">A14, posterior sampling</a> plugs Tweedie’s denoiser into a likelihood to solve inverse problems.</p>`,
  },
];
